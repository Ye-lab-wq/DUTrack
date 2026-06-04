import math

import torch
from torch import nn


class TrackingEvidenceUnitLayer(nn.Module):
    """Phrase-aware evidence-unit layer before the tracking head.

    This Stage-2R layer is intentionally separate from the old token-level
    TrackingEvidenceLayer. It first builds local phrase/evidence units around
    anchor tokens, then lets each search region read those units in a
    target-conditioned evidence space.
    """

    def __init__(
            self,
            dim,
            evidence_dim=128,
            gamma_init=0.01,
            beta=0.25,
            d_mag_max=1.0,
            d_norm_eps=1e-4,
            residual_init_scale=1e-3,
            lang_source="raw",
            target_pool="center",
            center_ratio=0.5,
            min_evidence_units=2,
            phrase_window=3,
            dropout=0.0):
        super().__init__()
        if lang_source not in ("raw", "fuse"):
            raise ValueError("Evidence unit layer lang_source must be 'raw' or 'fuse'.")
        if target_pool not in ("center", "mean"):
            raise ValueError("Evidence unit layer target_pool must be 'center' or 'mean'.")
        if beta <= 0.0 or beta >= 1.0:
            raise ValueError("Evidence unit layer beta should be in (0, 1).")
        if d_norm_eps <= 0.0:
            raise ValueError("Evidence unit layer d_norm_eps must be positive.")
        if residual_init_scale <= 0.0:
            raise ValueError("Evidence unit layer residual_init_scale must be positive.")
        if phrase_window < 1 or phrase_window % 2 != 1:
            raise ValueError("Evidence unit layer phrase_window must be an odd positive integer.")

        self.lang_source = lang_source
        self.target_pool = target_pool
        self.center_ratio = center_ratio
        self.min_evidence_units = int(min_evidence_units)
        self.phrase_window = int(phrase_window)
        self.beta = float(beta)
        self.d_mag_max = float(d_mag_max)
        self.d_norm_eps = float(d_norm_eps)
        self.residual_init_scale = float(residual_init_scale)
        self.scale = evidence_dim ** -0.5
        self.evidence_dim = evidence_dim
        self.enable_diagnostics = False

        self.lang_fuse = nn.Linear(dim * 2, dim) if lang_source == "fuse" else None

        self.phrase_pool = nn.Sequential(
            nn.Linear(dim * 3, dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim, bias=False),
        )
        self.search_proj = nn.Linear(dim, evidence_dim)
        self.phrase_proj = nn.Linear(dim, evidence_dim)
        self.target_to_search = nn.Linear(dim, evidence_dim)
        self.target_to_phrase = nn.Linear(dim, evidence_dim)

        self.search_norm = nn.LayerNorm(evidence_dim)
        self.phrase_norm = nn.LayerNorm(evidence_dim)

        self.query_proj = nn.Linear(evidence_dim, evidence_dim, bias=False)
        self.key_proj = nn.Linear(evidence_dim, evidence_dim, bias=False)
        self.value_proj = nn.Linear(evidence_dim, evidence_dim, bias=False)

        self.d_direction_norm = nn.LayerNorm(evidence_dim)
        self.interaction_norm = nn.LayerNorm(evidence_dim * 2)
        self.interaction_mlp = nn.Sequential(
            nn.Linear(evidence_dim * 2, evidence_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(evidence_dim, evidence_dim, bias=False),
        )
        self.evidence_norm = nn.LayerNorm(evidence_dim)
        self.evidence_mlp = nn.Sequential(
            nn.Linear(evidence_dim, evidence_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(evidence_dim, evidence_dim, bias=False),
        )

        self.strength_head = nn.Linear(evidence_dim, 1)
        self.out_norm = nn.LayerNorm(evidence_dim)
        self.out_linear = nn.Linear(evidence_dim, dim, bias=False)
        self.out_dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        self._init_strength_head()
        self._init_residual_path()

    def forward(
            self,
            h_x,
            l_raw,
            h_z,
            h_l=None,
            l_mask=None,
            evidence_anchor_mask=None,
            template_token_len=None):
        if l_raw is None or h_z is None:
            return h_x, self._empty_aux(h_x)

        l_src = self._select_language_source(l_raw, h_l)
        z_proto = self._target_pool(h_z, template_token_len)
        l_mask, anchor_mask, valid_token_count, raw_evidence_unit_count = self._sanitize_masks(
            l_mask,
            evidence_anchor_mask,
            l_src,
        )

        phrase_units, phrase_context_count = self._phrase_pool(l_src, l_mask)
        evidence_unit_count = anchor_mask.sum(dim=1)
        evidence_weight, evidence_availability, context_quality = self._evidence_availability(
            anchor_mask,
            phrase_context_count,
            l_src,
        )

        e_x = self.search_proj(h_x) + self.target_to_search(z_proto).unsqueeze(1)
        e_p = self.phrase_proj(phrase_units) + self.target_to_phrase(z_proto).unsqueeze(1)
        e_x = self.search_norm(e_x)
        e_p = self.phrase_norm(e_p)

        query = self.query_proj(e_x)
        key = self.key_proj(e_p)
        value = self.value_proj(e_p)

        attn_logits = torch.matmul(query, key.transpose(1, 2)) * self.scale
        evidence_attn = self._masked_softmax(attn_logits, anchor_mask)

        # Center target-conditioned values before readout. This is equivalent
        # to M_i - M_0 with exactly shared target conditioning, and it removes
        # any constant target bias from the evidence residual path.
        value_mean = self._masked_mean(value, anchor_mask).unsqueeze(1)
        value_centered = (value - value_mean) * anchor_mask.to(
            device=value.device, dtype=value.dtype).unsqueeze(-1)
        d_raw = torch.matmul(evidence_attn, value_centered)
        d_raw_norm = d_raw.norm(dim=-1, keepdim=True) / math.sqrt(float(self.evidence_dim))
        d_gate = d_raw_norm / (d_raw_norm + self.d_norm_eps)
        d_mag = d_raw_norm.clamp(max=self.d_mag_max)
        d_dir, _ = self._safe_layer_norm(self.d_direction_norm, d_raw)
        d_i = d_mag * d_dir

        interaction = torch.cat([d_i, e_x * d_i], dim=-1)
        interaction_input, _ = self._safe_layer_norm(self.interaction_norm, interaction)
        g_i = self.interaction_mlp(interaction_input) + d_i
        evidence_input, _ = self._safe_layer_norm(self.evidence_norm, g_i)
        c_i = self.evidence_mlp(evidence_input) + g_i

        strength_logits_raw = self.strength_head(c_i)
        strength_logits = strength_logits_raw - strength_logits_raw.mean(dim=1, keepdim=True)
        r_delta = torch.tanh(strength_logits)
        r_delta = r_delta - r_delta.mean(dim=1, keepdim=True)
        r_delta = r_delta.clamp(min=-1.0, max=1.0)
        r_i = self.beta * evidence_availability * r_delta

        delta_raw = self.out_dropout(self.out_linear(
            self._safe_layer_norm(self.out_norm, c_i)[0]))
        delta_direction, delta_direction_gate, delta_direction_raw_norm = self._safe_unit_direction(delta_raw)
        delta_after_strength = r_i * delta_direction
        gamma = torch.tanh(self.gamma)
        scaled_delta = gamma * delta_after_strength
        h_x_calibrated = h_x + scaled_delta

        aux = self._build_aux(
            h_x=h_x,
            d_raw=d_raw,
            d_gate=d_gate,
            d_i=d_i,
            g_i=g_i,
            c_i=c_i,
            strength_logits_raw=strength_logits_raw,
            strength_logits=strength_logits,
            r_i=r_i,
            evidence_availability=evidence_availability,
            delta=delta_direction,
            delta_direction_raw=delta_raw,
            delta_direction_gate=delta_direction_gate,
            delta_direction_raw_norm=delta_direction_raw_norm,
            delta_after_strength=delta_after_strength,
            scaled_delta=scaled_delta,
            gamma=gamma,
            evidence_attn=evidence_attn,
            valid_token_count=valid_token_count,
            raw_evidence_unit_count=raw_evidence_unit_count,
            evidence_unit_count=evidence_unit_count,
            phrase_context_count=phrase_context_count,
            evidence_weight=evidence_weight,
            context_quality=context_quality,
            z_proto=z_proto,
        )
        if self.enable_diagnostics:
            aux.update({
                "stage2r_diag_evidence_scalar": (
                    d_i.detach().norm(dim=-1)
                    * g_i.detach().norm(dim=-1)
                    * c_i.detach().norm(dim=-1)
                ),
                "stage2r_diag_calibration_scalar": (
                    scaled_delta.detach().norm(dim=-1)
                ),
                "stage2r_diag_strength": r_i.detach().squeeze(-1),
                "stage2r_diag_attention": evidence_attn.detach(),
            })
        return h_x_calibrated, aux

    def _select_language_source(self, l_raw, h_l):
        if self.lang_fuse is not None and h_l is not None and h_l.shape[:2] == l_raw.shape[:2]:
            return self.lang_fuse(torch.cat([l_raw, h_l], dim=-1))
        return l_raw

    def _phrase_pool(self, l_src, l_mask):
        l_mask = l_mask.to(device=l_src.device, dtype=l_src.dtype)
        masked_l = l_src * l_mask.unsqueeze(-1)
        pad = self.phrase_window // 2
        batch_size, token_len, channels = masked_l.shape
        zeros_l = masked_l.new_zeros(batch_size, pad, channels)
        zeros_m = l_mask.new_zeros(batch_size, pad)
        padded_l = torch.cat([zeros_l, masked_l, zeros_l], dim=1)
        padded_m = torch.cat([zeros_m, l_mask, zeros_m], dim=1)

        context_sum = masked_l.new_zeros(batch_size, token_len, channels)
        context_count = masked_l.new_zeros(batch_size, token_len, 1)
        for offset in range(self.phrase_window):
            if offset == pad:
                continue
            piece = padded_l[:, offset:offset + token_len]
            piece_mask = padded_m[:, offset:offset + token_len].unsqueeze(-1)
            context_sum = context_sum + piece * piece_mask
            context_count = context_count + piece_mask

        context_mean = context_sum / context_count.clamp_min(1.0)
        anchor_orthogonal = self._orthogonal_residual(masked_l, context_mean)
        anchor_context_interaction = anchor_orthogonal * context_mean

        # Anchor identity is not provided as a direct path. The anchor appears
        # through the component not explained by local context, while context
        # remains available for relation words and attributes.
        phrase_input = torch.cat([
            anchor_orthogonal,
            context_mean,
            anchor_context_interaction,
        ], dim=-1)
        context_available = (context_count > 0).to(dtype=l_src.dtype)
        phrase_units = self.phrase_pool(phrase_input) * l_mask.unsqueeze(-1)
        return phrase_units, context_count.squeeze(-1)

    def _evidence_availability(self, anchor_mask, phrase_context_count, l_src):
        anchor_float = anchor_mask.to(device=l_src.device, dtype=l_src.dtype)
        max_context = max(1.0, float(self.phrase_window - 1))
        context_quality = (phrase_context_count.to(device=l_src.device, dtype=l_src.dtype) / max_context).clamp(0.0, 1.0)

        # Weak unigram evidence is allowed, but it is deliberately capped far
        # below phrase evidence so category-only labels cannot dominate.
        unigram_floor = 0.25
        unit_weight = anchor_float * (unigram_floor + (1.0 - unigram_floor) * context_quality)
        weighted_evidence_count = unit_weight.sum(dim=1, keepdim=True)
        availability = (
            weighted_evidence_count / float(max(1, self.min_evidence_units))
        ).clamp(0.0, 1.0).view(-1, 1, 1)
        return unit_weight, availability, context_quality

    def _orthogonal_residual(self, anchor, context):
        context_norm_sq = context.square().sum(dim=-1, keepdim=True)
        projection_scale = (anchor * context).sum(dim=-1, keepdim=True) / context_norm_sq.clamp_min(
            self.d_norm_eps ** 2)
        projection = projection_scale * context
        has_context = (context_norm_sq > self.d_norm_eps ** 2).to(dtype=anchor.dtype)
        return anchor - projection * has_context

    def _target_pool(self, h_z, template_token_len):
        if self.target_pool != "center" or template_token_len is None:
            return h_z.mean(dim=1)

        template_token_len = int(template_token_len)
        side = int(math.sqrt(template_token_len))
        if side * side != template_token_len or h_z.shape[1] % template_token_len != 0:
            return h_z.mean(dim=1)

        batch_size, _, channels = h_z.shape
        num_templates = h_z.shape[1] // template_token_len
        h_z = h_z.view(batch_size, num_templates, side, side, channels)

        span = max(1, min(side, int(round(side * self.center_ratio))))
        start = (side - span) // 2
        h_z = h_z[:, :, start:start + span, start:start + span, :]
        return h_z.mean(dim=(1, 2, 3))

    @staticmethod
    def _sanitize_base_mask(mask, l_src):
        if mask is None:
            return torch.ones(l_src.shape[:2], device=l_src.device, dtype=torch.bool)

        mask = mask.to(device=l_src.device, dtype=torch.bool)
        if mask.shape[:2] != l_src.shape[:2]:
            return torch.ones(l_src.shape[:2], device=l_src.device, dtype=torch.bool)

        empty_rows = ~mask.any(dim=1)
        if empty_rows.any():
            mask = mask.clone()
            mask[empty_rows] = True
        return mask

    def _sanitize_masks(self, l_mask, evidence_anchor_mask, l_src):
        l_mask = self._sanitize_base_mask(l_mask, l_src)
        valid_token_count = l_mask.sum(dim=1)

        if evidence_anchor_mask is None:
            anchor_mask = l_mask
        else:
            anchor_mask = evidence_anchor_mask.to(device=l_src.device, dtype=torch.bool)
            if anchor_mask.shape[:2] != l_src.shape[:2]:
                anchor_mask = l_mask
            else:
                anchor_mask = anchor_mask & l_mask

        evidence_unit_count = anchor_mask.sum(dim=1)
        return l_mask, anchor_mask, valid_token_count, evidence_unit_count

    @staticmethod
    def _masked_softmax(logits, mask):
        mask = mask.to(device=logits.device, dtype=torch.bool)
        expanded_mask = mask.unsqueeze(1)
        logits = logits.masked_fill(~expanded_mask, torch.finfo(logits.dtype).min)
        probs = logits.softmax(dim=-1) * expanded_mask.to(dtype=logits.dtype)
        denom = probs.sum(dim=-1, keepdim=True)
        return torch.where(denom > 0, probs / denom.clamp_min(1e-6), torch.zeros_like(probs))

    @staticmethod
    def _masked_mean(values, mask):
        mask = mask.to(device=values.device, dtype=values.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (values * mask.unsqueeze(-1)).sum(dim=1) / denom

    def _safe_layer_norm(self, norm_layer, x):
        x_rms = x.norm(dim=-1, keepdim=True) / math.sqrt(float(x.shape[-1]))
        gate = x_rms / (x_rms + self.d_norm_eps)
        return gate * norm_layer(x), gate

    def _safe_unit_direction(self, x):
        raw_norm = x.norm(dim=-1, keepdim=True)
        raw_rms = raw_norm / math.sqrt(float(x.shape[-1]))
        denom = raw_norm + self.d_norm_eps * math.sqrt(float(x.shape[-1]))
        gate = raw_rms / (raw_rms + self.d_norm_eps)
        return x / denom, gate, raw_rms

    @staticmethod
    def _mean_norm(x):
        return x.detach().norm(dim=-1).mean()

    @staticmethod
    def _norm_std(x):
        return x.detach().norm(dim=-1).std(unbiased=False)

    @staticmethod
    def _spatial_norm_std(x):
        if x.shape[1] <= 1:
            return x.new_zeros(())
        return x.detach().norm(dim=-1).std(dim=1, unbiased=False).mean()

    @staticmethod
    def _entropy(attn):
        attn = attn.clamp_min(1e-6)
        return -(attn * attn.log()).sum(dim=-1)

    def _normalized_entropy(self, attn, evidence_unit_count):
        entropy = self._entropy(attn)
        denom = evidence_unit_count.detach().float().clamp_min(2.0).log()
        return entropy / denom.view(-1, 1).clamp_min(1e-6)

    def _build_aux(
            self,
            h_x,
            d_raw,
            d_gate,
            d_i,
            g_i,
            c_i,
            strength_logits_raw,
            strength_logits,
            r_i,
            evidence_availability,
            delta,
            delta_direction_raw,
            delta_direction_gate,
            delta_direction_raw_norm,
            delta_after_strength,
            scaled_delta,
            gamma,
            evidence_attn,
            valid_token_count,
            raw_evidence_unit_count,
            evidence_unit_count,
            phrase_context_count,
            evidence_weight,
            context_quality,
            z_proto):
        h_x_norm = self._mean_norm(h_x)
        r_detached = r_i.detach()
        legacy_s = 1.0 + r_detached
        attn_entropy = self._entropy(evidence_attn).detach().mean()
        attn_entropy_norm = self._normalized_entropy(evidence_attn, evidence_unit_count).detach().mean()
        top1_weight = evidence_attn.detach().max(dim=-1).values.mean()

        return {
            "stage2r_raw_gamma": self.gamma.detach().reshape(1),
            "stage2r_tanh_gamma": gamma.detach().reshape(1),
            "stage2r_beta": h_x.new_tensor([self.beta]),
            "stage2r_phrase_window": h_x.new_tensor([float(self.phrase_window)]),
            "stage2r_d_norm_eps": h_x.new_tensor([self.d_norm_eps]),
            "stage2r_residual_init_scale": h_x.new_tensor([self.residual_init_scale]),
            "stage2r_D_raw_norm_mean": self._mean_norm(d_raw).reshape(1),
            "stage2r_D_raw_norm_std": self._norm_std(d_raw).reshape(1),
            "stage2r_D_raw_spatial_std": self._spatial_norm_std(d_raw).reshape(1),
            "stage2r_D_gate_mean": d_gate.detach().mean().reshape(1),
            "stage2r_D_gate_std": d_gate.detach().std(unbiased=False).reshape(1),
            "stage2r_D_norm_mean": self._mean_norm(d_i).reshape(1),
            "stage2r_D_norm_std": self._norm_std(d_i).reshape(1),
            "stage2r_D_spatial_std": self._spatial_norm_std(d_i).reshape(1),
            "stage2r_G_norm_mean": self._mean_norm(g_i).reshape(1),
            "stage2r_G_norm_std": self._norm_std(g_i).reshape(1),
            "stage2r_G_spatial_std": self._spatial_norm_std(g_i).reshape(1),
            "stage2r_C_norm_mean": self._mean_norm(c_i).reshape(1),
            "stage2r_C_norm_std": self._norm_std(c_i).reshape(1),
            "stage2r_C_spatial_std": self._spatial_norm_std(c_i).reshape(1),
            "stage2r_evidence_availability_mean": evidence_availability.detach().mean().reshape(1),
            "stage2r_evidence_availability_std": evidence_availability.detach().std(unbiased=False).reshape(1),
            "stage2r_evidence_availability_min": evidence_availability.detach().min().reshape(1),
            "stage2r_evidence_availability_max": evidence_availability.detach().max().reshape(1),
            "stage2r_language_reliability_mean": evidence_availability.detach().mean().reshape(1),
            "stage2r_language_reliability_std": evidence_availability.detach().std(unbiased=False).reshape(1),
            "stage2r_language_reliability_min": evidence_availability.detach().min().reshape(1),
            "stage2r_language_reliability_max": evidence_availability.detach().max().reshape(1),
            "stage2r_u_raw_mean": strength_logits_raw.detach().mean().reshape(1),
            "stage2r_u_raw_std": strength_logits_raw.detach().std(unbiased=False).reshape(1),
            "stage2r_u_mean": strength_logits.detach().mean().reshape(1),
            "stage2r_u_std": strength_logits.detach().std(unbiased=False).reshape(1),
            "stage2r_r_mean": r_detached.mean().reshape(1),
            "stage2r_r_std": r_detached.std(unbiased=False).reshape(1),
            "stage2r_r_min": r_detached.min().reshape(1),
            "stage2r_r_max": r_detached.max().reshape(1),
            "stage2r_r_abs_mean": r_detached.abs().mean().reshape(1),
            "stage2r_s_mean": legacy_s.mean().reshape(1),
            "stage2r_s_std": legacy_s.std(unbiased=False).reshape(1),
            "stage2r_s_min": legacy_s.min().reshape(1),
            "stage2r_s_max": legacy_s.max().reshape(1),
            "stage2r_s_deviation_mean": r_detached.abs().mean().reshape(1),
            "stage2r_delta_direction_raw_norm_mean": delta_direction_raw_norm.detach().mean().reshape(1),
            "stage2r_delta_direction_raw_norm_std": delta_direction_raw_norm.detach().std(unbiased=False).reshape(1),
            "stage2r_delta_direction_gate_mean": delta_direction_gate.detach().mean().reshape(1),
            "stage2r_delta_direction_gate_std": delta_direction_gate.detach().std(unbiased=False).reshape(1),
            "stage2r_delta_direction_norm_mean": self._mean_norm(delta).reshape(1),
            "stage2r_delta_direction_norm_max": delta.detach().norm(dim=-1).max().reshape(1),
            "stage2r_delta_norm_before_strength": self._mean_norm(delta).reshape(1),
            "stage2r_delta_norm_after_strength": self._mean_norm(delta_after_strength).reshape(1),
            "stage2r_delta_norm_after_r": self._mean_norm(delta_after_strength).reshape(1),
            "stage2r_delta_norm_after_gamma": self._mean_norm(scaled_delta).reshape(1),
            "stage2r_delta_to_feature_ratio": (
                self._mean_norm(scaled_delta) / h_x_norm.clamp_min(1e-6)
            ).reshape(1),
            "stage2r_attention_entropy": attn_entropy.reshape(1),
            "stage2r_attention_entropy_norm": attn_entropy_norm.reshape(1),
            "stage2r_top1_evidence_weight": top1_weight.reshape(1),
            "stage2r_valid_token_count": valid_token_count.detach().float().mean().reshape(1),
            "stage2r_raw_anchor_token_count": raw_evidence_unit_count.detach().float().mean().reshape(1),
            "stage2r_anchor_token_count": evidence_unit_count.detach().float().mean().reshape(1),
            "stage2r_weighted_evidence_count": evidence_weight.detach().sum(dim=1).mean().reshape(1),
            "stage2r_anchor_context_quality_mean": (
                context_quality.detach() * (evidence_weight.detach() > 0).float()
            ).sum(dim=1).div(evidence_unit_count.detach().float().clamp_min(1.0)).mean().reshape(1),
            "stage2r_context_token_count": valid_token_count.detach().float().mean().reshape(1),
            "stage2r_phrase_context_count": phrase_context_count.detach().float().mean().reshape(1),
            "stage2r_phrase_has_context_ratio": (
                phrase_context_count.detach() > 0
            ).float().mean().reshape(1),
            "stage2r_evidence_unit_count": evidence_unit_count.detach().float().mean().reshape(1),
            "stage2r_no_raw_anchor_ratio": (
                raw_evidence_unit_count.detach() == 0
            ).float().mean().reshape(1),
            "stage2r_single_raw_anchor_ratio": (
                raw_evidence_unit_count.detach() == 1
            ).float().mean().reshape(1),
            "stage2r_no_effective_evidence_ratio": (
                evidence_unit_count.detach() == 0
            ).float().mean().reshape(1),
            "stage2r_single_effective_evidence_ratio": (
                evidence_unit_count.detach() == 1
            ).float().mean().reshape(1),
            "stage2r_low_evidence_unit_ratio": (
                evidence_unit_count.detach() < self.min_evidence_units
            ).float().mean().reshape(1),
            "stage2r_low_availability_ratio": (
                evidence_availability.detach().view(-1) < 0.5
            ).float().mean().reshape(1),
            "stage2r_z_proto_norm": z_proto.detach().norm(dim=-1).mean().reshape(1),
            "stage2r_z_proto_std": z_proto.detach().std(dim=-1).mean().reshape(1),
        }

    def _init_strength_head(self):
        nn.init.normal_(self.strength_head.weight, std=self.residual_init_scale)
        nn.init.zeros_(self.strength_head.bias)

    def _init_residual_path(self):
        nn.init.normal_(self.out_linear.weight, std=self.residual_init_scale)

    @staticmethod
    def _empty_aux(h_x):
        zero = h_x.new_zeros(1)
        keys = [
            "stage2r_raw_gamma",
            "stage2r_tanh_gamma",
            "stage2r_beta",
            "stage2r_phrase_window",
            "stage2r_d_norm_eps",
            "stage2r_residual_init_scale",
            "stage2r_D_raw_norm_mean",
            "stage2r_D_raw_norm_std",
            "stage2r_D_raw_spatial_std",
            "stage2r_D_gate_mean",
            "stage2r_D_gate_std",
            "stage2r_D_norm_mean",
            "stage2r_D_norm_std",
            "stage2r_D_spatial_std",
            "stage2r_G_norm_mean",
            "stage2r_G_norm_std",
            "stage2r_G_spatial_std",
            "stage2r_C_norm_mean",
            "stage2r_C_norm_std",
            "stage2r_C_spatial_std",
            "stage2r_evidence_availability_mean",
            "stage2r_evidence_availability_std",
            "stage2r_evidence_availability_min",
            "stage2r_evidence_availability_max",
            "stage2r_language_reliability_mean",
            "stage2r_language_reliability_std",
            "stage2r_language_reliability_min",
            "stage2r_language_reliability_max",
            "stage2r_u_raw_mean",
            "stage2r_u_raw_std",
            "stage2r_u_mean",
            "stage2r_u_std",
            "stage2r_r_mean",
            "stage2r_r_std",
            "stage2r_r_min",
            "stage2r_r_max",
            "stage2r_r_abs_mean",
            "stage2r_s_mean",
            "stage2r_s_std",
            "stage2r_s_min",
            "stage2r_s_max",
            "stage2r_s_deviation_mean",
            "stage2r_delta_direction_raw_norm_mean",
            "stage2r_delta_direction_raw_norm_std",
            "stage2r_delta_direction_gate_mean",
            "stage2r_delta_direction_gate_std",
            "stage2r_delta_direction_norm_mean",
            "stage2r_delta_direction_norm_max",
            "stage2r_delta_norm_before_strength",
            "stage2r_delta_norm_after_strength",
            "stage2r_delta_norm_after_r",
            "stage2r_delta_norm_after_gamma",
            "stage2r_delta_to_feature_ratio",
            "stage2r_attention_entropy",
            "stage2r_attention_entropy_norm",
            "stage2r_top1_evidence_weight",
            "stage2r_valid_token_count",
            "stage2r_raw_anchor_token_count",
            "stage2r_anchor_token_count",
            "stage2r_weighted_evidence_count",
            "stage2r_anchor_context_quality_mean",
            "stage2r_context_token_count",
            "stage2r_phrase_context_count",
            "stage2r_phrase_has_context_ratio",
            "stage2r_evidence_unit_count",
            "stage2r_no_raw_anchor_ratio",
            "stage2r_single_raw_anchor_ratio",
            "stage2r_no_effective_evidence_ratio",
            "stage2r_single_effective_evidence_ratio",
            "stage2r_low_evidence_unit_ratio",
            "stage2r_low_availability_ratio",
            "stage2r_z_proto_norm",
            "stage2r_z_proto_std",
        ]
        return {key: zero for key in keys}
