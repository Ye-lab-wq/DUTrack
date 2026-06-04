import math

import torch
from torch import nn


class TrackingEvidenceLayer(nn.Module):
    """Explicit region-level evidence layer before the tracking head.

    This is separate from the Stage-1 TEC residual adapter. It builds a
    language-conditioned evidence representation and applies only a bounded
    head-pre residual. It never writes directly to the score map.
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
            min_valid_tokens=3,
            num_evidence_slots=4,
            attention_uniform_mix=0.05,
            dropout=0.0):
        super().__init__()
        if lang_source not in ("raw", "fuse"):
            raise ValueError("Evidence layer lang_source must be 'raw' or 'fuse'.")
        if target_pool not in ("center", "mean"):
            raise ValueError("Evidence layer target_pool must be 'center' or 'mean'.")
        if beta <= 0.0 or beta >= 1.0:
            raise ValueError("Evidence layer beta should be in (0, 1).")
        if d_norm_eps <= 0.0:
            raise ValueError("Evidence layer d_norm_eps must be positive.")
        if residual_init_scale <= 0.0:
            raise ValueError("Evidence layer residual_init_scale must be positive.")
        if num_evidence_slots < 1:
            raise ValueError("Evidence layer num_evidence_slots must be >= 1.")
        if attention_uniform_mix < 0.0 or attention_uniform_mix >= 1.0:
            raise ValueError("Evidence layer attention_uniform_mix should be in [0, 1).")

        self.lang_source = lang_source
        self.target_pool = target_pool
        self.center_ratio = center_ratio
        self.min_valid_tokens = min_valid_tokens
        self.beta = float(beta)
        self.d_mag_max = float(d_mag_max)
        self.d_norm_eps = float(d_norm_eps)
        self.residual_init_scale = float(residual_init_scale)
        self.scale = evidence_dim ** -0.5
        self.evidence_dim = evidence_dim
        self.num_evidence_slots = int(num_evidence_slots)
        self.attention_uniform_mix = float(attention_uniform_mix)
        self.enable_diagnostics = False

        self.lang_fuse = nn.Linear(dim * 2, dim) if lang_source == "fuse" else None

        self.search_proj = nn.Linear(dim, evidence_dim)
        self.lang_proj = nn.Linear(dim, evidence_dim)
        self.target_to_search = nn.Linear(dim, evidence_dim)
        self.target_to_lang = nn.Linear(dim, evidence_dim)

        self.search_norm = nn.LayerNorm(evidence_dim)
        self.lang_norm = nn.LayerNorm(evidence_dim)

        self.query_proj = nn.Linear(evidence_dim, evidence_dim, bias=False)
        self.key_proj = nn.Linear(evidence_dim, evidence_dim, bias=False)
        self.value_proj = nn.Linear(evidence_dim, evidence_dim, bias=False)
        self.evidence_slot_embed = nn.Parameter(torch.zeros(self.num_evidence_slots, evidence_dim))
        self.slot_fuse = nn.Linear(evidence_dim * self.num_evidence_slots, evidence_dim, bias=False)

        self.d_direction_norm = nn.LayerNorm(evidence_dim)
        self.interaction_norm = nn.LayerNorm(evidence_dim * 2)
        self.interaction_mlp = nn.Sequential(
            nn.Linear(evidence_dim * 2, evidence_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(evidence_dim, evidence_dim, bias=False),
        )
        self.evidence_mlp = nn.Sequential(
            nn.LayerNorm(evidence_dim),
            nn.Linear(evidence_dim, evidence_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(evidence_dim, evidence_dim, bias=False),
        )

        self.strength_head = nn.Linear(evidence_dim, 1)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(evidence_dim),
            nn.Linear(evidence_dim, dim, bias=False),
            nn.Dropout(dropout),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        self._init_strength_head()
        self._init_evidence_slots()
        self._init_residual_path()

    def forward(self, h_x, l_raw, h_z, h_l=None, l_mask=None, semantic_l_mask=None, template_token_len=None):
        if l_raw is None or h_z is None:
            return h_x, self._empty_aux(h_x)

        l_src = self._select_language_source(l_raw, h_l)
        z_proto = self._target_pool(h_z, template_token_len)
        l_mask, semantic_l_mask, valid_token_count, semantic_token_count = self._sanitize_language_masks(
            l_mask,
            semantic_l_mask,
            l_src,
        )

        e_x = self.search_proj(h_x) + self.target_to_search(z_proto).unsqueeze(1)
        e_l = self.lang_proj(l_src) + self.target_to_lang(z_proto).unsqueeze(1)
        e_x = self.search_norm(e_x)
        e_l = self.lang_norm(e_l)

        query = self.query_proj(e_x)
        key = self.key_proj(e_l)
        value = self.value_proj(e_l)

        slot_query = query.unsqueeze(2) + self.evidence_slot_embed.view(
            1,
            1,
            self.num_evidence_slots,
            self.evidence_dim,
        )
        attn_logits = torch.einsum("bnsd,bld->bnsl", slot_query, key) * self.scale
        evidence_attn = self._masked_softmax(attn_logits, semantic_l_mask)
        evidence_attn = self._mix_with_uniform_attention(evidence_attn, semantic_l_mask)
        slot_evidence = torch.einsum("bnsl,bld->bnsd", evidence_attn, value)
        m_i = self.slot_fuse(slot_evidence.reshape(h_x.shape[0], h_x.shape[1], -1))
        m_0 = self._masked_mean(value, semantic_l_mask).unsqueeze(1)
        diagnostic_attn = evidence_attn.mean(dim=2)

        d_raw = m_i - m_0
        d_raw_norm = d_raw.norm(dim=-1, keepdim=True) / math.sqrt(float(self.evidence_dim))
        d_gate = d_raw_norm / (d_raw_norm + self.d_norm_eps)
        d_mag = d_raw_norm.clamp(max=self.d_mag_max)
        d_dir = self.d_direction_norm(d_raw)
        d_i = d_gate * d_mag * d_dir

        interaction = torch.cat([d_i, e_x * d_i], dim=-1)
        g_i = self.interaction_mlp(self.interaction_norm(interaction)) + d_i
        c_i = self.evidence_mlp(g_i)

        strength_logits_raw = self.strength_head(c_i)
        strength_logits = strength_logits_raw - strength_logits_raw.mean(dim=1, keepdim=True)
        strength_delta = torch.tanh(strength_logits)
        strength_delta = strength_delta - strength_delta.mean(dim=1, keepdim=True)
        strength_delta = strength_delta.clamp(min=-1.0, max=1.0)
        strength = 1.0 + self.beta * strength_delta
        delta = self.out_proj(c_i)
        delta_after_strength = strength * delta
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
            strength=strength,
            delta=delta,
            delta_after_strength=delta_after_strength,
            scaled_delta=scaled_delta,
            gamma=gamma,
            evidence_attn=diagnostic_attn,
            valid_token_count=valid_token_count,
            semantic_token_count=semantic_token_count,
            z_proto=z_proto,
        )
        if self.enable_diagnostics:
            aux.update({
                "stage2_diag_evidence_scalar": (
                    d_i.detach().norm(dim=-1)
                    * g_i.detach().norm(dim=-1)
                    * c_i.detach().norm(dim=-1)
                ),
                "stage2_diag_calibration_scalar": (
                    (strength.detach().squeeze(-1) - 1.0).abs()
                    * scaled_delta.detach().norm(dim=-1)
                ),
                "stage2_diag_strength": strength.detach().squeeze(-1),
                "stage2_diag_attention": diagnostic_attn.detach(),
            })
        return h_x_calibrated, aux

    def _select_language_source(self, l_raw, h_l):
        if self.lang_fuse is not None and h_l is not None and h_l.shape[:2] == l_raw.shape[:2]:
            return self.lang_fuse(torch.cat([l_raw, h_l], dim=-1))
        return l_raw

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
    def _sanitize_base_language_mask(l_mask, l_src):
        if l_mask is None:
            l_mask = torch.ones(l_src.shape[:2], device=l_src.device, dtype=torch.bool)
            return l_mask

        l_mask = l_mask.to(device=l_src.device, dtype=torch.bool)
        if l_mask.shape[:2] != l_src.shape[:2]:
            l_mask = torch.ones(l_src.shape[:2], device=l_src.device, dtype=torch.bool)
            return l_mask

        empty_rows = ~l_mask.any(dim=1)
        if empty_rows.any():
            l_mask = l_mask.clone()
            l_mask[empty_rows] = True
        return l_mask

    def _sanitize_language_masks(self, l_mask, semantic_l_mask, l_src):
        l_mask = self._sanitize_base_language_mask(l_mask, l_src)
        valid_token_count = l_mask.sum(dim=1)

        if semantic_l_mask is None:
            semantic_l_mask = l_mask
        else:
            semantic_l_mask = semantic_l_mask.to(device=l_src.device, dtype=torch.bool)
            if semantic_l_mask.shape[:2] != l_src.shape[:2]:
                semantic_l_mask = l_mask
            else:
                semantic_l_mask = semantic_l_mask & l_mask

        empty_rows = ~semantic_l_mask.any(dim=1)
        if empty_rows.any():
            semantic_l_mask = semantic_l_mask.clone()
            semantic_l_mask[empty_rows] = l_mask[empty_rows]

        semantic_token_count = semantic_l_mask.sum(dim=1)
        return l_mask, semantic_l_mask, valid_token_count, semantic_token_count

    @staticmethod
    def _masked_softmax(logits, mask):
        mask = mask.to(device=logits.device, dtype=torch.bool)
        while mask.dim() < logits.dim():
            mask = mask.unsqueeze(1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return logits.softmax(dim=-1)

    def _mix_with_uniform_attention(self, attn, mask):
        if self.attention_uniform_mix <= 0.0:
            return attn

        mask = mask.to(device=attn.device, dtype=attn.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        uniform = mask / denom
        while uniform.dim() < attn.dim():
            uniform = uniform.unsqueeze(1)
        return (1.0 - self.attention_uniform_mix) * attn + self.attention_uniform_mix * uniform

    @staticmethod
    def _masked_mean(values, mask):
        mask = mask.to(device=values.device, dtype=values.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (values * mask.unsqueeze(-1)).sum(dim=1) / denom

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

    def _normalized_entropy(self, attn, valid_token_count):
        entropy = self._entropy(attn)
        denom = valid_token_count.detach().float().clamp_min(2.0).log()
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
            strength,
            delta,
            delta_after_strength,
            scaled_delta,
            gamma,
            evidence_attn,
            valid_token_count,
            semantic_token_count,
            z_proto):
        h_x_norm = self._mean_norm(h_x)
        strength_detached = strength.detach()
        attn_entropy = self._entropy(evidence_attn).detach().mean()
        attn_entropy_norm = self._normalized_entropy(evidence_attn, semantic_token_count).detach().mean()

        return {
            "stage2_raw_gamma": self.gamma.detach().reshape(1),
            "stage2_tanh_gamma": gamma.detach().reshape(1),
            "stage2_beta": h_x.new_tensor([self.beta]),
            "stage2_num_evidence_slots": h_x.new_tensor([float(self.num_evidence_slots)]),
            "stage2_attention_uniform_mix": h_x.new_tensor([self.attention_uniform_mix]),
            "stage2_d_norm_eps": h_x.new_tensor([self.d_norm_eps]),
            "stage2_residual_init_scale": h_x.new_tensor([self.residual_init_scale]),
            "stage2_D_raw_norm_mean": self._mean_norm(d_raw).reshape(1),
            "stage2_D_raw_norm_std": self._norm_std(d_raw).reshape(1),
            "stage2_D_raw_spatial_std": self._spatial_norm_std(d_raw).reshape(1),
            "stage2_D_gate_mean": d_gate.detach().mean().reshape(1),
            "stage2_D_gate_std": d_gate.detach().std(unbiased=False).reshape(1),
            "stage2_D_norm_mean": self._mean_norm(d_i).reshape(1),
            "stage2_D_norm_std": self._norm_std(d_i).reshape(1),
            "stage2_D_spatial_std": self._spatial_norm_std(d_i).reshape(1),
            "stage2_G_norm_mean": self._mean_norm(g_i).reshape(1),
            "stage2_G_norm_std": self._norm_std(g_i).reshape(1),
            "stage2_G_spatial_std": self._spatial_norm_std(g_i).reshape(1),
            "stage2_C_norm_mean": self._mean_norm(c_i).reshape(1),
            "stage2_C_norm_std": self._norm_std(c_i).reshape(1),
            "stage2_C_spatial_std": self._spatial_norm_std(c_i).reshape(1),
            "stage2_u_raw_mean": strength_logits_raw.detach().mean().reshape(1),
            "stage2_u_raw_std": strength_logits_raw.detach().std(unbiased=False).reshape(1),
            "stage2_u_mean": strength_logits.detach().mean().reshape(1),
            "stage2_u_std": strength_logits.detach().std(unbiased=False).reshape(1),
            "stage2_s_mean": strength_detached.mean().reshape(1),
            "stage2_s_std": strength_detached.std(unbiased=False).reshape(1),
            "stage2_s_min": strength_detached.min().reshape(1),
            "stage2_s_max": strength_detached.max().reshape(1),
            "stage2_s_deviation_mean": (strength_detached - 1.0).abs().mean().reshape(1),
            "stage2_delta_norm_before_strength": self._mean_norm(delta).reshape(1),
            "stage2_delta_norm_after_strength": self._mean_norm(delta_after_strength).reshape(1),
            "stage2_delta_norm_after_gamma": self._mean_norm(scaled_delta).reshape(1),
            "stage2_delta_to_feature_ratio": (
                self._mean_norm(scaled_delta) / h_x_norm.clamp_min(1e-6)
            ).reshape(1),
            "stage2_attention_entropy": attn_entropy.reshape(1),
            "stage2_attention_entropy_norm": attn_entropy_norm.reshape(1),
            "stage2_valid_token_count": valid_token_count.detach().float().mean().reshape(1),
            "stage2_valid_semantic_token_count": semantic_token_count.detach().float().mean().reshape(1),
            "stage2_low_valid_token_ratio": (
                semantic_token_count.detach() < self.min_valid_tokens
            ).float().mean().reshape(1),
            "stage2_z_proto_norm": z_proto.detach().norm(dim=-1).mean().reshape(1),
            "stage2_z_proto_std": z_proto.detach().std(dim=-1).mean().reshape(1),
        }

    def _init_strength_head(self):
        nn.init.zeros_(self.strength_head.weight)
        nn.init.zeros_(self.strength_head.bias)

    def _init_evidence_slots(self):
        nn.init.normal_(self.evidence_slot_embed, std=0.02)

    def _init_residual_path(self):
        out_linear = self.out_proj[1]
        nn.init.normal_(out_linear.weight, std=self.residual_init_scale)

    @staticmethod
    def _empty_aux(h_x):
        zero = h_x.new_zeros(1)
        return {
            "stage2_raw_gamma": zero,
            "stage2_tanh_gamma": zero,
            "stage2_beta": zero,
            "stage2_num_evidence_slots": zero,
            "stage2_attention_uniform_mix": zero,
            "stage2_d_norm_eps": zero,
            "stage2_residual_init_scale": zero,
            "stage2_D_raw_norm_mean": zero,
            "stage2_D_raw_norm_std": zero,
            "stage2_D_raw_spatial_std": zero,
            "stage2_D_gate_mean": zero,
            "stage2_D_gate_std": zero,
            "stage2_D_norm_mean": zero,
            "stage2_D_norm_std": zero,
            "stage2_D_spatial_std": zero,
            "stage2_G_norm_mean": zero,
            "stage2_G_norm_std": zero,
            "stage2_G_spatial_std": zero,
            "stage2_C_norm_mean": zero,
            "stage2_C_norm_std": zero,
            "stage2_C_spatial_std": zero,
            "stage2_u_raw_mean": zero,
            "stage2_u_raw_std": zero,
            "stage2_u_mean": zero,
            "stage2_u_std": zero,
            "stage2_s_mean": zero,
            "stage2_s_std": zero,
            "stage2_s_min": zero,
            "stage2_s_max": zero,
            "stage2_s_deviation_mean": zero,
            "stage2_delta_norm_before_strength": zero,
            "stage2_delta_norm_after_strength": zero,
            "stage2_delta_norm_after_gamma": zero,
            "stage2_delta_to_feature_ratio": zero,
            "stage2_attention_entropy": zero,
            "stage2_attention_entropy_norm": zero,
            "stage2_valid_token_count": zero,
            "stage2_valid_semantic_token_count": zero,
            "stage2_low_valid_token_ratio": zero,
            "stage2_z_proto_norm": zero,
            "stage2_z_proto_std": zero,
        }
