import torch
import torch.nn as nn


class _RelationBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x, valid_mask=None):
        y = self.norm1(x)
        key_padding_mask = None
        if valid_mask is not None:
            if valid_mask.dim() == 3:
                valid_mask = valid_mask.squeeze(-1)
            key_padding_mask = ~valid_mask.to(device=x.device).bool()
        y = y.transpose(0, 1)
        attn_out, attn_weights = self.attn(
            y, y, y, key_padding_mask=key_padding_mask,
            need_weights=True)
        x = x + attn_out.transpose(0, 1)
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


def _masked_mean(x, mask):
    if mask is None:
        return x.mean(dim=1)
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    mask = mask.to(device=x.device, dtype=x.dtype)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return (x * mask).sum(dim=1) / denom


class LanguageTokenStateUpdater(nn.Module):
    """Conservative latent language-state updater.

    The module updates encoded language tokens directly. It is initialized near
    no-op so old checkpoints and diagnostic runs are not disturbed unless this
    path is explicitly enabled and trained.
    """

    def __init__(self, dim, hidden_dim=256, max_delta=0.1, dropout=0.0,
                 init_gate_bias=-4.0, init_delta_std=1e-4,
                 relation_layers=1, relation_heads=4,
                 visual_evidence_dim=8, update_mode="residual",
                 source_init_gate_bias=None,
                 alignment_mode="position", alignment_heads=4):
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.max_delta = float(max_delta)
        self.visual_evidence_dim = int(visual_evidence_dim)
        self.update_mode = str(update_mode).lower()
        if self.update_mode not in ("residual", "keep_absorb", "decoupled"):
            raise ValueError("Unsupported language state update mode: {}".format(update_mode))
        self.alignment_mode = str(alignment_mode).lower()
        if self.alignment_mode not in ("position", "cross_attn"):
            raise ValueError("Unsupported language state alignment mode: {}".format(alignment_mode))

        feature_dim = self.dim * 5
        self.token_norm = nn.LayerNorm(feature_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        relation_heads = max(1, min(int(relation_heads), int(hidden_dim)))
        if int(hidden_dim) % relation_heads != 0:
            relation_heads = 1
        self.relation_blocks = nn.ModuleList([
            _RelationBlock(hidden_dim, num_heads=relation_heads, dropout=dropout)
            for _ in range(max(0, int(relation_layers)))
        ])
        self.visual_mlp = nn.Sequential(
            nn.LayerNorm(self.visual_evidence_dim),
            nn.Linear(self.visual_evidence_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        alignment_heads = max(1, min(int(alignment_heads), self.dim))
        if self.dim % alignment_heads != 0:
            alignment_heads = 1
        self.align_query_norm = nn.LayerNorm(self.dim)
        self.align_key_norm = nn.LayerNorm(self.dim)
        self.align_attn = nn.MultiheadAttention(self.dim, alignment_heads, dropout=dropout)
        self.delta_head = nn.Linear(hidden_dim, self.dim)
        self.token_gate_head = nn.Linear(hidden_dim, 1)
        if self.update_mode == "decoupled":
            self.source_gate_head = nn.Linear(hidden_dim, 2)
            self.update_gate_head = nn.Linear(hidden_dim, 1)
        else:
            self.source_gate_head = nn.Linear(hidden_dim, 3)
            self.update_gate_head = None

        self.frame_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_conservative(init_gate_bias, init_delta_std, source_init_gate_bias)

    def _init_conservative(self, init_gate_bias, init_delta_std, source_init_gate_bias):
        nn.init.normal_(self.delta_head.weight, mean=0.0, std=float(init_delta_std))
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.token_gate_head.weight)
        nn.init.constant_(self.token_gate_head.bias, float(init_gate_bias))
        nn.init.zeros_(self.source_gate_head.weight)
        if self.update_mode == "decoupled":
            # Source blend: 2-way softmax between anchor and candidate.
            # Start balanced (equal bias for both).
            source_bias = 0.0 if source_init_gate_bias is None else float(source_init_gate_bias)
            nn.init.constant_(self.source_gate_head.bias, float(source_bias))
            # Update gate: sigmoid, initialized near 0 (keep prev).
            nn.init.zeros_(self.update_gate_head.weight)
            nn.init.constant_(self.update_gate_head.bias, float(init_gate_bias))
        else:
            # Source order: anchor, prev, candidate. Start near H_prev so the new
            # path is checkpoint-safe but still exposes explicit keep/absorb gates.
            source_bias = init_gate_bias if source_init_gate_bias is None else source_init_gate_bias
            nn.init.constant_(self.source_gate_head.bias, float(source_bias))
            self.source_gate_head.bias.data[1] = float(-source_bias)
        last = self.frame_gate[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, float(init_gate_bias))

    def _features(self, anchor_tokens, prev_tokens, candidate_tokens):
        return torch.cat([
            anchor_tokens,
            prev_tokens,
            candidate_tokens,
            candidate_tokens - anchor_tokens,
            candidate_tokens - prev_tokens,
        ], dim=-1)

    def _align_tokens_to_prev(self, prev_tokens, source_tokens, source_mask=None):
        if self.alignment_mode == "position":
            empty = torch.zeros((), device=prev_tokens.device, dtype=prev_tokens.dtype)
            return source_tokens, {
                "entropy": empty,
                "max": empty,
            }
        key_padding_mask = None
        if source_mask is not None:
            if source_mask.dim() == 3:
                source_mask = source_mask.squeeze(-1)
            key_padding_mask = ~source_mask.to(device=prev_tokens.device).bool()
        query = self.align_query_norm(prev_tokens).transpose(0, 1)
        key_value = self.align_key_norm(source_tokens).transpose(0, 1)
        aligned, attn_weights = self.align_attn(
            query, key_value, key_value,
            key_padding_mask=key_padding_mask,
            need_weights=True)
        aligned = aligned.transpose(0, 1)
        attn_probs = attn_weights.detach().clamp_min(1e-6)
        entropy = -(attn_probs * attn_probs.log()).sum(dim=-1).mean()
        return aligned, {
            "entropy": entropy,
            "max": attn_probs.max(dim=-1).values.mean(),
        }

    def _visual_embedding(self, visual_evidence, batch_size, device, dtype):
        if visual_evidence is None:
            visual_evidence = torch.zeros(
                batch_size, self.visual_evidence_dim, device=device, dtype=dtype)
        else:
            visual_evidence = visual_evidence.to(device=device, dtype=dtype)
            if visual_evidence.dim() == 1:
                visual_evidence = visual_evidence.unsqueeze(0)
            if visual_evidence.shape[0] != batch_size:
                raise ValueError(
                    "visual_evidence batch {} does not match token batch {}".format(
                        visual_evidence.shape[0], batch_size))
            if visual_evidence.shape[1] < self.visual_evidence_dim:
                pad = torch.zeros(
                    batch_size, self.visual_evidence_dim - visual_evidence.shape[1],
                    device=device, dtype=dtype)
                visual_evidence = torch.cat([visual_evidence, pad], dim=1)
            elif visual_evidence.shape[1] > self.visual_evidence_dim:
                visual_evidence = visual_evidence[:, :self.visual_evidence_dim]
        return self.visual_mlp(visual_evidence)

    def forward(self, anchor_tokens, prev_tokens, candidate_tokens,
                anchor_mask=None, prev_mask=None, candidate_mask=None,
                visual_evidence=None):
        if anchor_tokens.shape != prev_tokens.shape or anchor_tokens.shape != candidate_tokens.shape:
            raise ValueError(
                "LanguageTokenStateUpdater expects equal token shapes, got "
                "anchor {}, prev {}, candidate {}".format(
                    tuple(anchor_tokens.shape),
                    tuple(prev_tokens.shape),
                    tuple(candidate_tokens.shape)))
        if anchor_tokens.dim() != 3 or anchor_tokens.shape[-1] != self.dim:
            raise ValueError(
                "LanguageTokenStateUpdater expects (B,L,{}) tokens, got {}".format(
                    self.dim, tuple(anchor_tokens.shape)))

        anchor_aligned, anchor_align_stats = self._align_tokens_to_prev(
            prev_tokens, anchor_tokens, anchor_mask)
        candidate_aligned, candidate_align_stats = self._align_tokens_to_prev(
            prev_tokens, candidate_tokens, candidate_mask)

        features = self._features(anchor_aligned, prev_tokens, candidate_aligned)
        hidden = self.token_mlp(self.token_norm(features))
        valid_mask = prev_mask
        if valid_mask is None:
            valid_mask = candidate_mask if candidate_mask is not None else anchor_mask
        relation_attn_means = []
        relation_attn_raw = None
        for block in self.relation_blocks:
            hidden, attn_weights = block(hidden, valid_mask=valid_mask)
            relation_attn_means.append(attn_weights.detach().mean())
            relation_attn_raw = attn_weights.detach()
        visual_hidden = self._visual_embedding(
            visual_evidence, anchor_tokens.shape[0], anchor_tokens.device, anchor_tokens.dtype)
        hidden = hidden + visual_hidden.unsqueeze(1)
        token_gate = torch.sigmoid(self.token_gate_head(hidden))
        source_logits = self.source_gate_head(hidden)
        if self.update_mode == "decoupled":
            update_gate = torch.sigmoid(self.update_gate_head(hidden))
            source_weights = torch.softmax(source_logits, dim=-1)
            anchor_weight = source_weights[..., 0:1]
            candidate_weight = source_weights[..., 1:2]
            prev_weight = 1.0 - update_gate
        else:
            source_weights = torch.softmax(source_logits, dim=-1)
            anchor_weight = source_weights[..., 0:1]
            prev_weight = source_weights[..., 1:2]
            candidate_weight = source_weights[..., 2:3]
            update_gate = None

        frame_features = torch.cat([
            _masked_mean(hidden, valid_mask),
            visual_hidden,
        ], dim=-1)
        frame_gate = torch.sigmoid(self.frame_gate(frame_features)).unsqueeze(1)

        delta = torch.tanh(self.delta_head(hidden)) * self.max_delta
        combined_gate = token_gate * frame_gate
        if self.update_mode == "decoupled":
            source_mix = anchor_weight * anchor_aligned + candidate_weight * candidate_aligned
            state_tokens = prev_weight * prev_tokens + update_gate * (source_mix + combined_gate * delta)
        elif self.update_mode == "keep_absorb":
            state_tokens = (
                anchor_weight * anchor_aligned
                + prev_weight * prev_tokens
                + candidate_weight * candidate_aligned
                + combined_gate * delta
            )
        else:
            state_tokens = prev_tokens + combined_gate * delta

        if candidate_mask is not None or prev_mask is not None or anchor_mask is not None:
            valid_mask = prev_mask
            if valid_mask is None:
                valid_mask = candidate_mask if candidate_mask is not None else anchor_mask
            if valid_mask is not None:
                if valid_mask.dim() == 2:
                    valid_mask = valid_mask.unsqueeze(-1)
                valid_mask = valid_mask.to(device=state_tokens.device, dtype=state_tokens.dtype)
                state_tokens = state_tokens * valid_mask + prev_tokens * (1.0 - valid_mask)

        state_delta_abs = (state_tokens - prev_tokens).abs()
        if self.update_mode == "decoupled":
            source_mix = anchor_weight * anchor_aligned + candidate_weight * candidate_aligned
        else:
            source_mix = (
                anchor_weight * anchor_aligned
                + prev_weight * prev_tokens
                + candidate_weight * candidate_aligned
            )
        source_mix_delta_abs = (source_mix - prev_tokens).abs()
        residual_delta_abs = (combined_gate * delta).abs()
        source_entropy = -(source_weights.clamp_min(1e-6).log() * source_weights).sum(dim=-1)
        diagnostics = {
            "alignment_mode_cross_attn": torch.as_tensor(
                1.0 if self.alignment_mode == "cross_attn" else 0.0,
                device=state_tokens.device, dtype=state_tokens.dtype),
            "update_mode_keep_absorb": torch.as_tensor(
                1.0 if self.update_mode == "keep_absorb" else 0.0,
                device=state_tokens.device, dtype=state_tokens.dtype),
            "update_mode_decoupled": torch.as_tensor(
                1.0 if self.update_mode == "decoupled" else 0.0,
                device=state_tokens.device, dtype=state_tokens.dtype),
            "token_gate_mean": token_gate.detach().mean(),
            "frame_gate_mean": frame_gate.detach().mean(),
            "combined_gate_mean": combined_gate.detach().mean(),
            "combined_gate_x1e4": combined_gate.detach().mean() * 1.0e4,
            "delta_abs_mean": delta.detach().abs().mean(),
            "state_delta_abs_mean": state_delta_abs.detach().mean(),
            "state_delta_abs_x1e6": state_delta_abs.detach().mean() * 1.0e6,
            "source_mix_delta_abs_mean": source_mix_delta_abs.detach().mean(),
            "residual_delta_abs_mean": residual_delta_abs.detach().mean(),
            "source_mix_delta_abs_x1e6": source_mix_delta_abs.detach().mean() * 1.0e6,
            "residual_delta_abs_x1e6": residual_delta_abs.detach().mean() * 1.0e6,
            "anchor_weight_mean": anchor_weight.detach().mean(),
            "prev_keep_weight_mean": prev_weight.detach().mean(),
            "candidate_absorb_weight_mean": candidate_weight.detach().mean(),
            "source_entropy_mean": source_entropy.detach().mean(),
            "update_gate_mean": (
                update_gate.detach().mean() if update_gate is not None
                else torch.zeros((), device=state_tokens.device, dtype=state_tokens.dtype)
            ),
            "gate_reg_loss": (combined_gate + (1.0 - prev_weight)).mean(),
            "delta_reg_loss": state_delta_abs.mean(),
            "visual_evidence_abs_mean": (
                torch.zeros((), device=state_tokens.device)
                if visual_evidence is None else visual_evidence.detach().abs().mean()
            ),
            "anchor_alignment_entropy": anchor_align_stats["entropy"],
            "anchor_alignment_max": anchor_align_stats["max"],
            "candidate_alignment_entropy": candidate_align_stats["entropy"],
            "candidate_alignment_max": candidate_align_stats["max"],
            "_anchor_aligned_tokens": anchor_aligned,
            "_candidate_aligned_tokens": candidate_aligned,
            "_anchor_weight": anchor_weight,
            "_prev_weight": prev_weight,
            "_candidate_weight": candidate_weight,
            "_candidate_absorb_logit": (
                source_logits[..., 1] - source_logits[..., 0]
                if self.update_mode == "decoupled"
                else source_logits[..., 2] - torch.logsumexp(source_logits[..., :2], dim=-1)
            ),
        }
        if relation_attn_means:
            diagnostics["relation_attn_mean"] = torch.stack(relation_attn_means).mean()
        if relation_attn_raw is not None:
            attn_clamped = relation_attn_raw.clamp_min(1e-6)
            ent = -(attn_clamped * attn_clamped.log()).sum(dim=-1).mean()
            attn_max_val = attn_clamped.max(dim=-1).values.mean()
            diag_mass = attn_clamped.diagonal(dim1=-2, dim2=-1).sum(dim=-1).mean()
            diagnostics["relation_attn_entropy"] = ent
            diagnostics["relation_attn_max"] = attn_max_val
            diagnostics["relation_attn_diag_mass"] = diag_mass
            diagnostics["relation_attn_offdiag_mass"] = 1.0 - diag_mass
        return state_tokens, diagnostics
