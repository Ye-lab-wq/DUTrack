import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageGlobalScoreBias(nn.Module):
    """Global language state update → score map bias.

    h_prev: pooled language state from previous frame (B, hidden_dim)
    h_cand: pooled candidate language tokens from current frame (B, hidden_dim)

    gate  = sigmoid(MLP([h_prev, h_cand, h_cand - h_prev]))
    delta = tanh(MLP([h_prev, h_cand, h_cand - h_prev])) * max_delta
    h_new = h_prev + gate * delta

    score_bias = beta * normalize(search_feat) · normalize(proj(h_new))

    Init: gate_bias < 0, delta_head last layer zero → strict no-op at init.
    Zero auxiliary losses. Only tracking loss provides gradient.
    """

    def __init__(self, dim, hidden_dim=256, max_delta=0.02, dropout=0.0,
                 init_gate_bias=-4.0, init_delta_std=1e-4,
                 pool_mode="mean", beta=0.02):
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.max_delta = float(max_delta)
        self.pool_mode = str(pool_mode).lower()
        if self.pool_mode not in ("mean", "cls", "attention"):
            raise ValueError("Unsupported pool_mode: {}".format(pool_mode))
        self.beta = float(beta)

        # Token pooling: dim → hidden_dim
        self.token_proj = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.hidden_dim),
            nn.GELU(),
        )
        if self.pool_mode == "attention":
            self.pool_query = nn.Parameter(torch.empty(1, 1, self.hidden_dim))
            nn.init.trunc_normal_(self.pool_query, std=0.02)

        # Gate MLP: [h_prev, h_cand, h_cand - h_prev] → 1
        gate_in_dim = self.hidden_dim * 3
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )

        # Delta MLP: same input → hidden_dim
        self.delta_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Score projection: hidden_dim → dim, for normalized similarity with search features
        self.lang_score_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.dim),
        )
        self.search_score_proj = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
        )

        self._init_conservative(init_gate_bias, init_delta_std)

    def _init_conservative(self, gate_bias, delta_std):
        # Gate: last layer → zero weight, negative bias → gate ≈ sigmoid(-4) ≈ 0.018
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.constant_(self.gate_mlp[-1].bias, float(gate_bias))
        # Delta: last layer → near-zero
        nn.init.normal_(self.delta_mlp[-1].weight, std=float(delta_std))
        nn.init.zeros_(self.delta_mlp[-1].bias)
        # Score projections: small random init so initial score_bias ≈ 0
        # but gradients are non-degenerate (unlike zero-init which deadens the path).
        nn.init.normal_(self.lang_score_proj[-1].weight, std=float(delta_std))
        nn.init.zeros_(self.lang_score_proj[-1].bias)
        nn.init.normal_(self.search_score_proj[-1].weight, std=float(delta_std))
        nn.init.zeros_(self.search_score_proj[-1].bias)

    def pool_tokens(self, tokens, mask=None):
        """Pool (B, L, D) token sequence → (B, hidden_dim)."""
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.squeeze(-1)
            mask_f = mask.to(dtype=tokens.dtype, device=tokens.device)
        else:
            mask_f = torch.ones(tokens.shape[:2], dtype=tokens.dtype, device=tokens.device)

        proj = self.token_proj(tokens)  # (B, L, hidden_dim)

        if self.pool_mode == "mean":
            denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1e-6)
            return (proj * mask_f.unsqueeze(-1)).sum(dim=1) / denom

        if self.pool_mode == "cls":
            return proj[:, 0, :]  # first token (CLS)

        if self.pool_mode == "attention":
            query = self.pool_query.expand(tokens.shape[0], -1, -1)
            attn_logits = torch.matmul(
                query, proj.transpose(1, 2)) / math.sqrt(float(self.hidden_dim))
            attn_logits = attn_logits.masked_fill(
                ~mask_f.bool().unsqueeze(1), -1e4)
            attn = torch.softmax(attn_logits, dim=-1)
            pooled = torch.matmul(attn, proj).squeeze(1)  # (B, hidden_dim)
            return pooled

        raise ValueError("Unsupported pool_mode: {}".format(self.pool_mode))

    def forward(self, h_prev_pooled, h_cand_pooled, search_feat,
                return_diagnostics=True):
        """Compute updated language state and score bias.

        Args:
            h_prev_pooled: (B, hidden_dim) language state from previous frame
            h_cand_pooled: (B, hidden_dim) pooled candidate language tokens
            search_feat:   (B, N, D) search region features from backbone

        Returns:
            h_new:      (B, hidden_dim) updated language state
            score_bias: (B, N) score map bias (flattened spatial dim)
            diagnostics: dict of scalar tensors
        """
        feat = torch.cat([h_prev_pooled, h_cand_pooled, h_cand_pooled - h_prev_pooled], dim=-1)

        gate = torch.sigmoid(self.gate_mlp(feat))       # (B, 1)
        delta = torch.tanh(self.delta_mlp(feat)) * self.max_delta  # (B, hidden_dim)
        h_new = h_prev_pooled + gate * delta              # (B, hidden_dim)

        # Score bias via normalized similarity
        lang_proj = F.normalize(self.lang_score_proj(h_new), dim=-1)      # (B, dim)
        search_proj = F.normalize(self.search_score_proj(search_feat), dim=-1)  # (B, N, dim)
        score_bias = torch.matmul(search_proj, lang_proj.unsqueeze(-1)).squeeze(-1)  # (B, N)
        score_bias = score_bias * self.beta

        diagnostics = {}
        if return_diagnostics:
            diagnostics = {
                "gsb_gate_mean": gate.detach().mean(),
                "gsb_delta_abs_mean": delta.detach().abs().mean(),
                "gsb_bias_abs_mean": score_bias.detach().abs().mean(),
                "gsb_bias_max": score_bias.detach().max(),
                "gsb_bias_min": score_bias.detach().min(),
                "gsb_bias_std": score_bias.detach().std(),
            }

        return h_new, score_bias, diagnostics
