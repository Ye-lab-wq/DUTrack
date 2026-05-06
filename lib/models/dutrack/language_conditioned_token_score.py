import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageConditionedTokenScore(nn.Module):
    """Lightweight language-conditioned scoring for visual patch tokens."""

    def __init__(self, dim, hidden_dim=256, lang_pool="cls", lang_refine=False,
                 lang_refine_alpha=0.5, lang_refine_temp=1.0, lang_refine_mode="visual_soft",
                 lang_residual_beta=0.1, lang_subject_hard=True):
        super().__init__()
        self.lang_pool = lang_pool
        self.lang_refine = lang_refine
        self.lang_refine_alpha = lang_refine_alpha
        self.lang_refine_temp = lang_refine_temp
        self.lang_refine_mode = lang_refine_mode
        self.lang_residual_beta = lang_residual_beta
        self.lang_subject_hard = lang_subject_hard

        self.visual_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.lang_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.subject_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2),
        )
        self.consistency_scale = nn.Parameter(torch.tensor(1.0))

    def _pool_language(self, lang_tokens, lang_mask=None):
        if self.lang_pool == "mean":
            if lang_mask is not None:
                weight = lang_mask.to(dtype=lang_tokens.dtype).unsqueeze(-1)
                return (lang_tokens * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
            return lang_tokens.mean(dim=1)
        return lang_tokens[:, 0]

    def _refine_language(self, z_feat, lang_tokens, lang_mask=None):
        base_context = self._pool_language(lang_tokens, lang_mask)
        if not self.lang_refine:
            return base_context, None
        if self.lang_refine_mode == "gumbel_subject_residual":
            return self._refine_language_with_subject_residual(lang_tokens, base_context, lang_mask)

        visual_query = z_feat.mean(dim=1)
        visual_query = F.normalize(self.visual_proj(visual_query), dim=-1).unsqueeze(1)
        lang_key = F.normalize(self.lang_proj(lang_tokens), dim=-1)
        token_logits = (visual_query * lang_key).sum(dim=-1)
        token_logits = token_logits / max(float(self.lang_refine_temp), 1e-6)
        if lang_mask is not None:
            token_logits = token_logits.masked_fill(~lang_mask.bool(), float("-inf"))

        token_weight = F.softmax(token_logits, dim=-1)
        refined_context = torch.bmm(token_weight.unsqueeze(1), lang_tokens).squeeze(1)
        alpha = float(self.lang_refine_alpha)
        lang_context = (1.0 - alpha) * base_context + alpha * refined_context
        return lang_context, token_weight

    def _refine_language_with_subject_residual(self, lang_tokens, base_context, lang_mask=None):
        logits = self.subject_head(lang_tokens)
        if lang_mask is not None:
            pad_mask = ~lang_mask.bool()
            logits = logits.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            logits = logits.masked_fill(pad_mask.unsqueeze(-1) & torch.tensor([False, True], device=logits.device), -1e4)

        tau = max(float(self.lang_refine_temp), 1e-6)
        if self.training:
            subject_decision = F.gumbel_softmax(logits, tau=tau, hard=self.lang_subject_hard, dim=-1)
        else:
            subject_decision = F.softmax(logits / tau, dim=-1)
        subject_weight = subject_decision[..., 1]

        if lang_mask is not None:
            subject_weight = subject_weight * lang_mask.to(dtype=subject_weight.dtype)
        denom = subject_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        subject_context = torch.bmm(subject_weight.unsqueeze(1), lang_tokens).squeeze(1) / denom

        beta = float(self.lang_residual_beta)
        lang_context = base_context + beta * (subject_context - base_context)
        return lang_context, subject_weight

    def _score_tokens(self, visual_tokens, lang_context):
        visual = F.normalize(self.visual_proj(visual_tokens), dim=-1)
        lang = F.normalize(self.lang_proj(lang_context), dim=-1).unsqueeze(1)
        lang = lang.expand_as(visual)

        product = visual * lang
        distance = (visual - lang).abs()
        logits = self.score_head(torch.cat([visual, lang, product, distance], dim=-1)).squeeze(-1)
        consistency = product.sum(dim=-1)
        logits = logits + self.consistency_scale * consistency
        return logits, torch.sigmoid(logits), consistency

    def forward(self, z_feat, x_feat, l_feat, num_templates=1, lang_mask=None):
        batch_size, z_len, _ = z_feat.shape
        num_templates = max(int(num_templates), 1)
        if z_len % num_templates != 0:
            num_templates = 1
        tokens_per_template = z_len // num_templates

        lang_context, lang_weight = self._refine_language(z_feat, l_feat, lang_mask)
        z_logits_flat, z_score_flat, z_consistency_flat = self._score_tokens(z_feat, lang_context)
        x_logits, x_score, x_consistency = self._score_tokens(x_feat, lang_context)

        z_logits = z_logits_flat.view(batch_size, num_templates, tokens_per_template)
        z_score = z_score_flat.view(batch_size, num_templates, tokens_per_template)
        z_consistency = z_consistency_flat.view(batch_size, num_templates, tokens_per_template)

        out = {
            "vl_score_z": z_score,
            "vl_score_x": x_score,
            "vl_score_z_logits": z_logits,
            "vl_score_x_logits": x_logits,
            "language_consistency": z_consistency.mean(dim=-1),
            "search_language_consistency": x_consistency.mean(dim=-1),
            "template_quality": z_score.mean(dim=-1),
        }
        if lang_weight is not None:
            out["language_token_weights"] = lang_weight
        return out
