import math

import torch
from torch import nn


class TrackingEvidenceCalibration(nn.Module):
    """Task-specific feature calibration before the tracking head.

    This module updates search tokens, not the head score map. It uses raw
    language tokens and template tokens as narrow evidence for the target.
    """

    def __init__(
            self,
            dim,
            evidence_dim=128,
            gamma_init=1e-3,
            lang_source="raw",
            target_pool="center",
            center_ratio=0.5,
            min_valid_tokens=3,
            dropout=0.0):
        super().__init__()
        if lang_source not in ("raw", "fuse"):
            raise ValueError("TEC lang_source must be 'raw' or 'fuse'.")
        if target_pool not in ("center", "mean"):
            raise ValueError("TEC target_pool must be 'center' or 'mean'.")

        self.lang_source = lang_source
        self.target_pool = target_pool
        self.center_ratio = center_ratio
        self.min_valid_tokens = min_valid_tokens
        self.scale = evidence_dim ** -0.5

        self.lang_fuse = nn.Linear(dim * 2, dim) if lang_source == "fuse" else None
        self.lang_key = nn.Linear(dim, evidence_dim)
        self.lang_value = nn.Linear(dim, evidence_dim)
        self.search_query = nn.Linear(dim, evidence_dim)
        self.target_to_lang = nn.Linear(dim, evidence_dim)
        self.target_to_search = nn.Linear(dim, evidence_dim)

        self.lang_norm = nn.LayerNorm(evidence_dim)
        self.search_norm = nn.LayerNorm(evidence_dim)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(evidence_dim),
            nn.Linear(evidence_dim, dim),
            nn.Dropout(dropout),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, h_x, l_raw, h_z, h_l=None, l_mask=None, template_token_len=None):
        if l_raw is None or h_z is None:
            return h_x, self._empty_aux(h_x)

        l_src = self._select_language_source(l_raw, h_l)
        z_proto = self._target_pool(h_z, template_token_len)
        l_mask, valid_token_count = self._sanitize_language_mask(l_mask, l_src)

        lang_key = self.lang_key(l_src) + self.target_to_lang(z_proto).unsqueeze(1)
        search_query = self.search_query(h_x) + self.target_to_search(z_proto).unsqueeze(1)
        lang_key = self.lang_norm(lang_key)
        search_query = self.search_norm(search_query)

        evidence_logits = torch.matmul(search_query, lang_key.transpose(1, 2)) * self.scale
        evidence_attn = self._masked_softmax(evidence_logits, l_mask)
        evidence = torch.matmul(evidence_attn, self.lang_value(l_src))

        delta = self.out_proj(evidence)
        gamma = torch.tanh(self.gamma)
        scaled_delta = gamma * delta
        h_x_calibrated = h_x + scaled_delta

        delta_norm_before = delta.detach().norm(dim=-1).mean()
        delta_norm_after = scaled_delta.detach().norm(dim=-1).mean()
        h_x_norm = h_x.detach().norm(dim=-1).mean()

        aux = {
            "tec_raw_gamma": self.gamma.detach().reshape(1),
            "tec_tanh_gamma": gamma.detach().reshape(1),
            "tec_delta_norm_before_gamma": delta_norm_before.reshape(1),
            "tec_delta_norm_after_gamma": delta_norm_after.reshape(1),
            "tec_delta_to_feature_ratio": (delta_norm_after / h_x_norm.clamp_min(1e-6)).reshape(1),
            "tec_attn_entropy": self._entropy(evidence_attn).detach().reshape(1),
            "tec_valid_token_count": valid_token_count.detach().float().mean().reshape(1),
            "tec_low_valid_token_ratio": (valid_token_count.detach() < self.min_valid_tokens).float().mean().reshape(1),
            "tec_z_proto_norm": z_proto.detach().norm(dim=-1).mean().reshape(1),
            "tec_z_proto_std": z_proto.detach().std(dim=-1).mean().reshape(1),
        }
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
    def _sanitize_language_mask(l_mask, l_src):
        if l_mask is None:
            l_mask = torch.ones(l_src.shape[:2], device=l_src.device, dtype=torch.bool)
            return l_mask, l_mask.sum(dim=1)

        l_mask = l_mask.to(device=l_src.device, dtype=torch.bool)
        if l_mask.shape[:2] != l_src.shape[:2]:
            l_mask = torch.ones(l_src.shape[:2], device=l_src.device, dtype=torch.bool)
            return l_mask, l_mask.sum(dim=1)

        valid_token_count = l_mask.sum(dim=1)
        empty_rows = ~l_mask.any(dim=1)
        if empty_rows.any():
            l_mask = l_mask.clone()
            l_mask[empty_rows] = True
        return l_mask, valid_token_count

    @staticmethod
    def _masked_softmax(logits, mask):
        mask = mask.to(device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(~mask.unsqueeze(1), torch.finfo(logits.dtype).min)
        return logits.softmax(dim=-1)

    @staticmethod
    def _entropy(attn):
        attn = attn.clamp_min(1e-6)
        return -(attn * attn.log()).sum(dim=-1).mean()

    @staticmethod
    def _empty_aux(h_x):
        zero = h_x.new_zeros(1)
        return {
            "tec_raw_gamma": zero,
            "tec_tanh_gamma": zero,
            "tec_delta_norm_before_gamma": zero,
            "tec_delta_norm_after_gamma": zero,
            "tec_delta_to_feature_ratio": zero,
            "tec_attn_entropy": zero,
            "tec_valid_token_count": zero,
            "tec_low_valid_token_ratio": zero,
            "tec_z_proto_norm": zero,
            "tec_z_proto_std": zero,
        }
