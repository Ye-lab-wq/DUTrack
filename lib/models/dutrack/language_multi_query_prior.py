import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageMultiQueryPrior(nn.Module):
    """Learnable language queries that produce a search-space score prior."""

    def __init__(self, embed_dim, hidden_dim=256, num_queries=4, dropout=0.0,
                 seed_residual=False, seed_residual_gamma=0.1,
                 decoder_enable=False, decoder_num_heads=8,
                 decoder_dropout=0.1, decoder_ffn_ratio=2.0):
        super().__init__()
        self.num_queries = max(1, int(num_queries))
        self.hidden_dim = int(hidden_dim)
        self.seed_residual = bool(seed_residual)
        self.seed_residual_gamma = float(seed_residual_gamma)
        self.decoder_enable = bool(decoder_enable)
        self.decoder_num_heads = int(decoder_num_heads)
        self.decoder_dropout = float(decoder_dropout)
        self.decoder_ffn_ratio = float(decoder_ffn_ratio)
        if self.decoder_enable and self.hidden_dim % self.decoder_num_heads != 0:
            raise ValueError("LMQ decoder hidden_dim must be divisible by decoder_num_heads")

        self.lang_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.visual_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.query_seed = nn.Parameter(torch.empty(self.num_queries, hidden_dim))
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lang_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lang_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.search_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_norm = nn.LayerNorm(hidden_dim)
        if self.decoder_enable:
            self.query_self_attn = nn.MultiheadAttention(
                hidden_dim, self.decoder_num_heads,
                dropout=self.decoder_dropout, batch_first=True)
            self.query_cross_attn = nn.MultiheadAttention(
                hidden_dim, self.decoder_num_heads,
                dropout=self.decoder_dropout, batch_first=True)
            self.self_attn_norm = nn.LayerNorm(hidden_dim)
            self.cross_attn_norm = nn.LayerNorm(hidden_dim)
            self.decoder_ffn_norm = nn.LayerNorm(hidden_dim)
            ffn_dim = max(hidden_dim, int(round(hidden_dim * self.decoder_ffn_ratio)))
            self.decoder_ffn = nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(self.decoder_dropout),
                nn.Linear(ffn_dim, hidden_dim),
                nn.Dropout(self.decoder_dropout),
            )
            self.query_search_pair = nn.Sequential(
                nn.LayerNorm(hidden_dim * 3),
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(self.decoder_dropout),
                nn.Linear(hidden_dim, 1),
            )
        self.query_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.prior_scale = nn.Parameter(torch.tensor(1.0))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.query_seed, std=0.02)
        nn.init.zeros_(self.query_fusion[-1].bias)

    @staticmethod
    def _prepare_mask(tokens, mask):
        if mask is None:
            return torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        if mask.dim() == 3:
            mask = mask.squeeze(-1)
        return mask.to(device=tokens.device).bool()

    @staticmethod
    def _pairwise_cosine(values, center=False):
        if values.dim() == 2:
            values = values.unsqueeze(0)
        if center:
            values = values - values.mean(dim=-1, keepdim=True)
        values = F.normalize(values, dim=-1, eps=1e-6)
        cosine = torch.matmul(values, values.transpose(1, 2))
        k = cosine.shape[1]
        if k <= 1:
            return cosine.new_zeros(cosine.shape[0]), cosine.new_zeros(cosine.shape[0])
        mask = ~torch.eye(k, dtype=torch.bool, device=cosine.device)
        pair_values = cosine[:, mask].view(cosine.shape[0], k * (k - 1))
        return pair_values.mean(dim=1), pair_values.max(dim=1).values

    @staticmethod
    def _pairwise_prior_cosine(query_maps):
        # query_maps: B,K,N. Center each map so cosine measures spatial pattern,
        # not a shared global offset.
        return LanguageMultiQueryPrior._pairwise_cosine(query_maps, center=True)

    def forward(self, lang_tokens, search_tokens, lang_mask=None):
        lang_mask = self._prepare_mask(lang_tokens, lang_mask)
        lang = self.lang_proj(lang_tokens)
        search = self.visual_proj(search_tokens)

        batch_size = lang.shape[0]
        seeds = self.query_seed.unsqueeze(0).expand(batch_size, -1, -1)
        query_logits = torch.matmul(
            self.query_proj(seeds),
            self.lang_key(lang).transpose(1, 2)) / math.sqrt(float(self.hidden_dim))
        query_logits = query_logits.masked_fill(~lang_mask[:, None, :], -1e4)
        query_attn = torch.softmax(query_logits, dim=-1)
        pooled_queries = torch.matmul(query_attn, self.lang_value(lang))
        if self.seed_residual:
            queries = self.query_norm(pooled_queries + self.seed_residual_gamma * seeds)
        else:
            queries = self.query_norm(pooled_queries) if self.decoder_enable else pooled_queries

        query_search_attn_entropy = queries.new_zeros(batch_size)
        query_search_attn_max = queries.new_zeros(batch_size)
        decoder_query_delta_norm = queries.new_zeros(batch_size)
        if self.decoder_enable:
            query_input = queries
            self_out, _ = self.query_self_attn(
                query_input, query_input, query_input, need_weights=False)
            queries = self.self_attn_norm(query_input + self_out)
            cross_out, cross_attn = self.query_cross_attn(
                queries, search, search, need_weights=True)
            queries = self.cross_attn_norm(queries + cross_out)
            queries = self.decoder_ffn_norm(queries + self.decoder_ffn(queries))
            decoder_query_delta_norm = (queries - query_input).norm(dim=-1).mean(dim=1)

            query_expand = queries[:, :, None, :].expand(-1, -1, search.shape[1], -1)
            search_expand = search[:, None, :, :].expand(-1, self.num_queries, -1, -1)
            pair_feat = torch.cat([query_expand, search_expand, query_expand * search_expand], dim=-1)
            query_maps = self.query_search_pair(pair_feat).squeeze(-1)
            if cross_attn is not None:
                if cross_attn.dim() == 4:
                    cross_attn = cross_attn.mean(dim=1)
                query_search_attn_entropy = -(
                    cross_attn.clamp_min(1e-6).log() * cross_attn).sum(dim=-1).mean(dim=1)
                query_search_attn_max = cross_attn.max(dim=-1).values.mean(dim=1)
        else:
            queries = F.normalize(queries, dim=-1, eps=1e-6)
            search_keys = F.normalize(self.search_key(search), dim=-1, eps=1e-6)
            query_maps = torch.matmul(queries, search_keys.transpose(1, 2))

        fusion_logits = self.query_fusion(queries).squeeze(-1)
        fusion_weights = torch.softmax(fusion_logits, dim=-1)
        prior = (query_maps * fusion_weights.unsqueeze(-1)).sum(dim=1)
        prior = prior * self.prior_scale

        cosine_mean, cosine_max = self._pairwise_prior_cosine(query_maps)
        seed_cosine_mean, seed_cosine_max = self._pairwise_cosine(seeds, center=False)
        attn_cosine_mean, attn_cosine_max = self._pairwise_cosine(query_attn, center=False)
        pooled_query_cosine_mean, pooled_query_cosine_max = self._pairwise_cosine(pooled_queries, center=False)
        query_cosine_mean, query_cosine_max = self._pairwise_cosine(queries, center=False)
        attn_entropy = -(query_attn.clamp_min(1e-6).log() * query_attn).sum(dim=-1).mean(dim=1)
        attn_max = query_attn.max(dim=-1).values.mean(dim=1)
        query_map_between_std = query_maps.std(dim=1, unbiased=False).mean(dim=-1)
        prior_score_std = prior.std(dim=-1, unbiased=False)
        return {
            "prior_scores": prior,
            "query_prior_maps": query_maps,
            "query_lang_attn": query_attn,
            "query_fusion_weights": fusion_weights,
            "query_prior_cosine_mean": cosine_mean,
            "query_prior_cosine_max": cosine_max,
            "query_seed_cosine_mean": seed_cosine_mean,
            "query_seed_cosine_max": seed_cosine_max,
            "query_lang_attn_cosine_mean": attn_cosine_mean,
            "query_lang_attn_cosine_max": attn_cosine_max,
            "query_lang_attn_entropy": attn_entropy,
            "query_lang_attn_max": attn_max,
            "pooled_query_cosine_mean": pooled_query_cosine_mean,
            "pooled_query_cosine_max": pooled_query_cosine_max,
            "query_vector_cosine_mean": query_cosine_mean,
            "query_vector_cosine_max": query_cosine_max,
            "query_map_between_std": query_map_between_std,
            "prior_score_std": prior_score_std,
            "query_search_attn_entropy": query_search_attn_entropy,
            "query_search_attn_max": query_search_attn_max,
            "decoder_query_delta_norm": decoder_query_delta_norm,
        }
