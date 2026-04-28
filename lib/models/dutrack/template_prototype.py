import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemplatePrototypeHead(nn.Module):
    def __init__(self, hidden_dim, keep_topk=4, split_beta=0.35):
        super().__init__()
        self.keep_topk = keep_topk
        self.split_beta = split_beta

        self.feat_norm = nn.LayerNorm(hidden_dim)
        self.semantic_token_base = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.semantic_lang_proj = nn.Linear(hidden_dim, hidden_dim)
        self.semantic_visual_proj = nn.Linear(hidden_dim, hidden_dim)
        self.semantic_token_norm = nn.LayerNorm(hidden_dim)
        self.target_scorer = nn.Linear(hidden_dim, 1)
        self.distractor_scorer = nn.Linear(hidden_dim, 1)
        self.background_scorer = nn.Linear(hidden_dim, 1)

        self.target_attn_scale = nn.Parameter(torch.tensor(0.5))
        self.target_pos_scale = nn.Parameter(torch.tensor(0.5))
        self.split_attn_scale = nn.Parameter(torch.tensor(0.25))
        self.split_sim_scale = nn.Parameter(torch.tensor(0.5))

    @staticmethod
    def _normalize_scores(scores):
        scores = scores - scores.amin(dim=1, keepdim=True)
        return scores / scores.amax(dim=1, keepdim=True).clamp(min=1e-6)

    @staticmethod
    def _masked_softmax(logits, mask):
        masked_logits = logits.masked_fill(mask <= 0, float("-inf"))
        valid = (mask > 0).any(dim=1, keepdim=True)
        weights = F.softmax(masked_logits, dim=1)
        return torch.where(valid, weights, torch.zeros_like(weights))

    @staticmethod
    def _weighted_sum(feat, weight):
        denom = weight.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return torch.bmm(weight.unsqueeze(1), feat).squeeze(1) / denom

    @staticmethod
    def _build_box_mask(boxes, grid_hw, device):
        boxes = boxes.clamp(0.0, 1.0)
        cx, cy, w, h = boxes.unbind(dim=-1)
        x1 = (cx - 0.5 * w).clamp(0.0, 1.0)
        y1 = (cy - 0.5 * h).clamp(0.0, 1.0)
        x2 = (cx + 0.5 * w).clamp(0.0, 1.0)
        y2 = (cy + 0.5 * h).clamp(0.0, 1.0)

        scale = float(grid_hw)
        gx1 = torch.floor(x1 * scale).long().clamp(0, grid_hw - 1)
        gy1 = torch.floor(y1 * scale).long().clamp(0, grid_hw - 1)
        gx2 = torch.ceil(x2 * scale).long().clamp(1, grid_hw)
        gy2 = torch.ceil(y2 * scale).long().clamp(1, grid_hw)
        gx2 = torch.maximum(gx2, gx1 + 1)
        gy2 = torch.maximum(gy2, gy1 + 1)

        batch = boxes.shape[0]
        mask = torch.zeros((batch, grid_hw, grid_hw), device=device, dtype=torch.float32)
        for b in range(batch):
            mask[b, gy1[b]:gy2[b], gx1[b]:gx2[b]] = 1.0
        return mask.flatten(1)

    @staticmethod
    def _gaussian_prior(boxes, grid_hw, device):
        coords = torch.arange(grid_hw, device=device, dtype=torch.float32) + 0.5
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        xx = (xx / float(grid_hw)).flatten().unsqueeze(0)
        yy = (yy / float(grid_hw)).flatten().unsqueeze(0)

        cx, cy, w, h = boxes.unbind(dim=-1)
        cx = cx.unsqueeze(1)
        cy = cy.unsqueeze(1)
        sigma = (0.25 * torch.maximum(w, h)).clamp(min=1.0 / grid_hw).unsqueeze(1)
        dist2 = (xx - cx).pow(2) + (yy - cy).pow(2)
        return torch.exp(-dist2 / (2.0 * sigma.pow(2)))

    def _build_pseudo_distractor(self, split_signal, out_mask):
        batch, num_tokens = split_signal.shape
        pseudo = torch.zeros_like(split_signal)
        for b in range(batch):
            valid_idx = torch.nonzero(out_mask[b] > 0, as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                continue
            probs = F.softmax(split_signal[b, valid_idx], dim=0)
            sorted_probs, order = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=0)
            dist_count = int((cum_probs <= self.split_beta).sum().item())
            if dist_count <= 0 and sorted_probs[0] > 0:
                dist_count = 1
            dist_count = min(dist_count, valid_idx.numel())
            pseudo_idx = valid_idx[order[:dist_count]]
            pseudo[b, pseudo_idx] = 1.0
        return pseudo

    def forward(self, search_feat, attn_l2s, boxes, language_cls=None):
        batch, num_tokens, hidden_dim = search_feat.shape
        grid_hw = int(round(math.sqrt(num_tokens)))
        if grid_hw * grid_hw != num_tokens:
            raise ValueError("Search token count must be a square number.")

        feat = self.feat_norm(search_feat)
        attn = attn_l2s.view(batch, -1)
        attn = self._normalize_scores(attn)

        semantic_token = self.semantic_token_base.expand(batch, -1, -1).squeeze(1)
        if language_cls is not None:
            semantic_token = semantic_token + self.semantic_lang_proj(language_cls)

        in_mask = self._build_box_mask(boxes, grid_hw, feat.device)
        out_mask = 1.0 - in_mask
        empty_mask = in_mask.sum(dim=1, keepdim=True) <= 0
        if empty_mask.any():
            in_mask = torch.where(empty_mask, torch.ones_like(in_mask), in_mask)
            out_mask = torch.where(empty_mask, torch.zeros_like(out_mask), out_mask)

        pos_prior = self._gaussian_prior(boxes, grid_hw, feat.device)
        target_logits = self.target_scorer(feat).squeeze(-1)
        target_logits = target_logits + self.target_attn_scale * attn + self.target_pos_scale * pos_prior
        target_weight = self._masked_softmax(target_logits, in_mask)
        target_proto = self._weighted_sum(feat, target_weight)
        # Ground the semantic token with a detached in-box visual summary so it
        # remains language-conditioned while still aligning to the target region.
        visual_seed = self._weighted_sum(feat, target_weight.detach())
        semantic_token = self.semantic_token_norm(semantic_token + self.semantic_visual_proj(visual_seed))

        sim_to_target = F.cosine_similarity(feat, target_proto.unsqueeze(1), dim=-1)
        split_signal = sim_to_target.detach() + attn

        distractor_logits = self.distractor_scorer(feat).squeeze(-1)
        distractor_logits = distractor_logits + self.split_sim_scale * sim_to_target + self.split_attn_scale * attn
        background_logits = self.background_scorer(feat).squeeze(-1)
        background_logits = background_logits - self.split_sim_scale * sim_to_target

        distractor_weight = self._masked_softmax(distractor_logits, out_mask)
        background_weight = self._masked_softmax(background_logits, out_mask)

        distractor_proto = self._weighted_sum(feat, distractor_weight)
        background_proto = self._weighted_sum(feat, background_weight)

        distractor_sim = F.cosine_similarity(feat, distractor_proto.unsqueeze(1), dim=-1)
        background_sim = F.cosine_similarity(feat, background_proto.unsqueeze(1), dim=-1)
        neg_sim = torch.maximum(distractor_sim, background_sim).clamp_min(0.0)
        token_logits = target_logits - neg_sim
        token_score = torch.sigmoid(token_logits)

        masked_token_logits = token_logits.masked_fill(in_mask <= 0, float("-inf"))
        keep_k = min(self.keep_topk, num_tokens)
        topk_logits, token_idx = torch.topk(masked_token_logits, k=keep_k, dim=1)
        quality_logit = topk_logits.mean(dim=1)
        quality_score = torch.sigmoid(quality_logit)
        gather_idx = token_idx.unsqueeze(-1).expand(-1, -1, hidden_dim)
        template_token_feat = torch.gather(feat, 1, gather_idx)

        pseudo_distractor = self._build_pseudo_distractor(split_signal, out_mask)

        return {
            "semantic_token": semantic_token,
            "target_proto": target_proto,
            "distractor_proto": distractor_proto,
            "background_proto": background_proto,
            "proto_patch_feat": feat,
            "target_logits": target_logits,
            "distractor_logits": distractor_logits,
            "background_logits": background_logits,
            "token_logits": token_logits,
            "token_score": token_score,
            "template_quality_logit": quality_logit,
            "template_quality": quality_score,
            "template_token_idx": token_idx,
            "template_token_feat": template_token_feat,
            "in_mask": in_mask,
            "out_mask": out_mask,
            "pos_prior": pos_prior,
            "pseudo_distractor_mask": pseudo_distractor,
        }
