import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageGuidedTokenEmphasizer(nn.Module):
    """Language-conditioned token keep predictor for visual-language TE variants."""

    def __init__(self, embed_dim, hidden_dim=256, hard=False, tau=1.0,
                 lang_residual_beta=0.1, keep_vl_source="global",
                 proto_topk_target=4, proto_topk_negative=8,
                 proto_contrast_tau=0.2, safe_confirm_gamma=0.35,
                 safe_confirm_tau=0.0, safe_confirm_max=0.25,
                 negative_gate_scale=8.0, negative_gate_floor=0.05,
                 word_weight_tau=0.07, word_template_weight=1.0,
                 word_search_weight=0.5, word_learned_weight=0.1):
        super().__init__()
        self.hard = hard
        self.tau = tau
        self.lang_residual_beta = lang_residual_beta
        self.keep_vl_source = str(keep_vl_source).lower()
        if self.keep_vl_source not in (
                "global", "template_match", "safe_multi_proto",
                "word_safe_multi_proto", "word_direct"):
            raise ValueError("Unsupported keep_vl_source: {}".format(keep_vl_source))
        self.proto_topk_target = max(1, int(proto_topk_target))
        self.proto_topk_negative = max(1, int(proto_topk_negative))
        self.proto_contrast_tau = max(float(proto_contrast_tau), 1e-6)
        self.safe_confirm_gamma = float(safe_confirm_gamma)
        self.safe_confirm_tau = float(safe_confirm_tau)
        self.safe_confirm_max = max(float(safe_confirm_max), 0.0)
        self.negative_gate_scale = float(negative_gate_scale)
        self.negative_gate_floor = min(max(float(negative_gate_floor), 0.0), 1.0)
        self.word_weight_tau = max(float(word_weight_tau), 1e-6)
        self.word_template_weight = float(word_template_weight)
        self.word_search_weight = float(word_search_weight)
        self.word_learned_weight = float(word_learned_weight)

        self.visual_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.lang_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.visual_score_head = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.lang_score_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.word_score_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.word_score_head.weight)
        nn.init.zeros_(self.word_score_head.bias)

    def _binary_keep(self, logits):
        tau = max(float(self.tau), 1e-6)
        if self.training:
            decision = F.gumbel_softmax(logits, tau=tau, hard=self.hard, dim=-1)
        else:
            decision = F.softmax(logits / tau, dim=-1)
        probs = F.softmax(logits / tau, dim=-1)
        return probs, decision[..., 0:1]

    @staticmethod
    def _keep_to_logits(keep):
        keep = keep.clamp(min=1e-6, max=1.0 - 1e-6)
        return torch.cat([keep.log(), (1.0 - keep).log()], dim=-1)

    @staticmethod
    def _unit(tokens):
        return F.normalize(tokens, dim=-1, eps=1e-6)

    @staticmethod
    def _weighted_mean(tokens, weights):
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (tokens * weights).sum(dim=1, keepdim=True) / denom

    @staticmethod
    def _prepare_mask(tokens, mask):
        if mask is None:
            return torch.ones(tokens.shape[:2] + (1,), dtype=tokens.dtype, device=tokens.device)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        return mask.to(dtype=tokens.dtype, device=tokens.device)

    @staticmethod
    def _content_lang_mask(lang_mask):
        mask = lang_mask.clone()
        if mask.shape[1] <= 2:
            return mask
        content = mask.clone()
        content[:, 0, :] = 0
        lengths = (mask.squeeze(-1) > 0).sum(dim=1)
        sep_idx = (lengths - 1).clamp(min=0)
        content.scatter_(1, sep_idx.view(-1, 1, 1), 0)
        empty = content.sum(dim=1, keepdim=True) <= 0
        return torch.where(empty, mask, content)

    def _language_context(self, lang_tokens, lang_mask=None):
        mask = self._prepare_mask(lang_tokens, lang_mask)
        return self._weighted_mean(lang_tokens, mask)

    def _score_language(self, lang_tokens, visual_context, prev_decision_l, lang_mask=None):
        lang_mask = self._prepare_mask(lang_tokens, lang_mask)
        lang = self.lang_proj(lang_tokens)
        visual = self.visual_proj(visual_context).expand(-1, lang.shape[1], -1)
        relation = torch.cat([lang, visual, lang * visual, (lang - visual).abs()], dim=-1)
        logits = self.lang_score_head(relation)
        probs, keep = self._binary_keep(logits)
        keep = keep * prev_decision_l * lang_mask
        base_context = self._language_context(lang_tokens, lang_mask)
        refined_context = self._weighted_mean(lang_tokens, keep)
        beta = float(self.lang_residual_beta)
        lang_context = base_context + beta * (refined_context - base_context)
        return logits, probs, keep, lang_context

    def _target_prototype(self, template_tokens, prev_decision_z, lang_context):
        template = self.visual_proj(template_tokens)
        lang = self.lang_proj(lang_context)
        scale = template.shape[-1] ** -0.5
        affinity = (template * lang).sum(dim=-1, keepdim=True) * scale
        affinity = affinity + prev_decision_z.clamp_min(1e-6).log()
        weights = torch.softmax(affinity, dim=1)
        return self._weighted_mean(template_tokens, weights)

    def _score_visual(self, visual_tokens, prev_decision, lang_context, reference_context=None):
        visual = self.visual_proj(visual_tokens)
        if reference_context is None:
            global_visual = self._weighted_mean(visual, prev_decision).expand_as(visual)
            lang = self.lang_proj(lang_context).expand_as(visual)
            relation = torch.cat([visual, global_visual, lang, visual * lang, (visual - lang).abs()], dim=-1)
        else:
            reference = self.visual_proj(reference_context).expand_as(visual)
            lang = self.lang_proj(lang_context).expand_as(visual)
            relation = torch.cat([visual, reference, lang, visual * reference, (visual - reference).abs()], dim=-1)
        logits = self.visual_score_head(relation)
        probs, keep = self._binary_keep(logits)
        keep = keep * prev_decision
        return logits, probs, keep

    @staticmethod
    def _gather_by_indices(tokens, indices):
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        return tokens.gather(1, gather_idx)

    @staticmethod
    def _masked_topk(tokens, scores, k, largest=True, exclude_indices=None):
        scores = scores.squeeze(-1)
        k = max(1, min(int(k), scores.shape[1]))
        if exclude_indices is not None and exclude_indices.numel() > 0:
            exclude = torch.zeros_like(scores, dtype=torch.bool)
            exclude.scatter_(1, exclude_indices, True)
            fill = torch.finfo(scores.dtype).min if largest else torch.finfo(scores.dtype).max
            scores = scores.masked_fill(exclude, fill)
        indices = torch.topk(scores, k, dim=1, largest=largest).indices
        return LanguageGuidedTokenEmphasizer._gather_by_indices(tokens, indices), indices

    def _max_similarity(self, search_tokens, prototype_tokens):
        sim = torch.matmul(
            self._unit(search_tokens),
            self._unit(prototype_tokens).transpose(1, 2))
        return sim.max(dim=-1, keepdim=True).values

    def _negative_gate_from_direct(self, negative_score, direct_keep, prev_decision_x):
        active = prev_decision_x.clamp(min=0.0, max=1.0)
        target_weight = (direct_keep * active).clamp_min(0.0)
        background_weight = ((1.0 - direct_keep) * active).clamp_min(0.0)
        target_mean = self._weighted_mean(negative_score, target_weight)
        background_mean = self._weighted_mean(negative_score, background_weight)
        gate = torch.sigmoid(self.negative_gate_scale * (background_mean - target_mean))
        return gate.clamp(min=self.negative_gate_floor, max=1.0)

    def _smooth_peak_gap(self, sim_map, dim=1):
        tau = self.word_weight_tau
        smooth_peak = tau * torch.logsumexp(sim_map / tau, dim=dim)
        smooth_peak = smooth_peak - tau * math.log(max(sim_map.shape[dim], 1))
        return smooth_peak - sim_map.mean(dim=dim)

    def _word_level_scores(self, lang_tokens, template_tokens, search_tokens, lang_mask,
                           word_reliability=None):
        lang = self.lang_proj(lang_tokens)
        template = self.visual_proj(template_tokens)
        search = self.visual_proj(search_tokens)
        sim_z = torch.matmul(self._unit(template), self._unit(lang).transpose(1, 2))
        sim_x = torch.matmul(self._unit(search), self._unit(lang).transpose(1, 2))

        content_mask = self._content_lang_mask(lang_mask).squeeze(-1).bool()
        template_gap = self._smooth_peak_gap(sim_z, dim=1)
        search_gap = self._smooth_peak_gap(sim_x, dim=1)
        learned = self.word_score_head(lang).squeeze(-1)
        word_logits = (
            self.word_template_weight * template_gap
            + self.word_search_weight * search_gap
            + self.word_learned_weight * learned
        )
        word_logits = word_logits.masked_fill(~content_mask, -1e4)
        word_weights = torch.softmax(word_logits / self.word_weight_tau, dim=-1)
        word_weights = word_weights * content_mask.to(dtype=word_weights.dtype)
        reliability = None
        if word_reliability is not None:
            reliability = word_reliability
            if reliability.dim() == 3:
                reliability = reliability.squeeze(-1)
            reliability = reliability.to(device=word_weights.device, dtype=word_weights.dtype)
            if reliability.shape != word_weights.shape:
                raise ValueError(
                    "word_reliability shape {} must match word weights shape {}".format(
                        tuple(reliability.shape), tuple(word_weights.shape)))
            reliability = reliability.clamp_min(0.0)
            reliability = reliability * content_mask.to(dtype=reliability.dtype)
            word_weights = word_weights * reliability
        word_weights = word_weights / word_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        template_score = (sim_z * word_weights.unsqueeze(1)).sum(dim=-1, keepdim=True)
        direct_score = (sim_x * word_weights.unsqueeze(1)).sum(dim=-1, keepdim=True)
        if reliability is None:
            reliability = content_mask.to(dtype=word_weights.dtype)
        return template, search, template_score, direct_score, word_weights, sim_z, sim_x, reliability

    def _score_search_word_direct(self, lang_tokens, template_tokens, search_tokens,
                                  prev_decision_z, prev_decision_x, lang_mask,
                                  word_reliability=None):
        _, _, template_score, direct_score, word_weights, sim_z, sim_x, reliability = self._word_level_scores(
            lang_tokens, template_tokens, search_tokens, lang_mask, word_reliability=word_reliability)

        centered_template = template_score - template_score.mean(dim=1, keepdim=True)
        centered_direct = direct_score - direct_score.mean(dim=1, keepdim=True)
        template_keep = torch.sigmoid(centered_template / self.proto_contrast_tau) * prev_decision_z
        keep = torch.sigmoid(centered_direct / self.proto_contrast_tau) * prev_decision_x

        logits = self._keep_to_logits(keep)
        probs = torch.cat([keep, 1.0 - keep], dim=-1)
        z_logits = self._keep_to_logits(template_keep)
        z_probs = torch.cat([template_keep, 1.0 - template_keep], dim=-1)
        aux = {
            "word_level_template_scores": template_score,
            "word_level_direct_scores": direct_score,
            "word_level_weights": word_weights,
            "word_level_reliability": reliability,
            "word_level_template_token_scores": sim_z,
            "word_level_search_token_scores": sim_x,
        }
        return z_logits, z_probs, template_keep, logits, probs, keep, aux

    def _score_search_word_safe_multi_proto(self, lang_tokens, template_tokens, search_tokens,
                                            prev_decision_z, prev_decision_x, lang_mask,
                                            word_reliability=None):
        template, search, template_score, direct_score, word_weights, sim_z, sim_x, reliability = self._word_level_scores(
            lang_tokens, template_tokens, search_tokens, lang_mask, word_reliability=word_reliability)

        centered_template = template_score - template_score.mean(dim=1, keepdim=True)
        centered_direct = direct_score - direct_score.mean(dim=1, keepdim=True)
        template_keep = torch.sigmoid(centered_template / self.proto_contrast_tau) * prev_decision_z
        direct_keep = torch.sigmoid(centered_direct / self.proto_contrast_tau) * prev_decision_x

        target_scores = template_score + prev_decision_z.clamp_min(1e-6).log()
        target_proto, target_idx = self._masked_topk(
            template, target_scores, self.proto_topk_target, largest=True)
        context_proto, _ = self._masked_topk(
            template, template_score, self.proto_topk_negative, largest=True,
            exclude_indices=target_idx)
        background_proto, _ = self._masked_topk(
            template, template_score, self.proto_topk_negative, largest=False,
            exclude_indices=target_idx)
        distractor_score = prev_decision_z - template_score
        distractor_proto, _ = self._masked_topk(
            template, distractor_score, self.proto_topk_negative, largest=True,
            exclude_indices=target_idx)

        target = self._max_similarity(search, target_proto)
        context = self._max_similarity(search, context_proto)
        background = self._max_similarity(search, background_proto)
        distractor = self._max_similarity(search, distractor_proto)

        context_gate = self._negative_gate_from_direct(context, direct_keep, prev_decision_x)
        background_gate = self._negative_gate_from_direct(background, direct_keep, prev_decision_x)
        distractor_gate = self._negative_gate_from_direct(distractor, direct_keep, prev_decision_x)
        safe_negative = torch.stack([
            context * context_gate,
            background * background_gate,
            distractor * distractor_gate,
            torch.zeros_like(target),
        ], dim=0).max(dim=0).values
        safe_margin = target - safe_negative
        confirm = F.relu(safe_margin - self.safe_confirm_tau).clamp(max=self.safe_confirm_max)
        keep = (direct_keep + self.safe_confirm_gamma * confirm).clamp(min=0.0, max=1.0)
        keep = keep * prev_decision_x
        logits = self._keep_to_logits(keep)
        probs = torch.cat([keep, 1.0 - keep], dim=-1)
        z_logits = self._keep_to_logits(template_keep)
        z_probs = torch.cat([template_keep, 1.0 - template_keep], dim=-1)
        aux = {
            "safe_proto_target_scores": target,
            "safe_proto_negative_scores": safe_negative,
            "safe_proto_margins": safe_margin,
            "word_level_template_scores": template_score,
            "word_level_direct_scores": direct_score,
            "word_level_weights": word_weights,
            "word_level_reliability": reliability,
            "word_level_template_token_scores": sim_z,
            "word_level_search_token_scores": sim_x,
        }
        return z_logits, z_probs, template_keep, logits, probs, keep, aux

    def _score_search_safe_multi_proto(self, template_tokens, search_tokens,
                                       prev_decision_z, prev_decision_x,
                                       lang_context):
        direct_logits, _, direct_keep = self._score_visual(
            search_tokens, prev_decision_x, lang_context)

        template = self.visual_proj(template_tokens)
        search = self.visual_proj(search_tokens)
        lang = self.lang_proj(lang_context)
        lang_unit = self._unit(lang)
        affinity = (self._unit(template) * lang_unit).sum(dim=-1, keepdim=True)

        target_scores = affinity + prev_decision_z.clamp_min(1e-6).log()
        target_proto, target_idx = self._masked_topk(
            template, target_scores, self.proto_topk_target, largest=True)
        context_proto, _ = self._masked_topk(
            template, affinity, self.proto_topk_negative, largest=True,
            exclude_indices=target_idx)
        background_proto, _ = self._masked_topk(
            template, affinity, self.proto_topk_negative, largest=False,
            exclude_indices=target_idx)
        distractor_score = prev_decision_z - affinity
        distractor_proto, _ = self._masked_topk(
            template, distractor_score, self.proto_topk_negative, largest=True,
            exclude_indices=target_idx)

        target = self._max_similarity(search, target_proto)
        context = self._max_similarity(search, context_proto)
        background = self._max_similarity(search, background_proto)
        distractor = self._max_similarity(search, distractor_proto)

        context_gate = self._negative_gate_from_direct(context, direct_keep, prev_decision_x)
        background_gate = self._negative_gate_from_direct(background, direct_keep, prev_decision_x)
        distractor_gate = self._negative_gate_from_direct(distractor, direct_keep, prev_decision_x)
        safe_negative = torch.stack([
            context * context_gate,
            background * background_gate,
            distractor * distractor_gate,
            torch.zeros_like(target),
        ], dim=0).max(dim=0).values
        safe_margin = target - safe_negative
        confirm = F.relu(safe_margin - self.safe_confirm_tau).clamp(max=self.safe_confirm_max)
        keep = (direct_keep + self.safe_confirm_gamma * confirm).clamp(min=0.0, max=1.0)
        keep = keep * prev_decision_x
        logits = self._keep_to_logits(keep)
        probs = torch.cat([keep, 1.0 - keep], dim=-1)
        aux = {
            "safe_proto_target_scores": target,
            "safe_proto_negative_scores": safe_negative,
            "safe_proto_margins": safe_margin,
        }
        return logits, probs, keep, aux

    def forward(self, lang_tokens, template_tokens, search_tokens,
                prev_decision_l, prev_decision_z, prev_decision_x,
                keep_vl=True, keep_lv=True, bidir_mode="sequential",
                lang_mask=None, word_reliability=None):
        out = {}
        lang_mask = self._prepare_mask(lang_tokens, lang_mask)
        raw_lang_context = self._language_context(lang_tokens, lang_mask)
        if keep_lv:
            visual_tokens = torch.cat([template_tokens, search_tokens], dim=1)
            visual_decision = torch.cat([prev_decision_z, prev_decision_x], dim=1)
            visual_context = self._weighted_mean(visual_tokens, visual_decision)
            l_logits, l_probs, prev_decision_l, lang_context = self._score_language(
                lang_tokens, visual_context, prev_decision_l, lang_mask=lang_mask)
            out.update({
                "language_logits": l_logits,
                "language_probs": l_probs,
                "language_decision": prev_decision_l,
                "language_context": lang_context,
            })
        else:
            lang_context = raw_lang_context
            out["language_context"] = lang_context

        if keep_vl:
            if keep_lv and str(bidir_mode).lower() == "parallel":
                lang_context = raw_lang_context
                out["visual_language_context"] = lang_context
            elif str(bidir_mode).lower() not in ("sequential", "parallel"):
                raise ValueError("Unsupported bidir_mode: {}".format(bidir_mode))
            if self.keep_vl_source == "word_direct":
                z_logits, z_probs, prev_decision_z, x_logits, x_probs, prev_decision_x, proto_aux = (
                    self._score_search_word_direct(
                        lang_tokens, template_tokens, search_tokens,
                        prev_decision_z, prev_decision_x, lang_mask,
                        word_reliability=word_reliability))
                out.update(proto_aux)
            elif self.keep_vl_source == "word_safe_multi_proto":
                z_logits, z_probs, prev_decision_z, x_logits, x_probs, prev_decision_x, proto_aux = (
                    self._score_search_word_safe_multi_proto(
                        lang_tokens, template_tokens, search_tokens,
                        prev_decision_z, prev_decision_x, lang_mask,
                        word_reliability=word_reliability))
                out.update(proto_aux)
            else:
                z_logits, z_probs, prev_decision_z = self._score_visual(
                    template_tokens, prev_decision_z, lang_context)
                search_reference = None
                if self.keep_vl_source == "template_match":
                    search_reference = self._target_prototype(template_tokens, prev_decision_z, lang_context)
                if self.keep_vl_source == "safe_multi_proto":
                    x_logits, x_probs, prev_decision_x, proto_aux = self._score_search_safe_multi_proto(
                        template_tokens, search_tokens, prev_decision_z, prev_decision_x, lang_context)
                    out.update(proto_aux)
                else:
                    x_logits, x_probs, prev_decision_x = self._score_visual(
                        search_tokens, prev_decision_x, lang_context, reference_context=search_reference)
            out.update({
                "template_logits": z_logits,
                "template_probs": z_probs,
                "template_decision": prev_decision_z,
                "search_logits": x_logits,
                "search_probs": x_probs,
                "search_decision": prev_decision_x,
            })

        return out
