from . import BaseActor
from lib.utils.misc import NestedTensor, interpolate
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import math
import torch
import torch.nn.functional as F
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


def _actor_masked_mean(x, mask):
    if mask is None:
        return x.mean(dim=1)
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    mask = mask.to(device=x.device, dtype=x.dtype)
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)



class DUTrackActor(BaseActor):
    """ Actor for training ODTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        template_list = []
        search_list = []

        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_list.append(template_img_i)

        for i in range(self.settings.num_search):
            search_img_i = data['search_images'][i].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
            search_list.append(search_img_i)
            
        box_mask_z = []
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            for i in range(self.settings.num_template):
                box_mask_z.append(generate_mask_cond(self.cfg, template_list[i].shape[0], template_list[i].device,
                                                    data['template_anno'][i]))
            box_mask_z = torch.cat(box_mask_z, dim=1)

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])



        language_token_states, language_token_masks, language_state_diagnostics = \
            self._build_language_state_training_inputs(data, search_list[0].device)

        out_dict = self.net(template=template_list,
                            search=search_list,
                            descript=data['language_description'],
                            language_token_state=language_token_states,
                            language_token_mask=language_token_masks,
)
        if language_state_diagnostics is not None:
            for idx, diagnostics in enumerate(language_state_diagnostics):
                if idx < len(out_dict):
                    out_dict[idx]["language_state_diagnostics"] = diagnostics
                    if diagnostics.get("_prev_tokens", None) is not None:
                        out_dict[idx]["language_state_aux"] = {
                            "prev_tokens": diagnostics.get("_prev_tokens"),
                            "state_tokens": diagnostics.get("_state_tokens"),
                            "mask": diagnostics.get("_mask"),
                            "anchor_raw_tokens": diagnostics.get("_anchor_tokens"),
                            "anchor_raw_mask": diagnostics.get("_anchor_mask"),
                            "anchor_aligned_tokens": diagnostics.get("_anchor_aligned_tokens"),
                            "candidate_aligned_tokens": diagnostics.get("_candidate_aligned_tokens"),
                            "candidate_weight": diagnostics.get("_candidate_weight"),
                            "candidate_absorb_logit": diagnostics.get("_candidate_absorb_logit"),
                            "prev_weight": diagnostics.get("_prev_weight"),
                            "anchor_weight": diagnostics.get("_anchor_weight"),
                        }

        return out_dict

    def _search_descriptions(self, data):
        descriptions = data.get("language_description", None)
        if descriptions is None:
            return None
        if isinstance(descriptions, (list, tuple)):
            return list(descriptions)
        return [descriptions]

    def _encode_language_tokens(self, descriptions):
        if not isinstance(descriptions, (list, tuple)):
            descriptions = list(descriptions)
        detach_text = bool(getattr(self.cfg.TRAIN, "LANGUAGE_STATE_DETACH_TEXT", True))
        if detach_text:
            with torch.no_grad():
                tokens, mask = self.net.backbone._l_feat(list(descriptions))
            return tokens.detach(), mask.detach()
        return self.net.backbone._l_feat(list(descriptions))

    def _language_state_visual_evidence(self, data, search_idx, device, dtype):
        mode = str(getattr(self.cfg.TRAIN, "LANGUAGE_STATE_VISUAL_EVIDENCE", "gt_motion")).lower()
        search_box = data["search_anno"][search_idx].to(device=device, dtype=dtype)
        batch_size = search_box.shape[0]
        if mode in ("none", "zero", "zeros", "off"):
            return torch.zeros(batch_size, 8, device=device, dtype=dtype)

        template_box = data["template_anno"][-1].to(device=device, dtype=dtype)
        search_center = search_box[:, :2] + 0.5 * search_box[:, 2:4]
        template_center = template_box[:, :2] + 0.5 * template_box[:, 2:4]
        center_motion = (search_center - template_center).norm(dim=1, keepdim=True) / math.sqrt(2.0)

        search_area = (search_box[:, 2] * search_box[:, 3]).clamp_min(1e-6)
        template_area = (template_box[:, 2] * template_box[:, 3]).clamp_min(1e-6)
        scale_change = torch.log(search_area / template_area).abs().unsqueeze(1)

        color_change = torch.zeros_like(center_motion)
        score_peak_gap = torch.zeros_like(center_motion)
        score_entropy = torch.zeros_like(center_motion)
        box_jump = center_motion
        deploy_delta = torch.zeros_like(center_motion)
        partial_delta = torch.zeros_like(center_motion)
        return torch.cat([
            center_motion,
            scale_change,
            color_change,
            score_peak_gap,
            score_entropy,
            box_jump,
            deploy_delta,
            partial_delta,
        ], dim=1)

    def _build_gsb_language_inputs(self, data, device):
        """GSB path: encode descriptions, pass raw tokens (no updater).

        The model handles pooling and state management internally.
        """
        descriptions = self._search_descriptions(data)
        if not descriptions:
            return None, None, None
        tokens_list = []
        masks_list = []
        for desc in descriptions:
            tokens, mask = self._encode_language_tokens(desc)
            tokens_list.append(tokens)
            masks_list.append(mask)
        return tokens_list, masks_list, None

    def _build_language_state_training_inputs(self, data, device):
        te_cfg = getattr(self.cfg.MODEL, "TE", None)
        gsb_enabled = bool(getattr(te_cfg, "GLOBAL_SCORE_BIAS_ENABLE", False))
        if gsb_enabled:
            return self._build_gsb_language_inputs(data, device)
        if not bool(getattr(te_cfg, "LANGUAGE_STATE_ENABLE", False)):
            return None, None, None
        if not bool(getattr(self.cfg.TRAIN, "LANGUAGE_STATE_TRAIN_ENABLE", False)):
            return None, None, None
        updater = getattr(self.net.backbone, "language_state_updater", None)
        if updater is None:
            return None, None, None

        descriptions = self._search_descriptions(data)
        if not descriptions:
            return None, None, None

        states = []
        masks = []
        diagnostics = []
        anchor_desc = descriptions[0]
        anchor_tokens, anchor_mask = self._encode_language_tokens(anchor_desc)
        alignment_mode = str(getattr(te_cfg, "LANGUAGE_STATE_ALIGNMENT", "position")).lower()
        for search_idx, candidate_desc in enumerate(descriptions):
            prev_desc = descriptions[search_idx - 1] if search_idx > 0 else anchor_desc
            prev_tokens, prev_mask = self._encode_language_tokens(prev_desc)
            candidate_tokens, candidate_mask = self._encode_language_tokens(candidate_desc)
            visual_evidence = self._language_state_visual_evidence(
                data, search_idx, anchor_tokens.device, anchor_tokens.dtype)
            state_tokens, diag = updater(
                anchor_tokens, prev_tokens, candidate_tokens,
                anchor_mask=anchor_mask, prev_mask=prev_mask,
                candidate_mask=candidate_mask,
                visual_evidence=visual_evidence)
            diag = dict(diag)
            diag["visual_center_motion_mean"] = visual_evidence[:, 0].detach().mean()
            diag["visual_scale_change_mean"] = visual_evidence[:, 1].detach().mean()
            diag["visual_box_jump_mean"] = visual_evidence[:, 5].detach().mean()
            diag["_anchor_tokens"] = anchor_tokens.detach()
            diag["_anchor_mask"] = anchor_mask.detach() if anchor_mask is not None else None
            diag["_prev_tokens"] = prev_tokens.detach()
            diag["_state_tokens"] = state_tokens
            state_mask = prev_mask if alignment_mode == "cross_attn" else candidate_mask
            diag["_mask"] = state_mask.detach() if state_mask is not None else None
            states.append(state_tokens)
            masks.append(state_mask)
            diagnostics.append(diag)
        return states, masks, diagnostics

    def _search_token_centers(self, gt_bbox, num_tokens, device):
        feat_sz = int(num_tokens ** 0.5)
        if feat_sz * feat_sz != num_tokens:
            raise ValueError("Search token count must be a square grid, got {}".format(num_tokens))
        ys = (torch.arange(feat_sz, device=device, dtype=gt_bbox.dtype) + 0.5) / feat_sz
        xs = (torch.arange(feat_sz, device=device, dtype=gt_bbox.dtype) + 0.5) / feat_sz
        try:
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        except TypeError:
            yy, xx = torch.meshgrid(ys, xs)
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

    def _search_token_box_mask(self, gt_bbox, num_tokens, device):
        centers = self._search_token_centers(gt_bbox, num_tokens, device)
        x1 = gt_bbox[:, 0:1]
        y1 = gt_bbox[:, 1:2]
        x2 = x1 + gt_bbox[:, 2:3]
        y2 = y1 + gt_bbox[:, 3:4]
        return ((centers[None, :, 0] >= x1) & (centers[None, :, 0] <= x2) &
                (centers[None, :, 1] >= y1) & (centers[None, :, 1] <= y2))

    def _search_token_target(self, gt_bbox, num_tokens, device):
        centers = self._search_token_centers(gt_bbox, num_tokens, device)
        target_mode = str(getattr(getattr(self.cfg.MODEL, "TE", None), "AUX_SEARCH_TARGET", "box")).lower()
        if target_mode == "box":
            return self._search_token_box_mask(gt_bbox, num_tokens, device).to(dtype=gt_bbox.dtype)
        if target_mode in ("soft_center", "center", "gaussian"):
            x1 = gt_bbox[:, 0:1]
            y1 = gt_bbox[:, 1:2]
            cx = x1 + 0.5 * gt_bbox[:, 2:3]
            cy = y1 + 0.5 * gt_bbox[:, 3:4]
            sigma_scale = float(getattr(self.cfg.MODEL.TE, "AUX_CENTER_SIGMA_SCALE", 0.35))
            min_sigma = float(getattr(self.cfg.MODEL.TE, "AUX_CENTER_MIN_SIGMA", 0.02))
            sigma_x = (gt_bbox[:, 2:3] * sigma_scale).clamp_min(min_sigma)
            sigma_y = (gt_bbox[:, 3:4] * sigma_scale).clamp_min(min_sigma)
            dx = (centers[None, :, 0] - cx) / sigma_x
            dy = (centers[None, :, 1] - cy) / sigma_y
            return torch.exp(-0.5 * (dx * dx + dy * dy)).to(dtype=gt_bbox.dtype)
        raise ValueError("Unsupported TE AUX_SEARCH_TARGET: {}".format(target_mode))

    def _search_token_center_quality(self, gt_bbox, num_tokens, device):
        centers = self._search_token_centers(gt_bbox, num_tokens, device)
        x1 = gt_bbox[:, 0:1]
        y1 = gt_bbox[:, 1:2]
        cx = x1 + 0.5 * gt_bbox[:, 2:3]
        cy = y1 + 0.5 * gt_bbox[:, 3:4]
        sigma_scale = float(getattr(self.cfg.MODEL.TE, "AUX_CENTER_SIGMA_SCALE", 0.35))
        min_sigma = float(getattr(self.cfg.MODEL.TE, "AUX_CENTER_MIN_SIGMA", 0.04))
        sigma_x = (gt_bbox[:, 2:3] * sigma_scale).clamp_min(min_sigma)
        sigma_y = (gt_bbox[:, 3:4] * sigma_scale).clamp_min(min_sigma)
        dx = (centers[None, :, 0] - cx) / sigma_x
        dy = (centers[None, :, 1] - cy) / sigma_y
        gaussian = torch.exp(-0.5 * (dx * dx + dy * dy)).to(dtype=gt_bbox.dtype)
        box_mask = self._search_token_box_mask(gt_bbox, num_tokens, device)
        base_quality = float(getattr(self.cfg.MODEL.TE, "AUX_CENTER_BASE_QUALITY", 0.3))
        base_quality = max(0.0, min(1.0, base_quality))
        quality = box_mask.to(dtype=gt_bbox.dtype) * (base_quality + (1.0 - base_quality) * gaussian)

        # Very small objects can fall between token centers. In that case keep a
        # single center-nearest positive token instead of producing no positives.
        empty_pos = quality.sum(dim=1, keepdim=True) <= 1e-6
        if empty_pos.any():
            fallback = torch.zeros_like(quality)
            fallback.scatter_(1, gaussian.argmax(dim=1, keepdim=True), 1.0)
            quality = torch.where(empty_pos, fallback, quality)
        return quality

    def _compute_te_search_rank_loss(self, margin, gt_bbox):
        keep = margin.sigmoid()
        pos_mask = self._search_token_box_mask(gt_bbox, margin.shape[1], margin.device)
        neg_mask = ~pos_mask
        pos_den = pos_mask.sum(dim=1).clamp_min(1).to(dtype=keep.dtype)
        pos_mean = (keep * pos_mask.to(dtype=keep.dtype)).sum(dim=1) / pos_den

        hard_neg_ratio = float(getattr(self.cfg.MODEL.TE, "AUX_HARD_NEG_RATIO", 0.1))
        neg_values = keep.masked_fill(~neg_mask, -1.0)
        neg_count = neg_mask.sum(dim=1).clamp_min(1)
        hard_k = torch.clamp((neg_count.float() * hard_neg_ratio).ceil().long(), min=1)
        max_k = int(hard_k.max().item())
        top_values = torch.topk(neg_values, max_k, dim=1).values
        rank_ids = torch.arange(max_k, device=margin.device)[None, :]
        valid_top = rank_ids < hard_k[:, None]
        hard_neg_mean = (top_values.clamp_min(0.0) * valid_top.to(dtype=keep.dtype)).sum(dim=1) / hard_k.to(dtype=keep.dtype)

        margin_value = float(getattr(self.cfg.MODEL.TE, "AUX_RANK_MARGIN", 0.2))
        rank_loss = F.relu(margin_value - pos_mean + hard_neg_mean).mean()

        area_weight = float(getattr(self.cfg.MODEL.TE, "AUX_AREA_WEIGHT", 0.0))
        if area_weight <= 0.0:
            return rank_loss
        area_multiplier = float(getattr(self.cfg.MODEL.TE, "AUX_AREA_MULTIPLIER", 1.5))
        area_min = float(getattr(self.cfg.MODEL.TE, "AUX_AREA_MIN", 0.02))
        area_max = float(getattr(self.cfg.MODEL.TE, "AUX_AREA_MAX", 0.5))
        target_area = (gt_bbox[:, 2] * gt_bbox[:, 3] * area_multiplier).clamp(min=area_min, max=area_max)
        area_loss = (keep.mean(dim=1) - target_area.to(dtype=keep.dtype)).abs().mean()
        return rank_loss + area_weight * area_loss

    def _compute_te_search_center_rank_loss(self, margin, gt_bbox):
        keep = margin.sigmoid()
        quality = self._search_token_center_quality(gt_bbox, margin.shape[1], margin.device)
        center_den = quality.sum(dim=1).clamp_min(1e-6)
        center_mean = (keep * quality).sum(dim=1) / center_den

        neg_mask = quality <= 0.0
        fallback_mask = quality <= quality.mean(dim=1, keepdim=True)
        neg_mask = neg_mask | (~neg_mask.any(dim=1, keepdim=True) & fallback_mask)
        neg_values = keep.masked_fill(~neg_mask, -1.0)
        neg_count = neg_mask.sum(dim=1).clamp_min(1)
        hard_neg_ratio = float(getattr(self.cfg.MODEL.TE, "AUX_HARD_NEG_RATIO", 0.1))
        hard_k = torch.clamp((neg_count.float() * hard_neg_ratio).ceil().long(), min=1)
        max_k = int(hard_k.max().item())
        top_values = torch.topk(neg_values, max_k, dim=1).values
        rank_ids = torch.arange(max_k, device=margin.device)[None, :]
        valid_top = rank_ids < hard_k[:, None]
        hard_neg_mean = (top_values.clamp_min(0.0) * valid_top.to(dtype=keep.dtype)).sum(dim=1) / hard_k.to(dtype=keep.dtype)

        margin_value = float(getattr(self.cfg.MODEL.TE, "AUX_RANK_MARGIN", 0.2))
        rank_loss = F.relu(margin_value - center_mean + hard_neg_mean).mean()

        area_weight = float(getattr(self.cfg.MODEL.TE, "AUX_AREA_WEIGHT", 0.0))
        if area_weight <= 0.0:
            return rank_loss
        area_multiplier = float(getattr(self.cfg.MODEL.TE, "AUX_AREA_MULTIPLIER", 1.5))
        area_min = float(getattr(self.cfg.MODEL.TE, "AUX_AREA_MIN", 0.02))
        area_max = float(getattr(self.cfg.MODEL.TE, "AUX_AREA_MAX", 0.5))
        target_area = (quality.mean(dim=1) * area_multiplier).clamp(min=area_min, max=area_max)
        area_loss = (keep.mean(dim=1) - target_area.to(dtype=keep.dtype)).abs().mean()
        return rank_loss + area_weight * area_loss

    def _compute_te_search_loss(self, pred, gt_bbox):
        logits_list = pred.get("lang_te_search_logits", None)
        if not logits_list:
            return None
        losses = []
        loss_mode = str(getattr(getattr(self.cfg.MODEL, "TE", None), "AUX_SEARCH_LOSS", "bce")).lower()
        for logits in logits_list:
            margin = logits[..., 0] - logits[..., 1]
            if loss_mode == "rank":
                losses.append(self._compute_te_search_rank_loss(margin, gt_bbox.to(margin.device)))
                continue
            if loss_mode in ("center_rank", "quality_center_rank", "quality_rank"):
                losses.append(self._compute_te_search_center_rank_loss(margin, gt_bbox.to(margin.device)))
                continue
            if loss_mode != "bce":
                raise ValueError("Unsupported TE AUX_SEARCH_LOSS: {}".format(loss_mode))
            target = self._search_token_target(gt_bbox.to(margin.device), margin.shape[1], margin.device)
            pos = target.sum(dim=1, keepdim=True).clamp_min(1.0)
            neg = (target.shape[1] - pos).clamp_min(1.0)
            pos_weight = (neg / pos).clamp(max=20.0)
            weight = 1.0 + target * (pos_weight - 1.0)
            losses.append(F.binary_cross_entropy_with_logits(margin, target, weight=weight))
        return torch.stack(losses).mean()

    def _te_aux_weight(self, data):
        te_cfg = getattr(self.cfg.MODEL, "TE", None)
        base_weight = float(getattr(te_cfg, "AUX_SEARCH_LOSS_WEIGHT", 0.0))
        if base_weight <= 0.0:
            return 0.0
        anneal = str(getattr(te_cfg, "AUX_SEARCH_LOSS_ANNEAL", "none")).lower()
        if anneal in ("none", "off", "false"):
            return base_weight
        end_weight = float(getattr(te_cfg, "AUX_SEARCH_LOSS_WEIGHT_END", 0.0))
        total_epochs = int(getattr(te_cfg, "AUX_SEARCH_ANNEAL_EPOCHS", 0))
        if total_epochs <= 0:
            total_epochs = int(getattr(self.cfg.TRAIN, "EPOCH", 1))
        epoch = int(data.get("epoch", 1)) if isinstance(data, dict) else 1
        if total_epochs <= 1:
            progress = 1.0
        else:
            progress = max(0.0, min(1.0, float(epoch - 1) / float(total_epochs - 1)))
        if anneal == "linear":
            return base_weight + (end_weight - base_weight) * progress
        if anneal == "cosine":
            return end_weight + (base_weight - end_weight) * 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError("Unsupported TE AUX_SEARCH_LOSS_ANNEAL: {}".format(anneal))

    def _score_aux_weight(self, data):
        te_cfg = getattr(self.cfg.MODEL, "TE", None)
        base_weight = float(getattr(te_cfg, "AUX_SCORE_LOSS_WEIGHT", 0.0))
        if base_weight <= 0.0:
            return 0.0
        anneal = str(getattr(te_cfg, "AUX_SCORE_LOSS_ANNEAL", "none")).lower()
        if anneal in ("none", "off", "false"):
            return base_weight
        end_weight = float(getattr(te_cfg, "AUX_SCORE_LOSS_WEIGHT_END", 0.0))
        total_epochs = int(getattr(te_cfg, "AUX_SCORE_ANNEAL_EPOCHS", 0))
        if total_epochs <= 0:
            total_epochs = int(getattr(self.cfg.TRAIN, "EPOCH", 1))
        epoch = int(data.get("epoch", 1)) if isinstance(data, dict) else 1
        if total_epochs <= 1:
            progress = 1.0
        else:
            progress = max(0.0, min(1.0, float(epoch - 1) / float(total_epochs - 1)))
        if anneal == "linear":
            return base_weight + (end_weight - base_weight) * progress
        if anneal == "cosine":
            return end_weight + (base_weight - end_weight) * 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError("Unsupported TE AUX_SCORE_LOSS_ANNEAL: {}".format(anneal))

    def _expand_score_prior_bias(self, bias, total_rows):
        if bias is None:
            return None
        if bias.dim() == 3:
            bias = bias.unsqueeze(1)
        if bias.dim() != 4:
            raise ValueError("score_prior_bias must have shape (B,1,H,W), got {}".format(tuple(bias.shape)))
        if bias.shape[0] == total_rows:
            return bias
        if total_rows % bias.shape[0] != 0:
            raise ValueError(
                "score_prior_bias batch {} cannot expand to score logits batch {}".format(
                    bias.shape[0], total_rows))
        repeat = total_rows // bias.shape[0]
        return bias[:, None].expand(
            bias.shape[0], repeat, bias.shape[1], bias.shape[2], bias.shape[3]).reshape(
            total_rows, bias.shape[1], bias.shape[2], bias.shape[3])

    def _masked_mean(self, values, mask):
        mask_f = mask.to(dtype=values.dtype)
        return (values * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)

    def _score_positive_mask(self, quality):
        te_cfg = getattr(self.cfg.MODEL, "TE", None)
        max_pos = int(getattr(te_cfg, "AUX_SCORE_POS_MAX_TOKENS", 4))
        max_pos = max(1, min(max_pos, quality.shape[1]))
        rel_quality = float(getattr(te_cfg, "AUX_SCORE_POS_QUALITY", 0.7))
        min_quality = float(getattr(te_cfg, "AUX_SCORE_POS_MIN_QUALITY", 0.0))
        top_values, top_indices = torch.topk(quality, max_pos, dim=1)
        max_quality = quality.max(dim=1, keepdim=True).values
        valid = (top_values > 0.0) & (top_values >= max_quality * rel_quality) & (top_values >= min_quality)
        empty = ~valid.any(dim=1, keepdim=True)
        valid = valid | (empty & (torch.arange(max_pos, device=quality.device)[None, :] == 0))
        pos_mask = torch.zeros_like(quality, dtype=torch.bool)
        pos_mask.scatter_(1, top_indices, valid)
        return pos_mask

    def _score_hard_negative_mask(self, base_logits, quality, pos_mask):
        te_cfg = getattr(self.cfg.MODEL, "TE", None)
        neg_quality_max = float(getattr(te_cfg, "AUX_SCORE_HARD_NEG_MAX_QUALITY", 0.0))
        neg_mask = (quality <= neg_quality_max) & (~pos_mask)
        fallback_neg = (quality <= quality.mean(dim=1, keepdim=True)) & (~pos_mask)
        neg_mask = torch.where(neg_mask.any(dim=1, keepdim=True), neg_mask, fallback_neg)
        neg_count = neg_mask.sum(dim=1).clamp_min(1)

        fixed_topk = int(getattr(te_cfg, "AUX_SCORE_HARD_NEG_TOPK", 8))
        if fixed_topk > 0:
            hard_k = torch.clamp(neg_count, max=fixed_topk)
        else:
            hard_ratio = float(getattr(te_cfg, "AUX_SCORE_HARD_NEG_RATIO", 0.02))
            hard_k = torch.clamp((neg_count.float() * hard_ratio).ceil().long(), min=1)

        max_k = int(hard_k.max().item())
        neg_values = base_logits.masked_fill(~neg_mask, torch.finfo(base_logits.dtype).min)
        top_indices = torch.topk(neg_values, max_k, dim=1).indices
        valid = torch.arange(max_k, device=base_logits.device)[None, :] < hard_k[:, None]
        hard_mask = torch.zeros_like(neg_mask)
        hard_mask.scatter_(1, top_indices, valid)
        return hard_mask

    def _repeat_token_mask(self, mask, total_rows):
        if mask.shape[0] == total_rows:
            return mask
        if total_rows % mask.shape[0] != 0:
            raise ValueError("Mask batch {} cannot expand to {}".format(mask.shape[0], total_rows))
        repeat = total_rows // mask.shape[0]
        return mask[:, None].expand(mask.shape[0], repeat, mask.shape[1]).reshape(total_rows, mask.shape[1])

    def _repeat_token_values(self, values, total_rows):
        if values.dim() == 3 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        if values.dim() != 2:
            raise ValueError("Token values must have shape (B,N) or (B,N,1), got {}".format(tuple(values.shape)))
        if values.shape[0] == total_rows:
            return values
        if total_rows % values.shape[0] != 0:
            raise ValueError("Token value batch {} cannot expand to {}".format(values.shape[0], total_rows))
        repeat = total_rows // values.shape[0]
        return values[:, None].expand(values.shape[0], repeat, values.shape[1]).reshape(total_rows, values.shape[1])

    def _repeat_batch_values(self, values, total_rows):
        if values.dim() == 0:
            values = values.view(1)
        if values.shape[0] == total_rows:
            return values
        if total_rows % values.shape[0] != 0:
            raise ValueError("Batch values {} cannot expand to {}".format(values.shape[0], total_rows))
        repeat = total_rows // values.shape[0]
        return values[:, None].expand(values.shape[0], repeat).reshape(total_rows, *values.shape[1:])

    def _repeat_query_maps(self, maps, total_rows):
        if maps.dim() != 3:
            raise ValueError("Query maps must have shape (B,K,N), got {}".format(tuple(maps.shape)))
        if maps.shape[0] == total_rows:
            return maps
        if total_rows % maps.shape[0] != 0:
            raise ValueError("Query map batch {} cannot expand to {}".format(maps.shape[0], total_rows))
        repeat = total_rows // maps.shape[0]
        return maps[:, None].expand(
            maps.shape[0], repeat, maps.shape[1], maps.shape[2]).reshape(
            total_rows, maps.shape[1], maps.shape[2])

    @staticmethod
    def _last_aux_map(pred, key):
        values = pred.get(key, None)
        if not values:
            return None
        return values[-1]

    def _compute_lmq_status(self, pred, total_rows, pos_mask, hard_neg_mask):
        maps = self._last_aux_map(pred, "lmq_query_prior_maps")
        status = {}
        if maps is None:
            return status
        maps = self._repeat_query_maps(maps, total_rows).to(device=pos_mask.device, dtype=torch.float32)
        pos = pos_mask[:, None, :].to(dtype=maps.dtype)
        neg = hard_neg_mask[:, None, :].to(dtype=maps.dtype)
        pos_mean = (maps * pos).sum(dim=-1) / pos.sum(dim=-1).clamp_min(1.0)
        neg_mean = (maps * neg).sum(dim=-1) / neg.sum(dim=-1).clamp_min(1.0)
        gap = pos_mean - neg_mean

        centered = maps - maps.mean(dim=-1, keepdim=True)
        normalized = F.normalize(centered, dim=-1, eps=1e-6)
        cosine = torch.matmul(normalized, normalized.transpose(1, 2))
        num_queries = cosine.shape[1]
        if num_queries > 1:
            pair_mask = ~torch.eye(num_queries, dtype=torch.bool, device=cosine.device)
            pair_values = cosine[:, pair_mask].view(cosine.shape[0], num_queries * (num_queries - 1))
            cosine_mean = pair_values.mean()
            cosine_max = pair_values.max()
        else:
            cosine_mean = cosine.new_tensor(0.0)
            cosine_max = cosine.new_tensor(0.0)

        fusion = self._last_aux_map(pred, "lmq_query_fusion_weights")
        if fusion is not None:
            fusion = self._repeat_token_values(fusion, total_rows).to(device=pos_mask.device, dtype=torch.float32)
            fusion_entropy = -(fusion.clamp_min(1e-6).log() * fusion).sum(dim=1).mean()
            fusion_max = fusion.max(dim=1).values.mean()
        else:
            fusion_entropy = maps.new_tensor(0.0)
            fusion_max = maps.new_tensor(0.0)

        status.update({
            "lmq_query_prior_gap_mean": gap.detach().mean(),
            "lmq_query_prior_gap_max": gap.detach().max(dim=1).values.mean(),
            "lmq_query_prior_gap_min": gap.detach().min(dim=1).values.mean(),
            "lmq_query_prior_cosine_mean": cosine_mean.detach(),
            "lmq_query_prior_cosine_max": cosine_max.detach(),
            "lmq_query_fusion_entropy": fusion_entropy.detach(),
            "lmq_query_fusion_max": fusion_max.detach(),
        })
        for query_idx in range(num_queries):
            status["lmq_query_prior_gap_q{}".format(query_idx)] = gap[:, query_idx].detach().mean()
        diagnostic_keys = {
            "lmq_query_seed_cosine_mean": "lmq_query_seed_cosine_mean",
            "lmq_query_seed_cosine_max": "lmq_query_seed_cosine_max",
            "lmq_query_lang_attn_cosine_mean": "lmq_query_lang_attn_cosine_mean",
            "lmq_query_lang_attn_cosine_max": "lmq_query_lang_attn_cosine_max",
            "lmq_query_lang_attn_entropy": "lmq_query_lang_attn_entropy",
            "lmq_query_lang_attn_max": "lmq_query_lang_attn_max",
            "lmq_pooled_query_cosine_mean": "lmq_pooled_query_cosine_mean",
            "lmq_pooled_query_cosine_max": "lmq_pooled_query_cosine_max",
            "lmq_query_vector_cosine_mean": "lmq_query_vector_cosine_mean",
            "lmq_query_vector_cosine_max": "lmq_query_vector_cosine_max",
            "lmq_query_map_between_std": "lmq_query_map_between_std",
            "lmq_prior_score_std": "lmq_prior_score_std",
            "lmq_query_search_attn_entropy": "lmq_query_search_attn_entropy",
            "lmq_query_search_attn_max": "lmq_query_search_attn_max",
            "lmq_decoder_query_delta_norm": "lmq_decoder_query_delta_norm",
        }
        for status_key, aux_key in diagnostic_keys.items():
            values = self._last_aux_map(pred, aux_key)
            if values is None:
                continue
            values = self._repeat_batch_values(values, total_rows).to(
                device=pos_mask.device, dtype=torch.float32)
            status[status_key] = values.detach().mean()
        return status

    def _compute_safe_proto_status(self, pred, total_rows, pos_mask, hard_neg_mask):
        target = self._last_aux_map(pred, "safe_proto_target_scores")
        negative = self._last_aux_map(pred, "safe_proto_negative_scores")
        margin = self._last_aux_map(pred, "safe_proto_margins")
        word_direct = self._last_aux_map(pred, "word_level_direct_scores")
        status = {}
        if word_direct is not None:
            word_direct = self._repeat_token_values(word_direct, total_rows).to(device=pos_mask.device, dtype=torch.float32)
            word_pos = self._masked_mean(word_direct, pos_mask)
            word_hardneg = self._masked_mean(word_direct, hard_neg_mask)
            status.update({
                "word_direct_score_on_pos": word_pos.detach().mean(),
                "word_direct_score_on_hardneg": word_hardneg.detach().mean(),
                "word_direct_gap": (word_pos - word_hardneg).detach().mean(),
                "word_direct_hard_case_ratio": (word_pos <= word_hardneg).detach().float().mean(),
            })
        if target is None or negative is None or margin is None:
            return status

        target = self._repeat_token_values(target, total_rows).to(device=pos_mask.device, dtype=torch.float32)
        negative = self._repeat_token_values(negative, total_rows).to(device=pos_mask.device, dtype=torch.float32)
        margin = self._repeat_token_values(margin, total_rows).to(device=pos_mask.device, dtype=torch.float32)

        target_pos = self._masked_mean(target, pos_mask)
        target_hardneg = self._masked_mean(target, hard_neg_mask)
        negative_pos = self._masked_mean(negative, pos_mask)
        negative_hardneg = self._masked_mean(negative, hard_neg_mask)
        margin_pos = self._masked_mean(margin, pos_mask)
        margin_hardneg = self._masked_mean(margin, hard_neg_mask)

        status.update({
            "target_proto_score_on_pos": target_pos.detach().mean(),
            "target_proto_score_on_hardneg": target_hardneg.detach().mean(),
            "negative_proto_score_on_pos": negative_pos.detach().mean(),
            "negative_proto_score_on_hardneg": negative_hardneg.detach().mean(),
            "safe_margin_pos": margin_pos.detach().mean(),
            "safe_margin_hardneg": margin_hardneg.detach().mean(),
            "safe_margin_gap": (margin_pos - margin_hardneg).detach().mean(),
            "safe_margin_hard_case_ratio": (margin_pos <= margin_hardneg).detach().float().mean(),
            "target_proto_hard_case_ratio": (target_pos <= target_hardneg).detach().float().mean(),
            "negative_proto_conflict_ratio": (negative_pos >= negative_hardneg).detach().float().mean(),
        })
        return status

    def _compute_score_prior_loss(self, pred, gt_bbox):
        required = ("score_map_logits_base", "score_prior_bias")
        if any(k not in pred or pred[k] is None for k in required):
            return None, {}

        base_logits = pred["score_map_logits_base"].detach()
        total_rows = base_logits.shape[0]
        bias = self._expand_score_prior_bias(pred["score_prior_bias"], total_rows)
        base_flat = base_logits.flatten(1)
        bias_flat = bias.flatten(1).to(dtype=base_flat.dtype, device=base_flat.device)
        score_flat = base_flat + bias_flat

        gt_bbox = gt_bbox.to(device=base_flat.device, dtype=base_flat.dtype)
        quality = self._search_token_center_quality(gt_bbox, base_flat.shape[1], base_flat.device)
        pos_mask = self._score_positive_mask(quality)
        pos_mask = self._repeat_token_mask(pos_mask, total_rows)
        quality = self._repeat_token_mask(quality, total_rows)
        hard_neg_mask = self._score_hard_negative_mask(base_flat, quality, pos_mask)

        pos_score = self._masked_mean(score_flat, pos_mask)
        hard_neg_score = self._masked_mean(score_flat, hard_neg_mask)
        margin = float(getattr(self.cfg.MODEL.TE, "AUX_SCORE_CORRECTIVE_MARGIN", 0.05))
        corrective_per_sample = F.relu(margin - pos_score + hard_neg_score)
        corrective_loss = corrective_per_sample.mean()

        pos_gain = self._masked_mean(bias_flat, pos_mask)
        hard_neg_gain = self._masked_mean(bias_flat, hard_neg_mask)
        gain_margin = float(getattr(self.cfg.MODEL.TE, "AUX_SCORE_GAIN_MARGIN", 0.01))
        prior_gain_per_sample = F.relu(gain_margin - pos_gain + hard_neg_gain)
        prior_gain_loss = prior_gain_per_sample.mean()

        gain_weight = float(getattr(self.cfg.MODEL.TE, "AUX_SCORE_GAIN_WEIGHT", 0.1))
        bias_l2_weight = float(getattr(self.cfg.MODEL.TE, "AUX_SCORE_BIAS_L2_WEIGHT", 0.001))
        bias_l2 = (bias_flat * bias_flat).mean()
        loss = corrective_loss + gain_weight * prior_gain_loss + bias_l2_weight * bias_l2

        status = {
            "score_rank_loss": corrective_loss.detach(),
            "score_corrective": corrective_loss.detach(),
            "score_prior_gain": prior_gain_loss.detach(),
            "score_bias_l2": bias_l2.detach(),
            "active_corrective_ratio": (corrective_per_sample.detach() > 0).float().mean(),
            "active_prior_gain_ratio": (prior_gain_per_sample.detach() > 0).float().mean(),
            "score_pos_tokens": pos_mask.detach().float().sum(dim=1).mean(),
            "score_hard_neg_tokens": hard_neg_mask.detach().float().sum(dim=1).mean(),
            "score_pos_mean": pos_score.detach().mean(),
            "score_hard_neg_mean": hard_neg_score.detach().mean(),
            "prior_pos_gain": pos_gain.detach().mean(),
            "prior_hard_neg_gain": hard_neg_gain.detach().mean(),
            "prior_bias_mean": bias_flat.detach().mean(),
            "prior_bias_abs_mean": bias_flat.detach().abs().mean(),
            "prior_bias_abs_max": bias_flat.detach().abs().max(),
            "prior_bias_max": bias_flat.detach().max(),
            "prior_bias_min": bias_flat.detach().min(),
            "score_logits_base_mean": base_flat.detach().mean(),
            "score_logits_base_abs_mean": base_flat.detach().abs().mean(),
            "score_logits_base_abs_max": base_flat.detach().abs().max(),
            "prior_to_score_abs_ratio": (
                bias_flat.detach().abs().mean() / base_flat.detach().abs().mean().clamp_min(1e-6)),
        }
        clamp_value = float(getattr(self.cfg.MODEL.TE, "SCORE_PRIOR_BIAS_CLAMP", 0.0))
        if clamp_value > 0.0:
            status["prior_clamp_ratio"] = (
                bias_flat.detach().abs() >= (clamp_value - 1e-6)).float().mean()
        else:
            status["prior_clamp_ratio"] = bias_flat.new_tensor(0.0)
        status.update(self._compute_safe_proto_status(pred, total_rows, pos_mask, hard_neg_mask))
        status.update(self._compute_lmq_status(pred, total_rows, pos_mask, hard_neg_mask))
        return loss, status

    def _compute_language_state_loss(self, pred):
        diagnostics = pred.get("language_state_diagnostics", None)
        if not diagnostics:
            return None, {}

        gate_weight = float(getattr(self.cfg.TRAIN, "LANGUAGE_STATE_GATE_LOSS_WEIGHT", 0.0))
        delta_weight = float(getattr(self.cfg.TRAIN, "LANGUAGE_STATE_DELTA_LOSS_WEIGHT", 0.0))
        gain_weight = float(getattr(self.cfg.TRAIN, "LANGUAGE_STATE_GAIN_LOSS_WEIGHT", 0.0))
        token_absorb_weight = float(getattr(self.cfg.TRAIN, "LANGUAGE_STATE_TOKEN_ABSORB_LOSS_WEIGHT", 0.0))
        candidate_cap_weight = float(getattr(self.cfg.TRAIN, "LANGUAGE_STATE_CANDIDATE_CAP_LOSS_WEIGHT", 0.0))
        anchor_cap_weight = float(getattr(self.cfg.TRAIN, "LANGUAGE_STATE_ANCHOR_CAP_LOSS_WEIGHT", 0.0))
        prev_keep_weight = float(getattr(self.cfg.TRAIN, "LANGUAGE_STATE_PREV_KEEP_LOSS_WEIGHT", 0.0))
        align_contrast_weight = float(getattr(self.cfg.TRAIN, "LANGUAGE_STATE_ALIGN_CONTRAST_LOSS_WEIGHT", 0.0))
        if (gate_weight <= 0.0 and delta_weight <= 0.0 and gain_weight <= 0.0
                and token_absorb_weight <= 0.0 and candidate_cap_weight <= 0.0
                and anchor_cap_weight <= 0.0 and prev_keep_weight <= 0.0
                and align_contrast_weight <= 0.0):
            return None, {}

        loss = None
        status = {}
        gate_reg = diagnostics.get("gate_reg_loss", None)
        if gate_reg is not None and gate_weight > 0.0:
            gate_reg = gate_reg.float().mean()
            gate_loss = gate_weight * gate_reg
            loss = gate_loss if loss is None else loss + gate_loss
            status["gate_reg_loss"] = gate_reg.detach()
            status["gate_loss_weight"] = torch.as_tensor(
                gate_weight, device=gate_reg.device, dtype=gate_reg.dtype)

        delta_reg = diagnostics.get("delta_reg_loss", None)
        if delta_reg is not None and delta_weight > 0.0:
            delta_reg = delta_reg.float().mean()
            delta_loss = delta_weight * delta_reg
            loss = delta_loss if loss is None else loss + delta_loss
            status["delta_reg_loss"] = delta_reg.detach()
            status["delta_loss_weight"] = torch.as_tensor(
                delta_weight, device=delta_reg.device, dtype=delta_reg.dtype)

        if gain_weight > 0.0:
            gain_loss, gain_status = self._compute_language_state_gain_loss(pred)
            if gain_loss is not None:
                weighted_gain_loss = gain_weight * gain_loss
                loss = weighted_gain_loss if loss is None else loss + weighted_gain_loss
                status.update(gain_status)
                status["gain_loss_weight"] = torch.as_tensor(
                    gain_weight, device=gain_loss.device, dtype=gain_loss.dtype)

        if token_absorb_weight > 0.0:
            token_loss, token_status = self._compute_language_state_token_absorb_loss(pred)
            if token_loss is not None:
                weighted_token_loss = token_absorb_weight * token_loss
                loss = weighted_token_loss if loss is None else loss + weighted_token_loss
                status.update(token_status)
                status["token_absorb_loss_weight"] = torch.as_tensor(
                    token_absorb_weight, device=token_loss.device, dtype=token_loss.dtype)

        if candidate_cap_weight > 0.0:
            cap_loss, cap_status = self._compute_language_state_candidate_cap_loss(pred)
            if cap_loss is not None:
                weighted_cap_loss = candidate_cap_weight * cap_loss
                loss = weighted_cap_loss if loss is None else loss + weighted_cap_loss
                status.update(cap_status)
                status["candidate_cap_loss_weight"] = torch.as_tensor(
                    candidate_cap_weight, device=cap_loss.device, dtype=cap_loss.dtype)

        source_bound_loss, source_bound_status = self._compute_language_state_source_bound_loss(pred)
        if source_bound_loss is not None:
            if anchor_cap_weight > 0.0 and "anchor_cap_loss" in source_bound_status:
                anchor_loss = anchor_cap_weight * source_bound_status["anchor_cap_loss"]
                loss = anchor_loss if loss is None else loss + anchor_loss
                status["anchor_cap_loss"] = source_bound_status["anchor_cap_loss"].detach()
                status["anchor_cap_active_ratio"] = source_bound_status["anchor_cap_active_ratio"].detach()
                status["anchor_cap_weight_mean"] = source_bound_status["anchor_cap_weight_mean"].detach()
                status["anchor_cap_weight_max"] = source_bound_status["anchor_cap_weight_max"].detach()
                status["anchor_cap_value"] = source_bound_status["anchor_cap_value"].detach()
                status["anchor_cap_loss_weight"] = torch.as_tensor(
                    anchor_cap_weight,
                    device=source_bound_status["anchor_cap_loss"].device,
                    dtype=source_bound_status["anchor_cap_loss"].dtype)
            if prev_keep_weight > 0.0 and "prev_keep_loss" in source_bound_status:
                prev_loss = prev_keep_weight * source_bound_status["prev_keep_loss"]
                loss = prev_loss if loss is None else loss + prev_loss
                status["prev_keep_loss"] = source_bound_status["prev_keep_loss"].detach()
                status["prev_keep_active_ratio"] = source_bound_status["prev_keep_active_ratio"].detach()
                status["prev_keep_weight_mean_reg"] = source_bound_status["prev_keep_weight_mean"].detach()
                status["prev_keep_weight_min_reg"] = source_bound_status["prev_keep_weight_min"].detach()
                status["prev_keep_min_value"] = source_bound_status["prev_keep_min_value"].detach()
                status["prev_keep_loss_weight"] = torch.as_tensor(
                    prev_keep_weight,
                    device=source_bound_status["prev_keep_loss"].device,
                    dtype=source_bound_status["prev_keep_loss"].dtype)

        if align_contrast_weight > 0.0:
            align_loss, align_status = self._compute_language_state_align_contrast_loss(pred)
            if align_loss is not None:
                weighted_align_loss = align_contrast_weight * align_loss
                loss = weighted_align_loss if loss is None else loss + weighted_align_loss
                status.update(align_status)
                status["align_contrast_loss_weight"] = torch.as_tensor(
                    align_contrast_weight, device=align_loss.device, dtype=align_loss.dtype)

        if loss is None:
            return None, {}
        status["loss"] = loss.detach()
        return loss, status

    def _language_state_prior_scores(self, language_tokens, language_mask, search_tokens):
        query = _actor_masked_mean(language_tokens, language_mask)
        query = F.normalize(query, dim=-1)
        keys = F.normalize(search_tokens.detach(), dim=-1)
        return torch.matmul(keys, query.unsqueeze(-1)).squeeze(-1)

    def _language_state_token_scores(self, language_tokens, search_tokens):
        token_queries = F.normalize(language_tokens, dim=-1)
        keys = F.normalize(search_tokens.detach(), dim=-1)
        return torch.einsum("bld,bnd->bln", token_queries, keys)

    def _language_state_anchor_identity_support(self, candidate_tokens, anchor_tokens, anchor_mask):
        if anchor_tokens is None:
            return None
        candidate_norm = F.normalize(candidate_tokens.detach(), dim=-1)
        anchor_norm = F.normalize(anchor_tokens.detach(), dim=-1)
        support = torch.einsum("bld,bmd->blm", candidate_norm, anchor_norm)
        if anchor_mask is not None:
            if anchor_mask.dim() == 3:
                anchor_mask = anchor_mask.squeeze(-1)
            anchor_valid = anchor_mask.to(device=support.device).bool()
            support = support.masked_fill(~anchor_valid[:, None, :], -1.0)
        return support.max(dim=-1).values

    def _compute_language_state_source_bound_loss(self, pred):
        aux = pred.get("language_state_aux", None)
        if not aux:
            return None, {}
        anchor_weight = aux.get("anchor_weight", None)
        prev_weight = aux.get("prev_weight", None)
        mask = aux.get("mask", None)
        if anchor_weight is None and prev_weight is None:
            return None, {}
        ref_weight = anchor_weight if anchor_weight is not None else prev_weight
        ref_weight = ref_weight.squeeze(-1)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.squeeze(-1)
            valid = mask.to(device=ref_weight.device, dtype=ref_weight.dtype)
        else:
            valid = torch.ones_like(ref_weight)
        valid_bool = valid > 0
        status = {}
        loss = ref_weight.new_tensor(0.0)
        if anchor_weight is not None:
            anchor_weight = anchor_weight.squeeze(-1)
            anchor_max = float(getattr(self.cfg.TRAIN, "LANGUAGE_STATE_ANCHOR_WEIGHT_MAX", 0.5))
            anchor_over = F.relu(anchor_weight - anchor_max)
            anchor_loss = (anchor_over.pow(2) * valid).sum() / valid.sum().clamp_min(1.0)
            loss = loss + anchor_loss
            if valid_bool.any():
                valid_anchor = anchor_weight.detach()[valid_bool]
                valid_over = anchor_over.detach()[valid_bool]
                anchor_active = (valid_over > 0).float().mean()
                anchor_mean = valid_anchor.mean()
                anchor_max_value = valid_anchor.max()
            else:
                anchor_active = ref_weight.new_tensor(0.0)
                anchor_mean = ref_weight.new_tensor(0.0)
                anchor_max_value = ref_weight.new_tensor(0.0)
            status.update({
                "anchor_cap_loss": anchor_loss,
                "anchor_cap_active_ratio": anchor_active,
                "anchor_cap_weight_mean": anchor_mean,
                "anchor_cap_weight_max": anchor_max_value,
                "anchor_cap_value": torch.as_tensor(
                    anchor_max, device=ref_weight.device, dtype=ref_weight.dtype),
            })
        if prev_weight is not None:
            prev_weight = prev_weight.squeeze(-1)
            prev_min = float(getattr(self.cfg.TRAIN, "LANGUAGE_STATE_PREV_KEEP_MIN", 0.5))
            prev_under = F.relu(prev_min - prev_weight)
            prev_loss = (prev_under.pow(2) * valid).sum() / valid.sum().clamp_min(1.0)
            loss = loss + prev_loss
            if valid_bool.any():
                valid_prev = prev_weight.detach()[valid_bool]
                valid_under = prev_under.detach()[valid_bool]
                prev_active = (valid_under > 0).float().mean()
                prev_mean = valid_prev.mean()
                prev_min_value = valid_prev.min()
            else:
                prev_active = ref_weight.new_tensor(0.0)
                prev_mean = ref_weight.new_tensor(0.0)
                prev_min_value = ref_weight.new_tensor(0.0)
            status.update({
                "prev_keep_loss": prev_loss,
                "prev_keep_active_ratio": prev_active,
                "prev_keep_weight_mean": prev_mean,
                "prev_keep_weight_min": prev_min_value,
                "prev_keep_min_value": torch.as_tensor(
                    prev_min, device=ref_weight.device, dtype=ref_weight.dtype),
            })
        return loss, status

    def _limit_language_state_positive_targets(self, target, evidence_score, valid):
        top_ratio = float(getattr(self.cfg.TRAIN, "LANGUAGE_STATE_TOKEN_ABSORB_TOP_RATIO", 0.0))
        max_pos = int(getattr(self.cfg.TRAIN, "LANGUAGE_STATE_TOKEN_ABSORB_MAX_POS", 0))
        if top_ratio <= 0.0 and max_pos <= 0:
            return target

        target_bool = target.bool() & (valid > 0)
        limited = torch.zeros_like(target_bool)
        valid_counts = (valid > 0).sum(dim=1)
        for batch_idx in range(target_bool.shape[0]):
            eligible = target_bool[batch_idx]
            eligible_count = int(eligible.sum().item())
            if eligible_count <= 0:
                continue
            limit = eligible_count
            if top_ratio > 0.0:
                ratio_limit = int(math.ceil(float(valid_counts[batch_idx].item()) * top_ratio))
                limit = min(limit, max(1, ratio_limit))
            if max_pos > 0:
                limit = min(limit, max_pos)
            scores = evidence_score[batch_idx].masked_fill(~eligible, -1.0e6)
            top_idx = scores.topk(limit, dim=0).indices
            limited[batch_idx, top_idx] = True
        return limited.to(dtype=target.dtype)

    def _compute_language_state_token_absorb_components(self, pred):
        aux = pred.get("language_state_aux", None)
        if not aux:
            return None
        prev_tokens = aux.get("prev_tokens", None)
        anchor_tokens = aux.get("anchor_aligned_tokens", None)
        anchor_raw_tokens = aux.get("anchor_raw_tokens", None)
        anchor_raw_mask = aux.get("anchor_raw_mask", None)
        candidate_tokens = aux.get("candidate_aligned_tokens", None)
        candidate_weight = aux.get("candidate_weight", None)
        candidate_logit = aux.get("candidate_absorb_logit", None)
        mask = aux.get("mask", None)
        feat = pred.get("backbone_feat", None)
        score_map = pred.get("score_map", None)
        if (prev_tokens is None or candidate_tokens is None or candidate_weight is None
                or candidate_logit is None or feat is None or score_map is None):
            return None
        if isinstance(feat, list):
            feat = feat[-1]
        search_tokens = feat[:, -self.net.feat_len_s:].detach()
        if search_tokens.shape[1] != self.net.feat_len_s:
            return None

        batch_size, token_count = search_tokens.shape[:2]
        gt_bbox = pred.get("_language_state_gt_bbox", None)
        if gt_bbox is None:
            return None
        gt_bbox = gt_bbox.to(device=search_tokens.device, dtype=search_tokens.dtype)
        pos_quality = self._search_token_center_quality(gt_bbox, token_count, search_tokens.device)
        pos_mask = pos_quality > 0.0
        pos_den = pos_quality.sum(dim=1).clamp_min(1e-6)
        hard_mask, hard_topk = self._language_state_hardneg_mask(score_map, pos_mask, token_count)
        hard_den = hard_mask.sum(dim=1).to(dtype=search_tokens.dtype).clamp_min(1.0)

        prev_token_scores = self._language_state_token_scores(prev_tokens.detach(), search_tokens)
        candidate_token_scores = self._language_state_token_scores(candidate_tokens.detach(), search_tokens)
        prev_pos = (prev_token_scores * pos_quality[:, None, :]).sum(dim=-1) / pos_den[:, None]
        cand_pos = (candidate_token_scores * pos_quality[:, None, :]).sum(dim=-1) / pos_den[:, None]
        hard_weight = hard_mask[:, None, :].to(dtype=search_tokens.dtype)
        prev_hard = (prev_token_scores * hard_weight).sum(dim=-1) / hard_den[:, None]
        cand_hard = (candidate_token_scores * hard_weight).sum(dim=-1) / hard_den[:, None]
        prev_gap = prev_pos - prev_hard
        cand_gap = cand_pos - cand_hard
        token_gain = cand_gap - prev_gap
        hardneg_gain = cand_hard - prev_hard

        old_margin = float(getattr(self.cfg.TRAIN, "LANGUAGE_STATE_TOKEN_ABSORB_MARGIN", 0.005))
        margin_rel = float(getattr(
            self.cfg.TRAIN, "LANGUAGE_STATE_TOKEN_ABSORB_MARGIN_REL", old_margin))
        margin_abs = float(getattr(
            self.cfg.TRAIN, "LANGUAGE_STATE_TOKEN_ABSORB_MARGIN_ABS", 0.005))
        hardneg_margin = float(getattr(
            self.cfg.TRAIN, "LANGUAGE_STATE_TOKEN_ABSORB_HARDNEG_MARGIN", 0.0))
        rel_ok = token_gain > margin_rel
        abs_ok = cand_gap > margin_abs
        hard_ok = hardneg_gain < hardneg_margin

        identity_support_raw = self._language_state_anchor_identity_support(
            candidate_tokens, anchor_raw_tokens, anchor_raw_mask)
        identity_support = None
        if anchor_tokens is not None:
            identity_support = (
                F.normalize(anchor_tokens.detach(), dim=-1)
                * F.normalize(candidate_tokens.detach(), dim=-1)
            ).sum(dim=-1)
        elif identity_support_raw is not None:
            identity_support = identity_support_raw
        identity_enable = bool(getattr(
            self.cfg.TRAIN, "LANGUAGE_STATE_TOKEN_ABSORB_IDENTITY_ENABLE", False))
        identity_min = float(getattr(
            self.cfg.TRAIN, "LANGUAGE_STATE_TOKEN_ABSORB_IDENTITY_MIN", 0.2))
        if identity_support is not None and identity_enable:
            identity_ok = identity_support > identity_min
        else:
            identity_ok = torch.ones_like(rel_ok, dtype=torch.bool)

        target_base = rel_ok & abs_ok & hard_ok & identity_ok
        evidence_score = token_gain + cand_gap - hardneg_gain
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.squeeze(-1)
            valid = mask.to(device=candidate_weight.device, dtype=candidate_weight.dtype)
        else:
            valid = torch.ones_like(candidate_weight.squeeze(-1))

        return {
            "prev_gap": prev_gap,
            "cand_gap": cand_gap,
            "token_gain": token_gain,
            "hardneg_gain": hardneg_gain,
            "rel_ok": rel_ok,
            "abs_ok": abs_ok,
            "hard_ok": hard_ok,
            "identity_ok": identity_ok,
            "identity_support": identity_support,
            "identity_support_raw": identity_support_raw,
            "target_base": target_base,
            "evidence_score": evidence_score,
            "valid": valid,
            "candidate_weight": candidate_weight.squeeze(-1),
            "candidate_logit": candidate_logit,
            "hard_topk": hard_topk,
            "margin_rel": margin_rel,
            "margin_abs": margin_abs,
            "hardneg_margin": hardneg_margin,
            "identity_min": identity_min,
        }

    def _compute_language_state_candidate_cap_loss(self, pred):
        aux = pred.get("language_state_aux", None)
        if not aux:
            return None, {}
        candidate_weight = aux.get("candidate_weight", None)
        mask = aux.get("mask", None)
        if candidate_weight is None:
            return None, {}
        candidate_weight = candidate_weight.squeeze(-1)
        cap = float(getattr(self.cfg.TRAIN, "LANGUAGE_STATE_CANDIDATE_WEIGHT_MAX", 0.5))
        over_cap = F.relu(candidate_weight - cap)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.squeeze(-1)
            valid = mask.to(device=candidate_weight.device, dtype=candidate_weight.dtype)
        else:
            valid = torch.ones_like(candidate_weight)
        loss = (over_cap.pow(2) * valid).sum() / valid.sum().clamp_min(1.0)
        valid_bool = valid > 0
        if valid_bool.any():
            valid_weight = candidate_weight.detach()[valid_bool]
            valid_over = over_cap.detach()[valid_bool]
            active_ratio = (valid_over > 0).float().mean()
            weight_mean = valid_weight.mean()
            weight_max = valid_weight.max()
            over_mean = valid_over.mean()
        else:
            active_ratio = candidate_weight.new_tensor(0.0)
            weight_mean = candidate_weight.new_tensor(0.0)
            weight_max = candidate_weight.new_tensor(0.0)
            over_mean = candidate_weight.new_tensor(0.0)
        status = {
            "candidate_cap_loss": loss.detach(),
            "candidate_cap_active_ratio": active_ratio.detach(),
            "candidate_cap_weight_mean": weight_mean.detach(),
            "candidate_cap_weight_max": weight_max.detach(),
            "candidate_cap_over_mean": over_mean.detach(),
            "candidate_cap_value": torch.as_tensor(
                cap, device=candidate_weight.device, dtype=candidate_weight.dtype),
        }
        return loss, status

    def _compute_language_state_align_contrast_loss(self, pred):
        aux = pred.get("language_state_aux", None)
        if not aux:
            return None, {}
        candidate_aligned = aux.get("candidate_aligned_tokens", None)
        prev_tokens = aux.get("prev_tokens", None)
        mask = aux.get("mask", None)
        if candidate_aligned is None or prev_tokens is None:
            return None, {}
        candidate_aligned = F.normalize(candidate_aligned, dim=-1)
        prev_tokens = F.normalize(prev_tokens.detach(), dim=-1)
        tau = float(getattr(self.cfg.TRAIN, "LANGUAGE_STATE_ALIGN_CONTRAST_TAU", 0.07))
        sim = torch.matmul(candidate_aligned, prev_tokens.transpose(1, 2)) / tau
        L = sim.shape[1]
        diag_sim = sim.diagonal(dim1=-2, dim2=-1)
        loss_per_token = -diag_sim + torch.logsumexp(sim, dim=-1)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.squeeze(-1)
            valid = mask.to(device=loss_per_token.device, dtype=loss_per_token.dtype)
        else:
            valid = torch.ones_like(loss_per_token)
        loss = (loss_per_token * valid).sum() / valid.sum().clamp_min(1.0)
        valid_bool = valid > 0
        if valid_bool.any():
            valid_sim = sim.detach()[valid_bool]
            diag_mean = valid_sim.diagonal(dim1=-2, dim2=-1).mean()
            offdiag_mean = (valid_sim.sum(dim=-1) - valid_sim.diagonal(dim1=-2, dim2=-1)).mean() / max(L - 1, 1)
            diag_gap = diag_mean - offdiag_mean
        else:
            diag_mean = sim.new_tensor(0.0)
            offdiag_mean = sim.new_tensor(0.0)
            diag_gap = sim.new_tensor(0.0)
        status = {
            "align_contrast_loss": loss.detach(),
            "align_contrast_diag_sim": diag_mean.detach(),
            "align_contrast_offdiag_sim": offdiag_mean.detach(),
            "align_contrast_diag_gap": diag_gap.detach(),
            "align_contrast_tau": torch.as_tensor(tau, device=loss.device, dtype=loss.dtype),
        }
        return loss, status

    def _language_state_hardneg_mask(self, score_map, pos_mask, token_count):
        batch_size = pos_mask.shape[0]
        score_flat = score_map.detach().reshape(batch_size, -1)
        if score_flat.shape[1] != token_count:
            feat_sz = int(token_count ** 0.5)
            score_flat = F.interpolate(
                score_map.detach(), size=(feat_sz, feat_sz),
                mode="bilinear", align_corners=False).reshape(batch_size, -1)
        hard_source = score_flat.masked_fill(pos_mask, -1.0)
        hard_topk = int(getattr(self.cfg.TRAIN, "LANGUAGE_STATE_GAIN_HARDNEG_TOPK", 6))
        hard_topk = max(1, min(hard_topk, token_count))
        hard_idx = hard_source.topk(hard_topk, dim=1).indices
        hard_mask = torch.zeros_like(pos_mask)
        hard_mask.scatter_(1, hard_idx, True)
        return hard_mask, hard_topk

    def _compute_language_state_gain_loss(self, pred):
        aux = pred.get("language_state_aux", None)
        if not aux:
            return None, {}
        state_tokens = aux.get("state_tokens", None)
        prev_tokens = aux.get("prev_tokens", None)
        mask = aux.get("mask", None)
        feat = pred.get("backbone_feat", None)
        score_map = pred.get("score_map", None)
        if state_tokens is None or prev_tokens is None or feat is None or score_map is None:
            return None, {}
        if isinstance(feat, list):
            feat = feat[-1]
        search_tokens = feat[:, -self.net.feat_len_s:].detach()
        if search_tokens.shape[1] != self.net.feat_len_s:
            return None, {}

        state_scores = self._language_state_prior_scores(state_tokens, mask, search_tokens)
        prev_scores = self._language_state_prior_scores(prev_tokens, mask, search_tokens)

        batch_size, token_count = state_scores.shape
        if token_count != self.net.feat_len_s:
            return None, {}
        gt_bbox = pred.get("_language_state_gt_bbox", None)
        if gt_bbox is None:
            return None, {}
        gt_bbox = gt_bbox.to(device=state_scores.device, dtype=state_scores.dtype)
        pos_quality = self._search_token_center_quality(gt_bbox, token_count, state_scores.device)
        pos_mask = pos_quality > 0.0
        pos_den = pos_quality.sum(dim=1).clamp_min(1e-6)

        hard_mask, hard_topk = self._language_state_hardneg_mask(score_map, pos_mask, token_count)
        hard_den = hard_mask.sum(dim=1).to(dtype=state_scores.dtype).clamp_min(1.0)

        state_pos = (state_scores * pos_quality).sum(dim=1) / pos_den
        prev_pos = (prev_scores.detach() * pos_quality).sum(dim=1) / pos_den
        state_hard = (state_scores * hard_mask.to(dtype=state_scores.dtype)).sum(dim=1) / hard_den
        prev_hard = (prev_scores.detach() * hard_mask.to(dtype=state_scores.dtype)).sum(dim=1) / hard_den
        state_gap = state_pos - state_hard
        prev_gap = prev_pos - prev_hard
        gap_gain = state_gap - prev_gap
        margin = float(getattr(self.cfg.TRAIN, "LANGUAGE_STATE_GAIN_MARGIN", 0.01))
        per_sample_loss = F.relu(margin - gap_gain)
        loss = per_sample_loss.mean()
        status = {
            "gain_loss": loss.detach(),
            "gain_active_ratio": (per_sample_loss.detach() > 0).float().mean(),
            "gap_state": state_gap.detach().mean(),
            "gap_prev": prev_gap.detach().mean(),
            "gap_gain": gap_gain.detach().mean(),
            "state_pos": state_pos.detach().mean(),
            "state_hardneg": state_hard.detach().mean(),
            "prev_pos": prev_pos.detach().mean(),
            "prev_hardneg": prev_hard.detach().mean(),
            "hardneg_topk": torch.as_tensor(
                hard_topk, device=state_scores.device, dtype=state_scores.dtype),
        }
        return loss, status

    def _compute_language_state_token_absorb_loss(self, pred):
        components = self._compute_language_state_token_absorb_components(pred)
        if components is None:
            return None, {}

        target_base = components["target_base"]
        multiframe_target = pred.get("_language_state_token_absorb_multiframe_target", None)
        if multiframe_target is not None and multiframe_target.shape == target_base.shape:
            target_base = target_base & multiframe_target.to(device=target_base.device).bool()
            multiframe_ok = multiframe_target.to(device=target_base.device).bool()
        else:
            multiframe_ok = torch.ones_like(target_base, dtype=torch.bool)

        valid = components["valid"]
        target_limited = self._limit_language_state_positive_targets(
            target_base.to(dtype=valid.dtype), components["evidence_score"].detach(), valid)
        target_hard = target_limited.detach()
        pred_logit = components["candidate_logit"].to(dtype=torch.float32)
        pred_weight = components["candidate_weight"].clamp(1e-4, 1.0 - 1e-4)

        soft_target_enable = bool(getattr(
            self.cfg.TRAIN, "LANGUAGE_STATE_TOKEN_ABSORB_SOFT_TARGET", False))
        if soft_target_enable:
            soft_tau = float(getattr(
                self.cfg.TRAIN, "LANGUAGE_STATE_TOKEN_ABSORB_SOFT_TAU", 0.02))
            token_gain = components["token_gain"]
            target = torch.sigmoid((token_gain - components["margin_rel"]) / soft_tau).detach()
            target = target.clamp(0.0, 1.0)
            per_token_loss = F.binary_cross_entropy_with_logits(pred_logit, target, reduction="none")
        else:
            target = target_hard.to(dtype=torch.float32)
            per_token_loss = F.binary_cross_entropy_with_logits(pred_logit, target, reduction="none")

        hard_neg_weight = float(getattr(
            self.cfg.TRAIN, "LANGUAGE_STATE_TOKEN_ABSORB_HARD_NEG_WEIGHT", 1.0))
        target_bool = target > 0.5
        hard_negative = (~target_bool) & (valid > 0) & (
            pred_weight.detach() > pred_weight.detach().mean(dim=1, keepdim=True))
        sample_weight = valid.clone()
        if hard_neg_weight > 1.0:
            sample_weight = sample_weight + hard_negative.to(dtype=sample_weight.dtype) * (hard_neg_weight - 1.0)

        focal_gamma = float(getattr(
            self.cfg.TRAIN, "LANGUAGE_STATE_TOKEN_ABSORB_FOCAL_GAMMA", 0.0))
        focal_alpha = float(getattr(
            self.cfg.TRAIN, "LANGUAGE_STATE_TOKEN_ABSORB_FOCAL_ALPHA", 0.75))
        if focal_gamma > 0.0:
            bce = per_token_loss
            pt = torch.exp(-bce)
            focal_weight = focal_alpha * target + (1.0 - focal_alpha) * (1.0 - target)
            focal_mod = ((1.0 - pt) ** focal_gamma).detach()
            per_token_loss = focal_weight * focal_mod * per_token_loss

        bce_loss = (per_token_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)
        loss = bce_loss

        pos_floor_weight = float(getattr(
            self.cfg.TRAIN, "LANGUAGE_STATE_TOKEN_ABSORB_POS_FLOOR_WEIGHT", 0.0))
        pos_floor_min = float(getattr(
            self.cfg.TRAIN, "LANGUAGE_STATE_TOKEN_ABSORB_POS_FLOOR_MIN", 0.3))
        pos_floor_loss = pred_weight.new_tensor(0.0)
        if pos_floor_weight > 0.0:
            target_bool = target > 0.5
            pos_floor_per_token = target_bool.to(dtype=pred_weight.dtype) * F.relu(pos_floor_min - pred_weight) ** 2
            pos_floor_loss = (pos_floor_per_token * valid).sum() / valid.sum().clamp_min(1.0)
            loss = loss + pos_floor_weight * pos_floor_loss

        valid_bool = valid > 0
        if valid_bool.any():
            valid_gain = components["token_gain"].detach()[valid_bool]
            valid_prev_gap = components["prev_gap"].detach()[valid_bool]
            valid_cand_gap = components["cand_gap"].detach()[valid_bool]
            valid_hard_gain = components["hardneg_gain"].detach()[valid_bool]
            valid_target = target.detach()[valid_bool]
            valid_weight = pred_weight.detach()[valid_bool]
            valid_rel_ok = components["rel_ok"].detach()[valid_bool]
            valid_abs_ok = components["abs_ok"].detach()[valid_bool]
            valid_hard_ok = components["hard_ok"].detach()[valid_bool]
            valid_identity_ok = components["identity_ok"].detach()[valid_bool]
            valid_base_target = target_base.detach()[valid_bool]
            valid_multiframe_ok = multiframe_ok.detach()[valid_bool]
            valid_hard_negative = hard_negative.detach()[valid_bool]
            target_pos_ratio = valid_target.float().mean()
            base_pos_ratio = valid_base_target.float().mean()
            multiframe_ok_ratio = valid_multiframe_ok.float().mean()
            positive_weight = valid_weight[valid_target > 0.5].mean() if (valid_target > 0.5).any() else valid_weight.new_tensor(0.0)
            negative_weight = valid_weight[valid_target <= 0.5].mean() if (valid_target <= 0.5).any() else valid_weight.new_tensor(0.0)
            weight_pos_minus_neg = positive_weight - negative_weight
            gain_mean = valid_gain.mean()
            prev_gap_mean = valid_prev_gap.mean()
            cand_gap_mean = valid_cand_gap.mean()
            hard_gain_mean = valid_hard_gain.mean()
            rel_ok_ratio = valid_rel_ok.float().mean()
            abs_ok_ratio = valid_abs_ok.float().mean()
            hard_ok_ratio = valid_hard_ok.float().mean()
            identity_ok_ratio = valid_identity_ok.float().mean()
            hard_negative_ratio = valid_hard_negative.float().mean()
            # Per-condition pass rates on base positive tokens (before top-k limit)
            base_pos_bool = valid_base_target > 0.5
            if base_pos_bool.any():
                pos_rel_ok_ratio = valid_rel_ok[base_pos_bool].float().mean()
                pos_abs_ok_ratio = valid_abs_ok[base_pos_bool].float().mean()
                pos_hard_ok_ratio = valid_hard_ok[base_pos_bool].float().mean()
                pos_identity_ok_ratio = valid_identity_ok[base_pos_bool].float().mean()
                pos_all_four_ok = (valid_rel_ok[base_pos_bool] & valid_abs_ok[base_pos_bool]
                                   & valid_hard_ok[base_pos_bool] & valid_identity_ok[base_pos_bool]).float().mean()
            else:
                pos_rel_ok_ratio = pred_weight.new_tensor(0.0)
                pos_abs_ok_ratio = pred_weight.new_tensor(0.0)
                pos_hard_ok_ratio = pred_weight.new_tensor(0.0)
                pos_identity_ok_ratio = pred_weight.new_tensor(0.0)
                pos_all_four_ok = pred_weight.new_tensor(0.0)
        else:
            target_pos_ratio = pred_weight.new_tensor(0.0)
            base_pos_ratio = pred_weight.new_tensor(0.0)
            multiframe_ok_ratio = pred_weight.new_tensor(0.0)
            positive_weight = pred_weight.new_tensor(0.0)
            negative_weight = pred_weight.new_tensor(0.0)
            weight_pos_minus_neg = pred_weight.new_tensor(0.0)
            gain_mean = pred_weight.new_tensor(0.0)
            prev_gap_mean = pred_weight.new_tensor(0.0)
            cand_gap_mean = pred_weight.new_tensor(0.0)
            hard_gain_mean = pred_weight.new_tensor(0.0)
            rel_ok_ratio = pred_weight.new_tensor(0.0)
            abs_ok_ratio = pred_weight.new_tensor(0.0)
            hard_ok_ratio = pred_weight.new_tensor(0.0)
            identity_ok_ratio = pred_weight.new_tensor(0.0)
            hard_negative_ratio = pred_weight.new_tensor(0.0)
            pos_rel_ok_ratio = pred_weight.new_tensor(0.0)
            pos_abs_ok_ratio = pred_weight.new_tensor(0.0)
            pos_hard_ok_ratio = pred_weight.new_tensor(0.0)
            pos_identity_ok_ratio = pred_weight.new_tensor(0.0)
            pos_all_four_ok = pred_weight.new_tensor(0.0)
        status = {
            "token_absorb_loss": loss.detach(),
            "token_absorb_bce_loss": bce_loss.detach(),
            "token_absorb_pos_floor_loss": pos_floor_loss.detach(),
            "token_absorb_target_pos_ratio": target_pos_ratio.detach(),
            "token_absorb_base_pos_ratio": base_pos_ratio.detach(),
            "token_absorb_multiframe_ok_ratio": multiframe_ok_ratio.detach(),
            "token_absorb_candidate_weight_pos": positive_weight.detach(),
            "token_absorb_candidate_weight_neg": negative_weight.detach(),
            "token_absorb_weight_pos_minus_neg": weight_pos_minus_neg.detach(),
            "token_absorb_gain_mean": gain_mean.detach(),
            "token_absorb_prev_gap_mean": prev_gap_mean.detach(),
            "token_absorb_cand_gap_mean": cand_gap_mean.detach(),
            "token_absorb_hardneg_gain_mean": hard_gain_mean.detach(),
            "token_absorb_rel_ok_ratio": rel_ok_ratio.detach(),
            "token_absorb_abs_ok_ratio": abs_ok_ratio.detach(),
            "token_absorb_hard_ok_ratio": hard_ok_ratio.detach(),
            "token_absorb_identity_ok_ratio": identity_ok_ratio.detach(),
            "token_absorb_hard_negative_ratio": hard_negative_ratio.detach(),
            "token_absorb_pos_rel_ok_ratio": pos_rel_ok_ratio.detach(),
            "token_absorb_pos_abs_ok_ratio": pos_abs_ok_ratio.detach(),
            "token_absorb_pos_hard_ok_ratio": pos_hard_ok_ratio.detach(),
            "token_absorb_pos_identity_ok_ratio": pos_identity_ok_ratio.detach(),
            "token_absorb_pos_all_four_ok": pos_all_four_ok.detach(),
            "token_absorb_margin_rel": torch.as_tensor(
                components["margin_rel"], device=pred_weight.device, dtype=pred_weight.dtype),
            "token_absorb_margin_abs": torch.as_tensor(
                components["margin_abs"], device=pred_weight.device, dtype=pred_weight.dtype),
            "token_absorb_hardneg_margin": torch.as_tensor(
                components["hardneg_margin"], device=pred_weight.device, dtype=pred_weight.dtype),
            "token_absorb_identity_min": torch.as_tensor(
                components["identity_min"], device=pred_weight.device, dtype=pred_weight.dtype),
            "token_absorb_hardneg_topk": torch.as_tensor(
                components["hard_topk"], device=pred_weight.device, dtype=pred_weight.dtype),
            "token_absorb_pos_floor_min": torch.as_tensor(
                pos_floor_min, device=pred_weight.device, dtype=pred_weight.dtype),
            "token_absorb_pos_floor_weight": torch.as_tensor(
                pos_floor_weight, device=pred_weight.device, dtype=pred_weight.dtype),
            "token_absorb_soft_target": torch.as_tensor(
                1.0 if soft_target_enable else 0.0, device=pred_weight.device, dtype=pred_weight.dtype),
            "token_absorb_focal_gamma": torch.as_tensor(
                focal_gamma, device=pred_weight.device, dtype=pred_weight.dtype),
            "token_absorb_soft_target_mean": target.detach()[valid_bool].mean() if valid_bool.any() else pred_weight.new_tensor(0.0),
        }
        identity_support = components["identity_support"]
        if identity_support is not None:
            if valid_bool.any():
                valid_anchor_cos = identity_support.detach()[valid_bool]
                valid_target = target.detach()[valid_bool]
                anchor_cos_pos = (
                    valid_anchor_cos[valid_target > 0.5].mean()
                    if (valid_target > 0.5).any()
                    else valid_anchor_cos.new_tensor(0.0))
                anchor_cos_neg = (
                    valid_anchor_cos[valid_target <= 0.5].mean()
                    if (valid_target <= 0.5).any()
                    else valid_anchor_cos.new_tensor(0.0))
                anchor_cos_mean = valid_anchor_cos.mean()
            else:
                anchor_cos_mean = pred_weight.new_tensor(0.0)
                anchor_cos_pos = pred_weight.new_tensor(0.0)
                anchor_cos_neg = pred_weight.new_tensor(0.0)
            status.update({
                "token_absorb_anchor_cos_mean": anchor_cos_mean.detach(),
                "token_absorb_anchor_cos_pos": anchor_cos_pos.detach(),
                "token_absorb_anchor_cos_neg": anchor_cos_neg.detach(),
            })
        raw_identity_support = components.get("identity_support_raw", None)
        if raw_identity_support is not None and valid_bool.any():
            status["token_absorb_raw_anchor_cos_mean"] = raw_identity_support.detach()[valid_bool].mean()
        return loss, status

    def _prepare_language_state_token_absorb_multiframe_targets(self, pred_dict):
        if not bool(getattr(self.cfg.TRAIN, "LANGUAGE_STATE_TOKEN_ABSORB_MULTI_FRAME", False)):
            return
        if len(pred_dict) <= 1:
            return
        targets = []
        for pred in pred_dict:
            components = self._compute_language_state_token_absorb_components(pred)
            if components is None:
                return
            targets.append(components["target_base"].detach())
        first_shape = targets[0].shape
        if any(target.shape != first_shape for target in targets):
            return
        common = torch.stack(targets, dim=0).all(dim=0)
        for pred in pred_dict:
            pred["_language_state_token_absorb_multiframe_target"] = common

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # currently only support the type of pred_dict is list
        assert isinstance(pred_dict, list)
        loss_dict = {}
        total_status = {}
        total_loss = torch.tensor(0., dtype=torch.float).cuda() # 定义 0 tensor，并指定GPU设备
        
        # generate gt gaussian map
        gt_gaussian_maps_list = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        for i in range(len(pred_dict)):
            pred_dict[i]["_language_state_gt_bbox"] = gt_dict['search_anno'][i]
        self._prepare_language_state_token_absorb_multiframe_targets(pred_dict)
        
        for i in range(len(pred_dict)):
            # get GT
            gt_bbox = gt_dict['search_anno'][i]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
            gt_gaussian_maps = gt_gaussian_maps_list[i].unsqueeze(1)

            # Get boxes
            pred_boxes = pred_dict[i]['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)
            # (B,4) --> (B,1,4) --> (B,N,4)
            
            # compute giou and iou
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            loss_dict['giou'] = giou_loss
            
            # compute l1 loss
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            loss_dict['l1'] = l1_loss
            
            # compute location loss
            if 'score_map' in pred_dict[i]:
                location_loss = self.objective['focal'](pred_dict[i]['score_map'], gt_gaussian_maps)
            else:
                location_loss = torch.tensor(0.0, device=l1_loss.device)
            loss_dict['focal'] = location_loss
            te_weight = self._te_aux_weight(gt_dict)
            score_weight = self._score_aux_weight(gt_dict)
            te_search_loss = self._compute_te_search_loss(pred_dict[i], gt_bbox) if te_weight > 0.0 else None
            if score_weight > 0.0:
                score_prior_loss, score_prior_status = self._compute_score_prior_loss(pred_dict[i], gt_bbox)
            else:
                score_prior_loss, score_prior_status = None, {}
            pred_dict[i]["_language_state_gt_bbox"] = gt_bbox
            language_state_loss, language_state_status = self._compute_language_state_loss(pred_dict[i])

            # weighted sum
            loss = sum(loss_dict[k] * self.loss_weight[k] for k in loss_dict.keys() if k in self.loss_weight)
            if te_search_loss is not None:
                loss = loss + te_weight * te_search_loss
            if score_prior_loss is not None:
                loss = loss + score_weight * score_prior_loss
            if language_state_loss is not None:
                loss = loss + language_state_loss
            total_loss += loss
            
            if return_status:
                # status for log
                status = {}
                
                mean_iou = iou.detach().mean()
                status = {f"{i}frame_Loss/total": loss.item(),
                        f"{i}frame_Loss/giou": giou_loss.item(),
                        f"{i}frame_Loss/l1": l1_loss.item(),
                        f"{i}frame_Loss/location": location_loss.item(),
                        f"{i}frame_IoU": mean_iou.item()}
                if te_search_loss is not None:
                    status[f"{i}frame_Loss/te_search"] = te_search_loss.item()
                    status[f"{i}frame_Loss/te_search_weight"] = te_weight
                if score_prior_loss is not None:
                    status[f"{i}frame_Loss/score_prior"] = score_prior_loss.item()
                    status[f"{i}frame_Loss/score_prior_weight"] = score_weight
                    for key, value in score_prior_status.items():
                        status[f"{i}frame_ScorePrior/{key}"] = value.item()
                if language_state_loss is not None:
                    status[f"{i}frame_Loss/language_state"] = language_state_loss.item()
                    for key, value in language_state_status.items():
                        status[f"{i}frame_LanguageStateReg/{key}"] = value.item()
                language_state_diag = pred_dict[i].get("language_state_diagnostics", None)
                if language_state_diag:
                    for key, value in language_state_diag.items():
                        if str(key).startswith("_"):
                            continue
                        if torch.is_tensor(value):
                            status[f"{i}frame_LanguageState/{key}"] = value.detach().float().mean().item()
                gsb_diag = pred_dict[i].get("gsb_diagnostics", None)
                if gsb_diag:
                    for key, value in gsb_diag.items():
                        if torch.is_tensor(value):
                            status[f"{i}frame_GSB/{key}"] = value.detach().float().mean().item()

                total_status.update(status)

        if return_status:
            return total_loss, total_status
        else:
            return total_loss
