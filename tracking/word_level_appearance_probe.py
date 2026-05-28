import argparse
import csv
import math
import os
import sys

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker
from lib.train.data.processing_utils import sample_target
from tracking.language_visual_source_probe import (
    _as_1d,
    _minmax_01,
    _split_tokens,
    _tokenize_mask,
    _unit,
)
from tracking.visualte_diagnostic import (
    _feat_size,
    _heat_stats,
    _plain_tile,
    _read_rgb,
    _region_gap_stats,
    _render_map_tile,
    _run_label,
    _save_story_grid,
    _search_crop_box,
    _short_label,
    _token_box_mask,
    _draw_patch_grid,
)
from tracking.te_keep_gumbel_probe import _safe_tag


STOP_WORDS = {
    "a", "an", "the", "of", "on", "in", "at", "to", "for", "from", "with", "by", "and",
    "or", "as", "is", "are", "was", "were", "be", "being", "been", "this", "that", "these",
    "those", "left", "right", "top", "bottom", "front", "back", "behind", "under", "over",
    "near", "next", "around", "beside", "between", "middle", "center", "photo", "image",
    "picture", "background", "foreground",
}

CONTEXT_WORDS = {
    "road", "street", "tree", "grass", "sky", "ground", "water", "wall", "floor", "field",
    "person", "man", "woman", "hand", "bike", "bicycle", "motorcycle", "car", "truck",
    "chair", "table", "building",
}

ATTRIBUTE_WORDS = {
    "black", "white", "red", "green", "blue", "yellow", "brown", "gray", "grey", "orange",
    "small", "large", "big", "tiny", "bright", "dark", "striped", "round", "long", "short",
}


def _token_labels_and_masks(tracker, text, device):
    tokenizer = tracker.network.backbone.tokenizer
    encoded = tokenizer(
        [text], add_special_tokens=True, truncation=True, pad_to_max_length=True,
        max_length=16, return_attention_mask=True)
    ids = encoded["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    valid = _tokenize_mask(tracker, text, device).squeeze(0).squeeze(-1).bool().cpu()
    content = []
    for token, is_valid in zip(tokens, valid.tolist()):
        clean = token.replace("##", "").lower()
        content.append(bool(is_valid) and clean not in STOP_WORDS)
    return tokens, valid, torch.tensor(content, dtype=torch.bool)


def _manual_mask(tokens, subject_words):
    words = [w.strip().lower() for w in str(subject_words or "").split(",") if w.strip()]
    mask = torch.zeros(len(tokens), dtype=torch.bool)
    if not words:
        return mask
    for i, token in enumerate(tokens):
        clean = token.replace("##", "").lower()
        if clean in words:
            mask[i] = True
    return mask


def _context_mask(tokens, valid):
    mask = torch.zeros(len(tokens), dtype=torch.bool)
    for i, token in enumerate(tokens):
        clean = token.replace("##", "").lower()
        if bool(valid[i]) and clean in CONTEXT_WORDS:
            mask[i] = True
    return mask


def _clean_token(token):
    return str(token).replace("##", "").lower()


def _role_scores(tokens, valid_mask, content_mask, ctx_mask, manual_mask, args):
    scores = torch.zeros(len(tokens), dtype=torch.float32)
    labels = []
    for i, token in enumerate(tokens):
        clean = _clean_token(token)
        if not bool(valid_mask[i]):
            labels.append("invalid")
            continue
        if bool(manual_mask[i]):
            scores[i] = float(args.role_manual_weight)
            labels.append("manual_subject")
        elif clean in ATTRIBUTE_WORDS:
            scores[i] = float(args.role_attribute_weight)
            labels.append("attribute")
        elif bool(ctx_mask[i]):
            scores[i] = float(args.role_context_weight)
            labels.append("context")
        elif bool(content_mask[i]):
            scores[i] = float(args.role_content_weight)
            labels.append("content")
        else:
            scores[i] = 0.0
            labels.append("stop")
    return scores, labels


def _gap_per_word(sim_map, pos_mask, neg_mask=None):
    pos = pos_mask.to(device=sim_map.device).bool()
    if neg_mask is None:
        neg = ~pos
    else:
        neg = neg_mask.to(device=sim_map.device).bool()
    if pos.sum().item() == 0 or neg.sum().item() == 0:
        return torch.zeros(sim_map.shape[0], sim_map.shape[-1], device=sim_map.device)
    return sim_map[:, pos, :].mean(dim=1) - sim_map[:, neg, :].mean(dim=1)


def _hard_negative_mask(score_map, gt_mask, topk):
    score = _as_1d(score_map)
    gt = gt_mask.bool()
    mask = torch.zeros_like(gt, dtype=torch.bool)
    if score is None:
        return ~gt
    outside = ~gt
    count = int(outside.sum().item())
    if count <= 0:
        return mask
    k = max(1, min(int(topk), count))
    outside_idx = outside.nonzero(as_tuple=False).view(-1)
    top_local = torch.topk(score[outside], k).indices
    mask[outside_idx[top_local]] = True
    return mask


def _token_box_ring_mask(box, crop_box, feat_sz, scale=2.5):
    crop_x, crop_y, crop_w, crop_h = [float(v) for v in crop_box]
    x, y, w, h = [float(v) for v in box]
    cx, cy = x + 0.5 * w, y + 0.5 * h
    ew, eh = w * float(scale), h * float(scale)
    ex, ey = cx - 0.5 * ew, cy - 0.5 * eh
    cols = torch.arange(feat_sz, dtype=torch.float32) + 0.5
    rows = torch.arange(feat_sz, dtype=torch.float32) + 0.5
    yy, xx = torch.meshgrid(rows, cols, indexing='ij')
    px = crop_x + xx.reshape(-1) / feat_sz * crop_w
    py = crop_y + yy.reshape(-1) / feat_sz * crop_h
    inside = (px >= x) & (px <= x + w) & (py >= y) & (py <= y + h)
    expanded = (px >= ex) & (px <= ex + ew) & (py >= ey) & (py <= ey + eh)
    return expanded & ~inside


def _temporal_consistency(sim_x, prev_word_search):
    cur = sim_x[0].detach()
    if prev_word_search is None or prev_word_search.shape != cur.shape:
        return torch.full((cur.shape[-1],), 0.5, dtype=cur.dtype, device=cur.device)
    cur_t = cur.transpose(0, 1)
    prev_t = prev_word_search.transpose(0, 1).to(device=cur_t.device, dtype=cur_t.dtype)
    cur_t = cur_t - cur_t.mean(dim=1, keepdim=True)
    prev_t = prev_t - prev_t.mean(dim=1, keepdim=True)
    cos = F.cosine_similarity(cur_t, prev_t, dim=1, eps=1e-6)
    return ((cos + 1.0) * 0.5).clamp(0.0, 1.0)


def _tracking_word_weights(sim_z, sim_x, template_gt_mask, search_gt_mask, hardneg_mask,
                           prev_word_search, role_scores, valid_mask, args):
    template_gap = _gap_per_word(sim_z, template_gt_mask)[0].detach()
    search_gap = _gap_per_word(sim_x, search_gt_mask)[0].detach()
    hardneg_gap = _gap_per_word(sim_x, search_gt_mask, hardneg_mask)[0].detach()
    consistency = _temporal_consistency(sim_x, prev_word_search).detach()

    reliability_logit = (
        float(args.reliability_template_weight) * template_gap
        + float(args.reliability_search_weight) * search_gap
        + float(args.reliability_temporal_weight) * (consistency - 0.5)
    )
    reliability = torch.sigmoid(float(args.reliability_scale) * reliability_logit)
    discriminability = torch.sigmoid(float(args.discriminability_scale) * hardneg_gap)
    valid = valid_mask.to(dtype=torch.float32, device=role_scores.device)
    role = role_scores.to(dtype=torch.float32, device=role_scores.device) * valid
    role_rel = role * reliability.cpu()
    tracking = role_rel * discriminability.cpu()
    return {
        "template_gap": template_gap.cpu(),
        "search_gap": search_gap.cpu(),
        "hardneg_gap": hardneg_gap.cpu(),
        "temporal_consistency": consistency.cpu(),
        "role": role.cpu(),
        "reliability": reliability.cpu(),
        "discriminability": discriminability.cpu(),
        "role_rel": role_rel.cpu(),
        "tracking": tracking.cpu(),
    }


def _scores_from_word_weights(sim_z, sim_x, template_tokens, search_tokens, word_weights,
                              valid_mask, temperature):
    weights = word_weights.to(device=sim_z.device, dtype=sim_z.dtype)
    if weights.sum().item() <= 1e-8:
        weights = valid_mask.to(device=sim_z.device, dtype=sim_z.dtype)
    denom = weights.sum().clamp_min(1e-6)
    template_score = (sim_z * weights.view(1, 1, -1)).sum(dim=-1) / denom
    template_weights = torch.softmax(template_score / max(float(temperature), 1e-6), dim=1)
    proto_search = _search_from_template_weights(template_tokens, search_tokens, template_weights)
    direct_search = (sim_x * weights.view(1, 1, -1)).sum(dim=-1) / denom
    return template_score, proto_search, direct_search


def _topk_indices(score, mask, k, largest=True):
    score = _as_1d(score)
    mask = mask.bool().view(-1)
    indices = mask.nonzero(as_tuple=False).view(-1)
    if indices.numel() == 0:
        return torch.arange(min(max(int(k), 1), score.numel()))
    k = max(1, min(int(k), indices.numel()))
    local = torch.topk(score[indices], k, largest=largest).indices
    return indices[local]


def _max_sim_from_indices(template_tokens, search_tokens, indices):
    idx = indices.to(device=template_tokens.device)
    proto = template_tokens[:, idx, :]
    sim = torch.matmul(_unit(search_tokens), _unit(proto).transpose(1, 2))
    return sim.max(dim=-1).values


def _negative_gate(negative_map, target_gt_mask, hardneg_mask, args):
    neg = negative_map[0].detach().float().cpu().view(-1)
    target = target_gt_mask.bool().view(-1)
    hardneg = hardneg_mask.bool().view(-1)
    if target.sum().item() == 0 or hardneg.sum().item() == 0:
        return torch.tensor(float(args.negative_gate_floor), dtype=torch.float32)
    hard_score = neg[hardneg].mean()
    target_score = neg[target].mean()
    gate = torch.sigmoid(float(args.negative_gate_scale) * (hard_score - target_score))
    floor = float(args.negative_gate_floor)
    return gate.clamp(min=floor, max=1.0)


def _multi_prototype_maps(template_tokens, search_tokens, template_score, direct_search,
                          target_mask, context_mask, search_gt_mask, search_hardneg_mask, args):
    score = _as_1d(template_score)
    target_mask = target_mask.bool()
    context_mask = context_mask.bool() & ~target_mask
    out_mask = ~target_mask
    distractor_mask = out_mask & ~context_mask
    if distractor_mask.sum().item() == 0:
        distractor_mask = out_mask
    background_mask = out_mask & ~context_mask
    if background_mask.sum().item() == 0:
        background_mask = out_mask

    target_idx = _topk_indices(score, target_mask, args.proto_topk_target, largest=True)
    context_idx = _topk_indices(score, context_mask, args.proto_topk_negative, largest=True)
    distractor_idx = _topk_indices(score, distractor_mask, args.proto_topk_negative, largest=True)
    background_idx = _topk_indices(score, background_mask, args.proto_topk_negative, largest=False)

    target = _max_sim_from_indices(template_tokens, search_tokens, target_idx)
    context = _max_sim_from_indices(template_tokens, search_tokens, context_idx)
    distractor = _max_sim_from_indices(template_tokens, search_tokens, distractor_idx)
    background = _max_sim_from_indices(template_tokens, search_tokens, background_idx)
    negative = torch.stack([context, distractor, background, torch.zeros_like(target)], dim=0).max(dim=0).values
    contrast = torch.sigmoid((target - negative) / max(float(args.proto_contrast_tau), 1e-6))

    context_gate = _negative_gate(context, search_gt_mask, search_hardneg_mask, args)
    distractor_gate = _negative_gate(distractor, search_gt_mask, search_hardneg_mask, args)
    background_gate = _negative_gate(background, search_gt_mask, search_hardneg_mask, args)
    gated_context = context * context_gate.to(device=context.device, dtype=context.dtype)
    gated_distractor = distractor * distractor_gate.to(device=distractor.device, dtype=distractor.dtype)
    gated_background = background * background_gate.to(device=background.device, dtype=background.dtype)
    safe_negative = torch.stack(
        [gated_context, gated_distractor, gated_background, torch.zeros_like(target)], dim=0).max(dim=0).values
    safe_margin = target - safe_negative
    safe_contrast = torch.sigmoid(safe_margin / max(float(args.proto_contrast_tau), 1e-6))

    direct = direct_search.to(device=target.device, dtype=target.dtype)
    confirm = direct + float(args.confirm_gamma) * F.relu(target - float(args.confirm_tau))
    safe_confirm = direct + float(args.safe_confirm_gamma) * F.relu(
        safe_margin - float(args.safe_confirm_tau)).clamp(max=float(args.safe_confirm_max))
    return {
        "multi_target_max_search": target[0].detach().float().cpu(),
        "multi_context_max_search": context[0].detach().float().cpu(),
        "multi_distractor_max_search": distractor[0].detach().float().cpu(),
        "multi_background_max_search": background[0].detach().float().cpu(),
        "multi_negative_max_search": negative[0].detach().float().cpu(),
        "multi_contrast_search": contrast[0].detach().float().cpu(),
        "direct_multi_confirm_search": confirm[0].detach().float().cpu(),
        "safe_multi_negative_search": safe_negative[0].detach().float().cpu(),
        "safe_multi_margin_search": safe_margin[0].detach().float().cpu(),
        "safe_multi_contrast_search": safe_contrast[0].detach().float().cpu(),
        "direct_safe_multi_confirm_search": safe_confirm[0].detach().float().cpu(),
        "context_negative_gate": context_gate.detach().float().cpu(),
        "distractor_negative_gate": distractor_gate.detach().float().cpu(),
        "background_negative_gate": background_gate.detach().float().cpu(),
    }


def _weighted_proto(tokens, weights):
    weights = weights.clamp_min(0)
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (tokens * weights.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)


def _word_similarity(lang_tokens, template_tokens, search_tokens):
    lang = _unit(lang_tokens)
    template = _unit(template_tokens)
    search = _unit(search_tokens)
    sim_z = torch.matmul(template, lang.transpose(1, 2))
    sim_x = torch.matmul(search, lang.transpose(1, 2))
    return sim_z, sim_x


def _mask_from_words(sim_z, word_mask, valid_mask, temperature):
    word_mask = word_mask.to(device=sim_z.device)
    valid_mask = valid_mask.to(device=sim_z.device)
    selected = word_mask & valid_mask
    if selected.sum().item() == 0:
        selected = valid_mask
    selected_f = selected.float().view(1, 1, -1)
    target_score = (sim_z * selected_f).sum(dim=-1) / selected_f.sum(dim=-1).clamp_min(1.0)
    weights = torch.softmax(target_score / max(float(temperature), 1e-6), dim=1)
    return target_score, weights


def _search_from_template_weights(template_tokens, search_tokens, weights):
    proto = _weighted_proto(template_tokens, weights)
    return (_unit(search_tokens) * _unit(proto)).sum(dim=-1)


def _score_group(name, heat, gt_mask, row, top_ratio):
    raw = _as_1d(heat[0] if isinstance(heat, torch.Tensor) and heat.dim() == 2 else heat)
    heat01 = _minmax_01(raw)
    row.update(_heat_stats("{}_01".format(name), heat01, gt_mask, top_ratio))
    row.update(_region_gap_stats(name, raw, gt_mask))
    row["{}_min".format(name)] = raw.min().item()
    row["{}_max".format(name)] = raw.max().item()


def _word_row_stats(prefix, heat, gt_mask, row):
    raw = _as_1d(heat)
    gap = _region_gap_stats(prefix, raw, gt_mask)
    row.update(gap)
    row["{}_mean_in_gt".format(prefix)] = raw[gt_mask].mean().item() if gt_mask.any() else float("nan")
    row["{}_mean_out_gt".format(prefix)] = raw[~gt_mask].mean().item() if (~gt_mask).any() else float("nan")


def _text_tile(lines, tile_size=142):
    canvas = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
    y = 16
    for line in lines[:9]:
        cv.putText(canvas, _short_label(line, 24), (6, y), cv.FONT_HERSHEY_SIMPLEX,
                   0.34, (35, 35, 35), 1, cv.LINE_AA)
        y += 14
    return canvas


def _save_word_story(save_dir, frame_num, template_img, search_img, maps, top_words):
    tile_size = 142
    template_tile = _plain_tile(template_img, tile_size)
    search_tile = _plain_tile(search_img, tile_size)
    _draw_patch_grid(template_tile, 12)
    _draw_patch_grid(search_tile, 24)

    top_lines = ["top subject:"]
    for token, score in top_words[:6]:
        top_lines.append("{} {:.3f}".format(token.replace("##", ""), score))

    col_labels = [
        "template",
        "top words",
        "all mask z",
        "all -> search",
        "subject mask z",
        "subject -> search",
    ]
    specs = [
        template_tile,
        _text_tile(top_lines, tile_size),
        ("all_template_mask", template_img),
        ("all_search", search_img),
        ("subject_template_mask", template_img),
        ("subject_search", search_img),
    ]
    if maps.get("manual_search") is not None:
        col_labels.extend(["manual mask z", "manual -> search"])
        specs.extend([("manual_template_mask", template_img), ("manual_search", search_img)])
    col_labels.extend(["context -> search", "GT z -> search", "score map"])
    specs.extend([
        ("context_search", search_img),
        ("gt_template_search", search_img),
        ("score_map", search_img),
    ])
    tiles = []
    for spec in specs:
        if isinstance(spec, np.ndarray):
            tile = spec
        else:
            key, bg = spec
            heat = maps.get(key)
            tile = _render_map_tile({"heat": heat, "mode": "minmax", "factor": 0.55}, bg, tile_size)
            feat_sz = _feat_size(_as_1d(heat).numel())
            _draw_patch_grid(tile, feat_sz)
        tiles.append(tile)

    _save_story_grid(
        os.path.join(save_dir, "{:04d}_word_level_appearance_probe.jpg".format(frame_num)),
        "Frame {} word-level appearance modulation probe".format(frame_num),
        "word-template correspondence -> template mask -> search prototype",
        ["word-level"],
        col_labels,
        [tiles],
        note="subject is oracle top-k by template GT contrast for diagnosis, not a deployable selector.",
    )


def _save_tracking_story(save_dir, frame_num, search_img, maps, weighted_words):
    tile_size = 142
    search_tile = _plain_tile(search_img, tile_size)
    _draw_patch_grid(search_tile, 24)
    lines = ["top weights:"]
    for token, weight, role, rel, disc in weighted_words[:6]:
        lines.append("{} {:.2f}/{:.2f}/{:.2f}".format(
            token.replace("##", ""), float(role), float(rel), float(disc)))
    col_labels = [
        "search",
        "weights r/r/d",
        "role direct",
        "role proto",
        "role*rel direct",
        "role*rel proto",
        "full direct",
        "full proto",
        "score map",
    ]
    specs = [
        search_tile,
        _text_tile(lines, tile_size),
        ("role_direct_search", search_img),
        ("role_proto_search", search_img),
        ("role_rel_direct_search", search_img),
        ("role_rel_proto_search", search_img),
        ("tracking_direct_search", search_img),
        ("tracking_proto_search", search_img),
        ("score_map", search_img),
    ]
    tiles = []
    for spec in specs:
        if isinstance(spec, np.ndarray):
            tile = spec
        else:
            key, bg = spec
            heat = maps.get(key)
            tile = _render_map_tile({"heat": heat, "mode": "minmax", "factor": 0.55}, bg, tile_size)
            feat_sz = _feat_size(_as_1d(heat).numel())
            _draw_patch_grid(tile, feat_sz)
        tiles.append(tile)
    _save_story_grid(
        os.path.join(save_dir, "{:04d}_tracking_word_scoring_probe.jpg".format(frame_num)),
        "Frame {} tracking-aware word scoring".format(frame_num),
        "role, reliability, discriminability weighted word-to-visual scores",
        ["tracking"],
        col_labels,
        [tiles],
        note="full = role * reliability * discriminability; direct bypasses template prototype.",
    )


def _save_multi_proto_story(save_dir, frame_num, search_img, maps):
    tile_size = 142
    search_tile = _plain_tile(search_img, tile_size)
    _draw_patch_grid(search_tile, 24)
    col_labels = [
        "search",
        "direct",
        "target max",
        "contrast",
        "safe margin",
        "safe contrast",
        "safe confirm",
        "direct+confirm",
        "score map",
    ]
    keys = [
        None,
        "tracking_direct_search",
        "multi_target_max_search",
        "multi_contrast_search",
        "safe_multi_margin_search",
        "safe_multi_contrast_search",
        "direct_safe_multi_confirm_search",
        "direct_multi_confirm_search",
        "score_map",
    ]
    tiles = []
    for key in keys:
        if key is None:
            tile = search_tile.copy()
        else:
            heat = maps.get(key)
            tile = _render_map_tile({"heat": heat, "mode": "minmax", "factor": 0.55}, search_img, tile_size)
            _draw_patch_grid(tile, _feat_size(_as_1d(heat).numel()))
        tiles.append(tile)
    _save_story_grid(
        os.path.join(save_dir, "{:04d}_multi_prototype_probe.jpg".format(frame_num)),
        "Frame {} multi-prototype probe".format(frame_num),
        "target/context/distractor/background prototypes from template tokens",
        ["multi-proto"],
        col_labels,
        [tiles],
        note="target uses in-box top-k; context/distractor/background use out-box template tokens.",
    )


def _write_summary(save_dir, args, rows, word_rows, description):
    lines = [
        "# Word-Level Appearance Probe",
        "",
        "Config: `{}`".format(args.config),
        "Dataset/sequence: `{}:{}`".format(args.dataset_name, args.sequence),
        "Description: `{}`".format(description),
        "Frames: `{}` | word top-k: `{}` | temperature: `{}`".format(
            args.max_frames, args.word_topk, args.temperature),
        "",
        "| Source | GT mass | top10 precision | in-out gap | raw min | raw max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    source_names = (
        "all_search", "subject_search", "manual_search", "context_search",
        "role_direct_search", "role_proto_search",
        "role_rel_direct_search", "role_rel_proto_search",
        "tracking_direct_search", "tracking_proto_search",
        "multi_target_max_search", "multi_context_max_search",
        "multi_distractor_max_search", "multi_background_max_search",
        "multi_negative_max_search", "multi_contrast_search",
        "direct_multi_confirm_search",
        "safe_multi_negative_search", "safe_multi_margin_search",
        "safe_multi_contrast_search", "direct_safe_multi_confirm_search",
        "gt_template_search", "score_map")
    for name in source_names:
        if not any("{}_01_mass_in_gt".format(name) in row for row in rows):
            continue
        lines.append(
            "| {name} | {mass:.6g} | {top:.6g} | {gap:.6g} | {minv:.6g} | {maxv:.6g} |".format(
                name=name,
                mass=_mean(rows, "{}_01_mass_in_gt".format(name)),
                top=_mean(rows, "{}_01_top10_precision".format(name)),
                gap=_mean(rows, "{}_gap_in_minus_out".format(name)),
                minv=_mean(rows, "{}_min".format(name)),
                maxv=_mean(rows, "{}_max".format(name)),
            )
        )
    lines.extend([
        "",
        "Safe negative gates:",
        "",
        "| gate | mean | min | max |",
        "| --- | ---: | ---: | ---: |",
    ])
    for name in ("context_negative_gate", "distractor_negative_gate", "background_negative_gate"):
        values = []
        for row in rows:
            try:
                value = float(row[name])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        if values:
            lines.append("| {name} | {mean:.6g} | {minv:.6g} | {maxv:.6g} |".format(
                name=name, mean=sum(values) / len(values), minv=min(values), maxv=max(values)))
    lines.extend([
        "",
        "Word ranking by template GT contrast:",
        "",
        "| token | role | valid | content | context | rel | disc | final | template gap | direct gap | hardneg gap | proto gap |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    by_token = {}
    for row in word_rows:
        key = row["token"]
        by_token.setdefault(key, []).append(row)
    ranked = sorted(
        by_token.items(),
        key=lambda item: (
            _mean(item[1], "valid"),
            _mean(item[1], "content"),
            _mean(item[1], "template_word_gap_in_minus_out"),
        ),
        reverse=True)
    for token, token_rows in ranked[:12]:
        lines.append(
            "| {token} | {role_label} | {valid:.0f} | {content:.0f} | {context:.0f} | {rel:.3g} | {disc:.3g} | {final:.3g} | {zgap:.6g} | {xgap:.6g} | {hgap:.6g} | {pgap:.6g} |".format(
                token=token.replace("|", "\\|"),
                role_label=token_rows[0].get("role_label", ""),
                valid=_mean(token_rows, "valid"),
                content=_mean(token_rows, "content"),
                context=_mean(token_rows, "context"),
                rel=_mean(token_rows, "reliability_score"),
                disc=_mean(token_rows, "discriminability_score"),
                final=_mean(token_rows, "tracking_word_weight"),
                zgap=_mean(token_rows, "template_word_gap_in_minus_out"),
                xgap=_mean(token_rows, "direct_word_gap_in_minus_out"),
                hgap=_mean(token_rows, "hardneg_word_gap"),
                pgap=_mean(token_rows, "proto_word_gap_in_minus_out"),
            )
        )
    lines.extend([
        "",
        "Interpretation:",
        "",
        "- `all_search` uses all valid non-special words to build the template mask.",
        "- `subject_search` is an oracle diagnostic: top-k words by template GT in-out contrast.",
        "- `context_search` uses known context/background words when present.",
        "- `gt_template_search` bypasses language and uses GT template-box tokens as an upper bound for template-to-search matching.",
        "- If `subject_search` improves over `all_search`, word-level subject filtering is useful.",
        "- If `gt_template_search` is also weak, the main issue is template-to-search matching, not word selection.",
        "- `tracking_direct_search` tests whether tracking-aware word weights contain a useful search-space signal.",
        "- `tracking_proto_search` tests whether the same weights survive the template-prototype bridge.",
    ])
    with open(os.path.join(save_dir, "word_level_appearance_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _mean(rows, key):
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return sum(values) / len(values) if values else float("nan")


def run(args):
    dataset = get_dataset(args.dataset_name)
    seq = dataset[int(args.sequence)] if str(args.sequence).isdigit() else dataset[args.sequence]
    tracker_info = Tracker("dutrack", args.config, args.dataset_name, args.runid)
    params = tracker_info.get_parameters(run_id=args.runid)
    params.debug = 0
    tracker = tracker_info.create_tracker(params)
    tracker.network.backbone.te_policy_apply = "none"
    tracker.network.score_prior_enabled = False

    run_label = _run_label(args.config, args)
    if args.output_tag:
        run_label = args.output_tag
    elif args.tag:
        run_label = "{}_{}".format(run_label, _safe_tag(args.tag))
    save_dir = os.path.join(args.out_dir, run_label, seq.name)
    os.makedirs(save_dir, exist_ok=True)

    image0 = _read_rgb(seq.frames[0])
    init_info = seq.init_info()
    init_info["class"] = seq.object_class
    init_info["path"] = seq.name
    tracker.initialize(image0, init_info)
    if args.description:
        tracker.descript = args.description

    template_img = getattr(tracker, "z_patch_arr", None)
    if template_img is None:
        template_img, template_resize, _ = sample_target(
            image0, init_info["init_bbox"], params.template_factor, output_sz=params.template_size)
    else:
        _, template_resize, _ = sample_target(
            image0, init_info["init_bbox"], params.template_factor, output_sz=params.template_size)

    search_len = int(tracker.network.backbone.pos_embed_x.shape[1])
    template_len = int(tracker.network.backbone.pos_embed_z.shape[1])
    search_feat_sz = _feat_size(search_len)
    template_feat_sz = _feat_size(template_len)
    template_crop_box = _search_crop_box(init_info["init_bbox"], template_resize, params.template_size)
    template_gt_mask = _token_box_mask(init_info["init_bbox"], template_crop_box, template_feat_sz)
    template_context_mask = _token_box_ring_mask(
        init_info["init_bbox"], template_crop_box, template_feat_sz, scale=args.context_scale)

    rows = []
    word_rows = []
    prev_word_search = None
    max_frame = min(int(args.max_frames), len(seq.frames) - 1)
    for frame_num in range(1, max_frame + 1):
        image = _read_rgb(seq.frames[frame_num])
        crop_state = seq.ground_truth_rect[frame_num - 1].tolist() if args.crop_source == "gt_prev" else list(tracker.state)
        search_img, resize_factor, search_amask = sample_target(
            image, crop_state, params.search_factor, output_sz=params.search_size)
        crop_box = _search_crop_box(crop_state, resize_factor, params.search_size)
        gt_box = seq.ground_truth_rect[frame_num].tolist() if seq.ground_truth_rect is not None else None
        search_gt_mask = _token_box_mask(gt_box, crop_box, search_feat_sz) if gt_box is not None else torch.zeros(search_len, dtype=torch.bool)

        search_nt = tracker.preprocessor.process(search_img, search_amask)
        with torch.no_grad():
            out = tracker.network.forward(
                template=tracker.memory_frames.copy(),
                search=[search_nt.tensors],
                descript=[[getattr(tracker, "descript", "")]],
            )
        out = out[-1] if isinstance(out, list) else out
        _, lang_tokens, template_tokens, search_tokens = _split_tokens(tracker, out)
        tokens, valid_mask, content_mask = _token_labels_and_masks(
            tracker, getattr(tracker, "descript", ""), lang_tokens.device)
        manual_mask = _manual_mask(tokens, args.subject_words)
        ctx_mask = _context_mask(tokens, valid_mask)
        role_scores, role_labels = _role_scores(tokens, valid_mask, content_mask, ctx_mask, manual_mask, args)
        sim_z, sim_x = _word_similarity(lang_tokens, template_tokens, search_tokens)

        score_map = out.get("score_map")
        score_flat = score_map[0].detach().float().cpu().view(-1) if isinstance(score_map, torch.Tensor) else None
        hardneg_mask = _hard_negative_mask(score_flat, search_gt_mask, args.hardneg_topk)
        tracking_scores = _tracking_word_weights(
            sim_z, sim_x, template_gt_mask, search_gt_mask, hardneg_mask,
            prev_word_search, role_scores, valid_mask, args)

        template_scores = []
        for idx, token in enumerate(tokens):
            z_word = sim_z[0, :, idx].detach().float().cpu()
            x_word = sim_x[0, :, idx].detach().float().cpu()
            one_word = torch.zeros_like(valid_mask)
            one_word[idx] = bool(valid_mask[idx])
            _, one_weights = _mask_from_words(sim_z, one_word, valid_mask, args.temperature)
            proto_word = _search_from_template_weights(template_tokens, search_tokens, one_weights)[0].detach().float().cpu()
            word_row = {
                "frame": frame_num,
                "token_idx": idx,
                "token": token,
                "valid": float(valid_mask[idx].item()),
                "content": float(content_mask[idx].item()),
                "context": float(ctx_mask[idx].item()),
                "role_label": role_labels[idx],
                "role_score": float(tracking_scores["role"][idx].item()),
                "reliability_score": float(tracking_scores["reliability"][idx].item()),
                "discriminability_score": float(tracking_scores["discriminability"][idx].item()),
                "role_reliability_weight": float(tracking_scores["role_rel"][idx].item()),
                "tracking_word_weight": float(tracking_scores["tracking"][idx].item()),
                "hardneg_word_gap": float(tracking_scores["hardneg_gap"][idx].item()),
                "temporal_consistency": float(tracking_scores["temporal_consistency"][idx].item()),
            }
            _word_row_stats("template_word", z_word, template_gt_mask, word_row)
            _word_row_stats("direct_word", x_word, search_gt_mask, word_row)
            _word_row_stats("proto_word", proto_word, search_gt_mask, word_row)
            word_rows.append(word_row)
            if bool(content_mask[idx]):
                template_scores.append((idx, word_row["template_word_gap_in_minus_out"]))

        template_scores = sorted(template_scores, key=lambda item: item[1], reverse=True)
        subject_mask = torch.zeros_like(valid_mask)
        for idx, _ in template_scores[:max(1, int(args.word_topk))]:
            subject_mask[idx] = True
        all_mask = content_mask if content_mask.any() else valid_mask

        group_specs = {
            "all": all_mask,
            "subject": subject_mask,
            "context": ctx_mask,
        }
        if manual_mask.any():
            group_specs["manual"] = manual_mask
        maps = {}
        row = {"frame": frame_num, "sequence": seq.name}
        for group_name, word_mask in group_specs.items():
            template_mask, weights = _mask_from_words(sim_z, word_mask, valid_mask, args.temperature)
            search_map = _search_from_template_weights(template_tokens, search_tokens, weights)
            maps["{}_template_mask".format(group_name)] = template_mask[0].detach().float().cpu()
            maps["{}_search".format(group_name)] = search_map[0].detach().float().cpu()
            _score_group("{}_search".format(group_name), search_map[0].detach().float().cpu(), search_gt_mask, row, args.top_ratio)

        tracking_variants = {
            "role": tracking_scores["role"],
            "role_rel": tracking_scores["role_rel"],
            "tracking": tracking_scores["tracking"],
        }
        tracking_template_score = None
        tracking_direct_search = None
        for variant_name, word_weights in tracking_variants.items():
            template_mask, proto_search, direct_search = _scores_from_word_weights(
                sim_z, sim_x, template_tokens, search_tokens, word_weights,
                valid_mask, args.temperature)
            proto_name = "{}_proto_search".format(variant_name)
            direct_name = "{}_direct_search".format(variant_name)
            template_name = "{}_template_mask".format(variant_name)
            maps[template_name] = template_mask[0].detach().float().cpu()
            maps[proto_name] = proto_search[0].detach().float().cpu()
            maps[direct_name] = direct_search[0].detach().float().cpu()
            _score_group(proto_name, maps[proto_name], search_gt_mask, row, args.top_ratio)
            _score_group(direct_name, maps[direct_name], search_gt_mask, row, args.top_ratio)
            if variant_name == "tracking":
                tracking_template_score = template_mask
                tracking_direct_search = direct_search

        gt_weights = template_gt_mask.to(device=template_tokens.device, dtype=template_tokens.dtype).unsqueeze(0)
        gt_search = _search_from_template_weights(template_tokens, search_tokens, gt_weights)
        maps["gt_template_search"] = gt_search[0].detach().float().cpu()
        _score_group("gt_template_search", maps["gt_template_search"], search_gt_mask, row, args.top_ratio)

        if tracking_template_score is not None and tracking_direct_search is not None:
            multi_maps = _multi_prototype_maps(
                template_tokens, search_tokens, tracking_template_score[0].detach().float().cpu(),
                tracking_direct_search, template_gt_mask, template_context_mask,
                search_gt_mask, hardneg_mask, args)
            maps.update(multi_maps)
            for name, heat in multi_maps.items():
                if name.endswith("_gate"):
                    row[name] = float(heat.item() if isinstance(heat, torch.Tensor) else heat)
                else:
                    _score_group(name, heat, search_gt_mask, row, args.top_ratio)

        if score_flat is not None:
            maps["score_map"] = score_flat
            _score_group("score_map", maps["score_map"], search_gt_mask, row, args.top_ratio)

        rows.append(row)
        top_words = [(tokens[idx], score) for idx, score in template_scores]
        _save_word_story(save_dir, frame_num, template_img, search_img, maps, top_words)
        weighted_words = []
        for idx, token in enumerate(tokens):
            weighted_words.append((
                token,
                tracking_scores["tracking"][idx].item(),
                tracking_scores["role"][idx].item(),
                tracking_scores["reliability"][idx].item(),
                tracking_scores["discriminability"][idx].item(),
            ))
        weighted_words = sorted(weighted_words, key=lambda item: item[1], reverse=True)
        _save_tracking_story(save_dir, frame_num, search_img, maps, weighted_words)
        _save_multi_proto_story(save_dir, frame_num, search_img, maps)
        prev_word_search = sim_x[0].detach()

    csv_path = os.path.join(save_dir, "word_level_appearance_probe.csv")
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    word_csv_path = os.path.join(save_dir, "word_level_word_stats.csv")
    word_fieldnames = sorted(set().union(*(row.keys() for row in word_rows))) if word_rows else []
    with open(word_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=word_fieldnames)
        writer.writeheader()
        for row in word_rows:
            writer.writerow(row)

    _write_summary(save_dir, args, rows, word_rows, getattr(tracker, "descript", ""))
    print("Saved word-level appearance probe to {}".format(save_dir))


def main():
    parser = argparse.ArgumentParser(description="Probe word-level language-to-template appearance modulation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_name", default="otb_lang")
    parser.add_argument("--sequence", default="Biker")
    parser.add_argument("--runid", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=5)
    parser.add_argument("--crop_source", choices=("gt_prev", "tracker"), default="gt_prev")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--top_ratio", type=float, default=0.1)
    parser.add_argument("--word_topk", type=int, default=2)
    parser.add_argument("--subject_words", default="", help="Optional comma-separated manual subject tokens, e.g. head,biker.")
    parser.add_argument("--description", default="", help="Override tracker-generated language for source diagnosis.")
    parser.add_argument("--hardneg_topk", type=int, default=24)
    parser.add_argument("--proto_topk_target", type=int, default=4)
    parser.add_argument("--proto_topk_negative", type=int, default=8)
    parser.add_argument("--context_scale", type=float, default=2.5)
    parser.add_argument("--proto_contrast_tau", type=float, default=0.07)
    parser.add_argument("--confirm_gamma", type=float, default=0.5)
    parser.add_argument("--confirm_tau", type=float, default=0.0)
    parser.add_argument("--negative_gate_scale", type=float, default=8.0)
    parser.add_argument("--negative_gate_floor", type=float, default=0.05)
    parser.add_argument("--safe_confirm_gamma", type=float, default=0.35)
    parser.add_argument("--safe_confirm_tau", type=float, default=0.0)
    parser.add_argument("--safe_confirm_max", type=float, default=0.25)
    parser.add_argument("--role_manual_weight", type=float, default=1.0)
    parser.add_argument("--role_content_weight", type=float, default=0.75)
    parser.add_argument("--role_attribute_weight", type=float, default=0.8)
    parser.add_argument("--role_context_weight", type=float, default=0.35)
    parser.add_argument("--reliability_scale", type=float, default=5.0)
    parser.add_argument("--reliability_template_weight", type=float, default=0.45)
    parser.add_argument("--reliability_search_weight", type=float, default=0.35)
    parser.add_argument("--reliability_temporal_weight", type=float, default=0.20)
    parser.add_argument("--discriminability_scale", type=float, default=8.0)
    parser.add_argument("--tag", default="word_level")
    parser.add_argument("--output_tag", default=None)
    parser.add_argument("--out_dir", default="output/test/word_level_appearance_probe")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
