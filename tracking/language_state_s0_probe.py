import argparse
import csv
import math
import os
import sys

import torch
import torch.nn.functional as F

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import box_xywh_to_xyxy, box_iou
from tracking.visualte_diagnostic import (
    _as_tensor_1d,
    _feat_size,
    _hard_negative_stats,
    _heat_stats,
    _read_rgb,
    _region_gap_stats,
    _run_label,
    _search_crop_box,
    _token_box_mask,
)
from tracking.language_visual_source_probe import _tokenize_mask


def _safe_tag(text):
    return str(text).lower().replace("-", "m").replace(".", "p").replace("/", "_")


_STOP_WORDS = {
    "a", "an", "the", "of", "on", "in", "at", "by", "with", "and", "or", "to", "from",
    "is", "are", "was", "were", "this", "that", "there", "as", "for", "near",
}

CONTEXT_WORDS = {
    "road", "street", "tree", "grass", "sky", "ground", "water", "wall", "floor", "field",
    "person", "people", "man", "woman", "hand", "hands", "room", "table", "chair", "building",
    "background", "foreground", "car", "truck", "bike", "bicycle", "motorcycle",
}


def _mean(rows, key):
    values = []
    for row in rows:
        try:
            value = float(row.get(key, "nan"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return sum(values) / len(values) if values else float("nan")


def _sum(rows, key):
    total = 0.0
    seen = False
    for row in rows:
        try:
            value = float(row.get(key, "nan"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            total += value
            seen = True
    return total if seen else float("nan")


def _restore_track_query(network, saved_track_query):
    if saved_track_query is None:
        network.track_query = None
    else:
        network.track_query = saved_track_query.clone()


def _forward_description(tracker, template_list, search_tensor, description):
    saved_track_query = None
    if getattr(tracker.network, "track_query", None) is not None:
        saved_track_query = tracker.network.track_query.detach().clone()
    try:
        with torch.no_grad():
            out = tracker.network.forward(
                template=template_list.copy(),
                search=[search_tensor],
                descript=[[description]],
                language_word_reliability=[None],
            )
    finally:
        _restore_track_query(tracker.network, saved_track_query)
    if isinstance(out, list):
        out = out[-1]
    return out


def _forward_language_token_state(tracker, template_list, search_tensor, token_state, token_mask):
    saved_track_query = None
    if getattr(tracker.network, "track_query", None) is not None:
        saved_track_query = tracker.network.track_query.detach().clone()
    try:
        with torch.no_grad():
            out = tracker.network.forward(
                template=template_list.copy(),
                search=[search_tensor],
                descript=[[""]],
                language_word_reliability=[None],
                language_token_state=[token_state],
                language_token_mask=[token_mask],
            )
    finally:
        _restore_track_query(tracker.network, saved_track_query)
    if isinstance(out, list):
        out = out[-1]
    return out


def _encode_language_tokens(tracker, description):
    with torch.no_grad():
        tokens, mask = tracker.network.backbone._l_feat([str(description or "")])
    return tokens.detach(), mask.detach()


def _parse_float_list(text):
    values = []
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return values


def _alpha_tag(alpha):
    return "a{:03d}".format(int(round(float(alpha) * 100.0)))


def _last_stage(values):
    if not values:
        return None
    value = values[-1]
    if isinstance(value, torch.Tensor):
        return value
    return None


def _prior_from_output(out):
    prior = _last_stage(out.get("lmq_prior_scores", None))
    if prior is None:
        return None
    return prior[0].detach().float().cpu().view(-1)


def _score_from_output(out):
    score = out.get("score_map", None)
    if not isinstance(score, torch.Tensor):
        return None
    return score[0].detach().float().cpu().view(-1)


def _bbox_iou(pred_box, gt_box, device):
    if pred_box is None or gt_box is None:
        return float("nan")
    pred = torch.tensor(pred_box, dtype=torch.float32, device=device).view(1, 4)
    gt = torch.tensor(gt_box, dtype=torch.float32, device=device).view(1, 4)
    pred_xyxy = box_xywh_to_xyxy(pred)
    gt_xyxy = box_xywh_to_xyxy(gt)
    return box_iou(pred_xyxy, gt_xyxy)[0].item()


def _predict_box(tracker, out, resize_factor):
    score_map = out.get("score_map", None)
    size_map = out.get("size_map", None)
    offset_map = out.get("offset_map", None)
    if not all(isinstance(x, torch.Tensor) for x in (score_map, size_map, offset_map)):
        return None
    response = tracker.output_window * score_map
    pred_boxes = tracker.network.box_head.cal_bbox(response, size_map, offset_map)
    pred_boxes = pred_boxes.view(-1, 4)
    pred_box = (pred_boxes.mean(dim=0) * tracker.params.search_size / resize_factor).detach().cpu().tolist()
    return tracker.map_box_back(pred_box, resize_factor)


def _source_stats(prefix, prior, gt_mask, score_ref, top_ratio, hardneg_topk):
    row = {}
    if prior is None:
        return row
    row.update(_heat_stats(prefix, prior, gt_mask, top_ratio))
    row.update(_region_gap_stats(prefix, prior, gt_mask))
    row.update(_hard_negative_stats(prefix, prior, gt_mask, score_ref, hardneg_topk))
    row["{}_min".format(prefix)] = prior.min().item()
    row["{}_max".format(prefix)] = prior.max().item()
    row["{}_std".format(prefix)] = prior.std(unbiased=False).item()
    return row


def _gap_with_mask(heat, pos_mask, score_ref, hardneg_topk):
    heat = _as_tensor_1d(heat)
    score_ref = _as_tensor_1d(score_ref)
    if heat is None or score_ref is None or heat.numel() != score_ref.numel():
        return float("nan")
    pos_mask = pos_mask.bool()
    neg_mask = ~pos_mask
    if not pos_mask.any() or not neg_mask.any():
        return float("nan")
    hard_k = max(1, min(int(hardneg_topk), int(neg_mask.sum().item())))
    neg_indices = neg_mask.nonzero(as_tuple=False).view(-1)
    hard_local = torch.topk(score_ref[neg_indices], hard_k).indices
    hard_indices = neg_indices[hard_local]
    return heat[pos_mask].mean().item() - heat[hard_indices].mean().item()


def _score_confidence(score_map):
    score = _as_tensor_1d(score_map)
    if score is None or score.numel() == 0:
        return {
            "score_peak": float("nan"),
            "score_second_peak": float("nan"),
            "score_peak_second_gap": float("nan"),
        }
    values = torch.topk(score, min(2, score.numel())).values
    peak = values[0].item()
    second = values[1].item() if values.numel() > 1 else float("nan")
    return {
        "score_peak": peak,
        "score_second_peak": second,
        "score_peak_second_gap": peak - second if math.isfinite(second) else float("nan"),
    }


def _score_entropy(score_map):
    score = _as_tensor_1d(score_map)
    if score is None or score.numel() <= 1:
        return float("nan")
    prob = torch.softmax(score.float(), dim=0).clamp_min(1e-12)
    entropy = -(prob * prob.log()).sum()
    return (entropy / math.log(float(prob.numel()))).item()


def _finite_or_zero(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _state_change_evidence(tracker):
    center_distance = float(getattr(tracker, "language_trigger_center_distance", float("nan")))
    area_ratio = float(getattr(tracker, "language_trigger_area_ratio", float("nan")))
    color_delta = float(getattr(tracker, "language_trigger_color_delta", float("nan")))
    ref_box = getattr(tracker, "his_state", None)
    if ref_box is None:
        ref_box = getattr(tracker, "state", None)
    scale = float("nan")
    if ref_box is not None and len(ref_box) >= 4:
        try:
            scale = math.sqrt(max(float(ref_box[2]) * float(ref_box[3]), 1e-12))
        except (TypeError, ValueError):
            scale = float("nan")
    center_motion_norm = center_distance / scale if math.isfinite(center_distance) and math.isfinite(scale) else float("nan")
    scale_change_ratio = abs(math.log(max(area_ratio, 1e-12))) if math.isfinite(area_ratio) and area_ratio > 0 else float("nan")
    color_change_norm = color_delta / 255.0 if math.isfinite(color_delta) else float("nan")
    return center_motion_norm, scale_change_ratio, color_change_norm


def _learned_state_visual_evidence(tracker, prev_out, resize_factor, deploy_gaps):
    score = _score_from_output(prev_out)
    conf = _score_confidence(score)
    pred_box = _predict_box(tracker, prev_out, resize_factor)
    jump = _box_jump_ratio(pred_box, tracker.state)
    score_entropy = _score_entropy(score)
    center_motion_norm, scale_change_ratio, color_change_norm = _state_change_evidence(tracker)
    prev_gap = deploy_gaps.get("prev", float("nan"))
    blip_gap = deploy_gaps.get("blip", float("nan"))
    best_partial_gap = deploy_gaps.get("best_partial", float("nan"))
    deploy_score_delta = blip_gap - prev_gap if math.isfinite(blip_gap) and math.isfinite(prev_gap) else float("nan")
    partial_deploy_delta = (
        best_partial_gap - prev_gap
        if math.isfinite(best_partial_gap) and math.isfinite(prev_gap)
        else float("nan")
    )
    values = [
        _finite_or_zero(center_motion_norm),
        _finite_or_zero(scale_change_ratio),
        _finite_or_zero(color_change_norm),
        _finite_or_zero(conf.get("score_peak_second_gap", 0.0)),
        _finite_or_zero(score_entropy),
        _finite_or_zero(jump),
        _finite_or_zero(deploy_score_delta),
        _finite_or_zero(partial_deploy_delta),
    ]
    device = tracker.network.backbone.pos_embed_x.device
    dtype = tracker.network.backbone.pos_embed_x.dtype
    tensor = torch.tensor(values, device=device, dtype=dtype).view(1, -1)
    diagnostics = {
        "token_learned_state_center_motion_norm": values[0],
        "token_learned_state_scale_change_ratio": values[1],
        "token_learned_state_color_change_norm": values[2],
        "token_learned_conf_peak_gap": values[3],
        "token_learned_conf_score_entropy": values[4],
        "token_learned_conf_box_jump": values[5],
        "token_learned_candidate_deploy_score_delta": values[6],
        "token_learned_candidate_partial_deploy_delta": values[7],
    }
    return tensor, diagnostics


def _box_jump_ratio(pred_box, prev_box):
    if pred_box is None or prev_box is None:
        return float("nan")
    px, py, pw, ph = [float(v) for v in pred_box]
    qx, qy, qw, qh = [float(v) for v in prev_box]
    pcx, pcy = px + 0.5 * pw, py + 0.5 * ph
    qcx, qcy = qx + 0.5 * qw, qy + 0.5 * qh
    distance = math.sqrt((pcx - qcx) ** 2 + (pcy - qcy) ** 2)
    scale = math.sqrt(max(qw * qh, 1e-12))
    return distance / max(scale, 1e-12)


def _select_evidence_signal(name, scores, priors, evidence_source):
    if evidence_source == "score":
        return scores.get(name)
    if evidence_source in ("lmq", "lmq_prior"):
        return priors.get(name)
    raise ValueError("Unsupported evidence_source: {}".format(evidence_source))


def _select_best_source(gaps):
    best_name = ""
    best_gap = -float("inf")
    for name, gap in gaps.items():
        if gap is None:
            continue
        try:
            value = float(gap)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > best_gap:
            best_name = name
            best_gap = value
    return best_name, best_gap


def _unit(x):
    return F.normalize(x, dim=-1, eps=1e-6)


def _nan():
    return float("nan")


def _content_words(text):
    words = []
    for part in str(text).lower().replace("-", " ").replace("_", " ").split():
        token = "".join(ch for ch in part if ch.isalnum())
        if token and token not in _STOP_WORDS:
            words.append(token)
    return set(words)


def _content_word_list(text):
    words = []
    for part in str(text).lower().replace("-", " ").replace("_", " ").split():
        token = "".join(ch for ch in part if ch.isalnum())
        if token and token not in _STOP_WORDS:
            words.append(token)
    return words


def _compose_anchor_state_description(anchor, candidate, max_state_words=6):
    anchor = " ".join(str(anchor or "").split())
    candidate = " ".join(str(candidate or "").split())
    if not anchor:
        return candidate
    if not candidate:
        return anchor
    anchor_words = set(_content_word_list(anchor))
    state_words = []
    for word in _content_word_list(candidate):
        if word in anchor_words:
            continue
        if word in state_words:
            continue
        state_words.append(word)
        if len(state_words) >= int(max_state_words):
            break
    if not state_words:
        return anchor
    return "{} {}".format(anchor, " ".join(state_words))


def _compose_with_words(base, words):
    base = " ".join(str(base or "").split())
    selected = []
    seen = set(_content_word_list(base))
    for word in words:
        clean = "".join(ch for ch in str(word).lower() if ch.isalnum())
        if not clean or clean in seen or clean in selected:
            continue
        selected.append(clean)
    if not selected:
        return base
    if not base:
        return " ".join(selected)
    return "{} {}".format(base, " ".join(selected))


def _finite_gain(value, base):
    try:
        value = float(value)
        base = float(base)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(value) or not math.isfinite(base):
        return float("nan")
    return value - base


def _word_gate_empty():
    return {
        "word_gate_word_count": 0,
        "word_gate_selected_count": 0,
        "word_gate_selected_words": "",
        "word_gate_best_word": "",
        "word_gate_best_word_deploy_gain": float("nan"),
        "word_gate_mean_selected_deploy_gain": float("nan"),
        "word_gate_candidate_words": "",
    }


def _word_gate_select_words(tracker, template_list, search_tensor, base_description,
                            blip_description, pred_mask, score_ref, prev_deploy_gap, args):
    if not blip_description or pred_mask is None or not math.isfinite(float(prev_deploy_gap)):
        return [], _word_gate_empty()
    base_words = set(_content_word_list(base_description))
    candidate_words = []
    for word in _content_word_list(blip_description):
        if word in base_words or word in candidate_words:
            continue
        candidate_words.append(word)
        if len(candidate_words) >= int(args.word_gate_max_candidate_words):
            break
    if not candidate_words:
        return [], _word_gate_empty()

    scored = []
    for word in candidate_words:
        word_description = _compose_with_words(base_description, [word])
        word_out = _forward_description(tracker, template_list, search_tensor, word_description)
        word_score = _score_from_output(word_out)
        word_deploy_gap = _gap_with_mask(word_score, pred_mask, score_ref, args.hardneg_topk)
        word_deploy_gain = _finite_gain(word_deploy_gap, prev_deploy_gap)
        if math.isfinite(word_deploy_gain):
            scored.append((word, word_deploy_gain, word_deploy_gap))

    min_gain = max(float(args.quality_gate_gap_eps), float(args.word_gate_min_deploy_gain))
    selected = [
        (word, gain, gap) for word, gain, gap in scored
        if gain > min_gain and word not in CONTEXT_WORDS
    ]
    selected.sort(key=lambda item: item[1], reverse=True)
    selected = selected[:int(args.word_gate_max_selected_words)]
    selected_words = [word for word, _, _ in selected]
    selected_gains = [gain for _, gain, _ in selected]
    best = max(scored, key=lambda item: item[1]) if scored else ("", float("nan"), float("nan"))
    meta = {
        "word_gate_word_count": len(candidate_words),
        "word_gate_selected_count": len(selected_words),
        "word_gate_selected_words": " ".join(selected_words),
        "word_gate_best_word": best[0],
        "word_gate_best_word_deploy_gain": best[1],
        "word_gate_mean_selected_deploy_gain": (
            sum(selected_gains) / len(selected_gains) if selected_gains else float("nan")
        ),
        "word_gate_candidate_words": " ".join(candidate_words),
    }
    return selected_words, meta


def _add_token_state_candidates(outputs, descriptions, tracker, template_list, search_tensor,
                                anchor_description, prev_description, blip_description,
                                visual_evidence, visual_evidence_diag, args):
    if not (args.token_state_probe or args.learned_token_state_probe) or not blip_description:
        return [], {}
    anchor_tokens, anchor_mask = _encode_language_tokens(tracker, anchor_description)
    prev_tokens, prev_mask = _encode_language_tokens(tracker, prev_description)
    blip_tokens, blip_mask = _encode_language_tokens(tracker, blip_description)
    token_sources = []
    diagnostics = {}
    if args.token_state_probe:
        alphas = _parse_float_list(args.token_state_alphas)
        for alpha in alphas:
            tag = _alpha_tag(alpha)
            alpha = float(alpha)
            candidates = {
                "token_anchor_res_{}".format(tag): (
                    anchor_tokens + alpha * (blip_tokens - anchor_tokens),
                    torch.maximum(anchor_mask, blip_mask),
                ),
                "token_prev_res_{}".format(tag): (
                    prev_tokens + alpha * (blip_tokens - prev_tokens),
                    torch.maximum(prev_mask, blip_mask),
                ),
            }
            for name, (tokens, mask) in candidates.items():
                outputs[name] = _forward_language_token_state(
                    tracker, template_list, search_tensor, tokens, mask)
                descriptions[name] = name
                token_sources.append(name)
    if args.learned_token_state_probe:
        updater = getattr(tracker.network.backbone, "language_state_updater", None)
        if updater is None:
            diagnostics["token_learned_state_available"] = 0.0
        else:
            diagnostics.update(visual_evidence_diag)
            with torch.no_grad():
                learned_tokens, learned_diag = updater(
                    anchor_tokens, prev_tokens, blip_tokens,
                    anchor_mask=anchor_mask, prev_mask=prev_mask,
                    candidate_mask=blip_mask,
                    visual_evidence=visual_evidence)
            learned_mask = torch.maximum(prev_mask, blip_mask)
            name = "token_learned_state"
            outputs[name] = _forward_language_token_state(
                tracker, template_list, search_tensor, learned_tokens, learned_mask)
            descriptions[name] = name
            token_sources.append(name)
            diagnostics["token_learned_state_available"] = 1.0
            for key, value in learned_diag.items():
                if isinstance(value, torch.Tensor):
                    value = value.detach().float().cpu().item()
                diagnostics["token_learned_{}".format(key)] = float(value)
    return token_sources, diagnostics


def _token_words(tracker, text, device):
    tokenizer = tracker.network.backbone.tokenizer
    encoded = tokenizer(
        [text], add_special_tokens=True, truncation=True, pad_to_max_length=True,
        max_length=16, return_attention_mask=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    valid = _tokenize_mask(tracker, text, device).squeeze(0).squeeze(-1).bool().cpu()
    words = []
    content = []
    for token, is_valid in zip(tokens, valid.tolist()):
        clean = str(token).replace("##", "").lower()
        clean = "".join(ch for ch in clean if ch.isalnum())
        words.append(clean)
        content.append(bool(is_valid) and bool(clean) and clean not in _STOP_WORDS)
    return words, valid, torch.tensor(content, dtype=torch.bool)


def _mean_or_nan(values):
    if isinstance(values, torch.Tensor):
        if values.numel() == 0:
            return float("nan")
        return values.float().mean().item()
    return float("nan")


def _max_or_nan(values):
    if isinstance(values, torch.Tensor):
        if values.numel() == 0:
            return float("nan")
        return values.float().max().item()
    return float("nan")


def _gap_per_word(sim, pos_mask, neg_mask=None):
    pos = pos_mask.to(device=sim.device).bool().view(-1)
    if neg_mask is None:
        neg = ~pos
    else:
        neg = neg_mask.to(device=sim.device).bool().view(-1)
    if not pos.any() or not neg.any() or sim.shape[1] != pos.numel():
        return None
    return sim[:, pos, :].mean(dim=1).squeeze(0) - sim[:, neg, :].mean(dim=1).squeeze(0)


def _split_tokens_for_word_evidence(tracker, out, template_mask):
    feat = out.get("backbone_feat")
    if isinstance(feat, list):
        feat = feat[-1]
    if not isinstance(feat, torch.Tensor) or feat.dim() != 3:
        raise RuntimeError("backbone_feat must be a B x N x C tensor.")
    search_len = int(tracker.network.backbone.pos_embed_x.shape[1])
    search_tokens = feat[:, -search_len:, :]
    prefix = feat[:, :-search_len, :]
    lang_len = 16
    if template_mask is not None:
        template_len = int(template_mask.numel())
    else:
        template_unit_len = int(tracker.network.backbone.pos_embed_z.shape[1])
        template_len = min(template_unit_len, max(prefix.shape[1] - lang_len, 0))
    if template_len <= 0 or prefix.shape[1] < template_len + lang_len:
        raise RuntimeError(
            "Cannot split word-evidence tokens: prefix={}, template={}, lang={}.".format(
                prefix.shape[1], template_len, lang_len))
    template_tokens = prefix[:, -template_len:, :]
    lang_tokens = prefix[:, :prefix.shape[1] - template_len][:, -lang_len:, :]
    return lang_tokens, template_tokens, search_tokens


def _group_gap_stats(prefix, gaps, mask, row):
    if gaps is None:
        row["{}_mean".format(prefix)] = float("nan")
        row["{}_max".format(prefix)] = float("nan")
        row["{}_count".format(prefix)] = 0
        return
    mask = mask.to(device=gaps.device).bool()
    selected = gaps[mask]
    row["{}_mean".format(prefix)] = _mean_or_nan(selected)
    row["{}_max".format(prefix)] = _max_or_nan(selected)
    row["{}_count".format(prefix)] = int(mask.sum().item())


def _word_evidence_for_description(tracker, out, description, template_mask, search_pos_mask,
                                   anchor_words, prev_words, prefix):
    row = {}
    if not description or not isinstance(out, dict):
        return row
    try:
        lang_tokens, template_tokens, search_tokens = _split_tokens_for_word_evidence(
            tracker, out, template_mask)
    except Exception:
        return row
    device = lang_tokens.device
    words, _, content_mask = _token_words(tracker, description, device)
    content_mask = content_mask.to(device=device)
    target_set = set(anchor_words) | set(prev_words)
    target_mask = torch.tensor(
        [bool(content_mask[i].item()) and words[i] in target_set for i in range(len(words))],
        dtype=torch.bool, device=device)
    context_mask = torch.tensor(
        [bool(content_mask[i].item()) and words[i] in CONTEXT_WORDS for i in range(len(words))],
        dtype=torch.bool, device=device)
    new_mask = content_mask & ~target_mask
    target_for_gap = target_mask if target_mask.any() else content_mask

    sim_z = torch.matmul(_unit(template_tokens), _unit(lang_tokens).transpose(1, 2))
    sim_x = torch.matmul(_unit(search_tokens), _unit(lang_tokens).transpose(1, 2))
    template_mask = template_mask.to(device=device).bool().view(-1) if template_mask is not None else None
    if template_mask is not None and template_mask.numel() != template_tokens.shape[1]:
        template_mask = None
    search_pos_mask = search_pos_mask.to(device=device).bool().view(-1) if search_pos_mask is not None else None
    if search_pos_mask is not None and search_pos_mask.numel() != search_tokens.shape[1]:
        search_pos_mask = None

    template_gap = _gap_per_word(sim_z, template_mask) if template_mask is not None else None
    search_gap = _gap_per_word(sim_x, search_pos_mask) if search_pos_mask is not None else None

    row["{}_content_word_count".format(prefix)] = int(content_mask.sum().item())
    row["{}_target_word_overlap_count".format(prefix)] = int(target_mask.sum().item())
    row["{}_target_word_missing_flag".format(prefix)] = 0.0 if target_mask.any() else 1.0
    row["{}_context_word_count".format(prefix)] = int(context_mask.sum().item())
    denom = max(int(content_mask.sum().item()), 1)
    row["{}_context_dominance".format(prefix)] = float(context_mask.sum().item()) / float(denom)

    _group_gap_stats("{}_target_template_gap".format(prefix), template_gap, target_for_gap, row)
    _group_gap_stats("{}_context_template_gap".format(prefix), template_gap, context_mask, row)
    _group_gap_stats("{}_new_template_gap".format(prefix), template_gap, new_mask, row)
    _group_gap_stats("{}_target_search_deploy_gap".format(prefix), search_gap, target_for_gap, row)
    _group_gap_stats("{}_context_search_deploy_gap".format(prefix), search_gap, context_mask, row)
    _group_gap_stats("{}_new_search_deploy_gap".format(prefix), search_gap, new_mask, row)

    target_template = row.get("{}_target_template_gap_mean".format(prefix), float("nan"))
    context_template = row.get("{}_context_template_gap_mean".format(prefix), float("nan"))
    target_search = row.get("{}_target_search_deploy_gap_mean".format(prefix), float("nan"))
    context_search = row.get("{}_context_search_deploy_gap_mean".format(prefix), float("nan"))
    row["{}_target_minus_context_template_gap".format(prefix)] = (
        target_template - context_template
        if math.isfinite(target_template) and math.isfinite(context_template)
        else float("nan")
    )
    row["{}_target_minus_context_search_deploy_gap".format(prefix)] = (
        target_search - context_search
        if math.isfinite(target_search) and math.isfinite(context_search)
        else float("nan")
    )
    return row


def _selected_template_mask(tracker, template_list):
    if not getattr(tracker.cfg.MODEL.BACKBONE, "CE_LOC", False):
        return None
    masks = getattr(tracker, "memory_masks", [])
    if not masks:
        return None
    if tracker.frame_id <= tracker.cfg.TEST.TEMPLATE_NUMBER:
        return torch.cat(masks, dim=1).detach().bool().cpu().view(-1)
    try:
        _, selected = tracker.select_memory_frames()
    except Exception:
        selected = None
    if selected is not None:
        return selected.detach().bool().cpu().view(-1)
    expected = sum(int(frame.shape[1]) for frame in template_list)
    joined = torch.cat(masks, dim=1).detach().bool().cpu().view(-1)
    return joined[:expected] if joined.numel() >= expected else None


def _add_word_evidence(row, tracker, outputs, descriptions, template_mask, pred_mask):
    anchor_words = _content_word_list(descriptions.get("anchor", ""))
    prev_words = _content_word_list(descriptions.get("prev", ""))
    if "blip" in outputs:
        row.update(_word_evidence_for_description(
            tracker, outputs.get("blip"), descriptions.get("blip", ""),
            template_mask, pred_mask, anchor_words, prev_words, "blip_word"))
    if "prev" in outputs:
        row.update(_word_evidence_for_description(
            tracker, outputs.get("prev"), descriptions.get("prev", ""),
            template_mask, pred_mask, anchor_words, prev_words, "prev_word"))
    blip_template = row.get("blip_word_target_template_gap_mean", float("nan"))
    prev_template = row.get("prev_word_target_template_gap_mean", float("nan"))
    blip_search = row.get("blip_word_target_search_deploy_gap_mean", float("nan"))
    prev_search = row.get("prev_word_target_search_deploy_gap_mean", float("nan"))
    row["blip_minus_prev_target_template_gap"] = (
        blip_template - prev_template
        if math.isfinite(blip_template) and math.isfinite(prev_template)
        else float("nan")
    )
    row["blip_minus_prev_target_search_deploy_gap"] = (
        blip_search - prev_search
        if math.isfinite(blip_search) and math.isfinite(prev_search)
        else float("nan")
    )
    return row


def _text_jaccard(a, b):
    wa = _content_words(a)
    wb = _content_words(b)
    if not wa or not wb:
        return 0.0
    return float(len(wa & wb)) / float(len(wa | wb))


def _content_overlap_count(a, b):
    return len(_content_words(a) & _content_words(b))


def _quality_gate(row, descriptions, args, deploy_gate_available=True):
    blip_gap = row.get("blip_pos_hardneg_gap", float("nan"))
    prev_gap = row.get("prev_pos_hardneg_gap", float("nan"))
    if not (math.isfinite(float(blip_gap)) and math.isfinite(float(prev_gap))):
        return {
            "quality_gate_observable": 0.0,
            "quality_gate_accept": float("nan"),
            "quality_gate_gain_over_prev": float("nan"),
            "quality_gate_score_delta": float("nan"),
            "quality_gate_semantic": float("nan"),
            "quality_gate_source": "unavailable",
            "quality_gate_true_accept": float("nan"),
            "quality_gate_false_reject": float("nan"),
            "quality_gate_true_reject": float("nan"),
            "quality_gate_false_accept": float("nan"),
        }
    blip_gap = float(blip_gap)
    prev_gap = float(prev_gap)
    score_delta = blip_gap - prev_gap
    sim_anchor = _text_jaccard(descriptions.get("blip", ""), descriptions.get("anchor", ""))
    sim_prev = _text_jaccard(descriptions.get("blip", ""), descriptions.get("prev", ""))
    if args.quality_gate_semantic_ref == "anchor":
        semantic = sim_anchor
    elif args.quality_gate_semantic_ref == "prev":
        semantic = sim_prev
    elif args.quality_gate_semantic_ref == "max":
        semantic = max(sim_anchor, sim_prev)
    else:
        raise ValueError("Unsupported quality_gate_semantic_ref: {}".format(args.quality_gate_semantic_ref))

    useful = score_delta > float(args.quality_gate_gap_eps)
    harmful = score_delta < -float(args.quality_gate_gap_eps)
    oracle_accept = useful and semantic >= float(args.quality_gate_semantic_thr)
    deploy_delta = float(row.get("quality_gate_deploy_score_delta", float("nan")))
    confidence_ok = bool(row.get("quality_gate_confidence_ok", 1.0))
    deploy_accept = (
        deploy_gate_available
        and math.isfinite(deploy_delta)
        and deploy_delta > float(args.quality_gate_gap_eps)
        and semantic >= float(args.quality_gate_semantic_thr)
        and confidence_ok
    )
    accept = oracle_accept if args.quality_gate_mode == "oracle" else deploy_accept
    gate_gap = blip_gap if accept else prev_gap
    return {
        "quality_gate_observable": 1.0,
        "quality_gate_accept": 1.0 if accept else 0.0,
        "quality_gate_oracle_accept": 1.0 if oracle_accept else 0.0,
        "quality_gate_deploy_accept": 1.0 if deploy_accept else 0.0,
        "quality_gate_gain_over_prev": gate_gap - prev_gap,
        "quality_gate_score_delta": score_delta,
        "quality_gate_semantic": semantic,
        "quality_gate_semantic_anchor": sim_anchor,
        "quality_gate_semantic_prev": sim_prev,
        "quality_gate_source": "blip" if accept else "prev",
        "quality_gate_true_accept": 1.0 if useful and accept else 0.0,
        "quality_gate_false_reject": 1.0 if useful and not accept else 0.0,
        "quality_gate_true_reject": 1.0 if harmful and not accept else 0.0,
        "quality_gate_false_accept": 1.0 if harmful and accept else 0.0,
    }


def _gain_for_partial_source(row, source):
    if source == "anchor_delta":
        return float(row.get("anchor_delta_gain_over_prev", float("nan")))
    if source == "prev_delta":
        return float(row.get("prev_delta_gain_over_prev", float("nan")))
    return float("nan")


def _partial_absorption_gate(row, args):
    best_gain = float(row.get("best_partial_gain_over_prev", float("nan")))
    if not math.isfinite(best_gain):
        return {
            "partial_gate_observable": 0.0,
            "partial_label_useful": float("nan"),
            "partial_label_harmful": float("nan"),
            "partial_gate_accept": float("nan"),
            "partial_gate_source": "unavailable",
        }

    eps = float(args.quality_gate_gap_eps)
    useful = best_gain > eps
    harmful = best_gain < -eps
    semantic = float(row.get("quality_gate_semantic", float("nan")))
    confidence_ok = bool(row.get("quality_gate_confidence_ok", 1.0))
    semantic_ok = math.isfinite(semantic) and semantic >= float(args.quality_gate_semantic_thr)

    current_accept = bool(row.get("quality_gate_accept", 0.0))
    current_gain = best_gain if current_accept else 0.0

    deploy_gain = float(row.get("deploy_best_partial_gain_over_prev", float("nan")))
    deploy_accept = (
        math.isfinite(deploy_gain)
        and deploy_gain > eps
        and semantic_ok
        and confidence_ok
    )
    oracle_accept = useful and semantic_ok
    accept = oracle_accept if args.quality_gate_mode == "oracle" else deploy_accept
    selected_source = (
        row.get("best_partial_source", "") if args.quality_gate_mode == "oracle"
        else row.get("deploy_best_partial_source", "")
    )
    selected_gain = _gain_for_partial_source(row, selected_source) if accept else 0.0
    if accept and not math.isfinite(selected_gain):
        selected_gain = best_gain

    return {
        "partial_gate_observable": 1.0,
        "partial_label_useful": 1.0 if useful else 0.0,
        "partial_label_harmful": 1.0 if harmful else 0.0,
        "partial_current_gate_true_accept": 1.0 if useful and current_accept else 0.0,
        "partial_current_gate_false_reject": 1.0 if useful and not current_accept else 0.0,
        "partial_current_gate_true_reject": 1.0 if harmful and not current_accept else 0.0,
        "partial_current_gate_false_accept": 1.0 if harmful and current_accept else 0.0,
        "partial_current_gate_gain_over_prev": current_gain,
        "partial_gate_accept": 1.0 if accept else 0.0,
        "partial_gate_oracle_accept": 1.0 if oracle_accept else 0.0,
        "partial_gate_deploy_accept": 1.0 if deploy_accept else 0.0,
        "partial_gate_source": selected_source if accept else "prev",
        "partial_gate_gain_over_prev": selected_gain,
        "partial_gate_true_accept": 1.0 if useful and accept else 0.0,
        "partial_gate_false_reject": 1.0 if useful and not accept else 0.0,
        "partial_gate_true_reject": 1.0 if harmful and not accept else 0.0,
        "partial_gate_false_accept": 1.0 if harmful and accept else 0.0,
    }


def _candidate_description(tracker, image, cls, mode, deploy_trigger, anchor_description):
    mode = str(mode).lower()
    if mode == "deploy_like":
        if not deploy_trigger:
            return "", False
        candidate = tracker._generate_blip_description(image, cls=cls)
        return candidate or anchor_description, True
    if mode == "oracle_blip":
        candidate = tracker._generate_blip_description(image, cls=cls)
        return candidate or anchor_description, True
    if mode == "off":
        return "", False
    raise ValueError("Unsupported candidate_mode: {}".format(mode))


def _write_summary(save_dir, args, rows):
    frames = len(rows)
    lines = [
        "# Stage 3-S0 Language State Probe",
        "",
        "Config: `{}`".format(args.config),
        "Dataset/sequence: `{}:{}`".format(args.dataset_name, args.sequence),
        "Evidence source: `{}`".format(args.evidence_source),
        "Frames: `{}`".format(frames),
        "",
        "## Core Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        "| anchor gap | {:.6g} |".format(_mean(rows, "anchor_pos_hardneg_gap")),
        "| blip gap | {:.6g} |".format(_mean(rows, "blip_pos_hardneg_gap")),
        "| prev-state gap | {:.6g} |".format(_mean(rows, "prev_pos_hardneg_gap")),
        "| anchor-delta gap | {:.6g} |".format(_mean(rows, "anchor_delta_pos_hardneg_gap")),
        "| prev-delta gap | {:.6g} |".format(_mean(rows, "prev_delta_pos_hardneg_gap")),
        "| oracle gap | {:.6g} |".format(_mean(rows, "oracle_gap")),
        "| hard-replace gain over prev | {:.6g} |".format(_mean(rows, "hard_replace_gain_over_prev")),
        "| anchor-delta gain over prev | {:.6g} |".format(_mean(rows, "anchor_delta_gain_over_prev")),
        "| prev-delta gain over prev | {:.6g} |".format(_mean(rows, "prev_delta_gain_over_prev")),
        "| best partial gain over prev | {:.6g} |".format(_mean(rows, "best_partial_gain_over_prev")),
        "| deploy best partial gain over prev | {:.6g} |".format(_mean(rows, "deploy_best_partial_gain_over_prev")),
        "| partial beats hard-replace ratio | {:.6g} |".format(_mean(rows, "partial_beats_hard_replace")),
        "| partial useful when BLIP hurts ratio | {:.6g} |".format(_mean(rows, "partial_useful_when_blip_hurts")),
        "| partial label useful rate | {:.6g} |".format(_mean(rows, "partial_label_useful")),
        "| current gate gain on partial label | {:.6g} |".format(_mean(rows, "partial_current_gate_gain_over_prev")),
        "| current gate partial false accept rate | {:.6g} |".format(_mean(rows, "partial_current_gate_false_accept")),
        "| partial gate accept rate | {:.6g} |".format(_mean(rows, "partial_gate_accept")),
        "| partial gate gain | {:.6g} |".format(_mean(rows, "partial_gate_gain_over_prev")),
        "| partial gate false accept rate | {:.6g} |".format(_mean(rows, "partial_gate_false_accept")),
        "| word-gate selected words | {:.6g} |".format(_mean(rows, "word_gate_selected_count")),
        "| word-gate best gain over prev | {:.6g} |".format(_mean(rows, "word_gate_best_gain_over_prev")),
        "| deploy word-gate best gain over prev | {:.6g} |".format(_mean(rows, "deploy_word_gate_best_gain_over_prev")),
        "| anchor-word-gate gain over prev | {:.6g} |".format(_mean(rows, "anchor_word_gate_gain_over_prev")),
        "| prev-word-gate gain over prev | {:.6g} |".format(_mean(rows, "prev_word_gate_gain_over_prev")),
        "| token-state raw best gain over prev | {:.6g} |".format(_mean(rows, "token_state_raw_best_gain_over_prev")),
        "| token-state best gain over prev | {:.6g} |".format(_mean(rows, "token_state_best_gain_over_prev")),
        "| learned token-state available rate | {:.6g} |".format(_mean(rows, "token_learned_state_available")),
        "| learned token-state frame gate | {:.6g} |".format(_mean(rows, "token_learned_frame_gate_mean")),
        "| learned token-state token gate | {:.6g} |".format(_mean(rows, "token_learned_token_gate_mean")),
        "| learned token-state state delta abs | {:.6g} |".format(_mean(rows, "token_learned_state_delta_abs_mean")),
        "| learned token-state relation attn mean | {:.6g} |".format(_mean(rows, "token_learned_relation_attn_mean")),
        "| learned token-state visual evidence abs | {:.6g} |".format(_mean(rows, "token_learned_visual_evidence_abs_mean")),
        "| learned state center motion | {:.6g} |".format(_mean(rows, "token_learned_state_center_motion_norm")),
        "| learned state scale change | {:.6g} |".format(_mean(rows, "token_learned_state_scale_change_ratio")),
        "| learned confidence peak-gap | {:.6g} |".format(_mean(rows, "token_learned_conf_peak_gap")),
        "| learned confidence entropy | {:.6g} |".format(_mean(rows, "token_learned_conf_score_entropy")),
        "| learned candidate deploy delta | {:.6g} |".format(_mean(rows, "token_learned_candidate_deploy_score_delta")),
        "| learned candidate partial delta | {:.6g} |".format(_mean(rows, "token_learned_candidate_partial_deploy_delta")),
        "| BLIP better than anchor ratio | {:.6g} |".format(_mean(rows, "blip_better_anchor")),
        "| BLIP better than prev ratio | {:.6g} |".format(_mean(rows, "blip_better_prev")),
        "| BLIP hurts ratio | {:.6g} |".format(_mean(rows, "blip_hurts")),
        "| deploy trigger rate | {:.6g} |".format(_mean(rows, "deploy_trigger")),
        "| candidate available rate | {:.6g} |".format(_mean(rows, "candidate_available")),
        "| oracle update rate | {:.6g} |".format(_mean(rows, "oracle_update")),
        "| deploy/oracle agree ratio | {:.6g} |".format(_mean(rows, "deploy_oracle_agree")),
        "| deploy false-positive ratio | {:.6g} |".format(_mean(rows, "deploy_false_positive")),
        "| deploy missed-oracle ratio | {:.6g} |".format(_mean(rows, "deploy_missed_oracle")),
        "| quality gate accept rate | {:.6g} |".format(_mean(rows, "quality_gate_accept")),
        "| quality gate gain | {:.6g} |".format(_mean(rows, "quality_gate_gain_over_prev")),
        "| quality gate false accept rate | {:.6g} |".format(_mean(rows, "quality_gate_false_accept")),
        "| anchor IoU | {:.6g} |".format(_mean(rows, "anchor_iou")),
        "| blip IoU | {:.6g} |".format(_mean(rows, "blip_iou")),
        "| prev-state IoU | {:.6g} |".format(_mean(rows, "prev_iou")),
        "",
        "## Interpretation",
        "",
        "- `BLIP better than anchor ratio` answers whether current caption has usable incremental evidence.",
        "- `anchor_delta` is `anchor + new BLIP content words`; `prev_delta` is `prev + new BLIP content words`.",
        "- `best partial gain over prev` measures whether conservative text absorption is better than keeping the previous state.",
        "- `partial useful when BLIP hurts ratio` is the key counterfactual: hard replacement hurts, but a partial update still helps.",
        "- `partial_label_useful` redefines the oracle label as `best_partial_gain_over_prev > gap_eps`.",
        "- `partial_current_gate_*` evaluates the old BLIP gate against the partial-update label.",
        "- `partial_gate_*` evaluates a partial-update deploy gate using `deploy_best_partial_gain_over_prev`.",
        "- `word_gate` selects individual BLIP new words by deploy score-gap before composing `anchor_word_gate` / `prev_word_gate`.",
        "- `token_state` injects residual language tokens after BERT embedding: `H_base + alpha * (H_blip - H_base)`.",
        "- `oracle gap - prev gap` is the upper bound for a state updater on this probe.",
        "- `BLIP hurts ratio` measures how often current caption prior is worse than the previous state prior.",
        "- `deploy trigger` is the original DUTrack update trigger (`updata_key`).",
        "- In `deploy_like` mode BLIP is generated only when `deploy trigger` is true.",
        "- In `oracle_blip` mode BLIP is generated every frame to measure missed/false update triggers.",
        "- `evidence_source=score` uses the center score map with score-prior injection disabled by default.",
        "- `evidence_source=lmq_prior` keeps the previous LMQ-prior diagnostic and should be treated as prior-module evidence, not language-quality evidence.",
        "- This is a source/state diagnostic only. It does not prove CandidateAdapter is solved.",
    ]
    with open(os.path.join(save_dir, "stage3_s0_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run(args):
    dataset = get_dataset(args.dataset_name)
    seq = dataset[int(args.sequence)] if str(args.sequence).isdigit() else dataset[args.sequence]
    tracker_info = Tracker("dutrack", args.config, args.dataset_name, args.runid)
    params = tracker_info.get_parameters(run_id=args.runid)
    checkpoint_config = str(getattr(args, "checkpoint_config", "") or "").strip()
    if checkpoint_config:
        checkpoint_info = Tracker("dutrack", checkpoint_config, args.dataset_name, args.runid)
        checkpoint_params = checkpoint_info.get_parameters(run_id=args.runid)
        params.checkpoint = checkpoint_params.checkpoint
    params.debug = 0
    tracker = tracker_info.create_tracker(params)
    tracker.cfg.TEST.LANGUAGE_UPDATE_MODE = "anchor"

    saved_score_prior_enabled = getattr(tracker.network, "score_prior_enabled", None)
    if saved_score_prior_enabled is not None and not args.use_score_prior_effect:
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
    if args.language_description:
        init_info["init_text_description"] = str(args.language_description)
        init_info["text_description"] = str(args.language_description)
    tracker.initialize(image0, init_info)

    search_len = int(tracker.network.backbone.pos_embed_x.shape[1])
    search_feat_sz = _feat_size(search_len)
    rows = []
    prev_description = str(getattr(tracker, "language_anchor", ""))
    requested_frames = int(args.max_frames)
    max_frame = len(seq.frames) - 1 if requested_frames <= 0 else min(requested_frames, len(seq.frames) - 1)

    for frame_num in range(1, max_frame + 1):
        image = _read_rgb(seq.frames[frame_num])
        info = {"class": seq.object_class, "path": seq.name, "num": frame_num}

        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, tracker.state, params.search_factor, output_sz=params.search_size)
        search = tracker.preprocessor.process(x_patch_arr, x_amask_arr)
        if tracker.frame_id <= tracker.cfg.TEST.TEMPLATE_NUMBER:
            template_list = tracker.memory_frames.copy()
        else:
            template_list, _ = tracker.select_memory_frames()
        template_target_mask = _selected_template_mask(tracker, template_list) if args.word_evidence else None

        anchor_description = str(getattr(tracker, "language_anchor", ""))
        deploy_trigger = bool(getattr(tracker, "updata_key", False))
        blip_description, candidate_available = _candidate_description(
            tracker, image, seq.object_class, args.candidate_mode,
            deploy_trigger, anchor_description)

        descriptions = {
            "anchor": anchor_description,
            "prev": prev_description or anchor_description,
        }
        if candidate_available and blip_description:
            descriptions["blip"] = blip_description
            descriptions["anchor_delta"] = _compose_anchor_state_description(
                anchor_description, blip_description)
            descriptions["prev_delta"] = _compose_anchor_state_description(
                descriptions["prev"], blip_description)
        outputs = {}
        for name, desc in descriptions.items():
            outputs[name] = _forward_description(tracker, template_list, search.tensors, desc)
        base_priors = {name: _prior_from_output(out) for name, out in outputs.items()}
        base_scores = {name: _score_from_output(out) for name, out in outputs.items()}
        base_score_ref = base_scores.get("anchor")
        crop_box = _search_crop_box(tracker.state, resize_factor, params.search_size)
        prev_pred_box_for_evidence = _predict_box(
            tracker, outputs.get("prev", outputs.get("anchor")), resize_factor)
        deploy_gaps_for_evidence = {}
        if prev_pred_box_for_evidence is not None:
            pred_mask_for_evidence = _token_box_mask(prev_pred_box_for_evidence, crop_box, search_feat_sz)
            for source_name in ("prev", "blip", "anchor_delta", "prev_delta"):
                deploy_gaps_for_evidence[source_name] = _gap_with_mask(
                    _select_evidence_signal(source_name, base_scores, base_priors, args.evidence_source),
                    pred_mask_for_evidence, base_score_ref, args.hardneg_topk)
            partial_gap_values = {
                name: deploy_gaps_for_evidence.get(name, float("nan"))
                for name in ("anchor_delta", "prev_delta")
            }
            _, best_partial_deploy_gap = _select_best_source(partial_gap_values)
            deploy_gaps_for_evidence["best_partial"] = best_partial_deploy_gap
        visual_evidence, visual_evidence_diag = _learned_state_visual_evidence(
            tracker, outputs["prev"], resize_factor, deploy_gaps_for_evidence)
        token_state_sources, token_state_diag = _add_token_state_candidates(
            outputs, descriptions, tracker, template_list, search.tensors,
            anchor_description, descriptions["prev"], blip_description,
            visual_evidence, visual_evidence_diag, args)
        priors = {name: _prior_from_output(out) for name, out in outputs.items()}
        scores = {name: _score_from_output(out) for name, out in outputs.items()}
        score_ref = scores.get("anchor")

        gt_box = seq.ground_truth_rect[frame_num].tolist() if seq.ground_truth_rect is not None else None
        gt_mask = _token_box_mask(gt_box, crop_box, search_feat_sz) if gt_box is not None else torch.zeros(search_len, dtype=torch.bool)

        row = {
            "frame": frame_num,
            "sequence": seq.name,
            "anchor_description": anchor_description,
            "blip_description": blip_description,
            "prev_description": descriptions["prev"],
            "anchor_blip_content_overlap_count": _content_overlap_count(anchor_description, blip_description),
            "prev_blip_content_overlap_count": _content_overlap_count(descriptions["prev"], blip_description),
            "anchor_state_candidate_description": _compose_anchor_state_description(
                anchor_description, blip_description),
            "anchor_delta_candidate_description": descriptions.get("anchor_delta", ""),
            "prev_delta_candidate_description": descriptions.get("prev_delta", ""),
            "candidate_mode": args.candidate_mode,
            "deploy_trigger": 1.0 if deploy_trigger else 0.0,
            "candidate_available": 1.0 if candidate_available else 0.0,
            "trigger_by_position": 1.0 if bool(getattr(tracker, "language_trigger_by_position", False)) else 0.0,
            "trigger_by_scale": 1.0 if bool(getattr(tracker, "language_trigger_by_scale", False)) else 0.0,
            "trigger_by_color": 1.0 if bool(getattr(tracker, "language_trigger_by_color", False)) else 0.0,
            "trigger_area_ratio": float(getattr(tracker, "language_trigger_area_ratio", float("nan"))),
            "trigger_center_distance": float(getattr(tracker, "language_trigger_center_distance", float("nan"))),
            "trigger_color_delta": float(getattr(tracker, "language_trigger_color_delta", float("nan"))),
        }
        row.update(token_state_diag)
        gaps = {}
        pred_boxes = {}
        source_names = ("anchor", "blip", "prev", "anchor_delta", "prev_delta") + tuple(token_state_sources)
        for name in source_names:
            row.update(_source_stats(
                "{}_score".format(name), scores.get(name), gt_mask, score_ref,
                args.top_ratio, args.hardneg_topk))
            row.update(_source_stats(
                "{}_lmq".format(name), priors.get(name), gt_mask, score_ref,
                args.top_ratio, args.hardneg_topk))
            evidence = _select_evidence_signal(name, scores, priors, args.evidence_source)
            row.update(_source_stats(
                name, evidence, gt_mask, score_ref,
                args.top_ratio, args.hardneg_topk))
            gaps[name] = row.get("{}_pos_hardneg_gap".format(name), float("nan"))
            if name in outputs:
                pred_box = _predict_box(tracker, outputs[name], resize_factor)
                pred_boxes[name] = pred_box
                row["{}_iou".format(name)] = _bbox_iou(pred_box, gt_box, search.tensors.device)
            else:
                row["{}_iou".format(name)] = _nan()

        best_name, best_gap = _select_best_source(gaps)
        prev_gap = float(gaps.get("prev", float("nan")))
        anchor_gap = float(gaps.get("anchor", float("nan")))
        blip_gap = float(gaps.get("blip", float("nan")))
        anchor_delta_gap = float(gaps.get("anchor_delta", float("nan")))
        prev_delta_gap = float(gaps.get("prev_delta", float("nan")))
        partial_gaps = {
            "anchor_delta": anchor_delta_gap,
            "prev_delta": prev_delta_gap,
        }
        best_partial_source, best_partial_gap = _select_best_source(partial_gaps)
        hard_replace_gain = _finite_gain(blip_gap, prev_gap)
        anchor_delta_gain = _finite_gain(anchor_delta_gap, prev_gap)
        prev_delta_gain = _finite_gain(prev_delta_gap, prev_gap)
        best_partial_gain = _finite_gain(best_partial_gap, prev_gap)
        row["oracle_source"] = best_name
        row["oracle_gap"] = best_gap
        row["oracle_gain_over_prev"] = best_gap - prev_gap if math.isfinite(prev_gap) and math.isfinite(best_gap) else float("nan")
        row["hard_replace_gain_over_prev"] = hard_replace_gain
        row["anchor_delta_gain_over_prev"] = anchor_delta_gain
        row["prev_delta_gain_over_prev"] = prev_delta_gain
        row["best_partial_source"] = best_partial_source
        row["best_partial_gap"] = best_partial_gap
        row["best_partial_gain_over_prev"] = best_partial_gain
        row["partial_beats_hard_replace"] = (
            1.0 if math.isfinite(best_partial_gain) and math.isfinite(hard_replace_gain)
            and best_partial_gain > hard_replace_gain
            else (0.0 if math.isfinite(best_partial_gain) and math.isfinite(hard_replace_gain) else _nan())
        )
        row["partial_useful_when_blip_hurts"] = (
            1.0 if math.isfinite(best_partial_gain) and math.isfinite(hard_replace_gain)
            and best_partial_gain > 0.0 and hard_replace_gain < 0.0
            else (0.0 if math.isfinite(best_partial_gain) and math.isfinite(hard_replace_gain) else _nan())
        )
        if token_state_sources:
            token_gaps = {
                name: float(gaps.get(name, float("nan")))
                for name in token_state_sources
            }
            raw_token_best_source, raw_token_best_gap = _select_best_source(token_gaps)
            row["token_state_raw_best_source"] = raw_token_best_source
            row["token_state_raw_best_gap"] = raw_token_best_gap
            row["token_state_raw_best_gain_over_prev"] = _finite_gain(raw_token_best_gap, prev_gap)
            if math.isfinite(prev_gap):
                token_gaps["no_update"] = prev_gap
            token_best_source, token_best_gap = _select_best_source(token_gaps)
            row["token_state_best_source"] = token_best_source
            row["token_state_best_gap"] = token_best_gap
            row["token_state_best_gain_over_prev"] = _finite_gain(token_best_gap, prev_gap)
            for name in token_state_sources:
                row["{}_gain_over_prev".format(name)] = _finite_gain(
                    gaps.get(name, float("nan")), prev_gap)
        else:
            row["token_state_best_source"] = "unavailable"
            row["token_state_best_gap"] = float("nan")
            row["token_state_best_gain_over_prev"] = float("nan")
        row["blip_better_anchor"] = (
            1.0 if math.isfinite(blip_gap) and math.isfinite(anchor_gap) and blip_gap > anchor_gap
            else (0.0 if math.isfinite(blip_gap) and math.isfinite(anchor_gap) else _nan())
        )
        row["blip_better_prev"] = (
            1.0 if math.isfinite(blip_gap) and math.isfinite(prev_gap) and blip_gap > prev_gap
            else (0.0 if math.isfinite(blip_gap) and math.isfinite(prev_gap) else _nan())
        )
        row["blip_hurts"] = (
            1.0 if math.isfinite(blip_gap) and math.isfinite(prev_gap) and blip_gap < prev_gap
            else (0.0 if math.isfinite(blip_gap) and math.isfinite(prev_gap) else _nan())
        )
        row["oracle_update"] = 1.0 if best_name != "prev" and math.isfinite(best_gap) else 0.0
        oracle_trigger_observable = candidate_available and math.isfinite(blip_gap)
        oracle_trigger = oracle_trigger_observable and best_name == "blip"
        row["oracle_trigger_observable"] = 1.0 if oracle_trigger_observable else 0.0
        row["oracle_trigger"] = 1.0 if oracle_trigger else (0.0 if oracle_trigger_observable else _nan())
        row["deploy_oracle_agree"] = (
            1.0 if oracle_trigger_observable and bool(deploy_trigger) == bool(oracle_trigger)
            else (0.0 if oracle_trigger_observable else _nan())
        )
        row["deploy_false_positive"] = (
            1.0 if oracle_trigger_observable and deploy_trigger and not oracle_trigger
            else (0.0 if oracle_trigger_observable else _nan())
        )
        row["deploy_missed_oracle"] = (
            1.0 if oracle_trigger_observable and (not deploy_trigger) and oracle_trigger
            else (0.0 if oracle_trigger_observable else _nan())
        )
        prev_pred_box = pred_boxes.get("prev", pred_boxes.get("anchor"))
        row.update(_score_confidence(scores.get("prev")))
        row["pred_box_jump_ratio"] = _box_jump_ratio(prev_pred_box, tracker.state)
        confidence_ok = True
        if math.isfinite(row["score_peak"]) and row["score_peak"] < float(args.quality_gate_score_peak_thr):
            confidence_ok = False
        if math.isfinite(row["score_peak_second_gap"]) and row["score_peak_second_gap"] < float(args.quality_gate_peak_gap_thr):
            confidence_ok = False
        if math.isfinite(row["pred_box_jump_ratio"]) and row["pred_box_jump_ratio"] > float(args.quality_gate_box_jump_thr):
            confidence_ok = False
        row["quality_gate_confidence_ok"] = 1.0 if confidence_ok else 0.0
        deploy_gate_available = False
        if prev_pred_box is not None:
            pred_mask = _token_box_mask(prev_pred_box, crop_box, search_feat_sz)
            prev_deploy_gap = _gap_with_mask(
                _select_evidence_signal("prev", scores, priors, args.evidence_source),
                pred_mask, score_ref, args.hardneg_topk)
            blip_deploy_gap = _gap_with_mask(
                _select_evidence_signal("blip", scores, priors, args.evidence_source),
                pred_mask, score_ref, args.hardneg_topk)
            anchor_delta_deploy_gap = _gap_with_mask(
                _select_evidence_signal("anchor_delta", scores, priors, args.evidence_source),
                pred_mask, score_ref, args.hardneg_topk)
            prev_delta_deploy_gap = _gap_with_mask(
                _select_evidence_signal("prev_delta", scores, priors, args.evidence_source),
                pred_mask, score_ref, args.hardneg_topk)
            row["prev_deploy_pos_hardneg_gap"] = prev_deploy_gap
            row["blip_deploy_pos_hardneg_gap"] = blip_deploy_gap
            row["anchor_delta_deploy_pos_hardneg_gap"] = anchor_delta_deploy_gap
            row["prev_delta_deploy_pos_hardneg_gap"] = prev_delta_deploy_gap
            deploy_partial_gaps = {
                "anchor_delta": anchor_delta_deploy_gap,
                "prev_delta": prev_delta_deploy_gap,
            }
            deploy_best_partial_source, deploy_best_partial_gap = _select_best_source(deploy_partial_gaps)
            row["deploy_best_partial_source"] = deploy_best_partial_source
            row["deploy_best_partial_gap"] = deploy_best_partial_gap
            row["deploy_best_partial_gain_over_prev"] = _finite_gain(deploy_best_partial_gap, prev_deploy_gap)
            row["quality_gate_deploy_score_delta"] = (
                blip_deploy_gap - prev_deploy_gap
                if math.isfinite(blip_deploy_gap) and math.isfinite(prev_deploy_gap)
                else float("nan")
            )
            deploy_gate_available = math.isfinite(row["quality_gate_deploy_score_delta"])
            if args.word_absorption and candidate_available and blip_description:
                selected_words, word_gate_meta = _word_gate_select_words(
                    tracker,
                    template_list,
                    search.tensors,
                    descriptions["prev"],
                    blip_description,
                    pred_mask,
                    score_ref,
                    prev_deploy_gap,
                    args,
                )
                row.update(word_gate_meta)
                descriptions["anchor_word_gate"] = _compose_with_words(anchor_description, selected_words)
                descriptions["prev_word_gate"] = _compose_with_words(descriptions["prev"], selected_words)
                for name in ("anchor_word_gate", "prev_word_gate"):
                    outputs[name] = _forward_description(
                        tracker, template_list, search.tensors, descriptions[name])
                    priors[name] = _prior_from_output(outputs[name])
                    scores[name] = _score_from_output(outputs[name])
                    evidence = _select_evidence_signal(name, scores, priors, args.evidence_source)
                    row.update(_source_stats(
                        "{}_score".format(name), scores.get(name), gt_mask, score_ref,
                        args.top_ratio, args.hardneg_topk))
                    row.update(_source_stats(
                        "{}_lmq".format(name), priors.get(name), gt_mask, score_ref,
                        args.top_ratio, args.hardneg_topk))
                    row.update(_source_stats(
                        name, evidence, gt_mask, score_ref,
                        args.top_ratio, args.hardneg_topk))
                    pred_box = _predict_box(tracker, outputs[name], resize_factor)
                    row["{}_iou".format(name)] = _bbox_iou(pred_box, gt_box, search.tensors.device)
                    deploy_gap = _gap_with_mask(evidence, pred_mask, score_ref, args.hardneg_topk)
                    row["{}_deploy_pos_hardneg_gap".format(name)] = deploy_gap
                    row["{}_gain_over_prev".format(name)] = _finite_gain(
                        row.get("{}_pos_hardneg_gap".format(name), float("nan")), prev_gap)
                    row["{}_deploy_gain_over_prev".format(name)] = _finite_gain(
                        deploy_gap, prev_deploy_gap)
                word_gate_partial = {
                    "anchor_word_gate": float(row.get("anchor_word_gate_pos_hardneg_gap", float("nan"))),
                    "prev_word_gate": float(row.get("prev_word_gate_pos_hardneg_gap", float("nan"))),
                }
                word_gate_source, word_gate_gap = _select_best_source(word_gate_partial)
                row["word_gate_best_source"] = word_gate_source
                row["word_gate_best_gap"] = word_gate_gap
                row["word_gate_best_gain_over_prev"] = _finite_gain(word_gate_gap, prev_gap)
                deploy_word_gate_partial = {
                    "anchor_word_gate": float(row.get("anchor_word_gate_deploy_pos_hardneg_gap", float("nan"))),
                    "prev_word_gate": float(row.get("prev_word_gate_deploy_pos_hardneg_gap", float("nan"))),
                }
                deploy_word_gate_source, deploy_word_gate_gap = _select_best_source(deploy_word_gate_partial)
                row["deploy_word_gate_best_source"] = deploy_word_gate_source
                row["deploy_word_gate_best_gap"] = deploy_word_gate_gap
                row["deploy_word_gate_best_gain_over_prev"] = _finite_gain(
                    deploy_word_gate_gap, prev_deploy_gap)
            elif args.word_absorption:
                row.update(_word_gate_empty())
                row["anchor_word_gate_gain_over_prev"] = float("nan")
                row["prev_word_gate_gain_over_prev"] = float("nan")
                row["word_gate_best_source"] = "unavailable"
                row["word_gate_best_gap"] = float("nan")
                row["word_gate_best_gain_over_prev"] = float("nan")
                row["deploy_word_gate_best_source"] = "unavailable"
                row["deploy_word_gate_best_gap"] = float("nan")
                row["deploy_word_gate_best_gain_over_prev"] = float("nan")
            if args.word_evidence:
                row.update(_add_word_evidence(
                    row, tracker, outputs, descriptions, template_target_mask, pred_mask))
        row.update(_quality_gate(row, descriptions, args, deploy_gate_available=deploy_gate_available))
        row.update(_partial_absorption_gate(row, args))
        rows.append(row)

        if args.state_update_policy == "gate":
            if row.get("quality_gate_source") == "blip":
                prev_description = descriptions.get("blip", prev_description)
        elif args.state_update_policy == "anchor_state_gate":
            if row.get("quality_gate_source") == "blip":
                prev_description = row.get("anchor_state_candidate_description", prev_description)
        elif args.state_update_policy == "prev_delta_gate":
            if row.get("quality_gate_source") == "blip":
                prev_description = row.get("prev_delta_candidate_description", prev_description)
        elif args.state_update_policy == "best_partial_oracle":
            if math.isfinite(best_partial_gain) and best_partial_gain > 0.0:
                prev_description = descriptions.get(best_partial_source, prev_description)
        elif args.state_update_policy == "word_gate":
            source = row.get("deploy_word_gate_best_source", "")
            gain = float(row.get("deploy_word_gate_best_gain_over_prev", float("nan")))
            if math.isfinite(gain) and gain > float(args.quality_gate_gap_eps):
                prev_description = descriptions.get(source, prev_description)
        elif args.state_update_policy == "token_state":
            # Token-state rollout is not represented as a natural-language string.
            # Keep text state unchanged in this S1 diagnostic; use token_state_best_* for counterfactual quality.
            pass
        elif args.state_update_policy == "oracle" and best_name:
            prev_description = descriptions.get(best_name, prev_description)
        elif args.state_update_policy == "none":
            pass
        else:
            raise ValueError("Unsupported state_update_policy: {}".format(args.state_update_policy))

        tracker.track(image, info)

    csv_path = os.path.join(save_dir, "stage3_s0_probe.csv")
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    _write_summary(save_dir, args, rows)
    print("Saved Stage 3-S0 language state probe to {}".format(save_dir))
    return save_dir, rows


def main():
    parser = argparse.ArgumentParser(description="Stage 3-S0 probe for language state update value.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_config", default="",
                        help="Optional config name used only for checkpoint lookup. "
                             "The model is still built from --config.")
    parser.add_argument("--dataset_name", default="otb_lang")
    parser.add_argument("--sequence", default="Biker")
    parser.add_argument("--runid", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=5,
                        help="Number of frames after initialization to evaluate. Use <=0 for the full sequence.")
    parser.add_argument("--top_ratio", type=float, default=0.1)
    parser.add_argument("--hardneg_topk", type=int, default=6)
    parser.add_argument("--language_description", default="")
    parser.add_argument("--candidate_mode", default="deploy_like",
                        choices=("deploy_like", "oracle_blip", "off"),
                        help="deploy_like calls BLIP only when original updata_key triggers; oracle_blip calls BLIP every frame.")
    parser.add_argument("--evidence_source", default="score",
                        choices=("score", "lmq_prior"),
                        help="Primary source-quality signal. Use score for language quality; lmq_prior only diagnoses the LMQ prior module.")
    parser.add_argument("--oracle_state_update", type=int, default=1)
    parser.add_argument("--state_update_policy", default="oracle",
                        choices=("oracle", "gate", "anchor_state_gate", "prev_delta_gate",
                                 "best_partial_oracle", "word_gate", "token_state", "none"),
                        help="How to update the probe's previous language state.")
    parser.add_argument("--quality_gate_gap_eps", type=float, default=0.0)
    parser.add_argument("--quality_gate_semantic_thr", type=float, default=0.0)
    parser.add_argument("--quality_gate_semantic_ref", default="max",
                        choices=("anchor", "prev", "max"))
    parser.add_argument("--quality_gate_mode", default="deploy",
                        choices=("deploy", "oracle"),
                        help="deploy uses predicted-box score evidence; oracle uses GT score evidence for upper-bound diagnosis.")
    parser.add_argument("--quality_gate_score_peak_thr", type=float, default=-1e9)
    parser.add_argument("--quality_gate_peak_gap_thr", type=float, default=-1e9)
    parser.add_argument("--quality_gate_box_jump_thr", type=float, default=1e9)
    parser.add_argument("--use_score_prior_effect", action="store_true",
                        help="Keep configured score-prior bias during source forwards.")
    parser.add_argument("--word_evidence", action="store_true",
                        help="Record template-target and search-deploy word evidence for BLIP/prev captions.")
    parser.add_argument("--word_absorption", action="store_true",
                        help="Select individual BLIP new words by deploy score-gap and evaluate word-gated language states.")
    parser.add_argument("--word_gate_max_candidate_words", type=int, default=8)
    parser.add_argument("--word_gate_max_selected_words", type=int, default=4)
    parser.add_argument("--word_gate_min_deploy_gain", type=float, default=0.0)
    parser.add_argument("--token_state_probe", action="store_true",
                        help="Evaluate latent language token residual states without changing text state.")
    parser.add_argument("--token_state_alphas", default="0.1,0.3",
                        help="Comma-separated residual strengths for token-state candidates.")
    parser.add_argument("--learned_token_state_probe", action="store_true",
                        help="Evaluate backbone.language_state_updater if the config enables it.")
    parser.add_argument("--tag", default="s0")
    parser.add_argument("--output_tag", default=None)
    parser.add_argument("--out_dir", default="output/test/language_state_s0_probe")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
