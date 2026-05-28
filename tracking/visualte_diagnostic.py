import argparse
import csv
import math
import os
import sys
from collections import OrderedDict

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


def _beta_tag(beta):
    text = "{:.6g}".format(float(beta))
    return "beta{}".format(text.replace("-", "m").replace(".", "p"))


def _safe_tag(text):
    return str(text).lower().replace("-", "m").replace(".", "p").replace("/", "_")


def _run_label(config, args):
    if getattr(args, "output_tag", None):
        return args.output_tag
    parts = []
    source = getattr(args, "score_prior_source", None)
    if source:
        parts.append("source{}".format(_safe_tag(source)))
    beta = getattr(args, "score_prior_beta", None)
    if beta is not None:
        parts.append(_beta_tag(beta))
    layer = getattr(args, "score_prior_layer", None)
    if layer is not None:
        parts.append("layer{}".format(_safe_tag(layer)))
    lang_init = getattr(args, "language_init_source", None)
    if lang_init:
        parts.append("langinit{}".format(_safe_tag(lang_init)))
    lang_mode = getattr(args, "language_update_mode", None)
    if lang_mode:
        parts.append("lang{}".format(_safe_tag(lang_mode)))
    return "{}_{}".format(config, "_".join(parts)) if parts else config


def _read_rgb(path):
    image = cv.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def _as_tensor_1d(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().view(-1)
    return torch.tensor(x, dtype=torch.float32).view(-1)


def _feat_size(num_tokens):
    size = int(round(math.sqrt(num_tokens)))
    if size * size != num_tokens:
        raise ValueError("Token count is not square: {}".format(num_tokens))
    return size


def _search_crop_box(target_box, resize_factor, search_size):
    crop_sz = float(search_size) / float(resize_factor)
    x, y, w, h = [float(v) for v in target_box]
    return [x + 0.5 * w - 0.5 * crop_sz, y + 0.5 * h - 0.5 * crop_sz, crop_sz, crop_sz]


def _token_box_mask(box, crop_box, feat_sz):
    crop_x, crop_y, crop_w, crop_h = [float(v) for v in crop_box]
    x, y, w, h = [float(v) for v in box]
    cols = torch.arange(feat_sz, dtype=torch.float32) + 0.5
    rows = torch.arange(feat_sz, dtype=torch.float32) + 0.5
    yy, xx = torch.meshgrid(rows, cols, indexing='ij')
    px = crop_x + xx.reshape(-1) / feat_sz * crop_w
    py = crop_y + yy.reshape(-1) / feat_sz * crop_h
    return (px >= x) & (px <= x + w) & (py >= y) & (py <= y + h)


def _heat_stats(name, heat, gt_mask, top_ratio=0.1):
    heat = _as_tensor_1d(heat)
    if heat is None:
        return {}
    gt_mask = gt_mask.bool()
    out_mask = ~gt_mask
    top_k = max(1, int(round(heat.numel() * float(top_ratio))))
    top_idx = torch.topk(heat, top_k).indices
    top_mask = torch.zeros_like(heat, dtype=torch.bool)
    top_mask[top_idx] = True
    prob = heat.clamp_min(0)
    prob = prob / prob.sum().clamp_min(1e-12)
    entropy = -(prob * prob.clamp_min(1e-12).log()).sum() / math.log(max(heat.numel(), 2))
    mass_den = heat.sum().clamp_min(1e-12)
    return {
        "{}_mean_in_gt".format(name): heat[gt_mask].mean().item() if gt_mask.any() else float("nan"),
        "{}_mean_out_gt".format(name): heat[out_mask].mean().item() if out_mask.any() else float("nan"),
        "{}_mass_in_gt".format(name): (heat[gt_mask].sum() / mass_den).item() if gt_mask.any() else float("nan"),
        "{}_top{}_precision".format(name, int(top_ratio * 100)): (top_mask & gt_mask).sum().item() / float(top_k),
        "{}_min".format(name): heat.min().item(),
        "{}_max".format(name): heat.max().item(),
        "{}_entropy".format(name): entropy.item(),
    }


def _region_gap_stats(name, heat, gt_mask):
    heat = _as_tensor_1d(heat)
    if heat is None:
        return {}
    gt_mask = gt_mask.bool()
    out_mask = ~gt_mask
    mean_in = heat[gt_mask].mean().item() if gt_mask.any() else float("nan")
    mean_out = heat[out_mask].mean().item() if out_mask.any() else float("nan")
    ratio = mean_in / max(mean_out, 1e-12) if math.isfinite(mean_in) and math.isfinite(mean_out) else float("nan")
    return {
        "{}_gap_in_minus_out".format(name): mean_in - mean_out if math.isfinite(mean_in) and math.isfinite(mean_out) else float("nan"),
        "{}_ratio_in_over_out".format(name): ratio,
    }


def _hard_negative_stats(name, heat, gt_mask, score_map, hard_k=6):
    heat = _as_tensor_1d(heat)
    score_map = _as_tensor_1d(score_map)
    if heat is None or score_map is None or heat.numel() != score_map.numel():
        return {}
    gt_mask = gt_mask.bool()
    hard_mask_source = ~gt_mask
    if not gt_mask.any() or not hard_mask_source.any():
        return {}
    hard_k = max(1, min(int(hard_k), int(hard_mask_source.sum().item())))
    hard_source_indices = hard_mask_source.nonzero(as_tuple=False).view(-1)
    hard_local = torch.topk(score_map[hard_source_indices], hard_k).indices
    hard_indices = hard_source_indices[hard_local]
    pos_mean = heat[gt_mask].mean().item()
    hard_mean = heat[hard_indices].mean().item()
    gap = pos_mean - hard_mean
    return {
        "{}_pos_mean".format(name): pos_mean,
        "{}_hardneg_mean".format(name): hard_mean,
        "{}_pos_hardneg_gap".format(name): gap,
        "{}_hardneg_count".format(name): hard_k,
        "{}_hard_case".format(name): 1.0 if gap < 0 else 0.0,
    }


def _hard_negative_indices(gt_mask, score_map, hard_k=6):
    score_map = _as_tensor_1d(score_map)
    if score_map is None:
        return None
    gt_mask = gt_mask.bool()
    hard_mask_source = ~gt_mask
    if not gt_mask.any() or not hard_mask_source.any():
        return None
    hard_k = max(1, min(int(hard_k), int(hard_mask_source.sum().item())))
    hard_source_indices = hard_mask_source.nonzero(as_tuple=False).view(-1)
    hard_local = torch.topk(score_map[hard_source_indices], hard_k).indices
    return hard_source_indices[hard_local]


def _is_content_token(label):
    token = str(label).strip().lower()
    return token not in ("[cls]", "[sep]", "[pad]", "[unk]", "")


_ATTRIBUTE_WORDS = {
    "red", "blue", "green", "yellow", "black", "white", "gray", "grey", "brown",
    "orange", "purple", "pink", "dark", "bright", "light", "small", "large",
    "big", "tiny", "visible", "occluded", "round", "long", "short", "front",
    "back", "left", "right",
}

_CONTEXT_WORDS = {
    "the", "a", "an", "of", "on", "in", "at", "by", "with", "near", "under",
    "above", "below", "behind", "beside", "between", "and", "or", "to", "from",
    "held", "holding", "road", "tree", "hand", "floor", "ground", "street", "background",
}


def _clean_word_token(label):
    token = str(label).strip().lower()
    if token.startswith("##"):
        token = token[2:]
    return "".join(ch for ch in token if ch.isalnum() or ch == "_")


def _word_role(label):
    token = _clean_word_token(label)
    if not _is_content_token(label):
        return "special"
    if token in _ATTRIBUTE_WORDS:
        return "attribute"
    if token in _CONTEXT_WORDS:
        return "context"
    return "subject_candidate"


def _anchor_subject_terms(token_labels, object_class=None):
    content = []
    for label in token_labels:
        role = _word_role(label)
        token = _clean_word_token(label)
        if token and role == "subject_candidate":
            content.append(token)
    class_terms = set()
    if object_class:
        for part in str(object_class).replace("_", " ").replace("-", " ").split():
            token = _clean_word_token(part)
            if token:
                class_terms.add(token)
    matched = [token for token in content if token in class_terms]
    if matched:
        return set(matched)
    return {content[0]} if content else set()


def _word_response_stage(aux, name, stage_idx):
    items = aux.get(name)
    if not items or stage_idx >= len(items):
        return None
    value = items[stage_idx]
    if not isinstance(value, torch.Tensor):
        return None
    value = value[0].detach().float().cpu()
    if value.dim() != 2:
        return None
    return value


def _rank_desc(values):
    order = sorted(range(len(values)), key=lambda idx: values[idx], reverse=True)
    ranks = [0] * len(values)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def _safe_mean(values):
    values = [value for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _rank_corr(rank_a, rank_b):
    pairs = [(float(a), float(b)) for a, b in zip(rank_a, rank_b) if a > 0 and b > 0]
    n = len(pairs)
    if n < 2:
        return float("nan")
    mean_a = sum(a for a, _ in pairs) / n
    mean_b = sum(b for _, b in pairs) / n
    da = [a - mean_a for a, _ in pairs]
    db = [b - mean_b for _, b in pairs]
    denom_a = math.sqrt(sum(v * v for v in da))
    denom_b = math.sqrt(sum(v * v for v in db))
    if denom_a <= 0 or denom_b <= 0:
        return float("nan")
    return sum(a * b for a, b in zip(da, db)) / (denom_a * denom_b)


def _topk_overlap(rank_a, rank_b, k=3):
    a = {idx for idx, rank in enumerate(rank_a) if rank > 0 and rank <= k}
    b = {idx for idx, rank in enumerate(rank_b) if rank > 0 and rank <= k}
    if not a or not b:
        return float("nan")
    return len(a & b) / float(k)


def _word_evidence_rows(frame_num, sequence_name, loc, evidence_mode, word_scores, word_weights,
                        word_reliability,
                        token_labels, positive_mask, score_map, hard_k=6, tau=0.1,
                        object_class=None):
    if word_scores is None:
        return [], {}
    positive_mask = positive_mask.bool()
    hard_indices = _hard_negative_indices(positive_mask, score_map, hard_k)
    if hard_indices is None or not positive_mask.any():
        return [], {}
    word_weights = _as_tensor_1d(word_weights)
    word_reliability = _as_tensor_1d(word_reliability)
    anchor_terms = _anchor_subject_terms(token_labels, object_class)
    rows = []
    entries = []
    num_words = word_scores.shape[1]
    for word_idx in range(num_words):
        label = token_labels[word_idx] if word_idx < len(token_labels) else "tok{}".format(word_idx)
        is_content = _is_content_token(label)
        if not is_content:
            continue
        heat = word_scores[:, word_idx]
        if heat.numel() != positive_mask.numel():
            continue
        pos_score = heat[positive_mask].mean().item()
        hardneg_score = heat[hard_indices].mean().item()
        out_mask = ~positive_mask
        out_score = heat[out_mask].mean().item() if out_mask.any() else float("nan")
        gap = pos_score - hardneg_score
        visual_evidence = 1.0 / (1.0 + math.exp(-gap / max(float(tau), 1e-8)))
        weight = (float(word_weights[word_idx].item())
                  if word_weights is not None and word_idx < word_weights.numel() else float("nan"))
        reliability = (float(word_reliability[word_idx].item())
                       if word_reliability is not None and word_idx < word_reliability.numel() else float("nan"))
        role = _word_role(label)
        clean_word = _clean_word_token(label)
        entries.append({
            "frame": frame_num,
            "sequence": sequence_name,
            "layer": loc,
            "evidence_mode": evidence_mode,
            "word_index": word_idx,
            "word": str(label),
            "word_type": role,
            "word_is_content": 1.0 if is_content else 0.0,
            "word_is_subject_candidate": 1.0 if role == "subject_candidate" else 0.0,
            "word_is_anchor_subject": 1.0 if clean_word in anchor_terms else 0.0,
            "word_weight": weight,
            "word_reliability": reliability,
            "word_pos_score": pos_score,
            "word_hardneg_score": hardneg_score,
            "word_out_score": out_score,
            "word_gap": gap,
            "word_visual_evidence": visual_evidence,
            "word_hard_case": 1.0 if gap < 0 else 0.0,
            "pos_token_count": int(positive_mask.sum().item()),
            "neg_token_count": int(hard_indices.numel()),
            "hardneg_count": int(hard_indices.numel()),
        })
    if not entries:
        return rows, {}
    gap_ranks = _rank_desc([entry["word_gap"] for entry in entries])
    weight_ranks = _rank_desc([
        entry["word_weight"] if math.isfinite(float(entry["word_weight"])) else float("-inf")
        for entry in entries
    ])
    for entry, gap_rank, weight_rank in zip(entries, gap_ranks, weight_ranks):
        entry["word_rank_by_gap"] = gap_rank
        entry["word_rank_by_weight"] = weight_rank
        rows.append(entry)
    gaps = [entry["word_gap"] for entry in entries]
    evidences = [entry["word_visual_evidence"] for entry in entries]
    hard_cases = sum(1 for entry in entries if entry["word_gap"] < 0)
    content_count = len(entries)
    if not gaps:
        return rows, {}
    gaps_tensor = torch.tensor(gaps, dtype=torch.float32)
    evidence_tensor = torch.tensor(evidences, dtype=torch.float32)
    subject_gaps = [entry["word_gap"] for entry in entries if entry["word_type"] == "subject_candidate"]
    attribute_gaps = [entry["word_gap"] for entry in entries if entry["word_type"] == "attribute"]
    context_gaps = [entry["word_gap"] for entry in entries if entry["word_type"] == "context"]
    anchor_subject_gaps = [entry["word_gap"] for entry in entries if entry["word_is_anchor_subject"] > 0]
    subject_entries = [entry for entry in entries if entry["word_is_anchor_subject"] > 0]
    if not subject_entries:
        subject_entries = [entry for entry in entries if entry["word_is_subject_candidate"] > 0]
    subject_entry = subject_entries[0] if subject_entries else None
    best_entry = max(entries, key=lambda entry: entry["word_gap"])
    positive_ratio = sum(1 for entry in entries if entry["word_gap"] > 0) / float(max(content_count, 1))
    best_gap = max(gaps)
    prefix = "word_evidence_{}_L{}".format(evidence_mode, loc)
    summary = {
        "{}_mean_gap".format(prefix): gaps_tensor.mean().item(),
        "{}_max_gap".format(prefix): gaps_tensor.max().item(),
        "{}_min_gap".format(prefix): gaps_tensor.min().item(),
        "{}_mean_visual_evidence".format(prefix): evidence_tensor.mean().item(),
        "{}_hard_case_ratio".format(prefix): hard_cases / float(max(content_count, 1)),
        "{}_content_count".format(prefix): content_count,
        "{}_subject_gap_mean".format(prefix): _safe_mean(subject_gaps),
        "{}_attribute_gap_mean".format(prefix): _safe_mean(attribute_gaps),
        "{}_context_gap_mean".format(prefix): _safe_mean(context_gaps),
        "{}_best_word_gap_mean".format(prefix): best_gap,
        "{}_anchor_subject_gap_mean".format(prefix): _safe_mean(anchor_subject_gaps),
        "{}_content_word_positive_ratio".format(prefix): positive_ratio,
        "{}_subject_gap".format(prefix): subject_entry["word_gap"] if subject_entry is not None else float("nan"),
        "{}_subject_rank_by_gap".format(prefix): subject_entry["word_rank_by_gap"] if subject_entry is not None else float("nan"),
        "{}_subject_rank_by_weight".format(prefix): subject_entry["word_rank_by_weight"] if subject_entry is not None else float("nan"),
        "{}_weight_gap_rank_corr".format(prefix): _rank_corr(weight_ranks, gap_ranks),
        "{}_top3_weight_gap_overlap".format(prefix): _topk_overlap(weight_ranks, gap_ranks, k=3),
    }
    if subject_entry is not None:
        summary["{}_subject_candidate_word".format(prefix)] = subject_entry["word"]
    summary["{}_best_gap_word".format(prefix)] = best_entry["word"]
    return rows, summary


def _signed_heat_stats(name, heat, gt_mask, top_ratio=0.1):
    heat = _as_tensor_1d(heat)
    if heat is None:
        return {}
    gt_mask = gt_mask.bool()
    out_mask = ~gt_mask
    top_k = max(1, int(round(heat.numel() * float(top_ratio))))
    top_idx = torch.topk(heat, top_k).indices
    top_mask = torch.zeros_like(heat, dtype=torch.bool)
    top_mask[top_idx] = True
    return {
        "{}_mean_in_gt".format(name): heat[gt_mask].mean().item() if gt_mask.any() else float("nan"),
        "{}_mean_out_gt".format(name): heat[out_mask].mean().item() if out_mask.any() else float("nan"),
        "{}_top{}_precision".format(name, int(top_ratio * 100)): (top_mask & gt_mask).sum().item() / float(top_k),
        "{}_min".format(name): heat.min().item(),
        "{}_max".format(name): heat.max().item(),
        "{}_abs_sum".format(name): heat.abs().sum().item(),
    }


def _qkv_from_attention(module, x):
    B, N, _ = x.shape
    if getattr(module, "deepnorm", False) or getattr(module, "subln", False):
        q = F.linear(input=x, weight=module.q_proj.weight, bias=module.q_bias)
        k = F.linear(input=x, weight=module.k_proj.weight, bias=None)
        v = F.linear(input=x, weight=module.v_proj.weight, bias=module.v_bias)
        q = q.reshape(B, N, module.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B, N, module.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, N, module.num_heads, -1).permute(0, 2, 1, 3)
        return q, k, v

    qkv_bias = None
    if module.q_bias is not None:
        qkv_bias = torch.cat((module.q_bias, torch.zeros_like(module.v_bias, requires_grad=False), module.v_bias))
    qkv = F.linear(input=x, weight=module.qkv.weight, bias=qkv_bias)
    qkv = qkv.reshape(B, N, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
    return qkv[0], qkv[1], qkv[2]


def _attention_snapshot(module, inputs, output):
    if not inputs:
        return None
    x = inputs[0]
    if not isinstance(x, torch.Tensor) or x.dim() != 3:
        return None
    with torch.no_grad():
        q, k, v = _qkv_from_attention(module, x)
        q_scaled = q * module.scale
        if getattr(module, "qk_float", False):
            logits = q_scaled.float() @ k.float().transpose(-2, -1)
        else:
            logits = q_scaled @ k.transpose(-2, -1)
        raw_attn = logits.softmax(dim=-1).type_as(x)
        policy_attn = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        return {
            "qk_logits": logits.detach().float().cpu(),
            "raw_attn": raw_attn.detach().float().cpu(),
            "policy_attn": policy_attn.detach().float().cpu() if isinstance(policy_attn, torch.Tensor) else None,
            "v_norm": v.detach().float().norm(dim=-1).mean(dim=1).cpu(),
        }


def _search_attn_row(attn, query_index, lens_x):
    if attn is None:
        return None
    attn = attn.detach().float().cpu()
    if attn.dim() != 4:
        return None
    attn = attn[0].mean(dim=0)
    query_index = min(max(int(query_index), 0), attn.shape[0] - 1)
    return attn[query_index, -lens_x:]


def _attn_row_slice(attn, query_index, start, end):
    if attn is None:
        return None
    attn = attn.detach().float().cpu()
    if attn.dim() != 4:
        return None
    attn = attn[0].mean(dim=0)
    query_index = min(max(int(query_index), 0), attn.shape[0] - 1)
    start = max(0, int(start))
    end = min(int(end), attn.shape[1])
    if end <= start:
        return None
    return attn[query_index, start:end]


def _attn_row(attn, query_index):
    if attn is None:
        return None
    attn = attn.detach().float().cpu()
    if attn.dim() != 4:
        return None
    attn = attn[0].mean(dim=0)
    query_index = min(max(int(query_index), 0), attn.shape[0] - 1)
    return attn[query_index]


def _attn_rows_mean(attn, start, end, row_mask=None):
    if attn is None:
        return None
    attn = attn.detach().float().cpu()
    if attn.dim() != 4:
        return None
    attn = attn[0].mean(dim=0)
    start = max(0, int(start))
    end = min(int(end), attn.shape[0])
    if end <= start:
        return None
    rows = attn[start:end]
    if row_mask is not None:
        row_mask = _as_tensor_1d(row_mask).bool()
        if row_mask.numel() != rows.shape[0]:
            return None
        if not row_mask.any():
            return None
        rows = rows[row_mask]
    return rows.mean(dim=0)


def _infer_token_layout(snapshot, l_len, template_tokens, search_len):
    attn = snapshot.get("policy_attn") if snapshot is not None else None
    if attn is None:
        attn = snapshot.get("raw_attn") if snapshot is not None else None
    if attn is None or attn.dim() != 4:
        return None
    total_len = int(attn.shape[-1])
    l_len = int(l_len)
    template_tokens = int(template_tokens)
    search_len = int(search_len)
    temporal_len = total_len - l_len - template_tokens - search_len
    if temporal_len < 0:
        return None
    l_start = temporal_len
    z_start = l_start + l_len
    x_start = z_start + template_tokens
    return {
        "track": (0, temporal_len),
        "language": (l_start, z_start),
        "template": (z_start, x_start),
        "search": (x_start, x_start + search_len),
    }


def _attention_compare_stats(name, raw_row, policy_row, gt_mask, layout):
    raw_row = _as_tensor_1d(raw_row)
    policy_row = _as_tensor_1d(policy_row)
    if raw_row is None or policy_row is None or raw_row.numel() != policy_row.numel():
        return {}
    eps = 1e-12
    raw_prob = raw_row.clamp_min(eps)
    raw_prob = raw_prob / raw_prob.sum().clamp_min(eps)
    policy_prob = policy_row.clamp_min(eps)
    policy_prob = policy_prob / policy_prob.sum().clamp_min(eps)
    stats = {
        "{}_kl_raw_to_policy".format(name): (raw_prob * (raw_prob.log() - policy_prob.log())).sum().item(),
        "{}_l1_raw_policy".format(name): (policy_prob - raw_prob).abs().sum().item(),
    }
    if layout is not None:
        for group, (start, end) in layout.items():
            if end <= start:
                continue
            raw_mass = raw_row[start:end].sum().item()
            policy_mass = policy_row[start:end].sum().item()
            stats["{}_{}_mass_raw".format(name, group)] = raw_mass
            stats["{}_{}_mass_policy".format(name, group)] = policy_mass
            stats["{}_{}_mass_delta".format(name, group)] = policy_mass - raw_mass
        x_start, x_end = layout.get("search", (None, None))
        if x_start is not None and x_end is not None and x_end > x_start:
            gt_mask = gt_mask.bool()
            raw_search = raw_row[x_start:x_end]
            policy_search = policy_row[x_start:x_end]
            raw_abs_gt = raw_search[gt_mask].sum().item() if gt_mask.any() else float("nan")
            policy_abs_gt = policy_search[gt_mask].sum().item() if gt_mask.any() else float("nan")
            stats["{}_search_gt_abs_mass_raw".format(name)] = raw_abs_gt
            stats["{}_search_gt_abs_mass_policy".format(name)] = policy_abs_gt
            stats["{}_search_gt_abs_mass_delta".format(name)] = (
                policy_abs_gt - raw_abs_gt if math.isfinite(raw_abs_gt) and math.isfinite(policy_abs_gt) else float("nan")
            )
    return stats


def _non_track_delta_absmax(raw_attn, policy_attn):
    if raw_attn is None or policy_attn is None:
        return float("nan")
    raw = raw_attn.detach().float().cpu()
    policy = policy_attn.detach().float().cpu()
    if raw.dim() != 4 or policy.shape != raw.shape or raw.shape[2] <= 1:
        return float("nan")
    return (policy[:, :, 1:, :] - raw[:, :, 1:, :]).abs().max().item()


def _query_group_compare_stats(name, snapshot, layout, query_group, gt_mask, row_mask=None):
    if snapshot is None or layout is None or query_group not in layout:
        return {}
    q_start, q_end = layout[query_group]
    raw_row = _attn_rows_mean(snapshot.get("raw_attn"), q_start, q_end, row_mask=row_mask)
    policy_row = _attn_rows_mean(snapshot.get("policy_attn"), q_start, q_end, row_mask=row_mask)
    return _attention_compare_stats(name, raw_row, policy_row, gt_mask, layout)


def _query_group_search_row(snapshot, layout, query_group, attn_name, search_len, row_mask=None):
    if snapshot is None or layout is None or query_group not in layout:
        return None
    q_start, q_end = layout[query_group]
    row = _attn_rows_mean(snapshot.get(attn_name), q_start, q_end, row_mask=row_mask)
    row = _as_tensor_1d(row)
    if row is None or row.numel() < search_len:
        return None
    return row[-search_len:]


def _v_search_norm(snapshot, search_len):
    if snapshot is None:
        return None
    v_norm = snapshot.get("v_norm")
    if not isinstance(v_norm, torch.Tensor) or v_norm.dim() != 2 or v_norm.shape[-1] < search_len:
        return None
    return v_norm[0, -search_len:].detach().float().cpu().view(-1)


def _head_input_maps(out_dict, feat_len_s):
    if not isinstance(out_dict, dict):
        return None, None, None
    feat = out_dict.get("backbone_feat")
    if isinstance(feat, list):
        feat = feat[-1]
    if not isinstance(feat, torch.Tensor) or feat.dim() != 3 or feat.shape[1] <= feat_len_s:
        return None, None, None
    feat = feat[0].detach().float().cpu()
    enc_opt = feat[-feat_len_s:]
    q0 = feat[0]
    gate = enc_opt @ q0
    search_norm = enc_opt.norm(dim=-1)
    head_input_norm = (enc_opt * gate.unsqueeze(-1)).norm(dim=-1)
    return gate, search_norm, head_input_norm


def _diff_same_shape(a, b):
    a = _as_tensor_1d(a)
    b = _as_tensor_1d(b)
    if a is None or b is None or a.numel() != b.numel():
        return None
    return a - b


def _tensor_abs_max(x):
    x = _as_tensor_1d(x)
    if x is None:
        return 0.0
    return x.abs().max().item()


def _pair_positive_max(*items):
    max_value = 0.0
    for item in items:
        heat = _as_tensor_1d(item)
        if heat is not None:
            max_value = max(max_value, heat.max().item())
    return max(max_value, 1e-12)


def _short_label(text, limit=28):
    text = str(text)
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _stage_label(loc, stage_idx, total_stages):
    return "S{}/L{}".format(stage_idx + 1, loc) if total_stages > 1 else "L{}".format(loc)


def _active_keep_source(pruning_locs, layer_idx):
    active = [loc for loc in pruning_locs if loc <= layer_idx]
    if not active:
        return None, None
    loc = active[-1]
    return loc, len(active) - 1


def _layer_keep_label(pruning_locs, layer_idx):
    keep_loc, _ = _active_keep_source(pruning_locs, layer_idx)
    if keep_loc is None:
        return "L{:02d}/noK".format(layer_idx)
    return "L{:02d}/K{}".format(layer_idx, keep_loc)


def _draw_text(img, text, xy, scale=0.42):
    if not text:
        return
    cv.putText(img, text, xy, cv.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(img, text, xy, cv.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 1, cv.LINE_AA)


def _normalize_heat_array(heat, tile_size):
    heat = _as_tensor_1d(heat)
    if heat is None:
        return None, None
    feat_sz = _feat_size(heat.numel())
    heat_np = heat.reshape(feat_sz, feat_sz).numpy().astype(np.float32)
    heat_np = cv.resize(heat_np, (tile_size, tile_size), interpolation=cv.INTER_LINEAR)
    return heat_np, heat


def _linear_color_map(heat_np, vmin=None, vmax=None):
    if vmin is None:
        vmin = float(np.nanmin(heat_np))
    if vmax is None:
        vmax = float(np.nanmax(heat_np))
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax):
        vmax = vmin
    den = max(vmax - vmin, 1e-12)
    norm = np.clip((heat_np - vmin) / den, 0.0, 1.0)
    return cv.applyColorMap(np.uint8(norm * 255), cv.COLORMAP_JET)


def _signed_color_map(heat_np, limit):
    limit = max(float(limit), 1e-12)
    norm = np.clip(heat_np / limit, -1.0, 1.0)
    pos = np.clip(norm, 0.0, 1.0)
    neg = np.clip(-norm, 0.0, 1.0)
    color = np.full((*heat_np.shape, 3), 255, dtype=np.float32)
    color[..., 0] = 255.0 * (1.0 - pos)
    color[..., 1] = 255.0 * (1.0 - 0.75 * np.maximum(pos, neg))
    color[..., 2] = 255.0 * (1.0 - neg)
    return np.uint8(np.clip(color, 0, 255))


def _render_map_tile(spec, search_img, tile_size):
    heat_np, heat = _normalize_heat_array(spec.get("heat"), tile_size)
    if heat_np is None:
        return None
    search_bgr = cv.cvtColor(search_img, cv.COLOR_RGB2BGR)
    search_bgr = cv.resize(search_bgr, (tile_size, tile_size))
    mode = spec.get("mode", "minmax")
    if mode == "signed":
        color = _signed_color_map(heat_np, spec.get("limit", _tensor_abs_max(heat)))
    elif mode == "fixed":
        color = _linear_color_map(heat_np, vmin=spec.get("vmin", 0.0), vmax=spec.get("vmax", 1.0))
    else:
        color = _linear_color_map(heat_np)
    factor = float(spec.get("factor", 0.45))
    return np.uint8(search_bgr * (1.0 - factor) + color * factor)


def _plain_tile(search_img, tile_size):
    search_bgr = cv.cvtColor(search_img, cv.COLOR_RGB2BGR)
    return cv.resize(search_bgr, (tile_size, tile_size))


def _draw_xywh_box(img, box, color, label):
    if box is None:
        return
    x, y, w, h = [float(v) for v in box]
    p1 = (int(round(x)), int(round(y)))
    p2 = (int(round(x + w)), int(round(y + h)))
    cv.rectangle(img, p1, p2, color, 2, cv.LINE_AA)
    cv.putText(img, label, (p1[0], max(16, p1[1] - 5)), cv.FONT_HERSHEY_SIMPLEX,
               0.55, color, 2, cv.LINE_AA)


def _save_original_view(save_dir, frame_num, image_rgb, gt_box, pred_box, language_text):
    image_bgr = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR).copy()
    _draw_xywh_box(image_bgr, gt_box, (0, 220, 0), "GT")
    _draw_xywh_box(image_bgr, pred_box, (0, 0, 255), "Pred")
    cv.putText(image_bgr, _short_label(str(language_text), 90), (12, 24),
               cv.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv.LINE_AA)
    os.makedirs(save_dir, exist_ok=True)
    cv.imwrite(os.path.join(save_dir, "{:04d}_original_view.jpg".format(frame_num)), image_bgr)


def _should_save_original_view(args):
    mode = str(getattr(args, "original_view", "auto")).lower()
    if mode in ("1", "true", "yes", "on"):
        return True
    if mode in ("0", "false", "no", "off"):
        return False
    return str(getattr(args, "dataset_name", "")).lower().startswith("hoot")


def _draw_patch_grid(img, feat_sz, color=(230, 230, 230), thickness=1):
    if feat_sz <= 1:
        return
    h, w = img.shape[:2]
    for i in range(1, feat_sz):
        x = int(round(i * w / float(feat_sz)))
        y = int(round(i * h / float(feat_sz)))
        cv.line(img, (x, 0), (x, h - 1), color, thickness, cv.LINE_AA)
        cv.line(img, (0, y), (w - 1, y), color, thickness, cv.LINE_AA)


def _language_token_labels(tracker, text):
    try:
        tokenizer = tracker.network.backbone.tokenizer
        ids = tokenizer([text], add_special_tokens=True, truncation=True,
                        pad_to_max_length=True, max_length=16)["input_ids"][0]
        return tokenizer.convert_ids_to_tokens(ids)
    except Exception:
        return ["tok{}".format(i) for i in range(16)]


def _bar_tile(values, labels, tile_size=142):
    values = _as_tensor_1d(values)
    canvas = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
    if values is None or values.numel() == 0:
        return canvas
    values = values.clamp(0, 1)
    n = values.numel()
    bar_w = max(2, int((tile_size - 18) / max(n, 1)))
    base_y = tile_size - 18
    max_h = tile_size - 46
    for idx, value in enumerate(values):
        x0 = 8 + idx * bar_w
        x1 = min(tile_size - 8, x0 + max(bar_w - 1, 1))
        h = int(round(float(value.item()) * max_h))
        color = (40, int(220 - 120 * float(value.item())), int(80 + 140 * float(value.item())))
        cv.rectangle(canvas, (x0, base_y - h), (x1, base_y), color, -1)
    top_k = min(4, n)
    top_idx = torch.topk(values, top_k).indices.tolist()
    y = 13
    for idx in top_idx:
        label = labels[idx] if idx < len(labels) else "tok{}".format(idx)
        text = "{}:{:.2f}".format(label.replace("##", ""), float(values[idx].item()))
        cv.putText(canvas, _short_label(text, 18), (6, y), cv.FONT_HERSHEY_SIMPLEX,
                   0.34, (30, 30, 30), 1, cv.LINE_AA)
        y += 12
    return canvas


def _vector_stats(name, values):
    values = _as_tensor_1d(values)
    if values is None or values.numel() == 0:
        return {}
    prob = values.clamp_min(0)
    prob = prob / prob.sum().clamp_min(1e-12)
    entropy = -(prob * prob.clamp_min(1e-12).log()).sum() / math.log(max(values.numel(), 2))
    return {
        "{}_mean".format(name): values.mean().item(),
        "{}_min".format(name): values.min().item(),
        "{}_max".format(name): values.max().item(),
        "{}_entropy".format(name): entropy.item(),
    }


def _put_header(canvas, text, x, y, width, scale=0.43):
    cv.putText(canvas, _short_label(text, 24), (x, y), cv.FONT_HERSHEY_SIMPLEX, scale, (35, 35, 35), 1, cv.LINE_AA)


def _save_story_grid(save_path, title, subtitle, row_labels, col_labels, tile_rows, note=None):
    tile_size = 142
    label_w = 116
    gap = 8
    header_h = 102
    col_h = 46
    rows = len(tile_rows)
    cols = max(len(r) for r in tile_rows) if rows else 0
    if rows == 0 or cols == 0:
        return
    canvas_w = label_w + gap + cols * tile_size + (cols + 1) * gap
    canvas_h = header_h + col_h + rows * tile_size + (rows + 1) * gap
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    cv.putText(canvas, title, (12, 30), cv.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv.LINE_AA)
    cv.putText(canvas, subtitle, (12, 58), cv.FONT_HERSHEY_SIMPLEX, 0.48, (60, 60, 60), 1, cv.LINE_AA)
    if note is None:
        note = "TE stages are cumulative keep probabilities; attention maps use track token Q0."
    cv.putText(canvas, _short_label(note, 120),
               (12, 82), cv.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv.LINE_AA)

    x0 = label_w + 2 * gap
    for col, label in enumerate(col_labels):
        _put_header(canvas, label, x0 + col * (tile_size + gap), header_h + 18, tile_size)

    for row_idx, tiles in enumerate(tile_rows):
        y = header_h + col_h + gap + row_idx * (tile_size + gap)
        cv.putText(canvas, _short_label(row_labels[row_idx], 15), (10, y + 28),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv.LINE_AA)
        for col_idx, tile in enumerate(tiles):
            if tile is None:
                tile = np.full((tile_size, tile_size, 3), 245, dtype=np.uint8)
            x = x0 + col_idx * (tile_size + gap)
            canvas[y:y + tile_size, x:x + tile_size] = tile

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv.imwrite(save_path, canvas)


def _template_decision_to_map(decision, template_len):
    decision = _as_tensor_1d(decision)
    if decision is None:
        return None
    if template_len <= 0 or decision.numel() % template_len != 0:
        return decision
    num_templates = decision.numel() // template_len
    return decision.view(num_templates, template_len).mean(dim=0)


def _decision_stage(aux, name, stage_idx):
    items = aux.get(name)
    if not items or stage_idx >= len(items):
        return None
    value = items[stage_idx]
    if isinstance(value, torch.Tensor):
        return value[0].detach().float().cpu().view(-1)
    return None


def _query_prior_stage(aux, name, stage_idx):
    items = aux.get(name)
    if not items or stage_idx >= len(items):
        return None
    value = items[stage_idx]
    if not isinstance(value, torch.Tensor):
        return None
    value = value[0].detach().float().cpu()
    if value.dim() != 2:
        return None
    return value


def _prob_keep_stage(aux, name, stage_idx):
    items = aux.get(name)
    if not items or stage_idx >= len(items):
        return None
    value = items[stage_idx]
    if isinstance(value, torch.Tensor):
        return value[0, :, 0].detach().float().cpu().view(-1)
    return None


def _save_visualte_story(save_dir, frame_num, search_img, aux, score_map, track_delta, gt_box, pred_box,
                         pruning_locs, search_len):
    tile_size = 142
    feat_sz = _feat_size(search_len)
    search_tile = _plain_tile(search_img, tile_size)
    _draw_patch_grid(search_tile, feat_sz)

    cols = ["search", "score map"]
    tiles = [search_tile, _render_map_tile({"heat": score_map, "mode": "minmax"}, search_img, tile_size)]
    for stage_idx, loc in enumerate(pruning_locs):
        decision = _decision_stage(aux, "visual_te_search_decisions", stage_idx)
        cols.append("{} keep".format(_stage_label(loc, stage_idx, len(pruning_locs))))
        tiles.append(_render_map_tile({"heat": decision, "mode": "fixed", "vmin": 0.0, "vmax": 1.0}, search_img, tile_size))
    cols.append("track A delta")
    tiles.append(_render_map_tile({"heat": track_delta, "mode": "signed",
                                   "limit": max(_tensor_abs_max(track_delta), 1e-12)}, search_img, tile_size))

    title = "Frame {} Visual TE story".format(frame_num)
    subtitle = "GT {} | Pred {} | pruning {}".format(
        "[{:.1f},{:.1f},{:.1f},{:.1f}]".format(*gt_box) if gt_box is not None else "n/a",
        "[{:.1f},{:.1f},{:.1f},{:.1f}]".format(*pred_box) if pred_box is not None else "n/a",
        ",".join(str(i) for i in pruning_locs))
    save_path = os.path.join(save_dir, "{:04d}_visualte_story.jpg".format(frame_num))
    _save_story_grid(save_path, title, subtitle, ["search TE"], cols, [tiles])


def _save_q_story(save_dir, frame_num, search_img, layer_rows, gt_box, pred_box):
    col_labels = ["A raw", "TE keep", "A policy", "A delta", "A' * ||V_j||"]
    title = "Frame {} Visual TE track-Q story".format(frame_num)
    subtitle = "GT {} | Pred {} | query row Q0".format(
        "[{:.1f},{:.1f},{:.1f},{:.1f}]".format(*gt_box) if gt_box is not None else "n/a",
        "[{:.1f},{:.1f},{:.1f},{:.1f}]".format(*pred_box) if pred_box is not None else "n/a")
    save_path = os.path.join(save_dir, "{:04d}_visualte_q_story.jpg".format(frame_num))
    _save_story_grid(save_path, title, subtitle, [label for label, _ in layer_rows], col_labels,
                     [tiles for _, tiles in layer_rows],
                     note="Row label Lxx/Kyy means attention at layer xx using the latest keep generated at layer yy.")


def _save_track_tokens_story(save_dir, frame_num, search_img, snapshots, out_dict,
                             score_map, gt_box, pred_box, pruning_locs, search_len,
                             l_len, template_len, max_tokens=3):
    rows = []
    row_labels = []
    tile_size = 142
    template_tokens = _as_tensor_1d(_decision_stage(out_dict, "visual_te_template_decisions", 0))
    template_tokens = template_tokens.numel() if template_tokens is not None else template_len
    for layer_idx, snapshot in sorted(snapshots.items()):
        layout = _infer_token_layout(snapshot, l_len, template_tokens, search_len)
        track_start, track_end = layout.get("track", (0, 0)) if layout is not None else (0, 0)
        for q_idx in range(max_tokens):
            if q_idx < track_start or q_idx >= track_end:
                continue
            raw_query = _search_attn_row(snapshot.get("raw_attn"), q_idx, search_len)
            policy_query = _search_attn_row(snapshot.get("policy_attn"), q_idx, search_len)
            if raw_query is None or policy_query is None:
                continue
            delta = _as_tensor_1d(policy_query) - _as_tensor_1d(raw_query)
            v_norm = _v_search_norm(snapshot, search_len)
            contribution = _as_tensor_1d(policy_query) * _as_tensor_1d(v_norm) if v_norm is not None else None
            keep_stage = None
            past_locs = [loc for loc in pruning_locs if loc <= layer_idx]
            if past_locs:
                keep_stage = _decision_stage(out_dict, "visual_te_search_decisions", len(past_locs) - 1)
            scale = _pair_positive_max(raw_query, policy_query)
            row_labels.append("{}/Q{}".format(_layer_keep_label(pruning_locs, layer_idx), q_idx))
            rows.append([
                _render_map_tile({"heat": raw_query, "mode": "fixed", "vmin": 0.0, "vmax": scale}, search_img, tile_size),
                _render_map_tile({"heat": keep_stage, "mode": "fixed", "vmin": 0.0, "vmax": 1.0}, search_img, tile_size),
                _render_map_tile({"heat": policy_query, "mode": "fixed", "vmin": 0.0, "vmax": scale}, search_img, tile_size),
                _render_map_tile({"heat": delta, "mode": "signed",
                                  "limit": max(_tensor_abs_max(delta), 1e-12)}, search_img, tile_size),
                _render_map_tile({"heat": contribution, "mode": "minmax"}, search_img, tile_size),
                _render_map_tile({"heat": score_map, "mode": "minmax"}, search_img, tile_size),
            ])
    if not rows:
        return
    title = "Frame {} Q0-Q2 temporal query story".format(frame_num)
    subtitle = "GT {} | Pred {} | only valid temporal rows are shown".format(
        "[{:.1f},{:.1f},{:.1f},{:.1f}]".format(*gt_box) if gt_box is not None else "n/a",
        "[{:.1f},{:.1f},{:.1f},{:.1f}]".format(*pred_box) if pred_box is not None else "n/a")
    save_path = os.path.join(save_dir, "{:04d}_track_tokens_story.jpg".format(frame_num))
    _save_story_grid(save_path, title, subtitle, row_labels,
                     ["A raw", "TE keep", "A policy", "A delta", "A' * ||V_j||", "score"],
                     rows,
                     note="Lxx/Kyy means layer xx uses keep from TE layer yy; Q0 is used by the head.")


def _save_visual_query_story(save_dir, frame_num, search_img, snapshots, out_dict,
                             score_map, gt_box, pred_box, pruning_locs, search_len,
                             template_len, l_len, gt_mask):
    rows = []
    row_labels = []
    tile_size = 142
    template_tokens = _as_tensor_1d(_decision_stage(out_dict, "visual_te_template_decisions", 0))
    template_tokens = template_tokens.numel() if template_tokens is not None else template_len
    for layer_idx, snapshot in sorted(snapshots.items()):
        layout = _infer_token_layout(snapshot, l_len, template_tokens, search_len)
        if layout is None:
            continue
        keep_stage = None
        past_locs = [loc for loc in pruning_locs if loc <= layer_idx]
        if past_locs:
            keep_stage = _decision_stage(out_dict, "visual_te_search_decisions", len(past_locs) - 1)
        v_norm = _v_search_norm(snapshot, search_len)

        template_raw = _query_group_search_row(snapshot, layout, "template", "raw_attn", search_len)
        template_policy = _query_group_search_row(snapshot, layout, "template", "policy_attn", search_len)
        search_raw = _query_group_search_row(snapshot, layout, "search", "raw_attn", search_len)
        search_policy = _query_group_search_row(snapshot, layout, "search", "policy_attn", search_len)
        search_gt_raw = _query_group_search_row(
            snapshot, layout, "search", "raw_attn", search_len, row_mask=gt_mask)
        search_gt_policy = _query_group_search_row(
            snapshot, layout, "search", "policy_attn", search_len, row_mask=gt_mask)
        search_bg_raw = _query_group_search_row(
            snapshot, layout, "search", "raw_attn", search_len, row_mask=~gt_mask.bool())
        search_bg_policy = _query_group_search_row(
            snapshot, layout, "search", "policy_attn", search_len, row_mask=~gt_mask.bool())

        template_delta = _diff_same_shape(template_policy, template_raw)
        search_delta = _diff_same_shape(search_policy, search_raw)
        search_gt_delta = _diff_same_shape(search_gt_policy, search_gt_raw)
        search_bg_delta = _diff_same_shape(search_bg_policy, search_bg_raw)
        template_contrib = (_as_tensor_1d(template_policy) * _as_tensor_1d(v_norm)
                            if template_policy is not None and v_norm is not None else None)
        search_contrib = (_as_tensor_1d(search_policy) * _as_tensor_1d(v_norm)
                          if search_policy is not None and v_norm is not None else None)
        policy_scale = _pair_positive_max(template_policy, search_policy)
        delta_scale = max(_tensor_abs_max(template_delta), _tensor_abs_max(search_delta),
                          _tensor_abs_max(search_gt_delta), _tensor_abs_max(search_bg_delta), 1e-12)

        row_labels.append(_layer_keep_label(pruning_locs, layer_idx))
        rows.append([
            _render_map_tile({"heat": keep_stage, "mode": "fixed", "vmin": 0.0, "vmax": 1.0}, search_img, tile_size),
            _render_map_tile({"heat": template_policy, "mode": "fixed", "vmin": 0.0, "vmax": policy_scale}, search_img, tile_size),
            _render_map_tile({"heat": template_delta, "mode": "signed", "limit": delta_scale}, search_img, tile_size),
            _render_map_tile({"heat": template_contrib, "mode": "minmax"}, search_img, tile_size),
            _render_map_tile({"heat": search_policy, "mode": "fixed", "vmin": 0.0, "vmax": policy_scale}, search_img, tile_size),
            _render_map_tile({"heat": search_delta, "mode": "signed", "limit": delta_scale}, search_img, tile_size),
            _render_map_tile({"heat": search_contrib, "mode": "minmax"}, search_img, tile_size),
            _render_map_tile({"heat": search_gt_delta, "mode": "signed", "limit": delta_scale}, search_img, tile_size),
            _render_map_tile({"heat": search_bg_delta, "mode": "signed", "limit": delta_scale}, search_img, tile_size),
            _render_map_tile({"heat": score_map, "mode": "minmax"}, search_img, tile_size),
        ])
    if not rows:
        return
    title = "Frame {} template/search query story".format(frame_num)
    subtitle = "GT {} | Pred {} | averaged query rows, visual keys shown on search grid".format(
        "[{:.1f},{:.1f},{:.1f},{:.1f}]".format(*gt_box) if gt_box is not None else "n/a",
        "[{:.1f},{:.1f},{:.1f},{:.1f}]".format(*pred_box) if pred_box is not None else "n/a")
    save_path = os.path.join(save_dir, "{:04d}_visual_query_story.jpg".format(frame_num))
    _save_story_grid(save_path, title, subtitle, row_labels,
                     ["keep", "TE-Q A'", "TE-Q dA", "TE-Q A'||V||",
                      "SE-Q A'", "SE-Q dA", "SE-Q A'||V||", "SE-GTQ dA", "SE-BGQ dA", "score"],
                     rows,
                     note="Lxx/Kyy marks which keep controls that layer; TE-Q/template rows and SE-Q/search rows are averaged separately.")


def _save_score_input_story(save_dir, frame_num, search_img, out_dict, off_out,
                            score_map, off_score_map, gt_box, pred_box, feat_len_s):
    gate, search_norm, head_input_norm = _head_input_maps(out_dict, feat_len_s)
    off_gate, _, off_head_input_norm = _head_input_maps(off_out, feat_len_s)
    gate_delta = _diff_same_shape(gate, off_gate)
    head_input_delta = _diff_same_shape(head_input_norm, off_head_input_norm)
    score_delta = _diff_same_shape(score_map, off_score_map)
    if gate is None and head_input_norm is None and score_map is None:
        return
    tile_size = 142
    rows = [[
        _render_map_tile({"heat": gate, "mode": "signed", "limit": max(_tensor_abs_max(gate), 1e-12)}, search_img, tile_size),
        _render_map_tile({"heat": search_norm, "mode": "minmax"}, search_img, tile_size),
        _render_map_tile({"heat": head_input_norm, "mode": "minmax"}, search_img, tile_size),
        _render_map_tile({"heat": gate_delta, "mode": "signed",
                          "limit": max(_tensor_abs_max(gate_delta), 1e-12)}, search_img, tile_size),
        _render_map_tile({"heat": head_input_delta, "mode": "signed",
                          "limit": max(_tensor_abs_max(head_input_delta), 1e-12)}, search_img, tile_size),
        _render_map_tile({"heat": score_map, "mode": "minmax"}, search_img, tile_size),
        _render_map_tile({"heat": score_delta, "mode": "signed",
                          "limit": max(_tensor_abs_max(score_delta), 1e-12)}, search_img, tile_size),
    ]]
    title = "Frame {} final score-head input story".format(frame_num)
    subtitle = "GT {} | Pred {} | on-policy minus no-policy deltas when available".format(
        "[{:.1f},{:.1f},{:.1f},{:.1f}]".format(*gt_box) if gt_box is not None else "n/a",
        "[{:.1f},{:.1f},{:.1f},{:.1f}]".format(*pred_box) if pred_box is not None else "n/a")
    save_path = os.path.join(save_dir, "{:04d}_score_input_story.jpg".format(frame_num))
    _save_story_grid(save_path, title, subtitle, ["head"],
                     ["q0 dot x", "||x_j||", "||dot*x_j||", "dot delta",
                      "input delta", "score", "score delta"],
                     rows,
                     note="DUTrack head input is search feature x_j multiplied by raw q0 dot x_j, then passed to the CENTER head.")


def _save_lte_parallel_story(save_dir, frame_num, search_img, out_dict, snapshots, score_map,
                             gt_box, pred_box, pruning_locs, search_len, template_len,
                             l_len, token_labels):
    rows = []
    row_labels = []
    tile_size = 142
    z_len = _as_tensor_1d(_decision_stage(out_dict, "visual_te_template_decisions", 0))
    z_len = z_len.numel() if z_len is not None else template_len
    for stage_idx, loc in enumerate(pruning_locs):
        snapshot = snapshots.get(loc)
        language_keep = _decision_stage(out_dict, "lang_te_language_decisions", stage_idx)
        template_keep = _decision_stage(out_dict, "visual_te_template_decisions", stage_idx)
        template_keep = _template_decision_to_map(template_keep, template_len)
        search_keep = _decision_stage(out_dict, "visual_te_search_decisions", stage_idx)
        raw_search = policy_search = delta_search = None
        if snapshot is not None:
            raw_search = _search_attn_row(snapshot.get("raw_attn"), 0, search_len)
            policy_search = _search_attn_row(snapshot.get("policy_attn"), 0, search_len)
            if raw_search is not None and policy_search is not None:
                delta_search = _as_tensor_1d(policy_search) - _as_tensor_1d(raw_search)
        scale = _pair_positive_max(raw_search, policy_search)
        row_labels.append(_stage_label(loc, stage_idx, len(pruning_locs)))
        rows.append([
            _bar_tile(language_keep, token_labels, tile_size),
            _render_map_tile({"heat": template_keep, "mode": "fixed", "vmin": 0.0, "vmax": 1.0}, search_img, tile_size),
            _render_map_tile({"heat": search_keep, "mode": "fixed", "vmin": 0.0, "vmax": 1.0}, search_img, tile_size),
            _render_map_tile({"heat": raw_search, "mode": "fixed", "vmin": 0.0, "vmax": scale}, search_img, tile_size),
            _render_map_tile({"heat": policy_search, "mode": "fixed", "vmin": 0.0, "vmax": scale}, search_img, tile_size),
            _render_map_tile({"heat": delta_search, "mode": "signed",
                              "limit": max(_tensor_abs_max(delta_search), 1e-12)}, search_img, tile_size),
            _render_map_tile({"heat": score_map, "mode": "minmax"}, search_img, tile_size),
        ])
    if not rows:
        return
    title = "Frame {} Language-Visual TE story".format(frame_num)
    subtitle = "GT {} | Pred {} | stages {}".format(
        "[{:.1f},{:.1f},{:.1f},{:.1f}]".format(*gt_box) if gt_box is not None else "n/a",
        "[{:.1f},{:.1f},{:.1f},{:.1f}]".format(*pred_box) if pred_box is not None else "n/a",
        ",".join(str(i) for i in pruning_locs))
    save_path = os.path.join(save_dir, "{:04d}_lte_parallel_story.jpg".format(frame_num))
    _save_story_grid(save_path, title, subtitle, row_labels,
                     ["word keep", "template keep", "search keep", "Q0 raw->x",
                      "Q0 policy->x", "Q0 delta", "score map"], rows)


def _write_csv(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean_value(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _sum_value(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return sum(values) if values else 0.0


def _unique_text_count(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None and str(value) != "":
            values.append(str(value))
    return len(set(values))


def _write_summary(save_dir, rows, pruning_locs):
    has_language_te = any("language_keep_L{}_mean".format(loc) in row for row in rows for loc in pruning_locs)
    lines = [
        "# Visual TE Diagnostic",
        "",
        "This diagnostic checks token keep prediction -> attention policy -> final score map.",
        "",
        "TE update layers: `{}`. Between two update layers, the previous keep is reused by later blocks.".format(
            ",".join(str(loc) for loc in pruning_locs) if pruning_locs else "none"),
        "",
        "Language-aware TE fields are included when `lang_te_*` tensors are present.",
        "",
    ]
    if has_language_te:
        lines.extend([
            "| Stage | Lang Keep Mean | Search Keep In GT | Search Keep Out GT | Track Delta In GT | Track Delta Out GT | Non-track Delta Max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
    else:
        lines.extend([
            "| Stage | Search Keep In GT | Search Keep Out GT | Track Delta In GT | Track Delta Out GT |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
    detail_lines = []
    for stage_idx, loc in enumerate(pruning_locs):
        keep_in = _mean_value(rows, "search_keep_L{}_mean_in_gt".format(loc))
        keep_out = _mean_value(rows, "search_keep_L{}_mean_out_gt".format(loc))
        keep_gap = _mean_value(rows, "search_keep_L{}_gap_in_minus_out".format(loc))
        keep_ratio = _mean_value(rows, "search_keep_L{}_ratio_in_over_out".format(loc))
        delta_in = _mean_value(rows, "track_delta_L{}_mean_in_gt".format(loc))
        delta_out = _mean_value(rows, "track_delta_L{}_mean_out_gt".format(loc))
        if has_language_te:
            lang_mean = _mean_value(rows, "language_keep_L{}_mean".format(loc))
            nontrack_delta = _mean_value(rows, "nontrack_delta_absmax_L{}".format(loc))
            lines.append("| {} | {:.6f} | {:.6f} | {:.6f} | {:+.6f} | {:+.6f} | {:.8f} |".format(
                _stage_label(loc, stage_idx, len(pruning_locs)),
                lang_mean, keep_in, keep_out, delta_in, delta_out, nontrack_delta))
        else:
            lines.append("| {} | {:.6f} | {:.6f} | {:+.6f} | {:+.6f} |".format(
                _stage_label(loc, stage_idx, len(pruning_locs)), keep_in, keep_out, delta_in, delta_out))
        detail_lines.append("- {} keep gap/ratio: {:+.6f} / {:.6f}".format(
            _stage_label(loc, stage_idx, len(pruning_locs)), keep_gap, keep_ratio))
        for group in ("track", "language", "template", "search"):
            group_delta = _mean_value(rows, "q0_full_L{}_{}_mass_delta".format(loc, group))
            if math.isfinite(group_delta):
                detail_lines.append("- L{} Q0 {} mass delta: {:+.6f}".format(loc, group, group_delta))
        kl_value = _mean_value(rows, "q0_full_L{}_kl_raw_to_policy".format(loc))
        l1_value = _mean_value(rows, "q0_full_L{}_l1_raw_policy".format(loc))
        if math.isfinite(kl_value) or math.isfinite(l1_value):
            detail_lines.append("- L{} Q0 attention KL/L1: {:.6f} / {:.6f}".format(loc, kl_value, l1_value))
        visual_kl = _mean_value(rows, "visualq_full_L{}_kl_raw_to_policy".format(loc))
        visual_l1 = _mean_value(rows, "visualq_full_L{}_l1_raw_policy".format(loc))
        if math.isfinite(visual_kl) or math.isfinite(visual_l1):
            detail_lines.append("- L{} visual-Q attention KL/L1: {:.6f} / {:.6f}".format(loc, visual_kl, visual_l1))
        visual_search_delta = _mean_value(rows, "visualq_full_L{}_search_mass_delta".format(loc))
        visual_gt_delta = _mean_value(rows, "visualq_full_L{}_search_gt_abs_mass_delta".format(loc))
        if math.isfinite(visual_search_delta) or math.isfinite(visual_gt_delta):
            detail_lines.append("- L{} visual-Q search mass / GT-search delta: {:+.6f} / {:+.6f}".format(
                loc, visual_search_delta, visual_gt_delta))
        searchq_search_delta = _mean_value(rows, "searchq_full_L{}_search_mass_delta".format(loc))
        searchq_gt_delta = _mean_value(rows, "searchq_full_L{}_search_gt_abs_mass_delta".format(loc))
        if math.isfinite(searchq_search_delta) or math.isfinite(searchq_gt_delta):
            detail_lines.append("- L{} search-Q search mass / GT-search delta: {:+.6f} / {:+.6f}".format(
                loc, searchq_search_delta, searchq_gt_delta))
        gtrow_gt_delta = _mean_value(rows, "searchq_gtrows_full_L{}_search_gt_abs_mass_delta".format(loc))
        bgrow_gt_delta = _mean_value(rows, "searchq_bgrows_full_L{}_search_gt_abs_mass_delta".format(loc))
        if math.isfinite(gtrow_gt_delta) or math.isfinite(bgrow_gt_delta):
            detail_lines.append("- L{} search-Q GT-row / BG-row to GT-search delta: {:+.6f} / {:+.6f}".format(
                loc, gtrow_gt_delta, bgrow_gt_delta))
        templateq_gt_delta = _mean_value(rows, "templateq_full_L{}_search_gt_abs_mass_delta".format(loc))
        if math.isfinite(templateq_gt_delta):
            detail_lines.append("- L{} template-Q to GT-search delta: {:+.6f}".format(loc, templateq_gt_delta))
    if detail_lines:
        lines.extend(["", "Attention/keep detail averages:", ""] + detail_lines)
    if rows:
        lines.extend([
            "",
            "Language update checks:",
            "",
            "- Frames summarized: `{}`".format(len(rows)),
            "- Update requested frames: `{:.0f}`".format(_sum_value(rows, "language_update_requested")),
            "- Description changed frames: `{:.0f}`".format(_sum_value(rows, "language_changed")),
            "- Unique descriptions: `{}`".format(_unique_text_count(rows, "language_description")),
            "- Initial diagnostic description: `{}`".format(rows[0].get("language_description", "")),
            "- Final diagnostic description: `{}`".format(rows[-1].get("language_description", "")),
        ])
    lines.extend([
        "",
        "On/off score-map checks:",
        "",
        "- `score_onoff_*` compares the normal run with a diagnostic no-policy forward pass on the same search crop.",
        "- It is diagnostic-only and does not update tracker state.",
        "- `score_onoff_language_mismatch=1` marks frames where the normal run updated language after the no-policy diagnostic forward, so on/off score comparison is partially confounded by language update.",
        "- If these fields are NaN, the extra no-policy forward was unavailable for that frame.",
        "",
        "Interpretation:",
        "",
        "- A useful TE stage should keep more search mass inside the GT than outside it.",
        "- Row labels like `L23/K15` mean the attention is captured at block 23 while using the latest keep generated at block 15.",
        "- If a TE update layer is also the final layer, for example `L23/K23`, that row is not directly comparable to an old final `L23/K11` row.",
        "- `track_delta` is `policy attention - raw attention` for the track token row Q0.",
        "- `q0_full_*_mass_delta` shows where the Q0 attention mass moved across track/language/template/search token groups.",
        "- `q0_full_*_kl_raw_to_policy` and `q0_full_*_l1_raw_policy` quantify how strongly the policy changed Q0's whole attention distribution.",
        "- `visualq_full_*`, `searchq_full_*`, and `templateq_full_*` average the same attention statistics over visual query rows.",
        "- `searchq_gtrows_full_*` and `searchq_bgrows_full_*` split search query rows by whether the query patch is inside the GT box.",
        "- `*_pos_hardneg_gap` compares a keep/word map on GT tokens against the highest-score non-GT score-map tokens; negative values mean the signal favors hard negatives.",
        "- `word_reliability_diagnostics.csv` records per-word target-vs-hard-negative evidence for each TE stage without changing the tracker language.",
        "- In that CSV, `evidence_mode=oracle` uses GT tokens as positives; `evidence_mode=deploy` uses current predicted-box tokens as positives.",
        "- Compare `word_rank_by_gap` with `word_rank_by_weight`: disagreement means the learned word weight is not a reliable proxy for visual discriminability.",
        "- `weight_gap_rank_corr` and `top3_weight_gap_overlap` summarize whether the current word weights rank the same words as target-hard-negative gap.",
        "- `Non-track Delta Max` should stay near zero for target-Q-only policies.",
        "- `*_lte_parallel_story.jpg` shows word keep, template keep, search keep, and the resulting track-Q attention change per TE layer.",
        "- `*_track_tokens_story.jpg` compares Q0/Q1/Q2 temporal rows when those rows exist in the current frame.",
        "- `*_visual_query_story.jpg` shows template-query and search-query attention changes toward search-region keys.",
        "- `*_score_input_story.jpg` shows the closest available input to the CENTER head: q0 dot search feature, search feature norm, gated feature norm, and score-map deltas.",
        "- If keep maps are broad or high on background, the TE module is not yet target-discriminative.",
        "",
    ])
    path = os.path.join(save_dir, "visualte_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _score_map_compare_stats(name, on_score, off_score, gt_mask):
    on_score = _as_tensor_1d(on_score)
    off_score = _as_tensor_1d(off_score)
    if on_score is None or off_score is None or on_score.numel() != off_score.numel():
        return {}
    delta = on_score - off_score
    stats = {}
    stats.update(_signed_heat_stats("{}_delta".format(name), delta, gt_mask, top_ratio=0.1))
    stats["{}_peak_delta".format(name)] = on_score.max().item() - off_score.max().item()
    gt_mask = gt_mask.bool()
    if gt_mask.any():
        stats["{}_peak_in_gt_on".format(name)] = on_score[gt_mask].max().item()
        stats["{}_peak_in_gt_off".format(name)] = off_score[gt_mask].max().item()
        stats["{}_peak_in_gt_delta".format(name)] = stats["{}_peak_in_gt_on".format(name)] - stats["{}_peak_in_gt_off".format(name)]
    return stats


def run(args):
    dataset = get_dataset(args.dataset_name)
    seq = dataset[int(args.sequence)] if str(args.sequence).isdigit() else dataset[args.sequence]

    tracker_info = Tracker("dutrack", args.config, args.dataset_name, args.runid)
    params = tracker_info.get_parameters(run_id=args.runid)
    params.debug = 0
    tracker = tracker_info.create_tracker(params)
    if args.language_init_source is not None:
        tracker.cfg.TEST.LANGUAGE_INIT_SOURCE = str(args.language_init_source)
    if args.language_update_mode is not None:
        tracker.cfg.TEST.LANGUAGE_UPDATE_MODE = str(args.language_update_mode)
    if args.language_word_filter is not None:
        tracker.cfg.TEST.LANGUAGE_WORD_FILTER_ENABLE = bool(args.language_word_filter)
    if args.language_word_filter_threshold is not None:
        tracker.cfg.TEST.LANGUAGE_WORD_FILTER_THRESHOLD = float(args.language_word_filter_threshold)
    if args.language_word_reliability is not None:
        tracker.cfg.TEST.LANGUAGE_WORD_RELIABILITY_ENABLE = bool(args.language_word_reliability)
    if args.language_word_reliability_source is not None:
        tracker.cfg.TEST.LANGUAGE_WORD_RELIABILITY_SOURCE = str(args.language_word_reliability_source)
    if args.language_word_reliability_momentum is not None:
        tracker.cfg.TEST.LANGUAGE_WORD_RELIABILITY_MOMENTUM = float(args.language_word_reliability_momentum)
    if args.language_word_reliability_tau is not None:
        tracker.cfg.TEST.LANGUAGE_WORD_RELIABILITY_TAU = float(args.language_word_reliability_tau)
    if args.language_subject_min_reliability is not None:
        tracker.cfg.TEST.LANGUAGE_SUBJECT_MIN_RELIABILITY = float(args.language_subject_min_reliability)
    if args.language_context_max_weight is not None:
        tracker.cfg.TEST.LANGUAGE_CONTEXT_MAX_WEIGHT = float(args.language_context_max_weight)
    if args.language_subject_type_prior is not None:
        tracker.cfg.TEST.LANGUAGE_SUBJECT_TYPE_PRIOR = float(args.language_subject_type_prior)
    if args.language_attribute_type_prior is not None:
        tracker.cfg.TEST.LANGUAGE_ATTRIBUTE_TYPE_PRIOR = float(args.language_attribute_type_prior)
    if args.language_context_type_prior is not None:
        tracker.cfg.TEST.LANGUAGE_CONTEXT_TYPE_PRIOR = float(args.language_context_type_prior)
    if args.language_reliability_update_gate is not None:
        tracker.cfg.TEST.LANGUAGE_RELIABILITY_UPDATE_GATE = bool(args.language_reliability_update_gate)
    if args.language_reliability_gate_mode is not None:
        tracker.cfg.TEST.LANGUAGE_RELIABILITY_GATE_MODE = str(args.language_reliability_gate_mode)
    if args.language_reliability_score_thr is not None:
        tracker.cfg.TEST.LANGUAGE_RELIABILITY_SCORE_THR = float(args.language_reliability_score_thr)
    if args.language_reliability_score_gap_thr is not None:
        tracker.cfg.TEST.LANGUAGE_RELIABILITY_SCORE_GAP_THR = float(args.language_reliability_score_gap_thr)
    if args.score_prior_beta is not None:
        if not hasattr(tracker.network, "score_prior_beta"):
            raise AttributeError("Network does not expose score_prior_beta")
        tracker.network.score_prior_beta = float(args.score_prior_beta)
    if args.score_prior_source is not None:
        if not hasattr(tracker.network, "score_prior_source"):
            raise AttributeError("Network does not expose score_prior_source")
        tracker.network.score_prior_source = str(args.score_prior_source).lower()
    if args.score_prior_layer is not None:
        if not hasattr(tracker.network, "score_prior_layer"):
            raise AttributeError("Network does not expose score_prior_layer")
        tracker.network.score_prior_layer = int(args.score_prior_layer)

    save_dir = os.path.join(args.out_dir, _run_label(args.config, args), seq.name)
    os.makedirs(save_dir, exist_ok=True)

    main_blocks = tracker.network.backbone.blocks[-tracker.network.backbone.num_main_blocks:]
    pruning_locs = list(getattr(tracker.network.backbone, "visual_te_pruning_loc", []))
    lmq_locs = list(getattr(tracker.network.backbone, "language_query_prior_loc", []))
    key_layers = sorted(set(pruning_locs + lmq_locs + [len(main_blocks) - 1]))
    print("Visual TE diagnostic layers: {}".format(",".join(str(i) for i in key_layers)))

    captured = {"out": None, "layers": []}
    original_forward = tracker.network.forward

    def capture_forward(*f_args, **f_kwargs):
        captured["out"] = None
        captured["layers"] = []
        out = original_forward(*f_args, **f_kwargs)
        captured["out"] = out[-1] if isinstance(out, list) else out
        return out

    tracker.network.forward = capture_forward

    def _diagnostic_no_policy_output(search_tensor, frame_num):
        """Run a no-policy forward for score-map comparison without updating tracker state."""
        if not getattr(tracker.network.backbone, "visual_te_enabled", False):
            return None
        saved_query = tracker.network.track_query
        saved_policy_apply = getattr(tracker.network.backbone, "te_policy_apply", None)
        saved_score_prior_enabled = getattr(tracker.network, "score_prior_enabled", None)
        old_frame_id = tracker.frame_id
        try:
            if frame_num <= tracker.cfg.TEST.TEMPLATE_NUMBER:
                template_list = tracker.memory_frames.copy()
            else:
                tracker.frame_id = frame_num
                template_list, _ = tracker.select_memory_frames()
            tracker.network.backbone.te_policy_apply = "none"
            if saved_score_prior_enabled is not None:
                tracker.network.score_prior_enabled = False
            with torch.no_grad():
                out = tracker.network.forward(
                    template=template_list,
                    search=[search_tensor],
                    descript=[[getattr(tracker, "descript", "")]])
            return out[-1] if isinstance(out, list) else out
        finally:
            tracker.frame_id = old_frame_id
            tracker.network.track_query = saved_query
            if saved_policy_apply is not None:
                tracker.network.backbone.te_policy_apply = saved_policy_apply
            if saved_score_prior_enabled is not None:
                tracker.network.score_prior_enabled = saved_score_prior_enabled
            captured["out"] = None
            captured["layers"] = []

    handles = []
    for layer_idx, block in enumerate(main_blocks):
        if layer_idx not in key_layers or block.attn is None:
            continue

        def _hook(module, inputs, output, idx=layer_idx):
            snapshot = _attention_snapshot(module, inputs, output)
            if snapshot is not None:
                captured["layers"].append((idx, snapshot))

        handles.append(block.attn.register_forward_hook(_hook))

    try:
        image = _read_rgb(seq.frames[0])
        init_info = seq.init_info()
        if args.language_description:
            init_info["init_text_description"] = str(args.language_description)
            init_info["text_description"] = str(args.language_description)
        init_info["class"] = seq.object_class
        init_info["path"] = seq.name
        tracker.initialize(image, init_info)
        prev_output = OrderedDict({"target_bbox": init_info.get("init_bbox")})

        search_len = int(tracker.network.backbone.pos_embed_x.shape[1])
        template_len = int(tracker.network.backbone.pos_embed_z.shape[1])
        feat_sz = _feat_size(search_len)
        rows = []
        stat_frames = args.stat_frames if args.stat_frames is not None else args.max_frames
        vis_frames = args.vis_frames if args.vis_frames is not None else args.max_frames
        max_available_frame = len(seq.frames) - 1
        max_frame = max_available_frame if int(stat_frames) <= 0 else min(int(stat_frames), max_available_frame)
        vis_frame_limit = max(0, int(vis_frames))
        word_rows = []
        for frame_num in range(1, max_frame + 1):
            image = _read_rgb(seq.frames[frame_num])
            prev_state = list(tracker.state)
            language_before = str(getattr(tracker, "descript", ""))
            update_requested = bool(getattr(tracker, "updata_key", False))
            search_img, resize_factor, search_amask = sample_target(
                image, prev_state, params.search_factor, output_sz=params.search_size)
            crop_box = _search_crop_box(prev_state, resize_factor, params.search_size)

            info = seq.frame_info(frame_num)
            info["previous_output"] = prev_output
            info["class"] = seq.object_class
            info["path"] = seq.name
            info["num"] = frame_num
            off_out = None
            try:
                search_nt = tracker.preprocessor.process(search_img, search_amask)
                off_out = _diagnostic_no_policy_output(search_nt.tensors, tracker.frame_id + 1)
            except Exception as exc:
                print("No-policy diagnostic failed on frame {}: {}".format(frame_num, exc))
            track_out = tracker.track(image, info)
            prev_output = OrderedDict(track_out)
            language_after = str(getattr(tracker, "descript", ""))
            language_changed = language_after != language_before

            pred_box = track_out.get("target_bbox")
            gt_box = seq.ground_truth_rect[frame_num].tolist() if seq.ground_truth_rect is not None else None
            gt_mask = _token_box_mask(gt_box, crop_box, feat_sz) if gt_box is not None else torch.zeros(search_len, dtype=torch.bool)
            pred_mask = _token_box_mask(pred_box, crop_box, feat_sz) if pred_box is not None else torch.zeros(search_len, dtype=torch.bool)

            out_dict = captured["out"] or {}
            score_map = out_dict.get("score_map")
            if isinstance(score_map, torch.Tensor):
                score_map = score_map[0].detach().float().cpu().view(-1)
            score_prior_bias = out_dict.get("score_prior_bias")
            if isinstance(score_prior_bias, torch.Tensor):
                score_prior_bias = score_prior_bias[0].detach().float().cpu().view(-1)
            score_logits_base = out_dict.get("score_map_logits_base")
            if isinstance(score_logits_base, torch.Tensor):
                score_logits_base = score_logits_base[0].detach().float().cpu().view(-1)
            off_score_map = None
            if isinstance(off_out, dict) and isinstance(off_out.get("score_map"), torch.Tensor):
                off_score_map = off_out["score_map"][0].detach().float().cpu().view(-1)

            row = {
                "frame": frame_num,
                "sequence": seq.name,
                "pruning_locs": ",".join(str(i) for i in pruning_locs),
                "lmq_locs": ",".join(str(i) for i in lmq_locs),
                "score_prior_beta": getattr(tracker.network, "score_prior_beta", float("nan")),
                "score_prior_source": getattr(tracker.network, "score_prior_source", ""),
                "pred_x": pred_box[0],
                "pred_y": pred_box[1],
                "pred_w": pred_box[2],
                "pred_h": pred_box[3],
                "language_description_before": language_before,
                "language_description": language_after,
                "language_anchor": str(getattr(tracker, "language_anchor", "")),
                "language_source": str(getattr(tracker, "language_source", "")),
                "language_candidate_description": str(getattr(tracker, "language_candidate_description", "")),
                "language_filtered_description": str(getattr(tracker, "language_filtered_description", "")),
                "language_word_filter_active": 1.0 if bool(getattr(tracker, "language_word_filter_active", False)) else 0.0,
                "language_word_reliability_active": 1.0 if bool(getattr(tracker, "language_word_reliability_active", False)) else 0.0,
                "language_word_reliability_updated": 1.0 if bool(getattr(tracker, "language_word_reliability_updated", False)) else 0.0,
                "language_word_reliability_delta": float(getattr(tracker, "language_word_reliability_delta", 0.0)),
                "language_word_reliability_score_peak": float(getattr(tracker, "language_word_reliability_score_peak", float("nan"))),
                "language_word_reliability_hardneg_peak": float(getattr(tracker, "language_word_reliability_hardneg_peak", float("nan"))),
                "language_word_reliability_score_gap": float(getattr(tracker, "language_word_reliability_score_gap", float("nan"))),
                "language_word_reliability": ";".join(
                    "{:.3f}".format(float(v)) for v in (
                        getattr(tracker, "language_word_reliability", None).tolist()
                        if getattr(tracker, "language_word_reliability", None) is not None else [])),
                "language_changed": 1.0 if language_changed else 0.0,
                "language_update_requested": 1.0 if update_requested else 0.0,
                "language_update_next": 1.0 if bool(getattr(tracker, "updata_key", False)) else 0.0,
                "trigger_by_position": 1.0 if bool(getattr(tracker, "language_trigger_by_position", False)) else 0.0,
                "trigger_by_scale": 1.0 if bool(getattr(tracker, "language_trigger_by_scale", False)) else 0.0,
                "trigger_by_color": 1.0 if bool(getattr(tracker, "language_trigger_by_color", False)) else 0.0,
                "trigger_area_ratio": float(getattr(tracker, "language_trigger_area_ratio", float("nan"))),
                "trigger_center_distance": float(getattr(tracker, "language_trigger_center_distance", float("nan"))),
                "trigger_color_delta": float(getattr(tracker, "language_trigger_color_delta", float("nan"))),
                "score_onoff_language_mismatch": 1.0 if language_changed else 0.0,
            }
            if gt_box is not None:
                row.update({"gt_x": gt_box[0], "gt_y": gt_box[1], "gt_w": gt_box[2], "gt_h": gt_box[3]})
            row.update(_heat_stats("score_map", score_map, gt_mask, args.top_ratio))
            row.update(_signed_heat_stats("score_prior_bias", score_prior_bias, gt_mask, args.top_ratio))
            row.update(_vector_stats("score_prior_bias", score_prior_bias))
            row.update(_vector_stats("score_logits_base", score_logits_base))
            if score_prior_bias is not None:
                row["score_prior_bias_abs_mean"] = score_prior_bias.abs().mean().item()
            if score_logits_base is not None:
                row["score_logits_base_abs_mean"] = score_logits_base.abs().mean().item()
            if score_prior_bias is not None and score_logits_base is not None and score_prior_bias.numel() == score_logits_base.numel():
                bias_abs_mean = score_prior_bias.abs().mean().item()
                base_abs_mean = score_logits_base.abs().mean().item()
                row["score_prior_to_base_abs_ratio"] = bias_abs_mean / max(base_abs_mean, 1e-12)
                clamp_value = float(getattr(tracker.network, "score_prior_bias_clamp", 0.0))
                if clamp_value > 0:
                    row["score_prior_bias_clamp_ratio"] = (
                        score_prior_bias.abs() >= (clamp_value - 1e-6)).float().mean().item()
                else:
                    row["score_prior_bias_clamp_ratio"] = 0.0
            row.update(_score_map_compare_stats("score_onoff", score_map, off_score_map, gt_mask))
            head_gate, search_feat_norm, head_input_norm = _head_input_maps(out_dict, tracker.network.feat_len_s)
            off_head_gate, _, off_head_input_norm = _head_input_maps(off_out, tracker.network.feat_len_s)
            row.update(_signed_heat_stats("head_gate", head_gate, gt_mask, args.top_ratio))
            row.update(_heat_stats("head_search_feat_norm", search_feat_norm, gt_mask, args.top_ratio))
            row.update(_heat_stats("head_input_norm", head_input_norm, gt_mask, args.top_ratio))
            row.update(_signed_heat_stats("head_gate_onoff_delta",
                                          _diff_same_shape(head_gate, off_head_gate), gt_mask, args.top_ratio))
            row.update(_signed_heat_stats("head_input_norm_onoff_delta",
                                          _diff_same_shape(head_input_norm, off_head_input_norm), gt_mask, args.top_ratio))

            snapshots_by_layer = {idx: snapshot for idx, snapshot in captured["layers"]}
            first_language_keep = _decision_stage(out_dict, "lang_te_language_decisions", 0)
            l_len = _as_tensor_1d(first_language_keep).numel() if first_language_keep is not None else 16
            token_labels = _language_token_labels(tracker, getattr(tracker, "descript", ""))

            for stage_idx, loc in enumerate(pruning_locs):
                search_keep = _decision_stage(out_dict, "visual_te_search_decisions", stage_idx)
                search_prob = _prob_keep_stage(out_dict, "visual_te_search_probs", stage_idx)
                template_keep = _decision_stage(out_dict, "visual_te_template_decisions", stage_idx)
                template_keep = _template_decision_to_map(template_keep, template_len)
                language_keep = _decision_stage(out_dict, "lang_te_language_decisions", stage_idx)
                row.update(_heat_stats("search_keep_L{}".format(loc), search_keep, gt_mask, args.top_ratio))
                row.update(_region_gap_stats("search_keep_L{}".format(loc), search_keep, gt_mask))
                row.update(_hard_negative_stats(
                    "search_keep_L{}".format(loc), search_keep, gt_mask, score_map, args.hardneg_topk))
                row.update(_heat_stats("search_prob_L{}".format(loc), search_prob, gt_mask, args.top_ratio))
                row.update(_region_gap_stats("search_prob_L{}".format(loc), search_prob, gt_mask))
                row.update(_vector_stats("language_keep_L{}".format(loc), language_keep))
                proto_target = _decision_stage(out_dict, "safe_proto_target_scores", stage_idx)
                proto_negative = _decision_stage(out_dict, "safe_proto_negative_scores", stage_idx)
                proto_margin = _decision_stage(out_dict, "safe_proto_margins", stage_idx)
                word_direct = _decision_stage(out_dict, "word_level_direct_scores", stage_idx)
                word_template = _decision_stage(out_dict, "word_level_template_scores", stage_idx)
                word_search_token_scores = _word_response_stage(
                    out_dict, "word_level_search_token_scores", stage_idx)
                word_stage_weights = _decision_stage(out_dict, "word_level_weights", stage_idx)
                word_stage_reliability = _decision_stage(out_dict, "word_level_reliability", stage_idx)
                row.update(_signed_heat_stats("safe_proto_target_L{}".format(loc), proto_target, gt_mask, args.top_ratio))
                row.update(_region_gap_stats("safe_proto_target_L{}".format(loc), proto_target, gt_mask))
                row.update(_signed_heat_stats("safe_proto_negative_L{}".format(loc), proto_negative, gt_mask, args.top_ratio))
                row.update(_region_gap_stats("safe_proto_negative_L{}".format(loc), proto_negative, gt_mask))
                row.update(_signed_heat_stats("safe_proto_margin_L{}".format(loc), proto_margin, gt_mask, args.top_ratio))
                row.update(_region_gap_stats("safe_proto_margin_L{}".format(loc), proto_margin, gt_mask))
                row.update(_signed_heat_stats("word_direct_L{}".format(loc), word_direct, gt_mask, args.top_ratio))
                row.update(_region_gap_stats("word_direct_L{}".format(loc), word_direct, gt_mask))
                row.update(_hard_negative_stats(
                    "word_direct_L{}".format(loc), word_direct, gt_mask, score_map, args.hardneg_topk))
                oracle_word_rows, oracle_word_summary = _word_evidence_rows(
                    frame_num, seq.name, loc, "oracle", word_search_token_scores, word_stage_weights,
                    word_stage_reliability, token_labels, gt_mask, score_map, args.hardneg_topk, args.word_evidence_tau,
                    object_class=seq.object_class)
                deploy_word_rows, deploy_word_summary = _word_evidence_rows(
                    frame_num, seq.name, loc, "deploy", word_search_token_scores, word_stage_weights,
                    word_stage_reliability, token_labels, pred_mask, score_map, args.hardneg_topk, args.word_evidence_tau,
                    object_class=seq.object_class)
                word_rows.extend(oracle_word_rows)
                word_rows.extend(deploy_word_rows)
                row.update(oracle_word_summary)
                row.update(deploy_word_summary)
                row.update(_vector_stats("word_template_L{}".format(loc), word_template))
                row["template_keep_L{}_mean".format(loc)] = _as_tensor_1d(template_keep).mean().item() if template_keep is not None else float("nan")
                snapshot = snapshots_by_layer.get(loc)
                if snapshot is not None:
                    row["nontrack_delta_absmax_L{}".format(loc)] = _non_track_delta_absmax(
                        snapshot.get("raw_attn"), snapshot.get("policy_attn"))
                    template_tokens = _as_tensor_1d(_decision_stage(out_dict, "visual_te_template_decisions", stage_idx))
                    template_tokens = template_tokens.numel() if template_tokens is not None else template_len
                    layout = _infer_token_layout(snapshot, l_len, template_tokens, search_len)
                    raw_full = _attn_row(snapshot.get("raw_attn"), 0)
                    policy_full = _attn_row(snapshot.get("policy_attn"), 0)
                    row.update(_attention_compare_stats("q0_full_L{}".format(loc), raw_full, policy_full, gt_mask, layout))
                    row.update(_query_group_compare_stats(
                        "templateq_full_L{}".format(loc), snapshot, layout, "template", gt_mask))
                    row.update(_query_group_compare_stats(
                        "searchq_full_L{}".format(loc), snapshot, layout, "search", gt_mask))
                    row.update(_query_group_compare_stats(
                        "searchq_gtrows_full_L{}".format(loc), snapshot, layout, "search", gt_mask,
                        row_mask=gt_mask))
                    row.update(_query_group_compare_stats(
                        "searchq_bgrows_full_L{}".format(loc), snapshot, layout, "search", gt_mask,
                        row_mask=~gt_mask.bool()))
                    if layout is not None and "template" in layout and "search" in layout:
                        z_start, z_end = layout["template"]
                        x_start, x_end = layout["search"]
                        raw_visual = _attn_rows_mean(snapshot.get("raw_attn"), z_start, x_end)
                        policy_visual = _attn_rows_mean(snapshot.get("policy_attn"), z_start, x_end)
                        row.update(_attention_compare_stats(
                            "visualq_full_L{}".format(loc), raw_visual, policy_visual, gt_mask, layout))
                    if layout is not None:
                        l_start, l_end = layout["language"]
                    else:
                        l_start, l_end = 1, 1 + l_len
                    raw_lang = _attn_row_slice(snapshot.get("raw_attn"), 0, l_start, l_end)
                    policy_lang = _attn_row_slice(snapshot.get("policy_attn"), 0, l_start, l_end)
                    row.update(_vector_stats("track_lang_raw_L{}".format(loc), raw_lang))
                    row.update(_vector_stats("track_lang_policy_L{}".format(loc), policy_lang))
                    if raw_lang is not None and policy_lang is not None:
                        row.update(_vector_stats("track_lang_delta_abs_L{}".format(loc),
                                                 (_as_tensor_1d(policy_lang) - _as_tensor_1d(raw_lang)).abs()))

            for stage_idx, loc in enumerate(lmq_locs):
                lmq_prior = _decision_stage(out_dict, "lmq_prior_scores", stage_idx)
                row.update(_signed_heat_stats("lmq_prior_L{}".format(loc), lmq_prior, gt_mask, args.top_ratio))
                row.update(_region_gap_stats("lmq_prior_L{}".format(loc), lmq_prior, gt_mask))
                row.update(_hard_negative_stats(
                    "lmq_prior_L{}".format(loc), lmq_prior, gt_mask, score_map, args.hardneg_topk))
                lmq_query_maps = _query_prior_stage(out_dict, "lmq_query_prior_maps", stage_idx)
                if lmq_query_maps is not None:
                    for query_idx in range(lmq_query_maps.shape[0]):
                        query_prior = lmq_query_maps[query_idx]
                        query_name = "lmq_query_prior_q{}_L{}".format(query_idx, loc)
                        row.update(_signed_heat_stats(query_name, query_prior, gt_mask, args.top_ratio))
                        row.update(_region_gap_stats(query_name, query_prior, gt_mask))
                        row.update(_hard_negative_stats(
                            query_name, query_prior, gt_mask, score_map, args.hardneg_topk))
                cosine_mean = _decision_stage(out_dict, "lmq_query_prior_cosine_mean", stage_idx)
                cosine_max = _decision_stage(out_dict, "lmq_query_prior_cosine_max", stage_idx)
                fusion_weights = _decision_stage(out_dict, "lmq_query_fusion_weights", stage_idx)
                row.update(_vector_stats("lmq_query_cosine_mean_L{}".format(loc), cosine_mean))
                row.update(_vector_stats("lmq_query_cosine_max_L{}".format(loc), cosine_max))
                row.update(_vector_stats("lmq_query_fusion_L{}".format(loc), fusion_weights))
                lmq_diag_names = [
                    "lmq_query_seed_cosine_mean",
                    "lmq_query_seed_cosine_max",
                    "lmq_query_lang_attn_cosine_mean",
                    "lmq_query_lang_attn_cosine_max",
                    "lmq_query_lang_attn_entropy",
                    "lmq_query_lang_attn_max",
                    "lmq_pooled_query_cosine_mean",
                    "lmq_pooled_query_cosine_max",
                    "lmq_query_vector_cosine_mean",
                    "lmq_query_vector_cosine_max",
                    "lmq_query_map_between_std",
                    "lmq_prior_score_std",
                    "lmq_query_search_attn_entropy",
                    "lmq_query_search_attn_max",
                    "lmq_decoder_query_delta_norm",
                ]
                for diag_name in lmq_diag_names:
                    diag_value = _decision_stage(out_dict, diag_name, stage_idx)
                    row.update(_vector_stats("{}_L{}".format(diag_name, loc), diag_value))

            q_rows = []
            story_delta = None
            for layer_idx, snapshot in sorted(captured["layers"], key=lambda item: item[0]):
                raw_query = _search_attn_row(snapshot.get("raw_attn"), 0, search_len)
                policy_query = _search_attn_row(snapshot.get("policy_attn"), 0, search_len)
                if raw_query is None or policy_query is None:
                    continue
                delta = _as_tensor_1d(policy_query) - _as_tensor_1d(raw_query)
                story_delta = delta
                keep_stage = None
                past_locs = [loc for loc in pruning_locs if loc <= layer_idx]
                if past_locs:
                    keep_stage = _decision_stage(out_dict, "visual_te_search_decisions", len(past_locs) - 1)
                contribution = None
                v_norm = snapshot.get("v_norm")
                if v_norm is not None:
                    contribution = _as_tensor_1d(policy_query) * _as_tensor_1d(v_norm[0, -search_len:])
                scale = _pair_positive_max(raw_query, policy_query)
                delta_scale = max(_tensor_abs_max(delta), 1e-12)
                tile_size = 142
                q_rows.append((
                    _layer_keep_label(pruning_locs, layer_idx),
                    [
                        _render_map_tile({"heat": raw_query, "mode": "fixed", "vmin": 0.0, "vmax": scale}, search_img, tile_size),
                        _render_map_tile({"heat": keep_stage, "mode": "fixed", "vmin": 0.0, "vmax": 1.0}, search_img, tile_size),
                        _render_map_tile({"heat": policy_query, "mode": "fixed", "vmin": 0.0, "vmax": scale}, search_img, tile_size),
                        _render_map_tile({"heat": delta, "mode": "signed", "limit": delta_scale}, search_img, tile_size),
                        _render_map_tile({"heat": contribution, "mode": "minmax"}, search_img, tile_size),
                    ]))
                row.update(_heat_stats("track_raw_L{}".format(layer_idx), raw_query, gt_mask, args.top_ratio))
                row.update(_heat_stats("track_policy_L{}".format(layer_idx), policy_query, gt_mask, args.top_ratio))
                row.update(_signed_heat_stats("track_delta_L{}".format(layer_idx), delta, gt_mask, args.top_ratio))

            rows.append(row)
            if frame_num <= vis_frame_limit:
                if _should_save_original_view(args):
                    _save_original_view(save_dir, frame_num, image, gt_box, pred_box, getattr(tracker, "descript", ""))
                _save_visualte_story(save_dir, frame_num, search_img, out_dict, score_map, story_delta, gt_box, pred_box,
                                     pruning_locs, search_len)
                _save_q_story(save_dir, frame_num, search_img, q_rows, gt_box, pred_box)
                _save_lte_parallel_story(save_dir, frame_num, search_img, out_dict, snapshots_by_layer, score_map,
                                         gt_box, pred_box, pruning_locs, search_len, template_len,
                                         l_len, token_labels)
                _save_track_tokens_story(save_dir, frame_num, search_img, snapshots_by_layer, out_dict,
                                         score_map, gt_box, pred_box, pruning_locs, search_len,
                                         l_len, template_len)
                _save_visual_query_story(save_dir, frame_num, search_img, snapshots_by_layer, out_dict,
                                         score_map, gt_box, pred_box, pruning_locs, search_len,
                                         template_len, l_len, gt_mask)
                _save_score_input_story(save_dir, frame_num, search_img, out_dict, off_out,
                                        score_map, off_score_map, gt_box, pred_box,
                                        tracker.network.feat_len_s)

        _write_csv(os.path.join(save_dir, "diagnostics.csv"), rows)
        _write_csv(os.path.join(save_dir, "word_reliability_diagnostics.csv"), word_rows)
        _write_summary(save_dir, rows, pruning_locs)
        print("Saved Visual TE diagnostics to {}".format(save_dir))
    finally:
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description="Visual TE diagnostic visualization.")
    parser.add_argument("--config", default="dutrack_384_full_visualte_e5")
    parser.add_argument("--dataset_name", default="otb_lang")
    parser.add_argument("--sequence", default="Biker")
    parser.add_argument("--runid", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=5)
    parser.add_argument("--stat_frames", type=int, default=None,
                        help="Number of frames to include in diagnostics. 0 means the whole sequence. Defaults to --max_frames.")
    parser.add_argument("--vis_frames", type=int, default=None,
                        help="Number of early frames to save story images for. Defaults to --max_frames.")
    parser.add_argument("--original_view", default="auto", choices=("auto", "on", "off"),
                        help="Save official-style original-frame GT/Pred visualization. auto enables it for HOOT.")
    parser.add_argument("--top_ratio", type=float, default=0.1)
    parser.add_argument("--hardneg_topk", type=int, default=6,
                        help="Number of highest-score non-GT tokens used for diagnostic hard-negative word/keep gaps.")
    parser.add_argument("--word_evidence_tau", type=float, default=0.1,
                        help="Temperature for sigmoid(word_pos_score - word_hardneg_score) word evidence diagnostics.")
    parser.add_argument("--out_dir", default="output/test/visualte_diagnostic")
    parser.add_argument("--score_prior_beta", type=float, default=None,
                        help="Runtime override for network.score_prior_beta; reuses the same config/checkpoint.")
    parser.add_argument("--score_prior_source", default=None,
                        help="Runtime override for network.score_prior_source, e.g. logits or decision.")
    parser.add_argument("--score_prior_layer", type=int, default=None,
                        help="Runtime override for network.score_prior_layer; -1 means last TE stage.")
    parser.add_argument("--language_init_source", default=None,
                        help="Runtime override for TEST.LANGUAGE_INIT_SOURCE: blip, dataset_or_blip, dataset_or_class, class_or_blip.")
    parser.add_argument("--language_update_mode", default=None,
                        help="Runtime override for TEST.LANGUAGE_UPDATE_MODE: caption_replace, anchor, off.")
    parser.add_argument("--language_word_filter", type=int, default=None,
                        help="Runtime override for TEST.LANGUAGE_WORD_FILTER_ENABLE, 0/1.")
    parser.add_argument("--language_word_filter_threshold", type=float, default=None,
                        help="Runtime override for TEST.LANGUAGE_WORD_FILTER_THRESHOLD.")
    parser.add_argument("--language_word_reliability", type=int, default=None,
                        help="Runtime override for TEST.LANGUAGE_WORD_RELIABILITY_ENABLE, 0/1.")
    parser.add_argument("--language_word_reliability_source", default=None,
                        help="Runtime reliability source: target_hardneg_gap or word_weights.")
    parser.add_argument("--language_word_reliability_momentum", type=float, default=None,
                        help="Runtime override for TEST.LANGUAGE_WORD_RELIABILITY_MOMENTUM.")
    parser.add_argument("--language_word_reliability_tau", type=float, default=None,
                        help="Runtime override for TEST.LANGUAGE_WORD_RELIABILITY_TAU.")
    parser.add_argument("--language_subject_min_reliability", type=float, default=None,
                        help="Runtime override for TEST.LANGUAGE_SUBJECT_MIN_RELIABILITY.")
    parser.add_argument("--language_context_max_weight", type=float, default=None,
                        help="Runtime override for TEST.LANGUAGE_CONTEXT_MAX_WEIGHT.")
    parser.add_argument("--language_subject_type_prior", type=float, default=None,
                        help="Runtime override for TEST.LANGUAGE_SUBJECT_TYPE_PRIOR.")
    parser.add_argument("--language_attribute_type_prior", type=float, default=None,
                        help="Runtime override for TEST.LANGUAGE_ATTRIBUTE_TYPE_PRIOR.")
    parser.add_argument("--language_context_type_prior", type=float, default=None,
                        help="Runtime override for TEST.LANGUAGE_CONTEXT_TYPE_PRIOR.")
    parser.add_argument("--language_reliability_update_gate", type=int, default=None,
                        help="Runtime override for TEST.LANGUAGE_RELIABILITY_UPDATE_GATE, 0/1.")
    parser.add_argument("--language_reliability_gate_mode", default=None,
                        help="Runtime override for TEST.LANGUAGE_RELIABILITY_GATE_MODE: score_gap, score_peak, both.")
    parser.add_argument("--language_reliability_score_thr", type=float, default=None,
                        help="Runtime override for TEST.LANGUAGE_RELIABILITY_SCORE_THR.")
    parser.add_argument("--language_reliability_score_gap_thr", type=float, default=None,
                        help="Runtime override for TEST.LANGUAGE_RELIABILITY_SCORE_GAP_THR.")
    parser.add_argument("--language_description", default="",
                        help="Override the initial sequence language description for diagnostics.")
    parser.add_argument("--output_tag", default=None,
                        help="Optional output directory name under out_dir. Defaults to config or config_betaX.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
