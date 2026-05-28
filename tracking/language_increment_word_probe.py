import argparse
import csv
import math
import os
import sys

import torch

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker
from lib.train.data.processing_utils import sample_target
from tracking.language_state_s0_probe import (
    CONTEXT_WORDS,
    _bbox_iou,
    _candidate_description,
    _content_word_list,
    _forward_description,
    _gap_with_mask,
    _mean,
    _predict_box,
    _selected_template_mask,
    _score_from_output,
    _search_crop_box,
    _source_stats,
    _split_tokens_for_word_evidence,
    _token_words,
    _unit,
)
from tracking.visualte_diagnostic import _feat_size, _read_rgb, _token_box_mask


def _safe_tag(text):
    return str(text).lower().replace("-", "m").replace(".", "p").replace("/", "_")


def _float_or_nan(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _unique_new_words(candidate, base, max_words):
    base_words = set(_content_word_list(base))
    words = []
    for word in _content_word_list(candidate):
        if word in base_words or word in words:
            continue
        words.append(word)
        if len(words) >= int(max_words):
            break
    return words


def _word_role(word):
    if word in CONTEXT_WORDS:
        return "context"
    if word in {
        "red", "green", "blue", "black", "white", "yellow", "brown", "gray", "grey",
        "small", "large", "big", "tiny", "dark", "bright", "striped", "round",
    }:
        return "attribute"
    return "content"


def _append_word(base, word, mode):
    base = " ".join(str(base or "").split())
    word = " ".join(str(word or "").split())
    if not base:
        return word
    if not word:
        return base
    if mode == "space":
        return "{} {}".format(base, word)
    if mode == "with":
        return "{} with {}".format(base, word)
    if mode == "target_is":
        return "the target is {} with {}".format(base, word)
    raise ValueError("Unsupported prompt_mode: {}".format(mode))


def _word_token_index(tracker, description, word, device):
    words, valid, content = _token_words(tracker, description, device)
    target = "".join(ch for ch in str(word).lower() if ch.isalnum())
    if not target:
        return None
    matches = [
        idx for idx, token in enumerate(words)
        if token == target and bool(valid[idx].item()) and bool(content[idx].item())
    ]
    if matches:
        return matches[-1]
    return None


def _hard_negative_mask(score_ref, pos_mask, topk):
    if not isinstance(score_ref, torch.Tensor) or score_ref.numel() == 0:
        return None
    pos = pos_mask.to(device=score_ref.device).bool().view(-1)
    score = score_ref.to(device=score_ref.device).view(-1)
    if pos.numel() != score.numel():
        return None
    neg = ~pos
    if not neg.any():
        return None
    k = max(1, min(int(topk), int(neg.sum().item())))
    neg_indices = neg.nonzero(as_tuple=False).view(-1)
    hard_local = torch.topk(score[neg_indices], k).indices
    hard_indices = neg_indices[hard_local]
    mask = torch.zeros_like(pos)
    mask[hard_indices] = True
    return mask


def _similarity_gap(sim, pos_mask, neg_mask):
    if not isinstance(sim, torch.Tensor):
        return float("nan"), float("nan"), float("nan")
    pos = pos_mask.to(device=sim.device).bool().view(-1)
    neg = neg_mask.to(device=sim.device).bool().view(-1)
    if sim.numel() != pos.numel() or sim.numel() != neg.numel() or not pos.any() or not neg.any():
        return float("nan"), float("nan"), float("nan")
    pos_mean = sim[pos].float().mean().item()
    neg_mean = sim[neg].float().mean().item()
    return pos_mean, neg_mean, pos_mean - neg_mean


def _word_visual_consistency(tracker, out, description, word, template_mask, pred_mask, score_ref, hardneg_topk):
    empty = {
        "word_token_found": 0.0,
        "word_template_target_sim": float("nan"),
        "word_template_bg_sim": float("nan"),
        "word_template_gap": float("nan"),
        "word_search_pred_sim": float("nan"),
        "word_search_hardneg_sim": float("nan"),
        "word_search_hardneg_gap": float("nan"),
    }
    try:
        lang_tokens, template_tokens, search_tokens = _split_tokens_for_word_evidence(
            tracker, out, template_mask)
    except Exception:
        return empty
    device = lang_tokens.device
    token_idx = _word_token_index(tracker, description, word, device)
    if token_idx is None or token_idx >= lang_tokens.shape[1]:
        return empty

    word_token = _unit(lang_tokens[:, token_idx:token_idx + 1, :])
    sim_z = torch.matmul(_unit(template_tokens), word_token.transpose(1, 2)).view(-1)
    sim_x = torch.matmul(_unit(search_tokens), word_token.transpose(1, 2)).view(-1)

    out = dict(empty)
    out["word_token_found"] = 1.0
    template_mask = template_mask.to(device=device).bool().view(-1) if template_mask is not None else None
    if template_mask is not None and template_mask.numel() == sim_z.numel():
        template_pos, template_neg, template_gap = _similarity_gap(sim_z, template_mask, ~template_mask)
        out["word_template_target_sim"] = template_pos
        out["word_template_bg_sim"] = template_neg
        out["word_template_gap"] = template_gap

    pred_mask = pred_mask.to(device=device).bool().view(-1) if pred_mask is not None else None
    hardneg_mask = _hard_negative_mask(score_ref.to(device=device), pred_mask, hardneg_topk) if pred_mask is not None else None
    if pred_mask is not None and hardneg_mask is not None and pred_mask.numel() == sim_x.numel():
        pred_sim, hardneg_sim, search_gap = _similarity_gap(sim_x, pred_mask, hardneg_mask)
        out["word_search_pred_sim"] = pred_sim
        out["word_search_hardneg_sim"] = hardneg_sim
        out["word_search_hardneg_gap"] = search_gap
    return out


def _selected_indices(dataset, args):
    if args.sequence_indices:
        indices = []
        for part in args.sequence_indices.split(","):
            part = part.strip()
            if part:
                indices.append(int(part))
        return indices
    if args.sequence_names:
        wanted = {name.strip() for name in args.sequence_names.split(",") if name.strip()}
        return [idx for idx, seq in enumerate(dataset) if seq.name in wanted]
    indices = list(range(len(dataset)))
    if args.max_sequences > 0:
        indices = indices[:args.max_sequences]
    return indices


def _write_summary(save_dir, args, rows):
    positive = [r for r in rows if _float_or_nan(r.get("word_label_useful")) > 0.5]
    harmful = [r for r in rows if _float_or_nan(r.get("word_label_harmful")) > 0.5]
    by_role = {}
    for row in rows:
        role = row.get("word_role", "unknown")
        by_role.setdefault(role, []).append(row)

    lines = [
        "# Stage 3 Word Increment Weak-Label Probe",
        "",
        "Config: `{}`".format(args.config),
        "Datasets: `{}`".format(args.dataset_names),
        "Candidate mode: `{}`".format(args.candidate_mode),
        "Base source: `{}`".format(args.base_source),
        "Prompt mode: `{}`".format(args.prompt_mode),
        "Rows: `{}`".format(len(rows)),
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        "| word useful ratio | {:.6g} |".format(len(positive) / len(rows) if rows else float("nan")),
        "| word harmful ratio | {:.6g} |".format(len(harmful) / len(rows) if rows else float("nan")),
        "| mean word gain | {:.6g} |".format(_mean(rows, "word_gain_over_base")),
        "| mean deploy word gain | {:.6g} |".format(_mean(rows, "word_deploy_gain_over_base")),
        "| mean BLIP hard gain | {:.6g} |".format(_mean(rows, "blip_gain_over_base")),
        "| mean word-template gap | {:.6g} |".format(_mean(rows, "word_template_gap")),
        "| mean word-search hardneg gap | {:.6g} |".format(_mean(rows, "word_search_hardneg_gap")),
        "",
        "## By Role",
        "",
        "| Role | Rows | useful ratio | harmful ratio | mean gain | mean deploy gain | template gap | search hardneg gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for role in sorted(by_role):
        role_rows = by_role[role]
        useful = [r for r in role_rows if _float_or_nan(r.get("word_label_useful")) > 0.5]
        bad = [r for r in role_rows if _float_or_nan(r.get("word_label_harmful")) > 0.5]
        lines.append("| {} | {} | {:.6g} | {:.6g} | {:.6g} | {:.6g} | {:.6g} | {:.6g} |".format(
            role,
            len(role_rows),
            len(useful) / len(role_rows) if role_rows else float("nan"),
            len(bad) / len(role_rows) if role_rows else float("nan"),
            _mean(role_rows, "word_gain_over_base"),
            _mean(role_rows, "word_deploy_gain_over_base"),
            _mean(role_rows, "word_template_gap"),
            _mean(role_rows, "word_search_hardneg_gap"),
        ))

    lines.extend([
        "",
        "## Definition",
        "",
        "- A row is one candidate BLIP word appended to the base language state.",
        "- `word_gain_over_base = gap(base + word) - gap(base)` uses GT/oracle target region.",
        "- `word_deploy_gain_over_base` uses the predicted-box region and is deploy-like.",
        "- `word_template_gap` compares the appended word token against selected template target tokens versus template background tokens in fused DUTrack token space.",
        "- `word_search_hardneg_gap` compares the appended word token against predicted search tokens versus base-score hard negatives in fused DUTrack token space.",
        "- `word_label_useful = word_gain_over_base > gap_eps` is a weak label for later Word ReliabilityNet experiments.",
        "- This script is diagnostic only; it does not update tracker weights or change normal tracking.",
    ])
    with open(os.path.join(save_dir, "word_increment_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _run_sequence(args, dataset_name, seq_idx, seq):
    tracker_info = Tracker("dutrack", args.config, dataset_name, args.runid)
    params = tracker_info.get_parameters(run_id=args.runid)
    params.debug = 0
    tracker = tracker_info.create_tracker(params)
    tracker.cfg.TEST.LANGUAGE_UPDATE_MODE = "anchor"

    saved_score_prior_enabled = getattr(tracker.network, "score_prior_enabled", None)
    if saved_score_prior_enabled is not None and not args.use_score_prior_effect:
        tracker.network.score_prior_enabled = False

    image0 = _read_rgb(seq.frames[0])
    init_info = seq.init_info()
    init_info["class"] = seq.object_class
    init_info["path"] = seq.name
    tracker.initialize(image0, init_info)

    search_len = int(tracker.network.backbone.pos_embed_x.shape[1])
    search_feat_sz = _feat_size(search_len)
    prev_description = str(getattr(tracker, "language_anchor", ""))
    requested_frames = int(args.max_frames)
    max_frame = len(seq.frames) - 1 if requested_frames <= 0 else min(requested_frames, len(seq.frames) - 1)
    rows = []

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
        template_target_mask = _selected_template_mask(tracker, template_list)

        anchor_description = str(getattr(tracker, "language_anchor", ""))
        base_description = anchor_description if args.base_source == "anchor" else (prev_description or anchor_description)
        deploy_trigger = bool(getattr(tracker, "updata_key", False))
        blip_description, candidate_available = _candidate_description(
            tracker, image, seq.object_class, args.candidate_mode, deploy_trigger, anchor_description)

        crop_box = _search_crop_box(tracker.state, resize_factor, params.search_size)
        gt_box = seq.ground_truth_rect[frame_num].tolist() if seq.ground_truth_rect is not None else None
        gt_mask = _token_box_mask(gt_box, crop_box, search_feat_sz) if gt_box is not None else torch.zeros(search_len, dtype=torch.bool)

        base_out = _forward_description(tracker, template_list, search.tensors, base_description)
        base_score = _score_from_output(base_out)
        base_stats = _source_stats("base", base_score, gt_mask, base_score, args.top_ratio, args.hardneg_topk)
        base_gap = _float_or_nan(base_stats.get("base_pos_hardneg_gap"))
        base_pred_box = _predict_box(tracker, base_out, resize_factor)
        base_iou = _bbox_iou(base_pred_box, gt_box, search.tensors.device)
        pred_mask = _token_box_mask(base_pred_box, crop_box, search_feat_sz) if base_pred_box is not None else None
        base_deploy_gap = (
            _gap_with_mask(base_score, pred_mask, base_score, args.hardneg_topk)
            if pred_mask is not None else float("nan")
        )

        blip_gap = float("nan")
        if candidate_available and blip_description:
            blip_out = _forward_description(tracker, template_list, search.tensors, blip_description)
            blip_score = _score_from_output(blip_out)
            blip_stats = _source_stats("blip", blip_score, gt_mask, base_score, args.top_ratio, args.hardneg_topk)
            blip_gap = _float_or_nan(blip_stats.get("blip_pos_hardneg_gap"))

        words = _unique_new_words(blip_description, base_description, args.max_words_per_frame) if candidate_available else []
        for rank, word in enumerate(words):
            word_description = _append_word(base_description, word, args.prompt_mode)
            word_out = _forward_description(tracker, template_list, search.tensors, word_description)
            word_score = _score_from_output(word_out)
            visual_consistency = _word_visual_consistency(
                tracker,
                word_out,
                word_description,
                word,
                template_target_mask,
                pred_mask,
                base_score,
                args.hardneg_topk,
            )
            word_stats = _source_stats("word", word_score, gt_mask, base_score, args.top_ratio, args.hardneg_topk)
            word_gap = _float_or_nan(word_stats.get("word_pos_hardneg_gap"))
            word_gain = word_gap - base_gap if math.isfinite(word_gap) and math.isfinite(base_gap) else float("nan")
            word_deploy_gap = (
                _gap_with_mask(word_score, pred_mask, base_score, args.hardneg_topk)
                if pred_mask is not None else float("nan")
            )
            word_deploy_gain = (
                word_deploy_gap - base_deploy_gap
                if math.isfinite(word_deploy_gap) and math.isfinite(base_deploy_gap)
                else float("nan")
            )
            word_pred_box = _predict_box(tracker, word_out, resize_factor)
            word_iou = _bbox_iou(word_pred_box, gt_box, search.tensors.device)

            rows.append({
                "dataset": dataset_name,
                "sequence": seq.name,
                "frame": frame_num,
                "word": word,
                "word_rank": rank,
                "word_role": _word_role(word),
                "candidate_available": 1.0 if candidate_available else 0.0,
                "deploy_trigger": 1.0 if deploy_trigger else 0.0,
                "anchor_description": anchor_description,
                "base_description": base_description,
                "blip_description": blip_description,
                "word_description": word_description,
                "base_gap": base_gap,
                "blip_gap": blip_gap,
                "blip_gain_over_base": blip_gap - base_gap if math.isfinite(blip_gap) and math.isfinite(base_gap) else float("nan"),
                "word_gap": word_gap,
                "word_gain_over_base": word_gain,
                "base_deploy_gap": base_deploy_gap,
                "word_deploy_gap": word_deploy_gap,
                "word_deploy_gain_over_base": word_deploy_gain,
                "word_label_useful": 1.0 if math.isfinite(word_gain) and word_gain > float(args.gap_eps) else 0.0,
                "word_label_harmful": 1.0 if math.isfinite(word_gain) and word_gain < -float(args.gap_eps) else 0.0,
                "base_iou": base_iou,
                "word_iou": word_iou,
                "word_iou_gain_over_base": word_iou - base_iou if math.isfinite(word_iou) and math.isfinite(base_iou) else float("nan"),
                "base_score_peak": base_score.max().item() if isinstance(base_score, torch.Tensor) and base_score.numel() else float("nan"),
                "word_score_peak": word_score.max().item() if isinstance(word_score, torch.Tensor) and word_score.numel() else float("nan"),
                **visual_consistency,
            })

        tracker.track(image, info)

    return rows


def run(args):
    save_dir = os.path.join(args.out_dir, args.output_tag)
    os.makedirs(save_dir, exist_ok=True)
    all_rows = []
    for dataset_name in [name.strip() for name in args.dataset_names.split(",") if name.strip()]:
        dataset = get_dataset(dataset_name)
        for seq_idx in _selected_indices(dataset, args):
            seq = dataset[seq_idx]
            print("[word-increment] {}:{} ({}/{})".format(dataset_name, seq.name, seq_idx + 1, len(dataset)))
            all_rows.extend(_run_sequence(args, dataset_name, seq_idx, seq))

    csv_path = os.path.join(save_dir, "word_increment_probe.csv")
    fieldnames = sorted(set().union(*(row.keys() for row in all_rows))) if all_rows else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    _write_summary(save_dir, args, all_rows)
    print("Saved word increment probe to {}".format(save_dir))
    return save_dir, all_rows


def main():
    parser = argparse.ArgumentParser(description="Probe per-word BLIP semantic increments for weak labels.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_names", default="otb_lang")
    parser.add_argument("--sequence_names", default="")
    parser.add_argument("--sequence_indices", default="")
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--runid", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=50)
    parser.add_argument("--top_ratio", type=float, default=0.1)
    parser.add_argument("--hardneg_topk", type=int, default=6)
    parser.add_argument("--candidate_mode", default="deploy_like", choices=("deploy_like", "oracle_blip", "off"))
    parser.add_argument("--base_source", default="prev", choices=("prev", "anchor"))
    parser.add_argument("--prompt_mode", default="space", choices=("space", "with", "target_is"))
    parser.add_argument("--max_words_per_frame", type=int, default=8)
    parser.add_argument("--gap_eps", type=float, default=0.0)
    parser.add_argument("--use_score_prior_effect", action="store_true")
    parser.add_argument("--output_tag", default="stage3_word_increment_probe")
    parser.add_argument("--out_dir", default="output/test/language_increment_word_probe")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
