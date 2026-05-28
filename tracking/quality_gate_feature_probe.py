import argparse
import csv
import math
import os
import random
import re
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


STOP_WORDS = {
    "a", "an", "the", "of", "on", "in", "at", "by", "with", "and", "or", "to", "from",
    "is", "are", "was", "were", "this", "that", "there", "as", "for", "near", "over",
    "under", "beside", "behind", "front", "back", "left", "right", "top", "bottom",
}

GENERIC_WORDS = {
    "object", "thing", "item", "scene", "area", "picture", "image", "photo", "something",
    "someone", "stuff", "view",
}

CONTEXT_WORDS = {
    "road", "street", "tree", "grass", "sky", "ground", "water", "wall", "floor", "field",
    "person", "people", "man", "woman", "hand", "hands", "room", "table", "chair", "building",
    "background", "foreground", "car", "truck", "bike", "bicycle", "motorcycle",
}


BASE_FEATURES = [
    "quality_gate_deploy_score_delta",
    "quality_gate_semantic",
    "quality_gate_semantic_anchor",
    "quality_gate_semantic_prev",
    "quality_gate_confidence_ok",
    "score_peak",
    "score_second_peak",
    "score_peak_second_gap",
    "pred_box_jump_ratio",
    "trigger_by_position",
    "trigger_by_scale",
    "trigger_by_color",
    "trigger_area_ratio",
    "trigger_center_distance",
    "trigger_color_delta",
    "anchor_score_gap",
    "prev_score_gap",
    "blip_score_gap",
    "prev_deploy_gap",
    "blip_deploy_gap",
]

TEXT_FEATURES = [
    "anchor_blip_content_overlap",
    "prev_blip_content_overlap",
    "anchor_blip_content_jaccard",
    "prev_blip_content_jaccard",
    "blip_content_word_count",
    "blip_generic_word_count",
    "blip_generic_ratio",
    "blip_context_word_count",
    "blip_context_ratio",
    "blip_anchor_missing_ratio",
    "blip_prev_missing_ratio",
]

WORD_EVIDENCE_FEATURES = [
    "blip_word_target_word_overlap_count",
    "blip_word_target_word_missing_flag",
    "blip_word_context_dominance",
    "blip_word_target_template_gap_mean",
    "blip_word_context_template_gap_mean",
    "blip_word_new_template_gap_mean",
    "blip_word_target_search_deploy_gap_mean",
    "blip_word_context_search_deploy_gap_mean",
    "blip_word_new_search_deploy_gap_mean",
    "blip_word_target_minus_context_template_gap",
    "blip_word_target_minus_context_search_deploy_gap",
    "blip_minus_prev_target_template_gap",
    "blip_minus_prev_target_search_deploy_gap",
]


def _safe_float(row, key, default=float("nan")):
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _words(text):
    words = []
    for token in re.split(r"[\s_\-]+", str(text).lower()):
        clean = re.sub(r"[^a-z0-9]+", "", token)
        if clean and clean not in STOP_WORDS:
            words.append(clean)
    return words


def _text_features(row):
    anchor = set(_words(row.get("anchor_description", "")))
    prev = set(_words(row.get("prev_description", "")))
    blip_words = _words(row.get("blip_description", ""))
    blip = set(blip_words)
    content_count = len(blip_words)
    generic_count = sum(1 for word in blip_words if word in GENERIC_WORDS)
    context_count = sum(1 for word in blip_words if word in CONTEXT_WORDS)
    anchor_inter = len(anchor & blip)
    prev_inter = len(prev & blip)
    anchor_union = len(anchor | blip)
    prev_union = len(prev | blip)
    return {
        "anchor_blip_content_overlap": float(anchor_inter),
        "prev_blip_content_overlap": float(prev_inter),
        "anchor_blip_content_jaccard": float(anchor_inter) / anchor_union if anchor_union else 0.0,
        "prev_blip_content_jaccard": float(prev_inter) / prev_union if prev_union else 0.0,
        "blip_content_word_count": float(content_count),
        "blip_generic_word_count": float(generic_count),
        "blip_generic_ratio": float(generic_count) / content_count if content_count else 0.0,
        "blip_context_word_count": float(context_count),
        "blip_context_ratio": float(context_count) / content_count if content_count else 0.0,
        "blip_anchor_missing_ratio": float(len(anchor - blip)) / len(anchor) if anchor else 0.0,
        "blip_prev_missing_ratio": float(len(prev - blip)) / len(prev) if prev else 0.0,
    }


def _load_examples(args):
    wanted = {name.strip() for name in args.sequence_names.split(",") if name.strip()}
    examples = []
    with open(args.error_report, newline="") as f:
        for row in csv.DictReader(f):
            if wanted and row.get("sequence") not in wanted:
                continue
            oracle_delta = _safe_float(row, "quality_gate_score_delta")
            deploy_delta = _safe_float(row, "quality_gate_deploy_score_delta")
            if not math.isfinite(oracle_delta) or not math.isfinite(deploy_delta):
                continue
            if oracle_delta > args.label_eps:
                label = 1.0
            elif oracle_delta < -args.label_eps:
                label = 0.0
            else:
                continue
            feature_values = {}
            for key in BASE_FEATURES:
                feature_values[key] = _safe_float(row, key, 0.0)
            feature_values.update(_text_features(row))
            for key in WORD_EVIDENCE_FEATURES:
                feature_values[key] = _safe_float(row, key, 0.0)
            current_accept = 1.0 if _safe_float(row, "quality_gate_accept", 0.0) > 0.5 else 0.0
            examples.append({
                "dataset": row.get("dataset", ""),
                "sequence": row.get("sequence", ""),
                "frame": row.get("frame", ""),
                "label": label,
                "current_accept": current_accept,
                "oracle_delta": oracle_delta,
                "deploy_delta": deploy_delta,
                "features": feature_values,
            })
    return examples


def _feature_names(args):
    if args.feature_set == "base":
        return BASE_FEATURES
    if args.feature_set == "text":
        return TEXT_FEATURES
    if args.feature_set == "word":
        return BASE_FEATURES + WORD_EVIDENCE_FEATURES
    return BASE_FEATURES + TEXT_FEATURES + WORD_EVIDENCE_FEATURES


def _tensorize(examples, feature_names):
    xs = []
    ys = []
    for ex in examples:
        xs.append([float(ex["features"].get(name, 0.0)) for name in feature_names])
        ys.append(float(ex["label"]))
    if not xs:
        return None, None
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32).view(-1, 1)


def _standardize(train_x, test_x):
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (train_x - mean) / std, (test_x - mean) / std


def _train_logistic(train_x, train_y, args):
    model = nn.Linear(train_x.shape[1], 1)
    pos = train_y.sum().item()
    neg = train_y.numel() - pos
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(args.epochs):
        logits = model(train_x)
        loss = F.binary_cross_entropy_with_logits(logits, train_y, pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def _metrics(pred, label, oracle_delta):
    pred = [int(v > 0.5) for v in pred]
    label = [int(v > 0.5) for v in label]
    tp = sum(1 for p, y in zip(pred, label) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(pred, label) if p == 1 and y == 0)
    tn = sum(1 for p, y in zip(pred, label) if p == 0 and y == 0)
    fn = sum(1 for p, y in zip(pred, label) if p == 0 and y == 1)
    accepted = tp + fp
    useful = tp + fn
    harmful = tn + fp
    gains = [delta if p == 1 else 0.0 for p, delta in zip(pred, oracle_delta)]
    return {
        "count": len(label),
        "accept_rate": accepted / len(label) if label else float("nan"),
        "accept_precision": tp / accepted if accepted else float("nan"),
        "useful_recall": tp / useful if useful else float("nan"),
        "hurt_rejection": tn / harmful if harmful else float("nan"),
        "false_accept_rate_harmful": fp / harmful if harmful else float("nan"),
        "mean_gate_gain": sum(gains) / len(gains) if gains else float("nan"),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _evaluate_loso(examples, feature_names, args):
    by_seq = defaultdict(list)
    for ex in examples:
        by_seq[(ex["dataset"], ex["sequence"])].append(ex)
    rows = []
    predictions = []
    for (dataset, sequence), test_examples in sorted(by_seq.items()):
        train_examples = [
            ex for key, seq_examples in by_seq.items()
            if key != (dataset, sequence)
            for ex in seq_examples
        ]
        if len(train_examples) < args.min_train_examples:
            continue
        train_x, train_y = _tensorize(train_examples, feature_names)
        test_x, test_y = _tensorize(test_examples, feature_names)
        train_x, test_x = _standardize(train_x, test_x)
        model = _train_logistic(train_x, train_y, args)
        with torch.no_grad():
            prob = torch.sigmoid(model(test_x)).view(-1).tolist()
        learned_pred = [1.0 if p >= args.threshold else 0.0 for p in prob]
        current_pred = [ex["current_accept"] for ex in test_examples]
        labels = [ex["label"] for ex in test_examples]
        deltas = [ex["oracle_delta"] for ex in test_examples]
        learned = _metrics(learned_pred, labels, deltas)
        current = _metrics(current_pred, labels, deltas)
        row = {
            "dataset": dataset,
            "sequence": sequence,
            "frames": len(test_examples),
            "current_accept_precision": current["accept_precision"],
            "learned_accept_precision": learned["accept_precision"],
            "current_useful_recall": current["useful_recall"],
            "learned_useful_recall": learned["useful_recall"],
            "current_hurt_rejection": current["hurt_rejection"],
            "learned_hurt_rejection": learned["hurt_rejection"],
            "current_false_accept_rate_harmful": current["false_accept_rate_harmful"],
            "learned_false_accept_rate_harmful": learned["false_accept_rate_harmful"],
            "current_mean_gate_gain": current["mean_gate_gain"],
            "learned_mean_gate_gain": learned["mean_gate_gain"],
            "current_accept_rate": current["accept_rate"],
            "learned_accept_rate": learned["accept_rate"],
            "current_tp": current["tp"],
            "current_fp": current["fp"],
            "current_tn": current["tn"],
            "current_fn": current["fn"],
            "learned_tp": learned["tp"],
            "learned_fp": learned["fp"],
            "learned_tn": learned["tn"],
            "learned_fn": learned["fn"],
        }
        rows.append(row)
        for ex, p, pred in zip(test_examples, prob, learned_pred):
            predictions.append({
                "dataset": dataset,
                "sequence": sequence,
                "frame": ex["frame"],
                "label": ex["label"],
                "current_accept": ex["current_accept"],
                "learned_prob": p,
                "learned_accept": pred,
                "oracle_delta": ex["oracle_delta"],
                "deploy_delta": ex["deploy_delta"],
            })
    return rows, predictions


def _aggregate(rows, prefix):
    totals = defaultdict(float)
    count = 0
    for row in rows:
        count += 1
        for key, value in row.items():
            if key in ("dataset", "sequence"):
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                totals[key] += value
    out = {}
    for key, total in totals.items():
        out[prefix + key] = total / max(count, 1)
    return out


def _write_outputs(save_dir, args, rows, predictions, feature_names):
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, "loso_metrics.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    pred_path = os.path.join(save_dir, "loso_predictions.csv")
    if predictions:
        fieldnames = list(predictions[0].keys())
        with open(pred_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(predictions)

    agg = _aggregate(rows, "")
    lines = [
        "# Stage 3 Quality Gate Feature Probe",
        "",
        "This is an offline separability diagnostic, not the final online gate.",
        "",
        "Error report: `{}`".format(args.error_report),
        "Sequences: `{}`".format(args.sequence_names),
        "Feature set: `{}`".format(args.feature_set),
        "Features: `{}`".format(", ".join(feature_names)),
        "",
        "## LOSO Mean Metrics",
        "",
        "| Metric | Current gate | Learned gate |",
        "| --- | ---: | ---: |",
    ]
    pairs = [
        ("accept precision", "current_accept_precision", "learned_accept_precision"),
        ("useful recall", "current_useful_recall", "learned_useful_recall"),
        ("hurt rejection", "current_hurt_rejection", "learned_hurt_rejection"),
        ("false accept / harmful", "current_false_accept_rate_harmful", "learned_false_accept_rate_harmful"),
        ("mean gate gain", "current_mean_gate_gain", "learned_mean_gate_gain"),
        ("accept rate", "current_accept_rate", "learned_accept_rate"),
    ]
    for label, cur, learned in pairs:
        lines.append("| {} | {:.6g} | {:.6g} |".format(label, agg.get(cur, float("nan")), agg.get(learned, float("nan"))))
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `current gate` is the deployed score-delta gate already present in the error report.",
        "- `learned gate` is a leave-one-sequence-out logistic probe trained only on other selected sequences.",
        "- A useful feature set should reduce false accept without collapsing useful recall or mean gate gain.",
        "- If learned metrics do not consistently beat current gate, the available deploy features are not yet stable enough for an online learnable gate.",
    ])
    with open(os.path.join(save_dir, "feature_probe_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    examples = _load_examples(args)
    feature_names = _feature_names(args)
    rows, predictions = _evaluate_loso(examples, feature_names, args)
    save_dir = os.path.join(args.out_dir, args.output_tag)
    _write_outputs(save_dir, args, rows, predictions, feature_names)
    print("Loaded {} labeled examples from {}.".format(len(examples), args.error_report))
    print("Saved quality gate feature probe to {}".format(save_dir))


def main():
    parser = argparse.ArgumentParser(description="Offline LOSO feature probe for Stage 3 quality gate.")
    parser.add_argument(
        "--error_report",
        default="output/test/language_state_s0_screen/stage3_quality_gate_otb_hoot_error_report/s0_error_report.csv")
    parser.add_argument(
        "--sequence_names",
        default="Bird1,Dog,Gym,potted_plant-008,toilet_paper-001,koala-003")
    parser.add_argument("--feature_set", default="all", choices=("base", "text", "word", "all"))
    parser.add_argument("--label_eps", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--min_train_examples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output_tag", default="stage3_gate_feature_probe_loso")
    parser.add_argument("--out_dir", default="output/test/language_state_s0_gate_probe")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
