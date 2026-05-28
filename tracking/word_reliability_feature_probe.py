import argparse
import csv
import math
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


ROLE_NAMES = ["attribute", "content", "context", "unknown"]

NUMERIC_FEATURES = [
    "word_rank",
    "candidate_available",
    "deploy_trigger",
    "base_deploy_gap",
    "word_deploy_gap",
    "word_deploy_gain_over_base",
    "base_score_peak",
    "word_score_peak",
    "score_peak_delta",
    "word_token_found",
    "word_template_target_sim",
    "word_template_bg_sim",
    "word_template_gap",
    "word_search_pred_sim",
    "word_search_hardneg_sim",
    "word_search_hardneg_gap",
    "word_length",
    "word_has_digit",
    "word_is_alpha",
]


def _safe_float(row, key, default=float("nan")):
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _word_hash_features(word, dim):
    features = [0.0] * dim
    if dim <= 0:
        return features
    word = str(word or "").lower()
    if not word:
        return features
    grams = [word]
    grams.extend(word[i:i + 2] for i in range(max(0, len(word) - 1)))
    grams.extend(word[i:i + 3] for i in range(max(0, len(word) - 2)))
    for gram in grams:
        idx = hash(gram) % dim
        features[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in features))
    if norm > 0:
        features = [v / norm for v in features]
    return features


def _role_features(role):
    role = str(role or "unknown")
    if role not in ROLE_NAMES:
        role = "unknown"
    return [1.0 if role == name else 0.0 for name in ROLE_NAMES]


def _load_examples(paths, args):
    examples = []
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if args.sequence_names:
                    wanted = {name.strip() for name in args.sequence_names.split(",") if name.strip()}
                    if row.get("sequence") not in wanted:
                        continue
                label_value = _safe_float(row, "word_gain_over_base")
                deploy_gain = _safe_float(row, "word_deploy_gain_over_base")
                if not math.isfinite(label_value) or not math.isfinite(deploy_gain):
                    continue
                if label_value > float(args.label_eps):
                    label = 1.0
                elif label_value < -float(args.label_eps):
                    label = 0.0
                else:
                    continue
                word = row.get("word", "")
                base_peak = _safe_float(row, "base_score_peak", 0.0)
                word_peak = _safe_float(row, "word_score_peak", 0.0)
                feature_values = {
                    "word_rank": _safe_float(row, "word_rank", 0.0),
                    "candidate_available": _safe_float(row, "candidate_available", 0.0),
                    "deploy_trigger": _safe_float(row, "deploy_trigger", 0.0),
                    "base_deploy_gap": _safe_float(row, "base_deploy_gap", 0.0),
                    "word_deploy_gap": _safe_float(row, "word_deploy_gap", 0.0),
                    "word_deploy_gain_over_base": deploy_gain,
                    "base_score_peak": base_peak,
                    "word_score_peak": word_peak,
                    "score_peak_delta": word_peak - base_peak,
                    "word_token_found": _safe_float(row, "word_token_found", 0.0),
                    "word_template_target_sim": _safe_float(row, "word_template_target_sim", 0.0),
                    "word_template_bg_sim": _safe_float(row, "word_template_bg_sim", 0.0),
                    "word_template_gap": _safe_float(row, "word_template_gap", 0.0),
                    "word_search_pred_sim": _safe_float(row, "word_search_pred_sim", 0.0),
                    "word_search_hardneg_sim": _safe_float(row, "word_search_hardneg_sim", 0.0),
                    "word_search_hardneg_gap": _safe_float(row, "word_search_hardneg_gap", 0.0),
                    "word_length": float(len(str(word))),
                    "word_has_digit": 1.0 if any(ch.isdigit() for ch in str(word)) else 0.0,
                    "word_is_alpha": 1.0 if str(word).isalpha() else 0.0,
                }
                examples.append({
                    "dataset": row.get("dataset", ""),
                    "sequence": row.get("sequence", ""),
                    "frame": row.get("frame", ""),
                    "word": word,
                    "word_role": row.get("word_role", "unknown"),
                    "label": label,
                    "oracle_gain": label_value,
                    "deploy_gain": deploy_gain,
                    "features": feature_values,
                })
    return examples


def _feature_names(args):
    names = list(NUMERIC_FEATURES)
    names.extend("role_{}".format(role) for role in ROLE_NAMES)
    names.extend("hash_{}".format(i) for i in range(int(args.hash_dim)))
    return names


def _vectorize_example(example, args):
    values = [float(example["features"].get(name, 0.0)) for name in NUMERIC_FEATURES]
    values.extend(_role_features(example.get("word_role", "unknown")))
    values.extend(_word_hash_features(example.get("word", ""), int(args.hash_dim)))
    return values


def _tensorize(examples, args):
    if not examples:
        return None, None
    xs = [_vectorize_example(example, args) for example in examples]
    ys = [float(example["label"]) for example in examples]
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32).view(-1, 1)


def _standardize(train_x, test_x):
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (train_x - mean) / std, (test_x - mean) / std


class TinyMLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        if hidden <= 0:
            self.net = nn.Linear(dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(hidden, 1),
            )

    def forward(self, x):
        return self.net(x)


def _train_model(train_x, train_y, args):
    torch.manual_seed(int(args.seed))
    model = TinyMLP(train_x.shape[1], int(args.hidden_dim))
    pos = train_y.sum().item()
    neg = train_y.numel() - pos
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(int(args.epochs)):
        logits = model(train_x)
        loss = F.binary_cross_entropy_with_logits(logits, train_y, pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def _metrics(pred, labels, oracle_gains):
    pred = [int(v > 0.5) for v in pred]
    labels = [int(v > 0.5) for v in labels]
    tp = sum(1 for p, y in zip(pred, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(pred, labels) if p == 1 and y == 0)
    tn = sum(1 for p, y in zip(pred, labels) if p == 0 and y == 0)
    fn = sum(1 for p, y in zip(pred, labels) if p == 0 and y == 1)
    accepted = tp + fp
    useful = tp + fn
    harmful = tn + fp
    gains = [gain if p == 1 else 0.0 for p, gain in zip(pred, oracle_gains)]
    return {
        "count": len(labels),
        "accept_rate": accepted / len(labels) if labels else float("nan"),
        "accept_precision": tp / accepted if accepted else float("nan"),
        "useful_recall": tp / useful if useful else float("nan"),
        "hurt_rejection": tn / harmful if harmful else float("nan"),
        "false_accept_rate_harmful": fp / harmful if harmful else float("nan"),
        "mean_gain": sum(gains) / len(gains) if gains else float("nan"),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _evaluate_loso(examples, args):
    by_seq = defaultdict(list)
    for example in examples:
        by_seq[(example["dataset"], example["sequence"])].append(example)

    metric_rows = []
    prediction_rows = []
    for key, test_examples in sorted(by_seq.items()):
        train_examples = [
            example for seq_key, seq_examples in by_seq.items()
            if seq_key != key
            for example in seq_examples
        ]
        if len(train_examples) < int(args.min_train_examples):
            continue
        train_x, train_y = _tensorize(train_examples, args)
        test_x, test_y = _tensorize(test_examples, args)
        train_x, test_x = _standardize(train_x, test_x)
        model = _train_model(train_x, train_y, args)
        with torch.no_grad():
            prob = torch.sigmoid(model(test_x)).view(-1).tolist()
        learned_pred = [1.0 if value >= float(args.threshold) else 0.0 for value in prob]
        deploy_pred = [1.0 if example["deploy_gain"] > float(args.deploy_gain_thr) else 0.0 for example in test_examples]
        labels = [example["label"] for example in test_examples]
        gains = [example["oracle_gain"] for example in test_examples]
        learned = _metrics(learned_pred, labels, gains)
        deploy = _metrics(deploy_pred, labels, gains)
        row = {
            "dataset": key[0],
            "sequence": key[1],
            "count": len(test_examples),
        }
        for prefix, metrics in (("learned", learned), ("deploy_gain", deploy)):
            for name, value in metrics.items():
                row["{}_{}".format(prefix, name)] = value
        metric_rows.append(row)
        for example, p, lp, dp in zip(test_examples, prob, labels, deploy_pred):
            prediction_rows.append({
                "dataset": example["dataset"],
                "sequence": example["sequence"],
                "frame": example["frame"],
                "word": example["word"],
                "word_role": example["word_role"],
                "label": lp,
                "oracle_gain": example["oracle_gain"],
                "deploy_gain": example["deploy_gain"],
                "learned_prob": p,
                "learned_accept": 1.0 if p >= float(args.threshold) else 0.0,
                "deploy_gain_accept": dp,
            })
    return metric_rows, prediction_rows


def _weighted_mean(rows, key):
    total = 0.0
    weight = 0.0
    for row in rows:
        try:
            value = float(row.get(key, "nan"))
            count = float(row.get("count", "nan"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and math.isfinite(count) and count > 0:
            total += value * count
            weight += count
    return total / weight if weight > 0 else float("nan")


def _write_outputs(save_dir, args, feature_names, metric_rows, prediction_rows):
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, "word_reliability_loso_metrics.csv")
    fieldnames = sorted(set().union(*(row.keys() for row in metric_rows))) if metric_rows else []
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metric_rows:
            writer.writerow(row)

    pred_path = os.path.join(save_dir, "word_reliability_loso_predictions.csv")
    fieldnames = sorted(set().union(*(row.keys() for row in prediction_rows))) if prediction_rows else []
    with open(pred_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in prediction_rows:
            writer.writerow(row)

    lines = [
        "# Stage 3 Word Reliability Feature Probe",
        "",
        "Inputs: `{}`".format(args.inputs),
        "Feature count: `{}`".format(len(feature_names)),
        "Model hidden dim: `{}`".format(args.hidden_dim),
        "Rows: `{}`".format(sum(int(row.get("count", 0)) for row in metric_rows)),
        "",
        "## Weighted LOSO Metrics",
        "",
        "| Method | accept precision | useful recall | hurt rejection | false accept/harmful | mean gain | accept rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for prefix in ("deploy_gain", "learned"):
        lines.append("| {} | {:.6g} | {:.6g} | {:.6g} | {:.6g} | {:.6g} | {:.6g} |".format(
            prefix,
            _weighted_mean(metric_rows, "{}_accept_precision".format(prefix)),
            _weighted_mean(metric_rows, "{}_useful_recall".format(prefix)),
            _weighted_mean(metric_rows, "{}_hurt_rejection".format(prefix)),
            _weighted_mean(metric_rows, "{}_false_accept_rate_harmful".format(prefix)),
            _weighted_mean(metric_rows, "{}_mean_gain".format(prefix)),
            _weighted_mean(metric_rows, "{}_accept_rate".format(prefix)),
        ))
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `deploy_gain` is the simple baseline: accept a word when `word_deploy_gain_over_base > threshold`.",
        "- `learned` is a leave-one-sequence-out lightweight classifier over deploy-available numeric, role, and hashed word features.",
        "- If `learned` does not beat `deploy_gain`, the current feature set is not enough to justify a learned reliability module.",
    ])
    with open(os.path.join(save_dir, "word_reliability_feature_probe_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run(args):
    paths = [path.strip() for path in args.inputs.split(",") if path.strip()]
    examples = _load_examples(paths, args)
    feature_names = _feature_names(args)
    metric_rows, prediction_rows = _evaluate_loso(examples, args)
    save_dir = os.path.join(args.out_dir, args.output_tag)
    _write_outputs(save_dir, args, feature_names, metric_rows, prediction_rows)
    print("Saved word reliability feature probe to {}".format(save_dir))
    return save_dir, metric_rows, prediction_rows


def main():
    parser = argparse.ArgumentParser(description="Offline lightweight Word ReliabilityNet probe.")
    parser.add_argument("--inputs", required=True,
                        help="Comma-separated word_increment_probe.csv files.")
    parser.add_argument("--sequence_names", default="")
    parser.add_argument("--label_eps", type=float, default=0.0)
    parser.add_argument("--deploy_gain_thr", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hash_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=0,
                        help="0 uses logistic regression; >0 uses a tiny MLP.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_train_examples", type=int, default=50)
    parser.add_argument("--output_tag", default="stage3_word_reliability_probe")
    parser.add_argument("--out_dir", default="output/test/word_reliability_feature_probe")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
