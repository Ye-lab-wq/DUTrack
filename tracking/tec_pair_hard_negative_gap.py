import argparse
import csv
import os
from collections import defaultdict


DEFAULT_TRACKERS = {
    "normal": "dutrack_384_full_tec_stage1",
    "wrong": "dutrack_384_full_tec_stage1_wrong",
    "generic": "dutrack_384_full_tec_stage1_generic",
}


def _to_float(row, key):
    return float(row[key])


def _to_int(row, key):
    return int(float(row[key]))


def _optional_float(row, key):
    if key not in row or row[key] == "":
        return None
    return float(row[key])


def _summarize(values):
    values = list(values)
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "positive": 0,
            "negative": 0,
            "zero": 0,
        }
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "positive": sum(v > 0 for v in values),
        "negative": sum(v < 0 for v in values),
        "zero": sum(v == 0 for v in values),
    }


def _write_summary(path, paired_rows):
    metrics = [
        "normal_minus_wrong_gap",
        "normal_minus_generic_gap",
        "normal_minus_wrong_score_gap",
        "normal_minus_generic_score_gap",
        "normal_minus_wrong_peak_inside",
        "normal_minus_generic_peak_inside",
    ]
    optional_metrics = [
        "normal_minus_wrong_stage2_evidence_mean_gap",
        "normal_minus_generic_stage2_evidence_mean_gap",
        "normal_minus_wrong_stage2_evidence_max_gap",
        "normal_minus_generic_stage2_evidence_max_gap",
        "normal_minus_wrong_stage2_calibration_mean_gap",
        "normal_minus_generic_stage2_calibration_mean_gap",
        "normal_minus_wrong_stage2_strength_mean_gap",
        "normal_minus_generic_stage2_strength_mean_gap",
        "normal_minus_wrong_stage2r_evidence_mean_gap",
        "normal_minus_generic_stage2r_evidence_mean_gap",
        "normal_minus_wrong_stage2r_evidence_max_gap",
        "normal_minus_generic_stage2r_evidence_max_gap",
        "normal_minus_wrong_stage2r_calibration_mean_gap",
        "normal_minus_generic_stage2r_calibration_mean_gap",
        "normal_minus_wrong_stage2r_strength_mean_gap",
        "normal_minus_generic_stage2r_strength_mean_gap",
    ]
    metrics.extend(metric for metric in optional_metrics if any(metric in row for row in paired_rows))

    rows = []
    for scope, subset in [("all", paired_rows)]:
        for metric in metrics:
            summary = _summarize(float(row[metric]) for row in subset if metric in row and row[metric] != "")
            rows.append({
                "scope": scope,
                "metric": metric,
                **summary,
            })

    by_sequence = defaultdict(list)
    for row in paired_rows:
        by_sequence[row["sequence"]].append(row)
    for sequence in sorted(by_sequence):
        subset = by_sequence[sequence]
        for metric in metrics:
            summary = _summarize(float(row[metric]) for row in subset if metric in row and row[metric] != "")
            rows.append({
                "scope": sequence,
                "metric": metric,
                **summary,
            })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scope", "metric", "count", "mean", "positive", "negative", "zero"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Keep only sequence-frame keys shared by normal/wrong/generic and compute paired TEC diagnostics."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="output/diagnostics/tec_stage1_hard_negative_gap.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/diagnostics/tec_stage1_hard_negative_gap_paired.csv",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="output/diagnostics/tec_stage1_hard_negative_gap_paired_summary.csv",
    )
    parser.add_argument("--normal", type=str, default=DEFAULT_TRACKERS["normal"])
    parser.add_argument("--wrong", type=str, default=DEFAULT_TRACKERS["wrong"])
    parser.add_argument("--generic", type=str, default=DEFAULT_TRACKERS["generic"])
    args = parser.parse_args()

    tracker_map = {
        "normal": args.normal,
        "wrong": args.wrong,
        "generic": args.generic,
    }
    tracker_to_alias = {tracker: alias for alias, tracker in tracker_map.items()}

    with open(args.input, newline="") as f:
        source_rows = list(csv.DictReader(f))

    grouped = defaultdict(dict)
    tracker_counts = defaultdict(int)
    for row in source_rows:
        alias = tracker_to_alias.get(row["tracker_param"])
        if alias is None:
            continue
        key = (row["dataset"], row["sequence"], int(row["frame"]))
        grouped[key][alias] = row
        tracker_counts[alias] += 1

    paired_rows = []
    for key in sorted(grouped, key=lambda item: (item[0], item[1], item[2])):
        variants = grouped[key]
        if not all(alias in variants for alias in ("normal", "wrong", "generic")):
            continue

        dataset, sequence, frame = key
        normal = variants["normal"]
        wrong = variants["wrong"]
        generic = variants["generic"]

        normal_gap = _to_float(normal, "hard_negative_gap")
        wrong_gap = _to_float(wrong, "hard_negative_gap")
        generic_gap = _to_float(generic, "hard_negative_gap")
        normal_score_gap = _to_float(normal, "hard_negative_score_gap")
        wrong_score_gap = _to_float(wrong, "hard_negative_score_gap")
        generic_score_gap = _to_float(generic, "hard_negative_score_gap")
        normal_peak = _to_int(normal, "peak_inside_gt")
        wrong_peak = _to_int(wrong, "peak_inside_gt")
        generic_peak = _to_int(generic, "peak_inside_gt")

        paired_row = {
            "dataset": dataset,
            "sequence": sequence,
            "frame": frame,
            "normal_gap": normal_gap,
            "wrong_gap": wrong_gap,
            "generic_gap": generic_gap,
            "normal_minus_wrong_gap": normal_gap - wrong_gap,
            "normal_minus_generic_gap": normal_gap - generic_gap,
            "normal_score_gap": normal_score_gap,
            "wrong_score_gap": wrong_score_gap,
            "generic_score_gap": generic_score_gap,
            "normal_minus_wrong_score_gap": normal_score_gap - wrong_score_gap,
            "normal_minus_generic_score_gap": normal_score_gap - generic_score_gap,
            "normal_peak_inside": normal_peak,
            "wrong_peak_inside": wrong_peak,
            "generic_peak_inside": generic_peak,
            "normal_minus_wrong_peak_inside": normal_peak - wrong_peak,
            "normal_minus_generic_peak_inside": normal_peak - generic_peak,
        }

        for source_metric in [
                "stage2_evidence_mean_gap",
                "stage2_evidence_max_gap",
                "stage2_calibration_mean_gap",
                "stage2_strength_mean_gap",
                "stage2r_evidence_mean_gap",
                "stage2r_evidence_max_gap",
                "stage2r_calibration_mean_gap",
                "stage2r_strength_mean_gap"]:
            normal_value = _optional_float(normal, source_metric)
            wrong_value = _optional_float(wrong, source_metric)
            generic_value = _optional_float(generic, source_metric)
            if normal_value is None or wrong_value is None or generic_value is None:
                continue
            paired_row["normal_{}".format(source_metric)] = normal_value
            paired_row["wrong_{}".format(source_metric)] = wrong_value
            paired_row["generic_{}".format(source_metric)] = generic_value
            paired_row["normal_minus_wrong_{}".format(source_metric)] = normal_value - wrong_value
            paired_row["normal_minus_generic_{}".format(source_metric)] = normal_value - generic_value

        paired_rows.append(paired_row)

    if not paired_rows:
        raise RuntimeError("No paired sequence-frame rows found.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()))
        writer.writeheader()
        writer.writerows(paired_rows)

    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    _write_summary(args.summary, paired_rows)

    print("source_rows={}".format(len(source_rows)))
    print("tracker_rows={}".format(dict(sorted(tracker_counts.items()))))
    print("unique_sequence_frames={}".format(len(grouped)))
    print("paired_sequence_frames={}".format(len(paired_rows)))
    print("wrote {}".format(args.output))
    print("wrote {}".format(args.summary))

    for metric in [
        "normal_minus_wrong_gap",
        "normal_minus_generic_gap",
        "normal_minus_wrong_score_gap",
        "normal_minus_generic_score_gap",
        "normal_minus_wrong_peak_inside",
        "normal_minus_generic_peak_inside",
        "normal_minus_wrong_stage2_evidence_mean_gap",
        "normal_minus_generic_stage2_evidence_mean_gap",
        "normal_minus_wrong_stage2_calibration_mean_gap",
        "normal_minus_generic_stage2_calibration_mean_gap",
        "normal_minus_wrong_stage2_strength_mean_gap",
        "normal_minus_generic_stage2_strength_mean_gap",
        "normal_minus_wrong_stage2r_evidence_mean_gap",
        "normal_minus_generic_stage2r_evidence_mean_gap",
        "normal_minus_wrong_stage2r_calibration_mean_gap",
        "normal_minus_generic_stage2r_calibration_mean_gap",
        "normal_minus_wrong_stage2r_strength_mean_gap",
        "normal_minus_generic_stage2r_strength_mean_gap",
    ]:
        if not any(metric in row for row in paired_rows):
            continue
        summary = _summarize(float(row[metric]) for row in paired_rows if metric in row and row[metric] != "")
        print(
            "{}: mean={:.6f}, pos={}, neg={}, zero={}, n={}".format(
                metric,
                summary["mean"],
                summary["positive"],
                summary["negative"],
                summary["zero"],
                summary["count"],
            )
        )


if __name__ == "__main__":
    main()
