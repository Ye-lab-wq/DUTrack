import argparse
import csv
import os

import torch

import _init_paths
from lib.test.analysis.plot_results import check_and_load_precomputed_results
from lib.test.evaluation import get_dataset, trackerlist
from lib.test.evaluation.environment import env_settings


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


TRACKER_SPECS = [
    ("dutrack_384_full_047_updatekey", "a0_baseline"),
    ("dutrack_384_full_tec_stage1", "a1_tec_normal"),
    ("dutrack_384_full_evidence_stage2", "a2_evidence_normal"),
    ("dutrack_384_full_evidence_stage2_wrong", "a2_evidence_wrong"),
    ("dutrack_384_full_evidence_stage2_generic", "a2_evidence_generic"),
]


def _project_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _threshold_index(thresholds, target):
    return min(range(len(thresholds)), key=lambda idx: abs(float(thresholds[idx]) - target))


def _build_trackers(dataset_name):
    trackers = []
    for parameter_name, display_name in TRACKER_SPECS:
        trackers.extend(trackerlist(
            name="dutrack",
            parameter_name=parameter_name,
            dataset_name=dataset_name,
            display_name=display_name,
        ))
    return trackers


def _extract_scores(eval_data):
    overlap = torch.tensor(eval_data["ave_success_rate_plot_overlap"], dtype=torch.float32)
    center = torch.tensor(eval_data["ave_success_rate_plot_center"], dtype=torch.float32)
    center_norm = torch.tensor(eval_data["ave_success_rate_plot_center_norm"], dtype=torch.float32)

    overlap_thresholds = eval_data["threshold_set_overlap"]
    center_thresholds = eval_data["threshold_set_center"]
    center_norm_thresholds = eval_data["threshold_set_center_norm"]

    op50_idx = _threshold_index(overlap_thresholds, 0.50)
    op75_idx = _threshold_index(overlap_thresholds, 0.75)
    precision20_idx = _threshold_index(center_thresholds, 20.0)
    norm_precision_idx = _threshold_index(center_norm_thresholds, 0.50)

    return {
        "auc": overlap.mean(dim=-1) * 100.0,
        "op50": overlap[:, :, op50_idx] * 100.0,
        "op75": overlap[:, :, op75_idx] * 100.0,
        "precision20": center[:, :, precision20_idx] * 100.0,
        "norm_precision": center_norm[:, :, norm_precision_idx] * 100.0,
    }


def _best_variant(row_values):
    variants = {
        "normal": row_values["a2_evidence_normal"],
        "wrong": row_values["a2_evidence_wrong"],
        "generic": row_values["a2_evidence_generic"],
    }
    return max(variants, key=variants.get)


def _build_rows(eval_data):
    scores = _extract_scores(eval_data)
    tracker_names = [spec[1] for spec in TRACKER_SPECS]
    valid_sequence = eval_data.get("valid_sequence", [True] * len(eval_data["sequences"]))

    rows = []
    for seq_idx, sequence in enumerate(eval_data["sequences"]):
        row = {
            "sequence": sequence,
            "valid_sequence": int(bool(valid_sequence[seq_idx])),
        }

        per_metric_values = {}
        for metric_name, metric_values in scores.items():
            per_metric_values[metric_name] = {}
            for tracker_idx, tracker_name in enumerate(tracker_names):
                value = float(metric_values[seq_idx, tracker_idx].item())
                per_metric_values[metric_name][tracker_name] = value
                row["{}_{}".format(tracker_name, metric_name)] = value

        auc_values = per_metric_values["auc"]
        op75_values = per_metric_values["op75"]
        norm_values = per_metric_values["norm_precision"]
        for metric_name, values in [
                ("auc", auc_values),
                ("op75", op75_values),
                ("norm_precision", norm_values)]:
            row["a2_normal_minus_a0_{}".format(metric_name)] = (
                values["a2_evidence_normal"] - values["a0_baseline"]
            )
            row["a2_normal_minus_a1_{}".format(metric_name)] = (
                values["a2_evidence_normal"] - values["a1_tec_normal"]
            )
            row["a2_normal_minus_wrong_{}".format(metric_name)] = (
                values["a2_evidence_normal"] - values["a2_evidence_wrong"]
            )
            row["a2_normal_minus_generic_{}".format(metric_name)] = (
                values["a2_evidence_normal"] - values["a2_evidence_generic"]
            )
            row["a2_normal_minus_best_control_{}".format(metric_name)] = (
                values["a2_evidence_normal"]
                - max(values["a2_evidence_wrong"], values["a2_evidence_generic"])
            )

        row["a2_best_variant_auc"] = _best_variant(auc_values)
        row["a2_best_variant_op75"] = _best_variant(op75_values)
        row["a2_best_variant_norm_precision"] = _best_variant(norm_values)
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Force-evaluate and export Stage-2 A2 per-sequence CSV.")
    parser.add_argument("--dataset_name", type=str, default="otb_lang")
    parser.add_argument("--report_name", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Use existing eval_data.pkl if valid. Default is to force recompute.",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name
    report_name = args.report_name or "{}_stage2_a2_per_sequence_csv_eval".format(dataset_name)
    output = args.output or "output/diagnostics/stage2_a2_{}_per_sequence.csv".format(dataset_name)
    output = _project_path(output)

    trackers = _build_trackers(dataset_name)
    dataset = get_dataset(dataset_name)
    eval_data = check_and_load_precomputed_results(
        trackers,
        dataset,
        report_name,
        force_evaluation=not args.use_cache,
        skip_missing_seq=False,
        plot_bin_gap=0.05,
    )

    rows = _build_rows(eval_data)
    if not rows:
        raise RuntimeError("No per-sequence rows were generated.")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    normal_gt_a0 = sum(row["a2_normal_minus_a0_auc"] > 0 for row in rows)
    normal_gt_a1 = sum(row["a2_normal_minus_a1_auc"] > 0 for row in rows)
    normal_gt_controls = sum(row["a2_normal_minus_best_control_auc"] > 0 for row in rows)
    result_plot_path = os.path.join(env_settings().result_plot_path, report_name, "eval_data.pkl")

    print("Wrote {}".format(output))
    print("Eval cache {}".format(result_plot_path))
    print("force_evaluation={}".format(not args.use_cache))
    print("A2 normal AUC > A0: {}/{}".format(normal_gt_a0, len(rows)))
    print("A2 normal AUC > A1: {}/{}".format(normal_gt_a1, len(rows)))
    print("A2 normal AUC > max(wrong, generic): {}/{}".format(
        normal_gt_controls, len(rows)))


if __name__ == "__main__":
    main()
