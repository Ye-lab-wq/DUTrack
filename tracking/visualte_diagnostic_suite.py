import argparse
import csv
import math
import os
import subprocess
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset


DEFAULT_CASES = (
    "otb_lang:Biker",
    "olod:0",
    "hoot_balanced20:0",
)


def _parse_case(case):
    if ":" not in case:
        raise ValueError("Case must use dataset:sequence format, got {}".format(case))
    dataset_name, sequence = case.split(":", 1)
    dataset_name = dataset_name.strip()
    sequence = sequence.strip()
    if not dataset_name or not sequence:
        raise ValueError("Case must use dataset:sequence format, got {}".format(case))
    return dataset_name, sequence


def _parse_case_description(item):
    if "=" not in item:
        raise ValueError("Case description must use dataset:sequence=description format, got {}".format(item))
    case, description = item.split("=", 1)
    dataset_name, sequence = _parse_case(case)
    description = description.strip()
    if not description:
        raise ValueError("Empty description for {}".format(case))
    return "{}:{}".format(dataset_name, sequence), description


def _description_for_case(description_map, dataset_name, sequence, sequence_name):
    return (
        description_map.get("{}:{}".format(dataset_name, sequence))
        or description_map.get("{}:{}".format(dataset_name, sequence_name))
        or ""
    )


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
    lang_rel = getattr(args, "language_word_reliability", None)
    if lang_rel is not None:
        parts.append("wordrel{}".format(_safe_tag(lang_rel)))
    return "{}_{}".format(config, "_".join(parts)) if parts else config


def _resolve_sequence_name(dataset_name, sequence):
    dataset = get_dataset(dataset_name)
    seq = dataset[int(sequence)] if str(sequence).isdigit() else dataset[sequence]
    return seq.name


def _float(row, key):
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _mean(rows, key):
    values = [_float(row, key) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _sum(rows, key):
    values = [_float(row, key) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) if values else None


def _unique_count(rows, key):
    values = [str(row.get(key, "")) for row in rows if str(row.get(key, "")) != ""]
    return len(set(values)) if values else 0


def _iou_xywh(pred, gt):
    px, py, pw, ph = pred
    gx, gy, gw, gh = gt
    px2, py2 = px + pw, py + ph
    gx2, gy2 = gx + gw, gy + gh
    ix1, iy1 = max(px, gx), max(py, gy)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = pw * ph + gw * gh - inter
    return inter / union if union > 0 else 0.0


def _mean_iou(rows):
    values = []
    for row in rows:
        pred = [_float(row, key) for key in ("pred_x", "pred_y", "pred_w", "pred_h")]
        gt = [_float(row, key) for key in ("gt_x", "gt_y", "gt_w", "gt_h")]
        if any(value is None for value in pred + gt):
            continue
        values.append(_iou_xywh(pred, gt))
    return sum(values) / len(values) if values else None


def _max_layer_mean(rows, prefix, suffix):
    values = []
    if not rows:
        return None
    for key in rows[0].keys():
        if key.startswith(prefix) and key.endswith(suffix):
            value = _mean(rows, key)
            if value is not None:
                values.append(value)
    return max(values) if values else None


def _min_layer_mean(rows, prefix, suffix):
    values = []
    if not rows:
        return None
    for key in rows[0].keys():
        if key.startswith(prefix) and key.endswith(suffix):
            value = _mean(rows, key)
            if value is not None:
                values.append(value)
    return min(values) if values else None


def _summarize_diagnostics(dataset_name, sequence_name, diagnostics_path):
    with open(diagnostics_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        "dataset": dataset_name,
        "sequence": sequence_name,
        "frames": len(rows),
        "mean_iou": _mean_iou(rows),
        "score_map_mass_in_gt": _mean(rows, "score_map_mass_in_gt"),
        "score_map_top10_precision": _mean(rows, "score_map_top10_precision"),
        "score_onoff_peak_delta": _mean(rows, "score_onoff_peak_delta"),
        "score_onoff_delta_abs_sum": _mean(rows, "score_onoff_delta_abs_sum"),
        "head_input_norm_onoff_delta_abs_sum": _mean(rows, "head_input_norm_onoff_delta_abs_sum"),
        "language_update_requested_count": _sum(rows, "language_update_requested"),
        "language_changed_count": _sum(rows, "language_changed"),
        "language_unique_count": _unique_count(rows, "language_description"),
        "language_source_unique_count": _unique_count(rows, "language_source"),
        "language_anchor_unique_count": _unique_count(rows, "language_anchor"),
        "language_filtered_unique_count": _unique_count(rows, "language_filtered_description"),
        "language_word_filter_active_count": _sum(rows, "language_word_filter_active"),
        "language_word_reliability_active_count": _sum(rows, "language_word_reliability_active"),
        "language_reliability_update_rate": (
            _sum(rows, "language_word_reliability_updated") / float(len(rows)) if rows else None),
        "mean_reliability_delta": _mean(rows, "language_word_reliability_delta"),
        "score_onoff_language_mismatch_count": _sum(rows, "score_onoff_language_mismatch"),
        "word_direct_gap_max": _max_layer_mean(rows, "word_direct_L", "_gap_in_minus_out"),
        "word_direct_hardneg_gap_max": _max_layer_mean(rows, "word_direct_L", "_pos_hardneg_gap"),
        "word_direct_hard_case_ratio_min": _min_layer_mean(rows, "word_direct_L", "_hard_case"),
        "word_evidence_oracle_mean_gap_max": _max_layer_mean(rows, "word_evidence_oracle_L", "_mean_gap"),
        "word_evidence_oracle_hard_case_ratio_min": _min_layer_mean(rows, "word_evidence_oracle_L", "_hard_case_ratio"),
        "word_evidence_oracle_subject_gap_mean_max": _max_layer_mean(rows, "word_evidence_oracle_L", "_subject_gap_mean"),
        "word_evidence_oracle_anchor_subject_gap_mean_max": _max_layer_mean(rows, "word_evidence_oracle_L", "_anchor_subject_gap_mean"),
        "word_evidence_oracle_positive_ratio_max": _max_layer_mean(rows, "word_evidence_oracle_L", "_content_word_positive_ratio"),
        "word_evidence_oracle_rank_corr_max": _max_layer_mean(rows, "word_evidence_oracle_L", "_weight_gap_rank_corr"),
        "word_evidence_oracle_top3_overlap_max": _max_layer_mean(rows, "word_evidence_oracle_L", "_top3_weight_gap_overlap"),
        "word_evidence_deploy_mean_gap_max": _max_layer_mean(rows, "word_evidence_deploy_L", "_mean_gap"),
        "word_evidence_deploy_hard_case_ratio_min": _min_layer_mean(rows, "word_evidence_deploy_L", "_hard_case_ratio"),
        "word_evidence_deploy_subject_gap_mean_max": _max_layer_mean(rows, "word_evidence_deploy_L", "_subject_gap_mean"),
        "word_evidence_deploy_anchor_subject_gap_mean_max": _max_layer_mean(rows, "word_evidence_deploy_L", "_anchor_subject_gap_mean"),
        "word_evidence_deploy_positive_ratio_max": _max_layer_mean(rows, "word_evidence_deploy_L", "_content_word_positive_ratio"),
        "word_evidence_deploy_rank_corr_max": _max_layer_mean(rows, "word_evidence_deploy_L", "_weight_gap_rank_corr"),
        "word_evidence_deploy_top3_overlap_max": _max_layer_mean(rows, "word_evidence_deploy_L", "_top3_weight_gap_overlap"),
        "search_keep_hardneg_gap_max": _max_layer_mean(rows, "search_keep_L", "_pos_hardneg_gap"),
        "safe_margin_gap_max": _max_layer_mean(rows, "safe_proto_margin_L", "_gap_in_minus_out"),
        "lmq_prior_gap_max": _max_layer_mean(rows, "lmq_prior_L", "_gap_in_minus_out"),
        "lmq_prior_hardneg_gap_max": _max_layer_mean(rows, "lmq_prior_L", "_pos_hardneg_gap"),
        "lmq_query_prior_gap_max": _max_layer_mean(rows, "lmq_query_prior_q", "_pos_hardneg_gap"),
        "lmq_query_cosine_mean_max": _max_layer_mean(rows, "lmq_query_cosine_mean_L", "_mean"),
        "lmq_query_cosine_max_max": _max_layer_mean(rows, "lmq_query_cosine_max_L", "_mean"),
        "lmq_seed_cosine_mean_max": _max_layer_mean(rows, "lmq_query_seed_cosine_mean_L", "_mean"),
        "lmq_lang_attn_cosine_mean_max": _max_layer_mean(rows, "lmq_query_lang_attn_cosine_mean_L", "_mean"),
        "lmq_lang_attn_entropy_max": _max_layer_mean(rows, "lmq_query_lang_attn_entropy_L", "_mean"),
        "lmq_lang_attn_max_max": _max_layer_mean(rows, "lmq_query_lang_attn_max_L", "_mean"),
        "lmq_pooled_query_cosine_mean_max": _max_layer_mean(rows, "lmq_pooled_query_cosine_mean_L", "_mean"),
        "lmq_query_vector_cosine_mean_max": _max_layer_mean(rows, "lmq_query_vector_cosine_mean_L", "_mean"),
        "lmq_query_map_between_std_max": _max_layer_mean(rows, "lmq_query_map_between_std_L", "_mean"),
        "lmq_prior_score_std_max": _max_layer_mean(rows, "lmq_prior_score_std_L", "_mean"),
        "lmq_search_attn_entropy_max": _max_layer_mean(rows, "lmq_query_search_attn_entropy_L", "_mean"),
        "lmq_search_attn_max_max": _max_layer_mean(rows, "lmq_query_search_attn_max_L", "_mean"),
        "lmq_decoder_delta_norm_max": _max_layer_mean(rows, "lmq_decoder_query_delta_norm_L", "_mean"),
        "score_prior_bias_abs_mean": _mean(rows, "score_prior_bias_abs_mean"),
        "score_prior_bias_max": _mean(rows, "score_prior_bias_max"),
        "score_prior_bias_clamp_ratio": _mean(rows, "score_prior_bias_clamp_ratio"),
        "score_logits_base_abs_mean": _mean(rows, "score_logits_base_abs_mean"),
        "score_prior_to_base_abs_ratio": _mean(rows, "score_prior_to_base_abs_ratio"),
        "q0_l1_raw_policy_max": _max_layer_mean(rows, "q0_full_L", "_l1_raw_policy"),
        "q0_search_gt_delta_max": _max_layer_mean(rows, "q0_full_L", "_search_gt_abs_mass_delta"),
    }


def _format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return "{:.6g}".format(value)
    return str(value)


def _write_suite_summary(out_dir, config, run_label, summaries,
                         score_prior_beta=None, score_prior_source=None, score_prior_layer=None,
                         stat_frames=None, vis_frames=None):
    suite_dir = os.path.join(out_dir, run_label)
    os.makedirs(suite_dir, exist_ok=True)
    csv_path = os.path.join(suite_dir, "suite_summary.csv")
    fieldnames = [
        "dataset",
        "sequence",
        "frames",
        "mean_iou",
        "score_map_mass_in_gt",
        "score_map_top10_precision",
        "score_onoff_peak_delta",
        "score_onoff_delta_abs_sum",
        "head_input_norm_onoff_delta_abs_sum",
        "language_update_requested_count",
        "language_changed_count",
        "language_unique_count",
        "language_source_unique_count",
        "language_anchor_unique_count",
        "language_filtered_unique_count",
        "language_word_filter_active_count",
        "language_word_reliability_active_count",
        "language_reliability_update_rate",
        "mean_reliability_delta",
        "score_onoff_language_mismatch_count",
        "word_direct_gap_max",
        "word_direct_hardneg_gap_max",
        "word_direct_hard_case_ratio_min",
        "word_evidence_oracle_mean_gap_max",
        "word_evidence_oracle_hard_case_ratio_min",
        "word_evidence_oracle_subject_gap_mean_max",
        "word_evidence_oracle_anchor_subject_gap_mean_max",
        "word_evidence_oracle_positive_ratio_max",
        "word_evidence_oracle_rank_corr_max",
        "word_evidence_oracle_top3_overlap_max",
        "word_evidence_deploy_mean_gap_max",
        "word_evidence_deploy_hard_case_ratio_min",
        "word_evidence_deploy_subject_gap_mean_max",
        "word_evidence_deploy_anchor_subject_gap_mean_max",
        "word_evidence_deploy_positive_ratio_max",
        "word_evidence_deploy_rank_corr_max",
        "word_evidence_deploy_top3_overlap_max",
        "search_keep_hardneg_gap_max",
        "safe_margin_gap_max",
        "lmq_prior_gap_max",
        "lmq_prior_hardneg_gap_max",
        "lmq_query_prior_gap_max",
        "lmq_query_cosine_mean_max",
        "lmq_query_cosine_max_max",
        "lmq_seed_cosine_mean_max",
        "lmq_lang_attn_cosine_mean_max",
        "lmq_lang_attn_entropy_max",
        "lmq_lang_attn_max_max",
        "lmq_pooled_query_cosine_mean_max",
        "lmq_query_vector_cosine_mean_max",
        "lmq_query_map_between_std_max",
        "lmq_prior_score_std_max",
        "lmq_search_attn_entropy_max",
        "lmq_search_attn_max_max",
        "lmq_decoder_delta_norm_max",
        "score_prior_bias_abs_mean",
        "score_prior_bias_max",
        "score_prior_bias_clamp_ratio",
        "score_logits_base_abs_mean",
        "score_prior_to_base_abs_ratio",
        "q0_l1_raw_policy_max",
        "q0_search_gt_delta_max",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary)

    md_path = os.path.join(suite_dir, "suite_summary.md")
    lines = [
        "# Visual TE Suite Summary",
        "",
        "Config: `{}`".format(config),
        "Output label: `{}`".format(run_label),
    ]
    if score_prior_beta is not None:
        lines.append("Runtime SCORE_PRIOR_BETA: `{:.6g}`".format(float(score_prior_beta)))
    if score_prior_source is not None:
        lines.append("Runtime SCORE_PRIOR_SOURCE: `{}`".format(score_prior_source))
    if score_prior_layer is not None:
        lines.append("Runtime SCORE_PRIOR_LAYER: `{}`".format(score_prior_layer))
    if stat_frames is not None:
        lines.append("Diagnostic stat frames: `{}`".format(stat_frames))
    if vis_frames is not None:
        lines.append("Story image frames: `{}`".format(vis_frames))
    lines.extend([
        "",
        "| Dataset | Sequence | Frames | IoU | Score GT Mass | On/Off Peak Delta | LMQ Gap | LMQ HardNeg Gap | Best Query Gap | Query Cos Mean | Prior/Base | Clamp | Lang Changes | Unique Lang |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for summary in summaries:
        lines.append(
            "| {dataset} | {sequence} | {frames} | {mean_iou} | {score_map_mass_in_gt} | "
            "{score_onoff_peak_delta} | {lmq_prior_gap_max} | {lmq_prior_hardneg_gap_max} | "
            "{lmq_query_prior_gap_max} | {lmq_query_cosine_mean_max} | "
            "{score_prior_to_base_abs_ratio} | {score_prior_bias_clamp_ratio} | "
            "{language_changed_count} | {language_unique_count} |".format(
                **{key: _format_value(value) for key, value in summary.items()}
            )
        )
    lines.extend([
        "",
        "Per-case diagnostics are under `{}` grouped by dataset name.".format(out_dir),
        "",
    ])
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return csv_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Run Visual TE diagnostics on a small fixed sequence suite.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--runid", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=5)
    parser.add_argument("--stat_frames", type=int, default=None,
                        help="Number of frames to include in diagnostics. 0 means the whole sequence. Defaults to --max_frames.")
    parser.add_argument("--vis_frames", type=int, default=None,
                        help="Number of early frames to save story images for. Defaults to --max_frames.")
    parser.add_argument("--original_view", default="auto", choices=("auto", "on", "off"),
                        help="Forward original-frame visualization mode to visualte_diagnostic.py.")
    parser.add_argument("--top_ratio", type=float, default=0.1)
    parser.add_argument("--hardneg_topk", type=int, default=6,
                        help="Number of highest-score non-GT tokens used for diagnostic hard-negative gaps.")
    parser.add_argument("--word_evidence_tau", type=float, default=0.1,
                        help="Temperature for per-word visual evidence diagnostics.")
    parser.add_argument("--out_dir", default="output/test/visualte_diagnostic_suite")
    parser.add_argument("--score_prior_beta", type=float, default=None,
                        help="Runtime override forwarded to visualte_diagnostic.py.")
    parser.add_argument("--score_prior_source", default=None,
                        help="Runtime score-prior source override, e.g. logits or decision.")
    parser.add_argument("--score_prior_layer", type=int, default=None,
                        help="Runtime score-prior TE layer override, e.g. 15; -1 means last stage.")
    parser.add_argument("--language_init_source", default=None,
                        help="Runtime language init override forwarded to visualte_diagnostic.py.")
    parser.add_argument("--language_update_mode", default=None,
                        help="Runtime language update override forwarded to visualte_diagnostic.py.")
    parser.add_argument("--language_word_filter", type=int, default=None,
                        help="Runtime word-filter override forwarded to visualte_diagnostic.py, 0/1.")
    parser.add_argument("--language_word_filter_threshold", type=float, default=None,
                        help="Runtime word-filter threshold override forwarded to visualte_diagnostic.py.")
    parser.add_argument("--language_word_reliability", type=int, default=None,
                        help="Runtime soft word-reliability override forwarded to visualte_diagnostic.py, 0/1.")
    parser.add_argument("--language_word_reliability_source", default=None,
                        help="Runtime reliability source override: target_hardneg_gap or word_weights.")
    parser.add_argument("--language_word_reliability_momentum", type=float, default=None,
                        help="Runtime reliability momentum override.")
    parser.add_argument("--language_word_reliability_tau", type=float, default=None,
                        help="Runtime reliability tau override.")
    parser.add_argument("--language_subject_min_reliability", type=float, default=None,
                        help="Runtime subject floor override.")
    parser.add_argument("--language_context_max_weight", type=float, default=None,
                        help="Runtime context cap override.")
    parser.add_argument("--language_subject_type_prior", type=float, default=None,
                        help="Runtime subject soft type prior override.")
    parser.add_argument("--language_attribute_type_prior", type=float, default=None,
                        help="Runtime attribute soft type prior override.")
    parser.add_argument("--language_context_type_prior", type=float, default=None,
                        help="Runtime context soft type prior override.")
    parser.add_argument("--language_reliability_update_gate", type=int, default=None,
                        help="Runtime reliability update gate override, 0/1.")
    parser.add_argument("--language_reliability_gate_mode", default=None,
                        help="Runtime reliability gate mode: score_gap, score_peak, both.")
    parser.add_argument("--language_reliability_score_thr", type=float, default=None,
                        help="Runtime reliability score peak threshold.")
    parser.add_argument("--language_reliability_score_gap_thr", type=float, default=None,
                        help="Runtime reliability score_peak - hardneg_peak threshold.")
    parser.add_argument("--output_tag", default=None,
                        help="Optional output directory label. Defaults to config or config_betaX.")
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="Dataset and sequence in dataset:sequence format. Repeat to override the default suite.",
    )
    parser.add_argument(
        "--case_description",
        action="append",
        default=None,
        help="Initial language override in dataset:sequence=description format. Repeat for multiple cases.",
    )
    args = parser.parse_args()

    cases = args.case if args.case is not None else list(DEFAULT_CASES)
    description_map = dict(_parse_case_description(item) for item in (args.case_description or []))
    summaries = []
    run_label = _run_label(args.config, args)
    script_path = os.path.join(os.path.dirname(__file__), "visualte_diagnostic.py")
    for case in cases:
        dataset_name, sequence = _parse_case(case)
        sequence_name = _resolve_sequence_name(dataset_name, sequence)
        case_out_dir = os.path.join(args.out_dir, dataset_name)
        cmd = [
            sys.executable,
            script_path,
            "--config",
            args.config,
            "--dataset_name",
            dataset_name,
            "--sequence",
            sequence,
            "--runid",
            str(args.runid),
            "--max_frames",
            str(args.max_frames),
            "--top_ratio",
            str(args.top_ratio),
            "--hardneg_topk",
            str(args.hardneg_topk),
            "--word_evidence_tau",
            str(args.word_evidence_tau),
            "--out_dir",
            case_out_dir,
        ]
        if args.stat_frames is not None:
            cmd.extend(["--stat_frames", str(args.stat_frames)])
        if args.vis_frames is not None:
            cmd.extend(["--vis_frames", str(args.vis_frames)])
        if args.original_view is not None:
            cmd.extend(["--original_view", args.original_view])
        if args.score_prior_beta is not None:
            cmd.extend(["--score_prior_beta", str(args.score_prior_beta)])
        if args.score_prior_source is not None:
            cmd.extend(["--score_prior_source", args.score_prior_source])
        if args.score_prior_layer is not None:
            cmd.extend(["--score_prior_layer", str(args.score_prior_layer)])
        if args.language_init_source is not None:
            cmd.extend(["--language_init_source", args.language_init_source])
        if args.language_update_mode is not None:
            cmd.extend(["--language_update_mode", args.language_update_mode])
        if args.language_word_filter is not None:
            cmd.extend(["--language_word_filter", str(args.language_word_filter)])
        if args.language_word_filter_threshold is not None:
            cmd.extend(["--language_word_filter_threshold", str(args.language_word_filter_threshold)])
        if args.language_word_reliability is not None:
            cmd.extend(["--language_word_reliability", str(args.language_word_reliability)])
        if args.language_word_reliability_source is not None:
            cmd.extend(["--language_word_reliability_source", args.language_word_reliability_source])
        if args.language_word_reliability_momentum is not None:
            cmd.extend(["--language_word_reliability_momentum", str(args.language_word_reliability_momentum)])
        if args.language_word_reliability_tau is not None:
            cmd.extend(["--language_word_reliability_tau", str(args.language_word_reliability_tau)])
        if args.language_subject_min_reliability is not None:
            cmd.extend(["--language_subject_min_reliability", str(args.language_subject_min_reliability)])
        if args.language_context_max_weight is not None:
            cmd.extend(["--language_context_max_weight", str(args.language_context_max_weight)])
        if args.language_subject_type_prior is not None:
            cmd.extend(["--language_subject_type_prior", str(args.language_subject_type_prior)])
        if args.language_attribute_type_prior is not None:
            cmd.extend(["--language_attribute_type_prior", str(args.language_attribute_type_prior)])
        if args.language_context_type_prior is not None:
            cmd.extend(["--language_context_type_prior", str(args.language_context_type_prior)])
        if args.language_reliability_update_gate is not None:
            cmd.extend(["--language_reliability_update_gate", str(args.language_reliability_update_gate)])
        if args.language_reliability_gate_mode is not None:
            cmd.extend(["--language_reliability_gate_mode", args.language_reliability_gate_mode])
        if args.language_reliability_score_thr is not None:
            cmd.extend(["--language_reliability_score_thr", str(args.language_reliability_score_thr)])
        if args.language_reliability_score_gap_thr is not None:
            cmd.extend(["--language_reliability_score_gap_thr", str(args.language_reliability_score_gap_thr)])
        description = _description_for_case(description_map, dataset_name, sequence, sequence_name)
        if description:
            cmd.extend(["--language_description", description])
        if args.output_tag:
            cmd.extend(["--output_tag", args.output_tag])
        print("Running {}:{} ({})".format(dataset_name, sequence, sequence_name), flush=True)
        subprocess.run(cmd, check=True)
        diagnostics_path = os.path.join(case_out_dir, run_label, sequence_name, "diagnostics.csv")
        summaries.append(_summarize_diagnostics(dataset_name, sequence_name, diagnostics_path))

    csv_path, md_path = _write_suite_summary(
        args.out_dir, args.config, run_label, summaries,
        args.score_prior_beta, args.score_prior_source, args.score_prior_layer,
        args.stat_frames, args.vis_frames)
    print("Saved suite summary to {}".format(md_path))
    print("Saved suite metrics to {}".format(csv_path))


if __name__ == "__main__":
    main()
