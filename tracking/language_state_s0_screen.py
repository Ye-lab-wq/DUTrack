import argparse
import csv
import math
import os
import sys
from argparse import Namespace

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from tracking.language_state_s0_probe import run as run_s0_probe


def _safe_tag(text):
    return str(text).lower().replace("-", "m").replace(".", "p").replace("/", "_")


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


def _ratio(rows, key, predicate):
    values = []
    for row in rows:
        try:
            value = float(row.get(key, "nan"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(1.0 if predicate(value) else 0.0)
    return sum(values) / len(values) if values else float("nan")


def _class_labels(summary, args):
    labels = []
    mean_iou = summary["baseline_mean_iou"]
    if math.isfinite(mean_iou) and args.class_a_iou_low <= mean_iou <= args.class_a_iou_high:
        labels.append("A_mid_difficulty")
    if (
        math.isfinite(summary["oracle_score_gain"])
        and summary["oracle_score_gain"] > args.class_b_gain_thr
        and math.isfinite(summary["BLIP_better_anchor_ratio"])
        and summary["BLIP_better_anchor_ratio"] >= args.class_b_blip_better_thr
    ):
        labels.append("B_language_gain")
    if (
        math.isfinite(summary["BLIP_hurts_ratio"])
        and summary["BLIP_hurts_ratio"] >= args.class_c_hurts_thr
    ):
        labels.append("C_language_hurts")
    return ",".join(labels) if labels else "none"


def _summarize_sequence(dataset_name, sequence_name, rows, args):
    true_accept = _mean(rows, "quality_gate_true_accept")
    false_reject = _mean(rows, "quality_gate_false_reject")
    true_reject = _mean(rows, "quality_gate_true_reject")
    false_accept = _mean(rows, "quality_gate_false_accept")
    partial_current_true_accept = _mean(rows, "partial_current_gate_true_accept")
    partial_current_false_reject = _mean(rows, "partial_current_gate_false_reject")
    partial_current_true_reject = _mean(rows, "partial_current_gate_true_reject")
    partial_current_false_accept = _mean(rows, "partial_current_gate_false_accept")
    partial_gate_true_accept = _mean(rows, "partial_gate_true_accept")
    partial_gate_false_reject = _mean(rows, "partial_gate_false_reject")
    partial_gate_true_reject = _mean(rows, "partial_gate_true_reject")
    partial_gate_false_accept = _mean(rows, "partial_gate_false_accept")
    useful_den = true_accept + false_reject if math.isfinite(true_accept) and math.isfinite(false_reject) else float("nan")
    hurt_den = true_reject + false_accept if math.isfinite(true_reject) and math.isfinite(false_accept) else float("nan")
    partial_current_useful_den = (
        partial_current_true_accept + partial_current_false_reject
        if math.isfinite(partial_current_true_accept) and math.isfinite(partial_current_false_reject)
        else float("nan")
    )
    partial_current_hurt_den = (
        partial_current_true_reject + partial_current_false_accept
        if math.isfinite(partial_current_true_reject) and math.isfinite(partial_current_false_accept)
        else float("nan")
    )
    partial_gate_useful_den = (
        partial_gate_true_accept + partial_gate_false_reject
        if math.isfinite(partial_gate_true_accept) and math.isfinite(partial_gate_false_reject)
        else float("nan")
    )
    partial_gate_hurt_den = (
        partial_gate_true_reject + partial_gate_false_accept
        if math.isfinite(partial_gate_true_reject) and math.isfinite(partial_gate_false_accept)
        else float("nan")
    )
    summary = {
        "dataset": dataset_name,
        "sequence": sequence_name,
        "frames": len(rows),
        "baseline_mean_iou": _mean(rows, "anchor_iou"),
        "score_gap_mean": _mean(rows, "anchor_score_pos_hardneg_gap"),
        "score_gap_low_ratio": _ratio(
            rows, "anchor_score_pos_hardneg_gap",
            lambda value: value <= args.score_gap_low_thr),
        "deploy_trigger_rate": _mean(rows, "deploy_trigger"),
        "trigger_by_position_rate": _mean(rows, "trigger_by_position"),
        "trigger_by_scale_rate": _mean(rows, "trigger_by_scale"),
        "trigger_by_color_rate": _mean(rows, "trigger_by_color"),
        "trigger_color_delta_mean": _mean(rows, "trigger_color_delta"),
        "BLIP_available_rate": _mean(rows, "candidate_available"),
        "anchor_score_gap": _mean(rows, "anchor_score_pos_hardneg_gap"),
        "blip_score_gap": _mean(rows, "blip_score_pos_hardneg_gap"),
        "anchor_delta_score_gap": _mean(rows, "anchor_delta_score_pos_hardneg_gap"),
        "prev_delta_score_gap": _mean(rows, "prev_delta_score_pos_hardneg_gap"),
        "oracle_score_gap": _mean(rows, "oracle_gap"),
        "oracle_score_gain": _mean(rows, "oracle_gain_over_prev"),
        "hard_replace_gain": _mean(rows, "hard_replace_gain_over_prev"),
        "anchor_delta_gain": _mean(rows, "anchor_delta_gain_over_prev"),
        "prev_delta_gain": _mean(rows, "prev_delta_gain_over_prev"),
        "best_partial_gain": _mean(rows, "best_partial_gain_over_prev"),
        "deploy_best_partial_gain": _mean(rows, "deploy_best_partial_gain_over_prev"),
        "partial_beats_hard_replace_ratio": _mean(rows, "partial_beats_hard_replace"),
        "partial_useful_when_blip_hurts_ratio": _mean(rows, "partial_useful_when_blip_hurts"),
        "partial_label_useful_rate": _mean(rows, "partial_label_useful"),
        "partial_label_harmful_rate": _mean(rows, "partial_label_harmful"),
        "partial_current_gate_accept_rate": _mean(rows, "quality_gate_accept"),
        "partial_current_gate_gain": _mean(rows, "partial_current_gate_gain_over_prev"),
        "partial_current_gate_true_accept_rate": partial_current_true_accept,
        "partial_current_gate_false_reject_rate": partial_current_false_reject,
        "partial_current_gate_true_reject_rate": partial_current_true_reject,
        "partial_current_gate_false_accept_rate": partial_current_false_accept,
        "partial_current_useful_update_recall": (
            partial_current_true_accept / partial_current_useful_den
            if math.isfinite(partial_current_useful_den) and partial_current_useful_den > 0 else float("nan")
        ),
        "partial_current_hurt_rejection_rate": (
            partial_current_true_reject / partial_current_hurt_den
            if math.isfinite(partial_current_hurt_den) and partial_current_hurt_den > 0 else float("nan")
        ),
        "partial_gate_accept_rate": _mean(rows, "partial_gate_accept"),
        "partial_gate_gain": _mean(rows, "partial_gate_gain_over_prev"),
        "partial_gate_true_accept_rate": partial_gate_true_accept,
        "partial_gate_false_reject_rate": partial_gate_false_reject,
        "partial_gate_true_reject_rate": partial_gate_true_reject,
        "partial_gate_false_accept_rate": partial_gate_false_accept,
        "partial_gate_useful_update_recall": (
            partial_gate_true_accept / partial_gate_useful_den
            if math.isfinite(partial_gate_useful_den) and partial_gate_useful_den > 0 else float("nan")
        ),
        "partial_gate_hurt_rejection_rate": (
            partial_gate_true_reject / partial_gate_hurt_den
            if math.isfinite(partial_gate_hurt_den) and partial_gate_hurt_den > 0 else float("nan")
        ),
        "word_gate_selected_words": _mean(rows, "word_gate_selected_count"),
        "word_gate_best_gain": _mean(rows, "word_gate_best_gain_over_prev"),
        "deploy_word_gate_best_gain": _mean(rows, "deploy_word_gate_best_gain_over_prev"),
        "anchor_word_gate_gain": _mean(rows, "anchor_word_gate_gain_over_prev"),
        "prev_word_gate_gain": _mean(rows, "prev_word_gate_gain_over_prev"),
        "token_state_raw_best_gain": _mean(rows, "token_state_raw_best_gain_over_prev"),
        "token_state_best_gain": _mean(rows, "token_state_best_gain_over_prev"),
        "token_learned_state_available_rate": _mean(rows, "token_learned_state_available"),
        "token_learned_frame_gate": _mean(rows, "token_learned_frame_gate_mean"),
        "token_learned_token_gate": _mean(rows, "token_learned_token_gate_mean"),
        "token_learned_state_delta_abs": _mean(rows, "token_learned_state_delta_abs_mean"),
        "token_learned_relation_attn": _mean(rows, "token_learned_relation_attn_mean"),
        "token_learned_visual_evidence_abs": _mean(rows, "token_learned_visual_evidence_abs_mean"),
        "token_learned_state_center_motion": _mean(rows, "token_learned_state_center_motion_norm"),
        "token_learned_state_scale_change": _mean(rows, "token_learned_state_scale_change_ratio"),
        "token_learned_conf_peak_gap": _mean(rows, "token_learned_conf_peak_gap"),
        "token_learned_conf_entropy": _mean(rows, "token_learned_conf_score_entropy"),
        "token_learned_candidate_deploy_delta": _mean(rows, "token_learned_candidate_deploy_score_delta"),
        "token_learned_candidate_partial_delta": _mean(rows, "token_learned_candidate_partial_deploy_delta"),
        "BLIP_better_anchor_ratio": _mean(rows, "blip_better_anchor"),
        "BLIP_hurts_ratio": _mean(rows, "blip_hurts"),
        "deploy_false_positive_ratio": _mean(rows, "deploy_false_positive"),
        "deploy_missed_oracle_ratio": _mean(rows, "deploy_missed_oracle"),
        "quality_gate_accept_rate": _mean(rows, "quality_gate_accept"),
        "quality_gate_gain": _mean(rows, "quality_gate_gain_over_prev"),
        "quality_gate_semantic": _mean(rows, "quality_gate_semantic"),
        "quality_gate_oracle_accept_rate": _mean(rows, "quality_gate_oracle_accept"),
        "quality_gate_deploy_accept_rate": _mean(rows, "quality_gate_deploy_accept"),
        "quality_gate_confidence_ok_rate": _mean(rows, "quality_gate_confidence_ok"),
        "quality_gate_deploy_score_delta": _mean(rows, "quality_gate_deploy_score_delta"),
        "score_peak_mean": _mean(rows, "score_peak"),
        "score_peak_second_gap_mean": _mean(rows, "score_peak_second_gap"),
        "pred_box_jump_ratio_mean": _mean(rows, "pred_box_jump_ratio"),
        "quality_gate_true_accept_rate": true_accept,
        "quality_gate_false_reject_rate": false_reject,
        "quality_gate_true_reject_rate": true_reject,
        "quality_gate_false_accept_rate": false_accept,
        "useful_update_recall": true_accept / useful_den if math.isfinite(useful_den) and useful_den > 0 else float("nan"),
        "hurt_rejection_rate": true_reject / hurt_den if math.isfinite(hurt_den) and hurt_den > 0 else float("nan"),
    }
    summary["class_label"] = _class_labels(summary, args)
    return summary


def _float_or_nan(row, key):
    try:
        value = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _boolish(row, key):
    value = _float_or_nan(row, key)
    if not math.isfinite(value):
        return None
    return value > 0.5


def _trigger_error_type(row):
    deploy = _boolish(row, "deploy_trigger")
    oracle = _boolish(row, "oracle_trigger")
    observable = _boolish(row, "oracle_trigger_observable")
    if observable is False:
        return "not_observable"
    if deploy is None or oracle is None:
        return "unknown"
    if deploy and oracle:
        return "trigger_true_positive"
    if deploy and not oracle:
        return "trigger_false_positive"
    if (not deploy) and oracle:
        return "trigger_false_negative"
    return "trigger_true_negative"


def _gate_error_type(row):
    if _boolish(row, "quality_gate_observable") is False:
        return "not_observable"
    if _boolish(row, "quality_gate_true_accept"):
        return "true_accept"
    if _boolish(row, "quality_gate_false_reject"):
        return "false_reject"
    if _boolish(row, "quality_gate_true_reject"):
        return "true_reject"
    if _boolish(row, "quality_gate_false_accept"):
        return "false_accept"
    score_delta = _float_or_nan(row, "quality_gate_score_delta")
    if math.isfinite(score_delta) and abs(score_delta) <= float(0.0):
        return "neutral"
    return "unknown"


def _partial_gate_error_type(row, prefix):
    if _boolish(row, "partial_gate_observable") is False:
        return "not_observable"
    if _boolish(row, "{}_true_accept".format(prefix)):
        return "true_accept"
    if _boolish(row, "{}_false_reject".format(prefix)):
        return "false_reject"
    if _boolish(row, "{}_true_reject".format(prefix)):
        return "true_reject"
    if _boolish(row, "{}_false_accept".format(prefix)):
        return "false_accept"
    useful = _boolish(row, "partial_label_useful")
    harmful = _boolish(row, "partial_label_harmful")
    if useful is False and harmful is False:
        return "neutral"
    return "unknown"


def _copy_frame_field(row, key):
    value = row.get(key, "")
    return "" if value is None else value


WORD_EVIDENCE_FIELDS = [
    "blip_word_content_word_count",
    "blip_word_target_word_overlap_count",
    "blip_word_target_word_missing_flag",
    "blip_word_context_word_count",
    "blip_word_context_dominance",
    "blip_word_target_template_gap_mean",
    "blip_word_target_template_gap_max",
    "blip_word_context_template_gap_mean",
    "blip_word_context_template_gap_max",
    "blip_word_new_template_gap_mean",
    "blip_word_new_template_gap_max",
    "blip_word_target_search_deploy_gap_mean",
    "blip_word_target_search_deploy_gap_max",
    "blip_word_context_search_deploy_gap_mean",
    "blip_word_context_search_deploy_gap_max",
    "blip_word_new_search_deploy_gap_mean",
    "blip_word_new_search_deploy_gap_max",
    "blip_word_target_minus_context_template_gap",
    "blip_word_target_minus_context_search_deploy_gap",
    "prev_word_target_template_gap_mean",
    "prev_word_target_search_deploy_gap_mean",
    "blip_minus_prev_target_template_gap",
    "blip_minus_prev_target_search_deploy_gap",
]

WORD_GATE_FIELDS = [
    "word_gate_word_count",
    "word_gate_selected_count",
    "word_gate_selected_words",
    "word_gate_best_word",
    "word_gate_best_word_deploy_gain",
    "word_gate_best_source",
    "word_gate_best_gain_over_prev",
    "deploy_word_gate_best_source",
    "deploy_word_gate_best_gain_over_prev",
    "token_state_best_source",
    "token_state_best_gain_over_prev",
]


def _deploy_selected_source(row):
    deploy = _boolish(row, "deploy_trigger")
    available = _boolish(row, "candidate_available")
    if deploy and available:
        return "blip"
    if deploy is None:
        return "unknown"
    return "prev"


def _make_error_rows(dataset_name, sequence_name, rows, sequence_summary, args):
    error_rows = []
    for row in rows:
        trigger_error = _trigger_error_type(row)
        gate_error = _gate_error_type(row)
        partial_current_error = _partial_gate_error_type(row, "partial_current_gate")
        partial_gate_error = _partial_gate_error_type(row, "partial_gate")
        error_row = {
            "dataset": dataset_name,
            "sequence": sequence_name,
            "frame": _copy_frame_field(row, "frame"),
            "sequence_class_label": sequence_summary.get("class_label", "none"),
            "candidate_mode": args.candidate_mode,
            "state_update_policy": args.state_update_policy,
            "quality_gate_mode": args.quality_gate_mode,
            "evidence_source": args.evidence_source,
            "trigger_error_type": trigger_error,
            "gate_error_type": gate_error,
            "partial_current_gate_error_type": partial_current_error,
            "partial_gate_error_type": partial_gate_error,
            "oracle_selected_source": _copy_frame_field(row, "oracle_source"),
            "deploy_selected_source": _deploy_selected_source(row),
            "gate_selected_source": _copy_frame_field(row, "quality_gate_source"),
            "candidate_available": _copy_frame_field(row, "candidate_available"),
            "deploy_trigger": _copy_frame_field(row, "deploy_trigger"),
            "oracle_trigger": _copy_frame_field(row, "oracle_trigger"),
            "deploy_false_positive": _copy_frame_field(row, "deploy_false_positive"),
            "deploy_missed_oracle": _copy_frame_field(row, "deploy_missed_oracle"),
            "quality_gate_accept": _copy_frame_field(row, "quality_gate_accept"),
            "quality_gate_oracle_accept": _copy_frame_field(row, "quality_gate_oracle_accept"),
            "quality_gate_deploy_accept": _copy_frame_field(row, "quality_gate_deploy_accept"),
            "quality_gate_source": _copy_frame_field(row, "quality_gate_source"),
            "quality_gate_gain_over_prev": _copy_frame_field(row, "quality_gate_gain_over_prev"),
            "quality_gate_score_delta": _copy_frame_field(row, "quality_gate_score_delta"),
            "quality_gate_deploy_score_delta": _copy_frame_field(row, "quality_gate_deploy_score_delta"),
            "oracle_gain_over_prev": _copy_frame_field(row, "oracle_gain_over_prev"),
            "hard_replace_gain_over_prev": _copy_frame_field(row, "hard_replace_gain_over_prev"),
            "anchor_delta_gain_over_prev": _copy_frame_field(row, "anchor_delta_gain_over_prev"),
            "prev_delta_gain_over_prev": _copy_frame_field(row, "prev_delta_gain_over_prev"),
            "best_partial_source": _copy_frame_field(row, "best_partial_source"),
            "best_partial_gap": _copy_frame_field(row, "best_partial_gap"),
            "best_partial_gain_over_prev": _copy_frame_field(row, "best_partial_gain_over_prev"),
            "partial_label_useful": _copy_frame_field(row, "partial_label_useful"),
            "partial_label_harmful": _copy_frame_field(row, "partial_label_harmful"),
            "partial_current_gate_gain_over_prev": _copy_frame_field(row, "partial_current_gate_gain_over_prev"),
            "partial_current_gate_true_accept": _copy_frame_field(row, "partial_current_gate_true_accept"),
            "partial_current_gate_false_reject": _copy_frame_field(row, "partial_current_gate_false_reject"),
            "partial_current_gate_true_reject": _copy_frame_field(row, "partial_current_gate_true_reject"),
            "partial_current_gate_false_accept": _copy_frame_field(row, "partial_current_gate_false_accept"),
            "deploy_best_partial_source": _copy_frame_field(row, "deploy_best_partial_source"),
            "deploy_best_partial_gap": _copy_frame_field(row, "deploy_best_partial_gap"),
            "deploy_best_partial_gain_over_prev": _copy_frame_field(row, "deploy_best_partial_gain_over_prev"),
            "partial_gate_accept": _copy_frame_field(row, "partial_gate_accept"),
            "partial_gate_source": _copy_frame_field(row, "partial_gate_source"),
            "partial_gate_gain_over_prev": _copy_frame_field(row, "partial_gate_gain_over_prev"),
            "partial_gate_true_accept": _copy_frame_field(row, "partial_gate_true_accept"),
            "partial_gate_false_reject": _copy_frame_field(row, "partial_gate_false_reject"),
            "partial_gate_true_reject": _copy_frame_field(row, "partial_gate_true_reject"),
            "partial_gate_false_accept": _copy_frame_field(row, "partial_gate_false_accept"),
            "partial_beats_hard_replace": _copy_frame_field(row, "partial_beats_hard_replace"),
            "partial_useful_when_blip_hurts": _copy_frame_field(row, "partial_useful_when_blip_hurts"),
            "word_gate_word_count": _copy_frame_field(row, "word_gate_word_count"),
            "word_gate_selected_count": _copy_frame_field(row, "word_gate_selected_count"),
            "word_gate_selected_words": _copy_frame_field(row, "word_gate_selected_words"),
            "word_gate_best_word": _copy_frame_field(row, "word_gate_best_word"),
            "word_gate_best_word_deploy_gain": _copy_frame_field(row, "word_gate_best_word_deploy_gain"),
            "word_gate_best_source": _copy_frame_field(row, "word_gate_best_source"),
            "word_gate_best_gain_over_prev": _copy_frame_field(row, "word_gate_best_gain_over_prev"),
            "deploy_word_gate_best_source": _copy_frame_field(row, "deploy_word_gate_best_source"),
            "deploy_word_gate_best_gain_over_prev": _copy_frame_field(row, "deploy_word_gate_best_gain_over_prev"),
            "token_state_best_source": _copy_frame_field(row, "token_state_best_source"),
            "token_state_best_gain_over_prev": _copy_frame_field(row, "token_state_best_gain_over_prev"),
            "anchor_score_gap": _copy_frame_field(row, "anchor_score_pos_hardneg_gap"),
            "prev_score_gap": _copy_frame_field(row, "prev_score_pos_hardneg_gap"),
            "blip_score_gap": _copy_frame_field(row, "blip_score_pos_hardneg_gap"),
            "anchor_delta_score_gap": _copy_frame_field(row, "anchor_delta_score_pos_hardneg_gap"),
            "prev_delta_score_gap": _copy_frame_field(row, "prev_delta_score_pos_hardneg_gap"),
            "anchor_gap": _copy_frame_field(row, "anchor_pos_hardneg_gap"),
            "prev_gap": _copy_frame_field(row, "prev_pos_hardneg_gap"),
            "blip_gap": _copy_frame_field(row, "blip_pos_hardneg_gap"),
            "anchor_delta_gap": _copy_frame_field(row, "anchor_delta_pos_hardneg_gap"),
            "prev_delta_gap": _copy_frame_field(row, "prev_delta_pos_hardneg_gap"),
            "oracle_gap": _copy_frame_field(row, "oracle_gap"),
            "prev_deploy_gap": _copy_frame_field(row, "prev_deploy_pos_hardneg_gap"),
            "blip_deploy_gap": _copy_frame_field(row, "blip_deploy_pos_hardneg_gap"),
            "anchor_delta_deploy_gap": _copy_frame_field(row, "anchor_delta_deploy_pos_hardneg_gap"),
            "prev_delta_deploy_gap": _copy_frame_field(row, "prev_delta_deploy_pos_hardneg_gap"),
            "quality_gate_semantic": _copy_frame_field(row, "quality_gate_semantic"),
            "quality_gate_semantic_anchor": _copy_frame_field(row, "quality_gate_semantic_anchor"),
            "quality_gate_semantic_prev": _copy_frame_field(row, "quality_gate_semantic_prev"),
            "quality_gate_confidence_ok": _copy_frame_field(row, "quality_gate_confidence_ok"),
            "score_peak": _copy_frame_field(row, "score_peak"),
            "score_second_peak": _copy_frame_field(row, "score_second_peak"),
            "score_peak_second_gap": _copy_frame_field(row, "score_peak_second_gap"),
            "pred_box_jump_ratio": _copy_frame_field(row, "pred_box_jump_ratio"),
            "trigger_by_position": _copy_frame_field(row, "trigger_by_position"),
            "trigger_by_scale": _copy_frame_field(row, "trigger_by_scale"),
            "trigger_by_color": _copy_frame_field(row, "trigger_by_color"),
            "trigger_area_ratio": _copy_frame_field(row, "trigger_area_ratio"),
            "trigger_center_distance": _copy_frame_field(row, "trigger_center_distance"),
            "trigger_color_delta": _copy_frame_field(row, "trigger_color_delta"),
            "anchor_iou": _copy_frame_field(row, "anchor_iou"),
            "prev_iou": _copy_frame_field(row, "prev_iou"),
            "blip_iou": _copy_frame_field(row, "blip_iou"),
            "anchor_delta_iou": _copy_frame_field(row, "anchor_delta_iou"),
            "prev_delta_iou": _copy_frame_field(row, "prev_delta_iou"),
            "anchor_blip_content_overlap_count": _copy_frame_field(row, "anchor_blip_content_overlap_count"),
            "prev_blip_content_overlap_count": _copy_frame_field(row, "prev_blip_content_overlap_count"),
            "anchor_state_candidate_description": _copy_frame_field(row, "anchor_state_candidate_description"),
            "anchor_delta_candidate_description": _copy_frame_field(row, "anchor_delta_candidate_description"),
            "prev_delta_candidate_description": _copy_frame_field(row, "prev_delta_candidate_description"),
            "anchor_description": _copy_frame_field(row, "anchor_description"),
            "prev_description": _copy_frame_field(row, "prev_description"),
            "blip_description": _copy_frame_field(row, "blip_description"),
        }
        for field in WORD_GATE_FIELDS:
            error_row[field] = _copy_frame_field(row, field)
        for field in WORD_EVIDENCE_FIELDS:
            error_row[field] = _copy_frame_field(row, field)
        error_rows.append(error_row)
    return error_rows


def _count_values(rows, key):
    counts = {}
    for row in rows:
        value = str(row.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _mean_by_group(rows, group_key, value_key):
    grouped = {}
    for row in rows:
        group = str(row.get(group_key, "unknown"))
        value = _float_or_nan(row, value_key)
        if not math.isfinite(value):
            continue
        grouped.setdefault(group, []).append(value)
    return {
        key: sum(values) / len(values)
        for key, values in grouped.items()
        if values
    }


def _write_error_report(save_dir, args, error_rows):
    if not error_rows:
        return
    fieldnames = [
        "dataset", "sequence", "frame", "sequence_class_label",
        "candidate_mode", "state_update_policy", "quality_gate_mode", "evidence_source",
        "trigger_error_type", "gate_error_type",
        "partial_current_gate_error_type", "partial_gate_error_type",
        "oracle_selected_source", "deploy_selected_source", "gate_selected_source",
        "candidate_available", "deploy_trigger", "oracle_trigger",
        "deploy_false_positive", "deploy_missed_oracle",
        "quality_gate_accept", "quality_gate_oracle_accept", "quality_gate_deploy_accept",
        "quality_gate_source", "quality_gate_gain_over_prev",
        "quality_gate_score_delta", "quality_gate_deploy_score_delta",
        "oracle_gain_over_prev",
        "hard_replace_gain_over_prev", "anchor_delta_gain_over_prev",
        "prev_delta_gain_over_prev", "best_partial_source", "best_partial_gap",
        "best_partial_gain_over_prev",
        "partial_label_useful", "partial_label_harmful",
        "partial_current_gate_gain_over_prev",
        "partial_current_gate_true_accept", "partial_current_gate_false_reject",
        "partial_current_gate_true_reject", "partial_current_gate_false_accept",
        "deploy_best_partial_source",
        "deploy_best_partial_gap", "deploy_best_partial_gain_over_prev",
        "partial_gate_accept", "partial_gate_source", "partial_gate_gain_over_prev",
        "partial_gate_true_accept", "partial_gate_false_reject",
        "partial_gate_true_reject", "partial_gate_false_accept",
        "partial_beats_hard_replace",
        "partial_useful_when_blip_hurts",
        "anchor_score_gap", "prev_score_gap", "blip_score_gap",
        "anchor_delta_score_gap", "prev_delta_score_gap",
        "anchor_gap", "prev_gap", "blip_gap", "anchor_delta_gap",
        "prev_delta_gap", "oracle_gap",
        "prev_deploy_gap", "blip_deploy_gap",
        "anchor_delta_deploy_gap", "prev_delta_deploy_gap",
        "quality_gate_semantic", "quality_gate_semantic_anchor", "quality_gate_semantic_prev",
        "quality_gate_confidence_ok", "score_peak", "score_second_peak",
        "score_peak_second_gap", "pred_box_jump_ratio",
        "trigger_by_position", "trigger_by_scale", "trigger_by_color",
        "trigger_area_ratio", "trigger_center_distance", "trigger_color_delta",
        "anchor_iou", "prev_iou", "blip_iou", "anchor_delta_iou", "prev_delta_iou",
        "anchor_blip_content_overlap_count", "prev_blip_content_overlap_count",
        "anchor_state_candidate_description", "anchor_delta_candidate_description",
        "prev_delta_candidate_description",
        "anchor_description", "prev_description", "blip_description",
    ]
    fieldnames.extend(WORD_GATE_FIELDS)
    fieldnames.extend(WORD_EVIDENCE_FIELDS)
    csv_path = os.path.join(save_dir, "s0_error_report.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in error_rows:
            writer.writerow(row)

    trigger_counts = _count_values(error_rows, "trigger_error_type")
    gate_counts = _count_values(error_rows, "gate_error_type")
    partial_current_counts = _count_values(error_rows, "partial_current_gate_error_type")
    partial_gate_counts = _count_values(error_rows, "partial_gate_error_type")
    gate_gain_by_type = _mean_by_group(error_rows, "gate_error_type", "quality_gate_gain_over_prev")
    deploy_delta_by_type = _mean_by_group(error_rows, "gate_error_type", "quality_gate_deploy_score_delta")
    semantic_by_type = _mean_by_group(error_rows, "gate_error_type", "quality_gate_semantic")
    confidence_by_type = _mean_by_group(error_rows, "gate_error_type", "quality_gate_confidence_ok")
    partial_current_gain_by_type = _mean_by_group(
        error_rows, "partial_current_gate_error_type", "partial_current_gate_gain_over_prev")
    partial_gate_gain_by_type = _mean_by_group(
        error_rows, "partial_gate_error_type", "partial_gate_gain_over_prev")

    lines = [
        "# Stage 3-S0 Error Report",
        "",
        "Config: `{}`".format(args.config),
        "Datasets: `{}`".format(args.dataset_names),
        "Rows: `{}`".format(len(error_rows)),
        "",
        "## Trigger Error Types",
        "",
        "| Type | Count |",
        "| --- | ---: |",
    ]
    for key in sorted(trigger_counts):
        lines.append("| {} | {} |".format(key, trigger_counts[key]))
    lines.extend([
        "",
        "## Gate Error Types",
        "",
        "| Type | Count | mean gate gain | mean deploy delta | mean semantic | confidence ok rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for key in sorted(gate_counts):
        lines.append("| {} | {} | {:.6g} | {:.6g} | {:.6g} | {:.6g} |".format(
            key,
            gate_counts[key],
            gate_gain_by_type.get(key, float("nan")),
            deploy_delta_by_type.get(key, float("nan")),
            semantic_by_type.get(key, float("nan")),
            confidence_by_type.get(key, float("nan")),
        ))
    lines.extend([
        "",
        "## Partial-Absorption Current Gate Counts",
        "",
        "| Type | Count | mean partial gain |",
        "| --- | ---: | ---: |",
    ])
    for key in sorted(partial_current_counts):
        lines.append("| {} | {} | {:.6g} |".format(
            key,
            partial_current_counts[key],
            partial_current_gain_by_type.get(key, float("nan")),
        ))
    lines.extend([
        "",
        "## Partial-Absorption Deploy Gate Counts",
        "",
        "| Type | Count | mean partial gate gain |",
        "| --- | ---: | ---: |",
    ])
    for key in sorted(partial_gate_counts):
        lines.append("| {} | {} | {:.6g} |".format(
            key,
            partial_gate_counts[key],
            partial_gate_gain_by_type.get(key, float("nan")),
        ))
    lines.extend([
        "",
        "## Field Notes",
        "",
        "- `trigger_error_type` compares original deploy trigger with oracle need-to-update.",
        "- `gate_error_type` compares deploy quality gate accept/reject with oracle useful/harmful BLIP labels.",
        "- `false_accept` rows are the highest-risk updates: deploy gate accepted BLIP but oracle score-gap says it hurt.",
        "- `false_reject` rows are missed useful updates: BLIP would help but deploy gate rejected it.",
        "- `oracle_selected_source`, `deploy_selected_source`, and `gate_selected_source` make source decisions explicit.",
        "- `anchor_score_gap`, `prev_score_gap`, and `blip_score_gap` are raw center-score gaps for the three language sources.",
        "- `anchor_delta` means `anchor + new BLIP content words`; `prev_delta` means `prev + new BLIP content words`.",
        "- `best_partial_gain_over_prev` is the counterfactual gain of the best conservative text absorption source over `prev`.",
        "- `partial_useful_when_blip_hurts` marks frames where BLIP hard replacement hurts but a partial text update still helps.",
        "- `partial_current_gate_error_type` evaluates the old BLIP gate against the partial-update label.",
        "- `partial_gate_error_type` evaluates a deploy gate built from `deploy_best_partial_gain_over_prev`.",
        "- `word_gate` selects individual BLIP new words by deploy score-gap before composing a conservative text state.",
        "- `token_state` evaluates latent language residual tokens after BERT embedding and before visual-language fusion.",
        "- `quality_gate_score_delta` uses GT/oracle positive region; `quality_gate_deploy_score_delta` uses predicted-box positive region.",
        "- `blip_word_*` fields are optional and appear when `--word_evidence` is enabled.",
    ])
    with open(os.path.join(save_dir, "s0_error_report_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_suite_summary(save_dir, args, rows):
    csv_path = os.path.join(save_dir, "s0_screen_summary.csv")
    fieldnames = [
        "dataset", "sequence", "frames", "class_label",
        "baseline_mean_iou", "score_gap_mean", "score_gap_low_ratio",
        "deploy_trigger_rate", "trigger_by_position_rate",
        "trigger_by_scale_rate", "trigger_by_color_rate",
        "trigger_color_delta_mean", "BLIP_available_rate",
        "anchor_score_gap", "blip_score_gap", "anchor_delta_score_gap",
        "prev_delta_score_gap", "oracle_score_gap",
        "oracle_score_gain", "hard_replace_gain", "anchor_delta_gain",
        "prev_delta_gain", "best_partial_gain", "deploy_best_partial_gain",
        "partial_beats_hard_replace_ratio",
        "partial_useful_when_blip_hurts_ratio",
        "partial_label_useful_rate", "partial_label_harmful_rate",
        "partial_current_gate_accept_rate", "partial_current_gate_gain",
        "partial_current_gate_true_accept_rate", "partial_current_gate_false_reject_rate",
        "partial_current_gate_true_reject_rate", "partial_current_gate_false_accept_rate",
        "partial_current_useful_update_recall", "partial_current_hurt_rejection_rate",
        "partial_gate_accept_rate", "partial_gate_gain",
        "partial_gate_true_accept_rate", "partial_gate_false_reject_rate",
        "partial_gate_true_reject_rate", "partial_gate_false_accept_rate",
        "partial_gate_useful_update_recall", "partial_gate_hurt_rejection_rate",
        "word_gate_selected_words", "word_gate_best_gain",
        "deploy_word_gate_best_gain", "anchor_word_gate_gain", "prev_word_gate_gain",
        "token_state_raw_best_gain", "token_state_best_gain",
        "token_learned_state_available_rate", "token_learned_frame_gate",
        "token_learned_token_gate", "token_learned_state_delta_abs",
        "token_learned_relation_attn", "token_learned_visual_evidence_abs",
        "token_learned_state_center_motion", "token_learned_state_scale_change",
        "token_learned_conf_peak_gap", "token_learned_conf_entropy",
        "token_learned_candidate_deploy_delta", "token_learned_candidate_partial_delta",
        "BLIP_better_anchor_ratio", "BLIP_hurts_ratio",
        "deploy_false_positive_ratio", "deploy_missed_oracle_ratio",
        "quality_gate_accept_rate", "quality_gate_gain", "quality_gate_semantic",
        "quality_gate_oracle_accept_rate", "quality_gate_deploy_accept_rate",
        "quality_gate_confidence_ok_rate", "quality_gate_deploy_score_delta",
        "score_peak_mean", "score_peak_second_gap_mean", "pred_box_jump_ratio_mean",
        "quality_gate_true_accept_rate", "quality_gate_false_reject_rate",
        "quality_gate_true_reject_rate", "quality_gate_false_accept_rate",
        "useful_update_recall", "hurt_rejection_rate",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    counts = {"A_mid_difficulty": 0, "B_language_gain": 0, "C_language_hurts": 0, "none": 0}
    for row in rows:
        labels = str(row.get("class_label", "none")).split(",")
        for label in labels:
            if label:
                counts[label] = counts.get(label, 0) + 1

    lines = [
        "# Stage 3-S0 Sequence Screening",
        "",
        "Config: `{}`".format(args.config),
        "Datasets: `{}`".format(args.dataset_names),
        "Candidate mode: `{}`".format(args.candidate_mode),
        "Evidence source: `{}`".format(args.evidence_source),
        "State update policy: `{}`".format(args.state_update_policy),
        "Quality gate mode: `{}`".format(args.quality_gate_mode),
        "Quality gate: `score_delta > {}` and semantic >= `{}`".format(
            args.quality_gate_gap_eps, args.quality_gate_semantic_thr),
        "Confidence guard: peak >= `{}`, peak-second >= `{}`, box-jump <= `{}`".format(
            args.quality_gate_score_peak_thr, args.quality_gate_peak_gap_thr,
            args.quality_gate_box_jump_thr),
        "Max frames per sequence: `{}`".format("full" if int(args.max_frames) <= 0 else args.max_frames),
        "",
        "## Class Counts",
        "",
        "| Class | Count |",
        "| --- | ---: |",
    ]
    for label in ("A_mid_difficulty", "B_language_gain", "C_language_hurts", "none"):
        lines.append("| {} | {} |".format(label, counts.get(label, 0)))
    lines.extend([
        "",
        "## Selection Rules",
        "",
        "- A: `baseline_mean_iou` in [{:.3g}, {:.3g}]".format(
            args.class_a_iou_low, args.class_a_iou_high),
        "- B: `oracle_score_gain > {:.3g}` and `BLIP_better_anchor_ratio >= {:.3g}`".format(
            args.class_b_gain_thr, args.class_b_blip_better_thr),
        "- C: `BLIP_hurts_ratio >= {:.3g}`".format(args.class_c_hurts_thr),
        "",
        "## Notes",
        "",
        "- `baseline_mean_iou` uses the anchor-language forward in the S0 probe.",
        "- `score_gap_mean` is `anchor_score_pos_hardneg_gap`.",
        "- `oracle_score_gain` is the mean S0 source-oracle gain over previous state under the selected evidence source.",
        "- Use `oracle_blip` for screening optimization space; use `deploy_like` to inspect real trigger behavior.",
        "- Quality gate is a non-learning diagnostic: accept BLIP only when score-gap improves and text overlap is above threshold.",
    ])
    with open(os.path.join(save_dir, "s0_screen_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _selected_indices(dataset, args):
    total = len(dataset)
    if args.sequence_indices:
        indices = []
        for part in args.sequence_indices.split(","):
            part = part.strip()
            if not part:
                continue
            indices.append(int(part))
    elif args.sequence_names:
        wanted = {name.strip() for name in args.sequence_names.split(",") if name.strip()}
        indices = [idx for idx, seq in enumerate(dataset) if seq.name in wanted]
    else:
        indices = list(range(total))
    if args.max_sequences > 0:
        indices = indices[:args.max_sequences]
    return indices


def run(args):
    out_dir = os.path.join(args.out_dir, args.output_tag)
    os.makedirs(out_dir, exist_ok=True)
    summaries = []
    error_rows = []
    dataset_names = [name.strip() for name in args.dataset_names.split(",") if name.strip()]
    for dataset_name in dataset_names:
        dataset = get_dataset(dataset_name)
        indices = _selected_indices(dataset, args)
        for seq_idx in indices:
            seq = dataset[seq_idx]
            print("[S0-screen] {}:{} ({}/{})".format(
                dataset_name, seq.name, seq_idx + 1, len(dataset)))
            probe_args = Namespace(
                config=args.config,
                checkpoint_config=args.checkpoint_config,
                dataset_name=dataset_name,
                sequence=str(seq_idx),
                runid=args.runid,
                max_frames=args.max_frames,
                top_ratio=args.top_ratio,
                hardneg_topk=args.hardneg_topk,
                language_description="",
                candidate_mode=args.candidate_mode,
                evidence_source=args.evidence_source,
                oracle_state_update=args.oracle_state_update,
                state_update_policy=args.state_update_policy,
                quality_gate_gap_eps=args.quality_gate_gap_eps,
                quality_gate_semantic_thr=args.quality_gate_semantic_thr,
                quality_gate_semantic_ref=args.quality_gate_semantic_ref,
                quality_gate_mode=args.quality_gate_mode,
                quality_gate_score_peak_thr=args.quality_gate_score_peak_thr,
                quality_gate_peak_gap_thr=args.quality_gate_peak_gap_thr,
                quality_gate_box_jump_thr=args.quality_gate_box_jump_thr,
                use_score_prior_effect=args.use_score_prior_effect,
                word_evidence=args.word_evidence,
                word_absorption=args.word_absorption,
                word_gate_max_candidate_words=args.word_gate_max_candidate_words,
                word_gate_max_selected_words=args.word_gate_max_selected_words,
                word_gate_min_deploy_gain=args.word_gate_min_deploy_gain,
                token_state_probe=args.token_state_probe,
                token_state_alphas=args.token_state_alphas,
                learned_token_state_probe=args.learned_token_state_probe,
                tag="",
                output_tag="{}/{}".format(args.output_tag, _safe_tag(dataset_name)),
                out_dir=args.out_dir,
            )
            _, rows = run_s0_probe(probe_args)
            summary = _summarize_sequence(dataset_name, seq.name, rows, args)
            summaries.append(summary)
            error_rows.extend(_make_error_rows(dataset_name, seq.name, rows, summary, args))
    _write_suite_summary(out_dir, args, summaries)
    _write_error_report(out_dir, args, error_rows)
    print("Saved Stage 3-S0 screening summary to {}".format(out_dir))


def main():
    parser = argparse.ArgumentParser(description="Screen sequences for Stage 3-S language-state experiments.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_config", default="")
    parser.add_argument("--dataset_names", default="otb_lang,hoot_balanced20")
    parser.add_argument("--runid", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=20,
                        help="Number of frames after initialization per sequence. Use <=0 for full sequences.")
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--sequence_indices", default="")
    parser.add_argument("--sequence_names", default="")
    parser.add_argument("--top_ratio", type=float, default=0.1)
    parser.add_argument("--hardneg_topk", type=int, default=6)
    parser.add_argument("--candidate_mode", default="oracle_blip",
                        choices=("deploy_like", "oracle_blip", "off"))
    parser.add_argument("--evidence_source", default="score",
                        choices=("score", "lmq_prior"))
    parser.add_argument("--oracle_state_update", type=int, default=1)
    parser.add_argument("--state_update_policy", default="oracle",
                        choices=("oracle", "gate", "anchor_state_gate", "prev_delta_gate",
                                 "best_partial_oracle", "word_gate", "token_state", "none"))
    parser.add_argument("--quality_gate_gap_eps", type=float, default=0.0)
    parser.add_argument("--quality_gate_semantic_thr", type=float, default=0.0)
    parser.add_argument("--quality_gate_semantic_ref", default="max",
                        choices=("anchor", "prev", "max"))
    parser.add_argument("--quality_gate_mode", default="deploy",
                        choices=("deploy", "oracle"))
    parser.add_argument("--quality_gate_score_peak_thr", type=float, default=-1e9)
    parser.add_argument("--quality_gate_peak_gap_thr", type=float, default=-1e9)
    parser.add_argument("--quality_gate_box_jump_thr", type=float, default=1e9)
    parser.add_argument("--use_score_prior_effect", action="store_true")
    parser.add_argument("--word_evidence", action="store_true")
    parser.add_argument("--word_absorption", action="store_true")
    parser.add_argument("--word_gate_max_candidate_words", type=int, default=8)
    parser.add_argument("--word_gate_max_selected_words", type=int, default=4)
    parser.add_argument("--word_gate_min_deploy_gain", type=float, default=0.0)
    parser.add_argument("--token_state_probe", action="store_true")
    parser.add_argument("--token_state_alphas", default="0.1,0.3")
    parser.add_argument("--learned_token_state_probe", action="store_true")
    parser.add_argument("--score_gap_low_thr", type=float, default=0.0)
    parser.add_argument("--class_a_iou_low", type=float, default=0.3)
    parser.add_argument("--class_a_iou_high", type=float, default=0.7)
    parser.add_argument("--class_b_gain_thr", type=float, default=0.0)
    parser.add_argument("--class_b_blip_better_thr", type=float, default=0.3)
    parser.add_argument("--class_c_hurts_thr", type=float, default=0.5)
    parser.add_argument("--output_tag", default="stage3_s0_screen_score_oracle")
    parser.add_argument("--out_dir", default="output/test/language_state_s0_screen")
    args = parser.parse_args()
    if args.evidence_source != "score":
        args.output_tag = "{}_{}".format(args.output_tag, _safe_tag(args.evidence_source))
    run(args)


if __name__ == "__main__":
    main()
