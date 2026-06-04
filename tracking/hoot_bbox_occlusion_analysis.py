import argparse
import csv
import json
import os
from collections import defaultdict

import _init_paths
import numpy as np

from lib.test.evaluation import get_dataset
from lib.test.evaluation.environment import env_settings


THRESHOLDS = np.arange(0.0, 1.0 + 0.05, 0.05)
OCCLUSION_KEYS = (
    "full_occlusion",
    "partial_obj_occlusion",
    "similar_occluder",
    "cut_by_frame",
)


def _xywh_to_poly(box):
    x, y, w, h = [float(v) for v in box]
    return np.array([
        [x, y],
        [x + max(w - 1.0, 0.0), y],
        [x + max(w - 1.0, 0.0), y + max(h - 1.0, 0.0)],
        [x, y + max(h - 1.0, 0.0)],
    ], dtype=np.float64)


def _poly_to_xywh(points):
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.zeros(4, dtype=np.float64)
    xy_min = points.min(axis=0)
    xy_max = points.max(axis=0)
    return np.array([
        xy_min[0],
        xy_min[1],
        xy_max[0] - xy_min[0],
        xy_max[1] - xy_min[1],
    ], dtype=np.float64)


def _poly_area(poly):
    poly = np.asarray(poly, dtype=np.float64)
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return abs(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _signed_area(poly):
    poly = np.asarray(poly, dtype=np.float64)
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


def _inside(point, edge_start, edge_end, orientation):
    cross = (
        (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
        - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
    )
    return cross >= -1e-9 if orientation >= 0 else cross <= 1e-9


def _line_intersection(p1, p2, q1, q2):
    p = np.asarray(p1, dtype=np.float64)
    r = np.asarray(p2, dtype=np.float64) - p
    q = np.asarray(q1, dtype=np.float64)
    s = np.asarray(q2, dtype=np.float64) - q
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-12:
        return p2
    qp = q - p
    t = (qp[0] * s[1] - qp[1] * s[0]) / denom
    return p + t * r


def _convex_clip(subject, clip):
    output = [np.asarray(p, dtype=np.float64) for p in subject]
    clip = [np.asarray(p, dtype=np.float64) for p in clip]
    orientation = _signed_area(np.asarray(clip, dtype=np.float64))

    for i in range(len(clip)):
        edge_start = clip[i]
        edge_end = clip[(i + 1) % len(clip)]
        input_list = output
        output = []
        if not input_list:
            break
        prev = input_list[-1]
        prev_inside = _inside(prev, edge_start, edge_end, orientation)
        for cur in input_list:
            cur_inside = _inside(cur, edge_start, edge_end, orientation)
            if cur_inside:
                if not prev_inside:
                    output.append(_line_intersection(prev, cur, edge_start, edge_end))
                output.append(cur)
            elif prev_inside:
                output.append(_line_intersection(prev, cur, edge_start, edge_end))
            prev = cur
            prev_inside = cur_inside
    return np.asarray(output, dtype=np.float64)


def _poly_iou(poly_a, poly_b):
    area_a = _poly_area(poly_a)
    area_b = _poly_area(poly_b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    inter_poly = _convex_clip(poly_a, poly_b)
    inter = _poly_area(inter_poly)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _xywh_iou(pred, gt):
    pred_poly = _xywh_to_poly(pred)
    gt_poly = _xywh_to_poly(gt)
    return _poly_iou(pred_poly, gt_poly)


def _load_text_boxes(path):
    data = np.loadtxt(path, delimiter="\t")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 4:
        data = np.loadtxt(path, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(1, -1)
    return data.astype(np.float64)


def _load_anno(base_path, sequence_name):
    class_name, video_id = sequence_name.rsplit("-", 1)
    anno_path = os.path.join(base_path, class_name, video_id, "anno.json")
    with open(anno_path, "r", encoding="utf-8") as f:
        anno = json.load(f)
    return sorted(anno["frames"], key=lambda item: int(item["frame_id"]))


def _parse_tracker(spec):
    if "=" in spec:
        display, param = spec.split("=", 1)
    else:
        param = spec
        display = spec
    return display, param


def _scope_masks(attrs):
    absent = np.array([bool(a.get("absent", False)) for a in attrs], dtype=bool)
    masks = {
        "all_frames": np.ones(len(attrs), dtype=bool),
        "visible": ~absent,
    }
    any_occ = np.zeros(len(attrs), dtype=bool)
    for key in OCCLUSION_KEYS:
        cur = np.array([bool(a.get(key, False)) for a in attrs], dtype=bool) & ~absent
        masks[key] = cur
        any_occ |= cur
    masks["any_occlusion"] = any_occ
    masks["visible_no_occlusion"] = (~absent) & (~any_occ)
    return masks


def _summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "frames": 0,
            "mean_iou": "",
            "auc": "",
            "op50": "",
            "op75": "",
        }
    success = [(values > t).mean() for t in THRESHOLDS]
    return {
        "frames": int(values.size),
        "mean_iou": float(values.mean() * 100.0),
        "auc": float(np.mean(success) * 100.0),
        "op50": float((values > 0.5).mean() * 100.0),
        "op75": float((values > 0.75).mean() * 100.0),
    }


def _aggregate_sequence_summaries(summaries):
    if not summaries:
        return {
            "sequences": 0,
            "frames": 0,
            "mean_iou": "",
            "auc": "",
            "op50": "",
            "op75": "",
        }
    keys = ("mean_iou", "auc", "op50", "op75")
    return {
        "sequences": len(summaries),
        "frames": int(sum(summary["frames"] for summary in summaries)),
        **{key: float(np.mean([summary[key] for summary in summaries])) for key in keys},
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Supplementary HOOT evaluation: compare current aa_bb OPE with "
            "predicted-axis-aligned vs rot_bb polygon IoU, split by occlusion attributes."
        )
    )
    parser.add_argument("--dataset", default="hoot_all")
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=[
            "normal=dutrack_384_full_hoot_all_lang_normal",
            "wrong=dutrack_384_full_hoot_all_lang_wrong",
            "generic=dutrack_384_full_hoot_all_lang_generic",
            "no_update=dutrack_384_full_hoot_all_lang_no_update",
        ],
        help="Tracker specs as display=parameter_name.",
    )
    parser.add_argument(
        "--output",
        default="output/diagnostics/hoot_bbox_occlusion_eval.csv",
    )
    args = parser.parse_args()

    env = env_settings()
    dataset = get_dataset(args.dataset)
    tracker_specs = [_parse_tracker(spec) for spec in args.trackers]

    rows = []
    for display_name, param_name in tracker_specs:
        metric_values = defaultdict(list)
        for seq in dataset:
            result_path = os.path.join(
                env.results_path, "dutrack", param_name, seq.dataset, "{}.txt".format(seq.name)
            )
            if not os.path.isfile(result_path):
                raise FileNotFoundError(result_path)

            pred = _load_text_boxes(result_path)
            frames = _load_anno(env.hoot_path, seq.name)
            aa_gt = np.asarray([_poly_to_xywh(frame.get("aa_bb", [])) for frame in frames], dtype=np.float64)
            rot_gt = [np.asarray(frame.get("rot_bb", []), dtype=np.float64) for frame in frames]
            attrs = [frame.get("attributes", {}) or {} for frame in frames]

            if pred.shape[0] > len(frames):
                pred = pred[:len(frames)]
            elif pred.shape[0] < len(frames):
                pad = np.zeros((len(frames) - pred.shape[0], 4), dtype=np.float64)
                pred = np.concatenate([pred, pad], axis=0)
            pred[0, :] = aa_gt[0, :]

            aa_iou = np.asarray([_xywh_iou(p, g) for p, g in zip(pred, aa_gt)], dtype=np.float64)
            rot_iou = np.asarray([_poly_iou(_xywh_to_poly(p), g) for p, g in zip(pred, rot_gt)], dtype=np.float64)
            absent = np.array([bool(attr.get("absent", False)) for attr in attrs], dtype=bool)
            aa_iou[absent] = 0.0
            rot_iou[absent] = 0.0

            for scope, mask in _scope_masks(attrs).items():
                if not mask.any():
                    continue
                metric_values[(scope, "aa_bb_xywh_iou")].append(_summarize(aa_iou[mask]))
                metric_values[(scope, "rot_bb_polygon_iou")].append(_summarize(rot_iou[mask]))

        for (scope, metric), summaries in sorted(metric_values.items()):
            summary = _aggregate_sequence_summaries(summaries)
            rows.append({
                "dataset": args.dataset,
                "tracker": display_name,
                "tracker_param": param_name,
                "scope": scope,
                "metric": metric,
                **summary,
            })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fieldnames = [
        "dataset",
        "tracker",
        "tracker_param",
        "scope",
        "metric",
        "sequences",
        "frames",
        "mean_iou",
        "auc",
        "op50",
        "op75",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("wrote {}".format(args.output))


if __name__ == "__main__":
    main()
