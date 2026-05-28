import argparse
import csv
import math
import os
import sys

import cv2 as cv
import torch
import torch.nn.functional as F

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker
from lib.train.data.processing_utils import sample_target
from tracking.visualte_diagnostic import (
    _feat_size,
    _heat_stats,
    _plain_tile,
    _region_gap_stats,
    _render_map_tile,
    _run_label,
    _search_crop_box,
    _token_box_mask,
    _read_rgb,
    _save_story_grid,
)


def _safe_tag(text):
    return str(text).lower().replace("-", "m").replace(".", "p").replace("/", "_")


def _as_flat(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().view(-1)
    return None


def _gumbel_samples(logits, prev_samples, samples, tau, hard):
    # logits: B x N x 2, prev_samples: S x B x N x 1
    bsz, num_tokens, _ = logits.shape
    logits_rep = logits[None].expand(samples, bsz, num_tokens, 2).reshape(samples * bsz, num_tokens, 2)
    keep = F.gumbel_softmax(logits_rep, tau=tau, hard=hard, dim=-1)[..., 0:1]
    keep = keep.view(samples, bsz, num_tokens, 1) * prev_samples
    return keep


def _scalar(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _save_probe_story(save_dir, frame_num, search_img, layer_tiles):
    row_labels = []
    tile_rows = []
    col_labels = ["search", "logit margin", "soft keep", "gumbel mean", "gumbel std", "sample keep"]
    search_tile = _plain_tile(search_img, 142)
    _draw_grid(search_tile)
    for layer, margin, soft_keep, g_mean, g_std, sample_keep in layer_tiles:
        row_labels.append("L{}".format(layer))
        tiles = [
            search_tile.copy(),
            _render_map_tile({"heat": margin, "mode": "signed", "factor": 0.55}, search_img, 142),
            _render_map_tile({"heat": soft_keep, "mode": "fixed", "vmin": 0.0, "vmax": 1.0, "factor": 0.55}, search_img, 142),
            _render_map_tile({"heat": g_mean, "mode": "fixed", "vmin": 0.0, "vmax": 1.0, "factor": 0.55}, search_img, 142),
            _render_map_tile({"heat": g_std, "mode": "minmax", "factor": 0.55}, search_img, 142),
            _render_map_tile({"heat": sample_keep, "mode": "fixed", "vmin": 0.0, "vmax": 1.0, "factor": 0.55}, search_img, 142),
        ]
        for tile in tiles:
            if tile is not None:
                _draw_grid(tile)
        tile_rows.append(tiles)
    if not tile_rows:
        return
    _save_story_grid(
        os.path.join(save_dir, "{:04d}_gumbel_keep_probe.jpg".format(frame_num)),
        "Frame {} language keep Gumbel probe".format(frame_num),
        "logits -> softmax keep and sampled Gumbel-Softmax decisions",
        row_labels,
        col_labels,
        tile_rows,
        note="Rows are TE stages; gumbel mean/std are over repeated samples from the same logits.",
    )


def _draw_grid(tile):
    h, w = tile.shape[:2]
    feat_sz = 24
    for i in range(1, feat_sz):
        x = int(round(i * w / float(feat_sz)))
        y = int(round(i * h / float(feat_sz)))
        cv.line(tile, (x, 0), (x, h - 1), (230, 230, 230), 1, cv.LINE_AA)
        cv.line(tile, (0, y), (w - 1, y), (230, 230, 230), 1, cv.LINE_AA)


def _write_summary(save_dir, args, rows):
    by_layer = {}
    for row in rows:
        by_layer.setdefault(row["layer"], []).append(row)

    def mean(layer_rows, key):
        values = []
        for row in layer_rows:
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        return sum(values) / len(values) if values else float("nan")

    lines = [
        "# TE Keep Gumbel Probe",
        "",
        "Config: `{}`".format(args.config),
        "Dataset/sequence: `{}:{}`".format(args.dataset_name, args.sequence),
        "Samples per frame: `{}`".format(args.samples),
        "Tau: `{}` | hard: `{}`".format(args.tau, args.hard),
        "",
        "| Layer | soft GT mass | soft in-out gap | gumbel GT mass | gumbel in-out gap | gumbel std mean | sample top10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for layer in sorted(by_layer, key=lambda x: int(x)):
        layer_rows = by_layer[layer]
        lines.append(
            "| L{layer} | {soft_mass:.6g} | {soft_gap:.6g} | {g_mass:.6g} | {g_gap:.6g} | {g_std:.6g} | {sample_top:.6g} |".format(
                layer=layer,
                soft_mass=mean(layer_rows, "soft_keep_mass_in_gt"),
                soft_gap=mean(layer_rows, "soft_keep_gap_in_minus_out"),
                g_mass=mean(layer_rows, "gumbel_keep_mean_mass_in_gt"),
                g_gap=mean(layer_rows, "gumbel_keep_mean_gap_in_minus_out"),
                g_std=mean(layer_rows, "gumbel_keep_std_mean"),
                sample_top=mean(layer_rows, "gumbel_keep_sample_top10_precision"),
            )
        )
    lines.extend([
        "",
        "Read this as a source probe: if soft/gumbel GT mass and in-out gap stay near the GT area prior, language keep is not adding target-location information.",
        "",
    ])
    with open(os.path.join(save_dir, "gumbel_keep_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run(args):
    dataset = get_dataset(args.dataset_name)
    seq = dataset[int(args.sequence)] if str(args.sequence).isdigit() else dataset[args.sequence]
    tracker_info = Tracker("dutrack", args.config, args.dataset_name, args.runid)
    params = tracker_info.get_parameters(run_id=args.runid)
    params.debug = 0
    tracker = tracker_info.create_tracker(params)

    if args.keep_vl_source:
        for predictor in getattr(tracker.network.backbone, "visual_te_predictors", []):
            if hasattr(predictor, "keep_vl_source"):
                predictor.keep_vl_source = str(args.keep_vl_source).lower()

    tau = float(args.tau if args.tau is not None else getattr(tracker.network.backbone, "visual_te_tau", 1.0))
    hard = bool(args.hard)
    run_label = _run_label(args.config, args)
    if args.keep_vl_source:
        run_label = "{}_{}".format(run_label, _safe_tag(args.keep_vl_source))
    save_dir = os.path.join(args.out_dir, run_label, seq.name)
    os.makedirs(save_dir, exist_ok=True)

    image0 = _read_rgb(seq.frames[0])
    init_info = seq.init_info()
    init_info["class"] = seq.object_class
    init_info["path"] = seq.name
    tracker.initialize(image0, init_info)

    search_len = int(tracker.network.backbone.pos_embed_x.shape[1])
    feat_sz = _feat_size(search_len)
    layers = list(getattr(tracker.network.backbone, "visual_te_pruning_loc", []))
    rows = []
    max_frame = min(int(args.max_frames), len(seq.frames) - 1)
    for frame_num in range(1, max_frame + 1):
        image = _read_rgb(seq.frames[frame_num])
        if args.crop_source == "gt_prev" and seq.ground_truth_rect is not None:
            crop_state = seq.ground_truth_rect[frame_num - 1].tolist()
        else:
            crop_state = list(tracker.state)
        search_img, resize_factor, search_amask = sample_target(
            image, crop_state, params.search_factor, output_sz=params.search_size)
        crop_box = _search_crop_box(crop_state, resize_factor, params.search_size)
        gt_box = seq.ground_truth_rect[frame_num].tolist() if seq.ground_truth_rect is not None else None
        gt_mask = _token_box_mask(gt_box, crop_box, feat_sz) if gt_box is not None else torch.zeros(search_len, dtype=torch.bool)

        search_nt = tracker.preprocessor.process(search_img, search_amask)
        with torch.no_grad():
            out = tracker.network.forward(
                template=tracker.memory_frames.copy(),
                search=[search_nt.tensors],
                descript=[[getattr(tracker, "descript", "")]],
            )
        out = out[-1] if isinstance(out, list) else out
        logits_list = out.get("lang_te_search_logits", [])
        if not logits_list:
            raise RuntimeError("No lang_te_search_logits found. This config must enable KEEP_VL.")

        prev_soft = None
        prev_samples = None
        layer_tiles = []
        for idx, logits in enumerate(logits_list):
            logits = logits.detach().float().cpu()
            if prev_soft is None:
                prev_soft = torch.ones(logits.shape[0], logits.shape[1], 1)
            if prev_samples is None:
                prev_samples = torch.ones(int(args.samples), logits.shape[0], logits.shape[1], 1)

            margin = logits[..., 0] - logits[..., 1]
            soft_keep = F.softmax(logits / max(tau, 1e-6), dim=-1)[..., 0:1] * prev_soft
            gumbel_keep = _gumbel_samples(logits, prev_samples, int(args.samples), max(tau, 1e-6), hard)
            gumbel_mean = gumbel_keep.mean(dim=0)
            gumbel_std = gumbel_keep.std(dim=0, unbiased=False)
            gumbel_sample = gumbel_keep[0]
            prev_soft = soft_keep
            prev_samples = gumbel_keep

            layer = layers[idx] if idx < len(layers) else idx
            row = {"frame": frame_num, "sequence": seq.name, "stage": idx, "layer": layer}
            row.update(_heat_stats("logit_margin", margin[0], gt_mask, args.top_ratio))
            row.update(_region_gap_stats("logit_margin", margin[0], gt_mask))
            row.update(_heat_stats("soft_keep", soft_keep[0, :, 0], gt_mask, args.top_ratio))
            row.update(_region_gap_stats("soft_keep", soft_keep[0, :, 0], gt_mask))
            row.update(_heat_stats("gumbel_keep_mean", gumbel_mean[0, :, 0], gt_mask, args.top_ratio))
            row.update(_region_gap_stats("gumbel_keep_mean", gumbel_mean[0, :, 0], gt_mask))
            row.update(_heat_stats("gumbel_keep_std", gumbel_std[0, :, 0], gt_mask, args.top_ratio))
            row.update(_heat_stats("gumbel_keep_sample", gumbel_sample[0, :, 0], gt_mask, args.top_ratio))
            row["gumbel_keep_std_mean"] = gumbel_std[0, :, 0].mean().item()
            rows.append(row)
            layer_tiles.append((
                layer,
                margin[0],
                soft_keep[0, :, 0],
                gumbel_mean[0, :, 0],
                gumbel_std[0, :, 0],
                gumbel_sample[0, :, 0],
            ))

        _save_probe_story(save_dir, frame_num, search_img, layer_tiles)

    csv_path = os.path.join(save_dir, "gumbel_keep_probe.csv")
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    _write_summary(save_dir, args, rows)
    print("Saved Gumbel keep probe to {}".format(save_dir))


def main():
    parser = argparse.ArgumentParser(description="Probe language-conditioned TE keep logits and Gumbel decisions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_name", default="otb_lang")
    parser.add_argument("--sequence", default="Biker")
    parser.add_argument("--runid", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=5)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--crop_source", choices=("gt_prev", "tracker"), default="gt_prev")
    parser.add_argument("--top_ratio", type=float, default=0.1)
    parser.add_argument("--keep_vl_source", default=None, help="Runtime override: global or template_match.")
    parser.add_argument("--out_dir", default="output/test/te_keep_gumbel_probe")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
