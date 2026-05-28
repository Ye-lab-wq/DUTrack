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
    _read_rgb,
    _region_gap_stats,
    _render_map_tile,
    _run_label,
    _save_story_grid,
    _search_crop_box,
    _token_box_mask,
    _draw_patch_grid,
)
from tracking.te_keep_gumbel_probe import _safe_tag


def _as_1d(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().view(-1)
    return None


def _masked_mean(tokens, mask):
    if mask is None:
        return tokens.mean(dim=1, keepdim=True)
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=tokens.dtype, device=tokens.device)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (tokens * mask).sum(dim=1, keepdim=True) / denom


def _unit(x):
    return F.normalize(x, dim=-1, eps=1e-6)


def _minmax_01(x):
    x = _as_1d(x)
    if x is None:
        return None
    return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-6)


def _tokenize_mask(tracker, text, device):
    tokenizer = tracker.network.backbone.tokenizer
    encoded = tokenizer(
        [text], add_special_tokens=True, truncation=True, pad_to_max_length=True,
        max_length=16, return_attention_mask=True)
    input_ids = torch.tensor(encoded["input_ids"], device=device)
    attention_mask = torch.tensor(encoded["attention_mask"], device=device)
    mask = attention_mask.clone()
    for token_id in (tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id):
        if token_id is not None:
            mask = mask * (input_ids != token_id).long()
    empty_rows = mask.sum(dim=1, keepdim=True) == 0
    mask = torch.where(empty_rows, attention_mask, mask)
    return mask.unsqueeze(-1)


def _split_tokens(tracker, out_dict):
    feat = out_dict.get("backbone_feat")
    if isinstance(feat, list):
        feat = feat[-1]
    if not isinstance(feat, torch.Tensor) or feat.dim() != 3:
        raise RuntimeError("backbone_feat must be a B x N x C tensor.")

    search_len = int(tracker.network.backbone.pos_embed_x.shape[1])
    template_unit_len = int(tracker.network.backbone.pos_embed_z.shape[1])
    search_tokens = feat[:, -search_len:, :]
    prefix = feat[:, :-search_len, :]

    template_count = max(1, len(getattr(tracker, "memory_frames", [])))
    template_len = min(prefix.shape[1], template_unit_len * template_count)
    template_tokens = prefix[:, -template_len:, :]
    lang_and_temporal = prefix[:, :-template_len, :]
    lang_len = 16
    if lang_and_temporal.shape[1] < lang_len:
        raise RuntimeError(
            "Cannot split language tokens: prefix length {} is shorter than {}.".format(
                lang_and_temporal.shape[1], lang_len))
    lang_tokens = lang_and_temporal[:, -lang_len:, :]
    temporal_tokens = lang_and_temporal[:, :-lang_len, :]
    return temporal_tokens, lang_tokens, template_tokens, search_tokens


def _similarity_maps(lang_tokens, template_tokens, search_tokens, lang_mask, temperature):
    lang_context = _masked_mean(lang_tokens, lang_mask)
    direct_lang_search = (_unit(search_tokens) * _unit(lang_context)).sum(dim=-1)

    lang_for_template = _unit(lang_context)
    template_norm = _unit(template_tokens)
    template_affinity = (template_norm * lang_for_template).sum(dim=-1, keepdim=True)
    weights = torch.softmax(template_affinity / max(float(temperature), 1e-6), dim=1)
    template_proto = (template_tokens * weights).sum(dim=1, keepdim=True)
    template_proto_search = (_unit(search_tokens) * _unit(template_proto)).sum(dim=-1)

    template_mean = template_tokens.mean(dim=1, keepdim=True)
    template_mean_search = (_unit(search_tokens) * _unit(template_mean)).sum(dim=-1)

    return {
        "language_direct_search": direct_lang_search,
        "language_template_affinity": template_affinity.squeeze(-1),
        "template_proto_search": template_proto_search,
        "template_mean_search": template_mean_search,
    }


def _save_source_story(save_dir, frame_num, search_img, maps):
    tile_size = 142
    search_tile = _plain_tile(search_img, tile_size)
    _draw_patch_grid(search_tile, 24)
    col_labels = [
        "search",
        "lang -> search",
        "lang -> template",
        "template proto -> search",
        "template mean -> search",
        "score map",
    ]
    tiles = [search_tile]
    for key in (
            "language_direct_search",
            "language_template_affinity",
            "template_proto_search",
            "template_mean_search",
            "score_map"):
        heat = maps.get(key)
        if heat is None:
            tile = _plain_tile(search_img, tile_size)
            feat_sz = 24
        else:
            tile = _render_map_tile({"heat": heat, "mode": "minmax", "factor": 0.55}, search_img, tile_size)
            feat_sz = _feat_size(_as_1d(heat).numel())
        if tile is not None:
            _draw_patch_grid(tile, feat_sz)
        tiles.append(tile)
    _save_story_grid(
        os.path.join(save_dir, "{:04d}_language_visual_source_probe.jpg".format(frame_num)),
        "Frame {} language-visual source probe".format(frame_num),
        "No training: compare raw source signals before keep/score optimization.",
        ["source"],
        col_labels,
        [tiles],
        note="lang->template is rendered on the template-sized grid; other maps are search grids.",
    )


def _mean(rows, key):
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return sum(values) / len(values) if values else float("nan")


def _write_summary(save_dir, args, rows):
    lines = [
        "# Language Visual Source Probe",
        "",
        "Config: `{}`".format(args.config),
        "Dataset/sequence: `{}:{}`".format(args.dataset_name, args.sequence),
        "Frames: `{}` | template softmax temperature: `{}`".format(args.max_frames, args.temperature),
        "",
        "| Source | GT mass | top10 precision | in-out gap | raw min | raw max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("language_direct_search", "template_proto_search", "template_mean_search", "score_map"):
        lines.append(
            "| {name} | {mass:.6g} | {top:.6g} | {gap:.6g} | {minv:.6g} | {maxv:.6g} |".format(
                name=name,
                mass=_mean(rows, "{}_01_mass_in_gt".format(name)),
                top=_mean(rows, "{}_01_top10_precision".format(name)),
                gap=_mean(rows, "{}_gap_in_minus_out".format(name)),
                minv=_mean(rows, "{}_min".format(name)),
                maxv=_mean(rows, "{}_max".format(name)),
            )
        )
    lines.extend([
        "",
        "Interpretation:",
        "",
        "- `language_direct_search` checks whether the pooled language token is already spatially aligned with the target in search.",
        "- `template_proto_search` checks the newer template-prototype path: language first selects template patches, then the selected template prototype scores search patches.",
        "- `template_mean_search` is a language-free baseline; if it matches or beats `template_proto_search`, language is not adding useful localization information.",
        "- `score_map` is the existing center head output, included only as a reference.",
        "",
        "Use top10 precision and raw in-out gap first. GT mass is useful but can be dominated by target area and per-map min-max scaling.",
    ])
    with open(os.path.join(save_dir, "language_visual_source_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run(args):
    dataset = get_dataset(args.dataset_name)
    seq = dataset[int(args.sequence)] if str(args.sequence).isdigit() else dataset[args.sequence]
    tracker_info = Tracker("dutrack", args.config, args.dataset_name, args.runid)
    params = tracker_info.get_parameters(run_id=args.runid)
    params.debug = 0
    tracker = tracker_info.create_tracker(params)
    if not args.use_config_te_effects:
        # The probe asks whether the raw language/template/search source is useful.
        # Keep TE heads callable for inspection, but remove their downstream effect.
        tracker.network.backbone.te_policy_apply = "none"
        tracker.network.score_prior_enabled = False

    run_label = _run_label(args.config, args)
    if args.output_tag:
        run_label = args.output_tag
    elif args.tag:
        run_label = "{}_{}".format(run_label, _safe_tag(args.tag))
    save_dir = os.path.join(args.out_dir, run_label, seq.name)
    os.makedirs(save_dir, exist_ok=True)

    image0 = _read_rgb(seq.frames[0])
    init_info = seq.init_info()
    init_info["class"] = seq.object_class
    init_info["path"] = seq.name
    tracker.initialize(image0, init_info)

    search_len = int(tracker.network.backbone.pos_embed_x.shape[1])
    search_feat_sz = _feat_size(search_len)
    rows = []
    max_frame = min(int(args.max_frames), len(seq.frames) - 1)
    for frame_num in range(1, max_frame + 1):
        image = _read_rgb(seq.frames[frame_num])
        crop_state = seq.ground_truth_rect[frame_num - 1].tolist() if args.crop_source == "gt_prev" else list(tracker.state)
        search_img, resize_factor, search_amask = sample_target(
            image, crop_state, params.search_factor, output_sz=params.search_size)
        crop_box = _search_crop_box(crop_state, resize_factor, params.search_size)
        gt_box = seq.ground_truth_rect[frame_num].tolist() if seq.ground_truth_rect is not None else None
        gt_mask = _token_box_mask(gt_box, crop_box, search_feat_sz) if gt_box is not None else torch.zeros(search_len, dtype=torch.bool)

        search_nt = tracker.preprocessor.process(search_img, search_amask)
        with torch.no_grad():
            out = tracker.network.forward(
                template=tracker.memory_frames.copy(),
                search=[search_nt.tensors],
                descript=[[getattr(tracker, "descript", "")]],
            )
        out = out[-1] if isinstance(out, list) else out
        _, lang_tokens, template_tokens, search_tokens = _split_tokens(tracker, out)
        lang_mask = _tokenize_mask(tracker, getattr(tracker, "descript", ""), lang_tokens.device)
        maps = _similarity_maps(
            lang_tokens, template_tokens, search_tokens, lang_mask, args.temperature)

        score_map = out.get("score_map")
        if isinstance(score_map, torch.Tensor):
            maps["score_map"] = score_map[0].detach().float().cpu().view(-1)

        row = {"frame": frame_num, "sequence": seq.name}
        for name in ("language_direct_search", "template_proto_search", "template_mean_search", "score_map"):
            heat = maps.get(name)
            if heat is None:
                continue
            raw = _as_1d(heat[0] if isinstance(heat, torch.Tensor) and heat.dim() == 2 else heat)
            heat01 = _minmax_01(raw)
            row.update(_heat_stats("{}_01".format(name), heat01, gt_mask, args.top_ratio))
            row.update(_region_gap_stats(name, raw, gt_mask))
            row["{}_min".format(name)] = raw.min().item()
            row["{}_max".format(name)] = raw.max().item()
        rows.append(row)

        story_maps = {}
        for key, value in maps.items():
            if key == "language_template_affinity":
                template_len = value.shape[1] if value.dim() == 2 else value.numel()
                try:
                    _feat_size(template_len)
                    story_maps[key] = value[0].detach().float().cpu()
                except ValueError:
                    story_maps[key] = None
            elif isinstance(value, torch.Tensor):
                story_maps[key] = value[0].detach().float().cpu().view(-1)
            else:
                story_maps[key] = value
        _save_source_story(save_dir, frame_num, search_img, story_maps)

    csv_path = os.path.join(save_dir, "language_visual_source_probe.csv")
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    _write_summary(save_dir, args, rows)
    print("Saved language-visual source probe to {}".format(save_dir))


def main():
    parser = argparse.ArgumentParser(description="No-training probe for language-conditioned visual token source signals.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_name", default="otb_lang")
    parser.add_argument("--sequence", default="Biker")
    parser.add_argument("--runid", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=5)
    parser.add_argument("--crop_source", choices=("gt_prev", "tracker"), default="gt_prev")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--top_ratio", type=float, default=0.1)
    parser.add_argument("--tag", default="no_train_source")
    parser.add_argument("--output_tag", default=None)
    parser.add_argument("--out_dir", default="output/test/language_visual_source_probe")
    parser.add_argument("--use_config_te_effects", action="store_true",
                        help="Keep configured TE attention/score-prior effects during the probe.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
