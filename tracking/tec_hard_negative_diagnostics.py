import argparse
import csv
import os
from collections import OrderedDict

import _init_paths

from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker


DEFAULT_SEQUENCES = [
    "Human4",
    "Coupon",
    "Skater",
    "Bird1",
    "Car1",
    "Human9",
    "Trans",
]


def _build_info(seq, frame_num, prev_output):
    info = seq.frame_info(frame_num)
    info["previous_output"] = prev_output
    info["class"] = seq.object_class
    info["path"] = seq.name
    info["num"] = frame_num
    info["gt_bbox"] = seq.ground_truth_rect[frame_num].tolist()
    return info


def run_one_sequence(tracker_info, seq, frame_stride, max_frames):
    params = tracker_info.get_parameters(tracker_info.run_id)
    params.debug = 0
    params.enable_diagnostics = True
    tracker = tracker_info.create_tracker(params)

    image0 = tracker_info._read_image(seq.frames[0])
    init_info = seq.init_info()
    init_info["class"] = seq.object_class
    out = tracker.initialize(image0, init_info) or {}
    prev_output = OrderedDict(out)

    rows = []
    processed = 0
    for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
        image = tracker_info._read_image(frame_path)
        info = _build_info(seq, frame_num, prev_output)
        out = tracker.track(image, info)
        prev_output = OrderedDict(out)
        processed += 1

        if frame_num % frame_stride == 0 and "diagnostics" in out:
            row = dict(out["diagnostics"])
            row.update({
                "tracker_param": tracker_info.parameter_name,
                "dataset": seq.dataset,
                "sequence": seq.name,
                "frame": frame_num,
            })
            rows.append(row)

        if max_frames > 0 and processed >= max_frames:
            break

    return rows


def main():
    parser = argparse.ArgumentParser(description="Sample hard-negative gap diagnostics for TEC.")
    parser.add_argument("--dataset_name", type=str, default="otb_lang")
    parser.add_argument("--sequence", action="append", default=None,
                        help="Sequence name. Can be repeated. Defaults to representative TEC cases.")
    parser.add_argument("--all_sequences", action="store_true",
                        help="Run all sequences in the dataset instead of the representative default subset.")
    parser.add_argument("--frame_stride", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "dutrack_384_full_tec_stage1",
            "dutrack_384_full_tec_stage1_wrong",
            "dutrack_384_full_tec_stage1_generic",
        ],
    )
    parser.add_argument(
        "--run_id",
        type=int,
        default=None,
        help="Optional checkpoint epoch/run id passed to Tracker, e.g. 3 loads DUTrack_ep0003.pth.tar.",
    )
    parser.add_argument("--output", type=str,
                        default="output/diagnostics/tec_stage1_hard_negative_gap.csv")
    args = parser.parse_args()

    dataset = get_dataset(args.dataset_name)
    if args.all_sequences:
        sequences = list(dataset)
    else:
        selected = set(args.sequence or DEFAULT_SEQUENCES)
        sequences = [seq for seq in dataset if seq.name in selected]
        missing = sorted(selected - {seq.name for seq in sequences})
        if missing:
            raise ValueError("Missing sequences in {}: {}".format(args.dataset_name, ", ".join(missing)))

    all_rows = []
    for cfg_name in args.configs:
        tracker_info = Tracker("dutrack", cfg_name, args.dataset_name, run_id=args.run_id)
        for seq in sequences:
            print("diagnose {} {}".format(cfg_name, seq.name))
            all_rows.extend(run_one_sequence(tracker_info, seq, args.frame_stride, args.max_frames))

    if not all_rows:
        raise RuntimeError("No diagnostic rows were collected.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    preferred = [
        "tracker_param",
        "dataset",
        "sequence",
        "frame",
        "gt_response_max",
        "hard_negative_response_max",
        "hard_negative_gap",
        "peak_inside_gt",
        "gt_score_max",
        "hard_negative_score_max",
        "hard_negative_score_gap",
        "gt_grid_x1",
        "gt_grid_y1",
        "gt_grid_x2",
        "gt_grid_y2",
        "peak_x",
        "peak_y",
    ]
    all_keys = []
    seen = set(preferred)
    for row in all_rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            all_keys.append(key)
    extra = all_keys
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=preferred + extra)
        writer.writeheader()
        writer.writerows(all_rows)

    print("Wrote {}".format(args.output))
    for cfg_name in args.configs:
        rows = [row for row in all_rows if row["tracker_param"] == cfg_name]
        gap = sum(float(row["hard_negative_gap"]) for row in rows) / len(rows)
        inside = sum(int(row["peak_inside_gt"]) for row in rows) / len(rows)
        print("{}: mean_gap={:.5f}, peak_inside_gt={:.3f}, rows={}".format(
            cfg_name, gap, inside, len(rows)))


if __name__ == "__main__":
    main()
