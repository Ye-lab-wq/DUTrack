import json
import os

import numpy as np

from lib.test.evaluation.data import BaseDataset, Sequence, SequenceList
from lib.test.evaluation.language_annotations import lookup_language_description


class HOOTDataset(BaseDataset):
    """HOOT test/evaluation dataset."""

    def __init__(self, split="test"):
        super().__init__()
        self.base_path = getattr(self.env_settings, "hoot_path", "")
        self.split = split
        self.dataset_name = "hoot" if split in ("test", "all") else "hoot_{}".format(split)
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(*seq_info) for seq_info in self.sequence_list])

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        if not self.base_path or not os.path.isdir(self.base_path):
            raise RuntimeError(
                "HOOT path is not configured or does not exist. Set settings.hoot_path in "
                "lib/test/evaluation/local.py to the extracted HOOT root directory."
            )

        split_keys = None
        split_file = os.path.join(self.base_path, "{}.txt".format(self.split))
        if not os.path.isfile(split_file):
            repo_split_file = os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", "data_specs", "hoot", "{}.txt".format(self.split)))
            if os.path.isfile(repo_split_file):
                split_file = repo_split_file
        if self.split != "all" and os.path.isfile(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                split_keys = {line.strip() for line in f if line.strip()}

        sequence_list = []
        for class_name in sorted(os.listdir(self.base_path)):
            class_dir = os.path.join(self.base_path, class_name)
            if not os.path.isdir(class_dir):
                continue
            for video_id in sorted(os.listdir(class_dir)):
                video_dir = os.path.join(class_dir, video_id)
                if not os.path.isdir(video_dir):
                    continue
                if not os.path.isfile(os.path.join(video_dir, "anno.json")):
                    continue
                sequence_name = "{}-{}".format(class_name, video_id)
                if split_keys is not None and sequence_name not in split_keys:
                    continue
                sequence_list.append((class_name, video_id, sequence_name))

        if not sequence_list:
            raise RuntimeError(
                "No HOOT videos found under {}. Verify the class/video/anno.json layout.".format(
                    self.base_path)
            )
        return sequence_list

    @staticmethod
    def _polygon_to_xywh(points):
        if not points:
            return [0.0, 0.0, 0.0, 0.0]
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    @staticmethod
    def _frame_attributes(frame):
        return frame.get("attributes", {}) or {}

    def _construct_sequence(self, class_name, video_id, sequence_name):
        video_dir = os.path.join(self.base_path, class_name, video_id)
        anno_path = os.path.join(video_dir, "anno.json")
        with open(anno_path, "r", encoding="utf-8") as f:
            anno = json.load(f)

        anno_frames = sorted(anno["frames"], key=lambda item: int(item["frame_id"]))
        frame_paths = []
        boxes = []
        target_visible = []
        occlusion_attributes = []

        for frame in anno_frames:
            frame_id = int(frame["frame_id"])
            frame_paths.append(os.path.join(video_dir, "{:06d}.png".format(frame_id)))

            attrs = self._frame_attributes(frame)
            absent = bool(attrs.get("absent", False))
            boxes.append(self._polygon_to_xywh(frame.get("aa_bb", [])))

            target_visible.append(not absent)
            occlusion_attributes.append({
                "absent": absent,
                "full_occlusion": bool(attrs.get("full_occlusion", False)),
                "similar_occluder": bool(attrs.get("similar_occluder", False)),
                "cut_by_frame": bool(attrs.get("cut_by_frame", False)),
                "partial_obj_occlusion": bool(attrs.get("partial_obj_occlusion", False)),
            })

        target_class = class_name.replace("_", " ")
        # HOOT does not provide natural language descriptions by default.
        # Keep missing annotations as None so DUTrack falls back to BLIP.
        text_description = lookup_language_description(self.dataset_name, sequence_name, None)
        seq = Sequence(
            sequence_name,
            frame_paths,
            self.dataset_name,
            np.array(boxes, dtype=np.float64).reshape(-1, 4),
            object_class=target_class,
            target_visible=np.array(target_visible, dtype=bool),
            text_description=text_description,
        )
        seq.hoot_occlusion_attributes = occlusion_attributes
        return seq
