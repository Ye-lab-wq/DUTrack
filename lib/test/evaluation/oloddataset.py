import json
import os

import numpy as np

from lib.test.evaluation.data import BaseDataset, Sequence, SequenceList
from lib.test.evaluation.language_annotations import lookup_language_description
from lib.test.utils.load_text import load_text


class OLODDataset(BaseDataset):
    """OLOD test set."""

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.olod_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info["path"]
        nz = sequence_info["nz"]
        ext = sequence_info["ext"]
        start_frame = sequence_info["startFrame"]
        end_frame = sequence_info["endFrame"]
        init_omit = sequence_info.get("initOmit", 0)

        frames = [
            f"{self.base_path}/{sequence_path}/{frame_num:0{nz}}.{ext}"
            for frame_num in range(start_frame + init_omit, end_frame + 1)
        ]
        anno_path = os.path.join(self.base_path, sequence_info["anno_path"])
        ground_truth_rect = load_text(str(anno_path), delimiter=",", dtype=np.float64, backend="numpy")

        sequence_name = sequence_info["name"]
        text_description = lookup_language_description("olod", sequence_name, "")
        return Sequence(
            sequence_name,
            frames,
            "olod",
            ground_truth_rect[init_omit:, :].reshape(-1, 4),
            text_description=text_description or None,
        )

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        json_path = os.path.join(self.base_path, "olod.json")
        with open(json_path, "r") as f:
            return json.load(f)
