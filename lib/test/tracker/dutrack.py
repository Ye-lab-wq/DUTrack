import math
import numpy as np
from lib.models.dutrack import build_dutrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from lib.models.dutrack.i2d import descriptgenRefiner
from tracking.draw_heatmap import visualize_attn


class DUTrack(BaseTracker):
    def __init__(self, params):
        super(DUTrack, self).__init__(params)
        network = build_dutrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            # else:
            #     # self.add_hook()
            #     self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.descriptgenRefiner = None
        self.descript = ""
        self.language_anchor = ""
        self.language_source = ""
        self.language_candidate_description = ""
        self.language_filtered_description = ""
        self.language_word_reliability = None
        self.language_word_tokens = []
        self.language_word_filter_active = False
        self.language_word_reliability_active = False
        self.language_word_reliability_updated = False
        self.language_word_reliability_delta = 0.0
        self.language_word_reliability_score_peak = float("nan")
        self.language_word_reliability_hardneg_peak = float("nan")
        self.language_word_reliability_score_gap = float("nan")
        self.language_trigger_by_position = False
        self.language_trigger_by_scale = False
        self.language_trigger_by_color = False
        self.language_trigger_area_ratio = float("nan")
        self.language_trigger_center_distance = float("nan")
        self.language_trigger_color_delta = float("nan")
        self.his_image = None

    def _test_cfg_value(self, name, default):
        return getattr(getattr(self.cfg, "TEST", None), name, default)

    def _language_init_source(self):
        value = os.environ.get(
            "DUTRACK_LANGUAGE_INIT_SOURCE",
            self._test_cfg_value("LANGUAGE_INIT_SOURCE", "blip"),
        )
        return str(value).lower()

    def _language_update_mode(self):
        value = os.environ.get(
            "DUTRACK_LANGUAGE_UPDATE_MODE",
            self._test_cfg_value("LANGUAGE_UPDATE_MODE", "caption_replace"),
        )
        return str(value).lower()

    def _language_trigger_scale_thr(self):
        value = os.environ.get(
            "DUTRACK_LANGUAGE_TRIGGER_SCALE_THR",
            self._test_cfg_value("LANGUAGE_TRIGGER_SCALE_THR", 0.95),
        )
        return float(value)

    def _language_trigger_distance_stride(self):
        value = os.environ.get(
            "DUTRACK_LANGUAGE_TRIGGER_DISTANCE_STRIDE",
            self._test_cfg_value("LANGUAGE_TRIGGER_DISTANCE_STRIDE", 1.0 / 32.0),
        )
        return float(value)

    def _language_trigger_color_enabled(self):
        return self._bool_test_cfg_value("LANGUAGE_TRIGGER_COLOR_ENABLE", False)

    def _language_trigger_color_thr(self):
        value = os.environ.get(
            "DUTRACK_LANGUAGE_TRIGGER_COLOR_THR",
            self._test_cfg_value("LANGUAGE_TRIGGER_COLOR_THR", 35.0),
        )
        return float(value)

    @staticmethod
    def _clean_language_text(text):
        if text is None:
            return ""
        if isinstance(text, (list, tuple)):
            text = text[0] if len(text) > 0 else ""
        text = str(text).strip()
        return " ".join(text.split())

    def _ensure_descript_refiner(self):
        if self.descriptgenRefiner is None:
            self.descriptgenRefiner = descriptgenRefiner(
                self.params.cfg.MODEL.BACKBONE.BLIP_DIR,
                self.params.cfg.MODEL.BACKBONE.BERT_DIR,
            )
        return self.descriptgenRefiner

    def _generate_blip_description(self, image, cls):
        desc = self._ensure_descript_refiner()(image, cls=cls)
        return self._clean_language_text(desc)

    def _initial_language_description(self, image, info):
        source = self._language_init_source()
        dataset_desc = self._clean_language_text(
            info.get("init_text_description", info.get("text_description", ""))
        )
        class_desc = self._clean_language_text(info.get("class", ""))

        def blip_desc():
            return self._generate_blip_description(image, cls=info.get("class", None))

        if source == "dataset_or_class":
            desc = dataset_desc or class_desc or blip_desc()
            label = "dataset" if dataset_desc else ("class" if class_desc else "blip_fallback")
        elif source == "dataset_or_blip":
            desc = dataset_desc or blip_desc()
            label = "dataset" if dataset_desc else "blip_fallback"
        elif source == "class_or_blip":
            desc = class_desc or blip_desc()
            label = "class" if class_desc else "blip_fallback"
        elif source == "blip":
            desc = blip_desc()
            label = "blip"
        else:
            raise ValueError("Unsupported TEST.LANGUAGE_INIT_SOURCE: {}".format(source))

        if not desc:
            desc = "object"
            label = "fallback_object"
        return desc, label

    def _bool_test_cfg_value(self, name, default=False):
        value = os.environ.get("DUTRACK_{}".format(name), self._test_cfg_value(name, default))
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _word_filter_enabled(self):
        return self._bool_test_cfg_value("LANGUAGE_WORD_FILTER_ENABLE", False)

    def _word_filter_threshold(self):
        return float(self._test_cfg_value("LANGUAGE_WORD_FILTER_THRESHOLD", 0.4))

    def _word_filter_momentum(self):
        return float(self._test_cfg_value("LANGUAGE_WORD_FILTER_MOMENTUM", 0.8))

    def _word_filter_min_keep(self):
        return int(self._test_cfg_value("LANGUAGE_WORD_FILTER_MIN_KEEP", 2))

    def _word_reliability_enabled(self):
        return self._bool_test_cfg_value("LANGUAGE_WORD_RELIABILITY_ENABLE", False)

    def _word_reliability_source(self):
        return str(self._test_cfg_value("LANGUAGE_WORD_RELIABILITY_SOURCE", "target_hardneg_gap")).lower()

    def _word_reliability_momentum(self):
        return float(self._test_cfg_value("LANGUAGE_WORD_RELIABILITY_MOMENTUM", 0.8))

    def _word_reliability_tau(self):
        return float(self._test_cfg_value("LANGUAGE_WORD_RELIABILITY_TAU", 0.1))

    def _subject_min_reliability(self):
        return float(self._test_cfg_value("LANGUAGE_SUBJECT_MIN_RELIABILITY", 0.7))

    def _context_max_weight(self):
        return float(self._test_cfg_value("LANGUAGE_CONTEXT_MAX_WEIGHT", 0.4))

    def _subject_type_prior(self):
        return float(self._test_cfg_value("LANGUAGE_SUBJECT_TYPE_PRIOR", 1.0))

    def _attribute_type_prior(self):
        return float(self._test_cfg_value("LANGUAGE_ATTRIBUTE_TYPE_PRIOR", 1.0))

    def _context_type_prior(self):
        return float(self._test_cfg_value("LANGUAGE_CONTEXT_TYPE_PRIOR", 1.0))

    def _reliability_update_gate_enabled(self):
        return self._bool_test_cfg_value("LANGUAGE_RELIABILITY_UPDATE_GATE", True)

    def _reliability_gate_mode(self):
        return str(self._test_cfg_value("LANGUAGE_RELIABILITY_GATE_MODE", "score_gap")).lower()

    def _reliability_score_threshold(self):
        return float(self._test_cfg_value("LANGUAGE_RELIABILITY_SCORE_THR", 0.4))

    def _reliability_score_gap_threshold(self):
        return float(self._test_cfg_value("LANGUAGE_RELIABILITY_SCORE_GAP_THR", 0.05))

    def _language_token_ids(self, text):
        tokenizer = self.network.backbone.tokenizer
        encoded = tokenizer([text], add_special_tokens=True, truncation=True,
                            pad_to_max_length=True, max_length=16)
        return encoded["input_ids"][0]

    def _content_token_indices(self, ids):
        tokenizer = self.network.backbone.tokenizer
        special = {tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id}
        return [idx for idx, token_id in enumerate(ids) if token_id not in special]

    @staticmethod
    def _tokens_to_text(tokens):
        words = []
        for token in tokens:
            if token.startswith("##") and words:
                words[-1] = words[-1] + token[2:]
            elif token.startswith("##"):
                words.append(token[2:])
            else:
                words.append(token)
        return " ".join(words).strip()

    def _init_language_word_filter(self):
        self.language_word_filter_active = self._word_filter_enabled()
        self.language_word_reliability_active = self._word_reliability_enabled()
        self.language_filtered_description = self.language_anchor
        self.language_word_reliability = None
        self.language_word_tokens = []
        if not (self.language_word_filter_active or self.language_word_reliability_active):
            return
        ids = self._language_token_ids(self.language_anchor)
        labels = self.network.backbone.tokenizer.convert_ids_to_tokens(ids)
        content_idx = self._content_token_indices(ids)
        self.language_word_tokens = labels
        reliability = torch.zeros(len(ids), dtype=torch.float32)
        if content_idx:
            reliability[content_idx] = 1.0
        self.language_word_reliability = reliability
        if self.language_word_filter_active:
            self._apply_language_word_filter()

    def _apply_language_word_filter(self):
        if not self.language_word_filter_active or self.language_word_reliability is None:
            return
        ids = self._language_token_ids(self.language_anchor)
        labels = self.network.backbone.tokenizer.convert_ids_to_tokens(ids)
        content_idx = self._content_token_indices(ids)
        if not content_idx:
            self.language_filtered_description = self.language_anchor
            self.descript = self.language_anchor
            return
        scores = self.language_word_reliability[content_idx]
        threshold = self._word_filter_threshold()
        keep_positions = [idx for idx, score in zip(content_idx, scores.tolist()) if score >= threshold]
        min_keep = min(max(self._word_filter_min_keep(), 1), len(content_idx))
        if len(keep_positions) < min_keep:
            top_local = torch.topk(scores, min_keep).indices.tolist()
            keep_positions = sorted({content_idx[i] for i in top_local} | set(keep_positions))
        keep_tokens = [labels[idx] for idx in keep_positions]
        filtered = self._tokens_to_text(keep_tokens)
        if not filtered:
            filtered = self.language_anchor
        self.language_filtered_description = filtered
        self.descript = filtered
        self.language_source = "word_filter"

    def _update_language_word_filter(self, out_dict):
        if not self.language_word_filter_active:
            return
        weights_list = out_dict.get("word_level_weights") if isinstance(out_dict, dict) else None
        if not weights_list:
            return
        weights = weights_list[-1]
        if not isinstance(weights, torch.Tensor):
            return
        weights = weights[0].detach().float().cpu().view(-1)
        if weights.numel() == 0:
            return
        if self.language_word_reliability is None or self.language_word_reliability.numel() != weights.numel():
            self.language_word_reliability = torch.ones(weights.numel(), dtype=torch.float32)
        ids = self._language_token_ids(self.language_anchor)
        content_idx = self._content_token_indices(ids)
        evidence = torch.zeros_like(self.language_word_reliability)
        if content_idx:
            content_weights = weights[content_idx].clamp_min(0)
            if float(content_weights.max().item()) > 0:
                content_weights = content_weights / content_weights.max().clamp_min(1e-6)
            evidence[content_idx] = content_weights
        momentum = min(max(self._word_filter_momentum(), 0.0), 1.0)
        self.language_word_reliability = (
            momentum * self.language_word_reliability + (1.0 - momentum) * evidence
        ).clamp(0.0, 1.0)
        self._apply_language_word_filter()

    @staticmethod
    def _clean_word_token(label):
        token = str(label).strip().lower()
        if token.startswith("##"):
            token = token[2:]
        return "".join(ch for ch in token if ch.isalnum() or ch == "_")

    def _subject_token_indices(self, labels, content_idx):
        context = {"the", "a", "an", "of", "on", "in", "at", "by", "with", "near",
                   "under", "above", "below", "behind", "beside", "between", "and",
                   "or", "to", "from", "held", "holding", "road", "tree", "hand",
                   "floor", "ground", "street", "background"}
        attributes = {"red", "blue", "green", "yellow", "black", "white", "gray",
                      "grey", "brown", "orange", "purple", "pink", "dark", "bright",
                      "light", "small", "large", "big", "tiny", "visible", "occluded"}
        candidates = []
        for idx in content_idx:
            token = self._clean_word_token(labels[idx])
            if token and token not in context and token not in attributes:
                candidates.append((idx, token))
        return [candidates[0][0]] if candidates else []

    def _attribute_token_indices(self, labels, content_idx):
        attributes = {"red", "blue", "green", "yellow", "black", "white", "gray",
                      "grey", "brown", "orange", "purple", "pink", "dark", "bright",
                      "light", "small", "large", "big", "tiny", "visible", "occluded"}
        return [idx for idx in content_idx if self._clean_word_token(labels[idx]) in attributes]

    def _context_token_indices(self, labels, content_idx):
        context = {"the", "a", "an", "of", "on", "in", "at", "by", "with", "near",
                   "under", "above", "below", "behind", "beside", "between", "and",
                   "or", "to", "from", "held", "holding", "road", "tree", "hand",
                   "floor", "ground", "street", "background"}
        return [idx for idx in content_idx if self._clean_word_token(labels[idx]) in context]

    def _network_language_word_reliability(self):
        if not self.language_word_reliability_active or self.language_word_reliability is None:
            return None
        values = self.language_word_reliability.clone()
        ids = self._language_token_ids(self.language_anchor)
        labels = self.network.backbone.tokenizer.convert_ids_to_tokens(ids)
        content_idx = self._content_token_indices(ids)
        if values.numel() == len(ids) and content_idx:
            subject_idx = self._subject_token_indices(labels, content_idx)
            attribute_idx = self._attribute_token_indices(labels, content_idx)
            context_idx = self._context_token_indices(labels, content_idx)
            if subject_idx:
                values[subject_idx] = values[subject_idx] * self._subject_type_prior()
            if attribute_idx:
                values[attribute_idx] = values[attribute_idx] * self._attribute_type_prior()
            if context_idx:
                values[context_idx] = values[context_idx] * self._context_type_prior()
        return values.view(1, -1)

    def _search_box_mask_from_cxcywh(self, box, resize_factor, device="cpu"):
        if box is None:
            return None
        cx, cy, w, h = [float(v) for v in box]
        crop_side = float(self.params.search_size) / max(float(resize_factor), 1e-12)
        feat_sz = int(self.feat_sz)
        centers = (torch.arange(feat_sz, dtype=torch.float32, device=device) + 0.5) * (crop_side / feat_sz)
        yy, xx = torch.meshgrid(centers, centers, indexing="ij")
        x1, x2 = cx - 0.5 * w, cx + 0.5 * w
        y1, y2 = cy - 0.5 * h, cy + 0.5 * h
        mask = (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)
        return mask.view(-1)

    def _hard_negative_indices(self, positive_mask, score_map, topk=6):
        if positive_mask is None or score_map is None:
            return None
        positive_mask = positive_mask.bool().view(-1)
        score_map = score_map.detach().float().view(-1).cpu()
        if score_map.numel() != positive_mask.numel() or not positive_mask.any():
            return None
        negative = ~positive_mask
        if not negative.any():
            return None
        topk = max(1, min(int(topk), int(negative.sum().item())))
        source = negative.nonzero(as_tuple=False).view(-1)
        local = torch.topk(score_map[source], topk).indices
        return source[local]

    def _update_language_word_reliability(self, out_dict, pred_box, score_map, resize_factor):
        self.language_word_reliability_updated = False
        self.language_word_reliability_delta = 0.0
        self.language_word_reliability_score_peak = float("nan")
        self.language_word_reliability_hardneg_peak = float("nan")
        self.language_word_reliability_score_gap = float("nan")
        if not self.language_word_reliability_active:
            return
        positive_mask = self._search_box_mask_from_cxcywh(pred_box, resize_factor, device="cpu")
        hard_idx = self._hard_negative_indices(positive_mask, score_map, topk=6)
        score_flat = score_map.detach().float().view(-1).cpu() if isinstance(score_map, torch.Tensor) else None
        score_peak = float(score_flat.max().item()) if score_flat is not None and score_flat.numel() > 0 else float("nan")
        hardneg_peak = float(score_flat[hard_idx].max().item()) if score_flat is not None and hard_idx is not None else float("nan")
        score_gap = score_peak - hardneg_peak if math.isfinite(score_peak) and math.isfinite(hardneg_peak) else float("nan")
        self.language_word_reliability_score_peak = score_peak
        self.language_word_reliability_hardneg_peak = hardneg_peak
        self.language_word_reliability_score_gap = score_gap
        if self._reliability_update_gate_enabled():
            mode = self._reliability_gate_mode()
            peak_ok = math.isfinite(score_peak) and score_peak >= self._reliability_score_threshold()
            gap_ok = math.isfinite(score_gap) and score_gap >= self._reliability_score_gap_threshold()
            if mode == "score_peak":
                update_ok = peak_ok
            elif mode == "score_gap":
                update_ok = gap_ok
            elif mode == "both":
                update_ok = peak_ok and gap_ok
            else:
                raise ValueError("Unsupported LANGUAGE_RELIABILITY_GATE_MODE: {}".format(mode))
            if not update_ok:
                return
        ids = self._language_token_ids(self.language_anchor)
        labels = self.network.backbone.tokenizer.convert_ids_to_tokens(ids)
        content_idx = self._content_token_indices(ids)
        if not content_idx:
            return
        if self.language_word_reliability is None or self.language_word_reliability.numel() != len(ids):
            self.language_word_reliability = torch.zeros(len(ids), dtype=torch.float32)
            self.language_word_reliability[content_idx] = 1.0
        old_reliability = self.language_word_reliability.clone()
        source = self._word_reliability_source()
        evidence = torch.zeros_like(self.language_word_reliability)
        if source == "word_weights":
            weights_list = out_dict.get("word_level_weights") if isinstance(out_dict, dict) else None
            if not weights_list:
                return
            weights = weights_list[-1]
            if not isinstance(weights, torch.Tensor):
                return
            weights = weights[0].detach().float().cpu().view(-1)
            if weights.numel() != len(ids):
                return
            content_weights = weights[content_idx].clamp_min(0)
            if float(content_weights.max().item()) > 0:
                content_weights = content_weights / content_weights.max().clamp_min(1e-6)
            evidence[content_idx] = content_weights
        elif source == "target_hardneg_gap":
            score_list = out_dict.get("word_level_search_token_scores") if isinstance(out_dict, dict) else None
            if not score_list:
                return
            word_scores = score_list[-1]
            if not isinstance(word_scores, torch.Tensor):
                return
            word_scores = word_scores[0].detach().float().cpu()
            if positive_mask is None or hard_idx is None or word_scores.dim() != 2:
                return
            if word_scores.shape[0] != positive_mask.numel() or word_scores.shape[1] != len(ids):
                return
            tau = max(self._word_reliability_tau(), 1e-8)
            for idx in content_idx:
                heat = word_scores[:, idx]
                pos_score = heat[positive_mask].mean()
                neg_score = heat[hard_idx].mean()
                evidence[idx] = torch.sigmoid((pos_score - neg_score) / tau)
        else:
            raise ValueError("Unsupported LANGUAGE_WORD_RELIABILITY_SOURCE: {}".format(source))

        subject_idx = self._subject_token_indices(labels, content_idx)
        context_idx = self._context_token_indices(labels, content_idx)
        if subject_idx:
            evidence[subject_idx] = evidence[subject_idx].clamp_min(self._subject_min_reliability())
        if context_idx:
            evidence[context_idx] = evidence[context_idx].clamp_max(self._context_max_weight())
        momentum = min(max(self._word_reliability_momentum(), 0.0), 1.0)
        self.language_word_reliability = (
            momentum * self.language_word_reliability + (1.0 - momentum) * evidence
        ).clamp(0.0, 1.0)
        if content_idx:
            delta = (self.language_word_reliability[content_idx] - old_reliability[content_idx]).abs().mean()
            self.language_word_reliability_delta = float(delta.item())
        self.language_word_reliability_updated = True

    def _apply_language_update(self, image, info):
        mode = self._language_update_mode()
        self.language_candidate_description = ""
        if mode == "caption_replace":
            candidate = self._generate_blip_description(image, cls=info.get("class", None))
            self.language_candidate_description = candidate
            if candidate:
                self.descript = candidate
                self.language_source = "blip_update"
        elif mode in ("anchor", "off"):
            self.descript = self.language_filtered_description or self.language_anchor
            self.language_source = "word_filter" if self.language_word_filter_active else "anchor"
        else:
            raise ValueError("Unsupported TEST.LANGUAGE_UPDATE_MODE: {}".format(mode))
        self.his_state = self.state
        self.his_image = image.copy()

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)

        # Keep an identity language anchor. BLIP is only used when the configured
        # init/update policy explicitly requests it.
        self.descript, self.language_source = self._initial_language_description(image, info)
        self.language_anchor = self.descript
        self.language_candidate_description = ""
        self._init_language_word_filter()
        self.his_state = info['init_bbox']
        self.his_image = image.copy()
        self.updata_key = False

        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            # self.z_dict1 = template
            self.memory_frames = [template.tensors]

        self.memory_masks = []
        if self.cfg.MODEL.BACKBONE.CE_LOC:  # use CE module
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.memory_masks.append(generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox))
        
        # save states
        # self.H,self.W,_ = image.shape
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    @staticmethod
    def _mean_rgb_in_box(image, box):
        if image is None or box is None:
            return None
        x, y, w, h = [float(v) for v in box]
        height, width = image.shape[:2]
        x1 = max(0, min(width, int(round(x))))
        y1 = max(0, min(height, int(round(y))))
        x2 = max(0, min(width, int(round(x + w))))
        y2 = max(0, min(height, int(round(y + h))))
        if x2 <= x1 or y2 <= y1:
            return None
        patch = image[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        return patch.reshape(-1, patch.shape[-1]).mean(axis=0)

    def ifupdata(self, his, cur, h, w, image=None):
        # Original implementation (always returns True)
        # x1,y1,w1,h1 = his
        # x2,y2,w2,h2 = cur
        # stride = 1/32
        #
        # s1,s2 = w1*h1,w2*h2
        # distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        # if s1>s2:
        #     i = s2/s1
        # else:
        #     i = s1/s2
        # if i < 0.95 :
        #     return True
        # if distance > stride*h or distance > stride*w :
        #     return True
        # return True

        # Updated implementation (paper-style: scale + center displacement).
        # Color is recorded by default and can be enabled as an additional trigger
        # after checking its distribution on the target datasets.
        x1, y1, w1, h1 = his
        x2, y2, w2, h2 = cur
        stride = self._language_trigger_distance_stride()

        s1, s2 = w1 * h1, w2 * h2
        area_ratio = min(s1, s2) / (max(s1, s2) + 1e-12)

        c1x, c1y = x1 + 0.5 * w1, y1 + 0.5 * h1
        c2x, c2y = x2 + 0.5 * w2, y2 + 0.5 * h2
        distance = math.sqrt((c1x - c2x) ** 2 + (c1y - c2y) ** 2)

        prev_rgb = self._mean_rgb_in_box(self.his_image, his)
        cur_rgb = self._mean_rgb_in_box(image, cur)
        color_delta = float("nan")
        if prev_rgb is not None and cur_rgb is not None:
            color_delta = float(np.linalg.norm(prev_rgb - cur_rgb))

        trigger_by_scale = area_ratio < self._language_trigger_scale_thr()
        trigger_by_position = distance > stride * h or distance > stride * w
        trigger_by_color = (
            self._language_trigger_color_enabled()
            and math.isfinite(color_delta)
            and color_delta > self._language_trigger_color_thr()
        )

        self.language_trigger_by_scale = bool(trigger_by_scale)
        self.language_trigger_by_position = bool(trigger_by_position)
        self.language_trigger_by_color = bool(trigger_by_color)
        self.language_trigger_area_ratio = float(area_ratio)
        self.language_trigger_center_distance = float(distance)
        self.language_trigger_color_delta = float(color_delta)

        return bool(trigger_by_scale or trigger_by_position or trigger_by_color)

    def track(self, image, info: dict = None):
        info = {} if info is None else info
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        if self.updata_key:
            self._apply_language_update(image, info)

        # print(info['num'])
        # print(self.descript)
        # --------- select memory frames ---------
        box_mask_z = None
        if self.frame_id <= self.cfg.TEST.TEMPLATE_NUMBER:
            template_list = self.memory_frames.copy()
            if self.cfg.MODEL.BACKBONE.CE_LOC:  # use CE module
                box_mask_z = torch.cat(self.memory_masks, dim=1)
        else:
            template_list, box_mask_z = self.select_memory_frames()
        # --------- select memory frames ---------

        with torch.no_grad():
            out_dict = self.network.forward(
                template=template_list, search=[search.tensors], descript=[[self.descript]],
                language_word_reliability=[self._network_language_word_reliability()])

        if isinstance(out_dict, list):
            out_dict = out_dict[-1]
        self._update_language_word_filter(out_dict)

        # A = visualize_attn(out_dict['attn'],x_patch_arr,info['path'],info['num'])
            
        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        self._update_language_word_reliability(out_dict, pred_box, pred_score_map, resize_factor)
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        self.updata_key = self.ifupdata(self.his_state, self.state, H, W, image=image)






        # --------- save memory frames and masks ---------
        z_patch_arr, z_resize_factor, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                    output_sz=self.params.template_size)
        cur_frame = self.preprocessor.process(z_patch_arr, z_amask_arr)
        frame = cur_frame.tensors
        # mask = cur_frame.mask
        if self.frame_id > self.cfg.TEST.MEMORY_THRESHOLD:
            frame = frame.detach().cpu()
            # mask = mask.detach().cpu()
        self.memory_frames.append(frame)
        if self.cfg.MODEL.BACKBONE.CE_LOC:  # use CE module
            template_bbox = self.transform_bbox_to_crop(self.state, z_resize_factor, frame.device).squeeze(1)
            self.memory_masks.append(generate_mask_cond(self.cfg, 1, frame.device, template_bbox))
        if 'pred_iou' in out_dict.keys():      # use IoU Head
            pred_iou = out_dict['pred_iou'].squeeze(-1)
            self.memory_ious.append(pred_iou)
        # --------- save memory frames and masks ---------
        
        # for debug
        # if self.debug:
        #     if not self.use_visdom:
        #         x1, y1, w, h = self.state
        #         image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #         cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
        #         save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
        #         cv2.imwrite(save_path, image_BGR)
        #     else:
        #         self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')
        #
        #         self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
        #         self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
        #         self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
        #         self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')
        #
        #         if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
        #             removed_indexes_s = out_dict['removed_indexes_s']
        #             removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
        #             masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
        #             self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')
        #
        #         while self.pause_mode:
        #             if self.step:
        #                 self.step = False
        #                 break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def select_memory_frames(self):
        num_segments = self.cfg.TEST.TEMPLATE_NUMBER
        cur_frame_idx = self.frame_id
        if num_segments != 1:
            assert cur_frame_idx > num_segments
            dur = cur_frame_idx // num_segments
            indexes = np.concatenate([
                np.array([0]),
                np.array(list(range(num_segments))) * dur + dur // 2
            ])
        else:
            indexes = np.array([0])
        indexes = np.unique(indexes)

        select_frames, select_masks = [], []
        
        for idx in indexes:
            frames = self.memory_frames[idx]
            if not frames.is_cuda:
                frames = frames.cuda()
            select_frames.append(frames)
            
            if self.cfg.MODEL.BACKBONE.CE_LOC:
                box_mask_z = self.memory_masks[idx]
                select_masks.append(box_mask_z.cuda())
        
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            return select_frames, torch.cat(select_masks, dim=1)
        else:
            return select_frames, None
    
    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights

def get_tracker_class():
    return DUTrack
