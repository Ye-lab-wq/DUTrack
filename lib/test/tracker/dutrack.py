import math
import os
import time
from collections import deque

import cv2
import numpy as np
import torch

from lib.models.dutrack import build_dutrack
from lib.models.dutrack.i2d import descriptgenRefiner
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.data_utils import Preprocessor
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from tracking.draw_heatmap import visualize_attn, visualize_cls_l2s_with_context, visualize_language_token_weights


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
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug and not self.use_visdom:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.descriptgenRefiner = descriptgenRefiner(params.cfg.MODEL.BACKBONE.BLIP_DIR,
                                                     params.cfg.MODEL.BACKBONE.BERT_DIR)
        self.vlte_vis_interval = 20

        self.language_update_frame = 0
        self.language_update_stats = None
        self.language_update_log_path = None
        self.language_update_log_dir = None
        self.language_update_history = deque(maxlen=int(getattr(self.cfg.TEST, "LANGUAGE_UPDATE_HISTORY", 50)))
        self.initial_description = ""
        self.object_class = ""

        self.occlusion_state = "visible"
        self.occlusion_score = 0
        self.occlusion_stats = {}
        self.occlusion_log_path = None
        self.occlusion_log_dir = None

    def _initial_description(self, image, info):
        text_description = info.get('init_text_description') or info.get('text_description')
        if text_description is not None and str(text_description).strip():
            return str(text_description).strip()
        return self.descriptgenRefiner(image, cls=info.get('class'))

    def _config_name(self):
        config_name = os.path.basename(os.path.dirname(self.params.checkpoint))
        if getattr(self.cfg.TEST, "CHECKPOINT_CONFIG", ""):
            config_name = getattr(self.params, "param_name", config_name)
        return config_name

    def _init_language_update_log(self, info):
        if not getattr(self.cfg.TEST, "LANGUAGE_UPDATE_ENABLE", False):
            return
        if not getattr(self.cfg.TEST, "LANGUAGE_UPDATE_LOG", True):
            return

        seq_name = str(info.get('path', 'sequence')) if info is not None else 'sequence'
        self.language_update_log_dir = os.path.join('output', 'test', 'language_update_logs', self._config_name())
        os.makedirs(self.language_update_log_dir, exist_ok=True)
        self.language_update_log_path = os.path.join(self.language_update_log_dir, '{}.txt'.format(seq_name))
        with open(self.language_update_log_path, 'w') as f:
            f.write('initial_description: {}\n'.format(self.descript))
            f.write(
                'frame\tupdate_key\tinside\toutside\tcontrast\ttop30\tinside_thr\tcontrast_thr\t'
                'history_count\tgen_time\tcrop_factor\tocclusion_state\tdecision\told_description\tcandidate\n'
            )

    def _init_occlusion_log(self, info):
        if not getattr(self.cfg.TEST, "OCCLUSION_ENABLE", False):
            return
        if not getattr(self.cfg.TEST, "OCCLUSION_LOG", True):
            return

        seq_name = str(info.get('path', 'sequence')) if info is not None else 'sequence'
        self.occlusion_log_dir = os.path.join('output', 'test', 'occlusion_state_logs', self._config_name())
        os.makedirs(self.occlusion_log_dir, exist_ok=True)
        self.occlusion_log_path = os.path.join(self.occlusion_log_dir, '{}.txt'.format(seq_name))
        with open(self.occlusion_log_path, 'w') as f:
            f.write('initial_description: {}\n'.format(self.descript))
            f.write(
                'frame\tstate\tscore\tscore_peak\tscore_entropy\tvl_inside\tvl_outside\t'
                'vl_contrast\tcenter_motion\tarea_change\tupdate_key\n'
            )

    def _maybe_rename_log(self, info, attr_path, attr_dir):
        log_path = getattr(self, attr_path)
        log_dir = getattr(self, attr_dir)
        if log_path is None or log_dir is None:
            return
        if info is None or not info.get('path'):
            return
        seq_name = str(info.get('path'))
        target_path = os.path.join(log_dir, '{}.txt'.format(seq_name))
        if target_path == log_path or os.path.exists(target_path):
            return
        os.rename(log_path, target_path)
        setattr(self, attr_path, target_path)

    def _maybe_rename_logs(self, info):
        self._maybe_rename_log(info, "language_update_log_path", "language_update_log_dir")
        self._maybe_rename_log(info, "occlusion_log_path", "occlusion_log_dir")

    def _log_language_update(self, decision, update_key=False, stats=None, old_description=None, candidate=''):
        if self.language_update_log_path is None:
            return
        if stats is None:
            stats = {}
        inside = stats.get("inside", float("nan"))
        outside = stats.get("outside", float("nan"))
        contrast = stats.get("contrast", float("nan"))
        top30 = stats.get("top30", float("nan"))
        inside_thr = stats.get("inside_thr", float("nan"))
        contrast_thr = stats.get("contrast_thr", float("nan"))
        history_count = int(stats.get("history_count", 0))
        gen_time = float(stats.get("gen_time", 0.0))
        crop_factor = float(stats.get("crop_factor", 1.0))
        old_description = self.descript if old_description is None else old_description
        line = (
            '{}\t{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{}\t'
            '{:.6f}\t{:.3f}\t{}\t{}\t{}\t{}\n'
        ).format(
            self.frame_id, int(bool(update_key)), inside, outside, contrast, top30,
            inside_thr, contrast_thr, history_count, gen_time, crop_factor,
            self.occlusion_state, decision,
            str(old_description).replace('\t', ' '), str(candidate).replace('\t', ' '))
        with open(self.language_update_log_path, 'a') as f:
            f.write(line)

    def _log_occlusion_state(self):
        if self.occlusion_log_path is None:
            return
        s = self.occlusion_stats
        line = (
            '{}\t{}\t{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t'
            '{:.6f}\t{:.6f}\t{}\n'
        ).format(
            self.frame_id, self.occlusion_state, self.occlusion_score,
            s.get("score_peak", float("nan")),
            s.get("score_entropy", float("nan")),
            s.get("vl_inside", float("nan")),
            s.get("vl_outside", float("nan")),
            s.get("vl_contrast", float("nan")),
            s.get("center_motion", float("nan")),
            s.get("area_change", float("nan")),
            int(bool(self.updata_key)))
        with open(self.occlusion_log_path, 'a') as f:
            f.write(line)

    @staticmethod
    def _language_tokens(text):
        stop_words = {
            "a", "an", "the", "of", "on", "in", "to", "and", "with", "for", "at", "is",
            "photo", "image", "picture", "tracked", "target", "close", "up", "background",
        }
        tokens, word = [], []
        for ch in str(text).lower():
            if ch.isalnum():
                word.append(ch)
            elif word:
                token = ''.join(word)
                if token not in stop_words and len(token) > 1:
                    tokens.append(token)
                word = []
        if word:
            token = ''.join(word)
            if token not in stop_words and len(token) > 1:
                tokens.append(token)
        return set(tokens)

    @staticmethod
    def _expand_language_aliases(tokens):
        alias_groups = [
            {"person", "human", "man", "woman", "boy", "girl", "people", "head", "face"},
            {"car", "vehicle", "truck", "bus", "van"},
            {"bird", "owl", "chicken", "flamingo"},
            {"dog"},
            {"panda", "bear", "mammal", "animal"},
        ]
        expanded = set(tokens)
        for group in alias_groups:
            if expanded & group:
                expanded |= group
        return expanded

    def _candidate_language_quality(self, candidate):
        if not getattr(self.cfg.TEST, "LANGUAGE_UPDATE_QUALITY_GATE", True):
            return True, "accepted"
        candidate_tokens = self._language_tokens(candidate)
        if len(candidate_tokens) < 2:
            return False, "skip_bad_language_empty"

        anchor_text = " ".join([self.initial_description, self.descript, self.object_class])
        anchor_tokens = self._expand_language_aliases(self._language_tokens(anchor_text))
        if not anchor_tokens:
            return True, "accepted"
        if candidate_tokens & anchor_tokens:
            return True, "accepted"
        return False, "skip_bad_language_drift"

    def _language_history_summary(self):
        history = list(self.language_update_history)
        if not history:
            return {"history_count": 0, "inside_thr": float("nan"), "contrast_thr": float("nan")}
        inside = np.array([item["inside"] for item in history], dtype=np.float32)
        contrast = np.array([item["contrast"] for item in history], dtype=np.float32)
        scale = float(getattr(self.cfg.TEST, "LANGUAGE_UPDATE_STD_SCALE", 0.5))
        return {
            "history_count": len(history),
            "inside_thr": float(inside.mean() + scale * inside.std()),
            "contrast_thr": float(contrast.mean() + scale * contrast.std()),
        }

    def _update_language_history(self, stats):
        self.language_update_history.append({
            "inside": float(stats["inside"]),
            "contrast": float(stats["contrast"]),
        })

    def _passes_adaptive_language_gate(self, stats):
        if not getattr(self.cfg.TEST, "LANGUAGE_UPDATE_ADAPTIVE", True):
            stats.update({
                "history_count": len(self.language_update_history),
                "inside_thr": float("nan"),
                "contrast_thr": float("nan"),
            })
            return True, stats, None

        summary = self._language_history_summary()
        stats.update(summary)
        warmup = int(getattr(self.cfg.TEST, "LANGUAGE_UPDATE_WARMUP", 20))
        if summary["history_count"] < warmup:
            return False, stats, "skip_adaptive_warmup"
        if stats["inside"] < summary["inside_thr"] or stats["contrast"] < summary["contrast_thr"]:
            return False, stats, "skip_adaptive_gate"
        return True, stats, None

    def _language_update_prompt(self, info):
        prompt = getattr(self.cfg.TEST, "LANGUAGE_UPDATE_PROMPT", "")
        class_name = None if info is None else info.get('class', None)
        if prompt:
            return prompt.format(class_name=class_name or "target", description=self.descript)
        return class_name

    def _tracking_description(self):
        if not getattr(self.cfg.TEST, "OCCLUSION_USE_STATE_PROMPT", False):
            return self.descript
        if self.occlusion_state == "partial_occluded":
            return '{}. target is partially occluded'.format(self.descript)
        if self.occlusion_state == "heavy_occluded":
            return '{}. target is heavily occluded'.format(self.descript)
        return '{}. target is visible'.format(self.descript)

    def _crop_target_image(self, image, target_box):
        h, w = image.shape[:2]
        x, y, bw, bh = [float(v) for v in target_box]
        crop_factor = max(1.0, float(getattr(self.cfg.TEST, "LANGUAGE_UPDATE_CROP_FACTOR", 1.0)))
        if crop_factor > 1.0:
            cx = x + 0.5 * bw
            cy = y + 0.5 * bh
            bw *= crop_factor
            bh *= crop_factor
            x = cx - 0.5 * bw
            y = cy - 0.5 * bh
        x1 = max(0, int(math.floor(x)))
        y1 = max(0, int(math.floor(y)))
        x2 = min(w, int(math.ceil(x + bw)))
        y2 = min(h, int(math.ceil(y + bh)))
        if x2 <= x1 or y2 <= y1:
            return image
        return image[y1:y2, x1:x2]

    def _search_crop_box(self, target_box, resize_factor):
        crop_sz = self.params.search_size / resize_factor
        x, y, w, h = [float(v) for v in target_box]
        return [x + 0.5 * w - 0.5 * crop_sz, y + 0.5 * h - 0.5 * crop_sz, crop_sz, crop_sz]

    def _vl_score_language_stats(self, out_dict, prev_state, pred_state, resize_factor):
        if 'vl_score_x' not in out_dict:
            return None

        score = out_dict['vl_score_x'][0].detach().float().cpu()
        grid_size = int(math.sqrt(score.numel()))
        if grid_size * grid_size != score.numel():
            return None

        crop_x, crop_y, crop_w, crop_h = self._search_crop_box(prev_state, resize_factor)
        x, y, w, h = [float(v) for v in pred_state]
        x1 = (x - crop_x) / max(crop_w, 1e-6)
        y1 = (y - crop_y) / max(crop_h, 1e-6)
        x2 = (x + w - crop_x) / max(crop_w, 1e-6)
        y2 = (y + h - crop_y) / max(crop_h, 1e-6)

        coords = torch.arange(grid_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords)
        token_x = (xx.flatten() + 0.5) / grid_size
        token_y = (yy.flatten() + 0.5) / grid_size
        inside = (token_x >= x1) & (token_x <= x2) & (token_y >= y1) & (token_y <= y2)
        if inside.sum() == 0:
            cx = min(max((x1 + x2) * 0.5, 0.0), 1.0)
            cy = min(max((y1 + y2) * 0.5, 0.0), 1.0)
            nearest = ((token_x - cx) ** 2 + (token_y - cy) ** 2).argmin()
            inside[nearest] = True

        outside = ~inside
        inside_mean = score[inside].mean().item()
        outside_mean = score[outside].mean().item() if outside.any() else 0.0
        contrast = inside_mean - outside_mean
        top_k = max(1, int(round(score.numel() * 0.3)))
        top_mean = torch.topk(score, top_k).values.mean().item()
        return {"inside": inside_mean, "outside": outside_mean, "contrast": contrast, "top30": top_mean}

    @staticmethod
    def _box_motion_stats(prev_state, cur_state, image_h, image_w):
        px, py, pw, ph = [float(v) for v in prev_state]
        cx, cy, cw, ch = [float(v) for v in cur_state]
        pcx, pcy = px + 0.5 * pw, py + 0.5 * ph
        ccx, ccy = cx + 0.5 * cw, cy + 0.5 * ch
        diag = max(math.sqrt(image_h ** 2 + image_w ** 2), 1e-6)
        center_motion = math.sqrt((pcx - ccx) ** 2 + (pcy - ccy) ** 2) / diag
        prev_area = max(pw * ph, 1e-6)
        cur_area = max(cw * ch, 1e-6)
        area_change = 1.0 - min(prev_area, cur_area) / max(prev_area, cur_area)
        return center_motion, area_change

    def _score_map_stats(self, out_dict):
        if 'score_map' not in out_dict:
            return float("nan"), float("nan")
        score = out_dict['score_map'][0].detach().float().cpu().view(-1)
        score_peak = score.max().item()
        prob = score.clamp_min(0)
        total = prob.sum().item()
        if total <= 1e-12:
            return score_peak, 1.0
        prob = prob / total
        entropy = float(-(prob * (prob + 1e-12).log()).sum().item() / math.log(prob.numel()))
        return score_peak, entropy

    def _estimate_occlusion_state(self, out_dict, prev_state, pred_state, image_h, image_w, resize_factor):
        if not getattr(self.cfg.TEST, "OCCLUSION_ENABLE", False):
            self.occlusion_state = "visible"
            self.occlusion_score = 0
            self.occlusion_stats = {}
            return

        vl_stats = self._vl_score_language_stats(out_dict, prev_state, pred_state, resize_factor)
        score_peak, score_entropy = self._score_map_stats(out_dict)
        center_motion, area_change = self._box_motion_stats(prev_state, pred_state, image_h, image_w)

        min_inside = float(getattr(self.cfg.TEST, "OCCLUSION_MIN_VL_INSIDE", 0.20))
        min_contrast = float(getattr(self.cfg.TEST, "OCCLUSION_MIN_VL_CONTRAST", 0.05))
        high_entropy = float(getattr(self.cfg.TEST, "OCCLUSION_HIGH_ENTROPY", 0.85))
        motion_thr = float(getattr(self.cfg.TEST, "OCCLUSION_MOTION_THR", 0.08))
        area_thr = float(getattr(self.cfg.TEST, "OCCLUSION_AREA_CHANGE_THR", 0.35))

        inside = float("nan") if vl_stats is None else float(vl_stats["inside"])
        outside = float("nan") if vl_stats is None else float(vl_stats["outside"])
        contrast = float("nan") if vl_stats is None else float(vl_stats["contrast"])

        evidence = 0
        if vl_stats is not None and inside < min_inside:
            evidence += 1
        if vl_stats is not None and contrast < min_contrast:
            evidence += 1
        if not math.isnan(score_entropy) and score_entropy > high_entropy:
            evidence += 1
        if center_motion > motion_thr:
            evidence += 1
        if area_change > area_thr:
            evidence += 1

        partial_thr = int(getattr(self.cfg.TEST, "OCCLUSION_PARTIAL_THR", 2))
        heavy_thr = int(getattr(self.cfg.TEST, "OCCLUSION_HEAVY_THR", 3))
        if evidence >= heavy_thr:
            state = "heavy_occluded"
        elif evidence >= partial_thr:
            state = "partial_occluded"
        else:
            state = "visible"

        self.occlusion_state = state
        self.occlusion_score = evidence
        self.occlusion_stats = {
            "score_peak": score_peak,
            "score_entropy": score_entropy,
            "vl_inside": inside,
            "vl_outside": outside,
            "vl_contrast": contrast,
            "center_motion": center_motion,
            "area_change": area_change,
        }

    def _maybe_update_language(self, image, info, out_dict, prev_state, pred_state, resize_factor):
        if not getattr(self.cfg.TEST, "LANGUAGE_UPDATE_ENABLE", False):
            return
        if getattr(self.cfg.TEST, "OCCLUSION_PAUSE_LANGUAGE_UPDATE", False) and self.occlusion_state != "visible":
            self._log_language_update('skip_occlusion_state', update_key=self.updata_key, stats=self.language_update_stats)
            return
        if not self.updata_key:
            self._log_language_update('skip_updatekey_false', update_key=False)
            return

        stats = self._vl_score_language_stats(out_dict, prev_state, pred_state, resize_factor)
        self.language_update_stats = stats
        if stats is None:
            self._log_language_update('skip_no_vlscore', update_key=True)
            return
        adaptive_pass, stats, adaptive_reason = self._passes_adaptive_language_gate(stats)
        self._update_language_history(stats)
        min_interval = int(getattr(self.cfg.TEST, "LANGUAGE_UPDATE_MIN_INTERVAL", 20))
        if self.frame_id - self.language_update_frame < min_interval:
            self._log_language_update('skip_interval', update_key=True, stats=stats)
            return
        min_inside = float(getattr(self.cfg.TEST, "LANGUAGE_UPDATE_MIN_INSIDE", 0.45))
        min_contrast = float(getattr(self.cfg.TEST, "LANGUAGE_UPDATE_MIN_CONTRAST", 0.05))
        if stats["inside"] < min_inside or stats["contrast"] < min_contrast:
            self._log_language_update('skip_low_consistency', update_key=True, stats=stats)
            return
        if not adaptive_pass:
            self._log_language_update(adaptive_reason, update_key=True, stats=stats)
            return

        stats["crop_factor"] = max(1.0, float(getattr(self.cfg.TEST, "LANGUAGE_UPDATE_CROP_FACTOR", 1.0)))
        target_crop = self._crop_target_image(image, pred_state)
        old_description = self.descript
        start_time = time.perf_counter()
        candidate = self.descriptgenRefiner(target_crop, cls=self._language_update_prompt(info))
        stats["gen_time"] = time.perf_counter() - start_time
        self.language_update_frame = self.frame_id
        candidate = str(candidate).strip()
        quality_pass, quality_reason = self._candidate_language_quality(candidate)
        if not quality_pass:
            self._log_language_update(quality_reason, update_key=True, stats=stats,
                                      old_description=old_description, candidate=candidate)
            return
        if candidate and candidate.lower() != str(self.descript).strip().lower():
            self.descript = candidate
            self._log_language_update('accepted', update_key=True, stats=stats,
                                      old_description=old_description, candidate=candidate)
        else:
            self._log_language_update('skip_same_or_empty', update_key=True, stats=stats,
                                      old_description=old_description, candidate=candidate)

    def initialize(self, image, info: dict):
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'],
                                                                self.params.template_factor,
                                                                output_sz=self.params.template_size)

        self.descript = self._initial_description(image, info)
        self.initial_description = self.descript
        self.object_class = str(info.get('class', '')) if info is not None else ''
        self.his_state = info['init_bbox']
        self.updata_key = False

        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.memory_frames = [template.tensors]

        self.memory_masks = []
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.memory_masks.append(generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox))

        self.state = info['init_bbox']
        self.frame_id = 0
        self.language_update_frame = 0
        self.language_update_stats = None
        self.language_update_log_path = None
        self.language_update_log_dir = None
        self.language_update_history = deque(maxlen=int(getattr(self.cfg.TEST, "LANGUAGE_UPDATE_HISTORY", 50)))
        self.occlusion_state = "visible"
        self.occlusion_score = 0
        self.occlusion_stats = {}
        self.occlusion_log_path = None
        self.occlusion_log_dir = None
        self._init_language_update_log(info)
        self._init_occlusion_log(info)
        if self.save_all_boxes:
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def ifupdata(self, his, cur, h, w):
        x1, y1, w1, h1 = his
        x2, y2, w2, h2 = cur
        stride = 1 / 32

        s1, s2 = w1 * h1, w2 * h2
        area_ratio = min(s1, s2) / (max(s1, s2) + 1e-12)

        c1x, c1y = x1 + 0.5 * w1, y1 + 0.5 * h1
        c2x, c2y = x2 + 0.5 * w2, y2 + 0.5 * h2
        distance = math.sqrt((c1x - c2x) ** 2 + (c1y - c2y) ** 2)

        if area_ratio < 0.95:
            return True
        if distance > stride * h or distance > stride * w:
            return True
        return False

    def _maybe_save_vlte_vis(self, out_dict, search_img, orig_img, info, prev_state, pred_state, resize_factor):
        if self.debug < 2:
            return
        if self.frame_id > 5 and self.frame_id % self.vlte_vis_interval != 0:
            return

        seq_name = str(info.get('path', 'sequence')) if info is not None else 'sequence'
        save_dir = os.path.join('output', 'test', 'vis_vlte', self._config_name(), seq_name)
        crop_box = self._search_crop_box(prev_state, resize_factor)
        status = [
            'frame: {}'.format(self.frame_id),
            'prev: [{:.1f}, {:.1f}, {:.1f}, {:.1f}]'.format(*prev_state),
            'pred: [{:.1f}, {:.1f}, {:.1f}, {:.1f}]'.format(*pred_state),
            'occ: {} ({})'.format(self.occlusion_state, self.occlusion_score),
        ]

        if 'vl_score_x' in out_dict:
            vl_score = out_dict['vl_score_x'][0].detach().float().cpu()
            save_path = os.path.join(save_dir, '{:04d}_vl_score_x.jpg'.format(self.frame_id))
            visualize_cls_l2s_with_context(vl_score, search_img, orig_img, save_path,
                                           search_crop_box=crop_box, ref_box=prev_state, pred_box=pred_state,
                                           description=self.descript, status_lines=status, title='VL score x')

        if 'vl_te_mask_x' in out_dict:
            te_mask = out_dict['vl_te_mask_x'][0].detach().float().cpu()
            save_path = os.path.join(save_dir, '{:04d}_vl_te_mask_x.jpg'.format(self.frame_id))
            visualize_cls_l2s_with_context(te_mask, search_img, orig_img, save_path,
                                           search_crop_box=crop_box, ref_box=prev_state, pred_box=pred_state,
                                           description=self.descript, status_lines=status, title='Learned TE mask x')

        if 'attn_l2s' in out_dict:
            attn_l2s = out_dict['attn_l2s'][0].detach().float().cpu()
            save_path = os.path.join(save_dir, '{:04d}_attn_l2s.jpg'.format(self.frame_id))
            visualize_cls_l2s_with_context(attn_l2s, search_img, orig_img, save_path,
                                           search_crop_box=crop_box, ref_box=prev_state, pred_box=pred_state,
                                           description=self.descript, status_lines=status,
                                           title='Temporal/query attention to search')

        if 'score_map' in out_dict:
            score_map = out_dict['score_map'][0].detach().float().cpu().view(-1)
            save_path = os.path.join(save_dir, '{:04d}_score_map.jpg'.format(self.frame_id))
            visualize_cls_l2s_with_context(score_map, search_img, orig_img, save_path,
                                           search_crop_box=crop_box, ref_box=prev_state, pred_box=pred_state,
                                           description=self.descript, status_lines=status, title='Box score map')

        if 'language_token_weights' in out_dict and 'language_tokens' in out_dict:
            token_weights = out_dict['language_token_weights'][0].detach().float().cpu()
            tokens = out_dict['language_tokens'][0]
            save_path = os.path.join(save_dir, '{:04d}_language_tokens.jpg'.format(self.frame_id))
            visualize_language_token_weights(tokens, token_weights, save_path,
                                             description=self.descript, status_lines=status,
                                             title='Visual-guided language token weights')

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        self._maybe_rename_logs(info)
        prev_state = list(self.state)
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        if self.updata_key:
            self.his_state = self.state

        box_mask_z = None
        if self.frame_id <= self.cfg.TEST.TEMPLATE_NUMBER:
            template_list = self.memory_frames.copy()
            if self.cfg.MODEL.BACKBONE.CE_LOC:
                box_mask_z = torch.cat(self.memory_masks, dim=1)
        else:
            template_list, box_mask_z = self.select_memory_frames()

        with torch.no_grad():
            out_dict = self.network.forward(template=template_list, search=[search.tensors],
                                            descript=[[self._tracking_description()]])

        if isinstance(out_dict, list):
            out_dict = out_dict[-1]

        if self.cfg.MODEL.HEAD.TYPE == "CORNER":
            pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        else:
            pred_score_map = out_dict['score_map']
            response = self.output_window * pred_score_map
            pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
            pred_boxes = pred_boxes.view(-1, 4)

        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        self.updata_key = self.ifupdata(self.his_state, self.state, H, W)
        self._estimate_occlusion_state(out_dict, prev_state, self.state, H, W, resize_factor)
        self._log_occlusion_state()
        self._maybe_save_vlte_vis(out_dict, x_patch_arr, image, info, prev_state, self.state, resize_factor)
        self._maybe_update_language(image, info, out_dict, prev_state, self.state, resize_factor)

        pause_template = (
            getattr(self.cfg.TEST, "OCCLUSION_PAUSE_TEMPLATE_UPDATE", False)
            and self.occlusion_state != "visible"
        )
        if not pause_template:
            z_patch_arr, z_resize_factor, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                      output_sz=self.params.template_size)
            cur_frame = self.preprocessor.process(z_patch_arr, z_amask_arr)
            frame = cur_frame.tensors
            if self.frame_id > self.cfg.TEST.MEMORY_THRESHOLD:
                frame = frame.detach().cpu()
            self.memory_frames.append(frame)
            if self.cfg.MODEL.BACKBONE.CE_LOC:
                template_bbox = self.transform_bbox_to_crop(self.state, z_resize_factor, frame.device).squeeze(1)
                self.memory_masks.append(generate_mask_cond(self.cfg, 1, frame.device, template_bbox))
            if 'pred_iou' in out_dict.keys():
                pred_iou = out_dict['pred_iou'].squeeze(-1)
                self.memory_ious.append(pred_iou)

        if self.save_all_boxes:
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()
            return {"target_bbox": self.state, "all_boxes": all_boxes_save}
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
        return select_frames, None

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return DUTrack
