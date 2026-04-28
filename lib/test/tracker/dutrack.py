import math
import numpy as np
from lib.models.dutrack import build_dutrack
from lib.test.tracker.basetracker import BaseTracker
import torch
import torch.nn.functional as F

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.test.evaluation.environment import env_settings
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from lib.models.dutrack.i2d import descriptgenRefiner
from tracking.draw_heatmap import visualize_attn, visualize_cls_l2s_with_context


class DUTrack(BaseTracker):
    def __init__(self, params):
        super(DUTrack, self).__init__(params)
        network = build_dutrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.feat_len_s = self.network.feat_len_s
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
        self.descriptgenRefiner = descriptgenRefiner(params.cfg.MODEL.BACKBONE.BLIP_DIR,params.cfg.MODEL.BACKBONE.BERT_DIR)

        self.save_cls_l2s_vis = bool(getattr(self.cfg.TEST, 'SAVE_CLS_L2S_VIS', False)) or \
            os.environ.get('DUTRACK_SAVE_CLS_L2S', '0') == '1'
        vis_root = getattr(self.cfg.TEST, 'SAVE_CLS_L2S_VIS_DIR', '')
        if not vis_root:
            model_name = os.path.basename(os.path.dirname(self.params.checkpoint.rstrip(os.sep)))
            vis_root = os.path.join(env_settings().save_dir, 'test', 'vis_cls_l2s', model_name)
        self.cls_l2s_vis_root = vis_root
        self.last_update_metrics = None
        self.last_template_quality = None
        self.memory_frame_scores = []
        self.last_template_token_indices = None
        self.memory_template_tokens = []

    def initialize(self, image, info: dict):
        self.network.track_query = None
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)

        # Prefer the dataset-provided language description at initialization.
        # Fall back to BLIP captioning only when the dataset does not supply text.
        init_text = info.get('init_text_description')
        if init_text is not None and str(init_text).strip():
            self.descript = str(init_text).strip()
        else:
            self.descript = self.descriptgenRefiner(image, cls=info['class'])
        self.his_state = info['init_bbox']
        self.updata_key = False

        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            # self.z_dict1 = template
            self.memory_frames = [template.tensors]
            self.memory_frame_scores = [1.0]
            self.last_template_quality = 1.0
            self.last_template_token_indices = None
            self.memory_template_tokens = [None]

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

    def ifupdata(self, his, cur, h, w):
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

        # Updated implementation (paper-style: scale + center displacement)
        x1, y1, w1, h1 = his
        x2, y2, w2, h2 = cur
        stride = 1 / 32

        s1, s2 = w1 * h1, w2 * h2
        area_ratio = min(s1, s2) / (max(s1, s2) + 1e-12)

        c1x, c1y = x1 + 0.5 * w1, y1 + 0.5 * h1
        c2x, c2y = x2 + 0.5 * w2, y2 + 0.5 * h2
        distance = math.sqrt((c1x - c2x) ** 2 + (c1y - c2y) ** 2)

        triggered = False
        if area_ratio < 0.95:
            triggered = True
        if distance > stride * h or distance > stride * w:
            triggered = True

        self.last_update_metrics = {
            'triggered': triggered,
            'area_ratio': float(area_ratio),
            'distance': float(distance),
            'distance_h_thresh': float(stride * h),
            'distance_w_thresh': float(stride * w),
        }
        return triggered

    @staticmethod
    def _topk_indices(scores, k):
        k = min(int(k), int(scores.numel()))
        if k <= 0:
            return scores.new_zeros((0,), dtype=torch.long)
        return torch.topk(scores, k=k, largest=True).indices

    @staticmethod
    def _cosine_matrix(a, b, eps=1e-6):
        a = torch.nn.functional.normalize(a, dim=-1, eps=eps)
        b = torch.nn.functional.normalize(b, dim=-1, eps=eps)
        return a @ b.transpose(0, 1)

    def _build_box_mask(self, pred_box, grid_hw, crop_sz, device):
        cx, cy, w, h = [float(v) for v in pred_box]
        x1 = max(0.0, cx - 0.5 * w)
        y1 = max(0.0, cy - 0.5 * h)
        x2 = min(float(crop_sz), cx + 0.5 * w)
        y2 = min(float(crop_sz), cy + 0.5 * h)

        scale = float(grid_hw) / float(crop_sz)
        gx1 = max(0, min(grid_hw, int(math.floor(x1 * scale))))
        gy1 = max(0, min(grid_hw, int(math.floor(y1 * scale))))
        gx2 = max(0, min(grid_hw, int(math.ceil(x2 * scale))))
        gy2 = max(0, min(grid_hw, int(math.ceil(y2 * scale))))
        if gx2 <= gx1:
            gx2 = min(grid_hw, gx1 + 1)
        if gy2 <= gy1:
            gy2 = min(grid_hw, gy1 + 1)

        mask = torch.zeros((grid_hw, grid_hw), device=device, dtype=torch.float32)
        mask[gy1:gy2, gx1:gx2] = 1.0
        return mask.flatten()

    def _gaussian_center_prior(self, pred_box, grid_hw, crop_sz, device):
        cx, cy, w, h = [float(v) for v in pred_box]
        coords = torch.arange(grid_hw, device=device, dtype=torch.float32) + 0.5
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        token_cx = xx * (float(crop_sz) / float(grid_hw))
        token_cy = yy * (float(crop_sz) / float(grid_hw))
        sigma = max(0.25 * max(w, h), float(crop_sz) / max(2.0 * grid_hw, 1.0), 1.0)
        dist2 = (token_cx - cx).pow(2) + (token_cy - cy).pow(2)
        prior = torch.exp(-dist2 / (2.0 * sigma * sigma))
        return prior.flatten()

    @staticmethod
    def _masked_distribution(scores, mask):
        valid = mask > 0
        if valid.sum() <= 0:
            return None
        masked_scores = scores.masked_fill(~valid, float('-inf'))
        return F.softmax(masked_scores, dim=0)

    @staticmethod
    def _weighted_prototype(feat, weight):
        if weight is None:
            return None
        denom = weight.sum().clamp(min=1e-6)
        return (weight.unsqueeze(-1) * feat).sum(dim=0) / denom

    def _score_template_quality_legacy(
        self,
        attn_l2s,
        search_feat,
        pred_box,
        crop_sz,
        k_in=8,
        k_out=8,
        r=4,
        tau_d=0.75,
        alpha=1.0,
        beta=0.3,
        gamma=0.5,
    ):
        if attn_l2s.dim() > 1:
            attn_l2s = attn_l2s.view(-1)
        if search_feat.dim() != 2:
            search_feat = search_feat.view(-1, search_feat.shape[-1])

        num_tokens = int(search_feat.shape[0])
        grid_hw = int(round(num_tokens ** 0.5))
        if grid_hw * grid_hw != num_tokens:
            return float(attn_l2s.max().item()), None

        in_mask = self._build_box_mask(pred_box, grid_hw, crop_sz, search_feat.device)
        out_mask = 1.0 - in_mask
        if in_mask.sum() <= 0:
            in_mask = torch.ones_like(in_mask)
            out_mask = torch.zeros_like(out_mask)

        in_scores = attn_l2s * in_mask
        in_idx = self._topk_indices(in_scores, k_in)
        if in_idx.numel() == 0:
            return float(attn_l2s.max().item()), None

        target_feat = search_feat[in_idx]

        out_scores = attn_l2s * out_mask
        out_idx = self._topk_indices(out_scores, k_out)
        if out_idx.numel() > 0:
            out_feat = search_feat[out_idx]
            sim_to_target = self._cosine_matrix(out_feat, target_feat).max(dim=1).values
            dist_idx = out_idx[sim_to_target > tau_d]
        else:
            dist_idx = out_idx

        if dist_idx.numel() > 0:
            sim_to_distr = self._cosine_matrix(target_feat, search_feat[dist_idx]).max(dim=1).values
        else:
            sim_to_distr = torch.zeros(target_feat.shape[0], device=search_feat.device)

        pos_prior = self._gaussian_center_prior(pred_box, grid_hw, crop_sz, search_feat.device)[in_idx]
        norm_in_scores = in_scores[in_idx]
        norm_in_scores = norm_in_scores / norm_in_scores.max().clamp(min=1e-6)
        refined = alpha * norm_in_scores + beta * pos_prior - gamma * sim_to_distr
        keep = refined.topk(min(r, refined.numel())).indices
        selected_idx = in_idx[keep]
        score = float(refined[keep].mean().item())
        return score, selected_idx

    def _score_template_quality_prototype(
        self,
        attn_l2s,
        search_feat,
        pred_box,
        crop_sz,
        k_in=8,
        beta_split=0.35,
        r=4,
        alpha=0.6,
        beta=0.2,
        gamma=0.2,
    ):
        if attn_l2s.dim() > 1:
            attn_l2s = attn_l2s.view(-1)
        if search_feat.dim() != 2:
            search_feat = search_feat.view(-1, search_feat.shape[-1])

        num_tokens = int(search_feat.shape[0])
        grid_hw = int(round(num_tokens ** 0.5))
        if grid_hw * grid_hw != num_tokens:
            return float(attn_l2s.max().item()), None

        in_mask = self._build_box_mask(pred_box, grid_hw, crop_sz, search_feat.device)
        out_mask = 1.0 - in_mask
        if in_mask.sum() <= 0:
            in_mask = torch.ones_like(in_mask)
            out_mask = torch.zeros_like(out_mask)

        in_dist = self._masked_distribution(attn_l2s, in_mask)
        if in_dist is None:
            return float(attn_l2s.max().item()), None
        target_proto = self._weighted_prototype(search_feat, in_dist)

        out_valid = torch.nonzero(out_mask > 0, as_tuple=False).flatten()
        distractor_proto = None
        background_proto = None
        if out_valid.numel() > 0:
            out_dist = self._masked_distribution(attn_l2s, out_mask)
            out_probs = out_dist[out_valid]
            sorted_probs, order = torch.sort(out_probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=0)
            distractor_count = int((cum_probs <= beta_split).sum().item())
            if distractor_count <= 0 and sorted_probs[0] > 0:
                distractor_count = 1
            distractor_count = min(distractor_count, out_valid.numel())

            dist_idx = out_valid[order[:distractor_count]]
            bg_idx = out_valid[order[distractor_count:]]
            if dist_idx.numel() > 0:
                dist_weight = out_dist[dist_idx]
                distractor_proto = self._weighted_prototype(search_feat[dist_idx], dist_weight)
            if bg_idx.numel() > 0:
                bg_weight = out_dist[bg_idx]
                background_proto = self._weighted_prototype(search_feat[bg_idx], bg_weight)

        in_scores = attn_l2s * in_mask
        in_idx = self._topk_indices(in_scores, k_in)
        if in_idx.numel() == 0:
            return float(attn_l2s.max().item()), None

        cand_feat = search_feat[in_idx]
        target_sim = F.cosine_similarity(cand_feat, target_proto.unsqueeze(0), dim=-1)

        neg_terms = []
        if distractor_proto is not None:
            neg_terms.append(F.cosine_similarity(cand_feat, distractor_proto.unsqueeze(0), dim=-1))
        if background_proto is not None:
            neg_terms.append(F.cosine_similarity(cand_feat, background_proto.unsqueeze(0), dim=-1))
        if neg_terms:
            neg_sim = torch.stack(neg_terms, dim=0).max(dim=0).values.clamp_min(0.0)
        else:
            neg_sim = torch.zeros_like(target_sim)

        contrastive_prob = torch.exp(target_sim) / (torch.exp(target_sim) + torch.exp(neg_sim) + 1e-6)
        pos_prior = self._gaussian_center_prior(pred_box, grid_hw, crop_sz, search_feat.device)[in_idx]
        norm_in_scores = in_scores[in_idx]
        norm_in_scores = norm_in_scores / norm_in_scores.max().clamp(min=1e-6)

        refined = alpha * contrastive_prob + beta * pos_prior + gamma * norm_in_scores
        keep = refined.topk(min(r, refined.numel())).indices
        selected_idx = in_idx[keep]
        score = float(refined[keep].mean().item())
        return score, selected_idx

    def _score_template_quality(
        self,
        attn_l2s,
        search_feat,
        pred_box,
        crop_sz,
    ):
        mode = str(getattr(self.cfg.TEST, 'TEMPLATE_QUALITY_MODE', 'prototype')).lower()
        k_in = int(getattr(self.cfg.TEST, 'TEMPLATE_GATE_K_IN', 8))
        k_out = int(getattr(self.cfg.TEST, 'TEMPLATE_GATE_K_OUT', 8))
        r = int(getattr(self.cfg.TEST, 'TEMPLATE_GATE_KEEP', 4))

        if mode == 'legacy':
            return self._score_template_quality_legacy(
                attn_l2s,
                search_feat,
                pred_box,
                crop_sz,
                k_in=k_in,
                k_out=k_out,
                r=r,
            )

        return self._score_template_quality_prototype(
            attn_l2s,
            search_feat,
            pred_box,
            crop_sz,
            k_in=k_in,
            beta_split=float(getattr(self.cfg.TEST, 'TEMPLATE_PROTO_BETA', 0.35)),
            r=r,
            alpha=float(getattr(self.cfg.TEST, 'TEMPLATE_PROTO_ALPHA', 0.6)),
            beta=float(getattr(self.cfg.TEST, 'TEMPLATE_PROTO_POS', 0.2)),
            gamma=float(getattr(self.cfg.TEST, 'TEMPLATE_PROTO_ATTN', 0.2)),
        )

    @staticmethod
    def _compute_search_crop_box(target_bb, search_area_factor):
        x, y, w, h = target_bb
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
        x1 = round(x + 0.5 * w - crop_sz * 0.5)
        y1 = round(y + 0.5 * h - crop_sz * 0.5)
        return [x1, y1, crop_sz, crop_sz]

    @staticmethod
    def _merge_template_token_groups(token_groups):
        # Keep token roles separate until the backbone adds role-specific type embeddings.
        merged = {}
        for role, tokens in token_groups.items():
            valid_tokens = []
            for token in tokens:
                if token is None:
                    continue
                if not token.is_cuda:
                    token = token.cuda()
                valid_tokens.append(token)
            if valid_tokens:
                merged[role] = torch.cat(valid_tokens, dim=1)
        return merged or None

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        prev_state = list(self.state)
        search_crop_box = self._compute_search_crop_box(prev_state, self.params.search_factor)
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        if self.updata_key:
            self.descript = self.descriptgenRefiner(image,cls=info['class'])
            self.his_state = self.state

        # print(info['num'])
        # print(self.descript)
        # --------- select role-aware templates ---------
        template_list, box_mask_z, template_tokens = self.select_memory_frames()
        # --------- select role-aware templates ---------

        with torch.no_grad():
            out_dict = self.network.forward(template=template_list, search=[search.tensors],
                                            descript=[[self.descript]], template_tokens=template_tokens)

        if isinstance(out_dict, list):
            out_dict = out_dict[-1]

        # A = visualize_attn(out_dict['attn'],x_patch_arr,info['path'],info['num'])
            
        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        self.updata_key = self.ifupdata(self.his_state, self.state, H, W)

        if self.save_cls_l2s_vis and 'attn_l2s' in out_dict:
            seq_dir = os.path.join(self.cls_l2s_vis_root, info['path'])
            save_path = os.path.join(seq_dir, '{:04d}.jpg'.format(int(info['num'])))
            top_indices = None
            if 'attn_l2s' in out_dict:
                top_indices = out_dict['attn_l2s'][0].topk(self.network.token_len).indices
            metrics = self.last_update_metrics or {}
            status_lines = [
                'updatekey(prev frame): {} | area_ratio={:.4f} | distance={:.2f}'.format(
                    metrics.get('triggered', False),
                    metrics.get('area_ratio', float('nan')),
                    metrics.get('distance', float('nan')),
                ),
                'thresholds: area_ratio<0.95 or distance>max({:.2f}, {:.2f})'.format(
                    metrics.get('distance_h_thresh', float('nan')),
                    metrics.get('distance_w_thresh', float('nan')),
                )
            ]
            visualize_cls_l2s_with_context(
                out_dict['attn_l2s'][0],
                x_patch_arr,
                image,
                save_path,
                top_indices=top_indices,
                search_crop_box=search_crop_box,
                ref_box=prev_state,
                pred_box=self.state,
                description=self.descript,
                status_lines=status_lines,
            )

        template_quality = 0.0
        selected_token_idx = None
        if 'template_quality' in out_dict:
            template_quality = float(out_dict['template_quality'][0].item())
            if 'template_token_idx' in out_dict:
                selected_token_idx = out_dict['template_token_idx'][0]
            metrics = self.last_update_metrics or {}
            area_ratio = float(metrics.get('area_ratio', 1.0))
            distance = float(metrics.get('distance', 0.0))
            dist_thr = max(
                float(metrics.get('distance_h_thresh', 1.0)),
                float(metrics.get('distance_w_thresh', 1.0)),
                1.0,
            )
            box_consistency = area_ratio * math.exp(-distance / dist_thr)
            template_quality = float(template_quality * box_consistency)
        elif 'attn_l2s' in out_dict and 'backbone_feat' in out_dict:
            search_feat = out_dict['backbone_feat'][:, -self.feat_len_s:]
            pred_box_crop = pred_boxes.mean(dim=0) * self.params.search_size / resize_factor
            token_score, selected_token_idx = self._score_template_quality(
                out_dict['attn_l2s'][0],
                search_feat[0],
                pred_box_crop.detach(),
                self.params.search_size,
            )
            metrics = self.last_update_metrics or {}
            area_ratio = float(metrics.get('area_ratio', 1.0))
            distance = float(metrics.get('distance', 0.0))
            dist_thr = max(
                float(metrics.get('distance_h_thresh', 1.0)),
                float(metrics.get('distance_w_thresh', 1.0)),
                1.0,
            )
            box_consistency = area_ratio * math.exp(-distance / dist_thr)
            template_quality = float(token_score * box_consistency)
        self.last_template_quality = template_quality
        self.last_template_token_indices = selected_token_idx.detach().cpu() if selected_token_idx is not None else None
        self.memory_frame_scores.append(template_quality)

        # --------- save raw template crop, token template, and CE masks ---------
        z_patch_arr, z_resize_factor, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                    output_sz=self.params.template_size)

        cur_frame = self.preprocessor.process(z_patch_arr, z_amask_arr)
        frame = cur_frame.tensors
        # mask = cur_frame.mask
        if self.frame_id > self.cfg.TEST.MEMORY_THRESHOLD:
            frame = frame.detach().cpu()
            # mask = mask.detach().cpu()
        self.memory_frames.append(frame)
        template_token_feat = out_dict.get('template_token_feat', None)
        if template_token_feat is not None:
            token_mem = template_token_feat.detach()
            if self.frame_id > self.cfg.TEST.MEMORY_THRESHOLD:
                token_mem = token_mem.cpu()
            self.memory_template_tokens.append(token_mem)
        else:
            self.memory_template_tokens.append(None)
        if self.cfg.MODEL.BACKBONE.CE_LOC:  # use CE module
            template_bbox = self.transform_bbox_to_crop(self.state, z_resize_factor, frame.device).squeeze(1)
            self.memory_masks.append(generate_mask_cond(self.cfg, 1, frame.device, template_bbox))
        if 'pred_iou' in out_dict.keys():      # use IoU Head
            pred_iou = out_dict['pred_iou'].squeeze(-1)
            self.memory_ious.append(pred_iou)
        # --------- save raw template crop, token template, and CE masks ---------
        
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
        total_frames = len(self.memory_frames)
        scores = np.asarray(self.memory_frame_scores, dtype=np.float32)

        # Build a minimal role-aware template set:
        # init   -> first template frame
        # recent -> latest template frame
        # history -> highest-quality frames between init and recent
        init_idx = 0
        recent_idx = total_frames - 1 if total_frames > 1 and num_segments > 1 else None
        history_slots = max(num_segments - 1 - (1 if recent_idx is not None else 0), 0)

        history_indexes = []
        if recent_idx is not None and recent_idx > 1 and history_slots > 0:
            cand = np.arange(1, recent_idx, dtype=np.int64)
            cand_scores = scores[1:recent_idx]
            order = cand[np.argsort(-cand_scores, kind='stable')]
            history_indexes = order[:history_slots].tolist()

        ordered_roles = [('init', init_idx)]
        ordered_roles.extend([('history', idx) for idx in history_indexes])
        if recent_idx is not None:
            ordered_roles.append(('recent', recent_idx))

        select_frames, select_masks = [], []
        token_groups = {'init': [], 'history': [], 'recent': []}

        for role, idx in ordered_roles:
            frames = self.memory_frames[idx]
            if not frames.is_cuda:
                frames = frames.cuda()
            select_frames.append(frames)
            token_groups[role].append(self.memory_template_tokens[idx])

            if self.cfg.MODEL.BACKBONE.CE_LOC:
                box_mask_z = self.memory_masks[idx]
                select_masks.append(box_mask_z.cuda())

        merged_tokens = self._merge_template_token_groups(token_groups)
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            return select_frames, torch.cat(select_masks, dim=1), merged_tokens
        else:
            return select_frames, None, merged_tokens
    
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
