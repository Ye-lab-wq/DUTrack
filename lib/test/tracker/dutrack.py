import math
import random
import numpy as np
from lib.models.dutrack import build_dutrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from lib.models.dutrack.i2d import descriptgenRefiner
from lib.models.dutrack.language_masks import build_evidence_anchor_mask, build_semantic_token_mask
from tracking.draw_heatmap import visualize_attn


WRONG_LANG_POOL = {
    # OTB-Lang classes
    "person": "a red car driving on the highway",
    "person head": "a blue bicycle parked on the sidewalk",
    "car": "a person walking across the street",
    "face": "a white bird flying under the cloud",
    "bird": "a human face looking at the screen",
    "dog": "a silver airplane landing on the runway",
    "mammal": "a small drone hovering above the water",
    "other": "a black dog running through the grass",
    "vehicle": "a young man riding a skateboard",
    "cat": "a fish swimming in the ocean",
    # HOOT classes (class_name from dataset with underscores replaced by spaces)
    "apple": "a blue backpack on the table",
    "avocado": "a red bicycle parked outside",
    "backpack": "a green apple in the basket",
    "banana": "a white clock on the wall",
    "bear": "a yellow lemon on the plate",
    "bicycle": "a brown bear in the forest",
    "book": "a silver spoon on the napkin",
    "bottle": "a small turtle swimming",
    "bowl": "a black keyboard on the desk",
    "camel": "a striped zebra running",
    "carrot": "a ceramic cup on the shelf",
    "clock": "a ripe orange on the tree",
    "coaster": "a wool coat hanging",
    "coat": "a tennis ball bouncing",
    "coin": "a plush toy on the bed",
    "crocodile": "a wine glass on the bar",
    "cup": "a carrot stick on the cutting board",
    "deer": "a purple donut with sprinkles",
    "donut": "a camel walking in the desert",
    "egg": "a pair of scissors cutting paper",
    "elephant": "a small mouse on the cheese",
    "eye glasses": "an ostrich running fast",
    "fish": "a kangaroo hopping in the grass",
    "flamingo": "a sea lion resting on the rock",
    "fork": "a toothbrush in the holder",
    "giraffe": "a crocodile floating in the water",
    "hand": "a remote control on the couch",
    "hat": "a bottle of water on the counter",
    "kangaroo": "a flamingo standing in the water",
    "keyboard": "a potted plant on the windowsill",
    "kiwi": "a sports ball on the field",
    "knife": "an umbrella open on the beach",
    "koala": "a giraffe eating leaves",
    "lemon": "a penguin sliding on ice",
    "mouse": "a deer grazing in the meadow",
    "mouse animal": "a fork next to the plate",
    "orange": "a knife on the cutting board",
    "ostrich": "a gold coin on the table",
    "paper": "a lemon slice on the drink",
    "pen": "an egg in the frying pan",
    "penguin": "a kiwi fruit cut in half",
    "phone": "a toilet paper roll on the shelf",
    "plate": "a scissors cutting fabric",
    "plushie toy": "a shoe on the rack",
    "potted plant": "a cell phone on the desk",
    "purse": "a bowl of soup on the table",
    "rag": "an avocado cut in half",
    "remote": "a rhino charging forward",
    "rhino": "a hand waving hello",
    "rubiks cube": "a donut with pink frosting",
    "scissors": "a purse on the chair",
    "sea lion": "a hat on the coat rack",
    "shoe": "an eye glasses case open",
    "spoon": "a plate of pasta",
    "sports ball": "a rag on the counter",
    "toilet paper": "a phone charging on the desk",
    "toothbrush": "a book open on the table",
    "turtle": "a rubiks cube solved",
    "umbrella": "a fork on the napkin",
    "vase": "a clock ticking loudly",
    "wine glass": "a coaster under the cup",
    "zebra": "a fish swimming in the tank",
    "electric fan": "a red apple on the table",
    "frog": "a yellow banana on the counter",
    "gazelle": "a blue backpack beside a chair",
    "gorilla": "a silver spoon on the napkin",
    "leopard": "a ceramic cup on the shelf",
    "lizard": "a white clock on the wall",
    "motorcycle": "a green potted plant near the window",
    "robot": "a ripe orange on the table",
}


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
        self.descriptgenRefiner = descriptgenRefiner(params.cfg.MODEL.BACKBONE.BLIP_DIR,params.cfg.MODEL.BACKBONE.BERT_DIR)
        self.lang_mode = getattr(params.cfg.TEST, 'LANG_MODE', 'normal')
        self.enable_diagnostics = getattr(params, 'enable_diagnostics', False)
        if self.enable_diagnostics and getattr(self.network, "evidence_layer", None) is not None:
            self.network.evidence_layer.enable_diagnostics = True
        if self.enable_diagnostics and getattr(self.network, "evidence_unit_layer", None) is not None:
            self.network.evidence_unit_layer.enable_diagnostics = True

    def _perturb_language(self, descript, object_class=None):
        if self.lang_mode == 'normal':
            return descript
        elif self.lang_mode == 'shuffle':
            words = descript.split()
            if len(words) <= 1:
                return descript
            seed = sum(ord(ch) for ch in descript)
            random.Random(seed).shuffle(words)
            return ' '.join(words)
        elif self.lang_mode == 'wrong':
            if object_class and object_class in WRONG_LANG_POOL:
                return WRONG_LANG_POOL[object_class]
            pool = sorted(WRONG_LANG_POOL.values())
            key = object_class or descript or "unknown"
            index = sum(ord(ch) for ch in key) % len(pool)
            return pool[index]
        elif self.lang_mode == 'generic':
            return "a moving object in the scene"
        elif self.lang_mode == 'no_update':
            return descript
        return descript

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)

        #update descript: use dataset-provided text if available, otherwise BLIP
        if 'init_text_description' in info and info['init_text_description']:
            raw_descript = info['init_text_description']
        else:
            raw_descript = self.descriptgenRefiner(image, cls=info.get('class', None))
        self.descript = self._perturb_language(raw_descript, info.get('class', None))
        self.his_state = info['init_bbox']
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

        if area_ratio < 0.95:
            return True
        if distance > stride * h or distance > stride * w:
            return True
        return False

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        search_anchor_state = self.state
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        if self.updata_key:
            raw_descript = self.descriptgenRefiner(image,cls=info['class'])
            self.descript = self._perturb_language(raw_descript, info.get('class', None))
            self.his_state = self.state

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
            out_dict = self.network.forward(template=template_list, search=[search.tensors],descript=[[self.descript]])

        if isinstance(out_dict, list):
            out_dict = out_dict[-1]

        # A = visualize_attn(out_dict['attn'],x_patch_arr,info['path'],info['num'])
            
        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        diagnostics = self._compute_score_diagnostics(
            pred_score_map, response, resize_factor, search_anchor_state, info, out_dict=out_dict)
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        if self.lang_mode == 'no_update':
            self.updata_key = False
        else:
            self.updata_key = self.ifupdata(self.his_state,self.state,H,W)






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
            output = {"target_bbox": self.state,
                      "all_boxes": all_boxes_save}
        else:
            output = {"target_bbox": self.state}
        if diagnostics is not None:
            output["diagnostics"] = diagnostics
        return output

    def _compute_score_diagnostics(self, score_map, response, resize_factor, search_anchor_state, info, out_dict=None):
        if not self.enable_diagnostics or info is None or 'gt_bbox' not in info:
            return None

        gt_bbox = info['gt_bbox']
        if gt_bbox is None:
            return None

        score = score_map.detach().float().view(-1, self.feat_sz, self.feat_sz)[0]
        response_map = response.detach().float().view(-1, self.feat_sz, self.feat_sz)[0]
        device = response_map.device

        gt = torch.tensor(gt_bbox, dtype=torch.float32, device=device)
        if gt[2].item() <= 0 or gt[3].item() <= 0:
            return None

        anchor = torch.tensor(search_anchor_state, dtype=torch.float32, device=device)
        crop_sz = torch.tensor([self.params.search_size, self.params.search_size],
                               dtype=torch.float32, device=device)
        gt_crop = transform_image_to_crop(gt, anchor, resize_factor, crop_sz, normalize=False)

        x1, y1, w, h = gt_crop.tolist()
        x2, y2 = x1 + w, y1 + h
        if x2 <= 0 or y2 <= 0 or x1 >= self.params.search_size or y1 >= self.params.search_size:
            return None

        scale = self.feat_sz / float(self.params.search_size)
        gx1 = int(math.floor(max(0.0, x1) * scale))
        gy1 = int(math.floor(max(0.0, y1) * scale))
        gx2 = int(math.ceil(min(float(self.params.search_size), x2) * scale)) - 1
        gy2 = int(math.ceil(min(float(self.params.search_size), y2) * scale)) - 1
        gx1 = max(0, min(self.feat_sz - 1, gx1))
        gy1 = max(0, min(self.feat_sz - 1, gy1))
        gx2 = max(0, min(self.feat_sz - 1, gx2))
        gy2 = max(0, min(self.feat_sz - 1, gy2))
        if gx2 < gx1 or gy2 < gy1:
            return None

        target_mask = torch.zeros((self.feat_sz, self.feat_sz), dtype=torch.bool, device=device)
        target_mask[gy1:gy2 + 1, gx1:gx2 + 1] = True

        ignore_mask = target_mask.clone()
        pad = 1
        ignore_mask[max(0, gy1 - pad):min(self.feat_sz, gy2 + pad + 1),
                    max(0, gx1 - pad):min(self.feat_sz, gx2 + pad + 1)] = True
        hard_negative_mask = ~ignore_mask
        if not hard_negative_mask.any():
            return None

        gt_response_max = response_map[target_mask].max()
        hard_response_max = response_map[hard_negative_mask].max()
        gt_score_max = score[target_mask].max()
        hard_score_max = score[hard_negative_mask].max()

        peak_index = int(response_map.flatten().argmax().item())
        peak_y = peak_index // self.feat_sz
        peak_x = peak_index % self.feat_sz

        diagnostics = {
            "gt_response_max": float(gt_response_max.item()),
            "hard_negative_response_max": float(hard_response_max.item()),
            "hard_negative_gap": float((gt_response_max - hard_response_max).item()),
            "peak_inside_gt": int(target_mask[peak_y, peak_x].item()),
            "gt_score_max": float(gt_score_max.item()),
            "hard_negative_score_max": float(hard_score_max.item()),
            "hard_negative_score_gap": float((gt_score_max - hard_score_max).item()),
            "gt_grid_x1": gx1,
            "gt_grid_y1": gy1,
            "gt_grid_x2": gx2,
            "gt_grid_y2": gy2,
            "peak_x": peak_x,
            "peak_y": peak_y,
            "gt_crop_x": float(x1),
            "gt_crop_y": float(y1),
            "gt_crop_w": float(w),
            "gt_crop_h": float(h),
        }
        self._add_stage2_evidence_diagnostics(
            diagnostics, out_dict, target_mask, hard_negative_mask)
        return diagnostics

    def _add_stage2_evidence_diagnostics(self, diagnostics, out_dict, target_mask, hard_negative_mask):
        if out_dict is None:
            return

        for key, prefix in [
                ("stage2_diag_evidence_scalar", "stage2_evidence"),
                ("stage2_diag_calibration_scalar", "stage2_calibration"),
                ("stage2_diag_strength", "stage2_strength"),
                ("stage2r_diag_evidence_scalar", "stage2r_evidence"),
                ("stage2r_diag_calibration_scalar", "stage2r_calibration"),
                ("stage2r_diag_strength", "stage2r_strength")]:
            if key not in out_dict:
                continue
            metric = out_dict[key].detach().float().view(-1, self.feat_sz, self.feat_sz)[0]
            target_values = metric[target_mask]
            hard_values = metric[hard_negative_mask]
            if target_values.numel() == 0 or hard_values.numel() == 0:
                continue
            diagnostics.update({
                "{}_target_mean".format(prefix): float(target_values.mean().item()),
                "{}_hard_negative_mean".format(prefix): float(hard_values.mean().item()),
                "{}_mean_gap".format(prefix): float((target_values.mean() - hard_values.mean()).item()),
                "{}_target_max".format(prefix): float(target_values.max().item()),
                "{}_hard_negative_max".format(prefix): float(hard_values.max().item()),
                "{}_max_gap".format(prefix): float((target_values.max() - hard_values.max()).item()),
            })

        if "stage2_diag_attention" in out_dict:
            attn = out_dict["stage2_diag_attention"].detach().float()
            if attn.dim() == 3 and attn.shape[1] == self.feat_sz * self.feat_sz:
                flat_target = target_mask.flatten()
                flat_hard = hard_negative_mask.flatten()
                target_attn = attn[0, flat_target].mean(dim=0)
                hard_attn = attn[0, flat_hard].mean(dim=0)
                tokens = self._current_semantic_tokens(target_attn.shape[0])
                diagnostics.update(self._format_top_tokens(
                    target_attn, tokens, "stage2_attn_target"))
                diagnostics.update(self._format_top_tokens(
                    hard_attn, tokens, "stage2_attn_hard_negative"))

        if "stage2r_diag_attention" in out_dict:
            attn = out_dict["stage2r_diag_attention"].detach().float()
            if attn.dim() == 3 and attn.shape[1] == self.feat_sz * self.feat_sz:
                flat_target = target_mask.flatten()
                flat_hard = hard_negative_mask.flatten()
                target_attn = attn[0, flat_target].mean(dim=0)
                hard_attn = attn[0, flat_hard].mean(dim=0)
                units = self._current_evidence_units(target_attn.shape[0])
                diagnostics.update(self._format_top_tokens(
                    target_attn, units, "stage2r_attn_target"))
                diagnostics.update(self._format_top_tokens(
                    hard_attn, units, "stage2r_attn_hard_negative"))

    def _current_semantic_tokens(self, token_len):
        encoded = self.descriptgenRefiner.tokenizer(
            self.descript,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=token_len,
            return_special_tokens_mask=True,
        )
        token_ids = encoded["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        attention_mask = encoded["attention_mask"]
        if attention_mask and isinstance(attention_mask[0], list):
            attention_mask = attention_mask[0]
        special_mask = encoded["special_tokens_mask"]
        if special_mask and isinstance(special_mask[0], list):
            special_mask = special_mask[0]

        tokens = self.descriptgenRefiner.tokenizer.convert_ids_to_tokens(token_ids)
        valid_token_mask = torch.tensor([[
            bool(valid) and not bool(special)
            for valid, special in zip(attention_mask, special_mask)
        ]], dtype=torch.bool)
        semantic_mask = build_semantic_token_mask(
            self.descriptgenRefiner.tokenizer,
            [token_ids],
            valid_token_mask,
        )[0].tolist()
        cleaned = []
        for token, valid, special, semantic in zip(tokens, attention_mask, special_mask, semantic_mask):
            if valid and not special and semantic:
                cleaned.append(token)
            else:
                cleaned.append("[MASKED]")
        if len(cleaned) < token_len:
            cleaned.extend(["[MISSING]"] * (token_len - len(cleaned)))
        return cleaned[:token_len]

    def _current_evidence_units(self, token_len):
        encoded = self.descriptgenRefiner.tokenizer(
            self.descript,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=token_len,
            return_special_tokens_mask=True,
        )
        token_ids = encoded["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        attention_mask = encoded["attention_mask"]
        if attention_mask and isinstance(attention_mask[0], list):
            attention_mask = attention_mask[0]
        special_mask = encoded["special_tokens_mask"]
        if special_mask and isinstance(special_mask[0], list):
            special_mask = special_mask[0]

        tokens = self.descriptgenRefiner.tokenizer.convert_ids_to_tokens(token_ids)
        valid_token_mask = torch.tensor([[
            bool(valid) and not bool(special)
            for valid, special in zip(attention_mask, special_mask)
        ]], dtype=torch.bool)
        anchor_mask = build_evidence_anchor_mask(
            self.descriptgenRefiner.tokenizer,
            [token_ids],
            valid_token_mask,
        )[0].tolist()
        valid_tokens = valid_token_mask[0].tolist()

        evidence_layer = getattr(self.network, "evidence_unit_layer", None)
        phrase_window = getattr(evidence_layer, "phrase_window", 3)
        radius = int(phrase_window) // 2

        units = []
        for index, (token, is_valid, is_anchor) in enumerate(zip(tokens, valid_tokens, anchor_mask)):
            if not is_valid or not is_anchor:
                units.append("[MASKED]")
                continue

            start = max(0, index - radius)
            end = min(len(tokens), index + radius + 1)
            unit_tokens = [
                tokens[position]
                for position in range(start, end)
                if valid_tokens[position]
            ]
            units.append(self._clean_wordpiece_tokens(unit_tokens))

        if len(units) < token_len:
            units.extend(["[MISSING]"] * (token_len - len(units)))
        return units[:token_len]

    @staticmethod
    def _clean_wordpiece_tokens(tokens):
        pieces = []
        for token in tokens:
            if token.startswith("##") and pieces:
                pieces[-1] = pieces[-1] + token[2:]
            elif token.startswith("##"):
                pieces.append(token[2:])
            else:
                pieces.append(token)
        return " ".join(pieces)

    @staticmethod
    def _format_top_tokens(attn, tokens, prefix, topk=5):
        topk = min(topk, attn.numel())
        values, indices = torch.topk(attn, topk)
        selected_tokens = [tokens[int(index.item())] for index in indices]
        selected_weights = ["{:.6f}".format(float(value.item())) for value in values]
        return {
            "{}_top_tokens".format(prefix): "|".join(selected_tokens),
            "{}_top_weights".format(prefix): "|".join(selected_weights),
        }

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
