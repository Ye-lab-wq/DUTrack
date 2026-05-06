from . import BaseActor
from lib.utils.misc import NestedTensor, interpolate
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import math
import torch
import torch.nn.functional as F
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate



class DUTrackActor(BaseActor):
    """ Actor for training ODTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        template_list = []
        search_list = []

        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_list.append(template_img_i)

        for i in range(self.settings.num_search):
            search_img_i = data['search_images'][i].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
            search_list.append(search_img_i)
            
        box_mask_z = []
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            for i in range(self.settings.num_template):
                box_mask_z.append(generate_mask_cond(self.cfg, template_list[i].shape[0], template_list[i].device,
                                                    data['template_anno'][i]))
            box_mask_z = torch.cat(box_mask_z, dim=1)

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])



        out_dict = self.net(
            template=template_list,
            search=search_list,
            descript=data['language_description'],
        )

        return out_dict

    @staticmethod
    def _search_token_centers(gt_bbox, grid_size, device):
        batch_size = gt_bbox.shape[0]
        coords = torch.arange(grid_size, device=device)
        y, x = torch.meshgrid(coords, coords)
        token_x = (x.flatten().float() + 0.5) / grid_size
        token_y = (y.flatten().float() + 0.5) / grid_size

        gt_bbox = gt_bbox.to(device)
        token_x = token_x.unsqueeze(0).expand(batch_size, -1)
        token_y = token_y.unsqueeze(0).expand(batch_size, -1)
        return token_x, token_y, gt_bbox

    @staticmethod
    def _build_search_token_mask(gt_bbox, grid_size, device):
        token_x, token_y, gt_bbox = DUTrackActor._search_token_centers(gt_bbox, grid_size, device)
        x1 = gt_bbox[:, 0].clamp(0.0, 1.0).unsqueeze(1)
        y1 = gt_bbox[:, 1].clamp(0.0, 1.0).unsqueeze(1)
        x2 = (gt_bbox[:, 0] + gt_bbox[:, 2]).clamp(0.0, 1.0).unsqueeze(1)
        y2 = (gt_bbox[:, 1] + gt_bbox[:, 3]).clamp(0.0, 1.0).unsqueeze(1)

        target = ((token_x >= x1) & (token_x <= x2) & (token_y >= y1) & (token_y <= y2)).float()

        empty = target.sum(dim=1) == 0
        if empty.any():
            center_x = ((x1 + x2) * 0.5).expand_as(token_x)
            center_y = ((y1 + y2) * 0.5).expand_as(token_y)
            nearest = ((token_x - center_x) ** 2 + (token_y - center_y) ** 2).argmin(dim=1)
            target[empty, nearest[empty]] = 1.0

        return target

    def _build_search_token_gaussian(self, gt_bbox, grid_size, device):
        token_x, token_y, gt_bbox = self._search_token_centers(gt_bbox, grid_size, device)
        center_x = (gt_bbox[:, 0] + 0.5 * gt_bbox[:, 2]).clamp(0.0, 1.0).unsqueeze(1)
        center_y = (gt_bbox[:, 1] + 0.5 * gt_bbox[:, 3]).clamp(0.0, 1.0).unsqueeze(1)

        sigma_scale = float(getattr(self.cfg.MODEL.VLTE, "GAUSSIAN_SIGMA_SCALE", 0.25))
        min_sigma = 1.0 / grid_size
        sigma_x = (gt_bbox[:, 2].clamp(min=min_sigma) * sigma_scale).clamp(min=min_sigma).unsqueeze(1)
        sigma_y = (gt_bbox[:, 3].clamp(min=min_sigma) * sigma_scale).clamp(min=min_sigma).unsqueeze(1)

        dist = ((token_x - center_x) / sigma_x) ** 2 + ((token_y - center_y) / sigma_y) ** 2
        return torch.exp(-0.5 * dist).clamp(0.0, 1.0)

    def _build_search_token_box_gaussian(self, gt_bbox, grid_size, device):
        box_mask = self._build_search_token_mask(gt_bbox, grid_size, device)
        gaussian = self._build_search_token_gaussian(gt_bbox, grid_size, device)
        floor = float(getattr(self.cfg.MODEL.VLTE, "BOX_GAUSSIAN_FLOOR", 0.5))
        floor_map = torch.full_like(gaussian, floor)
        inside_target = torch.maximum(gaussian, floor_map)
        return torch.where(box_mask > 0, inside_target, torch.zeros_like(inside_target))

    def _compute_vl_score_loss(self, pred_dict, gt_bbox, fallback_device):
        if 'vl_score_x_logits' not in pred_dict:
            return torch.tensor(0.0, device=fallback_device)

        logits = pred_dict['vl_score_x_logits']
        grid_size = int(math.sqrt(logits.shape[1]))
        if grid_size * grid_size != logits.shape[1]:
            return torch.tensor(0.0, device=logits.device)

        target_mode = getattr(self.cfg.MODEL.VLTE, "SCORE_TARGET", "box")
        if target_mode == "gaussian":
            target = self._build_search_token_gaussian(gt_bbox, grid_size, logits.device)
        elif target_mode == "box_gaussian":
            target = self._build_search_token_box_gaussian(gt_bbox, grid_size, logits.device)
        elif target_mode == "box":
            target = self._build_search_token_mask(gt_bbox, grid_size, logits.device)
        else:
            raise ValueError("Unsupported VLTE score target: {}".format(target_mode))

        loss_map = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        if target_mode == "gaussian":
            weight = 1.0 + target
        else:
            weight = torch.where(target > 0, torch.full_like(target, 2.0), torch.ones_like(target))
        return (loss_map * weight).mean()

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # currently only support the type of pred_dict is list
        assert isinstance(pred_dict, list)
        loss_dict = {}
        total_status = {}
        total_loss = torch.tensor(0., dtype=torch.float).cuda() # 定义 0 tensor，并指定GPU设备
        
        # generate gt gaussian map
        gt_gaussian_maps_list = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        
        for i in range(len(pred_dict)):
            # get GT
            gt_bbox = gt_dict['search_anno'][i]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
            gt_gaussian_maps = gt_gaussian_maps_list[i].unsqueeze(1)

            # Get boxes
            pred_boxes = pred_dict[i]['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)
            # (B,4) --> (B,1,4) --> (B,N,4)
            
            # compute giou and iou
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            loss_dict['giou'] = giou_loss
            
            # compute l1 loss
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            loss_dict['l1'] = l1_loss
            
            # compute location loss
            if 'score_map' in pred_dict[i]:
                location_loss = self.objective['focal'](pred_dict[i]['score_map'], gt_gaussian_maps)
            else:
                location_loss = torch.tensor(0.0, device=l1_loss.device)
            loss_dict['focal'] = location_loss

            vl_score_loss = self._compute_vl_score_loss(pred_dict[i], gt_bbox, l1_loss.device)
            loss_dict['vl_score'] = vl_score_loss
                
            # weighted sum
            loss = sum(loss_dict[k] * self.loss_weight[k] for k in loss_dict.keys() if k in self.loss_weight)
            total_loss += loss
            
            if return_status:
                # status for log
                status = {}
                
                mean_iou = iou.detach().mean()
                status = {f"{i}frame_Loss/total": loss.item(),
                        f"{i}frame_Loss/giou": giou_loss.item(),
                        f"{i}frame_Loss/l1": l1_loss.item(),
                        f"{i}frame_Loss/location": location_loss.item(),
                        f"{i}frame_Loss/vl_score": vl_score_loss.item(),
                        f"{i}frame_IoU": mean_iou.item()}
                    
                total_status.update(status)

        if return_status:
            return total_loss, total_status
        else:
            return total_loss
