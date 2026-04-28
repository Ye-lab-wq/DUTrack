from . import BaseActor
from lib.utils.misc import NestedTensor, interpolate
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xywh_to_cxcywh
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



        search_boxes = [box_xywh_to_cxcywh(data['search_anno'][i]) for i in range(self.settings.num_search)]
        out_dict = self.net(template=template_list,
                            search=search_list,
                            descript=data['language_description'],
                            search_boxes=search_boxes,
)

        return out_dict

    def _compute_proto_ctr_loss(self, pred):
        anchor_token = pred.get('semantic_token', pred['target_proto'])
        patch_feat = pred['proto_patch_feat']
        in_mask = pred['in_mask']
        out_mask = pred['out_mask']
        pos_prior = pred['pos_prior']
        target_logits = pred['target_logits']

        anchor_token = F.normalize(anchor_token, dim=-1)
        patch_feat = F.normalize(patch_feat, dim=-1)

        temp = float(getattr(self.cfg.TRAIN, 'TEMPLATE_PROTO_CTR_TEMP', 0.2))
        max_neg = int(getattr(self.cfg.TRAIN, 'TEMPLATE_PROTO_CTR_NEG', 8))
        max_neg = max(1, min(max_neg, patch_feat.size(1)))

        # Use the GT-box center prior to pick one reliable positive patch,
        # then contrast it against hard negatives outside the GT box.
        pos_scores = pos_prior.masked_fill(in_mask <= 0, float('-inf'))
        pos_idx = pos_scores.argmax(dim=1, keepdim=True)
        pos_feat = torch.gather(patch_feat, 1, pos_idx.unsqueeze(-1).expand(-1, -1, patch_feat.size(-1))).squeeze(1)

        neg_scores = target_logits.masked_fill(out_mask <= 0, float('-inf'))
        neg_scores, neg_idx = torch.topk(neg_scores, k=max_neg, dim=1)
        neg_valid = torch.isfinite(neg_scores)
        neg_feat = torch.gather(patch_feat, 1, neg_idx.unsqueeze(-1).expand(-1, -1, patch_feat.size(-1)))

        pos_logit = torch.sum(anchor_token * pos_feat, dim=-1, keepdim=True) / temp
        neg_logits = torch.sum(anchor_token.unsqueeze(1) * neg_feat, dim=-1) / temp
        neg_logits = neg_logits.masked_fill(~neg_valid, -1e4)
        logits = torch.cat([pos_logit, neg_logits], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    @staticmethod
    def _safe_corrcoef(x, y):
        if x.numel() < 2 or y.numel() < 2:
            return torch.tensor(0.0, device=x.device)
        x = x - x.mean()
        y = y - y.mean()
        denom = x.norm() * y.norm()
        if denom.item() < 1e-6:
            return torch.tensor(0.0, device=x.device)
        return (x * y).sum() / denom

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

            proto_target_loss = torch.tensor(0.0, device=l1_loss.device)
            proto_split_loss = torch.tensor(0.0, device=l1_loss.device)
            proto_ctr_loss = torch.tensor(0.0, device=l1_loss.device)
            quality_iou_corr = torch.tensor(0.0, device=l1_loss.device)
            in_token_score_mean = torch.tensor(0.0, device=l1_loss.device)
            out_token_score_mean = torch.tensor(0.0, device=l1_loss.device)
            if 'target_logits' in pred_dict[i]:
                in_mask = pred_dict[i]['in_mask'].detach()
                out_mask = pred_dict[i]['out_mask'].detach()
                pseudo_distractor = pred_dict[i]['pseudo_distractor_mask'].detach()

                proto_target_loss = self.objective['cls'](pred_dict[i]['target_logits'], in_mask)

                if out_mask.sum() > 0:
                    distractor_target = torch.where(out_mask > 0, pseudo_distractor, torch.zeros_like(pseudo_distractor))
                    background_target = torch.where(out_mask > 0, 1.0 - pseudo_distractor, torch.zeros_like(pseudo_distractor))
                    distractor_loss = F.binary_cross_entropy_with_logits(
                        pred_dict[i]['distractor_logits'], distractor_target, reduction='none'
                    )
                    background_loss = F.binary_cross_entropy_with_logits(
                        pred_dict[i]['background_logits'], background_target, reduction='none'
                    )
                    norm = out_mask.sum().clamp(min=1.0)
                    proto_split_loss = ((distractor_loss + background_loss) * out_mask).sum() / norm

                if 'target_proto' in pred_dict[i] and 'template_token_feat' in pred_dict[i]:
                    proto_ctr_loss = self._compute_proto_ctr_loss(pred_dict[i])

                if 'template_quality' in pred_dict[i]:
                    sample_iou = iou.detach().view(-1, num_queries).mean(dim=1)
                    quality_iou_corr = self._safe_corrcoef(pred_dict[i]['template_quality'].detach(), sample_iou)

                if 'token_score' in pred_dict[i]:
                    token_score = pred_dict[i]['token_score'].detach()
                    in_token_score_mean = ((token_score * in_mask).sum() / in_mask.sum().clamp(min=1.0))
                    out_token_score_mean = ((token_score * out_mask).sum() / out_mask.sum().clamp(min=1.0))

            loss_dict['proto_target'] = proto_target_loss
            loss_dict['proto_split'] = proto_split_loss
            loss_dict['proto_ctr'] = proto_ctr_loss
                
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
                        f"{i}frame_Loss/proto_target": proto_target_loss.item(),
                        f"{i}frame_Loss/proto_split": proto_split_loss.item(),
                        f"{i}frame_Loss/proto_ctr": proto_ctr_loss.item(),
                        f"{i}frame_Proto/quality_iou_corr": quality_iou_corr.item(),
                        f"{i}frame_Proto/in_token_score_mean": in_token_score_mean.item(),
                        f"{i}frame_Proto/out_token_score_mean": out_token_score_mean.item(),
                        f"{i}frame_IoU": mean_iou.item()}
                    
                total_status.update(status)

        if return_status:
            return total_loss, total_status
        else:
            return total_loss
