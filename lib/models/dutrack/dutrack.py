import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head

from lib.models.dutrack.evidence_layer import TrackingEvidenceLayer
from lib.models.dutrack.evidence_unit_layer import TrackingEvidenceUnitLayer
from lib.models.dutrack.itpn import fast_itpn_base_3324_patch16_224
from lib.models.dutrack.tec import TrackingEvidenceCalibration
from lib.utils.box_ops import box_xyxy_to_cxcywh


class DUTrack(nn.Module):
    """ This is the base class for MMTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER",
                 token_len=1, tec=None, evidence_layer=None, evidence_unit_layer=None,
                 template_token_len=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        self.track_query = None
        self.token_len = token_len
        self.tec = tec
        self.evidence_layer = evidence_layer
        self.evidence_unit_layer = evidence_unit_layer
        self.template_token_len = template_token_len

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                descript,
                ):
        assert isinstance(search, list), "The type of search is not List"

        out_dict = []
        for i in range(len(search)):
            x, aux_dict = self.backbone(z=template.copy(), x=search[i], l=list(descript[i]), temporal_query=self.track_query, top_K=self.token_len)
            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]
                
            enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            enc_opt_before_calibration = enc_opt

            if self.tec is not None or self.evidence_layer is not None or self.evidence_unit_layer is not None:
                h_l, h_z = self._split_fusion_tokens(feat_last, aux_dict)

            if self.tec is not None:
                enc_opt, tec_aux = self.tec(
                    h_x=enc_opt,
                    l_raw=aux_dict.get("l_raw", None),
                    h_z=h_z,
                    h_l=h_l,
                    l_mask=aux_dict.get("l_mask", None),
                    template_token_len=self.template_token_len,
                )
                aux_dict.update(tec_aux)

            if self.evidence_layer is not None:
                enc_opt, evidence_aux = self.evidence_layer(
                    h_x=enc_opt,
                    l_raw=aux_dict.get("l_raw", None),
                    h_z=h_z,
                    h_l=h_l,
                    l_mask=aux_dict.get("l_mask", None),
                    semantic_l_mask=aux_dict.get("semantic_l_mask", None),
                    template_token_len=self.template_token_len,
                )
                aux_dict.update(evidence_aux)

            if self.evidence_unit_layer is not None:
                enc_opt, evidence_unit_aux = self.evidence_unit_layer(
                    h_x=enc_opt,
                    l_raw=aux_dict.get("l_raw", None),
                    h_z=h_z,
                    h_l=h_l,
                    l_mask=aux_dict.get("l_mask", None),
                    evidence_anchor_mask=aux_dict.get("evidence_anchor_mask", None),
                    template_token_len=self.template_token_len,
                )
                aux_dict.update(evidence_unit_aux)

            if self.backbone.add_cls_token:
                self.track_query = (feat_last[:, :self.token_len].clone()).detach()  # stop grad  (B, N, C)

            att = torch.matmul(enc_opt, feat_last[:, :1].transpose(1, 2))  # (B, HW, N)
            if self.tec is not None:
                aux_dict.update({
                    "tec_enc_norm_before": enc_opt_before_calibration.detach().norm(dim=-1).mean().reshape(1),
                    "tec_enc_norm_after": enc_opt.detach().norm(dim=-1).mean().reshape(1),
                    "tec_head_att_mean": att.detach().mean().reshape(1),
                    "tec_head_att_std": att.detach().std().reshape(1),
                })
            if self.evidence_layer is not None:
                aux_dict.update({
                    "stage2_enc_norm_before": enc_opt_before_calibration.detach().norm(dim=-1).mean().reshape(1),
                    "stage2_enc_norm_after": enc_opt.detach().norm(dim=-1).mean().reshape(1),
                    "stage2_head_att_mean": att.detach().mean().reshape(1),
                    "stage2_head_att_std": att.detach().std().reshape(1),
                })
            if self.evidence_unit_layer is not None:
                aux_dict.update({
                    "stage2r_enc_norm_before": enc_opt_before_calibration.detach().norm(dim=-1).mean().reshape(1),
                    "stage2r_enc_norm_after": enc_opt.detach().norm(dim=-1).mean().reshape(1),
                    "stage2r_head_att_mean": att.detach().mean().reshape(1),
                    "stage2r_head_att_std": att.detach().std().reshape(1),
                })
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            
            # Forward head
            out = self.forward_head(opt, None)

            out.update(aux_dict)
            out['backbone_feat'] = x
            
            out_dict.append(out)
            
        return out_dict

    def _split_fusion_tokens(self, feat_last, aux_dict):
        lang_len = int(aux_dict.get("len_l", 0) or 0)
        template_len = int(aux_dict.get("len_z", 0) or 0)
        if lang_len <= 0 or template_len <= 0:
            return None, None

        prefix_len = feat_last.shape[1] - self.feat_len_s - lang_len - template_len
        if prefix_len < 0:
            return None, None

        lang_start = prefix_len
        template_start = lang_start + lang_len
        h_l = feat_last[:, lang_start:template_start]
        h_z = feat_last[:, template_start:template_start + template_len]
        return h_l, h_z

    def freeze_for_tec(self, freeze_backbone=False, freeze_head=False):
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if freeze_head:
            for param in self.box_head.parameters():
                param.requires_grad = False

        if self.tec is not None:
            for param in self.tec.parameters():
                param.requires_grad = True

        if self.evidence_layer is not None:
            for param in self.evidence_layer.parameters():
                param.requires_grad = True

        if self.evidence_unit_layer is not None:
            for param in self.evidence_unit_layer.parameters():
                param.requires_grad = True

    def forward_head(self, opt, gt_score_map=None):
        """
        enc_opt: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            
            out = {'pred_boxes': outputs_coord_new,
                    'score_map': score_map_ctr,
                    'size_map': size_map,
                    'offset_map': offset_map}
            
            return out
        else:
            raise NotImplementedError


def build_dutrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')

    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'itpn_base':
        backbone = fast_itpn_base_3324_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,bert_dir=cfg.MODEL.BACKBONE.BERT_DIR)
    else:
        raise NotImplementedError

    hidden_dim = backbone.embed_dim
    patch_start_index = 1
    
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)



    tec = None
    evidence_layer = None
    evidence_unit_layer = None
    tec_enabled = getattr(cfg.MODEL.TEC, "ENABLE", False)
    evidence_layer_enabled = getattr(cfg.MODEL.EVIDENCE_LAYER, "ENABLE", False)
    evidence_unit_layer_enabled = getattr(cfg.MODEL.EVIDENCE_UNIT_LAYER, "ENABLE", False)
    enabled_count = int(tec_enabled) + int(evidence_layer_enabled) + int(evidence_unit_layer_enabled)
    if enabled_count > 1:
        raise ValueError("MODEL.TEC, MODEL.EVIDENCE_LAYER, and MODEL.EVIDENCE_UNIT_LAYER are mutually exclusive.")

    if getattr(cfg.MODEL.TEC, "ENABLE", False):
        tec = TrackingEvidenceCalibration(
            dim=hidden_dim,
            evidence_dim=cfg.MODEL.TEC.EVIDENCE_DIM,
            gamma_init=cfg.MODEL.TEC.GAMMA_INIT,
            lang_source=cfg.MODEL.TEC.LANG_SOURCE,
            target_pool=cfg.MODEL.TEC.TARGET_POOL,
            center_ratio=cfg.MODEL.TEC.CENTER_RATIO,
            min_valid_tokens=cfg.MODEL.TEC.MIN_VALID_TOKENS,
            dropout=cfg.MODEL.TEC.DROPOUT,
        )

    if evidence_layer_enabled:
        evidence_layer = TrackingEvidenceLayer(
            dim=hidden_dim,
            evidence_dim=cfg.MODEL.EVIDENCE_LAYER.EVIDENCE_DIM,
            gamma_init=cfg.MODEL.EVIDENCE_LAYER.GAMMA_INIT,
            beta=cfg.MODEL.EVIDENCE_LAYER.BETA,
            d_mag_max=cfg.MODEL.EVIDENCE_LAYER.D_MAG_MAX,
            d_norm_eps=cfg.MODEL.EVIDENCE_LAYER.D_NORM_EPS,
            residual_init_scale=cfg.MODEL.EVIDENCE_LAYER.RESIDUAL_INIT_SCALE,
            lang_source=cfg.MODEL.EVIDENCE_LAYER.LANG_SOURCE,
            target_pool=cfg.MODEL.EVIDENCE_LAYER.TARGET_POOL,
            center_ratio=cfg.MODEL.EVIDENCE_LAYER.CENTER_RATIO,
            min_valid_tokens=cfg.MODEL.EVIDENCE_LAYER.MIN_VALID_TOKENS,
            num_evidence_slots=cfg.MODEL.EVIDENCE_LAYER.NUM_EVIDENCE_SLOTS,
            attention_uniform_mix=cfg.MODEL.EVIDENCE_LAYER.ATTENTION_UNIFORM_MIX,
            dropout=cfg.MODEL.EVIDENCE_LAYER.DROPOUT,
        )

    if evidence_unit_layer_enabled:
        evidence_unit_layer = TrackingEvidenceUnitLayer(
            dim=hidden_dim,
            evidence_dim=cfg.MODEL.EVIDENCE_UNIT_LAYER.EVIDENCE_DIM,
            gamma_init=cfg.MODEL.EVIDENCE_UNIT_LAYER.GAMMA_INIT,
            beta=cfg.MODEL.EVIDENCE_UNIT_LAYER.BETA,
            d_mag_max=cfg.MODEL.EVIDENCE_UNIT_LAYER.D_MAG_MAX,
            d_norm_eps=cfg.MODEL.EVIDENCE_UNIT_LAYER.D_NORM_EPS,
            residual_init_scale=cfg.MODEL.EVIDENCE_UNIT_LAYER.RESIDUAL_INIT_SCALE,
            lang_source=cfg.MODEL.EVIDENCE_UNIT_LAYER.LANG_SOURCE,
            target_pool=cfg.MODEL.EVIDENCE_UNIT_LAYER.TARGET_POOL,
            center_ratio=cfg.MODEL.EVIDENCE_UNIT_LAYER.CENTER_RATIO,
            min_evidence_units=cfg.MODEL.EVIDENCE_UNIT_LAYER.MIN_EVIDENCE_UNITS,
            phrase_window=cfg.MODEL.EVIDENCE_UNIT_LAYER.PHRASE_WINDOW,
            dropout=cfg.MODEL.EVIDENCE_UNIT_LAYER.DROPOUT,
        )

    template_token_len = None
    if hasattr(backbone, "pos_embed_z"):
        template_token_len = backbone.pos_embed_z.shape[1]

    model = DUTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        token_len=cfg.MODEL.BACKBONE.TOP_K,
        tec=tec,
        evidence_layer=evidence_layer,
        evidence_unit_layer=evidence_unit_layer,
        template_token_len=template_token_len,
    )
    if 'DUTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
        pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
        file_name = cfg.MODEL.PRETRAIN_FILE
        pth = os.path.join(pretrained_path,file_name)
        checkpoint = torch.load(pth, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
    if tec_enabled:
        model.freeze_for_tec(
            freeze_backbone=cfg.MODEL.TEC.FREEZE_BACKBONE,
            freeze_head=cfg.MODEL.TEC.FREEZE_HEAD,
        )
    if evidence_layer_enabled:
        model.freeze_for_tec(
            freeze_backbone=cfg.MODEL.EVIDENCE_LAYER.FREEZE_BACKBONE,
            freeze_head=cfg.MODEL.EVIDENCE_LAYER.FREEZE_HEAD,
        )
    if evidence_unit_layer_enabled:
        model.freeze_for_tec(
            freeze_backbone=cfg.MODEL.EVIDENCE_UNIT_LAYER.FREEZE_BACKBONE,
            freeze_head=cfg.MODEL.EVIDENCE_UNIT_LAYER.FREEZE_HEAD,
        )
    return model
