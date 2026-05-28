import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head

from lib.models.dutrack.itpn import fast_itpn_base_3324_patch16_224
from lib.models.dutrack.language_global_score_bias import LanguageGlobalScoreBias
from lib.utils.box_ops import box_xyxy_to_cxcywh


class DUTrack(nn.Module):
    """ This is the base class for MMTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", token_len=1, cfg=None):
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
        self.freeze_nontrainable_bn = False
        te_cfg = getattr(getattr(cfg, "MODEL", None), "TE", None) if cfg is not None else None
        self.score_prior_enabled = bool(getattr(te_cfg, "SCORE_PRIOR_ENABLE", False))
        self.score_prior_beta = float(getattr(te_cfg, "SCORE_PRIOR_BETA", 0.0))
        self.score_prior_source = str(getattr(te_cfg, "SCORE_PRIOR_SOURCE", "logits")).lower()
        self.score_prior_center = str(getattr(te_cfg, "SCORE_PRIOR_CENTER", "mean")).lower()
        self.score_prior_layer = int(getattr(te_cfg, "SCORE_PRIOR_LAYER", -1))
        self.score_prior_pruning_loc = list(getattr(te_cfg, "PRUNING_LOC", []))
        self.score_prior_lmq_loc = list(getattr(te_cfg, "LMQ_LOC", self.score_prior_pruning_loc))
        self.score_prior_eps = float(getattr(te_cfg, "SCORE_PRIOR_EPS", 1e-4))
        self.score_prior_bias_clamp = float(getattr(te_cfg, "SCORE_PRIOR_BIAS_CLAMP", 0.0))
        self.score_prior_margin_tau = float(getattr(te_cfg, "SCORE_PRIOR_MARGIN_TAU", 0.1))
        self.gsb_enabled = bool(getattr(te_cfg, "GLOBAL_SCORE_BIAS_ENABLE", False))
        self.gsb_module = None
        if self.gsb_enabled:
            gsb_hidden_dim = int(getattr(te_cfg, "GLOBAL_SCORE_BIAS_HIDDEN_DIM", 256))
            gsb_max_delta = float(getattr(te_cfg, "GLOBAL_SCORE_BIAS_MAX_DELTA", 0.02))
            gsb_init_gate_bias = float(getattr(te_cfg, "GLOBAL_SCORE_BIAS_INIT_GATE_BIAS", -4.0))
            gsb_init_delta_std = float(getattr(te_cfg, "GLOBAL_SCORE_BIAS_INIT_DELTA_STD", 1e-4))
            gsb_pool_mode = str(getattr(te_cfg, "GLOBAL_SCORE_BIAS_POOL_MODE", "mean")).lower()
            gsb_beta = float(getattr(te_cfg, "GLOBAL_SCORE_BIAS_BETA", 0.02))
            self.gsb_module = LanguageGlobalScoreBias(
                dim=self.backbone.embed_dim,
                hidden_dim=gsb_hidden_dim,
                max_delta=gsb_max_delta,
                init_gate_bias=gsb_init_gate_bias,
                init_delta_std=gsb_init_delta_std,
                pool_mode=gsb_pool_mode,
                beta=gsb_beta,
            )

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_nontrainable_bn:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    params = list(module.parameters(recurse=False))
                    if params and not any(param.requires_grad for param in params):
                        module.eval()
        return self

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                descript,
                language_word_reliability=None,
                language_token_state=None,
                language_token_mask=None,
                ):
        assert isinstance(search, list), "The type of search is not List"

        out_dict = []
        h_prev_pooled = None
        for i in range(len(search)):
            word_reliability = None
            if language_word_reliability is not None:
                word_reliability = language_word_reliability
                if isinstance(language_word_reliability, (list, tuple)):
                    word_reliability = language_word_reliability[i]
            token_state = language_token_state
            if isinstance(language_token_state, (list, tuple)):
                token_state = language_token_state[i]
            token_mask = language_token_mask
            if isinstance(language_token_mask, (list, tuple)):
                token_mask = language_token_mask[i]
            x, aux_dict = self.backbone(
                z=template.copy(), x=search[i], l=list(descript[i]),
                temporal_query=self.track_query, top_K=self.token_len,
                word_reliability=word_reliability,
                language_token_state=token_state,
                language_token_mask=token_mask)
            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]

            enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)

            if self.backbone.add_cls_token:
                self.track_query = (x[:, :self.token_len].clone()).detach()  # stop grad  (B, N, C)

            att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))  # (B, HW, N)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)

            score_bias, score_prior_keep = self._score_prior_bias(aux_dict, enc_opt.shape[0], enc_opt.device, enc_opt.dtype)

            # Global Score Bias: lightweight language state update → additive score correction
            if self.gsb_enabled and self.gsb_module is not None and token_state is not None:
                gsb_mask = token_mask
                h_cand = self.gsb_module.pool_tokens(token_state, gsb_mask)
                if h_prev_pooled is None:
                    h_prev_pooled = h_cand
                search_feat = feat_last[:, -self.feat_len_s:]
                h_new, gsb_bias_flat, gsb_diag = self.gsb_module(
                    h_prev_pooled, h_cand, search_feat)
                h_prev_pooled = h_new
                gsb_bias = gsb_bias_flat.view(h_cand.shape[0], 1, self.feat_sz_s, self.feat_sz_s)
                if score_bias is None:
                    score_bias = gsb_bias
                else:
                    score_bias = score_bias + gsb_bias
                aux_dict["gsb_diagnostics"] = gsb_diag

            # Forward head
            out = self.forward_head(opt, None, score_bias=score_bias)
            if score_prior_keep is not None:
                out["score_prior_keep"] = score_prior_keep.detach()
            if score_bias is not None:
                out["score_prior_bias"] = score_bias

            out.update(aux_dict)
            out['backbone_feat'] = x

            out_dict.append(out)
            
        return out_dict

    def _score_prior_bias(self, aux_dict, batch_size, device, dtype):
        if not self.score_prior_enabled:
            return None, None
        source = self.score_prior_source
        if source in ("logits", "raw_logits", "logit", "prob_from_logits"):
            stage_idx = self._score_prior_stage_index()
            logits_list = aux_dict.get("lang_te_search_logits", None)
            if not logits_list:
                raise ValueError("SCORE_PRIOR_SOURCE=logits requires lang_te_search_logits.")
            logits = logits_list[stage_idx]
            keep = (logits[..., 0] - logits[..., 1]).sigmoid()
        elif source in ("decision", "cumulative", "cumulative_decision", "keep"):
            stage_idx = self._score_prior_stage_index()
            decision_list = aux_dict.get("score_prior_search_decisions", None)
            if not decision_list:
                decision_list = aux_dict.get("visual_te_search_decisions", None)
            if not decision_list:
                raise ValueError("SCORE_PRIOR_SOURCE=decision requires visual_te_search_decisions.")
            keep = decision_list[stage_idx]
            if keep.dim() == 3 and keep.shape[-1] == 1:
                keep = keep.squeeze(-1)
        elif source in ("word_direct", "word_direct_margin", "direct_word", "word_margin"):
            stage_idx = self._score_prior_stage_index()
            score_list = aux_dict.get("word_level_direct_scores", None)
            if not score_list:
                raise ValueError("SCORE_PRIOR_SOURCE=word_direct_margin requires word_level_direct_scores.")
            raw_score = score_list[stage_idx]
            if raw_score.dim() == 3 and raw_score.shape[-1] == 1:
                raw_score = raw_score.squeeze(-1)
            if raw_score.shape[1] != self.feat_len_s:
                raise ValueError(
                    "Word direct score length {} must match CENTER head grid length {}".format(
                        raw_score.shape[1], self.feat_len_s))
            bias = raw_score
            if self.score_prior_center in ("mean", "token_mean", "search_mean"):
                bias = bias - bias.mean(dim=1, keepdim=True)
            elif self.score_prior_center in ("none", "off", ""):
                pass
            else:
                raise ValueError("Unsupported SCORE_PRIOR_CENTER: {}".format(self.score_prior_center))
            tau = max(self.score_prior_margin_tau, 1e-8)
            bias = torch.tanh(bias / tau)
            bias = bias * self.score_prior_beta
            if self.score_prior_bias_clamp > 0.0:
                bias = bias.clamp(min=-self.score_prior_bias_clamp, max=self.score_prior_bias_clamp)
            bias = bias.view(batch_size, 1, self.feat_sz_s, self.feat_sz_s).to(device=device, dtype=dtype)
            return bias, raw_score
        elif source in ("lmq", "lmq_prior", "multi_query", "language_multi_query"):
            stage_idx = self._score_prior_stage_index(self.score_prior_lmq_loc)
            score_list = aux_dict.get("lmq_prior_scores", None)
            if not score_list:
                raise ValueError("SCORE_PRIOR_SOURCE=lmq_prior requires lmq_prior_scores.")
            raw_score = score_list[stage_idx]
            if raw_score.dim() == 3 and raw_score.shape[-1] == 1:
                raw_score = raw_score.squeeze(-1)
            if raw_score.shape[1] != self.feat_len_s:
                raise ValueError(
                    "LMQ prior score length {} must match CENTER head grid length {}".format(
                        raw_score.shape[1], self.feat_len_s))
            bias = raw_score
            if self.score_prior_center in ("mean", "token_mean", "search_mean"):
                bias = bias - bias.mean(dim=1, keepdim=True)
            elif self.score_prior_center in ("none", "off", ""):
                pass
            else:
                raise ValueError("Unsupported SCORE_PRIOR_CENTER: {}".format(self.score_prior_center))
            tau = max(self.score_prior_margin_tau, 1e-8)
            bias = torch.tanh(bias / tau)
            bias = bias * self.score_prior_beta
            if self.score_prior_bias_clamp > 0.0:
                bias = bias.clamp(min=-self.score_prior_bias_clamp, max=self.score_prior_bias_clamp)
            bias = bias.view(batch_size, 1, self.feat_sz_s, self.feat_sz_s).to(device=device, dtype=dtype)
            return bias, raw_score
        else:
            raise ValueError("Unsupported SCORE_PRIOR_SOURCE: {}".format(self.score_prior_source))

        if keep.shape[1] != self.feat_len_s:
            raise ValueError(
                "Search keep length {} must match CENTER head grid length {}".format(
                    keep.shape[1], self.feat_len_s))
        eps = max(self.score_prior_eps, 1e-8)
        keep = keep.clamp(min=eps, max=1.0)
        bias = keep.log()
        if self.score_prior_center in ("mean", "token_mean", "search_mean"):
            bias = bias - bias.mean(dim=1, keepdim=True)
        elif self.score_prior_center in ("none", "off", ""):
            pass
        else:
            raise ValueError("Unsupported SCORE_PRIOR_CENTER: {}".format(self.score_prior_center))
        bias = bias * self.score_prior_beta
        if self.score_prior_bias_clamp > 0.0:
            bias = bias.clamp(min=-self.score_prior_bias_clamp, max=self.score_prior_bias_clamp)
        bias = bias.view(batch_size, 1, self.feat_sz_s, self.feat_sz_s).to(device=device, dtype=dtype)
        return bias, keep

    def _score_prior_stage_index(self, locations=None):
        layer = int(self.score_prior_layer)
        locations = self.score_prior_pruning_loc if locations is None else list(locations)
        if layer < 0:
            return -1
        if layer not in locations:
            raise ValueError(
                "SCORE_PRIOR_LAYER={} must be -1 or one of {}".format(
                    layer, locations))
        return locations.index(layer)

    def forward_head(self, opt, gt_score_map=None, score_bias=None):
        """
        enc_opt: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        if score_bias is not None:
            if score_bias.dim() == 3:
                score_bias = score_bias.unsqueeze(1)
            if score_bias.dim() != 4:
                raise ValueError("score_bias must have shape (B,1,H,W) or (B*Nq,1,H,W)")
            if score_bias.shape[0] == bs:
                score_bias = score_bias[:, None].expand(
                    bs, Nq, 1, self.feat_sz_s, self.feat_sz_s).reshape(
                    bs * Nq, 1, self.feat_sz_s, self.feat_sz_s)
            elif score_bias.shape[0] != bs * Nq:
                raise ValueError(
                    "score_bias batch {} must be B={} or B*Nq={}".format(
                        score_bias.shape[0], bs, bs * Nq))

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
            head_out = self.box_head(opt_feat, gt_score_map, score_bias=score_bias)
            if len(head_out) == 6:
                score_map_ctr, bbox, size_map, offset_map, score_logits_base, score_logits = head_out
            else:
                score_map_ctr, bbox, size_map, offset_map = head_out
                score_logits_base, score_logits = None, None
            
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            
            out = {'pred_boxes': outputs_coord_new,
                    'score_map': score_map_ctr,
                    'size_map': size_map,
                    'offset_map': offset_map}
            if score_logits_base is not None and score_logits is not None:
                out['score_map_logits_base'] = score_logits_base
                out['score_map_logits'] = score_logits
            
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



    model = DUTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        token_len=cfg.MODEL.BACKBONE.TOP_K,
        cfg=cfg,
    )
    if 'DUTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
        pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
        file_name = cfg.MODEL.PRETRAIN_FILE
        pth = os.path.join(pretrained_path,file_name)
        checkpoint = torch.load(pth, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
    if training and bool(getattr(cfg.TRAIN, "TRAIN_TE_ONLY", False)):
        patterns = list(getattr(cfg.TRAIN, "TRAIN_TE_ONLY_PATTERNS", ["backbone.visual_te_predictors"]))
        if bool(getattr(getattr(cfg, "MODEL", None), "TE", None)) and bool(getattr(cfg.MODEL.TE, "GLOBAL_SCORE_BIAS_ENABLE", False)):
            if "gsb_module" not in patterns:
                patterns.append("gsb_module")
        if bool(getattr(cfg.TRAIN, "STAGED_TRAINING", False)):
            stage_patterns = []
            stage_patterns.extend(list(getattr(cfg.TRAIN, "STAGE1_PATTERNS", patterns)))
            stage_patterns.extend(list(getattr(cfg.TRAIN, "STAGE2_PATTERNS", patterns)))
            patterns = list(dict.fromkeys(stage_patterns))
        model.freeze_nontrainable_bn = bool(getattr(cfg.TRAIN, "FREEZE_NONTRAINABLE_BN", True))
        trainable = []
        for name, param in model.named_parameters():
            keep_trainable = any(pattern in name for pattern in patterns)
            param.requires_grad = keep_trainable
            if keep_trainable:
                trainable.append(name)
        if not trainable:
            raise ValueError("TRAIN_TE_ONLY matched no parameters. Patterns: {}".format(patterns))
        print("TRAIN_TE_ONLY enabled. Learnable parameter patterns: {}".format(patterns))
        print("TRAIN_TE_ONLY matched {} tensors.".format(len(trainable)))
    return model
