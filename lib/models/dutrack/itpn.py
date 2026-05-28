# --------------------------------------------------------
# Fast-iTPN: Integrally Pre-Trained Transformer Pyramid Network with Token Migration
# Github source: https://github.com/sunsmarterjie/iTPN/tree/main/fast_itpn
# Copyright (c) 2023 University of Chinese Academy of Sciences
# Licensed under The MIT License [see LICENSE for details]
# By Yunjie Tian
# Based on EVA02, timm and deit code bases
# https://github.com/baaivision/EVA/tree/master/EVA-02
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# --------------------------------------------------------'
from functools import partial

import math
from os.path import split

import torch
import torch.nn as nn
from timm.models.registry import register_model
import torch.nn.functional as F
from timm.models.layers import to_2tuple, drop_path, trunc_normal_

from torch import Tensor, Size
from typing import Union, List

from lib.models.dutrack.base_backbone import BaseBackbone
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from lib.models.dutrack import utils as utils
from lib.models.dutrack.utils import combine_tokens, recover_tokens
from lib.models.dutrack.visual_token_emphasizer import VisualTokenEmphasizer
from lib.models.dutrack.language_token_emphasizer import LanguageGuidedTokenEmphasizer
from lib.models.dutrack.language_multi_query_prior import LanguageMultiQueryPrior
from lib.models.dutrack.language_token_state_updater import LanguageTokenStateUpdater

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


_shape_t = Union[int, List[int], Size]


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 norm_layer=nn.LayerNorm, subln=False
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()

        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.ffn_ln(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 norm_layer=nn.LayerNorm, subln=False
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()

        self.ffn_ln = norm_layer(hidden_features) if subln else None

        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.ffn_ln is not None:
            x = x.permute(0, 2, 3, 1)
            x = self.ffn_ln(x)
            x = x.permute(0, 3, 1, 2)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.,
                 norm_layer=nn.LayerNorm, subln=False
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x


class ConvSwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.,
                 norm_layer=nn.LayerNorm, subln=False
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Conv2d(in_features, hidden_features, 1)
        self.w2 = nn.Conv2d(in_features, hidden_features, 1)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Conv2d(hidden_features, out_features, 1)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x1 = self.w1(x).flatten(2).transpose(1, 2)
        x2 = self.w2(x).flatten(2).transpose(1, 2)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden).transpose(1, 2).view(B, C, H, W)
        x = self.w3(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=None,
            attn_head_dim=None, use_decoupled_rel_pos_bias=False, deepnorm=False, subln=False
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.deepnorm = deepnorm
        self.subln = subln
        if self.deepnorm or self.subln:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=False)
        else:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.rel_pos_bias = None
        self.qk_float = True

        self.window_size = None
        self.relative_position_bias_table = None

        if window_size:
            if use_decoupled_rel_pos_bias:
                self.rel_pos_bias = DecoupledRelativePositionBias(window_size=window_size, num_heads=num_heads)
            else:
                self.window_size = window_size
                self.num_relative_distance = (2 * window_size[0] - 1) * (
                        2 * window_size[1] - 1) + 3  # (2*14-1) * (2*14-1) + 3
                self.relative_position_bias_table = nn.Parameter(
                    torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
                # cls to token & token 2 cls & cls to cls

                # get pair-wise relative position index for each token inside the window
                coords_h = torch.arange(window_size[0])
                coords_w = torch.arange(window_size[1])
                coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
                coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
                relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
                relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
                relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
                relative_coords[:, :, 1] += window_size[1] - 1
                relative_coords[:, :, 0] *= 2 * window_size[1] - 1
                relative_position_index = \
                    torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
                relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
                relative_position_index[0, 0:] = self.num_relative_distance - 3
                relative_position_index[0:, 0] = self.num_relative_distance - 2
                relative_position_index[0, 0] = self.num_relative_distance - 1

                self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, attn_mask=None, policy=None,
                policy_bias=None, policy_query_ranges=None):
        B, N, C = x.shape

        if self.deepnorm or self.subln:
            q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
            k = F.linear(input=x, weight=self.k_proj.weight, bias=None)
            v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)

            q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # B, num_heads, N, C
            k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
            v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        else:
            qkv_bias = None
            if self.q_bias is not None:
                qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)  # 3, B, num_heads, N, C
            q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        if self.qk_float:
            attn = (q.float() @ k.float().transpose(-2, -1))
        else:
            attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0).type_as(attn)

        if self.rel_pos_bias is not None:
            attn = attn + self.rel_pos_bias().type_as(attn)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias.type_as(attn)
        if attn_mask is not None:
            attn_mask = attn_mask.bool()
            attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
        if policy_bias is not None:
            if policy_bias.dim() == 3:
                policy_bias = policy_bias.squeeze(-1)
            policy_bias = policy_bias.to(dtype=attn.dtype, device=attn.device)
            key_bias = policy_bias[:, None, None, :]
            if policy_query_ranges is None:
                attn = attn + key_bias
            else:
                for q_start, q_end in policy_query_ranges:
                    if q_end > q_start:
                        attn[:, :, q_start:q_end, :] = attn[:, :, q_start:q_end, :] + key_bias
        attn = attn.softmax(dim=-1).type_as(x)
        if policy is not None:
            if policy.dim() == 3:
                policy = policy.squeeze(-1)
            policy = policy.to(dtype=attn.dtype, device=attn.device)
            attn = (attn + attn * policy[:, None, None, :]) * 0.5
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x,attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, norm_layer=nn.LayerNorm, window_size=None, attn_head_dim=None,
                 use_decoupled_rel_pos_bias=False,
                 depth=None,
                 postnorm=False,
                 deepnorm=False,
                 subln=False,
                 swiglu=False,
                 naiveswiglu=False,
                 ):
        super().__init__()

        with_attn = num_heads > 0

        self.norm1 = norm_layer(dim) if with_attn else None
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size,
            use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias, attn_head_dim=attn_head_dim,
            deepnorm=deepnorm,
            subln=subln
        ) if with_attn else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        if swiglu:
            self.mlp = xops.SwiGLU(
                in_features=dim,
                hidden_features=mlp_hidden_dim
            )  # hidden_features: 2/3
        elif naiveswiglu:
            self.mlp = SwiGLU(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                subln=subln,
                norm_layer=norm_layer,
            )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                subln=subln,
                norm_layer=norm_layer
            )

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),
                                        requires_grad=True) if self.attn is not None else None
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.deepnorm = deepnorm
        if self.deepnorm:
            self.alpha = math.pow(2.0 * depth, 0.25)

        self.postnorm = postnorm

    def forward(self, x, rel_pos_bias=None, attn_mask=None, policy=None,
                policy_bias=None, policy_query_ranges=None):
        attn = None
        if self.gamma_2 is None:
            if self.postnorm:
                if self.attn is not None:
                    feat, attn = self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask,
                                           policy=policy, policy_bias=policy_bias,
                                           policy_query_ranges=policy_query_ranges)
                    x = x + self.drop_path(
                        self.norm1(feat))
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            elif self.deepnorm:
                if self.attn is not None:
                    residual = x
                    x, attn = self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask,
                                        policy=policy, policy_bias=policy_bias,
                                        policy_query_ranges=policy_query_ranges)
                    x = self.drop_path(x)
                    x = residual * self.alpha + x
                    x = self.norm1(x)

                residual = x
                x = self.mlp(x)
                x = self.drop_path(x)
                x = residual * self.alpha + x
                x = self.norm2(x)
            else:
                if self.attn is not None:
                    feat, attn = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask,
                                           policy=policy, policy_bias=policy_bias,
                                           policy_query_ranges=policy_query_ranges)
                    x = x + self.drop_path(
                        feat)
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.postnorm:
                if self.attn is not None:
                    feat, attn = self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask,
                                           policy=policy, policy_bias=policy_bias,
                                           policy_query_ranges=policy_query_ranges)
                    x = x + self.drop_path(
                        self.gamma_1 * self.norm1(feat))
                x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
            else:
                if self.attn is not None:
                    feat,attn = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask,
                                          policy=policy, policy_bias=policy_bias,
                                          policy_query_ranges=policy_query_ranges)
                    x = x + self.drop_path(self.gamma_1 * feat)
                    # x = x + self.drop_path(
                    #     self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x,attn


class ConvMlpBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop_path=0., init_values=None, norm_layer=nn.LayerNorm,
                 depth=None,
                 postnorm=False,
                 deepnorm=False,
                 subln=False,
                 swiglu=False,
                 naiveswiglu=False,
                 ):
        super().__init__()

        self.attn = None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        if swiglu:
            self.mlp = xops.SwiGLU(
                in_features=dim,
                hidden_features=mlp_hidden_dim
            )  # hidden_features: 2/3
        elif naiveswiglu:
            self.mlp = ConvSwiGLU(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                subln=subln,
                norm_layer=norm_layer,
            )
        else:
            self.mlp = ConvMlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                subln=subln,
                norm_layer=norm_layer
            )

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),
                                        requires_grad=True) if self.attn is not None else None
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.deepnorm = deepnorm
        if self.deepnorm:
            self.alpha = math.pow(2.0 * depth, 0.25)

        self.postnorm = postnorm

    def forward(self, x):
        if self.gamma_2 is None:
            if self.postnorm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            elif self.deepnorm:
                residual = x
                x = self.mlp(x)
                x = self.drop_path(x)
                x = residual * self.alpha + x
                x = self.norm2(x)
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))
        else:
            if self.postnorm:
                x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
            else:
                m = self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
                x = x + self.drop_path(self.gamma_2 * m)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, inner_patches=4, in_chans=3, embed_dim=128, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.inner_patches = inner_patches
        self.patches_resolution = self.patch_shape = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        conv_size = [size // inner_patches for size in patch_size]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=conv_size, stride=conv_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        patches_resolution = (H // self.patch_size[0], W // self.patch_size[1])
        num_patches = patches_resolution[0] * patches_resolution[1]
        x = self.proj(x).view(
            B, -1,
            patches_resolution[0], self.inner_patches,
            patches_resolution[1], self.inner_patches,
        ).permute(0, 2, 4, 3, 5, 1).reshape(B, num_patches, self.inner_patches, self.inner_patches, -1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, inner_patches=4, in_chans=3, embed_dim=128, norm_layer=None,
                 stop_grad_conv1=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.stop_grad_conv1 = stop_grad_conv1
        self.inner_patches = inner_patches
        self.patches_resolution = self.patch_shape = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        conv_size = [size // inner_patches for size in patch_size]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=conv_size, stride=conv_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x, bool_masked_pos=None, mask_token=None):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.stop_grad_conv1:
            x = x.detach() * 0.9 + x * 0.1

        if bool_masked_pos is not None:
            x = torch.nn.functional.unfold(x, kernel_size=4, stride=4, padding=0).transpose(1, 2)

            seq_len = x.shape[1]
            mask_token = mask_token.expand(B, seq_len, -1)
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

            x = torch.nn.functional.fold(x.transpose(1, 2), output_size=(H // 4, W // 4), kernel_size=4, padding=0,
                                         stride=4)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerge(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)
        self.mlp = None

    def forward(self, x):
        x0 = x[..., 0::2, 0::2, :]
        x1 = x[..., 1::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class ConvPatchMerge(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim)
        self.reduction = nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2, padding=0)
        self.mlp = None

    def forward(self, x):
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.reduction(x)
        return x


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


def _mask_1d_rel_pos_index(seq_len):
    index = torch.arange(seq_len)
    return index.view(1, seq_len) - index.view(seq_len, 1) + seq_len - 1


def _add_cls_to_index_matrix(index, num_tokens, offset):
    index = index.contiguous().view(num_tokens, num_tokens)
    new_index = torch.zeros(size=(num_tokens + 1, num_tokens + 1), dtype=index.dtype)
    new_index[1:, 1:] = index
    new_index[0, 0:] = offset
    new_index[0:, 0] = offset + 1
    new_index[0, 0] = offset + 2
    return new_index


class DecoupledRelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] + 2, 2 * window_size[1] + 2)

        num_tokens = window_size[0] * window_size[1]

        self.relative_position_bias_for_high = nn.Parameter(torch.zeros(self.num_relative_distance[0], num_heads))
        self.relative_position_bias_for_width = nn.Parameter(torch.zeros(self.num_relative_distance[1], num_heads))
        # cls to token & token 2 cls & cls to cls

        h_index = _mask_1d_rel_pos_index(window_size[0]).view(
            window_size[0], 1, window_size[0], 1).expand(-1, window_size[1], -1, window_size[1])
        h_index = _add_cls_to_index_matrix(h_index, num_tokens, 2 * window_size[0] - 1)
        self.register_buffer("relative_position_high_index", h_index)

        w_index = _mask_1d_rel_pos_index(window_size[1]).view(
            1, window_size[1], 1, window_size[1]).expand(window_size[0], -1, window_size[0], -1)
        w_index = _add_cls_to_index_matrix(w_index, num_tokens, 2 * window_size[1] - 1)

        self.register_buffer("relative_position_width_index", w_index)

    def forward(self):
        relative_position_bias = \
            F.embedding(input=self.relative_position_high_index, weight=self.relative_position_bias_for_high) + \
            F.embedding(input=self.relative_position_width_index, weight=self.relative_position_bias_for_width)
        return relative_position_bias.permute(2, 0, 1).contiguous()


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.embed = nn.Embedding(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embed.weight)

    def forward(self, x):
        # x: (B, 5, C) or (B, mask, C) or (B, bbox+mask, C)
        n = x.size(1)
        i = torch.arange(n, device=x.device)
        pos = self.embed(i).unsqueeze(0) # (N,C) --> (1,N,C) --> (B,N,C)
        return pos


class Fast_iTPN(BaseBackbone):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=512, depth_stage1=3, depth_stage2=3, depth=24,
                 num_heads=8, bridge_mlp_ratio=3., mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.0, init_values=0.1, attn_head_dim=None, norm_layer=nn.LayerNorm,
                 patch_norm=False, num_classes=1000, use_mean_pooling=False,
                 init_scale=0.01,
                 cls_token=False,
                 grad_ckpt=False,
                 stop_grad_conv1=False,
                 use_abs_pos_emb=True,
                 use_rel_pos_bias=False,
                 use_shared_rel_pos_bias=False,
                 use_shared_decoupled_rel_pos_bias=False,
                 convmlp=False,
                 postnorm=False,
                 deepnorm=False,
                 subln=False,
                 swiglu=False,
                 naiveswiglu=False,
                 bert_dir=None,
                 **kwargs):
        super().__init__()
        self.img_size = img_size
        self.mlp_ratio = mlp_ratio
        self.grad_ckpt = grad_ckpt
        self.num_main_blocks = depth
        self.depth_stage1 = depth_stage1
        self.depth_stage2 = depth_stage2
        self.depth = depth
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim
        self.convmlp = convmlp
        self.stop_grad_conv1 = stop_grad_conv1
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_shared_rel_pos_bias = use_shared_rel_pos_bias
        self.use_shared_decoupled_rel_pos_bias = use_shared_decoupled_rel_pos_bias
        self.use_decoupled_rel_pos_bias = False
        self.tokenizer = BertTokenizer.from_pretrained(bert_dir)
        bert_config = BertConfig(
            vocab_size=30522,
            hidden_size=512,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=12 * 4,
            max_position_embeddings=40,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.descript_embedding = BertEmbeddings(bert_config)
        self.descript_embedding.apply(utils.init_weights)
        self.description_patch_pos_embed = PositionEmbeddingLearned(self.embed_dim, self.embed_dim)

        mlvl_dims = {'4': embed_dim // 4, '8': embed_dim // 2, '16': embed_dim}
        # split image into non-overlapping patches
        if convmlp:
            self.patch_embed = ConvPatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=mlvl_dims['4'],
                stop_grad_conv1=stop_grad_conv1,
                norm_layer=norm_layer if patch_norm else None)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=mlvl_dims['4'],
                norm_layer=norm_layer if patch_norm else None)
        num_patches = self.patch_embed.num_patches

        if cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None
        if use_abs_pos_emb:
            if cls_token:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            else:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        if use_shared_decoupled_rel_pos_bias:
            assert self.rel_pos_bias is None
            self.rel_pos_bias = DecoupledRelativePositionBias(window_size=self.patch_embed.patch_shape,
                                                              num_heads=num_heads)

        self.subln = subln
        self.swiglu = swiglu
        self.naiveswiglu = naiveswiglu

        self.build_blocks(
            depths=[depth_stage1, depth_stage2, depth],
            dims=mlvl_dims,
            num_heads=num_heads,
            bridge_mlp_ratio=bridge_mlp_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            attn_head_dim=attn_head_dim,
            postnorm=postnorm,
            deepnorm=deepnorm,
            subln=subln,
            swiglu=swiglu,
            naiveswiglu=naiveswiglu,
            convmlp=convmlp,
        )

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)


    def build_blocks(self,
                     depths=[3, 3, 24],
                     dims={'4': 128 // 4, '8': 256, '16': 512},
                     num_heads=8,
                     bridge_mlp_ratio=3.,
                     mlp_ratio=4.0,
                     qkv_bias=True,
                     qk_scale=None,
                     window_size=None,
                     drop=0.,
                     attn_drop=0.,
                     drop_path_rate=0.,
                     norm_layer=nn.LayerNorm,
                     init_values=0.,
                     attn_head_dim=None,
                     postnorm=False,
                     deepnorm=False,
                     subln=False,
                     swiglu=False,
                     naiveswiglu=False,
                     convmlp=False,
                     ):
        dpr = iter(x.item() for x in torch.linspace(0, drop_path_rate, depths[0] + depths[1] + depths[2]))

        self.blocks = nn.ModuleList()

        if convmlp:
            self.blocks.extend([
                ConvMlpBlock(
                    dim=dims['4'],
                    mlp_ratio=bridge_mlp_ratio,
                    drop_path=next(dpr),
                    norm_layer=norm_layer,
                    init_values=0.,
                    depth=depths[-1],
                    postnorm=postnorm,
                    deepnorm=deepnorm,
                    subln=subln,
                    swiglu=False,
                    naiveswiglu=False,
                ) for _ in range(depths[0])
            ])
            self.blocks.append(ConvPatchMerge(dims['4'], norm_layer))
            self.blocks.extend([
                ConvMlpBlock(
                    dim=dims['8'],
                    mlp_ratio=bridge_mlp_ratio,
                    drop_path=next(dpr),
                    norm_layer=norm_layer,
                    init_values=0.,
                    depth=depths[-1],
                    postnorm=postnorm,
                    deepnorm=deepnorm,
                    subln=subln,
                    swiglu=False,
                    naiveswiglu=False,
                ) for _ in range(depths[1])
            ])
            self.blocks.append(ConvPatchMerge(dims['8'], norm_layer))
        else:
            self.blocks.extend([
                Block(
                    dim=dims['4'],
                    num_heads=0,
                    mlp_ratio=bridge_mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=next(dpr),
                    norm_layer=norm_layer,
                    init_values=init_values,
                    window_size=window_size,
                    depth=depths[-1],
                    postnorm=postnorm,
                    deepnorm=deepnorm,
                    subln=subln,
                    swiglu=swiglu,
                    naiveswiglu=naiveswiglu,
                ) for _ in range(depths[0])
            ])
            self.blocks.append(PatchMerge(dims['4'], norm_layer))
            self.blocks.extend([
                Block(
                    dim=dims['8'],
                    num_heads=0,
                    mlp_ratio=bridge_mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=next(dpr),
                    norm_layer=norm_layer,
                    init_values=init_values,
                    window_size=window_size,
                    depth=depths[-1],
                    postnorm=postnorm,
                    deepnorm=deepnorm,
                    subln=subln,
                    swiglu=swiglu,
                    naiveswiglu=naiveswiglu,
                ) for _ in range(depths[1])
            ])
            self.blocks.append(PatchMerge(dims['8'], norm_layer))

        ######### stage 3 ########
        self.blocks.extend([
            Block(
                dim=dims['16'],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=next(dpr),
                norm_layer=norm_layer,
                init_values=init_values,
                window_size=window_size,
                attn_head_dim=attn_head_dim,
                depth=depths[-1],
                postnorm=postnorm,
                deepnorm=deepnorm,
                subln=subln,
                swiglu=swiglu,
                naiveswiglu=naiveswiglu,
            ) for _ in range(depths[2])
        ])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.cls_token is not None:
            return {'pos_embed', 'cls_token'}
        return {'pos_embed'}


    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def finetune_track(self, cfg, patch_start_index=1):
        super().finetune_track(cfg, patch_start_index=patch_start_index)

        te_cfg = getattr(cfg.MODEL, "TE", None)
        self.visual_te_enabled = bool(te_cfg is not None and getattr(te_cfg, "ENABLE", False))
        self.visual_te_pruning_loc = list(getattr(te_cfg, "PRUNING_LOC", [3, 7, 11])) if te_cfg is not None else []
        self.visual_te_hard = bool(getattr(te_cfg, "HARD", False)) if te_cfg is not None else False
        self.visual_te_tau = float(getattr(te_cfg, "TAU", 1.0)) if te_cfg is not None else 1.0
        self.te_keep_vl = bool(getattr(te_cfg, "KEEP_VL", False)) if te_cfg is not None else False
        self.te_keep_lv = bool(getattr(te_cfg, "KEEP_LV", False)) if te_cfg is not None else False
        self.te_bidir_mode = getattr(te_cfg, "BIDIR_MODE", "sequential") if te_cfg is not None else "sequential"
        self.te_keep_vl_source = getattr(te_cfg, "KEEP_VL_SOURCE", "global") if te_cfg is not None else "global"
        self.te_policy_apply = getattr(te_cfg, "POLICY_APPLY", "post_softmax") if te_cfg is not None else "post_softmax"
        self.te_query_scope = getattr(te_cfg, "QUERY_SCOPE", "all") if te_cfg is not None else "all"
        self.te_pre_softmax_lambda = float(getattr(te_cfg, "PRE_SOFTMAX_LAMBDA", 1.0)) if te_cfg is not None else 1.0
        self.te_pre_softmax_center = getattr(te_cfg, "PRE_SOFTMAX_CENTER", "none") if te_cfg is not None else "none"
        self.te_policy_eps = float(getattr(te_cfg, "POLICY_EPS", 1e-4)) if te_cfg is not None else 1e-4
        self.te_hidden_dim = int(getattr(te_cfg, "HIDDEN_DIM", 256)) if te_cfg is not None else 256
        self.te_lang_residual_beta = float(getattr(te_cfg, "LANG_RESIDUAL_BETA", 0.1)) if te_cfg is not None else 0.1
        self.te_proto_topk_target = int(getattr(te_cfg, "PROTO_TOPK_TARGET", 4)) if te_cfg is not None else 4
        self.te_proto_topk_negative = int(getattr(te_cfg, "PROTO_TOPK_NEGATIVE", 8)) if te_cfg is not None else 8
        self.te_proto_contrast_tau = float(getattr(te_cfg, "PROTO_CONTRAST_TAU", 0.2)) if te_cfg is not None else 0.2
        self.te_safe_confirm_gamma = float(getattr(te_cfg, "SAFE_CONFIRM_GAMMA", 0.35)) if te_cfg is not None else 0.35
        self.te_safe_confirm_tau = float(getattr(te_cfg, "SAFE_CONFIRM_TAU", 0.0)) if te_cfg is not None else 0.0
        self.te_safe_confirm_max = float(getattr(te_cfg, "SAFE_CONFIRM_MAX", 0.25)) if te_cfg is not None else 0.25
        self.te_negative_gate_scale = float(getattr(te_cfg, "NEGATIVE_GATE_SCALE", 8.0)) if te_cfg is not None else 8.0
        self.te_negative_gate_floor = float(getattr(te_cfg, "NEGATIVE_GATE_FLOOR", 0.05)) if te_cfg is not None else 0.05
        self.te_word_weight_tau = float(getattr(te_cfg, "WORD_WEIGHT_TAU", 0.07)) if te_cfg is not None else 0.07
        self.te_word_template_weight = float(getattr(te_cfg, "WORD_TEMPLATE_WEIGHT", 1.0)) if te_cfg is not None else 1.0
        self.te_word_search_weight = float(getattr(te_cfg, "WORD_SEARCH_WEIGHT", 0.5)) if te_cfg is not None else 0.5
        self.te_word_learned_weight = float(getattr(te_cfg, "WORD_LEARNED_WEIGHT", 0.1)) if te_cfg is not None else 0.1
        self.lmq_prior_enabled = bool(getattr(te_cfg, "LMQ_ENABLE", False)) if te_cfg is not None else False
        self.language_query_prior_loc = list(getattr(te_cfg, "LMQ_LOC", [15])) if te_cfg is not None else []
        self.lmq_num_queries = int(getattr(te_cfg, "LMQ_NUM_QUERIES", 4)) if te_cfg is not None else 4
        self.lmq_hidden_dim = int(getattr(te_cfg, "LMQ_HIDDEN_DIM", self.te_hidden_dim)) if te_cfg is not None else self.te_hidden_dim
        self.lmq_dropout = float(getattr(te_cfg, "LMQ_DROPOUT", 0.0)) if te_cfg is not None else 0.0
        self.lmq_seed_residual = bool(getattr(te_cfg, "LMQ_SEED_RESIDUAL", False)) if te_cfg is not None else False
        self.lmq_seed_residual_gamma = float(getattr(te_cfg, "LMQ_SEED_RESIDUAL_GAMMA", 0.1)) if te_cfg is not None else 0.1
        self.lmq_decoder_enable = bool(getattr(te_cfg, "LMQ_DECODER_ENABLE", False)) if te_cfg is not None else False
        self.lmq_decoder_num_heads = int(getattr(te_cfg, "LMQ_DECODER_NUM_HEADS", 8)) if te_cfg is not None else 8
        self.lmq_decoder_dropout = float(getattr(te_cfg, "LMQ_DECODER_DROPOUT", 0.1)) if te_cfg is not None else 0.1
        self.lmq_decoder_ffn_ratio = float(getattr(te_cfg, "LMQ_DECODER_FFN_RATIO", 2.0)) if te_cfg is not None else 2.0
        self.language_state_enabled = bool(getattr(te_cfg, "LANGUAGE_STATE_ENABLE", False)) if te_cfg is not None else False
        self.language_state_updater = None
        if self.language_state_enabled:
            self.language_state_updater = LanguageTokenStateUpdater(
                self.embed_dim,
                hidden_dim=int(getattr(te_cfg, "LANGUAGE_STATE_HIDDEN_DIM", self.te_hidden_dim)),
                max_delta=float(getattr(te_cfg, "LANGUAGE_STATE_MAX_DELTA", 0.1)),
                dropout=float(getattr(te_cfg, "LANGUAGE_STATE_DROPOUT", 0.0)),
                init_gate_bias=float(getattr(te_cfg, "LANGUAGE_STATE_INIT_GATE_BIAS", -4.0)),
                source_init_gate_bias=getattr(te_cfg, "LANGUAGE_STATE_SOURCE_INIT_GATE_BIAS", None),
                init_delta_std=float(getattr(te_cfg, "LANGUAGE_STATE_INIT_DELTA_STD", 1e-4)),
                relation_layers=int(getattr(te_cfg, "LANGUAGE_STATE_RELATION_LAYERS", 1)),
                relation_heads=int(getattr(te_cfg, "LANGUAGE_STATE_RELATION_HEADS", 4)),
                visual_evidence_dim=int(getattr(te_cfg, "LANGUAGE_STATE_VISUAL_EVIDENCE_DIM", 8)),
                update_mode=str(getattr(te_cfg, "LANGUAGE_STATE_UPDATE_MODE", "residual")),
                alignment_mode=str(getattr(te_cfg, "LANGUAGE_STATE_ALIGNMENT", "position")),
                alignment_heads=int(getattr(te_cfg, "LANGUAGE_STATE_ALIGNMENT_HEADS", 4)),
            )
        self.language_query_priors = nn.ModuleList()
        if self.lmq_prior_enabled:
            if self.cat_mode != "direct":
                raise ValueError("Language multi-query prior currently requires direct token concatenation")
            for _ in self.language_query_prior_loc:
                self.language_query_priors.append(
                    LanguageMultiQueryPrior(
                        self.embed_dim, hidden_dim=self.lmq_hidden_dim,
                        num_queries=self.lmq_num_queries, dropout=self.lmq_dropout,
                        seed_residual=self.lmq_seed_residual,
                        seed_residual_gamma=self.lmq_seed_residual_gamma,
                        decoder_enable=self.lmq_decoder_enable,
                        decoder_num_heads=self.lmq_decoder_num_heads,
                        decoder_dropout=self.lmq_decoder_dropout,
                        decoder_ffn_ratio=self.lmq_decoder_ffn_ratio)
                )
        self.visual_te_predictors = nn.ModuleList()
        if self.visual_te_enabled:
            if self.cat_mode != "direct":
                raise ValueError("Visual TE currently requires direct token concatenation")
            for _ in self.visual_te_pruning_loc:
                if self.te_keep_vl or self.te_keep_lv:
                    self.visual_te_predictors.append(
                        LanguageGuidedTokenEmphasizer(
                            self.embed_dim, hidden_dim=self.te_hidden_dim,
                            hard=self.visual_te_hard, tau=self.visual_te_tau,
                            lang_residual_beta=self.te_lang_residual_beta,
                            keep_vl_source=self.te_keep_vl_source,
                            proto_topk_target=self.te_proto_topk_target,
                            proto_topk_negative=self.te_proto_topk_negative,
                            proto_contrast_tau=self.te_proto_contrast_tau,
                            safe_confirm_gamma=self.te_safe_confirm_gamma,
                            safe_confirm_tau=self.te_safe_confirm_tau,
                            safe_confirm_max=self.te_safe_confirm_max,
                            negative_gate_scale=self.te_negative_gate_scale,
                            negative_gate_floor=self.te_negative_gate_floor,
                            word_weight_tau=self.te_word_weight_tau,
                            word_template_weight=self.te_word_template_weight,
                            word_search_weight=self.te_word_search_weight,
                            word_learned_weight=self.te_word_learned_weight)
                    )
                else:
                    self.visual_te_predictors.append(
                        VisualTokenEmphasizer(self.embed_dim, num_heads=self.blocks[-1].attn.num_heads,
                                              hard=self.visual_te_hard, tau=self.visual_te_tau)
                    )

    def _z_feat(self,z,B):
        z = torch.stack(z, dim=1)
        _, T_z, C_z, H_z, W_z = z.shape

        z = z.flatten(0, 1)

        z = self.patch_embed(z)

        for blk in self.blocks[:-self.num_main_blocks]:
            z = blk(z)

        z = z.flatten(2).transpose(1, 2)
        z += self.pos_embed_z

        if T_z > 1:  # multiple memory frames
            z = z.view(B, T_z, -1, z.size()[-1]).contiguous()
            z = z.flatten(1, 2)

        return z

    def _x_feat(self,x):
        x = self.patch_embed(x)
        if not self.convmlp and self.stop_grad_conv1:  # self.convmlp==True
            x = x.detach() * 0.9 + x * 0.1
            assert self.convmlp == True, '想像失败'

        for blk in self.blocks[:-self.num_main_blocks]:
            x = blk(x)

        x = x.flatten(2).transpose(1, 2)
        x += self.pos_embed_x

        return x

    def _l_feat(self,l):
        encoded = self.tokenizer(
            l, add_special_tokens=True, truncation=True, pad_to_max_length=True,
            max_length=16, return_attention_mask=True)
        descript_id_tensor = torch.tensor(encoded['input_ids'], device=self.pos_embed_x.device)
        attention_mask = torch.tensor(encoded['attention_mask'], device=self.pos_embed_x.device)
        lang_te_mask = attention_mask.clone()
        for token_id in (self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id):
            if token_id is not None:
                lang_te_mask = lang_te_mask * (descript_id_tensor != token_id).long()
        empty_rows = lang_te_mask.sum(dim=1, keepdim=True) == 0
        lang_te_mask = torch.where(empty_rows, attention_mask, lang_te_mask)
        lang_te_mask = lang_te_mask.unsqueeze(-1).to(dtype=self.pos_embed_x.dtype)
        l = self.descript_embedding(descript_id_tensor)
        l += self.description_patch_pos_embed(l)

        return l, lang_te_mask

    def _te_query_ranges(self, temporal_len, l_len, z_len, x_len):
        scope = str(getattr(self, "te_query_scope", "all")).lower()
        if scope == "all":
            return None
        ranges = []
        l_start = temporal_len
        z_start = l_start + l_len
        x_start = z_start + z_len
        if scope in ("q0", "track0", "target0"):
            if temporal_len > 0:
                ranges.append((0, 1))
        elif scope in ("track", "target"):
            if temporal_len > 0:
                ranges.append((0, temporal_len))
        elif scope in ("track_search", "target_search"):
            if temporal_len > 0:
                ranges.append((0, temporal_len))
            ranges.append((x_start, x_start + x_len))
        elif scope == "search":
            ranges.append((x_start, x_start + x_len))
        elif scope == "visual":
            ranges.append((z_start, z_start + z_len))
            ranges.append((x_start, x_start + x_len))
        elif scope in ("track_visual", "target_visual", "track_template_search", "target_template_search"):
            if temporal_len > 0:
                ranges.append((0, temporal_len))
            ranges.append((z_start, z_start + z_len))
            ranges.append((x_start, x_start + x_len))
        else:
            raise ValueError("Unsupported TE query scope: {}".format(self.te_query_scope))
        return ranges

    def _build_post_te_policy(self, B, temporal_len, l_len, prev_decision_l,
                              prev_decision_z, prev_decision_x, dtype, device):
        prefix = []
        if temporal_len > 0:
            prefix.append(torch.ones(B, temporal_len, 1, dtype=dtype, device=device))
        if prev_decision_l is None:
            prefix.append(torch.ones(B, l_len, 1, dtype=dtype, device=device))
        else:
            prefix.append(prev_decision_l)
        return torch.cat(prefix + [prev_decision_z, prev_decision_x], dim=1)

    def _build_pre_te_bias(self, B, temporal_len, l_len, z_len, x_len,
                           prev_decision_l, prev_decision_z, prev_decision_x,
                           dtype, device):
        total_len = temporal_len + l_len + z_len + x_len
        bias = torch.zeros(B, total_len, dtype=dtype, device=device)
        l_start = temporal_len
        z_start = l_start + l_len
        x_start = z_start + z_len
        eps = max(float(getattr(self, "te_policy_eps", 1e-4)), 1e-8)
        scale = float(getattr(self, "te_pre_softmax_lambda", 1.0))
        center_mode = str(getattr(self, "te_pre_softmax_center", "none")).lower()
        visual_logs = []
        z_center = None
        x_center = None
        if center_mode == "visual":
            if prev_decision_z is not None:
                visual_logs.append(prev_decision_z.squeeze(-1).clamp_min(eps).log())
            if prev_decision_x is not None:
                visual_logs.append(prev_decision_x.squeeze(-1).clamp_min(eps).log())
        elif center_mode == "separate":
            if prev_decision_z is not None:
                z_center = prev_decision_z.squeeze(-1).clamp_min(eps).log().mean(dim=1, keepdim=True)
            if prev_decision_x is not None:
                x_center = prev_decision_x.squeeze(-1).clamp_min(eps).log().mean(dim=1, keepdim=True)
        elif center_mode != "none":
            raise ValueError("Unsupported PRE_SOFTMAX_CENTER mode: {}".format(self.te_pre_softmax_center))
        if visual_logs:
            visual_center = torch.cat(visual_logs, dim=1).mean(dim=1, keepdim=True)
        else:
            visual_center = None
        if prev_decision_l is not None:
            bias[:, l_start:z_start] = scale * prev_decision_l.squeeze(-1).clamp_min(eps).log()
        if prev_decision_z is not None:
            z_log = prev_decision_z.squeeze(-1).clamp_min(eps).log()
            if visual_center is not None:
                z_log = z_log - visual_center
            elif z_center is not None:
                z_log = z_log - z_center
            bias[:, z_start:x_start] = scale * z_log
        if prev_decision_x is not None:
            x_log = prev_decision_x.squeeze(-1).clamp_min(eps).log()
            if visual_center is not None:
                x_log = x_log - visual_center
            elif x_center is not None:
                x_log = x_log - x_center
            bias[:, x_start:x_start + x_len] = scale * x_log
        return bias

    def _fusion_feat(self,z,x,l,B,temporal_query, l_mask=None, word_reliability=None):
        temporal_len = 0
        if self.add_cls_token:
            if temporal_query is None:
                temporal_init = self.temporal_token.expand(B, 1, -1)
                temporal_init = temporal_init + self.temporal_pos_embed
                temporal_len = temporal_init.shape[1]
            else:
                temporal_len = temporal_query.shape[1]

        z_len = z.shape[1]
        x_len = x.shape[1]
        l_len = l.shape[1]

        x = combine_tokens(z, x, mode=self.cat_mode)
        x = combine_tokens(l, x, mode=self.cat_mode)

        if self.add_cls_token:
            if temporal_query is None:
                x = torch.cat([temporal_init, x], dim=1)
            else:
                x = torch.cat([temporal_query, x], dim=1)

        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        assert rel_pos_bias == None, 'rel_pos_bias not None'
        assert self.grad_ckpt == False, 'grad_ckpt != Fasle'
        prefix_len = l_len + temporal_len
        z_start = prefix_len
        x_start = z_start + z_len
        prev_decision_z = None
        prev_decision_x = None
        prev_decision_l = None
        te_predictor_idx = 0
        lmq_prior_idx = 0
        te_policy = None
        te_policy_bias = None
        te_query_ranges = self._te_query_ranges(temporal_len, l_len, z_len, x_len)
        te_aux = {}
        if self.visual_te_enabled or self.lmq_prior_enabled:
            te_aux = {
                "lang_te_language_decisions": [],
                "lang_te_language_probs": [],
                "lang_te_language_logits": [],
                "lang_te_template_logits": [],
                "lang_te_search_logits": [],
                "score_prior_search_decisions": [],
                "safe_proto_target_scores": [],
                "safe_proto_negative_scores": [],
                "safe_proto_margins": [],
                "word_level_template_scores": [],
                "word_level_direct_scores": [],
                "word_level_weights": [],
                "word_level_reliability": [],
                "word_level_template_token_scores": [],
                "word_level_search_token_scores": [],
                "lmq_prior_scores": [],
                "lmq_query_prior_maps": [],
                "lmq_query_fusion_weights": [],
                "lmq_query_prior_cosine_mean": [],
                "lmq_query_prior_cosine_max": [],
                "lmq_query_seed_cosine_mean": [],
                "lmq_query_seed_cosine_max": [],
                "lmq_query_lang_attn_cosine_mean": [],
                "lmq_query_lang_attn_cosine_max": [],
                "lmq_query_lang_attn_entropy": [],
                "lmq_query_lang_attn_max": [],
                "lmq_pooled_query_cosine_mean": [],
                "lmq_pooled_query_cosine_max": [],
                "lmq_query_vector_cosine_mean": [],
                "lmq_query_vector_cosine_max": [],
                "lmq_query_map_between_std": [],
                "lmq_prior_score_std": [],
                "lmq_query_search_attn_entropy": [],
                "lmq_query_search_attn_max": [],
                "lmq_decoder_query_delta_norm": [],
                "visual_te_template_decisions": [],
                "visual_te_search_decisions": [],
                "visual_te_template_probs": [],
                "visual_te_search_probs": [],
            }
        if self.visual_te_enabled:
            if l_mask is None:
                prev_decision_l = torch.ones(B, l_len, 1, dtype=x.dtype, device=x.device)
            else:
                prev_decision_l = l_mask.to(dtype=x.dtype, device=x.device)
            prev_decision_z = torch.ones(B, z_len, 1, dtype=x.dtype, device=x.device)
            prev_decision_x = torch.ones(B, x_len, 1, dtype=x.dtype, device=x.device)
        for block_idx, blk in enumerate(self.blocks[-self.num_main_blocks:]):
            if self.lmq_prior_enabled and block_idx in self.language_query_prior_loc:
                if lmq_prior_idx >= len(self.language_query_priors):
                    raise ValueError("Not enough language query prior modules for LMQ locations")
                l_tokens = x[:, temporal_len:temporal_len + l_len, :]
                x_tokens = x[:, x_start:x_start + x_len, :]
                lmq_out = self.language_query_priors[lmq_prior_idx](
                    l_tokens, x_tokens, lang_mask=l_mask)
                # Keep prior scores attached so tracking loss can train the prior module.
                te_aux["lmq_prior_scores"].append(lmq_out["prior_scores"])
                te_aux["lmq_query_prior_maps"].append(lmq_out["query_prior_maps"].detach())
                te_aux["lmq_query_fusion_weights"].append(lmq_out["query_fusion_weights"].detach())
                te_aux["lmq_query_prior_cosine_mean"].append(lmq_out["query_prior_cosine_mean"].detach())
                te_aux["lmq_query_prior_cosine_max"].append(lmq_out["query_prior_cosine_max"].detach())
                te_aux["lmq_query_seed_cosine_mean"].append(lmq_out["query_seed_cosine_mean"].detach())
                te_aux["lmq_query_seed_cosine_max"].append(lmq_out["query_seed_cosine_max"].detach())
                te_aux["lmq_query_lang_attn_cosine_mean"].append(lmq_out["query_lang_attn_cosine_mean"].detach())
                te_aux["lmq_query_lang_attn_cosine_max"].append(lmq_out["query_lang_attn_cosine_max"].detach())
                te_aux["lmq_query_lang_attn_entropy"].append(lmq_out["query_lang_attn_entropy"].detach())
                te_aux["lmq_query_lang_attn_max"].append(lmq_out["query_lang_attn_max"].detach())
                te_aux["lmq_pooled_query_cosine_mean"].append(lmq_out["pooled_query_cosine_mean"].detach())
                te_aux["lmq_pooled_query_cosine_max"].append(lmq_out["pooled_query_cosine_max"].detach())
                te_aux["lmq_query_vector_cosine_mean"].append(lmq_out["query_vector_cosine_mean"].detach())
                te_aux["lmq_query_vector_cosine_max"].append(lmq_out["query_vector_cosine_max"].detach())
                te_aux["lmq_query_map_between_std"].append(lmq_out["query_map_between_std"].detach())
                te_aux["lmq_prior_score_std"].append(lmq_out["prior_score_std"].detach())
                te_aux["lmq_query_search_attn_entropy"].append(lmq_out["query_search_attn_entropy"].detach())
                te_aux["lmq_query_search_attn_max"].append(lmq_out["query_search_attn_max"].detach())
                te_aux["lmq_decoder_query_delta_norm"].append(lmq_out["decoder_query_delta_norm"].detach())
                lmq_prior_idx += 1
            if self.visual_te_enabled and block_idx in self.visual_te_pruning_loc:
                if te_predictor_idx >= len(self.visual_te_predictors):
                    raise ValueError("Not enough Visual TE predictors for pruning locations")
                te_predictor = self.visual_te_predictors[te_predictor_idx]
                l_tokens = x[:, temporal_len:temporal_len + l_len, :]
                z_tokens = x[:, z_start:z_start + z_len, :]
                x_tokens = x[:, x_start:x_start + x_len, :]
                if self.te_keep_vl or self.te_keep_lv:
                    te_out = te_predictor(
                        l_tokens, z_tokens, x_tokens,
                        prev_decision_l, prev_decision_z, prev_decision_x,
                        keep_vl=self.te_keep_vl, keep_lv=self.te_keep_lv,
                        bidir_mode=self.te_bidir_mode,
                        lang_mask=l_mask,
                        word_reliability=word_reliability)
                    if self.te_keep_lv:
                        prev_decision_l = te_out["language_decision"]
                        te_aux["lang_te_language_decisions"].append(prev_decision_l.detach())
                        te_aux["lang_te_language_probs"].append(te_out["language_probs"].detach())
                        te_aux["lang_te_language_logits"].append(te_out["language_logits"])
                    if self.te_keep_vl:
                        prev_decision_z = te_out["template_decision"]
                        prev_decision_x = te_out["search_decision"]
                        z_prob = te_out["template_probs"]
                        x_prob = te_out["search_probs"]
                        te_aux["lang_te_template_logits"].append(te_out["template_logits"])
                        te_aux["lang_te_search_logits"].append(te_out["search_logits"])
                        for aux_key in ("safe_proto_target_scores",
                                        "safe_proto_negative_scores",
                                        "safe_proto_margins",
                                        "word_level_template_scores",
                                        "word_level_direct_scores",
                                        "word_level_weights",
                                        "word_level_reliability",
                                        "word_level_template_token_scores",
                                        "word_level_search_token_scores"):
                            if aux_key in te_out:
                                te_aux[aux_key].append(te_out[aux_key].detach())
                    else:
                        z_prob = torch.cat([prev_decision_z, 1.0 - prev_decision_z], dim=-1)
                        x_prob = torch.cat([prev_decision_x, 1.0 - prev_decision_x], dim=-1)
                else:
                    _, z_prob, prev_decision_z = te_predictor(z_tokens, prev_decision_z)
                    _, x_prob, prev_decision_x = te_predictor(x_tokens, prev_decision_x)
                te_policy = None
                te_policy_bias = None
                if self.te_policy_apply == "post_softmax":
                    policy_l = prev_decision_l if self.te_keep_lv else None
                    te_policy = self._build_post_te_policy(
                        B, temporal_len, l_len, policy_l, prev_decision_z, prev_decision_x, x.dtype, x.device)
                elif self.te_policy_apply == "pre_softmax":
                    policy_l = prev_decision_l if self.te_keep_lv else None
                    policy_z = prev_decision_z if (self.te_keep_vl or not (self.te_keep_vl or self.te_keep_lv)) else None
                    policy_x = prev_decision_x if (self.te_keep_vl or not (self.te_keep_vl or self.te_keep_lv)) else None
                    te_policy_bias = self._build_pre_te_bias(
                        B, temporal_len, l_len, z_len, x_len,
                        policy_l, policy_z, policy_x, x.dtype, x.device)
                elif self.te_policy_apply == "none":
                    pass
                else:
                    raise ValueError("Unsupported TE policy apply mode: {}".format(self.te_policy_apply))
                te_aux["visual_te_template_decisions"].append(prev_decision_z.detach())
                te_aux["visual_te_search_decisions"].append(prev_decision_x.detach())
                te_aux["score_prior_search_decisions"].append(prev_decision_x)
                te_aux["visual_te_template_probs"].append(z_prob.detach())
                te_aux["visual_te_search_probs"].append(x_prob.detach())
                te_predictor_idx += 1
            x,attn = blk(x, policy=te_policy, policy_bias=te_policy_bias,
                         policy_query_ranges=te_query_ranges)

        x = self.norm(x)
        if self.visual_te_enabled and prev_decision_z is not None:
            if prev_decision_l is not None:
                te_aux["lang_te_language_quality"] = prev_decision_l.sum(dim=1).squeeze(-1).detach()
            te_aux["visual_te_template_quality"] = prev_decision_z.sum(dim=1).squeeze(-1).detach()
            te_aux["visual_te_search_quality"] = prev_decision_x.sum(dim=1).squeeze(-1).detach()

        return x,attn,te_aux

    def _split_feat(self,attn,topk):
        #fusion_feat(bs,temporal_l + descript_l + z_l + x_l)
        lens_x = self.pos_embed_x.shape[1]
        attn = torch.mean(attn,dim=1)
        # x = fusion_feat[:, -lens_x:]
        l2s = attn[:,topk,-lens_x:]
        max,index = torch.sort(l2s,dim=1,descending=True)
        top_index = index[:,:topk]

        return top_index,l2s

    def _finder(self,x,index):
        index_expanded = index.unsqueeze(2)
        result = torch.gather(x, 1, index_expanded.expand(-1, -1, 512))
        return result


    def forward_features(self, z, x, l, temporal_query=None, top_K=None,
                         word_reliability=None, language_token_state=None,
                         language_token_mask=None):
        B = x.shape[0]
        z_feat = self._z_feat(z,B)
        x_feat = self._x_feat(x)
        if language_token_state is None:
            l_feat, l_mask = self._l_feat(l)
        else:
            l_feat = language_token_state.to(device=x_feat.device, dtype=x_feat.dtype)
            if l_feat.dim() != 3 or l_feat.shape[0] != B or l_feat.shape[-1] != self.embed_dim:
                raise ValueError(
                    "language_token_state must have shape (B,L,{}) but got {}".format(
                        self.embed_dim, tuple(l_feat.shape)))
            if language_token_mask is None:
                l_mask = torch.ones(
                    l_feat.shape[0], l_feat.shape[1], 1,
                    dtype=x_feat.dtype, device=x_feat.device)
            else:
                l_mask = language_token_mask.to(device=x_feat.device, dtype=x_feat.dtype)
                if l_mask.dim() == 2:
                    l_mask = l_mask.unsqueeze(-1)
                if l_mask.shape[:2] != l_feat.shape[:2]:
                    raise ValueError(
                        "language_token_mask shape {} must match language_token_state first dims {}".format(
                            tuple(l_mask.shape), tuple(l_feat.shape[:2])))
        fusion_feat,attn,te_aux = self._fusion_feat(
            z_feat,x_feat,l_feat,B,temporal_query,l_mask=l_mask,
            word_reliability=word_reliability)  #attn(bs,head_num,l,l)
        top_index,att_l2s = self._split_feat(attn,top_K)
        l2s = self._finder(x_feat,top_index)
        if self.training:
            attn = attn.detach()
            att_l2s = att_l2s.detach()
            l2s = l2s.detach()
        aux_dict = {"attn": attn,
                    "attn_l2s": att_l2s,
                    "temproal_token": l2s}
        aux_dict.update(te_aux)

        return fusion_feat, aux_dict

    def forward(self, z, x, l, temporal_query, top_K, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic ViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]
            l (list.str): descript of search refion

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        x, aux_dict = self.forward_features(
            z, x, l, temporal_query, top_K,
            word_reliability=kwargs.get("word_reliability", None),
            language_token_state=kwargs.get("language_token_state", None),
            language_token_mask=kwargs.get("language_token_mask", None))

        return x, aux_dict


@register_model
def fast_itpn_tiny_1112_patch16_224(pretrained=False, **kwargs):
    model = Fast_iTPN(
        patch_size=16, embed_dim=384, depth_stage1=1, depth_stage2=1, depth=12, num_heads=6, bridge_mlp_ratio=3.,
        mlp_ratio=3., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        convmlp=True,
        naiveswiglu=True,
        subln=True,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def fast_itpn_small_2220_patch16_224(pretrained=False, **kwargs):
    model = Fast_iTPN(
        patch_size=16, embed_dim=384, depth_stage1=2, depth_stage2=2, depth=20, num_heads=6, bridge_mlp_ratio=3.,
        mlp_ratio=3., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        convmlp=True,
        naiveswiglu=True,
        subln=True,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def fast_itpn_base_3324_patch16_224(pretrained=False,bert_dir=None, **kwargs ):
    model = Fast_iTPN(
        patch_size=16, embed_dim=512, depth_stage1=3, depth_stage2=3, depth=24, num_heads=8, bridge_mlp_ratio=3.,
        mlp_ratio=3., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        convmlp=True,
        naiveswiglu=True,
        subln=True,
        bert_dir=bert_dir,
        **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
            checkpoint = torch.load(pretrained, map_location="cpu")
            #print(checkpoint.keys())
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['net'], strict=False)
            print(missing_keys, unexpected_keys)
            print('Load pretrained model from: ' + pretrained)

    return model


@register_model
def fast_itpn_large_2240_patch16_256(pretrained=False, **kwargs):
    model = Fast_iTPN(
        patch_size=16, embed_dim=768, depth_stage1=2, depth_stage2=2, depth=40, num_heads=12, bridge_mlp_ratio=3.,
        mlp_ratio=3., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        convmlp=True,
        naiveswiglu=True,
        subln=True,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
