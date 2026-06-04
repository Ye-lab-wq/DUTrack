# DUTrack TEC 实现修改记录

更新时间：2026-06-01

## 1. 修改目标

本次修改开始实现 TEC（Tracking Evidence Calibration）第一版。核心目标不是重新训练 backbone，也不是在 score map 上叠加语言先验，而是在 DUTrack 原始 tracking head 之前，对 search token 做一个轻量、任务相关、target-centric 的后验特征校准。

基线保持默认不变：`cfg.MODEL.TEC.ENABLE = False`。只有显式打开 TEC 时，才会引入新增模块。

## 2. 数学定义

原 DUTrack 可以抽象为：

```text
H = F_backbone(Z, X, L)
H_X = SearchTokens(H)
s_i^0, b_i^0 = Head(H_X)
```

其中 `s_i^0` 是原 head 自己产生的 score，例如当前 CENTER head 的 `score_map_ctr`。本次不把它强行解释为空间 softmax，也不直接改写它。

TEC 的第一版改成：

```text
L_src = L_raw
M_L = valid_language_mask
p_Z = Pool_center(H_Z)
Delta_i = TEC_theta(H_X_i, L_src, p_Z)
H_X_i' = H_X_i + tanh(gamma) * Delta_i
s_i, b_i = Head(H_X')
```

具体实现为 token-level evidence attention：

```text
q_i = LN(W_x H_X_i + W_z^x p_Z)
k_j = LN(W_l L_src_j + W_z^l p_Z)
a_ij = masked_softmax_j(q_i k_j / sqrt(d), M_L)
e_i = sum_j a_ij V_l L_src_j
Delta_i = W_o e_i
```

这里 `Delta_i` 只作用在 head 前的 search feature 上。它不是 score prior，因为它不直接修改 `score_map`、不乘 Hann window、不重定义原始 head 的分类分数。

## 3. 关键边界

1. TEC 不直接改 `score_map`，避免退化成 score prior / GSB 新版本。
2. TEC 默认使用 `L_raw`，不完全依赖弱融合后可能已被稀释的 `H_L`。
3. `LANG_SOURCE=fuse` 只作为后续保留后路：`L_src = Fuse(L_raw, H_L)`，但第一阶段不启用。
4. `s_i^0` 保持为原 DUTrack head 的输出，不引入额外 spatial softmax 训练范式。
5. 不做 token-level pseudo label，不做规则堆砌；若第一版无效，后续最多只考虑轻量 ranking / contrastive 约束。
6. TEC attention 必须使用有效语言 token mask，padding、`[CLS]`、`[SEP]` 不允许参与 evidence attention。

## 4. 代码修改

### 4.1 新增 `lib/models/dutrack/tec.py`

新增 `TrackingEvidenceCalibration`：

- 输入：`H_X`、`L_raw`、`H_Z`，可选 `H_L`。
- 输出：校准后的 `H_X'` 和诊断量。
- `gamma` 用 `tanh(gamma)` 限幅，初始值为 `0.01`；baseline 默认关闭，因此不会影响旧实验。
- `H_Z` 使用模板中心区域池化作为 target prototype；多模板时先按单模板 token 数拆分，再对中心区域和模板维度求平均。
- `LANG_SOURCE=fuse` 时才创建 `lang_fuse` 参数；第一阶段 `raw` 模式下可训练参数全部是实际参与前向的 TEC 参数。
- attention logits 会用 `l_mask` 屏蔽 padding 和 special token；mask 在 softmax 前强制转到 logits 的 device，dtype 为 `bool`。
- 如果某个样本出现全 mask 异常，前向会临时回退为全 token 可见以避免 NaN，但 `tec_valid_token_count` 仍记录为 0，便于定位数据问题。
- 诊断量包括 `raw_gamma/tanh_gamma`、gamma 前后的 delta norm、`z_proto_norm/z_proto_std`、有效语言 token 数和 delta 相对 feature 的比例。

### 4.2 修改 `lib/models/dutrack/itpn.py`

在 backbone 的 `aux_dict` 中暴露：

```text
l_raw: 进入跨模态 transformer 之前的 language embedding
l_mask: 有效语言 token mask，排除 padding / [CLS] / [SEP]
len_l: language token 数
len_z: template token 数
```

这样 TEC 可以从原始语言分支取证据，而不是只依赖已经融合后的 `H_L`。

注意：当前 backbone 自身的跨模态 fusion 仍然接收完整 language embedding；本次 mask 首先用于 TEC evidence attention，目的是避免 TEC 的后验校准被 padding/special token 主导。

### 4.3 修改 `lib/models/dutrack/dutrack.py`

在 `enc_opt = feat_last[:, -self.feat_len_s:]` 之后、`forward_head()` 之前插入 TEC：

```text
enc_opt = TEC(enc_opt, l_raw, h_z, h_l, l_mask)
```

随后仍然走原来的：

```text
att = enc_opt @ temporal_query
opt = enc_opt * att
out = forward_head(opt)
```

因此 head 的输出结构、loss、测试阶段的 `score_map + Hann window + cal_bbox` 流程都不改变。

由于 TEC 会改变 `enc_opt` 的尺度，`att = enc_opt @ temporal_query` 的数值尺度也可能变化。因此新增 `tec_enc_norm_before/after` 和 `tec_head_att_mean/std`，专门观察 head 前乘性门控是否被 TEC 放大或压扁。

### 4.4 修改 `lib/config/dutrack/config.py`

新增默认关闭配置：

```yaml
MODEL:
  TEC:
    ENABLE: false
    EVIDENCE_DIM: 128
    GAMMA_INIT: 0.01
    LANG_SOURCE: raw
    TARGET_POOL: center
    CENTER_RATIO: 0.5
    MIN_VALID_TOKENS: 3
    DROPOUT: 0.0
    FREEZE_BACKBONE: false
    FREEZE_HEAD: false
```

### 4.5 修改 `lib/train/actors/dutrack.py`

训练日志新增诊断量：

```text
TEC/tec_raw_gamma
TEC/tec_tanh_gamma
TEC/tec_delta_norm_before_gamma
TEC/tec_delta_norm_after_gamma
TEC/tec_delta_to_feature_ratio
TEC/tec_attn_entropy
TEC/tec_valid_token_count
TEC/tec_low_valid_token_ratio
TEC/tec_z_proto_norm
TEC/tec_z_proto_std
TEC/tec_enc_norm_before
TEC/tec_enc_norm_after
TEC/tec_head_att_mean
TEC/tec_head_att_std
```

这些值不参与 loss，只用于观察 TEC 是否真的在产生非零、非坍缩的校准。

### 4.6 CENTER head 梯度路径边界

当前 `CENTER` head 的前向为：

```text
opt_feat -> score_map_ctr, size_map, offset_map
bbox = cal_bbox(score_map_ctr, size_map, offset_map)
```

训练 loss 为：

```text
loss = giou/l1(pred_boxes, gt_boxes) + focal(score_map_ctr, gt_gaussian_map)
```

需要注意：

- `focal` 对 `score_map_ctr` 是连续可导的，因此会通过 center score 分支回传到 `opt_feat -> enc_opt -> TEC`。
- `giou/l1` 通过 `pred_boxes` 回传。`pred_boxes` 中的 `size/offset` 来自 `size_map/offset_map` 在选中位置的 gather，因此会通过 size/offset 分支回传到 `opt_feat -> enc_opt -> TEC`。
- 但 `cal_bbox()` 中的位置由 `torch.max(score_map_ctr)` 得到，`idx` 是离散索引，不可导。因此 `giou/l1` 不能直接教 TEC 把 center peak 移到正确位置；这部分主要依赖 focal loss。
- 第一阶段冻结 head 时，TEC 仍能收到来自 frozen conv head 的输入梯度，但不能改变 head 参数本身。如果 `focal` 没有有效改变 score 分布，`giou/l1` 对位置纠错会比较弱。

因此第一阶段的判断不能只看 `giou/l1` 是否下降，还要看 `score_map` 相关指标和 `tec_head_att_mean/std` 是否稳定。

### 4.7 修改 `.gitignore`

仓库已有 `models/` 忽略规则，会连带忽略新建的 `lib/models/dutrack/tec.py`。因此新增精确反忽略规则，让 TEC 源码文件后续可以被正常版本管理。

## 5. 第一阶段实验配置

新增配置文件：

```text
experiments/dutrack/dutrack_384_full_tec_stage1.yaml
```

第一阶段设置为：

```yaml
MODEL:
  TEC:
    ENABLE: true
    LANG_SOURCE: raw
    GAMMA_INIT: 0.01
    MIN_VALID_TOKENS: 3
    FREEZE_BACKBONE: true
    FREEZE_HEAD: true
```

这意味着：

- backbone 不更新；
- 原 tracking head 不更新；
- 只训练 TEC；
- `BertEmbeddings` 和 `description_patch_pos_embed` 属于 backbone，因此第一阶段也被冻结；
- loss 仍然是原来的 `giou + l1 + focal`；
- TEC 必须通过改变 head 前 search token，让固定 head 得到更好的 box 和 score map。

当前 384 full 配置不是单模板：`DATA.TEMPLATE.NUMBER=3`，测试侧 `TEST.TEMPLATE_NUMBER=3`，tracker 也有 memory frame 选择逻辑。TEC 的 `p_Z` 不是重新做一个模板 prototype 打分器，而是把每个模板按单模板 token 数拆开，在中心区域池化后跨模板平均，作为 language-search evidence attention 的 target anchor。它不直接产生 score，不参与模板相似度 ranking，因此和之前失败的模板 prototype / score prior 路线不同。

`CENTER_RATIO=0.5` 对 192 模板、stride 16 来说是 12x12 token 中取中心 6x6；对 128 模板是 8x8 中取中心 4x4。它覆盖 25% token，不是全模板平均，也不是过窄的中心点。第一阶段先保留 0.5，通过 `tec_z_proto_norm/std` 和结果判断是否过多引入背景；若后续发现 prototype 方差过低或语言敏感性仍无差异，再做 0.33 的轻量 sanity ablation。

当前 HOOT 默认不提供自然语言描述。若数据集中没有人工语言标注，`HOOTDataset` 必须让 `text_description=None`，使 tracker 回退到 BLIP 生成描述；不能把类别名直接作为 `init_text_description`，否则会绕过 BLIP，退化成 class-level conditioning。若 `tec_valid_token_count < 3`，TEC 更接近弱类别条件，而不是 phrase-level grounding。第一阶段仍可用 HOOT 做 normal / wrong / generic 的语言敏感性检查，但不应期待它直接解决属性、实例级描述或遮挡状态表达。`tec_low_valid_token_ratio` 用于统计 batch 中低信息量语言的比例；若长期接近 1，需要优先检查语言来源，再判断 TEC 的上限。

如果这一步完全无效，优先检查：

1. `tec_delta_norm_after_gamma / tec_enc_norm_before` 是否长期接近 0；
2. `tec_raw_gamma` 和 `tec_tanh_gamma` 是否一直停留在初始值附近；
3. `tec_attn_entropy` 是否接近均匀分布；
4. `tec_valid_token_count` 是否正常，不能被 padding/special token 污染；
5. `tec_low_valid_token_ratio` 是否长期接近 1；若是，说明当前语言基本只是类别名，TEC 只能做弱 class-level 校准；
6. `tec_head_att_mean/std` 是否因 `enc_opt @ temporal_query` 被异常放大或压扁；
7. normal / wrong / generic language 是否仍然无法拉开差距。

只有确认 TEC 不是实现无效后，才进入第二阶段：冻结 backbone，放开 head + TEC 微调。仍不允许直接改 score map。

## 6. 兼容性说明

1. 默认 `ENABLE=False`，旧实验配置不受影响。
2. 加载旧 DUTrack checkpoint 时，TEC 参数会作为 missing keys 出现，这是预期行为。
3. 测试 TEC 时必须使用训练过 TEC 的 checkpoint；否则打开 TEC 但加载旧 checkpoint 会得到随机初始化的 TEC。

## 7. 最小验证

建议先做语法和配置验证：

```bash
python -m py_compile \
  lib/models/dutrack/tec.py \
  lib/models/dutrack/dutrack.py \
  lib/models/dutrack/itpn.py \
  lib/train/actors/dutrack.py \
  lib/config/dutrack/config.py
```

然后做一次短训练 smoke test，确认 learnable parameters 里只有 `tec.*`：

```bash
python tracking/train.py --script dutrack --config dutrack_384_full_tec_stage1
```

## 8. 阶段性实验结果与结论

### 8.1 OTB-Lang：TEC Stage-1 标准评测

评测文件：

```text
output/test/result_plots/otb_lang_tec_stage1/eval_data.pkl
```

结果如下：

| Tracker | AUC | OP50 | OP75 | Precision | Norm Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 baseline ep47 | 70.76 | 89.59 | 53.97 | 94.38 | 97.14 |
| A1 TEC normal ep5 | 71.82 | 90.48 | 57.18 | 94.72 | 96.97 |
| A1 TEC wrong ep5 | 71.57 | 90.07 | 56.93 | 94.45 | 96.67 |
| A1 TEC generic ep5 | 71.74 | 90.09 | 57.25 | 94.64 | 96.83 |

观察：

- TEC normal 相对 baseline 有整体收益：AUC `+1.06`，OP75 `+3.21`。
- wrong/generic 也接近 normal，说明整体收益不能直接归因于语言语义接地。
- 当前更稳妥的结论是：Stage-1 TEC 证明了 head 前 posterior feature calibration 有效，但尚未证明稳定 language-grounded calibration 有效。

### 8.2 OTB-Lang：per-sequence 与 hard-negative 诊断

per-sequence 诊断文件：

```text
output/diagnostics/tec_stage1_per_sequence.csv
```

统计结论：

```text
normal > A0: 34 / 48
normal > max(wrong, generic): 20 / 48
```

hard-negative 配对诊断文件：

```text
output/diagnostics/tec_stage1_hard_negative_gap_paired_summary.csv
```

只保留 normal / wrong / generic 都存在的 `(dataset, sequence, frame)` 后，共有 `166` 个 paired sequence-frame。核心结果：

| Metric | Mean | Positive | Negative | Zero |
| --- | ---: | ---: | ---: | ---: |
| normal - wrong hard-negative gap | 0.0174 | 66 | 100 | 0 |
| normal - generic hard-negative gap | 0.0055 | 98 | 68 | 0 |
| normal - wrong peak_inside_gt | 0.0301 | 5 | 0 | 161 |
| normal - generic peak_inside_gt | 0.0060 | 2 | 1 | 163 |

观察：

- Coupon 上 normal 对 wrong/generic 的 hard-negative gap 优势明显。
- Bird1、Human4、Human9、Trans 等序列不支持稳定 normal 优势。
- hard-negative gap 不足以支撑“TEC 已经稳定语言接地”的结论。

### 8.3 HOOT：评估口径与数据来源

HOOT 官方信息：

```text
https://hootbenchmark.org/
https://hootbenchmark.org/download
https://github.com/gzdshn/hoot-toolkit
```

HOOT 标注中同时包含：

```text
rot_bb: rotated bounding box
aa_bb: axis-aligned bounding box
```

当前 DUTrack 本地 `HOOTDataset` 使用 `aa_bb` 转为 `xywh` 后走 pytracking/DUTrack OPE 指标，即 AUC、OP50、OP75、Precision、Norm Precision。该结果适合内部相同口径对比，但不是 HOOT 官方完整的 rotated-bbox / occlusion-aware 分组评估。

本地数据位置：

```text
/media/b520/KESU1/HOOT
```

本地检查结果：

```text
size: 131G
classes: 74
sequence dirs: 130
test.txt: 130
train.txt: 451
实际 anno.json 数量: 130
```

这说明当前本地只下载了解压后的 HOOT test split，而不是完整 581 序列。shell history 中保留的下载命令为：

```bash
python tracking/download_hoot.py --dest /media/b520/KESU1/HOOT --split test
```

注意：当前工作区中 `tracking/download_hoot.py` 源码已经不存在，只剩：

```text
tracking/__pycache__/download_hoot.cpython-311.pyc
output/hoot_download.pid
output/hoot_download.log
```

其中 `output/hoot_download.log` 为空，不能恢复完整下载日志。若后续需要重新下载，优先使用官方 toolkit 或重新恢复当时的 `tracking/download_hoot.py`。

### 8.4 HOOT-random50：class fallback 旧口径

旧评测文件：

```text
output/test/result_plots/hoot_random50_tec_stage1/eval_data.pkl
```

该版本中 HOOT 缺失人工语言时曾错误回退为类别名，导致 normal 不是 BLIP-normal，而是 class-level normal。结果如下：

| Tracker | AUC | OP50 | OP75 | Precision | Norm Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 baseline ep47 | 63.39 | 78.20 | 46.85 | 69.47 | 89.50 |
| A1 TEC normal ep5 | 59.64 | 72.36 | 42.93 | 63.18 | 84.28 |
| A1 TEC wrong ep5 | 62.45 | 76.25 | 45.68 | 66.52 | 87.84 |
| A1 TEC generic ep5 | 61.28 | 74.32 | 44.88 | 65.41 | 86.10 |

该结果不能作为 BLIP-normal 的结论，只能说明单靠类别词初始化在 HOOT 上明显不足。

### 8.5 HOOT-random50：BLIP normal 修正口径

修正后评测文件：

```text
output/test/result_plots/hoot_random50_tec_stage1_blip/eval_data.pkl
```

修正内容：

- HOOT 无人工语言时 `text_description=None`，tracker 回退到 BLIP。
- wrong/generic 不依赖 BLIP 内容，因此与旧结果一致。
- wrong fallback 和 shuffle 已改为确定性扰动，避免随机性污染语言敏感性评估。

结果如下：

| Tracker | AUC | OP50 | OP75 | Precision | Norm Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 baseline ep47 | 63.80 | 78.98 | 47.60 | 70.36 | 90.02 |
| A1 TEC normal ep5 | 62.48 | 75.98 | 45.60 | 66.34 | 88.06 |
| A1 TEC wrong ep5 | 62.45 | 76.25 | 45.68 | 66.52 | 87.84 |
| A1 TEC generic ep5 | 61.28 | 74.32 | 44.88 | 65.41 | 86.10 |

与 class fallback 旧口径相比：

```text
A1 TEC normal: 59.64 -> 62.48  (+2.84 AUC)
A1 TEC wrong:  62.45 -> 62.45  (+0.00 AUC)
A1 TEC generic:61.28 -> 61.28  (+0.00 AUC)
```

观察：

- BLIP 对 HOOT normal 明显重要，单靠类别词初始化不够。
- TEC normal 仍低于 A0 baseline：`62.48` vs `63.80`。
- normal 与 wrong 几乎没有差距：`62.48` vs `62.45`，per-sequence 上 normal-wrong mean 约 `+0.03`，normal 优于 wrong 为 `22/50`，弱于 wrong 为 `27/50`。
- generic 低于 normal/wrong，说明完全泛化语言有一定损失，但 wrong 接近 normal 表明 TEC 未稳定利用语义正确性。

### 8.6 HOOT-all baseline 语言敏感性

评测文件：

```text
output/test/result_plots/hoot_all_compare/eval_data.pkl
```

结果如下：

| Tracker | AUC | OP50 | OP75 | Precision | Norm Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline normal | 63.89 | 78.64 | 48.05 | 69.54 | 89.57 |
| baseline shuffle | 63.50 | 78.48 | 47.62 | 69.20 | 89.15 |
| baseline wrong | 63.18 | 78.01 | 47.62 | 69.06 | 88.16 |
| baseline generic | 64.17 | 79.08 | 48.10 | 70.01 | 89.99 |
| baseline initial-only BLIP control | 63.83 | 78.67 | 47.94 | 69.52 | 89.64 |

观察：

- baseline normal 整体正常，但 generic 反而最高，说明 HOOT 当前不是强语言语义评估集合。
- normal 与 initial-only BLIP control 接近，说明动态 BLIP 更新收益有限，甚至可能在重遮挡下引入噪声。
- wrong 低于 normal，但差距不大，仍不能证明 DUTrack 已经稳定利用语言语义。

### 8.7 阶段性结论

当前阶段最稳结论：

1. TEC Stage-1 在 OTB-Lang 上能提升整体定位表现，尤其 OP75 提升明显，说明 head 前 posterior feature calibration 是有效方向。
2. normal / wrong / generic 的差距不足，hard-negative gap 也不稳定，因此不能声称 TEC 已经实现稳定 language-grounded calibration。
3. HOOT 进一步说明 BLIP 描述比类别词初始化更重要；但在重遮挡场景中，BLIP 动态更新可能引入噪声，wrong/generic 的稳定文本反而可能局部更稳。
4. HOOT 上 TEC normal 仍低于 A0 baseline，说明 OTB-Lang 5 轮 Stage-1 TEC 对 HOOT 重遮挡分布泛化不足。
5. 下一步不应立刻加复杂 loss。当前优先进入 Stage-2 explicit evidence layer，结合 per-sequence 归因和 hard-negative gap 判断 evidence representation 是否成立。

---

## 9. Stage-2 Evidence Layer 方案修订记录

### 9.1 本次修订动机

Stage-1 TEC 已经证明 head 前 residual calibration 位置有效，但没有证明稳定 language-grounded calibration。Stage-2 不再沿用旧的 TEC residual 变体，也不把 head 微调作为当前主线，否则会重新混入普通 head adaptation，无法判断收益是否来自 explicit evidence layer。

本次修订将 Stage-2 收紧为：

```text
freeze backbone
freeze head
train explicit tracking evidence layer only
```

核心不是比较 `Δscore` 改了多少，而是构造一个目标条件化的 region-level evidence layer。

---

### 9.2 两个退化风险与建模层面处理

风险 1：

```text
s_i / reliability 退化成 visual score prior 或 targetness gate。
```

设计处理：

```text
s_i 不直接从 raw H_X_i 或 E_x_i 产生；
s_i 只从 language-visual interaction feature G_i 经过 C_i / u_i 间接产生；
s_i 不直接改 score_map，只调节 head-pre residual strength；
s_i = 1 + β · tanh(u_i)，u_i = W_e C_i，β 较小；
因此 s_i 只能在 [1 - β, 1 + β] 内调节 residual，不能像 score prior 一样打开或关闭空间位置。
```

风险 2：

```text
C_i 忽略语言，退化成 visual-only adapter。
```

设计处理：

```text
M_i = Σ_j A_ij · W_v E_l_j
M_0 = masked_mean_j(W_v E_l_j, M_L^sem)

D_raw_i = M_i - M_0
d_mag_i = clamp(||D_raw_i||_2 / sqrt(d_e), max=τ_d)
D_gate_i = ||D_raw_i||_2 / (||D_raw_i||_2 + eps_d)
D_dir_i = LN(D_raw_i)
D_i = D_gate_i · d_mag_i · D_dir_i

G_i = MLP_g([D_i, E_x_i ⊙ D_i]) + D_i
C_i = MLP_c(G_i)
u_i = W_e C_i
ΔH_i = W_o C_i
```

这里 `M_i` 只是 region i 从 language tokens 读出的候选证据，不直接作为 evidence。`D_raw_i = M_i - M_0` 才是 region-conditioned language residual。不能直接写成 `D_i = LN(D_raw_i)`，否则当 `D_raw_i` 很小时，LayerNorm 会重新放大小差异，破坏“语言无信息时 residual 自然变弱”的设计。

`D_i = D_gate_i · d_mag_i · LN(D_raw_i)` 保留残差幅值，同时在 `D_raw_i` 极小时用 `D_gate_i` 压制 LayerNorm 的数值噪声。若 language attention 退化为均匀，或者语言没有提供 token-selective 信息，则 `D_raw_i / D_gate_i / D_i` 都会接近 0，后续 `G_i / C_i / ΔH_i` 自然变弱。

`G_i` 使用 `MLP_g([D_i, E_x_i ⊙ D_i]) + D_i`，目的是给 evidence layer 一个语言残差启动路径，避免乘性交互 `E_x_i ⊙ D_i` 在短训早期启动过慢。这个 skip 仍然只来自 `D_i`，不引入 raw visual 到 residual 的直接旁路。

关键约束：

```text
ΔH_i 不允许存在 raw H_X_i -> ΔH_i 的 visual-only direct residual path；
C_i 只从 interaction feature G_i 得到；
语言必须通过 D_raw_i / D_i 进入 residual path；
G_i 不直接拼接 raw E_x_i，避免 visual-only bypass；
u_i = W_e C_i 提供跨 region 共享 evidence coordinate。
```

这样不是靠训练后检测模型有没有退化，而是在结构上削弱退化路径。

---

### 9.3 Stage-2 数学形式

建议第一版写成：

```text
z_proto = TargetPool(H_Z)

E_x_i = LN(W_x H_X_i + W_z^x z_proto)
E_l_j = LN(W_l L_src_j + W_z^l z_proto)

A_ij = masked_softmax_j((E_x_i W_q)(E_l_j W_k)^T / τ, M_L)
M_i  = Σ_j A_ij · W_v E_l_j

M_0      = masked_mean_j(W_v E_l_j, M_L^sem)
D_raw_i  = M_i - M_0
d_mag_i  = clamp(||D_raw_i||_2 / sqrt(d_e), max=τ_d)
D_gate_i = ||D_raw_i||_2 / (||D_raw_i||_2 + eps_d)
D_dir_i  = LN(D_raw_i)
D_i      = D_gate_i · d_mag_i · D_dir_i

G_i  = MLP_g([D_i, E_x_i ⊙ D_i]) + D_i
C_i  = MLP_c(G_i)

u_i  = W_e C_i
s_i  = 1 + β · tanh(u_i)
ΔH_i = W_o C_i

H_X'_i = H_X_i + γ · s_i · ΔH_i
```

含义：

| 符号 | 含义 |
|---|---|
| `D_raw_i` | unnormalized region-conditioned language residual |
| `D_gate_i` | tiny-residual damping factor before direction normalization |
| `D_i` | magnitude-preserving and noise-damped region-conditioned language evidence residual |
| `G_i` | language-visual interaction feature |
| `C_i` | region-level tracking evidence representation |
| `u_i` | shared evidence coordinate / readout for region comparability |
| `s_i` | bounded calibration strength |
| `ΔH_i` | evidence-conditioned head-pre residual |

初始化约束：

```text
β = 0.25；
W_e.weight = 0；
W_e.bias = 0；
=> u_i = 0, s_i = 1；

W_o 使用受控小随机初始化，当前 σ_o = 1e-3；
γ 使用 A1 可比初始化，当前 γ_init = 0.01；
不能同时把 W_o 和 γ 初始化为 0。
```

这使初始 residual 很小但非零：不会一开始破坏 frozen head 的输入分布，同时 `W_o / γ / C_i` 仍有梯度。`s_i` 初始为 1 是正确的中性状态，但必须记录 `s_i` 偏离 1 的幅度，防止它长期不动。

`Δscore_i` 只作为 head readout diagnostic：

```text
Δscore_i = score_i(H_X + γ · s_i · ΔH_i) - score_i(H_X)
```

它用于判断 frozen head 是否读出了 evidence-conditioned residual，不定义 Stage-2 的主对象。

---

### 9.4 Stage-2 验收分层

Stage-2 的验收拆成三层，不能把结构健康、指标收益和语言接地混成一个结论。

**Structure pass**：

```text
D_raw / G / C / ΔH 非塌缩；
D_raw_norm 不长期接近 0；
D_gate 不长期接近 0；
D_i 在空间上有方差；
G_i 在空间上有方差；
C_i 在空间上有方差；
s_i 的 std 不为 0，但不贴近 1 ± β；
训练中后期 ΔH_i / H_X 在 1%~10% 的可控范围；初始化瞬间可以更小；
attention normalized entropy 低于均匀但不坍缩到单一 token；
训练稳定；
A2 不明显低于 A0；
score map 不崩；
不存在 visual-only direct residual bypass。
```

**Performance pass**：

```text
A2 接近或超过 A1；
OP75 / Norm Precision 不低于 A1；
target region 的 evidence scalar 高于 hard negative；
head_readout_delta_score 与 evidence map 方向一致；
hard-negative 局部改善；
Bird1 / HOOT harmful case 不进一步恶化。
```

**Language pass**：

```text
wrong / generic 应改变 D_raw_i 分布以及 D_i / G_i / C_i / s_i；
不能只改变 attention entropy 或最终 AUC 噪声；
A2-normal - wrong/generic 在 language-needed / hard-negative subset 上优于 A1；
normal 相对 wrong/generic 在 evidence representation 或 hard-negative case 上稳定更好；
若 normal / wrong / generic 同步提升，只能解释为 localization-aware evidence calibration。
```

因此 Stage-2 可能出现三种分层结论：

```text
Structure pass:
explicit evidence layer improves localization-aware evidence calibration.

Performance pass:
explicit evidence layer is read by the tracking posterior and is not weaker than A1.

Language pass:
在 structure / performance pass 成立基础上，normal 相对 wrong/generic 有稳定 evidence 差异。
```

若只有 structure pass，没有 language pass，不能写成 language grounding 已解决。

---

### 9.5 工程落点建议

为了最小化风险，Stage-2 实现不建议直接替换 DUTrack 主干。建议：

```text
新增 TrackingEvidenceLayer；
接口沿用 h_x, l_raw, h_z, h_l, l_mask, template_token_len；
保持 freeze backbone + freeze head；
保持原 score_map / size_map / offset_map / loss 不变。
```

不建议在旧 `TrackingEvidenceCalibration` 里继续增加 `mode=evidence_layer`。Stage-1 residual TEC 和 Stage-2 explicit evidence layer 应该工程上分开，便于后续汇报和避免 config 变体消融失控。

必须新增的日志：

```text
stage2_D_raw_norm_mean / std
stage2_D_raw_spatial_std
stage2_D_gate_mean / std
stage2_D_norm_mean / std
stage2_D_spatial_std
stage2_G_norm_mean / std
stage2_G_spatial_std
stage2_C_norm_mean / std
stage2_C_spatial_std
stage2_u_mean / std
stage2_s_mean / std / min / max
stage2_s_deviation_mean
stage2_delta_norm_before_strength
stage2_delta_norm_after_strength
stage2_delta_norm_after_gamma
stage2_delta_to_feature_ratio
stage2_attention_entropy
stage2_attention_entropy_norm
stage2_valid_token_count
stage2_valid_semantic_token_count
stage2_d_norm_eps
stage2_residual_init_scale
```

离线诊断再计算：

```text
e_i = |u_i| 或 ||D_i|| · ||G_i|| · ||C_i||
q_i = |s_i - 1| · ||ΔH_i||
head_readout_delta_score
target-hard_negative evidence gap
```

注意：这些离线 target / hard-negative 统计只用于诊断，不进入训练，不构造 pseudo label。

`M_0` 的 `masked_mean` 必须只使用 valid semantic tokens：

```text
排除 PAD / CLS / SEP；
保留真实语义 token；
记录 valid_semantic_token_count；
当 valid_semantic_token_count 过低时，只能解释为弱类别条件校准。
```

### 9.6 本次代码落地

本次实现选择独立 Stage-2 类，不在旧 TEC 类里增加 mode：

```text
lib/models/dutrack/evidence_layer.py
TrackingEvidenceLayer
```

接入位置：

```text
lib/models/dutrack/dutrack.py
backbone output H_X / H_Z / L_raw
        ↓
TrackingEvidenceLayer
        ↓
原 DUTrack center head
```

配置新增：

```text
MODEL.EVIDENCE_LAYER
```

并新增 5 轮短训配置：

```text
experiments/dutrack/dutrack_384_full_evidence_stage2.yaml
```

当前约束：

```text
MODEL.TEC.ENABLE = false
MODEL.EVIDENCE_LAYER.ENABLE = true
FREEZE_BACKBONE = true
FREEZE_HEAD = true
TRAIN.EPOCH = 5
TEST.EPOCH = 5
```

模型构建时显式禁止：

```text
MODEL.TEC.ENABLE == true
and
MODEL.EVIDENCE_LAYER.ENABLE == true
```

防止 Stage-1 residual TEC 与 Stage-2 evidence layer 同时启用，导致结果不可解释。

训练日志接入：

```text
lib/train/actors/dutrack.py
```

日志命名：

```text
frame_Stage2Evidence/stage2_*
```

实现中的关键初始化：

```text
strength_head.weight = 0
strength_head.bias = 0
=> u_i = 0
=> s_i = 1
```

这保证 Stage-2 初始时不会额外形成空间 gate；真正的校准来自训练后 `C_i -> ΔH_i` 的 evidence-conditioned residual。

同时新增两个受控初始化参数：

```text
D_NORM_EPS = 1e-4
RESIDUAL_INIT_SCALE = 1e-3
```

`D_NORM_EPS` 防止极小 `D_raw` 被 `LN(D_raw)` 放大成噪声方向；`RESIDUAL_INIT_SCALE` 控制 `W_o` 的初始幅度，使 `W_o` 小随机 + `γ` 非零不会破坏 frozen head 的输入分布。

需要重点检查的健康指标：

```text
stage2_D_raw_norm_mean
stage2_D_gate_mean / std
stage2_D_raw_spatial_std
stage2_C_spatial_std
stage2_s_std / min / max / deviation_mean
stage2_delta_to_feature_ratio
stage2_attention_entropy_norm
stage2_valid_semantic_token_count
```

其中：

```text
stage2_delta_to_feature_ratio
```

第 0 步由于 `RESIDUAL_INIT_SCALE=1e-3` 可以明显低于 `1%`；短训中后期建议进入约 `1%~10%` 的量级。若长期接近 0，说明 evidence residual 没有被激活；若过大，则可能破坏 frozen head 的输入分布。

---

## 10. Stage-2R：Phrase-aware Evidence Unit 实现记录

### 10.1 本阶段主线

Stage-2R 不继续把旧 A2 的 token attention 往细节上修。旧 A2 的问题不是少了几个 stop-word rule，而是语言仍然以孤立 raw token 被 search region 读取，容易退化成固定 token residual calibration。

因此本阶段主线改为：

```text
raw language tokens
        ↓
phrase-aware evidence units
        ↓
target-conditioned evidence space
        ↓
region-level evidence residual ΔH_i
        ↓
原 DUTrack tracking head
```

主实验只验证：

```text
phrase-aware evidence unit 是否能替代 raw token shortcut。
```

不把 semantic mask、multi-slot、uniform mixing 作为 Stage-2R 的核心方法。它们只保留为旧 A2 的失败诊断或局部补丁，不进入本阶段主线。

### 10.2 新增模块

新增独立模块：

```text
lib/models/dutrack/evidence_unit_layer.py
TrackingEvidenceUnitLayer
```

输入接口：

```text
H_X: search region tokens
L_raw: BERT raw language embeddings
H_Z: template tokens / memory template tokens
H_L: fused language tokens，可选
l_mask: valid token mask
evidence_anchor_mask: evidence unit anchor mask
template_token_len: 单模板 token 数，用于 center target pooling
```

当前主实验使用：

```text
LANG_SOURCE = raw
TARGET_POOL = center
PHRASE_WINDOW = 3
FREEZE_BACKBONE = true
FREEZE_HEAD = true
```

即 backbone 和 head 均冻结，只训练 `TrackingEvidenceUnitLayer`。

### 10.3 Evidence Unit 构造

本阶段不让 search region 直接 attention 到 raw token，而是先构造 evidence unit：

```text
U_j = PhrasePool(l_{j-r}, ..., l_j, ..., l_{j+r})
```

其中 `j` 必须是 evidence anchor token。anchor token 过滤掉：

```text
a / an / the
and / or / but
of / to / from / for / by / with / without
in / on / at / as
near / behind / beside / under / over
```

这些 token 不作为 evidence center，但仍可作为 phrase window 的上下文参与 `PhrasePool`。例如：

```text
a red car
```

`a` 不会成为 evidence unit，但 `red` 或 `car` 的 evidence unit 可以看到 `a` 作为上下文。

实现位置：

```text
lib/models/dutrack/language_masks.py
build_evidence_anchor_mask
```

关键修正：

```text
anchor mask 不再在全过滤时 fallback 到 valid token。
```

这样可以避免极端情况下 `a` 被重新选成 evidence token。若某个样本没有任何有效 anchor，Stage-2R 的 masked softmax 输出零 attention，该样本的 evidence residual 近似 no-op，而不是用功能词做伪证据。

### 10.4 Evidence Space 与 Residual

Stage-2R 的核心不是改 score，而是在 head 前形成 target-conditioned evidence residual：

```text
E_x_i = LN(W_x H_X_i + W_z z_proto)
E_u_j = LN(W_u U_j + W_z' z_proto)
```

search region 在同一 evidence space 中读取 phrase evidence units：

```text
A_ij = masked_softmax(q(E_x_i)^T k(E_u_j))
M_i  = Σ_j A_ij v(E_u_j)
M_0  = masked_mean_j v(E_u_j)
D_raw_i = M_i - M_0
```

`D_raw_i` 表示该 region 相对全局语言 evidence 均值的目标相关证据偏移。

本次收紧后，代码不再先分别计算 `M_i` 和 `M_0` 后相减，而是先在同一 target-conditioned value space 中中心化：

```text
V_j  = v(E_u_j)
V_0  = masked_mean_j V_j
\tilde V_j = V_j - V_0
D_raw_i = Σ_j A_ij \tilde V_j
```

这样 `M_i` 和 `M_0` 必然共享完全相同的 target conditioning。若 target bias 对所有 evidence units 是常量，它会在 `V_j - V_0` 中被精确抵消，不会混入 `D_raw_i`。

同时，若没有有效 anchor：

```text
A_ij = 0
\tilde V_j = 0
D_raw_i = 0
ΔH_i = 0
```

即严格 no-op，而不是 fallback 到功能词。

为避免极小 `D_raw` 被 LayerNorm 放大，使用：

```text
D_gate = ||D_raw|| / (||D_raw|| + eps)
D_i = D_gate · clamp(||D_raw||) · LN(D_raw)
```

实现上所有可能破坏 zero path 的 LayerNorm 都改成 safe LayerNorm：

```text
SafeLN(x) = gate(x) · LN(x)
gate(x) = rms(x) / (rms(x) + eps)
```

因此：

```text
x = 0  => SafeLN(x) = 0
x 很小 => 不会被 LN 放大成 O(1) 噪声
```

然后构造 region evidence：

```text
G_i = MLP([D_i, E_x_i ⊙ D_i]) + D_i
C_i = MLP(G_i)
```

最终以 residual 形式进入 tracking head：

```text
r_i = β · q_e · centered_tanh(W_r C_i)
dir_i = SafeNormalize(W_o C_i)
ΔH_i = tanh(γ) · r_i · dir_i
H'_X_i = H_X_i + ΔH_i
```

这里不再把 `s_i = 1 + ...` 作为 residual 幅度。原因是 `W_o C_i` 可能绕过 `s_i` 自行放大，导致 `s_i` 只剩日志意义。

新的 `r_i` 是 signed region-relative residual coefficient：

```text
r_i > 0：沿 evidence direction 校准；
r_i < 0：沿相反方向抑制 / 去证据；
r_i = 0：严格 no-op。
```

允许 `r_i` 为负值是有意设计。它不是分类 score 的正负先验，而是 feature residual 的有符号系数，用来表达 target region 与 hard-negative region 的相反校准方向。

`dir_i` 使用 safe normalize：

```text
dir_i = W_o C_i / max(||W_o C_i||, eps)
```

不会对极小向量强行归一化成随机单位方向。

`q_e` 不是 learnable gate，而是非参数、连续的 evidence availability：

```text
context_quality_j = clamp(context_count_j / (phrase_window - 1), 0, 1)
w_j = anchor_j · (0.25 + 0.75 · context_quality_j)
q_e = clamp(Σ_j w_j / MIN_EVIDENCE_UNITS, 0, 1)
```

这里故意删除了 learnable `q_L`，避免它学成新的全局 gate shortcut。当前 `q_e` 只由 evidence unit 数量和上下文覆盖度决定，不由网络学习。

含义：

```text
无 anchor: q_e = 0
单个无上下文 anchor: q_e 很低，但不再强制 no-op
带上下文 phrase evidence: q_e 更高
多个 phrase evidence: q_e 接近 1
```

这样可以表达“有一点但不可靠”的语言证据，避免二值开关导致训练样本不足；同时单词类别标签仍被低权重约束，不能和完整 phrase evidence 等价。

需要注意：这版约束明显强于旧 A2。训练初期 `ΔH_i` 会非常小，AUC 可能低于旧 A2，甚至短训低于 A0。这不是 bug，而是本阶段为了验证 language evidence path 是否真实有效而主动牺牲了一部分 shortcut 能力。

### 10.4.1 Phrase unit 进一步收紧

旧实现的 `PhrasePool(window tokens)` 仍可能退化成平滑后的 token shortcut。当前改为：

```text
context_j = mean(valid neighbors around anchor_j)
anchor_perp_j = l_j - proj_{context_j}(l_j)
U_j = MLP([
    anchor_perp_j,
    context_j,
    anchor_perp_j ⊙ context_j
])
```

也就是说，不直接提供：

```text
U_j = MLP(l_j)
```

这并不是禁止 anchor token，而是禁止 `U_j ≈ l_j` 成为唯一稳定解。当前实现把 anchor 分解为：

```text
l_j = proj_context(l_j) + anchor_perp_j
```

只把 `anchor_perp_j` 给 phrase unit，同时保留 `context_j` 用于关系词和属性词。这样 anchor 本身仍作为必要语义进入，但它不是裸 token shortcut；context 也不会被完全丢掉。

### 10.5 接入位置

模型接入：

```text
lib/models/dutrack/dutrack.py
```

当前互斥关系：

```text
MODEL.TEC
MODEL.EVIDENCE_LAYER
MODEL.EVIDENCE_UNIT_LAYER
```

三者只能启用一个，避免 Stage-1 TEC、旧 A2、Stage-2R 同时生效导致归因混乱。

语言 mask 由 backbone 侧输出：

```text
lib/models/dutrack/itpn.py

l_mask
semantic_l_mask
evidence_anchor_mask
```

Stage-2R 只使用：

```text
l_mask
evidence_anchor_mask
```

`semantic_l_mask` 仍服务旧 A2 诊断，不作为 Stage-2R 主线。

### 10.6 配置文件

新增配置：

```text
experiments/dutrack/dutrack_384_full_evidence_stage2r_phraseunit.yaml
experiments/dutrack/dutrack_384_full_evidence_stage2r_phraseunit_wrong.yaml
experiments/dutrack/dutrack_384_full_evidence_stage2r_phraseunit_generic.yaml
```

主训练配置：

```text
TRAIN.EPOCH = 5
DATA.TRAIN.SAMPLE_PER_EPOCH = 16000
TRAIN.BATCH_SIZE = 8
TRAIN.KEEP_LAST_CHECKPOINTS = 1
TRAIN.KEEP_CHECKPOINT_EPOCHS = [3, 5]
MODEL.EVIDENCE_UNIT_LAYER.ENABLE = true
MODEL.EVIDENCE_UNIT_LAYER.FREEZE_BACKBONE = true
MODEL.EVIDENCE_UNIT_LAYER.FREEZE_HEAD = true
```

wrong/generic 配置只改变：

```text
TEST.LANG_MODE
TEST.CHECKPOINT_NAME = dutrack_384_full_evidence_stage2r_phraseunit
```

即 wrong/generic 使用同一个 Stage-2R normal 训练得到的 checkpoint，避免重新训练导致变量混入。

### 10.7 训练日志

训练 actor 新增日志前缀：

```text
frame_Stage2REvidenceUnit/stage2r_*
```

重点观察：

```text
stage2r_D_raw_norm_mean / std / spatial_std
stage2r_D_gate_mean / std
stage2r_G_norm_mean / spatial_std
stage2r_C_norm_mean / spatial_std
stage2r_s_mean / std / min / max / deviation_mean
stage2r_r_mean / std / min / max / abs_mean
stage2r_evidence_availability_mean / std / min / max
stage2r_language_reliability_mean / std / min / max
stage2r_delta_direction_raw_norm_mean / std
stage2r_delta_direction_gate_mean / std
stage2r_delta_direction_norm_mean / max
stage2r_phrase_context_count
stage2r_phrase_has_context_ratio
stage2r_raw_anchor_token_count
stage2r_weighted_evidence_count
stage2r_anchor_context_quality_mean
stage2r_no_raw_anchor_ratio
stage2r_single_raw_anchor_ratio
stage2r_no_effective_evidence_ratio
stage2r_single_effective_evidence_ratio
stage2r_low_availability_ratio
stage2r_delta_to_feature_ratio
stage2r_attention_entropy_norm
stage2r_top1_evidence_weight
stage2r_valid_token_count
stage2r_anchor_token_count
stage2r_low_evidence_unit_ratio
```

结构健康判断：

```text
D_raw / G / C / ΔH 不塌缩；
delta_to_feature_ratio 不长期接近 0，也不明显过大；
r_i 有 region-relative 方差，但不贴边；
evidence_availability 不能长期接近 0，否则说明 anchor 过滤过严或语言本身缺少 phrase evidence；
low_availability_ratio 不能长期过高，否则说明训练主要依赖弱 unigram evidence；
delta_direction_gate 不长期接近 0；
delta_direction_norm_max 不应在极小 raw_norm 时接近 1；
attention entropy 低于均匀但不坍缩到单一 evidence unit；
normal/wrong/generic 改变 evidence representation，而不是只改变 top attention token。
```

### 10.10 本轮 shortcut 风险处理

#### 风险 1：`q_L` 成为新的全局 gate shortcut

处理：

```text
删除 learnable language_reliability_head；
删除 q_L = sigmoid(MLP(...))；
改为非参数连续 evidence availability。
```

`q_e` 不再由网络学习，不能学成全局 gate；但它也不是二值开关，可以区分弱 unigram evidence 和强 phrase evidence。

#### 风险 2：phrase unit 仍退化成类别 anchor shortcut

处理：

```text
保留 context_j，避免丢掉关系词；
不直接提供 l_j；
改为 anchor_perp_j = l_j - proj_context(l_j)；
使用 [anchor_perp_j, context_j, anchor_perp_j ⊙ context_j]；
单 anchor 只给低 q_e，不再强制 no-op。
```

这样类别词仍能作为必要 anchor 信息参与训练，但单独类别词只有低 availability，不能和完整 phrase evidence 同等影响 residual。

#### 风险 3：`C_i = MLP(G_i)` 的 bias 破坏 zero path

当前 zero path 约束：

```text
interaction_mlp: Linear bias=False
evidence_mlp: Linear bias=False
out_linear: Linear bias=False
SafeLN(x) = gate(x) · LN(x)
```

即使 LayerNorm 自带 affine bias，`x=0` 时 `gate=0`，输出仍为 0。`strength_head` 虽有 bias，但后续做 region mean-centering，常量 bias 被抵消。

#### 风险 4：anchor 过滤过严，导致“看似干净，实际无证据”

处理不是放松规则，而是把 raw/effective evidence 分开记录：

```text
raw_anchor_token_count
anchor_token_count / evidence_unit_count
no_raw_anchor_ratio
single_raw_anchor_ratio
no_effective_evidence_ratio
single_effective_evidence_ratio
phrase_has_context_ratio
weighted_evidence_count
anchor_context_quality_mean
low_availability_ratio
```

如果 `no_effective_evidence_ratio` 长期很高，这一版不能解释为“语言证据干净”，只能解释为“当前语言不足以构造 phrase evidence”。如果 `single_effective_evidence_ratio` 高且 `low_availability_ratio` 高，则说明模型主要依赖弱 unigram evidence，这时也不能声称形成了稳定 phrase-level grounding。

#### 风险 6：SafeNormalize 把极小 residual direction 拉成单位向量

旧 hard clamp：

```text
dir = x / max(||x||, eps)
```

在 `||x||` 接近 `eps` 附近会有硬切换。当前改为 soft normalize：

```text
dir = x / (||x|| + eps)
```

若 `x` 极小，则：

```text
||dir|| ≈ ||x|| / eps << 1
```

不会被拉成单位向量。新增：

```text
stage2r_delta_direction_norm_mean
stage2r_delta_direction_norm_max
```

用于检查极小 `W_o C_i` 是否被异常放大。

#### 风险 7：wrong/generic 更容易无 evidence

wrong/generic 可能因为语句模板不同导致 `q_e` 更低，造成“normal 更好”只是因为 normal 有 residual、wrong/generic no-op。后续语言敏感性解释必须同时报告：

```text
no_effective_evidence_ratio
single_effective_evidence_ratio
weighted_evidence_count
low_availability_ratio
delta_to_feature_ratio
```

若 normal/wrong/generic 的 availability 分布不一致，性能差异不能直接解释为语义 grounding。

#### 风险 5：只保留最新 checkpoint 不适合本轮判断

新增配置：

```text
TRAIN.KEEP_CHECKPOINT_EPOCHS = [3, 5]
TRAIN.KEEP_LAST_CHECKPOINTS = 1
```

训练中始终保留当前最新 checkpoint，最终保留 ep3 和 ep5。这样既能节省空间，也能对比中期/末期是否从 no-op 逐步启动，避免只看 ep5。

### 10.8 测试诊断

测试端新增 Stage-2R 诊断：

```text
stage2r_evidence_*_gap
stage2r_calibration_*_gap
stage2r_strength_*_gap
stage2r_attn_target_top_tokens
stage2r_attn_hard_negative_top_tokens
```

注意：这里的 top token 字段实际显示的是 phrase evidence unit，例如：

```text
a red car
red car on
```

而不是旧 A2 的孤立 raw token。

实现位置：

```text
lib/test/tracker/dutrack.py
```

diagnostics 开启时会自动打开：

```text
network.evidence_unit_layer.enable_diagnostics = True
```

### 10.9 当前阶段验收标准

Stage-2R 不以单次 AUC 最高作为唯一目标，验收分三层：

```text
Structure pass:
phrase-aware evidence units 被构造；
region evidence C_i 有空间差异；
ΔH_i 被 head 稳定读到；
score map 不崩。

Performance pass:
A2R normal 接近或超过旧 A2/A1；
OP75 / Norm Precision 不明显下降；
hard-negative gap 局部改善。

Language pass:
normal 相对 wrong/generic 在 evidence representation 或 hard-negative case 上稳定更好；
top evidence units 不再坍缩到固定功能词；
normal/wrong/generic 对 D_raw / C_i 分布产生可解释差异。
```

若只有 Structure pass，不能声称语言 grounding 已经解决；只能说 phrase-aware evidence residual 在 tracking posterior 中可用。
