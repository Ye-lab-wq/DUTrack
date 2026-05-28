# Stage 3-C1 Learnable Multi-Query Prior 实现记录

更新时间：2026-05-24

## 1. 实现目标

本次实现 Stage 3-C1：

```text
language tokens -> K learnable language queries -> search prior -> center-score bias
```

边界：

```text
1. 不改 backbone attention；
2. 不使用 TE keep/policy；
3. 不影响 size branch / offset branch；
4. prior 只作为 bounded additive bias 接入 center score；
5. tracking loss 保持主导；
6. aux score-rank loss 只作为小权重退火辅助。
```

配置名：

```text
dutrack_384_full_lmq_k4_e10
```

其中：

```text
lmq = learnable multi-query
k4 = 4 个 language queries
e10 = 10 轮短训
```

---

## 2. 代码改动

### 2.1 新增 learnable multi-query prior 模块

文件：

```text
lib/models/dutrack/language_multi_query_prior.py
```

核心结构：

```text
lang_tokens -> lang_proj
search_tokens -> visual_proj
K learnable query seeds -> attend language tokens -> Q_lang
Q_lang 与 search tokens 做 cosine compatibility
K 张 query prior map -> learned fusion -> prior score
```

输出：

```text
prior_scores: B x N_search
query_prior_maps: B x K x N_search
query_fusion_weights: B x K
query_prior_cosine_mean / max
```

注意：

```text
prior_scores 不 detach，用于 tracking loss 反传；
query maps 和统计项只用于诊断，进入 aux_dict 时 detach。
```

### 2.2 iTPN 中加入 LMQ prior 生成

文件：

```text
lib/models/dutrack/itpn.py
```

新增配置读取：

```text
MODEL.TE.LMQ_ENABLE
MODEL.TE.LMQ_LOC
MODEL.TE.LMQ_NUM_QUERIES
MODEL.TE.LMQ_HIDDEN_DIM
MODEL.TE.LMQ_DROPOUT
```

新增模块：

```text
self.language_query_priors = nn.ModuleList(...)
```

在 `_fusion_feat` 的指定 block index 处读取：

```text
l_tokens = language tokens
x_tokens = search tokens
```

生成：

```text
te_aux["lmq_prior_scores"]
te_aux["lmq_query_prior_maps"]
te_aux["lmq_query_fusion_weights"]
te_aux["lmq_query_prior_cosine_mean"]
te_aux["lmq_query_prior_cosine_max"]
```

本阶段配置里：

```text
MODEL.TE.ENABLE: false
MODEL.TE.LMQ_ENABLE: true
MODEL.TE.LMQ_LOC: [15]
```

因此 LMQ 不会触发旧 TE keep，也不会修改 attention。

#### 为什么先取 L15 的 l_tokens / x_tokens

这里的 L15 指的是融合主干中的第 15 个 main block 之后、进入该 block 前可读取到的当前融合 token 状态。使用该层的目的不是复用 TE 的 7/15/23 规则，而是做一个中后层的最小变量验证：

```text
1. l_tokens 已经过若干层视觉-语言融合，比原始 BERT token 更接近 DUTrack 的融合空间；
2. x_tokens 已经包含模板、语言、历史 query 影响后的 search 表征，比浅层 patch 更接近 center head 输入；
3. L15 仍早于最终 score head，有机会让 prior 对最终 center score 产生增量；
4. 只选一层可以先验证正向信号，避免层数选择成为额外变量。
```

所以本阶段的含义是：

```text
在中后层融合表示上，用语言 query 读取 search tokens，生成 score prior。
```

如果 L15 的 `lmq_prior_gap` 有正信号但 score 没有改善，下一步优先考虑 score adapter 或改 prior/score 对齐；如果 L15 没有正信号，再比较 L7 / L23 或多层 prior。

### 2.3 DUTrack score prior source 支持 LMQ

文件：

```text
lib/models/dutrack/dutrack.py
```

新增：

```text
SCORE_PRIOR_SOURCE: lmq_prior
```

读取：

```text
aux_dict["lmq_prior_scores"][stage_idx]
```

处理方式与 word-direct margin 一致：

```text
bias = raw_score
bias = bias - mean(bias)
bias = tanh(bias / tau)
bias = beta * bias
bias = clamp(bias, -bias_clamp, bias_clamp)
score_logits_final = score_logits_base + bias
```

关键点：

```text
lmq_prior_scores 保持梯度；
tracking loss 可以反传到 LMQ module；
detach 只在 aux score-rank loss 内部用于 S_base。
```

### 2.4 训练日志增加 LMQ 诊断项

文件：

```text
lib/train/actors/dutrack.py
```

新增训练状态：

```text
ScorePrior/lmq_query_prior_gap_mean
ScorePrior/lmq_query_prior_gap_max
ScorePrior/lmq_query_prior_gap_min
ScorePrior/lmq_query_prior_gap_q0
ScorePrior/lmq_query_prior_gap_q1
ScorePrior/lmq_query_prior_gap_q2
ScorePrior/lmq_query_prior_gap_q3
ScorePrior/lmq_query_prior_cosine_mean
ScorePrior/lmq_query_prior_cosine_max
ScorePrior/lmq_query_fusion_entropy
ScorePrior/lmq_query_fusion_max
```

用途：

```text
1. 判断 query prior 是否能区分 pos 与 hard negative；
2. 判断 K 个 query 是否塌缩；
3. 判断 fusion 是否退化成单 query。
```

如果：

```text
lmq_query_prior_cosine_mean / max 长期接近 1
```

说明 K 个 query prior map 没有形成互补分工。2026-05-24 的短训日志中已经观察到：

```text
lmq_query_prior_cosine_mean = 1.00000
lmq_query_fusion_entropy = 1.38629 ~= ln(4)
lmq_query_fusion_max ~= 0.25
lmq_query_prior_gap_q0/q1/q2/q3 几乎完全一致
```

进一步检查 checkpoint：

```text
query_seed 本身并未相同；
4 个 seed 的 pairwise cosine 约为 -0.05 到 0.04。
```

因此当前问题不是 seed 初始化重复，而是后续链路塌缩：

```text
query_seed -> language attention pooling -> query vector -> search prior map
```

某一步把不同 seed 变成了几乎相同的 query prior。

### 2.5 LMQ 塌缩诊断字段

为定位塌缩位置，新增以下只读诊断字段，不改变训练前向行为：

```text
lmq_query_seed_cosine_mean / max
lmq_query_lang_attn_cosine_mean / max
lmq_query_lang_attn_entropy
lmq_query_lang_attn_max
lmq_query_vector_cosine_mean / max
lmq_query_map_between_std
lmq_prior_score_std
```

判断方式：

```text
1. seed cosine 低，但 lang_attn cosine 高：
   不同 seed 在语言 token 上读到了相同内容，语言读取阶段塌缩。

2. lang_attn cosine 不高，但 query_vector cosine 高：
   lang_value / projection 后把不同词组合压成近似同一向量。

3. query_vector cosine 不高，但 prior map cosine 高：
   search matching 阶段分辨率不足，视觉 token 对不同 query 的响应相同。

4. lmq_prior_score_std 很小：
   raw prior 自身空间对比度不足，即使接入 score head 也难以产生有效中心偏置。
```

注意：

```text
当前 LMQ 不使用简单 mean language pooling；
它是 K 个 learnable query seed 对 language tokens 做 attention pooling。
但如果 query attention 过于相似，效果上仍会退化成近似单一语言摘要。
```

---

## 7. C1-R seed residual 验证版本

更新时间：2026-05-24

### 7.1 修改目标

上一轮诊断定位到：

```text
query_seed 本身不同；
query_lang_attn 几乎完全相同；
query_vector / query_prior 随后完全塌缩。
```

因此 C1-R 只处理一个问题：

```text
在 language attention pooling 后保留 query identity。
```

### 7.2 实现方式

文件：

```text
lib/models/dutrack/language_multi_query_prior.py
```

当前 C1：

```text
Q = Attn(E, L)
```

C1-R：

```text
Q_pool = Attn(E, L)
Q = LN(Q_pool + gamma * E)
```

其中：

```text
E: K 个 learnable query seed
gamma: LMQ_SEED_RESIDUAL_GAMMA，默认 0.1
```

配置新增：

```text
MODEL.TE.LMQ_SEED_RESIDUAL
MODEL.TE.LMQ_SEED_RESIDUAL_GAMMA
```

新增配置文件：

```text
experiments/dutrack/dutrack_384_full_lmq_r01_e10.yaml
experiments/dutrack/dutrack_384_full_lmq_r02_e10.yaml
```

其中：

```text
r01: gamma = 0.1，最保守验证；
r02: gamma = 0.2，模块级扫参后建议优先训练的版本。
```

### 7.3 额外诊断

为了区分 pooling 前后，新增：

```text
lmq_pooled_query_cosine_mean / max
```

判断逻辑：

```text
如果 pooled_query_cosine 仍接近 1，
但 query_vector_cosine 下降，
说明 seed residual 起到了保留 query identity 的作用。

如果 query_vector_cosine 下降，
但 query_prior_cosine 仍接近 1，
说明后续 search matching 仍然塌缩，需要进入 light query-search decoder。
```

### 7.4 验证标准

第一阶段只看结构是否解除天然塌缩，不先看最终性能：

```text
query_vector_cosine_mean 明显低于原 C1 的 0.99999；
query_map_between_std 明显大于 1e-5；
query_prior_cosine_mean 不再长期等于 1；
lmq_query_prior_gap_q0/q1/q2/q3 出现差异。
```

训练策略仍保持：

```text
只训练 backbone.language_query_priors；
不开放 center head / score adapter；
不改 beta；
不改 size/offset；
不改 backbone attention。
```

---

## 8. C1-K1 单 query 验证版本

更新时间：2026-05-24

### 8.1 修改目标

`dutrack_384_full_lmq_r02_e10` 第 1 个 epoch 已经显示：

```text
query seed 本身不同；
language attention / pooled query / final query / prior map 全部 cosine 接近 1；
4 个 query 的 prior gap 几乎相同。
```

因此 K=1 不是为了解决 multi-query 分化，而是先回答一个更基础的问题：

```text
单个 learnable language query prior 本身有没有可用正向信号？
```

如果 K=1 的 prior gap 仍长期为负，说明问题不只是 query collapse，而是当前 LMQ prior 信息源或 search matching 本身不可靠。

如果 K=1 的 prior gap 能稳定转正，再回头处理 K>1 的 query 分化才有意义。

### 8.2 配置

新增配置：

```text
experiments/dutrack/dutrack_384_full_lmq_k1_e10.yaml
```

核心差异：

```yaml
MODEL:
  TE:
    LMQ_ENABLE: true
    LMQ_LOC: [15]
    LMQ_NUM_QUERIES: 1
    LMQ_SEED_RESIDUAL: false
    LMQ_SEED_RESIDUAL_GAMMA: 0.0
    SCORE_PRIOR_SOURCE: lmq_prior
    SCORE_PRIOR_BETA: 0.1
    SCORE_PRIOR_LAYER: 15
TRAIN:
  TRAIN_TE_ONLY: true
  TRAIN_TE_ONLY_PATTERNS:
  - backbone.language_query_priors
```

### 8.3 诊断注意

K=1 时以下 pairwise 指标会自然为 0，不再用于判断塌缩：

```text
lmq_query_prior_cosine_mean / max
lmq_query_seed_cosine_mean / max
lmq_query_lang_attn_cosine_mean / max
lmq_pooled_query_cosine_mean / max
lmq_query_vector_cosine_mean / max
lmq_query_map_between_std
```

应重点看：

```text
lmq_query_prior_gap_mean
prior_pos_gain
prior_hard_neg_gain
prior_to_score_abs_ratio
score_rank_loss
active_corrective_ratio
score_onoff_peak_delta
score_map_mass_in_gt
```

判断：

```text
prior gap 为正，且 score_on/off 指标改善:
  单 query prior 有效，后续再做 K>1 分化。

prior gap 为正，但 score_on/off 不改善:
  prior 信息存在，但接入 center score 的尺度或 head 适配不足。

prior gap 长期为负:
  当前 LMQ 信息源本身不可靠，继续做多 query 没意义。
```

### 8.4 指令

训练：

```bash
python tracking/train.py --script dutrack --config dutrack_384_full_lmq_k1_e10 --save_dir ./output --mode single --use_wandb 0
```

如果 shell 没激活环境：

```bash
conda run -n DUTrack python tracking/train.py --script dutrack --config dutrack_384_full_lmq_k1_e10 --save_dir ./output --mode single --use_wandb 0
```

第 1 个 epoch 后优先看日志：

```bash
rg -n "lmq_query_prior_gap_mean|prior_pos_gain|prior_hard_neg_gain|prior_to_score_abs_ratio|Loss/total|IoU" output/logs/dutrack-dutrack_384_full_lmq_k1_e10.log
```

可视化/统计：

```bash
python tracking/visualte_diagnostic_suite.py --config dutrack_384_full_lmq_k1_e10 --runid 10 --max_frames 5 --stat_frames 0 --vis_frames 5 --top_ratio 0.1 --hardneg_topk 6 --language_init_source dataset_or_class --language_update_mode anchor --case otb_lang:Biker --case hoot_balanced20:0 --out_dir output/test/visualte_diagnostic_suite --output_tag lmq_k1_fullstats
```

### 7.5 模块级验证

随机 token 下的 off / residual 对比：

```text
off:
  query_lang_attn_cosine ~= 0.99996
  query_vector_cosine ~= 0.99997
  query_prior_cosine ~= 0.99997
  query_map_between_std ~= 0.00028

gamma=0.1:
  query_vector_cosine ~= 0.99976
  query_prior_cosine ~= 0.99977
  query_map_between_std ~= 0.00076

gamma=0.2:
  query_vector_cosine ~= 0.99923
  query_prior_cosine ~= 0.99927
  query_map_between_std ~= 0.00138
```

结论：

```text
seed residual 方向有效，但 0.1 偏保守；
优先跑 r02，确认真实数据和训练后是否继续分化。
```

说明 multi-query 没有真实分化。

另外新增 score/prior 尺度诊断：

```text
ScorePrior/score_rank_loss
ScorePrior/prior_bias_mean
ScorePrior/prior_bias_abs_mean
ScorePrior/prior_bias_abs_max
ScorePrior/prior_bias_max
ScorePrior/prior_clamp_ratio
ScorePrior/score_logits_base_mean
ScorePrior/score_logits_base_abs_mean
ScorePrior/score_logits_base_abs_max
ScorePrior/prior_to_score_abs_ratio
```

用途：

```text
1. 判断 tanh + clamp 是否让 prior 太保守；
2. 判断 prior bias 相比 base score logits 是否量级过小；
3. 判断 prior 是否频繁触发 clamp，若 clamp ratio 高则说明 beta 或 tau 过强；
4. 判断 aux loss 是否真的还有 active 样本。
```

### 2.5 可视化诊断脚本兼容 LMQ

文件：

```text
tracking/visualte_diagnostic.py
```

新增：

```text
lmq_locs
lmq_prior_L{layer}
lmq_query_prior_q{k}_L{layer}
lmq_query_cosine_mean_L{layer}
lmq_query_cosine_max_L{layer}
lmq_query_fusion_L{layer}
score_prior_bias
score_logits_base
score_prior_to_base_abs_ratio
score_prior_bias_clamp_ratio
```

同时 attention hook 会包含 LMQ 层，便于后续可视化分析。

`visualte_diagnostic_suite.py` 也补充了 LMQ 汇总字段：

```text
lmq_prior_gap_max
lmq_prior_hardneg_gap_max
lmq_query_prior_gap_max
lmq_query_cosine_mean_max
lmq_query_cosine_max_max
score_prior_bias_abs_mean
score_prior_bias_max
score_prior_bias_clamp_ratio
score_logits_base_abs_mean
score_prior_to_base_abs_ratio
```

### 2.6 新增训练配置

文件：

```text
experiments/dutrack/dutrack_384_full_lmq_k4_e10.yaml
```

关键配置：

```yaml
MODEL:
  TE:
    ENABLE: false
    LMQ_ENABLE: true
    LMQ_LOC: [15]
    LMQ_NUM_QUERIES: 4
    SCORE_PRIOR_ENABLE: true
    SCORE_PRIOR_SOURCE: lmq_prior
    SCORE_PRIOR_LAYER: 15
    SCORE_PRIOR_BETA: 0.1
    SCORE_PRIOR_BIAS_CLAMP: 0.35
    AUX_SCORE_LOSS_WEIGHT: 0.003
    AUX_SCORE_LOSS_ANNEAL: cosine
    AUX_SCORE_LOSS_WEIGHT_END: 0.0

TRAIN:
  EPOCH: 10
  TRAIN_TE_ONLY: true
  TRAIN_TE_ONLY_PATTERNS:
  - backbone.language_query_priors
```

训练参数只开放：

```text
backbone.language_query_priors
```

冻结：

```text
backbone 主体
language encoder
center head
size branch
offset branch
```

---

## 3. 训练策略

### 3.1 当前 C1 只训练 query prior module

第一轮不开放 center head adapter。

原因：

```text
1. 先判断 language prior 自身是否有正向信号；
2. 避免 head 适配掩盖 prior 是否有效；
3. 降低短训不稳定性。
```

### 3.2 Loss 设计

主损失：

```text
tracking loss = location focal + L1 + GIoU
```

辅助损失：

```text
score-space aux rank loss
```

权重：

```text
AUX_SCORE_LOSS_WEIGHT: 0.003
AUX_SCORE_LOSS_WEIGHT_END: 0.0
AUX_SCORE_LOSS_ANNEAL: cosine
```

注意：

```text
S_base.detach() 只在 aux loss 内部使用；
主 tracking path 不 detach。
```

### 3.3 重点看哪些结果

训练日志：

```text
ScorePrior/prior_pos_gain
ScorePrior/prior_hard_neg_gain
ScorePrior/prior_to_score_abs_ratio
ScorePrior/prior_clamp_ratio
ScorePrior/score_logits_base_abs_mean
ScorePrior/score_pos_mean
ScorePrior/score_hard_neg_mean
ScorePrior/score_rank_loss
ScorePrior/active_corrective_ratio
ScorePrior/active_prior_gain_ratio
ScorePrior/lmq_query_prior_gap_mean
ScorePrior/lmq_query_prior_gap_q0/q1/q2/q3
ScorePrior/lmq_query_prior_cosine_mean
ScorePrior/lmq_query_prior_cosine_max
```

判断逻辑：

```text
active_corrective_ratio 很低:
  大部分样本中 base score 已经把 pos/hardneg 拉开，aux 提供的梯度弱。

active_corrective_ratio 很高但 tracking loss 不改善:
  prior 可能在学习一个和最终 tracking head 不完全一致的辅助目标。

prior_to_score_abs_ratio 很低:
  prior 量级太小，即使方向正确也难以改变 center peak。

prior_clamp_ratio 很高:
  tanh/clamp 过强或 beta/tau 设置不合适，prior 可能被截断。

lmq_query_prior_gap_q{k} 差异很小且 cosine 高:
  K 个 query 可能塌缩成同一张 prior map。
```

可视化/诊断：

```text
lmq_prior_L15_pos_hardneg_gap
score_onoff_peak_delta
score_map_mass_in_gt
score_map_top10_precision
mean_iou
```

---

## 4. 训练与诊断指令

训练：

```bash
python tracking/train.py --script dutrack --config dutrack_384_full_lmq_k4_e10 --save_dir ./output --mode single --use_wandb 0
```

如果当前 shell 没有进入 DUTrack conda 环境：

```bash
conda run -n DUTrack python tracking/train.py --script dutrack --config dutrack_384_full_lmq_k4_e10 --save_dir ./output --mode single --use_wandb 0
```

诊断套件：

```bash
python tracking/visualte_diagnostic_suite.py --config dutrack_384_full_lmq_k4_e10 --runid 10 --max_frames 5 --stat_frames 0 --vis_frames 5 --top_ratio 0.1 --hardneg_topk 6 --language_init_source dataset_or_class --language_update_mode anchor --case otb_lang:Biker --case hoot_balanced20:0 --out_dir output/test/visualte_diagnostic_suite --output_tag lmq_k4_e10_fullstats
```

---

## 5. 当前风险

### 5.1 Query collapse

如果：

```text
lmq_query_prior_cosine_mean > 0.9
```

说明多个 query prior map 高度相似，K=4 没有真正分化。

第一轮只监控，不加 diversity loss。因为强行 diversity 可能把 query 推向背景。

### 5.2 Prior 与 center score 单位不匹配

如果：

```text
lmq_prior_L15_pos_hardneg_gap 为正
但 score_onoff_peak_delta / score_map_mass_in_gt 不提升
```

说明 prior 有信息，但和 center score 的尺度/分布没有对齐。

下一步应考虑：

```text
small center score adapter
```

而不是回到手工筛词规则。

### 5.3 学习信号仍可能偏弱

C1 只训练 prior module，head 冻结。它适合验证语言 prior 是否有正信号，但不一定能直接提升 IoU。

若 C1 有 prior 信号但 IoU 不变，进入 C2：

```text
train query prior module + small center score adapter
```

---

## 9. C1-D light query-search decoder

更新时间：2026-05-24

### 9.1 修改动机

`C1` / `C1-R` / `C1-K1` 的主要问题不是单纯 seed 不同，而是：

```text
language query 读完语言后，和 search token 的交互仍然太浅；
dot-product matching 容易产生低对比度或 query 间高度相似的 prior。
```

因此 C1-D 不再只做：

```text
query -> attend language -> cosine(search)
```

而是在 language-conditioned query 之后加入一层很轻的 query-search decoder：

```text
query self-attention
-> query cross-attention over search tokens
-> query FFN
-> query-search pair MLP
-> K 个 query prior maps
-> learned fusion
-> center score additive bias
```

它借鉴 ReferFormer 的关键思想：

```text
query identity 需要在 decoder 内持续保留；
query 需要和 visual/search memory 做真正 cross-attention；
最终仍由 tracking loss 驱动，而不是人工 gap router。
```

但为了控制训练压力，本版本仍然保持：

```text
不改 backbone attention；
不接入 size / offset branch；
不开放 head；
只训练 backbone.language_query_priors；
aux score-rank loss 继续小权重退火。
```

### 9.2 代码改动

修改文件：

```text
lib/models/dutrack/language_multi_query_prior.py
lib/models/dutrack/itpn.py
lib/config/dutrack/config.py
lib/train/actors/dutrack.py
tracking/visualte_diagnostic.py
tracking/visualte_diagnostic_suite.py
experiments/dutrack/dutrack_384_full_lmq_d1_e10.yaml
```

新增配置：

```yaml
MODEL:
  TE:
    LMQ_DECODER_ENABLE: true
    LMQ_DECODER_NUM_HEADS: 8
    LMQ_DECODER_DROPOUT: 0.1
    LMQ_DECODER_FFN_RATIO: 2.0
```

模块内新增诊断：

```text
lmq_query_search_attn_entropy
lmq_query_search_attn_max
lmq_decoder_query_delta_norm
```

用途：

```text
query_search_attn_entropy:
  query cross-attention 是否仍然近似均匀。

query_search_attn_max:
  query 是否在 search tokens 上形成局部关注。

decoder_query_delta_norm:
  decoder 是否实际改变了 language-conditioned query。
```

### 9.3 和 C1-R 的区别

`C1-R` 只是在语言 pooling 后加：

```text
Q = LN(Q_pool + gamma * seed)
```

目的是保留 query identity。

`C1-D` 进一步让 query 读 search tokens：

```text
Q = Decoder(Q, X_search)
prior_k(j) = MLP([Q_k, X_j, Q_k * X_j])
```

因此它检查的是另一个问题：

```text
query identity 保留下来以后，能否通过 search cross-attention 形成更有判别性的 spatial prior。
```

### 9.4 预期观察

第一轮不要先看最终 IoU，应先看：

```text
lmq_query_prior_gap_mean / q{k}
lmq_query_prior_cosine_mean
lmq_query_map_between_std
lmq_query_search_attn_entropy
lmq_query_search_attn_max
lmq_decoder_query_delta_norm
prior_to_score_abs_ratio
score_onoff_peak_delta
```

判断逻辑：

```text
query_search_attn_entropy 下降，query_search_attn_max 上升：
  decoder 开始让 query 在 search tokens 上形成局部读取。

decoder_query_delta_norm 明显大于 0：
  cross-attention/FFN 确实改变了 query 表示。

query_prior_cosine 仍接近 1：
  多 query 仍未分化，后续再考虑 K=1 decoder 或弱 diversity 诊断。

query_prior_gap 仍长期为负：
  说明 light decoder 没解决语言 prior 的信息源问题，不应继续放大 beta。
```

### 9.5 训练指令

```bash
python tracking/train.py --script dutrack --config dutrack_384_full_lmq_d1_e10 --save_dir ./output --mode single --use_wandb 0
```

如果当前 shell 没激活 DUTrack 环境：

```bash
conda run -n DUTrack python tracking/train.py --script dutrack --config dutrack_384_full_lmq_d1_e10 --save_dir ./output --mode single --use_wandb 0
```

建议第一轮先看第 1 个 epoch 的日志，不要等满 10 轮才判断：

```bash
rg -n "lmq_query_prior_gap|lmq_query_search_attn|lmq_decoder_query_delta|lmq_query_prior_cosine|prior_to_score_abs_ratio|Loss/total|IoU" output/logs/dutrack-dutrack_384_full_lmq_d1_e10.log
```

诊断套件：

```bash
python tracking/visualte_diagnostic_suite.py --config dutrack_384_full_lmq_d1_e10 --runid 10 --max_frames 5 --stat_frames 0 --vis_frames 5 --top_ratio 0.1 --hardneg_topk 6 --language_init_source dataset_or_class --language_update_mode anchor --case otb_lang:Biker --case hoot_balanced20:0 --out_dir output/test/visualte_diagnostic_suite --output_tag lmq_d1_fullstats
```
