# Stage 3 Learnable Multi-Query Prior 方案

更新时间：2026-05-24

## 1. 本阶段必须明确的四点

### 1.1 Stage 3-A / 3-B 是诊断，不是最终端到端方案

Stage 3-A / 3-B 中的：

```text
gap_anchor / gap_blip / gap_current
alpha = softmax(gap / tau)
P_final = sum(alpha_m * P_m)
```

只用于回答诊断问题：

```text
1. anchor language 是否仍然稳定；
2. BLIP/current language 是否在部分帧提供额外定位证据；
3. 不同语言源产生的 prior 是否能区分 target 与 hard negative；
4. 语言源是否存在漂移、背景偏置或上下文误导。
```

它们不是最终端到端模块。原因是 gap router 依赖人工定义的 `P_pred`、`N_hard` 和 `tau`，属于运行时启发式选择器，不应作为长期主机制。

### 1.2 最终模块应是 learnable multi-query prior

最终主线应从：

```text
人工词筛选 / 手工 gap 路由 / floor-cap 规则
```

转为：

```text
language tokens -> learnable multi-query generator -> search prior -> bounded center-score bias
```

也就是让语言状态通过可学习 query 与 search tokens 交互，生成一个对 center score 有增量作用的 prior。

### 1.3 detach 只用于 aux score-rank loss

训练时需要区分：

```text
tracking loss:
  S_final = S_base + prior_bias
  不 detach S_base，让最终预测误差驱动 prior module 学习。

aux score-rank loss:
  S_aux = S_base.detach() + prior_bias
  detach 只用于让辅助约束主要优化 prior，而不是破坏原 center head。
```

不能把 detach 用到整个训练路径里，否则 prior module 只能学习局部修正，难以真正服务最终 tracking loss。

### 1.4 Stage 2 reliability 只作为 warm-start / diagnostic

Stage 2 的 `word reliability` 已证明可以改善：

```text
word weight 与 target-hardneg gap 的 rank alignment
```

但它不应成为长期主机制。后续只建议作为：

```text
1. 诊断指标；
2. query 初始化或 warm-start 权重；
3. early-stage 的弱先验。
```

最终仍应让 query generator 学习词权重、词组合和跨模态兼容关系。

---

## 2. 当前阶段推荐方案

### 2.1 模块定位

模块名称建议：

```text
Language-Conditioned Multi-Query Prior
```

职责：

```text
1. 从 anchor / current / BLIP language tokens 中生成 K 个 language-conditioned queries；
2. 用 queries 与 search tokens 做 compatibility scoring；
3. 生成 search-space prior；
4. 将 prior 作为有界 additive bias 注入 center score；
5. 不修改 backbone attention；
6. 不影响 size branch 和 offset branch。
```

### 2.2 输入输出

输入：

```text
L_anchor: 初始语言 token，身份参考；
L_current: 当前语言状态 token，可选；
L_blip: 当前 BLIP caption token，可选；
X_search: search tokens；
S_base: 原 center score logits。
```

输出：

```text
prior_bias: 与 center score map 对齐的有界 bias；
S_final = S_base + prior_bias。
```

### 2.3 推荐结构

#### 2.3.1 共享语言投影

不同语言源必须共享同一套投影：

```text
H_l = LN(L)
Z_l = normalize(W_l H_l)
```

原因：

```text
1. 保证 anchor / BLIP / current 的 prior 分数尺度可比较；
2. 避免每路语言单独拟合；
3. 控制参数量。
```

#### 2.3.2 共享视觉投影

search tokens 进入同一个视觉投影：

```text
Z_x = normalize(W_x X_search)
```

不能假设 raw language token 和 raw visual token 天然可比较，必须通过可学习投影进入统一 compatibility space。

#### 2.3.3 Multi-query generator

使用 K 个可学习 query seed 从语言 token 中聚合信息：

```text
E = {e_1, e_2, ..., e_K}

A_lang = softmax((E W_q) (Z_l W_k)^T / sqrt(d))
Q_lang = A_lang (Z_l W_v)
```

其中：

```text
K = 4 或 5
```

不要手工规定 query 1 是 subject、query 2 是 attribute。让模型在训练中自然分化。

#### 2.3.4 Query-search compatibility

每个 query 对 search tokens 产生一张 prior map：

```text
C_k(j) = q_k · z_xj
```

融合方式第一版保持轻量：

```text
P(j) = MLP_or_weighted_sum({C_k(j)}_{k=1..K})
```

不建议第一版使用复杂 transformer decoder，避免训练压力过大。

#### 2.3.5 有界 score bias

prior 不直接替换 score，只作为有界增量：

```text
prior_bias = beta * clamp(norm(P), -c, c)
S_final = S_base + prior_bias
```

关键约束：

```text
1. beta 可学习或配置；
2. clamp 防止 prior 过强；
3. 记录 clamp ratio；
4. 只接 center score，不接 size/offset。
```

---

## 3. 训练目标

### 3.1 Tracking loss 为主

主损失保持原 tracking loss：

```text
L_track = L_cls/location + L_box/l1/giou
```

训练路径：

```text
S_final = S_base + prior_bias
pred_box = head(S_final, size, offset)
L_track -> query prior module
```

这里不 detach `S_base`，否则最终预测误差无法正常驱动 prior module。

### 3.2 Aux score-rank loss 为辅

辅助 loss 只用于让 prior 更关注 center score 的 target-hardneg 竞争：

```text
S_aux = S_base.detach() + prior_bias

L_score_rank =
  ReLU(m - mean(S_aux on P_pos) + mean(S_aux on N_hard))
```

其中：

```text
P_pos: GT center 附近或高质量目标区域；
N_hard: GT 外 score 较高的 hard negative 区域。
```

### 3.3 Prior regularization

避免 prior 学成全局扰动：

```text
L_bias = mean(prior_bias^2)
```

总损失：

```text
L = L_track + lambda_rank * L_score_rank + lambda_bias * L_bias
```

建议：

```text
lambda_rank 小；
lambda_bias 小；
aux loss 前期启用，后期退火；
记录 active ratio。
```

---

## 4. 训练策略

第一轮建议冻结：

```text
backbone
language encoder
size branch
offset branch
```

训练：

```text
language query generator
visual projection W_x
prior fusion head
center score adapter，可选
```

不建议第一轮训练 full head。原因：

```text
1. 目标是验证 language prior 是否能提供增量；
2. full head 会引入更多变量；
3. 当前数据和短训设置不足以稳定重训整个 head。
```

---

## 5. 实验分组

### 5.1 诊断组

```text
A0: 原 anchor baseline
A1: Stage 2 reliability only
A2: anchor prior diagnostic
A3: BLIP prior diagnostic
A4: anchor + BLIP prior diagnostic
```

这些组不作为最终方法，只用于判断语言源是否有可用视觉证据。

### 5.2 端到端模块组

```text
B1: single-query prior
B2: per-word query prior
B3: multi-query prior K=4
B4: multi-query prior K=5
B5: multi-query prior + score adapter
```

主判断：

```text
B3/B4 是否比 B1/B2 更能提升 target-hardneg gap；
B5 是否能把 prior 增益转成 score map / IoU 增益。
```

---

## 6. 必须记录的指标

### 6.1 Tracking 指标

```text
mean_iou
center error
score_map_mass_in_gt
score_onoff_peak_delta
peak distance
```

### 6.2 Prior 指标

```text
prior_pos_mean
prior_hardneg_mean
prior_gap
prior_bias_mean
prior_bias_max
prior_clamp_ratio
```

### 6.3 Query 指标

```text
query_attention_entropy
query_diversity
query_prior_gap_k
best_query_id
```

### 6.4 语言源指标

```text
gap_anchor
gap_blip
gap_current
anchor_blip_similarity
language_changes
unique_language_count
```

### 6.5 Aux loss 诊断

```text
score_rank_active_ratio
score_rank_loss
bias_reg_loss
```

---

## 7. 可视化要求

每个序列至少输出：

```text
1. base score map
2. prior bias map
3. final score map
4. P_anchor / P_blip / P_current 诊断图
5. K 个 query 的 attention/prior map
6. prior before/after clamp
7. hard negative token 位置
```

重点不是看 prior 是否“像目标区域”，而是看：

```text
1. prior 是否提高目标中心峰值；
2. prior 是否压低 hard negative；
3. final score peak 是否更接近 GT center；
4. query 是否出现不同关注模式，而不是 K 个 query 完全相同。
```

---

## 8. 当前执行顺序

### Step 1：固定 Stage 3-A 诊断

先输出：

```text
anchor prior
BLIP prior
anchor+BLIP prior
```

只判断可用性，不作为最终方法。

### Step 2：实现 multi-query prior module

第一版：

```text
K=4
shared W_l / W_x
simple compatibility
bounded score bias
score-only injection
```

### Step 3：短训验证

训练：

```text
query prior module
prior fusion head
可选 center score adapter
```

冻结：

```text
backbone
size branch
offset branch
```

### Step 4：判断是否继续

如果出现：

```text
prior_gap 提升
score_onoff_peak_delta 为正
score_map_mass_in_gt 提升
hard negative 被压低
```

再考虑扩大训练集和训练轮数。

如果只出现：

```text
prior_gap 提升，但 score map 不变
```

说明 prior 与 center head 仍未对齐，需要优先调整 score adapter，而不是继续加规则。

---

## 9. 本阶段的核心边界

当前阶段不要继续引入：

```text
1. subject floor；
2. context cap；
3. hard word filtering；
4. 手工 one-hot language source switch；
5. 直接改 backbone feature；
6. 直接改 size/offset。
```

当前阶段保留：

```text
1. Stage 2 reliability 作为诊断和 warm-start；
2. language source gap 作为诊断；
3. multi-query prior 作为主模块；
4. score-only bounded injection；
5. tracking loss 主导训练；
6. aux score-rank loss 只作为弱辅助。
```

最终目标：

```text
不是构造一个更复杂的规则系统，
而是验证一个轻量可训练的 language-conditioned prior
是否能稳定提升 center score 的目标/干扰区分能力。
```
