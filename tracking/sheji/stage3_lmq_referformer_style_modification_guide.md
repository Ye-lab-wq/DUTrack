# Stage 3-C1 后续修改指导：从 Dot-Product LMQ 到 ReferFormer-style Light Query Decoder

> 目标：结合当前 DUTrack/LMQ 实现与 ReferFormer “Language as Queries” 的设计思想，修正当前 `K=4` query prior 塌缩问题，并把 Stage 3-C1 推进到更合理的 **language-conditioned multi-query prior** 版本。

---

## 0. 当前问题结论

当前 `dutrack_384_full_lmq_k4_e10` 的核心现象是：

```text
query_seed 本身不同；
但经过 language attention / query pooling / search matching 后，
K 个 query prior map 几乎完全相同。
```

典型日志：

```text
lmq_query_prior_cosine_mean = 1.00000
lmq_query_prior_cosine_max  = 1.00000
lmq_query_fusion_entropy    = ln(4)
lmq_query_fusion_max        ≈ 0.25
q0/q1/q2/q3 prior gap 几乎完全一样
```

这说明：

```text
K 个 query 没有形成有效分化；
multi-query 实际退化成 single-query；
raw prior 本身 target-hardneg gap 不稳定；
当前还不到开放 score adapter 或放大 beta 的阶段。
```

当前主要矛盾不是：

```text
query_seed 初始化相同
beta 太小
score adapter 没开放
```

而是：

```text
query identity 在 language pooling / search matching 后被抹掉；
query 与 search tokens 的交互太浅；
多个 query 缺少持续的身份约束和分化机制。
```

---

## 1. ReferFormer 对当前 LMQ 的关键启发

ReferFormer 的 “Language as Queries” 不是简单：

```text
K 个 seed -> attend language -> dot search
```

而是更接近：

```text
text sentence feature 作为 decoder tgt
learned query embedding 作为 query identity / query position
query embedding 生成 reference points
decoder self-attention + visual cross-attention
query features 端到端输出 box / class / mask kernel
```

它避免 query 塌缩的关键不在于 “query seed 不同” 这一点本身，而是：

```text
1. query identity 持续保留在 decoder 内；
2. 不同 query 具有不同的 query_pos / reference point / decoder trajectory；
3. queries 与 visual memory 进行真正 cross-attention；
4. 最终由端到端检测/分割 loss 驱动 query 分化。
```

对 DUTrack 当前任务，不能直接照搬 mask decoder，但可以迁移其思想：

```text
language-conditioned query identity
query-search cross-attention
small query set
end-to-end loss-driven differentiation
score-only output
```

也就是说，DUTrack 中更合理的版本应该是：

```text
language tokens
-> K language-conditioned queries
-> light query-search decoder
-> K search prior maps
-> fused center-score prior bias
```

而不是当前过浅的：

```text
query_seed -> language attention -> Q_lang -> cosine(search)
```

---

## 2. 当前应先做的诊断：四级 Forward Probe

在继续改结构前，先定位 query 塌缩发生在哪一层。

### 2.1 需要记录的四级表示

在 `lib/models/dutrack/language_multi_query_prior.py` 中额外输出：

```text
query_seed:          [K, D]
query_lang_attn:     [B, K, T_lang]
Q_lang:              [B, K, D]
query_prior_maps:    [B, K, N_search]
```

### 2.2 需要记录的指标

#### Seed 层

```text
seed_pairwise_cosine_mean
seed_pairwise_cosine_max
```

#### Language attention 层

```text
query_lang_attn_cosine_mean
query_lang_attn_cosine_max
query_lang_attn_entropy_mean
query_lang_attn_entropy_min
query_lang_attn_entropy_max
```

#### Query 表示层

```text
Q_lang_cosine_mean
Q_lang_cosine_max
Q_lang_norm_mean
Q_lang_std_across_queries
```

#### Prior map 层

```text
query_prior_cosine_mean
query_prior_cosine_max
query_prior_std_across_queries
query_prior_gap_q0 ... query_prior_gap_qK
```

### 2.3 判断逻辑

#### 情况 A：`query_lang_attn` 已经相同

```text
seed 不同
query_lang_attn 接近相同
Q_lang 接近相同
prior map 接近相同
```

说明塌缩发生在 language pooling 阶段。

优先修改：

```text
Q_lang = Attn(E, L) + gamma * E
query-specific projection
attention temperature
```

#### 情况 B：`Q_lang` 不同，但 `prior map` 相同

```text
seed 不同
query_lang_attn 不同
Q_lang 不同
prior map 相同
```

说明塌缩发生在 search matching 阶段。

优先检查：

```text
visual_proj 是否输出过于集中
search token variance 是否足够
cosine / tanh / normalization 是否压平差异
K 维是否被错误广播或平均
```

#### 情况 C：prior map 实际不同，但日志显示 cosine=1

说明诊断统计有 bug。

优先检查：

```text
cosine 计算维度
是否误用了 fused prior
是否在统计前把 K 维平均掉
```

---

## 3. 修改路线总览

建议按下面顺序推进，不要一次性加入太多变量。

```text
C1-Probe:
  只加四级 forward probe，不改结构。

C1-R:
  在当前 dot-product LMQ 中加入 seed residual，保留 query identity。

C1-K1 / C1-PW:
  做 K=1 和 per-word query baseline，判断 multi-query 是否真的必要。

C1-D:
  加 light query-search decoder，让 query 与 search tokens 进行真正 cross-attention。

C2:
  在 prior 有稳定正向 gap 后，再开放 small center score adapter。
```

---

## 4. C1-Probe：只加诊断，不改结构

### 4.1 修改文件

```text
lib/models/dutrack/language_multi_query_prior.py
lib/train/actors/dutrack.py
tracking/visualte_diagnostic.py
```

### 4.2 新增输出

在 `LanguageMultiQueryPrior.forward()` 的 `aux` 中加入：

```python
aux = {
    "lmq_query_seed": query_seed.detach(),
    "lmq_query_lang_attn": attn.detach(),
    "lmq_queries": q_lang.detach(),
    "lmq_query_prior_maps": query_prior_maps.detach(),
}
```

注意：

```text
训练主路径不要 detach prior_scores；
诊断输出可以 detach。
```

### 4.3 训练日志新增项

```text
ScorePrior/lmq_seed_cosine_mean
ScorePrior/lmq_seed_cosine_max
ScorePrior/lmq_lang_attn_cosine_mean
ScorePrior/lmq_lang_attn_cosine_max
ScorePrior/lmq_lang_attn_entropy
ScorePrior/lmq_query_cosine_mean
ScorePrior/lmq_query_cosine_max
ScorePrior/lmq_query_prior_cosine_mean
ScorePrior/lmq_query_prior_cosine_max
ScorePrior/lmq_query_prior_std_across_queries
```

### 4.4 目的

先回答：

```text
query 在哪一步塌缩？
是 language attention 塌缩？
是 Q_lang 塌缩？
还是 search matching 塌缩？
```

---

## 5. C1-R：加入 seed residual，保留 query identity

当前结构可能是：

```text
Q_lang = Attn(E, L)
```

建议改为：

```text
Q_lang = LN(Attn(E, L) + gamma * E)
```

其中：

```text
E: learnable query seed
gamma: 固定 0.1 或可学习标量
```

### 5.1 推荐配置

```yaml
MODEL:
  TE:
    LMQ_SEED_RESIDUAL: true
    LMQ_SEED_RESIDUAL_GAMMA: 0.1
```

### 5.2 为什么这样改

当前问题是：

```text
seed 不同，但经过 language pooling 后被抹掉。
```

seed residual 的作用是：

```text
让 query identity 在 language pooling 之后仍然存在；
避免所有 query 都变成同一组语言 token 的平均表达。
```

### 5.3 注意事项

不要一开始加太强的 residual：

```text
gamma = 1.0
```

可能让 query 过度依赖 seed、忽略语言。第一版建议：

```text
gamma = 0.1 / 0.2
```

---

## 6. C1-K1 / C1-PW：必要对照

### 6.1 K=1 single-query baseline

新增配置：

```text
dutrack_384_full_lmq_k1_e10
```

目的：

```text
判断当前 K=4 是否真的优于 single-query；
如果 K=4 query maps 全塌缩，那么理论上它应该接近 K=1。
```

必须保持其他条件一致：

```text
same layer
same beta
same clamp
same aux loss
same trainable params except query count
same epoch
```

### 6.2 Per-word query baseline

实现一个更接近 Stage 2 的版本：

```text
q_i = W_l word_i
P_i(j) = q_i · W_x x_j
P(j) = Σ_i w_i P_i(j)
```

其中 `w_i` 可用：

```text
softmax word logits
或 Stage2 reliability
```

目的：

```text
如果 per-word query 有正 gap，而 LMQ 没有，
说明问题在 multi-query generator。

如果 per-word 也没有正 gap，
说明问题在 language-search compatibility / prior injection。
```

---

## 7. C1-D：加入 Light Query-Search Decoder

如果 C1-R 仍不能解决 query collapse，建议进入轻量 decoder 版本。

### 7.1 结构

```text
L tokens -> shared lang projection
E seeds -> query-language attention
Q0 = LN(Attn(E, L) + gamma * E)

Q1 = SelfAttention(Q0)
Q2 = CrossAttention(Q1, K=X_search, V=X_search)

prior_k(j) = MLP([Q2_k, X_j])
P = Fusion({prior_k})
prior_bias = beta * bounded_norm(P)
S_final = S_base + prior_bias
```

### 7.2 为什么需要 query-search decoder

当前 dot-product matching 太浅：

```text
C_k(j) = q_k · x_j
```

这容易让所有 query 都匹配到同一个最显著 visual pattern。

加入 query-search cross-attention 后：

```text
每个 query 可以从 search tokens 中聚合不同视觉证据；
query identity 可以在视觉交互中继续保留；
更接近 ReferFormer 的 language-as-query 思想。
```

### 7.3 推荐轻量实现

不要上完整 Deformable DETR decoder。第一版只做：

```text
1 层 self-attention
1 层 cross-attention
hidden dim = 256
num heads = 8
dropout = 0.1
```

输出仍然只接：

```text
center-score prior
```

不要修改 backbone feature，不接 size/offset。

---

## 8. Loss 与训练策略保持不变

### 8.1 训练参数

C1-R / C1-D 第一轮仍然只训练：

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

### 8.2 损失

保持：

```text
L = L_track + lambda_rank * L_score_rank + lambda_bias * L_bias
```

其中：

```text
tracking loss:
  S_final = S_base + prior_bias

aux score-rank loss:
  S_aux = S_base.detach() + prior_bias
```

不要把 detach 用到主 tracking path。

### 8.3 不要马上开放 score adapter

进入 C2 的前提是：

```text
lmq prior gap 稳定为正；
query prior maps 不再完全塌缩；
score_onoff_peak_delta 不为负；
prior_bias 有可测影响。
```

在此之前不要开放 center score adapter。

---

## 9. 成功标准

### 9.1 C1-Probe 成功

能明确定位塌缩位置：

```text
language attention 塌缩
或 Q_lang 塌缩
或 search matching 塌缩
或统计 bug
```

### 9.2 C1-R 成功

相对原 C1：

```text
query_prior_cosine_mean 明显下降
query_prior_gap 至少部分 query 为正
fusion entropy 不一定下降，但 query maps 应有差异
prior_pos_gain > prior_hard_neg_gain
```

### 9.3 C1-D 成功

相对 C1-R：

```text
prior_gap 更稳定为正
query maps 有可解释差异
score_onoff_peak_delta 不为负
score_map_mass_in_gt 不下降
IoU 至少不明显下降
```

---

## 10. 当前不要做的事

暂时不要：

```text
1. 放大 beta；
2. 开放 score adapter；
3. 加 strong diversity loss；
4. 加 subject floor / context cap；
5. 重新引入 hard filtering；
6. 让 BLIP 直接替换 anchor；
7. 修改 size/offset；
8. 修改 backbone attention。
```

原因：

```text
当前 prior 本身还没有稳定正向；
query 尚未分化；
过早加这些会掩盖真正问题。
```

---

## 11. 最终建议执行顺序

```text
Step 1:
  加 C1-Probe 四级诊断，确认塌缩位置。

Step 2:
  跑 K=1 / K=4 / per-word baseline，确认 multi-query 是否真正有必要。

Step 3:
  如果塌缩发生在 language pooling：
    加 seed residual，跑 C1-R。

Step 4:
  如果塌缩发生在 search matching：
    检查 visual projection / prior norm；
    再考虑 C1-D light decoder。

Step 5:
  如果 C1-R 仍不分化：
    进入 C1-D light query-search decoder。

Step 6:
  只有 prior gap 稳定为正后，
    才进入 C2 small center score adapter。
```

---

## 12. 一句话总结

当前 C1 的失败不是否定 “language as queries” 方向，而是说明当前实现还太浅：

```text
query_seed -> language pooling -> dot search
```

不足以产生稳定分化的 language-conditioned queries。

下一步应该借鉴 ReferFormer 的核心思想：

```text
query identity 要持续保留；
query 要和 visual memory 做真正 cross-attention；
query 分化要由端到端 loss 驱动。
```

但在 DUTrack 中仍需保持边界：

```text
只输出 center-score prior；
不改 backbone；
不接 size/offset；
先诊断，再训练。
```
