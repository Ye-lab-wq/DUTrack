# Stage 3 语言调制实验主线总结

## 1. 核心目标

当前主线要解决的问题不是“让语言参与网络”本身，而是：

```text
如何让语言模态通过与视觉的可靠交互，稳定改善 DUTrack 的最终 center score / box prediction。
```

DUTrack 原始模型已经有较强的视觉跟踪能力。语言调制只有在下面条件成立时才有价值：

1. 语言提供了 base tracker 没有充分利用的目标信息；
2. 这部分信息能被转成 search 空间中对目标中心更有利、对 hard negative 更抑制的信号；
3. 这种信号不会破坏已经训练好的 backbone / head 分布；
4. 推理阶段可以获得同样类型的信号，不能依赖 GT 或 oracle。

因此当前所有实验都围绕一个数学问题展开：

```text
给定语言状态 H 和视觉搜索特征 X，
能否学习到一个可靠的调制项 Δ，
使最终 score map S_final = S_base + Δ
在目标中心更尖锐，同时不抬高 hard negative。
```

## 2. 数学本质设计演进

### 2.1 早期 VLTE / TE-style policy

最初尝试把语言或 VLScore 转成 token keep / policy，作用在 Transformer attention 中：

```text
A = softmax(QK^T / sqrt(d))
A' = policy(A, keep)
Y = A'V
```

核心问题是：

- policy 主要改变 attention 分布，但最终 center head 要的是中心峰值；
- attention 中不同 Q 的语义不同，不能默认所有 Q 都应该看目标；
- keep 作用在中间层后，传到 score map 的路径很长，效果经常被后续层和 head 吸收；
- 可视化中 keep 有时看起来贴近目标，但 `A policy`、`A*V`、最终 score map 的变化很弱。

阶段结论：

```text
仅在 attention/feature 流中做轻量 keep，不一定能稳定影响 center score。
```

### 2.2 Score-prior 方案

为更贴近 center head，后续改为把 keep / prior 直接作为 center score bias：

```text
S_final = S_base + beta * bounded_bias(P_language)
```

并尝试：

- raw logits prior；
- decision prior；
- word-direct prior；
- safe multi-prototype prior；
- score-space auxiliary loss。

这条线更贴近最终任务，但暴露了新问题：

- raw prior 对比度低；
- decision / keep 与 base score 高度重复，容易退化成 score 的影子；
- score-space auxiliary 可能和原 center loss 重复或冲突；
- prior 对 GT mass / peak delta 的增益不稳定。

阶段结论：

```text
直接改 score 是更贴近目标的方向，但 prior 本身必须包含 base score 没有的新信息。
如果 prior 只是 score 的低对比度复制，beta 再大也难稳定提升。
```

### 2.3 Word-level language evidence

为避免整句语言混入 context，开始做词级语言证据：

```text
word_weight_i =
  role_score_i
  × reliability_score_i
  × discriminability_score_i

visual_score_j = Σ_i word_weight_i · sim(word_i, visual_token_j)
```

参考思想：

- subject words 描述目标本体；
- attribute words 可能补充外观；
- context / relation words 可能误导；
- BLIP/current caption 可能带来状态增量，也可能引入背景词。

已做诊断包括：

- direct language-search；
- template prototype；
- multi-prototype；
- safe multi-prototype；
- word reliability；
- target/context gap；
- anchor/BLIP gate。

阶段发现：

- 直接 language-search 常常不可靠；
- subject filtering 有帮助但不足；
- template mean prototype 会抹掉局部信息；
- multi-prototype / safe prior 可改善 probe 指标，但转成最终 score 收益仍弱；
- word reliability 可改善 rank alignment，但没有稳定转成 IoU。

阶段结论：

```text
词级建模是必要的，但当前词-视觉相似度本身仍不够可靠。
语言 token 与视觉 token 是否处在可比较空间，是核心风险。
```

### 2.4 Language state updater

后来意识到文本级 hard replacement 过于粗糙，因此转向 latent language state：

```text
H_anchor: 初始目标语言状态
H_prev:   上一帧语言状态
H_cand:   当前 BLIP / candidate 语言状态

H_t = Update(H_anchor, H_prev, H_cand, visual evidence)
```

目标不是输出一句更好的自然语言，而是输出一组更适合跟踪的 token embedding：

```text
H_t: 保留 anchor 主体、吸收 candidate 有用增量、拒绝噪声后的 latent language tokens
```

当前 keep-absorb 形式：

```text
H_state =
  w_anchor * H_anchor_aligned
  + w_prev * H_prev
  + w_candidate * H_candidate_aligned
  + g_token * g_frame * Δ
```

对应代码：

```text
lib/models/dutrack/language_token_state_updater.py
```

训练目标包括：

- final tracking loss；
- token absorb BCE；
- candidate cap；
- anchor cap；
- prev keep lower-bound；
- 当前 v3 已关闭 gain loss，避免 source gate shortcut。

token absorb positive target 由多条件构成：

```text
cand_gap > prev_gap + margin_rel
cand_gap > margin_abs
cand_hardneg_gain < hardneg_margin
identity / anchor consistency
multi-frame consistency
top-k sparse positive selection
```

## 3. 当前实验进展

### 3.1 TE / VLTE attention policy

已尝试：

- post-softmax policy；
- pre-softmax bias；
- query scope: q0 / track / search / visual；
- layer choices: 3/7/11, 7/15/23, 5/11/17；
- head freeze / head open；
- rank / center-rank auxiliary。

主要结论：

- keep 图像上有时明显，但最终 score map 变化弱；
- Q0/track Q 对最终结果更直接，但只改 Q0 容易陷入局部；
- search/template Q 的作用需要区分，不能全部当成“应该看目标”；
- 只靠中间 attention policy 很难稳定提升 center peak。

### 3.2 Score-prior / center-aware experiments

已尝试：

- score prior 只作用 score branch；
- 不改 size/offset；
- beta 大小实验；
- raw / decision prior；
- score-space corrective auxiliary；
- score-space prior gain。

主要结论：

- score-prior 路径更合理，因为直接对 center score 起作用；
- 但当前 prior 信息量不足，容易退化成 base score 的重复表达；
- peak delta / GT mass 的变化不稳定；
- 辅助 loss 如果直接优化 score，可能和原 tracking center loss 重复。

### 3.3 Word-level / prototype experiments

已尝试：

- word-level appearance probe；
- manual initial description；
- direct-only；
- single prototype；
- multi-prototype；
- direct + prototype confirmation；
- safe multi-prototype；
- word-direct margin prior；
- fullstats 统计语言变化。

主要结论：

- 单一 template mean prototype 不适合 DUTrack 多输入交互，会抹掉局部信息；
- direct language-search 有时比 prototype 更有效，但不稳定；
- multi-prototype probe 可改善 target/hard-negative contrast，但工程化后仍没有稳定收益；
- 语言源质量和语言状态稳定性成为更上游的问题。

### 3.4 Language update / quality gate

已尝试：

- original trigger；
- color trigger；
- deploy-like gate；
- oracle/deploy 对照；
- anchor baseline；
- BLIP candidate；
- text hard replacement；
- word-level conservative absorption。

主要结论：

- BLIP candidate 有时有增量，但也经常引入背景/错误状态；
- gate 可以降低部分 false accept，但 false reject / false accept 仍明显；
- hard replacement 不是合理长期方案；
- 文本级拼接和筛词新意有限，且语法/上下文风险大。

### 3.5 Learnable token-state updater

当前实现状态：

- `LanguageTokenStateUpdater` 已接入训练；
- 支持 cross-attn token alignment；
- 支持 relation block；
- 支持 visual evidence 输入；
- 支持 keep-absorb source gate；
- 支持 token absorb auxiliary。

近期实验观察：

#### v2

问题：

- `token_absorb_target_pos_ratio = 0`；
- `identity_ok_ratio = 0`；
- raw/aligned identity 空间不一致。

结论：

```text
identity 约束过严，且空间不对齐。
```

#### v3 / v3_2

修正：

- identity 改为 aligned candidate vs aligned anchor；
- raw identity 只作诊断；
- source gate 加 anchor cap 和 prev keep lower-bound；
- gain loss 关闭。

观察：

```text
v3 target_pos_ratio ≈ 0.0058
v3_2 target_pos_ratio ≈ 0.0129

candidate_weight_pos ≈ 0.0012 / 0.0028
candidate_weight_neg ≈ 0.0175

prev_keep_weight_mean ≈ 0.965
candidate_absorb_weight_mean ≈ 0.0175
anchor_weight_mean ≈ 0.0177

alignment_entropy ≈ 1.8
alignment_max ≈ 0.21
relation_attn_mean = 0.0625
```

当前结论：

- positive token 稀疏；
- positive candidate weight 没被拉高；
- source gate 仍接近 no-op；
- aligned identity 从过严变成过宽，几乎失去筛选作用；
- relation_attn_mean 指标本身无意义，因为 softmax attention 的均值天然接近 `1 / token_len`；
- `state_delta_abs_mean` 偏大主要来自 source mixture，不是 residual delta。

## 4. 当前阶段结论

### 4.1 主要矛盾

当前主要矛盾不是 layer 数、head 是否开放、beta 是否够大，而是：

```text
语言侧生成的 token / prior 是否真的包含可部署、可学习、能区分 target 与 hard negative 的新信息。
```

更具体地说：

1. token-level positive label 不够可靠；
2. token alignment 没有学出清晰结构；
3. candidate absorption 没有足够梯度；
4. source gate 容易选择 no-op；
5. 语言 token 与视觉 token 的对齐空间仍然不明确。

### 4.2 当前不能过早下的结论

不能说：

```text
语言模态没有用。
```

目前只能说：

```text
现有 lightweight language-to-visual / language-state 更新路径，
还没有稳定学出可转化为 center score 收益的可靠信号。
```

原因可能是机制问题，也可能是训练信号与数据构造问题。

## 5. 亟待解决的问题

### 5.1 token alignment 问题

当前 cross-attn alignment 接近均匀：

```text
alignment entropy 高
alignment max 低
```

这说明 candidate token 没有明确对齐到 prev/anchor 的语义位置。若 alignment 本身不可靠，再用 aligned candidate 生成 positive label，会形成自举误差。

需要解决：

- alignment 是否应预训练 / 冻结；
- identity 是否应基于 raw token、aligned token，还是额外投影空间；
- relation block 是否需要更明确的结构监督。

### 5.2 positive label 可靠性

当前 positive label 由 candidate token 对 search token 的 gap 构造：

```text
cand_gap - prev_gap
```

风险：

- 如果 token-search 相似度空间本身不可靠，positive label 就是噪声；
- 如果 search hard negative 来自错误 score peak，会自举错误；
- 如果 positive 极少，模型会学到全部拒绝更安全。

需要增加：

- positive label invariant 检查；
- positive token 的 identity / gap / hardneg 逐项统计；
- conservative positive + explicit positive weight。

### 5.3 candidate_weight_pos 没有被拉高

当前 BCE 没有显式处理正负极不平衡：

```text
positive token < 1% 或约 1%
negative token 占绝大多数
```

需要考虑：

```text
L_pos_candidate = pos * ReLU(w_pos_min - w_candidate)^2
```

或者：

```text
positive BCE weight
focal-style token absorb loss
```

否则 `candidate_weight_pos` 很难超过 `candidate_weight_neg`。

### 5.4 state_delta_abs_mean 指标拆分

当前：

```text
state_delta_abs = |H_state - H_prev|
```

但在 keep-absorb 模式下：

```text
H_state =
  source_mix + residual_delta
```

所以 `state_delta_abs_mean` 同时混合了：

- source mixture 带来的偏移；
- residual delta 带来的偏移。

需要新增：

```text
source_mix_delta_abs_mean
residual_delta_abs_mean
state_delta_abs_mean
```

否则会误判 residual 更新是否过大。

### 5.5 relation 指标无效

当前 `relation_attn_mean` 恒接近 `1 / token_len`，不能判断是否学出结构。

需要记录：

```text
relation_attn_entropy
relation_attn_max
relation_attn_diag_mass
relation_attn_offdiag_mass
```

### 5.6 deploy / oracle mismatch

很多有效诊断依赖 GT 或 oracle BLIP，但最终推理不能用 GT。

必须保持区分：

```text
oracle diagnostic: 判断有没有上界
deploy-like diagnostic: 判断真实推理可用性
```

任何只在 oracle 下有效的机制，都不能直接作为最终方案。

## 6. 可能存在的风险

### 6.1 机制退化风险

模型可能退化为：

- no-op：全部保持 prev；
- anchor shortcut：全部回退 anchor；
- score-copy：prior 复制 base score；
- hard replacement：接受 BLIP 后污染语言状态；
- over-regularized：所有 gate 被压死。

### 6.2 语言质量风险

BLIP caption 可能：

- 描述背景；
- 改变主体类别；
- 丢失关键目标词；
- 引入错误状态；
- 只描述当前可见区域而非长期目标 identity。

### 6.3 评价风险

当前很多实验是：

- 短训；
- 小序列；
- 单序列；
- OTB/HOOT/OLOD 少量样本；
- 单帧或前若干帧诊断。

这些结果只能说明机制趋势，不能直接说明整体数据集性能。

### 6.4 创新性风险

如果最终方案只是：

```text
BLIP 生成文本 + 简单筛词 + 拼接
```

创新性较弱。更有价值的方向是：

```text
从生成语言中学习可靠语义增量，
并以 latent language state 的形式服务 tracking。
```

### 6.5 空间对齐风险

当前语言 token、visual token、aligned token、score map 并不天然处在同一语义空间。直接点积或 cosine 可能没有稳定语义。

这会影响：

- word-level scoring；
- token absorb label；
- identity consistency；
- prior generation。

## 7. 下一步建议

### 7.1 先补诊断，不急着继续训练

优先新增：

```text
source_mix_delta_abs_mean
residual_delta_abs_mean
relation_attn_entropy
relation_attn_max
positive_identity_min / mean
positive_candidate_gap_mean
positive_hardneg_gain_mean
candidate_weight_pos_minus_neg
```

目的：

```text
确认当前 label / alignment / source gate 的定义是否可信。
```

### 7.2 修 token absorb loss

下一版应加入：

```text
positive candidate lower-bound
positive class weight
更严格的 positive invariant check
```

同时避免让正样本数量过少到没有梯度。

### 7.3 暂时不要扩大模块

当前问题不是模块容量不足，而是：

```text
监督目标、对齐空间、正负样本定义还不够可靠。
```

在这些问题没解决前，增加更多 query、更多 relation layer 或更复杂 gate，大概率只会增加不稳定性。

### 7.4 保持主线判断标准

后续每个版本都应回答三件事：

1. `candidate` 是否真的含有比 `prev/anchor` 更有用的视觉语言信息？
2. updater 是否能吸收这些信息而不是 no-op / shortcut？
3. 吸收后的语言状态是否能稳定提升 center score，而不是只改善 probe 指标？

## 8. 当前一句话总结

当前主线已经从“让语言直接筛视觉 token”推进到“学习一个跟踪可用的 latent language state”。这个方向比文本替换和人工规则更合理，但目前最关键的问题仍未解决：

```text
语言 token 与视觉 evidence 的对齐和 positive token 学习信号还不可靠，
导致 updater 倾向保持 prev，不愿真正吸收 candidate。
```

下一步应先修正诊断和 token absorb 目标，再判断是否继续训练更完整的端到端模块。
