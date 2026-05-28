# Stage 3-S Language State Prior 实验主线

更新时间：2026-05-25

## 0. 先回答一个核心问题：更新后的语言状态是什么

`H_t` 不是一句完整自然语言，也不应该被强行解码成人能读的 caption。

更准确地说：

```text
H_t 是一组模型内部的 latent language state tokens。
```

它来自：

```text
初始语言 anchor
上一帧语言状态 H_{t-1}
当前 BLIP/current caption 候选语言
当前帧视觉证据 search tokens / base score
```

经过更新模块后，得到一组更适合当前跟踪目标的连续向量 token：

```text
H_t = g_t * H_candidate_t + (1 - g_t) * H_{t-1}
```

因此它不是：

```text
1. 原始 anchor 句子；
2. 原始 BLIP 句子；
3. anchor 和 BLIP 的文本级拼接；
4. 一句新的自然语言描述。
```

它是：

```text
一组融合了初始身份约束、当前候选语言和视觉证据的 tracking-oriented language tokens。
```

这些 tokens 的目标不是可读性，而是：

```text
1. 保持目标身份；
2. 吸收可靠的当前外观变化；
3. 避免 BLIP/current caption 漂移到背景或遮挡物；
4. 为 search-space center score prior 提供更稳定的语言条件。
```

可以把它类比成视觉 tracking 里的 template memory：

```text
template memory 不等于原图；
language state memory 也不等于原句。
```

区别是：

```text
template memory 维护目标视觉外观；
language state memory 维护目标语言语义和可迁移外观提示。
```

## 1. 当前问题回顾

### 1.1 C1-D 仍然在做什么

当前 `dutrack_384_full_lmq_d1_e10` 的路径是：

```text
language tokens
-> learnable multi-query prior
-> lmq_prior_scores
-> bounded score bias
-> center score logits
```

它仍然属于：

```text
语言影响视觉侧 center score。
```

但它默认输入语言已经可靠，没有处理：

```text
1. anchor 可能过时；
2. BLIP/current caption 可能漂移；
3. 词级证据与原始 word weight 不一致；
4. 语言 prior 应该启用、减弱还是关闭。
```

### 1.2 C1-D 结果暴露的问题

第 1 epoch 日志说明：

```text
lmq_query_lang_attn_cosine_mean = 1.0
lmq_pooled_query_cosine_mean = 1.0
lmq_query_search_attn_entropy 接近 log(576)
lmq_query_search_attn_max 接近 1/576
```

也就是说：

```text
语言侧 query 读到的内容高度一致；
search cross-attention 没有形成局部选择；
继续放大 beta 或开放 head 并不能解决语言质量和状态不稳的问题。
```

## 2. Stage 3-S 总体结构

Stage 3-S 将语言调制拆成两层：

```text
Language State Updater
Language-to-Score Prior Generator
```

完整数据流：

```text
H_anchor
H_{t-1}
H_blip_t / H_current_t
X_search
S_base
      |
      v
Language State Updater
      |
      v
H_t
      |
      v
Language-to-Score Prior Generator
      |
      v
P_t
      |
      v
S_final = S_base + bounded_bias(P_t)
```

其中：

```text
H_anchor:
  初始语言编码，是身份锚点。

H_{t-1}:
  上一帧 latent language state tokens。

H_blip_t / H_current_t:
  当前帧候选语言编码，可能包含新外观，也可能包含背景漂移。

X_search:
  当前搜索区域视觉 tokens。

S_base:
  原始 center score logits/map。

H_t:
  更新后的 latent language state tokens。

P_t:
  由 H_t 生成的 search-space prior。
```

## 3. 数学形式

### 3.1 CandidateAdapter

候选语言不能直接使用 BLIP，需要先被 anchor 约束：

```text
H_candidate_t = CandidateAdapter(H_blip_t, H_anchor)
```

目标：

```text
保留与目标身份一致的候选信息；
抑制只描述背景、遮挡物或上下文的候选信息。
```

### 3.2 Update Gate

更新门控：

```text
g_t = Gate(H_{t-1}, H_candidate_t, H_anchor, X_search, S_base)
```

它不是简单文本相似度，而应关注：

```text
candidate prior 是否优于 previous prior；
candidate 是否把响应推向 hard negative；
base score 当前是否可靠；
candidate 是否仍与 anchor 身份一致。
```

### 3.3 State Update

```text
H_t = g_t * H_candidate_t + (1 - g_t) * H_{t-1}
```

`g_t` 可以先实现为全局 scalar，再扩展为 token-wise gate：

```text
scalar gate:
  每帧一个更新强度，最稳定。

token-wise gate:
  每个 language token 一个更新强度，更灵活，但更容易漂。
```

第一阶段建议从 scalar gate 开始。

### 3.4 Prior Generator

```text
P_t = PriorGenerator(H_t, X_search)
```

这里可以复用或改造当前 LMQ：

```text
H_t -> K queries -> search prior maps -> fused prior
```

最终：

```text
S_final = S_base + bounded_bias(P_t)
```

继续只影响 center score，不影响 size / offset。

## 4. 分阶段实验路线

## Stage 3-S0：离线诊断，不改模型行为

目标：

```text
验证语言状态更新是否有必要。
```

做法：

```text
1. 分别计算 anchor prior、BLIP/current prior、previous-state prior；
2. 统计它们的 target-hardneg gap；
3. 检查什么时候 BLIP 优于 anchor，什么时候 BLIP 伤害目标；
4. 记录一个 oracle gate 上限。
```

输出诊断：

```text
P_anchor_gap
P_blip_gap
P_prev_gap
P_oracle_gap = max(P_anchor_gap, P_blip_gap, P_prev_gap)
oracle_update_rate
blip_better_than_prev_ratio
blip_hurts_ratio
```

判断：

```text
如果 P_oracle 明显优于 P_prev / P_anchor:
  说明语言状态更新有上限价值。

如果 P_blip 经常为负:
  说明必须有 gate，不能直接更新。
```

## Stage 3-S1：非训练版 latent state updater

目标：

```text
先建立 H_t 状态流，不训练端到端。
```

实现：

```text
H_anchor = encode(anchor)
H_blip_t = encode(blip_t)
H_candidate_t = CandidateAdapter(H_blip_t, H_anchor)
g_t = heuristic_gate based on deploy-like gap / score_gap
H_t = g_t * H_candidate_t + (1 - g_t) * H_{t-1}
```

注意：

```text
这个阶段是验证状态机制，不是最终方法。
```

诊断：

```text
g_t_mean
g_t_active_ratio
candidate_vs_anchor_cosine
candidate_vs_prev_cosine
state_vs_anchor_cosine
state_drift = 1 - cosine(H_t, H_anchor)
P_candidate_gap
P_prev_gap
P_state_gap
gate_correct_ratio = mean(g_t high and P_candidate_gap > P_prev_gap)
```

判断：

```text
如果 H_t gap 高于 H_prev 且 state_drift 可控:
  状态更新有价值。

如果 H_t 经常被 BLIP 拉向负 gap:
  gate 或 CandidateAdapter 不可靠。
```

## Stage 3-S2：可学习 gate，只训练语言状态模块

目标：

```text
让模型学习什么时候更新语言状态。
```

训练范围：

```text
LanguageStateUpdater
PriorGenerator
```

冻结：

```text
backbone
center head
size / offset branch
```

损失：

```text
tracking loss 仍为主导；
aux score-rank loss 小权重退火；
gate regularization 只作为轻约束。
```

建议：

```text
不要直接监督 g_t 等于 oracle gate。
可以用 oracle gate 只做诊断。
```

新增诊断：

```text
gate_mean
gate_entropy
gate_active_ratio
state_update_norm
state_drift
P_state_gap
score_onoff_peak_delta
prior_to_score_abs_ratio
```

## Stage 3-S3：可学习 CandidateAdapter

目标：

```text
让 BLIP/current caption 的候选语言先经过 anchor-conditioned adapter。
```

结构：

```text
H_candidate_t = H_anchor + Adapter([H_anchor, H_blip_t, H_anchor * H_blip_t])
```

或：

```text
H_candidate_t = CrossAttn(query=H_anchor, key/value=H_blip_t)
```

约束：

```text
identity consistency:
  H_candidate_t 不应远离 H_anchor。

visual usefulness:
  P_candidate 应该改善 target-hardneg gap。
```

注意：

```text
identity consistency 不能太强，否则永远不更新；
visual usefulness 不能太强，否则容易过拟合 hard negative 定义。
```

## Stage 3-S4：小范围开放 score adapter

只有当前面满足：

```text
P_state_gap 稳定为正；
gate_correct_ratio 高于 baseline；
prior_to_score_abs_ratio 不过小；
score_onoff_peak_delta 有正趋势；
```

才开放：

```text
small score adapter 或 center score branch 的小部分参数。
```

不建议一开始开放 head，因为会掩盖语言状态是否真的有用。

## 5. 与当前 Stage 2 / C1-D 的关系

### 5.1 和 Stage 2 reliability 的关系

Stage 2 是词级 runtime reliability：

```text
word_i reliability
```

Stage 3-S 是状态级 latent language memory：

```text
H_t tokens
```

Stage 2 可以作为诊断信号或 warm-start，但不应继续作为长期主机制。

### 5.2 和 C1-D 的关系

C1-D 默认输入语言可靠：

```text
H_language -> prior
```

Stage 3-S 先产生更可靠的语言状态：

```text
H_anchor, H_prev, H_blip, visual evidence -> H_t -> prior
```

因此 C1-D 的 PriorGenerator 可以被复用，但输入从原始 language tokens 改成 `H_t`。

## 6. 风险与约束

### 6.1 状态漂移

如果 `g_t` 长期偏大，BLIP/current caption 会把状态带向背景。

必须记录：

```text
state_drift
P_state_hardneg_gap
gate_active_ratio
```

### 6.2 状态不更新

如果 `g_t` 长期接近 0，退化为 anchor-only。

必须记录：

```text
update_rate
state_update_norm
blip_better_than_prev_ratio
```

### 6.3 语言状态不可解释

`H_t` 不是自然语言，因此不要用“生成了更好的句子”描述。

更准确表述：

```text
learned tracking-oriented latent language state
```

### 6.4 训练压力

本方案比 Stage 2 更复杂，但可以控制训练范围：

```text
先冻结 backbone/head；
只训练 updater + prior generator；
短训看 prior gap 和 gate correctness；
不要一开始追求完整数据集性能。
```

## 7. 第一轮建议实现顺序

1. 新增只读诊断脚本或诊断模式：

```text
Stage 3-S0 oracle source/state gap probe
```

2. 实现 `LanguageState` 容器：

```text
H_anchor
H_prev
H_candidate
H_state
```

3. 实现非训练版 scalar gate：

```text
g_t based on candidate_gap - prev_gap and score_gap
```

4. 接入当前 score prior generator：

```text
H_state -> lmq/prior -> score bias
```

5. 只跑 Biker / HOOT：

```text
先看状态更新是否提高 P_state_gap；
暂时不要用 OLOD 做主判断。
```

## 8. 推荐命名

实验主线：

```text
Stage 3-S: Language State Prior
```

第一版诊断：

```text
stage3_s0_state_probe
```

第一版非训练状态更新：

```text
stage3_s1_state_gate
```

第一版可学习状态更新：

```text
dutrack_384_full_lstate_s2_e10
```

其中：

```text
lstate = latent language state
s2 = learnable state updater
e10 = 10 epoch short training
```

