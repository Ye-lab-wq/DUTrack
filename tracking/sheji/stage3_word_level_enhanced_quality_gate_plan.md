# Stage 3：Word-Level Enhanced Language Quality Gate 与语言状态更新实验指导

更新时间：2026-05-26

## 0. 当前阶段定位

当前实验已经基本明确：

```text
原 DUTrack dynamic language update trigger 更适合作为“候选语言生成器”，
而不是最终语言状态更新器。
```

原 trigger 主要基于：

```text
position change
scale change
color change
```

它可以较高召回可能需要更新的帧，但精度不足。全量诊断中，trigger precision 约 `30.9%`，说明如果触发后直接采用 BLIP，会引入大量不可靠语言。

当前 deploy-like quality gate 已经取得正向结果：

```text
accept precision ≈ 81.7%
useful recall ≈ 76.3%
hurt rejection ≈ 92.0%
false accept / harmful ≈ 8.0%
mean gate gain ≈ +5.33e-5
```

这说明 quality gate 方向成立。但当前 gate 仍存在核心问题：

```text
deploy gate 主要依赖预测框区域 P_pred 来计算 deploy_score_delta，
因此会受到预测框质量影响。
```

如果 `P_pred` 已经偏向干扰物或背景，gate 可能错误地认为 BLIP 对当前预测区域有帮助，从而接受有害 BLIP。

因此下一阶段不应继续单纯依赖 score-gap gate，也不应立刻进入复杂 latent language state updater，而应引入之前 word-level 实验中的有效思想：

```text
词级 target/context 建模
+
tracking-aware score evidence
+
semantic / confidence protection
```

目标是构建一个轻量、可解释、结合词级语言质量证据的 quality gate，为后续语言状态更新打基础。

---

## 1. 总体目标

下一阶段目标不是直接做完整语言状态更新，而是先提高动态语言候选的采用质量。

核心目标：

```text
提高 BLIP/current caption 的采用质量；
减少有害 BLIP 进入语言状态；
保留一部分真正有用的动态语言；
缓解 deploy_score_delta 的预测框自举问题；
为后续 latent language state updater 提供更可靠的输入。
```

当前阶段应收敛为：

```text
Trigger 负责生成候选语言；
Word-level enhanced quality gate 负责判断候选语言是否采用；
通过 gate 后再考虑语言状态更新。
```

---

## 2. 当前 gate 的问题

### 2.1 当前 deploy gate 机制

当前 deploy-like gate 大致是：

```text
原 trigger 触发
→ 生成 BLIP caption
→ 分别用 prev language 和 BLIP language forward
→ 以当前预测框 P_pred 作为正区域
→ 计算 deploy_score_gap
→ 比较 deploy_score_gap(BLIP) - deploy_score_gap(prev)
→ 若为正且通过其他条件，则接受 BLIP
```

形式上：

```text
deploy_score_gap(L)
= mean(score_L on P_pred)
- mean(score_L on hard negative)
```

其中：

```text
L = prev language 或 BLIP language
P_pred = 当前 tracker 预测框区域
```

### 2.2 主要风险：预测框自举

如果 `P_pred` 准确：

```text
P_pred ≈ GT
```

则 deploy_score_gap 较可信。

但如果 `P_pred` 已经偏移：

```text
P_pred → background / distractor / occluder
```

则 gate 实际判断的是：

```text
BLIP 是否支持当前错误预测区域
```

而不是真正判断：

```text
BLIP 是否支持真实目标区域
```

这会导致：

```text
deploy_score_delta > 0
但 oracle_score_delta < 0
```

即 false accept。

### 2.3 全量诊断暴露的问题

当前全量诊断中：

```text
true_accept deploy_delta  ≈ +0.000450
false_accept deploy_delta ≈ +0.000470
true_accept semantic      ≈ 0.471
false_accept semantic     ≈ 0.466
```

说明 true_accept 与 false_accept 在当前 deploy 侧特征上非常接近。单纯调 score_delta 阈值或整句 semantic 阈值，难以稳定区分两者。

因此需要更细粒度的语言侧证据。

---

## 3. 为什么要参考之前 word-level 实验

之前 word-level 实验的价值不应被丢弃。它的结论可以重新定位为：

```text
不适合作为直接更新语言状态的最终手工规则；
但适合作为 quality gate 的词级语言质量证据。
```

之前实验得到的关键经验：

```text
1. word_direct_score / word_gap 具有 target-hardneg 判别信息；
2. 直接转 keep / sigmoid / prior 容易压缩 margin；
3. reliability only 相对更稳；
4. subject floor / context cap / type prior 等强规则容易破坏词权重排序；
5. 词级视觉证据可以诊断某个词是否指向目标；
6. 但人工词分类和硬阈值更新不适合作为最终主机制。
```

因此新阶段不应回到复杂手工词级更新，而应把 word-level 信息作为 gate 的辅助证据：

```text
旧方向：
word reliability → 直接筛词 / 改语言状态

新方向：
word-level evidence → 判断 BLIP 是否值得被采用
```

---

## 4. 新阶段核心思路

### 4.1 不再只依赖预测框自举

当前 gate 的问题是过度依赖：

```text
P_pred-based deploy_score_delta
```

下一阶段要引入语言侧约束：

```text
target-word consistency
context dominance
generic caption flag
word-level visual evidence
```

这些证据不完全依赖预测框区域，可以缓解 deploy evidence 被错误预测框带偏的问题。

### 4.2 借鉴 target/context 建模

参考 ATCTrack 一类 target/context 建模思想：

```text
候选语言中并不是所有词都同等重要；
目标主体词、属性词、上下文词应区别处理；
动态 caption 可能描述 context，而不是 target。
```

但当前不复现完整 ATCTrack，而只做轻量 gate 特征。

目标是区分：

```text
target cue:
  主体类别、核心身份、目标属性

context cue:
  背景、载体、遮挡物、周围人/物、空间关系

generic cue:
  object, thing, item, scene, area 等过泛化描述
```

---

## 5. 下一阶段模块定义

推荐模块名称：

```text
Word-Level Enhanced Language Quality Gate
```

或：

```text
Target-Context Aware Quality Gate
```

它不是语言状态更新器，而是语言状态更新之前的候选筛选器。

输入：

```text
anchor_description
prev_description
blip_description
deploy_score_delta
score confidence features
trigger features
word-level target/context features
```

输出：

```text
accept_blip / reject_blip
```

后续可扩展为：

```text
soft accept score
update strength g_t
```

但第一版只做二值 gate。

---

## 6. 需要补充到 error report 的 word-level 字段

当前 `s0_error_report.csv` 已包含：

```text
trigger error type
gate error type
deploy_score_delta
oracle_score_delta
semantic
score_peak / peak-second gap
pred_box_jump_ratio
trigger reason
anchor_iou / prev_iou / blip_iou
anchor / prev / blip descriptions
```

下一阶段建议补充以下字段。

### 6.1 目标词一致性

```text
anchor_content_words
prev_content_words
blip_content_words

target_word_overlap_anchor_blip
target_word_overlap_prev_blip
class_word_presence_blip
target_word_missing_flag
```

含义：

```text
BLIP 是否仍然包含与 anchor / class 一致的目标主体词。
```

### 6.2 上下文占比

```text
context_word_count
context_word_ratio
person_context_flag
background_context_flag
carrier_context_flag
occluder_context_flag
```

含义：

```text
BLIP 是否主要描述 person / background / room / road / carrier / occluder 等上下文。
```

### 6.3 泛化描述检测

```text
generic_word_count
generic_caption_flag
blip_num_words
blip_content_word_count
```

典型 generic words：

```text
object
thing
item
scene
area
picture
image
person
people
someone
something
```

注意：person 在人体目标中可能是 target，在非人体目标中可能是 context，需要结合目标类别判断。

### 6.4 词级视觉证据

如果可以接入之前 word_direct / word_gap 计算，建议记录：

```text
target_word_gap_mean
target_word_gap_max
context_word_gap_mean
context_word_gap_max
target_minus_context_gap
blip_word_gap_mean
blip_word_gap_max
```

含义：

```text
BLIP 中 target words 是否比 context words 更支持当前目标区域。
```

第一版可先不进入决策，只用于诊断 true_accept / false_accept 的差异。

---

## 7. 实验路线

### Step 1：Word-level diagnostic only

先不要修改 gate 逻辑，只补充 word-level 字段到 error report。

当前实现采用可选开关：

```text
tracking/language_state_s0_probe.py --word_evidence
tracking/language_state_s0_screen.py --word_evidence
```

新增字段分两类：

```text
template target consistency:
  blip_word_target_template_gap_mean
  blip_word_context_template_gap_mean
  blip_word_new_template_gap_mean
  blip_word_target_minus_context_template_gap
  blip_minus_prev_target_template_gap

search deploy evidence:
  blip_word_target_search_deploy_gap_mean
  blip_word_context_search_deploy_gap_mean
  blip_word_new_search_deploy_gap_mean
  blip_word_target_minus_context_search_deploy_gap
  blip_minus_prev_target_search_deploy_gap
```

其中 template target consistency 使用 DUTrack memory template 的目标 mask，
用于降低当前预测框漂移导致的自举风险；search deploy evidence 仍依赖当前预测框，
只作为对照诊断。

重点比较：

```text
true_accept vs false_accept
false_reject vs true_reject
```

看这些特征是否有区分度：

```text
target_word_overlap
target_word_missing_flag
context_word_ratio
generic_caption_flag
target_minus_context_gap
```

目标：

```text
判断 false_accept 是否更常出现 target 缺失、context 占比高、generic 描述；
判断 true_accept 是否更常包含有效 target/attribute 词。
```

如果这些差异不明显，不要急着加入 gate。

---

### Step 2：Rule-lite target/context gate

如果 Step 1 显示 word-level 特征有区分度，则加入轻量规则。

第一版建议：

```text
accept_blip =
    deploy_score_delta > eps
    and confidence_ok
    and target_consistency_ok
    and not context_dominant
```

其中：

```text
target_consistency_ok:
  target_word_overlap_anchor_blip > 0
  or class_word_presence_blip = True

context_dominant:
  context_word_ratio > context_thr
  and target_word_missing_flag = True
```

注意：

```text
不要使用 subject floor / context cap 这类强行改词权重的机制；
只作为 accept/reject 的弱约束。
```

---

### Step 3：Gate feature ablation

比较：

```text
G0: deploy_score_delta only
G1: score + confidence
G2: score + target_consistency
G3: score + context_dominance filtering
G4: score + confidence + target/context
```

指标：

```text
false_accept_rate
useful_update_recall
hurt_rejection_rate
accept_precision
gate_gain
accepted_update_rate
```

核心目标：

```text
target/context 特征是否降低 false_accept；
是否不会大幅牺牲 useful_update_recall；
是否在 HOOT 上更有效。
```

---

### Step 4：与原 deploy 对比

仍然保留四组：

```text
A0: no update / anchor-prev baseline
A1: original deploy direct update
A2: deploy + current quality gate
A3: deploy + word-level enhanced quality gate
A4: oracle gate upper bound
```

目标：

```text
证明 A3 相比 A2 能进一步减少 false accept；
证明 A3 相比 A1 更稳定；
评估 A3 是否接近 A4 的部分上界。
```

---

### Step 5：进入语言状态更新前的判断

只有当 word-level enhanced gate 满足以下条件，才进入下一阶段语言状态更新：

```text
false_accept_rate 进一步下降；
useful_update_recall 不明显崩；
gate_gain 非负且优于当前 gate；
在 HOOT 困难序列上更稳；
tracking AUC / precision 不下降。
```

否则不要进入 latent language state updater。

---

## 8. 与语言状态更新的关系

当前阶段仍是：

```text
hard text replacement:
  gate accept -> prev_description = blip_description
```

这不是最终目标。

新增诊断对照：

```text
anchor-preserving state update:
  gate accept -> prev_description = compact(anchor_description + BLIP state words)
```

该对照用于验证：

```text
anchor 保主体、BLIP 补状态
```

是否比直接替换 BLIP 更稳定。它仍然是文本级诊断，不是最终 latent state。

最终方向应是：

```text
gate 提高候选语言质量
→ 高质量候选进入 language state updater
→ 状态更新不是简单文本替换，而是连续语言状态维护
```

未来形式可以是：

```text
H_t = (1 - g_t) * H_{t-1} + g_t * H_candidate
```

其中：

```text
g_t 由 quality gate / target-context evidence / tracking confidence 共同决定。
```

但这必须建立在 gate 已经能稳定筛选候选语言的基础上。

当前不要直接做 latent updater。

---

## 9. 当前不建议做的事情

暂时不要：

```text
1. 直接复现完整 ATCTrack；
2. 重新引入 subject floor / context cap / type prior 强规则；
3. 直接用 word-level reliability 修改语言状态；
4. 把 BLIP 整句直接替换 anchor；
5. 继续加深 LMQ query decoder；
6. 在 gate 不稳定前训练端到端 updater。
```

原因：

```text
当前目标是提高候选语言质量判断；
不是直接设计完整语言更新系统。
```

---

## 10. 推荐最小实验包

下一步最小可行实验：

```text
Experiment 1:
  在 s0_error_report 中补充 word-level target/context 字段。

Experiment 2:
  统计 true_accept / false_accept 的 word-level 特征差异。

Experiment 3:
  加入 target_consistency + context_dominance 的 rule-lite gate。

Experiment 4:
  对比 current gate vs word-level enhanced gate。

Experiment 5:
  若有效，再接入真实 tracker test，观察 AUC / precision / IoU。
```

---

## 11. 阶段性表述建议

中文：

```text
前期实验表明，仅依赖预测框区域计算的 deploy score gap 会受到 tracker 自举误差影响。
当预测框偏离真实目标时，gate 可能错误接受支持错误区域的 BLIP 描述。
为缓解这一问题，我们不再单纯依赖结果导向的 score-gap gate，
而是引入轻量的词级 target/context 建模，将目标词一致性、上下文占比和词级视觉证据作为候选语言质量判断的辅助信息。
该模块的目标不是直接更新语言状态，而是在语言状态更新前过滤不可靠候选，从而提高动态语言更新的稳定性。
```

英文：

```text
Our previous deploy-like gate relies on prediction-box-based score evidence,
which may be biased when the tracker prediction is unreliable.
To reduce this self-bootstrapping risk, we introduce lightweight word-level
target/context cues into the language quality gate.
Instead of directly rewriting the language state with handcrafted word rules,
we use target-word consistency, context dominance, and word-level visual evidence
as auxiliary reliability cues for candidate caption filtering.
The goal is to improve the quality of accepted dynamic language observations
before moving toward latent language state updating.
```

---

## 12. 一句话总结

下一阶段目标应从：

```text
单纯依赖 deploy score-gap 的 quality gate
```

推进到：

```text
结合 word-level target/context 建模的轻量 language quality gate
```

用它提高候选 BLIP 语言的采用质量，再考虑真正的连续语言状态更新。
