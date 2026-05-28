# Stage 3：Quality Gate 后续实验计划

更新时间：2026-05-25

## 0. 当前阶段核心结论

当前实验已经基本明确：

```text
原 DUTrack language update trigger 适合看作“候选语言生成器”，
但不能直接看作“语言状态更新器”。
```

原因是：

```text
1. 原 trigger 触发率较高，missed-oracle 较低，说明召回不差；
2. 但 deploy false-positive 较高，说明很多触发帧的 BLIP 并不值得采用；
3. BLIP/current caption 有时有用，但经常有害；
4. deploy-like quality gate 已经能明显降低 false accept；
5. gate gain 为正但量级较小，说明当前主要是稳定性收益，而不是强性能提升。
```

下一阶段不要继续发散到复杂 latent language state updater，而应围绕一个清晰假设展开：

```text
在原 trigger 之后加入 tracking-aware language quality gate，
可以减少有害 BLIP 更新，同时保留部分有益更新，
从而提高动态语言更新的稳定性。
```

---

## 1. 两个问题必须分开

### 1.1 Trigger：什么时候生成候选语言？

原论文 trigger 主要基于：

```text
position change
scale change
color change
```

其作用是：

```text
判断目标状态是否变化明显，是否值得调用 BLIP 生成候选语言。
```

它解决的是：

```text
need-to-observe
```

而不是：

```text
candidate language reliability
```

### 1.2 Quality Gate：生成出来的语言是否采用？

quality gate 判断：

```text
BLIP/current caption 是否对当前 tracking 有用；
是否仍然保持目标身份；
是否会损害 target-hardneg score gap。
```

它解决的是：

```text
whether-to-use
```

完整流程应是：

```text
position/scale/color trigger
        ↓
调用 BLIP 生成 candidate caption
        ↓
quality gate 判断是否采用
        ↓
通过才更新 prev language / language state
```

---

## 2. 当前已完成实验与结论

### 2.1 Trigger diagnosis

已观察到：

```text
deploy trigger rate 较高；
deploy missed-oracle ratio 较低；
deploy false-positive ratio 较高。
```

说明：

```text
原 trigger 更像高召回候选生成器，
但精度不足。
```

### 2.2 Color trigger 对照

加入 color trigger 后：

```text
Bird1: 只多约 1 帧；
Dog: 无变化；
potted_plant-008: color 命中但与 position/scale 重叠；
toilet_paper-001: color 命中但与 position/scale 重叠。
```

结论：

```text
color trigger 当前边际贡献很小；
主要矛盾不是触发不足，而是触发后候选语言质量不稳定。
```

### 2.3 Oracle quality gate

使用 GT 区域定义：

```text
score_gap(BLIP) - score_gap(prev) > 0
```

作为 oracle gate 时，可以做到：

```text
useful_update_recall = 1.0
hurt_rejection_rate = 1.0
false_accept = 0
false_reject = 0
```

但这是定义上成立的 oracle 上界，不是部署机制。

结论：

```text
理论上存在可过滤空间，
但不能说明真实 deploy gate 已经可用。
```

### 2.4 Deploy-like quality gate

当前结果：

| 序列 | 原 deploy false positive | gate false accept | useful recall | gate gain |
| --- | ---: | ---: | ---: | ---: |
| Bird1 | 65.0% | 4.4% | 59.2% | +0.00030 |
| Dog | 73.6% | 16.7% | 65.5% | +0.00060 |

结论：

```text
deploy-like gate 明显降低有害 BLIP 接受率；
仍保留一半以上有益更新；
gate gain 为正但量级较小。
```

当前阶段可以认为：

```text
quality gate 方向成立，但收益仍弱，需要扩大验证和做特征消融。
```

---

## 3. 下一阶段总目标

下一阶段集中回答一个问题：

```text
deploy trigger + quality gate 是否优于原 deploy trigger 直接更新？
```

重点指标：

```text
false accept 是否下降；
useful update 是否保留；
score-gap gain 是否为正；
IoU / AUC 是否不下降或小幅提升。
```

---

## 4. 实验 A：原 deploy vs deploy + quality gate

### 4.0 Error report：逐帧错误归因

在正式比较 A0/A1/A2/A3 前，先生成跨序列逐帧错误报告。

新增输出：

```text
s0_error_report.csv
s0_error_report_summary.md
```

该报告由 `tracking/language_state_s0_screen.py` 汇总每个序列的
`stage3_s0_probe.csv` 得到，目标是一次性保留后续分析需要的关键字段。

核心字段包括：

```text
dataset / sequence / frame
trigger_error_type
gate_error_type
oracle_selected_source / deploy_selected_source / gate_selected_source
candidate_available
deploy_trigger
oracle_trigger
quality_gate_accept
quality_gate_source
quality_gate_score_delta
quality_gate_deploy_score_delta
oracle_gain_over_prev
anchor_score_gap / prev_score_gap / blip_score_gap
anchor_gap / prev_gap / blip_gap / oracle_gap
prev_deploy_gap / blip_deploy_gap
quality_gate_semantic
quality_gate_confidence_ok
score_peak / score_second_peak / score_peak_second_gap
pred_box_jump_ratio
trigger_by_position / trigger_by_scale / trigger_by_color
trigger_area_ratio / trigger_center_distance / trigger_color_delta
anchor_iou / prev_iou / blip_iou
anchor_description / prev_description / blip_description
```

`trigger_error_type` 用来诊断原 trigger 的触发时机：

```text
trigger_true_positive: 应该更新，也触发
trigger_false_positive: 不该更新，却触发
trigger_false_negative: 应该更新，却没触发
trigger_true_negative: 不该更新，也没触发
```

`gate_error_type` 用来诊断 quality gate 的采用决策：

```text
true_accept: 接受了有益 BLIP
false_reject: 拒绝了有益 BLIP
true_reject: 拒绝了有害 BLIP
false_accept: 接受了有害 BLIP
```

该步骤的目的不是调阈值，而是回答：

```text
false_accept 是 BLIP 描述错误、deploy score_delta 误判、预测框漂移，
还是 confidence / semantic 特征没有起作用？

false_reject 是 deploy score_delta 太弱、semantic 太低，
还是 trigger 时机过晚导致 candidate 不稳定？
```

只有先完成该错误归因，后续 gate 特征消融才不会退化成单序列经验调参。

### 4.1 对比组

建议固定四组：

```text
A0: anchor / prev baseline
    不使用 BLIP 或不更新语言。

A1: original deploy
    原 trigger 触发后直接采用 BLIP。

A2: deploy + quality gate
    原 trigger 只生成候选；
    通过 gate 后才采用 BLIP。

A3: oracle gate
    使用 GT score-gap 选择最优语言源；
    只作为上界诊断。
```

### 4.2 主要指标

```text
accepted_update_rate
false_accept_rate
useful_update_recall
hurt_rejection_rate
score_gap_gain
mean_iou
success_auc
precision
normalized_precision
```

### 4.3 判断标准

如果 A2 相比 A1：

```text
false_accept_rate 明显下降；
useful_update_recall 不崩；
score_gap_gain 为正；
IoU/AUC 不下降或小幅提升；
```

则说明 quality gate 具有实际价值。

---

## 5. 实验 B：多序列验证

不要只看 Bird1 / Dog。建议按三类序列验证。

### 5.1 正负混合困难例

特征：

```text
baseline 较低；
BLIP 有时有用，但经常有害；
oracle_gain 有一定正值。
```

建议：

```text
Bird1
Dog
potted_plant-008
toilet_paper-001
```

用途：

```text
验证 gate 能否保留有用 BLIP，同时拒绝有害 BLIP。
```

### 5.2 稳定负例

特征：

```text
anchor/prev 已经稳定；
BLIP 增量很小；
原 trigger 可能过度触发。
```

建议：

```text
Biker
otb_lang:0
其他 anchor IoU 高且 oracle_gain 低的序列
```

用途：

```text
验证 gate 是否能避免无效更新。
```

### 5.3 高风险 BLIP 例

特征：

```text
BLIP_hurts_ratio 高；
caption 容易描述背景、遮挡物或上下文。
```

用途：

```text
验证 gate 的鲁棒性。
```

建议从全量筛选表中选择：

```text
BLIP_hurts_ratio 高
deploy_false_positive 高
baseline 中等或偏低
```

的序列。

---

## 6. 实验 C：Gate 特征消融

当前 deploy-like gate 可能包含：

```text
deploy_score_delta
semantic_consistency
confidence_ok
```

建议做最小消融：

```text
C1: score only
    deploy_score_delta > eps

C2: score + confidence
    deploy_score_delta > eps and confidence_ok

C3: score + semantic
    deploy_score_delta > eps and semantic_consistency > threshold

C4: score + confidence + semantic
```

### 6.1 指标

```text
false_accept_rate
useful_update_recall
hurt_rejection_rate
gate_gain
accepted_update_rate
```

### 6.2 目标

回答：

```text
score evidence 是否已经足够？
confidence protection 是否能减少自举误判？
semantic consistency 是否能减少 caption drift？
```

---

## 7. 实验 D：Hard Replacement vs Conservative Update

当前通过 gate 后的状态更新是：

```python
if quality_gate_source == "blip":
    prev_description = blip_description
```

这是 hard text replacement。

风险：

```text
一次错误接受会污染后续 prev_description。
```

建议比较两个轻量版本。

### 7.1 Hard Replacement

```text
gate accept:
  prev_description = blip_description
```

### 7.2 Conservative Update

第一版可以不做 latent H_t，只做文本级保守策略：

```text
gate accept:
  prev_description = anchor_identity + blip_description
```

或更简单：

```text
连续 N 次 gate accept 后才更新 prev_description
```

建议先试：

```text
N = 2
```

### 7.3 评价

```text
state_drift
false_accept_after_update
long-term IoU/AUC
accepted_update_rate
```

目标：

```text
减少单次错误 BLIP 对后续状态的污染。
```

---

## 8. 实验 E：最终 tracking 指标验证

前面的 gate 指标都是中间诊断，最终仍需验证 tracking 性能。

### 8.1 必须报告

```text
Success AUC
Precision
Normalized Precision
Mean IoU
per-sequence IoU
```

### 8.2 建议报告方式

不要只报平均值，应按序列类型分组：

```text
困难正负混合组
稳定负例组
高风险 BLIP 组
```

如果 gate 主要提升稳定性，可能表现为：

```text
高风险组下降减少；
困难组小幅提升；
稳定组不下降。
```

这比全局平均更有解释力。

---

## 9. 当前不建议做的事情

下一阶段暂时不要做：

```text
1. 复杂 latent language state updater；
2. 继续加深 LMQ query decoder；
3. 训练端到端 gate；
4. 大规模人工标注所有 BLIP 错误类型；
5. 继续大量调 trigger 阈值；
6. 只根据 Bird1 / Dog 反复调参数。
```

原因：

```text
当前阶段的核心是验证 quality gate 是否能稳定减少坏更新；
不是构建完整语言状态系统。
```

---

## 10. 推荐最小实验包

最小可行实验只需要三组：

```text
Experiment 1:
  original deploy vs deploy + quality gate vs oracle gate

Experiment 2:
  gate feature ablation
  score / score+confidence / score+semantic / all

Experiment 3:
  多序列验证
  正负混合、稳定负例、高风险 BLIP
```

如果这三组成立，就可以进入下一阶段：

```text
learnable quality gate
或
latent language state updater
```

---

## 11. 阶段性论文表述建议

英文：

```text
The original dynamic language update trigger mainly detects visual state changes
and provides high-recall candidate language observations.
However, the generated BLIP captions are not always beneficial for tracking.
We therefore introduce a tracking-aware quality gate after candidate generation,
which evaluates whether the candidate language should be adopted.
Preliminary results show that the gate significantly reduces harmful BLIP updates
while preserving a considerable portion of useful updates.
```

中文：

```text
原动态语言更新机制主要判断目标视觉状态是否变化，
适合作为候选语言生成触发器。
但触发后生成的 BLIP 描述质量不稳定，直接采用容易引入错误更新。
因此，我们将动态语言更新拆分为“候选生成触发”和“候选质量判断”两步，
并在触发后加入 tracking-aware quality gate。
初步实验表明，该 gate 能显著降低有害 BLIP 的接受率，
同时保留一部分有益更新。
```

---

## 12. 一句话总结

下一阶段实验应围绕：

```text
验证 deploy-like quality gate 是否能稳定减少错误 BLIP 更新，
并将这种稳定性转化为 score-gap 或 tracking 指标收益。
```

不要继续扩展模块，而是先把 gate 的实际价值验证清楚。
