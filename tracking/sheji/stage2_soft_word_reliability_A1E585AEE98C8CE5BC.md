# Stage 2 补充实验指导：Soft Word Reliability

## 1. 当前阶段定位

Stage 2 的核心目标不是直接追求 IoU 大幅提升，而是验证：

> 在固定 anchor language 不变的前提下，基于 target-hardneg gap 的 soft word reliability 是否能让词权重更接近真实视觉判别性。

当前 A/B/C/D 消融的阶段性结论是：

```text
A. anchor baseline:
   稳定基准，语言不变化。

B. reliability only:
   当前最优，显著提升 word weight 与 word gap 的对齐。

C. reliability + subject floor:
   rank alignment 下降，说明硬性主体保底当前不稳。

D. reliability + subject floor + context cap:
   rank alignment 进一步下降，说明硬性 context cap 当前不适合。
```

因此，下一步不建议直接进入 Stage 3 的 BLIP candidate supplement，而应先补扎实 Stage 2。

---

## 2. 当前最推荐主线

当前 Stage 2 主线应以 B 组为基础：

```text
固定 anchor 文本
不删词
不改写句子
不加入 BLIP 新词
不使用 hard subject floor
不使用 hard context cap
只用 deploy target-hardneg gap 更新 soft word reliability
```

即：

```text
word_gap_i(t) = mean_pos(word_i) - mean_hardneg(word_i)

e_i(t) = sigmoid(word_gap_i(t) / tau)

r_i(t) = momentum * r_i(t-1) + (1 - momentum) * e_i(t)

final_word_weight_i ∝ base_word_weight_i * r_i(t)
```

本阶段重点证明：

```text
word_rank_by_weight 更接近 word_rank_by_gap
```

而不是只看最终 IoU。

---

## 3. 必须补充的实验一：Update Gate 消融

### 3.1 为什么要做

Stage 2 使用当前预测框 `P_pred` 作为正样本区域，因此存在自举误差：

```text
预测框准确 → reliability 更新可信
预测框偏移 → reliability 可能被错误更新
```

因此需要验证 update gate 是否能降低错误更新。

### 3.2 推荐实验组

在 B 组 reliability only 基础上做：

```text
B0: no gate
B1: score_peak gate
B2: score_gap gate
B3: both gate
```

含义：

```text
no gate:
  每帧都更新 reliability。

score_peak gate:
  只有 score_peak 足够高时更新。

score_gap gate:
  只有 score_peak - hardneg_peak 足够大时更新。

both gate:
  同时满足 score_peak 和 score_gap 条件才更新。
```

### 3.3 推荐参数

```yaml
LANGUAGE_RELIABILITY_UPDATE_GATE: True
LANGUAGE_RELIABILITY_GATE_MODE: score_gap
LANGUAGE_RELIABILITY_SCORE_THR: 0.4
LANGUAGE_RELIABILITY_SCORE_GAP_THR: 0.05
```

建议测试：

```text
score_gap_thr = 0.03 / 0.05 / 0.08
```

### 3.4 重点指标

```text
update_skip_ratio
reliability_std
weight_gap_rank_corr
top3_weight_gap_overlap
word_hardneg_gap
Score GT Mass
On/Off Peak Delta
IoU
```

### 3.5 判断标准

如果 `score_gap gate` 能做到：

```text
weight_gap_rank_corr 不下降或提升
top3_weight_gap_overlap 不下降或提升
reliability 曲线更平滑
IoU / Score GT Mass 不下降
```

则保留 score_gap gate。

如果 gate 后 rank alignment 明显下降，说明门控过严或阈值不合适。

---

## 4. 必须补充的实验二：Soft Type Prior 消融

### 4.1 为什么要做

A/B/C/D 已经显示：

```text
hard subject floor 和 hard context cap 会破坏词权重排序。
```

因此不建议继续使用：

```text
subject reliability 强制不低于 0.7
context weight 强制不高于 0.4
```

但词类型信息仍然有价值。更合理的做法是使用轻微软先验，而不是硬约束。

### 4.2 推荐形式

```text
final_word_weight_i ∝ base_weight_i * reliability_i * type_prior_i
```

其中：

```text
subject_prior   = 1.05 或 1.10
attribute_prior = 1.00
context_prior   = 0.90 或 0.95
```

注意：

```text
type_prior 只能轻微影响权重；
最终排序仍应主要由 word_gap / reliability 决定。
```

### 4.3 推荐实验组

```text
T0: reliability only
T1: subject prior 1.05
T2: subject prior 1.10
T3: context prior 0.95
T4: subject prior 1.05 + context prior 0.95
```

不建议第一轮使用过强参数，例如：

```text
subject_prior > 1.2
context_prior < 0.8
```

### 4.4 重点指标

```text
weight_gap_rank_corr
top3_weight_gap_overlap
subject_rank_by_gap
subject_rank_by_weight
best_gap_word
context_gap_mean
context_weight_mean
Score GT Mass
IoU
```

### 4.5 判断标准

如果 soft prior 能保持 B 组的 rank alignment，同时改善 subject/context 的合理性，则可以保留。

如果 soft prior 仍然降低 `weight_gap_rank_corr`，说明当前词类型规则不可靠，暂时不应加入类型约束。

---

## 5. 必须补充的实验三：词级日志与可视化

### 5.1 为什么要做

Stage 2 的核心不是语言字符串变化，而是内部词权重变化。因此必须记录：

```text
哪些词被升权？
哪些词被降权？
升降权是否有 target-hardneg gap 支持？
```

否则很难解释方法是否真的合理。

### 5.2 建议记录字段

每帧记录：

```text
frame_id
anchor_description
subject_candidate_word
best_gap_word

top3_weight_words
top3_gap_words
weight_gap_rank_corr
top3_weight_gap_overlap

update_gate_pass
score_peak
hardneg_peak
score_gap

word
word_type
base_word_weight
word_reliability
final_word_weight
word_pos_score
word_hardneg_score
word_gap
word_rank_by_weight
word_rank_by_gap
```

### 5.3 建议可视化

至少输出三类图：

```text
1. word_gap 曲线
   看每个词是否长期支持目标区域。

2. word_reliability 曲线
   看 reliability 是否平滑，有无剧烈震荡。

3. word_weight vs word_gap 排名对比图
   看权重最高的词是否也是最有判别性的词。
```

### 5.4 建议重点观察词

Biker：

```text
head / man / bike / track
```

HOOT backpack-004：

```text
red / backpack / held / person
```

---

## 6. 当前不建议继续的方向

### 6.1 不建议直接进入 Stage 3

Stage 3 会引入 BLIP 候选词，变量更多：

```text
BLIP 何时触发
BLIP 生成哪些词
候选词如何分类
候选词如何初始化 reliability
候选词是否替换或补充 anchor
候选词错误时如何回退
```

如果 Stage 2 的 soft reliability 还没稳定，直接进入 Stage 3 会导致实验难以解释。

### 6.2 不建议继续使用 hard filtering

hard filtering 会改变文本字符串，并导致语言状态频繁变化：

```text
anchor 语言稳定性被破坏
词被删除后恢复困难
filtered description 反复切换
```

Stage 2 应坚持：

```text
文本不变
词权重变
```

### 6.3 不建议继续使用 hard floor / hard cap

A/B/C/D 已经显示，硬性 subject floor 和 context cap 当前会破坏 rank alignment。

后续应改成：

```text
soft type prior
```

而不是：

```text
hard subject floor / hard context cap
```

---

## 7. 推荐补充实验顺序

### Step 1：确认 B 组主线

重新跑或整理 B 组：

```text
reliability only
score_gap gate 默认关闭或开启两版
```

确认它相对 A 组稳定提升：

```text
weight_gap_rank_corr
top3_weight_gap_overlap
```

### Step 2：Update Gate 消融

跑：

```text
no gate
score_peak gate
score_gap gate
both gate
```

选择最稳的 gate。

### Step 3：Soft Type Prior 消融

在最佳 gate 上测试：

```text
subject prior 1.05 / 1.10
context prior 0.95
subject + context soft prior
```

### Step 4：词级曲线可视化

输出 Biker 和 HOOT 的：

```text
word_gap curve
word_reliability curve
word_weight curve
top3 weight/gap 对比
```

### Step 5：形成 Stage 2 结论

如果结果支持：

```text
soft reliability 稳定提升 rank alignment
gate 能降低错误更新
soft type prior 不破坏 rank alignment
```

则进入 Stage 3。

否则继续修 Stage 2，不进入 BLIP 候选词阶段。

---

## 8. 成功标准

Stage 2 补充实验成功不一定要求 IoU 明显提升。

更合理的成功标准是：

```text
Lang Changes = 0
Unique Lang = 1
weight_gap_rank_corr 提升
top3_weight_gap_overlap 提升
reliability 曲线平滑
word_hardneg_gap 不下降
Score GT Mass / On-Off Peak Delta 不下降
IoU 至少不明显下降
```

如果这些成立，说明 Stage 2 的语言调制是稳定且可解释的。

---

## 9. 阶段性结论模板

可以在汇报中这样总结：

```text
Stage 2 的目标不是生成新语言，而是在固定 identity anchor 的基础上，
根据每个词对目标区域和 hard negative 的区分能力，动态调整词权重。

当前 A/B/C/D 消融表明：
soft reliability 本身可以提升词权重与视觉判别性的对齐；
但 hard subject floor 和 hard context cap 会破坏这种对齐。

因此下一步不直接进入 BLIP 更新，
而是补充 update gate、soft type prior 和词级可视化，
先把固定词集合下的语言调制做扎实。
```

---

## 10. 最终建议

当前 Stage 2 的补充实验应围绕三点展开：

```text
1. 用 update gate 降低自举误差；
2. 用 soft type prior 替代 hard floor/cap；
3. 用词级日志和可视化证明 reliability 更新确实合理。
```

完成这些之后，再进入 Stage 3 的 BLIP 候选词补充会更稳。
