# Stage 2 Soft Word Reliability 消融结果汇总

日期：2026-05-24

## 结果来源

结果来自：

```text
output/test/visualte_diagnostic_suite/stage2_*/suite_summary.csv
```

当前目录下实际包含 12 组 Stage 2 结果：

```text
A/B/C/D reliability 约束消融
B0/B1/B2/B3 update gate 消融
T1/T2/T3/T4 soft type prior 消融
```

测试序列：

```text
otb_lang:Biker
hoot_balanced20:backpack-004
```

以下表格均为两个序列的均值。

## 总表

| Experiment | IoU | Score GT Mass | Update Rate | Rel Delta | Deploy Rank Corr | Deploy Top3 Overlap | Word HardNeg Gap | Peak Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A anchor baseline | 0.768669 | 0.487510 | 0.000 | 0.000000 | 0.334884 | 0.587000 | 0.042994 | 0.015262 |
| B reliability only | 0.768669 | 0.487521 | 1.000 | 0.004517 | 0.492831 | 0.702323 | 0.043168 | 0.015274 |
| C subject floor | 0.768669 | 0.487503 | 1.000 | 0.004162 | 0.288524 | 0.545434 | 0.042976 | 0.015256 |
| D subject + context | 0.768669 | 0.487529 | 1.000 | 0.003421 | 0.152175 | 0.474053 | 0.043191 | 0.015327 |
| B0 no gate | 0.768669 | 0.487521 | 1.000 | 0.004517 | 0.492831 | 0.702323 | 0.043168 | 0.015274 |
| B1 score peak gate | 0.768669 | 0.487521 | 1.000 | 0.004517 | 0.492831 | 0.702323 | 0.043168 | 0.015274 |
| B2 score gap gate 0.05 | 0.768669 | 0.487521 | 1.000 | 0.004517 | 0.492831 | 0.702323 | 0.043168 | 0.015274 |
| B3 both gate | 0.768669 | 0.487521 | 1.000 | 0.004517 | 0.492831 | 0.702323 | 0.043168 | 0.015274 |
| T1 subject prior 1.05 | 0.768669 | 0.487519 | 1.000 | 0.004517 | 0.433821 | 0.641144 | 0.043139 | 0.015272 |
| T2 subject prior 1.10 | 0.768669 | 0.487517 | 1.000 | 0.004517 | 0.341779 | 0.559664 | 0.043110 | 0.015270 |
| T3 context prior 0.95 | 0.768669 | 0.487530 | 1.000 | 0.004517 | 0.400181 | 0.630942 | 0.043251 | 0.015291 |
| T4 subject 1.05 + context 0.95 | 0.768669 | 0.487527 | 1.000 | 0.004517 | 0.350783 | 0.600989 | 0.043221 | 0.015289 |

## 主要结论

### 1. Reliability only 是当前最稳的 Stage 2 主线

相对 A 组 anchor baseline：

```text
Deploy Rank Corr: 0.334884 -> 0.492831
Deploy Top3 Overlap: 0.587000 -> 0.702323
Word HardNeg Gap: 0.042994 -> 0.043168
IoU: 基本不变
Score GT Mass: 基本不变
```

这说明 soft word reliability 的主要收益不是立即改变最终框，而是让 `word_weight` 排名更接近 `word_gap` 排名。

当前更准确的表述是：

```text
Stage 2 已经改善了词权重与视觉判别性的对齐；
但这种改善还没有明显转化成最终 tracking 指标提升。
```

### 2. Hard subject floor / context cap 当前不适合

C/D 组结果明显下降：

```text
B reliability only Deploy Rank Corr: 0.492831
C subject floor Deploy Rank Corr: 0.288524
D subject + context Deploy Rank Corr: 0.152175
```

Top3 overlap 也下降：

```text
B: 0.702323
C: 0.545434
D: 0.474053
```

结论：

```text
当前 subject 启发式和 context cap 规则还不够可靠。
硬约束会破坏 reliability 从 target-hardneg gap 学到的排序。
```

后续不建议继续把 subject floor/context cap 作为默认方案。

### 3. Update gate 在当前两个序列上没有发挥作用

B0/B1/B2/B3 完全一致：

```text
Update Rate = 1.0
Mean Reliability Delta = 0.004517
Deploy Rank Corr = 0.492831
Deploy Top3 Overlap = 0.702323
```

原因很直接：

```text
当前 Biker 和 HOOT backpack-004 上，
score_peak / score_gap 阈值都没有过滤掉任何帧。
```

因此这轮不能证明 gate 有效，只能说明默认阈值对当前两个序列过松，或者这两个序列没有触发低置信更新场景。

后续要验证 gate，需要：

```text
1. 加困难序列；
2. 或扫更高 score_gap_thr；
3. 或单独统计 score_gap 分布再选阈值。
```

### 4. Soft type prior 当前也不如 reliability only

T1-T4 均低于 B 组：

```text
B reliability only Deploy Rank Corr: 0.492831
T1 subject 1.05: 0.433821
T2 subject 1.10: 0.341779
T3 context 0.95: 0.400181
T4 subject 1.05 + context 0.95: 0.350783
```

其中 T1 损失最小，但仍然低于 B。

结论：

```text
当前词类型先验还不可靠。
即使是 soft prior，也会干扰 target-hardneg gap 学到的自然排序。
```

因此 Stage 2 主线暂时应保留：

```text
reliability only
subject floor = 0
context cap = 1
type prior = 1
```

## 分序列观察

### Biker

```text
A Deploy Rank Corr: 0.363712
B Deploy Rank Corr: 0.525768
C Deploy Rank Corr: 0.399054
D Deploy Rank Corr: 0.199173
```

B 组提升最明显，说明 Biker 上 soft reliability 确实让词权重排序更接近视觉判别性。

### HOOT backpack-004

```text
A Deploy Rank Corr: 0.306056
B Deploy Rank Corr: 0.459894
C Deploy Rank Corr: 0.177994
D Deploy Rank Corr: 0.105178
```

HOOT 上趋势更清楚：hard subject/context 约束破坏更严重，reliability only 更稳。

## 当前推荐配置

下一步主线建议使用：

```bash
--language_word_reliability 1
--language_word_reliability_source target_hardneg_gap
--language_word_reliability_momentum 0.9
--language_word_reliability_tau 0.1
--language_subject_min_reliability 0
--language_context_max_weight 1
--language_subject_type_prior 1.0
--language_attribute_type_prior 1.0
--language_context_type_prior 1.0
```

gate 可保留为 score gap，但这轮没有验证出作用：

```bash
--language_reliability_update_gate 1
--language_reliability_gate_mode score_gap
--language_reliability_score_gap_thr 0.05
```

如果要专门验证 gate，建议单独扫：

```text
score_gap_thr = 0.05 / 0.08 / 0.12 / 0.16
```

并重点看：

```text
language_reliability_update_rate
mean_reliability_delta
word_evidence_deploy_rank_corr_max
word_evidence_deploy_top3_overlap_max
```

## 下一步建议

1. 暂时不要进入 Stage 3 BLIP candidate supplement。

2. 先保留 `reliability only` 作为 Stage 2 主线。

3. 对 gate 做更严格阈值或困难序列测试，否则当前 gate 只是形式上存在。

4. 词类型规则暂时不要参与默认方案。当前 subject/context 的启发式分类会降低 rank alignment。

5. 下一步最有价值的是做词级曲线图：

```text
word_gap curve
word_reliability curve
word_weight vs word_gap rank curve
```

这能解释为什么 reliability only 有 rank alignment 提升，但最终 IoU 尚未变化。

