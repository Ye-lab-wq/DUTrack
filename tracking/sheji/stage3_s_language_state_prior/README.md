# Stage 3-S Language State Prior

## 目标

本阶段不再继续堆叠人工规则式的 word reliability / source routing，而是把语言更新问题拆成两部分：

1. 语言状态是否有必要更新；
2. 更新后的语言状态能否通过 prior 影响 center score。

这里的语言状态 `H_t` 不是一句可读文本，而是一组 latent language tokens / embeddings。它由初始语言、当前 BLIP/current caption 和视觉证据共同决定，用于后续 `Language-to-Score Prior Generator`。

完整链路：

```text
H_anchor, H_{t-1}, H_blip_t, X_search, S_base
        -> Language State Updater
        -> H_t
        -> PriorGenerator
        -> P_t
        -> S_final = S_base + bounded_bias(P_t)
```

## 必须先证明的三件事

### 1. BLIP/current 是否有可用增量

如果当前 caption 相比 anchor / previous state 没有更好的 target-hard negative gap，那么 `CandidateAdapter(H_blip, H_anchor)` 没有稳定信息源，只会学习到噪声融合。

S0 先不训练，只比较不同语言源对同一帧同一 tracker 状态的影响。默认不每帧调用 BLIP，而是复用原 DUTrack 的语言更新触发条件：

```text
deploy_trigger = tracker.updata_key
```

也就是上一帧 `ifupdata(his_state, state, H, W)` 判断需要更新时，当前帧才生成 BLIP/current candidate。

默认 `deploy_like` 模式比较：

- `anchor`: 初始语言；
- `prev`: 上一帧语言状态文本近似。
- `blip`: 仅当原始触发条件为真时才生成的当前帧 BLIP/current caption。

如果需要完整 oracle 上界，才使用 `oracle_blip` 模式，每帧调用 BLIP。

核心指标：

- `blip_pos_hardneg_gap`
- `anchor_pos_hardneg_gap`
- `prev_pos_hardneg_gap`
- `blip_better_anchor`
- `blip_better_prev`
- `blip_hurts`
- `deploy_trigger`
- `candidate_available`

S0 默认主评价信号使用 `score_map`，不是 `lmq_prior_scores`。原因是当前问题是“语言状态是否值得更新”，不能把已经验证不稳定的 LMQ prior 当成语言质量评估器。

```text
--evidence_source score      # 默认，判断语言源对 center score 的影响
--evidence_source lmq_prior  # 只诊断 LMQ prior 模块本身
```

脚本仍会同时导出 `*_score_*` 和 `*_lmq_*` 辅助字段，但 summary 里的 `anchor gap / blip gap / prev-state gap` 取决于 `--evidence_source`。

### 2. 状态更新能否提高 prior gap 且不漂移

状态更新不能只看“当前 BLIP 偶尔更好”，还要看更新后是否会破坏上一帧稳定状态。

S0 使用 source-level oracle 做上界诊断：

```text
oracle_source = argmax(anchor_gap, blip_gap, prev_gap)
oracle_gain_over_prev = oracle_gap - prev_gap
```

这只能回答“有没有可更新空间”，还不能证明真实 `Language State Updater` 能学会这个决策。后续 S1 需要引入 learnable update gate，并记录 state drift。

关于触发条件：

- `deploy_like`: 只能判断原始触发后生成的 BLIP 是否有用，不能判断未触发帧是否漏掉了有用 BLIP。
- `oracle_blip`: 每帧生成 BLIP，可以计算 `deploy_missed_oracle` 和 `deploy_false_positive`，但它是昂贵上界诊断，不适合作为默认实验。

### 3. PriorGenerator 能否把 H_t 的改进传到 center score

S0 默认检查 score-map gap，因此更适合判断语言源是否有用；如果使用 `--evidence_source lmq_prior`，则只是在检查当前 LMQ prior 的质量。

真正要证明传递有效，需要后续比较：

- prior off / on 的 `score_peak_delta`
- `score_gt_mass_delta`
- `score_pos_hardneg_gap_delta`
- 最终 IoU / tracking loss 变化

所以 S0 的定位是：先证明语言状态候选里是否有新信息，而不是证明完整端到端方案已经成立。

## 阶段规划

| Stage | 目的 | 是否训练 |
| --- | --- | --- |
| S0 | 语言源与 oracle state update 诊断 | 否 |
| S1 | learnable language state updater，小门控更新 | 是 |
| S2 | state prior on/off，验证是否传到 center score | 是 |
| S3 | 多序列稳定性与语言漂移分析 | 否/评测 |

## S0 Screening

单序列 Biker 不足以判断语言状态更新是否有优化空间。新增筛选脚本会对 OTB-Lang / HOOT 逐序列统计：

- `baseline_mean_iou`
- `score_gap_mean`
- `score_gap_low_ratio`
- `deploy_trigger_rate`
- `BLIP_available_rate`
- `anchor_score_gap`
- `blip_score_gap`
- `oracle_score_gain`
- `BLIP_better_anchor_ratio`
- `BLIP_hurts_ratio`

并自动打三类标签：

- A: baseline 中等困难，`baseline_mean_iou` 在 0.3 到 0.7；
- B: 语言有潜在增量，`oracle_score_gain > 0` 且 `BLIP_better_anchor_ratio` 较高；
- C: 语言容易误伤，`BLIP_hurts_ratio` 较高。

筛选默认使用 `candidate_mode=oracle_blip` 和 `evidence_source=score`，因为目标是找“可能有优化空间”的序列；如果要评估原始触发条件，再改成 `candidate_mode=deploy_like`。

## 当前新增文件

- `tracking/language_state_s0_probe.py`
- `tracking/language_state_s0_screen.py`
- `tracking/sheji/stage3_s_language_state_prior/s0_probe_implementation.md`
