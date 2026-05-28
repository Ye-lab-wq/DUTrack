# S0 Source/State Probe 实现记录

## 实现目的

S0 是只读诊断脚本，不改模型权重，不训练 `CandidateAdapter`。它用于回答：

1. 当前 BLIP/current caption 是否提供了比 anchor 更好的语言 prior；
2. 如果允许 oracle 选择语言源，prior gap 是否能提升；
3. 这种提升是否可能成为后续 `Language State Updater` 的学习信号。

默认模式是 `deploy_like`：复用原项目语言更新触发条件，只有 `tracker.updata_key=True` 时才调用 BLIP。每帧调用 BLIP 的 `oracle_blip` 只作为上界诊断，不作为默认实验。

## 新增脚本

`tracking/language_state_s0_probe.py`

每一帧执行：

```text
anchor_description = tracker.language_anchor
deploy_trigger     = tracker.updata_key
prev_description   = previous selected state description

if candidate_mode == "deploy_like":
    blip_description = BLIP(image) only when deploy_trigger is true
elif candidate_mode == "oracle_blip":
    blip_description = BLIP(image) every frame

for source in available(anchor, blip, prev):
    forward(template_memory, search, source_description)
    collect lmq_prior_scores
    collect score_map / predicted box

oracle_source = argmax(pos_hardneg_gap(anchor, blip, prev))
if oracle_state_update:
    prev_description = oracle_source_description

tracker.track(image)  # 正常推进 tracker 状态
```

脚本会临时保存并恢复 `network.track_query`，避免多次 probe forward 污染 tracker 的真实递推状态。

默认关闭 score prior 的 score-map 注入，避免 prior bias 反过来污染 source-quality 判断：

```text
tracker.network.score_prior_enabled = False
```

如果需要观察 prior 已经接入 score 后的效果，可以加：

```text
--use_score_prior_effect
```

默认主评价信号使用 `score_map`：

```text
--evidence_source score
```

这表示 S0 问的是：同一个 tracker 状态、同一批模板和 search crop 下，换成 anchor / BLIP / prev-state 语言后，原 DUTrack center score 是否更能区分目标和 hard negative。

如果使用：

```text
--evidence_source lmq_prior
```

则问的是：当前 LMQ prior 模块能否给不同语言源产生更好的 prior。这个结果不能直接解释为语言质量，因为它混入了 LMQ prior 的建模能力。

## 输出文件

输出目录：

```text
output/test/language_state_s0_probe/<output_tag>/<sequence>/
```

主要文件：

- `stage3_s0_probe.csv`
- `stage3_s0_summary.md`

## 关键字段

语言源字段：

- `anchor_description`
- `blip_description`
- `prev_description`
- `oracle_source`
- `candidate_mode`
- `deploy_trigger`
- `candidate_available`

主评价信号诊断：

- `anchor_pos_hardneg_gap`
- `blip_pos_hardneg_gap`
- `prev_pos_hardneg_gap`
- `oracle_gap`
- `oracle_gain_over_prev`

辅助字段会同时导出：

- `anchor_score_*`, `blip_score_*`, `prev_score_*`
- `anchor_lmq_*`, `blip_lmq_*`, `prev_lmq_*`

summary 中不带 `_score/_lmq` 的字段是当前 `--evidence_source` 选中的主判断信号。

语言增量判断：

- `blip_better_anchor`
- `blip_better_prev`
- `blip_hurts`
- `oracle_update`
- `oracle_trigger_observable`
- `oracle_trigger`
- `deploy_oracle_agree`
- `deploy_false_positive`
- `deploy_missed_oracle`

跟踪输出参考：

- `anchor_iou`
- `blip_iou`
- `prev_iou`

## 推荐运行指令

### 单序列 Probe

OTB/Biker，deploy-like 默认模式：

```bash
python tracking/language_state_s0_probe.py \
  --config dutrack_384_full_lmq_d1_e10 \
  --dataset_name otb_lang \
  --sequence Biker \
  --runid 1 \
  --max_frames 5 \
  --candidate_mode deploy_like \
  --evidence_source score \
  --output_tag stage3_s0_biker_lmq_d1_ep1
```

为了避免和旧的 LMQ-prior 诊断混淆，建议新结果命名显式带 `score`：

```bash
python tracking/language_state_s0_probe.py \
  --config dutrack_384_full_lmq_d1_e10 \
  --dataset_name otb_lang \
  --sequence Biker \
  --runid 1 \
  --max_frames 5 \
  --candidate_mode deploy_like \
  --evidence_source score \
  --output_tag stage3_s0_biker_score_deploy
```

OTB/Biker，oracle 上界模式：

```bash
python tracking/language_state_s0_probe.py \
  --config dutrack_384_full_lmq_d1_e10 \
  --dataset_name otb_lang \
  --sequence Biker \
  --runid 1 \
  --max_frames 5 \
  --candidate_mode oracle_blip \
  --evidence_source score \
  --output_tag stage3_s0_biker_score_oracle
```

只诊断 LMQ prior 模块：

```bash
python tracking/language_state_s0_probe.py \
  --config dutrack_384_full_lmq_d1_e10 \
  --dataset_name otb_lang \
  --sequence Biker \
  --runid 1 \
  --max_frames 5 \
  --candidate_mode oracle_blip \
  --evidence_source lmq_prior \
  --output_tag stage3_s0_biker_lmqprior_oracle
```

HOOT 示例：

```bash
python tracking/language_state_s0_probe.py \
  --config dutrack_384_full_lmq_d1_e10 \
  --dataset_name hoot_balanced20 \
  --sequence 0 \
  --runid 1 \
  --max_frames 5 \
  --candidate_mode deploy_like \
  --evidence_source score \
  --output_tag stage3_s0_hoot_lmq_d1_ep1
```

如果当前 shell 没有激活 `DUTrack` 环境，使用：

```bash
conda run -n DUTrack python tracking/language_state_s0_probe.py \
  --config dutrack_384_full_lmq_d1_e10 \
  --dataset_name otb_lang \
  --sequence Biker \
  --runid 1 \
  --max_frames 5 \
  --candidate_mode deploy_like \
  --evidence_source score \
  --output_tag stage3_s0_biker_lmq_d1_ep1
```

## 结果判读

如果 `candidate_available` 很低，说明原始更新触发很保守，默认 deploy 路径很少真正使用 BLIP。

如果 `candidate_available` 不低，但 `blip_better_anchor` 和 `blip_better_prev` 很低，说明 current caption 本身没有稳定增量，不应该直接训练 `H_candidate = CandidateAdapter(H_blip, H_anchor)`。

如果 `oracle_gain_over_prev` 为正但 `blip_hurts` 也高，说明语言状态更新需要强门控，不能每帧更新。

如果 `oracle_blip` 模式下 `deploy_missed_oracle` 高，说明原始触发条件错过了一些有用语言更新；如果 `deploy_false_positive` 高，说明原始触发条件经常触发无用或有害更新。

如果 `evidence_source=score` 下 BLIP 有提升，而 `evidence_source=lmq_prior` 下没有提升，说明语言源本身可能有用，但当前 LMQ prior generator 没把它转成有效 prior。

如果两者都没有提升，才更接近“当前语言源没有稳定增量”的结论。

如果 `evidence_source=lmq_prior` 有提升，但 `evidence_source=score` 没提升，说明 LMQ prior 自己的响应不一定能服务最终 center score。

## 多序列 Screening

新增脚本：

`tracking/language_state_s0_screen.py`

它会循环调用单序列 S0 probe，对每个序列汇总：

- `baseline_mean_iou`
- `score_gap_mean`
- `score_gap_low_ratio`
- `deploy_trigger_rate`
- `trigger_by_position_rate`
- `trigger_by_scale_rate`
- `trigger_by_color_rate`
- `trigger_color_delta_mean`
- `BLIP_available_rate`
- `anchor_score_gap`
- `blip_score_gap`
- `oracle_score_gain`
- `BLIP_better_anchor_ratio`
- `BLIP_hurts_ratio`

Part B quality gate 诊断还会汇总：

- `quality_gate_accept_rate`
- `quality_gate_gain`
- `quality_gate_true_accept_rate`
- `quality_gate_false_reject_rate`
- `quality_gate_true_reject_rate`
- `quality_gate_false_accept_rate`
- `useful_update_recall`
- `hurt_rejection_rate`

并按三类自动筛选：

- A: `baseline_mean_iou` 在 0.3 到 0.7，表示 baseline 中等困难；
- B: `oracle_score_gain > 0` 且 `BLIP_better_anchor_ratio` 足够高，表示语言有潜在增量；
- C: `BLIP_hurts_ratio` 高，表示语言容易误伤。

先小规模试跑：

```bash
python tracking/language_state_s0_screen.py \
  --config dutrack_384_full_lmq_d1_e10 \
  --dataset_names otb_lang,hoot_balanced20 \
  --runid 1 \
  --max_frames 10 \
  --max_sequences 3 \
  --candidate_mode oracle_blip \
  --evidence_source score \
  --output_tag stage3_s0_screen_smoke
```

完整筛选 OTB-Lang + HOOT：

```bash
python tracking/language_state_s0_screen.py \
  --config dutrack_384_full_lmq_d1_e10 \
  --dataset_names otb_lang,hoot_balanced20 \
  --runid 1 \
  --max_frames 20 \
  --candidate_mode oracle_blip \
  --evidence_source score \
  --output_tag stage3_s0_screen_otb_hoot_score_oracle
```

如果要检查原始触发条件实际会不会调用 BLIP：

```bash
python tracking/language_state_s0_screen.py \
  --config dutrack_384_full_lmq_d1_e10 \
  --dataset_names otb_lang,hoot_balanced20 \
  --runid 1 \
  --max_frames 20 \
  --candidate_mode deploy_like \
  --evidence_source score \
  --state_update_policy gate \
  --output_tag stage3_s0_screen_otb_hoot_score_deploy
```

输出目录：

```text
output/test/language_state_s0_screen/<output_tag>/
```

核心文件：

- `s0_screen_summary.csv`
- `s0_screen_summary.md`

注意：`oracle_blip` 会每帧调用 BLIP，适合筛选优化空间，但计算较慢；先用 `--max_sequences` 小规模试跑更稳。

`--max_frames` 表示初始化帧之后最多评估多少帧。使用 `--max_frames 0` 表示跑完整序列：

```bash
python tracking/language_state_s0_screen.py \
  --config dutrack_384_full_lmq_d1_e10 \
  --dataset_names otb_lang,hoot_balanced20 \
  --runid 1 \
  --max_frames 0 \
  --candidate_mode oracle_blip \
  --evidence_source score \
  --output_tag stage3_s0_screen_otb_hoot_score_oracle_full
```

完整序列的 `oracle_blip` 会非常慢，因为它会对每帧调用 BLIP。更实际的组合是先跑完整序列的 `deploy_like`，再只对筛出的重点序列跑 `oracle_blip`。

## 当前限制

S0 使用 source-level text state 近似 latent state，不是真正的 `H_t` token 状态。

S0 的 oracle update 依赖 GT 区域计算 gap，只能做诊断上界，不能作为部署机制。

S0 不训练 CandidateAdapter，也不解决 BLIP caption 质量问题。
