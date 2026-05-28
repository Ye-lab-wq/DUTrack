# Stage 3 Quality Gate Feature Probe

更新时间：2026-05-26

## 目的

该实验不是最终在线 gate，也不是正式训练策略。

它只回答一个诊断问题：

```text
当前可部署特征是否存在跨序列可学习判别边界，
能否比现有 deploy score-delta gate 更好地区分 useful / harmful BLIP？
```

## 输入

读取已有逐帧诊断表：

```text
output/test/language_state_s0_screen/stage3_quality_gate_otb_hoot_error_report/s0_error_report.csv
```

默认小集合：

```text
OTB: Bird1, Dog, Gym
HOOT: potted_plant-008, toilet_paper-001, koala-003
```

## 标签

离线标签只用于诊断：

```text
y = 1  if quality_gate_score_delta > 0
y = 0  if quality_gate_score_delta < 0
```

其中 `quality_gate_score_delta` 是 oracle/GT 区域下的 BLIP 相对 prev 的 score-gap 增益。
该标签不能用于部署，只用于判断 feature separability。

## 特征

只使用推理时可获得或可近似获得的特征，不使用 GT IoU / oracle gap 作为输入。

基础特征：

```text
quality_gate_deploy_score_delta
quality_gate_semantic
score_peak
score_peak_second_gap
pred_box_jump_ratio
trigger reason
anchor/prev/blip score gap
prev/blip deploy gap
```

文本统计特征：

```text
anchor_blip_content_overlap
prev_blip_content_overlap
anchor_blip_content_jaccard
prev_blip_content_jaccard
blip_content_word_count
blip_generic_ratio
blip_context_ratio
blip_anchor_missing_ratio
blip_prev_missing_ratio
```

可选词级视觉证据（需要先用 `--word_evidence` 重跑 S0 screen）：

```text
blip_word_target_template_gap_mean
blip_word_context_template_gap_mean
blip_word_new_template_gap_mean
blip_word_target_search_deploy_gap_mean
blip_word_context_search_deploy_gap_mean
blip_word_new_search_deploy_gap_mean
blip_word_target_minus_context_template_gap
blip_word_target_minus_context_search_deploy_gap
blip_minus_prev_target_template_gap
blip_minus_prev_target_search_deploy_gap
```

其中 template gap 使用 DUTrack memory template 的目标 mask 作为稳定参考；
search deploy gap 使用当前预测框 mask，仅作为对照，不能单独作为可靠证据。

## 方法

使用 leave-one-sequence-out：

```text
每次留出一个序列测试，
其他小集合序列训练一个 logistic gate。
```

模型非常轻：

```text
Linear(features -> accept_logit)
weighted BCE
```

## 输出

```text
output/test/language_state_s0_gate_probe/<tag>/
  loso_metrics.csv
  loso_predictions.csv
  feature_probe_summary.md
```

## 判断标准

如果 learned gate 相比 current gate：

```text
false_accept / harmful 下降；
useful_recall 不明显下降；
mean_gate_gain 不下降；
不同留出序列方向一致；
```

则说明这些可部署特征具有进一步做在线 learnable quality gate 的价值。

如果 LOSO 不稳定或只在某个序列上好，说明目前特征泛化不足，不应进入在线模块。

## Anchor-Preserving State Update 诊断

为验证 hard replacement 是否是主要风险，新增一个诊断策略：

```text
state_update_policy = anchor_state_gate
```

原 hard replacement：

```text
gate accept -> prev_description = blip_description
```

anchor-preserving 版本：

```text
gate accept -> prev_description = compact(anchor_description + BLIP state words)
```

其中 anchor 始终保留主体，BLIP 只补充不在 anchor 中的 content words，最多保留 6 个状态词。
该策略不是最终 latent language state updater，只用于比较：

```text
保留 anchor 主体是否能降低错误 BLIP 对后续语言状态的污染。
```

新增诊断字段：

```text
anchor_state_candidate_description
anchor_blip_content_overlap_count
prev_blip_content_overlap_count
```
