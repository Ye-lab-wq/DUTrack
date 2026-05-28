# Stage 1 词级视觉证据诊断实现记录

## 目标

本阶段只做诊断，不改变 tracker 的语言输入，不启用新的语言更新策略。

目的：验证每个语言 token 是否真的能在 search 区域上区分目标区域和 hard negative 区域。

## 核心公式

对第 `i` 个词、第 `t` 帧、第 `j` 个 search token：

```text
R_i^t(j) = sim(word_i, search_token_j)
```

定义目标 token 集合 `P_t` 和 hard negative token 集合 `N_t`。

本阶段同时记录两种 `P_t`：

```text
oracle mode:
  P_t = GT box 覆盖的 search tokens
  用于判断语言-视觉打分机制的上限。

deploy-like mode:
  P_t = 当前预测框覆盖的 search tokens
  用于模拟真实推理阶段，因为推理时没有 GT。
```

公式：

```text
s_pos(i,t) = mean_{j in P_t} R_i^t(j)
s_neg(i,t) = mean_{j in N_t} R_i^t(j)
gap(i,t) = s_pos(i,t) - s_neg(i,t)
visual_evidence(i,t) = sigmoid(gap(i,t) / tau)
```

解释：

```text
gap > 0: 该词更支持目标区域
gap = 0: 该词区分性弱
gap < 0: 该词更支持 hard negative 或背景
```

## 代码改动

### 1. 导出 per-word token response

文件：

```text
lib/models/dutrack/language_token_emphasizer.py
lib/models/dutrack/itpn.py
```

新增 aux 字段：

```text
word_level_template_token_scores  # shape: [B, template_tokens, language_tokens]
word_level_search_token_scores    # shape: [B, search_tokens, language_tokens]
```

这两个字段来自 word-level language-to-visual similarity，不改变 keep、score prior 或 tracking 输出。

### 2. 诊断脚本记录 per-word evidence

文件：

```text
tracking/visualte_diagnostic.py
```

新增输出：

```text
word_reliability_diagnostics.csv
```

字段包括：

```text
frame
sequence
layer
evidence_mode
word_index
word
word_type
word_is_content
word_is_subject_candidate
word_is_anchor_subject
word_weight
word_rank_by_gap
word_rank_by_weight
word_pos_score
word_hardneg_score
word_out_score
word_gap
word_visual_evidence
word_hard_case
pos_token_count
neg_token_count
hardneg_count
```

其中：

```text
evidence_mode = oracle:
  正样本区域来自 GT box。

evidence_mode = deploy:
  正样本区域来自当前预测框。
```

hard negative 的定义沿用现有诊断逻辑：

```text
GT 外部或目标区域外部
+ 当前 score_map 最高的 top-k tokens
```

### 3. diagnostics.csv 增加帧级摘要

每个 TE 层新增：

```text
word_evidence_oracle_L{layer}_mean_gap
word_evidence_oracle_L{layer}_max_gap
word_evidence_oracle_L{layer}_min_gap
word_evidence_oracle_L{layer}_mean_visual_evidence
word_evidence_oracle_L{layer}_hard_case_ratio
word_evidence_oracle_L{layer}_content_count
word_evidence_oracle_L{layer}_subject_gap_mean
word_evidence_oracle_L{layer}_attribute_gap_mean
word_evidence_oracle_L{layer}_context_gap_mean
word_evidence_oracle_L{layer}_best_word_gap_mean
word_evidence_oracle_L{layer}_anchor_subject_gap_mean
word_evidence_oracle_L{layer}_content_word_positive_ratio

word_evidence_deploy_L{layer}_...
```

`oracle` 和 `deploy` 字段完全同构。

### 4. suite 汇总增加跨序列摘要

文件：

```text
tracking/visualte_diagnostic_suite.py
```

新增 suite 字段：

```text
word_evidence_oracle_mean_gap_max
word_evidence_oracle_hard_case_ratio_min
word_evidence_oracle_subject_gap_mean_max
word_evidence_oracle_anchor_subject_gap_mean_max
word_evidence_oracle_positive_ratio_max

word_evidence_deploy_mean_gap_max
word_evidence_deploy_hard_case_ratio_min
word_evidence_deploy_subject_gap_mean_max
word_evidence_deploy_anchor_subject_gap_mean_max
word_evidence_deploy_positive_ratio_max
```

## 新增参数

```bash
--word_evidence_tau 0.1
```

用于：

```text
visual_evidence = sigmoid(word_gap / tau)
```

默认值为 `0.1`。

建议后续做敏感性检查：

```text
tau = 0.05 / 0.1 / 0.2
```

## 使用建议

第一轮建议固定语言，不开启 hard filtering：

```bash
python tracking/visualte_diagnostic_suite.py \
  --config dutrack_384_full_lte_keepvl_worddirect_71523_l15prior_stagehead_e10 \
  --runid 10 \
  --stat_frames 0 \
  --vis_frames 5 \
  --top_ratio 0.1 \
  --hardneg_topk 6 \
  --word_evidence_tau 0.1 \
  --score_prior_source word_direct_margin \
  --language_init_source dataset_or_class \
  --language_update_mode anchor \
  --original_view auto \
  --output_tag word_evidence_stage1_anchor_fullstats
```

## 判断标准

如果可行，应看到：

```text
word_evidence_oracle_mean_gap_max > 0
word_evidence_deploy_mean_gap_max > 0
word_evidence_*_hard_case_ratio_min 降低
主体词 / 关键属性词的 word_gap 稳定为正
上下文词或背景词的 word_gap 不应长期高于主体词
```

如果大量词 `word_gap < 0`，说明语言词响应本身仍然更偏向 hard negative，下一步不应急着做语言更新，而应继续修 language-to-visual scoring 或词角色建模。

最关键的判断：

```text
如果 word_rank_by_weight 靠前的词，不等于 word_rank_by_gap 靠前的词，
说明当前 word weighting 机制不能直接用于 reliability 更新。
```

原因：

```text
word_weight 高只能说明模型当前更依赖这个词；
word_gap 高才说明该词有 target-hardneg 判别性。
```

如果两者不一致，后续 soft reliability 应从 `word_gap` 更新，而不是从 `word_level_weights` 更新。
