# Stage 2 Soft Word Reliability 实现记录

## 目标

Stage 2 不再删除词、不改写句子，而是在保持 anchor language 不变的前提下，动态调整每个词进入 word-level prior 的权重。

核心区别：

```text
Stage 1:
  只记录 per-word target-hardneg gap，不改变模型行为。

Stage 2:
  用上一帧 deploy-like target-hardneg gap 更新 word reliability，
  下一帧 forward 时将 reliability 乘进 word-level prior 的词权重。
```

## 数据流

```text
anchor description
  -> tokenizer 得到固定 word tokens
  -> 初始化 reliability: content words = 1.0, special tokens = 0
  -> 第 t 帧 forward 时传入 reliability
  -> LanguageGuidedTokenEmphasizer 内部:
       word_weights = softmax(word_logits)
       word_weights = normalize(word_weights * reliability)
       word_direct_score = Σ_i word_weights_i * sim(word_i, search_token_j)
  -> 跟踪输出 score_map / pred_box
  -> 用 pred_box 区域作为 deploy-like positive tokens
  -> 用 score_map 高分框外 token 作为 hard negative
  -> 更新 reliability，供第 t+1 帧使用
```

注意：本阶段的更新不依赖 GT，因此可以部署；GT oracle 只保留在诊断 CSV 里做上限分析。

## 可靠性公式

对词 `i`：

```text
gap_i(t) = mean_{j in P_pred} R_i(j) - mean_{j in N_hard} R_i(j)
e_i(t) = sigmoid(gap_i(t) / tau)
r_i(t) = momentum * r_i(t-1) + (1 - momentum) * e_i(t)
```

其中：

```text
P_pred:
  当前预测框覆盖的 search tokens

N_hard:
  预测框外 score_map 最高的 top-k tokens
```

### 更新安全门

当前帧置信度不足时不更新 reliability，避免错误预测框把语言状态带偏。

单独使用 `max(score_map)` 对不同序列不一定稳，因为有些困难场景整体 score 偏低，但目标和 hard negative 仍然可分。因此默认门控优先使用：

```text
score_gap = score_peak - hardneg_peak
```

这里的 `hardneg_peak` 是预测框外 hard negative tokens 的最高 score。这个门控更贴近 reliability 更新本身要处理的 target-hardneg 竞争。

```text
if LANGUAGE_RELIABILITY_GATE_MODE == "score_gap":
  update only when score_peak - hardneg_peak >= LANGUAGE_RELIABILITY_SCORE_GAP_THR

if LANGUAGE_RELIABILITY_GATE_MODE == "score_peak":
  update only when score_peak >= LANGUAGE_RELIABILITY_SCORE_THR

if LANGUAGE_RELIABILITY_GATE_MODE == "both":
  update only when both conditions pass

otherwise:
  skip reliability update
```

默认：

```yaml
LANGUAGE_RELIABILITY_UPDATE_GATE: True
LANGUAGE_RELIABILITY_GATE_MODE: score_gap
LANGUAGE_RELIABILITY_SCORE_THR: 0.4
LANGUAGE_RELIABILITY_SCORE_GAP_THR: 0.05
```

## 类型约束

当前实现是轻量启发式版本：

```text
subject candidate:
  取 anchor 中第一个非属性、非上下文 content token
  reliability 下限为 LANGUAGE_SUBJECT_MIN_RELIABILITY

context words:
  常见介词、关系词、背景词
  reliability 上限为 LANGUAGE_CONTEXT_MAX_WEIGHT
```

这一步是为了避免主体词短期低证据被压没，也避免上下文词长期压过主体词。

## 代码改动

### 1. 配置项

文件：

```text
lib/config/dutrack/config.py
```

新增：

```yaml
TEST:
  LANGUAGE_WORD_RELIABILITY_ENABLE: False
  LANGUAGE_WORD_RELIABILITY_SOURCE: target_hardneg_gap
  LANGUAGE_WORD_RELIABILITY_MOMENTUM: 0.8
  LANGUAGE_WORD_RELIABILITY_TAU: 0.1
  LANGUAGE_SUBJECT_MIN_RELIABILITY: 0.7
  LANGUAGE_CONTEXT_MAX_WEIGHT: 0.4
  LANGUAGE_RELIABILITY_UPDATE_GATE: True
  LANGUAGE_RELIABILITY_GATE_MODE: score_gap
  LANGUAGE_RELIABILITY_SCORE_THR: 0.4
  LANGUAGE_RELIABILITY_SCORE_GAP_THR: 0.05
```

### 2. Tracker 状态更新

文件：

```text
lib/test/tracker/dutrack.py
```

新增：

```text
language_word_reliability_active
_update_language_word_reliability(...)
_network_language_word_reliability(...)
```

`track()` 中先用上一帧 reliability forward，得到当前预测后再更新 reliability，供下一帧使用。

### 3. 网络传参

文件：

```text
lib/models/dutrack/dutrack.py
lib/models/dutrack/itpn.py
lib/models/dutrack/language_token_emphasizer.py
```

新增 `language_word_reliability / word_reliability` 传参。

在 `LanguageGuidedTokenEmphasizer._word_level_scores()` 内部执行：

```text
word_weights = word_weights * reliability
word_weights = word_weights / sum(word_weights)
```

同时继续导出：

```text
word_level_reliability
word_level_weights
word_level_search_token_scores
```

用于诊断 `reliability -> weight -> gap` 是否一致。

### 4. 诊断参数

文件：

```text
tracking/visualte_diagnostic.py
tracking/visualte_diagnostic_suite.py
```

新增运行时参数：

```bash
--language_word_reliability 1
--language_word_reliability_source target_hardneg_gap
--language_word_reliability_momentum 0.8
--language_word_reliability_tau 0.1
--language_subject_min_reliability 0.7
--language_context_max_weight 0.4
--language_reliability_gate_mode score_gap
--language_reliability_score_gap_thr 0.05
```

`diagnostics.csv` 继续记录：

```text
language_word_reliability_active
language_word_reliability
```

`word_reliability_diagnostics.csv` 额外记录：

```text
word_reliability
language_word_reliability_updated
language_word_reliability_delta
language_word_reliability_score_peak
language_word_reliability_hardneg_peak
language_word_reliability_score_gap
subject_candidate_word
subject_gap
subject_rank_by_gap
subject_rank_by_weight
best_gap_word
weight_gap_rank_corr
top3_weight_gap_overlap
```

其中：

```text
weight_gap_rank_corr:
  word_rank_by_weight 与 word_rank_by_gap 的 Spearman-style 相关性。

top3_weight_gap_overlap:
  权重 top3 词和 gap top3 词的重合率。
```

suite 汇总新增：

```text
language_reliability_update_rate
mean_reliability_delta
```

其中：

```text
update_rate = 实际发生 reliability update 的帧数 / 总帧数
mean_reliability_delta = 每帧 content-word reliability 平均绝对变化幅度
```

## 建议实验：只跑 OTB + HOOT

暂时不跑 OLOD，避免极小目标和标注语言质量把第一轮判断复杂化。

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
  --language_word_filter 0 \
  --language_word_reliability 1 \
  --language_word_reliability_source target_hardneg_gap \
  --language_word_reliability_momentum 0.9 \
  --language_word_reliability_tau 0.1 \
  --language_subject_min_reliability 0.7 \
  --language_context_max_weight 0.4 \
  --language_reliability_gate_mode score_gap \
  --language_reliability_score_gap_thr 0.05 \
  --original_view auto \
  --case otb_lang:Biker \
  --case hoot_balanced20:0 \
  --output_tag word_reliability_stage2_otb_hoot_fullstats
```

## 判断重点

重点不是只看 IoU，而是看：

```text
1. language_word_reliability 是否稳定，而不是剧烈震荡
2. word_rank_by_weight 是否比 Stage 1 更接近 word_rank_by_gap
3. anchor_subject_gap_mean 是否保持为正
4. context_gap_mean 是否被压低，避免上下文词主导
5. deploy 指标是否接近 oracle 指标
```

如果 Stage 2 没有让 `word_rank_by_weight` 更接近 `word_rank_by_gap`，说明可靠性虽然更新了，但还没有真正改善 word-level prior 的判别性。

## 推荐 reliability 消融

四组对照：

```text
A. anchor baseline
   --language_word_reliability 0

B. reliability only
   --language_word_reliability 1
   --language_subject_min_reliability 0
   --language_context_max_weight 1

C. reliability + subject floor
   --language_word_reliability 1
   --language_subject_min_reliability 0.7
   --language_context_max_weight 1

D. reliability + subject floor + context cap
   --language_word_reliability 1
   --language_subject_min_reliability 0.7
   --language_context_max_weight 0.4
```

当前命令行已经支持打开/关闭 reliability，也支持运行时覆盖 subject floor 和 context cap，因此 B/C/D 不需要改配置文件。

重点比较：

```text
1. weight_gap_rank_corr 是否提升
2. top3_weight_gap_overlap 是否提升
3. subject_rank_by_gap / subject_rank_by_weight 是否合理
4. context_gap_mean 是否下降
5. IoU 和 score peak 是否没有明显受损
```

## Stage 2 补充实验矩阵

参考 `stage2_soft_word_reliability_A1E585AEE98C8CE5BC.md`，当前 Stage 2 先不进入 BLIP candidate supplement。下一轮实验目标是把固定词集合下的 soft reliability 做扎实。

阶段定位：

```text
Stage 2 不是直接追求 IoU 大幅提升；
核心是验证 word_weight 排名是否更接近 target-hardneg word_gap 排名。
```

当前优先主线：

```text
B 组 reliability only:
  固定 anchor 文本
  不删词
  不改写句子
  不加入 BLIP 新词
  不使用 hard subject floor
  不使用 hard context cap
  只用 deploy target-hardneg gap 更新 soft word reliability
```

当前 A/B/C/D 的阶段性判断：

```text
A. anchor baseline:
   稳定基准，语言不变化。

B. reliability only:
   当前最值得保留，重点看 rank alignment 是否提升。

C. reliability + subject floor:
   如果 rank alignment 下降，说明硬 subject floor 当前不稳。

D. reliability + subject floor + context cap:
   如果 rank alignment 进一步下降，说明硬 context cap 当前不适合。
```

### 1. 基础命令

后续补充实验都以 OTB + HOOT 为小套件，暂时不跑 OLOD：

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
  --language_word_filter 0 \
  --original_view auto \
  --case otb_lang:Biker \
  --case hoot_balanced20:0
```

下面每组实验是在基础命令后追加对应参数和 `--output_tag`。

### 2. Update Gate 消融

目的：

```text
验证当前预测框作为 P_pred 时，是否需要 gate 降低错误 self-training 更新。
```

#### B0：no gate

```bash
  --language_word_reliability 1 \
  --language_word_reliability_source target_hardneg_gap \
  --language_word_reliability_momentum 0.9 \
  --language_word_reliability_tau 0.1 \
  --language_subject_min_reliability 0 \
  --language_context_max_weight 1 \
  --language_reliability_update_gate 0 \
  --output_tag stage2_gate_B0_no_gate
```

#### B1：score peak gate

```bash
  --language_word_reliability 1 \
  --language_word_reliability_source target_hardneg_gap \
  --language_word_reliability_momentum 0.9 \
  --language_word_reliability_tau 0.1 \
  --language_subject_min_reliability 0 \
  --language_context_max_weight 1 \
  --language_reliability_update_gate 1 \
  --language_reliability_gate_mode score_peak \
  --language_reliability_score_thr 0.4 \
  --output_tag stage2_gate_B1_score_peak
```

#### B2：score gap gate

```bash
  --language_word_reliability 1 \
  --language_word_reliability_source target_hardneg_gap \
  --language_word_reliability_momentum 0.9 \
  --language_word_reliability_tau 0.1 \
  --language_subject_min_reliability 0 \
  --language_context_max_weight 1 \
  --language_reliability_update_gate 1 \
  --language_reliability_gate_mode score_gap \
  --language_reliability_score_gap_thr 0.05 \
  --output_tag stage2_gate_B2_score_gap_005
```

#### B3：both gate

```bash
  --language_word_reliability 1 \
  --language_word_reliability_source target_hardneg_gap \
  --language_word_reliability_momentum 0.9 \
  --language_word_reliability_tau 0.1 \
  --language_subject_min_reliability 0 \
  --language_context_max_weight 1 \
  --language_reliability_update_gate 1 \
  --language_reliability_gate_mode both \
  --language_reliability_score_thr 0.4 \
  --language_reliability_score_gap_thr 0.05 \
  --output_tag stage2_gate_B3_both
```

建议额外扫 `score_gap_thr`：

```text
0.03 / 0.05 / 0.08
```

重点指标：

```text
language_reliability_update_rate
mean_reliability_delta
weight_gap_rank_corr
top3_weight_gap_overlap
word_direct_hardneg_gap_max
word_evidence_deploy_mean_gap_max
word_evidence_deploy_rank_corr_max
Score GT Mass
On/Off Peak Delta
IoU
```

判断：

```text
如果 score_gap gate 让 rank alignment 更稳且 IoU/Score GT Mass 不下降，保留 score_gap gate。
如果 gate 后 rank alignment 明显下降，说明阈值过严或 gate 条件不适合。
```

### 3. Soft Type Prior 消融

目的：

```text
用轻微软先验替代 hard subject floor / hard context cap。
```

实现形式：

```text
final_word_weight_i ∝ base_weight_i * reliability_i * type_prior_i
```

当前已支持运行时参数：

```bash
--language_subject_type_prior
--language_attribute_type_prior
--language_context_type_prior
```

注意：

```text
soft type prior 只轻微影响排序；
最终排序仍应主要由 word_gap / reliability 决定。
```

统一设置：

```text
--language_subject_min_reliability 0
--language_context_max_weight 1
```

即关闭 hard floor/cap，只测试 soft prior。

#### T0：reliability only

```bash
  --language_word_reliability 1 \
  --language_word_reliability_source target_hardneg_gap \
  --language_word_reliability_momentum 0.9 \
  --language_word_reliability_tau 0.1 \
  --language_subject_min_reliability 0 \
  --language_context_max_weight 1 \
  --language_subject_type_prior 1.0 \
  --language_attribute_type_prior 1.0 \
  --language_context_type_prior 1.0 \
  --language_reliability_update_gate 1 \
  --language_reliability_gate_mode score_gap \
  --language_reliability_score_gap_thr 0.05 \
  --output_tag stage2_type_T0_reliability_only
```

#### T1：subject prior 1.05

```bash
  --language_word_reliability 1 \
  --language_word_reliability_source target_hardneg_gap \
  --language_word_reliability_momentum 0.9 \
  --language_word_reliability_tau 0.1 \
  --language_subject_min_reliability 0 \
  --language_context_max_weight 1 \
  --language_subject_type_prior 1.05 \
  --language_attribute_type_prior 1.0 \
  --language_context_type_prior 1.0 \
  --language_reliability_update_gate 1 \
  --language_reliability_gate_mode score_gap \
  --language_reliability_score_gap_thr 0.05 \
  --output_tag stage2_type_T1_subject105
```

#### T2：subject prior 1.10

```bash
  --language_word_reliability 1 \
  --language_word_reliability_source target_hardneg_gap \
  --language_word_reliability_momentum 0.9 \
  --language_word_reliability_tau 0.1 \
  --language_subject_min_reliability 0 \
  --language_context_max_weight 1 \
  --language_subject_type_prior 1.10 \
  --language_attribute_type_prior 1.0 \
  --language_context_type_prior 1.0 \
  --language_reliability_update_gate 1 \
  --language_reliability_gate_mode score_gap \
  --language_reliability_score_gap_thr 0.05 \
  --output_tag stage2_type_T2_subject110
```

#### T3：context prior 0.95

```bash
  --language_word_reliability 1 \
  --language_word_reliability_source target_hardneg_gap \
  --language_word_reliability_momentum 0.9 \
  --language_word_reliability_tau 0.1 \
  --language_subject_min_reliability 0 \
  --language_context_max_weight 1 \
  --language_subject_type_prior 1.0 \
  --language_attribute_type_prior 1.0 \
  --language_context_type_prior 0.95 \
  --language_reliability_update_gate 1 \
  --language_reliability_gate_mode score_gap \
  --language_reliability_score_gap_thr 0.05 \
  --output_tag stage2_type_T3_context095
```

#### T4：subject 1.05 + context 0.95

```bash
  --language_word_reliability 1 \
  --language_word_reliability_source target_hardneg_gap \
  --language_word_reliability_momentum 0.9 \
  --language_word_reliability_tau 0.1 \
  --language_subject_min_reliability 0 \
  --language_context_max_weight 1 \
  --language_subject_type_prior 1.05 \
  --language_attribute_type_prior 1.0 \
  --language_context_type_prior 0.95 \
  --language_reliability_update_gate 1 \
  --language_reliability_gate_mode score_gap \
  --language_reliability_score_gap_thr 0.05 \
  --output_tag stage2_type_T4_subject105_context095
```

不建议第一轮使用：

```text
subject_type_prior > 1.2
context_type_prior < 0.8
```

### 4. 词级日志与可视化观察

当前已有：

```text
diagnostics.csv
word_reliability_diagnostics.csv
```

重点查看字段：

```text
frame
anchor description / language_description
subject_candidate_word
best_gap_word
word
word_type
word_reliability
word_weight
word_gap
word_rank_by_weight
word_rank_by_gap
weight_gap_rank_corr
top3_weight_gap_overlap
```

建议重点观察词：

```text
Biker:
  head / man / bike / track

HOOT backpack-004:
  red / backpack / held / person
```

如果需要更直观的图，下一步可以从 `word_reliability_diagnostics.csv` 生成：

```text
1. word_gap 曲线
2. word_reliability 曲线
3. word_weight vs word_gap 排名对比图
```

### 5. Stage 2 成功标准

Stage 2 成功不要求 IoU 明显提升。更合理的标准是：

```text
Lang Changes = 0
Unique Lang = 1
update_rate 合理，不应为 0，也不应在低置信场景无条件接近 1
mean_reliability_delta 平滑，不应出现大幅震荡
weight_gap_rank_corr 提升
top3_weight_gap_overlap 提升
reliability 曲线更平滑
word_hardneg_gap 不下降
Score GT Mass / On-Off Peak Delta 不下降
IoU 至少不明显下降
```

如果这些成立，说明固定词集合下的语言调制是稳定且可解释的，再进入 Stage 3 的 BLIP candidate supplement 会更稳。
