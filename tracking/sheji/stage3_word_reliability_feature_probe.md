# Stage 3 Word Reliability Feature Probe

更新时间：2026-05-26

## 目标

在词级弱标签已经生成后，先做一个离线轻量可学习实验：

```text
输入: deploy 可用的词级特征
输出: word_label_useful
```

目的不是立刻接入 tracker，而是判断：

```text
“可靠语义增量”是否能被一个轻量模型从现有特征中学出来。
```

如果这个离线实验都不能超过简单 baseline，那么后续端到端 Word ReliabilityNet 很容易退化。

## 新增脚本

```text
tracking/word_reliability_feature_probe.py
```

输入：

```text
word_increment_probe.csv
```

标签：

```text
word_label_useful = 1 if word_gain_over_base > label_eps
```

其中 `word_gain_over_base` 使用 GT/oracle score-gap，只作为离线弱标签。

## 特征

当前只使用部署侧可获得或近似可获得的特征：

```text
word_rank
candidate_available
deploy_trigger
base_deploy_gap
word_deploy_gap
word_deploy_gain_over_base
base_score_peak
word_score_peak
score_peak_delta
word_token_found
word_template_target_sim
word_template_bg_sim
word_template_gap
word_search_pred_sim
word_search_hardneg_sim
word_search_hardneg_gap
word_length
word_has_digit
word_is_alpha
word_role one-hot
hashed word n-gram features
```

注意：

```text
不使用 word_gain_over_base 作为输入；
不使用 GT 信息作为输入；
```

否则会造成标签泄漏。

## 模型

第一版支持：

```text
hidden_dim = 0:
  logistic regression

hidden_dim > 0:
  tiny MLP
```

训练方式：

```text
leave-one-sequence-out
```

每次留一个 sequence 测试，其余 sequence 训练。

## 对照 baseline

脚本同时输出：

```text
deploy_gain baseline:
  accept if word_deploy_gain_over_base > deploy_gain_thr
```

这是最重要的基线。

如果 learned 模型不能超过这个 baseline，则说明当前特征还不支持学习式 reliability。

## 输出

```text
output/test/word_reliability_feature_probe/<output_tag>/
```

包含：

```text
word_reliability_loso_metrics.csv
word_reliability_loso_predictions.csv
word_reliability_feature_probe_summary.md
```

重点指标：

```text
accept_precision
useful_recall
hurt_rejection
false_accept_rate_harmful
mean_gain
```

## 判断标准

如果 learned 模型相比 deploy_gain baseline：

```text
1. accept_precision 更高；
2. false_accept_rate_harmful 更低；
3. mean_gain 不下降；
4. leave-one-sequence-out 下稳定；
```

则可以进入下一步：

```text
把 Word ReliabilityNet 接入语言状态更新。
```

否则说明当前可用特征不足，需要更强视觉 grounding / template consistency 特征。

## 首轮结果

输入：

```text
output/test/language_increment_word_probe/stage3_word_increment_oracle_small/word_increment_probe.csv
output/test/language_increment_word_probe/stage3_word_increment_deploy_small/word_increment_probe.csv
```

### Logistic Regression

输出：

```text
output/test/word_reliability_feature_probe/stage3_word_reliability_logistic_small/
```

加权 LOSO：

```text
deploy_gain baseline:
  accept_precision: 0.764936
  useful_recall: 0.740354
  hurt_rejection: 0.793544
  false_accept_rate_harmful: 0.206456
  mean_gain: 0.000233908
  accept_rate: 0.513274

learned logistic:
  accept_precision: 0.629811
  useful_recall: 0.480079
  hurt_rejection: 0.662725
  false_accept_rate_harmful: 0.337275
  mean_gain: 0.000162067
  accept_rate: 0.408555
```

结论：

```text
logistic 明显弱于 deploy_gain baseline。
```

### Tiny MLP hidden_dim=32

输出：

```text
output/test/word_reliability_feature_probe/stage3_word_reliability_mlp32_small/
```

加权 LOSO：

```text
deploy_gain baseline:
  accept_precision: 0.764936
  useful_recall: 0.740354
  hurt_rejection: 0.793544
  false_accept_rate_harmful: 0.206456
  mean_gain: 0.000233908
  accept_rate: 0.513274

learned MLP:
  accept_precision: 0.596162
  useful_recall: 0.638688
  hurt_rejection: 0.474204
  false_accept_rate_harmful: 0.525796
  mean_gain: 0.000136512
  accept_rate: 0.578663
```

结论：

```text
MLP 容量更大，但 false accept 明显增加，仍然弱于 deploy_gain baseline。
```

## 当前判断

这轮实验说明：

```text
1. 词级可靠增量存在；
2. 但用当前 deploy 数值特征 + 简单词面 hash/role 特征训练轻量分类器，不能稳定超过 deploy_gain 阈值；
3. 当前还不适合把 Word ReliabilityNet 接入主 tracker；
4. 下一步应补更直接的视觉一致性特征，而不是继续加分类器容量。
```

最需要补的特征：

```text
1. word-template target consistency
2. word-search hard-negative contrast
3. word 与 anchor target memory 的一致性
4. word 是否稳定跨帧重复支持目标
```

也就是说，主要缺口不是 MLP 是否够大，而是：

```text
当前特征没有真正表达“这个词是否和模板目标一致，并能压制 hard negative”。
```

## 本轮补充：融合 token 一致性特征

为了处理上面的缺口，`language_increment_word_probe.py` 现在额外导出：

```text
word_template_gap
word_search_hardneg_gap
```

`word_reliability_feature_probe.py` 已将这些字段加入可学习输入。它们的定位是：

```text
1. word_template_gap:
   检查新增词是否更贴近当前模板目标 token，而不是模板背景 token；

2. word_search_hardneg_gap:
   检查新增词是否更贴近当前预测目标区域，而不是 base score 的 hard negative 区域。
```

两者都在 DUTrack fused token space 中计算，因此比直接原始 word-token / visual-token 点积更接近项目真实前向。它们仍然不是最终规则：

```text
它们只作为 Word ReliabilityNet 的候选输入，用 LOSO 检查是否能改善 false accept / useful recall。
```

下一轮判断标准保持不变：

```text
learned reliability 必须超过 deploy_gain baseline；
如果仍然不能超过，说明问题不只是特征数量，而可能是当前语言/视觉融合表征本身不足以稳定支撑词级可靠性判断。
```
