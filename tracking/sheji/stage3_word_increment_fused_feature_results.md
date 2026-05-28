# Stage 3 Word Increment Fused Feature Results

更新时间：2026-05-26

## 实验目的

本轮是 `stage3_word_increment_weak_label_probe` 的最小补充实验：

```text
在单词级 BLIP 增量 probe 中，额外记录 fused-token 视觉一致性证据。
```

新增字段：

```text
word_template_gap
word_search_hardneg_gap
word_template_target_sim
word_template_bg_sim
word_search_pred_sim
word_search_hardneg_sim
```

这些字段在 DUTrack `backbone_feat` 的 fused token space 中计算，不直接使用原始 BERT token 和原始视觉 patch。

## Probe 结果

### Oracle BLIP

路径：

```text
output/test/language_increment_word_probe/stage3_word_increment_oracle_fused_small/
```

核心结果：

```text
Rows: 1356
word useful ratio: 0.581858
word harmful ratio: 0.417404
mean word gain: 9.86323e-05
mean deploy word gain: 5.56268e-05
mean BLIP hard gain: -0.00036053
mean word-template gap: -0.00326668
mean word-search hardneg gap: 0.0122036
```

分组上：

```text
attribute: search hardneg gap = 0.077246
content:   search hardneg gap = 0.0210125
context:   search hardneg gap = -0.0439657
```

这说明 search hard-negative contrast 对 context 词有一定抑制趋势，但还不能直接说明它能稳定预测 useful label。

### Deploy-like BLIP

路径：

```text
output/test/language_increment_word_probe/stage3_word_increment_deploy_fused_small/
```

核心结果：

```text
Rows: 680
word useful ratio: 0.479412
word harmful ratio: 0.519118
mean word gain: 0.000121142
mean deploy word gain: 5.24536e-05
mean BLIP hard gain: -0.000479662
mean word-template gap: 0.00730814
mean word-search hardneg gap: 0.0451671
```

分组上：

```text
attribute: search hardneg gap = 0.133375
content:   search hardneg gap = 0.0519434
context:   search hardneg gap = -0.00700367
```

同样能看到 context 的 hardneg gap 更弱，但这个趋势不足以直接转化为可靠接受策略。

## Reliability Probe 结果

路径：

```text
output/test/word_reliability_feature_probe/stage3_word_reliability_fused_logistic_small/
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

learned fused logistic:
  accept_precision: 0.610915
  useful_recall: 0.492533
  hurt_rejection: 0.59665
  false_accept_rate_harmful: 0.40335
  mean_gain: 0.000165743
  accept_rate: 0.454769
```

对比旧 logistic：

```text
old learned logistic:
  accept_precision: 0.629811
  useful_recall: 0.480079
  hurt_rejection: 0.662725
  false_accept_rate_harmful: 0.337275
  mean_gain: 0.000162067
```

新增 fused 特征让 recall 和 mean gain 略有变化，但 precision / harmful rejection 变差，没有超过简单 `word_deploy_gain_over_base > 0` baseline。

## 相关性检查

合并 oracle + deploy 两组 CSV 后，关键字段与 `word_gain_over_base` 的相关性：

```text
word_deploy_gain_over_base: corr_gain = 0.6572
word_template_gap:          corr_gain = 0.0449
word_search_hardneg_gap:    corr_gain = 0.0637
```

useful / harmful 分组均值：

```text
word_template_gap:
  useful  = -0.000354
  harmful =  0.001287

word_search_hardneg_gap:
  useful  = 0.011495
  harmful = 0.037454
```

这说明当前新增 fused gap 与真正的 oracle useful label 不一致，甚至 harmful 词的 fused gap 更高。

## 判断

本轮最重要结论：

```text
单词级语义增量仍然存在；
整句 BLIP 仍然整体偏负；
但当前 fused token 的 template/search 相似度不能可靠判断“哪个新增词有用”。
```

更具体地说：

```text
1. word_search_hardneg_gap 能粗略反映 context 词风险，但不是 useful/harmful 的稳定判别信号；
2. word_template_gap 基本没有提供有效增量；
3. learned reliability 仍弱于 deploy_gain baseline；
4. 当前最强 deploy 信号仍是 word_deploy_gain_over_base，而不是词-视觉相似度本身。
```

## 后续含义

这说明下一步不适合继续简单堆叠相似度特征或扩大分类器容量。

更合理的方向是：

```text
1. 保留 word_deploy_gain 作为当前最强 deploy-side 判断信号；
2. 对 fused token 相似度只作为辅助诊断，不直接作为 gate 主信号；
3. 若要做可学习模块，需要让模块端到端学习“语言增量如何影响 score”，而不是只学习静态 word-visual similarity；
4. 如果继续做语言状态更新，应优先研究如何稳定生成/吸收候选语言，而不是把当前 fused similarity 当作可靠 grounding。
```

