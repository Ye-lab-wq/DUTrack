# Stage 3 Word Increment Weak-Label Probe

更新时间：2026-05-26

## 当前目标

这一步不是训练端到端模块，而是先生成词级诊断数据，回答：

```text
BLIP 生成的新增词中，是否存在可以稳定改善 center score 判别的语义增量？
```

如果单词级增量本身没有可分辨信号，后续训练 Word ReliabilityNet 或 token-state updater 都容易退化。

## 新增脚本

新增：

```text
tracking/language_increment_word_probe.py
```

它逐帧执行：

```text
1. 使用当前 tracker 状态裁 search；
2. 获取 base description，目前可选 prev 或 anchor；
3. 如果 trigger/candidate_mode 允许，则生成 BLIP candidate；
4. 从 BLIP 中提取相对 base 的新增 content words；
5. 对每个新增词单独构造：

   word_description = base + word

6. 分别 forward：

   base
   BLIP 整句
   base + 单个新增词

7. 记录每个词对 score-gap / IoU / deploy-gap 的影响。
```

这一步不训练、不改主模型、不改正常 tracker 推理链路。

## 弱标签定义

每个 CSV row 是：

```text
一个 sequence / frame / BLIP word
```

核心字段：

```text
base_gap
word_gap
word_gain_over_base = word_gap - base_gap

base_deploy_gap
word_deploy_gap
word_deploy_gain_over_base

blip_gap
blip_gain_over_base
```

弱标签：

```text
word_label_useful  = 1 if word_gain_over_base > gap_eps
word_label_harmful = 1 if word_gain_over_base < -gap_eps
```

解释：

```text
word_label_useful = 1
```

表示：

```text
只加入这个 BLIP 新增词，就能让当前帧 GT 区域相对 hard negative 的 center score gap 变好。
```

它不是最终监督，只是给后续轻量可学习模块提供第一版可分析弱标签。

## 词角色

脚本暂时只做很弱的角色标注：

```text
context: 处在 CONTEXT_WORDS 中的词
attribute: 常见颜色/大小/外观词
content: 其他内容词
```

这不是最终规则模块，只用于结果分组统计，避免把规则当成方法本身。

## 输出

输出目录：

```text
output/test/language_increment_word_probe/<output_tag>/
```

包含：

```text
word_increment_probe.csv
word_increment_summary.md
```

summary 会统计：

```text
word useful ratio
word harmful ratio
mean word gain
mean deploy word gain
mean BLIP hard gain
mean word-template gap
mean word-search hardneg gap
role-wise useful/harmful/gain
```

## 最小补充实验：融合 token 视觉证据

本轮补充不是把规则接入 tracker，而是在词级 weak-label probe 中额外导出两个 deploy 侧可用的视觉一致性特征：

```text
word_template_gap
word_search_hardneg_gap
```

计算空间：

```text
DUTrack backbone 输出的 fused token space
```

也就是 language / template / search 已经经过项目现有跨模态融合后的 `backbone_feat`，避免直接拿原始 BERT token 和原始视觉 patch 做未对齐点积。

### word_template_gap

```text
sim(word_token, selected_template_target_tokens)
-
sim(word_token, selected_template_background_tokens)
```

其中 template target mask 来自当前 tracker 可用的 template memory mask。它不是只看初帧模板，而是跟随当前被选择的 template_list / memory 走，更接近后续推理可用形态。

### word_search_hardneg_gap

```text
sim(word_token, predicted_search_region_tokens)
-
sim(word_token, base_score_hard_negative_tokens)
```

其中正区域来自当前 base 预测框，hard negative 来自 base score 在预测框外的 top-k 高响应位置。这个字段仍然是自举证据，所以只作为可学习模块输入候选，不作为最终标签。

## 风险控制

这两个字段不使用 GT 作为输入：

```text
GT 只用于 word_label_useful / word_label_harmful 弱标签和诊断。
```

因此它们可以用于后续离线 Word ReliabilityNet 的 deploy-like 输入，但需要重点观察：

```text
1. word_template_gap 是否比简单 word_role 更能区分 useful/harmful；
2. word_search_hardneg_gap 是否会被当前错误预测框带偏；
3. learned reliability 是否能超过 word_deploy_gain baseline。
```

## 判断标准

如果结果显示：

```text
1. useful ratio 明显大于随机噪声水平；
2. attribute/content 的 mean gain 高于 context；
3. word_deploy_gain 与 word_gain 有一定一致性；
4. 部分单词 gain > 0，而整句 BLIP gain <= 0；
```

则说明：

```text
生成语言里确实存在可学习的可靠语义增量。
```

后续可以进入：

```text
Word ReliabilityNet
```

如果这些条件不成立，则说明当前 BLIP/DUTrack 内部语言表征下，词级增量本身不稳定，直接端到端训练风险较高。
