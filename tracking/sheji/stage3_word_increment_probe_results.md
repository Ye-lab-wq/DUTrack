# Stage 3 Word Increment Probe Results

更新时间：2026-05-26

## 实验设置

本轮使用：

```text
config: dutrack_384_full_lmq_d1_e10
datasets: otb_lang, hoot_balanced20
sequences: Bird1, Dog, Gym, potted_plant-008, toilet_paper-001, koala-003
base_source: anchor
prompt_mode: space
max_frames: 80
```

两组：

```text
stage3_word_increment_oracle_small:
  candidate_mode = oracle_blip

stage3_word_increment_deploy_small:
  candidate_mode = deploy_like
```

输出：

```text
output/test/language_increment_word_probe/stage3_word_increment_oracle_small/
output/test/language_increment_word_probe/stage3_word_increment_deploy_small/
```

## 核心结果

### Oracle BLIP

```text
rows: 1356
word useful ratio: 0.581858
word harmful ratio: 0.417404
mean word gain: 9.86323e-05
mean deploy word gain: 5.56268e-05
mean BLIP hard gain: -0.00036053
```

### Deploy-like BLIP

```text
rows: 680
word useful ratio: 0.479412
word harmful ratio: 0.519118
mean word gain: 0.000121142
mean deploy word gain: 5.24536e-05
mean BLIP hard gain: -0.000479662
```

## 主要发现

### 1. 单词级增量确实存在

两组实验里：

```text
mean word gain > 0
mean BLIP hard gain < 0
```

说明：

```text
BLIP 整句替换整体有害，
但 BLIP 里的部分单词单独加入 anchor 后，平均能带来正向 score-gap 增益。
```

这支持“从生成语言中筛选可靠语义增量”的方向。

### 2. deploy 信号和 oracle 信号有中等一致性

统计结果：

```text
oracle/deploy gain corr:
  oracle_blip: 0.6737
  deploy_like: 0.6359

sign agree:
  oracle_blip: 0.7898
  deploy_like: 0.7294
```

说明 deploy-like 预测框证据不是完全不可用，但也不足够稳定。

### 3. 词级效果强烈依赖序列

Oracle 模式下：

```text
koala-003 useful ratio: 0.904
toilet_paper-001 useful ratio: 0.593
Dog useful ratio: 0.607
Bird1 useful ratio: 0.282
Gym useful ratio: 0.298
```

Deploy-like 模式下：

```text
Dog useful ratio: 0.677
potted_plant-008 useful ratio: 0.685
toilet_paper-001 useful ratio: 0.558
Bird1 useful ratio: 0.240
Gym useful ratio: 0.306
```

说明同一机制不是全局稳定正收益。后续训练/诊断必须按序列类型分组，不能只看均值。

### 4. 词面 role 规则并不可靠

当前弱 role 统计显示：

```text
context words useful ratio 并不低；
context mean gain 也可能为正。
```

例如 deploy-like：

```text
context useful ratio: 0.532609
context mean gain: 0.000197165
context deploy mean gain: -8.37814e-05
```

这说明：

```text
不能简单按 subject/context/attribute 手工规则筛词。
```

部分 context 词可能通过场景相关性间接帮助当前 score，但部署侧未必稳定。

### 5. BLIP 词质量存在明显噪声

高增益/低增益词里出现：

```text
de
dekloen
disply
screenshot
quote
```

这说明 BLIP caption 或 tokenizer 后文本质量存在噪声，后续 reliability 模块必须具备拒绝无意义词的能力。

## 当前判断

这轮结果说明：

```text
可靠语义增量是存在的；
整句 BLIP 不能直接更新；
简单人工词类规则不足；
deploy-like 证据有一定可用性，但需要学习式筛选。
```

因此下一步不应该继续扩展人工规则，而应该构建轻量可学习的：

```text
Word ReliabilityNet
```

它的目标不是生成语言，而是预测：

```text
某个 BLIP 新增词是否是当前跟踪状态下可吸收的可靠增量。
```

## 下一步建议

### Step 1：构造训练/验证 CSV

使用 `word_increment_probe.csv` 作为弱监督数据：

```text
label = word_label_useful
```

候选输入特征：

```text
word text embedding
word_role one-hot
word_rank
deploy_trigger
base_gap
base_deploy_gap
word_deploy_gain_over_base
blip_gain_over_base
base_score_peak
word_score_peak
```

### Step 2：先做离线轻量分类器

先不接入 tracker，做 LOSO 或 leave-sequence-out：

```text
输入: deploy 可用特征
输出: word_label_useful
```

重点看：

```text
precision
recall
false accept
sequence generalization
```

### Step 3：再考虑接入更新

只有当离线分类器能超过简单 baseline，才进入：

```text
Word ReliabilityNet -> weighted language delta -> score prior / token state
```

