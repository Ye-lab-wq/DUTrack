# Stage 3 Word-Gated Text Absorption Probe

更新时间：2026-05-26

## 背景

上一轮 fused-token 最小补充实验说明：

```text
1. BLIP 整句 hard replacement 仍然经常有害；
2. 单词级增量确实存在；
3. word-template / word-search fused similarity 不能稳定判断 useful/harmful；
4. 当前最强 deploy-side 信号仍然是 word_deploy_gain_over_base。
```

因此本轮不继续堆相似度特征，而是直接验证一个更保守的问题：

```text
只吸收那些在 deploy score-gap 上带来正向增益的 BLIP 新词，是否优于全量拼接或整句替换？
```

## 新增开关

修改文件：

```text
tracking/language_state_s0_probe.py
tracking/language_state_s0_screen.py
```

新增参数：

```text
--word_absorption
--word_gate_max_candidate_words
--word_gate_max_selected_words
--word_gate_min_deploy_gain
```

新增状态更新策略：

```text
--state_update_policy word_gate
```

## 数据流

每帧仍然先得到：

```text
anchor_description
prev_description
blip_description
```

然后在 `--word_absorption` 开启时：

```text
1. 从 BLIP 中取出相对 prev_description 的新增 content words；
2. 对每个新增词构造：

   word_description = prev_description + word

3. 单独 forward，得到 word_score_map；
4. 用当前 prev 预测框作为 deploy 正区域，用 base score 的 hard negative 作为负区域；
5. 计算：

   word_deploy_gain = gap(word_description) - gap(prev_description)

6. 选择：

   word_deploy_gain > max(quality_gate_gap_eps, word_gate_min_deploy_gain)
   且不在 CONTEXT_WORDS 中的词

7. 构造两个候选状态：

   anchor_word_gate = anchor + selected_words
   prev_word_gate   = prev   + selected_words
```

这一步仍然是 probe，不改正常 tracker 推理。

## 输出字段

新增逐帧字段：

```text
word_gate_word_count
word_gate_selected_count
word_gate_selected_words
word_gate_best_word
word_gate_best_word_deploy_gain
anchor_word_gate_gain_over_prev
prev_word_gate_gain_over_prev
word_gate_best_source
word_gate_best_gain_over_prev
deploy_word_gate_best_source
deploy_word_gate_best_gain_over_prev
```

套件 summary 中新增：

```text
word_gate_selected_words
word_gate_best_gain
deploy_word_gate_best_gain
anchor_word_gate_gain
prev_word_gate_gain
```

## 判断标准

主要比较：

```text
hard_replace_gain
anchor_delta_gain
prev_delta_gain
best_partial_gain
word_gate_best_gain
deploy_word_gate_best_gain
```

如果：

```text
word_gate_best_gain > best_partial_gain
或 deploy_word_gate_best_gain > deploy_best_partial_gain
```

说明“逐词保守吸收”比“把 BLIP 新词全拼进去”更合理。

如果没有提升，则说明：

```text
当前 deploy score-gap 逐词筛选仍然不能稳定发现可吸收语义；
后续应转向 token-state / learnable updater，而不是继续文本规则。
```

## 风险

该方法依赖当前预测框作为 deploy 正区域，因此仍有自举风险：

```text
如果 prev 预测框已经漂移，word_deploy_gain 可能强化错误区域。
```

所以它只能作为下一步诊断，不应直接作为最终算法。

