# Stage 3 Conservative Text Absorption Probe

更新时间：2026-05-26

## 目标

当前需要区分两个问题：

```text
1. BLIP 候选语言本身是否包含可用增量；
2. 当前 hard replacement / full reject 的吸收方式是否过于粗糙。
```

因此这版只做文本级反事实诊断，不训练模型，也不改正常 tracker 推理逻辑。

## 新增语言源

在 `tracking/language_state_s0_probe.py` 中，除了已有的：

```text
anchor: 初始语言
prev: 当前维护语言状态
blip: BLIP 整句候选
```

新增两种保守吸收候选：

```text
anchor_delta = anchor + new BLIP content words
prev_delta   = prev   + new BLIP content words
```

其中 `new BLIP content words` 使用当前已有的轻量规则：

```text
1. 去掉 stop words；
2. 去掉已经出现在 base description 中的词；
3. 去重；
4. 最多保留 6 个新增词。
```

这不是最终语言更新方法，只是用来判断“部分吸收”是否比“整句替换”更合理。

## 新增诊断指标

逐帧 CSV 新增：

```text
hard_replace_gain_over_prev = gap(blip) - gap(prev)
anchor_delta_gain_over_prev = gap(anchor_delta) - gap(prev)
prev_delta_gain_over_prev   = gap(prev_delta) - gap(prev)

best_partial_source
best_partial_gap
best_partial_gain_over_prev

partial_beats_hard_replace
partial_useful_when_blip_hurts
```

其中最关键的是：

```text
partial_useful_when_blip_hurts = 1
```

它表示：

```text
BLIP 整句替换会伤害当前语言状态，
但保守部分吸收仍然有正收益。
```

这类帧说明 BLIP 不是完全没用，而是 hard replacement 的吸收方式有问题。

同时增加 deploy 视角的部分吸收指标：

```text
anchor_delta_deploy_gap
prev_delta_deploy_gap
deploy_best_partial_source
deploy_best_partial_gain_over_prev
```

用于判断部署可用的预测框证据是否也能看到同样趋势。

## 新增状态更新策略

新增两个 probe-only 策略：

```text
prev_delta_gate:
  如果 deploy quality gate 接受 BLIP，则 prev = prev_delta

best_partial_oracle:
  离线 oracle 诊断。若 best_partial_gain > 0，则 prev = best_partial_source
```

注意：

```text
best_partial_oracle 依赖 GT score-gap，只能做上界诊断，不能部署。
```

## 可吸收部分的定义

这版把“可吸收部分”严格定义为：

```text
在同一帧、同一个 BLIP 候选下，
anchor_delta 或 prev_delta 至少有一个比 prev 的 score-gap 更高。
```

公式：

```text
best_partial_gain = max(
  gap(anchor_delta) - gap(prev),
  gap(prev_delta)   - gap(prev)
)

partial_label_useful = best_partial_gain > gap_eps
partial_label_harmful = best_partial_gain < -gap_eps
```

含义：

```text
partial_label_useful = 1
```

不表示 BLIP 整句可靠，而只表示：

```text
BLIP 中存在一种保守文本吸收方式，
能让当前帧的目标中心判别比 prev 更好。
```

这和旧标签不同：

```text
旧标签: quality_gate_score_delta = gap(blip) - gap(prev)
新标签: best_partial_gain = max(gap(anchor_delta), gap(prev_delta)) - gap(prev)
```

因此它更适合回答：

```text
当前 gate 是否能发现“可部分吸收”的 BLIP？
```

而不是：

```text
BLIP 整句是否应该替换 prev？
```

## 新增 partial gate 诊断

为了避免混淆，保留原有 `quality_gate_*`，新增两套 partial 诊断：

```text
partial_current_gate_*
```

含义：

```text
使用旧的 BLIP deploy gate 接受/拒绝结果，
但用 partial_label_useful / harmful 重新判断 true/false accept/reject。
```

用途：

```text
检查旧 gate 是否本来就能识别“可吸收部分”。
```

```text
partial_gate_*
```

含义：

```text
使用 deploy_best_partial_gain_over_prev 作为 deploy 证据，
构造一个专门面向部分吸收的 deploy gate。
```

用途：

```text
检查如果 gate 目标改成“部分吸收”，部署侧是否有可用信号。
```

## 判断方式

如果结果显示：

```text
hard_replace_gain <= 0
best_partial_gain > 0
partial_useful_when_blip_hurts 比例不低
```

则说明：

```text
BLIP 可能包含可用状态信息；
此前 hard replacement / full reject 低估了 BLIP 的价值；
后续应该研究“部分吸收 / token-state residual”，而不是直接否定 BLIP。
```

如果：

```text
hard_replace_gain <= 0
best_partial_gain <= 0
```

仍然占主导，则说明当前主要瓶颈更可能是：

```text
BLIP 候选质量不足，或 DUTrack 对这些语言差异不敏感。
```
