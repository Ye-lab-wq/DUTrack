# Stage 3 Token-State Residual Update Design

更新时间：2026-05-26

## 1. 当前定位

这一步从文本级语言更新转向 token-state 语言更新。

已经不再把以下方案作为主线：

```text
1. BLIP 整句替换；
2. anchor + BLIP 全量新词拼接；
3. prev + BLIP 全量新词拼接；
4. 手工 word-gate 文本拼接。
```

这些已经作为诊断或 baseline 跑过，不需要继续重复作为主要实验。

当前主线是：

```text
在 tokenizer/BERT embedding 之后维护 latent language state tokens。
```

## 2. 为什么要做 token-state

文本级 word-gate 结果说明：

```text
1. BLIP 整句经常有害；
2. 局部语义增量存在；
3. 逐词筛选信号为正；
4. 但把词重新拼成自然语言后，收益会明显衰减。
```

因此问题不只是“选哪些词”，还有：

```text
自然语言字符串拼接不是 DUTrack 内部语言状态更新的稳定表达方式。
```

更合理的对象是：

```text
H_t: BERT embedding + language position embedding 后的一组 language tokens。
```

## 3. DUTrack 当前语言入口

当前代码路径：

```text
lib/models/dutrack/dutrack.py
  DUTrack.forward(..., descript=...)

lib/models/dutrack/itpn.py
  Fast_iTPN.forward_features(...)
    _l_feat(l)
      tokenizer
      BertEmbeddings
      description_patch_pos_embed
    _fusion_feat(z_feat, x_feat, l_feat, ...)
```

所以 token-state 的正确注入位置是：

```text
_l_feat(l) 之后
_fusion_feat(...) 之前
```

而不是在 tracker 里继续拼接文本。

## 4. 本轮实现

新增模型接口：

```text
DUTrack.forward(..., language_token_state=None, language_token_mask=None)
Fast_iTPN.forward_features(..., language_token_state=None, language_token_mask=None)
```

如果没有传入 token state：

```text
保持原始文本路径不变。
```

如果传入 token state：

```text
跳过 _l_feat(l)，直接用 language_token_state 作为 language tokens。
```

这保证已有训练/测试命令默认行为不变。

## 5. S1 最小诊断：Residual Token State

先不训练，先构造两个 counterfactual token 状态：

```text
H_anchor = Encode(anchor_text)
H_prev   = Encode(prev_text)
H_blip   = Encode(blip_text)
```

### Anchor residual

```text
H_anchor_res(alpha) = H_anchor + alpha * (H_blip - H_anchor)
```

含义：

```text
以 anchor identity 为中心，只吸收一小部分 BLIP token delta。
```

### Prev residual

```text
H_prev_res(alpha) = H_prev + alpha * (H_blip - H_prev)
```

含义：

```text
以上一帧语言状态为中心，小幅吸收当前 BLIP token delta。
```

默认测试：

```text
alpha = 0.1, 0.3
```

## 6. 诊断指标

新增字段：

```text
token_anchor_res_a010_gain_over_prev
token_anchor_res_a030_gain_over_prev
token_prev_res_a010_gain_over_prev
token_prev_res_a030_gain_over_prev

token_state_best_source
token_state_best_gap
token_state_best_gain_over_prev
```

套件 summary 记录：

```text
token_state_best_gain
```

对照字段：

```text
hard_replace_gain
anchor_delta_gain
prev_delta_gain
word_gate_best_gain
deploy_word_gate_best_gain
```

## 7. 判断标准

如果：

```text
token_state_best_gain > hard_replace_gain
token_state_best_gain > anchor_delta_gain / prev_delta_gain
```

说明 token-level residual 比文本级更新更稳。

如果：

```text
token_state_best_gain 仍接近 0 或为负
```

说明仅做线性 residual 不够，需要 learnable CandidateAdapter / Gate。

## 8. 与最终可学习模块的关系

本轮 S1 不是最终方法。

它只是在验证：

```text
H_t = H_base + alpha * (H_blip - H_base)
```

这种 token 层小幅更新是否比文本级替换/拼接更接近有效方向。

如果 S1 有正向信号，下一步才进入：

```text
H_candidate = CandidateAdapter(H_blip, H_anchor, H_prev)
g = Gate(H_candidate, H_prev, search evidence)
H_t = H_prev + g * Delta(H_candidate)
```

训练时仍应：

```text
1. 冻结 backbone / head；
2. 只训练 updater / prior；
3. tracking loss 为主；
4. aux score-rank loss 低权重退火；
5. 不用文本可读性作为目标。
```

