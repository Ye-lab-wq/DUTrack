# Stage 3：Anchor-Preserving Token State Update 实验指导

更新时间：2026-05-26

## 0. 当前阶段核心判断

当前语言更新路线已经经历多轮诊断，结论逐渐清楚：

```text
1. 原 DUTrack trigger 更适合作为候选语言生成器，而不是最终语言更新决策器；
2. BLIP/current caption 有时有用，但经常有害，不能直接替换当前语言状态；
3. deploy-like score-gap gate 能减少有害 BLIP，但依赖 P_pred，存在预测框自举问题；
4. word-level visual support / token dot-product evidence 当前没有可靠区分 true_accept 与 false_accept；
5. 继续在文本字符串层面拼接、删词、加词，会让规则系统越来越臃肿；
6. 更合理的下一步是从“文本状态更新”转向“language token state update”。
```

当前推荐的新主线是：

```text
Anchor 保主体；
BLIP 只提供动态状态候选；
语言状态不再主要维护为自然语言字符串；
而是维护为一组连续 language token / embedding state。
```

---

## 1. 为什么要从文本层更新转向 token 层更新

### 1.1 当前文本级更新流程

当前系统维护：

```text
prev_description
```

初始化：

```text
prev_description = language_anchor
```

每帧流程：

```text
anchor_description = tracker.language_anchor
prev_description   = 上一帧维护的语言状态
blip_description   = 当前帧候选语言，只有 trigger 后才生成
```

然后分别 forward：

```text
DUTrack(anchor_description)
DUTrack(prev_description)
DUTrack(blip_description)
```

如果 deploy gate 接受 BLIP：

```text
gate:
  prev_description = blip_description

anchor_state_gate:
  prev_description = compact(anchor_description + BLIP_state_words)
```

其中 `anchor_state_gate` 当前逻辑是：

```python
def compose(anchor, blip):
    anchor_words = content_words(anchor)
    state_words = []

    for word in content_words(blip):
        if word in anchor_words:
            continue
        if word already selected:
            continue
        state_words.append(word)
        if len(state_words) >= 6:
            break

    return anchor + " " + " ".join(state_words)
```

因此当前语言状态本质是：

```text
anchor 主体文本 + BLIP 补充词
```

---

### 1.2 文本级更新的问题

文本级更新会带来很多工程与建模问题：

```text
1. 语法不稳定；
2. 关键词堆叠后不一定符合 BERT 原始训练分布；
3. 需要手工决定哪些词添加、哪些词删除；
4. 需要处理重复词、BERT token 长度、WordPiece 截断；
5. context/background/person 等词很容易被误加入状态；
6. 一旦 hard replacement 接受错误 BLIP，prev_description 会被污染；
7. 更新发生在 tokenizer / embedding 之前，和真正进入模型的 language tokens 之间存在间接映射。
```

所以继续在文本字符串层面做规则，会越来越复杂。

---

## 2. 新目标：维护连续 language token state

更合理的状态形式是：

```text
H_lang_t = 当前语言状态 tokens
```

其中：

```text
H_anchor = anchor language tokens
H_prev   = 上一帧语言状态 tokens
H_blip   = 当前 BLIP candidate tokens
```

更新不再是：

```text
prev_description = compact(anchor + BLIP words)
```

而是：

```text
H_t = Update(H_anchor, H_prev, H_blip, gate)
```

这样语言状态更新发生在 tokenizer / BERT 之后，避免自然语言拼接问题。

---

## 3. 推荐建模：Identity Tokens + State Tokens

建议将语言状态拆成两部分：

```text
Identity tokens:
  来自 anchor，长期稳定，不被 BLIP 覆盖。

State tokens:
  来自 BLIP / current observation，可动态更新。
```

形式上：

```text
H_lang_t = [H_id ; H_state_t]
```

其中：

```text
H_id = Encode(anchor)
H_state_t = dynamic language state memory
```

更新公式：

```text
H_state_t = (1 - g_t) * H_state_{t-1}
          + g_t * Adapter(H_blip)
```

最终输入 tracker 的语言 token：

```text
H_lang_t = [H_id ; H_state_t]
```

这样可以实现：

```text
anchor 保主体；
BLIP 补状态；
不再需要维护自然语言语法。
```

---

## 4. 最小实现方案

不要一开始做复杂训练。建议分三阶段。

### Stage T0：Token-State Diagnostic，不训练

目标：

```text
验证 token-level state composition 是否比文本 hard replacement / anchor-word 拼接更稳定。
```

保持模型不训练，只在 forward 前构造混合 language tokens。

当前流程：

```text
text -> tokenizer -> BERT -> language tokens
```

改成：

```text
anchor_text -> BERT -> H_anchor
prev_text   -> BERT -> H_prev
blip_text   -> BERT -> H_blip
```

构造几种 token 状态：

#### T0-A：Prev / BLIP 线性混合

```text
H_mix = (1 - alpha) * H_prev + alpha * H_blip
```

测试：

```text
alpha = 0.1 / 0.3 / 0.5
```

#### T0-B：Anchor-preserving residual

```text
H_mix = H_anchor + alpha * (H_blip - H_anchor)
```

测试：

```text
alpha = 0.1 / 0.3 / 0.5
```

#### T0-C：Gate-controlled residual

```text
H_mix = H_anchor + g_t * alpha * (H_blip - H_anchor)
```

第一版：

```text
g_t = 1 if quality_gate_accept else 0
```

暂时不训练 Adapter，可以先用 identity projection 或极小线性投影。

---

### Stage T1：Anchor-Preserving Token Update

如果 T0 有正趋势，进入状态递推。

推荐形式：

```text
H_t = LN(H_anchor + beta * H_state_t)
```

其中：

```text
H_state_t = (1 - g_t) * H_state_{t-1}
          + g_t * Adapter(H_blip - H_anchor)
```

解释：

```text
H_anchor 永远存在，提供目标身份；
H_state_t 只表示动态状态补充；
beta 控制动态状态强度；
g_t 控制当前帧是否吸收 BLIP。
```

第一版可以固定：

```text
beta = 0.1 / 0.3
g_t = quality_gate_accept
```

---

### Stage T2：Learnable Token State Updater

只有 T0/T1 显示有效，再做可学习版本。

形式：

```text
g_t = MLP([
    language_internal_consistency,
    tracker_confidence,
    trigger_features,
    score_delta,
    state_delta_norm
])
```

更新：

```text
H_state_t = (1 - g_t) * H_state_{t-1}
          + g_t * Adapter(H_blip - H_anchor)
```

训练目标可包括：

```text
tracking loss
score-space auxiliary loss
state smoothness regularization
identity consistency regularization
```

这一步不是当前立即执行的目标。

---

## 5. 与当前文本级方案的对照实验

建议对比三类策略。

### A. Hard Replacement

```text
gate accept:
  prev_description = blip_description
```

优点：

```text
直接使用 BLIP 当前描述。
```

缺点：

```text
容易整句漂移；
一次错误接受会污染后续状态。
```

### B. Text-level Anchor-Preserving

```text
gate accept:
  prev_description = anchor_description + selected BLIP words
```

优点：

```text
保留 anchor 主体；
降低整句替换风险。
```

缺点：

```text
仍然有语法和词筛选问题；
可能加入背景/上下文词；
BERT token 长度不可控。
```

### C. Token-level Anchor-Preserving

```text
gate accept:
  H_state_t = update(H_state_{t-1}, H_blip, H_anchor)
  H_lang_t  = [H_anchor ; H_state_t]
```

优点：

```text
避免文本拼接；
避免硬删词/加词；
更接近端到端连续状态更新；
可后续训练。
```

缺点：

```text
可解释性较弱；
需要处理 language token mask / position embedding / BERT output 接入；
可能破坏原 DUTrack 语言 token 分布。
```

---

## 6. 关键工程检查点

实现 token-level update 前必须确认以下内容。

### 6.1 DUTrack 语言输入位置

需要确认：

```text
language text 在哪里 tokenizer；
BERT embeddings / BERT encoder 输出在哪里；
language position embedding 在哪里加；
language tokens 在哪里送入 multimodal fusion；
language mask 如何构造。
```

目标是找到可以插入：

```text
H_lang_t
```

的位置。

### 6.2 token shape 与 mask

需要记录：

```text
H_anchor.shape
H_prev.shape
H_blip.shape
language_attention_mask.shape
有效 token 数
padding token 位置
```

如果 anchor / BLIP token 长度不同，第一版建议统一到固定长度：

```text
max_len = 16
```

保持和原 DUTrack 一致。

### 6.3 position embedding

如果直接替换 BERT output token，需要明确：

```text
position embedding 是否已经包含在 BERT output 中；
DUTrack 是否额外加 language position embedding；
替换后的 H_lang_t 是否还需要位置编码。
```

避免重复加或漏加 position embedding。

### 6.4 状态递推是否污染 batch

tracking 是逐序列递推，必须保证：

```text
每个序列有独立 H_state_t；
新序列初始化 H_state_0；
batch 测试时不同 sequence 不混状态。
```

---

## 7. 诊断指标

token-level 更新不像文本更新那样可读，因此必须增加诊断指标。

### 7.1 状态强度

```text
state_delta_norm = ||H_state_t - H_state_{t-1}||
anchor_delta_norm = ||H_lang_t - H_anchor||
blip_delta_norm = ||H_lang_t - H_blip||
```

### 7.2 状态相似度

```text
cos(H_lang_t, H_anchor)
cos(H_lang_t, H_prev)
cos(H_lang_t, H_blip)
```

### 7.3 更新频率

```text
token_state_update_count
frames_since_last_token_update
g_t_mean
g_t_max
```

### 7.4 跟踪效果

```text
score_gap
mean_iou
success_auc
precision
normalized_precision
```

### 7.5 稳定性

```text
state_norm_explosion_flag
token_nan_flag
language_attention_entropy
```

---

## 8. 当前不建议做的事情

暂时不要：

```text
1. 直接训练复杂 token updater；
2. 直接删除自然语言输入通道；
3. 完全绕过 anchor tokens；
4. 用 BLIP tokens 覆盖 anchor tokens；
5. 在 token state 不稳定前接入 LMQ query prior；
6. 用小集合弱标签训练最终 gate。
```

当前阶段只是验证：

```text
token-level anchor-preserving update 是否比文本级更新更稳定。
```

---

## 9. 推荐最小实验包

### Experiment 1：文本级基线

```text
E1-A: no update / anchor only
E1-B: hard replacement
E1-C: text anchor_state_gate
```

### Experiment 2：token-level 混合

```text
E2-A: H_mix = (1-alpha)H_prev + alpha H_blip
E2-B: H_mix = H_anchor + alpha(H_blip - H_anchor)
```

测试：

```text
alpha = 0.1 / 0.3 / 0.5
```

### Experiment 3：gate-controlled token update

```text
E3:
H_mix = H_anchor + g_t * alpha(H_blip - H_anchor)
g_t = quality_gate_accept
```

### Experiment 4：状态递推

```text
H_state_t = (1-g_t)H_state_{t-1} + g_t(H_blip - H_anchor)
H_lang_t = H_anchor + beta H_state_t
```

测试：

```text
beta = 0.1 / 0.3
```

---

## 10. 判断标准

如果 token-level 方案相比文本方案：

```text
1. false_accept 后的后续漂移更少；
2. score_gap 更稳定；
3. IoU/AUC 不下降或提升；
4. 状态更新不产生 token norm 异常；
5. 对困难序列比 hard replacement 更稳；
```

则说明：

```text
语言状态更适合在 token / embedding 层维护。
```

如果无改善，则说明：

```text
当前主要问题不是文本拼接，而是 BLIP 候选质量或 DUTrack 对语言状态变化不敏感。
```

---

## 11. 后续长期方向

如果 token-level update 有效，可以继续发展为：

```text
Identity-Preserving Latent Language State Updater
```

最终形式：

```text
H_id = Encode(anchor)
H_obs = Encode(BLIP/current)
H_state_t = Update(H_state_{t-1}, H_obs, H_id, confidence)
H_lang_t = Fuse(H_id, H_state_t)
```

其中：

```text
H_id:
  长期主体身份，不被覆盖。

H_state_t:
  可更新状态记忆。

confidence:
  控制更新强度。
```

这才是比文本替换更合理的动态语言更新。

---

## 12. 一句话总结

当前应从：

```text
维护 prev_description 字符串
```

转向：

```text
维护 anchor-preserving language token state
```

核心思路是：

```text
anchor tokens 保主体；
BLIP tokens 提供动态状态残差；
状态更新在 embedding/token 层完成；
文本层 hard replacement / 词拼接只作为 baseline。
```

这样可以避免自然语言拼接和词筛选规则过度复杂化，并为后续真正的 latent language state updater 打基础。
