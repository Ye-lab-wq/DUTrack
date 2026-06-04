# DUTrack 后续实验主线与注意事项

## 1. 当前实验阶段的核心判断

本项目已经不适合继续采用“想到一个模块就加一个模块”的探索方式。前期实验已经排除了一批低收益方向，后续实验应转为：

> 链路拆解式验证：先判断 baseline 是否真正使用语言，再判断视觉是否能有效校准语言，最后判断保守语言更新是否优于 BLIP 整句替换。

当前主线应围绕 DUTrack baseline 的实际实现展开：

```text
DUTrack backbone
    +
初始语言描述 / 当前语言状态
    +
帧内双向视觉语言调制
    +
触发式保守语言更新
```

整体目标不是重新设计一个完全独立的跟踪器，而是在 DUTrack 已有动态语言更新框架上，解决两个关键问题：

1. 帧内语言和视觉交互不足；
2. 触发更新后 BLIP 整句替换不可靠。

---

## 2. 必须记住的已有实验教训

### 2.1 不再重复 TE Policy 类实验

已做内容：

```text
language tokens → keep / policy → 修改 Transformer attention
```

包括 post-softmax policy、pre-softmax bias、不同 query scope、不同 layer choice 等。

已有结论：

- keep map 有时明显；
- 但 center score 变化弱；
- 中间 attention 层修改距离最终 localization head 太远；
- 后续 Transformer 层和 head 容易吸收或抹平这种扰动。

后续原则：

> 不再继续调更多 layer、更多 top-k、更多 policy 形式。该方向只作为负面探索证据保留。

---

### 2.2 不再重复 Score Prior / GSB 类实验

已做内容：

```text
language → score prior
global pooled language → score bias
```

已有结论：

- 直接 score bias 路径更靠近输出，但 language prior 信息量不足；
- GSB 中 global pooling 后 gate 和 score_bias 基本不起作用；
- 全局语义向量太粗，无法承接词级或状态级语言增量。

后续原则：

> 不再继续调 GSB 的 beta、max_delta、gate_bias、pooling 方式。该方向作为“global language signal 不足”的证据。

---

### 2.3 不再重复 per-token Language State Updater

已做内容：

```text
H_anchor / H_prev / H_cand
cross-attn alignment
source gate
token absorb BCE
多个辅助 loss
```

已有结论：

- positive label 稀疏；
- source gate 学成 no-op；
- alignment entropy 高；
- relation attention 退化；
- candidate absorption 很弱；
- 多个辅助 loss 无法根治问题。

根本原因：

> BERT token position 不是稳定语义槽位。不同 caption 的第 i 个 token 不一定表达同一种语义，per-token position-wise language state maintenance 结构上不稳。

后续原则：

> 不再堆更多 gate、更多 loss、更多 AND 条件、更多 relation block。

---

### 2.4 不再重复 LMQ 多 query 语言先验

已有结论：

- K query 语义坍缩；
- 多个 query attend 到相似 language token；
- language-only semantic decomposition 缺少视觉约束。

后续原则：

> 如果未来再做 query / slot，必须引入视觉条件或竞争机制；不再单独做 language-only multi-query prior。

---

### 2.5 不再重复 BLIP 整句替换是否不可靠

当前 DUTrack 推理阶段的 baseline 更新方式是：

```text
触发 update_key
    ↓
BLIP 生成 candidate caption
    ↓
直接替换当前 description
```

已有判断：

- 整句替换容易丢失初始主体身份；
- BLIP 可能引入背景词、场景词或错误主体；
- 状态词可能有用，但整句替换把有用和有害信息一起引入。

后续原则：

> 不再大规模重复证明“整句替换不可靠”。最多补少量可视化 case 作为动机。

---

## 3. 后续实验的核心断点

后续只补三个关键断点，避免重复无用功。

### 3.1 断点一：DUTrack baseline 到底有多依赖语言？

DUTrack 已经将 language tokens 与 template/search tokens concat 后送入 backbone，但这不代表模型真的强依赖语言。

必须先做语言敏感性诊断：

```text
正常语言
错误语言 / shuffled language
generic language / empty-like language
```

目的：

> 判断当前 baseline 是否真的“吃语言”。

若错误语言几乎不影响结果，说明后续重点应放在增强帧内语言视觉交互；若错误语言明显伤害结果，说明语言状态更新才更有实际价值。

---

### 3.2 断点二：视觉能否端到端校准语言 token？

之前虽然使用过视觉证据，但多数是规则、后处理 reliability、pseudo label 或 AND 条件，不是结构中的可学习 Visual → Language 调制。

需要补一个最小实验：

```text
template visual prototype → recalibrate language tokens
```

但注意：

- 不写回长期 L_state；
- 只生成当前帧临时语言 L_frame；
- 只用 tracking loss；
- 不使用 pseudo label；
- 不使用 subject floor / context cap 等规则。

目的：

> 验证视觉特征能不能端到端校准语言 token，而不是依赖人工规则筛词。

---

### 3.3 断点三：保守语言更新是否优于 BLIP 整句替换？

不再直接做复杂 latent updater，而是先验证更新原则：

```text
No update
BLIP replace
Conservative text update
```

其中 Conservative text update 的核心是：

```text
保留主体身份
拒绝错误主体
只吸收可靠状态词 / 空间词 / 动作词
不整句替换
```

目的：

> 验证“主体锚定 + 增量吸收”这个原则是否比 BLIP 整句替换更稳。

只有这个原则有效，才值得进一步做可学习语言更新模块。

---

## 4. 推荐后续实验路线

### Step 1：Baseline 语言敏感性诊断

实验组：

```text
A0: DUTrack + 正常语言
A1: DUTrack + shuffled / wrong language
A2: DUTrack + generic language
A3: DUTrack + no update，仅使用初始语言
A4: DUTrack + trigger BLIP replace
```

关注指标：

- AUC / Precision / Normalized Precision；
- wrong language 是否明显掉点；
- generic language 是否接近正常语言；
- BLIP replace 是否优于 no update；
- update 触发次数；
- 更新前后 caption 主体是否漂移。

判断：

```text
如果 A1/A2 几乎不降：
    baseline 对语言不敏感，优先做帧内交互增强。

如果 A1/A2 明显下降：
    baseline 确实使用语言，保守语言更新更有意义。

如果 A4 低于 A3：
    BLIP 整句替换有害，应改为保守更新。
```

停止条件：

> 如果 baseline 对语言完全不敏感，不要直接做复杂语言更新，应先解决语言进入视觉/head 的有效路径。

---

### Step 2：Visual → Language 最小结构验证

实验组：

```text
B0: DUTrack baseline
B1: DUTrack + Visual → Language recalibration
B2: B1 + shuffled template prototype
B3: B1 + all-one gate
B4: B1 + wrong / shuffled language
```

建议结构：

```text
p_z = TargetPool(template_tokens)

g_i = sigmoid(MLP([l_i, p_z, l_i * p_z, l_i - p_z]))
delta_i = tanh(MLP([l_i, p_z, l_i * p_z, l_i - p_z]))

l_i^V = l_i + alpha · g_i · delta_i
```

注意：

- `l_i^V` 是当前帧临时语言；
- 不写回长期 L_state；
- 不引入 candidate caption；
- 不引入复杂辅助 loss；
- 先在 head 前或 fusion 后使用，避免插入过深。

判断：

```text
B1 > B0 且 B2/B3/B4 下降：
    Visual → Language 调制有效。

B1 ≈ B0：
    当前 V→L 结构无效，不急着做完整双向。

B2 不下降：
    说明模块没有真正利用视觉。

B4 不下降：
    说明模块没有真正利用语言。
```

停止条件：

> 如果 Visual → Language 分支没有正信号，不要直接叠双向结构；先检查插入位置、语言敏感性和 gate 是否启动。

---

### Step 3：帧内双向调制验证

实验组：

```text
C0: DUTrack baseline
C1: Language → Visual only
C2: Visual → Language only
C3: 并行双向调制 L↔V
C4: 普通 language cross-attn，无 visual gate
C5: shuffled visual / shuffled language 负对照
```

建议使用并行残差，不要强串行：

```text
X_L = X + gamma_x · CrossAttn(Q=X, K=L_state, V=L_state)

L_V = L_state + alpha · VisualCondDelta(L_state, Z)

X_final = X_L + eta · CrossAttn(Q=X_L, K=L_V, V=L_V)
```

注意：

- 不要一开始插入 backbone 多层；
- 优先在 head 前 / fusion 后；
- alpha、gamma、eta 小值初始化；
- 先只用 tracking loss；
- 不写回长期语言记忆。

判断：

```text
C3 > C1/C2 > C0：
    双向调制成立。

C3 ≈ C1：
    V→L 没有额外贡献。

C3 ≈ C2：
    L→V 路径不强。

C4 ≈ C3：
    visual gate 贡献不足。
```

停止条件：

> 如果 C1/C2 均无效，不要继续组合完整框架；说明当前语言视觉交互路径还没有打通。

---

### Step 4：保守语言更新原则验证

实验组：

```text
D0: No update，只用初始语言
D1: Trigger + BLIP replace
D2: Conservative text update
D3: Conservative update + wrong candidate check
```

Conservative text update 的基本原则：

```text
保留 L_anchor / L_state 中的主体身份；
拒绝与 anchor 主体冲突的 candidate；
从 L_cand 中只吸收状态词、动作词、空间词；
低置信或主体冲突时不更新；
更新后不允许整句替换。
```

注意：

- 这一步可以先用文本级启发式实现；
- 它不是最终方法，而是验证更新原则；
- 不要重新做 per-token latent state updater；
- 不要引入复杂 token alignment。

判断：

```text
D2 > D1：
    保守增量更新优于整句替换。

D0 ≥ D1：
    BLIP replace 确实有害。

D2 ≈ D0：
    保守更新至少避免了 BLIP 损害，但收益有限。

D2 < D0：
    当前更新策略仍会引入噪声，应收紧吸收条件。
```

停止条件：

> 如果 Conservative text update 都不能优于 BLIP replace 或 no update，不要急着做可学习 updater，应先分析 candidate caption 质量和更新触发时机。

---

### Step 5：最终组合验证

只有 Step 2/3/4 有正信号后再组合。

实验组：

```text
E0: DUTrack baseline
E1: 帧内双向调制 only
E2: 保守语言更新 only
E3: 帧内双向调制 + 保守语言更新
```

判断：

```text
E3 最好：
    完整框架成立。

E1 有效，E2 无效：
    主线应聚焦帧内双向调制。

E2 有效，E1 无效：
    主线应聚焦语言更新。

E1/E2 都无效：
    当前语言信息接入 DUTrack 的路径仍有问题。
```

---

## 5. 必须避免的重复实验

后续不要再做以下内容，除非明确发现代码 bug：

```text
继续调 TE policy 的层数、top-k、query scope；
继续调 GSB 的 beta / max_delta / gate_bias；
继续做 language-only LMQ K 值消融；
继续给 Language State Updater 加 loss；
继续构造更复杂 token absorb positive label；
继续做 anchor/prev/candidate token position-wise alignment；
继续依赖 subject floor / context cap 等硬规则修补模型；
继续把 BLIP candidate 整句替换作为主方法。
```

这些方向已经被前期实验基本排除。

---

## 6. 边界条件与注意事项

### 6.1 区分帧内临时语言与长期语言状态

必须区分：

```text
L_anchor：初始语言锚点，不允许被覆盖；
L_state：当前长期语言状态；
L_frame：当前帧临时视觉校准语言；
L_cand：BLIP candidate，只作为增量来源。
```

不要把 `L_frame` 直接写回 `L_state`。

---

### 6.2 不要把触发机制和更新机制混在一起

当前重点是语言自身更新机制，不是触发机制。

触发机制可以暂时沿用 baseline：

```text
area ratio
center displacement
confidence if available
```

更新机制才是重点：

```text
不整句替换；
主体保留；
状态/空间增量吸收。
```

---

### 6.3 所有新模块必须有负对照

每个模块至少配一个对应负对照：

```text
语言模块 → wrong / shuffled language
视觉调语言 → shuffled visual prototype
gate 模块 → all-one gate / no-gate
candidate 更新 → shuffled / wrong candidate
score bias → zero-delta / candidate shuffle
```

没有负对照的提升不可靠。

---

### 6.4 不要只看可视化

可视化可以辅助说明，但不能替代指标。

必须同时看：

```text
AUC
Precision
Normalized Precision
score peak
score peak sharpness
pos-neg gap
update frequency
caption drift cases
gate statistics
```

---

### 6.5 不要用过多辅助 loss

优先只用 tracking loss。

只有在最小结构有正信号但训练不稳定时，才考虑轻量正则。

禁止一开始加入：

```text
多重 auxiliary loss；
复杂 pseudo label；
多条件 AND label；
hard threshold rule；
大规模 hand-crafted word role rule。
```

---

### 6.6 插入位置要克制

优先位置：

```text
backbone fusion output 后
head 前 search tokens
```

谨慎位置：

```text
多层 Transformer 内部
早期 patch embedding 层
多层重复插入
```

原因：

> 之前 TE Policy 已说明，中间层扰动可能传不到最终 score。

---

### 6.7 所有实验必须能回答一个明确问题

每个实验开始前写清楚：

```text
这个实验验证什么？
成功说明什么？
失败排除什么？
下一步是什么？
```

禁止无目标地做模块堆叠。

---

## 7. 推荐最终最小实验包

如果资源有限，建议只做以下核心实验：

```text
1. Baseline 语言敏感性：
   normal / wrong / generic language

2. Visual → Language 最小验证：
   baseline / V→L / shuffled visual / all-one gate

3. 帧内双向调制：
   baseline / L→V / V→L / L↔V

4. 保守语言更新原则：
   no update / BLIP replace / conservative text update

5. 最终组合：
   baseline / bi-modulation / conservative update / both
```

这已经足够支撑一条完整科研主线。

---

## 8. 最终实验主线总结

后续实验不应重复旧模块，而应围绕三个核心问题展开：

```text
1. DUTrack baseline 是否真正依赖语言？
2. 视觉能否端到端校准语言 token？
3. 保守增量更新是否优于 BLIP 整句替换？
```

如果这三个问题成立，则完整框架可以自然定义为：

> 基于 DUTrack 的语言记忆驱动双向视觉语言校准框架。

最终目标是：

```text
初始语言作为身份锚点；
当前语言参与每帧跟踪；
语言调制视觉以增强定位；
视觉调制语言以抑制错误词；
BLIP 只作为候选增量来源；
更新时保留主体，吸收可靠状态/空间增量；
避免整句替换、global pooling、per-token position alignment 和规则堆砌。
```
