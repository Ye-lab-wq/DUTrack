# DUTrack 模态对齐问题与 Tracking-Specific Evidence Calibration 修正版

## 1. 修正后的核心立场

前一版文档强调了 **Tracking-Specific Evidence Space**，但表述上容易让方案看起来像是在 DUTrack backbone 后面临时加一个小 adapter，并围绕大量负对照反复试探。结合后续反思，现在需要重新收紧主线：

> DUTrack 已经在 backbone 中通过 joint self-attention 将 language tokens、template tokens 和 search tokens 放在一起做隐式融合。问题不是“语言完全没有进入 backbone”，而是这种隐式融合缺少 tracking posterior 约束，导致语言没有稳定转化为目标定位证据。

因此，新的方案不应被理解为：

```text
忽略前面大层 backbone，只在 head 前补一个 adapter
```

而应被理解为：

```text
在 DUTrack 已有隐式多模态融合之后，引入一个 tracking-specific evidence calibration 层，
将 backbone 输出的 fused language / template / search representations 显式重组为目标条件化证据，
再作用到 localization head。
```

也就是说，完整方法主张是：

> **Implicit multimodal fusion is not enough; tracking-specific evidence calibration is needed before localization.**

中文表述：

> **仅有隐式多模态融合还不够，需要在定位头之前进行面向跟踪任务的证据校准。**

---

## 2. 当前 DUTrack 语言路径的真实问题

### 2.1 语言已经进入 backbone，但以弱形式进入

当前 DUTrack 中，语言 token 并不是完全缺失，而是以如下方式进入模型：

```text
language tokens + template tokens + search tokens
        ↓
joint transformer self-attention
        ↓
backbone fused representations
        ↓
search tokens → tracking head
```

这说明：

- 语言已经参与 backbone 内的 token interaction；
- 但这种交互是隐式的；
- tracking loss 只监督最终 bbox；
- 模型没有被显式要求学习“语言证据如何支持目标区域、抑制干扰物、影响 center score”。

因此，当前问题不是：

```text
没有语言输入
```

而是：

```text
语言输入没有成为 localization posterior 的有效条件变量。
```

---

### 2.2 语言编码本身也偏弱

当前项目中语言部分并没有使用完整 `BertModel` 做深层上下文编码，而是主要使用：

```text
BertTokenizer + BertEmbeddings + position embedding
```

这意味着语言 token 更接近：

```text
词 / 子词 embedding + 位置 embedding
```

而不是：

```text
上下文化语言表示
视觉接地语言表示
tracking-aware language evidence
```

因此，富文本描述虽然被输入模型，但其内部的主体、属性、空间关系并不一定被充分建模。

---

### 2.3 语言容易被视觉 shortcut 旁路化

从数学上，当前模型容易退化为：

```text
p(bbox | X, Z, L) ≈ p(bbox | X, Z)
```

其中：

- `X` 是 search visual tokens；
- `Z` 是 template visual tokens；
- `L` 是 language tokens。

由于 template-search visual matching 已经很强，模型可以主要依赖视觉路径完成跟踪，而不必真正使用语言。

这解释了语言敏感性实验中的现象：

```text
normal / generic / shuffled / wrong language 差距很小
```

以及 HOOT 中：

```text
wrong category 会造成一定下降，但 generic language 反而优于类别名 fallback
```

这说明语言路径不是完全不存在，而是没有稳定形成有效定位证据。

---

## 3. 前期实验教训与新方案的关系

### 3.1 TE Policy 的教训

前期实验尝试：

```text
language → attention keep / policy → 修改 backbone attention
```

结果：

- keep map 有时可见；
- 但 center score 变化弱；
- 中间层扰动容易被后续层和 head 吸收。

对应新方案：

> 不再只在 backbone 中间层做 attention perturbation，而是在 backbone 已融合输出之后进行 tracking-specific evidence calibration，使语言证据更直接地影响定位头。

---

### 3.2 Score Prior / GSB 的教训

前期实验尝试：

```text
language → score prior
global pooled language → score bias
```

结果：

- 全局语言向量太粗；
- gate 和 score_bias 容易接近无效；
- 有效语言信息在词级 / 片段级，而不是整句 global vector。

对应新方案：

> 不再把语言压成单一全局向量，而是保留 token / phrase level evidence，并在 target-conditioned evidence space 中校准后使用。

---

### 3.3 Word-Level Evidence 的教训

前期实验发现：

- oracle 条件下，部分 word-level signal 有正向作用；
- deploy 条件下，raw word-visual similarity 不可靠；
- 词权重排序改善并不一定转化为 IoU 收益。

对应新方案：

> 不再直接相信 BERT token 与 visual token 的原始相似度，而是学习 tracking-specific projection，使语言和视觉在目标条件化证据空间中比较。

---

### 3.4 LMQ 的教训

前期实验尝试：

```text
language-only multi-query prior
```

结果：

- K query 语义坍缩；
- 多个 query attend 到相同 language token 子集；
- language-only decomposition 缺少视觉约束。

对应新方案：

> evidence 不应由 language-only query 自行分解，而应由 template target prototype 和 fused visual context 条件化。

---

### 3.5 Language State Updater 的教训

前期实验尝试：

```text
H_anchor / H_prev / H_cand
per-token cross-frame state maintenance
```

结果：

- source gate no-op；
- candidate absorption 很弱；
- alignment entropy 高；
- 多辅助 loss 也无法解决结构不稳。

对应新方案：

> 不再维护跨帧 per-position token state，不做 anchor/prev/candidate token_i 对齐。后续若做语言更新，应在 evidence-level 做增量吸收，而不是 token-position state update。

---

### 3.6 BLIP 整句替换的教训

当前 baseline 中 BLIP 更新相当于：

```text
L_state ← BLIP caption
```

问题：

- BLIP caption 不保证 target-centric；
- 不保证 identity-preserving；
- 容易生成背景词、场景词或错误主体；
- 整句替换容易丢失初始身份锚点。

对应新方案：

> Stage-1 不引入 BLIP 或语言更新。后续若使用 BLIP，也只能作为 candidate evidence generator，不应作为 language state replacement。语言更新应在 evidence calibration 有效后再讨论，不应和第一阶段混在一起。

---

## 4. 修正后的总体架构定位

实现上，第一版可以落为一个 lightweight head-side calibration block；但方法上不能被描述成“绕开 backbone 的普通小 adapter”。更准确的描述是：它作用在 DUTrack backbone 已经融合后的 multimodal representations 上，在定位头前做 tracking-specific evidence calibration。

整体可描述成三层结构：

```text
Layer 1: DUTrack Implicit Multimodal Fusion Backbone
Layer 2: Tracking-Specific Evidence Calibration
Layer 3: Localization Posterior Head
```

### 4.1 Layer 1：DUTrack 隐式多模态融合

```text
L, Z, X → joint transformer → H_L, H_Z, H_X
```

作用：

- 保留 DUTrack 原有 backbone；
- 让 language / template / search token 先进行基础隐式交互；
- 继承原模型视觉跟踪能力。

这一层不是被忽略，而是作为 fused representation provider。

---

### 4.2 Layer 2：Tracking-Specific Evidence Calibration

这是新增核心层。

输入：

```text
H_L: backbone-fused language representations
H_Z: backbone-fused template / memory representations
H_X: backbone-fused search representations
```

目标：

```text
将隐式融合表示重组为 target-conditioned tracking evidence
```

核心思想：

```text
template target prototype 提供目标身份锚点；
language representations 提供候选语义证据；
search representations 提供当前位置证据；
evidence calibration 判断哪些信息支持目标、抑制干扰物、或应被忽略。
```

输出：

```text
H_X_calibrated
```

也就是被 evidence calibration 调整后的 search representation。

---

### 4.3 Layer 3：Localization Posterior Head

```text
H_X_calibrated → center / size / offset head → bbox
```

这里的关键不是 attention 图是否好看，而是：

```text
目标区域相对于 hard negative / distractor 区域的定位后验是否被改善。
```

---

## 5. 数学表述

当前 DUTrack 可以简化为：

```text
H = F_backbone([q, L, Z, X])
Y = Head(H_search, H_q)
loss = L_track(Y, bbox_gt)
```

问题是：

```text
L_track 只监督最终 bbox，
不直接监督 language token 与目标区域之间的对应关系。
```

在不训练 backbone 的前提下，TEC 的数学本质不应被理解为“重新学习完整视觉语言对齐”，而应被理解为：

```text
在 frozen backbone 给出的 fused representations 上，
学习一个窄的、面向 localization posterior 的任务相关校准项。
```

DUTrack baseline 可以看作先给出一个视觉主导的定位响应：

```text
S^0 = Head_base(H_X, H_q)
s_i^0 = S^0_i
```

其中 `i` 是 search feature map 上的位置，`s_i^0` 指原 DUTrack center head 输出的当前位置响应。注意这里不强行把它定义为 spatial softmax probability；当前 DUTrack 的 center head 使用的是原有 score map / size map / offset map 训练范式，TEC 不能改写这一范式。

TEC 要学习的不是全局图文匹配，也不是完整 phrase grounding，而是一个语言和目标条件化的 head-pre 后验校准：

```text
L_src = L_raw
     or Fuse(L_raw, H_L)

ΔH_i = TEC_theta(H_X_i, L_src, H_Z)

H_X_calibrated_i = H_X_i + γ · ΔH_i
S^TEC = Head_base(H_X_calibrated, H_q)
```

这里默认选择 **head-pre feature calibration**，而不是直接给 `score_map` 加语言 prior。这样可以避免退化成前期失败过的 score prior / GSB 新版本。

对应的实现形式可以写成：

```text
H_L, H_Z, H_X = F_backbone(L, Z, X)
L_raw = RawLanguageEmbedding(L)

L_src = L_raw
     or Fuse(L_raw, H_L)

z_proto = TargetPool(H_Z)

E_L = φ_L(L_src, z_proto)
E_X = φ_X(H_X, z_proto)

M = EvidenceInteraction(E_X, E_L)

H_X_calibrated = H_X + γ · M

Y = Head(H_X_calibrated)
```

如果为了诊断写成校准项，本质应理解为：

```text
S^TEC = Head_base(H_X + TEC_theta(H_X, L_src, H_Z))
```

而不是：

```text
S^TEC = S^0 + language-only score prior
```

因此 `φ_L, φ_X` 不是为了学习通用语义空间，而是为了学习一个 **narrow task-specific evidence space**：

```text
只对“当前位置是不是目标”这个 tracking posterior 有用；
只需要区分 target / distractor / background；
不追求开放词汇语义对齐；
不追求完整 token-level grounding 可解释性。
```

目标不是学习：

```text
word k 是否对应 region i
```

而是学习：

```text
某个语言 / 视觉证据是否会改变 region i 成为目标的 posterior。
```

---

## 6. 与 word-region grounding 的区别

本方案不是简单的：

```text
G_{k,i} = language token k grounds to visual region i
```

因为 tracking 中的语言作用不止是区域对应：

- subject / local target word 可能对应目标区域；
- attribute word 可能用于区分同类干扰物；
- spatial relation word 可能对应区域之间关系；
- context word 有时是噪声，有时是关键参照物；
- wrong word 不一定应强行负向打分，可能应被识别为低可靠证据并忽略。

因此，本方案关注的是：

```text
value(e_m | target prototype, search scene, history)
```

而不是固定词语角色。

这避免了回到：

```text
subject floor
context cap
state word whitelist
space word threshold
```

这类规则堆砌。

---

## 7. 实现方向：frozen-backbone posterior calibration

当前资源约束下，不应重训 backbone，也不应在 backbone 内部继续插多层模块。TEC 的设计目标是：

```text
冻结 DUTrack backbone；
保留原有隐式多模态融合；
只在 localization head 前学习一个窄的后验校准函数。
```

因此，TEC 不是为了做“大而全”的模态对齐，而是为了学习：

```text
region i 是否因为当前语言证据和目标原型而更像目标？
```

这是一种 narrow task-specific alignment，只服务于 target-vs-distractor posterior。

更准确的实现命名：

```text
Tracking-Specific Evidence Calibration Block
```

最小结构：

```text
z_proto = TargetPool(H_Z)

L_src = L_raw
     or Fuse(L_raw, H_L)

L_e = LN(W_l L_src + W_lz z_proto)
X_e = LN(W_x H_X + W_xz z_proto)

A = softmax((X_e W_q)(L_e W_k)^T / τ)
M = A (L_e W_v)

H_X_calibrated = H_X + γ · W_o M
```

这里要强调：

- `H_Z, H_X` 是 backbone 融合后的视觉表示；
- `L_src` 不应完全依赖 `H_L`，因为 `H_L` 可能已经在弱融合中被稀释；
- `z_proto` 不是额外视觉旁路，而是从 fused template representation 中提取的目标身份锚点；
- `A` 不是最终解释性 grounding，而是 evidence interaction；
- `H_X_calibrated - H_X` 只应被理解为 posterior correction，不应被包装成完整语义对齐；
- 最终评价看 localization posterior，而不是 attention 可视化。

`L_src` 第一版建议采用更稳的来源：

```text
优先：L_src = L_raw
可选：L_src = Fuse(L_raw, H_L)
暂不建议：L_src = H_L only
```

其中 `L_raw` 指进入 backbone 前的浅层 language token embedding，保留原始语言信号；`H_L` 只作为可选补充，表示 backbone 隐式融合后的上下文语言状态。

为了避免 TEC 变成另一个大模型，第一版应限制容量：

```text
低维 evidence projection；
小残差系数 γ；
不改 backbone；
不直接改 score_map；
不引入 BLIP；
不引入 token-level pseudo label；
不做多分支规则融合。
```

`TargetPool(H_Z)` 第一版应保持可控，优先选择一种简单稳定的定义：

```text
方案 1：template token 中心区域平均；
方案 2：使用已有 template bbox mask 做加权平均；
方案 3：多 memory template 时，对每帧 target pool 后再平均。
```

如果当前训练接口不方便传 bbox mask，先用中心区域平均，避免把 Step-1 变成数据接口改造实验。

---

## 8. 实验思路的重新修正

前一版实验仍然容易落入 adapter variant 消融。修正后，实验主线必须围绕一个问题展开：

```text
冻结 backbone 后，
TEC 能否作为 head 前后验校准项，
让语言从弱条件变量变成 localization posterior 的有效证据？
```

因此第一阶段不是做一组小模块横向比较，而是做一个主实验：

```text
DUTrack baseline
    vs
DUTrack + TEC posterior calibration
```

其他对照只作为健康检查，不作为主线。

### 8.1 主验证 1：baseline 隐式融合语言敏感性弱

目的：

```text
证明 DUTrack 原有 implicit fusion 没有稳定利用语言。
```

已有结果可以支持：

```text
OTB-Lang: normal / generic / wrong 差距很小
HOOT: wrong 有一定影响，但 generic 反而优于类别名
```

该部分作为问题诊断，不再继续大规模重复。

---

### 8.2 主验证 2：evidence calibration 后语言成为有效条件变量

目的：

```text
证明 evidence calibration 后，模型对正常语言、通用语言、错误语言的响应被拉开。
```

期望从：

```text
baseline:
normal ≈ generic ≈ wrong
```

变成：

```text
ours:
overall:
normal 与 wrong 的 gap 大于 baseline

language-needed subset:
normal 优于 generic / wrong

generic:
作为 low-information baseline，不强制总是低于 normal
```

这不是为了追求小差值，而是证明：

```text
语言从可忽略条件变量，变成会影响 localization posterior 的条件证据。
```

---

### 8.3 主验证 3：hard-negative / language-needed 场景改善

目的：

```text
证明 evidence calibration 改善的是目标与干扰物的判别，而不是普通特征适配。
```

重点分析：

- same-class distractor；
- occlusion；
- target/background ambiguity；
- fine-grained phrase description；
- OTB-Lang 中 wrong/generic 掉点明显的序列；
- HOOT 中 wrong category 影响明显的序列。

指标：

```text
target-hard negative score gap
score peak sharpness
target region score
distractor region score
```

---

### 8.4 主验证 4：TEC 是否成为有效后验校准项

目的不是比较一堆 adapter，而是确认 TEC 没有学成 no-op。

```text
TEC 的输出是否改变 posterior？
这种改变是否与语言和目标原型相关？
```

重点看：

```text
Δ_i 是否非零；
Δ_i 是否集中在 target / hard-negative 判别区域；
wrong language 是否改变 Δ_i；
generic language 是否接近 low-information correction。
```

只有当主实验出现正信号后，才有必要补 ordinary adapter 等结构对照。

---

## 9. 消融策略的收缩

消融不应成为主线。第一阶段只保留主实验和必要健康检查。

建议保留：

| 对照 | 目的 |
|---|---|
| DUTrack baseline | 原始隐式融合 |
| DUTrack + TEC | frozen-backbone 后验校准 |
| wrong / generic language | 检查语言因果作用 |
| hard-negative subset | 检查判别性提升 |

暂不建议第一轮加入：

```text
ordinary adapter
no-z-proto
all-one gate
tau variant
gamma variant
多层插入
```

这些可以作为后续定位问题的工具，但不能成为第一个实验的主线。

---

## 10. 边界条件与必须遵守的准则

### 10.1 不否定 backbone

本方案不是绕开 backbone，而是建立在 backbone 隐式融合表示之上。

必须表述为：

```text
implicit fusion + explicit evidence calibration
```

而不是：

```text
backbone 没用，直接 head 前补模块
```

---

### 10.2 不做多层插入

第一版只在 backbone 输出后、tracking head 前做 calibration。

原因：

```text
前期 TE policy 已说明，中间层扰动可能无法稳定传到 score map。
```

---

### 10.3 不做语言更新

第一阶段只验证：

```text
语言 evidence 是否能影响 localization posterior
```

不做：

```text
BLIP candidate 注入
language memory update
conservative update
```

语言更新应在 evidence calibration 有效后再讨论。后续若重新引入 BLIP，只允许把 BLIP caption 拆成 candidate evidence source；不允许回到整句替换。

---

### 10.4 不加复杂辅助 loss

第一版只用 tracking loss。

不加：

```text
grounding loss
token reliability loss
candidate absorb loss
multi-condition pseudo label
```

否则会重回规则堆砌。

同时必须保留 no-op 失败判据：

```text
gamma 长期接近 0
feature_delta_norm 接近 0
adapter gradient 很小
normal / wrong / generic gap 不变
TEC-on 与 TEC-off posterior 无差异
```

如果出现这些现象，不能直接否定 evidence calibration 假设，更可能说明 tracking loss 仍然绕开了语言路径。此时可以保留后路，但范围必须严格限定：

```text
只允许轻量 ranking / contrastive regularization；
不允许 token-level pseudo label；
不允许 candidate absorb label；
不允许 subject floor / context cap / whitelist 等规则堆砌。
```

---

### 10.5 不做词语分类规则

不手工定义：

```text
subject / attribute / context / state / space
```

不引入：

```text
subject floor
context cap
state whitelist
spatial threshold
```

第一阶段优先让 evidence 作用由 target condition 和 tracking loss 学习。若出现明确 no-op，只能考虑第 10.4 中限定的轻量 ranking / contrastive regularization，不能回到 token-level pseudo label 和词类规则。

---

### 10.6 不把负对照当训练规则

wrong/generic 只用于评价模块是否真正使用语言，不参与 loss 构造。其他健康检查只有在定位 no-op 原因时使用，不能扩展成消融主线。

---

### 10.7 不把 TEC 做成 Score Prior / GSB 新版本

前期 `Score Prior / GSB` 的问题在于：

```text
language/global vector → score bias
```

这条路径容易退化成信息量不足的语言先验，或者复制 base score 的低对比度偏置，无法稳定提供 target-centric 判别证据。

因此 TEC 第一版不直接对 `score_map` 做：

```text
S_final = S_base + language_prior
```

而应优先做：

```text
H_X_calibrated = H_X + γ · TEC(H_X, L_src, H_Z)
S_final = Head_base(H_X_calibrated)
```

区别在于：

| 旧 Score Prior / GSB | TEC |
|---|---|
| 从语言或全局语言向量直接生成 score bias | 在 head 前校准 search feature |
| 容易是 language-only prior | 显式依赖 `H_X_i + L_src + z_proto` |
| 可能绕开目标局部证据 | 必须以 search location 为条件 |
| 直接改 score，容易破坏 head 语义 | 保持原 head 训练范式 |

如果后续为了诊断引入 score/logit correction，也必须是有界、小幅、location-conditioned residual，不能成为主方法。

---

### 10.8 不完全依赖被弱融合稀释后的 `H_L`

如果 DUTrack 的隐式融合本来就没有稳定利用语言，那么 backbone 输出中的 `H_L` 可能已经被视觉主路径稀释。此时 TEC 再只从 `H_L` 挖语言，会存在逻辑风险：

```text
弱语言融合 → H_L 语言信息不足 → TEC 仍然没有可靠语言源
```

因此 TEC 的语言源应写成：

```text
L_src = L_raw
     or Fuse(L_raw, H_L)
```

其中：

- `L_raw` 是进入 backbone 前的浅层 language token embedding；
- `H_L` 是 backbone 融合后的 language slice；
- 第一版优先使用 `L_raw`，保证 TEC 能拿到未被融合稀释的语言信号；
- 若使用 `Fuse(L_raw, H_L)`，应保持轻量残差融合，不做复杂 language updater。

不建议第一版使用：

```text
L_src = H_L only
```

---

### 10.9 不改变原 DUTrack Head 的 Score 定义

文档中的 `posterior` 是方法论表述，不等于要把原 head 改成 spatial softmax。

当前 DUTrack center head 有自己的训练和推理范式：

```text
score_map / size_map / offset_map
focal loss on score_map
inference: score_map with hann window + max location
```

因此：

```text
s_i^0
```

应理解为原 head 在位置 `i` 的 score response 或对应 logit/activation，而不是新定义的 spatial softmax probability。

第一版 TEC 必须保持：

```text
Head_base 不改；
loss 不改；
score_map 语义不改；
只改变 head 前输入特征 H_X。
```

这样才能保证改动是在 baseline 训练范式上的后验校准，而不是重写一个新的 score head。

---

## 11. 修正后的方法叙事

可以将方法叙事写成：

> DUTrack already introduces language into the backbone through joint self-attention, but this implicit fusion is weakly constrained by the final tracking loss and does not reliably make language a discriminative condition for localization. Instead of replacing the backbone or updating language captions directly, we propose a tracking-specific evidence calibration module. It operates on the backbone-fused language, template, and search representations, uses the target prototype to condition language-search evidence interaction, and produces calibrated search features before the tracking head.

中文：

> DUTrack 已经通过 joint self-attention 将语言引入 backbone，但这种隐式融合只受最终跟踪损失间接约束，难以稳定让语言成为定位判别条件。我们不替换 backbone，也不直接更新 caption，而是在 backbone 融合后的语言、模板和搜索表示上引入 tracking-specific evidence calibration。该模块利用目标原型条件化语言与搜索证据交互，并在跟踪头前生成校准后的 search features。

---

## 12. 最终方案定位

最终方案不应被理解成：

```text
一个绕开 backbone 的普通 head-side 小 adapter
```

而应被理解成：

```text
DUTrack 隐式融合后的任务显式证据校准层
```

更准确地说：

```text
实现上：lightweight head-side calibration block
方法上：operating on backbone-fused multimodal representations
关键点：target-conditioned evidence interaction
```

完整定位：

```text
DUTrack implicit multimodal backbone
        +
Tracking-Specific Evidence Calibration
        +
Localization posterior head
```

目标：

```text
让语言不再只是 concat 到 backbone 中的弱条件变量，
而是成为能够改变 target-vs-distractor localization posterior 的有效证据。
```

核心原则：

```text
不重做 backbone；
Stage-1 不做 BLIP / language update；
只允许 tracking loss，必要时仅考虑轻量 ranking / contrastive；
不做词类规则；
不陷入碎消融；
用架构级验证证明 implicit fusion 不足，而 evidence calibration 有效。
```

---

## 13. 第一个实验方案：Frozen-Backbone TEC Posterior Calibration

### 13.1 实验目标

第一个实验只回答一个问题：

```text
在不训练 DUTrack backbone 的前提下，
head 前 TEC 能否学习到任务相关的语言-目标后验校准？
```

这不是 adapter 消融实验，也不是完整视觉语言对齐实验。它的数学目标是：

```text
原 DUTrack head response S^0
    ↓
TEC 根据 H_X、L_src 和 z_proto 校准 head 前特征
    ↓
原 DUTrack head 在 H_X_calibrated 上输出 S^TEC
```

也就是说，TEC 要学的是一个窄的 localization posterior correction，而不是通用语义对齐。

---

### 13.2 固定部分

必须固定：

```text
DUTrack backbone
BERT / BertEmbeddings 路径
BLIP 更新路径
template-search 主视觉路径
原 center head 结构与 score 定义
```

第一轮优先固定 head，只训练 TEC：

```text
freeze backbone
freeze original localization head
train TEC only
```

这样 TEC 的作用更接近真正的 posterior correction：

```text
baseline head 给出 s_i^0；
TEC 只能通过校准 H_X 改变原 head 输出。
```

如果 TEC-only 明确 no-op，当前不立刻进入 head 微调；应先回到 evidence representation、语言输入质量和 hard-negative 诊断，避免把普通 head adaptation 误判为语言接地。

---

### 13.3 插入位置

在 `DUTrack.forward()` 中，backbone 输出之后、head 前插入。

当前路径：

```text
feat_last = backbone(z, x, l)

H_X = feat_last[:, -feat_len_s:]
q   = feat_last[:, :1]

att = H_X · q
opt = H_X * att
score / size / offset = head(opt)
```

TEC 路径：

```text
feat_last = backbone(z, x, l)

L_raw = language token embedding before backbone fusion
H_L   = feat_last[:, 1:17]
H_X = feat_last[:, -feat_len_s:]
H_Z = feat_last[:, 17:-feat_len_s]
q   = feat_last[:, :1]

L_src = L_raw
     or Fuse(L_raw, H_L)

H_X_calibrated = TEC(H_X, L_src, H_Z)

att = H_X_calibrated · q
opt = H_X_calibrated * att
score / size / offset = head(opt)
```

这里 `H_L = feat_last[:, 1:17]` 依赖当前 tokenizer `max_length=16`。后续如果改语言长度，必须把 `lang_len` 配置化。

实现时需要从 backbone 中返回或缓存 `L_raw`。这不是新语言模块，而是把已有 `BertEmbeddings + position embedding` 的输出暴露给 TEC 使用，避免 TEC 只依赖可能被弱融合稀释的 `H_L`。

---

### 13.4 TEC 模块设计

第一版保持低容量：

```text
z_proto = TargetPool(H_Z)

L_src = L_raw
     or Fuse(L_raw, H_L)

L_e = LN(W_l L_src + W_lz z_proto)
X_e = LN(W_x H_X + W_xz z_proto)

A = softmax((X_e W_q)(L_e W_k)^T / τ)
M = A (L_e W_v)

H_X_calibrated = H_X + γ · W_o M
```

设计约束：

- `TargetPool(H_Z)` 第一版用中心区域 template token 平均；
- `L_src` 第一版优先用 `L_raw`；
- `Fuse(L_raw, H_L)` 只作为可选轻量残差融合；
- 不使用 `H_L only` 作为第一版语言源；
- `γ` 初始化为小非零值，例如 `1e-3`；
- evidence hidden dim 可以低于 512，例如 128 或 256；
- `τ` 固定，不做温度搜索；
- 不加 gate variant；
- 不直接修改 `score_map`；
- 不引入 spatial softmax；
- 不接 BLIP；
- 不写回语言状态；
- 不引入 token-level label。

这个模块要学的是：

```text
language evidence 在 target prototype 条件下，
对每个 search location 的 posterior correction。
```

---

### 13.5 训练方式

训练数据使用正常语言描述，不构造 wrong/generic 训练样本。

```text
train input:
template image
search image
normal language
gt bbox

loss:
原 DUTrack tracking loss
```

训练时保持原 head 输出和损失定义：

```text
不改 score_map / size_map / offset_map；
不把 score_map 改成 spatial softmax；
不新增 language score prior。
```

第一轮不加：

```text
ranking loss
contrastive loss
grounding loss
pseudo label
词类规则
```

如果 TEC-only no-op，先排查实现和梯度；只有在确认 tracking loss 无法激活 TEC 时，才允许轻量 ranking / contrastive regularization。

该后路必须满足：

```text
只基于 region-level target-vs-distractor score；
不构造 token-level pseudo label；
不做 subject/context/state 词类规则；
不做 candidate absorb label。
```

---

### 13.6 评价设计

第一轮不做 adapter variant 消融，只比较：

| 组别 | 含义 |
|---|---|
| A0 DUTrack baseline | 原始模型 |
| A1 DUTrack + TEC | frozen-backbone head-pre feature calibration |
| A1-wrong | A1 推理时换 wrong language |
| A1-generic | A1 推理时换 generic language |

评价数据：

```text
OTB-Lang
HOOT-all 或 HOOT balanced subset
language-needed / hard-negative subset
```

language-needed subset 可以来自：

```text
wrong language 掉点明显的序列；
same-class distractor；
target/background ambiguity；
fine-grained phrase 有实际区分作用的序列。
```

---

### 13.7 关键指标

常规指标：

```text
AUC
Precision
Normalized Precision
```

后验校准指标：

```text
normal - wrong gap
normal - generic gap
target-hard negative score gap
score peak sharpness
target region score mean
distractor region score mean
```

TEC 健康指标：

```text
feature_delta_norm = ||H_X_calibrated - H_X||
gamma
TEC parameter gradient norm
attention entropy
z_proto variance
L_src sensitivity: L_raw vs wrong/generic language
```

这些指标的作用不是做碎消融，而是确认 TEC 是否真的参与了 posterior correction。

---

### 13.8 预期结果

理想结果：

```text
A1 在 hard-negative / language-needed subset 上优于 A0；
normal - wrong gap 大于 baseline；
target-hard negative score gap 变大；
feature_delta_norm 非零；
gamma 不塌到 0。
score_map 仍保持原 DUTrack head 的数值语义。
```

可接受结果：

```text
整体 AUC 基本持平，
但语言敏感性和 target-vs-distractor posterior 更清楚。
```

失败结果：

```text
A0 与 A1 完全无差异；
normal / wrong / generic 仍重合；
feature_delta_norm ≈ 0；
TEC gradient 很小；
z_proto 几乎无方差或无作用。
wrong/generic 改变 L_src 但不改变 TEC 输出。
```

失败后处理：

```text
先查 token slicing；
再查 TargetPool；
再查 gamma 和梯度；
再尝试 TEC + head 训练；
最后才允许轻量 ranking / contrastive；
不允许回到 token-level pseudo label 或规则堆砌。
```

---

## 14. Stage-1.5：评估边界澄清与结果归因

Stage-1 已经完成了一个关键验证：在冻结 backbone 和冻结 head 的严格设置下，TEC 能在 OTB-Lang 上改善整体定位表现，说明 **head 前 posterior feature calibration 这个位置是有效的**。

但 Stage-1 结果也同时说明：

```text
normal / wrong / generic 的差异没有稳定拉开；
paired hard-negative gap 中 normal 对 generic / wrong 的优势很弱；
HOOT 中 BLIP-normal 比 class fallback 更合理，但 normal / wrong / generic / initial-only BLIP control 仍未稳定分离。
```

因此，Stage-1.5 的作用不是继续扩大消融，而是给 Stage-1 结果划定解释边界：

```text
Stage-1 证明了 localization-aware posterior calibration 有效；
Stage-1 没有证明稳定的 language-grounded calibration 已经成立。
```

换句话说，Stage-1.5 是 **Evaluation Boundary Clarification**，不是新的方法阶段，也不是新的训练阶段。

---

### 14.1 HOOT 评估口径边界

当前 HOOT 结果不能只看一张标准 AUC 表。HOOT 的标注天然比 OTB-Lang 更复杂：

```text
aa_bb: axis-aligned bounding box
rot_bb: rotated bounding box
occ_masks / attributes: 遮挡与可见性属性
```

当前 DUTrack 本地接入方式是：

```text
HOOTDataset: aa_bb -> xywh
extract_results: axis-aligned IoU / center precision / normalized precision
```

也就是说，当前结果是 **DUTrack / pytracking 风格的 axis-aligned OPE 口径**，适合做内部 normal / wrong / generic 对比，但不能直接声称是 HOOT 完整官方 rotated-box / occlusion-aware 评估。

因此 HOOT 结果必须加边界条件：

```text
当前 HOOT 结果用于重遮挡场景下的相对趋势分析；
不用于宣称官方 rotated / occlusion-aware benchmark 性能；
不用于证明完整语言 grounding 已经成立。
```

---

### 14.2 aa_bb OPE 与 rot_bb polygon IoU 的诊断边界

当前 tracker 输出仍然是：

```text
pred_bbox = [x, y, w, h]
```

因此 rot_bb 诊断只能做：

```text
pred xywh -> axis-aligned polygon
GT rot_bb -> rotated polygon
polygon IoU(pred, rot_bb)
```

这个指标比 `aa_bb` 更严格，但要注意：

- 它会惩罚 DUTrack 输出形式本身不是旋转框；
- 它不能直接拿来和专门输出 rotated box 的方法公平比较；
- 它只适合作为内部诊断：看 TEC / language variant 的相对排序是否只是 `aa_bb` 口径带来的假象。

当前 HOOT-all 默认分析中：

```text
normal aa_bb AUC: 63.75
normal rot_bb AUC: 60.82
```

rot_bb 口径整体下降是合理的，因为预测框仍是 axis-aligned box。更重要的是，normal / wrong / generic 在 rot_bb 下仍然非常接近：

```text
normal  rot_bb AUC: 60.82
wrong   rot_bb AUC: 60.12
generic rot_bb AUC: 61.13
```

因此 HOOT 当前结论应写成：

```text
BLIP 初始化能提升 HOOT normal 的整体可用性，
但无论 aa_bb 还是 rot_bb 诊断口径，
normal 与 wrong / generic 仍未拉开稳定语义差距。
```

不能强行解释为 TEC 已经实现 HOOT 上的语言接地。

---

### 14.3 occlusion-aware 分组诊断边界

HOOT annotation 中已有：

```text
absent
full_occlusion
partial_obj_occlusion
similar_occluder
cut_by_frame
```

当前 `HOOTDataset` 已经把这些属性挂到：

```text
seq.hoot_occlusion_attributes
```

因此可以用以下 scope 做诊断：

```text
all_frames
visible
visible_no_occlusion
any_occlusion
full_occlusion
partial_obj_occlusion
similar_occluder
cut_by_frame
```

这些分组的解释优先级是：

| scope | 用途 |
|---|---|
| `visible_no_occlusion` | 普通可见定位能力 |
| `any_occlusion` | 遮挡总体鲁棒性 |
| `similar_occluder` | 最接近语言可能发挥作用的同类 / 相似干扰场景 |
| `full_occlusion` | 严重遮挡，通常不应用来证明语言接地 |
| `cut_by_frame` | 视野截断与框口径敏感性 |

当前 HOOT-all 默认分析中，normal 在 `similar_occluder` 上并没有稳定优于 wrong / generic / initial-only BLIP control：

```text
similar_occluder aa_bb AUC:
normal    52.62
wrong     51.02
generic   52.45
initial-only BLIP control 53.07

similar_occluder rot_bb AUC:
normal    49.00
wrong     47.98
generic   48.81
initial-only BLIP control 49.61
```

这说明 normal 在相似遮挡场景里相对 wrong 有一点信号，但它与 generic / initial-only BLIP control 非常接近，不能证明动态语言更新或 TEC 语言接地已经稳定成立。

更稳的解释是：

```text
HOOT 中 BLIP / 初始描述对可用语言状态很重要；
但当前模型主要仍是视觉定位 + 后验校准；
语言语义只在部分 case 中提供增益。
```

---

### 14.4 Stage-1.5 的必须边界

Stage-1.5 只做结果归因与评估口径澄清，不引入新的训练目标。

必须遵守：

```text
不新增模型结构；
不新增 loss；
不新增 token-level pseudo label；
不把 wrong / generic / initial-only BLIP control 作为训练样本；
不把 HOOT rot_bb 诊断包装成官方 rotated benchmark；
不把 occlusion split 中的局部优势解释成全局语言接地；
不继续扩大 zero-language / shuffled-z-proto / random-language 等碎消融矩阵。
```

Stage-1.5 的输出只用于给 Stage-2 定边界：

```text
Stage-2 不能被写成“语言 grounding 已经成立后的增强阶段”；
Stage-2 必须重新回到 evidence space 本身的结构设计。
```

---

## 15. 第二阶段实验：Explicit Tracking Evidence Layer

### 15.1 为什么 Stage-2 不能回到旧 TEC residual 路线

Stage-2 不再沿用旧的 TEC residual 变体，也不把 head 微调作为当前主线。

原因是：如果 A2 提升，我们仍然无法判断收益来自：

```text
1. 更好的 tracking-specific evidence space；
2. 普通 head adaptation；
3. head 对 TEC residual 的吸收；
4. 语言无关的 localization calibration；
5. 数据集/评估口径带来的局部增益。
```

这样实验又会回到：

```text
A0 / A1 / A2 / wrong / generic
```

然后围绕小差值反复解释，偏离原始主线。

因此 Stage-2 的主问题不是：

```text
换一个更强 adapter 后 AUC 能不能继续涨？
```

而是：

```text
能否把 Stage-1 的 residual calibration 升级为显式 tracking evidence layer，
让 language、target prototype、search region 在同一个任务证据空间中交互？
```

---

### 15.2 Stage-2 的核心目标

Stage-2 的目标是从 Stage-1 的：

```text
H_X' = H_X + γ · TEC(H_X, L_src, H_Z)
```

升级为显式的：

```text
region-level tracking evidence representation
```

也就是说，Stage-2 要让模型显式产生：

```text
E_x(i): search region i 的目标条件化视觉证据
E_l(j): language token j 的目标条件化语义证据
c_i: region i 的 tracking evidence representation
s_i: region i 的 bounded calibration strength
ΔH_i: 由 evidence representation 产生的 feature correction
```

Stage-2 要回答的是：

```text
在 frozen DUTrack backbone 之后，
能否构建一个与 localization posterior 同层面的 evidence layer，
而不是只做一个隐式 feature residual adapter？
```

这个目标与文档主线一致：

```text
Implicit multimodal fusion is not enough;
tracking-specific evidence calibration is needed before localization.
```

---

### 15.3 Stage-2 的方法设计

Stage-1 TEC 的核心形式是：

```text
z_proto = TargetPool(H_Z)

L_e = LN(W_l L_src + W_lz z_proto)
X_e = LN(W_x H_X + W_xz z_proto)

A = softmax((X_e W_q)(L_e W_k)^T / τ)
M = A (L_e W_v)

H_X_calibrated = H_X + γ · W_o M
```

Stage-2 保留这个基本方向，但把 `M` 从隐式 attention aggregation 升级为显式 evidence representation。

这里必须先解决两个退化风险：

```text
风险 1：
s_i 退化成新的 visual score prior / targetness gate。

风险 2：
C_i 走视觉 shortcut，忽略 language evidence。
```

这两个风险不应只靠后期诊断发现，而应在数学建模层面限制信息路径：

```text
不让 s_i 直接从 raw H_X_i 或 E_x_i 产生；
不让 ΔH_i 存在 visual-only residual path；
不把 attention pooled language M_i 直接当完整 evidence；
显式构造 region-conditioned language residual D_i；
当 language attention 退化为均匀或语言无信息时，D_i 接近 0，Stage-2 残差自然变弱。
```

修正后的建议结构：

```text
z_proto = TargetPool(H_Z)

E_x_i = LN(W_x H_X_i + W_z^x z_proto)
E_l_j = LN(W_l L_src_j + W_z^l z_proto)

A_ij = masked_softmax_j((E_x_i W_q)(E_l_j W_k)^T / τ, M_L)
M_i  = Σ_j A_ij · W_v E_l_j

M_0      = masked_mean_j(W_v E_l_j, M_L^sem)
D_raw_i  = M_i - M_0
d_mag_i  = clamp(||D_raw_i||_2 / sqrt(d_e), max=τ_d)
D_gate_i = ||D_raw_i||_2 / (||D_raw_i||_2 + eps_d)
D_dir_i  = LN(D_raw_i)
D_i      = D_gate_i · d_mag_i · D_dir_i

G_i  = MLP_g([D_i, E_x_i ⊙ D_i]) + D_i
C_i  = MLP_c(G_i)

u_i  = W_e C_i
s_i  = 1 + β · tanh(u_i)
ΔH_i = W_o C_i

H_X'_i = H_X_i + γ · s_i · ΔH_i
```

其中：

```text
D_raw_i: unnormalized region-conditioned language residual
D_i: magnitude-preserving region-conditioned language evidence residual
G_i: language-visual interaction feature
C_i: region-level tracking evidence representation
u_i: shared evidence coordinate / readout for region comparability
s_i: bounded calibration strength
ΔH_i: evidence-conditioned feature correction
```

这里的关键变化是：

```text
M_i 不是 evidence 本身，只是 region i 从 language tokens 中取到的候选语言证据；
M_0 是当前语言的 neutral evidence baseline；
D_raw_i = M_i - M_0 才表示 region i 相对中性语言上下文获得了什么 token-selective evidence；
D_i 保留 D_raw_i 的幅值，并通过 D_gate_i 抑制极小 D_raw_i 被 LayerNorm 重新放大；
G_i 增加 +D_i 作为语言残差启动路径，避免短训早期只靠乘性交互启动过慢；
G_i 不直接拼接 raw E_x_i，避免给 visual-only residual 留旁路；
C_i 只从 interaction feature G_i 得到；
u_i = W_e C_i 提供跨 region 共享 evidence coordinate，使 C_i 具备可比较读出；
s_i 只调节 residual strength，不直接作用 score map。
```

`s_i` 不是人为规则 gate，也不是 targetness score。它的作用是：

```text
当 language-visual evidence 支持校准时增强 residual；
当 evidence 不可靠或接近中性时降低 residual；
从结构上缓解 Bird1 / HOOT 这类 harmful calibration。
```

建议第一版取较小的 `β`，例如：

```text
β = 0.25
s_i ∈ [1 - β, 1 + β]
```

初始化必须保持中性：

```text
W_e.weight = 0
W_e.bias = 0
=> u_i = 0
=> s_i = 1

W_o 使用受控小随机初始化，当前 σ_o = 1e-3；
γ 使用 A1 可比初始化，当前 γ_init = 0.01；
不能同时把 W_o 和 γ 初始化为 0，否则 Stage-2 容易 no-op。
```

`W_o` 小随机 + `γ` 非零必须非常可控。当前设定让初始 residual 很小但非零：不会一开始破坏 frozen head 的输入分布，同时仍保留 `W_o / γ / C_i` 的梯度路径。`s_i` 初始为 1 是中性设置，但必须记录 `s_i` 偏离 1 的幅度，防止 reliability 分支长期不动。

这样 `s_i` 不能像 score prior 一样把某些位置直接打开或关闭，只能在小范围内调节 evidence residual 的强度。真正改变 localization posterior 的仍然是：

```text
H_X'_i = H_X_i + γ · s_i · ΔH_i
Head_base(H_X')
```

也就是说，Stage-2 的数学约束是：

```text
score prior 被结构性禁止；
visual-only adapter 被结构性削弱；
language token-selective evidence 通过 D_raw_i / D_i 被显式放到残差路径中。
```

---

### 15.4 与 Stage-1 TEC 的关系

Stage-2 不是推翻 Stage-1，而是结构升级。

| 项目 | Stage-1 TEC | Stage-2 Evidence Layer |
|---|---|---|
| 核心形式 | attention aggregation + residual | evidence representation + reliability residual |
| 输出 | `ΔH_i` | `D_raw_i`, `D_i`, `C_i`, `u_i`, `s_i`, `ΔH_i` |
| 语言作用 | 隐式参与 residual | 显式参与 region-level evidence |
| 可靠性 | 仅靠小 `γ` 保守 | 由 bounded `s_i` 学习残差强度 |
| 目标 | 验证 head-pre 位置有效 | 建立同一 tracking evidence space |
| 是否改 head | 不改 | 默认仍不改 |

Stage-1 已经证明 head-pre 位置有效；Stage-2 要证明的是：

```text
显式 evidence representation 是否比隐式 residual calibration 更符合 tracking posterior 建模。
```

---

### 15.5 训练设置

Stage-2 默认仍然冻结：

```text
backbone
BertEmbeddings / description_patch_pos_embed
original tracking head
```

默认训练：

```text
Stage-2 Evidence Layer only
```

不建议一开始放开 head。

原因：

```text
如果同时放开 head，A2 的收益会再次混入 head adaptation，
不利于判断 evidence layer 本身是否成立。
```

因此 Stage-2 的主设置是：

```text
freeze backbone
freeze head
train explicit evidence layer only
```

工程上必须与 Stage-1 TEC 分离：

```text
Stage-1: TrackingEvidenceCalibration
Stage-2: TrackingEvidenceLayer
```

不建议在旧 TEC 类里继续增加 `mode=evidence_layer`。否则后续会变成多个 mode / variant 的碎消融，也不利于汇报时区分 residual TEC 与 explicit evidence layer。

Stage-2 的初始化也要保持可解释：

```text
若结构不兼容，则 residual branch 初始化为近 no-op；
γ 使用与 A1 可比的初始化；
W_e 零初始化，使 s_i 初始为 1；
W_o 使用受控小随机初始化，保证 residual path 有可训练梯度但初始幅度很小；
D_NORM_EPS 抑制极小 D_raw 下的方向噪声；
A1 / A2 使用相同训练预算。
```

否则 A2 的结果会混入：

```text
随机初始化难度；
短训 5 epoch 收敛不足；
残差幅度变化；
```

从而难以判断 explicit evidence layer 是否真的优于 Stage-1 residual calibration。

---

### 15.6 损失函数

Stage-2 继续使用原 DUTrack tracking loss：

```text
loss = giou + l1 + focal
```

不新增：

```text
InfoNCE
ranking loss
contrastive loss
grounding loss
token-level pseudo label
candidate absorb loss
subject / context / state 词类规则
BLIP 更新监督
```

理由：Stage-2 的核心是结构升级，不是用新的 loss 去强行拉语言差异。

如果 Stage-2 失败，也不应立刻加 loss，而应判断：

```text
C_i 是否退化为常量？
D_raw_i 是否长期接近 0？
s_i 是否退化为常量，或者长期贴近边界？
ΔH_i 是否仍有有效幅度？
evidence gap 是否没有改善？
```

只有确认 tracking loss 完全无法激活 evidence layer，才可以在后续阶段考虑极轻量 region-level ranking；但它不属于 Stage-2。

---

### 15.7 与前期失败实验的区别

#### 不同于 Score Prior / GSB

Stage-2 不做：

```text
language/global vector → score bias
```

而是：

```text
H_X_i + z_proto + token-level L_src → D_i / G_i / C_i → head-pre residual
```

它仍然是 region-wise、target-conditioned、head-pre feature calibration。

---

#### 不同于 LMQ

Stage-2 不做 language-only query decomposition。

前期 LMQ 的失败在于：

```text
K query 语义坍缩；
language-only decomposition 缺少视觉约束。
```

Stage-2 的每个 evidence representation 都以 search region `H_X_i` 和 target prototype `z_proto` 为条件：

```text
C_i = f(G_i),   G_i = g(D_i, E_x_i ⊙ D_i),   D_raw_i = M_i - M_0
```

因此它不是语言自己分解语义，而是 tracking region 主动索取与目标相关的语言 evidence。

---

#### 不同于 Language State Updater

Stage-2 不维护跨帧 language state，不做：

```text
H_anchor / H_prev / H_cand
per-token alignment
candidate absorb label
```

也不引入 BLIP 整句替换。

BLIP 后续只能作为 candidate evidence source，但不属于 Stage-2。

---

### 15.8 评价设计

Stage-2 的评价不再围绕大量 normal / wrong / generic 消融展开，而围绕 evidence layer 是否成立。

主表只保留：

| 组别 | 含义 |
|---|---|
| A0 | DUTrack baseline |
| A1 | Stage-1 TEC residual calibration |
| A2 | Stage-2 explicit evidence layer |

主数据集：

```text
OTB-Lang
```

补充数据集：

```text
HOOT-aa_bb / HOOT-rot_bb diagnostic
HOOT occlusion-aware split
```

HOOT 只用于边界分析，不用于宣称完整官方 HOOT benchmark。

---

### 15.9 必要边界检查

Stage-2 只保留最小语言边界检查：

```text
A2-normal
A2-wrong
A2-generic
```

但这些不是主实验表的中心，而是边界检查：

```text
如果 A2-normal 明显优于 wrong/generic，说明 evidence layer 开始更好利用语言；
如果 A2-normal、wrong、generic 同步提升，说明 evidence layer 主要增强 localization calibration；
```

这些结果不能反过来主导方法设计。

但边界检查不能省略。否则 A2 只能被解释为：

```text
explicit evidence layer improves localization posterior
```

不能进一步解释为：

```text
explicit evidence layer improves language-grounded calibration
```

---

### 15.10 Evidence-layer 诊断指标

Stage-2 要少看无穷小消融，多看 evidence layer 是否真的形成。

建议记录：

```text
D_raw_norm_mean
D_raw_norm_std
D_raw_spatial_std
D_gate_mean
D_gate_std
D_norm_mean
D_norm_std
D_spatial_std
G_norm_mean
G_norm_std
G_spatial_std
C_norm_mean
C_norm_std
C_spatial_std
u_mean
u_std
s_mean
s_std
s_min
s_max
s_deviation_mean
s_target_region_mean
s_hard_negative_mean
ΔH_norm_before_strength
ΔH_norm_after_strength
ΔH_to_feature_ratio
evidence_attn_entropy_norm
evidence_target_hardneg_gap
head_readout_delta_score
valid_semantic_token_count
d_norm_eps
residual_init_scale
```

这里的主线指标是 evidence layer 自身是否形成：

```text
D_raw_i 不长期接近 0；
D_gate_i 不长期接近 0；
D_i 是否携带保幅值的 region-conditioned language residual；
G_i 是否体现 language-visual interaction；
C_i 是否形成非平凡 region-level evidence representation；
s_i 的 std 不为 0，但不贴近 1 ± β；
s_i 是否只在有限范围内调节 residual strength；
ΔH_i 是否由 C_i 产生有效 head-pre residual。
```

不能把 `C_i` 向量本身直接叫作 score。离线分析中如果要比较 target / hard-negative，应明确诊断标量，建议按优先级使用：

```text
primary evidence scalar:
e_i = |u_i| 或 ||D_i|| · ||G_i|| · ||C_i||

calibration scalar:
q_i = |s_i - 1| · ||ΔH_i||

head readout diagnostic:
Δscore_i = score_i(H_X + γ · s_i · ΔH_i) - score_i(H_X)
```

其中 `target region` 和 `hard negative region` 只在离线诊断中由 GT / score map 定义，不进入推理，也不构造训练 pseudo label。`Δscore_i` 只用于验证 frozen head 是否读出了 evidence-conditioned residual，不是 Stage-2 的主对象。

attention 诊断必须使用归一化熵：

```text
H_norm = H(attn) / log(valid_semantic_token_count)
```

长期接近 `1` 表示近似均匀，长期接近 `0` 表示坍缩到单一 token；二者都不是健康的 language evidence interaction。

`M_0` 的 `masked_mean` 只允许使用 valid semantic tokens：

```text
排除 PAD / CLS / SEP；
保留真实语义 token；
记录 valid_semantic_token_count；
当 valid_semantic_token_count 过低时，Stage-2 只能解释为弱类别条件校准。
```

如果 Stage-2 成功，应该看到：

```text
目标区域 evidence 更高；
hard negative 区域 evidence 更低；
Bird1 / HOOT harmful case 中 s_i 不应盲目贴近上边界；
Δscore 与 evidence map 有一致性，但不作为 evidence 的定义。
```

这比单纯 normal/wrong/generic 的 AUC 差值更能说明 evidence layer 是否成立。

---

### 15.11 可视化要求

Stage-2 只做少量 case-level 可视化，不做大规模图表堆叠。

选择：

```text
Human4 / Coupon / Skater: positive case
Bird1: harmful calibration case
HOOT similar_occluder: 遮挡 / 干扰 case
```

可视化：

```text
A0 score map
A1 score map
A2 evidence map D / G / C / s
A2 score map
GT / prediction trajectory
```

目标是展示：

```text
A2 是否形成了更清楚的 target evidence；
A2 是否在 failure case 中降低不可靠 residual；
A2 是否比 A1 更像 evidence layer，而不是普通 residual adapter。
```

---

### 15.12 通过条件

Stage-2 通过条件不能混成一个结论，必须分成三层判断。

第一层是 **Structure pass**：证明 explicit evidence layer 本身没有塌缩，训练稳定，并且没有破坏 baseline tracking posterior。

第二层是 **Performance pass**：证明 explicit evidence layer 被 tracking head 读出来，整体性能接近或超过 Stage-1 TEC。

第三层是 **Language pass**：在前两层成立之后，再判断语言语义是否稳定参与 evidence representation。

也就是说：

```text
structure pass 可以成立，但 performance / language pass 仍然不成立。
```

这时结论只能写成：

```text
explicit evidence layer improves localization-aware evidence calibration
```

不能写成：

```text
explicit evidence layer solves language grounding
```

#### 结构证据

```text
D_raw / G / C / ΔH 非塌缩；
C_i 不是常量；
D_raw_i 不长期接近 0；
D_gate_i 不长期接近 0；
D_i / G_i 具有空间方差；
C_i 具有空间方差；
s_i 不贴边，也不退化为全局常量；
训练中后期 ΔH_i / H_X 保持在约 1%~10% 的可控范围；初始化瞬间可以更小；
attention normalized entropy 低于均匀但不坍缩到单一 token；
训练稳定；
A2 不明显低于 A0；
score map 不崩；
不存在 visual-only direct residual bypass。
```

#### posterior 证据

```text
A2 接近或超过 A1；
OP75 / Norm Precision 不低于 A1；
evidence_target_hardneg_gap 优于 Stage-1；
head_readout_delta_score 与 evidence map 方向一致；
hard-negative 局部改善；
Bird1 / harmful case 不进一步恶化；
```

结构证据成立只能说明：

```text
Structure pass
```

结构证据和 posterior 证据同时成立，才能说明：

```text
Performance pass
```

#### 语言边界证据

```text
若 normal / wrong / generic 同步提升，
则只能解释为 localization-aware evidence calibration，而不是语言 grounding。

A2-normal 不应系统性低于 wrong/generic；
A2-normal - wrong/generic 应在 language-needed / hard-negative subset 上优于 A1；
wrong/generic 应改变 D_raw_i 分布以及 D_i / G_i / C_i / s_i，而不是只改变 attention entropy 或最终 AUC 噪声；
normal 相对 wrong/generic 在 evidence representation 或 hard-negative case 上稳定更好。
```

只有语言边界证据也成立，才能写成：

```text
Language pass
```

否则 Stage-2 的正确结论仍然是：

```text
显式 evidence layer 改善了后验定位证据；
语言接地仍然只在局部 case 中有迹象，尚未稳定成立。
```

---

### 15.13 停止条件

如果出现以下情况，应停止 Stage-2，不继续加模块：

```text
C_i 接近常量；
D_raw_i 长期接近 0；
D_i / G_i / C_i 空间方差接近 0；
s_i 长期贴近边界，或者退化为全局常量；
A2 只提升所有语言模式，但 evidence_target_hardneg_gap 没有改善；
A2 在 HOOT / Bird1 上负迁移更重；
A2 的 score map 改善无法对应 evidence map；
normal / wrong / generic 的差异仍然完全随机；
```

触发停止条件后，不应直接进入：

```text
InfoNCE
ranking loss
BLIP candidate update
token-level grounding
subject/context/state 规则
```

而应回到：

```text
1. target prototype 是否有效；
2. L_raw 是否足以表达语言；
3. evidence layer 是否容量过小或过大；
4. tracking loss 是否仍绕开 evidence representation；
5. 是否需要极轻量 region-level ranking 作为后续阶段，而不是 Stage-2。
```

---

### 15.14 Stage-2 与后续阶段的关系

当前阶段只处理 explicit evidence layer 是否成立。若 Stage-2 已经通过 Structure / Performance pass 但 Language pass 仍弱，再考虑：

```text
Stage-3: anchor-preserving candidate evidence selection
```

也就是：

```text
保留 L_anchor；
BLIP 只作为 candidate evidence source；
candidate token 经过 target-consistency gate 后进入 evidence layer；
不做整句替换。
```

---

### 15.15 一句话总结

Stage-2 的最终定位是：

```text
从 Stage-1 的 head-pre residual calibration，
升级为显式 region-level tracking evidence layer。
```

它要证明的不是：

```text
语言 grounding 已经解决。
```

而是：

```text
在不重训 backbone 的前提下，
language、target prototype、search region 能否在同一 tracking-specific evidence space 中交互，
形成可用于 localization posterior 的 region-level evidence。
```

normal / wrong / generic 只是边界检查，不能再成为实验主线。

---

## 16. 当前阶段的最终实验路线

结合 Stage-1、Stage-1.5 与新的 Stage-2，当前路线应收束为：

```text
Stage-1:
验证 head-pre TEC residual calibration 位置有效。

Stage-1.5:
澄清 OTB / HOOT / rot_bb / occlusion split / hard-negative 的解释边界，
防止把 Stage-1 过度解释成 language grounding。

Stage-2:
构建 explicit tracking evidence layer，
让 language、target prototype、search region 在同一任务证据空间中交互。

Stage-3:
若需要增强语义，做 anchor-preserving candidate evidence selection，
而不是 BLIP 整句替换。
```

核心主线始终是：

```text
Implicit fusion is weak;
head-pre task-specific evidence calibration is necessary;
Stage-2 要把 calibration 从 residual adapter 推进到 explicit evidence layer。
```

这条线不再被 normal / wrong / generic 的细碎消融牵着走。

17. Stage-2R：Evidence-Unit Aware Tracking Evidence Calibration

17.0 审查后收紧的核心边界

当前版本方案方向正确，但 Stage-2R 必须进一步收紧为：

```text
Evidence Unit 重定义
```

也就是说，下一阶段不是继续修 token attention，而是回答一个更根本的问题：

```text
raw token 不是可靠 tracking evidence 单元；
能否把语言先组织成 phrase-aware evidence units，
再让 search region 在 target-conditioned evidence space 中读取这些 units？
```

因此，上一轮围绕旧 A2 做的：

```text
semantic mask
multi-slot reading
uniform mixing
```

只能被视为旧 A2 失败后的局部补丁或诊断工具，不应被写成 Stage-2R 的方法主线。

Stage-2R 必须新建独立模块：

```text
TrackingEvidenceUnitLayer
```

而不是继续在旧 `TrackingEvidenceLayer` 中增加 mode 或分支。这样可以在汇报时明确区分：

```text
Stage-2 old A2: token-conditioned evidence residual
Stage-2R: phrase/evidence-unit aware tracking evidence calibration
```

17.1 为什么需要 Stage-2R

旧 A2 暴露出三个核心问题：

1. raw token attention 会退化到固定 token，例如 `a/person/bird`；
2. `s_i` 会退化成全局 residual amplification；
3. normal / wrong / generic 没有形成稳定 evidence 差异。

这说明旧 A2 的问题不只是 attention 太尖，而是：

```text
language evidence unit 的定义太弱。
```

旧 A2 直接让 search region attend 到 raw token：

```text
region -> token
```

但 raw token 不是可靠 evidence 单元。孤立的 `a/the` 没有意义；孤立的 `on/of/in` 也未必有意义；但 `on the bike`、`head of the man`、`red car`、`black dog` 这类短语却可能是目标判别证据。

因此 Stage-2R 的主问题是：

```text
如何让模型读取 phrase-aware evidence unit，
而不是孤立 token shortcut？
```

17.2 Stage-2R 的核心目标

Stage-2R 要回答：

```text
在不重训 backbone、不改变主实验 BLIP 口径、不引入新监督的前提下，
能否把语言表示组织成 phrase-aware evidence units，
让 search region 基于 target prototype 读取这些 evidence units，
从而减少 fixed-token shortcut 和 global residual amplification？
```

Stage-2R 不是为了马上追求 AUC 超过旧 A2，而是为了验证：

1. top evidence unit 不再是 `a` 或固定裸 token；
2. target 和 hard-negative 不再 100% 读取同一个 evidence unit；
3. `s_i` 不再整体全部大于 1；
4. normal / wrong / generic 在 evidence-unit representation 上出现更合理差异；
5. AUC 不明显低于 A0，最好接近或超过旧 A2。

17.3 固定部分与允许变化

Stage-2R 主实验继续固定：

```text
DUTrack backbone
BertEmbeddings / language embedding path
tracking head
score_map / size_map / offset_map 定义
原 DUTrack tracking loss
```

Stage-2R 主实验不做：

```text
BLIP candidate 引入
BLIP 整句替换策略改动
language state update
token-level pseudo label
InfoNCE / contrastive / ranking loss
直接 score prior
backbone 微调
head 全量微调
```

主实验设置：

```text
freeze backbone
freeze head
train TrackingEvidenceUnitLayer only
```

如果 frozen-head 版本通过 Structure pass，但 tracking head 明显读不出 evidence residual，可以设第二档实验：

```text
freeze backbone
train TrackingEvidenceUnitLayer + score branch only
keep size_map / offset_map branch frozen
```

第二档只能作为 readout-capacity 检查，不能和主实验混在一张主表中，也不能同时改变 BLIP、语言更新、训练数据或 loss。优先开放 score branch 的原因是：当前问题首先表现为 localization posterior / score map 对 evidence 的读取不足，而不是 box regression 几何能力不足。

17.4 BLIP 口径必须保持一致

Stage-2R 主实验的 BLIP / language source 口径必须和 A0 / A1 / A2 保持一致。

也就是说，比较：

```text
A0 baseline
A1 TEC
A2 old evidence
A2R evidence-unit
```

时，只允许改变 evidence layer 本身，不允许同时改变：

```text
是否使用 BLIP
是否动态更新 caption
normal language 的来源
wrong / generic 的生成方式
```

如果 A0 / A1 / A2 normal 使用 dataset language 或 BLIP fallback，那么 A2R normal 也必须使用相同路径。如果要研究 BLIP no-update、BLIP candidate evidence 或 caption 更新策略，必须另列后续实验，不能混入 Stage-2R 主实验。

17.5 语言 evidence unit 的重新定义

旧 A2 的输入单位是：

```text
raw token
```

Stage-2R 的输入单位改为：

```text
phrase-aware evidence unit
```

其中 evidence unit 不是人工语义类别，也不是手写 subject / attribute / context 规则，而是由轻量局部语言建模得到。

第一版只做结构卫生，不做语义规则堆砌：

```text
anchor token:
    非 special；
    非 padding；
    非纯标点；
    非 article: a / an / the。

context token:
    anchor 周围窗口内的 valid token；
    可以包含 in / on / of / with / near / behind 等介词或关系词；
    但这些词第一版不单独作为 evidence unit 中心。
```

边界条件：

```text
不允许 a / an / the 成为独立 evidence；
不粗暴删除 in / on / of，因为它们可能参与短语结构；
不手工定义 subject / attribute / relation 类别；
不根据词性解析器或外部 NLP 工具决定 evidence 类别。
```

17.6 Evidence-unit pooling

给定原始语言 token：

```text
L = [l_1, l_2, ..., l_n]
```

对每个 anchor token `j`，构建局部 phrase/evidence unit：

```text
P_j = PhraseEvidencePool(l_{j-w}, ..., l_j, ..., l_{j+w})
```

第一版保持轻量：

```text
window size = 3
P_j = MLP([l_{j-1}, l_j, l_{j+1}])
```

或者等价实现为：

```text
P = Conv1D_window(L)
P_j = P[j] where j is anchor
```

但第一版不引入 parser，不引入复杂 language model，不做外部词法分析。

这样：

```text
head of the man
man on the bike
red car
black dog
object in the scene
```

都能通过局部上下文进入 evidence unit，而不是把 `of/on/in/the` 当成孤立 evidence token。

17.7 Stage-2R 方法公式

旧 A2 是：

```text
E_l_j = φ_l(L_j, z_proto)
A_ij = softmax(E_x_i · E_l_j)
M_i = Σ_j A_ij E_l_j
```

Stage-2R 改成：

```text
P_j = PhraseEvidencePool(L, anchor_j, context_window)

E_p_j = LN(W_p P_j + W_pz z_proto)
E_x_i = LN(W_x H_X_i + W_xz z_proto)

A_ij = masked_softmax_j((E_x_i W_q)(E_p_j W_k)^T / τ, M_P)

M_i = Σ_j A_ij · W_v E_p_j
M_0 = masked_mean_j(W_v E_p_j, M_P)

D_raw_i = M_i - M_0
D_i = magnitude_preserving_norm(D_raw_i)

G_i = MLP_g([D_i, E_x_i ⊙ D_i]) + D_i
C_i = MLP_c(G_i)

u_i = W_e C_i
u_i_centered = u_i - mean_i(u_i)

s_i = 1 + β · tanh(u_i_centered)
ΔH_i = W_o C_i

H_X'_i = H_X_i + γ · s_i · ΔH_i
```

其中关键变化有两个：

1. region 读取的是 phrase/evidence units，不是 raw tokens；
2. `s_i` 做 region-relative strength，不能全局抬高所有 residual。

17.8 为什么不把 semantic mask / multi-slot / uniform mixing 作为主线

Stage-2R 不把下面三件事作为核心方法：

```text
semantic token mask
multi-slot reading
uniform attention mixing
```

原因：

1. semantic mask 只能防止 `a/the` 作为裸 token 被读出，但不能构造 `head of the man` 这种 evidence unit；
2. multi-slot 只是增加读取通道，不解决语言 unit 本身弱的问题；
3. uniform mixing 只能防止数值上过尖，不能保证 attention 关注语义正确的 evidence；
4. slot 很可能复制同一个高响应 token，LMQ 已经显示过类似坍缩风险。

因此 Stage-2R 第一版：

```text
不用 multi-slot；
ATTENTION_UNIFORM_MIX = 0.0；
semantic mask 只作为 evidence-unit anchor 构造的结构卫生，不作为方法贡献。
```

如果训练早期直接 one-hot collapse，可以把 uniform mixing 作为 safety option：

```text
0.02 ~ 0.05
```

但它不进入主实验叙事。

17.9 训练设置

Stage-2R 主训练采用：

```text
正常语言训练；
不构造 wrong/generic 训练样本；
只用原 DUTrack tracking loss；
freeze backbone；
freeze head；
train TrackingEvidenceUnitLayer only。
```

训练轮数：

```text
3 ~ 5 epoch
```

checkpoint 保留策略要兼顾诊断和磁盘空间：

```text
短训期间临时保留 ep1 / ep3 / ep5；
完成诊断后只保留最佳或最新 checkpoint。
```

旧 A2 已经说明中间 epoch 可能比最终 epoch 更合理，后期可能出现 residual amplification，因此不能只看最后一轮。

17.10 关键日志指标

Stage-2R 不再只看 token attention，而要看 evidence-unit 诊断。

必须记录：

Evidence-unit 数量：

```text
valid_token_count
anchor_token_count
context_token_count
evidence_unit_count
low_evidence_unit_ratio
```

Evidence-unit attention：

```text
evidence_attention_entropy_norm
top1_evidence_weight
top1_evidence_text
target_hardneg_top1_same_ratio
```

Phrase/evidence 质量：

```text
top evidence unit examples
normal / wrong / generic top evidence distribution
```

Strength：

```text
s_mean
s_std
s_min
s_max
s_deviation_mean
```

Region-relative 后希望：

```text
s_mean 接近 1；
s_min 和 s_max 分布更对称；
不再全部大于 1。
```

Residual / head readout：

```text
delta_to_feature_ratio
enc_norm_before / after
head_att_mean / std
```

Evidence gap：

```text
target-hard_negative evidence gap
target-hard_negative Δscore gap
peak_inside_gt
```

17.11 评价设计

主表保持简洁：

| 组别 | 目的 |
| --- | --- |
| A0 baseline | 原始 DUTrack |
| A1 TEC | Stage-1 residual calibration |
| A2 old evidence | 旧 token-level evidence |
| A2R-FH | Stage-2R evidence-unit, frozen head |
| A2R-score | 第二档，只开放 score branch |

第一轮只跑：

```text
A2R-FH normal on OTB-Lang
```

如果 A2R-FH normal 不明显低于 A0，再跑：

```text
A2R-FH wrong
A2R-FH generic
hard-negative evidence diagnostics
attention / evidence-unit top examples
```

只有当 A2R-FH 通过 Structure pass，但 Performance pass 明显受限时，才进入第二档：

```text
A2R-score normal
```

A2R-score 的解释必须限定为：

```text
验证 frozen head 是否限制了 evidence residual readout。
```

不能把它直接等同于 evidence unit 本身更强。

17.12 预期结果

Stage-2R 的目标不是立刻 AUC 最高。

更合理的预期是：

Structure pass：

```text
top evidence 不再是 a；
target / hard-negative top evidence 不再 100% 相同；
s_i 不再整体全部 > 1；
attention 可以集中，但不能稳定塌到无意义 unit；
evidence unit 数量合理，不大量为空。
```

Performance pass：

```text
A2R-FH 不明显低于 A0；
如果接近或超过旧 A2，则非常好；
OP75 不明显下降。
```

Language pass：

```text
normal / wrong / generic 在 evidence-unit 分布上出现差异；
normal 在 hard-negative subset 上比 wrong/generic 更合理；
generic 不再因为固定 residual shortcut 获得最高收益。
```

17.13 成功与失败判断

成功：

```text
A2R-FH normal 接近或超过旧 A2；
attention / evidence unit 不再 fixed-token collapse；
s_i 不再全局增强；
hard-negative evidence gap 有正常语言优势。
```

部分成功：

```text
AUC 略低于旧 A2；
但 evidence diagnostics 明显更合理；
normal/wrong/generic 的 evidence 表现更有差异。
```

这说明方向正确，但性能还需优化。

失败：

```text
AUC 低于 A0；
evidence unit 仍塌到固定词；
target / hard-negative 仍读同一 evidence；
s_i 仍整体增强；
normal / wrong / generic 完全无差异。
```

失败后不要立刻加 loss，应先看：

```text
evidence unit 构造是否过弱；
anchor token 是否太少；
phrase pooling 是否退化；
L_raw 本身是否不足；
frozen head 是否读不出 evidence residual；
是否需要更强语言编码，而不是继续调 attention。
```

17.14 边界条件

Stage-2R 必须遵守：

```text
不重训 backbone；
主实验不解冻 head；
第二档最多只开放 score branch；
不改变 BLIP / language source 口径；
不使用 BLIP candidate；
不做整句语言更新；
不引入 token pseudo label；
不手工定义 subject / attribute / context 规则；
不强制多峰；
不强制过滤所有介词；
不把 semantic mask / multi-slot / uniform mixing 写成核心方法；
不让 wrong/generic 参与训练。
```

Stage-2R 只解决一个问题：

```text
让语言 evidence 的基本单位从 raw token 变成 phrase-aware evidence unit。
```
