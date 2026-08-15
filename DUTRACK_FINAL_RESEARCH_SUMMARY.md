# DUTrack 视觉语言跟踪实验线最终总结

日期：2026-08-15

状态：研究线关闭；仅保留本总结作为科学结论与实验设计档案

## 1. 最终结论

DUTrack 语言实验线已经系统验证了从“图文表示是否含有信息”到“语言是否真正改善在线跟踪”的完整证据链。最强且可重复的结论是：

1. TNL2K 目标裁剪与描述中存在非随机的多模态信息；轻量辅助空间可以学到区域定位和同类细节区分。
2. 部分冻结表示和路由位置保留了语言特异性，反事实语言也确实会改变响应图、预测框和在线状态，因此语言路径不是数值上完全失活。
3. 这些表示收益没有稳定传递到 DUTrack 原生空间决策。对齐、语言广播、候选修正、token 增量、QK 空间偏置、决策适配器等方案均未形成可靠的 held-out caption-specific decision。
4. 在官方 EP47 原始路径上，正确静态语言没有优于纯视觉或人工审核的安全错误语言；完整在线累积后反而显著降低 AUC/精度并增加严重漂移。
5. 动态 BLIP 更新没有可靠优于纯视觉或跨序列打乱后的 BLIP 更新。它只优于有害的静态正确语言，说明主要作用更像是削弱旧语言干扰，而不是提供有效新语义。

因此，本研究线支持：

- 优化目标能够收敛；
- 辅助空间存在区域 grounding；
- 某些上游表示存在语言特异性；
- 原生路径存在语言到决策的因果耦合。

但不支持：

- DUTrack 已学会稳定使用正确描述区分视觉困难目标；
- 静态或动态语言带来受控的 tracking improvement；
- 继续增加对齐容量、修复 BLIP 或恢复 JA1--JA6 能自然解决问题。

项目由此停止以 DUTrack 作为语言机制开发 baseline，后续工作转向具有更显式视觉语言交互和目标解码路径的 JointNLT。

## 2. 基线、数据与证据规则

### 2.1 冻结基线

| 项目 | 冻结值 |
|---|---|
| 官方架构锚点 | `abd57a8` (`updatekey`) |
| 配置 | `experiments/dutrack/dutrack_384_full.yaml` |
| EP47 checkpoint | `DUTrack_ep0047.pth.tar` |
| checkpoint SHA-256 | `c0fa34aa8355b6c3fa6102ba7bfc16d8bf514858c26d8e2ecfb821b6e59e0fcd` |
| 参数量 | 105,298,437 |
| B0 模式 | 全参数冻结、eval、无 optimizer、无训练 loss |
| 正式随机种子 | `20260701` |
| B0 协议 SHA-256 | `7bbf5506ea4370690f96ff80700bb4f8fd800b66970631a33ba820c9233c70dd` |
| batch 约束 | 正式反事实推理严格 sequential batch=1 |

EP47 checkpoint 严格加载通过，没有 missing、unexpected 或 shape-mismatch key。正式 B0 协议包含 train 200 和 test 100 个 TNL2K 序列；100 个错误描述在查看模型输出前完成人工审核，最终映射无自配对、100 个来源唯一。

### 2.2 证据等级

本项目始终区分以下等级，不能从较低等级跨越推断较高等级：

1. **优化成功**：训练 loss 或指定 ranking objective 按预期改善。
2. **区域 grounding**：同一搜索区域中，语言关联表示将 GT 排在困难负样本或背景之前。
3. **语言特异性**：正确描述在 held-out 序列上稳定优于合适的错误描述。
4. **决策路径耦合**：反事实语言改变响应图、head 输入、预测框或在线状态。
5. **跟踪收益**：在受控评估中，AUC、precision 或漂移指标可靠改善。

训练 loss、全局 cosine、attention 可视化、PCA、特征发生变化或错误语言更差，都不能单独证明 tracking improvement。

### 2.3 数值与统计合同

- 正式推理采用单样本顺序 forward。Ampere TF32 下 batch=3 相对 batch=1 的最大 logit 差约 `3.4e-3`、box 差约 `9.8e-4`；关闭 TF32 后仍有 `1.62e-5`，超过预注册的 `1e-5`，因此没有放宽容差。
- B0 主要统计以序列为 bootstrap 单位，使用 10,000 次 paired cluster bootstrap。
- 视觉困难帧阈值只由 train split 的 `LANG_OFF` 响应确定，不能由语言或结果反向挑选。
- 正确语言必须同时击败 `LANG_OFF` 和合适的 wrong/shuffled control，才能声明语义价值。

## 3. 实验路线与阶段结果

### 3.1 Stage2：辅助表示、区域监督与原生决策接口

#### A-clean / A-all-frame 表示学习

- 冻结参考模型确认目标 crop-caption 对含有非随机多模态信号。
- 轻量对齐头在 held-out retrieval 上优于原始几何；A-all-frame held-out top-1 从 E0 的 `0.0008` 提高到 `0.0959`，同时保留代表帧排名。
- 加入区域监督后三个种子的 held-out GT-hard-negative margin 为 `+0.03195`，GT-background margin 为 `+0.03479`，region top-1 提升 `+0.10636`。
- 7,113 个同类细节配对的 GT-versus-distractor 平均 margin 为 `0.3594`，正率 `0.9363`；只使用类别名时 margin 为 `-0.0221`、正率 `0.4195`。

这些结果证明辅助空间的优化成功与区域 grounding，但没有证明 DUTrack 原生 head 使用了正确语言。

#### 原生 posterior 与 evidence coupling

- Stage2C 原生 posterior 中，same-category-wrong score L1 仅 `0.00195`，predicted-IoU difference `-0.00186`，GT-hard-negative gap delta `-0.00106`，方向错误且幅度很小。
- 保守 score-map evidence coupling 虽改变 margin，但三个种子的 exact argmax 只改变 `5/608`、`5/608`、`6/608`；部署强度下没有观察到 outside-to-inside rescue。
- Oracle alpha 可以修复部分 baseline-wrong case，但多数需要超过安全部署上限，只能说明容量存在，不能形成可用策略。
- 冻结 static routing 在上游有正确对打乱语言的差异：raw margin `+0.06612`、GT mass `+0.09923`、enrichment `+2.81563`。经过 tracking-only TemporalFuse 后差异几乎消失：final-response relative L2 `0.000917`，50 个 case 的 response peak 改变为 `0/50`。

结论：上游语言证据存在，但在原生时序/响应接口中被衰减，未形成稳定目标决策。

#### BLIP、约束生成与 Florence 审计

- 20 序列在线 replay 中，官方静态文本 peak-in-GT 为 `0.2157`，普通 BLIP 为 `0.0980`，目标约束 BLIP 为 `0.1569`。
- 人工检查的 17 个 anchor 中，普通 BLIP 有 `15/17` 主体错误，约束 BLIP 有 `12/17` 主体错误。
- Stage2N-A 正式审计覆盖 50 序列、150 anchors、3,000 crop/prompt candidates、2,955 个有效 caption；人工检查与自动判断分歧 `22.21%`，100 个自动 stable-detail positive 中仅 `22` 个获人工确认。
- Florence 虽提高部分辅助 margin，但所有冻结 DUTrack response-gap delta 均为负（`-0.0026` 至 `-0.0202`）；`95.0%`--`98.3%` 的详细 caption 超出 24-token 接口。

结论：生成描述是噪声观测，不是可靠在线状态；无条件替换、拼接或更新语言均停止。

### 3.2 Stage3：联合训练与对齐传递

#### Stage3-A2 mixed-v2

- region loss 下降 `13.35%`，train GT-background margin 提高。
- held-out caption-specific localization 未改善；correct-hardest-safe train margin 从 `-0.03104` 恶化到 `-0.04250`。
- fixed-batch 审计显示 shortcut：positive similarity `+0.008032`，safe negative similarity 同时 `+0.007750`，rank-1 从 `3.94%` 降至 `3.15%`，335 次更新全部触发 `0.1` gradient clip。

#### Stage3-S1 与 Stage3-J1A

- Stage3-S1 学会 BI objective，但 official-static dev100 gate 失败：correct-shuffle localization delta `+0.000117`、GT mass `-0.000366`、Hann gap `-0.000730`。
- Stage3-J1A Phase-H 有较强 dev grounding：`S_loc=0.62832`、GT mass `0.83462`。
- hard-pair Phase-J 在 125 个 train 序列上过拟合，并显著损害 held-out grounding：`S_loc -0.08424`、GT mass `-0.03593`，置信区间均完全低于零。

结论：更强、更选择性的对齐监督可以优化局部目标，但没有获得稳健 held-out caption-specific decision。

### 3.3 Stage4：JA1--JA6 机制尝试

| 路线 | 主要设计/正证据 | 决定性失败 |
|---|---|---|
| JA1 contextual True-K16 | 正确接入上下文语言，粗类别文本有信息 | 100-step held-out trained-minus-step0 margin `-0.000560`，CI `[-0.003573,+0.002575]`；instance gate 失败 |
| JA1 fixed-pair replay | 402 个确定性 pair，checkpoint/crop 审计完整 | primary retrieval gate 失败，pair 仍是自动生成且未人工审核 |
| JA4 spatial/QK bias | proxy spatial/alignment objective 可学习 | 没有稳健 caption-specific decision；250-step 延长未获授权 |
| JA5 aligned fusion / SP1 | full-gallery top-1 `1.67%→22.0%`；fixed-safe accuracy `52.67%→73.0%` | hardest-margin treatment `-0.00603`，CI `[-0.01366,+0.00162]`；Hann treatment 近零 |
| JA5 G1 K48 | relative hardest-margin `+0.021460`，CI 全正 | absolute margin `-0.12450`；correct-above-hardest 仅 `25.0%`，hub gate 失败 |
| JA5 G2 mean center | 改善小型近邻描述 strata | primary SAFE treatment `-0.000607`；lexically-disjoint `-0.005773` |
| JA5 G3 query bank | 固定 hub caption 被压低 `-0.03584` | SAFE hardest-margin `-0.01723`，CI 全负；fixed-safe accuracy 下降 4 pp |
| JA5 T1 token dose | correct-caption GT-minus-HN gap `+0.008196` | correct-minus-shuffle `+0.000269`，CI `[-0.006550,+0.007074]` |
| JA5 SP1 online | 完成四个 100-sequence condition | ALIGN-minus-TRACK AUC `-1.63 pp`；FULL-minus-LANG_OFF `-1.38 pp` |
| JA6 semantic decision | 成对实现有效，causal gradient 有限且非零 | step1250 Hann-treatment gain `+0.00000615`、box-IoU gain `+0.000229`，CI 均跨零 |

JA1--JA6 的共同失败不是“无法训练”，而是训练目标、检索或 proxy 指标的改善没有转化为正确描述相对错误/关闭语言的空间决策与跟踪收益。主要 shortcut 是视觉容易样本上的 generic targetness、query-insensitive correction 和候选 hub。

### 3.4 EP47 frozen candidate diagnostic

只训练 512→128 的视觉/语言投影，在 132 序列训练、60 个未见序列评估：

- 正确 caption 的五选一 GT-vs-outside top-1：`98.33%`。
- safe wrong caption 的 top-1：同样 `98.33%`。
- correct-minus-full-shuffle margin：`+0.000372`，95% CI `[-0.001625,+0.002479]`。
- correct/wrong projected-text cosine：`0.99683`。
- correct/wrong tracker candidate-feature cosine：`0.999987`。
- 纯视觉 EP47 GT-minus-highest-outside response gap：`+0.66389`，60 个 held-out case 中 `100%` 为正。

这说明普通随机/容易候选主要测试视觉 targetness，不能作为语言特异性训练或评估人群，因此进入冻结 B0 因果审计。

## 4. B0：官方 EP47 无训练因果审计

### 4.1 B0-Local：相同在线状态的局部反事实

问题：在由 `LANG_OFF` 定义的视觉困难帧上，保持完全相同的 box、search crop、template、temporal query 和 history，正确官方描述是否同时优于 `LANG_OFF` 与人工审核错误描述？

- train 200 / test 100 全部完成，零失败。
- Hann ambiguity population：95 序列、22,362 帧。
- 所有参数冻结；`LANG_OFF` 独占在线状态；correct/wrong 仅做同帧只读 forward。

| 比较 | 序列宏平均 | 95% sequence-bootstrap CI | 结论 |
|---|---:|---:|---|
| ambiguous correct - `LANG_OFF` IoU | `-0.0000444` | `[-0.0002518,+0.0001522]` | 失败 |
| ambiguous correct - wrong IoU | `-0.0001017` | `[-0.0002701,+0.0000331]` | 失败 |
| ambiguous correct - wrong GT-vs-outside Hann gap | `+0.0000458` | `[-0.0000392,+0.0001336]` | 失败 |
| easy correct - `LANG_OFF` IoU | `-0.0000027` | `[-0.0000511,+0.0000403]` | noninferiority 通过 |

全序列诊断 AUC：correct `0.746754`，`LANG_OFF` `0.746768`，wrong `0.746803`。正确语言相对 `LANG_OFF` 的 box 在 `99.99%` 帧上发生数值变化，中位最大坐标变化约 0.053 pixel；这证明路径激活，但变化太小且方向无益。ambiguous set 上 correct 相对 off 为 6 次 rescue、13 次 harm。

证据结论：存在微弱 decision-path coupling；不存在局部语言特异性或 ambiguity-resolution benefit。B1 gate 不通过。

### 4.2 B0-Native Online：独立演化完整轨迹

问题：从共同 GT 初始化后，让 correct、wrong、`LANG_OFF` 分别独占自己的预测框、搜索区域、模板记忆和 temporal query，正确静态描述的长期累积是否有益？

- 100/100 序列完成，60,501 个 predicted frames，零失败。
- correct/wrong 均使用静态官方首帧描述，BLIP dynamic update 关闭。

| 全轨迹指标 | correct | `LANG_OFF` | wrong |
|---|---:|---:|---:|
| predicted-frame success AUC | `0.729423` | `0.746768` | `0.741562` |
| OPE-compatible AUC | `0.730071` | `0.747383` | `0.742180` |
| normalized precision@0.20 | `0.807440` | `0.826957` | `0.823178` |
| precision@20px | `0.773334` | `0.790969` | `0.782797` |
| severe drift (IoU<0.10) | `0.128479` | `0.108014` | `0.116769` |

关键 paired effect：

- correct - `LANG_OFF` AUC：`-0.017345`，95% CI `[-0.035867,-0.003313]`。
- correct - wrong AUC：`-0.012139`，95% CI `[-0.029305,+0.001120]`。
- correct - `LANG_OFF` normalized precision：`-0.019517`，CI `[-0.040755,-0.003127]`。
- correct - `LANG_OFF` precision@20px：`-0.017635`，CI `[-0.037736,-0.001790]`。
- correct - `LANG_OFF` severe drift：`+0.020466`，CI `[+0.004089,+0.041953]`。
- correct 相对 off：676 rescues、2,316 harms；相对 wrong：755 rescues、1,647 harms。

证据结论：静态正确语言在原生在线历史中产生显著负 tracking effect；小的局部错误会经 box、crop、template、history 和 temporal query 累积放大。正确语言仍未可靠优于安全错误语言，因此不能称为语言特异性。

### 4.3 B0-Dynamic：真实、打乱与静态语言五条件审计

条件：`DYNAMIC_REAL`、`DYNAMIC_SHUFFLED`、`LANG_OFF`、`STATIC_CORRECT`、`STATIC_WRONG`。

- 100/100 序列，70,176 video frames，27,298 个 BLIP update events，零失败。
- real/shuffled replay 与冻结 native state 在 box、pre-state、temporal query 和 query norm 上满足 `1e-5` 合同。
- shuffled event 一对一、跨序列、无相同 normalized caption，平均有效 BERT 长度差 `0.050187`。

| 条件 | OPE macro AUC |
|---|---:|
| `DYNAMIC_REAL` | `0.748429` |
| `DYNAMIC_SHUFFLED` | `0.747373` |
| `LANG_OFF` | `0.747383` |
| `STATIC_CORRECT` | `0.730071` |
| `STATIC_WRONG` | `0.742180` |

| 配对比较 | AUC delta | 95% CI | 结果 |
|---|---:|---:|---|
| real - `LANG_OFF` | `+0.001046` | `[-0.011709,+0.014022]` | 未支持 |
| real - shuffled | `+0.001056` | `[-0.021538,+0.024456]` | 未支持 |
| real - static correct | `+0.018357` | `[+0.002734,+0.038497]` | 仅解释性正结果 |

BLIP 质量诊断：

- 每 100 predicted frames 产生 38.95 次更新；
- `80.36%` 与上一条 caption 重复；
- normalized caption 唯一率仅 `3.42%`；
- `89.30%` 在 EP47 的 16-slot BERT 接口被截断；
- 性能盲审的 299 个事件中，只有 `3/299` 被标为主体正确。

证据结论：动态真实语义没有可靠优于纯视觉或打乱语义。real 优于 static correct 只说明动态替换缓解了有害的静态语言，不说明 BLIP 新语义有价值。

两个必要 point estimate 略为正，因此原预注册规则只允许一次 full-TNL2K no-training confirmation；该确认未运行。随着 DUTrack 研究线关闭，这个未完成项不再构成正证据，也不授权 BLIP repair 或 B1 training。

## 5. 跨阶段根因判断

反复出现的失败机制可以归纳为四层：

1. **候选人群不对**：大量样本视觉上过于容易，GT 由视觉 targetness 即可确定，正确和错误描述都会得到高分。
2. **辅助目标 shortcut**：InfoNCE、projection 或 region loss 可通过共同抬高正负相似度、类别线索或固定 hub 降低 loss，不必学会实例级语言区分。
3. **接口衰减**：上游存在语言差异，但 TemporalFuse、response head 或残差视觉路径将差异压缩到无法稳定改变空间决策。
4. **在线累积有害**：微小、非特异的语言 perturbation 在同帧几乎中性，但一旦拥有在线 state，会经错误框和模板更新放大为显著漂移。

因此，DUTrack 的本质问题不是简单的“语言 token 太少”或“对齐 loss 不够强”，而是缺少一个在视觉竞争情形下，将可靠语言证据与候选空间决策绑定的受控接口。继续全局拉近 cosine、广播语言或固定增强语言权重，既可能破坏已有视觉几何，也没有解决什么时候应使用语言以及使用哪部分语义的问题。

## 6. 对后续 JointNLT 研究的可复用资产

不迁移 JA1--JA6 模块、DUTrack tensor slot、阈值或 checkpoint。只迁移以下研究协议：

1. 把 grounding、tracking fusion 和 independently evolving online trajectory 分开审计。
2. 同时保留 common-state local 与 independently evolving online 两类反事实，避免把微小耦合与轨迹累积混为一谈。
3. 主负样本使用同图、同类、视觉困难或人工审核 safe wrong language，随机异类文本只作辅助。
4. 正确语言必须同时击败 wrong/shuffled 与 `LANG_OFF`。
5. 视觉困难程度必须由语言无关条件定义；阈值只在 train split 冻结。
6. batch=1 数值路径是正式标准，除非批量路径逐元素等价已被证明。
7. 对视觉语言 Transformer，进一步定位 Q/K/V、carrier、target decoder 和 box head 的读取关系，而不是以全局 cosine 或 attention 变化作为语言有效性的结论。
8. 动态语言必须允许 abstain，并同时建模视觉需要度和语言可靠度；低质量生成 caption 不能强制写入长期状态。

## 7. 关键 provenance

### 7.1 实验提交标识

以下是实验运行时记录的本地 commit 标识，仅用于历史对照；按最终归档决定，这些实验提交本身不推送到远端：

- B0-Local formal：`12faf38`。
- B0-Native implementation/formal：`806f011` / `fc24934`。
- B0-Dynamic implementation/smoke/formal：`6e14923` / `bd928a5` / `b9ae52a`。
- JA1 clean snapshot：`9fb0447`。
- JA2/JA3 snapshot：`eca34d7`。
- JA5 C2/SP1 snapshots：`ca2e3ad` / `724b641`。

### 7.2 正式产物哈希

| 产物 | SHA-256 |
|---|---|
| B0 protocol | `7bbf5506ea4370690f96ff80700bb4f8fd800b66970631a33ba820c9233c70dd` |
| wrong-caption mapping | `98bccd33eadaf3e8e9f45d6339fd276ea83df6e428df2862761db5e5eea8359e` |
| frozen thresholds | `3dafae5c993ccb7cceb12b471723edf4d851b39617b5314341365bfee30cd75a` |
| B0-Local report | `5045784887b7d1cc30da181d760cb25235908c76c2536000a2db04b097fbac9e` |
| B0-Native report | `fe18817d72c3e666b6a3d2e8d48cdd887099eceaac9dd999d7157d2be1c0600b` |
| B0-Dynamic report | `ce738eab992abfb68b5ff8954e366a8b63b7463c4e66b29a5dc356df70f1eeba` |
| dynamic real-event manifest | `af1293f9c295b5f2088faa8af70f7d3db38fa30b256bf536711f509ca2ee3320` |
| dynamic shuffle manifest | `82fdb4e5542fef780f1f27015bc571d25e3a65964eceef3b04e3d0430e1c55a5` |
| manual-quality ZIP | `0e79dd4f8c958ceb7ca1130e5bdaec32600546b8d73a2f17a9509272f1dd6428` |
| JA5-SP1 online report | `fe25c500dcf129538ee7c7ec3f9507d1a46133f971f7f88eace36e6bcd5055f0` |
| JA6 step1250 report | `34fc03b5a790dd60660a46f2f03e58544fc0c96e5615e4f1c31b48d68cc8399f` |

### 7.3 本总结的本地来源文档

| 来源文档 | SHA-256 |
|---|---|
| Stage2--Stage4 consolidated summary | `0f42b6327daadb5e06d33f10ec18ac14febe018157197cf16e0fa68616380f03` |
| final dynamic research state | `88dc1f421dd2e18bdae670c1678d88a8cef608fc5410a5631259cbe814855dee` |
| B0-Local human report | `3bf2dc3faf3ca223895aceb6ba1d37e880f39589f4019eda55f5a6b051258c31` |
| B0-Native human report | `812c224635460a7f1454fc7aabc913afb56069bf00cba7ae608d5ef527fc079b` |
| B0-Dynamic human report | `117c7436c7f5de09e603b68fdf690df95b88ca02f95f1e5fcb399329b050c25b` |
| JointNLT transition memo | `f93f34289aac7028ed5c170c12b4d268b55bba65aec45557d53fae1347577310` |

## 8. 保存与可复现边界

本文件是最终归档 commit 相对远端 `main` 的唯一新增文件。按项目关闭决定：

- 不推送 38 个本地实验 commit、stash、实验脚本、JA 模块、checkpoint、预训练模型或本机 output；
- 不上传数据集、私有图像、人工审核图片或 KESU/KESU1 大产物；
- KESU/KESU1 上已有正式产物保持原状，其哈希作为结果核验锚点；
- GitHub 保留原有 DUTrack baseline 历史和本最终总结，但不保证能够从 GitHub 单独重建所有已关闭实验实现；
- 本总结支持复核研究问题、设计、关键数字、证据等级和停止理由，不等价于完整代码级复现包。

这是有意的归档取舍：保留科学证据链和失败经验，停止维护已证明没有稳定收益的实现分支。

## 9. 最终研究决策

DUTrack 的语言开发路线正式关闭。full-TNL2K 动态语言确认、B1 selective intervention、BLIP repair 和 JA1--JA6 均不再继续。后续以 JointNLT 为 baseline，优先研究：

- 视觉困难何时产生真实语言需求；
- 正确语义在 Transformer Q/K/V 与 target decoder 中如何形成候选选择；
- 如何让可靠语言影响决策，同时让冗余或不可靠语言 abstain；
- 如何区分表示对齐成功、下游读取成功与最终 tracking improvement。

DUTrack 对后续工作的最大价值不是一个可继续堆模块的模型，而是一条完整的负结果链：语言信息存在、路径也会响应，但如果语义没有在视觉竞争位置与空间决策绑定，更强对齐和更大语言权重只会产生 shortcut、衰减或在线漂移。
