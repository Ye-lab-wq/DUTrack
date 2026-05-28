# Stage 3 Learnable Token-State Updater 实现记录

## 目标

本次修改只做一件事：把“文本字符串更新”推进到“语言 token 状态更新”的工程接口。

之前的 S0 诊断已经说明：

- 直接替换整句 BLIP caption 经常引入噪声；
- 文本级逐词吸收有正信号，但重新拼成文本后收益很弱；
- 固定 token residual 有 oracle 上界，但固定 `alpha` 不稳定。

因此当前实现一个保守的可学习 updater：

```text
H_anchor, H_prev, H_blip
        ↓
LanguageTokenStateUpdater
        ↓
H_state
        ↓
DUTrack.forward(..., language_token_state=H_state)
```

这里的 `H_state` 不是一句自然语言，而是一组 BERT token embedding 状态，表示“保留 anchor 主体、吸收候选语言增量后的潜在语言状态”。

## 代码改动

### 1. 新增模块

文件：

```text
lib/models/dutrack/language_token_state_updater.py
```

核心输入：

```text
anchor_tokens:    (B, L, C)
prev_tokens:      (B, L, C)
candidate_tokens: (B, L, C)
```

构造特征：

```text
[H_anchor, H_prev, H_candidate,
 H_candidate - H_anchor,
 H_candidate - H_prev]
```

输出：

```text
H_state = H_prev + frame_gate * token_gate * tanh(delta) * max_delta
```

设计约束：

- `delta_head` 小方差初始化，避免 delta 完全为 0 时 gate 初期没有有效梯度；
- token gate / frame gate bias 默认 `-4.0`；
- 初始状态接近 no-op；
- 有 mask 时无效 token 保持 `H_prev`。

### 风险修正

针对当前审查提出的三个风险，本轮补充如下。

#### 1. 显式视觉证据输入

`LanguageTokenStateUpdater.forward()` 新增：

```python
visual_evidence=None
```

S0 诊断中只使用 deploy 可获得证据，不用 GT：

```text
State-change evidence:
  center_motion_norm
  scale_change_ratio
  color_change_norm

Tracker-confidence evidence:
  score_peak_second_gap
  score_entropy
  box_jump_ratio

Candidate-usefulness evidence:
  deploy_score_delta = gap(BLIP) - gap(prev)
  partial_deploy_delta = best_partial_gap - gap(prev)
```

这些值先经过 `visual_mlp`，再注入 token hidden 和 frame gate。`deploy_trigger` 是旧阈值规则的输出，只保留为日志，不再作为模型输入。

#### 2. 词间关系建模

token MLP 后新增轻量 relation block：

```text
token hidden -> MultiheadSelfAttention -> MLP
```

配置项：

```yaml
LANGUAGE_STATE_RELATION_LAYERS: 1
LANGUAGE_STATE_RELATION_HEADS: 4
```

这不是重型语言模型，只用于让 anchor/prev/BLIP token 增量之间可以相互参照，避免完全独立的逐 token gate。

#### 3. gate 长期打不开

原先 `delta_head=0` 会导致初始 `state_delta=0`。虽然安全，但 gate 初期梯度很弱。现在改为：

```yaml
LANGUAGE_STATE_INIT_DELTA_STD: 0.0001
LANGUAGE_STATE_INIT_GATE_BIAS: -4.0
```

即：

- delta 非零但极小；
- gate 仍然接近关闭；
- 初始输出仍接近 no-op；
- 训练时 gate 有机会收到非零梯度。

新增诊断字段：

```text
token_learned_relation_attn_mean
token_learned_visual_evidence_abs_mean
token_learned_state_center_motion_norm
token_learned_state_scale_change_ratio
token_learned_state_color_change_norm
token_learned_conf_peak_gap
token_learned_conf_score_entropy
token_learned_conf_box_jump
token_learned_candidate_deploy_score_delta
token_learned_candidate_partial_deploy_delta
```

### 2. 配置默认值

文件：

```text
lib/config/dutrack/config.py
```

新增默认关闭配置：

```yaml
MODEL:
  TE:
    LANGUAGE_STATE_ENABLE: false
    LANGUAGE_STATE_HIDDEN_DIM: 256
    LANGUAGE_STATE_MAX_DELTA: 0.1
    LANGUAGE_STATE_DROPOUT: 0.0
    LANGUAGE_STATE_INIT_GATE_BIAS: -4.0
```

### 3. Backbone 注册

文件：

```text
lib/models/dutrack/itpn.py
```

在 `finetune_track()` 中，如果 `LANGUAGE_STATE_ENABLE=True`，创建：

```python
self.language_state_updater = LanguageTokenStateUpdater(...)
```

默认关闭时为 `None`，不影响原训练和测试。

### 4. S0 诊断接入

文件：

```text
tracking/language_state_s0_probe.py
tracking/language_state_s0_screen.py
```

新增参数：

```bash
--learned_token_state_probe
```

启用后，如果当前 config 中存在 `backbone.language_state_updater`，每帧额外评估：

```text
token_learned_state
```

并记录：

```text
token_learned_state_available
token_learned_frame_gate_mean
token_learned_token_gate_mean
token_learned_combined_gate_mean
token_learned_delta_abs_mean
token_learned_state_delta_abs_mean
token_learned_state_gain_over_prev
```

`token_learned_state` 也会参与 `token_state_best_source / token_state_best_gain_over_prev` 的比较。

## 重要边界

当前训练 actor 只有：

```python
descript=data["language_description"]
```

也就是说，训练 batch 里还没有：

```text
anchor language
previous language state
BLIP/current candidate language
```

所以这个 updater 现在不能被当前常规训练命令真正端到端训练。强行加入 `TRAIN_TE_ONLY_PATTERNS` 也没有意义，因为训练前向没有调用 `language_state_updater`。

当前阶段正确用途是：

1. S0/Suite 诊断：验证模块接口、gate 初始状态、候选状态对 score gap 的影响；
2. 下一步再改训练数据流：把候选语言状态输入 actor，才谈得上训练 updater。

## 风险

- 未训练 updater 初始接近 no-op，短期不会产生明显收益；
- 如果直接放大 gate 或 delta，容易回到“BLIP 噪声污染语言状态”的问题；
- 真正训练前必须补齐 candidate language 数据来源，否则没有有效梯度路径。

## 建议下一步

先在 S0 中只做诊断。由于新配置名会默认寻找新 checkpoint，本次补了：

```bash
--checkpoint_config dutrack_384_full
```

这样可以用 `dutrack_384_full_lmq_state_e10` 建模并实例化 `language_state_updater`，同时加载稳定的原始 baseline 权重 `dutrack_384_full/DUTrack_ep0047.pth.tar`。新增的 `language_state_updater` 与 LMQ 相关参数会因为 `strict=False` 保持随机初始化。

这里不应使用已经验证坍塌或不存在完整 checkpoint 的 LMQ 权重。当前阶段要隔离诊断“语言状态更新器是否有可用信号”，而不是混入 LMQ score-prior 的训练残留。

## TokenStrict v3：identity 空间校正与 source gate 双边界

### 背景问题

`dutrack_384_full_langstate_tokenstrict_v2_e5` 的训练结果暴露了两个核心问题：

1. `token_absorb_target_pos_ratio` 长期为 0，但 `rel_ok / abs_ok / hard_ok` 很高，说明视觉侧条件不是主要瓶颈；
2. `token_absorb_identity_ok_ratio` 为 0，`token_absorb_anchor_cos_mean` 只有约 0.05，说明 identity 约束几乎把所有 candidate token 都拒绝了。

原因不是候选词一定没有身份一致性，而是之前 identity 在两个不同空间里比较：

```text
candidate_aligned_tokens  vs  raw anchor BERT tokens
```

`candidate_aligned_tokens` 已经经过 updater 的对齐/交互投影，而 raw anchor token 仍是原始语言 encoder 输出。这个余弦相似度不能作为训练标签。

另一个问题是 source gate 存在 shortcut：

```text
gain_loss 可以通过提高 anchor weight、压低 candidate weight 来满足，
不一定真的学会“吸收有用 candidate token”。
```

### 代码修正

#### 1. identity 空间校正

文件：

```text
lib/train/actors/dutrack.py
```

当前 token absorb target 的 identity 约束优先使用：

```text
candidate_aligned_tokens  vs  anchor_aligned_tokens
```

只有在 aligned anchor 不存在时，才回退到 raw anchor。raw 空间相似度仍然记录为诊断：

```text
token_absorb_raw_anchor_cos_mean
```

因此后续判断 identity 是否有效，优先看：

```text
token_absorb_anchor_cos_mean
token_absorb_identity_ok_ratio
```

raw 指标只用于确认“原始 token 空间和 updater 对齐空间差异有多大”。

#### 2. source gate 双边界

新增两个正则：

```text
L_anchor_cap = mean(ReLU(w_anchor - anchor_max)^2)
L_prev_keep  = mean(ReLU(prev_min - w_prev)^2)
```

对应配置：

```yaml
TRAIN:
  LANGUAGE_STATE_ANCHOR_CAP_LOSS_WEIGHT: 0.03
  LANGUAGE_STATE_ANCHOR_WEIGHT_MAX: 0.35
  LANGUAGE_STATE_PREV_KEEP_LOSS_WEIGHT: 0.03
  LANGUAGE_STATE_PREV_KEEP_MIN: 0.5
```

含义：

- `anchor_cap` 防止 source gate 退化成“全部用 anchor”；
- `prev_keep` 防止语言状态被 candidate/anchor 快速覆盖；
- candidate 仍然由已有 `LANGUAGE_STATE_CANDIDATE_CAP_LOSS_WEIGHT` 控制上限。

这比只给 candidate 加 cap 更完整，因为 source gate 是三路竞争：

```text
state = w_prev * prev + w_anchor * anchor + w_candidate * candidate
```

如果只限制 candidate，模型仍可能走 anchor shortcut。

#### 3. gain loss 暂时关闭

新配置中：

```yaml
LANGUAGE_STATE_GAIN_LOSS_WEIGHT: 0.0
```

原因是当前阶段的主要目标不是直接追 score gap，而是验证：

```text
是否存在干净、稀疏、identity-consistent 的可吸收 token target。
```

如果继续保留 gain loss，模型可能绕过 token absorb 目标，直接通过 source gate 的 anchor/prev 比例制造局部收益，掩盖 token 级学习是否成立。

### 新配置

文件：

```text
experiments/dutrack/dutrack_384_full_langstate_tokenstrict_v3_e5.yaml
```

关键差异：

```yaml
TRAIN:
  LANGUAGE_STATE_GAIN_LOSS_WEIGHT: 0.0
  LANGUAGE_STATE_TOKEN_ABSORB_LOSS_WEIGHT: 0.001
  LANGUAGE_STATE_TOKEN_ABSORB_IDENTITY_ENABLE: true
  LANGUAGE_STATE_TOKEN_ABSORB_IDENTITY_MIN: 0.2
  LANGUAGE_STATE_TOKEN_ABSORB_TOP_RATIO: 0.1
  LANGUAGE_STATE_TOKEN_ABSORB_MAX_POS: 6
  LANGUAGE_STATE_CANDIDATE_CAP_LOSS_WEIGHT: 0.03
  LANGUAGE_STATE_CANDIDATE_WEIGHT_MAX: 0.35
  LANGUAGE_STATE_ANCHOR_CAP_LOSS_WEIGHT: 0.03
  LANGUAGE_STATE_ANCHOR_WEIGHT_MAX: 0.35
  LANGUAGE_STATE_PREV_KEEP_LOSS_WEIGHT: 0.03
  LANGUAGE_STATE_PREV_KEEP_MIN: 0.5
```

### 训练时重点看

这版不是优先追最终 IoU，而是先判断可学习目标是否健康。

关键健康指标：

```text
token_absorb_identity_ok_ratio
token_absorb_target_pos_ratio
token_absorb_anchor_cos_mean
token_absorb_raw_anchor_cos_mean
token_absorb_loss_active_ratio
```

期望：

- `identity_ok_ratio` 不再恒为 0；
- `target_pos_ratio` 非零但保持稀疏，理想范围大约 0.02 到 0.15；
- `anchor_cos_mean` 明显高于 raw anchor cos，说明 aligned identity 空间更合理；
- `loss_active_ratio` 不应全 0，也不应长期接近 1。

source gate 重点看：

```text
LanguageState/anchor_weight_mean
LanguageState/prev_keep_weight_mean
LanguageState/candidate_absorb_weight_mean
LanguageState/anchor_cap_active_ratio
LanguageState/prev_keep_active_ratio
```

期望：

- `anchor_weight_mean` 不应长期高于 0.35；
- `prev_keep_weight_mean` 应保持在 0.5 附近或以上；
- `candidate_absorb_weight_mean` 应受控，而不是直接塌到 0 或冲到上限；
- `anchor_cap_active_ratio / prev_keep_active_ratio` 如果长期接近 1，说明边界和主任务仍在冲突。

### 训练指令

建议从稳定 baseline 权重重新开始，不沿用 v2 checkpoint。v2 已经把 source gate 推向 anchor shortcut，继续训练会污染这次诊断。

```bash
python tracking/train.py \
  --script dutrack \
  --config dutrack_384_full_langstate_tokenstrict_v3_e5 \
  --save_dir ./output \
  --mode single \
  --use_wandb 0
```

### 验证命令

```bash
conda run -n DUTrack python -m py_compile \
  lib/train/actors/dutrack.py \
  lib/config/dutrack/config.py

conda run -n DUTrack python -c "from lib.config.dutrack.config import cfg, update_config_from_file; update_config_from_file('experiments/dutrack/dutrack_384_full_langstate_tokenstrict_v3_e5.yaml'); print(cfg.TRAIN.LANGUAGE_STATE_GAIN_LOSS_WEIGHT); print(cfg.TRAIN.LANGUAGE_STATE_ANCHOR_CAP_LOSS_WEIGHT); print(cfg.TRAIN.LANGUAGE_STATE_PREV_KEEP_MIN); print(cfg.TRAIN.LANGUAGE_STATE_TOKEN_ABSORB_IDENTITY_ENABLE)"
```

```bash
python tracking/language_state_s0_probe.py \
  --config dutrack_384_full_lmq_state_e10 \
  --checkpoint_config dutrack_384_full \
  --dataset_name otb_lang \
  --sequence Dog \
  --runid 10 \
  --candidate_mode oracle_blip \
  --state_update_policy none \
  --token_state_probe \
  --learned_token_state_probe \
  --max_frames 20 \
  --output_tag stage3_learned_token_state_diag
```

如果 `token_learned_state_available=0`，说明当前配置没有实例化 updater。

小套件诊断：

```bash
python tracking/language_state_s0_screen.py \
  --config dutrack_384_full_lmq_state_e10 \
  --checkpoint_config dutrack_384_full \
  --dataset_names otb_lang,hoot_balanced20 \
  --sequence_names Dog,Gym,potted_plant-008,toilet_paper-001 \
  --runid 10 \
  --candidate_mode oracle_blip \
  --state_update_policy none \
  --token_state_probe \
  --token_state_alphas 0.05,0.1,0.3 \
  --learned_token_state_probe \
  --max_frames 50 \
  --output_tag stage3_token_state_learned_diag_small
```

当前 `language_state_updater` 还是未训练模块，保守初始化下应该接近 no-op。这个阶段主要检查：

- `token_learned_state_available_rate` 是否为 1；
- `token_learned_frame_gate / token_learned_token_gate` 是否接近初始小门控；
- `token_learned_state_delta_abs` 是否很小；
- `token_state_raw_best_gain_over_prev`：只在 token 更新候选中取最好；
- `token_state_best_gain_over_prev`：显式允许 `no_update` 后的上界，应当不小于 0；
- 固定 residual 的 oracle 上界是否仍存在。

若这些都正常，下一步才是改训练数据流，让 actor 同时拿到 anchor / prev / candidate 语言状态。

## Gate-free 诊断版：确认 no-op 的来源

`dutrack_384_full_langstate_mid_e5` 相比强正则版稍微放松，但训练过程中仍然出现明显 no-op 趋势：

```text
token_gate_mean / frame_gate_mean 持续下降
combined_gate_mean 约 1e-4 -> 5e-5
state_delta_abs_mean 接近 0
```

这说明问题已经不是“gate 饱和”，而是 updater 没有获得足够强的正向学习信号。这里有两个可能：

1. gate 正则仍然把更新压死；
2. 即使不压 gate，最终 tracking loss 对 token-state update 的收益也太弱，模型自然选择 no-op。

为区分这两种情况，新增一个诊断配置：

```text
experiments/dutrack/dutrack_384_full_langstate_gatefree_e5.yaml
```

关键差异：

```yaml
MODEL:
  TE:
    LANGUAGE_STATE_MAX_DELTA: 0.02
    LANGUAGE_STATE_INIT_GATE_BIAS: -4.0
    LANGUAGE_STATE_INIT_DELTA_STD: 0.0001
TRAIN:
  LR: 0.00001
  LANGUAGE_STATE_VISUAL_EVIDENCE: none
  LANGUAGE_STATE_GATE_LOSS_WEIGHT: 0.0
  LANGUAGE_STATE_DELTA_LOSS_WEIGHT: 0.02
  TRAIN_TE_ONLY_PATTERNS:
  - backbone.language_state_updater
```

设计意图：

- 去掉 gate regularization，避免直接惩罚“打开更新”；
- 保留很弱的 delta regularization，防止状态偏移无约束扩大；
- 不使用 `gt_motion`，避免训练时引入 deploy 阶段不可用的捷径；
- 仍然只训练 `language_state_updater`，不开放 backbone/head，隔离变量。

训练指令：

```bash
python tracking/train.py \
  --script dutrack \
  --config dutrack_384_full_langstate_gatefree_e5 \
  --save_dir ./output \
  --mode single \
  --use_wandb 0
```

如果没有激活环境：

```bash
conda run -n DUTrack python tracking/train.py \
  --script dutrack \
  --config dutrack_384_full_langstate_gatefree_e5 \
  --save_dir ./output \
  --mode single \
  --use_wandb 0
```

判断标准：

- 如果 gate 仍然关闭：说明 tracking loss 给 updater 的正向信号不足，单靠当前训练链路很难学出有效语言状态更新；
- 如果 gate 打开但 tracking loss 不改善：说明 token-state perturbation 可以发生，但没有转化成有效 center/box 输出；
- 如果 gate 打开且 loss/IoU 改善：说明前一版 no-op 主要由 gate 正则导致，再考虑把 gate 正则作为退火项或质量门控约束重新引入。

需要重点记录：

```text
LanguageState/token_gate_mean
LanguageState/frame_gate_mean
LanguageState/combined_gate_mean
LanguageState/state_delta_abs_mean
LanguageStateReg/delta_reg_loss
Loss/language_state
```

## Token-level keep-absorb updater

`gatefree` 诊断版说明：只靠

```text
H_state = H_prev + gate * delta
```

这一条 residual 更新路径时，模块容易停在近似 no-op。即使去掉 gate regularization，`state_delta_abs` 仍然非常小。这说明问题不只是正则强度，而是当前结构没有显式表达“保留旧语言 / 吸收候选语言 / 回退 anchor 主体”这三种动作。

因此新增 `keep_absorb` 更新模式。

### 数学形式

文件：

```text
lib/models/dutrack/language_token_state_updater.py
```

新增配置：

```yaml
MODEL:
  TE:
    LANGUAGE_STATE_UPDATE_MODE: keep_absorb
```

每个 token 生成三路 source weight：

```text
[w_anchor_i, w_prev_i, w_candidate_i] = softmax(SourceGate(h_i))
```

然后更新：

```text
H_state_i =
    w_anchor_i    * H_anchor_i
  + w_prev_i      * H_prev_i
  + w_candidate_i * H_candidate_i
  + frame_gate * token_gate_i * delta_i
```

其中：

- `w_prev_i` 表示保留上一帧语言状态；
- `w_candidate_i` 表示吸收当前候选语言 token；
- `w_anchor_i` 表示回退到初始主体语言，防止 candidate 漂移；
- `delta_i` 仍然只保留小幅连续修正能力。

初始化方式：

```text
source bias = [init_gate_bias, -init_gate_bias, init_gate_bias]
```

当 `init_gate_bias=-4` 时，初始几乎等价于 `H_state ≈ H_prev`，保证 checkpoint-safe；但相比 residual 版本，candidate/anchor 有显式 source gate，可以观察是否真的发生选择性吸收。

### 新增诊断字段

```text
LanguageState/update_mode_keep_absorb
LanguageState/anchor_weight_mean
LanguageState/prev_keep_weight_mean
LanguageState/candidate_absorb_weight_mean
LanguageState/source_entropy_mean
```

关键判断：

- `prev_keep_weight_mean` 长期接近 1：仍然 no-op；
- `candidate_absorb_weight_mean` 上升但 tracking loss 不改善：候选语言被吸收，但没有转成有效跟踪收益；
- `anchor_weight_mean` 上升：模块在回退初始主体描述，可能说明 candidate/prev 状态不稳定；
- `source_entropy_mean` 很低：几乎 hard select；很高：三路混合不明确。

### 新配置

文件：

```text
experiments/dutrack/dutrack_384_full_langstate_keepabsorb_e5.yaml
```

关键设置：

```yaml
MODEL:
  TE:
    LANGUAGE_STATE_UPDATE_MODE: keep_absorb
    LANGUAGE_STATE_MAX_DELTA: 0.02
    LANGUAGE_STATE_INIT_GATE_BIAS: -4.0
TRAIN:
  LR: 0.00001
  LANGUAGE_STATE_VISUAL_EVIDENCE: none
  LANGUAGE_STATE_GATE_LOSS_WEIGHT: 0.0
  LANGUAGE_STATE_DELTA_LOSS_WEIGHT: 0.0
  TRAIN_TE_ONLY_PATTERNS:
  - backbone.language_state_updater
```

这版不加 language-state 正则，是为了先看 tracking loss 是否会主动推动 source gate 从 `H_prev` 转向 candidate 或 anchor。若一开始就加入 `delta/state` 正则，会再次把 source mixing 拉回 no-op。

训练指令：

```bash
python tracking/train.py \
  --script dutrack \
  --config dutrack_384_full_langstate_keepabsorb_e5 \
  --save_dir ./output \
  --mode single \
  --use_wandb 0
```

如果没有激活环境：

```bash
conda run -n DUTrack python tracking/train.py \
  --script dutrack \
  --config dutrack_384_full_langstate_keepabsorb_e5 \
  --save_dir ./output \
  --mode single \
  --use_wandb 0
```

主要风险：

- `prev_keep_weight_mean` 仍然接近 1：说明 tracking loss 对语言状态更新仍然没有足够驱动力；
- `candidate_absorb_weight_mean` 快速接近 1：说明变成无条件替换 candidate，需要重新加入 source regularization 或质量 gate；
- `anchor_weight_mean` 异常升高：可能说明训练在利用 anchor 作为 shortcut，而不是学习动态更新。

## Incremental gap gain auxiliary

`keep_absorb` 版本结构上提供了三路 source gate，但纯 tracking loss 下仍然基本停在：

```text
prev_keep_weight_mean ≈ 0.999
candidate_absorb_weight_mean ≈ 0.0003
```

这说明仅靠最终框回归/center loss，语言状态更新器仍缺少足够直接的学习信号。因此新增一个轻量辅助目标，只回答一个问题：

```text
H_state 是否比 H_prev 更能区分 GT center token 和当前 score hard negative token？
```

### 数学定义

使用同一个非参数 prior probe，不引入额外可学习 prior head：

```text
q_prev  = normalize(masked_mean(H_prev))
q_state = normalize(masked_mean(H_state))
K_x     = normalize(detach(search_tokens))

P_prev(j)  = q_prev  · K_x(j)
P_state(j) = q_state · K_x(j)
```

正样本区域：

```text
P_pos = GT center quality / gaussian tokens
```

hard negative：

```text
N_hard = base score map 中最高的非 GT-center tokens
```

gap：

```text
gap_prev  = mean(P_prev  on P_pos) - mean(P_prev  on N_hard)
gap_state = mean(P_state on P_pos) - mean(P_state on N_hard)
```

loss：

```text
L_gain = ReLU(margin - (gap_state - gap_prev))
```

### 风险处理

- 不直接监督 `candidate_absorb_weight`，避免把人工规则写进 source gate；
- `K_x` 使用 `detach(search_tokens)`，aux 主要训练 updater，不反向改 backbone；
- `P_prev` 的 gap 分支 detach，只作为 baseline，不让模型通过移动 baseline 逃避；
- 优化的是 `gap_state - gap_prev`，不是 absolute `gap_state`，避免退化为重新做一个弱 score head；
- hard negative 来自当前 center score 的高响应背景，更贴近 center head 的实际竞争点。

### 新配置

文件：

```text
experiments/dutrack/dutrack_384_full_langstate_keepabsorb_gain_e5.yaml
```

关键设置：

```yaml
TRAIN:
  LANGUAGE_STATE_GAIN_LOSS_WEIGHT: 0.02
  LANGUAGE_STATE_GAIN_MARGIN: 0.01
  LANGUAGE_STATE_GAIN_HARDNEG_TOPK: 6
  LANGUAGE_STATE_GATE_LOSS_WEIGHT: 0.0
  LANGUAGE_STATE_DELTA_LOSS_WEIGHT: 0.0
```

这版先不加 gate/source 正则，目的是观察 gain auxiliary 是否能让 source gate 从 `prev-only` 中走出来。

训练指令：

```bash
python tracking/train.py \
  --script dutrack \
  --config dutrack_384_full_langstate_keepabsorb_gain_e5 \
  --save_dir ./output \
  --mode single \
  --use_wandb 0
```

如果没有激活环境：

```bash
conda run -n DUTrack python tracking/train.py \
  --script dutrack \
  --config dutrack_384_full_langstate_keepabsorb_gain_e5 \
  --save_dir ./output \
  --mode single \
  --use_wandb 0
```

重点观察：

```text
LanguageStateReg/gain_loss
LanguageStateReg/gain_active_ratio
LanguageStateReg/gap_state
LanguageStateReg/gap_prev
LanguageStateReg/gap_gain
LanguageState/prev_keep_weight_mean
LanguageState/candidate_absorb_weight_mean
LanguageState/anchor_weight_mean
```

判断标准：

- `gain_active_ratio` 长期接近 0：base gap 已足够，aux 没有训练信号；
- `gain_active_ratio` 高但 source gate 不动：当前 language/search token cosine probe 不足以驱动 updater；
- `candidate_absorb_weight_mean` 上升且 `gap_gain > 0`：candidate 语言确实开始提供增量；
- `anchor_weight_mean` 上升且 `gap_gain > 0`：当前 candidate/prev 不稳，anchor 回退更有用。

## 轻量训练实验闭环

本轮补齐了一个最小可训练链路，目标不是最终方案，而是验证 `language_state_updater` 是否能通过最终 tracking loss 学到非 no-op 的 token-state 更新。

### 新增训练输入

文件：`lib/train/actors/dutrack.py`

新增开关：

```yaml
TRAIN:
  LANGUAGE_STATE_TRAIN_ENABLE: true
  LANGUAGE_STATE_DETACH_TEXT: true
  LANGUAGE_STATE_VISUAL_EVIDENCE: gt_motion
```

训练时 actor 会从 batch 内已有的 frame-level `language_description` 构造：

```text
H_anchor    = 第一个 search 描述的 BERT token embedding
H_prev      = 当前 search 的上一帧描述 embedding；第 0 个 search 使用 H_anchor
H_candidate = 当前 search 描述 embedding
```

然后调用：

```text
H_state = LanguageTokenStateUpdater(
  H_anchor,
  H_prev,
  H_candidate,
  visual_evidence
)
```

并把 `H_state` 作为 `language_token_state` 传入 backbone，替代原来的文本重新编码路径。`LANGUAGE_STATE_DETACH_TEXT=true` 时，BERT embedding 只作为常量输入，梯度只更新 updater。

### 轻量视觉证据

当前训练版不调用 BLIP，也不使用预测 score 自举信号。`gt_motion` 只提供低成本训练诊断证据：

```text
center_motion_norm
scale_change_ratio
color_change_norm = 0
score_peak_second_gap = 0
score_entropy = 0
box_jump_ratio = center_motion_norm
deploy_score_delta = 0
partial_deploy_delta = 0
```

其中 center/scale 来自训练 batch 的 template/search GT 框。这个证据不是最终 deploy 输入，只用于先验证 updater 能不能在受控条件下被 tracking loss 驱动。

### 新配置

文件：`experiments/dutrack/dutrack_384_full_langstate_e5.yaml`

设计原则：

- 关闭旧 LMQ：`LMQ_ENABLE: false`
- 关闭 score prior：`SCORE_PRIOR_ENABLE: false`
- 只启用 latent token state updater：`LANGUAGE_STATE_ENABLE: true`
- 只训练 updater：`TRAIN_TE_ONLY_PATTERNS: [backbone.language_state_updater]`
- 不加 aux loss：只靠 `L1/GIoU/location` 反向传播
- `SAVE_LATEST_ONLY: true`，短训只保留一个 checkpoint

### 训练指令

```bash
python tracking/train.py \
  --script dutrack \
  --config dutrack_384_full_langstate_e5 \
  --save_dir ./output \
  --mode single \
  --use_wandb 0
```

如果当前 shell 没有激活 `DUTrack` 环境，用：

```bash
conda run -n DUTrack python tracking/train.py \
  --script dutrack \
  --config dutrack_384_full_langstate_e5 \
  --save_dir ./output \
  --mode single \
  --use_wandb 0
```

### 训练中重点观察

日志里新增：

```text
LanguageState/frame_gate_mean
LanguageState/token_gate_mean
LanguageState/combined_gate_mean
LanguageState/delta_abs_mean
LanguageState/state_delta_abs_mean
LanguageState/visual_center_motion_mean
LanguageState/visual_scale_change_mean
```

判断标准：

- 如果 `state_delta_abs_mean` 长期接近 0，说明 updater 没学开；
- 如果 gate 快速变大但 IoU/loss 变差，说明更新污染语言状态；
- 如果 IoU 不变但 gate/delta 有稳定非零，说明 tracking loss 对该模块的有效梯度可能仍偏弱。

## 保守正则版：抑制 gate 饱和

短训 `dutrack_384_full_langstate_e5` 出现了明确的饱和现象：

```text
epoch 1: token_gate_mean 约 0.20，combined_gate_mean 约 0.04
epoch 2: token_gate_mean 约 0.999，combined_gate_mean 约 0.91
epoch 3 early: token_gate_mean 约 1.0，delta_abs_mean 接近 max_delta
```

这说明 updater 学到的不是“选择性吸收语言增量”，而是“几乎所有 token 全量更新”。训练 loss 在 epoch 2 变好不能直接说明机制正确，因为这可能是短训集上的局部适配。

### 修改点

文件：

```text
lib/models/dutrack/language_token_state_updater.py
lib/train/actors/dutrack.py
lib/config/dutrack/config.py
experiments/dutrack/dutrack_384_full_langstate_reg_e5.yaml
```

新增可反传诊断项：

```text
gate_reg_loss  = mean(frame_gate * token_gate)
delta_reg_loss = mean(abs(H_state - H_prev))
```

actor 中新增训练正则：

```text
L_language_state =
  lambda_gate  * gate_reg_loss
  + lambda_delta * delta_reg_loss
```

它不是替代 tracking loss，而是给“打开更新”和“偏离上一帧语言状态”一个小代价。只有当最终 `L1/GIoU/location` 的收益足够大时，模型才应该打开 gate。

### 新配置

文件：

```text
experiments/dutrack/dutrack_384_full_langstate_reg_e5.yaml
```

相对 `dutrack_384_full_langstate_e5` 的关键差异：

```yaml
MODEL:
  TE:
    LANGUAGE_STATE_MAX_DELTA: 0.02
    LANGUAGE_STATE_INIT_GATE_BIAS: -4.0
    LANGUAGE_STATE_INIT_DELTA_STD: 0.0001
TRAIN:
  LR: 0.00001
  LANGUAGE_STATE_VISUAL_EVIDENCE: none
  LANGUAGE_STATE_GATE_LOSS_WEIGHT: 0.01
  LANGUAGE_STATE_DELTA_LOSS_WEIGHT: 0.2
```

`LANGUAGE_STATE_VISUAL_EVIDENCE: none` 是为了去掉 `gt_motion` shortcut，先看纯 token-state updater 在 tracking loss 和轻正则下是否还能学出非 no-op 更新。

### 训练指令

```bash
python tracking/train.py \
  --script dutrack \
  --config dutrack_384_full_langstate_reg_e5 \
  --save_dir ./output \
  --mode single \
  --use_wandb 0
```

如果没有激活环境：

```bash
conda run -n DUTrack python tracking/train.py \
  --script dutrack \
  --config dutrack_384_full_langstate_reg_e5 \
  --save_dir ./output \
  --mode single \
  --use_wandb 0
```

### 重点观察

需要同时看 tracking loss 和 updater 行为：

```text
LanguageState/token_gate_mean
LanguageState/frame_gate_mean
LanguageState/combined_gate_mean
LanguageState/state_delta_abs_mean
LanguageStateReg/gate_reg_loss
LanguageStateReg/delta_reg_loss
Loss/language_state
```

合理现象：

- gate 不应在 epoch 2 直接接近 1；
- `state_delta_abs_mean` 不应快速贴近 `LANGUAGE_STATE_MAX_DELTA`；
- 如果 gate 被压住但 IoU 完全无变化，说明当前语言状态增量信号仍弱；
- 如果 gate 再次饱和，说明需要更强的 no-update 负样本、identity 约束，或者将候选语言质量监督显式化。

### 仍然存在的风险

- 正则过强会把 updater 压回 no-op；
- 纯 tracking loss 的梯度仍可能不足以学出细粒度 token 选择；
- 关闭 visual evidence 后更接近 deploy，但也减少了训练信号；
- 当前训练里的 `H_anchor/H_prev/H_candidate` 仍来自 batch 内已有描述，不等同于真实推理中的 BLIP 候选更新链路。

## 中间档参数：避免 no-op 退化

`dutrack_384_full_langstate_reg_e5` 成功避免了 gate 饱和，但训练进展显示：

```text
token_gate_mean       约 0.001
frame_gate_mean       约 0.003
combined_gate_mean    约 0
state_delta_abs_mean  约 0
```

这说明它已经退化成接近 no-op。这里的“退化”不是 tracking 崩溃，而是 language-state updater 被正则压住，基本没有参与模型输出。

### 新增配置

文件：

```text
experiments/dutrack/dutrack_384_full_langstate_mid_e5.yaml
```

相对强正则版只改几个参数：

```yaml
TRAIN:
  LANGUAGE_STATE_GATE_LOSS_WEIGHT: 0.001
  LANGUAGE_STATE_DELTA_LOSS_WEIGHT: 0.05
MODEL:
  TE:
    LANGUAGE_STATE_MAX_DELTA: 0.02
    LANGUAGE_STATE_INIT_GATE_BIAS: -4.0
```

保留：

```yaml
LANGUAGE_STATE_VISUAL_EVIDENCE: none
LR: 0.00001
TRAIN_TE_ONLY_PATTERNS:
- backbone.language_state_updater
```

这版的目标不是追求马上提升 IoU，而是把 updater 从 `combined_gate_mean ~= 0` 拉回可观测区间，同时不回到旧版 `combined_gate_mean > 0.8` 的全开状态。

建议观察区间：

```text
combined_gate_mean: 1e-3 到 1e-2
state_delta_abs_mean: 非零但显著低于 0.02
```

为了避免日志五位小数看不出变化，`LanguageTokenStateUpdater` 额外记录：

```text
combined_gate_x1e4
state_delta_abs_x1e6
```

### 训练指令

```bash
python tracking/train.py \
  --script dutrack \
  --config dutrack_384_full_langstate_mid_e5 \
  --save_dir ./output \
  --mode single \
  --use_wandb 0
```

如果没有激活环境：

```bash
conda run -n DUTrack python tracking/train.py \
  --script dutrack \
  --config dutrack_384_full_langstate_mid_e5 \
  --save_dir ./output \
  --mode single \
  --use_wandb 0
```

### 风险判断

- 如果 gate 仍然接近 0：说明只有最终 tracking loss 的学习信号仍太弱，需要引入更明确的候选语言质量监督或 no-update/positive-update 对比。
- 如果 gate 再次快速接近 1：说明正则仍不足，或者模块找到了“全量更新”的短训捷径。
- 如果 gate 有中等打开但 IoU 不变：说明 token-state 更新有行为，但未传导成有效 tracking 增益，需要看 S0/可视化中的 score gap 和语言状态质量。

## keep-absorb + gain 的 source gate 软化版

### 背景

`dutrack_384_full_langstate_keepabsorb_gain_e5` 在 epoch 2 后仍然基本保持：

```text
prev_keep_weight_mean          约 0.9993
candidate_absorb_weight_mean  约 0.00034
anchor_weight_mean            约 0.00034
gap_state - gap_prev          约 0
gain_active_ratio             1.0
```

这说明 gain auxiliary 已经处于 active 状态，但三路 source softmax 的初始化过硬，candidate/anchor 分支概率太低，梯度很难把它们拉起来。此时继续单纯增加训练轮数意义不大。

### 代码改动

新增配置项：

```yaml
MODEL:
  TE:
    LANGUAGE_STATE_SOURCE_INIT_GATE_BIAS: null
```

该项只控制 keep-absorb 模式的三路 source gate 初始化，不再和 `LANGUAGE_STATE_INIT_GATE_BIAS` 绑定。

初始化逻辑：

```text
source order = [anchor, prev, candidate]
source bias  = [source_bias, -source_bias, source_bias]
```

例如：

```yaml
LANGUAGE_STATE_INIT_GATE_BIAS: -4.0
LANGUAGE_STATE_SOURCE_INIT_GATE_BIAS: -2.0
```

含义是：

- token/frame update gate 仍保持保守；
- source gate 从近似 `[0.00034, 0.99932, 0.00034]` 放松到约 `[0.0177, 0.9647, 0.0177]`；
- candidate 分支初始仍小，但不再几乎无梯度。

涉及文件：

```text
lib/models/dutrack/language_token_state_updater.py
lib/models/dutrack/itpn.py
lib/config/dutrack/config.py
```

### 新配置

文件：

```text
experiments/dutrack/dutrack_384_full_langstate_gainsoft_e5.yaml
```

关键差异：

```yaml
MODEL:
  TE:
    LANGUAGE_STATE_UPDATE_MODE: keep_absorb
    LANGUAGE_STATE_INIT_GATE_BIAS: -4.0
    LANGUAGE_STATE_SOURCE_INIT_GATE_BIAS: -2.0
TRAIN:
  LANGUAGE_STATE_GAIN_LOSS_WEIGHT: 0.05
  LANGUAGE_STATE_GAIN_MARGIN: 0.01
  LANGUAGE_STATE_GATE_LOSS_WEIGHT: 0.0
  LANGUAGE_STATE_DELTA_LOSS_WEIGHT: 0.0
```

### 观察重点

这版首先看 source gate 是否能离开 prev-only：

```text
LanguageState/prev_keep_weight_mean
LanguageState/candidate_absorb_weight_mean
LanguageState/anchor_weight_mean
LanguageState/source_entropy_mean
LanguageStateReg/gap_gain
LanguageStateReg/gain_loss
LanguageStateReg/gain_active_ratio
```

合理方向：

- `candidate_absorb_weight_mean` 不再长期固定在 `0.00034`；
- `prev_keep_weight_mean` 可以下降，但不应快速低于约 `0.8`；
- `gap_gain` 应该从 0 附近出现可观测变化；
- 如果 `gap_gain` 仍为 0，说明当前 cosine prior probe 本身仍不能给 language state 提供有效训练信号。

### 风险

- source gate 放松后，candidate 可能吸收错误语言，短训 IoU 可能波动；
- `LANGUAGE_STATE_GAIN_LOSS_WEIGHT=0.05` 仍然很小，但比旧版 0.02 更容易观察是否有梯度效果；
- 如果 source gate 大幅离开 prev 但 tracking loss 变差，说明 auxiliary 与最终 tracking 目标仍不一致。

## Stage 3 token 对齐修正

### 问题

前面的 `keep_absorb` 版本默认做逐位置混合：

```text
H_state_i =
  w_anchor_i * H_anchor_i
+ w_prev_i   * H_prev_i
+ w_cand_i   * H_candidate_i
```

这个公式隐含了一个强假设：

```text
anchor / prev / candidate 的第 i 个 token 表示同一个语义位置
```

但真实语言更新里，BLIP/current caption 会改变词序、长度和词集合。例如：

```text
anchor:    the dog running
candidate: a brown dog on grass
```

这时按 index 做 `candidate_i - prev_i` 没有稳定语义，source gate 学到的也不是“吸收哪个语义 token”，而是混合了一组错位的 token embedding。这是当前模块难以学习的基础问题之一。

### 改动

新增对齐模式：

```yaml
MODEL:
  TE:
    LANGUAGE_STATE_ALIGNMENT: position   # 默认旧行为
    LANGUAGE_STATE_ALIGNMENT_HEADS: 4
```

新实验使用：

```yaml
LANGUAGE_STATE_ALIGNMENT: cross_attn
```

实现逻辑：

```text
candidate_aligned = CrossAttention(query=H_prev, key=H_candidate, value=H_candidate)
anchor_aligned    = CrossAttention(query=H_prev, key=H_anchor,    value=H_anchor)
```

然后再做：

```text
features_i = [
  anchor_aligned_i,
  prev_i,
  candidate_aligned_i,
  candidate_aligned_i - anchor_aligned_i,
  candidate_aligned_i - prev_i
]
```

也就是说，状态 token 坐标系固定在 `H_prev` 上，candidate 和 anchor 先被投影到 prev slots。后续 token gate 的含义变成：

```text
在 prev 的第 i 个状态位置上，是否吸收与它最相关的 candidate 语义信息
```

这比原来的 index-wise 混合更接近 latent language state update。

### mask 修正

打开 `cross_attn` 后，`H_state` 的 token 坐标属于 `prev`，所以训练和 backbone 输入使用 `prev_mask`，而不是 `candidate_mask`。

涉及文件：

```text
lib/models/dutrack/language_token_state_updater.py
lib/models/dutrack/itpn.py
lib/train/actors/dutrack.py
lib/config/dutrack/config.py
```

### 新增诊断字段

```text
LanguageState/alignment_mode_cross_attn
LanguageState/anchor_alignment_entropy
LanguageState/anchor_alignment_max
LanguageState/candidate_alignment_entropy
LanguageState/candidate_alignment_max
```

需要重点看：

- `candidate_alignment_max` 是否明显高于均匀注意力；
- `candidate_alignment_entropy` 是否不是完全均匀；
- 若 alignment 始终均匀，说明文本 token embedding 本身不足以完成语义对齐，需要更强的词级/短语级 grounding 或显式 token-id 辅助。

### 新配置

文件：

```text
experiments/dutrack/dutrack_384_full_langstate_align_e5.yaml
```

它在 `gainsoft` 基础上只额外启用：

```yaml
LANGUAGE_STATE_ALIGNMENT: cross_attn
LANGUAGE_STATE_ALIGNMENT_HEADS: 4
```

训练目标不变，先单独验证“token 对齐”是否能让 source gate 和 `gap_gain` 出现更有效变化。

### 后续逐步处理的问题

1. token 对齐：当前已处理，先验证 cross-attn alignment 是否有效。
2. token-level visual evidence：下一步把每个 candidate token 对 target/hardneg 的响应作为 updater 输入。
3. token-level absorb supervision：将整体 `gap_state - gap_prev` 改成逐 token 的 absorb/keep 目标。
4. temporal reliability：再引入多帧稳定性，避免单帧视觉证据自举误导。

## Stage 3 token-level absorb auxiliary

### 动机

`gainsoft` 和 `align` 的主要问题是：即使 source gate 初始更软，整体 `gap_state - gap_prev` 仍然几乎不给出有效方向。原因是它只监督混合后的整句状态：

```text
gap(H_state) - gap(H_prev)
```

但真正需要学习的是：

```text
candidate 中哪些 token 应该吸收？
candidate 中哪些 token 应该拒绝？
```

因此新增逐 token 的吸收辅助目标。

### 目标定义

对每个 prev-slot 上的 token，计算：

```text
prev_gap_i = sim(prev_i, target_tokens) - sim(prev_i, hardneg_tokens)
cand_gap_i = sim(candidate_aligned_i, target_tokens) - sim(candidate_aligned_i, hardneg_tokens)
token_gain_i = cand_gap_i - prev_gap_i
```

其中：

- `target_tokens` 来自 GT center quality；
- `hardneg_tokens` 来自当前 score map 中目标外 top-k 高响应 token；
- `candidate_aligned_i` 是已经对齐到 `prev_i` slot 的候选语言 token。

监督信号：

```text
target_i = 1, if token_gain_i > margin
target_i = 0, otherwise
```

然后用 candidate source gate 作为预测：

```text
L_token_absorb = BCE(candidate_weight_i, target_i)
```

这个目标比整体 gain 更直接：它不要求整个 `H_state` 立刻变好，而是先告诉 source gate 哪些候选 token 有吸收价值。

实现时不直接对 softmax 后的 `candidate_weight_i` 调用 `binary_cross_entropy`，因为 AMP/autocast 下这不安全。代码里使用：

```text
candidate_absorb_logit_i =
  source_logit_candidate_i - logsumexp(source_logit_anchor_i, source_logit_prev_i)

L_token_absorb =
  BCEWithLogits(candidate_absorb_logit_i, target_i)
```

这样等价于把三路 source gate 转成“candidate vs non-candidate”的二分类 logit，同时兼容 AMP。

### 实现细节

`LanguageTokenStateUpdater` 额外把以下训练用张量放入 diagnostics 私有字段：

```text
_candidate_aligned_tokens
_candidate_weight
_prev_weight
_anchor_weight
```

actor 中新增：

```text
_compute_language_state_token_absorb_loss()
```

并新增配置：

```yaml
TRAIN:
  LANGUAGE_STATE_TOKEN_ABSORB_LOSS_WEIGHT: 0.0
  LANGUAGE_STATE_TOKEN_ABSORB_MARGIN: 0.005
```

### 新诊断字段

```text
LanguageStateReg/token_absorb_loss
LanguageStateReg/token_absorb_target_pos_ratio
LanguageStateReg/token_absorb_candidate_weight_pos
LanguageStateReg/token_absorb_candidate_weight_neg
LanguageStateReg/token_absorb_gain_mean
LanguageStateReg/token_absorb_margin
LanguageStateReg/token_absorb_hardneg_topk
```

重点看：

- `token_absorb_target_pos_ratio`：有多少 token 被判定为候选有增益；
- `token_absorb_candidate_weight_pos` 是否逐渐高于 `token_absorb_candidate_weight_neg`；
- 如果正样本比例长期接近 0，说明 candidate token 在当前视觉空间里确实没有稳定增益；
- 如果正样本比例正常但 gate 不分化，说明 updater 参数或 loss 权重仍不足。

### 新配置

文件：

```text
experiments/dutrack/dutrack_384_full_langstate_aligntoken_e5.yaml
```

关键配置：

```yaml
MODEL:
  TE:
    LANGUAGE_STATE_ALIGNMENT: cross_attn
    LANGUAGE_STATE_SOURCE_INIT_GATE_BIAS: -2.0
TRAIN:
  LANGUAGE_STATE_GAIN_LOSS_WEIGHT: 0.02
  LANGUAGE_STATE_TOKEN_ABSORB_LOSS_WEIGHT: 0.01
  LANGUAGE_STATE_TOKEN_ABSORB_MARGIN: 0.005
```

这版的目的不是直接追求 IoU，而是验证：

```text
token-level target 是否存在；
candidate gate 能否对 positive/negative token 产生区分。
```

## 2026-05-27：tokenstrict 版

上一轮 `aligntoken` 训练里，`token_absorb_target_pos_ratio` 很快升到接近 1，`candidate_absorb_weight_mean` 也被推到接近全吸收。这说明旧目标：

```text
cand_gap - prev_gap > 0.005
```

过宽。它会把“负 gap 里略好一点”的 candidate token 也当成正例，也没有约束 hard negative 是否被同步抬高。

### 新正例定义

本版把 token positive 从单条件改成三条件同时满足：

```text
relative gain:
  cand_gap > prev_gap + margin_rel

absolute positive:
  cand_gap > margin_abs

hard-negative suppression:
  cand_hardneg - prev_hardneg < hardneg_margin
```

默认配置：

```yaml
TRAIN:
  LANGUAGE_STATE_TOKEN_ABSORB_LOSS_WEIGHT: 0.002
  LANGUAGE_STATE_TOKEN_ABSORB_MARGIN_REL: 0.02
  LANGUAGE_STATE_TOKEN_ABSORB_MARGIN_ABS: 0.005
  LANGUAGE_STATE_TOKEN_ABSORB_HARDNEG_MARGIN: 0.0
```

这里的含义是：candidate token 必须比 prev token 明显更好，自己也必须是正 gap，并且不能靠同步抬高 hard negative 来获得表面 gap。

### candidate weight 上限正则

为避免 source gate 再次退化成“全 candidate”，新增：

```yaml
TRAIN:
  LANGUAGE_STATE_CANDIDATE_CAP_LOSS_WEIGHT: 0.01
  LANGUAGE_STATE_CANDIDATE_WEIGHT_MAX: 0.5
```

损失形式：

```text
L_cap = mean(ReLU(candidate_weight - max_weight)^2)
```

它不是禁止吸收 candidate，而是防止未被强证据支持的训练早期直接把候选语言压成主导状态。

### identity / anchor consistency

本版先不把 identity 条件写进监督目标，避免过早引入人工规则。只记录：

```text
token_absorb_anchor_cos_mean
token_absorb_anchor_cos_pos
token_absorb_anchor_cos_neg
```

如果后续仍有 false accept，再考虑把 anchor consistency 作为软权重或第四个 target 条件。

### 新诊断字段

重点新增：

```text
token_absorb_rel_ok_ratio
token_absorb_abs_ok_ratio
token_absorb_hard_ok_ratio
token_absorb_target_pos_ratio
token_absorb_prev_gap_mean
token_absorb_cand_gap_mean
token_absorb_hardneg_gain_mean
candidate_cap_active_ratio
candidate_cap_weight_mean
candidate_cap_weight_max
candidate_cap_over_mean
```

期望现象：

- `token_absorb_target_pos_ratio` 不应再快速接近 1；
- `candidate_absorb_weight_pos` 应该高于 `candidate_absorb_weight_neg`；
- `candidate_cap_active_ratio` 可以阶段性非零，但不应长期接近 1；
- `token_absorb_hardneg_gain_mean` 越低越好，正例不应该来自 hard negative 同步抬升。

### 配置

文件：

```text
experiments/dutrack/dutrack_384_full_langstate_tokenstrict_e5.yaml
```

## 2026-05-27：tokenstrict-v2 版

`tokenstrict` 第一版仍然在 ep2 退化成大面积 candidate absorb：

```text
token_absorb_target_pos_ratio ≈ 0.8
candidate_absorb_weight_mean ≈ 0.63
candidate_cap_active_ratio = 1.0
```

原因不是单纯 margin 太小，而是 target 仍然由当前 candidate 表征自举产生。一旦 candidate 分支把 hard negative 压低，三条件会同时大面积成立，导致后续标签继续偏向 candidate。

### v2 目标

v2 的目标是把 token absorb 改成“少数高置信语义增量”：

```text
positive token =
  relative gain ok
  AND absolute positive gap ok
  AND hard-negative suppression ok
  AND anchor identity ok
  AND multi-frame consistency ok
  AND top-ratio selected
```

### 新增多证据条件

1. `identity_ok`

使用 candidate aligned token 到 raw anchor token 集合的 max cosine：

```text
identity_support_i = max_j cos(candidate_aligned_i, anchor_raw_j)
identity_ok_i = identity_support_i > identity_min
```

这样比 aligned-anchor 逐位置相似更稳，因为后者本身也会被训练/对齐过程带偏。

2. `multi-frame consistency`

训练中两个 search frame 都满足 base positive 时，才进入候选正例：

```text
positive_base_i = positive_frame0_i AND positive_frame1_i
```

这只用于训练监督，避免单帧偶然正例；推理阶段不依赖未来帧。

3. `top-ratio positive`

即使满足全部条件，也只保留 evidence score 最高的一小部分：

```text
evidence_i = token_gain_i + cand_gap_i - hardneg_gain_i
final_positive = top 10% / max 6 tokens from positive_base
```

这一步是防止 `target_pos_ratio` 再次冲到 0.8 的关键。

4. hard negative token mining

对非正例但模型当前 candidate weight 偏高的 token 增加 BCE 权重：

```text
hard_negative_i =
  not positive_i
  AND candidate_weight_i > mean(candidate_weight)
```

默认负例权重倍率为 `2.0`。

### v2 配置

配置文件：

```text
experiments/dutrack/dutrack_384_full_langstate_tokenstrict_v2_e5.yaml
```

关键值调整为：

```yaml
TRAIN:
  LANGUAGE_STATE_GAIN_LOSS_WEIGHT: 0.005
  LANGUAGE_STATE_TOKEN_ABSORB_LOSS_WEIGHT: 0.001
  LANGUAGE_STATE_TOKEN_ABSORB_MARGIN_REL: 0.03
  LANGUAGE_STATE_TOKEN_ABSORB_MARGIN_ABS: 0.01
  LANGUAGE_STATE_TOKEN_ABSORB_HARDNEG_MARGIN: -0.005
  LANGUAGE_STATE_TOKEN_ABSORB_IDENTITY_ENABLE: true
  LANGUAGE_STATE_TOKEN_ABSORB_IDENTITY_MIN: 0.2
  LANGUAGE_STATE_TOKEN_ABSORB_TOP_RATIO: 0.1
  LANGUAGE_STATE_TOKEN_ABSORB_MAX_POS: 6
  LANGUAGE_STATE_TOKEN_ABSORB_MULTI_FRAME: true
  LANGUAGE_STATE_TOKEN_ABSORB_HARD_NEG_WEIGHT: 2.0
  LANGUAGE_STATE_CANDIDATE_CAP_LOSS_WEIGHT: 0.03
  LANGUAGE_STATE_CANDIDATE_WEIGHT_MAX: 0.35
```

### 新增/重点诊断

```text
token_absorb_base_pos_ratio
token_absorb_multiframe_ok_ratio
token_absorb_identity_ok_ratio
token_absorb_hard_negative_ratio
token_absorb_target_pos_ratio
candidate_cap_active_ratio
candidate_cap_weight_mean
candidate_absorb_weight_mean
prev_keep_weight_mean
```

期望：

- `token_absorb_target_pos_ratio` 应被压到约 `0.02~0.15`；
- `base_pos_ratio` 可以高，但 `target_pos_ratio` 不能高；
- `candidate_absorb_weight_mean` 不应再次超过 `0.5`；
- `candidate_weight_pos > candidate_weight_neg` 才说明可学习区分正在发生；
- 如果 `identity_ok_ratio` 很低，说明 anchor 证据过严；
- 如果 `multiframe_ok_ratio` 很低，说明当前 candidate 增益缺少跨帧稳定性。
