# Safe Multi-Prototype 训练版改动说明

## 目标

这版不是继续把 keep 插入 backbone attention，而是把 probe 中更稳的 `direct_safe_multi_confirm` 思路做成可训练近似版本：

```text
language/search direct keep
  + template target-vs-negative prototype bounded confirmation
  -> search keep
  -> center score logits prior
```

核心目标是让 keep 更直接服务 center head 的中心峰值，而不是大范围改变 QKV attention 流。

## 数据流

当前训练版的数据流是：

```text
language tokens
template tokens
search tokens
        |
        v
LanguageGuidedTokenEmphasizer
        |
        |-- template tokens + language context -> target prototypes
        |-- low/high language affinity template tokens -> negative prototypes
        |-- language-search direct keep -> deployable direct signal
        |
        v
safe_multi_proto search keep
        |
        v
DUTrack._score_prior_bias()
        |
        v
center score logits = base score logits + beta * normalized log(keep)
```

这版保持：

- 不使用 `POLICY_APPLY` 改 backbone attention。
- 不影响 size branch。
- 不影响 offset branch。
- 只影响 center score branch。

## 代码改动

### 1. `LanguageGuidedTokenEmphasizer`

文件：

```text
lib/models/dutrack/language_token_emphasizer.py
```

新增 `KEEP_VL_SOURCE = safe_multi_proto`。

主要函数：

```text
_score_search_safe_multi_proto()
```

它做三件事：

1. 用语言和 search tokens 先生成 direct keep。
2. 用语言和 template tokens 选 target prototypes，同时构造 context/background/distractor 风格的 negative prototypes。
3. 用 target score - gated negative score 得到 `safe_margin`，再用 bounded confirmation 修正 direct keep。

最终：

```text
keep = direct_keep + gamma * clamp(ReLU(safe_margin - tau), max=max_confirm)
keep = clamp(keep, 0, 1) * prev_decision_x
```

注意：这里的 prototype top-k 选择本身不是连续可微的，但被选中的 token 特征和后面的 score-prior 路径仍然可以收到梯度。它适合做轻量训练验证，不应该被理解成最终最优实现。

### 2. `itpn.py`

文件：

```text
lib/models/dutrack/itpn.py
```

新增从 config 读取以下参数，并传入 `LanguageGuidedTokenEmphasizer`：

```text
PROTO_TOPK_TARGET
PROTO_TOPK_NEGATIVE
PROTO_CONTRAST_TAU
SAFE_CONFIRM_GAMMA
SAFE_CONFIRM_TAU
SAFE_CONFIRM_MAX
NEGATIVE_GATE_SCALE
NEGATIVE_GATE_FLOOR
```

### 3. `config.py`

文件：

```text
lib/config/dutrack/config.py
```

新增默认配置键，避免 yaml 加载时报 `not exist in config.py`。

### 4. 新实验配置

文件：

```text
experiments/dutrack/dutrack_384_full_lte_keepvl_scoreprior_safemultiproto_71523_stagehead_e10.yaml
```

关键配置：

```text
KEEP_VL_SOURCE: safe_multi_proto
POLICY_APPLY: none
SCORE_PRIOR_ENABLE: true
SCORE_PRIOR_SOURCE: decision
SCORE_PRIOR_BETA: 0.1
SCORE_PRIOR_CENTER: mean
SCORE_PRIOR_BIAS_CLAMP: 0.35
```

## 训练策略

训练 10 epoch，保留 latest checkpoint：

```text
SAVE_LATEST_ONLY: true
```

分两阶段：

```text
epoch 1-5:
  train backbone.visual_te_predictors

epoch 6-10:
  train backbone.visual_te_predictors
  train box_head center score branch
```

也就是说，前半段先让 prior adapter 学会生成较稳定的 search keep，后半段再让 center score branch 轻微适应新的 prior 分布。

当前没有开放：

- backbone 主干参数
- size branch
- offset branch

## Loss 设置

这版沿用 score-space auxiliary，而不是 keep-space rank：

```text
AUX_SEARCH_LOSS_WEIGHT: 0.0
AUX_SCORE_LOSS_WEIGHT: 0.005
AUX_SCORE_LOSS_ANNEAL: cosine
AUX_SCORE_LOSS_WEIGHT_END: 0.001
AUX_SCORE_ANNEAL_EPOCHS: 10
```

score auxiliary 包含：

```text
L_corrective
L_prior_gain
bias_l2
```

其中：

- `L_corrective` 约束加入 prior 之后的 score map 正样本高于 hard negative。
- `L_prior_gain` 防止 prior 退化成 no-op，但权重较小。
- `bias_l2` 限制 prior bias 幅度，避免强行改坏原 head。

## 训练指令

如果已经激活环境：

```bash
python tracking/train.py \
  --script dutrack \
  --config dutrack_384_full_lte_keepvl_scoreprior_safemultiproto_71523_stagehead_e10 \
  --save_dir ./output \
  --mode single \
  --use_wandb 0
```

如果没有激活环境：

```bash
conda run -n DUTrack python tracking/train.py \
  --script dutrack \
  --config dutrack_384_full_lte_keepvl_scoreprior_safemultiproto_71523_stagehead_e10 \
  --save_dir ./output \
  --mode single \
  --use_wandb 0
```

## 已做的轻量验证

语法检查：

```bash
conda run -n DUTrack python -m py_compile \
  lib/models/dutrack/language_token_emphasizer.py \
  lib/models/dutrack/itpn.py \
  lib/models/dutrack/dutrack.py \
  lib/config/dutrack/config.py
```

配置加载检查：

```bash
conda run -n DUTrack python -c "from lib.config.dutrack.config import cfg, update_config_from_file; update_config_from_file('experiments/dutrack/dutrack_384_full_lte_keepvl_scoreprior_safemultiproto_71523_stagehead_e10.yaml'); print(cfg.MODEL.TE.KEEP_VL_SOURCE, cfg.MODEL.TE.PROTO_TOPK_TARGET, cfg.MODEL.TE.SAFE_CONFIRM_GAMMA, cfg.TRAIN.STAGED_TRAINING)"
```

输出：

```text
safe_multi_proto 4 0.35 True
```

模块形状检查：

```text
search_decision: B x search_tokens x 1
search_probs:    B x search_tokens x 2
```

## 预期观察点

训练日志重点看：

- `Loss/score_prior`
- `ScorePrior/active_corrective_ratio`
- `ScorePrior/active_prior_gain_ratio`
- `ScorePrior/prior_pos_gain`
- `ScorePrior/prior_hard_neg_gain`
- `ScorePrior/prior_bias_abs_mean`
- `ScorePrior/target_proto_score_on_pos`
- `ScorePrior/target_proto_score_on_hardneg`
- `ScorePrior/negative_proto_score_on_pos`
- `ScorePrior/negative_proto_score_on_hardneg`
- `ScorePrior/safe_margin_pos`
- `ScorePrior/safe_margin_hardneg`
- `ScorePrior/safe_margin_gap`
- `ScorePrior/safe_margin_hard_case_ratio`
- `IoU`
- `location`

可视化重点看：

- safe multi-proto 后的 search keep 是否比原 decision keep 更贴近目标。
- score prior bias 是否在目标中心附近给出正增益。
- score map peak delta 是否由负转正或至少不再明显压低目标峰值。
- size/offset 是否保持稳定。

新增的 prototype 诊断字段：

```text
safe_proto_target_L{layer}_*
safe_proto_negative_L{layer}_*
safe_proto_margin_L{layer}_*
```

其中 `{layer}` 对应当前配置中的 `7/15/23`。这些字段会写入 `visualte_diagnostic.py` 生成的 `diagnostics.csv`，用于判断 prototype 分组本身是否正确，而不是只看最后的 keep。

判断方式：

```text
target_proto_score_on_pos      > target_proto_score_on_hardneg
negative_proto_score_on_pos    < negative_proto_score_on_hardneg
safe_margin_pos                > safe_margin_hardneg
safe_margin_gap                > 0
safe_margin_hard_case_ratio    越低越好
```

如果这些条件不成立，说明问题主要在 target/negative prototype 分组；如果成立但 score map 没改善，问题更可能在 prior 接入强度、score head 适配或 base head 已经足够强。

## 风险

1. 当前 negative prototypes 仍然是可部署近似，不是 probe 中的 GT/context oracle。
2. top-k prototype selection 不完全可微，训练信号可能仍偏弱。
3. 如果 base center score 已经很强，prior 可能退化为很小修正。
4. 如果 `SCORE_PRIOR_BETA` 太大，可能把中心峰值推偏；如果太小，则看不出效果。

## 当前判断

这版的意义是验证一个更贴近当前结论的方向：

```text
不要让 keep 广泛改变 backbone QKV；
让 language/template/search 生成一个 bounded prior；
只把 prior 接入 center score logits；
用 staged head 训练让 score branch 适配。
```

如果这版仍然没有明显收益，主要问题就更可能是 keep 信息源本身仍不足，而不是 keep 接入位置的问题。
