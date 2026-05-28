# Stage 3-S：语言更新触发与文本质量判断实验指导

更新时间：2026-05-25

## 0. 当前实验目标

当前语言更新问题应拆成两个独立部分：

```text
Part A: 触发更新
判断什么时候需要调用 BLIP / current caption 生成候选语言。

Part B: 更新文本质量判断
判断生成出来的候选语言是否值得进入 tracker。
```

这两个问题不能混在一起。原论文的更新判决主要解决 **Part A**，而不是 **Part B**。

---

## 1. 总体实验逻辑

当前推荐流程是：

```text
Step 1:
复现并补全原论文 trigger：position + scale + color。

Step 2:
诊断 trigger 本身的召回和误触发情况。

Step 3:
在 trigger 后加入 language quality gate，判断 BLIP/current caption 是否值得采用。

Step 4:
只有当 quality gate 能稳定拒绝坏 BLIP、保留有用 BLIP 后，再考虑 latent language state updater。
```

核心思想：

```text
trigger 负责“是否需要生成候选语言”；
quality gate 负责“候选语言是否值得使用”。
```

---

## 2. Part A：触发更新实验

### 2.1 原论文 trigger 的含义

原论文动态语言更新触发条件关注目标状态变化，主要包括：

```text
1. position change：目标中心位置变化；
2. scale change：目标框尺度变化；
3. color change：目标框内 RGB 均值变化。
```

它的目的不是判断 BLIP 文本质量，而是判断：

```text
目标视觉状态是否变化明显？
是否有必要重新生成语言描述？
```

### 2.2 当前代码需要确认的内容

检查当前实现是否完整包含：

```text
center distance / position change
area ratio / scale change
RGB mean difference / color change
```

如果当前只包含 position 和 scale，需要补充 color change。

建议记录三个独立标志：

```text
trigger_by_position
trigger_by_scale
trigger_by_color
```

以及最终：

```text
deploy_trigger = trigger_by_position or trigger_by_scale or trigger_by_color
```

### 2.3 position / scale trigger 参考

当前已有形式类似：

```python
area_ratio = min(area_prev, area_cur) / (max(area_prev, area_cur) + 1e-12)
center_distance = sqrt((cx_prev - cx_cur)^2 + (cy_prev - cy_cur)^2)

if area_ratio < scale_thr:
    trigger = True
if center_distance > distance_thr:
    trigger = True
```

### 2.4 color trigger 建议

在 `his_bbox` 和 `cur_bbox` 对应区域内分别计算 RGB 均值：

```text
mean_rgb_prev = mean(image_prev[bbox_prev])
mean_rgb_cur  = mean(image_cur[bbox_cur])

color_delta = ||mean_rgb_prev - mean_rgb_cur||_2
```

触发条件：

```text
if color_delta > color_thr:
    trigger_by_color = True
```

第一轮不要急着调阈值，先记录统计分布：

```text
color_delta_mean
color_delta_max
color_trigger_rate
```

再决定阈值。

---

## 3. Part A 的诊断指标

对每个序列统计：

```text
deploy_trigger_rate
trigger_by_position_rate
trigger_by_scale_rate
trigger_by_color_rate
candidate_available_rate
```

同时结合 oracle 结果统计：

```text
deploy/oracle agree ratio
deploy false-positive ratio
deploy missed-oracle ratio
```

解释：

```text
deploy false-positive 高：
  触发太频繁，很多 BLIP 并没有带来更好语言证据。

deploy missed-oracle 高：
  触发太保守，漏掉了潜在有用更新。

missed-oracle 低但 false-positive 高：
  trigger 召回较高，但精度较低。
  后续应重点做 quality gate，而不是单纯增加触发。
```

当前全量结果倾向于：

```text
missed-oracle 较低；
false-positive 较高。
```

因此原 trigger 更像高召回候选生成器，而不是可靠更新器。

---

## 4. Part B：更新文本质量判断实验

### 4.1 为什么需要 quality gate

即使 trigger 正确，BLIP 生成的语言也可能不可靠，例如：

```text
描述背景；
描述遮挡物；
描述载体人物；
过于泛化；
描述错误目标。
```

因此需要第二道门：

```text
if trigger:
    blip_caption = BLIP(image)
    if quality_gate(blip_caption):
        update language
    else:
        keep previous language
```

### 4.2 quality gate 的目标

quality gate 不判断 caption 是否语法好，而是判断：

```text
候选语言是否对当前 tracking 有用；
候选语言是否仍然保持目标身份；
候选语言是否会伤害 center score / hard negative 区分。
```

### 4.3 当前最重要的定义

给定语言源 `L`：

```text
score_gap(L) = mean(score on target positive region)
             - mean(score on hard negative region)
```

当前使用 `score` evidence，而不是 LMQ prior，作为主评价信号。

定义：

```text
BLIP better than prev:
  score_gap(BLIP) > score_gap(prev) + epsilon

BLIP hurts:
  score_gap(BLIP) < score_gap(prev) - epsilon

oracle_gap:
  max(score_gap(anchor), score_gap(BLIP), score_gap(prev))

oracle_gain_over_prev:
  oracle_gap - score_gap(prev)
```

注意：

```text
score_gap gain 是 score-space 判别性增益，
不等价于最终 IoU 增益。
```

---

## 5. 先做非学习版 quality gate

第一版不建议直接端到端训练。先做一个简单 gate 验证方向：

```text
accept_blip =
    score_gap(BLIP) > score_gap(prev) + epsilon
    and semantic_consistency(BLIP, anchor) > semantic_thr
```

其中：

```text
score_gap 条件：
  判断候选语言是否提升 target-hardneg 区分。

semantic_consistency 条件：
  判断候选语言是否仍然接近目标身份。
```

### 5.1 semantic consistency 可选实现

第一版可以先用文本 embedding cosine：

```text
sim_anchor_blip = cosine(Embed(anchor), Embed(BLIP))
sim_prev_blip   = cosine(Embed(prev), Embed(BLIP))
```

也可以用简单词级规则做诊断：

```text
目标类别词是否仍出现；
BLIP 是否明显变成 person / background / room / road 等上下文；
BLIP 是否过泛化。
```

这些先用于记录，不要一开始写成强规则。

---

## 6. quality gate 的评价指标

不要只看更新率，应统计四类结果：

```text
true accept:
  BLIP better，gate 接受

false reject:
  BLIP better，gate 拒绝

true reject:
  BLIP hurts，gate 拒绝

false accept:
  BLIP hurts，gate 接受
```

核心指标：

```text
useful_update_recall = true_accept / (true_accept + false_reject)

hurt_rejection_rate = true_reject / (true_reject + false_accept)

false_accept_rate = false_accept / (true_reject + false_accept)

gate_gain = score_gap(after_gate) - score_gap(prev)
```

目标：

```text
尽量保留有用 BLIP；
尽量拒绝有害 BLIP；
至少不降低 anchor / prev 的稳定性。
```

---

## 7. 触发与质量判断的组合实验

建议比较四组：

### A. Anchor / Prev baseline

```text
不使用 BLIP；
始终保持 anchor 或 prev-state。
```

用途：

```text
稳定基线。
```

### B. 原 deploy trigger

```text
只要原 trigger 触发，就采用 BLIP。
```

用途：

```text
复现原项目动态语言更新行为。
```

### C. deploy trigger + quality gate

```text
trigger 只负责生成候选；
quality gate 决定是否采用。
```

用途：

```text
验证 gate 是否能降低 false-positive update。
```

### D. oracle upper bound

```text
每帧在 anchor / BLIP / prev 中选择 score_gap 最大者。
```

用途：

```text
诊断上界，不是部署机制。
```

最终报告中重点比较：

```text
B vs C:
  quality gate 是否改善原 deploy 策略。

C vs D:
  gate 离 oracle 上界还有多远。

A vs C:
  动态语言机制是否真正优于不更新。
```

---

## 8. 序列分组建议

根据当前 S0 筛选结果，将序列分为三类。

### 8.1 稳定负例

特征：

```text
anchor/prev 已经很好；
BLIP 没有明显增量；
deploy false-positive 高。
```

用途：

```text
验证 gate 能否拒绝无效更新。
```

### 8.2 正负混合困难例

特征：

```text
baseline IoU 较低；
BLIP 有时有用，但经常有害；
oracle_gain 有一定正值。
```

用途：

```text
验证 gate 能否同时保留有用更新、拒绝有害更新。
```

### 8.3 BLIP 高风险例

特征：

```text
BLIP_hurts_ratio 高；
caption 容易描述背景、遮挡物或上下文。
```

用途：

```text
验证 gate 的鲁棒性。
```

---

## 9. 当前不建议马上做的事

暂时不要：

```text
1. 直接训练复杂 latent language state updater；
2. 直接开放 center head / score adapter；
3. 放大 LMQ prior beta；
4. 只靠 LMQ prior gap 判断语言质量；
5. 直接用 BLIP 替换 anchor；
6. 只调 trigger 阈值而不做 quality gate。
```

原因：

```text
当前主要问题是：
trigger 召回不差，但触发后的候选语言质量不稳定。
```

---

## 10. 推荐执行顺序

### Step 1：补全 trigger 诊断

```text
position + scale + color
```

记录：

```text
trigger_by_position
trigger_by_scale
trigger_by_color
```

### Step 2：跑 S0 全量 score evidence

输出：

```text
score_gap_anchor
score_gap_blip
score_gap_prev
oracle_gain
deploy_false_positive
deploy_missed_oracle
```

### Step 3：分析 BLIP caption 质量

对 hurt 帧记录：

```text
frame_id
anchor_caption
blip_caption
prev_caption
score_gap_anchor
score_gap_blip
score_gap_prev
```

人工归类：

```text
background drift
occluder drift
carrier/context drift
too generic
wrong object
useful but score not reflected
```

### Step 4：实现非学习 quality gate

先用：

```text
score_gap_delta + semantic_consistency
```

验证 gate 是否能降低 harmful update。

### Step 5：再考虑 learnable quality gate

输入：

```text
score_gap_blip - score_gap_prev
score_gap_blip - score_gap_anchor
base_score_confidence
semantic_consistency_blip_anchor
semantic_consistency_blip_prev
bbox_area_ratio
bbox_center_distance
color_delta
```

输出：

```text
accept_blip probability
```

训练方式：

```text
先做离线二分类诊断；
再考虑端到端接入 tracker。
```

### Step 6：只有 gate 成立后，再做 latent language state updater

进入：

```text
H_t = g_t * H_candidate + (1 - g_t) * H_{t-1}
```

否则不要过早进入复杂状态维护。

---

## 11. 最终结论

当前实验应优先处理：

```text
触发更新：
  什么时候生成候选语言。

更新质量判断：
  生成出来的语言是否值得采用。
```

原论文 trigger 更适合作为高召回的候选生成器，而不是可靠更新器。当前全量 S0 结果显示，BLIP 有时有用但经常有害，因此后续最有价值的方向是：

```text
deploy trigger + language quality gate
```

先让 gate 过滤有害 BLIP，再考虑更复杂的 latent language state updater。

---

## 12. 工程实现记录：Trigger 诊断补全

更新时间：2026-05-25

### 12.1 修改范围

本轮只处理 Part A：语言候选生成触发。

修改文件：

```text
lib/config/dutrack/config.py
lib/test/tracker/dutrack.py
tracking/language_state_s0_probe.py
tracking/language_state_s0_screen.py
tracking/visualte_diagnostic.py
```

暂时不接入 quality gate，也不训练 latent language state updater。

### 12.2 新增配置

```python
cfg.TEST.LANGUAGE_TRIGGER_SCALE_THR = 0.95
cfg.TEST.LANGUAGE_TRIGGER_DISTANCE_STRIDE = 0.03125
cfg.TEST.LANGUAGE_TRIGGER_COLOR_ENABLE = False
cfg.TEST.LANGUAGE_TRIGGER_COLOR_THR = 35.0
```

其中：

- `LANGUAGE_TRIGGER_SCALE_THR`：面积比例低于该阈值触发 scale update；
- `LANGUAGE_TRIGGER_DISTANCE_STRIDE`：中心位移阈值相对图像宽高的比例；
- `LANGUAGE_TRIGGER_COLOR_ENABLE`：是否让 color delta 参与最终触发；
- `LANGUAGE_TRIGGER_COLOR_THR`：RGB 均值 L2 差异阈值。

注意：color trigger 默认关闭。当前阶段先记录 `color_delta` 分布，不默认改变原有 deploy 行为。

### 12.3 新增 tracker 诊断字段

`DUTrack.ifupdata()` 现在会拆分记录：

```text
language_trigger_by_position
language_trigger_by_scale
language_trigger_by_color
language_trigger_area_ratio
language_trigger_center_distance
language_trigger_color_delta
```

最终：

```text
updata_key =
    trigger_by_position
    or trigger_by_scale
    or trigger_by_color
```

当 `LANGUAGE_TRIGGER_COLOR_ENABLE=False` 时，`trigger_by_color` 恒为 false，但仍记录 `language_trigger_color_delta`。

### 12.4 color delta 计算

当前实现使用上一次语言更新参考帧 `his_image` 与当前帧 `image`：

```text
prev_rgb = mean(his_image[his_bbox])
cur_rgb  = mean(image[cur_bbox])
color_delta = ||prev_rgb - cur_rgb||_2
```

`his_image` 在初始化和 `_apply_language_update()` 时更新。

### 12.5 诊断输出

S0 probe / screen 新增：

```text
trigger_by_position
trigger_by_scale
trigger_by_color
trigger_area_ratio
trigger_center_distance
trigger_color_delta
```

S0 screen 汇总新增：

```text
trigger_by_position_rate
trigger_by_scale_rate
trigger_by_color_rate
trigger_color_delta_mean
```

`visualte_diagnostic.py` 的 `diagnostics.csv` 也同步记录这些字段。

### 12.6 建议实验

先不打开 color trigger，只看分布：

```bash
python tracking/language_state_s0_screen.py \
  --config dutrack_384_full_lmq_d1_e10 \
  --dataset_names otb_lang,hoot_balanced20 \
  --runid 1 \
  --max_frames 0 \
  --candidate_mode deploy_like \
  --evidence_source score \
  --output_tag stage3_s0_trigger_deploy_full
```

如果 color delta 分布显示有明显区分，再新建配置打开：

```yaml
TEST:
  LANGUAGE_TRIGGER_COLOR_ENABLE: true
  LANGUAGE_TRIGGER_COLOR_THR: 35.0
```

也可以临时用环境变量覆盖：

```bash
DUTRACK_LANGUAGE_TRIGGER_COLOR_ENABLE=1 \
DUTRACK_LANGUAGE_TRIGGER_COLOR_THR=35 \
python tracking/language_state_s0_screen.py ...
```

然后比较：

```text
deploy_trigger_rate
trigger_by_color_rate
deploy_false_positive_ratio
deploy_missed_oracle_ratio
```

如果 color trigger 降低 missed-oracle 但显著增加 false-positive，则后续必须依赖 quality gate，而不是继续放宽 trigger。

---

## 13. 工程实现记录：Part B 非学习 Quality Gate 诊断

更新时间：2026-05-25

### 13.1 当前实现范围

本轮先实现非学习版 quality gate 的诊断，不直接改正式 tracker 推理路径。

修改文件：

```text
tracking/language_state_s0_probe.py
tracking/language_state_s0_screen.py
```

### 13.2 Gate 定义

对触发后生成的 BLIP candidate，计算：

```text
score_delta = score_gap(BLIP) - score_gap(prev)
semantic = Jaccard(content_words(BLIP), content_words(anchor/prev))
```

默认 deploy gate：

```text
accept_blip =
    deploy_score_delta > quality_gate_gap_eps
    and semantic >= quality_gate_semantic_thr
    and confidence_ok
```

其中：

```text
deploy_score_delta =
    deploy_score_gap(BLIP) - deploy_score_gap(prev)
```

`deploy_score_gap` 使用预测框 token 作为正样本区域，不使用 GT。

保留 `--quality_gate_mode oracle` 作为上界诊断；默认 `deploy` 更接近推理场景。

默认参数：

```text
quality_gate_gap_eps = 0.0
quality_gate_semantic_thr = 0.0
quality_gate_semantic_ref = max(anchor, prev)
```

这相当于第一版主要看 score-gap 是否改善，语义一致性先只记录/轻约束。

### 13.3 新增字段

单帧 probe 新增：

```text
quality_gate_observable
quality_gate_accept
quality_gate_gain_over_prev
quality_gate_score_delta
quality_gate_semantic
quality_gate_semantic_anchor
quality_gate_semantic_prev
quality_gate_source
quality_gate_true_accept
quality_gate_false_reject
quality_gate_true_reject
quality_gate_false_accept
quality_gate_oracle_accept
quality_gate_deploy_accept
quality_gate_confidence_ok
quality_gate_deploy_score_delta
score_peak
score_peak_second_gap
pred_box_jump_ratio
```

screen 汇总新增：

```text
quality_gate_accept_rate
quality_gate_gain
quality_gate_semantic
quality_gate_true_accept_rate
quality_gate_false_reject_rate
quality_gate_true_reject_rate
quality_gate_false_accept_rate
useful_update_recall
hurt_rejection_rate
quality_gate_oracle_accept_rate
quality_gate_deploy_accept_rate
quality_gate_confidence_ok_rate
quality_gate_deploy_score_delta
score_peak_mean
score_peak_second_gap_mean
pred_box_jump_ratio_mean
```

### 13.4 状态更新策略

新增参数：

```text
--state_update_policy oracle | gate | none
```

含义：

- `oracle`：沿用 S0 上界诊断，在 anchor / BLIP / prev 中选最大 gap；
- `gate`：只有 quality gate 接受 BLIP 时才更新 prev language；
- `none`：prev language 不更新。

Part B 应优先用：

```text
--state_update_policy gate
```

### 13.5 代表序列运行指令

OTB：

```bash
python tracking/language_state_s0_screen.py \
  --config dutrack_384_full_lmq_d1_e10 \
  --dataset_names otb_lang \
  --sequence_names Bird1,Dog \
  --runid 1 \
  --max_frames 0 \
  --candidate_mode deploy_like \
  --evidence_source score \
  --state_update_policy gate \
  --quality_gate_mode deploy \
  --quality_gate_gap_eps 0.0 \
  --quality_gate_semantic_thr 0.0 \
  --output_tag stage3_quality_gate_otb_score
```

HOOT：

```bash
python tracking/language_state_s0_screen.py \
  --config dutrack_384_full_lmq_d1_e10 \
  --dataset_names hoot_balanced20 \
  --sequence_names potted_plant-008,toilet_paper-001 \
  --runid 1 \
  --max_frames 0 \
  --candidate_mode deploy_like \
  --evidence_source score \
  --state_update_policy gate \
  --quality_gate_mode deploy \
  --quality_gate_gap_eps 0.0 \
  --quality_gate_semantic_thr 0.0 \
  --output_tag stage3_quality_gate_hoot_score
```

如果发现 false accept 偏高，再提高阈值：

```text
--quality_gate_gap_eps 0.0005
--quality_gate_semantic_thr 0.1
```

如果困难帧中 deploy gate 被错误预测带偏，加入置信度保护：

```text
--quality_gate_score_peak_thr 0.3
--quality_gate_peak_gap_thr 0.02
--quality_gate_box_jump_thr 2.0
```

三个保护条件分别对应：

- base score peak 足够高；
- peak-to-second-peak gap 足够大；
- 预测框相对上一帧目标尺度没有剧烈跳变。

### 13.6 结果判读

优先看：

```text
quality_gate_gain > 0
hurt_rejection_rate 高
quality_gate_false_accept_rate 低
useful_update_recall 不要过低
```

如果 `hurt_rejection_rate` 高但 `useful_update_recall` 很低，说明 gate 太保守。

如果 `useful_update_recall` 高但 `quality_gate_false_accept_rate` 也高，说明 gate 无法过滤坏 BLIP。

如果 `quality_gate_gain` 仍接近 0，说明当前 score-gap 证据下的语言更新收益本身很弱，后续不应急着做复杂 latent state updater。
