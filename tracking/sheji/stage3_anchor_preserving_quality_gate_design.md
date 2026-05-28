# Stage 3 Anchor-Preserving Quality Gate 设计说明

更新时间：2026-05-26

## 1. 当前版本定位

当前版本不是最终的端到端语言状态更新器，也不是训练模块。

它是一个诊断与轻量验证版本，用来回答：

```text
原 DUTrack 动态语言更新是否应该从“直接替换语言描述”
改成“anchor 保主体，BLIP 补状态”的更新方式？
```

核心思想：

```text
anchor description 维护目标身份；
BLIP/current caption 只作为当前状态候选；
tracking evidence 辅助判断候选是否采用；
采用后也不直接替换主体语言，而是保守组合。
```

---

## 2. 整体数据流

### 2.1 初始化

输入：

```text
first frame
initial bbox
dataset/class/BLIP initial language
```

初始化后得到：

```text
language_anchor
```

`language_anchor` 是长期身份描述，原则上不被 BLIP 替换。

示例：

```text
the dog
the small vehicle on the road
the red backpack held by the person
```

---

### 2.2 每帧 tracking 前的候选生成

原 DUTrack trigger 仍然保留，但重新定位为：

```text
candidate language generator trigger
```

而不是最终更新器。

触发依据包括：

```text
position change
scale change
color change
```

相关诊断字段：

```text
deploy_trigger
trigger_by_position
trigger_by_scale
trigger_by_color
trigger_area_ratio
trigger_center_distance
trigger_color_delta
```

如果 `candidate_mode=deploy_like`：

```text
trigger = false -> 不调用 BLIP
trigger = true  -> 调用 BLIP 生成 candidate caption
```

---

## 3. 三路语言源

每个可观测帧会比较三路语言：

```text
anchor: 初始主体语言
prev: 当前维护的语言状态
blip: 当前帧 BLIP candidate caption
```

当前诊断脚本会分别 forward：

```text
DUTrack(anchor)
DUTrack(prev)
DUTrack(blip)
```

并记录三路输出：

```text
anchor_score_gap
prev_score_gap
blip_score_gap

anchor_iou
prev_iou
blip_iou

anchor_description
prev_description
blip_description
```

其中 `score_gap` 指：

```text
mean(center score on positive region)
-
mean(center score on hard negative region)
```

---

## 4. Oracle 与 Deploy 的区别

### 4.1 Oracle 只用于离线诊断

Oracle 使用 GT 区域计算：

```text
quality_gate_score_delta
= score_gap(blip, GT region) - score_gap(prev, GT region)
```

由此定义：

```text
useful BLIP: quality_gate_score_delta > 0
harmful BLIP: quality_gate_score_delta < 0
```

Oracle 相关字段：

```text
oracle_selected_source
oracle_gain_over_prev
quality_gate_score_delta
quality_gate_true_accept
quality_gate_false_accept
quality_gate_true_reject
quality_gate_false_reject
```

注意：

```text
Oracle 不能部署，只用于评估 trigger/gate 判断是否正确。
```

---

### 4.2 Deploy gate 使用预测区域

部署可用证据使用当前预测框 `P_pred`：

```text
quality_gate_deploy_score_delta
= score_gap(blip, P_pred) - score_gap(prev, P_pred)
```

当前 deploy quality gate：

```text
accept =
    quality_gate_deploy_score_delta > eps
    and semantic_consistency >= threshold
    and confidence_ok
```

其中：

```text
semantic_consistency = content-word Jaccard(BLIP, anchor/prev)
confidence_ok = score peak / peak-second gap / box jump 保护
```

当前默认实验中 confidence 阈值通常关闭，因此：

```text
quality_gate_confidence_ok = 1
```

相关字段：

```text
quality_gate_deploy_score_delta
quality_gate_semantic
quality_gate_semantic_anchor
quality_gate_semantic_prev
quality_gate_confidence_ok
score_peak
score_second_peak
score_peak_second_gap
pred_box_jump_ratio
```

---

## 5. 当前发现的问题

全量诊断显示：

```text
原 trigger precision 约 30.9%
quality gate accept precision 约 81.7%
quality gate useful recall 约 76.3%
quality gate hurt rejection 约 92.0%
```

说明：

```text
quality gate 明显优于直接 deploy 更新。
```

但 false accept 仍然存在。

主要原因：

```text
deploy gate 依赖 P_pred；
当 P_pred 已经偏到背景或干扰物时，
BLIP 支持错误预测区域也会被误判为有益。
```

表现为：

```text
false_accept deploy_score_delta > 0
但 oracle_score_delta < 0
```

---

## 6. Word-Level 诊断

### 6.1 当前 word-level 怎么做

当前 word-level 只做诊断，不参与 gate。

实现方式：

```text
取 DUTrack backbone 输出中的 language tokens / template tokens / search tokens
L2 normalize
直接点积得到 word-token 与 visual-token 相似度
```

公式：

```text
sim_z = normalize(template_tokens) @ normalize(language_tokens)^T
sim_x = normalize(search_tokens)   @ normalize(language_tokens)^T
```

词分组：

```text
target words:
  BLIP 中与 anchor/prev content words 重合的词

context words:
  person / hand / road / tree / background 等高风险上下文词

new words:
  BLIP 中不在 anchor/prev 中的新 content words
```

记录字段：

```text
blip_word_target_template_gap_mean
blip_word_context_template_gap_mean
blip_word_new_template_gap_mean
blip_word_target_minus_context_template_gap
blip_minus_prev_target_template_gap

blip_word_target_search_deploy_gap_mean
blip_word_context_search_deploy_gap_mean
blip_word_new_search_deploy_gap_mean
blip_word_target_minus_context_search_deploy_gap
blip_minus_prev_target_search_deploy_gap
```

---

### 6.2 Word-level 诊断结论

当前结果为负。

`true_accept` 与 `false_accept` 在 word-level 证据上没有明显分开：

```text
false_accept 的 template word evidence 不比 true_accept 差；
target overlap / context dominance 也没有明显差异。
```

说明：

```text
当前 DUTrack 内部 language token 与 visual token 的简单点积，
不足以作为可靠 word grounding 信号。
```

因此：

```text
word-level evidence 目前保留为诊断字段；
不建议作为核心 gate 规则或在线可学习 gate 输入。
```

---

## 7. Anchor-Preserving State Update

### 7.1 为什么需要它

旧策略：

```text
gate accept -> prev_description = blip_description
```

风险：

```text
一次错误接受会让 BLIP 直接替换主体语言；
后续 prev_description 被污染；
动态语言状态可能漂移。
```

因此新增诊断策略：

```text
state_update_policy = anchor_state_gate
```

---

### 7.2 新策略数据流

如果 gate 拒绝：

```text
prev_description 不变
```

如果 gate 接受：

```text
prev_description = compact(anchor_description + BLIP state words)
```

其中：

```text
anchor_description 始终保留；
BLIP 只补充不在 anchor 中的 content words；
最多保留 6 个补充词，避免超过 BERT 16 token 限制。
```

当前实现函数：

```text
_compose_anchor_state_description(anchor, candidate, max_state_words=6)
```

示例：

```text
anchor: the dog
BLIP: a brown dog running on grass
state:  the dog brown running grass
```

该策略仍是文本级诊断，不是最终 latent state。

---

### 7.3 新增字段

```text
anchor_state_candidate_description
anchor_blip_content_overlap_count
prev_blip_content_overlap_count
```

用于检查：

```text
BLIP 是否保留 anchor 主体词；
组合后的语言是否仍以 anchor 为主体；
hard replacement 与 anchor-preserving 更新是否造成不同后续状态。
```

---

## 8. 代码位置

### 8.1 S0 单序列诊断

```text
tracking/language_state_s0_probe.py
```

关键逻辑：

```text
_candidate_description
_quality_gate
_compose_anchor_state_description
_add_word_evidence
state_update_policy branch
```

支持策略：

```text
oracle
gate
anchor_state_gate
none
```

---

### 8.2 S0 多序列筛选与 error report

```text
tracking/language_state_s0_screen.py
```

输出：

```text
s0_screen_summary.csv
s0_screen_summary.md
s0_error_report.csv
s0_error_report_summary.md
```

关键字段：

```text
trigger_error_type
gate_error_type
oracle_selected_source
deploy_selected_source
gate_selected_source
anchor_state_candidate_description
word evidence fields
```

---

### 8.3 离线 feature probe

```text
tracking/quality_gate_feature_probe.py
```

用途：

```text
读取 s0_error_report.csv
做 leave-one-sequence-out logistic gate 诊断
判断可部署特征是否有跨序列可学习边界
```

当前结果：

```text
learned gate 未超过 current gate；
说明现有统计特征和 word-level 点积证据泛化不足。
```

---

## 9. 当前推荐实验

### 9.1 Hard Replacement

```bash
python tracking/language_state_s0_screen.py \
  --config dutrack_384_full_lmq_d1_e10 \
  --dataset_names otb_lang,hoot_balanced20 \
  --sequence_names Bird1,Dog,Gym,potted_plant-008,toilet_paper-001,koala-003 \
  --runid 1 \
  --max_frames 0 \
  --candidate_mode deploy_like \
  --evidence_source score \
  --state_update_policy gate \
  --quality_gate_mode deploy \
  --quality_gate_gap_eps 0.0 \
  --quality_gate_semantic_thr 0.0 \
  --output_tag stage3_quality_gate_hard_replace_small
```

### 9.2 Anchor-Preserving

```bash
python tracking/language_state_s0_screen.py \
  --config dutrack_384_full_lmq_d1_e10 \
  --dataset_names otb_lang,hoot_balanced20 \
  --sequence_names Bird1,Dog,Gym,potted_plant-008,toilet_paper-001,koala-003 \
  --runid 1 \
  --max_frames 0 \
  --candidate_mode deploy_like \
  --evidence_source score \
  --state_update_policy anchor_state_gate \
  --quality_gate_mode deploy \
  --quality_gate_gap_eps 0.0 \
  --quality_gate_semantic_thr 0.0 \
  --output_tag stage3_quality_gate_anchor_state_small
```

---

## 10. 评价重点

对比 hard replacement 与 anchor-preserving：

```text
quality_gate_false_accept_rate
useful_update_recall
hurt_rejection_rate
quality_gate_gain
prev_iou / blip_iou / anchor_iou
后续帧 prev_description 是否漂移
```

如果 anchor-preserving 更好，说明：

```text
语言状态污染主要来自 hard replacement；
后续应进入 identity-preserving latent language state。
```

如果没有改善，说明：

```text
问题主要不在文本替换方式，
而在 candidate 质量判断或视觉状态可靠性。
```

---

## 11. 当前结论边界

当前可以说：

```text
quality gate 比原 trigger 直接更新更稳定；
word-level dot-product grounding 暂时不可靠；
anchor-preserving state update 是更合理的下一步诊断方向。
```

当前不能说：

```text
已经实现端到端语言状态更新；
已经证明 BLIP 动态语言能提升最终 tracking；
word-level grounding 已经解决目标/上下文区分；
learned quality gate 已经可部署。
```

