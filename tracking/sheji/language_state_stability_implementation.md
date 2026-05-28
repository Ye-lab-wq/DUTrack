# 语言状态稳定性实现记录

## 目标

当前问题不是 BLIP caption 语法不好，而是它会把跟踪目标身份改掉。典型风险是：

```text
head of the man on the bike -> a man riding a bike
```

这会把局部目标扩展成载体或背景上下文，后续 word-level prior 即使数学形式更合理，也会被错误语言源带偏。

## 当前实现

### 1. Tracker 语言策略可配置

修改位置：

```text
lib/config/dutrack/config.py
lib/test/tracker/dutrack.py
```

新增配置：

```yaml
TEST:
  LANGUAGE_INIT_SOURCE: blip            # blip / dataset_or_blip / dataset_or_class / class_or_blip
  LANGUAGE_UPDATE_MODE: caption_replace # caption_replace / anchor / off
```

含义：

- `caption_replace`：保留旧逻辑，触发更新时用 BLIP 整句替换当前描述。
- `anchor`：初始化时确定 identity anchor，后续即使触发 `updata_key`，也不再整句替换，只保持 anchor。
- `off`：与 `anchor` 一样不做整句替换，作为禁用动态语言更新的显式写法。

### 2. 初始语言优先级

新增稳定配置：

```text
dutrack_384_full_lte_keepvl_worddirect_margin_71523_l15prior_stagehead_langanchor_e10.yaml
```

该配置使用：

```yaml
LANGUAGE_INIT_SOURCE: dataset_or_class
LANGUAGE_UPDATE_MODE: anchor
```

优先级是：

```text
数据集/手工初始描述 -> object_class -> BLIP fallback
```

这样 OTB/HOOT/OLOD 如果已有人工或类别描述，会优先作为身份锚点；只有缺失时才退到 BLIP。

### 3. BLIP 延迟加载

旧代码在 tracker 初始化时直接加载 BLIP。现在改成 lazy load：只有 `LANGUAGE_INIT_SOURCE=blip` 或 `LANGUAGE_UPDATE_MODE=caption_replace` 真正需要生成 caption 时，才加载 BLIP。

这避免 anchor 模式下无意义的 BLIP 开销，也减少实验变量。

### 4. 诊断脚本支持运行时覆盖

修改位置：

```text
tracking/visualte_diagnostic.py
tracking/visualte_diagnostic_suite.py
```

新增参数：

```bash
--language_init_source dataset_or_class
--language_update_mode anchor
```

诊断 CSV 新增字段：

```text
language_anchor
language_source
language_candidate_description
```

Suite 汇总新增：

```text
language_source_unique_count
language_anchor_unique_count
```

用于确认当前测试是否真的稳定在一个语言锚点上。

## 当前边界

这一步只处理最危险的“整句替换漂移”。它还没有实现完整的 word reliability 状态更新。

后续更合理的方向是：

```text
固定原始词集合
+ 每帧根据 visual evidence 更新 word reliability
+ momentum 平滑
+ BLIP 只作为低权重候选词来源
```

也就是说，本版本是语言稳定性的第一阶段：先固定身份锚点，再讨论词级可靠性调制。

## 推荐诊断命令

同一 checkpoint 下对比旧动态语言与 anchor 语言：

```bash
python tracking/visualte_diagnostic_suite.py \
  --config dutrack_384_full_lte_keepvl_worddirect_margin_71523_l15prior_stagehead_e10 \
  --runid 10 \
  --stat_frames 0 \
  --vis_frames 5 \
  --top_ratio 0.1 \
  --language_init_source dataset_or_class \
  --language_update_mode anchor \
  --output_tag worddirect_margin_l15prior_langanchor_fullstats
```

训练 anchor 版本：

```bash
python tracking/train.py \
  --script dutrack \
  --config dutrack_384_full_lte_keepvl_worddirect_margin_71523_l15prior_stagehead_langanchor_e10 \
  --save_dir ./output \
  --mode single \
  --use_wandb 0
```

## 2026-05-23 更新：数据集语言标注、HOOT 原图视角、词可靠性筛选

### 1. 数据集侧语言标注

新增文件：

```text
lib/test/data_specs/language_descriptions.csv
lib/test/evaluation/language_annotations.py
```

当前标注：

```csv
dataset,sequence,description
olod,038,the small vehicle on the right road
hoot_balanced20,backpack-004,the red backpack held by the person
```

OLOD 和 HOOT dataset 构造 `Sequence` 时会自动读取该 CSV，并写入 `text_description`。这样更接近 TNL2K 的数据集侧语言输入，不需要在命令里硬编码描述。

### 2. HOOT 原图视角可视化

HOOT `backpack-004` 的大黑边来自 search crop padding，而不是图像读取错误。该序列首帧目标框高度接近整张图，`SEARCH_FACTOR=5.0` 后 crop 远大于原图，所以 `sample_target` 会产生大量黑色 padding。

新增诊断参数：

```bash
--original_view auto|on|off
```

`auto` 会对 HOOT 自动保存原图视角：

```text
0001_original_view.jpg
```

该图直接在原图上画 GT / Pred，不走 search crop，因此用于检查 HOOT 标注和预测位置更可靠。它不改变模型输入和评测逻辑。

### 3. 第一版视觉证据筛词

新增默认关闭配置：

```yaml
TEST:
  LANGUAGE_WORD_FILTER_ENABLE: False
  LANGUAGE_WORD_FILTER_THRESHOLD: 0.4
  LANGUAGE_WORD_FILTER_MOMENTUM: 0.8
  LANGUAGE_WORD_FILTER_MIN_KEEP: 2
```

新增诊断参数：

```bash
--language_word_filter 1
--language_word_filter_threshold 0.4
```

机制：

```text
anchor language tokens
  -> 每帧 forward 后读取 word_level_weights
  -> 将词权重归一化成 current visual evidence
  -> reliability_t = momentum * reliability_{t-1} + (1 - momentum) * evidence_t
  -> 下一帧使用 reliability 过滤后的语言短语
```

诊断 CSV 新增：

```text
language_filtered_description
language_word_filter_active
language_word_reliability
```

当前边界：

- 这是测试时的 hard word filtering，不是训练内的可微语言调制。
- 词一旦被过滤，后续恢复能力有限，因为模型下一帧看到的是过滤后的短语。
- 它用于先验证“视觉证据筛词”方向是否有正信号；如果有效，再考虑更稳的 soft word reliability 接入模型内部。

Smoke test：

```bash
python tracking/visualte_diagnostic_suite.py   --config dutrack_384_full_lte_keepvl_worddirect_71523_l15prior_stagehead_e10   --runid 10   --max_frames 1   --stat_frames 1   --vis_frames 1   --top_ratio 0.1   --score_prior_source word_direct_margin   --language_init_source dataset_or_class   --language_update_mode anchor   --language_word_filter 1   --original_view on   --case hoot_balanced20:0   --output_tag wordfilter_originalview_smoketest
```

已验证输出：

```text
output/test/visualte_diagnostic_suite/hoot_balanced20/wordfilter_originalview_smoketest/backpack-004/0001_original_view.jpg
```

## 2026-05-23 更新：Stage 1 词级视觉证据诊断

新增实现记录：

```text
tracking/sheji/language_update_stage1_word_evidence.md
```

该阶段只做诊断，不改变语言输入。核心输出是：

```text
word_reliability_diagnostics.csv
```

用于检查每个词在 search token 上的响应是否能区分目标区域和 hard negative 区域。

## 2026-05-23 更新：Stage 2 Soft Word Reliability

新增实现记录：

```text
tracking/sheji/language_update_stage2_soft_reliability.md
```

该阶段保持 anchor 句子不变，不做 hard filtering。tracker 根据上一帧 deploy-like target-hardneg gap 更新每个词的 reliability，并在下一帧将 reliability 乘进 word-level prior 的词权重。
