# Multi-Prototype Safe Prior Notes

## Motivation

The previous probe showed two different facts:

- `tracking_direct_search` is useful: tracking-aware word weights produce a positive target-vs-background search signal.
- single-prototype and target-only prototype matching are unstable: template-to-search matching can turn the useful language signal into background response.

The multi-prototype contrast probe improved the signal by comparing target prototypes against context, distractor, and background prototypes. However, two risks remain:

1. Context prototypes may be misused as negatives and suppress the true target.
2. A very strong `multi_contrast_search` may overfit the diagnostic probe because it uses oracle target and hard-negative regions.

## Modified Design

The updated probe keeps the original metrics and adds a safer variant:

```text
target_score = max sim(search_token, target_prototypes)
context_score = max sim(search_token, context_prototypes)
distractor_score = max sim(search_token, distractor_prototypes)
background_score = max sim(search_token, background_prototypes)
```

Each negative branch receives a gate:

```text
gate_n = sigmoid(scale * (mean(score_n on hard negatives) - mean(score_n on target)))
gate_n = clamp(gate_n, floor, 1)
```

Then:

```text
safe_negative = max(
  gate_context * context_score,
  gate_distractor * distractor_score,
  gate_background * background_score,
  0
)

safe_margin = target_score - safe_negative
safe_contrast = sigmoid(safe_margin / tau)
```

The deployable-style confirmation prior is bounded and anchored to direct search:

```text
direct_safe_multi_confirm =
  tracking_direct_search
  + safe_confirm_gamma * clamp(ReLU(safe_margin - safe_confirm_tau), max=safe_confirm_max)
```

This keeps the direct language-search branch as the main signal and only lets template prototypes add bounded confirmation.

## New Outputs

The probe now reports:

- `safe_multi_negative_search`
- `safe_multi_margin_search`
- `safe_multi_contrast_search`
- `direct_safe_multi_confirm_search`
- `context_negative_gate`
- `distractor_negative_gate`
- `background_negative_gate`

It also writes `*_multi_prototype_probe.jpg` with:

```text
direct | target max | contrast | safe margin | safe contrast | safe confirm | direct+confirm | score map
```

## Important Interpretation

Use these comparisons:

```text
tracking_direct_search
multi_contrast_search
safe_multi_contrast_search
direct_safe_multi_confirm_search
score_map
```

Expected decision logic:

- If `safe_multi_contrast_search` remains strong, target-vs-negative prototypes are genuinely useful.
- If `direct_safe_multi_confirm_search >= tracking_direct_search`, prototype confirmation adds useful information.
- If `direct_safe_multi_confirm_search < tracking_direct_search`, keep the direct branch and avoid prototype confirmation.
- If `safe` versions drop sharply while raw `multi_contrast_search` stays high, the previous contrast was likely too oracle/probe-specific.

## Command

```bash
python tracking/word_level_appearance_probe.py \
  --config dutrack_384_full_lte_keepvl_scoreprior_decision_71523_centerrank_anneal_e10 \
  --runid 10 \
  --dataset_name otb_lang \
  --sequence Biker \
  --max_frames 5 \
  --description "head of the man on the bike" \
  --subject_words head \
  --tag safe_multi_proto_head_desc
```

Optional knobs:

```bash
--negative_gate_scale 8.0
--negative_gate_floor 0.05
--safe_confirm_gamma 0.35
--safe_confirm_tau 0.0
--safe_confirm_max 0.25
```

## Training Implication

This is still a feasibility probe, not a final training module. It uses GT and score hard negatives for diagnosis. A later training version should approximate these with previous-frame boxes, base score hard negatives, and memory templates.

The lowest-risk implementation path remains:

```text
freeze backbone
do not alter QKV attention
do not alter size/offset branches
inject a bounded prior only into center score logits
```
