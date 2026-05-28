# Multi-Prototype Safe Prior Results

## Experiment Setup

- Config: `dutrack_384_full_lte_keepvl_scoreprior_decision_71523_centerrank_anneal_e10`
- Dataset/sequence: `otb_lang:Biker`
- Frames: `5`
- Fixed description: `head of the man on the bike`
- Manual subject word: `head`
- Output directory:

```text
output/test/word_level_appearance_probe/dutrack_384_full_lte_keepvl_scoreprior_decision_71523_centerrank_anneal_e10_safe_multi_proto_head_desc/Biker
```

## Main Results

| Source | GT mass | Top10 precision | In-out gap |
| --- | ---: | ---: | ---: |
| `tracking_direct_search` | 0.0699 | 0.2103 | 0.1219 |
| `multi_contrast_search` | 0.1961 | 0.3793 | 0.2987 |
| `safe_multi_margin_search` | 0.0674 | 0.3310 | 0.1347 |
| `safe_multi_contrast_search` | 0.0692 | 0.3310 | 0.2528 |
| `direct_safe_multi_confirm_search` | 0.0727 | 0.2724 | 0.1556 |
| `score_map` | 0.6399 | 0.3379 | 0.0691 |

## Negative Gates

| Gate | Mean | Min | Max |
| --- | ---: | ---: | ---: |
| `context_negative_gate` | 0.6823 | 0.6213 | 0.7591 |
| `distractor_negative_gate` | 0.6823 | 0.6213 | 0.7591 |
| `background_negative_gate` | 0.7248 | 0.5942 | 0.9294 |

## Interpretation

The safe variant keeps most of the useful signal from raw multi-prototype contrast:

- Raw `multi_contrast_search` is strongest, with gap `0.2987` and Top10 `0.3793`.
- `safe_multi_contrast_search` remains strong, with gap `0.2528` and Top10 `0.3310`.
- The deployable-style `direct_safe_multi_confirm_search` improves over `tracking_direct_search`: gap rises from `0.1219` to `0.1556`, and Top10 rises from `0.2103` to `0.2724`.

This suggests that target-vs-negative prototype contrast is not only a visualization artifact. The safe gates reduce the raw contrast strength but do not destroy the target signal.

## Risk Check

### Risk 1: Context Prototype Misuse

The gates are not saturated at `1.0`, and they are not collapsed to the floor:

```text
context gate mean    = 0.6823
distractor gate mean = 0.6823
background gate mean = 0.7248
```

This means the negative branches are partially active. They are not blindly treated as hard negatives everywhere. This reduces the risk that context prototypes suppress the real target.

### Risk 2: Raw Contrast Overfitting

Raw contrast is much stronger than the bounded deployable version:

```text
multi_contrast_search              gap = 0.2987
direct_safe_multi_confirm_search   gap = 0.1556
```

So raw `multi_contrast_search` should still be treated as an oracle-like probe signal. It is useful for proving that target-vs-negative prototype contrast exists, but it should not be directly injected into the tracker without constraints.

## Practical Conclusion

The most reasonable next design is not the raw contrast heatmap. It is the bounded confirmation prior:

```text
prior = tracking_direct_search + gamma * clamp(ReLU(safe_margin - tau), max=prior_max)
```

Then inject it only into the center score logits:

```text
score_logits = base_score_logits + beta * normalize(prior)
```

The safer training direction is:

```text
freeze backbone
do not modify QKV attention
do not affect size/offset branches
train only the word scoring / prototype prior adapter
optionally open the center score branch later with a small learning rate
```

## Current Decision

Keep these three candidates for the next training/probe stage:

1. `tracking_direct_search` as the simplest baseline.
2. `safe_multi_contrast_search` as the strongest diagnostic target-vs-negative signal.
3. `direct_safe_multi_confirm_search` as the most deployable prior candidate.

Do not use single prototype or target-only prototype as the main design, because previous probes showed:

```text
tracking_proto_search < 0
gt_template_search < 0
multi_target_max_search near 0 but Top10 = 0
```
