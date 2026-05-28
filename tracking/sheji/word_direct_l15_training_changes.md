# Word-Direct L15 Prior Training Changes

## Purpose

This is a cleaner ablation of the word-level prior path.

Previous results showed:

- L15 `word_direct` is positive on the diagnostic suite.
- L23 `word_direct` is negative.
- Prototype confirmation can still produce negative margin even when direct word-search is positive.

This version removes prototype confirmation from the score prior path.

## Config

```text
experiments/dutrack/dutrack_384_full_lte_keepvl_worddirect_71523_l15prior_stagehead_e10.yaml
```

Key settings:

```yaml
KEEP_VL_SOURCE: word_direct
SCORE_PRIOR_LAYER: 15
SCORE_PRIOR_SOURCE: decision
SCORE_PRIOR_BETA: 0.1
POLICY_APPLY: none
SAFE_CONFIRM_GAMMA: 0.0
SAFE_CONFIRM_MAX: 0.0
```

## Mechanism

The direct prior uses word-to-search similarity only:

```text
word_weight_i = softmax(template_peak_i + search_peak_i + learned_word_i)
direct_score_j = sum_i word_weight_i * sim(word_i, search_token_j)
keep_j = sigmoid(center(direct_score_j) / tau)
score_logits = base_score_logits + beta * center(log(keep_j))
```

No template target/negative prototype confirmation is used in this variant.

## Diagnostics To Watch

Training log:

```text
word_direct_gap
prior_pos_gain
prior_hard_neg_gain
active_prior_gain_ratio
```

Visual diagnostic:

```text
word_direct_L15_gap_in_minus_out
score_map_mass_in_gt
score_onoff_peak_delta
```

Expected useful pattern:

```text
word_direct_gap > 0
prior_pos_gain > prior_hard_neg_gain
score_map_mass_in_gt(default) > score_map_mass_in_gt(beta0)
```
