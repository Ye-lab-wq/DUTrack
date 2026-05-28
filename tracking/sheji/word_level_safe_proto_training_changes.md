# Word-Level Safe Prototype Training Changes

## Motivation

The previous `safe_multi_proto` training version still pooled the whole language sentence into one `lang_context`.
That differs from the probe, where useful signal came from word-level scoring:

```text
word_weight_i -> sim(word_i, search_token_j) -> direct search signal
```

The new version keeps the deployable constraint, but moves the language-to-visual scoring closer to the probe.

## New Config

```text
experiments/dutrack/dutrack_384_full_lte_keepvl_wordlevel_safeproto_71523_stagehead_e10.yaml
```

Key settings:

```yaml
KEEP_VL_SOURCE: word_safe_multi_proto
POLICY_APPLY: none
SCORE_PRIOR_ENABLE: true
SCORE_PRIOR_SOURCE: decision
SCORE_PRIOR_BETA: 0.1
```

This keeps the prior attached only to center score logits. It still does not alter backbone QKV attention and does not affect size/offset branches.

## Module Change

File:

```text
lib/models/dutrack/language_token_emphasizer.py
```

New source:

```text
word_safe_multi_proto
```

Main flow:

```text
1. project language tokens, template tokens, search tokens
2. compute word-template similarity sim_z
3. compute word-search similarity sim_x
4. compute deployable word weights from:
   - template peakiness
   - search peakiness
   - a small learnable word score head
5. form direct search score:
   direct_j = sum_i word_weight_i * sim(word_i, search_j)
6. form template target/negative prototypes from word-weighted template score
7. combine:
   keep = direct_keep + gamma * bounded_positive_safe_margin
8. inject keep as score prior
```

Special tokens are approximated by dropping the first language token and the last valid token.
This is not as exact as the probe's tokenizer-string filtering, but it keeps the model interface deployable.

## New Diagnostics

`tracking/visualte_diagnostic.py` now writes:

```text
word_direct_L*_gap_in_minus_out
word_direct_L*_top10_precision
word_direct_L*_abs_sum
word_template_L*_mean/min/max/entropy
```

`tracking/visualte_diagnostic_suite.py` now includes:

```text
Word Direct Gap
Safe Margin Gap
```

These two columns are important:

- `Word Direct Gap`: whether word-level direct search signal points inside GT.
- `Safe Margin Gap`: whether target prototypes beat negative prototypes inside GT.

## What This Version Tests

This version tests whether the signal seen in the probe survives in the actual model forward path.

Expected useful pattern:

```text
word_direct_gap > 0
safe_margin_gap > 0
score_onoff_peak_delta >= 0
score_map_mass_in_gt increases compared with beta0
```

If `word_direct_gap` is positive but `safe_margin_gap` is negative, the direct word-search branch is useful but prototype confirmation is still hurting.

If both are negative, the deployable word weighting is not reproducing the probe.
