# Language-to-Visual TE

## Goal

This branch keeps the TE objective focused on visual token emphasis:

```text
language + visual tokens -> keep_z, keep_x
keep_z / keep_x -> modulate template/search tokens as K/V contributors
```

The keep value is generated per visual token `j`, not per attention pair `(i, j)`.
The normal attention score still comes from:

```text
S_ij = Q_i K_j^T / sqrt(d)
A_ij = softmax_j(S_ij)
Y_i = sum_j A_ij V_j
```

The TE branch predicts:

```text
keep_j = LTE(v_j, global_visual, language_context)
```

and uses it to suppress background visual tokens when they are read as Key/Value:

```text
post: A'_ij = (A_ij + A_ij * keep_j) / 2
pre:  A'_ij = softmax_j(S_ij + lambda * log(keep_j))
```

So the direct effect is not to overwrite `Q`, `K`, or `V`. It changes how much each template/search visual token contributes through the attention-weighted `V_j`.

## Token Roles

The fusion stream contains four token groups:

```text
[track] + [language] + [template] + [search]
```

- `track`: target query / temporal query. DUTrack's final head directly uses the first track token to match search features.
- `language`: semantic constraint. It describes category, attributes, and relations. It affects final tracking through fusion, not by directly producing the box.
- `template`: appearance reference from target crops. It tells the model what the target looks like.
- `search`: current-frame candidates. It contains target and background and is the most important group to suppress with TE.

The L->V TE path should primarily score template/search tokens:

```text
keep_z = f(template_token, template_global, language_context)
keep_x = f(search_token, search_global, language_context)
```

## Preferred L->V Mechanism

Use the language as a condition for the visual scorer, while preserving TE's original token-emphasis structure:

```text
language_context = pool(real_language_tokens)

for visual token v_j in template/search:
    global_visual = weighted_mean(visual_tokens, previous_keep)
    relation_j = concat(
        visual_j,
        global_visual,
        language_context,
        visual_j * language_context,
        abs(visual_j - language_context)
    )
    keep_j = MLP(relation_j)
    keep_j = keep_j * previous_keep_j
```

Then apply `keep_j` to the attention readout:

```text
S_ij = Q_i K_j^T / sqrt(d)
S'_ij = S_ij + lambda * log(keep_j)      # pre-softmax variant
A'_ij = softmax_j(S'_ij)
Y_i = sum_j A'_ij V_j
```

The first controlled experiment should keep `QUERY_SCOPE=track`, so only the target query reads the filtered template/search K/V. An `all` query-scope variant should remain an ablation because it may also suppress visual context needed by language/template/search tokens.

## Current Scope

For now, keep the visual-to-language branch available but do not treat it as the main mechanism. The main TE experiment should be:

```text
KEEP_VL=true
KEEP_LV=false
POLICY_APPLY=pre_softmax
QUERY_SCOPE=track
```

This is the cleanest version of language-conditioned TE: language helps decide which visual tokens should be emphasized or suppressed.
