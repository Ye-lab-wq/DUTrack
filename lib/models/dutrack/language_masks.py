import string

import torch


TRACKING_STOP_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "of",
    "to",
    "from",
    "for",
    "by",
    "with",
    "without",
    "in",
    "on",
    "at",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "this",
    "that",
    "these",
    "those",
}

EVIDENCE_UNIT_ANCHOR_BLOCKLIST = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "of",
    "to",
    "from",
    "for",
    "by",
    "with",
    "without",
    "in",
    "on",
    "at",
    "as",
    "near",
    "behind",
    "beside",
    "under",
    "over",
}


def is_semantic_token(token):
    """Return False for padding/special tokens and high-frequency function words."""
    if token is None:
        return False

    token = str(token).strip().lower()
    if not token or token.startswith("["):
        return False

    # BERT wordpiece continuations can carry object semantics, e.g. ##board.
    base_token = token[2:] if token.startswith("##") else token
    if not base_token:
        return False
    if all(ch in string.punctuation for ch in base_token):
        return False
    return base_token not in TRACKING_STOP_WORDS


def is_evidence_anchor_token(token):
    """Return True when a token can be the center of a phrase evidence unit."""
    if token is None:
        return False

    token = str(token).strip().lower()
    if not token or token.startswith("["):
        return False

    base_token = token[2:] if token.startswith("##") else token
    if not base_token:
        return False
    if all(ch in string.punctuation for ch in base_token):
        return False
    return base_token not in EVIDENCE_UNIT_ANCHOR_BLOCKLIST


def _normalize_input_ids(input_ids):
    if torch.is_tensor(input_ids):
        input_ids_list = input_ids.detach().cpu().tolist()
    else:
        input_ids_list = input_ids

    if input_ids_list and isinstance(input_ids_list[0], int):
        input_ids_list = [input_ids_list]
    return input_ids_list


def _build_token_rule_mask(tokenizer, input_ids, valid_token_mask, predicate, fallback_to_valid=True):
    input_ids_list = _normalize_input_ids(input_ids)

    rows = []
    for row_ids in input_ids_list:
        tokens = tokenizer.convert_ids_to_tokens(row_ids)
        rows.append([predicate(token) for token in tokens])

    rule_mask = torch.tensor(
        rows,
        device=valid_token_mask.device,
        dtype=torch.bool,
    )
    if rule_mask.shape != valid_token_mask.shape:
        rule_mask = torch.zeros_like(valid_token_mask)

    rule_mask = rule_mask & valid_token_mask

    empty_rows = ~rule_mask.any(dim=1)
    if fallback_to_valid and empty_rows.any():
        rule_mask = rule_mask.clone()
        rule_mask[empty_rows] = valid_token_mask[empty_rows]
    return rule_mask


def build_semantic_token_mask(tokenizer, input_ids, valid_token_mask):
    """Build a semantic-token mask aligned with BERT token ids.

    The input valid_token_mask is still the source of truth for padding and
    special-token filtering. This function only removes non-semantic function
    words such as articles.
    """
    return _build_token_rule_mask(
        tokenizer,
        input_ids,
        valid_token_mask,
        is_semantic_token,
        fallback_to_valid=True,
    )


def build_evidence_anchor_mask(tokenizer, input_ids, valid_token_mask):
    """Build anchor positions for phrase-aware evidence units.

    Articles and lightweight function/relation words are not used as evidence
    centers, but they can still contribute as context in the local phrase pool.
    """
    return _build_token_rule_mask(
        tokenizer,
        input_ids,
        valid_token_mask,
        is_evidence_anchor_token,
        fallback_to_valid=False,
    )
