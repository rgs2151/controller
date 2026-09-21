"""Explicit tokenization modes for raw and preformatted model inputs."""

from __future__ import annotations


RAW_TEXT_TOKENIZATION = "raw_text_with_special_tokens_v1"
PREFORMATTED_CHAT_TOKENIZATION = "preformatted_chat_no_added_special_tokens_v1"


def tokenize_prompts(tokenizer, texts, *, prompt_tokenization: str, **kwargs):
    """Tokenize raw text or an already-rendered chat template exactly once."""

    if prompt_tokenization == RAW_TEXT_TOKENIZATION:
        return tokenizer(texts, add_special_tokens=True, **kwargs)
    if prompt_tokenization != PREFORMATTED_CHAT_TOKENIZATION:
        raise ValueError(f"Unknown prompt tokenization mode {prompt_tokenization!r}")

    values = [texts] if isinstance(texts, str) else list(texts)
    bos_token = getattr(tokenizer, "bos_token", None)
    if bos_token is not None and any(not value.startswith(bos_token) for value in values):
        raise ValueError(
            "Preformatted chat input must begin with the tokenizer BOS token"
        )
    encoded = tokenizer(texts, add_special_tokens=False, **kwargs)

    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is None or "input_ids" not in encoded:
        return encoded
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    rows = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
    masks = (
        attention_mask.tolist()
        if attention_mask is not None and hasattr(attention_mask, "tolist")
        else attention_mask
    )
    if rows and isinstance(rows[0], int):
        rows = [rows]
        masks = [masks] if masks is not None else None
    for index, row in enumerate(rows):
        mask = masks[index] if masks is not None else [1] * len(row)
        nonpadding = [token for token, keep in zip(row, mask, strict=True) if keep]
        if not nonpadding or nonpadding[0] != bos_token_id:
            raise ValueError("Preformatted chat input lost its leading BOS token")
        if len(nonpadding) > 1 and nonpadding[1] == bos_token_id:
            raise ValueError("Preformatted chat input contains a duplicated BOS token")
    return encoded

