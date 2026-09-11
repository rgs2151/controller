"""Generation calls matching the A-LQR evaluation scripts."""

from __future__ import annotations

from collections.abc import Callable

import torch

from robust_steerability.modeling.interventions import register_generation_policy_hooks
from robust_steerability.source_methods.protocol import GENERATION


def generate_full_texts(
    model,
    tokenizer,
    prompts: list[str],
    *,
    behavior: str,
    use_cache: bool,
) -> list[str]:
    """Generate one source batch and decode prompt plus continuation."""

    if tokenizer.padding_side != "left":
        raise ValueError("source generation requires left padding")
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
        next(model.parameters()).device
    )
    with torch.no_grad():
        output = model.generate(
            **encoded,
            **GENERATION[behavior],
            use_cache=use_cache,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.batch_decode(output.sequences, skip_special_tokens=True)


def source_completions(prompts: list[str], full_texts: list[str]) -> list[str]:
    """Apply the source scripts' decoded-string continuation extraction."""

    if len(prompts) != len(full_texts):
        raise ValueError("prompt and generation counts differ")
    return [text[len(prompt):].strip() for prompt, text in zip(prompts, full_texts, strict=True)]


def generate_unsteered(model, tokenizer, prompts: list[str], *, behavior: str) -> list[str]:
    return source_completions(
        prompts,
        generate_full_texts(model, tokenizer, prompts, behavior=behavior, use_cache=False),
    )


def generate_with_control_policy(model, tokenizer, prompts: list[str], policy, *, behavior: str) -> list[str]:
    """Generate with A-LQR, S-PID, or ActAddLFS at every decoder call."""

    handles = register_generation_policy_hooks(model, policy)
    try:
        full_texts = generate_full_texts(model, tokenizer, prompts, behavior=behavior, use_cache=True)
    finally:
        for handle in handles:
            handle.remove()
    return source_completions(prompts, full_texts)


def generate_batched(
    model,
    tokenizer,
    prompts: list[str],
    *,
    behavior: str,
    batch_size: int,
    seed: int,
    use_cache: bool,
    register_hooks: Callable[[], list[torch.utils.hooks.RemovableHandle]] | None = None,
    reset: Callable[[], None] | None = None,
) -> list[str]:
    """Generate source-style completions in bounded batches.

    The preserved scripts submit one sampling call per batch.  Resetting and
    registering hooks for each batch prevents PID state and first-call ActAdd
    state from leaking between independent prompt batches.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        torch.manual_seed(seed + start)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + start)
        if reset is not None:
            reset()
        handles = register_hooks() if register_hooks is not None else []
        try:
            full_texts = generate_full_texts(
                model,
                tokenizer,
                batch,
                behavior=behavior,
                use_cache=use_cache,
            )
        finally:
            for handle in handles:
                handle.remove()
        output.extend(source_completions(batch, full_texts))
    return output
