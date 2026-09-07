"""Matched-seed text generation under activation policies."""

from __future__ import annotations

import torch

from robust_steerability.modeling.interventions import register_generation_policy_hooks


def generate_completions(
    model,
    tokenizer,
    records: list[dict[str, object]],
    *,
    policy,
    seed: int,
    max_length: int,
    max_new_tokens: int,
    do_sample: bool,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
) -> list[dict[str, object]]:
    """Generate one continuation per record with stable per-prompt seeds."""

    model_device = next(model.parameters()).device
    output_rows = []
    for index, record in enumerate(records):
        prompt = str(record["prompt"] if "prompt" in record else record["text"])
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model_device)
        prompt_length = int(encoded["input_ids"].shape[1])
        sample_seed = seed + index
        torch.manual_seed(sample_seed)
        torch.cuda.manual_seed_all(sample_seed)
        handles = register_generation_policy_hooks(model, policy) if policy is not None else []
        generation_kwargs: dict[str, object] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "use_cache": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update({"top_p": top_p, "temperature": temperature})
        try:
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    **generation_kwargs,
                )
        finally:
            for handle in handles:
                handle.remove()
        completion_ids = generated[:, prompt_length:]
        output_rows.append(
            {
                "prompt_id": str(record["prompt_id"]),
                "prompt": prompt,
                "completion": tokenizer.decode(
                    completion_ids[0], skip_special_tokens=True
                ).strip(),
                "seed": sample_seed,
                "generated_tokens": int(completion_ids.shape[1]),
            }
        )
    return output_rows
