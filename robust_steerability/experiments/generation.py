"""Matched-seed text generation under activation policies."""

from __future__ import annotations

from pathlib import Path

import torch

from robust_steerability.artifacts import configuration_hash
from robust_steerability.experiments.diagnostics import read_json, sha256, write_json
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
    trace_directory: Path | None = None,
) -> list[dict[str, object]]:
    """Generate one continuation per record with stable per-prompt seeds."""

    model_device = next(model.parameters()).device
    if trace_directory is not None:
        if policy is not None and policy.recorder is None:
            raise ValueError("Trace caching requires a recording policy")
        trace_directory.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for index, record in enumerate(records):
        prompt = str(record["prompt"] if "prompt" in record else record["text"])
        sample_seed = seed + index
        prompt_key = configuration_hash({
            "prompt_id": str(record["prompt_id"]), "prompt": prompt, "seed": sample_seed,
            "max_length": max_length, "max_new_tokens": max_new_tokens, "do_sample": do_sample,
            "top_p": top_p, "temperature": temperature, "repetition_penalty": repetition_penalty,
        })
        checkpoint = None if trace_directory is None else trace_directory / (prompt_key + ".json")
        if checkpoint is not None and checkpoint.exists():
            saved = read_json(checkpoint)
            if sha256(trace_directory / saved["trace_file"]) != saved["trace_sha256"]:
                raise ValueError("Cached online trace changed")
            output_rows.append(saved)
            continue
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model_device)
        prompt_length = int(encoded["input_ids"].shape[1])
        torch.manual_seed(sample_seed)
        torch.cuda.manual_seed_all(sample_seed)
        handles = register_generation_policy_hooks(model, policy) if policy is not None else []
        generation_kwargs: dict[str, object] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty if do_sample else 1.0,
            "use_cache": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update({"top_p": top_p, "temperature": temperature, "top_k": 50})
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
                "input_token_ids": encoded["input_ids"][0].detach().cpu().tolist(),
                "completion_token_ids": completion_ids[0].detach().cpu().tolist(),
            }
        )
        if trace_directory is not None:
            trace = policy.recorder.finish() if policy is not None else {
                "schema_version": 1, "controller": "original",
                "control_energy": 0.0, "hidden_delta_energy_total": 0.0,
                "control_energy_definition": "Unsteered baseline: no interventions.",
            }
            trace.update({"prompt_id": str(record["prompt_id"]), "seed": sample_seed,
                          "prompt_tokens": prompt_length, "generated_tokens": int(completion_ids.shape[1])})
            trace["intervention_site"] = "none" if policy is None else policy.site
            if policy is None:
                trace["coordinate_system"] = "none"
            elif getattr(policy, "coordinate_system", None):
                trace["coordinate_system"] = policy.coordinate_system
            elif policy.site == "attention_heads":
                trace["coordinate_system"] = "physical attention-head coordinates"
            else:
                trace["coordinate_system"] = "physical post-block hidden-state coordinates"
            trace_path = trace_directory / (prompt_key + ".pt")
            temporary = trace_path.with_suffix(".pt.tmp")
            torch.save(trace, temporary)
            temporary.replace(trace_path)
            output_rows[-1].update({
                "trace_file": trace_path.name, "trace_sha256": sha256(trace_path),
                "control_energy": trace["control_energy"],
                "hidden_delta_energy": trace["hidden_delta_energy_total"],
            })
            write_json(checkpoint, output_rows[-1])
    return output_rows
