"""Multilingual MGSM generation and independent deterministic scoring."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from robust_steerability.benchmarks.layout import (
    artifact_root,
    calibration_root,
    dataset_root,
    evaluation_root,
    results_root,
)
from robust_steerability.benchmarks.mgsm_artifacts import (
    BENCHMARK,
    CONCEPT,
    load_model,
    model_load_spec,
)
from robust_steerability.benchmarks.specs import MODELS
from robust_steerability.calibration.nominal_artifact import load_shared_nominal_dynamics
from robust_steerability.datasets.mgsm import LANGUAGES, LANGUAGE_NAMES
from robust_steerability.experiments.methods import ControllerArtifact, build_policy
from robust_steerability.judges.exact import harmonic_mean
from robust_steerability.judges.mgsm import exact_match, spanish_rule_score
from robust_steerability.judges.specs import scorer_cache_path
from robust_steerability.modeling.interventions import register_generation_policy_hooks
from robust_steerability.source_methods.control import (
    SetpointCalibration,
    build_alqr_policy,
    build_spid_policy,
)
from robust_steerability.source_methods.id_benchmark import runtime_provenance


METHODS = ("original", "spid", "alqr", "h_infinity")
DEFAULT_BATCH_SIZE = {"qwen3_4b": 32, "qwen3_8b": 16}
EVALUATION_SAMPLE_SEED = 42
MAX_NEW_TOKENS = 256


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def data_path(model_key: str) -> Path:
    return dataset_root(BENCHMARK, model_key) / "mgsm.json"


def cache_root(model_key: str, use_cache: bool) -> Path:
    return evaluation_root(BENCHMARK, model_key, use_cache=use_cache)


def generation_path(
    model_key: str, language: str, method: str, *, use_cache: bool
) -> Path:
    return (
        cache_root(model_key, use_cache)
        / "generations"
        / f"mgsm_{language}"
        / method
        / "final.json"
    )


def generation_complete(path: Path) -> bool:
    return path.exists() and json.loads(path.read_text()).get("status") == "complete"


def _selection_parameters(model_key: str, method: str, calibration_id: str) -> dict:
    if method == "original":
        return {}
    path = calibration_root(BENCHMARK, model_key, method, calibration_id) / "selection.json"
    if not path.exists():
        raise FileNotFoundError(f"Run MGSM calibration before {method}: {path}")
    payload = json.loads(path.read_text())
    if payload.get("model") != [MODELS[model_key].model_id, MODELS[model_key].revision]:
        raise ValueError(f"MGSM selection model mismatch: {path}")
    return dict(
        payload["selected"]["parameters"]
        if method == "h_infinity"
        else payload["parameters"]
    )


def _policy(model_key: str, method: str, calibration_id: str, device: str):
    if method == "original":
        return None
    setpoint_payload = torch.load(
        artifact_root(BENCHMARK, model_key) / "setpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    setpoint = SetpointCalibration(
        setpoint_payload["contrast"], setpoint_payload["feature_norm"]
    )
    parameters = _selection_parameters(model_key, method, calibration_id)
    if method == "spid":
        return build_spid_policy(
            setpoint,
            multiplier=float(parameters["lambda"]),
            kp=float(parameters["kp"]),
            ki=float(parameters["ki"]),
            kd=float(parameters["kd"]),
        )
    if method == "alqr":
        dynamics = load_shared_nominal_dynamics(
            artifact_root(BENCHMARK, model_key) / "dynamics.pt",
            behavior=CONCEPT,
            model_id=MODELS[model_key].model_id,
            model_revision=MODELS[model_key].revision,
        )
        return build_alqr_policy(
            dynamics,
            setpoint,
            multiplier=float(parameters["lambda"]),
            q=float(parameters["q"]),
            r=float(parameters["r"]),
            q_final=float(parameters["q_final"]),
            device=device,
        )
    root = calibration_root(BENCHMARK, model_key, method, calibration_id)
    base_payload = torch.load(
        root / "base/controller.pt", map_location="cpu", weights_only=True, mmap=True
    )
    selected = torch.load(
        root / "controller.pt", map_location="cpu", weights_only=True, mmap=True
    )
    base = ControllerArtifact(**base_payload["artifact"])
    artifact = replace(
        base,
        hinf_gains=selected["gains"],
        hinf_feasible=bool(selected["feasible"]),
        gamma_star=float(selected["gamma_star"]),
        hinf_diagnostics=selected["diagnostics"],
    )
    return build_policy("hinf", artifact, kp=0.0, ki=0.0, kd=0.0)


def format_calibration_prompt(tokenizer, text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _eos_token_ids(model, tokenizer) -> list[int]:
    values = model.config.eos_token_id
    eos = list(values) if isinstance(values, list) else [int(values)]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        eos.append(im_end)
    return sorted(set(eos))


def generate_completions(
    model,
    tokenizer,
    prompts: list[str],
    *,
    policy,
    use_cache: bool,
    batch_size: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> tuple[list[str], list[int]]:
    completions = []
    generated_counts = []
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(
            prompts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(next(model.parameters()).device)
        input_width = int(encoded["input_ids"].shape[1])
        handles = register_generation_policy_hooks(model, policy) if policy is not None else []
        try:
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=use_cache,
                    eos_token_id=_eos_token_ids(model, tokenizer),
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        continuation = output.sequences[:, input_width:]
        for tokens in continuation:
            nonpad = tokens[tokens != tokenizer.pad_token_id]
            completions.append(tokenizer.decode(nonpad, skip_special_tokens=True).strip())
            generated_counts.append(int(nonpad.numel()))
    return completions, generated_counts


def _shard_path(
    model_key: str,
    language: str,
    method: str,
    shard_index: int,
    shard_count: int,
    *,
    use_cache: bool,
) -> Path:
    return (
        cache_root(model_key, use_cache)
        / "generation_shards"
        / f"mgsm_{language}"
        / method
        / f"shard_{shard_index:02d}_of_{shard_count:02d}.json"
    )


def generate_shard(
    model_key: str,
    language: str,
    sample_count: int,
    method: str,
    device: str,
    calibration_id: str,
    use_cache: bool,
    shard_index: int,
    shard_count: int,
    generation_batch_size: int | None,
) -> None:
    available = json.loads(data_path(model_key).read_text())["evaluation"][language]
    if sample_count < 1 or sample_count > len(available):
        raise ValueError(
            f"sample_count must be between 1 and {len(available)} for {language}"
        )
    selected_indices = sorted(
        random.Random(EVALUATION_SAMPLE_SEED).sample(range(len(available)), sample_count)
    )
    selected = [available[index] for index in selected_indices]
    records = selected[shard_index::shard_count]
    destination = _shard_path(
        model_key, language, method, shard_index, shard_count, use_cache=use_cache
    )
    if generation_complete(destination):
        return
    batch_size = generation_batch_size or DEFAULT_BATCH_SIZE[model_key]
    identity = {
        "schema_version": 1,
        "model": [MODELS[model_key].model_id, MODELS[model_key].revision],
        "input_language": language,
        "evaluation_sample_count": sample_count,
        "evaluation_sample_seed": EVALUATION_SAMPLE_SEED,
        "evaluation_problem_indices": selected_indices,
        "method": method,
        "parameters": _selection_parameters(model_key, method, calibration_id),
        "calibration_id": calibration_id,
        "evaluated_model_kv_cache": use_cache,
        "generation_batch_size": batch_size,
        "max_new_tokens": MAX_NEW_TOKENS,
        "model_loading": asdict(model_load_spec(model_key)),
        "shard": {"index": shard_index, "count": shard_count},
    }
    payload = {"identity": identity, "status": "partial", "attempts": [], "rows": []}
    started = time.perf_counter()
    attempt = {
        "started_at_utc": _utc_now(),
        "status": "running",
        "runtime": runtime_provenance(device),
    }
    payload["attempts"].append(attempt)
    _write_json(destination, payload)
    model, tokenizer = load_model(model_key, device)
    torch.cuda.reset_peak_memory_stats(device)
    policy = _policy(model_key, method, calibration_id, device)
    completions, generated_counts = generate_completions(
        model,
        tokenizer,
        [str(row["model_input"]) for row in records],
        policy=policy,
        use_cache=use_cache,
        batch_size=batch_size,
    )
    payload["rows"] = [
        {
            "prompt_id": row["prompt_id"],
            "problem_index": row["problem_index"],
            "input_language": language,
            "input_language_name": LANGUAGE_NAMES[language],
            "input_tokens": row["input_tokens"],
            "question": row["question"],
            "text": (
                "Solve the current grade-school math problem with reasoning and give "
                "the final Arabic-numeral answer.\n\n" + row["question"]
            ),
            "answer_number": row["answer_number"],
            "completion": completion,
            "generated_tokens": generated,
        }
        for row, completion, generated in zip(
            records, completions, generated_counts, strict=True
        )
    ]
    attempt.update(
        {
            "status": "complete",
            "finished_at_utc": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(
                device
            ),
            "gpu_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(
                device
            ),
        }
    )
    payload["status"] = "complete"
    _write_json(destination, payload)


def merge_shards(
    model_key: str,
    language: str,
    method: str,
    shard_count: int,
    *,
    use_cache: bool,
) -> Path:
    paths = [
        _shard_path(model_key, language, method, index, shard_count, use_cache=use_cache)
        for index in range(shard_count)
    ]
    shards = [json.loads(path.read_text()) for path in paths]
    if any(shard.get("status") != "complete" for shard in shards):
        raise ValueError(f"Incomplete MGSM shards for {language}/{method}")
    rows = [row for shard in shards for row in shard["rows"]]
    rows.sort(key=lambda row: int(row["problem_index"]))
    expected = [
        int(value)
        for value in shards[0]["identity"]["evaluation_problem_indices"]
    ]
    if [int(row["problem_index"]) for row in rows] != expected:
        raise ValueError(f"MGSM merge changed matched problem order for {language}/{method}")
    common = dict(shards[0]["identity"])
    common.pop("shard")
    destination = generation_path(model_key, language, method, use_cache=use_cache)
    _write_json(
        destination,
        {
            "identity": common,
            "status": "complete",
            "attempts": [attempt for shard in shards for attempt in shard["attempts"]],
            "repetitions": [
                {"repetition": 0, "sample_count": len(expected), "rows": rows}
            ],
        },
    )
    return destination


def score_deterministic(
    model_key: str, generation: Path, scorer_key: str, *, use_cache: bool
) -> Path:
    destination = scorer_cache_path(cache_root(model_key, use_cache), generation, scorer_key)
    if destination.exists() and json.loads(destination.read_text()).get("status") == "complete":
        return destination
    payload = json.loads(generation.read_text())
    rows = []
    for repetition in payload["repetitions"]:
        for row in repetition["rows"]:
            result = (
                exact_match(str(row["completion"]), row["answer_number"])
                if scorer_key == "mgsm_exact_match"
                else spanish_rule_score(str(row["completion"]))
            )
            rows.append({"prompt_id": row["prompt_id"], **result})
    _write_json(destination, {"status": "complete", "rows": rows})
    return destination


def score_overall(model_key: str, generation: Path, *, use_cache: bool) -> Path:
    root = cache_root(model_key, use_cache)
    components = (
        "axbench_rule_spanish",
        "axbench_instruction_relevance",
        "axbench_fluency",
    )
    values = []
    for key in components:
        payload = json.loads(scorer_cache_path(root, generation, key).read_text())
        if payload.get("status") != "complete":
            raise ValueError(f"Incomplete MGSM steering component: {key}")
        values.append({row["prompt_id"]: float(row["score"]) for row in payload["rows"]})
    prompt_ids = list(values[0])
    rows = [
        {
            "prompt_id": prompt_id,
            "score": harmonic_mean([mapping[prompt_id] for mapping in values]),
        }
        for prompt_id in prompt_ids
    ]
    destination = scorer_cache_path(root, generation, "mgsm_axbench_overall")
    _write_json(destination, {"status": "complete", "rows": rows})
    return destination


def summarize(
    model_key: str,
    language: str,
    method: str,
    scorer_keys: tuple[str, ...],
    *,
    use_cache: bool,
) -> Path:
    root = cache_root(model_key, use_cache)
    generation = generation_path(model_key, language, method, use_cache=use_cache)
    metrics = {}
    for key in scorer_keys:
        payload = json.loads(scorer_cache_path(root, generation, key).read_text())
        metrics[f"{key}.score"] = float(
            np.mean([float(row["score"]) for row in payload["rows"]])
        )
        if key == "mgsm_exact_match":
            metrics[f"{key}.valid_rate"] = float(
                np.mean([float(row["valid"]) for row in payload["rows"]])
            )
    summary = {
        "model": [MODELS[model_key].model_id, MODELS[model_key].revision],
        "input_language": language,
        "method": method,
        "sample_count": 250,
        "metrics": metrics,
    }
    destination = root / "summaries" / f"mgsm_{language}" / f"{method}.json"
    _write_json(destination, summary)
    _write_json(
        results_root(BENCHMARK, use_cache=use_cache)
        / model_key
        / f"mgsm_{language}"
        / f"{method}.json",
        summary,
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("generate-shard",), required=True)
    parser.add_argument("--model", choices=tuple(DEFAULT_BATCH_SIZE), required=True)
    parser.add_argument("--language", choices=LANGUAGES, required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--calibration-id", default="selected")
    parser.add_argument("--kv-cache", choices=("off", "on"), default="off")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--generation-batch-size", type=int)
    arguments = parser.parse_args()
    generate_shard(
        arguments.model,
        arguments.language,
        arguments.sample_count,
        arguments.method,
        arguments.device,
        arguments.calibration_id,
        arguments.kv_cache == "on",
        arguments.shard_index,
        arguments.shard_count,
        arguments.generation_batch_size,
    )


if __name__ == "__main__":
    main()
