"""HarmBench generation and independent result aggregation."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from robust_steerability.benchmarks.harmful_artifacts import (
    BENCHMARK,
    CONCEPT,
    MODEL_KEYS,
    load_model,
    model_load_spec,
)
from robust_steerability.benchmarks.layout import (
    artifact_root,
    calibration_root,
    dataset_root,
    evaluation_root,
    results_root,
)
from robust_steerability.benchmarks.specs import MODELS
from robust_steerability.calibration.nominal_artifact import (
    load_shared_nominal_dynamics,
    nominal_dynamics_signature,
)
from robust_steerability.datasets.harmful import SAFE_CONCEPT
from robust_steerability.experiments.methods import ControllerArtifact, build_policy
from robust_steerability.judges.exact import harmonic_mean
from robust_steerability.judges.specs import scorer_cache_path, scorer_spec
from robust_steerability.modeling.huggingface import cuda_device_index
from robust_steerability.modeling.interventions import register_generation_policy_hooks
from robust_steerability.source_methods.control import SetpointCalibration, build_alqr_policy
from robust_steerability.source_methods.id_benchmark import runtime_provenance


METHODS = ("original", "alqr", "h_infinity")
CONDITIONS = ("direct", "human_jailbreak")
DEFAULT_BATCH_SIZE = {
    "llama32_1b_instruct": {"direct": 16, "human_jailbreak": 8},
    "llama32_3b_instruct": {"direct": 16, "human_jailbreak": 8},
    "llama31_8b_instruct": {"direct": 8, "human_jailbreak": 4},
}
MAX_NEW_TOKENS = 512


def evaluation_records(source: dict, condition: str, behavior_count: int | None) -> list[dict]:
    """Select the same deterministic behavior identities across all conditions."""

    direct = source["evaluation"]["direct"]
    count = len(direct) if behavior_count is None else behavior_count
    if not 1 <= count <= len(direct):
        raise ValueError(f"Requested {count} behaviors from {len(direct)} available")
    behavior_ids = {row["behavior_id"] for row in direct[:count]}
    records = [
        row for row in source["evaluation"][condition]
        if row["behavior_id"] in behavior_ids
    ]
    expected_multiplier = len(source["evaluation"][condition]) // len(direct)
    expected = count * expected_multiplier
    if len(records) != expected:
        raise ValueError(
            f"Expected {expected} matched {condition} rows, found {len(records)}"
        )
    return records


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def data_path(model_key: str) -> Path:
    return dataset_root(BENCHMARK, model_key) / "harmbench.json"


def cache_root(model_key: str, use_cache: bool) -> Path:
    return evaluation_root(BENCHMARK, model_key, use_cache=use_cache)


def generation_path(
    model_key: str, condition: str, method: str, *, use_cache: bool
) -> Path:
    return (
        cache_root(model_key, use_cache)
        / "generations"
        / f"harmbench_{condition}"
        / method
        / "final.json"
    )


def generation_complete(path: Path) -> bool:
    return path.exists() and json.loads(path.read_text()).get("status") == "complete"


def selection_parameters(
    model_key: str, method: str, calibration_id: str
) -> dict:
    if method == "original":
        return {}
    path = calibration_root(BENCHMARK, model_key, method, calibration_id) / "selection.json"
    if not path.exists():
        raise FileNotFoundError(f"Run HarmBench calibration before {method}: {path}")
    payload = json.loads(path.read_text())
    if payload.get("model") != [MODELS[model_key].model_id, MODELS[model_key].revision]:
        raise ValueError(f"HarmBench selection model mismatch: {path}")
    return dict(
        payload["selected"]["parameters"]
        if method == "h_infinity"
        else payload["parameters"]
    )


def policy(model_key: str, method: str, calibration_id: str, device: str):
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
    parameters = selection_parameters(model_key, method, calibration_id)
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
    selection = json.loads((root / "selection.json").read_text())
    diagnostic_bundle = root / selection["diagnostic_bundle"]
    if not diagnostic_bundle.exists():
        raise FileNotFoundError(f"Missing Hannah diagnostic bundle: {diagnostic_bundle}")
    base_payload = torch.load(
        root / "base/controller.pt", map_location="cpu", weights_only=True, mmap=True
    )
    if base_payload["metadata"]["nominal_dynamics"] != nominal_dynamics_signature(
        artifact_root(BENCHMARK, model_key) / "dynamics.pt"
    ):
        raise ValueError("H-infinity was not synthesized from the current shared A matrix")
    selected = torch.load(
        root / "controller.pt", map_location="cpu", weights_only=True, mmap=True
    )
    base = ControllerArtifact(**base_payload["artifact"])
    expected_feature = setpoint_payload["contrast"][:-1] / setpoint_payload[
        "feature_norm"
    ][:-1].clamp_min(1e-4).unsqueeze(1)
    if not torch.allclose(base.raw_feature_unit, expected_feature, atol=1e-6, rtol=1e-5):
        raise ValueError("H-infinity does not use the shared HarmBench direction")
    artifact = replace(
        base,
        hinf_gains=selected["gains"],
        hinf_feasible=bool(selected["feasible"]),
        gamma_star=float(selected["gamma_star"]),
        hinf_diagnostics=selected["diagnostics"],
    )
    return build_policy("hinf", artifact, kp=0.0, ki=0.0, kd=0.0)


def generate_completions(
    model,
    tokenizer,
    prompts: list[str],
    *,
    steering_policy,
    use_cache: bool,
    batch_size: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> tuple[list[str], list[int]]:
    completions = []
    generated_counts = []
    eos = model.config.eos_token_id
    eos_ids = list(eos) if isinstance(eos, list) else [int(eos)]
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=False
        ).to(next(model.parameters()).device)
        input_width = int(encoded["input_ids"].shape[1])
        handles = (
            register_generation_policy_hooks(model, steering_policy)
            if steering_policy is not None
            else []
        )
        try:
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=use_cache,
                    eos_token_id=eos_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        continuation = output.sequences[:, input_width:]
        for tokens in continuation:
            retained = tokens[tokens != tokenizer.pad_token_id]
            completions.append(tokenizer.decode(retained, skip_special_tokens=True).strip())
            generated_counts.append(int(retained.numel()))
    return completions, generated_counts


def _shard_path(
    model_key: str,
    condition: str,
    method: str,
    shard_index: int,
    shard_count: int,
    *,
    use_cache: bool,
) -> Path:
    return (
        cache_root(model_key, use_cache)
        / "generation_shards"
        / f"harmbench_{condition}"
        / method
        / f"shard_{shard_index:02d}_of_{shard_count:02d}.json"
    )


def generate_shard(
    model_key: str,
    condition: str,
    method: str,
    device: str,
    calibration_id: str,
    use_cache: bool,
    shard_index: int,
    shard_count: int,
    generation_batch_size: int | None,
    behavior_count: int | None,
    max_new_tokens: int,
) -> None:
    source = json.loads(data_path(model_key).read_text())
    all_records = evaluation_records(source, condition, behavior_count)
    records = all_records[shard_index::shard_count]
    destination = _shard_path(
        model_key,
        condition,
        method,
        shard_index,
        shard_count,
        use_cache=use_cache,
    )
    batch_size = generation_batch_size or DEFAULT_BATCH_SIZE[model_key][condition]
    identity = {
        "schema_version": 1,
        "model": [MODELS[model_key].model_id, MODELS[model_key].revision],
        "condition": condition,
        "method": method,
        "parameters": selection_parameters(model_key, method, calibration_id),
        "calibration_id": calibration_id,
        "evaluated_model_kv_cache": use_cache,
        "generation_batch_size": batch_size,
        "evaluation_behavior_count": behavior_count or len(source["evaluation"]["direct"]),
        "max_new_tokens": max_new_tokens,
        "model_loading": asdict(model_load_spec(model_key)),
        "shard": {"index": shard_index, "count": shard_count},
    }
    if generation_complete(destination):
        return
    payload = (
        json.loads(destination.read_text())
        if destination.exists()
        else {"identity": identity, "status": "partial", "attempts": [], "rows": []}
    )
    if payload["identity"] != identity:
        raise ValueError(f"Existing shard has a different immutable identity: {destination}")
    completed = len(payload["rows"])
    if [row["prompt_id"] for row in payload["rows"]] != [
        row["prompt_id"] for row in records[:completed]
    ]:
        raise ValueError(f"Existing shard is not a valid prompt prefix: {destination}")
    started = time.perf_counter()
    attempt = {
        "started_at_utc": _utc_now(),
        "status": "running",
        "runtime": runtime_provenance(device),
    }
    payload["attempts"].append(attempt)
    _write_json(destination, payload)
    torch.cuda.reset_peak_memory_stats(cuda_device_index(device))
    try:
        model, tokenizer = load_model(model_key, device)
        steering_policy = policy(model_key, method, calibration_id, device)
        for start in range(completed, len(records), batch_size):
            batch = records[start : start + batch_size]
            completions, counts = generate_completions(
                model,
                tokenizer,
                [str(row["model_input"]) for row in batch],
                steering_policy=steering_policy,
                use_cache=use_cache,
                batch_size=len(batch),
                max_new_tokens=max_new_tokens,
            )
            payload["rows"].extend(
                {
                    "prompt_id": row["prompt_id"],
                    "condition": condition,
                    "behavior_id": row["behavior_id"],
                    "behavior": row["behavior"],
                    "context": row["context"],
                    "functional_category": row["functional_category"],
                    "semantic_category": row["semantic_category"],
                    "text": row["text"],
                    "concept": SAFE_CONCEPT,
                    "completion": completion,
                    "generated_tokens": count,
                }
                for row, completion, count in zip(
                    batch, completions, counts, strict=True
                )
            )
            _write_json(destination, payload)
    except BaseException as error:
        attempt.update(
            {
                "status": "failed",
                "finished_at_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        _write_json(destination, payload)
        raise
    else:
        attempt.update(
            {
                "status": "complete",
                "finished_at_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "gpu_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(
                    cuda_device_index(device)
                ),
                "gpu_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(
                    cuda_device_index(device)
                ),
            }
        )
        payload["status"] = "complete"
        _write_json(destination, payload)


def merge_shards(
    model_key: str,
    condition: str,
    method: str,
    shard_count: int,
    *,
    use_cache: bool,
    behavior_count: int | None = None,
) -> Path:
    paths = [
        _shard_path(
            model_key, condition, method, index, shard_count, use_cache=use_cache
        )
        for index in range(shard_count)
    ]
    shards = [json.loads(path.read_text()) for path in paths]
    if any(shard.get("status") != "complete" for shard in shards):
        raise ValueError(f"Incomplete HarmBench shards for {condition}/{method}")
    payload = json.loads(data_path(model_key).read_text())
    source = evaluation_records(payload, condition, behavior_count)
    order = {row["prompt_id"]: index for index, row in enumerate(source)}
    rows = [row for shard in shards for row in shard["rows"]]
    rows.sort(key=lambda row: order[row["prompt_id"]])
    if [row["prompt_id"] for row in rows] != [row["prompt_id"] for row in source]:
        raise ValueError(f"HarmBench merge changed row order for {condition}/{method}")
    common = dict(shards[0]["identity"])
    common.pop("shard")
    destination = generation_path(model_key, condition, method, use_cache=use_cache)
    _write_json(
        destination,
        {
            "identity": common,
            "status": "complete",
            "attempts": [attempt for shard in shards for attempt in shard["attempts"]],
            "repetitions": [
                {"repetition": 0, "sample_count": len(rows), "rows": rows}
            ],
        },
    )
    return destination


def score_axbench_overall(
    model_key: str, generation: Path, *, use_cache: bool
) -> Path:
    root = cache_root(model_key, use_cache)
    components = (
        "axbench_concept_relevance",
        "axbench_instruction_relevance",
        "axbench_fluency",
    )
    maps = []
    for component in components:
        payload = json.loads(scorer_cache_path(root, generation, component).read_text())
        if payload.get("status") != "complete":
            raise ValueError(f"Incomplete AXBench score for {component}")
        maps.append({row["prompt_id"]: float(row["score"]) for row in payload["rows"]})
    rows = [
        {
            "prompt_id": prompt_id,
            "score": harmonic_mean([mapping[prompt_id] for mapping in maps]),
        }
        for prompt_id in maps[0]
    ]
    destination = scorer_cache_path(root, generation, "axbench_overall")
    _write_json(destination, {"status": "complete", "rows": rows})
    return destination


def summarize(
    model_key: str,
    condition: str,
    method: str,
    scorer_keys: tuple[str, ...],
    *,
    use_cache: bool,
) -> Path:
    root = cache_root(model_key, use_cache)
    generation = generation_path(model_key, condition, method, use_cache=use_cache)
    generation_payload = json.loads(generation.read_text())
    generation_rows = generation_payload["repetitions"][0]["rows"]
    row_by_id = {row["prompt_id"]: row for row in generation_rows}
    metrics = {}
    category_metrics = {"functional": {}, "semantic": {}}
    for key in scorer_keys:
        score_payload = json.loads(scorer_cache_path(root, generation, key).read_text())
        numeric = {
            field
            for field in scorer_spec(key).output_fields
            if any(
                isinstance(row.get(field), (int, float))
                and not isinstance(row.get(field), bool)
                for row in score_payload["rows"]
            )
        }
        for field in numeric:
            metrics[f"{key}.{field}"] = float(
                np.mean([float(row[field]) for row in score_payload["rows"]])
            )
        if key == "harmbench_test_success":
            for category_name, row_field in (
                ("functional", "functional_category"),
                ("semantic", "semantic_category"),
            ):
                groups: dict[str, list[float]] = {}
                for score_row in score_payload["rows"]:
                    category = str(row_by_id[score_row["prompt_id"]][row_field])
                    groups.setdefault(category, []).append(float(score_row["score"]))
                category_metrics[category_name] = {
                    category: float(np.mean(values))
                    for category, values in sorted(groups.items())
                }
    summary = {
        "identity": generation_payload["identity"],
        "model": [MODELS[model_key].model_id, MODELS[model_key].revision],
        "condition": condition,
        "method": method,
        "sample_count": len(generation_rows),
        "created_at_utc": _utc_now(),
        "metrics": metrics,
        "attack_success_rate_by_category": category_metrics,
    }
    destination = root / "summaries" / f"harmbench_{condition}" / f"{method}.json"
    _write_json(destination, summary)
    _write_json(
        results_root(BENCHMARK, use_cache=use_cache)
        / model_key
        / f"harmbench_{condition}"
        / f"{method}.json",
        summary,
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("generate-shard",), required=True)
    parser.add_argument("--model", choices=MODEL_KEYS, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--calibration-id", default="selected")
    parser.add_argument("--kv-cache", choices=("off", "on"), default="off")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument("--evaluation-behaviors", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    arguments = parser.parse_args()
    generate_shard(
        arguments.model,
        arguments.condition,
        arguments.method,
        arguments.device,
        arguments.calibration_id,
        arguments.kv_cache == "on",
        arguments.shard_index,
        arguments.shard_count,
        arguments.generation_batch_size,
        arguments.evaluation_behaviors,
        arguments.max_new_tokens,
    )


if __name__ == "__main__":
    main()
