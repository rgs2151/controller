"""Long-context generation and independent scoring for L-CiteEval."""

from __future__ import annotations

import argparse
import json
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
from robust_steerability.benchmarks.lciteeval_artifacts import (
    BENCHMARK,
    CONTEXT_WINDOW,
    load_model,
    model_load_spec,
)
from robust_steerability.benchmarks.specs import MODELS
from robust_steerability.calibration.nominal_artifact import load_shared_nominal_dynamics
from robust_steerability.calibration.nominal_artifact import nominal_dynamics_signature
from robust_steerability.datasets.lciteeval import AXBENCH_CONCEPT
from robust_steerability.experiments.methods import ControllerArtifact, build_policy
from robust_steerability.judges.exact import harmonic_mean, lcite_answer_overlap
from robust_steerability.judges.lciteeval import (
    lcite_citation_scores,
    load_lcite_entailer,
    pipeline_entailment,
)
from robust_steerability.judges.specs import scorer_cache_path
from robust_steerability.modeling.huggingface import cuda_device_index
from robust_steerability.modeling.interventions import register_generation_policy_hooks
from robust_steerability.source_methods.control import (
    SetpointCalibration,
    build_alqr_policy,
    build_spid_policy,
)
from robust_steerability.source_methods.id_benchmark import runtime_provenance


METHODS = ("original", "spid", "alqr", "h_infinity")
CONDITIONS = ("8k", "16k", "32k")
DEFAULT_BATCH_SIZE = {
    "qwen25_3b_instruct": {"8k": 4, "16k": 2, "32k": 1},
    "llama31_8b_instruct": {"8k": 2, "16k": 1, "32k": 1},
    "llama32_1b_instruct": {"8k": 8, "16k": 4, "32k": 2},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def data_path(model_key: str) -> Path:
    return dataset_root(BENCHMARK, model_key) / "lciteeval.json"


def cache_root(
    model_key: str, use_cache: bool, calibration_id: str = "selected"
) -> Path:
    """Route named recalibrations away from the historical selected run."""

    root = evaluation_root(BENCHMARK, model_key, use_cache=use_cache)
    if calibration_id == "selected":
        return root
    if not calibration_id or "/" in calibration_id or "\\" in calibration_id:
        raise ValueError("calibration_id must be a simple name")
    return root / "calibrations" / calibration_id


def compact_results_root(use_cache: bool, calibration_id: str = "selected") -> Path:
    """Mirror the evaluation namespace for compact, Git-tracked summaries."""

    root = results_root(BENCHMARK, use_cache=use_cache)
    if calibration_id == "selected":
        return root
    if not calibration_id or "/" in calibration_id or "\\" in calibration_id:
        raise ValueError("calibration_id must be a simple name")
    return root / "calibrations" / calibration_id


def generation_path(
    model_key: str,
    condition: str,
    method: str,
    *,
    use_cache: bool,
    calibration_id: str = "selected",
) -> Path:
    return (
        cache_root(model_key, use_cache, calibration_id)
        / "generations"
        / f"hotpotqa_{condition}"
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
        raise FileNotFoundError(f"Run L-CiteEval calibration before {method}: {path}")
    payload = json.loads(path.read_text())
    if payload.get("model") != [MODELS[model_key].model_id, MODELS[model_key].revision]:
        raise ValueError(f"L-CiteEval selection model mismatch: {path}")
    return dict(payload["selected"]["parameters"] if method == "h_infinity" else payload["parameters"])


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
            behavior=AXBENCH_CONCEPT,
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
        raise ValueError("H-infinity does not use the current shared AXBench direction")
    artifact = replace(
        base,
        hinf_gains=selected["gains"],
        hinf_feasible=bool(selected["feasible"]),
        gamma_star=float(selected["gamma_star"]),
        hinf_diagnostics=selected["diagnostics"],
    )
    return build_policy("hinf", artifact, kp=0.0, ki=0.0, kd=0.0)


def format_short_instruction(tokenizer, instruction: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _stop_token_ids(model, tokenizer) -> list[int]:
    eos = model.config.eos_token_id
    values = list(eos) if isinstance(eos, list) else [int(eos)]
    values.extend(tokenizer.encode("\n", add_special_tokens=False))
    return sorted(set(int(value) for value in values))


def generate_completions(
    model,
    tokenizer,
    prompts: list[str],
    *,
    policy,
    use_cache: bool,
    batch_size: int,
    max_new_tokens: int = 200,
) -> tuple[list[str], list[int]]:
    completions = []
    generated_counts = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(next(model.parameters()).device)
        input_width = int(encoded["input_ids"].shape[1])
        handles = (
            register_generation_policy_hooks(model, policy)
            if policy is not None
            else []
        )
        try:
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=use_cache,
                    eos_token_id=_stop_token_ids(model, tokenizer),
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
    condition: str,
    method: str,
    shard_index: int,
    shard_count: int,
    *,
    use_cache: bool,
    calibration_id: str = "selected",
) -> Path:
    return (
        cache_root(model_key, use_cache, calibration_id)
        / "generation_shards"
        / f"hotpotqa_{condition}"
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
) -> None:
    data = json.loads(data_path(model_key).read_text())
    records = data["evaluation"][condition][shard_index::shard_count]
    destination = _shard_path(
        model_key,
        condition,
        method,
        shard_index,
        shard_count,
        use_cache=use_cache,
        calibration_id=calibration_id,
    )
    batch_size = generation_batch_size or DEFAULT_BATCH_SIZE[model_key][condition]
    identity = {
        "schema_version": 1,
        "model": [MODELS[model_key].model_id, MODELS[model_key].revision],
        "condition": condition,
        "method": method,
        "parameters": _selection_parameters(model_key, method, calibration_id),
        "calibration_id": calibration_id,
        "evaluated_model_kv_cache": use_cache,
        "generation_batch_size": batch_size,
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
    expected_prefix = [str(row["prompt_id"]) for row in records[:completed]]
    actual_prefix = [str(row["prompt_id"]) for row in payload["rows"]]
    if actual_prefix != expected_prefix:
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
        policy = _policy(model_key, method, calibration_id, device)
        for start in range(completed, len(records), batch_size):
            batch = records[start : start + batch_size]
            completions, generated_counts = generate_completions(
                model,
                tokenizer,
                [str(row["model_input"]) for row in batch],
                policy=policy,
                use_cache=use_cache,
                batch_size=len(batch),
            )
            payload["rows"].extend(
                {
                    "prompt_id": row["prompt_id"],
                    "condition": condition,
                    "matched_question_index": row["matched_question_index"],
                    "input_tokens": row["input_tokens"],
                    "question": row["question"],
                    "text": f"{row['instruction']}\n\nQuestion: {row['question']}",
                    "concept": AXBENCH_CONCEPT,
                    "completion": completion,
                    "generated_tokens": generated,
                }
                for row, completion, generated in zip(
                    batch, completions, generated_counts, strict=True
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
    calibration_id: str = "selected",
) -> Path:
    paths = [
        _shard_path(
            model_key,
            condition,
            method,
            index,
            shard_count,
            use_cache=use_cache,
            calibration_id=calibration_id,
        )
        for index in range(shard_count)
    ]
    shards = [json.loads(path.read_text()) for path in paths]
    if any(shard.get("status") != "complete" for shard in shards):
        raise ValueError(f"Incomplete L-CiteEval shards for {condition}/{method}")
    rows = [row for shard in shards for row in shard["rows"]]
    rows.sort(key=lambda row: int(row["matched_question_index"]))
    if [int(row["matched_question_index"]) for row in rows] != list(range(40)):
        raise ValueError(f"L-CiteEval merge changed matched order for {condition}/{method}")
    common = dict(shards[0]["identity"])
    common.pop("shard")
    destination = generation_path(
        model_key,
        condition,
        method,
        use_cache=use_cache,
        calibration_id=calibration_id,
    )
    _write_json(
        destination,
        {
            "identity": common,
            "status": "complete",
            "attempts": [attempt for shard in shards for attempt in shard["attempts"]],
            "repetitions": [{"repetition": 0, "sample_count": 40, "rows": rows}],
        },
    )
    return destination


def _dataset_map(model_key: str) -> dict[str, dict]:
    data = json.loads(data_path(model_key).read_text())
    return {
        row["prompt_id"]: row
        for condition in CONDITIONS
        for row in data["evaluation"][condition]
    }


def score_answer_overlap(
    model_key: str,
    generation: Path,
    *,
    use_cache: bool,
    calibration_id: str = "selected",
) -> Path:
    destination = scorer_cache_path(
        cache_root(model_key, use_cache, calibration_id),
        generation,
        "lcite_answer_overlap",
    )
    if destination.exists() and json.loads(destination.read_text()).get("status") == "complete":
        return destination
    source = _dataset_map(model_key)
    payload = json.loads(generation.read_text())
    rows = []
    for repetition in payload["repetitions"]:
        for row in repetition["rows"]:
            rows.append(
                {
                    "prompt_id": row["prompt_id"],
                    **lcite_answer_overlap(row["completion"], source[row["prompt_id"]]["answer"]),
                }
            )
    _write_json(destination, {"status": "complete", "rows": rows})
    return destination


def score_citations(
    model_key: str,
    generation: Path,
    device: str,
    *,
    use_cache: bool,
    calibration_id: str = "selected",
) -> Path:
    destination = scorer_cache_path(
        cache_root(model_key, use_cache, calibration_id),
        generation,
        "lcite_citation_nli",
    )
    if destination.exists() and json.loads(destination.read_text()).get("status") == "complete":
        return destination
    source = _dataset_map(model_key)
    pipeline = load_lcite_entailer(cuda_device_index(device))
    entails = pipeline_entailment(pipeline)
    payload = json.loads(generation.read_text())
    source_rows = [
        row for repetition in payload["repetitions"] for row in repetition["rows"]
    ]
    saved = (
        json.loads(destination.read_text())
        if destination.exists()
        else {"status": "partial", "rows": []}
    )
    rows = list(saved["rows"])
    expected_prefix = [str(row["prompt_id"]) for row in source_rows[: len(rows)]]
    actual_prefix = [str(row["prompt_id"]) for row in rows]
    if actual_prefix != expected_prefix:
        raise ValueError(f"Citation score cache is not a valid prompt prefix: {destination}")
    for row in source_rows[len(rows) :]:
        rows.append(
            {
                "prompt_id": row["prompt_id"],
                **lcite_citation_scores(
                    row["completion"], source[row["prompt_id"]]["docs"], entails
                ),
            }
        )
        _write_json(destination, {"status": "partial", "rows": rows})
    _write_json(destination, {"status": "complete", "rows": rows})
    return destination


def score_axbench_overall(
    model_key: str,
    generation: Path,
    *,
    use_cache: bool,
    calibration_id: str = "selected",
) -> Path:
    root = cache_root(model_key, use_cache, calibration_id)
    components = (
        "axbench_concept_relevance",
        "axbench_instruction_relevance",
        "axbench_fluency",
    )
    values = []
    for key in components:
        path = scorer_cache_path(root, generation, key)
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete":
            raise ValueError(f"Incomplete AXBench component score: {path}")
        values.append({row["prompt_id"]: float(row["score"]) for row in payload["rows"]})
    prompt_ids = list(values[0])
    rows = [
        {
            "prompt_id": prompt_id,
            "score": harmonic_mean([mapping[prompt_id] for mapping in values]),
        }
        for prompt_id in prompt_ids
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
    calibration_id: str = "selected",
) -> Path:
    root = cache_root(model_key, use_cache, calibration_id)
    generation = generation_path(
        model_key,
        condition,
        method,
        use_cache=use_cache,
        calibration_id=calibration_id,
    )
    generation_payload = json.loads(generation.read_text())
    metrics = {}
    for key in scorer_keys:
        path = scorer_cache_path(root, generation, key)
        payload = json.loads(path.read_text())
        numeric = {
            field
            for row in payload["rows"]
            for field, value in row.items()
            if field != "prompt_id" and isinstance(value, (int, float))
        }
        for field in numeric:
            metrics[f"{key}.{field}"] = float(
                np.mean([float(row[field]) for row in payload["rows"]])
            )
    destination = root / "summaries" / f"hotpotqa_{condition}" / f"{method}.json"
    summary = {
        "identity": generation_payload["identity"],
        "model": [MODELS[model_key].model_id, MODELS[model_key].revision],
        "condition": condition,
        "method": method,
        "sample_count": 40,
        "created_at_utc": _utc_now(),
        "metrics": metrics,
    }
    if destination.exists():
        existing = json.loads(destination.read_text())
        summary["metrics"] = {**existing.get("metrics", {}), **metrics}
    _write_json(destination, summary)
    _write_json(
        compact_results_root(use_cache, calibration_id)
        / model_key
        / f"hotpotqa_{condition}"
        / f"{method}.json",
        summary,
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("generate-shard", "score-citations"), required=True)
    parser.add_argument("--model", choices=tuple(DEFAULT_BATCH_SIZE), required=True)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--device", required=True)
    parser.add_argument("--calibration-id", default="selected")
    parser.add_argument("--kv-cache", choices=("off", "on"), default="off")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument("--generation-path")
    arguments = parser.parse_args()
    use_cache = arguments.kv_cache == "on"
    if arguments.stage == "generate-shard":
        generate_shard(
            arguments.model,
            arguments.condition,
            arguments.method,
            arguments.device,
            arguments.calibration_id,
            use_cache,
            arguments.shard_index,
            arguments.shard_count,
            arguments.generation_batch_size,
        )
    else:
        score_citations(
            arguments.model,
            Path(arguments.generation_path),
            arguments.device,
            use_cache=use_cache,
            calibration_id=arguments.calibration_id,
        )


if __name__ == "__main__":
    main()
