"""Independent multiple-choice evaluation against a frozen base controller."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from robust_steerability.benchmarks.composition import EvaluationDataset
from robust_steerability.benchmarks.composition import load_composition
from robust_steerability.benchmarks.layout import (
    artifact_root,
    calibration_root,
    dataset_root,
    evaluation_root,
    results_root,
)
from robust_steerability.benchmarks.launcher import run_data_shards
from robust_steerability.benchmarks.methods import method_spec
from robust_steerability.benchmarks.specs import MODELS
from robust_steerability.datasets.registry import dataset_adapter
from robust_steerability.experiments.methods import build_policy
from robust_steerability.judges.exact import score_multiple_choice_generation
from robust_steerability.judges.specs import scorer_cache_path
from robust_steerability.modeling.huggingface import load_access_token
from robust_steerability.modeling.interventions import register_generation_policy_hooks
from robust_steerability.source_methods.generation import generate_batched
from robust_steerability.source_methods.id_benchmark import (
    _batch_assignments,
    _batch_size as source_generation_batch_size,
    _output_rows,
    generation_shard_path,
    merge_generation_shards,
    run_generation_job,
    runtime_provenance,
)
from robust_steerability.source_methods.modeling import load_source_model
from robust_steerability.source_methods.protocol import SOURCE_RANDOM_SEED


REPO = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def data_path(base_benchmark: str, model_key: str, dataset: EvaluationDataset) -> Path:
    return dataset_root(base_benchmark, model_key) / f"{dataset.cache_namespace}.json"


def generation_path(
    base_benchmark: str,
    model_key: str,
    dataset: EvaluationDataset,
    method: str,
    *,
    use_cache: bool,
) -> Path:
    return (
        evaluation_root(base_benchmark, model_key, use_cache=use_cache)
        / "generations"
        / dataset.cache_namespace
        / method
        / "final.json"
    )


def generation_complete(path: Path) -> bool:
    return path.exists() and json.loads(path.read_text()).get("status") == "complete"


def prepare(base_benchmark: str, model_key: str, dataset: EvaluationDataset) -> Path:
    adapter = dataset_adapter(dataset.dataset)
    if dataset.runtime != "multiple_choice" or adapter.task != "multiple_choice":
        raise ValueError(f"Dataset {dataset.dataset!r} is not multiple choice")
    destination = data_path(base_benchmark, model_key, dataset)
    if destination.exists():
        return destination
    repetitions = {}
    for repetition in range(dataset.repetitions):
        records = adapter.prepare(
            SOURCE_RANDOM_SEED + repetition * 100_000,
            dataset.samples,
        )
        repetitions[str(repetition)] = [
            {
                "prompt_id": row["prompt_id"],
                "text": row["prompt"],
                "answer_index": row["answer_index"],
                "subject": row["subject"],
            }
            for row in records
        ]
    _write_json(
        destination,
        {
            "schema_version": 1,
            "dataset": [adapter.dataset_id, adapter.revision],
            "role": dataset.role,
            "base_benchmark": base_benchmark,
            "protocol": {
                "shots": 5,
                "samples_per_repetition": dataset.samples,
                "repetitions": dataset.repetitions,
                "sampling": "seeded subject-uniform test-question sample",
                "few_shot_source": "same-subject development split",
            },
            "evaluation": {dataset.cache_namespace: repetitions},
        },
    )
    return destination


def _truthfulness_generation(
    model_key: str,
    dataset: EvaluationDataset,
    method: str,
    device: str,
    calibration_id: str,
    generation_batch_size: int | None,
    use_cache: bool,
    shard_index: int,
    shard_count: int,
) -> None:
    from robust_steerability.benchmarks import truthfulness_runtime

    truthfulness_runtime._configure_runtime(
        model_key,
        calibration_id,
        generation_batch_size,
        use_cache=use_cache,
    )
    spec = method_spec(method)
    if spec.runtime == "h_infinity":
        artifact, parameters = truthfulness_runtime._load_selected_hinf(model_key)
        _generate_with_policy(
            "truthfulness",
            model_key,
            dataset,
            method,
            device,
            calibration_id,
            generation_batch_size,
            use_cache,
            build_policy(str(spec.policy_key), artifact, kp=0.0, ki=0.0, kd=0.0),
            parameters,
            shard_index,
            shard_count,
        )
        return
    run_generation_job(
        cache_root=evaluation_root("truthfulness", model_key, use_cache=use_cache),
        behavior="truthfulness",
        model_id=MODELS[model_key].model_id,
        revision=MODELS[model_key].revision,
        method=method,
        device=device,
        token=load_access_token(REPO),
        alqr_artifact_root=artifact_root("truthfulness", model_key),
        method_calibration_root=calibration_root(
            "truthfulness", model_key, method, calibration_id
        ),
        calibration_data_path=dataset_root("truthfulness", model_key)
        / "truthfulness.json",
        data_path=data_path("truthfulness", model_key, dataset),
        evaluation_key=dataset.cache_namespace,
        cache_namespace=dataset.cache_namespace,
        generation_profile=dataset.generation_profile,
        use_cache=use_cache,
        generation_batch_size=generation_batch_size,
        shard_index=shard_index,
        shard_count=shard_count,
    )


def _toxicity_generation(
    model_key: str,
    dataset: EvaluationDataset,
    method: str,
    device: str,
    calibration_id: str,
    generation_batch_size: int | None,
    use_cache: bool,
    shard_index: int,
    shard_count: int,
) -> None:
    from robust_steerability.benchmarks import toxicity_runtime

    toxicity_runtime.configure_model(
        model_key,
        calibration_id,
        generation_batch_size,
        use_cache=use_cache,
    )
    policy = toxicity_runtime._selected_policy(method, device)
    parameters = toxicity_runtime._method_parameters(method)
    _generate_with_policy(
        "toxicity",
        model_key,
        dataset,
        method,
        device,
        calibration_id,
        generation_batch_size,
        use_cache,
        policy,
        parameters,
        shard_index,
        shard_count,
    )


def _harmful_generation(
    model_key: str,
    dataset: EvaluationDataset,
    method: str,
    device: str,
    calibration_id: str,
    generation_batch_size: int | None,
    use_cache: bool,
    shard_index: int,
    shard_count: int,
) -> None:
    from robust_steerability.benchmarks import harmful_artifacts, harmful_runtime

    model, tokenizer = harmful_artifacts.load_model(model_key, device)
    steering_policy = harmful_runtime.policy(
        model_key, method, calibration_id, device
    )

    def format_prompt(text: str) -> str:
        return tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": (
                        "Answer the final multiple-choice question with only "
                        "A, B, C, or D.\n\n" + text
                    ),
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate_suffixes(
        loaded_model,
        loaded_tokenizer,
        prompts: list[str],
        policy,
        cache_enabled: bool,
        batch_size: int,
    ) -> list[str]:
        del cache_enabled
        letters = "ABCD"
        candidate_ids = []
        for letter in letters:
            token_ids = loaded_tokenizer.encode(letter, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(f"MMLU answer {letter!r} is not one token")
            candidate_ids.append(token_ids[0])
        completions = []
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            encoded = loaded_tokenizer(
                batch, return_tensors="pt", padding=True, truncation=False
            ).to(next(loaded_model.parameters()).device)
            handles = (
                register_generation_policy_hooks(loaded_model, policy)
                if policy is not None
                else []
            )
            try:
                with torch.inference_mode():
                    output = loaded_model(
                        **encoded, use_cache=False, return_dict=True
                    )
            finally:
                for handle in handles:
                    handle.remove()
            selected = output.logits[:, -1, candidate_ids].argmax(dim=-1).tolist()
            completions.extend(letters[index] for index in selected)
        return completions

    _generate_with_policy(
        "harmful",
        model_key,
        dataset,
        method,
        device,
        calibration_id,
        generation_batch_size,
        use_cache,
        steering_policy,
        harmful_runtime.selection_parameters(model_key, method, calibration_id),
        shard_index,
        shard_count,
        model_and_tokenizer=(model, tokenizer),
        prompt_transform=format_prompt,
        completion_generator=generate_suffixes,
    )


def _generate_with_policy(
    base_benchmark: str,
    model_key: str,
    dataset: EvaluationDataset,
    method: str,
    device: str,
    calibration_id: str,
    generation_batch_size: int | None,
    use_cache: bool,
    policy,
    parameters: dict,
    shard_index: int,
    shard_count: int,
    *,
    model_and_tokenizer=None,
    prompt_transform=None,
    completion_generator=None,
) -> None:
    root = evaluation_root(base_benchmark, model_key, use_cache=use_cache)
    destination = generation_shard_path(
        root, dataset.cache_namespace, method, shard_index, shard_count
    )
    if destination.exists() and json.loads(destination.read_text()).get("status") == "complete":
        return
    source = json.loads(data_path(base_benchmark, model_key, dataset).read_text())
    model_spec = MODELS[model_key]
    if model_and_tokenizer is None:
        model, tokenizer = load_source_model(
            method_spec(method).model_loader,
            base_benchmark,
            model_spec.model_id,
            model_spec.revision,
            device,
            load_access_token(REPO),
        )
    else:
        model, tokenizer = model_and_tokenizer
    batch_size = generation_batch_size or model_spec.activation_batch_size
    common_identity = {
            "schema_version": 1,
            "base_benchmark": base_benchmark,
            "model": [model_spec.model_id, model_spec.revision],
            "method": method,
            "parameters": parameters,
            "calibration_id": calibration_id,
            "dataset": dataset.dataset,
            "dataset_role": dataset.role,
            "cache_namespace": dataset.cache_namespace,
            "generation_profile": dataset.generation_profile,
            "kv_cache": use_cache,
            "batch_size": batch_size,
    }
    payload = {
        "identity": {
            "base": common_identity,
            "shard": {"index": shard_index, "count": shard_count},
        },
        "status": "partial",
        "attempts": [],
        "batches": [],
    }
    if destination.exists():
        payload = json.loads(destination.read_text())
    attempt_started = time.perf_counter()
    attempt = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "runtime": runtime_provenance(device),
    }
    payload["attempts"].append(attempt)
    _write_json(destination, payload)
    assignments = _batch_assignments(source, dataset.cache_namespace, batch_size)
    assigned = [
        assignment
        for index, assignment in enumerate(assignments)
        if index % shard_count == shard_index
    ]
    completed = [
        (int(row["repetition"]), int(row["start"])) for row in payload["batches"]
    ]
    expected = [(repetition, start) for repetition, start, _records in assigned]
    if completed != expected[: len(completed)]:
        raise ValueError(f"Multiple-choice shard batches are not a valid prefix: {destination}")
    for repetition_index, start, records in assigned[len(completed) :]:
        seed = SOURCE_RANDOM_SEED + repetition_index * 100_000 + start
        batch_started = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        prompts = [str(row["text"]) for row in records]
        if prompt_transform is not None:
            prompts = [prompt_transform(prompt) for prompt in prompts]
        if completion_generator is None:
            completions = generate_batched(
                model,
                tokenizer,
                prompts,
                behavior=dataset.generation_profile,
                batch_size=batch_size,
                seed=seed,
                use_cache=use_cache,
                register_hooks=(
                    None
                    if policy is None
                    else lambda: register_generation_policy_hooks(model, policy)
                ),
                reset=None if policy is None else policy.reset,
            )
        else:
            completions = completion_generator(
                model, tokenizer, prompts, policy, use_cache, batch_size
            )
        payload["batches"].append(
            {
                "repetition": repetition_index,
                "start": start,
                "started_at_utc": started_at,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - batch_started,
                "generation_seed": seed,
                "rows": _output_rows(records, completions),
            }
        )
        _write_json(destination, payload)
    attempt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    attempt["elapsed_seconds"] = time.perf_counter() - attempt_started
    attempt["status"] = "complete"
    payload["status"] = "complete"
    _write_json(destination, payload)
    del model, tokenizer, policy
    gc.collect()
    torch.cuda.empty_cache()


def generate(
    base_benchmark: str,
    model_key: str,
    dataset: EvaluationDataset,
    method: str,
    device: str,
    calibration_id: str,
    generation_batch_size: int | None,
    use_cache: bool,
    shard_index: int,
    shard_count: int,
) -> None:
    prepare(base_benchmark, model_key, dataset)
    if generation_complete(
        generation_path(base_benchmark, model_key, dataset, method, use_cache=use_cache)
    ):
        return
    if base_benchmark == "truthfulness":
        _truthfulness_generation(
            model_key,
            dataset,
            method,
            device,
            calibration_id,
            generation_batch_size,
            use_cache,
            shard_index,
            shard_count,
        )
    elif base_benchmark == "toxicity":
        _toxicity_generation(
            model_key,
            dataset,
            method,
            device,
            calibration_id,
            generation_batch_size,
            use_cache,
            shard_index,
            shard_count,
        )
    elif base_benchmark == "harmful":
        _harmful_generation(
            model_key,
            dataset,
            method,
            device,
            calibration_id,
            generation_batch_size,
            use_cache,
            shard_index,
            shard_count,
        )
    else:
        raise ValueError(f"Unknown base benchmark {base_benchmark!r}")


def launch_generation(
    base_benchmark: str,
    model_key: str,
    dataset: EvaluationDataset,
    method: str,
    devices: list[str],
    calibration_id: str,
    generation_batch_size: int | None,
    use_cache: bool,
    log_root: Path,
) -> Path:
    """Run one method across all GPUs, then merge its data shards."""

    import sys

    command = [
        sys.executable,
        "-m", "robust_steerability.benchmarks.multiple_choice",
        "--benchmark", base_benchmark,
        "--model", model_key,
        "--dataset", dataset.key,
        "--method", method,
        "--calibration-id", calibration_id,
        "--kv-cache", "on" if use_cache else "off",
        "--device", "{device}",
        *(
            ["--generation-batch-size", str(generation_batch_size)]
            if generation_batch_size is not None
            else []
        ),
    ]
    spec = method_spec(method)
    batch_size = generation_batch_size or (
        source_generation_batch_size(MODELS[model_key].model_id, method)
        if base_benchmark == "truthfulness" and spec.runtime == "source"
        else MODELS[model_key].activation_batch_size
    )
    batch_count = dataset.repetitions * math.ceil(dataset.samples / batch_size)
    active_devices = devices[: min(len(devices), batch_count)]
    run_data_shards(
        f"generate-{dataset.key}-{method}", command, active_devices, log_root
    )
    return merge_generation_shards(
        cache_root=evaluation_root(base_benchmark, model_key, use_cache=use_cache),
        cache_namespace=dataset.cache_namespace,
        method=method,
        data_path=data_path(base_benchmark, model_key, dataset),
        evaluation_key=dataset.cache_namespace,
        batch_size=batch_size,
        shard_count=len(active_devices),
    )


def score_and_summarize(
    base_benchmark: str,
    model_key: str,
    dataset: EvaluationDataset,
    method: str,
    *,
    use_cache: bool,
) -> dict:
    root = evaluation_root(base_benchmark, model_key, use_cache=use_cache)
    generation = generation_path(
        base_benchmark, model_key, dataset, method, use_cache=use_cache
    )
    score_path = score_multiple_choice_generation(generation, root)
    score_payload = json.loads(score_path.read_text())
    source_payload = json.loads(data_path(base_benchmark, model_key, dataset).read_text())
    by_repetition = {}
    for row in score_payload["rows"]:
        by_repetition.setdefault(int(row["repetition"]), []).append(float(row["score"]))
    per_repetition = [
        {
            "repetition": repetition,
            "mmlu_accuracy": 100.0 * float(np.mean(values)),
            "samples": len(values),
        }
        for repetition, values in sorted(by_repetition.items())
    ]
    values = [float(row["score"]) for row in score_payload["rows"]]
    mean = float(np.mean(values))
    standard_error = math.sqrt(mean * (1.0 - mean) / len(values))
    result = {
        "identity": {
            "base_benchmark": base_benchmark,
            "model": [MODELS[model_key].model_id, MODELS[model_key].revision],
            "method": method,
            "dataset": source_payload["dataset"],
            "distribution": dataset.cache_namespace,
            "role": dataset.role,
            "kv_cache": use_cache,
            "scorers": ["mmlu_accuracy"],
        },
        "evaluation_samples_per_repetition": dataset.samples,
        "evaluation_repetitions": dataset.repetitions,
        "per_repetition": per_repetition,
        "metrics": {
            "mmlu_accuracy": {
                "mean": 100.0 * mean,
                "standard_error": 100.0 * standard_error,
            }
        },
    }
    local_destination = root / "results" / dataset.cache_namespace / f"{method}.json"
    tracked_destination = (
        results_root(base_benchmark, use_cache=use_cache)
        / model_key
        / dataset.cache_namespace
        / f"{method}.json"
    )
    _write_json(local_destination, result)
    _write_json(tracked_destination, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", choices=("truthfulness", "toxicity", "harmful"), required=True
    )
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--calibration-id", default="selected")
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument("--kv-cache", choices=("off", "on"), default="off")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    arguments = parser.parse_args()
    composition = load_composition(arguments.benchmark)
    dataset = composition.dataset(arguments.dataset)
    generate(
        arguments.benchmark,
        arguments.model,
        dataset,
        arguments.method,
        arguments.device,
        arguments.calibration_id,
        arguments.generation_batch_size,
        arguments.kv_cache == "on",
        arguments.shard_index,
        arguments.shard_count,
    )


if __name__ == "__main__":
    main()
