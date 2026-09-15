"""Generate and score TruthfulQA responses one model, method, and dataset at a time."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from robust_steerability.benchmarks.layout import (
    artifact_root,
    benchmark_root,
    calibration_root,
    dataset_root,
    evaluation_root,
    results_root,
)
from robust_steerability.benchmarks.specs import MODELS
from robust_steerability.modeling.huggingface import cuda_device_index, load_access_token
from robust_steerability.experiments.methods import ControllerArtifact, build_policy
from robust_steerability.experiments.resources import resolve_cuda_devices
from robust_steerability.modeling.interventions import register_generation_policy_hooks
from robust_steerability.judges import huggingface as huggingface_scoring
from robust_steerability.judges.specs import scorer_cache_path, scorer_spec
from robust_steerability.source_methods.generation import generate_batched
from robust_steerability.source_methods.id_benchmark import (
    fit_source_method_calibration,
    load_frozen_alqr_artifacts,
    run_generation_job,
    runtime_provenance,
)
from robust_steerability.source_methods.protocol import (
    ALQR_CALIBRATION_COUNTS,
    GENERATION,
    METHODS as SOURCE_METHODS,
    SOURCE_RANDOM_SEED,
    calibration_counts,
    paper_alqr_setting,
    selected_parameters as resolve_selected_parameters,
)
from robust_steerability.source_methods.modeling import load_source_model, source_model_spec


REPO = Path(__file__).resolve().parents[2]
UNIT = benchmark_root("truthfulness")
CURRENT_MODEL_KEY = "gemma2b"
CURRENT_CALIBRATION_ID = "selected"
CURRENT_USE_CACHE = False
CACHE_ROOT = evaluation_root(
    "truthfulness", CURRENT_MODEL_KEY, use_cache=CURRENT_USE_CACHE
)
SPANISH_DATA_PATH = (
    REPO / "parking/truthfulqa_spanish/data/truthfulqa_spanish.json"
)


TRUTHFULQA_ID = "truthful_qa"
TRUTHFULQA_REVISION = "741b8276f2d1982aa3d5b832d3ee81ed3b896490"
EVALUATION_REPETITIONS = 5
EVALUATION_SAMPLES = {"truthfulness": 817}
HINF_METHOD = "h_infinity"
HINF_GENERATION_BATCH_SIZE = 8
GENERATION_BATCH_SIZE_OVERRIDE: int | None = None


def _configure_runtime(
    model_key: str,
    calibration_id: str = "selected",
    generation_batch_size: int | None = None,
    *,
    use_cache: bool = False,
) -> None:
    """Select one model, calibration, and explicit decoding-cache condition."""

    global CACHE_ROOT, CURRENT_MODEL_KEY, CURRENT_CALIBRATION_ID
    global CURRENT_USE_CACHE, GENERATION_BATCH_SIZE_OVERRIDE
    if model_key not in MODELS:
        raise ValueError(f"Unknown model {model_key!r}")
    if not calibration_id or "/" in calibration_id:
        raise ValueError("calibration_id must be a simple name")
    if generation_batch_size is not None and generation_batch_size < 1:
        raise ValueError("generation_batch_size must be positive")
    CURRENT_MODEL_KEY = model_key
    CURRENT_CALIBRATION_ID = calibration_id
    CURRENT_USE_CACHE = use_cache
    GENERATION_BATCH_SIZE_OVERRIDE = generation_batch_size
    CACHE_ROOT = evaluation_root("truthfulness", model_key, use_cache=use_cache)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sample(records: list[dict], count: int, seed: int) -> list[dict]:
    if len(records) < count:
        raise ValueError(f"Requested {count} records from a pool of {len(records)}")
    indices = random.Random(seed).sample(range(len(records)), count)
    return [records[index] for index in indices]


def _data_path(behavior: str) -> Path:
    return dataset_root("truthfulness", CURRENT_MODEL_KEY) / f"{behavior}.json"


def _distribution_spec(behavior: str, distribution: str) -> tuple[Path, str, str]:
    if distribution == "id":
        return _data_path(behavior), behavior, behavior
    if behavior == "truthfulness" and distribution == "spanish":
        return SPANISH_DATA_PATH, "truthfulness_spanish", "truthfulness_spanish"
    raise ValueError(f"Unsupported benchmark distribution: {behavior}/{distribution}")


def build_truthfulqa_data(generation_rows: list[dict], multiple_choice_rows: list[dict]) -> dict:
    """Create pinned calibration selections and five full evaluation permutations."""

    true_records: list[dict] = []
    false_records: list[dict] = []
    for row_index, row in enumerate(multiple_choice_rows):
        question = str(row["question"]).strip()
        targets = row["mc2_targets"]
        for answer_index, (answer, label) in enumerate(
            zip(targets["choices"], targets["labels"], strict=True)
        ):
            record = {
                "prompt_id": f"truthfulqa:mc2:{row_index}:{answer_index}",
                "question_id": f"truthfulqa:{row_index}",
                "text": f"Q: {question} A: {str(answer).strip()}",
            }
            (true_records if int(label) == 1 else false_records).append(record)

    evaluation = [
        {
            "prompt_id": f"truthfulqa:{row_index}",
            "question": str(row["question"]).strip(),
            "text": f"Q: {str(row['question']).strip()} A:",
        }
        for row_index, row in enumerate(generation_rows)
    ]
    counts = ALQR_CALIBRATION_COUNTS["truthfulness"]
    maximum_per_class = max(
        calibration_counts(method, "truthfulness").desired for method in SOURCE_METHODS
    )
    payload = {
        "schema_version": 4,
        "behavior": "truthfulness",
        "seed": SOURCE_RANDOM_SEED,
        "evaluation_repetitions": EVALUATION_REPETITIONS,
        "evaluation_samples": len(evaluation),
        "datasets": {
            "truthfulness": {
                "id": TRUTHFULQA_ID,
                "revision": TRUTHFULQA_REVISION,
                "split": "validation",
            }
        },
        "calibration_protocol": {
            "shared_pool_count_per_class": maximum_per_class,
            "nested_prefixes": {
                method: calibration_counts(method, "truthfulness").__dict__
                for method in SOURCE_METHODS
            },
            "alqr_negative_count": counts.undesired,
            "alqr_positive_count": counts.desired,
            "jacobian_count": counts.jacobian,
            "jacobian_class": counts.jacobian_class,
            "jacobian_max_length": counts.jacobian_max_length,
            "negative_sampling_seed": SOURCE_RANDOM_SEED,
            "positive_sampling_seed": SOURCE_RANDOM_SEED + 1,
            "jacobian_sampling_seed": SOURCE_RANDOM_SEED + 2,
            "jacobian_selection": "independent sample from the true-answer pool",
        },
        "evaluation_protocol": {
            "sampling": "full-set permutation without replacement",
            "seed_start": SOURCE_RANDOM_SEED,
            "repetition_seed_stride": 100_000,
        },
        "calibration": {
            "truthfulness": {
                "undesired": _sample(false_records, maximum_per_class, SOURCE_RANDOM_SEED),
                "desired": _sample(true_records, maximum_per_class, SOURCE_RANDOM_SEED + 1),
                "jacobian": _sample(true_records, counts.jacobian, SOURCE_RANDOM_SEED + 2),
            }
        },
        "evaluation": {
            "truthfulness": {
                str(repetition): _sample(
                    evaluation,
                    len(evaluation),
                    SOURCE_RANDOM_SEED + 100_000 * repetition,
                )
                for repetition in range(EVALUATION_REPETITIONS)
            }
        },
    }
    return payload


def prepare(behavior: str, distribution: str) -> None:
    if behavior != "truthfulness":
        raise ValueError("This runtime owns only the truthfulness benchmark")
    if distribution == "spanish":
        if behavior != "truthfulness":
            raise ValueError("Spanish distribution is available only for truthfulness")
        id_path = _data_path(behavior)
        if not id_path.exists():
            raise ValueError(f"Missing frozen ID dataset cache: {id_path}")
        if not SPANISH_DATA_PATH.exists():
            raise ValueError(f"Missing frozen Spanish dataset: {SPANISH_DATA_PATH}")
        return
    if distribution != "id":
        raise ValueError(f"Unknown benchmark distribution: {distribution}")
    destination = _data_path(behavior)
    if destination.exists():
        return
    started = time.perf_counter()
    started_at = _utc_now()
    generation = list(
        load_dataset(
            TRUTHFULQA_ID,
            "generation",
            split="validation",
            revision=TRUTHFULQA_REVISION,
        )
    )
    multiple_choice = list(
        load_dataset(
            TRUTHFULQA_ID,
            "multiple_choice",
            split="validation",
            revision=TRUTHFULQA_REVISION,
        )
    )
    if len(generation) != EVALUATION_SAMPLES["truthfulness"]:
        raise ValueError(
            f"Expected {EVALUATION_SAMPLES['truthfulness']} TruthfulQA questions; "
            f"found {len(generation)}"
        )
    payload = build_truthfulqa_data(generation, multiple_choice)
    source_counts = {
        "generation_source_rows": len(generation),
        "multiple_choice_source_rows": len(multiple_choice),
    }
    payload["preparation"] = {
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": runtime_provenance("cpu"),
        **source_counts,
    }
    _write_json(destination, payload)


def _hinf_paths(model_key: str) -> dict[str, Path]:
    artifacts = artifact_root("truthfulness", model_key)
    selected = calibration_root(
        "truthfulness", model_key, HINF_METHOD, CURRENT_CALIBRATION_ID
    )
    return {
        "alqr_data": artifacts / "data.json",
        "alqr_setpoint": artifacts / "setpoint.pt",
        "alqr_dynamics": artifacts / "dynamics.pt",
        "hinf_controller": selected / "base" / "controller.pt",
        "selected_controller": selected / "controller.pt",
        "hyperparameter_calibration": selected / "selection.json",
    }


def _load_selected_hinf(
    model_key: str,
) -> tuple[ControllerArtifact, dict[str, float]]:
    paths = _hinf_paths(model_key)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise ValueError(f"Missing frozen H-infinity artifacts: {missing}")

    calibration = json.loads(paths["hyperparameter_calibration"].read_text())
    configuration = calibration["selected"]
    parameters = {
        name: float(configuration["parameters"][name])
        for name in ("lambda", "q", "r", "q_final")
    }
    strategy = calibration.get("protocol", {}).get("selection_strategy")
    if (
        calibration.get("schema_version") != 2
        or calibration.get("model")
        != [MODELS[model_key].model_id, MODELS[model_key].revision]
        or calibration.get("benchmark") != "truthfulness"
        or calibration.get("calibration_id") != CURRENT_CALIBRATION_ID
        or strategy not in {"fixed", "grid"}
        or configuration.get("source")
        not in {
            "fixed configuration supplied at calibration launch",
            "TruthfulQA True calibration-grid argmax",
        }
    ):
        raise ValueError("Frozen H-infinity calibration metadata is invalid")
    diagnostic_bundle = calibration_root(
        "truthfulness", model_key, HINF_METHOD, CURRENT_CALIBRATION_ID
    ) / calibration["diagnostic_bundle"]
    if not diagnostic_bundle.exists():
        raise FileNotFoundError(f"Missing Hannah diagnostic bundle: {diagnostic_bundle}")

    controller = torch.load(
        paths["selected_controller"], map_location="cpu", weights_only=True, mmap=True
    )
    controller_identity = controller.get("identity", {})
    if (
        controller_identity.get("model")
        != [MODELS[model_key].model_id, MODELS[model_key].revision]
        or controller_identity.get("task") != "truthfulness"
        or controller_identity.get("parameters") != parameters
        or controller_identity.get("configuration_source")
        != configuration.get("source")
        or controller_identity.get("configuration_id")
        != configuration.get("configuration_id")
        or not bool(controller.get("feasible"))
    ):
        raise ValueError("Frozen H-infinity controller does not match its selection record")

    source = torch.load(
        paths["hinf_controller"], map_location="cpu", weights_only=True, mmap=True
    )
    base = ControllerArtifact(**source["artifact"])
    artifact = replace(
        base,
        setpoints=base.setpoints,
        hinf_gains=controller["gains"],
        hinf_feasible=bool(controller["feasible"]),
        gamma_star=float(controller["gamma_star"]),
        hinf_diagnostics=controller["diagnostics"],
    )
    return artifact, parameters


def _hinf_generation_identity(
    model_key: str, behavior: str, distribution: str
) -> dict:
    if behavior != "truthfulness":
        raise ValueError("This runtime only evaluates truthfulness")
    data_path, evaluation_key, cache_namespace = _distribution_spec(
        behavior, distribution
    )
    data = json.loads(data_path.read_text())
    _artifact, parameters = _load_selected_hinf(model_key)
    model = MODELS[model_key]
    return {
        "schema_version": 1,
        "model_id": model.model_id,
        "checkpoint_revision": model.revision,
        "behavior": behavior,
        "distribution": distribution,
        "evaluation_key": evaluation_key,
        "cache_namespace": cache_namespace,
        "method": HINF_METHOD,
        "parameters": parameters,
        "calibration_id": CURRENT_CALIBRATION_ID,
        "model_loading": asdict(
            source_model_spec("alqr", behavior, model.model_id, model.revision)
        ),
        "protocol": {
            "evaluation_repetitions": EVALUATION_REPETITIONS,
            "evaluation_samples_per_repetition": len(
                data["evaluation"][evaluation_key]["0"]
            ),
            "generation": GENERATION[behavior],
            "batch_size": (
                GENERATION_BATCH_SIZE_OVERRIDE or HINF_GENERATION_BATCH_SIZE
            ),
            "use_cache": CURRENT_USE_CACHE,
            "seed": SOURCE_RANDOM_SEED,
            "repetition_seed_stride": 100_000,
            "batch_seed_rule": "repetition_seed_plus_batch_start",
            "feedback": "full rank-8 reduced-state deviation",
        },
    }


def _hinf_shard_path(
    model_key: str,
    behavior: str,
    distribution: str,
    shard_index: int,
    shard_count: int,
) -> Path:
    _data_path_value, _evaluation_key, cache_namespace = _distribution_spec(
        behavior, distribution
    )
    return (
        CACHE_ROOT
        / "generation_shards"
        / cache_namespace
        / HINF_METHOD
        / f"shard_{shard_index:02d}_of_{shard_count:02d}.json"
    )


def _generate_hinf_shard(
    model_key: str,
    device: str,
    behavior: str,
    distribution: str,
    shard_index: int,
    shard_count: int,
) -> None:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("H-infinity generation requires a valid nonempty shard set")
    if not device.startswith("cuda:"):
        raise ValueError("H-infinity benchmark generation requires an explicit CUDA device")
    prepare(behavior, distribution)
    common_identity = _hinf_generation_identity(model_key, behavior, distribution)
    assigned_repetitions = list(range(shard_index, EVALUATION_REPETITIONS, shard_count))
    destination = _hinf_shard_path(
        model_key, behavior, distribution, shard_index, shard_count
    )
    identity = {
        **common_identity,
        "shard": {
            "index": shard_index,
            "count": shard_count,
            "assigned_repetitions": assigned_repetitions,
        },
    }
    payload = {"identity": identity, "status": "partial", "attempts": [], "repetitions": []}
    if destination.exists():
        payload = json.loads(destination.read_text())
        if payload.get("status") == "complete":
            return

    completed = [row["repetition"] for row in payload["repetitions"]]
    if completed != assigned_repetitions[: len(completed)]:
        raise ValueError(f"H-infinity shard repetitions are not a valid prefix: {destination}")

    model_spec = MODELS[model_key]
    token = load_access_token(REPO)
    attempt_started = time.perf_counter()
    attempt = {
        "started_at_utc": _utc_now(),
        "status": "running",
        "runtime": runtime_provenance(device),
    }
    payload["attempts"].append(attempt)
    _write_json(destination, payload)
    torch.manual_seed(SOURCE_RANDOM_SEED)
    torch.cuda.manual_seed_all(SOURCE_RANDOM_SEED)
    torch.cuda.reset_peak_memory_stats(cuda_device_index(device))

    model_load_started = time.perf_counter()
    model, tokenizer = load_source_model(
        "alqr", behavior, model_spec.model_id, model_spec.revision, device, token
    )
    attempt["model_load_elapsed_seconds"] = time.perf_counter() - model_load_started
    artifact, _parameters = _load_selected_hinf(model_key)
    policy = build_policy("hinf", artifact, kp=0.0, ki=0.0, kd=0.0)
    data_path, evaluation_key, _cache_namespace = _distribution_spec(
        behavior, distribution
    )
    data = json.loads(data_path.read_text())

    for repetition in assigned_repetitions[len(completed) :]:
        records = data["evaluation"][evaluation_key][str(repetition)]
        if len(records) != EVALUATION_SAMPLES[behavior]:
            raise ValueError("H-infinity evaluation repetition has the wrong sample count")
        repetition_started = time.perf_counter()
        started_at = _utc_now()
        repetition_seed = SOURCE_RANDOM_SEED + repetition * 100_000
        completions = generate_batched(
            model,
            tokenizer,
            [str(row["text"]) for row in records],
            behavior=behavior,
            batch_size=(
                GENERATION_BATCH_SIZE_OVERRIDE or HINF_GENERATION_BATCH_SIZE
            ),
            seed=repetition_seed,
            use_cache=CURRENT_USE_CACHE,
            register_hooks=lambda: register_generation_policy_hooks(model, policy),
        )
        output_rows = []
        for record, completion in zip(records, completions, strict=True):
            output_rows.append(
                {
                    "prompt_id": record["prompt_id"],
                    "question": record["question"],
                    "text": record["text"],
                    "completion": completion,
                }
            )
        payload["repetitions"].append(
            {
                "repetition": repetition,
                "started_at_utc": started_at,
                "finished_at_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - repetition_started,
                "sample_count": len(records),
                "generation_seed": repetition_seed,
                "rows": output_rows,
            }
        )
        _write_json(destination, payload)

    attempt["finished_at_utc"] = _utc_now()
    attempt["elapsed_seconds"] = time.perf_counter() - attempt_started
    attempt["completed_repetitions"] = len(payload["repetitions"])
    attempt["gpu_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
        cuda_device_index(device)
    )
    attempt["gpu_peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(
        cuda_device_index(device)
    )
    attempt["status"] = "complete"
    payload["status"] = "complete"
    _write_json(destination, payload)


def merge_hinf_generation(
    model_key: str, behavior: str, distribution: str, shard_count: int
) -> Path:
    common_identity = _hinf_generation_identity(model_key, behavior, distribution)
    _data_path_value, _evaluation_key, cache_namespace = _distribution_spec(
        behavior, distribution
    )
    shard_paths = [
        _hinf_shard_path(
            model_key, behavior, distribution, index, shard_count
        )
        for index in range(shard_count)
    ]
    repetitions: list[dict] = []
    attempts: list[dict] = []
    for index, path in enumerate(shard_paths):
        if not path.exists():
            raise ValueError(f"Missing H-infinity generation shard: {path}")
        shard = json.loads(path.read_text())
        if shard.get("status") != "complete":
            raise ValueError(f"Invalid H-infinity generation shard: {path}")
        repetitions.extend(shard["repetitions"])
        attempts.extend(shard["attempts"])
    repetitions.sort(key=lambda row: row["repetition"])
    if [row["repetition"] for row in repetitions] != list(range(EVALUATION_REPETITIONS)):
        raise ValueError("Merged H-infinity repetitions are incomplete")
    if any(len(row["rows"]) != EVALUATION_SAMPLES[behavior] for row in repetitions):
        raise ValueError("Merged H-infinity repetition has the wrong sample count")

    parameters = common_identity["parameters"]
    destination = (
        CACHE_ROOT
        / "generations"
        / cache_namespace
        / HINF_METHOD
        / "final.json"
    )
    payload = {
        "identity": common_identity,
        "status": "complete",
        "attempts": attempts,
        "repetitions": repetitions,
        "capability_evaluation": {},
    }
    if not destination.exists():
        _write_json(destination, payload)
    return destination


def launch_hinf_generation(
    model_key: str,
    behavior: str,
    distribution: str,
    devices: list[str],
    log_root: Path,
) -> None:
    prepare(behavior, distribution)
    _data_path_value, _evaluation_key, cache_namespace = _distribution_spec(
        behavior, distribution
    )
    log_root.mkdir(parents=True, exist_ok=True)
    running = []
    if not devices:
        raise ValueError("H-infinity generation requires at least one CUDA device")
    shard_count = len(devices)
    for shard_index, device in enumerate(devices):
        log_path = log_root / (
            f"generate_{cache_namespace}_{model_key}_{HINF_METHOD}_shard_{shard_index}.log"
        )
        handle = log_path.open("a")
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--stage", "generate",
                "--model", model_key,
                "--method", HINF_METHOD,
                "--behavior", behavior,
                "--distribution", distribution,
                "--calibration-id", CURRENT_CALIBRATION_ID,
                "--kv-cache", "on" if CURRENT_USE_CACHE else "off",
                "--device", device,
                "--shard-index", str(shard_index),
                "--shard-count", str(shard_count),
                *(
                    ["--generation-batch-size", str(GENERATION_BATCH_SIZE_OVERRIDE)]
                    if GENERATION_BATCH_SIZE_OVERRIDE is not None
                    else []
                ),
            ],
            cwd=REPO,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        running.append((shard_index, process, handle, log_path))
    failures = []
    for shard_index, process, handle, log_path in running:
        return_code = process.wait()
        handle.close()
        if return_code:
            failures.append(f"shard {shard_index} failed; see {log_path}")
    if failures:
        raise RuntimeError("; ".join(failures))
    merge_hinf_generation(model_key, behavior, distribution, shard_count)


def generate(
    model_key: str,
    method: str,
    device: str,
    behavior: str,
    distribution: str,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> None:
    if method == HINF_METHOD:
        if shard_index is None or shard_count is None:
            raise ValueError("H-infinity generation requires shard index and count")
        _generate_hinf_shard(
            model_key,
            device,
            behavior,
            distribution,
            shard_index,
            shard_count,
        )
        return
    if method not in SOURCE_METHODS:
        raise ValueError(f"Unsupported source method {method!r}")
    data_path, evaluation_key, cache_namespace = _distribution_spec(
        behavior, distribution
    )
    if not data_path.exists():
        raise ValueError(
            f"Run the prepare stage for {behavior}/{distribution} before generation"
        )
    model = MODELS[model_key]
    selected_parameters = None
    if method != "original":
        selection_path = calibration_root(
            "truthfulness", model_key, method, CURRENT_CALIBRATION_ID
        ) / "selection.json"
        if not selection_path.exists():
            raise FileNotFoundError(
                f"Run the calibration stage before evaluating {method}: {selection_path}"
            )
        selection = json.loads(selection_path.read_text())
        if (
            selection.get("model") != [model.model_id, model.revision]
            or selection.get("behavior") != behavior
            or selection.get("method") != method
            or selection.get("calibration_id") != CURRENT_CALIBRATION_ID
        ):
            raise ValueError(f"Calibration selection mismatch: {selection_path}")
        if method in {"spid", "iti"}:
            selected_parameters = selection["parameters"]
        elif selection.get("parameters") != resolve_selected_parameters(
            method, behavior, model.model_id
        ):
            raise ValueError(f"Fixed source selection changed: {selection_path}")
    run_generation_job(
        cache_root=CACHE_ROOT,
        behavior=behavior,
        model_id=model.model_id,
        revision=model.revision,
        method=method,
        device=device,
        token=load_access_token(REPO),
        alqr_artifact_root=artifact_root("truthfulness", model_key),
        method_calibration_root=calibration_root(
            "truthfulness", model_key, method, CURRENT_CALIBRATION_ID
        ),
        calibration_data_path=_data_path(behavior),
        data_path=data_path,
        evaluation_key=evaluation_key,
        cache_namespace=cache_namespace,
        generation_profile="truthfulness",
        use_cache=CURRENT_USE_CACHE,
        selected_parameters=selected_parameters,
        generation_batch_size=GENERATION_BATCH_SIZE_OVERRIDE,
    )


def fit_source(
    model_key: str,
    method: str,
    device: str,
    calibration_id: str,
) -> None:
    """Fit one source-method artifact using its frozen parameter selection."""

    if method in {"original", "alqr"} or method not in SOURCE_METHODS:
        raise ValueError(f"{method!r} has no source-method fit")
    _configure_runtime(model_key, calibration_id)
    prepare("truthfulness", "id")
    model = MODELS[model_key]
    fit_source_method_calibration(
        behavior="truthfulness",
        model_id=model.model_id,
        revision=model.revision,
        method=method,
        device=device,
        token=load_access_token(REPO),
        calibration_root=calibration_root(
            "truthfulness", model_key, method, calibration_id
        ),
        calibration_data_path=_data_path("truthfulness"),
        selected_parameters=None,
    )


def _generation_files(
    model_key: str, method: str, behavior: str, distribution: str
) -> list[Path]:
    _data_path_value, _evaluation_key, cache_namespace = _distribution_spec(
        behavior, distribution
    )
    path = CACHE_ROOT / "generations" / cache_namespace / method / "final.json"
    return [path] if path.exists() else []


def score_scorer(
    model_key: str,
    method: str,
    device: str,
    behavior: str,
    distribution: str,
    scorer_key: str,
) -> None:
    if behavior != "truthfulness":
        raise ValueError("This runtime owns only the truthfulness benchmark")
    files = _generation_files(model_key, method, behavior, distribution)
    if len(files) != 1:
        raise ValueError(f"Expected one {method} generation cache; found {len(files)}")
    huggingface_scoring.score_generation(
        files[0], CACHE_ROOT, scorer_key, device, load_access_token(REPO)
    )


def _mean_se(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1) / math.sqrt(len(array)))


def summarize_truthfulness(
    model_key: str,
    method: str,
    distribution: str,
    scorer_keys: tuple[str, ...],
) -> dict:
    model = MODELS[model_key]
    data_path, evaluation_key, cache_namespace = _distribution_spec(
        "truthfulness", distribution
    )
    files = _generation_files(model_key, method, "truthfulness", distribution)
    if len(files) != 1:
        raise ValueError(f"Expected one {method} generation cache; found {len(files)}")
    generation_path = files[0]
    generation = json.loads(generation_path.read_text())
    if generation["status"] != "complete":
        raise ValueError(f"Cannot summarize incomplete generation for {method}")
    invalid_scorer_outputs = {}
    scores = {}
    for key in scorer_keys:
        path = scorer_cache_path(CACHE_ROOT, generation_path, key)
        if not path.exists():
            raise FileNotFoundError(f"Missing {key} scorer output: {path}")
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete":
            raise ValueError(f"Incomplete {key} scorer output: {path}")
        scores[key] = payload["rows"]
        invalid_scorer_outputs[key] = [
            {
                "prompt_id": row["prompt_id"],
                "raw_answer": row["raw_answer"],
                "score": row["score"],
            }
            for row in payload["rows"]
            if not row.get("valid", True)
        ]

    offset = 0
    per_repetition = []
    for repetition in generation["repetitions"]:
        count = len(repetition["rows"])
        row = {"repetition": repetition["repetition"]}
        for key in scorer_keys:
            spec = scorer_spec(key)
            value = float(
                np.mean([item["score"] for item in scores[key][offset:offset + count]])
            )
            row[spec.metric] = 100.0 * value if spec.maximum == 1.0 else value
        per_repetition.append(row)
        offset += count
    metrics = {}
    for key in scorer_keys:
        metric = scorer_spec(key).metric
        mean, standard_error = _mean_se([row[metric] for row in per_repetition])
        metrics[metric] = {"mean": mean, "standard_error": standard_error}
    result = {
        "identity": {
            "model_id": model.model_id,
            "model_revision": model.revision,
            "method": method,
            "distribution": distribution,
            "dataset": [TRUTHFULQA_ID, TRUTHFULQA_REVISION],
            "evaluation_key": evaluation_key,
            "kv_cache": CURRENT_USE_CACHE,
            "scorers": list(scorer_keys),
        },
        "evaluation_samples_per_repetition": EVALUATION_SAMPLES["truthfulness"],
        "evaluation_repetitions": EVALUATION_REPETITIONS,
        "created_at_utc": _utc_now(),
        "per_repetition": per_repetition,
        "metrics": metrics,
        "invalid_scorer_outputs": invalid_scorer_outputs,
    }
    destination = CACHE_ROOT / "results" / cache_namespace / f"{method}.json"
    if destination.exists():
        existing = json.loads(destination.read_text())
        result["metrics"] = {**existing.get("metrics", {}), **result["metrics"]}
        existing_rows = {
            int(row["repetition"]): row for row in existing.get("per_repetition", [])
        }
        for row in result["per_repetition"]:
            existing_rows.setdefault(int(row["repetition"]), {}).update(row)
        result["per_repetition"] = [existing_rows[index] for index in sorted(existing_rows)]
        result["invalid_scorer_outputs"] = {
            **existing.get("invalid_scorer_outputs", {}),
            **result["invalid_scorer_outputs"],
        }
        result["identity"]["scorers"] = sorted(
            set(existing.get("identity", {}).get("scorers", [])) | set(scorer_keys)
        )
    _write_json(destination, result)
    _write_json(
        results_root("truthfulness", use_cache=CURRENT_USE_CACHE)
        / model_key
        / cache_namespace
        / f"{method}.json",
        result,
    )
    return result


def summarize(
    model_key: str,
    method: str,
    behavior: str,
    distribution: str,
    scorer_keys: tuple[str, ...],
) -> dict:
    if behavior != "truthfulness":
        raise ValueError("This runtime owns only the truthfulness benchmark")
    return summarize_truthfulness(model_key, method, distribution, scorer_keys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "prepare",
            "fit-source",
            "generate",
            "generate-hinf",
            "merge-hinf",
            "score-scorer",
            "summarize",
        ),
        required=True,
    )
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--method", choices=(*SOURCE_METHODS, HINF_METHOD))
    parser.add_argument("--behavior", choices=("truthfulness",), required=True)
    parser.add_argument("--distribution", choices=("id", "spanish"), required=True)
    parser.add_argument("--calibration-id", default="selected")
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument(
        "--scorer", choices=("truthfulqa_true", "truthfulqa_informative")
    )
    parser.add_argument(
        "--scorers", default="truthfulqa_true,truthfulqa_informative"
    )
    parser.add_argument("--kv-cache", choices=("off", "on"), default="off")
    parser.add_argument("--device")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    arguments = parser.parse_args()
    _configure_runtime(
        arguments.model,
        arguments.calibration_id,
        arguments.generation_batch_size,
        use_cache=arguments.kv_cache == "on",
    )
    if arguments.stage == "prepare":
        prepare(arguments.behavior, arguments.distribution)
    elif arguments.stage == "fit-source":
        if arguments.method is None or arguments.device is None:
            raise ValueError("fit-source requires --method and --device")
        fit_source(
            arguments.model,
            arguments.method,
            arguments.device,
            arguments.calibration_id,
        )
    elif arguments.stage == "generate":
        if arguments.method is None or arguments.device is None:
            raise ValueError(f"{arguments.stage} requires --method and --device")
        generate(
            arguments.model,
            arguments.method,
            arguments.device,
            arguments.behavior,
            arguments.distribution,
            arguments.shard_index,
            arguments.shard_count,
        )
    elif arguments.stage == "generate-hinf":
        launch_hinf_generation(
            arguments.model,
            arguments.behavior,
            arguments.distribution,
            resolve_cuda_devices(arguments.devices),
            CACHE_ROOT / "logs",
        )
    elif arguments.stage == "merge-hinf":
        if arguments.shard_count is None:
            raise ValueError("merge-hinf requires --shard-count")
        merge_hinf_generation(
            arguments.model,
            arguments.behavior,
            arguments.distribution,
            arguments.shard_count,
        )
    elif arguments.stage == "score-scorer":
        if (
            arguments.method is None
            or arguments.device is None
            or arguments.scorer is None
            or arguments.behavior != "truthfulness"
        ):
            raise ValueError(
                "score-scorer requires a method, --scorer, and --device"
            )
        score_scorer(
            arguments.model,
            arguments.method,
            arguments.device,
            arguments.behavior,
            arguments.distribution,
            arguments.scorer,
        )
    else:
        if arguments.method is None:
            raise ValueError("summarize requires --method")
        summarize(
            arguments.model,
            arguments.method,
            arguments.behavior,
            arguments.distribution,
            tuple(item.strip() for item in arguments.scorers.split(",") if item.strip()),
        )


if __name__ == "__main__":
    main()
