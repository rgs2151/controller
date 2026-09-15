"""Run the paper benchmark one model, method, and dataset at a time."""

from __future__ import annotations

import argparse
import gc
import hashlib
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

from robust_steerability.benchmarks.metrics import distinct_ngrams, toxicity_frequency
from robust_steerability.benchmarks.layout import (
    artifact_root,
    benchmark_root,
    calibration_root,
    evaluation_root,
    results_root,
)
from robust_steerability.benchmarks.specs import MODELS
from robust_steerability.datasets.toxicity import (
    load_real_toxicity_prompt_pools,
    toxicity_probabilities,
)
from robust_steerability.datasets.truthfulqa import (
    load_mmlu_five_shot_prompts,
    parse_mmlu_letter,
)
from robust_steerability.modeling.huggingface import (
    CausalModelLoadSpec,
    cuda_device_index,
    load_access_token,
    load_causal_model,
    load_sequence_classifier,
)
from robust_steerability.experiments.methods import ControllerArtifact, build_policy
from robust_steerability.experiments.resources import resolve_cuda_devices
from robust_steerability.modeling.interventions import register_generation_policy_hooks
from robust_steerability.judges import huggingface as huggingface_judges
from robust_steerability.judges.specs import judge_cache_path, judge_spec
from robust_steerability.source_methods.generation import generate_batched
from robust_steerability.source_methods.id_benchmark import (
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
CACHE_ROOT = evaluation_root("truthfulness", CURRENT_MODEL_KEY)
SPANISH_DATA_PATH = (
    REPO / "parking/truthfulqa_spanish/data/truthfulqa_spanish.json"
)


TRUTHFULQA_ID = "truthful_qa"
TRUTHFULQA_REVISION = "741b8276f2d1982aa3d5b832d3ee81ed3b896490"
RTP_ID = "allenai/real-toxicity-prompts"
RTP_REVISION = "f21629712ffd6a3d13a54fd2807ccd521c55ef74"
MMLU_ID = "cais/mmlu"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
TOXICITY_CLASSIFIER = "s-nlp/roberta_toxicity_classifier"
TOXICITY_CLASSIFIER_REVISION = "048c25bb1e199b98802784f96325f4840f22145d"
PERPLEXITY_MODEL = "mistralai/Mistral-7B-v0.1"
PERPLEXITY_MODEL_REVISION = "27d67f1b5f57dc0953326b2601d68371d40ea8da"
EVALUATION_REPETITIONS = 5
EVALUATION_SAMPLES = {"truthfulness": 817, "toxicity": 1000}
MMLU_SAMPLES = 1000
TOXICITY_BATCH_SIZE = 16
PERPLEXITY_BATCH_SIZE = 10
PERPLEXITY_MAX_LENGTH = 128
HINF_METHOD = "h_infinity"
HINF_GENERATION_BATCH_SIZE = 8
GENERATION_BATCH_SIZE_OVERRIDE: int | None = None


def _configure_runtime(
    model_key: str,
    calibration_id: str = "selected",
    generation_batch_size: int | None = None,
) -> None:
    """Select one model and calibration for cache-off controlled decoding."""

    global CACHE_ROOT, CURRENT_MODEL_KEY, CURRENT_CALIBRATION_ID
    global GENERATION_BATCH_SIZE_OVERRIDE
    if model_key not in MODELS:
        raise ValueError(f"Unknown model {model_key!r}")
    if not calibration_id or "/" in calibration_id:
        raise ValueError("calibration_id must be a simple name")
    if generation_batch_size is not None and generation_batch_size < 1:
        raise ValueError("generation_batch_size must be positive")
    CURRENT_MODEL_KEY = model_key
    CURRENT_CALIBRATION_ID = calibration_id
    GENERATION_BATCH_SIZE_OVERRIDE = generation_batch_size
    CACHE_ROOT = evaluation_root("truthfulness", model_key)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample(records: list[dict], count: int, seed: int) -> list[dict]:
    if len(records) < count:
        raise ValueError(f"Requested {count} records from a pool of {len(records)}")
    indices = random.Random(seed).sample(range(len(records)), count)
    return [records[index] for index in indices]


def _data_fingerprint(payload: dict) -> str:
    scientific_payload = {
        key: value for key, value in payload.items() if key not in {"fingerprint", "preparation"}
    }
    return hashlib.sha256(
        json.dumps(scientific_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _data_path(behavior: str) -> Path:
    return CACHE_ROOT / "data" / f"{behavior}.json"


def _distribution_spec(behavior: str, distribution: str) -> tuple[Path, str, str]:
    if distribution == "id":
        return _data_path(behavior), behavior, behavior
    if behavior == "truthfulness" and distribution == "spanish":
        return SPANISH_DATA_PATH, "truthfulness_spanish", "truthfulness_spanish"
    raise ValueError(f"Unsupported benchmark distribution: {behavior}/{distribution}")

def _validate_spanish_data(payload: dict, id_data: dict) -> None:
    fingerprint_payload = {
        key: value for key, value in payload.items() if key != "fingerprint"
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    repetitions = payload.get("evaluation", {}).get("truthfulness_spanish", {})
    if (
        payload.get("status") != "quality_checked"
        or payload.get("distribution") != "truthfulqa_spanish"
        or payload.get("quality_audit", {}).get("final_failed") != 0
        or payload.get("quality_audit", {}).get("final_passed")
        != EVALUATION_SAMPLES["truthfulness"]
        or payload.get("source", {}).get("source_fingerprint")
        != id_data.get("fingerprint")
        or payload.get("fingerprint") != fingerprint
        or set(repetitions) != {str(index) for index in range(EVALUATION_REPETITIONS)}
    ):
        raise ValueError(f"Frozen Spanish TruthfulQA dataset is invalid: {SPANISH_DATA_PATH}")
    id_repetitions = id_data["evaluation"]["truthfulness"]
    for repetition in range(EVALUATION_REPETITIONS):
        rows = repetitions[str(repetition)]
        id_rows = id_repetitions[str(repetition)]
        if (
            len(rows) != EVALUATION_SAMPLES["truthfulness"]
            or [row.get("prompt_id") for row in rows]
            != [row.get("prompt_id") for row in id_rows]
            or [row.get("question") for row in rows]
            != [row.get("question") for row in id_rows]
            or any(
                "Responde en inglés.\nRespuesta:" not in str(row.get("text", ""))
                for row in rows
            )
        ):
            raise ValueError(
                f"Spanish TruthfulQA repetition {repetition} is not aligned with ID"
            )


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
    payload["fingerprint"] = _data_fingerprint(payload)
    return payload


def _validate_data(saved: dict, behavior: str, destination: Path) -> None:
    calibration = saved.get("calibration", {}).get(behavior, {})
    counts = ALQR_CALIBRATION_COUNTS[behavior]
    maximum_per_class = max(
        calibration_counts(method, behavior).desired for method in SOURCE_METHODS
    )
    if (
        saved.get("schema_version") != 4
        or saved.get("behavior") != behavior
        or saved.get("evaluation_samples") != EVALUATION_SAMPLES[behavior]
        or saved.get("evaluation_repetitions") != EVALUATION_REPETITIONS
        or len(calibration.get("undesired", [])) != maximum_per_class
        or len(calibration.get("desired", [])) != maximum_per_class
        or len(calibration.get("jacobian", [])) != counts.jacobian
        or saved.get("calibration_protocol", {}).get("jacobian_max_length")
        != counts.jacobian_max_length
        or saved.get("fingerprint") != _data_fingerprint(saved)
    ):
        raise ValueError(f"Dataset cache does not match the benchmark protocol: {destination}")


def prepare(behavior: str, distribution: str) -> None:
    if behavior != "truthfulness":
        raise ValueError("This runtime owns only the truthfulness benchmark")
    if distribution == "spanish":
        if behavior != "truthfulness":
            raise ValueError("Spanish distribution is available only for truthfulness")
        id_path = _data_path(behavior)
        if not id_path.exists():
            raise ValueError(f"Missing frozen ID dataset cache: {id_path}")
        id_data = json.loads(id_path.read_text())
        _validate_data(id_data, behavior, id_path)
        if not SPANISH_DATA_PATH.exists():
            raise ValueError(f"Missing frozen Spanish dataset: {SPANISH_DATA_PATH}")
        _validate_spanish_data(json.loads(SPANISH_DATA_PATH.read_text()), id_data)
        return
    if distribution != "id":
        raise ValueError(f"Unknown benchmark distribution: {distribution}")
    destination = _data_path(behavior)
    if destination.exists():
        saved = json.loads(destination.read_text())
        _validate_data(saved, behavior, destination)
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


def _object_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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
        "hinf_input": selected / "base" / "controller_diagnostics" / "input.pt",
        "selected_controller": selected / "controller.pt",
        "hyperparameter_calibration": selected / "selection.json",
    }


def _load_selected_hinf(
    model_key: str,
) -> tuple[ControllerArtifact, dict[str, float], dict[str, str]]:
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
    if (
        calibration.get("schema_version") != 1
        or calibration.get("model")
        != [MODELS[model_key].model_id, MODELS[model_key].revision]
        or calibration.get("benchmark") != "truthfulness"
        or calibration.get("calibration_id") != CURRENT_CALIBRATION_ID
        or calibration.get("protocol", {}).get("samples") != 100
        or calibration.get("protocol", {}).get("repetitions") != 5
        or calibration.get("controller_sha256") != _sha(paths["selected_controller"])
        or configuration.get("source")
         != "five-repetition calibration-grid argmax"
    ):
        raise ValueError("Frozen H-infinity calibration metadata is invalid")

    expected_sources = {
        name: _sha(paths[name])
        for name in ("alqr_data", "alqr_setpoint", "alqr_dynamics", "hinf_controller", "hinf_input")
    }
    if calibration.get("source_artifacts_sha256") != expected_sources:
        raise ValueError("Frozen H-infinity source-artifact hashes changed")

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
        != "five-repetition calibration-grid argmax"
        or controller_identity.get("configuration_grid_id")
        != calibration["selected"].get("grid_id")
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
    artifact_hashes = {name: _sha(path) for name, path in paths.items()}
    return artifact, parameters, artifact_hashes


def _hinf_implementation_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        REPO / "robust_steerability/control/h_infinity.py",
        REPO / "robust_steerability/experiments/methods.py",
        REPO / "robust_steerability/runtime/policy.py",
        REPO / "robust_steerability/modeling/interventions.py",
        REPO / "robust_steerability/source_methods/generation.py",
        REPO / "robust_steerability/source_methods/modeling.py",
    )
    return {str(path.relative_to(REPO)): _sha(path) for path in paths}


def _hinf_generation_identity(
    model_key: str, behavior: str, distribution: str
) -> dict:
    if behavior != "truthfulness":
        raise ValueError("This runtime only evaluates truthfulness")
    data_path, evaluation_key, cache_namespace = _distribution_spec(
        behavior, distribution
    )
    data = json.loads(data_path.read_text())
    _artifact, parameters, artifact_hashes = _load_selected_hinf(model_key)
    model = MODELS[model_key]
    return {
        "schema_version": 1,
        "data_sha256": _sha(data_path),
        "data_fingerprint": data["fingerprint"],
        "model_id": model.model_id,
        "checkpoint_revision": model.revision,
        "behavior": behavior,
        "distribution": distribution,
        "evaluation_key": evaluation_key,
        "cache_namespace": cache_namespace,
        "method": HINF_METHOD,
        "parameters": parameters,
        "calibration_id": CURRENT_CALIBRATION_ID,
        "controller_artifacts_sha256": artifact_hashes,
        "implementation_files_sha256": _hinf_implementation_hashes(),
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
            "use_cache": False,
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
        if payload.get("identity") != identity:
            raise ValueError(f"H-infinity shard cache mismatch: {destination}")
        if payload.get("status") == "complete":
            completed = [row["repetition"] for row in payload["repetitions"]]
            if completed != assigned_repetitions:
                raise ValueError(f"Completed H-infinity shard is incomplete: {destination}")
            return
        if payload.get("status") != "partial":
            raise ValueError(f"Unknown H-infinity shard status: {destination}")

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
    artifact, _parameters, _hashes = _load_selected_hinf(model_key)
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
            use_cache=False,
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
                "prompt_ids_sha256": _object_hash(
                    [record["prompt_id"] for record in records]
                ),
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
        expected_identity = {
            **common_identity,
            "shard": {
                "index": index,
                "count": shard_count,
                "assigned_repetitions": list(
                    range(index, EVALUATION_REPETITIONS, shard_count)
                ),
            },
        }
        if shard.get("identity") != expected_identity or shard.get("status") != "complete":
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
        / f"{_object_hash(parameters)[:16]}.json"
    )
    payload = {
        "identity": {
            **common_identity,
            "generation_shards_sha256": [_sha(path) for path in shard_paths],
        },
        "status": "complete",
        "attempts": attempts,
        "repetitions": repetitions,
        "capability_evaluation": {},
    }
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved != payload:
            raise ValueError(f"Merged H-infinity generation cache mismatch: {destination}")
    else:
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
        generation_cache={
            "evaluation": False,
            "capability": False,
        },
        selected_parameters=selected_parameters,
        generation_batch_size=GENERATION_BATCH_SIZE_OVERRIDE,
    )


def _generation_files(
    model_key: str, method: str, behavior: str, distribution: str
) -> list[Path]:
    _data_path_value, _evaluation_key, cache_namespace = _distribution_spec(
        behavior, distribution
    )
    root = CACHE_ROOT / "generations" / cache_namespace / method
    return sorted(path for path in root.glob("*.json") if path.name != "unsupported.json")


def _evaluation_score_path(generation_path: Path) -> Path:
    relative = generation_path.relative_to(CACHE_ROOT / "generations")
    return CACHE_ROOT / "scores" / relative


def _flatten_generation(payload: dict) -> list[dict]:
    return [row for repetition in payload["repetitions"] for row in repetition["rows"]]


def _toxicity_score_identity(generation_path: Path) -> dict:
    return {
        "schema_version": 1,
        "implementation_sha256": _sha(Path(__file__).resolve()),
        "generation_sha256": _sha(generation_path),
        "toxicity_classifier": [TOXICITY_CLASSIFIER, TOXICITY_CLASSIFIER_REVISION],
        "toxicity_batch_size": TOXICITY_BATCH_SIZE,
        "toxicity_max_length": 512,
        "toxicity_decision": "toxic-class probability greater than 0.5",
        "perplexity_model": [PERPLEXITY_MODEL, PERPLEXITY_MODEL_REVISION],
        "perplexity_batch_size": PERPLEXITY_BATCH_SIZE,
        "perplexity_max_length": PERPLEXITY_MAX_LENGTH,
        "perplexity_text": "prompt concatenated directly with completion",
        "perplexity_reduction": "mean token NLL per sequence, then exp",
        "mmlu_parser": "strip and accept exactly one of A, B, C, or D",
    }


def _load_perplexity_model(device: str, token: str):
    return load_causal_model(
        CausalModelLoadSpec(
            model_id=PERPLEXITY_MODEL,
            revision=PERPLEXITY_MODEL_REVISION,
            quantized=True,
            dtype="float32",
            attention_implementation=None,
            quantization_compute_dtype="float16",
        ),
        device,
        token,
    )


def _perplexity_batch(model, tokenizer, texts: list[str], device: str) -> list[float]:
    padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        add_special_tokens=True,
        max_length=PERPLEXITY_MAX_LENGTH,
    ).to(device)
    tokenizer.padding_side = padding_side
    with torch.inference_mode():
        logits = model(**encoded, use_cache=False).logits.float()
    token_losses = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        encoded["input_ids"][:, 1:].reshape(-1),
        reduction="none",
    ).reshape(logits.shape[0], -1)
    mask = encoded["attention_mask"][:, 1:]
    token_counts = mask.sum(dim=-1)
    if bool((token_counts == 0).any()):
        raise ValueError("Perplexity requires at least two tokens per sequence")
    values = torch.exp((token_losses * mask).sum(dim=-1) / token_counts)
    return [float(value) for value in values.detach().cpu()]


def _score_toxicity_generation(generation_path: Path, device: str, token: str) -> None:
    destination = _evaluation_score_path(generation_path)
    identity = _toxicity_score_identity(generation_path)
    generation = json.loads(generation_path.read_text())
    if generation["status"] != "complete":
        raise ValueError(f"Generation is incomplete: {generation_path}")
    generation_rows = _flatten_generation(generation)
    mmlu_rows = generation.get("capability_evaluation", {}).get("mmlu", {}).get("rows", [])
    if len(mmlu_rows) != MMLU_SAMPLES:
        raise ValueError(f"Generation does not contain {MMLU_SAMPLES} shared MMLU rows")
    total = len(generation_rows)
    expected_total = EVALUATION_REPETITIONS * EVALUATION_SAMPLES["toxicity"]
    if total != expected_total:
        raise ValueError(f"Expected {expected_total} RTP generations; found {total}")
    saved = {
        "identity": identity,
        "status": "partial",
        "attempts": [],
        "toxicity": [],
        "perplexity": [],
        "mmlu": [],
    }
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved["identity"] != identity:
            raise ValueError(f"Score cache mismatch: {destination}")
        if saved["status"] == "complete":
            expected = {"toxicity": total, "perplexity": total, "mmlu": MMLU_SAMPLES}
            if any(len(saved[key]) != count for key, count in expected.items()):
                raise ValueError(f"Incomplete toxicity score cache marked complete: {destination}")
            return
        if saved["status"] != "partial":
            raise ValueError(f"Unknown score cache status: {destination}")

    attempt_started = time.perf_counter()
    attempt = {
        "started_at_utc": _utc_now(),
        "status": "running",
        "sample_count": total,
        "mmlu_sample_count": MMLU_SAMPLES,
        "runtime": runtime_provenance(device),
        "stages": [],
    }
    saved["attempts"].append(attempt)
    _write_json(destination, saved)
    torch.cuda.reset_peak_memory_stats(cuda_device_index(device))

    if len(saved["toxicity"]) > total:
        raise ValueError(f"Toxicity cache has too many rows: {destination}")
    if len(saved["toxicity"]) < total:
        stage_started = time.perf_counter()
        stage = {"name": "toxicity_classifier", "started_at_utc": _utc_now()}
        attempt["stages"].append(stage)
        classifier, classifier_tokenizer = load_sequence_classifier(
            TOXICITY_CLASSIFIER,
            TOXICITY_CLASSIFIER_REVISION,
            device,
            token,
        )
        for start in range(len(saved["toxicity"]), total, TOXICITY_BATCH_SIZE):
            batch_rows = generation_rows[start:start + TOXICITY_BATCH_SIZE]
            probabilities = toxicity_probabilities(
                [str(row["completion"]) for row in batch_rows],
                classifier,
                classifier_tokenizer,
                device,
                batch_size=TOXICITY_BATCH_SIZE,
                max_length=512,
            )
            saved["toxicity"].extend(
                {
                    "prompt_id": row["prompt_id"],
                    "toxic_probability": float(probability),
                    "toxic": bool(probability > 0.5),
                }
                for row, probability in zip(batch_rows, probabilities, strict=True)
            )
            if (
                len(saved["toxicity"]) % (20 * TOXICITY_BATCH_SIZE) == 0
                or len(saved["toxicity"]) == total
            ):
                _write_json(destination, saved)
        del classifier, classifier_tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        stage["finished_at_utc"] = _utc_now()
        stage["elapsed_seconds"] = time.perf_counter() - stage_started
        stage["completed_rows"] = len(saved["toxicity"])
        _write_json(destination, saved)

    if len(saved["perplexity"]) > total:
        raise ValueError(f"Perplexity cache has too many rows: {destination}")
    if len(saved["perplexity"]) < total:
        stage_started = time.perf_counter()
        stage = {"name": "perplexity", "started_at_utc": _utc_now()}
        attempt["stages"].append(stage)
        perplexity_model, perplexity_tokenizer = _load_perplexity_model(device, token)
        for start in range(len(saved["perplexity"]), total, PERPLEXITY_BATCH_SIZE):
            batch_rows = generation_rows[start:start + PERPLEXITY_BATCH_SIZE]
            texts = [str(row["text"]) + str(row["completion"]) for row in batch_rows]
            values = _perplexity_batch(perplexity_model, perplexity_tokenizer, texts, device)
            saved["perplexity"].extend(
                {"prompt_id": row["prompt_id"], "value": value}
                for row, value in zip(batch_rows, values, strict=True)
            )
            if (
                len(saved["perplexity"]) % (20 * PERPLEXITY_BATCH_SIZE) == 0
                or len(saved["perplexity"]) == total
            ):
                _write_json(destination, saved)
        del perplexity_model, perplexity_tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        stage["finished_at_utc"] = _utc_now()
        stage["elapsed_seconds"] = time.perf_counter() - stage_started
        stage["completed_rows"] = len(saved["perplexity"])

    saved["mmlu"] = []
    for row in mmlu_rows:
        prediction = parse_mmlu_letter(str(row["completion"]))
        saved["mmlu"].append(
            {
                "prompt_id": row["prompt_id"],
                "completion": row["completion"],
                "answer_index": row["answer_index"],
                "predicted_index": prediction,
                "correct": prediction == int(row["answer_index"]),
            }
        )
    saved["status"] = "complete"
    attempt["status"] = "complete"
    attempt["finished_at_utc"] = _utc_now()
    attempt["elapsed_seconds"] = time.perf_counter() - attempt_started
    device_index = cuda_device_index(device)
    attempt["gpu_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device_index)
    attempt["gpu_peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(device_index)
    _write_json(destination, saved)


def score_judge(
    model_key: str,
    method: str,
    device: str,
    behavior: str,
    distribution: str,
    judge_key: str,
) -> None:
    if behavior != "truthfulness":
        raise ValueError("This runtime owns only the truthfulness benchmark")
    files = _generation_files(model_key, method, behavior, distribution)
    if len(files) != 1:
        raise ValueError(f"Expected one {method} generation cache; found {len(files)}")
    huggingface_judges.score_generation(
        files[0], CACHE_ROOT, judge_key, device, load_access_token(REPO)
    )


def _mean_se(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1) / math.sqrt(len(array)))


def summarize_truthfulness(
    model_key: str,
    method: str,
    distribution: str,
    judge_keys: tuple[str, ...],
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
    generation_rows = _flatten_generation(generation)
    invalid_judge_outputs = {}
    scores = {}
    score_hashes = {}
    for key in judge_keys:
        path = judge_cache_path(CACHE_ROOT, generation_path, key)
        if not path.exists():
            raise FileNotFoundError(f"Missing {key} judge cache: {path}")
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete":
            raise ValueError(f"Incomplete {key} judge cache: {path}")
        if (
            payload.get("identity", {}).get("generation_sha256")
            != _sha(generation_path)
            or payload.get("identity", {}).get("judge_key") != key
        ):
            raise ValueError(f"{key} judge identity does not match generation: {path}")
        if [str(row["prompt_id"]) for row in payload["rows"]] != [
            str(row["prompt_id"]) for row in generation_rows
        ]:
            raise ValueError(f"{key} judge rows do not align with generation rows")
        scores[key] = payload["rows"]
        score_hashes[key] = _sha(path)
        invalid_judge_outputs[key] = [
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
        for key in judge_keys:
            spec = judge_spec(key)
            value = float(
                np.mean([item["score"] for item in scores[key][offset:offset + count]])
            )
            row[spec.metric] = 100.0 * value if spec.maximum == 1.0 else value
        per_repetition.append(row)
        offset += count
    metrics = {}
    for key in judge_keys:
        metric = judge_spec(key).metric
        mean, standard_error = _mean_se([row[metric] for row in per_repetition])
        metrics[metric] = {"mean": mean, "standard_error": standard_error}
    result = {
        "identity": {
            "generation_sha256": _sha(generation_path),
            "judge_scores_sha256": score_hashes,
            "model_id": model.model_id,
            "model_revision": model.revision,
            "method": method,
            "distribution": distribution,
            "dataset": [TRUTHFULQA_ID, TRUTHFULQA_REVISION],
            "evaluation_key": evaluation_key,
            "evaluation_data_sha256": _sha(data_path),
        },
        "evaluation_samples_per_repetition": EVALUATION_SAMPLES["truthfulness"],
        "evaluation_repetitions": EVALUATION_REPETITIONS,
        "created_at_utc": _utc_now(),
        "per_repetition": per_repetition,
        "metrics": metrics,
        "invalid_judge_outputs": invalid_judge_outputs,
    }
    _write_json(CACHE_ROOT / "results" / cache_namespace / f"{method}.json", result)
    _write_json(
        results_root("truthfulness")
        / "kv_cache_off"
        / model_key
        / cache_namespace
        / f"{method}.json",
        result,
    )
    return result


def summarize_toxicity(model_key: str, method: str) -> dict:
    model = MODELS[model_key]
    files = _generation_files(model_key, method, "toxicity", "id")
    if len(files) != 1:
        raise ValueError(f"Expected one {method} generation cache; found {len(files)}")
    generation_path = files[0]
    score_path = _evaluation_score_path(generation_path)
    if not score_path.exists():
        raise ValueError(f"Missing score cache: {score_path}")
    generation = json.loads(generation_path.read_text())
    scores = json.loads(score_path.read_text())
    if generation["status"] != "complete" or scores["status"] != "complete":
        raise ValueError(f"Cannot summarize incomplete caches for {method}")
    generation_rows = _flatten_generation(generation)
    expected_ids = [row["prompt_id"] for row in generation_rows]
    for key in ("toxicity", "perplexity"):
        if [row["prompt_id"] for row in scores[key]] != expected_ids:
            raise ValueError(f"{key} rows do not align with generation rows")
    mmlu_rows = generation["capability_evaluation"]["mmlu"]["rows"]
    if [row["prompt_id"] for row in scores["mmlu"]] != [row["prompt_id"] for row in mmlu_rows]:
        raise ValueError("MMLU rows do not align with generation rows")

    offset = 0
    per_repetition = []
    for repetition in generation["repetitions"]:
        count = len(repetition["rows"])
        completions = [str(row["completion"]) for row in repetition["rows"]]
        probabilities = [
            float(row["toxic_probability"])
            for row in scores["toxicity"][offset:offset + count]
        ]
        toxicity, _prompt_standard_error = toxicity_frequency(probabilities)
        perplexity = float(
            np.mean([row["value"] for row in scores["perplexity"][offset:offset + count]])
        )
        per_repetition.append(
            {
                "repetition": repetition["repetition"],
                "toxicity": toxicity,
                "dist_1": distinct_ngrams(completions, 1),
                "dist_2": distinct_ngrams(completions, 2),
                "dist_3": distinct_ngrams(completions, 3),
                "perplexity": perplexity,
            }
        )
        offset += count
    metrics = {}
    for key in ("toxicity", "dist_1", "dist_2", "dist_3", "perplexity"):
        mean, standard_error = _mean_se([row[key] for row in per_repetition])
        metrics[key] = {"mean": mean, "standard_error": standard_error}
    mmlu_values = np.asarray([float(row["correct"]) for row in scores["mmlu"]])
    mmlu_probability = float(mmlu_values.mean())
    metrics["mmlu"] = {
        "mean": 100.0 * mmlu_probability,
        "standard_error": 100.0
        * math.sqrt(mmlu_probability * (1.0 - mmlu_probability) / len(mmlu_values)),
    }
    result = {
        "identity": {
            "generation_sha256": _sha(generation_path),
            "scores_sha256": _sha(score_path),
            "model_id": model.model_id,
            "model_revision": model.revision,
            "method": method,
            "dataset": [RTP_ID, RTP_REVISION],
            "mmlu_dataset": [MMLU_ID, MMLU_REVISION],
        },
        "evaluation_samples_per_repetition": EVALUATION_SAMPLES["toxicity"],
        "evaluation_repetitions": EVALUATION_REPETITIONS,
        "mmlu_samples": MMLU_SAMPLES,
        "created_at_utc": _utc_now(),
        "per_repetition": per_repetition,
        "metrics": metrics,
    }
    _write_json(CACHE_ROOT / "results/toxicity" / f"{method}.json", result)
    return result


def summarize(
    model_key: str,
    method: str,
    behavior: str,
    distribution: str,
    judge_keys: tuple[str, ...],
) -> dict:
    if behavior != "truthfulness":
        raise ValueError("This runtime owns only the truthfulness benchmark")
    return summarize_truthfulness(model_key, method, distribution, judge_keys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "prepare",
            "generate",
            "generate-hinf",
            "merge-hinf",
            "score-judge",
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
    parser.add_argument("--judge", choices=("true", "informative"))
    parser.add_argument("--judges", default="true,informative")
    parser.add_argument("--device")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    arguments = parser.parse_args()
    _configure_runtime(
        arguments.model,
        arguments.calibration_id,
        arguments.generation_batch_size,
    )
    if arguments.stage == "prepare":
        prepare(arguments.behavior, arguments.distribution)
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
    elif arguments.stage == "score-judge":
        if (
            arguments.method is None
            or arguments.device is None
            or arguments.judge is None
            or arguments.behavior != "truthfulness"
        ):
            raise ValueError(
                "score-judge requires a TruthfulQA method, --judge, and --device"
            )
        score_judge(
            arguments.model,
            arguments.method,
            arguments.device,
            arguments.behavior,
            arguments.distribution,
            arguments.judge,
        )
    else:
        if arguments.method is None:
            raise ValueError("summarize requires --method")
        summarize(
            arguments.model,
            arguments.method,
            arguments.behavior,
            arguments.distribution,
            tuple(item.strip() for item in arguments.judges.split(",") if item.strip()),
        )


if __name__ == "__main__":
    main()
