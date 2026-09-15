"""Prepare frozen source-faithful A-LQR artifacts by model and behavior."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset

from robust_steerability.datasets.toxicity import load_real_toxicity_prompt_pools
from robust_steerability.benchmarks.layout import artifact_root
from robust_steerability.benchmarks.specs import MODELS
from robust_steerability.calibration.nominal import average_prompt_jacobians
from robust_steerability.calibration.nominal_artifact import (
    nominal_dynamics_identity,
    save_nominal_dynamics,
)
from robust_steerability.modeling.huggingface import cuda_device_index, load_access_token
from robust_steerability.experiments.resources import resolve_cuda_devices
from robust_steerability.source_methods.calibration import fit_setpoint_from_records
from robust_steerability.source_methods.id_benchmark import runtime_provenance
from robust_steerability.source_methods.modeling import load_source_model, source_model_spec
from robust_steerability.source_methods.protocol import (
    ALQR_CALIBRATION_COUNTS,
    SOURCE_RANDOM_SEED,
    paper_alqr_setting,
)


REPO = Path(__file__).resolve().parents[2]
DATASETS = {
    "truthfulness": {
        "id": "truthful_qa",
        "revision": "741b8276f2d1982aa3d5b832d3ee81ed3b896490",
        "configuration": "multiple_choice",
        "split": "validation",
    },
    "toxicity": {
        "id": "allenai/real-toxicity-prompts",
        "revision": "f21629712ffd6a3d13a54fd2807ccd521c55ef74",
        "configuration": None,
        "split": "train",
    },
}


def _cache(behavior: str, model_key: str) -> Path:
    return artifact_root(behavior, model_key)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _save_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _sample(records: list[dict], count: int, seed: int) -> list[dict]:
    if len(records) < count:
        raise ValueError(f"Requested {count} records from a pool of {len(records)}")
    indices = random.Random(seed).sample(range(len(records)), count)
    return [records[index] for index in indices]


def build_truthfulness_calibration_data(rows: list[dict]) -> dict:
    """Select the frozen false, true, and independent true-Jacobian prompts."""

    false_records = []
    true_records = []
    for row_index, row in enumerate(rows):
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

    behavior = "truthfulness"
    counts = ALQR_CALIBRATION_COUNTS[behavior]
    dataset = DATASETS[behavior]
    payload = {
        "schema_version": 1,
        "behavior": behavior,
        "dataset": dataset,
        "seed": SOURCE_RANDOM_SEED,
        "selection": {
            "undesired": {"class": "false MC2 answers", "count": counts.undesired},
            "desired": {"class": "true MC2 answers", "count": counts.desired},
            "jacobian": {
                "class": "true MC2 answers",
                "count": counts.jacobian,
                "independent_sample": True,
                "max_length": counts.jacobian_max_length,
            },
            "seeds": {
                "undesired": SOURCE_RANDOM_SEED,
                "desired": SOURCE_RANDOM_SEED + 1,
                "jacobian": SOURCE_RANDOM_SEED + 2,
            },
        },
        "calibration": {
            "undesired": _sample(false_records, counts.undesired, SOURCE_RANDOM_SEED),
            "desired": _sample(true_records, counts.desired, SOURCE_RANDOM_SEED + 1),
            "jacobian": _sample(true_records, counts.jacobian, SOURCE_RANDOM_SEED + 2),
        },
        "source_counts": {"false_answers": len(false_records), "true_answers": len(true_records)},
    }
    return payload


def build_toxicity_calibration_data(
    toxic_records: list[dict], nontoxic_records: list[dict]
) -> dict:
    """Select the frozen RTP toxic, non-toxic, and non-toxic Jacobian prompts."""

    behavior = "toxicity"
    counts = ALQR_CALIBRATION_COUNTS[behavior]
    payload = {
        "schema_version": 1,
        "behavior": behavior,
        "dataset": DATASETS[behavior],
        "seed": SOURCE_RANDOM_SEED,
        "selection": {
            "undesired": {"class": "RTP prompt toxicity in [0.8, 1.0]", "count": counts.undesired},
            "desired": {"class": "RTP prompt toxicity in [0.0, 0.1]", "count": counts.desired},
            "jacobian": {
                "class": "RTP prompt toxicity in [0.0, 0.1]",
                "count": counts.jacobian,
                "independent_sample": True,
                "max_length": counts.jacobian_max_length,
            },
            "seeds": {
                "undesired": SOURCE_RANDOM_SEED,
                "desired": SOURCE_RANDOM_SEED + 1,
                "jacobian": SOURCE_RANDOM_SEED + 2,
            },
        },
        "calibration": {
            "undesired": _sample(toxic_records, counts.undesired, SOURCE_RANDOM_SEED),
            "desired": _sample(nontoxic_records, counts.desired, SOURCE_RANDOM_SEED + 1),
            "jacobian": _sample(nontoxic_records, counts.jacobian, SOURCE_RANDOM_SEED + 2),
        },
        "source_counts": {
            "toxic_prompts": len(toxic_records),
            "nontoxic_prompts": len(nontoxic_records),
        },
    }
    return payload


def _update_timings(behavior: str, model_key: str, stage: str, timing: dict) -> None:
    destination = _cache(behavior, model_key) / "timings.json"
    payload = json.loads(destination.read_text()) if destination.exists() else {"schema_version": 1}
    payload[stage] = timing
    _write_json(destination, payload)


def prepare(model_key: str, behavior: str) -> dict:
    destination = _cache(behavior, model_key) / "data.json"
    if destination.exists():
        return json.loads(destination.read_text())

    started_at = _utc_now()
    started = time.perf_counter()
    dataset = DATASETS[behavior]
    if behavior == "truthfulness":
        rows = list(load_dataset(
            dataset["id"],
            "multiple_choice",
            split="validation",
            revision=dataset["revision"],
        ))
        data = build_truthfulness_calibration_data(rows)
        source_rows = len(rows)
    else:
        _all_records, toxic_records, nontoxic_records = load_real_toxicity_prompt_pools(
            dataset["id"], dataset["revision"]
        )
        data = build_toxicity_calibration_data(toxic_records, nontoxic_records)
        source_rows = len(toxic_records) + len(nontoxic_records)
    _write_json(destination, data)
    _update_timings(
        behavior,
        model_key,
        "prepare",
        {
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "source_rows": source_rows,
            "runtime": runtime_provenance("cpu"),
        },
    )
    return data


def calibration_identity(data: dict, model_key: str, behavior: str) -> dict:
    model = MODELS[model_key]
    return {
        "schema_version": 1,
        "behavior": behavior,
        "model": {"id": model.model_id, "revision": model.revision},
        "model_loading": asdict(
            source_model_spec("alqr", behavior, model.model_id, model.revision)
        ),
        "counts": asdict(ALQR_CALIBRATION_COUNTS[behavior]),
        "alqr_setting": asdict(paper_alqr_setting(behavior, model.model_id)),
        "activation_batch_size": model.activation_batch_size,
        "jacobian_vjp_chunk_size": model.jacobian_vjp_chunk_size,
    }


def _load_run(path: Path, identity: dict) -> dict:
    if not path.exists():
        return {"identity": identity, "status": "partial", "attempts": []}
    return json.loads(path.read_text())


def fit_setpoint(model_key: str, behavior: str, device: str) -> None:
    model_config = MODELS[model_key]
    cache = _cache(behavior, model_key)
    data = prepare(model_key, behavior)
    identity = calibration_identity(data, model_key, behavior)
    run_path = cache / "runs/setpoint.json"
    artifact_path = cache / "setpoint.pt"
    if artifact_path.exists():
        return
    run = _load_run(run_path, identity)
    if run["status"] == "complete":
        if not artifact_path.exists():
            raise ValueError("Setpoint run is complete but setpoint.pt is missing")
        return

    attempt = {
        "started_at_utc": _utc_now(),
        "status": "running",
        "runtime": runtime_provenance(device),
    }
    run["attempts"].append(attempt)
    _write_json(run_path, run)
    device_index = cuda_device_index(device)
    torch.cuda.reset_peak_memory_stats(device_index)
    stage_started = time.perf_counter()
    model_started = time.perf_counter()
    model, tokenizer = load_source_model(
        "alqr",
        behavior,
        model_config.model_id,
        model_config.revision,
        device,
        load_access_token(REPO),
    )
    attempt["model_load_elapsed_seconds"] = time.perf_counter() - model_started
    fit_started = time.perf_counter()
    calibration = fit_setpoint_from_records(
        model,
        tokenizer,
        behavior=behavior,
        negative_records=data["calibration"]["undesired"],
        positive_records=data["calibration"]["desired"],
        activation_batch_size=model_config.activation_batch_size,
    )
    attempt["fit_elapsed_seconds"] = time.perf_counter() - fit_started
    _save_torch(
        artifact_path,
        {
            "identity": identity,
            "contrast": calibration.contrast.cpu(),
            "feature_norm": calibration.feature_norm.cpu(),
        },
    )
    attempt["finished_at_utc"] = _utc_now()
    attempt["elapsed_seconds"] = time.perf_counter() - stage_started
    attempt["gpu_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device_index)
    attempt["gpu_peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(device_index)
    attempt["status"] = "complete"
    run["status"] = "complete"
    _write_json(run_path, run)
    _update_timings(behavior, model_key, "setpoint", attempt)


def _partition_records(records: list[dict], shard_index: int, shard_count: int) -> list[dict]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid Jacobian shard")
    return records[shard_index::shard_count]


def fit_jacobian_shard(
    model_key: str, behavior: str, device: str, shard_index: int, shard_count: int
) -> None:
    model_config = MODELS[model_key]
    cache = _cache(behavior, model_key)
    data = prepare(model_key, behavior)
    identity = {
        **calibration_identity(data, model_key, behavior),
        "shard_index": shard_index,
        "shard_count": shard_count,
    }
    records = _partition_records(data["calibration"]["jacobian"], shard_index, shard_count)
    run_path = cache / "runs" / f"jacobian_shard_{shard_index:02d}.json"
    cache_dir = cache / "jacobian_partials" / f"shard_{shard_index:02d}"
    run = _load_run(run_path, identity)
    if run["status"] == "complete":
        partial = torch.load(cache_dir / "partial.pt", map_location="cpu", weights_only=True)
        if int(partial["count"]) != len(records):
            raise ValueError(f"Completed Jacobian shard {shard_index} is incomplete")
        return

    attempt = {
        "started_at_utc": _utc_now(),
        "status": "running",
        "runtime": runtime_provenance(device),
        "prompt_ids": [record["prompt_id"] for record in records],
    }
    run["attempts"].append(attempt)
    _write_json(run_path, run)
    device_index = cuda_device_index(device)
    torch.cuda.reset_peak_memory_stats(device_index)
    stage_started = time.perf_counter()
    model_started = time.perf_counter()
    model, tokenizer = load_source_model(
        "alqr",
        behavior,
        model_config.model_id,
        model_config.revision,
        device,
        load_access_token(REPO),
    )
    attempt["model_load_elapsed_seconds"] = time.perf_counter() - model_started
    jacobian_started = time.perf_counter()
    shard_mean = average_prompt_jacobians(
        model,
        tokenizer,
        records,
        cache_dir=cache_dir,
        max_length=ALQR_CALIBRATION_COUNTS[behavior].jacobian_max_length,
        vjp_chunk_size=model_config.jacobian_vjp_chunk_size,
        model_revision=model_config.revision,
    )
    attempt["jacobian_elapsed_seconds"] = time.perf_counter() - jacobian_started
    attempt["finished_at_utc"] = _utc_now()
    attempt["elapsed_seconds"] = time.perf_counter() - stage_started
    attempt["gpu_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device_index)
    attempt["gpu_peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(device_index)
    attempt["status"] = "complete"
    run["status"] = "complete"
    run["record_count"] = len(records)
    run["layer_count"] = int(shard_mean.shape[0])
    run["hidden_size"] = int(shard_mean.shape[-1])
    _write_json(run_path, run)


def aggregate_jacobian_partials(
    shard_directories: list[Path], expected_counts: list[int]
) -> torch.Tensor:
    """Combine worker sums into the exact float64 prompt-weighted mean."""

    total = None
    count = 0
    expected_shape = None
    for shard_directory, expected_count in zip(
        shard_directories, expected_counts, strict=True
    ):
        path = shard_directory / "partial.pt"
        if not path.exists():
            raise ValueError(f"Missing Jacobian partial: {path}")
        partial = torch.load(path, map_location="cpu", weights_only=True)
        shard_count = int(partial["count"])
        shard_sum = partial["sum"]
        if shard_count != expected_count or shard_sum.dtype != torch.float64:
            raise ValueError(f"Invalid Jacobian partial: {path}")
        if expected_shape is None:
            expected_shape = shard_sum.shape
            total = torch.zeros(expected_shape, dtype=torch.float64)
        if shard_sum.shape != expected_shape or not torch.isfinite(shard_sum).all():
            raise ValueError(f"Invalid Jacobian partial tensor: {path}")
        total.add_(shard_sum)
        count += shard_count
    if total is None or count != sum(expected_counts):
        raise ValueError("Jacobian partial counts do not match the requested prompts")
    return (total / count).float()


def fit_jacobians(
    model_key: str,
    behavior: str,
    devices: list[str],
    *,
    log_root: Path | None = None,
) -> None:
    if len(devices) != len(set(devices)) or not devices:
        raise ValueError("Jacobian devices must be a nonempty list of distinct CUDA devices")
    model_config = MODELS[model_key]
    cache = _cache(behavior, model_key)
    data = prepare(model_key, behavior)
    identity = calibration_identity(data, model_key, behavior)
    run_path = cache / "runs/jacobians.json"
    artifact_path = cache / "dynamics.pt"
    if artifact_path.exists():
        return
    run = _load_run(run_path, identity)
    if run["status"] == "complete":
        return

    attempt = {
        "started_at_utc": _utc_now(),
        "status": "running",
        "runtime": runtime_provenance("cpu"),
        "devices": devices,
        "shard_count": len(devices),
    }
    run["attempts"].append(attempt)
    _write_json(run_path, run)
    started = time.perf_counter()
    processes = []
    worker_log_root = log_root or cache / "logs"
    for shard_index, device in enumerate(devices):
        log_path = worker_log_root / f"jacobian_shard_{shard_index:02d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a")
        command = [
            sys.executable,
            "-m",
            "robust_steerability.benchmarks.artifacts",
            "--stage",
            "jacobian-shard",
            "--model",
            model_key,
            "--behavior",
            behavior,
            "--device",
            device,
            "--shard-index",
            str(shard_index),
            "--shard-count",
            str(len(devices)),
        ]
        process = subprocess.Popen(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
        processes.append((shard_index, device, process, handle, log_path))

    failures = []
    for shard_index, device, process, handle, log_path in processes:
        process.wait()
        handle.close()
        if process.returncode != 0:
            failures.append(
                {
                    "shard_index": shard_index,
                    "device": device,
                    "returncode": process.returncode,
                    "log": str(log_path),
                }
            )
    if failures:
        attempt["status"] = "failed"
        attempt["failures"] = failures
        attempt["finished_at_utc"] = _utc_now()
        attempt["elapsed_seconds"] = time.perf_counter() - started
        _write_json(run_path, run)
        raise RuntimeError(f"Jacobian workers failed: {failures}")

    records = data["calibration"]["jacobian"]
    expected_counts = [
        len(_partition_records(records, shard_index, len(devices)))
        for shard_index in range(len(devices))
    ]
    aggregation_started = time.perf_counter()
    partial_directories = [
        cache / "jacobian_partials" / f"shard_{index:02d}"
        for index in range(len(devices))
    ]
    dynamics = aggregate_jacobian_partials(
        partial_directories,
        expected_counts,
    )
    attempt["aggregation_elapsed_seconds"] = time.perf_counter() - aggregation_started
    shard_runs = [
        json.loads((cache / "runs" / f"jacobian_shard_{index:02d}.json").read_text())
        for index in range(len(devices))
    ]
    attempt["finished_at_utc"] = _utc_now()
    attempt["elapsed_seconds"] = time.perf_counter() - started
    attempt["matrix_shape"] = list(dynamics.shape)
    attempt["status"] = "complete"
    shared_identity = nominal_dynamics_identity(
        behavior=behavior,
        model_id=model_config.model_id,
        model_revision=model_config.revision,
        records=records,
        max_length=int(ALQR_CALIBRATION_COUNTS[behavior].jacobian_max_length),
        vjp_chunk_size=model_config.jacobian_vjp_chunk_size,
    )
    save_nominal_dynamics(
        artifact_path,
        shared_identity,
        dynamics,
        attempts=[
            {
                "status": "complete",
                "created_at_utc": _utc_now(),
                "operation": "weighted mean of worker float64 accumulators",
            }
        ],
    )
    run["status"] = "complete"
    run["shards"] = [
        {
            "index": index,
            "device": devices[index],
            "record_count": shard["record_count"],
            "latest_attempt": shard["attempts"][-1],
        }
        for index, shard in enumerate(shard_runs)
    ]
    _write_json(run_path, run)
    for directory in partial_directories:
        if directory.parent != cache / "jacobian_partials":
            raise RuntimeError(f"Refusing to remove unexpected path: {directory}")
        shutil.rmtree(directory)
    attempt["temporary_partials_removed"] = True
    _write_json(run_path, run)
    _update_timings(behavior, model_key, "jacobians", {**attempt, "shards": run["shards"]})


def write_manifest(model_key: str, behavior: str) -> None:
    cache = _cache(behavior, model_key)
    data = prepare(model_key, behavior)
    identity = calibration_identity(data, model_key, behavior)
    setpoint_run = cache / "runs/setpoint.json"
    jacobian_run = cache / "runs/jacobians.json"
    status = {
        "setpoint": "complete"
        if setpoint_run.exists() and json.loads(setpoint_run.read_text()).get("status") == "complete"
        else "missing",
        "jacobians": "complete"
        if jacobian_run.exists() and json.loads(jacobian_run.read_text()).get("status") == "complete"
        else "missing",
        "full_benchmark": "not_run",
    }
    _write_json(
        cache / "manifest.json",
        {
            "identity": identity,
            "status": status,
            "artifacts": {
                "data": "data.json",
                "setpoint": "setpoint.pt",
                "temporary_jacobian_partials": "jacobian_partials/",
                "averaged_dynamics": "dynamics.pt",
                "timings": "timings.json",
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("prepare", "setpoint", "jacobian-shard", "jacobians", "all", "status"),
        required=True,
    )
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--behavior", choices=tuple(DATASETS), required=True)
    parser.add_argument("--device")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    arguments = parser.parse_args()
    devices = (
        resolve_cuda_devices(arguments.devices)
        if arguments.stage in {"jacobians", "all"}
        else []
    )
    paper_alqr_setting(arguments.behavior, MODELS[arguments.model].model_id)

    if arguments.stage == "prepare":
        prepare(arguments.model, arguments.behavior)
    elif arguments.stage == "setpoint":
        if arguments.device is None:
            raise ValueError("setpoint requires --device")
        fit_setpoint(arguments.model, arguments.behavior, arguments.device)
    elif arguments.stage == "jacobian-shard":
        if arguments.device is None or arguments.shard_index is None or arguments.shard_count is None:
            raise ValueError("jacobian-shard requires --device, --shard-index, and --shard-count")
        fit_jacobian_shard(
            arguments.model,
            arguments.behavior,
            arguments.device,
            arguments.shard_index,
            arguments.shard_count,
        )
    elif arguments.stage == "jacobians":
        fit_jacobians(arguments.model, arguments.behavior, devices)
    elif arguments.stage == "all":
        started_at = _utc_now()
        started = time.perf_counter()
        prepare(arguments.model, arguments.behavior)
        fit_setpoint(arguments.model, arguments.behavior, devices[0])
        fit_jacobians(arguments.model, arguments.behavior, devices)
        write_manifest(arguments.model, arguments.behavior)
        timings_path = _cache(arguments.behavior, arguments.model) / "timings.json"
        timings = json.loads(timings_path.read_text())
        if "calibration_total" not in timings:
            _update_timings(
                arguments.behavior,
                arguments.model,
                "calibration_total",
                {
                    "started_at_utc": started_at,
                    "finished_at_utc": _utc_now(),
                    "elapsed_seconds": time.perf_counter() - started,
                    "full_benchmark": "not_run",
                },
            )
    else:
        write_manifest(arguments.model, arguments.behavior)
        print(
            (_cache(arguments.behavior, arguments.model) / "manifest.json").read_text()
        )


if __name__ == "__main__":
    main()
