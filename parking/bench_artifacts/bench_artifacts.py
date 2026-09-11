"""Prepare the frozen Gemma-2-2B TruthfulQA A-LQR calibration artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset

from robust_steerability.calibration.nominal import average_prompt_jacobians
from robust_steerability.calibration.nominal_artifact import (
    load_nominal_dynamics,
    nominal_dynamics_identity,
    save_nominal_dynamics,
)
from robust_steerability.modeling.huggingface import cuda_device_index, load_access_token
from robust_steerability.source_methods.calibration import fit_setpoint_from_records
from robust_steerability.source_methods.id_benchmark import runtime_provenance
from robust_steerability.source_methods.modeling import load_source_model, source_model_spec
from robust_steerability.source_methods.protocol import (
    ALQR_CALIBRATION_COUNTS,
    SOURCE_RANDOM_SEED,
    paper_alqr_setting,
)


UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[1]
CACHE = UNIT / "cache"
MODEL_ID = "google/gemma-2-2b"
MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
DATASET_ID = "truthful_qa"
DATASET_REVISION = "741b8276f2d1982aa3d5b832d3ee81ed3b896490"
BEHAVIOR = "truthfulness"
ACTIVATION_BATCH_SIZE = 16
JACOBIAN_VJP_CHUNK_SIZE = 32


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


def _hash_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
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


def build_calibration_data(rows: list[dict]) -> dict:
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

    counts = ALQR_CALIBRATION_COUNTS[BEHAVIOR]
    payload = {
        "schema_version": 1,
        "behavior": BEHAVIOR,
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "configuration": "multiple_choice",
            "split": "validation",
        },
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
    payload["fingerprint"] = _hash_json(payload)
    return payload


def _validate_data(data: dict) -> None:
    counts = ALQR_CALIBRATION_COUNTS[BEHAVIOR]
    scientific = {key: value for key, value in data.items() if key != "fingerprint"}
    if data.get("schema_version") != 1 or data.get("behavior") != BEHAVIOR:
        raise ValueError("Calibration data schema does not match this unit")
    if data.get("fingerprint") != _hash_json(scientific):
        raise ValueError("Calibration data fingerprint mismatch")
    calibration = data["calibration"]
    expected = {
        "undesired": counts.undesired,
        "desired": counts.desired,
        "jacobian": counts.jacobian,
    }
    actual = {key: len(calibration[key]) for key in expected}
    if actual != expected:
        raise ValueError(f"Calibration counts changed: expected {expected}, found {actual}")


def _update_timings(stage: str, timing: dict) -> None:
    destination = CACHE / "timings.json"
    payload = json.loads(destination.read_text()) if destination.exists() else {"schema_version": 1}
    payload[stage] = timing
    _write_json(destination, payload)


def prepare() -> dict:
    destination = CACHE / "data.json"
    if destination.exists():
        data = json.loads(destination.read_text())
        _validate_data(data)
        return data

    started_at = _utc_now()
    started = time.perf_counter()
    rows = list(
        load_dataset(
            DATASET_ID,
            "multiple_choice",
            split="validation",
            revision=DATASET_REVISION,
        )
    )
    data = build_calibration_data(rows)
    _write_json(destination, data)
    _update_timings(
        "prepare",
        {
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "source_rows": len(rows),
            "runtime": runtime_provenance("cpu"),
        },
    )
    return data


def _implementation_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        REPO / "robust_steerability/calibration/nominal.py",
        REPO / "robust_steerability/calibration/nominal_artifact.py",
        REPO / "robust_steerability/modeling/jacobians.py",
        REPO / "robust_steerability/source_methods/calibration.py",
        REPO / "robust_steerability/source_methods/modeling.py",
        REPO / "robust_steerability/source_methods/protocol.py",
    )
    return {str(path.relative_to(REPO)): _sha256(path) for path in paths}


def calibration_identity(data: dict) -> dict:
    return {
        "schema_version": 1,
        "behavior": BEHAVIOR,
        "data_fingerprint": data["fingerprint"],
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "model_loading": asdict(source_model_spec("alqr", BEHAVIOR, MODEL_ID, MODEL_REVISION)),
        "counts": asdict(ALQR_CALIBRATION_COUNTS[BEHAVIOR]),
        "alqr_setting": asdict(paper_alqr_setting(BEHAVIOR, MODEL_ID)),
        "activation_batch_size": ACTIVATION_BATCH_SIZE,
        "jacobian_vjp_chunk_size": JACOBIAN_VJP_CHUNK_SIZE,
        "implementation_sha256": _implementation_hashes(),
    }


def _load_run(path: Path, identity: dict) -> dict:
    if not path.exists():
        return {"identity": identity, "status": "partial", "attempts": []}
    payload = json.loads(path.read_text())
    if payload.get("identity") != identity:
        raise ValueError(f"Run identity mismatch: {path}")
    if payload.get("status") not in {"partial", "complete"}:
        raise ValueError(f"Invalid run status: {path}")
    return payload


def fit_setpoint(device: str) -> None:
    data = prepare()
    identity = calibration_identity(data)
    run_path = CACHE / "runs/setpoint.json"
    artifact_path = CACHE / "setpoint.pt"
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
        "alqr", BEHAVIOR, MODEL_ID, MODEL_REVISION, device, load_access_token(REPO)
    )
    attempt["model_load_elapsed_seconds"] = time.perf_counter() - model_started
    fit_started = time.perf_counter()
    calibration = fit_setpoint_from_records(
        model,
        tokenizer,
        behavior=BEHAVIOR,
        negative_records=data["calibration"]["undesired"],
        positive_records=data["calibration"]["desired"],
        activation_batch_size=ACTIVATION_BATCH_SIZE,
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
    attempt["artifact_sha256"] = _sha256(artifact_path)
    attempt["status"] = "complete"
    run["status"] = "complete"
    _write_json(run_path, run)
    _update_timings("setpoint", attempt)


def _partition_records(records: list[dict], shard_index: int, shard_count: int) -> list[dict]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid Jacobian shard")
    return records[shard_index::shard_count]


def fit_jacobian_shard(device: str, shard_index: int, shard_count: int) -> None:
    data = prepare()
    identity = {
        **calibration_identity(data),
        "shard_index": shard_index,
        "shard_count": shard_count,
    }
    records = _partition_records(data["calibration"]["jacobian"], shard_index, shard_count)
    run_path = CACHE / "runs" / f"jacobian_shard_{shard_index:02d}.json"
    cache_dir = CACHE / "jacobians" / f"shard_{shard_index:02d}"
    run = _load_run(run_path, identity)
    if run["status"] == "complete":
        expected_layers = int(run["layer_count"])
        if len(list(cache_dir.glob("*/layer_*.pt"))) != len(records) * expected_layers:
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
        "alqr", BEHAVIOR, MODEL_ID, MODEL_REVISION, device, load_access_token(REPO)
    )
    attempt["model_load_elapsed_seconds"] = time.perf_counter() - model_started
    jacobian_started = time.perf_counter()
    shard_mean = average_prompt_jacobians(
        model,
        tokenizer,
        records,
        cache_dir=cache_dir,
        max_length=ALQR_CALIBRATION_COUNTS[BEHAVIOR].jacobian_max_length,
        vjp_chunk_size=JACOBIAN_VJP_CHUNK_SIZE,
        model_revision=MODEL_REVISION,
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


def aggregate_raw_jacobians(shard_directories: list[Path], expected_counts: list[int]) -> torch.Tensor:
    """Average all cached prompt Jacobians in float64, one layer at a time."""

    prompt_directories = []
    for shard_directory, expected_count in zip(shard_directories, expected_counts, strict=True):
        current = sorted(path for path in shard_directory.iterdir() if path.is_dir())
        if len(current) != expected_count:
            raise ValueError(
                f"Expected {expected_count} prompt caches in {shard_directory}; found {len(current)}"
            )
        prompt_directories.extend(current)
    if not prompt_directories:
        raise ValueError("No Jacobians were cached")
    layer_names = sorted(path.name for path in prompt_directories[0].glob("layer_*.pt"))
    if not layer_names:
        raise ValueError("The first prompt cache contains no layer Jacobians")

    averaged_layers = []
    for layer_name in layer_names:
        total = None
        expected_shape = None
        for prompt_directory in prompt_directories:
            path = prompt_directory / layer_name
            if not path.exists():
                raise ValueError(f"Missing cached Jacobian: {path}")
            derivative = torch.load(path, map_location="cpu", weights_only=True)
            if expected_shape is None:
                expected_shape = derivative.shape
                total = torch.zeros(expected_shape, dtype=torch.float64)
            if derivative.shape != expected_shape or not torch.isfinite(derivative).all():
                raise ValueError(f"Invalid cached Jacobian: {path}")
            total.add_(derivative)
        averaged_layers.append((total / len(prompt_directories)).float())
    return torch.stack(averaged_layers)


def fit_jacobians(devices: list[str]) -> None:
    if len(devices) != len(set(devices)) or not devices:
        raise ValueError("Jacobian devices must be a nonempty list of distinct CUDA devices")
    data = prepare()
    identity = {**calibration_identity(data), "devices": devices, "shard_count": len(devices)}
    run_path = CACHE / "runs/jacobians.json"
    artifact_path = CACHE / "dynamics.pt"
    nominal_identity = nominal_dynamics_identity(
        behavior=BEHAVIOR,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        records=data["calibration"]["jacobian"],
        max_length=ALQR_CALIBRATION_COUNTS[BEHAVIOR].jacobian_max_length,
        vjp_chunk_size=JACOBIAN_VJP_CHUNK_SIZE,
    )
    run = _load_run(run_path, identity)
    if run["status"] == "complete":
        load_nominal_dynamics(artifact_path, nominal_identity)
        return

    attempt = {
        "started_at_utc": _utc_now(),
        "status": "running",
        "runtime": runtime_provenance("cpu"),
    }
    run["attempts"].append(attempt)
    _write_json(run_path, run)
    started = time.perf_counter()
    processes = []
    for shard_index, device in enumerate(devices):
        log_path = CACHE / "logs" / f"jacobian_shard_{shard_index:02d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--stage",
            "jacobian-shard",
            "--device",
            device,
            "--shard-index",
            str(shard_index),
            "--shard-count",
            str(len(devices)),
        ]
        process = subprocess.Popen(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
        processes.append((shard_index, device, process, handle, log_path))

    last_report = 0.0
    while any(process.poll() is None for _index, _device, process, _handle, _log in processes):
        elapsed = time.perf_counter() - started
        if elapsed - last_report >= 30:
            cached_layers = len(list((CACHE / "jacobians").glob("shard_*/*/layer_*.pt")))
            states = [process.poll() for _index, _device, process, _handle, _log in processes]
            print(
                f"Jacobian calibration: {elapsed / 60:.1f} min, "
                f"{cached_layers} layer matrices cached, shard states={states}",
                flush=True,
            )
            last_report = elapsed
        time.sleep(5)

    failures = []
    for shard_index, device, process, handle, log_path in processes:
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
    dynamics = aggregate_raw_jacobians(
        [CACHE / "jacobians" / f"shard_{index:02d}" for index in range(len(devices))],
        expected_counts,
    )
    attempt["aggregation_elapsed_seconds"] = time.perf_counter() - aggregation_started
    shard_runs = [
        json.loads((CACHE / "runs" / f"jacobian_shard_{index:02d}.json").read_text())
        for index in range(len(devices))
    ]
    attempt["finished_at_utc"] = _utc_now()
    attempt["elapsed_seconds"] = time.perf_counter() - started
    attempt["matrix_shape"] = list(dynamics.shape)
    attempt["status"] = "complete"
    save_nominal_dynamics(
        artifact_path,
        nominal_identity,
        dynamics,
        attempts=run["attempts"],
    )
    attempt["artifact_sha256"] = _sha256(artifact_path)
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
    _update_timings("jacobians", {**attempt, "shards": run["shards"]})


def write_manifest() -> None:
    data = prepare()
    identity = calibration_identity(data)
    setpoint_run = CACHE / "runs/setpoint.json"
    jacobian_run = CACHE / "runs/jacobians.json"
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
        CACHE / "manifest.json",
        {
            "identity": identity,
            "status": status,
            "artifacts": {
                "data": str((CACHE / "data.json").resolve()),
                "setpoint": str((CACHE / "setpoint.pt").resolve()),
                "raw_jacobians": str((CACHE / "jacobians").resolve()),
                "averaged_dynamics": str((CACHE / "dynamics.pt").resolve()),
                "timings": str((CACHE / "timings.json").resolve()),
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
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"))
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    arguments = parser.parse_args()
    devices = arguments.devices.split(",")

    if arguments.stage == "prepare":
        prepare()
    elif arguments.stage == "setpoint":
        if arguments.device is None:
            raise ValueError("setpoint requires --device")
        fit_setpoint(arguments.device)
    elif arguments.stage == "jacobian-shard":
        if arguments.device is None or arguments.shard_index is None or arguments.shard_count is None:
            raise ValueError("jacobian-shard requires --device, --shard-index, and --shard-count")
        fit_jacobian_shard(arguments.device, arguments.shard_index, arguments.shard_count)
    elif arguments.stage == "jacobians":
        fit_jacobians(devices)
    elif arguments.stage == "all":
        started_at = _utc_now()
        started = time.perf_counter()
        prepare()
        fit_setpoint(devices[0])
        fit_jacobians(devices)
        write_manifest()
        timings_path = CACHE / "timings.json"
        timings = json.loads(timings_path.read_text())
        if "calibration_total" not in timings:
            _update_timings(
                "calibration_total",
                {
                    "started_at_utc": started_at,
                    "finished_at_utc": _utc_now(),
                    "elapsed_seconds": time.perf_counter() - started,
                    "full_benchmark": "not_run",
                },
            )
    else:
        write_manifest()
        print((CACHE / "manifest.json").read_text())


if __name__ == "__main__":
    main()
