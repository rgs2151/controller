"""Cache-first ID benchmark runner for frozen A-LQR comparison methods."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from robust_steerability.modeling.interventions import register_generation_policy_hooks
from robust_steerability.source_methods.actadd import ActAddSteerer
from robust_steerability.source_methods.calibration import (
    fit_actadd_calibration,
    fit_iti_calibration,
    fit_odesteer_calibration,
    fit_setpoint_from_records,
    fit_transport_stack,
)
from robust_steerability.source_methods.control import (
    SetpointCalibration,
    build_alqr_policy,
    build_spid_policy,
)
from robust_steerability.source_methods.generation import generate_batched
from robust_steerability.source_methods.iti import register_iti_hooks
from robust_steerability.source_methods.modeling import load_source_model
from robust_steerability.source_methods.odesteer import register_odesteer_hook
from robust_steerability.source_methods.protocol import (
    ACT_FIT_SAMPLES_PER_CLASS,
    GENERATION_CACHE,
    ITI_FIT_SAMPLES_PER_CLASS,
    ITI_MAX_LENGTH,
    METHODS,
    ODESTEER_FIT_SAMPLES_PER_CLASS,
    SOURCE_RANDOM_SEED,
    act_module_patterns,
    calibration_counts,
    model_key,
    protocol_manifest,
    selected_parameters as resolve_selected_parameters,
)
from robust_steerability.source_methods.transport import register_transport_hooks


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_files() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[2]
    relative_paths = (
        "robust_steerability/calibration/nominal.py",
        "robust_steerability/calibration/nominal_artifact.py",
        "robust_steerability/control/base.py",
        "robust_steerability/control/lqr.py",
        "robust_steerability/control/pid.py",
        "robust_steerability/control/types.py",
        "robust_steerability/control/validation.py",
        "robust_steerability/modeling/huggingface.py",
        "robust_steerability/modeling/interventions.py",
        "robust_steerability/modeling/jacobians.py",
        "robust_steerability/runtime/policy.py",
        "robust_steerability/source_methods/calibration.py",
        "robust_steerability/source_methods/actadd.py",
        "robust_steerability/source_methods/control.py",
        "robust_steerability/source_methods/generation.py",
        "robust_steerability/source_methods/id_benchmark.py",
        "robust_steerability/source_methods/iti.py",
        "robust_steerability/source_methods/modeling.py",
        "robust_steerability/source_methods/odesteer.py",
        "robust_steerability/source_methods/protocol.py",
        "robust_steerability/source_methods/transport.py",
    )
    return {relative: _sha(repo / relative) for relative in relative_paths}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime_provenance(device: str) -> dict:
    """Capture the software, source, command, and requested compute device."""

    repo = Path(__file__).resolve().parents[2]
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    )
    packages = {}
    for package in ("accelerate", "bitsandbytes", "datasets", "numpy", "transformers"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    result = {
        "recorded_at_utc": _utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "process_id": os.getpid(),
        "working_directory": str(Path.cwd()),
        "command": list(sys.argv),
        "git_commit": git_commit,
        "git_dirty": bool(git_status.strip()),
        "python": platform.python_version(),
        "packages": {"torch": str(torch.__version__), **packages},
        "cuda_runtime": torch.version.cuda,
        "requested_device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if device.startswith("cuda:"):
        device_index = int(device.split(":", 1)[1])
        properties = torch.cuda.get_device_properties(device_index)
        result["gpu"] = {
            "logical_index": device_index,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "uuid": str(getattr(properties, "uuid", "")),
        }
    elif device != "cpu":
        raise ValueError(f"Unsupported provenance device {device!r}")
    return result


def _json_identity(payload: dict) -> dict:
    """Return the exact JSON representation used on disk for comparisons."""

    return json.loads(json.dumps(payload, sort_keys=True))


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


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _records(data: dict, behavior: str, repetition: int) -> list[dict]:
    return data["evaluation"][behavior][str(repetition)]


def _texts(records: list[dict]) -> list[str]:
    return [str(row["text"]) for row in records]


def _batch_size(model_id: str, method: str) -> int:
    key = model_key(model_id)
    if key == "qwen14b":
        return 1 if method in {"iti", "mean_act", "linear_act", "pid_act"} else 2
    if key == "llama8b":
        return 2 if method in {"iti", "mean_act", "linear_act", "pid_act"} else 4
    return 8


def _activation_batch_size(model_id: str, method: str) -> int:
    key = model_key(model_id)
    if key == "qwen14b":
        return 1 if method in {"iti", "mean_act", "linear_act", "pid_act"} else 4
    if key == "llama8b":
        return 2 if method in {"iti", "mean_act", "linear_act", "pid_act"} else 8
    return 16


def _candidate_name(parameters: dict) -> str:
    return _hash(parameters)[:16]


def _output_rows(records: list[dict], completions: list[str]) -> list[dict]:
    rows = []
    for record, completion in zip(records, completions, strict=True):
        row = {
            "prompt_id": record["prompt_id"],
            "text": record["text"],
            "completion": completion,
        }
        for key in ("question", "answer_index", "subject"):
            if key in record:
                row[key] = record[key]
        rows.append(row)
    return rows


def _generate_candidate(
    *,
    output: Path,
    model,
    tokenizer,
    data: dict,
    behavior: str,
    model_id: str,
    revision: str,
    method: str,
    device: str,
    parameters: dict,
    controller_artifacts: dict[str, str],
    register_hooks: Callable[[], list[torch.utils.hooks.RemovableHandle]] | None,
    reset: Callable[[], None] | None = None,
) -> None:
    repetitions_expected = int(data["evaluation_repetitions"])
    evaluation_samples = len(_records(data, behavior, 0))
    auxiliary_expected = data.get("capability_evaluation", {})
    cache_policy = GENERATION_CACHE[method]
    manifest = protocol_manifest(
        method,
        behavior,
        model_id,
        revision,
        evaluation_samples,
        requested_parameters=parameters if method in {"iti", "spid"} else None,
    )
    identity = _json_identity({
        "schema_version": 5,
        "data_fingerprint": data["fingerprint"],
        "implementation_files_sha256": _implementation_files(),
        "protocol": manifest,
        "method": method,
        "parameters": parameters,
        "controller_artifacts_sha256": controller_artifacts,
        "execution": {
            "batch_size": _batch_size(model_id, method),
            "seed": SOURCE_RANDOM_SEED,
            "repetition_seed_stride": 100_000,
            "batch_seed_rule": "repetition_seed_plus_batch_start",
            "capability_seed_start": SOURCE_RANDOM_SEED + 900_000,
        },
    })
    payload = {
        "identity": identity,
        "status": "partial",
        "attempts": [],
        "repetitions": [],
        "capability_evaluation": {},
    }
    if output.exists():
        payload = json.loads(output.read_text())
        if payload["identity"] != identity:
            raise ValueError(f"Generation cache mismatch: {output}")
        if payload["status"] == "complete":
            if len(payload["repetitions"]) != repetitions_expected:
                raise ValueError(f"Incomplete generation cache marked complete: {output}")
            incomplete = [
                name
                for name, records in auxiliary_expected.items()
                if len(payload["capability_evaluation"].get(name, {}).get("rows", []))
                != len(records)
            ]
            if incomplete:
                raise ValueError(
                    f"Incomplete capability evaluation marked complete for {incomplete}: {output}"
                )
            return
        if payload["status"] != "partial":
            raise ValueError(f"Unknown generation cache status: {output}")
    repetitions = payload["repetitions"]
    if len(repetitions) > repetitions_expected:
        raise ValueError(f"Generation cache has too many repetitions: {output}")
    attempt_started = time.perf_counter()
    attempt = {
        "started_at_utc": _utc_now(),
        "status": "running",
        "runtime": runtime_provenance(device),
    }
    payload["attempts"].append(attempt)
    _write_json(output, payload)


    for repetition in range(len(repetitions), repetitions_expected):
        records = _records(data, behavior, repetition)
        if len(records) != evaluation_samples:
            raise ValueError("Every evaluation repetition must contain the same number of prompts")
        repetition_started = time.perf_counter()
        started_at = _utc_now()
        repetition_seed = SOURCE_RANDOM_SEED + repetition * 100_000
        completions = generate_batched(
            model,
            tokenizer,
            _texts(records),
            behavior=behavior,
            batch_size=_batch_size(model_id, method),
            seed=repetition_seed,
            use_cache=cache_policy["evaluation"],
            register_hooks=register_hooks,
            reset=reset,
        )
        repetitions.append(
            {
                "repetition": repetition,
                "started_at_utc": started_at,
                "finished_at_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - repetition_started,
                "sample_count": len(records),
                "generation_seed": repetition_seed,
                "prompt_ids_sha256": _hash([record["prompt_id"] for record in records]),
                "rows": _output_rows(records, completions),
            }
        )
        _write_json(output, payload)
    for auxiliary_index, (name, records) in enumerate(sorted(auxiliary_expected.items())):
        if name in payload["capability_evaluation"]:
            saved_rows = payload["capability_evaluation"][name]["rows"]
            if len(saved_rows) != len(records):
                raise ValueError(f"Incomplete cached {name} evaluation: {output}")
            continue
        auxiliary_started = time.perf_counter()
        auxiliary_started_at = _utc_now()
        auxiliary_seed = SOURCE_RANDOM_SEED + 900_000 + auxiliary_index * 100_000
        completions = generate_batched(
            model,
            tokenizer,
            _texts(records),
            behavior=name,
            batch_size=_batch_size(model_id, method),
            seed=auxiliary_seed,
            use_cache=cache_policy["capability"],
            register_hooks=register_hooks,
            reset=reset,
        )
        payload["capability_evaluation"][name] = {
            "started_at_utc": auxiliary_started_at,
            "finished_at_utc": _utc_now(),
            "elapsed_seconds": time.perf_counter() - auxiliary_started,
            "sample_count": len(records),
            "generation_seed": auxiliary_seed,
            "prompt_ids_sha256": _hash([record["prompt_id"] for record in records]),
            "rows": _output_rows(records, completions),
        }
        _write_json(output, payload)
    payload["status"] = "complete"
    attempt["finished_at_utc"] = _utc_now()
    attempt["elapsed_seconds"] = time.perf_counter() - attempt_started
    attempt["completed_repetitions"] = len(repetitions)
    attempt["status"] = "complete"
    _write_json(output, payload)


def load_frozen_alqr_artifacts(
    *,
    artifact_root: Path,
    behavior: str,
    model_id: str,
    revision: str,
    parameters: dict,
) -> tuple[SetpointCalibration, torch.Tensor, dict[str, str], dict[str, list[str]]]:
    """Load and strictly validate the precomputed source-faithful A-LQR artifacts."""

    data_path = artifact_root / "data.json"
    setpoint_path = artifact_root / "setpoint.pt"
    dynamics_path = artifact_root / "dynamics.pt"
    dynamics_metadata_path = dynamics_path.with_suffix(".json")
    missing = [
        str(path)
        for path in (data_path, setpoint_path, dynamics_path, dynamics_metadata_path)
        if not path.exists()
    ]
    if missing:
        raise ValueError(f"Missing frozen A-LQR artifacts: {missing}")

    artifact_data = json.loads(data_path.read_text())
    scientific_data = {
        key: value for key, value in artifact_data.items() if key != "fingerprint"
    }
    if artifact_data.get("fingerprint") != _hash(scientific_data):
        raise ValueError(f"Frozen A-LQR data fingerprint mismatch: {data_path}")
    if artifact_data.get("behavior") != behavior:
        raise ValueError("Frozen A-LQR behavior does not match the evaluation")

    setpoint_payload = torch.load(setpoint_path, map_location="cpu", weights_only=True)
    dynamics_payload = torch.load(dynamics_path, map_location="cpu", weights_only=True)
    dynamics_metadata = json.loads(dynamics_metadata_path.read_text())
    if setpoint_payload.get("identity") != dynamics_payload.get("identity"):
        raise ValueError("Frozen A-LQR setpoint and dynamics identities differ")
    identity = setpoint_payload["identity"]
    if (
        dynamics_metadata.get("identity") != identity
        or dynamics_metadata.get("status") != "complete"
        or dynamics_metadata.get("artifact_sha256") != _sha(dynamics_path)
    ):
        raise ValueError("Frozen A-LQR dynamics metadata is invalid")
    counts = calibration_counts("alqr", behavior)
    expected_setting = {
        "multiplier": float(parameters["lambda"]),
        "q": float(parameters["q"]),
        "r": float(parameters["r"]),
        "q_final": float(parameters["q_final"]),
    }
    expected_identity = {
        "schema_version": 1,
        "behavior": behavior,
        "data_fingerprint": artifact_data["fingerprint"],
        "model": {"id": model_id, "revision": revision},
        "counts": counts.__dict__,
        "alqr_setting": expected_setting,
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise ValueError(
                f"Frozen A-LQR identity mismatch for {key}: "
                f"expected {expected!r}, found {identity.get(key)!r}"
            )

    artifact_calibration = artifact_data["calibration"]
    calibration_selection = {}
    for name, count in (
        ("undesired", counts.undesired),
        ("desired", counts.desired),
        ("jacobian", counts.jacobian),
    ):
        artifact_ids = [row["prompt_id"] for row in artifact_calibration[name]]
        if len(artifact_ids) != count:
            raise ValueError(f"Frozen A-LQR {name} prompt count changed")
        calibration_selection[f"{name}_prompt_ids"] = artifact_ids

    contrast = setpoint_payload["contrast"]
    feature_norm = setpoint_payload["feature_norm"]
    dynamics = dynamics_payload["dynamics"]
    if contrast.ndim != 2 or feature_norm.shape != contrast.shape[:1]:
        raise ValueError("Frozen A-LQR setpoint tensors have invalid shapes")
    if (
        dynamics.ndim != 3
        or dynamics.shape[0] + 1 != contrast.shape[0]
        or dynamics.shape[1] != dynamics.shape[2]
        or dynamics.shape[1] != contrast.shape[1]
    ):
        raise ValueError("Frozen A-LQR dynamics shape does not match the setpoint")
    if not all(torch.isfinite(tensor).all() for tensor in (contrast, feature_norm, dynamics)):
        raise ValueError("Frozen A-LQR artifacts contain non-finite values")
    if not torch.allclose(
        feature_norm,
        torch.linalg.vector_norm(contrast, dim=1),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("Frozen A-LQR feature norms do not match the contrast vectors")

    hashes = {
        "calibration_data": _sha(data_path),
        "setpoint": _sha(setpoint_path),
        "dynamics": _sha(dynamics_path),
        "dynamics_metadata": _sha(dynamics_metadata_path),
    }
    return SetpointCalibration(contrast, feature_norm), dynamics, hashes, calibration_selection


def _artifact(
    path: Path,
    identity: dict,
    fit: Callable[[], object],
    device: str,
) -> object:
    identity = _json_identity(identity)
    metadata = path.with_suffix(".json")
    record = {"identity": identity, "status": "partial", "attempts": []}
    if path.exists():
        if not metadata.exists():
            raise ValueError(f"Calibration artifact has no metadata: {path}")
        record = json.loads(metadata.read_text())
        if record["identity"] != identity:
            raise ValueError(f"Calibration cache mismatch: {path}")
        if record["status"] != "complete" or record["artifact_sha256"] != _sha(path):
            raise ValueError(f"Invalid completed calibration cache: {path}")
        return torch.load(path, map_location="cpu", weights_only=False)
    if metadata.exists():
        record = json.loads(metadata.read_text())
        if record["identity"] != identity or record["status"] != "partial":
            raise ValueError(f"Calibration cache mismatch: {path}")
    started = time.perf_counter()
    attempt = {
        "started_at_utc": _utc_now(),
        "status": "running",
        "runtime": runtime_provenance(device),
    }
    record["attempts"].append(attempt)
    _write_json(metadata, record)
    fitted = fit()
    _save_torch(path, fitted)
    attempt["finished_at_utc"] = _utc_now()
    attempt["elapsed_seconds"] = time.perf_counter() - started
    attempt["status"] = "complete"
    record["status"] = "complete"
    record["artifact_sha256"] = _sha(path)
    _write_json(metadata, record)
    return fitted


def run_generation_job(
    *,
    unit: Path,
    behavior: str,
    model_id: str,
    revision: str,
    method: str,
    device: str,
    token: str,
    alqr_artifact_root: Path,
    selected_parameters: dict | None = None,
) -> None:
    """Fit one source method and cache every configured ID repetition."""

    if method not in METHODS:
        raise ValueError(f"Unsupported method {method!r}")
    if not device.startswith("cuda:"):
        raise ValueError("Source benchmark jobs require an explicit CUDA device")
    data_path = unit / "cache/data" / f"{behavior}.json"
    data = json.loads(data_path.read_text())
    key = model_key(model_id)
    evaluation_samples = len(_records(data, behavior, 0))
    parameters = resolve_selected_parameters(
        method, behavior, model_id, selected_parameters
    )
    manifest = protocol_manifest(
        method,
        behavior,
        model_id,
        revision,
        evaluation_samples,
        requested_parameters=selected_parameters,
    )
    counts = calibration_counts(method, behavior)
    undesired = data["calibration"][behavior]["undesired"]
    desired = data["calibration"][behavior]["desired"]
    jacobian = data["calibration"][behavior].get("jacobian", [])
    if method != "alqr" and (
        len(undesired) < counts.undesired or len(desired) < counts.desired
    ):
        raise ValueError(
            f"Calibration cache has {len(undesired)}/{len(desired)} class prompts; "
            f"{method} requires {counts.undesired}/{counts.desired}"
        )
    if method != "alqr" and len(jacobian) < counts.jacobian:
        raise ValueError(
            f"Calibration cache has {len(jacobian)} Jacobian prompts; "
            f"{method} requires {counts.jacobian}"
        )
    frozen_setpoint = None
    frozen_dynamics = None
    frozen_artifact_hashes: dict[str, str] = {}
    calibration_selection = {
        "undesired_prompt_ids": [row["prompt_id"] for row in undesired[:counts.undesired]],
        "desired_prompt_ids": [row["prompt_id"] for row in desired[:counts.desired]],
        "jacobian_prompt_ids": [row["prompt_id"] for row in jacobian[:counts.jacobian]],
    }
    if method == "alqr":
        (
            frozen_setpoint,
            frozen_dynamics,
            frozen_artifact_hashes,
            calibration_selection,
        ) = (
            load_frozen_alqr_artifacts(
                artifact_root=alqr_artifact_root,
                behavior=behavior,
                model_id=model_id,
                revision=revision,
                parameters=parameters,
            )
        )
    job_root = unit / "cache/generations" / behavior / key / method
    job_root.mkdir(parents=True, exist_ok=True)
    run_identity = _json_identity({
        "schema_version": 1,
        "data_sha256": _sha(data_path),
        "data_fingerprint": data["fingerprint"],
        "model_id": model_id,
        "checkpoint_revision": revision,
        "behavior": behavior,
        "method": method,
        "requested_device": device,
        "implementation_files_sha256": _implementation_files(),
        "protocol": manifest,
        "frozen_alqr_artifacts_sha256": frozen_artifact_hashes,
        "calibration_selection": calibration_selection,
    })
    run_path = unit / "cache/run_records" / behavior / key / f"{method}.json"
    run_record = {"identity": run_identity, "attempts": []}
    if run_path.exists():
        run_record = json.loads(run_path.read_text())
        if run_record["identity"] != run_identity:
            raise ValueError(f"Run-record identity mismatch: {run_path}")
    run_started = time.perf_counter()
    run_attempt = {
        "started_at_utc": _utc_now(),
        "status": "running",
        "runtime": runtime_provenance(device),
    }
    run_record["attempts"].append(run_attempt)
    _write_json(run_path, run_record)

    _seed(SOURCE_RANDOM_SEED)
    device_index = int(device.split(":", 1)[1])
    torch.cuda.reset_peak_memory_stats(device_index)
    model_method = "alqr" if method in {"alqr", "spid"} else method
    model_load_started = time.perf_counter()
    model, tokenizer = load_source_model(model_method, behavior, model_id, revision, device, token)
    run_attempt["model_loaded_at_utc"] = _utc_now()
    run_attempt["model_load_elapsed_seconds"] = time.perf_counter() - model_load_started
    _write_json(run_path, run_record)
    common_identity = {
        "schema_version": 1,
        "data_fingerprint": data["fingerprint"],
        "implementation_files_sha256": _implementation_files(),
        "model_id": model_id,
        "revision": revision,
        "behavior": behavior,
    }
    artifact_root = unit / "cache/calibrations" / behavior / key

    if method == "original":
        _generate_candidate(
            output=job_root / "original.json",
            model=model,
            tokenizer=tokenizer,
            data=data,
            behavior=behavior,
            model_id=model_id,
            revision=revision,
            method=method,
            device=device,
            parameters={},
            controller_artifacts={},
            register_hooks=None,
        )
    elif method in {"alqr", "spid"}:
        if method == "alqr":
            if frozen_setpoint is None or frozen_dynamics is None:
                raise RuntimeError("Frozen A-LQR artifacts were not loaded")
            policy = build_alqr_policy(
                frozen_dynamics,
                frozen_setpoint,
                multiplier=parameters["lambda"],
                q=parameters["q"],
                r=parameters["r"],
                q_final=parameters["q_final"],
                device=device,
            )
            _generate_candidate(
                output=job_root / f"{_candidate_name(parameters)}.json",
                model=model,
                tokenizer=tokenizer,
                data=data,
                behavior=behavior,
                model_id=model_id,
                revision=revision,
                method=method,
                device=device,
                parameters=parameters,
                controller_artifacts=frozen_artifact_hashes,
                register_hooks=lambda: register_generation_policy_hooks(model, policy),
            )
            del policy
            torch.cuda.empty_cache()
        else:
            setpoint_path = artifact_root / "setpoint.pt"
            setpoint = _artifact(
                setpoint_path,
                {
                    **common_identity,
                    "artifact": "shared_alqr_spid_setpoint",
                    "undesired_count": counts.undesired,
                    "desired_count": counts.desired,
                    "undesired_prompt_ids": [
                        row["prompt_id"] for row in undesired[:counts.undesired]
                    ],
                    "desired_prompt_ids": [
                        row["prompt_id"] for row in desired[:counts.desired]
                    ],
                    "activation_batch_size": _activation_batch_size(model_id, method),
                },
                lambda: fit_setpoint_from_records(
                    model,
                    tokenizer,
                    behavior=behavior,
                    negative_records=undesired[:counts.undesired],
                    positive_records=desired[:counts.desired],
                    activation_batch_size=_activation_batch_size(model_id, method),
                ),
                device,
            )
            policy = build_spid_policy(
                setpoint,
                multiplier=parameters["lambda"],
                kp=parameters["kp"],
                ki=parameters["ki"],
                kd=parameters["kd"],
            )
            _generate_candidate(
                output=job_root / f"{_candidate_name(parameters)}.json",
                model=model,
                tokenizer=tokenizer,
                data=data,
                behavior=behavior,
                model_id=model_id,
                revision=revision,
                method=method,
                device=device,
                parameters=parameters,
                controller_artifacts={"setpoint": _sha(setpoint_path)},
                register_hooks=lambda: register_generation_policy_hooks(model, policy),
            )
            del policy
            torch.cuda.empty_cache()
    elif method == "actadd":
        required = counts.undesired
        direction_path = artifact_root / "actadd.pt"
        direction = _artifact(
            direction_path,
            {**common_identity, "fit_samples_per_class": required},
            lambda: fit_actadd_calibration(
                model,
                tokenizer,
                undesired_texts=_texts(undesired[:required]),
                desired_texts=_texts(desired[:required]),
                batch_size=_activation_batch_size(model_id, method),
            ),
            device,
        )
        steerer = ActAddSteerer(direction, parameters["layer"], parameters["strength"])
        _generate_candidate(
            output=job_root / f"{_candidate_name(parameters)}.json",
            model=model,
            tokenizer=tokenizer,
            data=data,
            behavior=behavior,
            model_id=model_id,
            revision=revision,
            method=method,
            device=device,
            parameters=parameters,
            controller_artifacts={"actadd": _sha(direction_path)},
            register_hooks=lambda: steerer.register(model),
            reset=steerer.reset,
        )
    elif method == "iti":
        required = ITI_FIT_SAMPLES_PER_CLASS
        fitted_path = artifact_root / "iti.pt"
        fitted = _artifact(
            fitted_path,
            {**common_identity, "fit_samples_per_class": required, "max_length": ITI_MAX_LENGTH},
            lambda: fit_iti_calibration(
                model,
                tokenizer,
                undesired_texts=_texts(undesired[:required]),
                desired_texts=_texts(desired[:required]),
                batch_size=_activation_batch_size(model_id, method),
                max_length=ITI_MAX_LENGTH,
                seed=SOURCE_RANDOM_SEED,
            ),
            device,
        )
        _generate_candidate(
            output=job_root / f"{_candidate_name(parameters)}.json",
            model=model,
            tokenizer=tokenizer,
            data=data,
            behavior=behavior,
            model_id=model_id,
            revision=revision,
            method=method,
            device=device,
            parameters=parameters,
            controller_artifacts={"iti": _sha(fitted_path)},
            register_hooks=lambda: register_iti_hooks(
                model,
                fitted,
                top_heads=parameters["top_heads"],
                alpha=parameters["alpha"],
            ),
        )
    elif method in {"mean_act", "linear_act", "pid_act"}:
        required = ACT_FIT_SAMPLES_PER_CLASS[behavior]
        fitted_path = artifact_root / f"{method}.pt"
        fitted = _artifact(
            fitted_path,
            {**common_identity, "fit_samples_per_class": required},
            lambda: fit_transport_stack(
                model,
                tokenizer,
                source_texts=_texts(undesired[:required]),
                target_texts=_texts(desired[:required]),
                module_patterns=act_module_patterns(model_id),
                behavior=behavior,
                method=method,
                batch_size=_activation_batch_size(model_id, method),
                seed=SOURCE_RANDOM_SEED,
            ),
            device,
        )
        _generate_candidate(
            output=job_root / f"{_candidate_name(parameters)}.json",
            model=model,
            tokenizer=tokenizer,
            data=data,
            behavior=behavior,
            model_id=model_id,
            revision=revision,
            method=method,
            device=device,
            parameters=parameters,
            controller_artifacts={method: _sha(fitted_path)},
            register_hooks=lambda: register_transport_hooks(
                model, fitted, strength=parameters["strength"]
            ),
        )
    else:
        required = ODESTEER_FIT_SAMPLES_PER_CLASS[behavior]
        fitted_path = artifact_root / "odesteer.pt"
        fitted = _artifact(
            fitted_path,
            {**common_identity, "fit_samples_per_class": required, "layer": parameters["layer"]},
            lambda: fit_odesteer_calibration(
                model,
                tokenizer,
                behavior=behavior,
                layer_index=parameters["layer"],
                undesired_texts=_texts(undesired[:required]),
                desired_texts=_texts(desired[:required]),
                batch_size=_activation_batch_size(model_id, method),
            ),
            device,
        )
        _generate_candidate(
            output=job_root / f"{_candidate_name(parameters)}.json",
            model=model,
            tokenizer=tokenizer,
            data=data,
            behavior=behavior,
            model_id=model_id,
            revision=revision,
            method=method,
            device=device,
            parameters=parameters,
            controller_artifacts={"odesteer": _sha(fitted_path)},
            register_hooks=lambda: register_odesteer_hook(
                model,
                fitted,
                layer_index=parameters["layer"],
                time=parameters["time"],
            ),
        )

    output_files = sorted(path for path in job_root.glob("*.json") if path.name != "unsupported.json")
    run_attempt["status"] = "complete"
    run_attempt["finished_at_utc"] = _utc_now()
    run_attempt["elapsed_seconds"] = time.perf_counter() - run_started
    run_attempt["outputs"] = [
        {"path": str(path.relative_to(unit)), "sha256": _sha(path)} for path in output_files
    ]
    calibration_metadata = sorted(artifact_root.glob("*.json"))
    run_attempt["calibration_records"] = [
        {
            "path": str(path.relative_to(unit)),
            "metadata_sha256": _sha(path),
            "artifact_sha256": json.loads(path.read_text()).get("artifact_sha256"),
        }
        for path in calibration_metadata
    ]
    run_attempt["gpu_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device_index)
    run_attempt["gpu_peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(device_index)
    _write_json(run_path, run_record)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
