"""Generate benchmark responses with frozen upstream comparison methods."""

from __future__ import annotations

import gc
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

from robust_steerability.calibration.nominal_artifact import (
    load_shared_nominal_dynamics,
)
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


def _records(data: dict, evaluation_key: str, repetition: int) -> list[dict]:
    return data["evaluation"][evaluation_key][str(repetition)]


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
    generation_profile: str,
    evaluation_key: str,
    model_id: str,
    revision: str,
    method: str,
    device: str,
    parameters: dict,
    use_cache: bool,
    batch_size: int,
    register_hooks: Callable[[], list[torch.utils.hooks.RemovableHandle]] | None,
    reset: Callable[[], None] | None = None,
) -> None:
    repetitions_expected = len(data["evaluation"][evaluation_key])
    if set(data["evaluation"][evaluation_key]) != {
        str(index) for index in range(repetitions_expected)
    }:
        raise ValueError("Evaluation repetitions must be consecutively numbered from zero")
    evaluation_samples = len(_records(data, evaluation_key, 0))
    manifest = protocol_manifest(
        method,
        behavior,
        model_id,
        revision,
        evaluation_samples,
        repetitions_expected,
        generation_profile,
        use_cache=use_cache,
        requested_parameters=parameters if method in {"iti", "spid"} else None,
    )
    identity = _json_identity({
        "schema_version": 5,
        "model": [model_id, revision],
        "protocol": manifest,
        "method": method,
        "evaluation_key": evaluation_key,
        "parameters": parameters,
        "execution": {
            "batch_size": batch_size,
            "seed": SOURCE_RANDOM_SEED,
            "repetition_seed_stride": 100_000,
            "batch_seed_rule": "repetition_seed_plus_batch_start",
        },
    })
    payload = {
        "identity": identity,
        "status": "partial",
        "attempts": [],
        "repetitions": [],
    }
    if output.exists():
        payload = json.loads(output.read_text())
        if payload["status"] == "complete":
            return
    repetitions = payload["repetitions"]
    attempt_started = time.perf_counter()
    attempt = {
        "started_at_utc": _utc_now(),
        "status": "running",
        "runtime": runtime_provenance(device),
    }
    payload["attempts"].append(attempt)
    _write_json(output, payload)


    for repetition in range(len(repetitions), repetitions_expected):
        records = _records(data, evaluation_key, repetition)
        if len(records) != evaluation_samples:
            raise ValueError("Every evaluation repetition must contain the same number of prompts")
        repetition_started = time.perf_counter()
        started_at = _utc_now()
        repetition_seed = SOURCE_RANDOM_SEED + repetition * 100_000
        completions = generate_batched(
            model,
            tokenizer,
            _texts(records),
            behavior=generation_profile,
            batch_size=batch_size,
            seed=repetition_seed,
            use_cache=use_cache,
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
                "rows": _output_rows(records, completions),
            }
        )
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
    """Load source-faithful A-LQR artifacts from their explicit benchmark path."""

    data_path = artifact_root / "data.json"
    setpoint_path = artifact_root / "setpoint.pt"
    dynamics_path = artifact_root / "dynamics.pt"
    missing = [
        str(path)
        for path in (data_path, setpoint_path, dynamics_path)
        if not path.exists()
    ]
    if missing:
        raise ValueError(f"Missing frozen A-LQR artifacts: {missing}")

    artifact_data = json.loads(data_path.read_text())
    setpoint_payload = torch.load(setpoint_path, map_location="cpu", weights_only=True)
    dynamics = load_shared_nominal_dynamics(
        dynamics_path,
        behavior=behavior,
        model_id=model_id,
        model_revision=revision,
    )
    counts = calibration_counts("alqr", behavior)
    artifact_calibration = artifact_data["calibration"]
    calibration_selection = {
        f"{name}_prompt_ids": [row["prompt_id"] for row in artifact_calibration[name]]
        for name in ("undesired", "desired", "jacobian")
    }
    contrast = setpoint_payload["contrast"]
    feature_norm = setpoint_payload["feature_norm"]
    return SetpointCalibration(contrast, feature_norm), dynamics, {}, calibration_selection


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
        return torch.load(path, map_location="cpu", weights_only=False)
    if metadata.exists():
        record = json.loads(metadata.read_text())
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
    _write_json(metadata, record)
    return fitted


def _load_artifact(path: Path, identity: dict) -> object:
    """Load one completed calibration artifact without fitting during evaluation."""

    if not path.exists():
        raise FileNotFoundError(
            f"Missing selected calibration artifact; run calibrate first: {path}"
        )
    return torch.load(path, map_location="cpu", weights_only=False)


def _calibration_records(data: dict, behavior: str, method: str) -> tuple[
    object, list[dict], list[dict], list[dict]
]:
    counts = calibration_counts(method, behavior)
    calibration = data.get("calibration", {}).get(behavior, {})
    undesired = calibration.get("undesired", [])
    desired = calibration.get("desired", [])
    jacobian = calibration.get("jacobian", [])
    if len(undesired) < counts.undesired or len(desired) < counts.desired:
        raise ValueError(
            f"Calibration cache has {len(undesired)}/{len(desired)} class prompts; "
            f"{method} requires {counts.undesired}/{counts.desired}"
        )
    if len(jacobian) < counts.jacobian:
        raise ValueError(
            f"Calibration cache has {len(jacobian)} Jacobian prompts; "
            f"{method} requires {counts.jacobian}"
        )
    return counts, undesired, desired, jacobian


def _common_calibration_identity(
    data: dict, behavior: str, model_id: str, revision: str
) -> dict:
    return {
        "schema_version": 1,
        "model_id": model_id,
        "revision": revision,
        "behavior": behavior,
    }


def fit_source_method_calibration(
    *,
    behavior: str,
    model_id: str,
    revision: str,
    method: str,
    device: str,
    token: str,
    calibration_root: Path,
    calibration_data_path: Path,
    selected_parameters: dict | None = None,
) -> dict:
    """Fit one non-A-LQR source method before final evaluation."""

    if method in {"original", "alqr"} or method not in METHODS:
        raise ValueError(f"{method!r} has no source-method calibration fit")
    if not device.startswith("cuda:"):
        raise ValueError("Source calibration requires an explicit CUDA device")
    data = json.loads(calibration_data_path.read_text())
    parameters = resolve_selected_parameters(
        method, behavior, model_id, selected_parameters
    )
    counts, undesired, desired, _jacobian = _calibration_records(
        data, behavior, method
    )
    common_identity = _common_calibration_identity(
        data, behavior, model_id, revision
    )
    model_method = "alqr" if method == "spid" else method
    model, tokenizer = load_source_model(
        model_method, behavior, model_id, revision, device, token
    )

    if method == "spid":
        path = calibration_root / "setpoint.pt"
        identity = {
            **common_identity,
            "artifact": "shared_alqr_spid_setpoint",
            "undesired_count": counts.undesired,
            "desired_count": counts.desired,
            "undesired_prompt_ids": [
                row["prompt_id"] for row in undesired[: counts.undesired]
            ],
            "desired_prompt_ids": [
                row["prompt_id"] for row in desired[: counts.desired]
            ],
            "activation_batch_size": _activation_batch_size(model_id, method),
        }
        _artifact(
            path,
            identity,
            lambda: fit_setpoint_from_records(
                model,
                tokenizer,
                behavior=behavior,
                negative_records=undesired[: counts.undesired],
                positive_records=desired[: counts.desired],
                activation_batch_size=_activation_batch_size(model_id, method),
            ),
            device,
        )
    elif method == "actadd":
        path = calibration_root / "actadd.pt"
        identity = {
            **common_identity,
            "fit_samples_per_class": counts.undesired,
        }
        _artifact(
            path,
            identity,
            lambda: fit_actadd_calibration(
                model,
                tokenizer,
                undesired_texts=_texts(undesired[: counts.undesired]),
                desired_texts=_texts(desired[: counts.desired]),
                batch_size=_activation_batch_size(model_id, method),
            ),
            device,
        )
    elif method == "iti":
        path = calibration_root / "iti.pt"
        identity = {
            **common_identity,
            "fit_samples_per_class": ITI_FIT_SAMPLES_PER_CLASS,
            "max_length": ITI_MAX_LENGTH,
        }
        _artifact(
            path,
            identity,
            lambda: fit_iti_calibration(
                model,
                tokenizer,
                undesired_texts=_texts(undesired[:ITI_FIT_SAMPLES_PER_CLASS]),
                desired_texts=_texts(desired[:ITI_FIT_SAMPLES_PER_CLASS]),
                batch_size=_activation_batch_size(model_id, method),
                max_length=ITI_MAX_LENGTH,
                seed=SOURCE_RANDOM_SEED,
            ),
            device,
        )
    elif method in {"mean_act", "linear_act", "pid_act"}:
        path = calibration_root / f"{method}.pt"
        identity = {
            **common_identity,
            "fit_samples_per_class": counts.undesired,
        }
        _artifact(
            path,
            identity,
            lambda: fit_transport_stack(
                model,
                tokenizer,
                source_texts=_texts(undesired[: counts.undesired]),
                target_texts=_texts(desired[: counts.desired]),
                module_patterns=act_module_patterns(model_id),
                behavior=behavior,
                method=method,
                batch_size=_activation_batch_size(model_id, method),
                seed=SOURCE_RANDOM_SEED,
            ),
            device,
        )
    else:
        path = calibration_root / "odesteer.pt"
        identity = {
            **common_identity,
            "fit_samples_per_class": counts.undesired,
            "layer": parameters["layer"],
        }
        _artifact(
            path,
            identity,
            lambda: fit_odesteer_calibration(
                model,
                tokenizer,
                behavior=behavior,
                layer_index=parameters["layer"],
                undesired_texts=_texts(undesired[: counts.undesired]),
                desired_texts=_texts(desired[: counts.desired]),
                batch_size=_activation_batch_size(model_id, method),
            ),
            device,
        )

    selection = {
        "schema_version": 1,
        "model": [model_id, revision],
        "behavior": behavior,
        "method": method,
        "calibration_id": calibration_root.name,
        "parameters": parameters,
        "source": (
            "recorded development-set selection from the preserved source grid"
            if method in {"iti", "spid"}
            else "fixed source setting"
        ),
    }
    _write_json(calibration_root / "selection.json", selection)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return selection


def run_generation_job(
    *,
    cache_root: Path,
    behavior: str,
    model_id: str,
    revision: str,
    method: str,
    device: str,
    token: str,
    alqr_artifact_root: Path,
    method_calibration_root: Path,
    calibration_data_path: Path,
    data_path: Path,
    evaluation_key: str,
    cache_namespace: str,
    generation_profile: str,
    use_cache: bool,
    selected_parameters: dict | None = None,
    generation_batch_size: int | None = None,
) -> None:
    """Run one source method on an explicit frozen evaluation dataset."""

    if method not in METHODS:
        raise ValueError(f"Unsupported method {method!r}")
    if not device.startswith("cuda:"):
        raise ValueError("Source benchmark jobs require an explicit CUDA device")
    data = json.loads(data_path.read_text())
    evaluation_samples = len(_records(data, evaluation_key, 0))
    parameters = resolve_selected_parameters(
        method, behavior, model_id, selected_parameters
    )
    batch_size = generation_batch_size or _batch_size(model_id, method)
    if batch_size < 1:
        raise ValueError("generation_batch_size must be positive")
    manifest = protocol_manifest(
        method,
        behavior,
        model_id,
        revision,
        evaluation_samples,
        len(data["evaluation"][evaluation_key]),
        generation_profile,
        use_cache=use_cache,
        requested_parameters=selected_parameters,
    )
    calibration_data = json.loads(calibration_data_path.read_text())
    counts, undesired, desired, _jacobian = _calibration_records(
        calibration_data, behavior, method
    ) if method not in {"original", "alqr"} else (
        calibration_counts(method, behavior), [], [], []
    )
    frozen_setpoint = None
    frozen_dynamics = None
    frozen_artifact_hashes: dict[str, str] = {}
    calibration_selection: dict[str, list[str]] = {}
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
    job_root = cache_root / "generations" / cache_namespace / method
    job_root.mkdir(parents=True, exist_ok=True)
    run_identity = _json_identity({
        "schema_version": 1,
        "model_id": model_id,
        "checkpoint_revision": revision,
        "behavior": behavior,
        "evaluation_key": evaluation_key,
        "cache_namespace": cache_namespace,
        "method": method,
        "requested_device": device,
        "protocol": manifest,
        "calibration_selection": calibration_selection,
    })
    run_path = cache_root / "run_records" / cache_namespace / f"{method}.json"
    run_record = {"identity": run_identity, "attempts": []}
    if run_path.exists():
        run_record = json.loads(run_path.read_text())
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
    common_identity = _common_calibration_identity(
        calibration_data, behavior, model_id, revision
    )
    artifact_root = method_calibration_root

    if method == "original":
        _generate_candidate(
            output=job_root / "final.json",
            model=model,
            tokenizer=tokenizer,
            data=data,
            behavior=behavior,
            generation_profile=generation_profile,
            evaluation_key=evaluation_key,
            model_id=model_id,
            revision=revision,
            method=method,
            device=device,
            parameters={},
            use_cache=use_cache,
            batch_size=batch_size,
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
                output=job_root / "final.json",
                model=model,
                tokenizer=tokenizer,
                data=data,
                behavior=behavior,
                generation_profile=generation_profile,
                evaluation_key=evaluation_key,
                model_id=model_id,
                revision=revision,
                method=method,
                device=device,
                parameters=parameters,
                use_cache=use_cache,
                batch_size=batch_size,
                register_hooks=lambda: register_generation_policy_hooks(model, policy),
            )
            del policy
            torch.cuda.empty_cache()
        else:
            setpoint_path = artifact_root / "setpoint.pt"
            setpoint = _load_artifact(
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
            )
            policy = build_spid_policy(
                setpoint,
                multiplier=parameters["lambda"],
                kp=parameters["kp"],
                ki=parameters["ki"],
                kd=parameters["kd"],
            )
            _generate_candidate(
                output=job_root / "final.json",
                model=model,
                tokenizer=tokenizer,
                data=data,
                behavior=behavior,
                generation_profile=generation_profile,
                evaluation_key=evaluation_key,
                model_id=model_id,
                revision=revision,
                method=method,
                device=device,
                parameters=parameters,
                use_cache=use_cache,
                batch_size=batch_size,
                register_hooks=lambda: register_generation_policy_hooks(model, policy),
            )
            del policy
            torch.cuda.empty_cache()
    elif method == "actadd":
        required = counts.undesired
        direction_path = artifact_root / "actadd.pt"
        direction = _load_artifact(
            direction_path,
            {**common_identity, "fit_samples_per_class": required},
        )
        steerer = ActAddSteerer(
            direction[parameters["layer"]],
            parameters["layer"],
            parameters["strength"],
        )
        _generate_candidate(
            output=job_root / "final.json",
            model=model,
            tokenizer=tokenizer,
            data=data,
            behavior=behavior,
            generation_profile=generation_profile,
            evaluation_key=evaluation_key,
            model_id=model_id,
            revision=revision,
            method=method,
            device=device,
            parameters=parameters,
            use_cache=use_cache,
            batch_size=batch_size,
            register_hooks=lambda: steerer.register(model),
            reset=steerer.reset,
        )
    elif method == "iti":
        required = ITI_FIT_SAMPLES_PER_CLASS
        fitted_path = artifact_root / "iti.pt"
        fitted = _load_artifact(
            fitted_path,
            {**common_identity, "fit_samples_per_class": required, "max_length": ITI_MAX_LENGTH},
        )
        _generate_candidate(
            output=job_root / "final.json",
            model=model,
            tokenizer=tokenizer,
            data=data,
            behavior=behavior,
            generation_profile=generation_profile,
            evaluation_key=evaluation_key,
            model_id=model_id,
            revision=revision,
            method=method,
            device=device,
            parameters=parameters,
            use_cache=use_cache,
            batch_size=batch_size,
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
        fitted = _load_artifact(
            fitted_path,
            {**common_identity, "fit_samples_per_class": required},
        )
        _generate_candidate(
            output=job_root / "final.json",
            model=model,
            tokenizer=tokenizer,
            data=data,
            behavior=behavior,
            generation_profile=generation_profile,
            evaluation_key=evaluation_key,
            model_id=model_id,
            revision=revision,
            method=method,
            device=device,
            parameters=parameters,
            use_cache=use_cache,
            batch_size=batch_size,
            register_hooks=lambda: register_transport_hooks(
                model, fitted, strength=parameters["strength"]
            ),
        )
    else:
        required = ODESTEER_FIT_SAMPLES_PER_CLASS[behavior]
        fitted_path = artifact_root / "odesteer.pt"
        fitted = _load_artifact(
            fitted_path,
            {**common_identity, "fit_samples_per_class": required, "layer": parameters["layer"]},
        )
        _generate_candidate(
            output=job_root / "final.json",
            model=model,
            tokenizer=tokenizer,
            data=data,
            behavior=behavior,
            generation_profile=generation_profile,
            evaluation_key=evaluation_key,
            model_id=model_id,
            revision=revision,
            method=method,
            device=device,
            parameters=parameters,
            use_cache=use_cache,
            batch_size=batch_size,
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
    run_attempt["outputs"] = [str(path.relative_to(cache_root)) for path in output_files]
    calibration_metadata = sorted(artifact_root.glob("*.json"))
    run_attempt["calibration_records"] = [
        str(path.resolve())
        for path in calibration_metadata
    ]
    run_attempt["gpu_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device_index)
    run_attempt["gpu_peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(device_index)
    _write_json(run_path, run_record)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
