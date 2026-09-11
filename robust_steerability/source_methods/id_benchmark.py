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
    fit_control_calibration,
    fit_iti_calibration,
    fit_odesteer_calibration,
    fit_transport_stack,
)
from robust_steerability.source_methods.control import build_alqr_policy, build_spid_policy
from robust_steerability.source_methods.generation import generate_batched
from robust_steerability.source_methods.iti import register_iti_hooks
from robust_steerability.source_methods.modeling import load_source_model
from robust_steerability.source_methods.odesteer import register_odesteer_hook
from robust_steerability.source_methods.protocol import (
    ACT_FIT_SAMPLES_PER_CLASS,
    ACT_SWEEPS,
    ACTADD_PAPER_SELECTIONS,
    CALIBRATION_COUNTS,
    ITI_FIT_SAMPLES_PER_CLASS,
    ITI_SWEEPS,
    ODESTEER_FIT_SAMPLES,
    ODESTEER_PAPER_SELECTIONS,
    SPID_SWEEPS,
    SOURCE_RANDOM_SEED,
    act_module_patterns,
    model_key,
    paper_alqr_setting,
    protocol_manifest,
)
from robust_steerability.source_methods.transport import register_transport_hooks


METHODS = (
    "original",
    "iti",
    "actadd",
    "mean_act",
    "linear_act",
    "pid_act",
    "odesteer",
    "spid",
    "alqr",
)


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
        "robust_steerability/source_methods/control.py",
        "robust_steerability/source_methods/generation.py",
        "robust_steerability/source_methods/id_benchmark.py",
        "robust_steerability/source_methods/modeling.py",
        "robust_steerability/source_methods/protocol.py",
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
    register_hooks: Callable[[], list[torch.utils.hooks.RemovableHandle]] | None,
    reset: Callable[[], None] | None = None,
    use_cache: bool = True,
) -> None:
    repetitions_expected = int(data["evaluation_repetitions"])
    evaluation_samples = len(_records(data, behavior, 0))
    manifest = protocol_manifest(behavior, model_id, revision, evaluation_samples)
    identity = _json_identity({
        "schema_version": 3,
        "data_fingerprint": data["fingerprint"],
        "implementation_files_sha256": _implementation_files(),
        "protocol": manifest,
        "method": method,
        "parameters": parameters,
        "execution": {
            "batch_size": _batch_size(model_id, method),
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
        if payload["identity"] != identity:
            raise ValueError(f"Generation cache mismatch: {output}")
        if payload["status"] == "complete":
            if len(payload["repetitions"]) != repetitions_expected:
                raise ValueError(f"Incomplete generation cache marked complete: {output}")
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
                "prompt_ids_sha256": _hash([record["prompt_id"] for record in records]),
                "rows": [
                    {
                        "prompt_id": record["prompt_id"],
                        "text": record["text"],
                        "question": record.get("question"),
                        "completion": completion,
                    }
                    for record, completion in zip(records, completions, strict=True)
                ],
            }
        )
        payload["status"] = "complete" if len(repetitions) == repetitions_expected else "partial"
        _write_json(output, payload)
    attempt["finished_at_utc"] = _utc_now()
    attempt["elapsed_seconds"] = time.perf_counter() - attempt_started
    attempt["completed_repetitions"] = len(repetitions)
    attempt["status"] = "complete"
    _write_json(output, payload)


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
) -> None:
    """Fit one source method and cache every configured ID repetition."""

    if method not in METHODS:
        raise ValueError(f"Unsupported method {method!r}")
    if not device.startswith("cuda:"):
        raise ValueError("Source benchmark jobs require an explicit CUDA device")
    alqr_setting = paper_alqr_setting(behavior, model_id) if method == "alqr" else None
    data_path = unit / "cache/data.json"
    data = json.loads(data_path.read_text())
    key = model_key(model_id)
    job_root = unit / "cache/generations" / behavior / key / method
    job_root.mkdir(parents=True, exist_ok=True)
    if method == "odesteer" and key not in ODESTEER_PAPER_SELECTIONS[behavior]:
        _write_json(
            job_root / "unsupported.json",
            {
                "status": "unsupported",
                "reason": "The preserved comparison source does not record this paper selection.",
                "behavior": behavior,
                "model_id": model_id,
                "method": method,
            },
        )
        return

    evaluation_samples = len(_records(data, behavior, 0))
    undesired = data["calibration"][behavior]["undesired"]
    desired = data["calibration"][behavior]["desired"]
    jacobian = data["calibration"][behavior]["jacobian"]
    counts = CALIBRATION_COUNTS[behavior]
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
        "protocol": protocol_manifest(behavior, model_id, revision, evaluation_samples),
        "calibration_selection": {
            "negative_prompt_ids": [row["prompt_id"] for row in undesired[:counts.negative]],
            "positive_prompt_ids": [row["prompt_id"] for row in desired[:counts.positive]],
            "jacobian_prompt_ids": [row["prompt_id"] for row in jacobian[:counts.jacobian]],
        },
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
        "method": method,
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
            register_hooks=None,
            use_cache=False,
        )
    elif method in {"alqr", "spid"}:
        calibration = _artifact(
            artifact_root / "control.pt",
            {
                **common_identity,
                "method": "control",
                "counts": counts.__dict__,
                "negative_prompt_ids": [row["prompt_id"] for row in undesired[:counts.negative]],
                "positive_prompt_ids": [row["prompt_id"] for row in desired[:counts.positive]],
                "jacobian_prompt_ids": [row["prompt_id"] for row in jacobian[:counts.jacobian]],
                "activation_batch_size": _activation_batch_size(model_id, method),
                "jacobian_vjp_chunk_size": 32,
            },
            lambda: fit_control_calibration(
                model,
                tokenizer,
                behavior=behavior,
                negative_records=undesired[:counts.negative],
                positive_records=desired[:counts.positive],
                jacobian_records=jacobian[:counts.jacobian],
                checkpoint_revision=revision,
                jacobian_cache=artifact_root / "jacobians",
                activation_batch_size=_activation_batch_size(model_id, method),
                jacobian_vjp_chunk_size=32,
            ),
            device,
        )
        if method == "alqr":
            if alqr_setting is None:
                raise RuntimeError("A-LQR paper setting was not resolved before model loading")
            parameters = {
                "lambda": alqr_setting.multiplier,
                "q": alqr_setting.q,
                "r": alqr_setting.r,
                "q_final": alqr_setting.q_final,
            }
            policy = build_alqr_policy(
                calibration.dynamics,
                calibration.setpoint,
                multiplier=alqr_setting.multiplier,
                q=alqr_setting.q,
                r=alqr_setting.r,
                q_final=alqr_setting.q_final,
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
                register_hooks=lambda: register_generation_policy_hooks(model, policy),
            )
            del policy
            torch.cuda.empty_cache()
        else:
            spid_sweep = SPID_SWEEPS[behavior][key]
            for multiplier in spid_sweep.lambdas:
                parameters = {
                    "lambda": multiplier,
                    "kp": spid_sweep.kp,
                    "ki": spid_sweep.ki,
                    "kd": spid_sweep.kd,
                }
                policy = build_spid_policy(
                    calibration.setpoint,
                    multiplier=multiplier,
                    kp=spid_sweep.kp,
                    ki=spid_sweep.ki,
                    kd=spid_sweep.kd,
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
                    register_hooks=lambda policy=policy: register_generation_policy_hooks(model, policy),
                )
                del policy
                torch.cuda.empty_cache()
    elif method == "actadd":
        required = 100
        direction = _artifact(
            artifact_root / "actadd.pt",
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
        layer, strength = ACTADD_PAPER_SELECTIONS[key]
        parameters = {"layer": layer, "strength": strength}
        steerer = ActAddSteerer(direction, layer, strength)
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
            register_hooks=lambda: steerer.register(model),
            reset=steerer.reset,
        )
    elif method == "iti":
        required = ITI_FIT_SAMPLES_PER_CLASS
        fitted = _artifact(
            artifact_root / "iti.pt",
            {**common_identity, "fit_samples_per_class": required, "max_length": 50},
            lambda: fit_iti_calibration(
                model,
                tokenizer,
                undesired_texts=_texts(undesired[:required]),
                desired_texts=_texts(desired[:required]),
                batch_size=_activation_batch_size(model_id, method),
                max_length=50,
                seed=SOURCE_RANDOM_SEED,
            ),
            device,
        )
        for top_heads in ITI_SWEEPS[behavior]["top_heads"]:
            for alpha in ITI_SWEEPS[behavior]["alphas"]:
                parameters = {"top_heads": top_heads, "alpha": alpha}
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
                    register_hooks=lambda top_heads=top_heads, alpha=alpha: register_iti_hooks(
                        model, fitted, top_heads=top_heads, alpha=alpha
                    ),
                )
    elif method in {"mean_act", "linear_act", "pid_act"}:
        required = ACT_FIT_SAMPLES_PER_CLASS[behavior]
        fitted = _artifact(
            artifact_root / f"{method}.pt",
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
        for strength in ACT_SWEEPS[behavior]:
            parameters = {"strength": strength}
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
                register_hooks=lambda strength=strength: register_transport_hooks(
                    model, fitted, strength=strength
                ),
            )
    else:
        required = ODESTEER_FIT_SAMPLES[behavior]
        layer, time = ODESTEER_PAPER_SELECTIONS[behavior][key]
        fitted = _artifact(
            artifact_root / "odesteer.pt",
            {**common_identity, "fit_samples_per_class": required, "layer": layer},
            lambda: fit_odesteer_calibration(
                model,
                tokenizer,
                behavior=behavior,
                layer_index=layer,
                undesired_texts=_texts(undesired[:required]),
                desired_texts=_texts(desired[:required]),
                batch_size=_activation_batch_size(model_id, method),
            ),
            device,
        )
        parameters = {"layer": layer, "time": time}
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
            register_hooks=lambda: register_odesteer_hook(
                model, fitted, layer_index=layer, time=time
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
