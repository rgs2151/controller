"""Cache-first ID benchmark runner for frozen A-LQR comparison methods."""

from __future__ import annotations

import gc
import hashlib
import json
import random
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
    SOURCE_RANDOM_SEED,
    act_module_patterns,
    control_sweeps,
    model_key,
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
    parameters: dict,
    register_hooks: Callable[[], list[torch.utils.hooks.RemovableHandle]] | None,
    reset: Callable[[], None] | None = None,
    use_cache: bool = True,
) -> None:
    repetitions_expected = int(data["evaluation_repetitions"])
    evaluation_samples = len(_records(data, behavior, 0))
    manifest = protocol_manifest(behavior, model_id, revision, evaluation_samples)
    identity = _json_identity({
        "schema_version": 2,
        "data_fingerprint": data["fingerprint"],
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
    repetitions = []
    if output.exists():
        saved = json.loads(output.read_text())
        if saved["identity"] != identity:
            raise ValueError(f"Generation cache mismatch: {output}")
        repetitions = saved["repetitions"]
        if saved["status"] == "complete":
            if len(repetitions) != repetitions_expected:
                raise ValueError(f"Incomplete generation cache marked complete: {output}")
            return
        if saved["status"] != "partial":
            raise ValueError(f"Unknown generation cache status: {output}")
    if len(repetitions) > repetitions_expected:
        raise ValueError(f"Generation cache has too many repetitions: {output}")
    for repetition in range(len(repetitions), repetitions_expected):
        records = _records(data, behavior, repetition)
        if len(records) != evaluation_samples:
            raise ValueError("Every evaluation repetition must contain the same number of prompts")
        completions = generate_batched(
            model,
            tokenizer,
            _texts(records),
            behavior=behavior,
            batch_size=_batch_size(model_id, method),
            seed=SOURCE_RANDOM_SEED + repetition * 100_000,
            use_cache=use_cache,
            register_hooks=register_hooks,
            reset=reset,
        )
        repetitions.append(
            {
                "repetition": repetition,
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
        status = "complete" if len(repetitions) == repetitions_expected else "partial"
        _write_json(output, {"identity": identity, "status": status, "repetitions": repetitions})


def _artifact(
    path: Path,
    identity: dict,
    fit: Callable[[], object],
) -> object:
    identity = _json_identity(identity)
    metadata = path.with_suffix(".json")
    if path.exists() or metadata.exists():
        if not path.exists() or not metadata.exists():
            raise ValueError(f"Incomplete calibration cache: {path}")
        if json.loads(metadata.read_text()) != identity:
            raise ValueError(f"Calibration cache mismatch: {path}")
        return torch.load(path, map_location="cpu", weights_only=False)
    fitted = fit()
    _save_torch(path, fitted)
    _write_json(metadata, identity)
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
    data = json.loads((unit / "cache/data.json").read_text())
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

    _seed(SOURCE_RANDOM_SEED)
    model_method = "alqr" if method in {"alqr", "spid"} else method
    model, tokenizer = load_source_model(model_method, behavior, model_id, revision, device, token)
    undesired = data["calibration"][behavior]["undesired"]
    desired = data["calibration"][behavior]["desired"]
    jacobian = data["calibration"][behavior]["jacobian"]
    common_identity = {
        "schema_version": 1,
        "data_fingerprint": data["fingerprint"],
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
            parameters={},
            register_hooks=None,
            use_cache=False,
        )
    elif method in {"alqr", "spid"}:
        counts = CALIBRATION_COUNTS[behavior]
        calibration = _artifact(
            artifact_root / "control.pt",
            {**common_identity, "method": "control", "counts": counts.__dict__},
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
        )
        alqr_sweep, spid_sweep = control_sweeps(behavior, model_id)
        if method == "alqr":
            for multiplier in alqr_sweep.lambdas:
                parameters = {
                    "lambda": multiplier,
                    "q": alqr_sweep.q,
                    "r": alqr_sweep.r,
                    "q_final": alqr_sweep.q_final,
                }
                policy = build_alqr_policy(
                    calibration.dynamics,
                    calibration.setpoint,
                    multiplier=multiplier,
                    q=alqr_sweep.q,
                    r=alqr_sweep.r,
                    q_final=alqr_sweep.q_final,
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
                    parameters=parameters,
                    register_hooks=lambda policy=policy: register_generation_policy_hooks(model, policy),
                )
                del policy
                torch.cuda.empty_cache()
        else:
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
            parameters=parameters,
            register_hooks=lambda: register_odesteer_hook(
                model, fitted, layer_index=layer, time=time
            ),
        )

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
