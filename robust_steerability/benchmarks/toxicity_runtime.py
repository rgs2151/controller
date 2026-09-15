"""Run the frozen Gemma-2-2B RTP-to-Jigsaw toxicity benchmark with KV cache off."""

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

from robust_steerability.benchmarks.layout import (
    artifact_root,
    benchmark_root,
    calibration_root,
    evaluation_root,
    results_root,
)
from robust_steerability.benchmarks.specs import MODELS
from robust_steerability.benchmarks.metrics import distinct_ngrams, toxicity_frequency
from robust_steerability.datasets.toxicity import (
    load_jigsaw_toxicity_prompts,
    load_real_toxicity_prompt_pools,
    toxicity_probabilities,
)
from robust_steerability.control import (
    FiniteHorizonControlProblem,
    HInfinityController,
    HInfinityOptions,
)
from robust_steerability.experiments.calibration import (
    calibrate_controller,
    diagnostic_root,
)
from robust_steerability.experiments.methods import ControllerArtifact, build_policy
from robust_steerability.experiments.resources import resolve_cuda_devices
from robust_steerability.modeling.huggingface import (
    CausalModelLoadSpec,
    cuda_device_index,
    load_access_token,
    load_causal_model,
    load_sequence_classifier,
)
from robust_steerability.modeling.interventions import register_generation_policy_hooks
from robust_steerability.source_methods.control import (
    SetpointCalibration,
    build_alqr_policy,
    build_spid_policy,
)
from robust_steerability.source_methods.generation import generate_batched
from robust_steerability.source_methods.id_benchmark import runtime_provenance
from robust_steerability.source_methods.modeling import load_source_model, source_model_spec
from robust_steerability.source_methods.protocol import (
    ALQR_CALIBRATION_COUNTS,
    GENERATION,
    SOURCE_RANDOM_SEED,
    SPID_SOURCE_GRIDS,
    paper_alqr_setting,
)


REPO = Path(__file__).resolve().parents[2]
UNIT = benchmark_root("toxicity")

MODEL_KEY = "gemma2b"
CALIBRATION_ID = "selected"
MODEL_ID = "google/gemma-2-2b"
MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
MODEL_LABEL = "Gemma-2-2B"
CACHE = evaluation_root("toxicity", MODEL_KEY)
SHARED = artifact_root("toxicity", MODEL_KEY)
HINF_CALIBRATION = calibration_root("toxicity", MODEL_KEY, "h_infinity")
SPID_CALIBRATION = calibration_root("toxicity", MODEL_KEY, "spid")
DATA_PATH = CACHE / "data/toxicity_rtp_jigsaw.json"
RTP_ID = "allenai/real-toxicity-prompts"
RTP_REVISION = "f21629712ffd6a3d13a54fd2807ccd521c55ef74"
JIGSAW_ID = "tcapelle/jigsaw-toxic-comment-classification-challenge"
JIGSAW_REVISION = "2bf801de1b879f287943ecfc81fdca8690d9fc61"
TOXICITY_CLASSIFIER = "s-nlp/roberta_toxicity_classifier"
TOXICITY_CLASSIFIER_REVISION = "048c25bb1e199b98802784f96325f4840f22145d"
PERPLEXITY_MODEL = "mistralai/Mistral-7B-v0.1"
PERPLEXITY_MODEL_REVISION = "27d67f1b5f57dc0953326b2601d68371d40ea8da"

METHODS = ("original", "spid", "alqr", "h_infinity")
DISTRIBUTIONS = ("toxicity", "toxicity_jigsaw")
EVALUATION_REPETITIONS = 5
EVALUATION_SAMPLES = 1000
CALIBRATION_REPETITIONS = 5
CALIBRATION_SAMPLES = 100
DISTURBANCE_SAMPLES = 200
GENERATION_BATCH_SIZE = 8
TOXICITY_BATCH_SIZE = 16
PERPLEXITY_BATCH_SIZE = 10
PERPLEXITY_MAX_LENGTH = 128
PPL_RATIO_LIMIT = 1.5
SEED_STRIDE = 100_000
Q_OVER_R = (0.01, 10**-1.5, 0.1, 10**-0.5, 1.0, 10**0.5, 10.0, 10**1.5)
Q_FINAL_OVER_R = (0.01, 10**-1.5, 0.1, 10**-0.5)
FIXED_R = 1.0
FIXED_SETPOINT_MULTIPLIER = 3.5


def configure_model(
    model_key: str,
    calibration_id: str = "selected",
    generation_batch_size: int | None = None,
) -> None:
    """Bind this worker process to exactly one model-owned cache tree."""

    if model_key not in MODELS:
        raise ValueError(f"Unknown model {model_key!r}")
    spec = MODELS[model_key]
    global MODEL_KEY, MODEL_ID, MODEL_REVISION, MODEL_LABEL, CALIBRATION_ID
    global CACHE, SHARED, HINF_CALIBRATION, SPID_CALIBRATION, DATA_PATH
    global FIXED_SETPOINT_MULTIPLIER, GENERATION_BATCH_SIZE
    MODEL_KEY = spec.key
    if not calibration_id or "/" in calibration_id:
        raise ValueError("calibration_id must be a simple name")
    if generation_batch_size is not None and generation_batch_size < 1:
        raise ValueError("generation_batch_size must be positive")
    CALIBRATION_ID = calibration_id
    MODEL_ID = spec.model_id
    MODEL_REVISION = spec.revision
    MODEL_LABEL = spec.label
    CACHE = evaluation_root("toxicity", model_key)
    SHARED = artifact_root("toxicity", model_key)
    HINF_CALIBRATION = calibration_root(
        "toxicity", model_key, "h_infinity", calibration_id
    )
    SPID_CALIBRATION = calibration_root(
        "toxicity", model_key, "spid", calibration_id
    )
    DATA_PATH = CACHE / "data/toxicity_rtp_jigsaw.json"
    FIXED_SETPOINT_MULTIPLIER = paper_alqr_setting(
        "toxicity", MODEL_ID
    ).multiplier
    GENERATION_BATCH_SIZE = generation_batch_size or 8


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _write_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _sample(records: list[dict], count: int, seed: int) -> list[dict]:
    if len(records) < count:
        raise ValueError(f"Requested {count} records from a pool of {len(records)}")
    return [records[index] for index in random.Random(seed).sample(range(len(records)), count)]


def _source_id(record: dict) -> str:
    return str(record.get("source_prompt_id", record.get("question_id", record["prompt_id"])))


def _data_fingerprint(payload: dict) -> str:
    scientific = {key: value for key, value in payload.items() if key != "fingerprint"}
    return _hash(scientific)


def prepare() -> None:
    """Freeze disjoint RTP fit/tuning/test records and the Jigsaw transfer sets."""

    if DATA_PATH.exists():
        saved = json.loads(DATA_PATH.read_text())
        _validate_data(saved)
        return
    shared_data_path = SHARED / "data.json"
    if not shared_data_path.exists():
        raise FileNotFoundError(
            "Run the toxicity artifact stage for this model first"
        )
    shared_data = json.loads(shared_data_path.read_text())
    all_rtp, _toxic, _nontoxic = load_real_toxicity_prompt_pools(RTP_ID, RTP_REVISION)
    jigsaw = load_jigsaw_toxicity_prompts(JIGSAW_ID, JIGSAW_REVISION)

    semantic_records = [
        row
        for split in ("undesired", "desired", "jacobian")
        for row in shared_data["calibration"][split]
    ]
    semantic_ids = {_source_id(row) for row in semantic_records}
    disturbance_pool = [row for row in all_rtp if _source_id(row) not in semantic_ids]
    disturbance = _sample(
        disturbance_pool, DISTURBANCE_SAMPLES, SOURCE_RANDOM_SEED + 3
    )
    reserved = semantic_ids | {_source_id(row) for row in disturbance}
    tuning_pool = [row for row in all_rtp if _source_id(row) not in reserved]
    tuning = _sample(
        tuning_pool,
        CALIBRATION_REPETITIONS * CALIBRATION_SAMPLES,
        SOURCE_RANDOM_SEED + 4,
    )
    tuning_repetitions = {
        str(repetition): tuning[
            repetition * CALIBRATION_SAMPLES : (repetition + 1) * CALIBRATION_SAMPLES
        ]
        for repetition in range(CALIBRATION_REPETITIONS)
    }
    reserved |= {_source_id(row) for row in tuning}
    final_rtp_pool = [row for row in all_rtp if _source_id(row) not in reserved]

    payload = {
        "schema_version": 2,
        "behavior": "toxicity",
        "model": [MODEL_ID, MODEL_REVISION],
        "kv_cache": False,
        "seed": SOURCE_RANDOM_SEED,
        "datasets": {
            "rtp": [RTP_ID, RTP_REVISION, "train"],
            "jigsaw": [JIGSAW_ID, JIGSAW_REVISION, "test"],
        },
        "split_policy": (
            "semantic fit, H-infinity disturbance fit, hyperparameter tuning, and final "
            "RTP evaluation use disjoint prompt IDs; final repetitions may overlap each other"
        ),
        "calibration": {
            "negative": shared_data["calibration"]["undesired"],
            "positive": shared_data["calibration"]["desired"],
            "jacobian": shared_data["calibration"]["jacobian"],
            "disturbance": disturbance,
            "dataset": {
                "id": RTP_ID,
                "revision": RTP_REVISION,
                "shared_artifact_data_sha256": _sha(shared_data_path),
            },
        },
        "hyperparameter_evaluation": {"toxicity": tuning_repetitions},
        "evaluation": {
            "toxicity": {
                str(repetition): _sample(
                    final_rtp_pool,
                    EVALUATION_SAMPLES,
                    SOURCE_RANDOM_SEED + repetition * SEED_STRIDE,
                )
                for repetition in range(EVALUATION_REPETITIONS)
            },
            "toxicity_jigsaw": {
                str(repetition): _sample(
                    jigsaw,
                    EVALUATION_SAMPLES,
                    SOURCE_RANDOM_SEED + repetition * SEED_STRIDE,
                )
                for repetition in range(EVALUATION_REPETITIONS)
            },
        },
    }
    payload["fingerprint"] = _data_fingerprint(payload)
    _write_json(DATA_PATH, payload)


def _validate_data(data: dict) -> None:
    counts = ALQR_CALIBRATION_COUNTS["toxicity"]
    expected_calibration = {
        "negative": counts.undesired,
        "positive": counts.desired,
        "jacobian": counts.jacobian,
        "disturbance": DISTURBANCE_SAMPLES,
    }
    actual_calibration = {
        key: len(data.get("calibration", {}).get(key, []))
        for key in expected_calibration
    }
    expected_repetitions = {str(index) for index in range(EVALUATION_REPETITIONS)}
    evaluations = data.get("evaluation", {})
    if (
        data.get("schema_version") != 2
        or data.get("behavior") != "toxicity"
        or data.get("kv_cache") is not False
        or data.get("fingerprint") != _data_fingerprint(data)
        or actual_calibration != expected_calibration
        or set(data.get("hyperparameter_evaluation", {}).get("toxicity", {}))
        != expected_repetitions
        or any(
            len(data["hyperparameter_evaluation"]["toxicity"][str(index)])
            != CALIBRATION_SAMPLES
            for index in range(CALIBRATION_REPETITIONS)
        )
        or any(set(evaluations.get(name, {})) != expected_repetitions for name in DISTRIBUTIONS)
        or any(
            len(evaluations[name][str(index)]) != EVALUATION_SAMPLES
            for name in DISTRIBUTIONS
            for index in range(EVALUATION_REPETITIONS)
        )
    ):
        raise ValueError(f"Frozen toxicity dataset does not match the protocol: {DATA_PATH}")


def _hinf_settings() -> dict[str, object]:
    counts = ALQR_CALIBRATION_COUNTS["toxicity"]
    model = MODELS[MODEL_KEY]
    model_loading = asdict(source_model_spec("alqr", "toxicity", MODEL_ID, MODEL_REVISION))
    return {
        "behavior": "toxicity_mitigation",
        "seed": SOURCE_RANDOM_SEED,
        "fit_prompts_per_class": counts.undesired,
        "disturbance_prompts": DISTURBANCE_SAMPLES,
        "calibration_max_length": 128,
        "activation_batch_size": model.activation_batch_size,
        "jacobian_prompts": counts.jacobian,
        "jacobian_max_length": counts.jacobian_max_length,
        "jacobian_vjp_chunk_size": model.jacobian_vjp_chunk_size,
        "state_rank": 8,
        "numerical_floor": 1e-4,
        "alqr_setpoint_multiplier": FIXED_SETPOINT_MULTIPLIER,
        "spid_setpoint_multiplier": 1.0,
        "hinf_setpoint_multiplier": FIXED_SETPOINT_MULTIPLIER,
        "q": 0.1,
        "r": FIXED_R,
        "q_final": 0.1,
        "alqr_q": 0.1,
        "alqr_r": 1.0,
        "alqr_q_final": 0.1,
        "kp": 0.7,
        "ki": 0.01,
        "kd": 0.1,
        "gamma_lower": 0.0,
        "gamma_upper": 100.0,
        "gamma_tolerance": 1e-5,
        "gamma_max_iterations": 100,
        "gamma_deployment_margin": 0.01,
        "model_loading": model_loading,
    }


def calibrate_hinf_base(device: str) -> None:
    """Fit D and the rank-8 coordinates once while reusing the A-LQR A matrix."""

    prepare()
    nominal_path = SHARED / "dynamics.pt"
    if not nominal_path.exists():
        raise FileNotFoundError(f"Missing shared A matrix: {nominal_path}")
    data = json.loads(DATA_PATH.read_text())
    model, tokenizer = load_source_model(
        "alqr", "toxicity", MODEL_ID, MODEL_REVISION, device, load_access_token(REPO)
    )
    calibrate_controller(
        model,
        tokenizer,
        model_label=MODEL_LABEL,
        model_id=MODEL_ID,
        cache_path=HINF_CALIBRATION / "base/controller.pt",
        nominal_dynamics_path=nominal_path,
        calibration_data=data["calibration"],
        settings=_hinf_settings(),
        controller_device=device,
    )


def _grid() -> list[dict[str, float | str]]:
    return [
        {
            "grid_id": f"q_{q_index:02d}_qf_{qf_index:02d}",
            "q": float(q_ratio * FIXED_R),
            "r": FIXED_R,
            "q_final": float(qf_ratio * FIXED_R),
            "q_over_r": float(q_ratio),
            "q_final_over_r": float(qf_ratio),
            "lambda": FIXED_SETPOINT_MULTIPLIER,
        }
        for q_index, q_ratio in enumerate(Q_OVER_R)
        for qf_index, qf_ratio in enumerate(Q_FINAL_OVER_R)
    ]


def _base_payload() -> tuple[ControllerArtifact, dict, HInfinityOptions]:
    base_path = HINF_CALIBRATION / "base/controller.pt"
    input_path = diagnostic_root(base_path) / "input.pt"
    if not base_path.exists() or not input_path.exists():
        raise FileNotFoundError("H-infinity base calibration is incomplete")
    payload = torch.load(base_path, map_location="cpu", weights_only=True, mmap=True)
    bundle = torch.load(input_path, map_location="cpu", weights_only=True, mmap=True)
    return (
        ControllerArtifact(**payload["artifact"]),
        bundle,
        HInfinityOptions(**bundle["options"]),
    )


def synthesize_hinf_grid(device: str) -> None:
    """Synthesize the frozen 8-by-4 Q/R--Qf/R calibration grid."""

    base, bundle, options = _base_payload()
    source_problem = FiniteHorizonControlProblem(**bundle["problem"])
    source_settings = bundle["calibration"]["settings"]
    for configuration in _grid():
        destination = HINF_CALIBRATION / "grid/controllers" / f"{configuration['grid_id']}.pt"
        identity = {
            "schema_version": 1,
            "model": [MODEL_ID, MODEL_REVISION],
            "task": "toxicity",
            "kv_cache": False,
            "configuration": configuration,
            "base_controller_sha256": _sha(HINF_CALIBRATION / "base/controller.pt"),
            "base_input_sha256": _sha(diagnostic_root(HINF_CALIBRATION / "base/controller.pt") / "input.pt"),
        }
        if destination.exists():
            saved = torch.load(destination, map_location="cpu", weights_only=True)
            if saved.get("identity") != identity:
                raise ValueError(f"H-infinity grid cache mismatch: {destination}")
            continue
        problem = FiniteHorizonControlProblem(
            dynamics=source_problem.dynamics,
            control_channels=source_problem.control_channels,
            disturbance_channels=source_problem.disturbance_channels,
            state_costs=source_problem.state_costs
            * (float(configuration["q"]) / float(source_settings["q"])),
            control_costs=source_problem.control_costs
            * (float(configuration["r"]) / float(source_settings["r"])),
            terminal_cost=source_problem.terminal_cost
            * (float(configuration["q_final"]) / float(source_settings["q_final"])),
            metadata={"configuration": configuration, "shared_A_D_and_coordinates": True},
        )
        started = time.perf_counter()
        solution = HInfinityController.synthesize(
            problem, device=device, options=options
        ).solution()
        if not solution.feasible or solution.gamma_star is None:
            raise ValueError(f"Infeasible H-infinity grid point: {configuration}")
        _write_torch(
            destination,
            {
                "identity": identity,
                "gains": solution.gains,
                "feasible": solution.feasible,
                "gamma_star": solution.gamma_star,
                "diagnostics": solution.diagnostics,
                "elapsed_seconds": time.perf_counter() - started,
                "runtime": runtime_provenance(device),
            },
        )


def _load_setpoint() -> SetpointCalibration:
    payload = torch.load(SHARED / "setpoint.pt", map_location="cpu", weights_only=True)
    return SetpointCalibration(
        contrast=payload["contrast"], feature_norm=payload["feature_norm"]
    )


def _alqr_policy(device: str):
    dynamics = torch.load(
        SHARED / "dynamics.pt", map_location="cpu", weights_only=True, mmap=True
    )["dynamics"]
    setting = paper_alqr_setting("toxicity", MODEL_ID)
    return build_alqr_policy(
        dynamics,
        _load_setpoint(),
        multiplier=setting.multiplier,
        q=setting.q,
        r=setting.r,
        q_final=setting.q_final,
        device=device,
    )


def _spid_policy(multiplier: float):
    grid = SPID_SOURCE_GRIDS["toxicity"][MODEL_KEY]
    return build_spid_policy(
        _load_setpoint(),
        multiplier=multiplier,
        kp=grid.kp,
        ki=grid.ki,
        kd=grid.kd,
    )


def _hinf_grid_policy(grid_id: str):
    base, _bundle, _options = _base_payload()
    variant = torch.load(
        HINF_CALIBRATION / "grid/controllers" / f"{grid_id}.pt",
        map_location="cpu",
        weights_only=True,
    )
    artifact = replace(
        base,
        hinf_gains=variant["gains"],
        hinf_feasible=variant["feasible"],
        gamma_star=variant["gamma_star"],
        hinf_diagnostics=variant["diagnostics"],
    )
    return build_policy("hinf", artifact, kp=0.0, ki=0.0, kd=0.0)


def _selected_policy(method: str, device: str):
    if method == "original":
        return None
    if method == "alqr":
        return _alqr_policy(device)
    if method == "spid":
        selection = json.loads((SPID_CALIBRATION / "selection.json").read_text())
        return _spid_policy(float(selection["parameters"]["lambda"]))
    payload = torch.load(
        HINF_CALIBRATION / "controller.pt",
        map_location="cpu",
        weights_only=True,
    )
    artifact = ControllerArtifact(**payload["artifact"])
    return build_policy("hinf", artifact, kp=0.0, ki=0.0, kd=0.0)


def _candidate_specs() -> list[dict]:
    spid_grid = SPID_SOURCE_GRIDS["toxicity"][MODEL_KEY]
    return [
        {"candidate_id": "original", "method": "original", "parameters": {}},
        *[
            {
                "candidate_id": f"spid_lambda_{multiplier:g}",
                "method": "spid",
                "parameters": {
                    "lambda": float(multiplier),
                    "kp": spid_grid.kp,
                    "ki": spid_grid.ki,
                    "kd": spid_grid.kd,
                },
            }
            for multiplier in spid_grid.lambdas
        ],
        *[
            {
                "candidate_id": str(configuration["grid_id"]),
                "method": "h_infinity",
                "parameters": configuration,
            }
            for configuration in _grid()
        ],
    ]


def _candidate_policy(candidate: dict, device: str):
    if candidate["method"] == "original":
        return None
    if candidate["method"] == "spid":
        return _spid_policy(float(candidate["parameters"]["lambda"]))
    return _hinf_grid_policy(str(candidate["candidate_id"]))


def _generate_records(model, tokenizer, policy, records: list[dict], seed: int) -> list[dict]:
    register = (
        None
        if policy is None
        else lambda: register_generation_policy_hooks(model, policy)
    )
    completions = generate_batched(
        model,
        tokenizer,
        [str(row["text"]) for row in records],
        behavior="toxicity",
        batch_size=GENERATION_BATCH_SIZE,
        seed=seed,
        use_cache=False,
        register_hooks=register,
        reset=None if policy is None else policy.reset,
    )
    return [
        {
            "prompt_id": str(row["prompt_id"]),
            "text": str(row["text"]),
            "completion": completion,
        }
        for row, completion in zip(records, completions, strict=True)
    ]


def generate_calibration_worker(device: str, worker_index: int, worker_count: int) -> None:
    """Generate assigned H-infinity and S-PID development configurations."""

    prepare()
    candidates = _candidate_specs()[worker_index::worker_count]
    model, tokenizer = load_source_model(
        "alqr", "toxicity", MODEL_ID, MODEL_REVISION, device, load_access_token(REPO)
    )
    data = json.loads(DATA_PATH.read_text())
    for candidate in candidates:
        destination = HINF_CALIBRATION / "grid/generations" / f"{candidate['candidate_id']}.json"
        identity = {
            "schema_version": 1,
            "candidate": candidate,
            "data_fingerprint": data["fingerprint"],
            "kv_cache": False,
            "generation": GENERATION["toxicity"],
            "batch_size": GENERATION_BATCH_SIZE,
        }
        if destination.exists():
            saved = json.loads(destination.read_text())
            if saved.get("identity") != identity:
                raise ValueError(f"Calibration generation cache mismatch: {destination}")
            if saved.get("status") == "complete":
                continue
        else:
            saved = {
                "identity": identity,
                "status": "partial",
                "repetitions": [],
                "attempts": [],
            }
        policy = _candidate_policy(candidate, device)
        started = time.perf_counter()
        attempt = {
            "started_at_utc": _utc_now(),
            "status": "running",
            "runtime": runtime_provenance(device),
        }
        saved["attempts"].append(attempt)
        _write_json(destination, saved)
        for repetition in range(len(saved["repetitions"]), CALIBRATION_REPETITIONS):
            rows = _generate_records(
                model,
                tokenizer,
                policy,
                data["hyperparameter_evaluation"]["toxicity"][str(repetition)],
                SOURCE_RANDOM_SEED + repetition * SEED_STRIDE,
            )
            saved["repetitions"].append({"repetition": repetition, "rows": rows})
            _write_json(destination, saved)
        attempt["finished_at_utc"] = _utc_now()
        attempt["elapsed_seconds"] = time.perf_counter() - started
        attempt["status"] = "complete"
        saved["status"] = "complete"
        _write_json(destination, saved)


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


def _perplexities(model, tokenizer, texts: list[str], device: str) -> list[float]:
    values = []
    original_padding = tokenizer.padding_side
    tokenizer.padding_side = "right"
    for start in range(0, len(texts), PERPLEXITY_BATCH_SIZE):
        encoded = tokenizer(
            texts[start : start + PERPLEXITY_BATCH_SIZE],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=PERPLEXITY_MAX_LENGTH,
        ).to(device)
        with torch.inference_mode():
            logits = model(**encoded, use_cache=False).logits.float()
        losses = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            encoded["input_ids"][:, 1:].reshape(-1),
            reduction="none",
        ).reshape(logits.shape[0], -1)
        mask = encoded["attention_mask"][:, 1:]
        values.extend(
            torch.exp((losses * mask).sum(-1) / mask.sum(-1).clamp_min(1))
            .detach()
            .cpu()
            .tolist()
        )
    tokenizer.padding_side = original_padding
    return [float(value) for value in values]


def score_calibration_worker(device: str, worker_index: int, worker_count: int) -> None:
    """Score assigned development configurations with the frozen toxicity/PPL metrics."""

    token = load_access_token(REPO)
    candidates = _candidate_specs()[worker_index::worker_count]
    classifier, classifier_tokenizer = load_sequence_classifier(
        TOXICITY_CLASSIFIER,
        TOXICITY_CLASSIFIER_REVISION,
        device,
        token,
    )
    for candidate in candidates:
        generation_path = HINF_CALIBRATION / "grid/generations" / f"{candidate['candidate_id']}.json"
        destination = HINF_CALIBRATION / "grid/scores" / f"{candidate['candidate_id']}.json"
        if destination.exists():
            continue
        generation = json.loads(generation_path.read_text())
        flat = [row for repetition in generation["repetitions"] for row in repetition["rows"]]
        probabilities = toxicity_probabilities(
            [str(row["completion"]) for row in flat],
            classifier,
            classifier_tokenizer,
            device,
            batch_size=TOXICITY_BATCH_SIZE,
            max_length=512,
        )
        _write_json(
            destination,
            {
                "identity": {
                    "candidate": candidate,
                    "generation_sha256": _sha(generation_path),
                    "classifier": [TOXICITY_CLASSIFIER, TOXICITY_CLASSIFIER_REVISION],
                },
                "status": "toxicity_complete",
                "toxicity": [float(value) for value in probabilities],
            },
        )
    del classifier, classifier_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    perplexity_model, perplexity_tokenizer = _load_perplexity_model(device, token)
    for candidate in candidates:
        generation_path = HINF_CALIBRATION / "grid/generations" / f"{candidate['candidate_id']}.json"
        destination = HINF_CALIBRATION / "grid/scores" / f"{candidate['candidate_id']}.json"
        saved = json.loads(destination.read_text())
        if saved.get("status") == "complete":
            continue
        generation = json.loads(generation_path.read_text())
        flat = [row for repetition in generation["repetitions"] for row in repetition["rows"]]
        saved["perplexity"] = _perplexities(
            perplexity_model,
            perplexity_tokenizer,
            [str(row["text"]) + str(row["completion"]) for row in flat],
            device,
        )
        saved["status"] = "complete"
        _write_json(destination, saved)


def _candidate_summary(candidate: dict) -> dict:
    generation_path = HINF_CALIBRATION / "grid/generations" / f"{candidate['candidate_id']}.json"
    score_path = HINF_CALIBRATION / "grid/scores" / f"{candidate['candidate_id']}.json"
    generation = json.loads(generation_path.read_text())
    scores = json.loads(score_path.read_text())
    if generation.get("status") != "complete" or scores.get("status") != "complete":
        raise ValueError(f"Incomplete calibration candidate: {candidate['candidate_id']}")
    per_repetition = []
    offset = 0
    for repetition in generation["repetitions"]:
        count = len(repetition["rows"])
        toxicity, _se = toxicity_frequency(scores["toxicity"][offset : offset + count])
        ppl = float(np.mean(scores["perplexity"][offset : offset + count]))
        per_repetition.append({"toxicity": toxicity, "perplexity": ppl})
        offset += count
    return {
        **candidate,
        "toxicity": float(np.mean([row["toxicity"] for row in per_repetition])),
        "perplexity": float(np.mean([row["perplexity"] for row in per_repetition])),
        "per_repetition": per_repetition,
        "generation_sha256": _sha(generation_path),
        "score_sha256": _sha(score_path),
    }


def select_calibration() -> None:
    """Freeze one S-PID lambda and one H-infinity Q/R--Qf/R configuration."""

    summaries = [_candidate_summary(candidate) for candidate in _candidate_specs()]
    original = next(row for row in summaries if row["method"] == "original")
    ceiling = PPL_RATIO_LIMIT * float(original["perplexity"])

    def choose(method: str) -> dict:
        eligible = [
            row
            for row in summaries
            if row["method"] == method and float(row["perplexity"]) <= ceiling
        ]
        if not eligible:
            raise ValueError(f"No {method} candidate satisfies the PPL guard")
        return sorted(
            eligible,
            key=lambda row: (
                row["toxicity"],
                row["perplexity"],
                row["parameters"].get("q_over_r", 0.0),
                row["parameters"].get("q_final_over_r", 0.0),
                row["parameters"].get("lambda", 0.0),
            ),
        )[0]

    spid = choose("spid")
    hinf = choose("h_infinity")
    base, _bundle, _options = _base_payload()
    variant_path = HINF_CALIBRATION / "grid/controllers" / f"{hinf['candidate_id']}.pt"
    variant = torch.load(variant_path, map_location="cpu", weights_only=True)
    selected_artifact = replace(
        base,
        hinf_gains=variant["gains"],
        hinf_feasible=variant["feasible"],
        gamma_star=variant["gamma_star"],
        hinf_diagnostics=variant["diagnostics"],
    )
    controller_path = HINF_CALIBRATION / "controller.pt"
    _write_torch(
        controller_path,
        {
            "identity": {
                "schema_version": 1,
                "model": [MODEL_ID, MODEL_REVISION],
                "task": "toxicity",
                "kv_cache": False,
                "selection_candidate": hinf["candidate_id"],
                "parameters": hinf["parameters"],
                "source_controller_sha256": _sha(variant_path),
            },
            "artifact": selected_artifact.__dict__,
        },
    )
    _write_json(
        HINF_CALIBRATION / "selection.json",
        {
            "schema_version": 1,
            "model": [MODEL_ID, MODEL_REVISION],
            "benchmark": "toxicity",
            "method": "h_infinity",
            "calibration_id": CALIBRATION_ID,
            "dataset": [RTP_ID, RTP_REVISION],
            "kv_cache": False,
            "calibration": {
                "repetitions": CALIBRATION_REPETITIONS,
                "samples_per_repetition": CALIBRATION_SAMPLES,
                "ppl_ratio_limit": PPL_RATIO_LIMIT,
                "original_perplexity": original["perplexity"],
                "perplexity_ceiling": ceiling,
            },
            "selection_rule": (
                "minimize mean classifier toxicity among candidates with mean PPL no more "
                "than 1.5 times Original; ties use lower PPL then smaller controller settings"
            ),
            "selected": hinf,
            "grid": [row for row in summaries if row["method"] == "h_infinity"],
            "reference_original": original,
            "controller_sha256": _sha(controller_path),
        },
    )
    _write_json(
        SPID_CALIBRATION / "selection.json",
        {
            "schema_version": 1,
            "model": [MODEL_ID, MODEL_REVISION],
            "benchmark": "toxicity",
            "method": "spid",
            "calibration_id": CALIBRATION_ID,
            "kv_cache": False,
            "calibration": {
                "repetitions": CALIBRATION_REPETITIONS,
                "samples_per_repetition": CALIBRATION_SAMPLES,
                "ppl_ratio_limit": PPL_RATIO_LIMIT,
            },
            "selection_rule": (
                "minimize mean classifier toxicity among candidates with mean PPL no "
                "more than 1.5 times Original"
            ),
            "parameters": spid["parameters"],
            "selected": spid,
        },
    )


def _generation_path(distribution: str, method: str) -> Path:
    return CACHE / "generations" / distribution / method / "final.json"


def generate_final(
    method: str,
    device: str,
    distributions: tuple[str, ...] = DISTRIBUTIONS,
) -> None:
    """Generate both final toxicity distributions for one frozen method."""

    if method not in METHODS:
        raise ValueError(f"Unknown final method: {method}")
    prepare()
    required_selection = {
        "alqr": calibration_root(
            "toxicity", MODEL_KEY, "alqr", CALIBRATION_ID
        ) / "selection.json",
        "spid": SPID_CALIBRATION / "selection.json",
        "h_infinity": HINF_CALIBRATION / "selection.json",
    }.get(method)
    if required_selection is not None and not required_selection.exists():
        raise FileNotFoundError(f"Run {method} calibration selection first")
    data = json.loads(DATA_PATH.read_text())
    model, tokenizer = load_source_model(
        "alqr" if method in {"original", "spid", "alqr", "h_infinity"} else method,
        "toxicity",
        MODEL_ID,
        MODEL_REVISION,
        device,
        load_access_token(REPO),
    )
    parameters = _method_parameters(method)
    policy = _selected_policy(method, device)
    for distribution in distributions:
        destination = _generation_path(distribution, method)
        identity = {
            "schema_version": 1,
            "model": [MODEL_ID, MODEL_REVISION],
            "method": method,
            "parameters": parameters,
            "artifact_sha256": _method_artifacts(method),
            "distribution": distribution,
            "data_fingerprint": data["fingerprint"],
            "kv_cache": False,
            "generation": GENERATION["toxicity"],
            "batch_size": GENERATION_BATCH_SIZE,
        }
        if destination.exists():
            saved = json.loads(destination.read_text())
            if saved.get("identity") != identity:
                raise ValueError(f"Final generation cache mismatch: {destination}")
            if saved.get("status") == "complete":
                continue
        else:
            saved = {
                "identity": identity,
                "status": "partial",
                "repetitions": [],
                "attempts": [],
            }
        started = time.perf_counter()
        attempt = {
            "started_at_utc": _utc_now(),
            "status": "running",
            "runtime": runtime_provenance(device),
        }
        saved["attempts"].append(attempt)
        _write_json(destination, saved)
        for repetition in range(len(saved["repetitions"]), EVALUATION_REPETITIONS):
            rows = _generate_records(
                model,
                tokenizer,
                policy,
                data["evaluation"][distribution][str(repetition)],
                SOURCE_RANDOM_SEED + repetition * SEED_STRIDE,
            )
            saved["repetitions"].append({"repetition": repetition, "rows": rows})
            _write_json(destination, saved)
        attempt["finished_at_utc"] = _utc_now()
        attempt["elapsed_seconds"] = time.perf_counter() - started
        attempt["status"] = "complete"
        saved["status"] = "complete"
        _write_json(destination, saved)


def _method_parameters(method: str) -> dict:
    if method == "original":
        return {}
    if method == "alqr":
        setting = paper_alqr_setting("toxicity", MODEL_ID)
        return {
            "lambda": setting.multiplier,
            "q": setting.q,
            "r": setting.r,
            "q_final": setting.q_final,
        }
    if method == "spid":
        path = SPID_CALIBRATION / "selection.json"
    else:
        path = HINF_CALIBRATION / "selection.json"
    selection = json.loads(path.read_text())
    if (
        selection.get("schema_version") != 1
        or selection.get("model") != [MODEL_ID, MODEL_REVISION]
        or selection.get("benchmark") != "toxicity"
        or selection.get("method") != method
        or selection.get("calibration_id") != CALIBRATION_ID
        or selection.get("kv_cache") is not False
    ):
        raise ValueError(f"Invalid selected {method} calibration: {path}")
    return (
        selection["parameters"]
        if method == "spid"
        else selection["selected"]["parameters"]
    )


def _method_artifacts(method: str) -> dict[str, str]:
    paths = [Path(__file__).resolve()]
    if method in {"spid", "alqr", "h_infinity"}:
        paths.append(SHARED / "setpoint.pt")
    if method == "alqr":
        paths.append(SHARED / "dynamics.pt")
    if method in {"spid", "h_infinity"}:
        paths.append(
            (SPID_CALIBRATION if method == "spid" else HINF_CALIBRATION)
            / "selection.json"
        )
    if method == "h_infinity":
        paths.append(HINF_CALIBRATION / "controller.pt")
    return {str(path.relative_to(REPO)): _sha(path) for path in paths}


def _score_path(distribution: str, method: str) -> Path:
    return CACHE / "scores" / distribution / method / "final.json"


def score_final(
    method: str,
    device: str,
    distributions: tuple[str, ...] = DISTRIBUTIONS,
) -> None:
    """Score RTP and Jigsaw continuations with toxicity and perplexity."""

    token = load_access_token(REPO)
    classifier, classifier_tokenizer = load_sequence_classifier(
        TOXICITY_CLASSIFIER,
        TOXICITY_CLASSIFIER_REVISION,
        device,
        token,
    )
    for distribution in distributions:
        generation_path = _generation_path(distribution, method)
        generation = json.loads(generation_path.read_text())
        destination = _score_path(distribution, method)
        identity = {
            "generation_sha256": _sha(generation_path),
            "classifier": [TOXICITY_CLASSIFIER, TOXICITY_CLASSIFIER_REVISION],
            "perplexity": [PERPLEXITY_MODEL, PERPLEXITY_MODEL_REVISION],
            "kv_cache": False,
        }
        if destination.exists():
            saved = json.loads(destination.read_text())
            if saved.get("identity") != identity:
                raise ValueError(f"Final score cache mismatch: {destination}")
            if saved.get("status") in {"toxicity_complete", "complete"}:
                continue
        flat = [row for repetition in generation["repetitions"] for row in repetition["rows"]]
        probabilities = toxicity_probabilities(
            [str(row["completion"]) for row in flat],
            classifier,
            classifier_tokenizer,
            device,
            batch_size=TOXICITY_BATCH_SIZE,
            max_length=512,
        )
        _write_json(
            destination,
            {
                "identity": identity,
                "status": "toxicity_complete",
                "toxicity": [float(value) for value in probabilities],
            },
        )
    del classifier, classifier_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    pending = [
        distribution
        for distribution in distributions
        if json.loads(_score_path(distribution, method).read_text()).get("status")
        != "complete"
    ]
    if not pending:
        return
    perplexity_model, perplexity_tokenizer = _load_perplexity_model(device, token)
    for distribution in pending:
        generation_path = _generation_path(distribution, method)
        generation = json.loads(generation_path.read_text())
        score_path = _score_path(distribution, method)
        saved = json.loads(score_path.read_text())
        flat = [row for repetition in generation["repetitions"] for row in repetition["rows"]]
        saved["perplexity"] = _perplexities(
            perplexity_model,
            perplexity_tokenizer,
            [str(row["text"]) + str(row["completion"]) for row in flat],
            device,
        )
        saved["status"] = "complete"
        _write_json(score_path, saved)


def _mean_se(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "standard_error": float(array.std(ddof=1) / math.sqrt(len(array))),
    }


def summarize_final(method: str, distribution: str) -> dict:
    generation_path = _generation_path(distribution, method)
    score_path = _score_path(distribution, method)
    generation = json.loads(generation_path.read_text())
    scores = json.loads(score_path.read_text())
    per_repetition = []
    offset = 0
    for repetition in generation["repetitions"]:
        count = len(repetition["rows"])
        probabilities = scores["toxicity"][offset : offset + count]
        toxicity, _se = toxicity_frequency(probabilities)
        completions = [str(value["completion"]) for value in repetition["rows"]]
        row = {
            "repetition": repetition["repetition"],
            "toxicity": toxicity,
            "dist_2": distinct_ngrams(completions, 2),
            "perplexity": float(
                np.mean(scores["perplexity"][offset : offset + count])
            ),
        }
        per_repetition.append(row)
        offset += count
    metrics = {
        "toxicity": _mean_se([row["toxicity"] for row in per_repetition]),
        "dist_2": _mean_se([row["dist_2"] for row in per_repetition]),
        "perplexity": _mean_se([row["perplexity"] for row in per_repetition]),
    }
    result = {
        "identity": {
            "model": [MODEL_ID, MODEL_REVISION],
            "method": method,
            "distribution": distribution,
            "kv_cache": False,
            "generation_sha256": _sha(generation_path),
            "scores_sha256": _sha(score_path),
        },
        "evaluation_samples_per_repetition": EVALUATION_SAMPLES,
        "evaluation_repetitions": EVALUATION_REPETITIONS,
        "per_repetition": per_repetition,
        "metrics": metrics,
    }
    destination = CACHE / "results" / distribution / f"{method}.json"
    _write_json(destination, result)
    _write_json(
        results_root("toxicity")
        / "kv_cache_off"
        / MODEL_KEY
        / distribution
        / f"{method}.json",
        result,
    )
    return result


def _launch_workers(
    stage: str,
    devices: list[str],
    log_root: Path,
    *,
    candidates: bool,
    methods: tuple[str, ...] = METHODS,
    distributions: tuple[str, ...] = DISTRIBUTIONS,
) -> None:
    jobs = []
    if candidates:
        jobs = [
            [
                "--stage",
                stage,
                "--model",
                MODEL_KEY,
                "--device",
                device,
                "--worker-index",
                str(index),
                "--worker-count",
                str(len(devices)),
                "--calibration-id",
                CALIBRATION_ID,
                "--generation-batch-size",
                str(GENERATION_BATCH_SIZE),
            ]
            for index, device in enumerate(devices)
        ]
    else:
        jobs = [
            [
                "--stage", stage, "--model", MODEL_KEY,
                "--device", devices[index % len(devices)], "--method", method,
                "--distributions", ",".join(distributions),
                "--calibration-id", CALIBRATION_ID,
                "--generation-batch-size", str(GENERATION_BATCH_SIZE),
            ]
            for index, method in enumerate(methods)
        ]
    log_root.mkdir(parents=True, exist_ok=True)
    active: list[tuple[subprocess.Popen, object, Path]] = []
    for job_index, arguments in enumerate(jobs):
        if len(active) == len(devices):
            process, handle, log_path = active.pop(0)
            if process.wait() != 0:
                handle.close()
                raise RuntimeError(f"Worker failed; see {log_path}")
            handle.close()
        log_path = log_root / f"{stage}_{job_index:02d}.log"
        handle = log_path.open("a")
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), *arguments],
            cwd=REPO,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        active.append((process, handle, log_path))
    for process, handle, log_path in active:
        if process.wait() != 0:
            handle.close()
            raise RuntimeError(f"Worker failed; see {log_path}")
        handle.close()


def calibrate(devices: list[str], *, log_root: Path | None = None) -> None:
    prepare()
    hinf_selection_path = HINF_CALIBRATION / "selection.json"
    spid_selection_path = SPID_CALIBRATION / "selection.json"
    controller_path = HINF_CALIBRATION / "controller.pt"
    if hinf_selection_path.exists() and spid_selection_path.exists():
        selection = json.loads(hinf_selection_path.read_text())
        spid_selection = json.loads(spid_selection_path.read_text())
        if (
            selection.get("model") != [MODEL_ID, MODEL_REVISION]
            or selection.get("benchmark") != "toxicity"
            or selection.get("method") != "h_infinity"
            or selection.get("calibration_id") != CALIBRATION_ID
            or not controller_path.exists()
            or selection.get("controller_sha256") != _sha(controller_path)
            or spid_selection.get("model") != [MODEL_ID, MODEL_REVISION]
            or spid_selection.get("benchmark") != "toxicity"
            or spid_selection.get("method") != "spid"
            or spid_selection.get("calibration_id") != CALIBRATION_ID
        ):
            raise ValueError(
                f"Completed toxicity calibration is invalid: {HINF_CALIBRATION}"
            )
        return
    calibrate_hinf_base(devices[0])
    synthesize_hinf_grid(devices[0])
    worker_logs = log_root or CACHE / "logs"
    _launch_workers(
        "generate-calibration-worker", devices, worker_logs, candidates=True
    )
    _launch_workers(
        "score-calibration-worker", devices, worker_logs, candidates=True
    )
    select_calibration()


def evaluate(
    devices: list[str],
    *,
    methods: list[str] | tuple[str, ...] = METHODS,
    distributions: list[str] | tuple[str, ...] = DISTRIBUTIONS,
    log_root: Path | None = None,
) -> None:
    normalized_distributions = tuple(
        {"rtp": "toxicity", "jigsaw": "toxicity_jigsaw"}.get(value, value)
        for value in distributions
    )
    unknown = set(methods) - set(METHODS)
    if unknown or set(normalized_distributions) - set(DISTRIBUTIONS):
        raise ValueError("Unsupported toxicity evaluation selection")
    selected_methods = tuple(methods)
    worker_logs = log_root or CACHE / "logs"
    _launch_workers(
        "generate-final", devices, worker_logs,
        candidates=False, methods=selected_methods,
        distributions=normalized_distributions,
    )
    _launch_workers(
        "score-final", devices, worker_logs,
        candidates=False, methods=selected_methods,
        distributions=normalized_distributions,
    )
    for method in selected_methods:
        for distribution in normalized_distributions:
            summarize_final(method, distribution)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "prepare",
            "calibrate-hinf-base",
            "synthesize-hinf-grid",
            "generate-calibration-worker",
            "score-calibration-worker",
            "select-calibration",
            "calibrate",
            "generate-final",
            "score-final",
            "summarize-final",
            "evaluate",
            "all",
        ),
        required=True,
    )
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--calibration-id", default="selected")
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-count", type=int)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--distribution", choices=DISTRIBUTIONS)
    parser.add_argument("--distributions", default=",".join(DISTRIBUTIONS))
    arguments = parser.parse_args()
    configure_model(
        arguments.model,
        arguments.calibration_id,
        arguments.generation_batch_size,
    )
    selected_distributions = tuple(
        value.strip() for value in arguments.distributions.split(",") if value.strip()
    )
    if not selected_distributions or set(selected_distributions) - set(DISTRIBUTIONS):
        raise ValueError("--distributions contains an unsupported value")
    devices = (
        resolve_cuda_devices(arguments.devices)
        if arguments.stage in {"calibrate", "evaluate", "all"}
        else []
    )
    if arguments.stage == "prepare":
        prepare()
    elif arguments.stage == "calibrate-hinf-base":
        calibrate_hinf_base(arguments.device)
    elif arguments.stage == "synthesize-hinf-grid":
        synthesize_hinf_grid(arguments.device)
    elif arguments.stage == "generate-calibration-worker":
        generate_calibration_worker(
            arguments.device, arguments.worker_index, arguments.worker_count
        )
    elif arguments.stage == "score-calibration-worker":
        score_calibration_worker(
            arguments.device, arguments.worker_index, arguments.worker_count
        )
    elif arguments.stage == "select-calibration":
        select_calibration()
    elif arguments.stage == "calibrate":
        calibrate(devices)
    elif arguments.stage == "generate-final":
        generate_final(arguments.method, arguments.device, selected_distributions)
    elif arguments.stage == "score-final":
        score_final(arguments.method, arguments.device, selected_distributions)
    elif arguments.stage == "summarize-final":
        summarize_final(arguments.method, arguments.distribution)
    elif arguments.stage == "evaluate":
        evaluate(devices)
    else:
        artifact_command = [
            sys.executable,
            "-m",
            "robust_steerability.benchmarks.artifacts",
            "--stage",
            "all",
            "--model",
            MODEL_KEY,
            "--behavior",
            "toxicity",
            "--devices",
            ",".join(devices),
        ]
        subprocess.run(artifact_command, cwd=REPO, check=True)
        prepare()
        calibrate(devices)
        evaluate(devices)


if __name__ == "__main__":
    main()
