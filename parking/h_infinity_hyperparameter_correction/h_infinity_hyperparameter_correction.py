"""Calibrate H-infinity costs and compare matched TruthfulQA controllers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

from robust_steerability.benchmarks.metrics import judge_label, truth_judge_prompt
from robust_steerability.control import (
    FiniteHorizonControlProblem,
    HInfinityController,
    HInfinityOptions,
    LQRController,
)
from robust_steerability.experiments.methods import ControllerArtifact, build_policy
from robust_steerability.modeling.huggingface import (
    CausalModelLoadSpec,
    cuda_device_index,
    load_access_token,
    load_causal_model,
)
from robust_steerability.modeling.interventions import (
    capture_last_token_policy_rollout,
    register_generation_policy_hooks,
)
from robust_steerability.runtime.policy import SemanticSetpointPolicy
from robust_steerability.source_methods.control import (
    SetpointCalibration,
    build_spid_policy,
)
from robust_steerability.source_methods.generation import generate_batched
from robust_steerability.source_methods.id_benchmark import runtime_provenance
from robust_steerability.source_methods.modeling import load_source_model, source_model_spec
from robust_steerability.source_methods.protocol import GENERATION, paper_alqr_setting


UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[1]
CACHE = UNIT / "cache"
PLOTS = UNIT / "plots"
BENCH_ARTIFACTS = REPO / "benchmarks/truthfulness/cache/gemma2b/artifacts"
BENCH_HINF_CALIBRATION = (
    REPO
    / "benchmarks/truthfulness/cache/gemma2b/calibrations/h_infinity/selected"
)
HINF_SOURCE_ROOT = REPO / "parking/dist_changes/cache"

MODEL_ID = "google/gemma-2-2b"
MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
MODEL_LABEL = "Gemma-2-2B"
DATASET_ID = "truthful_qa"
DATASET_REVISION = "741b8276f2d1982aa3d5b832d3ee81ed3b896490"
TRUTH_JUDGE_ID = "allenai/truthfulqa-truth-judge-llama2-7B"
TRUTH_JUDGE_REVISION = "8f718cb4e68bc30a488727a0e37190835a7c129e"
INFO_JUDGE_ID = "allenai/truthfulqa-info-judge-llama2-7B"
INFO_JUDGE_REVISION = "72601eff294cbf7e8d7188ae82397e19cb9bb9bd"

SEED = 42
EVALUATION_COUNT = 100
CALIBRATION_COUNT = 100
CALIBRATION_REPETITIONS = 5
CALIBRATION_SEED_STRIDE = 100_000
GENERATION_BATCH_SIZE = 8
ROLLOUT_BATCH_SIZE = 8
JUDGE_BATCH_SIZE = 16
JUDGE_MAX_NEW_TOKENS = 20
UNIT_PROTOCOL_VERSION = 1

BASE_METHOD_ORDER = (
    "original",
    "spid",
    "alqr",
    "hinf_previous",
    "hinf_candidate",
)
METHOD_ORDER = (*BASE_METHOD_ORDER, "hinf_best")
METHOD_LABELS = {
    "original": "Original",
    "spid": "S-PID",
    "alqr": "A-LQR",
    "hinf_previous": "H∞ — adopted",
    "hinf_candidate": "H∞ — Kaz",
    "hinf_best": "H∞ — best",
}
SPID_PARAMETERS = {"lambda": 1.5, "kp": 0.7, "ki": 0.01, "kd": 0.1}
HINF_CONFIGURATIONS = {
    "hinf_previous": {"lambda": 3.0, "q": 0.1, "r": 1.0, "q_final": 1.0},
    "hinf_candidate": {"lambda": 3.0, "q": 0.5, "r": 1.0, "q_final": 0.3},
}
HINF_Q_VALUES = tuple(
    float(10**exponent)
    for exponent in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
)
HINF_Q_FINAL_VALUES = tuple(
    float(10**exponent)
    for exponent in (-2.0, -1.5, -1.0, -0.5)
)
HINF_GRID = tuple(
    {
        "grid_id": f"q_{q_index:02d}_qf_{q_final_index:02d}",
        "q_index": q_index,
        "q_final_index": q_final_index,
        "lambda": 3.0,
        "q": q,
        "r": 1.0,
        "q_final": q_final,
    }
    for q_index, q in enumerate(HINF_Q_VALUES)
    for q_final_index, q_final in enumerate(HINF_Q_FINAL_VALUES)
)
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _source_hashes() -> dict[str, str]:
    paths = (
        REPO / "robust_steerability/control/h_infinity.py",
        REPO / "robust_steerability/control/lqr.py",
        REPO / "robust_steerability/control/pid.py",
        REPO / "robust_steerability/experiments/methods.py",
        REPO / "robust_steerability/modeling/interventions.py",
        REPO / "robust_steerability/runtime/policy.py",
        REPO / "robust_steerability/source_methods/control.py",
        REPO / "robust_steerability/source_methods/generation.py",
    )
    return {str(path.relative_to(REPO)): _sha256(path) for path in paths}


def _required_sources() -> dict[str, Path]:
    return {
        "alqr_data": BENCH_ARTIFACTS / "data.json",
        "alqr_setpoint": BENCH_ARTIFACTS / "setpoint.pt",
        "alqr_dynamics": BENCH_ARTIFACTS / "dynamics.pt",
        "hinf_controller": HINF_SOURCE_ROOT / "hinf_controller.pt",
        "hinf_input": HINF_SOURCE_ROOT / "hinf_controller_diagnostics/input.pt",
    }


def _source_artifact_hashes() -> dict[str, str]:
    missing = [str(path) for path in _required_sources().values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing reusable calibration artifacts: {missing}")
    return {name: _sha256(path) for name, path in _required_sources().items()}


def prepare() -> None:
    """Freeze one source-style permutation of 100 unique TruthfulQA questions."""

    CACHE.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    destination = CACHE / "data.json"
    identity = {
        "schema_version": 1,
        "unit_protocol_version": UNIT_PROTOCOL_VERSION,
        "dataset": [DATASET_ID, DATASET_REVISION, "generation", "validation"],
        "selection": "first 100 rows of the seed-42 full-set permutation",
        "seed": SEED,
        "evaluation_count": EVALUATION_COUNT,
        "source_artifacts_sha256": _source_artifact_hashes(),
        "source_hashes": _source_hashes(),
    }
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved.get("identity") != identity:
            raise ValueError(f"Prepared data identity mismatch: {destination}")
        if len(saved.get("records", [])) != EVALUATION_COUNT:
            raise ValueError("Prepared data does not contain 100 records")
        return

    dataset = list(
        load_dataset(
            DATASET_ID,
            "generation",
            split="validation",
            revision=DATASET_REVISION,
        )
    )
    indices = random.Random(SEED).sample(range(len(dataset)), len(dataset))[:EVALUATION_COUNT]
    records = []
    for index in indices:
        question = str(dataset[index]["question"]).strip()
        records.append(
            {
                "prompt_id": f"truthfulqa:{index}",
                "question": question,
                "text": f"Q: {question} A:",
            }
        )
    if len({row["prompt_id"] for row in records}) != EVALUATION_COUNT:
        raise ValueError("TruthfulQA diagnostic selection is not unique")
    _write_json(destination, {"identity": identity, "records": records})


def prepare_hinf_calibration() -> None:
    """Freeze calibration questions disjoint from the 100 evaluation questions."""

    prepare()
    destination = CACHE / "sweep/data.json"
    identity = {
        "schema_version": 1,
        "model": [MODEL_ID, MODEL_REVISION],
        "task": "truthfulness",
        "dataset": [DATASET_ID, DATASET_REVISION, "generation", "validation"],
        "selection": (
            f"rows 101-{EVALUATION_COUNT + CALIBRATION_COUNT} of the seed-42 "
            "full-set permutation; disjoint from the first 100 evaluation rows"
        ),
        "seed": SEED,
        "calibration_count": CALIBRATION_COUNT,
        "evaluation_data_sha256": _sha256(CACHE / "data.json"),
        "source_artifacts_sha256": _source_artifact_hashes(),
    }
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved.get("identity") != identity:
            raise ValueError(f"H-infinity calibration data mismatch: {destination}")
        if len(saved.get("records", [])) != CALIBRATION_COUNT:
            raise ValueError("H-infinity calibration data has the wrong record count")
        return

    dataset = list(
        load_dataset(
            DATASET_ID,
            "generation",
            split="validation",
            revision=DATASET_REVISION,
        )
    )
    permutation = random.Random(SEED).sample(range(len(dataset)), len(dataset))
    indices = permutation[EVALUATION_COUNT : EVALUATION_COUNT + CALIBRATION_COUNT]
    records = []
    for index in indices:
        question = str(dataset[index]["question"]).strip()
        records.append(
            {
                "prompt_id": f"truthfulqa:{index}",
                "question": question,
                "text": f"Q: {question} A:",
            }
        )
    evaluation_ids = {
        row["prompt_id"] for row in json.loads((CACHE / "data.json").read_text())["records"]
    }
    calibration_ids = {row["prompt_id"] for row in records}
    if len(calibration_ids) != CALIBRATION_COUNT or evaluation_ids & calibration_ids:
        raise ValueError("H-infinity calibration questions are not unique and disjoint")
    _write_json(destination, {"identity": identity, "records": records})


def _load_setpoint() -> SetpointCalibration:
    payload = torch.load(
        BENCH_ARTIFACTS / "setpoint.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    return SetpointCalibration(payload["contrast"], payload["feature_norm"])


def _alqr_policy(device: str) -> SemanticSetpointPolicy:
    setting = paper_alqr_setting("truthfulness", MODEL_ID)
    cache_path = CACHE / "controllers/alqr.pt"
    identity = {
        "schema_version": 1,
        "parameters": asdict(setting),
        "setpoint_sha256": _sha256(BENCH_ARTIFACTS / "setpoint.pt"),
        "dynamics_sha256": _sha256(BENCH_ARTIFACTS / "dynamics.pt"),
        "lqr_source_sha256": _source_hashes()["robust_steerability/control/lqr.py"],
    }
    if cache_path.exists():
        saved = torch.load(cache_path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError(f"A-LQR controller cache mismatch: {cache_path}")
        gains = saved["gains"]
    else:
        dynamics = torch.load(
            BENCH_ARTIFACTS / "dynamics.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )["dynamics"]
        from robust_steerability.control.lqr import solve_identity_input_lqr

        gains = solve_identity_input_lqr(
            dynamics,
            device,
            setting.q,
            setting.r,
            setting.q_final,
        )
        _write_torch(cache_path, {"identity": identity, "gains": gains})
    calibration = _load_setpoint()
    return SemanticSetpointPolicy(
        LQRController.from_tracking_gains(gains),
        calibration.unit_features(0.0),
        calibration.setpoints(setting.multiplier),
    )


def _spid_policy() -> SemanticSetpointPolicy:
    calibration = _load_setpoint()
    return build_spid_policy(
        calibration,
        multiplier=SPID_PARAMETERS["lambda"],
        kp=SPID_PARAMETERS["kp"],
        ki=SPID_PARAMETERS["ki"],
        kd=SPID_PARAMETERS["kd"],
    )


def _hinf_problem(bundle: dict, parameters: dict[str, float]) -> FiniteHorizonControlProblem:
    source_problem = FiniteHorizonControlProblem(**bundle["problem"])
    source_settings = bundle["calibration"]["settings"]
    state_costs = source_problem.state_costs * (
        parameters["q"] / float(source_settings["q"])
    )
    control_costs = source_problem.control_costs * (
        parameters["r"] / float(source_settings["r"])
    )
    terminal_cost = source_problem.terminal_cost * (
        parameters["q_final"] / float(source_settings["q_final"])
    )
    return FiniteHorizonControlProblem(
        dynamics=source_problem.dynamics,
        control_channels=source_problem.control_channels,
        disturbance_channels=source_problem.disturbance_channels,
        state_costs=state_costs,
        control_costs=control_costs,
        terminal_cost=terminal_cost,
        metadata={
            "configuration": parameters,
            "shared_A_D_and_coordinates": True,
        },
    )


def _load_hinf_source() -> tuple[ControllerArtifact, dict, HInfinityOptions]:
    source_artifact_path = HINF_SOURCE_ROOT / "hinf_controller.pt"
    input_path = HINF_SOURCE_ROOT / "hinf_controller_diagnostics/input.pt"
    source_payload = torch.load(
        source_artifact_path, map_location="cpu", weights_only=True, mmap=True
    )
    source_artifact = ControllerArtifact(**source_payload["artifact"])
    bundle = torch.load(input_path, map_location="cpu", weights_only=True, mmap=True)
    source_settings = bundle["calibration"]["settings"]
    if (
        float(source_settings["hinf_setpoint_multiplier"]) != 3.0
        or float(source_settings["q"]) != 0.1
        or float(source_settings["r"]) != 1.0
        or float(source_settings["q_final"]) != 1.0
    ):
        raise ValueError("Reusable H-infinity artifact is not the expected previous configuration")
    if bundle["record"]["model_id"] != MODEL_ID or bundle["record"]["behavior"] != "truthfulness":
        raise ValueError("Reusable H-infinity artifact has the wrong model or behavior")
    return source_artifact, bundle, HInfinityOptions(**bundle["options"])


def _synthesize_hinf(
    bundle: dict,
    parameters: dict[str, float],
    device: str,
    options: HInfinityOptions,
) -> dict:
    synthesized = HInfinityController.synthesize(
        _hinf_problem(bundle, parameters), device=device, options=options
    ).solution()
    if not synthesized.feasible or synthesized.gamma_star is None:
        raise ValueError(f"Infeasible H-infinity configuration: {parameters}")
    return {
        "gains": synthesized.gains,
        "feasible": synthesized.feasible,
        "gamma_star": synthesized.gamma_star,
        "diagnostics": synthesized.diagnostics,
    }


def synthesize_hinf_variants(device: str) -> None:
    """Synthesize the two prespecified H-infinity configurations."""

    prepare()
    source_artifact_path = HINF_SOURCE_ROOT / "hinf_controller.pt"
    input_path = HINF_SOURCE_ROOT / "hinf_controller_diagnostics/input.pt"
    source_artifact, bundle, options = _load_hinf_source()

    previous_problem = _hinf_problem(bundle, HINF_CONFIGURATIONS["hinf_previous"])
    source_problem = FiniteHorizonControlProblem(**bundle["problem"])
    torch.testing.assert_close(previous_problem.state_costs, source_problem.state_costs)
    torch.testing.assert_close(previous_problem.control_costs, source_problem.control_costs)
    torch.testing.assert_close(previous_problem.terminal_cost, source_problem.terminal_cost)

    for method in ("hinf_previous", "hinf_candidate"):
        started_at = _utc_now()
        started = time.perf_counter()
        destination = CACHE / "controllers" / f"{method}.pt"
        parameters = HINF_CONFIGURATIONS[method]
        identity = {
            "schema_version": 1,
            "method": method,
            "parameters": parameters,
            "selection": "explicit diagnostic candidate; no hyperparameter sweep",
            "source_controller_sha256": _sha256(source_artifact_path),
            "source_input_sha256": _sha256(input_path),
            "hinf_source_sha256": _source_hashes()["robust_steerability/control/h_infinity.py"],
                "shared_target_coordinates_A_D": True,
            }
        if destination.exists():
            saved = torch.load(destination, map_location="cpu", weights_only=True)
            if saved["identity"] != identity:
                raise ValueError(f"H-infinity variant cache mismatch: {destination}")
            continue
        if method == "hinf_previous":
            result = {
                "gains": source_artifact.hinf_gains,
                "feasible": source_artifact.hinf_feasible,
                "gamma_star": source_artifact.gamma_star,
                "diagnostics": source_artifact.hinf_diagnostics,
            }
        else:
            result = _synthesize_hinf(bundle, parameters, device, options)
        if not result["feasible"] or result["gamma_star"] is None:
            raise ValueError(f"{method} did not produce a feasible H-infinity controller")
        _write_torch(
            destination,
            {
                "identity": identity,
                **result,
                "synthesis": {
                    "started_at_utc": started_at,
                    "finished_at_utc": _utc_now(),
                    "elapsed_seconds": time.perf_counter() - started,
                    "runtime": runtime_provenance(device),
                },
            },
        )


def _grid_controller_path(grid_id: str) -> Path:
    return CACHE / "sweep/controllers" / f"{grid_id}.pt"


def synthesize_hinf_grid(device: str) -> None:
    """Synthesize every point in the fixed 8-by-4 ratio grid."""

    prepare_hinf_calibration()
    source_artifact_path = HINF_SOURCE_ROOT / "hinf_controller.pt"
    input_path = HINF_SOURCE_ROOT / "hinf_controller_diagnostics/input.pt"
    _source_artifact, bundle, options = _load_hinf_source()
    for configuration in HINF_GRID:
        destination = _grid_controller_path(configuration["grid_id"])
        parameters = {
            "lambda": configuration["lambda"],
            "q": configuration["q"],
            "r": configuration["r"],
            "q_final": configuration["q_final"],
        }
        identity = {
            "schema_version": 1,
            "model": [MODEL_ID, MODEL_REVISION],
            "task": "truthfulness",
            "grid_id": configuration["grid_id"],
            "parameters": parameters,
            "cost_parameterization": "R=1; sweep Q/R and Qf/R on a half-decade grid",
            "source_controller_sha256": _sha256(source_artifact_path),
            "source_input_sha256": _sha256(input_path),
            "hinf_source_sha256": _source_hashes()[
                "robust_steerability/control/h_infinity.py"
            ],
            "shared_target_coordinates_A_D": True,
        }
        if destination.exists():
            saved = torch.load(destination, map_location="cpu", weights_only=True)
            if saved["identity"] != identity:
                raise ValueError(f"H-infinity grid controller mismatch: {destination}")
            continue
        started_at = _utc_now()
        started = time.perf_counter()
        result = _synthesize_hinf(bundle, parameters, device, options)
        _write_torch(
            destination,
            {
                "identity": identity,
                **result,
                "synthesis": {
                    "started_at_utc": started_at,
                    "finished_at_utc": _utc_now(),
                    "elapsed_seconds": time.perf_counter() - started,
                    "runtime": runtime_provenance(device),
                },
            },
        )


def _grid_policy(grid_id: str) -> object:
    source_payload = torch.load(
        HINF_SOURCE_ROOT / "hinf_controller.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    base = ControllerArtifact(**source_payload["artifact"])
    variant = torch.load(
        _grid_controller_path(grid_id), map_location="cpu", weights_only=True
    )
    artifact = replace(
        base,
        hinf_gains=variant["gains"],
        hinf_feasible=variant["feasible"],
        gamma_star=variant["gamma_star"],
        hinf_diagnostics=variant["diagnostics"],
    )
    return build_policy("hinf", artifact, kp=0.0, ki=0.0, kd=0.0)


def _hinf_policy(method: str) -> object:
    source_payload = torch.load(
        HINF_SOURCE_ROOT / "hinf_controller.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    base = ControllerArtifact(**source_payload["artifact"])
    variant_path = (
        BENCH_HINF_CALIBRATION / "controller.pt"
        if method == "hinf_best"
        else CACHE / "controllers" / f"{method}.pt"
    )
    variant = torch.load(variant_path, map_location="cpu", weights_only=True)
    artifact = replace(
        base,
        hinf_gains=variant["gains"],
        hinf_feasible=variant["feasible"],
        gamma_star=variant["gamma_star"],
        hinf_diagnostics=variant["diagnostics"],
    )
    return build_policy("hinf", artifact, kp=0.0, ki=0.0, kd=0.0)


def _policy(method: str, device: str):
    if method == "original":
        return None
    if method == "spid":
        return _spid_policy()
    if method == "alqr":
        return _alqr_policy(device)
    return _hinf_policy(method)


def _method_parameters(method: str) -> dict[str, float]:
    if method == "original":
        return {}
    if method == "spid":
        return SPID_PARAMETERS
    if method == "alqr":
        setting = paper_alqr_setting("truthfulness", MODEL_ID)
        return {
            "lambda": setting.multiplier,
            "q": setting.q,
            "r": setting.r,
            "q_final": setting.q_final,
        }
    if method == "hinf_best":
        calibration = json.loads(
            (BENCH_HINF_CALIBRATION / "hyperparameter_calibration.json").read_text()
        )
        return calibration["selected"]["parameters"]
    return HINF_CONFIGURATIONS[method]


def _generation_identity(method: str) -> dict:
    artifact_paths = {}
    if method == "spid":
        artifact_paths["setpoint"] = BENCH_ARTIFACTS / "setpoint.pt"
    elif method == "alqr":
        artifact_paths.update(
            {
                "setpoint": BENCH_ARTIFACTS / "setpoint.pt",
                "dynamics": BENCH_ARTIFACTS / "dynamics.pt",
                "controller": CACHE / "controllers/alqr.pt",
            }
        )
    elif method.startswith("hinf_"):
        controller_path = (
            BENCH_HINF_CALIBRATION / "controller.pt"
            if method == "hinf_best"
            else CACHE / "controllers" / f"{method}.pt"
        )
        artifact_paths.update(
            {
                "hinf_source": HINF_SOURCE_ROOT / "hinf_controller.pt",
                "hinf_input": HINF_SOURCE_ROOT / "hinf_controller_diagnostics/input.pt",
                "controller": controller_path,
            }
        )
        if method == "hinf_best":
            artifact_paths["hyperparameter_calibration"] = (
                BENCH_HINF_CALIBRATION / "hyperparameter_calibration.json"
            )
    return {
        "schema_version": 1,
        "unit_protocol_version": UNIT_PROTOCOL_VERSION,
        "method": method,
        "model": [MODEL_ID, MODEL_REVISION],
        "model_loading": asdict(
            source_model_spec("alqr", "truthfulness", MODEL_ID, MODEL_REVISION)
        ),
        "data_sha256": _sha256(CACHE / "data.json"),
        "parameters": _method_parameters(method),
        "generation": {
            **GENERATION["truthfulness"],
            "batch_size": GENERATION_BATCH_SIZE,
            "batch_seed_rule": "seed 42 plus batch start",
            "use_cache": method != "original",
        },
        "artifacts_sha256": {
            name: _sha256(path) for name, path in artifact_paths.items()
        },
        "source_hashes": _source_hashes(),
    }


def _rollout_metrics(
    states: torch.Tensor,
    controls: torch.Tensor,
    *,
    raw_dynamics: torch.Tensor,
    reduced_dynamics: torch.Tensor,
    means: torch.Tensor,
    encoders: torch.Tensor,
    reference_states: torch.Tensor,
) -> dict[str, torch.Tensor]:
    centered = states - means.unsqueeze(0)
    raw_prediction = torch.einsum("kij,bkj->bki", raw_dynamics, centered[:, :-1]) + controls
    raw_residuals = centered[:, 1:] - raw_prediction
    raw_residual_norm = torch.linalg.vector_norm(raw_residuals, dim=-1)
    next_state_norm = torch.linalg.vector_norm(states[:, 1:], dim=-1)
    reduced_states = torch.einsum("bkh,khr->bkr", centered, encoders)
    reduced_deviations = reduced_states - reference_states.unsqueeze(0)
    reduced_controls = torch.einsum("bkh,khr->bkr", controls, encoders[1:])
    reduced_prediction = torch.einsum(
        "kij,bkj->bki", reduced_dynamics, reduced_deviations[:, :-1]
    ) + reduced_controls
    reduced_residuals = reduced_deviations[:, 1:] - reduced_prediction
    return {
        "raw_residuals": raw_residuals.detach().cpu().to(torch.float16),
        "raw_residual_norm": raw_residual_norm.detach().cpu(),
        "next_state_norm": next_state_norm.detach().cpu(),
        "layer_relative_residual": (
            raw_residual_norm / next_state_norm.clamp_min(1e-12)
        ).detach().cpu(),
        "reduced_states": reduced_states.detach().cpu(),
        "reduced_deviations": reduced_deviations.detach().cpu(),
        "reduced_controls": reduced_controls.detach().cpu(),
        "reduced_residuals": reduced_residuals.detach().cpu(),
        "reduced_residual_energy": reduced_residuals.square().sum(dim=(1, 2)).detach().cpu(),
    }


def _capture_rollouts(model, tokenizer, records: list[dict], policy, method: str, device: str) -> None:
    destination = CACHE / "rollouts" / f"{method}.pt"
    identity = {
        "schema_version": 1,
        "generation_identity": _generation_identity(method),
        "position": "last prompt token",
        "states": "decoder inputs plus controlled final decoder output",
        "controls": "actual post-block hidden-space deltas",
        "raw_storage_dtype": "float16",
        "derived_storage_dtype": "float32 except raw_residuals float16",
    }
    if destination.exists():
        saved = torch.load(destination, map_location="cpu", weights_only=True, mmap=True)
        if saved["identity"] != identity or len(saved["prompt_ids"]) != EVALUATION_COUNT:
            raise ValueError(f"Rollout cache mismatch: {destination}")
        return

    started_at = _utc_now()
    started = time.perf_counter()
    input_bundle = torch.load(
        HINF_SOURCE_ROOT / "hinf_controller_diagnostics/input.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    calibration = input_bundle["calibration"]
    problem = input_bundle["problem"]
    raw_dynamics = torch.load(
        BENCH_ARTIFACTS / "dynamics.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )["dynamics"].to(device=device, dtype=torch.float32)
    reduced_dynamics = problem["dynamics"].to(device=device, dtype=torch.float32)
    means = calibration["means"].to(device=device, dtype=torch.float32)
    encoders = calibration["encoders"].to(device=device, dtype=torch.float32)
    reference_states = calibration["reference_states"].to(device=device, dtype=torch.float32)
    if method.startswith("hinf_"):
        reference_states = reference_states * (_method_parameters(method)["lambda"] / 3.0)

    state_batches = []
    control_batches = []
    metric_batches: dict[str, list[torch.Tensor]] = {}
    input_token_counts = []
    model_device = next(model.parameters()).device
    for start in range(0, len(records), ROLLOUT_BATCH_SIZE):
        batch = records[start : start + ROLLOUT_BATCH_SIZE]
        encoded = tokenizer(
            [row["text"] for row in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model_device)
        input_token_counts.extend(int(value) for value in encoded["attention_mask"].sum(dim=1))
        states, controls = capture_last_token_policy_rollout(model, encoded, policy)
        metrics = _rollout_metrics(
            states.to(device),
            controls.to(device),
            raw_dynamics=raw_dynamics,
            reduced_dynamics=reduced_dynamics,
            means=means,
            encoders=encoders,
            reference_states=reference_states,
        )
        state_batches.append(states.to(torch.float16))
        control_batches.append(controls.to(torch.float16))
        for name, values in metrics.items():
            metric_batches.setdefault(name, []).append(values)
    _write_torch(
        destination,
        {
            "identity": identity,
            "prompt_ids": [row["prompt_id"] for row in records],
            "input_token_counts": torch.tensor(input_token_counts),
            "states": torch.cat(state_batches),
            "controls": torch.cat(control_batches),
            **{name: torch.cat(values) for name, values in metric_batches.items()},
            "capture": {
                "started_at_utc": started_at,
                "finished_at_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "runtime": runtime_provenance(device),
            },
        },
    )


def _generate_method(model, tokenizer, records: list[dict], method: str, device: str) -> None:
    destination = CACHE / "generations" / f"{method}.json"
    policy = _policy(method, device)
    identity = _generation_identity(method)
    saved = {"identity": identity, "status": "partial", "rows": [], "attempts": []}
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved["identity"] != identity:
            raise ValueError(f"Generation cache mismatch: {destination}")
        if saved["status"] == "complete":
            if len(saved["rows"]) != EVALUATION_COUNT:
                raise ValueError(f"Completed generation has wrong row count: {destination}")
            _capture_rollouts(model, tokenizer, records, policy, method, device)
            return

    attempt = {
        "started_at_utc": _utc_now(),
        "status": "running",
        "runtime": runtime_provenance(device),
    }
    saved["attempts"].append(attempt)
    _write_json(destination, saved)
    started = time.perf_counter()
    completed = len(saved["rows"])
    if completed % GENERATION_BATCH_SIZE != 0 and completed != EVALUATION_COUNT:
        raise ValueError("Partial generation cache does not end at a batch boundary")
    for start in range(completed, len(records), GENERATION_BATCH_SIZE):
        batch = records[start : start + GENERATION_BATCH_SIZE]
        completions = generate_batched(
            model,
            tokenizer,
            [row["text"] for row in batch],
            behavior="truthfulness",
            batch_size=GENERATION_BATCH_SIZE,
            seed=SEED + start,
            use_cache=method != "original",
            register_hooks=(
                None
                if policy is None
                else lambda: register_generation_policy_hooks(model, policy)
            ),
            reset=None if policy is None else policy.reset,
        )
        for row, completion in zip(batch, completions, strict=True):
            saved["rows"].append(
                {
                    "method": method,
                    "prompt_id": row["prompt_id"],
                    "question": row["question"],
                    "prompt": row["text"],
                    "completion": completion,
                    "seed": SEED + start,
                }
            )
        _write_json(destination, saved)
        print(f"{method}: {len(saved['rows'])}/{EVALUATION_COUNT}", flush=True)
    saved["status"] = "complete"
    attempt["status"] = "complete"
    attempt["finished_at_utc"] = _utc_now()
    attempt["elapsed_seconds"] = time.perf_counter() - started
    attempt["gpu_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
        cuda_device_index(device)
    )
    _write_json(destination, saved)
    _capture_rollouts(model, tokenizer, records, policy, method, device)


def _sweep_generation_identity(configuration: dict, controller_path: Path) -> dict:
    return {
        "schema_version": 1,
        "model": [MODEL_ID, MODEL_REVISION],
        "task": "truthfulness",
        "model_loading": asdict(
            source_model_spec("alqr", "truthfulness", MODEL_ID, MODEL_REVISION)
        ),
        "calibration_data_sha256": _sha256(CACHE / "sweep/data.json"),
        "grid_id": configuration["grid_id"],
        "parameters": {
            "lambda": configuration["lambda"],
            "q": configuration["q"],
            "r": configuration["r"],
            "q_final": configuration["q_final"],
        },
        "generation": {
            **GENERATION["truthfulness"],
            "batch_size": GENERATION_BATCH_SIZE,
            "calibration_repetitions": CALIBRATION_REPETITIONS,
            "repetition_seed_stride": CALIBRATION_SEED_STRIDE,
            "batch_seed_rule": "seed 42 plus repetition stride plus batch start",
            "use_cache": True,
        },
        "controller_sha256": _sha256(controller_path),
        "source_hashes": _source_hashes(),
    }


def _generate_sweep_configuration(
    model,
    tokenizer,
    records: list[dict],
    configuration: dict,
    device: str,
) -> None:
    grid_id = configuration["grid_id"]
    destination = CACHE / "sweep/generations" / f"{grid_id}.json"
    controller_path = _grid_controller_path(grid_id)
    identity = _sweep_generation_identity(configuration, controller_path)
    saved = {"identity": identity, "status": "partial", "rows": [], "attempts": []}
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved["identity"] != identity:
            raise ValueError(f"Sweep generation cache mismatch: {destination}")
        if saved["status"] == "complete":
            if len(saved["rows"]) != CALIBRATION_COUNT * CALIBRATION_REPETITIONS:
                raise ValueError(f"Completed sweep generation has wrong size: {destination}")
            return

    policy = _grid_policy(grid_id)
    attempt = {
        "started_at_utc": _utc_now(),
        "status": "running",
        "runtime": runtime_provenance(device),
    }
    saved["attempts"].append(attempt)
    _write_json(destination, saved)
    started = time.perf_counter()
    completed = len(saved["rows"])
    expected = CALIBRATION_COUNT * CALIBRATION_REPETITIONS
    while completed < expected:
        repetition = completed // CALIBRATION_COUNT
        repetition_offset = completed % CALIBRATION_COUNT
        batch_end = min(repetition_offset + GENERATION_BATCH_SIZE, CALIBRATION_COUNT)
        batch = records[repetition_offset:batch_end]
        batch_seed = (
            SEED
            + repetition * CALIBRATION_SEED_STRIDE
            + repetition_offset
        )
        completions = generate_batched(
            model,
            tokenizer,
            [row["text"] for row in batch],
            behavior="truthfulness",
            batch_size=GENERATION_BATCH_SIZE,
            seed=batch_seed,
            use_cache=True,
            register_hooks=lambda: register_generation_policy_hooks(model, policy),
            reset=policy.reset,
        )
        for row, completion in zip(batch, completions, strict=True):
            saved["rows"].append(
                {
                    "method": grid_id,
                    "calibration_repetition": repetition,
                    "prompt_id": row["prompt_id"],
                    "question": row["question"],
                    "prompt": row["text"],
                    "completion": completion,
                    "seed": batch_seed,
                }
            )
        completed = len(saved["rows"])
        _write_json(destination, saved)
    saved["status"] = "complete"
    attempt["status"] = "complete"
    attempt["finished_at_utc"] = _utc_now()
    attempt["elapsed_seconds"] = time.perf_counter() - started
    attempt["gpu_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
        cuda_device_index(device)
    )
    _write_json(destination, saved)
    print(f"sweep {grid_id}: {expected}/{expected}", flush=True)


def generate_sweep_shard(device: str, shard_index: int, shard_count: int) -> None:
    prepare_hinf_calibration()
    configurations = HINF_GRID[shard_index::shard_count]
    records = json.loads((CACHE / "sweep/data.json").read_text())["records"]
    model, tokenizer = load_source_model(
        "alqr",
        "truthfulness",
        MODEL_ID,
        MODEL_REVISION,
        device,
        load_access_token(REPO),
    )
    torch.cuda.reset_peak_memory_stats(cuda_device_index(device))
    for configuration in configurations:
        _generate_sweep_configuration(model, tokenizer, records, configuration, device)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def _sweep_generation_rows() -> list[dict]:
    rows = []
    for configuration in HINF_GRID:
        path = CACHE / "sweep/generations" / f"{configuration['grid_id']}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing sweep generation cache: {path}")
        payload = json.loads(path.read_text())
        if (
            payload.get("status") != "complete"
            or len(payload.get("rows", []))
            != CALIBRATION_COUNT * CALIBRATION_REPETITIONS
        ):
            raise ValueError(f"Incomplete sweep generation cache: {path}")
        rows.extend(payload["rows"])
    return rows


def _q_selection_path() -> Path:
    return CACHE / "sweep/selection.json"


def generate_group(group: str, device: str) -> None:
    prepare()
    if group == "baselines":
        methods = ("original", "spid", "alqr")
    elif group == "hinf":
        synthesize_hinf_variants(device)
        methods = ("hinf_previous", "hinf_candidate")
    elif group == "best":
        if not (BENCH_HINF_CALIBRATION / "controller.pt").exists():
            raise FileNotFoundError("Run the H-infinity sweep before generating hinf_best")
        methods = ("hinf_best",)
    else:
        raise ValueError(f"Unknown generation group: {group}")
    data = json.loads((CACHE / "data.json").read_text())
    records = data["records"]
    token = load_access_token(REPO)
    model, tokenizer = load_source_model(
        "alqr", "truthfulness", MODEL_ID, MODEL_REVISION, device, token
    )
    torch.cuda.reset_peak_memory_stats(cuda_device_index(device))
    for method in methods:
        _generate_method(model, tokenizer, records, method, device)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def _generation_rows(methods: tuple[str, ...] = METHOD_ORDER) -> list[dict]:
    rows = []
    for method in methods:
        path = CACHE / "generations" / f"{method}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing generation cache: {path}")
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete" or len(payload.get("rows", [])) != EVALUATION_COUNT:
            raise ValueError(f"Incomplete generation cache: {path}")
        rows.extend(payload["rows"])
    return rows


def _load_judge(label: str, device: str, token: str):
    model_id, revision = (
        (TRUTH_JUDGE_ID, TRUTH_JUDGE_REVISION)
        if label == "truth"
        else (INFO_JUDGE_ID, INFO_JUDGE_REVISION)
    )
    model, tokenizer = load_causal_model(
        CausalModelLoadSpec(
            model_id=model_id,
            revision=revision,
            quantized=True,
            dtype="float32",
            attention_implementation=None,
            quantization_compute_dtype="bfloat16",
        ),
        device,
        token,
    )
    return model, tokenizer, model_id, revision


def judge(label: str, device: str, group: str) -> None:
    if group == "base":
        methods = BASE_METHOD_ORDER
        rows = _generation_rows(methods)
        destination = CACHE / "judges" / f"{label}.json"
        generation_hashes = {
            method: _sha256(CACHE / "generations" / f"{method}.json")
            for method in methods
        }
    elif group == "best":
        rows = _generation_rows(("hinf_best",))
        destination = CACHE / "judges" / f"{label}_hinf_best.json"
        generation_hashes = {
            "hinf_best": _sha256(CACHE / "generations/hinf_best.json")
        }
    elif group == "sweep":
        rows = _sweep_generation_rows()
        destination = CACHE / "sweep/judges" / f"{label}.json"
        generation_hashes = {
            configuration["grid_id"]: _sha256(
                CACHE / "sweep/generations" / f"{configuration['grid_id']}.json"
            )
            for configuration in HINF_GRID
        }
    else:
        raise ValueError(f"Unknown judge group: {group}")
    model_id, revision = (
        (TRUTH_JUDGE_ID, TRUTH_JUDGE_REVISION)
        if label == "truth"
        else (INFO_JUDGE_ID, INFO_JUDGE_REVISION)
    )
    prompt_label = "True" if label == "truth" else "Helpful"
    identity = {
        "schema_version": 1,
        "judge": [model_id, revision],
        "generation_sha256": generation_hashes,
        "rubric": f"Q: {{question}}\\nA: {{answer}}\\n{prompt_label}:",
        "batch_size": JUDGE_BATCH_SIZE,
        "max_new_tokens": JUDGE_MAX_NEW_TOKENS,
        "input_max_length": 1024,
        "generation": {"do_sample": False, "use_cache": True},
        "parser": "strip and lowercase; exact yes scores 1, every other output scores 0",
    }
    saved = {"identity": identity, "status": "partial", "rows": [], "attempts": []}
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved["identity"] != identity:
            raise ValueError(f"Judge cache mismatch: {destination}")
        if saved["status"] == "complete":
            if len(saved["rows"]) != len(rows):
                raise ValueError(f"Completed judge cache has wrong row count: {destination}")
            return
    token = load_access_token(REPO)
    model, tokenizer, _, _ = _load_judge(label, device, token)
    attempt = {
        "started_at_utc": _utc_now(),
        "status": "running",
        "runtime": runtime_provenance(device),
    }
    saved["attempts"].append(attempt)
    _write_json(destination, saved)
    torch.cuda.reset_peak_memory_stats(cuda_device_index(device))
    started = time.perf_counter()
    for start in range(len(saved["rows"]), len(rows), JUDGE_BATCH_SIZE):
        batch = rows[start : start + JUDGE_BATCH_SIZE]
        prompts = [
            truth_judge_prompt(row["question"], row["completion"], prompt_label)
            for row in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=JUDGE_MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        token_rows = generated.sequences[:, encoded["input_ids"].shape[1] :].cpu().tolist()
        answers = tokenizer.batch_decode(token_rows, skip_special_tokens=True)
        for row, prompt, answer, token_ids in zip(
            batch, prompts, answers, token_rows, strict=True
        ):
            raw_answer = answer.strip()
            score, valid = judge_label(raw_answer)
            saved["rows"].append(
                {
                    "method": row["method"],
                    **(
                        {"calibration_repetition": row["calibration_repetition"]}
                        if "calibration_repetition" in row
                        else {}
                    ),
                    "prompt_id": row["prompt_id"],
                    "judge_prompt": prompt,
                    "raw_answer": raw_answer,
                    "generated_token_ids": token_ids,
                    "score": score,
                    "valid": valid,
                }
            )
        _write_json(destination, saved)
        print(f"{label}: {len(saved['rows'])}/{len(rows)}", flush=True)
    saved["status"] = "complete"
    attempt["status"] = "complete"
    attempt["finished_at_utc"] = _utc_now()
    attempt["elapsed_seconds"] = time.perf_counter() - started
    attempt["gpu_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
        cuda_device_index(device)
    )
    _write_json(destination, saved)


def _plot_hinf_grid(rows: list[dict], selected: dict) -> None:
    matrix = np.full((len(HINF_Q_VALUES), len(HINF_Q_FINAL_VALUES)), np.nan)
    for row in rows:
        matrix[row["q_index"], row["q_final_index"]] = row["truth_x_info_percent"]
    if not np.isfinite(matrix).all():
        raise ValueError("H-infinity grid contains missing scores")

    sns.set_theme(context="talk", style="ticks", palette="dark")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )
    fig, axis = plt.subplots(figsize=(5.4, 7.2))
    q_labels = [f"{value:.3g}" for value in HINF_Q_VALUES]
    q_final_labels = [f"{value:.3g}" for value in HINF_Q_FINAL_VALUES]
    sns.heatmap(
        matrix,
        ax=axis,
        cmap="Greys",
        square=True,
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": 10},
        xticklabels=q_final_labels,
        yticklabels=q_labels,
        cbar_kws={"shrink": 0.8, "ticks": [0, 100]},
    )
    axis.add_patch(
        Rectangle(
            (selected["q_final_index"], selected["q_index"]),
            1,
            1,
            fill=False,
            edgecolor="darkred",
            linewidth=2.5,
        )
    )
    axis.set_xlabel(r"Terminal-state cost ratio $Q_f/R$")
    axis.set_ylabel(r"Running-state cost ratio $Q/R$")
    axis.set_title(
        rf"$H_\infty$ calibration ($n={CALIBRATION_COUNT}\times{CALIBRATION_REPETITIONS}$)"
    )
    axis.tick_params(axis="x", rotation=35)
    axis.tick_params(axis="y", rotation=0)
    axis.collections[0].colorbar.ax.set_ylabel(
        r"$T\times I$ (\%)", rotation=270, labelpad=18
    )
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        destination = PLOTS / f"h_infinity_calibration_heatmap.{suffix}"
        if destination.exists():
            destination.unlink()
        fig.savefig(destination, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)


def _plot_hinf_gamma_star_grid(rows: list[dict], selected: dict) -> None:
    matrix = np.full((len(HINF_Q_VALUES), len(HINF_Q_FINAL_VALUES)), np.nan)
    for row in rows:
        matrix[row["q_index"], row["q_final_index"]] = row["gamma_star"]
    if not np.isfinite(matrix).all():
        raise ValueError("H-infinity grid contains missing gamma-star values")

    sns.set_theme(context="talk", style="ticks", palette="dark")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )
    fig, axis = plt.subplots(figsize=(5.4, 7.2))
    q_labels = [f"{value:.3g}" for value in HINF_Q_VALUES]
    q_final_labels = [f"{value:.3g}" for value in HINF_Q_FINAL_VALUES]
    minimum = float(matrix.min())
    maximum = float(matrix.max())
    sns.heatmap(
        matrix,
        ax=axis,
        cmap="Greys",
        square=True,
        vmin=minimum,
        vmax=maximum,
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 9},
        xticklabels=q_final_labels,
        yticklabels=q_labels,
        cbar_kws={"shrink": 0.8, "ticks": [minimum, maximum]},
    )
    axis.add_patch(
        Rectangle(
            (selected["q_final_index"], selected["q_index"]),
            1,
            1,
            fill=False,
            edgecolor="darkred",
            linewidth=2.5,
        )
    )
    axis.set_xlabel(r"Terminal-state cost ratio $Q_f/R$")
    axis.set_ylabel(r"Running-state cost ratio $Q/R$")
    axis.set_title(r"$H_\infty$ attenuation boundary")
    axis.tick_params(axis="x", rotation=35)
    axis.tick_params(axis="y", rotation=0)
    axis.collections[0].colorbar.ax.set_ylabel(
        r"Minimum feasible $\gamma^\star$", rotation=270, labelpad=20
    )
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        destination = PLOTS / f"h_infinity_gamma_star_heatmap.{suffix}"
        if destination.exists():
            destination.unlink()
        fig.savefig(destination, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)


def _summarize_calibration_rows(rows: list[dict]) -> dict:
    per_repetition = []
    for repetition in range(CALIBRATION_REPETITIONS):
        repetition_rows = [
            row for row in rows if row["calibration_repetition"] == repetition
        ]
        if len(repetition_rows) != CALIBRATION_COUNT:
            raise ValueError(f"Calibration repetition {repetition} has the wrong size")
        truth = 100.0 * float(np.mean([row["truth_score"] for row in repetition_rows]))
        info = 100.0 * float(np.mean([row["info_score"] for row in repetition_rows]))
        per_repetition.append(
            {
                "repetition": repetition,
                "truth_x_info_percent": truth * info / 100.0,
                "truth_percent": truth,
                "info_percent": info,
            }
        )

    def mean_se(name: str) -> tuple[float, float]:
        values = np.asarray([row[name] for row in per_repetition], dtype=float)
        return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))

    truth_x_info, truth_x_info_se = mean_se("truth_x_info_percent")
    truth, truth_se = mean_se("truth_percent")
    info, info_se = mean_se("info_percent")
    return {
        "n_per_repetition": CALIBRATION_COUNT,
        "repetitions": CALIBRATION_REPETITIONS,
        "total_generations": CALIBRATION_COUNT * CALIBRATION_REPETITIONS,
        "truth_x_info_percent": truth_x_info,
        "truth_x_info_standard_error": truth_x_info_se,
        "truth_percent": truth,
        "truth_standard_error": truth_se,
        "info_percent": info,
        "info_standard_error": info_se,
        "per_repetition": per_repetition,
        "invalid_truth_judgments": sum(not row["truth_judge_valid"] for row in rows),
        "invalid_info_judgments": sum(not row["info_judge_valid"] for row in rows),
    }


def summarize_hinf_sweep() -> dict:
    generation_rows = _sweep_generation_rows()
    truth = json.loads((CACHE / "sweep/judges/truth.json").read_text())
    info = json.loads((CACHE / "sweep/judges/info.json").read_text())
    if truth.get("status") != "complete" or info.get("status") != "complete":
        raise ValueError("Both H-infinity sweep judge caches must be complete")
    key = lambda row: (
        row["method"], row["calibration_repetition"], row["prompt_id"]
    )
    truth_by_key = {key(row): row for row in truth["rows"]}
    info_by_key = {key(row): row for row in info["rows"]}
    joined = []
    for row in generation_rows:
        row_key = key(row)
        if row_key not in truth_by_key or row_key not in info_by_key:
            raise ValueError(f"Missing H-infinity sweep judgment for {row_key}")
        joined.append(
            {
                **row,
                "truth_score": truth_by_key[row_key]["score"],
                "truth_judge_answer": truth_by_key[row_key]["raw_answer"],
                "truth_judge_valid": truth_by_key[row_key]["valid"],
                "info_score": info_by_key[row_key]["score"],
                "info_judge_answer": info_by_key[row_key]["raw_answer"],
                "info_judge_valid": info_by_key[row_key]["valid"],
            }
        )
    _write_json(CACHE / "sweep/scored_generations.json", joined)

    summaries = []
    for configuration in HINF_GRID:
        rows = [row for row in joined if row["method"] == configuration["grid_id"]]
        if len(rows) != CALIBRATION_COUNT * CALIBRATION_REPETITIONS:
            raise ValueError(f"Wrong sweep sample count for {configuration['grid_id']}")
        controller = torch.load(
            _grid_controller_path(configuration["grid_id"]),
            map_location="cpu",
            weights_only=True,
        )
        summaries.append(
            {
                **configuration,
                **_summarize_calibration_rows(rows),
                "gamma_star": float(controller["gamma_star"]),
            }
        )
    selected = sorted(
        summaries,
        key=lambda row: (
            -row["truth_x_info_percent"],
            -row["info_percent"],
            -row["truth_percent"],
            row["q"],
            row["q_final"],
        ),
    )[0]
    q_selection = {
        "schema_version": 2,
        "calibration_count_per_repetition": CALIBRATION_COUNT,
        "calibration_repetitions": CALIBRATION_REPETITIONS,
        "selection_rule": (
            "maximize mean repetition-level T×I; ties maximize mean Informative, "
            "then mean True, then choose smaller Q/R and Qf/R"
        ),
        "grid": summaries,
        "selected": selected,
        "source_artifacts_sha256": _source_artifact_hashes(),
        "cache_sha256": {
            "truth_judgments": _sha256(CACHE / "sweep/judges/truth.json"),
            "info_judgments": _sha256(CACHE / "sweep/judges/info.json"),
            "scored_generations": _sha256(CACHE / "sweep/scored_generations.json"),
        },
    }
    _write_json(_q_selection_path(), q_selection)

    calibration_parameters = {
        "lambda": 3.0,
        "q": selected["q"],
        "r": selected["r"],
        "q_final": selected["q_final"],
    }
    final_selected = {**selected, "parameters": calibration_parameters}
    final_benchmark_configuration = {
        **selected,
        "parameters": calibration_parameters,
        "source": "five-repetition calibration-grid argmax",
    }
    source_controller = _grid_controller_path(selected["grid_id"])
    synthesized = torch.load(source_controller, map_location="cpu", weights_only=True)
    selection_rule = q_selection["selection_rule"]
    controller_identity = {
        "schema_version": 3,
        "model": [MODEL_ID, MODEL_REVISION],
        "task": "truthfulness",
        "dataset": [DATASET_ID, DATASET_REVISION, "generation", "validation"],
        "calibration_data_sha256": _sha256(CACHE / "sweep/data.json"),
        "configuration_source": "five-repetition calibration-grid argmax",
        "configuration_grid_id": selected["grid_id"],
        "parameters": calibration_parameters,
        "source_grid_controller_sha256": _sha256(source_controller),
    }
    controller_payload = {
        "identity": controller_identity,
        "gains": synthesized["gains"],
        "feasible": synthesized["feasible"],
        "gamma_star": synthesized["gamma_star"],
        "diagnostics": synthesized["diagnostics"],
        "synthesis": synthesized["synthesis"],
    }
    controller_destination = BENCH_HINF_CALIBRATION / "controller.pt"
    if controller_destination.exists():
        saved_controller = torch.load(
            controller_destination, map_location="cpu", weights_only=True
        )
        if saved_controller.get("identity") != controller_identity:
            raise ValueError("Saved calibrated H-infinity controller changed")
        torch.testing.assert_close(
            saved_controller["gains"], controller_payload["gains"], rtol=0, atol=0
        )
    else:
        _write_torch(controller_destination, controller_payload)

    calibration_data = json.loads((CACHE / "sweep/data.json").read_text())
    artifact = {
        "schema_version": 3,
        "created_at_utc": _utc_now(),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "task": "truthfulness",
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "configuration": "generation",
            "split": "validation",
        },
        "calibration": {
            "n": CALIBRATION_COUNT,
            "repetitions": CALIBRATION_REPETITIONS,
            "seeds": [
                SEED + repetition * CALIBRATION_SEED_STRIDE
                for repetition in range(CALIBRATION_REPETITIONS)
            ],
            "records": calibration_data["records"],
            "data_sha256": _sha256(CACHE / "sweep/data.json"),
        },
        "cost_parameterization": {
            "fixed_r": 1.0,
            "q_over_r": list(HINF_Q_VALUES),
            "q_final_over_r": list(HINF_Q_FINAL_VALUES),
            "spacing": "half-decade logarithmic",
            "configuration_count": len(HINF_GRID),
            "fixed_setpoint_multiplier": 3.0,
            "procedure": "joint Q/R–Qf/R grid with fixed setpoint multiplier",
        },
        "generation": {
            **GENERATION["truthfulness"],
            "batch_size": GENERATION_BATCH_SIZE,
            "repetitions": CALIBRATION_REPETITIONS,
            "repetition_seed_stride": CALIBRATION_SEED_STRIDE,
            "batch_seed_rule": "seed 42 plus repetition stride plus batch start",
        },
        "judges": {
            "truth": [TRUTH_JUDGE_ID, TRUTH_JUDGE_REVISION],
            "informative": [INFO_JUDGE_ID, INFO_JUDGE_REVISION],
            "parser": "exact case-insensitive yes scores 1",
        },
        "selection_rule": selection_rule,
        "grid": summaries,
        "selected": final_selected,
        "benchmark_configuration": final_benchmark_configuration,
        "source_artifacts_sha256": _source_artifact_hashes(),
        "controller_sha256": _sha256(controller_destination),
        "sweep_cache_sha256": {
            "selection": _sha256(_q_selection_path()),
            "truth_judgments": _sha256(CACHE / "sweep/judges/truth.json"),
            "info_judgments": _sha256(CACHE / "sweep/judges/info.json"),
            "scored_generations": _sha256(CACHE / "sweep/scored_generations.json"),
        },
    }
    calibration_destination = BENCH_HINF_CALIBRATION / "hyperparameter_calibration.json"
    if calibration_destination.exists():
        saved = json.loads(calibration_destination.read_text())
        comparable = {key: value for key, value in artifact.items() if key != "created_at_utc"}
        saved_comparable = {key: value for key, value in saved.items() if key != "created_at_utc"}
        if saved_comparable != comparable:
            raise ValueError("Saved H-infinity calibration changed")
    else:
        _write_json(calibration_destination, artifact)

    csv_lines = [
        "grid_id,q_over_r,q_final_over_r,n_per_repetition,repetitions,truth_x_info_percent,truth_x_info_standard_error,truth_percent,truth_standard_error,info_percent,info_standard_error,gamma_star,selected"
    ]
    for row in summaries:
        csv_lines.append(
            f"{row['grid_id']},{row['q']:.12g},{row['q_final']:.12g},"
            f"{row['n_per_repetition']},{row['repetitions']},"
            f"{row['truth_x_info_percent']:.6f},{row['truth_x_info_standard_error']:.6f},"
            f"{row['truth_percent']:.6f},{row['truth_standard_error']:.6f},"
            f"{row['info_percent']:.6f},{row['info_standard_error']:.6f},"
            f"{row['gamma_star']:.12g},"
            f"{str(row['grid_id'] == selected['grid_id']).lower()}"
        )
    (PLOTS / "h_infinity_calibration_grid.csv").write_text(
        "\n".join(csv_lines) + "\n"
    )
    _plot_hinf_grid(summaries, selected)
    _plot_hinf_gamma_star_grid(summaries, selected)
    return artifact


def summarize() -> None:
    generation_rows = _generation_rows()
    truth = json.loads((CACHE / "judges/truth.json").read_text())
    info = json.loads((CACHE / "judges/info.json").read_text())
    truth_best = json.loads((CACHE / "judges/truth_hinf_best.json").read_text())
    info_best = json.loads((CACHE / "judges/info_hinf_best.json").read_text())
    if any(
        payload.get("status") != "complete"
        for payload in (truth, info, truth_best, info_best)
    ):
        raise ValueError("Both judge caches must be complete")
    key = lambda row: (row["method"], row["prompt_id"])
    truth_by_key = {
        key(row): row for payload in (truth, truth_best) for row in payload["rows"]
    }
    info_by_key = {
        key(row): row for payload in (info, info_best) for row in payload["rows"]
    }
    joined = []
    for row in generation_rows:
        row_key = key(row)
        if row_key not in truth_by_key or row_key not in info_by_key:
            raise ValueError(f"Missing judge score for {row_key}")
        joined.append(
            {
                **row,
                "truth_score": truth_by_key[row_key]["score"],
                "truth_judge_answer": truth_by_key[row_key]["raw_answer"],
                "truth_judge_valid": truth_by_key[row_key]["valid"],
                "info_score": info_by_key[row_key]["score"],
                "info_judge_answer": info_by_key[row_key]["raw_answer"],
                "info_judge_valid": info_by_key[row_key]["valid"],
            }
        )
    _write_json(CACHE / "results/scored_generations.json", joined)

    summaries = []
    for method in METHOD_ORDER:
        rows = [row for row in joined if row["method"] == method]
        if len(rows) != EVALUATION_COUNT:
            raise ValueError(f"Expected 100 scored rows for {method}")
        truth_percent = 100.0 * float(np.mean([row["truth_score"] for row in rows]))
        info_percent = 100.0 * float(np.mean([row["info_score"] for row in rows]))
        summaries.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "parameters": _method_parameters(method),
                "n": len(rows),
                "truth_x_info_percent": truth_percent * info_percent / 100.0,
                "truth_percent": truth_percent,
                "info_percent": info_percent,
                "invalid_truth_judgments": sum(not row["truth_judge_valid"] for row in rows),
                "invalid_info_judgments": sum(not row["info_judge_valid"] for row in rows),
            }
        )
    _write_json(PLOTS / "summary.json", {"created_at_utc": _utc_now(), "rows": summaries})

    table_lines = [
        "| Method | $\\lambda$ | $Q$ | $R$ | $Q_f$ | $T \\times I$ (%) | True (%) | Informative (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    csv_lines = ["method,lambda,q,r,q_final,n,truth_x_info_percent,truth_percent,info_percent"]
    for row in summaries:
        parameters = row["parameters"]
        display = lambda value: "—" if value == "—" else f"{float(value):.4g}"
        lambda_value = display(parameters.get("lambda", "—"))
        q_value = display(parameters.get("q", "—"))
        r_value = display(parameters.get("r", "—"))
        q_final = display(parameters.get("q_final", "—"))
        table_lines.append(
            f"| {row['method_label']} | {lambda_value} | {q_value} | {r_value} | {q_final} | "
            f"{row['truth_x_info_percent']:.2f} | {row['truth_percent']:.2f} | {row['info_percent']:.2f} |"
        )
        csv_lines.append(
            f"{row['method']},{lambda_value},{q_value},{r_value},{q_final},{row['n']},"
            f"{row['truth_x_info_percent']:.6f},{row['truth_percent']:.6f},{row['info_percent']:.6f}"
        )
    (PLOTS / "truthfulness_table.md").write_text("\n".join(table_lines) + "\n")
    (PLOTS / "truthfulness_table.csv").write_text("\n".join(csv_lines) + "\n")


def _run_parallel(jobs: list[tuple[str, list[str]]]) -> None:
    processes = []
    for label, arguments in jobs:
        log_path = CACHE / "logs" / f"{label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a")
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), *arguments],
            cwd=REPO,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((label, process, handle, log_path))
    failures = []
    for label, process, handle, log_path in processes:
        return_code = process.wait()
        handle.close()
        if return_code != 0:
            failures.append(f"{label} failed; see {log_path}")
    if failures:
        raise RuntimeError("; ".join(failures))


def run_all() -> None:
    prepare_hinf_calibration()
    synthesize_hinf_grid("cuda:0")
    _run_parallel(
        [
            (
                "sweep_generate_0",
                [
                    "--stage", "generate-sweep", "--device", "cuda:0",
                    "--shard-index", "0", "--shard-count", "2",
                ],
            ),
            (
                "sweep_generate_1",
                [
                    "--stage", "generate-sweep", "--device", "cuda:1",
                    "--shard-index", "1", "--shard-count", "2",
                ],
            ),
        ]
    )
    _run_parallel(
        [
            (
                "sweep_judge_truth",
                ["--stage", "judge", "--group", "sweep", "--judge", "truth", "--device", "cuda:0"],
            ),
            (
                "sweep_judge_info",
                ["--stage", "judge", "--group", "sweep", "--judge", "info", "--device", "cuda:1"],
            ),
        ]
    )
    summarize_hinf_sweep()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "prepare",
            "prepare-sweep",
            "synthesize-hinf",
            "synthesize-grid",
            "generate",
            "generate-sweep",
            "judge",
            "summarize-sweep",
            "summarize",
            "all",
        ),
        required=True,
    )
    parser.add_argument(
        "--group",
        choices=("baselines", "hinf", "best", "base", "sweep"),
    )
    parser.add_argument("--judge", choices=("truth", "info"))
    parser.add_argument("--device")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    arguments = parser.parse_args()
    if arguments.stage == "prepare":
        prepare()
    elif arguments.stage == "prepare-sweep":
        prepare_hinf_calibration()
    elif arguments.stage == "synthesize-hinf":
        if arguments.device is None:
            raise ValueError("synthesize-hinf requires --device")
        synthesize_hinf_variants(arguments.device)
    elif arguments.stage == "synthesize-grid":
        if arguments.device is None:
            raise ValueError("synthesize-grid requires --device")
        synthesize_hinf_grid(arguments.device)
    elif arguments.stage == "generate":
        if arguments.device is None or arguments.group is None:
            raise ValueError("generate requires --device and --group")
        generate_group(arguments.group, arguments.device)
    elif arguments.stage == "generate-sweep":
        if (
            arguments.device is None
            or arguments.shard_index is None
            or arguments.shard_count is None
        ):
            raise ValueError(
                "generate-sweep requires --device, --shard-index, and --shard-count"
            )
        generate_sweep_shard(
            arguments.device, arguments.shard_index, arguments.shard_count
        )
    elif arguments.stage == "judge":
        if arguments.device is None or arguments.judge is None or arguments.group is None:
            raise ValueError("judge requires --device, --judge, and --group")
        judge(arguments.judge, arguments.device, arguments.group)
    elif arguments.stage == "summarize-sweep":
        summarize_hinf_sweep()
    elif arguments.stage == "summarize":
        summarize()
    else:
        run_all()


if __name__ == "__main__":
    main()
