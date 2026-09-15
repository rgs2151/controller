"""H-infinity calibration for one model's TruthfulQA task."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from robust_steerability.benchmarks import truthfulness_runtime as evaluation
from robust_steerability.benchmarks.launcher import run_jobs
from robust_steerability.benchmarks.layout import (
    artifact_root,
    calibration_root,
    dataset_root,
)
from robust_steerability.benchmarks.specs import MODELS
from robust_steerability.control import (
    FiniteHorizonControlProblem,
    HInfinityController,
    HInfinityOptions,
)
from robust_steerability.artifacts import configuration_hash
from robust_steerability.experiments.calibration import calibrate_controller, diagnostic_root
from robust_steerability.experiments.diagnostics import score as freeze_diagnostic_score
from robust_steerability.experiments.methods import ControllerArtifact, build_policy
from robust_steerability.judges import huggingface as huggingface_scoring
from robust_steerability.judges.specs import scorer_spec
from robust_steerability.benchmarks.metrics import truth_judge_prompt
from robust_steerability.modeling.huggingface import load_access_token
from robust_steerability.modeling.interventions import register_generation_policy_hooks
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
CALIBRATION_SAMPLES = 50
CALIBRATION_REPETITIONS = 1
DISTURBANCE_SAMPLES = 200
SEED_STRIDE = 100_000
Q_OVER_R = (0.01, 10**-1.5, 0.1, 10**-0.5, 1.0, 10**0.5, 10.0, 10**1.5)
Q_FINAL_OVER_R = (0.01, 10**-1.5, 0.1, 10**-0.5)
FIXED_R = 1.0
GRID_SELECTION_SOURCE = "TruthfulQA True calibration-grid argmax"
FIXED_SELECTION_SOURCE = "fixed configuration supplied at calibration launch"


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


def _root(model_key: str, calibration_id: str) -> Path:
    return calibration_root(
        "truthfulness", model_key, "h_infinity", calibration_id
    )


def _grid() -> list[dict[str, float | str]]:
    return [
        {
            "grid_id": f"q_{q_index:02d}_qf_{qf_index:02d}",
            "lambda": None,
            "q": float(q_ratio * FIXED_R),
            "r": FIXED_R,
            "q_final": float(qf_ratio * FIXED_R),
            "q_over_r": float(q_ratio),
            "q_final_over_r": float(qf_ratio),
        }
        for q_index, q_ratio in enumerate(Q_OVER_R)
        for qf_index, qf_ratio in enumerate(Q_FINAL_OVER_R)
    ]


def prepare(model_key: str, calibration_id: str) -> dict:
    """Freeze one disturbance split and one 50-question tuning set."""

    model = MODELS[model_key]
    evaluation._configure_runtime(model_key, calibration_id)
    evaluation.prepare("truthfulness", "id")
    root = _root(model_key, calibration_id)
    destination = root / "data.json"
    if destination.exists():
        return json.loads(destination.read_text())
    fit_path = artifact_root("truthfulness", model_key) / "data.json"
    fit = json.loads(fit_path.read_text())
    eval_path = dataset_root("truthfulness", model_key) / "truthfulness.json"
    eval_data = json.loads(eval_path.read_text())
    fit_records = [
        row
        for split in ("undesired", "desired", "jacobian")
        for row in fit["calibration"][split]
    ]
    excluded = {
        str(row.get("question_id", row.get("source_prompt_id", row["prompt_id"])))
        for row in fit_records
    }
    pool = [
        row for row in eval_data["evaluation"]["truthfulness"]["0"]
        if str(row["prompt_id"]) not in excluded
    ]
    if len(pool) < DISTURBANCE_SAMPLES + CALIBRATION_SAMPLES:
        raise ValueError("TruthfulQA cannot provide the disjoint H-infinity splits")
    selected = random.Random(SOURCE_RANDOM_SEED + 3).sample(
        pool, DISTURBANCE_SAMPLES + CALIBRATION_SAMPLES
    )
    payload = {
        "schema_version": 1,
        "model": [model.model_id, model.revision],
        "benchmark": "truthfulness",
        "disturbance": selected[:DISTURBANCE_SAMPLES],
        "tuning": selected[DISTURBANCE_SAMPLES:],
        "protocol": {
            "disturbance_samples": DISTURBANCE_SAMPLES,
            "tuning_samples": CALIBRATION_SAMPLES,
            "tuning_repetitions": CALIBRATION_REPETITIONS,
            "same_questions_across_repetitions": True,
            "disjoint_from_semantic_and_jacobian_fit_questions": True,
        },
    }
    _write_json(destination, payload)
    return payload


def _settings(
    model_key: str,
    *,
    q: float = 0.1,
    r: float = FIXED_R,
    q_final: float = 0.1,
) -> dict[str, object]:
    model = MODELS[model_key]
    counts = ALQR_CALIBRATION_COUNTS["truthfulness"]
    alqr = paper_alqr_setting("truthfulness", model.model_id)
    pid = SPID_SOURCE_GRIDS["truthfulness"][model_key]
    return {
        "behavior": "truthfulness", "seed": SOURCE_RANDOM_SEED,
        "fit_prompts_per_class": counts.undesired,
        "disturbance_prompts": DISTURBANCE_SAMPLES,
        "calibration_max_length": 512,
        "activation_batch_size": model.activation_batch_size,
        "jacobian_prompts": counts.jacobian,
        "jacobian_max_length": counts.jacobian_max_length,
        "jacobian_vjp_chunk_size": model.jacobian_vjp_chunk_size,
        "state_rank": 8, "numerical_floor": 1e-4,
        "alqr_setpoint_multiplier": alqr.multiplier,
        "spid_setpoint_multiplier": 1.0,
        "hinf_setpoint_multiplier": alqr.multiplier,
        "q": float(q), "r": float(r), "q_final": float(q_final),
        "alqr_q": alqr.q, "alqr_r": alqr.r, "alqr_q_final": alqr.q_final,
        "kp": pid.kp, "ki": pid.ki, "kd": pid.kd,
        "gamma_lower": 0.0, "gamma_upper": 100.0,
        "gamma_tolerance": 1e-5, "gamma_max_iterations": 100,
        "gamma_deployment_margin": 0.01,
        "model_loading": asdict(source_model_spec(
            "alqr", "truthfulness", model.model_id, model.revision
        )),
    }


def fit_base(
    model_key: str,
    device: str,
    calibration_id: str,
    *,
    q: float = 0.1,
    r: float = FIXED_R,
    q_final: float = 0.1,
) -> dict:
    model_spec = MODELS[model_key]
    data = prepare(model_key, calibration_id)
    fit = json.loads(
        (artifact_root("truthfulness", model_key) / "data.json").read_text()
    )
    calibration_data = {
        "negative": fit["calibration"]["undesired"],
        "positive": fit["calibration"]["desired"],
        "jacobian": fit["calibration"]["jacobian"],
        "disturbance": data["disturbance"],
        "dataset": fit["dataset"],
    }
    model, tokenizer = load_source_model(
        "alqr", "truthfulness", model_spec.model_id, model_spec.revision,
        device, load_access_token(REPO),
    )
    _artifact, metadata = calibrate_controller(
        model, tokenizer, model_label=model_spec.label, model_id=model_spec.model_id,
        cache_path=_root(model_key, calibration_id) / "base/controller.pt",
        nominal_dynamics_path=artifact_root("truthfulness", model_key) / "dynamics.pt",
        calibration_data=calibration_data,
        settings=_settings(model_key, q=q, r=r, q_final=q_final),
        controller_device=device,
    )
    return metadata


def _controller_from_base(
    model_key: str,
    calibration_id: str,
    parameters: dict[str, float],
    source: str,
    configuration_id: str,
) -> dict:
    root = _root(model_key, calibration_id)
    base = torch.load(
        root / "base/controller.pt", map_location="cpu", weights_only=True, mmap=True
    )
    artifact = ControllerArtifact(**base["artifact"])
    controller = {
        "identity": {
            "model": [MODELS[model_key].model_id, MODELS[model_key].revision],
            "task": "truthfulness",
            "calibration_id": calibration_id,
            "parameters": parameters,
            "configuration_source": source,
            "configuration_id": configuration_id,
        },
        "gains": artifact.hinf_gains.cpu(),
        "feasible": artifact.hinf_feasible,
        "gamma_star": artifact.gamma_star,
        "diagnostics": artifact.hinf_diagnostics,
    }
    if not controller["feasible"]:
        raise ValueError("The fixed H-infinity configuration is infeasible")
    _write_torch(root / "controller.pt", controller)
    return controller


def select_fixed(
    model_key: str,
    device: str,
    calibration_id: str,
    *,
    q_over_r: float,
    q_final_over_r: float,
    r: float,
) -> dict:
    """Synthesize and freeze one supplied H-infinity configuration without a sweep."""

    if min(q_over_r, q_final_over_r, r) <= 0:
        raise ValueError("Fixed H-infinity Q/R, Qf/R, and R must be positive")
    q = float(q_over_r * r)
    q_final = float(q_final_over_r * r)
    multiplier = paper_alqr_setting(
        "truthfulness", MODELS[model_key].model_id
    ).multiplier
    parameters = {
        "lambda": float(multiplier),
        "q": q,
        "r": float(r),
        "q_final": q_final,
    }
    prepare(model_key, calibration_id)
    metadata = fit_base(
        model_key,
        device,
        calibration_id,
        q=q,
        r=r,
        q_final=q_final,
    )
    controller = _controller_from_base(
        model_key,
        calibration_id,
        parameters,
        FIXED_SELECTION_SOURCE,
        "fixed",
    )
    diagnostic = diagnostic_root(_root(model_key, calibration_id) / "base/controller.pt")
    diagnostic_run = diagnostic / "runs" / (
        "calibration-" + str(metadata["fingerprint"])[:20]
    )
    if not diagnostic_run.exists():
        raise FileNotFoundError(f"Missing Hannah diagnostic bundle: {diagnostic_run}")
    payload = {
        "schema_version": 2,
        "model": [MODELS[model_key].model_id, MODELS[model_key].revision],
        "benchmark": "truthfulness",
        "calibration_id": calibration_id,
        "protocol": {
            "selection_strategy": "fixed",
            "selection_metric": None,
            "tuning_samples": 0,
            "tuning_repetitions": 0,
            "evaluated_model_kv_cache": False,
            "fixed_setpoint_multiplier": parameters["lambda"],
        },
        "selected": {
            "configuration_id": "fixed",
            "q_over_r": float(q_over_r),
            "q_final_over_r": float(q_final_over_r),
            "parameters": parameters,
            "gamma_star": controller["gamma_star"],
            "source": FIXED_SELECTION_SOURCE,
        },
        "diagnostic_bundle": str(diagnostic_run.relative_to(_root(model_key, calibration_id))),
    }
    _write_json(_root(model_key, calibration_id) / "selection.json", payload)
    return payload


def synthesize_grid(model_key: str, device: str, calibration_id: str) -> None:
    root = _root(model_key, calibration_id)
    base_path = root / "base/controller.pt"
    input_path = diagnostic_root(base_path) / "input.pt"
    base_payload = torch.load(base_path, map_location="cpu", weights_only=True, mmap=True)
    bundle = torch.load(input_path, map_location="cpu", weights_only=True, mmap=True)
    problem = FiniteHorizonControlProblem(**bundle["problem"])
    options = HInfinityOptions(**bundle["options"])
    source_settings = bundle["calibration"]["settings"]
    model = MODELS[model_key]
    multiplier = paper_alqr_setting("truthfulness", model.model_id).multiplier
    for configuration in _grid():
        configuration["lambda"] = multiplier
        destination = root / "grid/controllers" / f"{configuration['grid_id']}.pt"
        controller_identity = {
            "model": [model.model_id, model.revision],
            "task": "truthfulness",
            "calibration_id": calibration_id,
            "parameters": {
                name: float(configuration[name])
                for name in ("lambda", "q", "r", "q_final")
            },
            "configuration_source": GRID_SELECTION_SOURCE,
            "configuration_id": configuration["grid_id"],
        }
        if destination.exists():
            continue
        candidate = FiniteHorizonControlProblem(
            dynamics=problem.dynamics,
            control_channels=problem.control_channels,
            disturbance_channels=problem.disturbance_channels,
            state_costs=problem.state_costs * (
                float(configuration["q"]) / float(source_settings["q"])
            ),
            control_costs=problem.control_costs * (
                float(configuration["r"]) / float(source_settings["r"])
            ),
            terminal_cost=problem.terminal_cost * (
                float(configuration["q_final"]) / float(source_settings["q_final"])
            ),
        )
        solution = HInfinityController.synthesize(
            candidate, device=device, options=options
        ).solution()
        if not solution.feasible:
            raise ValueError(f"Infeasible H-infinity grid point: {configuration}")
        _write_torch(destination, {
            "identity": {
                **controller_identity,
            },
            "gains": solution.gains.cpu(), "feasible": solution.feasible,
            "gamma_star": solution.gamma_star,
            "diagnostics": solution.diagnostics,
        })


def generate_worker(
    model_key: str,
    device: str,
    shard_index: int,
    shard_count: int,
    calibration_id: str,
    generation_batch_size: int | None,
) -> None:
    root = _root(model_key, calibration_id)
    data = prepare(model_key, calibration_id)
    model_spec = MODELS[model_key]
    model, tokenizer = load_source_model(
        "alqr", "truthfulness", model_spec.model_id, model_spec.revision,
        device, load_access_token(REPO),
    )
    base_payload = torch.load(
        root / "base/controller.pt", map_location="cpu", weights_only=True, mmap=True
    )
    base = ControllerArtifact(**base_payload["artifact"])
    configurations = _grid()[shard_index::shard_count]
    multiplier = paper_alqr_setting("truthfulness", model_spec.model_id).multiplier
    batch_size = generation_batch_size or model_spec.activation_batch_size
    if batch_size < 1:
        raise ValueError("generation_batch_size must be positive")
    for configuration in configurations:
        configuration["lambda"] = multiplier
        grid_id = str(configuration["grid_id"])
        destination = root / "grid/generations" / f"{grid_id}.json"
        controller_path = root / "grid/controllers" / f"{grid_id}.pt"
        generation_identity = {
            "model": [model_spec.model_id, model_spec.revision],
            "configuration": configuration,
            "evaluated_model_kv_cache": False,
            "calibration_id": calibration_id,
            "generation_batch_size": batch_size,
        }
        saved = None
        if destination.exists():
            saved = json.loads(destination.read_text())
            if saved.get("status") == "complete":
                continue
        controller = torch.load(controller_path, map_location="cpu", weights_only=True)
        artifact = replace(
            base, hinf_gains=controller["gains"],
            hinf_feasible=bool(controller["feasible"]),
            gamma_star=float(controller["gamma_star"]),
            hinf_diagnostics=controller["diagnostics"],
        )
        policy = build_policy("hinf", artifact, kp=0.0, ki=0.0, kd=0.0)
        payload = saved or {
            "identity": generation_identity,
            "status": "partial", "repetitions": [],
            "runtime": runtime_provenance(device),
        }
        completed = [row["repetition"] for row in payload["repetitions"]]
        if completed != list(range(len(completed))):
            raise ValueError(f"Invalid calibration repetition prefix: {destination}")
        _write_json(destination, payload)
        prompts = [str(row["text"]) for row in data["tuning"]]
        for repetition in range(len(completed), CALIBRATION_REPETITIONS):
            completions = generate_batched(
                model, tokenizer, prompts, behavior="truthfulness",
                batch_size=batch_size,
                seed=SOURCE_RANDOM_SEED + repetition * SEED_STRIDE,
                use_cache=False,
                register_hooks=lambda: register_generation_policy_hooks(model, policy),
            )
            payload["repetitions"].append({
                "repetition": repetition,
                "rows": [
                    {
                        "grid_id": grid_id, "repetition": repetition,
                        "prompt_id": row["prompt_id"], "question": row["question"],
                        "text": row["text"], "completion": completion,
                    }
                    for row, completion in zip(data["tuning"], completions, strict=True)
                ],
            })
            _write_json(destination, payload)
        payload["status"] = "complete"
        _write_json(destination, payload)


def score_truth_grid(model_key: str, device: str, calibration_id: str) -> None:
    root = _root(model_key, calibration_id)
    specification = scorer_spec("truthfulqa_true")
    model_id = specification.model_id
    revision = specification.revision
    prompt_label = str(specification.prompt_label)
    destination = root / "grid/scores/truthfulqa_true.json"
    generation_paths = sorted((root / "grid/generations").glob("*.json"))
    identity = {
        "model": [model_id, revision],
        "scorer": "truthfulqa_true",
        "rubric": specification.rubric,
    }
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved.get("status") == "complete":
            return
    rows = []
    for path in generation_paths:
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete":
            raise ValueError(f"Incomplete calibration generation: {path}")
        rows.extend(
            row for repetition in payload["repetitions"] for row in repetition["rows"]
        )
    scorer_model, tokenizer = huggingface_scoring.load_scorer(
        model_id, revision, device, load_access_token(REPO)
    )
    if not destination.exists():
        saved = {"identity": identity, "status": "partial", "rows": []}
    _write_json(destination, saved)
    batch_size = huggingface_scoring.BATCH_SIZE
    for start in range(len(saved["rows"]), len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = [
            truth_judge_prompt(row["question"], row["completion"], prompt_label)
            for row in batch
        ]
        outputs = huggingface_scoring.score_batch(
            scorer_model, tokenizer, prompts, device
        )
        saved["rows"].extend([
            {
                "grid_id": row["grid_id"], "repetition": row["repetition"],
                "prompt_id": row["prompt_id"], **output,
            }
            for row, output in zip(batch, outputs, strict=True)
        ])
        _write_json(destination, saved)
    saved["status"] = "complete"
    _write_json(destination, saved)


def _freeze_selected_diagnostics(
    model_key: str,
    calibration_id: str,
    parameters: dict[str, float],
    controller: dict,
) -> Path:
    """Freeze Hannah's complete bundle for the controller selected from the grid."""

    root = _root(model_key, calibration_id)
    source_path = diagnostic_root(root / "base/controller.pt") / "input.pt"
    bundle = torch.load(source_path, map_location="cpu", weights_only=True)
    problem = dict(bundle["problem"])
    horizon = int(problem["dynamics"].shape[0])
    dimension = int(problem["dynamics"].shape[1])
    identity = torch.eye(dimension, dtype=problem["dynamics"].dtype)
    problem["state_costs"] = (
        float(parameters["q"]) * identity
    ).unsqueeze(0).repeat(horizon, 1, 1)
    problem["control_costs"] = (
        float(parameters["r"]) * identity
    ).unsqueeze(0).repeat(horizon, 1, 1)
    problem["terminal_cost"] = float(parameters["q_final"]) * identity
    bundle["problem"] = problem
    bundle["calibration"]["settings"].update({
        name: parameters[name] for name in ("q", "r", "q_final")
    })
    fingerprint = configuration_hash({
        "base_calibration_fingerprint": bundle["record"]["calibration_fingerprint"],
        "selected_parameters": parameters,
    })
    protocol_settings = {
        key: value
        for key, value in bundle["calibration"]["settings"].items()
        if key != "model_loading"
    }
    bundle["record"].update({
        "run_id": "calibration-" + fingerprint[:20],
        "calibration_fingerprint": fingerprint,
        "protocol_id": "full-reduced-state-" + configuration_hash(protocol_settings)[:16],
    })
    destination_root = root / "controller_diagnostics"
    input_path = destination_root / "input.pt"
    _write_torch(input_path, bundle)
    solution = HInfinityController(
        gains=controller["gains"],
        control_channels=problem["control_channels"],
        feasible=bool(controller["feasible"]),
        gamma_star=float(controller["gamma_star"]),
        diagnostics=controller["diagnostics"],
    ).solution()
    return freeze_diagnostic_score(
        input_path,
        "cpu",
        cache_root=destination_root,
        solution=solution,
    )


def select(model_key: str, calibration_id: str) -> dict:
    root = _root(model_key, calibration_id)
    truth = json.loads(
        (root / "grid/scores/truthfulqa_true.json").read_text()
    )["rows"]
    truth_map = {(r["grid_id"], r["repetition"], r["prompt_id"]): r["score"] for r in truth}
    summaries = []
    for configuration in _grid():
        grid_id = str(configuration["grid_id"])
        per_repetition = []
        for repetition in range(CALIBRATION_REPETITIONS):
            keys = [key for key in truth_map if key[0] == grid_id and key[1] == repetition]
            t = 100.0 * float(np.mean([truth_map[key] for key in keys]))
            per_repetition.append({"repetition": repetition, "truth": t})
        configuration["lambda"] = paper_alqr_setting(
            "truthfulness", MODELS[model_key].model_id
        ).multiplier
        summaries.append({
            **configuration,
            "truth": float(np.mean([row["truth"] for row in per_repetition])),
            "per_repetition": per_repetition,
        })
    selected = sorted(
        summaries,
        key=lambda row: (-row["truth"], row["q"], row["q_final"]),
    )[0]
    parameters = {
        name: float(selected[name]) for name in ("lambda", "q", "r", "q_final")
    }
    source_controller = root / "grid/controllers" / f"{selected['grid_id']}.pt"
    controller = torch.load(source_controller, map_location="cpu", weights_only=True)
    destination = root / "controller.pt"
    _write_torch(destination, controller)
    model = MODELS[model_key]
    payload = {
        "schema_version": 2,
        "model": [model.model_id, model.revision],
        "benchmark": "truthfulness",
        "calibration_id": calibration_id,
        "protocol": {
            "selection_strategy": "grid",
            "tuning_samples": CALIBRATION_SAMPLES,
            "tuning_repetitions": CALIBRATION_REPETITIONS,
            "evaluated_model_kv_cache": False,
            "selection_metric": "mean True percentage across repetitions",
            "q_over_r": list(Q_OVER_R), "q_final_over_r": list(Q_FINAL_OVER_R),
            "fixed_r": FIXED_R,
            "fixed_setpoint_multiplier": parameters["lambda"],
        },
        "selected": {
            **selected,
            "configuration_id": selected["grid_id"],
            "parameters": parameters,
            "source": GRID_SELECTION_SOURCE,
        },
        "grid": summaries,
    }
    payload["diagnostic_bundle"] = str(
        _freeze_selected_diagnostics(
            model_key,
            calibration_id,
            parameters,
            controller,
        ).relative_to(root)
    )
    _write_json(root / "selection.json", payload)
    return payload


def calibrate(
    model_key: str,
    devices: list[str],
    log_root: Path,
    calibration_id: str,
    generation_batch_size: int | None = None,
    fixed_parameters: dict[str, float] | None = None,
) -> None:
    evaluation._configure_runtime(model_key, calibration_id)
    selection = _root(model_key, calibration_id) / "selection.json"
    if selection.exists() and (_root(model_key, calibration_id) / "controller.pt").exists():
        return
    if fixed_parameters is not None:
        select_fixed(
            model_key,
            devices[0],
            calibration_id,
            q_over_r=float(fixed_parameters["q_over_r"]),
            q_final_over_r=float(fixed_parameters["q_final_over_r"]),
            r=float(fixed_parameters["r"]),
        )
        return
    prepare(model_key, calibration_id)
    fit_base(model_key, devices[0], calibration_id)
    synthesize_grid(model_key, devices[0], calibration_id)
    generation_jobs = [
        (
            f"hinf-grid-generate-{index:02d}",
            [
                sys.executable, "-m",
                "robust_steerability.benchmarks.truthfulness_calibration",
                "--stage", "generate-worker", "--model", model_key,
                "--device", "{device}", "--shard-index", str(index),
                "--shard-count", str(len(devices)),
                "--calibration-id", calibration_id,
                *(
                    ["--generation-batch-size", str(generation_batch_size)]
                    if generation_batch_size is not None
                    else []
                ),
            ],
        )
        for index, _device in enumerate(devices)
    ]
    run_jobs(generation_jobs, devices, log_root / "hinf-grid-generation")
    scoring_jobs = [(
        "hinf-grid-score-truthfulqa-true",
        [
            sys.executable, "-m",
            "robust_steerability.benchmarks.truthfulness_calibration",
            "--stage", "score-grid", "--model", model_key,
            "--device", "{device}",
            "--calibration-id", calibration_id,
        ],
    )]
    run_jobs(scoring_jobs, devices, log_root / "hinf-grid-scoring")
    select(model_key, calibration_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "prepare", "base", "synthesize", "generate-worker", "score-grid",
            "select", "select-fixed",
        ),
        required=True,
    )
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--device")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--calibration-id", default="selected")
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument("--q-over-r", type=float)
    parser.add_argument("--q-final-over-r", type=float)
    parser.add_argument("--r", type=float)
    arguments = parser.parse_args()
    if arguments.stage == "prepare":
        prepare(arguments.model, arguments.calibration_id)
    elif arguments.stage == "base":
        fit_base(arguments.model, arguments.device, arguments.calibration_id)
    elif arguments.stage == "synthesize":
        synthesize_grid(arguments.model, arguments.device, arguments.calibration_id)
    elif arguments.stage == "generate-worker":
        generate_worker(
            arguments.model,
            arguments.device,
            arguments.shard_index,
            arguments.shard_count,
            arguments.calibration_id,
            arguments.generation_batch_size,
        )
    elif arguments.stage == "score-grid":
        score_truth_grid(
            arguments.model,
            arguments.device,
            arguments.calibration_id,
        )
    elif arguments.stage == "select":
        print(json.dumps(
            select(arguments.model, arguments.calibration_id)["selected"],
            indent=2,
        ))
    else:
        if None in (arguments.q_over_r, arguments.q_final_over_r, arguments.r):
            raise ValueError("select-fixed requires --q-over-r, --q-final-over-r, and --r")
        if arguments.device is None:
            raise ValueError("select-fixed requires --device")
        print(json.dumps(
            select_fixed(
                arguments.model,
                arguments.device,
                arguments.calibration_id,
                q_over_r=arguments.q_over_r,
                q_final_over_r=arguments.q_final_over_r,
                r=arguments.r,
            )["selected"],
            indent=2,
        ))


if __name__ == "__main__":
    main()
