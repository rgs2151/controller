"""H-infinity calibration for one model's TruthfulQA task."""

from __future__ import annotations

import argparse
import json
import random
import sys
import tomllib
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from robust_steerability.benchmarks import truthfulness_runtime as evaluation
from robust_steerability.benchmarks.calibration import (
    require_nonzero_selection_metric,
    weighted_harmonic_mean,
)
from robust_steerability.benchmarks.composition import load_composition
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
from robust_steerability.benchmarks.metrics import truth_judge_prompt
from robust_steerability.judges import huggingface as huggingface_scoring
from robust_steerability.judges import openai as openai_scoring
from robust_steerability.judges.exact import harmonic_mean
from robust_steerability.judges.specs import scorer_cache_path, scorer_spec
from robust_steerability.modeling.huggingface import load_access_token, release_cuda_memory
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
COMPOSITION = load_composition("truthfulness")
CALIBRATION_CONFIG = tomllib.loads(
    (REPO / "benchmarks/truthfulness/benchmark.toml").read_text()
)["calibration"]
AVAILABLE_SELECTION_METRICS = COMPOSITION.calibration.available_selection_metrics
DEFAULT_SELECTION_METRIC = COMPOSITION.calibration.selection_metric
CALIBRATION_SAMPLES = int(CALIBRATION_CONFIG["h_infinity_tuning_samples"])
CALIBRATION_REPETITIONS = int(
    CALIBRATION_CONFIG["h_infinity_tuning_repetitions"]
)
if CALIBRATION_SAMPLES < 1 or CALIBRATION_REPETITIONS < 1:
    raise ValueError("Truthfulness calibration sample and repetition counts must be positive")
DISTURBANCE_SAMPLES = 200
SEED_STRIDE = 100_000
Q_OVER_R = (0.01, 10**-1.5, 0.1, 10**-0.5, 1.0, 10**0.5, 10.0, 10**1.5)
Q_FINAL_OVER_R = (0.01, 10**-1.5, 0.1, 10**-0.5)
FIXED_R = 1.0
GRID_SELECTION_SOURCES = {
    "truthfulqa_txi": "aggregate TruthfulQA True-times-Informative calibration-grid argmax",
    "truthfulqa_true_mean_percentage": "TruthfulQA True calibration-grid argmax",
    "mean_axbench_overall": "AXBench three-score harmonic-mean calibration argmax",
    "truthfulness_quality_composite": (
        "TruthfulQA True/instruction-relevance/fluency weighted-harmonic "
        "calibration argmax"
    ),
    "truthfulqa_txi_fluency_composite": (
        "TruthfulQA True-times-Informative/fluency weighted-sum "
        "calibration argmax"
    ),
}
GRID_CONFIGURATION_SOURCE = "truthfulness H-infinity Q/R and Qf/R calibration grid"
FIXED_SELECTION_SOURCE = "fixed configuration supplied at calibration launch"
API_SCORERS = (
    "axbench_concept_relevance",
    "axbench_instruction_relevance",
    "axbench_fluency",
)


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
    """Freeze disjoint disturbance and configured-size tuning splits."""

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
    # The shared H-infinity artifact schema records PID gains, but H-infinity
    # fitting does not use them.  Small-model truthfulness runs intentionally
    # omit S-PID, so they do not have (and must not require) an S-PID source
    # grid entry.  Preserve the source gains for models that do have one and
    # use inert placeholders otherwise.
    pid = SPID_SOURCE_GRIDS["truthfulness"].get(model_key)
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
        "kp": 0.0 if pid is None else pid.kp,
        "ki": 0.0 if pid is None else pid.ki,
        "kd": 0.0 if pid is None else pid.kd,
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
            "configuration_source": GRID_CONFIGURATION_SOURCE,
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


def _score_truthfulqa_grid(
    model_key: str,
    device: str,
    calibration_id: str,
    scorer_key: str,
) -> None:
    root = _root(model_key, calibration_id)
    if scorer_key not in {"truthfulqa_true", "truthfulqa_informative"}:
        raise ValueError(f"Unsupported TruthfulQA calibration scorer {scorer_key!r}")
    specification = scorer_spec(scorer_key)
    destination = root / "grid/scores" / f"{scorer_key}.json"
    generation_paths = sorted((root / "grid/generations").glob("*.json"))
    if len(generation_paths) != len(_grid()):
        raise ValueError("Truthfulness H-infinity grid generations are incomplete")
    identity = {
        "model": [specification.model_id, specification.revision],
        "scorer": scorer_key,
        "rubric": specification.rubric,
    }
    saved = {"identity": identity, "status": "partial", "rows": []}
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
    _write_json(destination, saved)
    scorer_model, tokenizer = huggingface_scoring.load_scorer(
        specification.model_id,
        str(specification.revision),
        device,
        load_access_token(REPO),
    )
    for start in range(len(saved["rows"]), len(rows), huggingface_scoring.BATCH_SIZE):
        batch = rows[start:start + huggingface_scoring.BATCH_SIZE]
        prompts = [
            truth_judge_prompt(
                str(row["question"]),
                str(row["completion"]),
                str(specification.prompt_label),
            )
            for row in batch
        ]
        outputs = huggingface_scoring.score_batch(
            scorer_model, tokenizer, prompts, device
        )
        saved["rows"].extend([
            {
                "grid_id": row["grid_id"],
                "repetition": row["repetition"],
                "prompt_id": row["prompt_id"],
                **output,
            }
            for row, output in zip(batch, outputs, strict=True)
        ])
        _write_json(destination, saved)
    del scorer_model, tokenizer
    release_cuda_memory(device)
    saved["status"] = "complete"
    _write_json(destination, saved)


def _required_api_scorers(selection_metric: str) -> tuple[str, ...]:
    if selection_metric in {"truthfulqa_true_mean_percentage", "truthfulqa_txi"}:
        return ()
    if selection_metric == "truthfulness_quality_composite":
        return ("axbench_instruction_relevance", "axbench_fluency")
    if selection_metric == "truthfulqa_txi_fluency_composite":
        return ("axbench_fluency",)
    if selection_metric == "mean_axbench_overall":
        return API_SCORERS
    raise ValueError(f"Unknown truthfulness calibration metric {selection_metric!r}")


def score_grid(
    model_key: str,
    device: str,
    calibration_id: str,
    *,
    selection_metric: str,
    api_concurrency: int,
    api_batch_size: int,
) -> None:
    if selection_metric not in AVAILABLE_SELECTION_METRICS:
        raise ValueError(f"Unknown truthfulness calibration metric {selection_metric!r}")
    root = _root(model_key, calibration_id)
    generation_paths = sorted((root / "grid/generations").glob("*.json"))
    if len(generation_paths) != len(_grid()):
        raise ValueError("Truthfulness H-infinity grid generations are incomplete")
    if selection_metric in {
        "truthfulqa_true_mean_percentage",
        "truthfulqa_txi",
        "truthfulness_quality_composite",
        "truthfulqa_txi_fluency_composite",
    }:
        _score_truthfulqa_grid(
            model_key, device, calibration_id, "truthfulqa_true"
        )
    if selection_metric in {"truthfulqa_txi", "truthfulqa_txi_fluency_composite"}:
        _score_truthfulqa_grid(
            model_key, device, calibration_id, "truthfulqa_informative"
        )
    api_scorers = _required_api_scorers(selection_metric)
    if api_scorers:
        openai_scoring.score_generations(
            generation_paths,
            root / "grid",
            list(api_scorers),
            concurrency=api_concurrency,
            batch_size=api_batch_size,
            row_defaults={"concept": evaluation.TRUTHFULNESS_CONCEPT},
        )
    if selection_metric == "mean_axbench_overall":
        for generation_path in generation_paths:
            mappings = []
            for scorer in API_SCORERS:
                payload = json.loads(
                    scorer_cache_path(root / "grid", generation_path, scorer).read_text()
                )
                mappings.append({
                    str(row["prompt_id"]): float(row["score"])
                    for row in payload["rows"]
                })
            rows = [
                {
                    "prompt_id": prompt_id,
                    "score": harmonic_mean([mapping[prompt_id] for mapping in mappings]),
                }
                for prompt_id in mappings[0]
            ]
            _write_json(
                scorer_cache_path(root / "grid", generation_path, "axbench_overall"),
                {"status": "complete", "rows": rows},
            )


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


def _truthfulqa_score_map(
    root: Path, scorer_key: str
) -> dict[tuple[str, int, str], float]:
    payload = json.loads((root / "grid/scores" / f"{scorer_key}.json").read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"TruthfulQA calibration scores are incomplete: {scorer_key}")
    return {
        (str(row["grid_id"]), int(row["repetition"]), str(row["prompt_id"])):
        float(row["score"])
        for row in payload["rows"]
    }


def _scorer_rows(root: Path, generation_path: Path, scorer: str) -> list[dict]:
    payload = json.loads(
        scorer_cache_path(root / "grid", generation_path, scorer).read_text()
    )
    if payload.get("status") != "complete":
        raise ValueError(f"Incomplete calibration score: {scorer}")
    return list(payload["rows"])


def _calibration_profile(
    model_key: str,
    calibration_id: str,
    selection_metric: str,
) -> dict:
    if selection_metric not in AVAILABLE_SELECTION_METRICS:
        raise ValueError(f"Unknown truthfulness calibration metric {selection_metric!r}")
    root = _root(model_key, calibration_id)
    truth_map = (
        _truthfulqa_score_map(root, "truthfulqa_true")
        if selection_metric in {
            "truthfulqa_true_mean_percentage",
            "truthfulqa_txi",
            "truthfulness_quality_composite",
            "truthfulqa_txi_fluency_composite",
        }
        else None
    )
    informative_map = (
        _truthfulqa_score_map(root, "truthfulqa_informative")
        if selection_metric in {"truthfulqa_txi", "truthfulqa_txi_fluency_composite"}
        else None
    )
    metric_config = CALIBRATION_CONFIG.get(selection_metric)
    if selection_metric == "truthfulness_quality_composite":
        quality_weights = (
            float(metric_config["truthfulqa_true_weight"]),
            float(metric_config["axbench_instruction_relevance_weight"]),
            float(metric_config["axbench_fluency_weight"]),
        )
        normalizer = float(metric_config["axbench_score_normalizer"])
    elif selection_metric == "truthfulqa_txi_fluency_composite":
        quality_weights = (
            float(metric_config["truthfulqa_true_informative_weight"]),
            float(metric_config["axbench_fluency_weight"]),
        )
        normalizer = float(metric_config["axbench_score_normalizer"])
    else:
        quality_weights = ()
        normalizer = 1.0
    if quality_weights and not np.isclose(sum(quality_weights), 1.0):
        raise ValueError("Truthfulness calibration weights must sum to one")
    if normalizer <= 0:
        raise ValueError("AXBench calibration score normalizer must be positive")
    summaries = []
    for configuration in _grid():
        grid_id = str(configuration["grid_id"])
        generation_path = (
            root / "grid/generations" / f"{grid_id}.json"
        )
        generation = json.loads(generation_path.read_text())
        flattened = [
            (int(repetition["repetition"]), row)
            for repetition in generation["repetitions"]
            for row in repetition["rows"]
        ]
        summary = {}
        if selection_metric == "mean_axbench_overall":
            for scorer in (*API_SCORERS, "axbench_overall"):
                rows = _scorer_rows(root, generation_path, scorer)
                summary[scorer] = float(np.mean([
                    float(row["score"]) for row in rows
                ]))
        elif selection_metric == "truthfulqa_true_mean_percentage":
            per_repetition = []
            for repetition in range(CALIBRATION_REPETITIONS):
                scores = [
                    truth_map[(grid_id, row_repetition, str(row["prompt_id"]))]
                    for row_repetition, row in flattened
                    if row_repetition == repetition
                ]
                per_repetition.append({
                    "repetition": repetition,
                    "truth": 100.0 * float(np.mean(scores)),
                })
            summary = {
                "truth": float(np.mean([row["truth"] for row in per_repetition])),
                "per_repetition": per_repetition,
            }
        elif selection_metric == "truthfulness_quality_composite":
            instruction_rows = _scorer_rows(
                root, generation_path, "axbench_instruction_relevance"
            )
            fluency_rows = _scorer_rows(
                root, generation_path, "axbench_fluency"
            )
            if len(flattened) != len(instruction_rows) or len(flattened) != len(fluency_rows):
                raise ValueError(f"Calibration scorer row-count mismatch for {grid_id}")
            response_scores = []
            truth_scores = []
            instruction_scores = []
            fluency_scores = []
            for (repetition, row), instruction, fluency in zip(
                flattened, instruction_rows, fluency_rows, strict=True
            ):
                prompt_id = str(row["prompt_id"])
                if (
                    str(instruction["prompt_id"]) != prompt_id
                    or str(fluency["prompt_id"]) != prompt_id
                ):
                    raise ValueError(f"Calibration scorer alignment mismatch for {grid_id}")
                truth = float(truth_map[(grid_id, repetition, prompt_id)])
                instruction_score = float(instruction["score"])
                fluency_score = float(fluency["score"])
                truth_scores.append(truth)
                instruction_scores.append(instruction_score)
                fluency_scores.append(fluency_score)
                response_scores.append(weighted_harmonic_mean(
                    (
                        truth,
                        float(np.clip(instruction_score / normalizer, 0.0, 1.0)),
                        float(np.clip(fluency_score / normalizer, 0.0, 1.0)),
                    ),
                    quality_weights,
                ))
            summary = {
                "truth": 100.0 * float(np.mean(truth_scores)),
                "axbench_instruction_relevance": float(np.mean(instruction_scores)),
                "axbench_fluency": float(np.mean(fluency_scores)),
                "truthfulness_quality_composite": float(np.mean(response_scores)),
            }
        else:
            fluency_rows = (
                _scorer_rows(root, generation_path, "axbench_fluency")
                if selection_metric == "truthfulqa_txi_fluency_composite"
                else None
            )
            if fluency_rows is not None and len(flattened) != len(fluency_rows):
                raise ValueError(f"Calibration scorer row-count mismatch for {grid_id}")
            truth_scores = []
            informative_scores = []
            fluency_scores = []
            for index, (repetition, row) in enumerate(flattened):
                prompt_id = str(row["prompt_id"])
                fluency = fluency_rows[index] if fluency_rows is not None else None
                if fluency is not None and str(fluency["prompt_id"]) != prompt_id:
                    raise ValueError(
                        f"Calibration scorer alignment mismatch for {grid_id}"
                    )
                truth = float(truth_map[(grid_id, repetition, prompt_id)])
                informative = float(
                    informative_map[(grid_id, repetition, prompt_id)]
                )
                truth_scores.append(truth)
                informative_scores.append(informative)
                if fluency is not None:
                    fluency_scores.append(float(fluency["score"]))
            truth_mean = float(np.mean(truth_scores))
            informative_mean = float(np.mean(informative_scores))
            # TruthfulQA's historical T×I statistic is the product of the two
            # aggregate acceptance rates, not the per-response joint pass rate.
            txi = truth_mean * informative_mean
            summary = {
                "truth": 100.0 * truth_mean,
                "info": 100.0 * informative_mean,
                "txi": 100.0 * txi,
            }
            if selection_metric == "truthfulqa_txi_fluency_composite":
                normalized_fluency_mean = float(np.mean([
                    np.clip(score / normalizer, 0.0, 1.0)
                    for score in fluency_scores
                ]))
                summary.update({
                    "axbench_fluency": float(np.mean(fluency_scores)),
                    "truthfulqa_txi_fluency_composite": (
                        quality_weights[0] * txi
                        + quality_weights[1] * normalized_fluency_mean
                    ),
                })
        configuration["lambda"] = paper_alqr_setting(
            "truthfulness", MODELS[model_key].model_id
        ).multiplier
        summaries.append({**configuration, **summary})
    if selection_metric == "truthfulqa_true_mean_percentage":
        metric_key = "truth"
        description = "mean TruthfulQA True percentage across repetitions"
        tie_breakers = ["smaller Q/R", "smaller Qf/R"]
        rank_key = lambda row: (-row["truth"], row["q"], row["q_final"])
    elif selection_metric == "mean_axbench_overall":
        metric_key = "axbench_overall"
        description = "mean per-response AXBench three-judge harmonic mean"
        tie_breakers = [
            "higher concept relevance",
            "higher instruction relevance",
            "higher fluency",
            "smaller Q/R",
            "smaller Qf/R",
        ]
        rank_key = lambda row: (
            -row["axbench_overall"],
            -row["axbench_concept_relevance"],
            -row["axbench_instruction_relevance"],
            -row["axbench_fluency"],
            row["q"],
            row["q_final"],
        )
    elif selection_metric == "truthfulness_quality_composite":
        metric_key = "truthfulness_quality_composite"
        description = (
            "mean per-response weighted harmonic mean of TruthfulQA True, "
            "AXBench instruction relevance, and AXBench fluency"
        )
        tie_breakers = [
            "higher True percentage",
            "higher instruction relevance",
            "higher fluency",
            "smaller Q/R",
            "smaller Qf/R",
        ]
        rank_key = lambda row: (
            -row["truthfulness_quality_composite"],
            -row["truth"],
            -row["axbench_instruction_relevance"],
            -row["axbench_fluency"],
            row["q"],
            row["q_final"],
        )
    elif selection_metric == "truthfulqa_txi_fluency_composite":
        metric_key = "truthfulqa_txi_fluency_composite"
        description = (
            "weighted sum of aggregate TruthfulQA True-times-Informative "
            "and mean normalized AXBench fluency"
        )
        tie_breakers = [
            "higher True-times-Informative percentage",
            "higher True percentage",
            "higher Informative percentage",
            "higher fluency",
            "smaller Q/R",
            "smaller Qf/R",
        ]
        rank_key = lambda row: (
            -row["truthfulqa_txi_fluency_composite"],
            -row["txi"],
            -row["truth"],
            -row["info"],
            -row["axbench_fluency"],
            row["q"],
            row["q_final"],
        )
    else:
        metric_key = "txi"
        description = (
            "product of aggregate TruthfulQA True and Informative acceptance rates"
        )
        tie_breakers = [
            "higher True percentage",
            "higher Informative percentage",
            "smaller Q/R",
            "smaller Qf/R",
        ]
        rank_key = lambda row: (
            -row["txi"],
            -row["truth"],
            -row["info"],
            row["q"],
            row["q_final"],
        )
    require_nonzero_selection_metric(
        summaries, metric_key, context="Truthfulness H-infinity calibration"
    )
    ranked = sorted(summaries, key=rank_key)
    model = MODELS[model_key]
    return {
        "schema_version": 1,
        "model": [model.model_id, model.revision],
        "benchmark": "truthfulness",
        "calibration_id": calibration_id,
        "selection_metric": selection_metric,
        "selection_metric_description": description,
        "metric_configuration": metric_config,
        "tie_breakers": tie_breakers,
        "selected": ranked[0],
        "ranking": [str(row["grid_id"]) for row in ranked],
        "grid": summaries,
    }


def select(model_key: str, calibration_id: str, selection_metric: str) -> dict:
    root = _root(model_key, calibration_id)
    previous_selection = (
        json.loads((root / "selection.json").read_text())
        if (root / "selection.json").exists()
        else None
    )
    profile = _calibration_profile(model_key, calibration_id, selection_metric)
    profile_path = root / "grid/selection_profiles" / f"{selection_metric}.json"
    _write_json(profile_path, profile)
    selected = dict(profile["selected"])
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
            "selection_metric": selection_metric,
            "selection_metric_description": profile["selection_metric_description"],
            "metric_configuration": profile["metric_configuration"],
            "tie_breakers": profile["tie_breakers"],
            "q_over_r": list(Q_OVER_R), "q_final_over_r": list(Q_FINAL_OVER_R),
            "fixed_r": FIXED_R,
            "fixed_setpoint_multiplier": parameters["lambda"],
        },
        "selected": {
            **selected,
            "configuration_id": selected["grid_id"],
            "parameters": parameters,
            "source": GRID_SELECTION_SOURCES[selection_metric],
        },
        "selection_profile": str(profile_path.relative_to(root)),
        "grid": profile["grid"],
    }
    previous_grid_id = (
        str(previous_selection.get("selected", {}).get("grid_id"))
        if previous_selection is not None
        else None
    )
    previous_bundle = (
        root / str(previous_selection.get("diagnostic_bundle", ""))
        if previous_selection is not None
        else None
    )
    if (
        previous_grid_id == str(selected["grid_id"])
        and previous_bundle is not None
        and previous_bundle.exists()
    ):
        payload["diagnostic_bundle"] = str(previous_bundle.relative_to(root))
    else:
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
    *,
    api_concurrency: int = openai_scoring.DEFAULT_CONCURRENCY,
    api_batch_size: int = openai_scoring.DEFAULT_BATCH_SIZE,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
    fixed_parameters: dict[str, float] | None = None,
) -> None:
    if selection_metric not in AVAILABLE_SELECTION_METRICS:
        raise ValueError(f"Unknown truthfulness calibration metric {selection_metric!r}")
    evaluation._configure_runtime(model_key, calibration_id)
    selection = _root(model_key, calibration_id) / "selection.json"
    if fixed_parameters is not None:
        controller = _root(model_key, calibration_id) / "controller.pt"
        if selection.exists() or controller.exists():
            if not selection.exists() or not controller.exists():
                raise ValueError(
                    f"Incomplete H-infinity calibration directory: {_root(model_key, calibration_id)}"
                )
            saved = json.loads(selection.read_text())
            expected = {
                "q_over_r": float(fixed_parameters["q_over_r"]),
                "q_final_over_r": float(fixed_parameters["q_final_over_r"]),
                "r": float(fixed_parameters["r"]),
            }
            actual_parameters = saved["selected"]["parameters"]
            actual = {
                "q_over_r": float(actual_parameters["q"] / actual_parameters["r"]),
                "q_final_over_r": float(
                    actual_parameters["q_final"] / actual_parameters["r"]
                ),
                "r": float(actual_parameters["r"]),
            }
            if actual != expected:
                raise ValueError(
                    f"Calibration ID {calibration_id!r} already contains {actual}, "
                    f"not requested {expected}; use a new calibration ID"
                )
            return
        select_fixed(
            model_key,
            devices[0],
            calibration_id,
            q_over_r=float(fixed_parameters["q_over_r"]),
            q_final_over_r=float(fixed_parameters["q_final_over_r"]),
            r=float(fixed_parameters["r"]),
        )
        return
    if selection.exists() and (_root(model_key, calibration_id) / "controller.pt").exists():
        saved = json.loads(selection.read_text())
        protocol = saved.get("protocol", {})
        if (
            protocol.get("selection_metric") == selection_metric
            and protocol.get("metric_configuration")
            == CALIBRATION_CONFIG.get(selection_metric)
        ):
            return
    prepare(model_key, calibration_id)
    fit_base(model_key, devices[0], calibration_id)
    release_cuda_memory(devices[0])
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
    score_grid(
        model_key,
        devices[0],
        calibration_id,
        selection_metric=selection_metric,
        api_concurrency=api_concurrency,
        api_batch_size=api_batch_size,
    )
    select(model_key, calibration_id, selection_metric)


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
    parser.add_argument(
        "--selection-metric",
        choices=AVAILABLE_SELECTION_METRICS,
        default=DEFAULT_SELECTION_METRIC,
    )
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument(
        "--api-concurrency", type=int, default=openai_scoring.DEFAULT_CONCURRENCY
    )
    parser.add_argument(
        "--api-batch-size", type=int, default=openai_scoring.DEFAULT_BATCH_SIZE
    )
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
        if arguments.device is None:
            raise ValueError("score-grid requires --device")
        score_grid(
            arguments.model,
            arguments.device,
            arguments.calibration_id,
            selection_metric=arguments.selection_metric,
            api_concurrency=arguments.api_concurrency,
            api_batch_size=arguments.api_batch_size,
        )
    elif arguments.stage == "select":
        print(json.dumps(
            select(
                arguments.model,
                arguments.calibration_id,
                arguments.selection_metric,
            )["selected"],
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
