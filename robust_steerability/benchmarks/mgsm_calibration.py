"""H-infinity calibration for MGSM Spanish language steering."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tomllib
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from robust_steerability.artifacts import configuration_hash
from robust_steerability.benchmarks import mgsm_artifacts as artifacts
from robust_steerability.benchmarks import mgsm_runtime as runtime
from robust_steerability.benchmarks.calibration import (
    require_nonzero_selection_metric,
    weighted_harmonic_mean,
)
from robust_steerability.benchmarks.composition import load_composition
from robust_steerability.benchmarks.launcher import run_jobs
from robust_steerability.benchmarks.layout import artifact_root, calibration_root
from robust_steerability.benchmarks.specs import MODELS
from robust_steerability.control import (
    FiniteHorizonControlProblem,
    HInfinityController,
    HInfinityOptions,
)
from robust_steerability.experiments.calibration import calibrate_controller, diagnostic_root
from robust_steerability.experiments.diagnostics import score as freeze_diagnostic_score
from robust_steerability.experiments.methods import ControllerArtifact, build_policy
from robust_steerability.judges import openai as openai_scoring
from robust_steerability.judges.exact import harmonic_mean
from robust_steerability.judges.mgsm import exact_match, extract_final_number
from robust_steerability.judges.specs import scorer_cache_path
from robust_steerability.modeling.huggingface import release_cuda_memory


BENCHMARK = artifacts.BENCHMARK
REPO = Path(__file__).resolve().parents[2]
Q_OVER_R = (0.01, 0.1, 1.0, 10.0)
Q_FINAL_OVER_R = (0.01, 0.1, 10**-0.5)
FIXED_R = 1.0
SETPOINT_MULTIPLIER = 1.5
COMPONENT_SCORERS = (
    "axbench_concept_relevance",
    "axbench_instruction_relevance",
    "axbench_fluency",
)
OVERALL_SCORER = "axbench_overall"
EXACT_SCORER = "mgsm_exact_match"
QUALITY_SCORER = "mgsm_quality_composite"
COMPOSITION = load_composition(BENCHMARK)
CALIBRATION_CONFIG = tomllib.loads(
    (REPO / "benchmarks/mgsm/benchmark.toml").read_text()
)["calibration"]
LAMBDA_SWEEP = COMPOSITION.calibration.h_infinity_lambda_sweep


def _selection_scorer(metric: str) -> str:
    scorers = {
        "mean_axbench_overall": OVERALL_SCORER,
        QUALITY_SCORER: QUALITY_SCORER,
    }
    try:
        return scorers[metric]
    except KeyError as error:
        raise ValueError(
            f"Unsupported MGSM H-infinity selection metric: {metric}"
        ) from error


LAMBDA_SELECTION_SCORER = _selection_scorer(LAMBDA_SWEEP.selection_metric)
GRID_SELECTION_SCORER = _selection_scorer(COMPOSITION.calibration.selection_metric)


def _root(model_key: str, calibration_id: str) -> Path:
    return calibration_root(BENCHMARK, model_key, "h_infinity", calibration_id)


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


def grid(multiplier: float = SETPOINT_MULTIPLIER) -> list[dict[str, float | str]]:
    return [
        {
            "grid_id": f"q_{q_index:02d}_qf_{qf_index:02d}",
            "lambda": float(multiplier),
            "q": float(q_ratio * FIXED_R),
            "r": FIXED_R,
            "q_final": float(qf_ratio * FIXED_R),
            "q_over_r": float(q_ratio),
            "q_final_over_r": float(qf_ratio),
        }
        for q_index, q_ratio in enumerate(Q_OVER_R)
        for qf_index, qf_ratio in enumerate(Q_FINAL_OVER_R)
    ]


def _settings(
    model_key: str,
    q: float,
    r: float,
    q_final: float,
    multiplier: float = SETPOINT_MULTIPLIER,
) -> dict[str, object]:
    model = MODELS[model_key]
    return {
        "behavior": BENCHMARK,
        "seed": 42,
        "fit_prompts_per_class": artifacts.DIRECTION_RECORDS_PER_CLASS,
        "disturbance_prompts": 200,
        "calibration_max_length": artifacts.MAX_CALIBRATION_LENGTH,
        "activation_batch_size": model.activation_batch_size,
        "jacobian_prompts": artifacts.JACOBIAN_PROMPTS,
        "jacobian_max_length": artifacts.MAX_CALIBRATION_LENGTH,
        "jacobian_vjp_chunk_size": model.jacobian_vjp_chunk_size,
        "state_rank": 8,
        "numerical_floor": 1e-4,
        "alqr_setpoint_multiplier": float(multiplier),
        "spid_setpoint_multiplier": float(multiplier),
        "hinf_setpoint_multiplier": float(multiplier),
        "q": float(q),
        "r": float(r),
        "q_final": float(q_final),
        "alqr_q": 0.1,
        "alqr_r": 1.0,
        "alqr_q_final": 0.1,
        "kp": 0.5,
        "ki": 0.5,
        "kd": 0.01,
        "gamma_lower": 0.0,
        "gamma_upper": 100.0,
        "gamma_tolerance": 1e-5,
        "gamma_max_iterations": 100,
        "gamma_deployment_margin": 0.01,
        "model_loading": asdict(artifacts.model_load_spec(model_key)),
    }


def fit_base(
    model_key: str,
    device: str,
    calibration_id: str,
    *,
    q: float = 0.1,
    r: float = 1.0,
    q_final: float = 0.1,
    multiplier: float = SETPOINT_MULTIPLIER,
    destination_root: Path | None = None,
) -> dict:
    data = artifacts.prepare(model_key)
    model, tokenizer = artifacts.load_model(model_key, device)
    setpoint = torch.load(
        artifact_root(BENCHMARK, model_key) / "setpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    calibration_data = {
        "negative": data["calibration"]["undesired"],
        "positive": data["calibration"]["desired"],
        "jacobian": data["calibration"]["jacobian"],
        "disturbance": data["calibration"]["disturbance"],
        "dataset": {
            "direction": "all 250 matched MGSM English/Spanish test questions",
            "disturbance": "frozen 200 GSM8K train questions",
            "tuning": "disjoint frozen 50 GSM8K train questions",
        },
    }
    root = destination_root or _root(model_key, calibration_id)
    _artifact, metadata = calibrate_controller(
        model,
        tokenizer,
        model_label=MODELS[model_key].label,
        model_id=MODELS[model_key].model_id,
        cache_path=root / "base/controller.pt",
        nominal_dynamics_path=artifact_root(BENCHMARK, model_key) / "dynamics.pt",
        calibration_data=calibration_data,
        settings=_settings(model_key, q, r, q_final, multiplier),
        controller_device=device,
        semantic_calibration={
            "contrast": setpoint["contrast"],
            "feature_norm": setpoint["feature_norm"],
        },
    )
    return metadata


def lambda_candidates() -> list[dict[str, float | str]]:
    """Return the configured first-phase setpoint sweep."""

    return [
        {
            "configuration_id": f"lambda_{index:02d}",
            "lambda": float(multiplier),
            "q": float(LAMBDA_SWEEP.fixed_q_over_r * LAMBDA_SWEEP.fixed_r),
            "r": float(LAMBDA_SWEEP.fixed_r),
            "q_final": float(
                LAMBDA_SWEEP.fixed_q_final_over_r * LAMBDA_SWEEP.fixed_r
            ),
            "q_over_r": float(LAMBDA_SWEEP.fixed_q_over_r),
            "q_final_over_r": float(LAMBDA_SWEEP.fixed_q_final_over_r),
        }
        for index, multiplier in enumerate(LAMBDA_SWEEP.values)
    ]


def _lambda_candidate_root(
    model_key: str, calibration_id: str, configuration_id: str
) -> Path:
    return (
        _root(model_key, calibration_id)
        / "lambda_sweep"
        / "candidates"
        / configuration_id
    )


def _lambda_selection_path(model_key: str, calibration_id: str) -> Path:
    return _root(model_key, calibration_id) / "lambda_sweep" / "selection.json"


def _lambda_generation_path(
    model_key: str, calibration_id: str, configuration_id: str
) -> Path:
    return (
        _root(model_key, calibration_id)
        / "lambda_sweep"
        / "generations"
        / f"{configuration_id}.json"
    )


def selected_multiplier(model_key: str, calibration_id: str) -> float:
    if not LAMBDA_SWEEP.enabled:
        return SETPOINT_MULTIPLIER
    payload = json.loads(_lambda_selection_path(model_key, calibration_id).read_text())
    return float(payload["selected"]["lambda"])


def _q_grid_root(model_key: str, calibration_id: str) -> Path:
    root = _root(model_key, calibration_id)
    if not LAMBDA_SWEEP.enabled:
        return root / "grid"
    selection = json.loads(_lambda_selection_path(model_key, calibration_id).read_text())
    return root / "q_qf_sweep" / str(selection["selected"]["configuration_id"])


def generate_lambda_candidate(
    model_key: str,
    device: str,
    candidate_index: int,
    calibration_id: str,
    generation_batch_size: int | None,
) -> None:
    """Fit and evaluate one genuine H-infinity setpoint candidate."""

    configuration = lambda_candidates()[candidate_index]
    candidate_root = _lambda_candidate_root(
        model_key, calibration_id, str(configuration["configuration_id"])
    )
    destination = _lambda_generation_path(
        model_key, calibration_id, str(configuration["configuration_id"])
    )
    if runtime.generation_complete(destination):
        return
    fit_base(
        model_key,
        device,
        calibration_id,
        q=float(configuration["q"]),
        r=float(configuration["r"]),
        q_final=float(configuration["q_final"]),
        multiplier=float(configuration["lambda"]),
        destination_root=candidate_root,
    )
    release_cuda_memory(device)
    data = artifacts.prepare(model_key)
    model, tokenizer = artifacts.load_model(model_key, device)
    base_payload = torch.load(
        candidate_root / "base/controller.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    artifact = ControllerArtifact(**base_payload["artifact"])
    policy = build_policy("hinf", artifact, kp=0.0, ki=0.0, kd=0.0)
    prompts = [
        runtime.format_calibration_prompt(tokenizer, str(row["text"]))
        for row in data["calibration"]["tuning"]
    ]
    batch_size = generation_batch_size or MODELS[model_key].activation_batch_size
    completions, generated = runtime.generate_completions(
        model,
        tokenizer,
        prompts,
        policy=policy,
        use_cache=False,
        batch_size=batch_size,
        max_new_tokens=256,
    )
    rows = [
        {
            "prompt_id": row["prompt_id"],
            "configuration_id": configuration["configuration_id"],
            "text": row["text"],
            "completion": completion,
            "concept": artifacts.CONCEPT,
            "generated_tokens": count,
        }
        for row, completion, count in zip(
            data["calibration"]["tuning"], completions, generated, strict=True
        )
    ]
    _write_json(
        destination,
        {
            "identity": {
                "model": [MODELS[model_key].model_id, MODELS[model_key].revision],
                "configuration": configuration,
                "calibration_id": calibration_id,
                "evaluated_model_kv_cache": False,
                "generation_batch_size": batch_size,
            },
            "status": "complete",
            "repetitions": [
                {"repetition": 0, "sample_count": len(rows), "rows": rows}
            ],
        },
    )


def _score_axbench_generations(
    generations: list[Path],
    scorer_root: Path,
    *,
    api_concurrency: int,
    api_batch_size: int,
) -> None:
    openai_scoring.score_generations(
        generations,
        scorer_root,
        list(COMPONENT_SCORERS),
        concurrency=api_concurrency,
        batch_size=api_batch_size,
    )
    for generation in generations:
        maps = []
        for scorer in COMPONENT_SCORERS:
            payload = json.loads(
                scorer_cache_path(scorer_root, generation, scorer).read_text()
            )
            maps.append(
                {row["prompt_id"]: float(row["score"]) for row in payload["rows"]}
            )
        rows = [
            {
                "prompt_id": prompt_id,
                "score": harmonic_mean([mapping[prompt_id] for mapping in maps]),
            }
            for prompt_id in maps[0]
        ]
        _write_json(
            scorer_cache_path(scorer_root, generation, OVERALL_SCORER),
            {"status": "complete", "rows": rows},
        )


def _score_exact_generations(
    model_key: str,
    generations: list[Path],
    scorer_root: Path,
) -> None:
    """Score calibration answers with the same exact-number rule as evaluation."""

    tuning = artifacts.prepare(model_key)["calibration"]["tuning"]
    answers = {}
    for row in tuning:
        answer = extract_final_number(str(row["answer"]))
        if answer is None:
            raise ValueError(f"Could not parse GSM8K calibration answer: {row['prompt_id']}")
        answers[str(row["prompt_id"])] = answer
    for generation in generations:
        destination = scorer_cache_path(scorer_root, generation, EXACT_SCORER)
        if destination.exists():
            saved = json.loads(destination.read_text())
            if saved.get("status") == "complete":
                continue
        payload = json.loads(generation.read_text())
        rows = []
        for repetition in payload["repetitions"]:
            for row in repetition["rows"]:
                prompt_id = str(row["prompt_id"])
                result = exact_match(str(row["completion"]), answers[prompt_id])
                rows.append({
                    "prompt_id": prompt_id,
                    "repetition": int(repetition["repetition"]),
                    **result,
                })
        _write_json(destination, {"status": "complete", "rows": rows})


def score_lambda_sweep(
    model_key: str,
    calibration_id: str,
    *,
    api_concurrency: int,
    api_batch_size: int,
) -> None:
    sweep_root = _root(model_key, calibration_id) / "lambda_sweep"
    generations = [
        _lambda_generation_path(
            model_key, calibration_id, str(configuration["configuration_id"])
        )
        for configuration in lambda_candidates()
    ]
    if any(not runtime.generation_complete(path) for path in generations):
        raise ValueError("MGSM H-infinity lambda sweep generations are incomplete")
    _score_axbench_generations(
        generations,
        sweep_root,
        api_concurrency=api_concurrency,
        api_batch_size=api_batch_size,
    )


def select_lambda(model_key: str, calibration_id: str) -> dict:
    root = _root(model_key, calibration_id)
    sweep_root = root / "lambda_sweep"
    summaries = []
    for configuration in lambda_candidates():
        generation = _lambda_generation_path(
            model_key, calibration_id, str(configuration["configuration_id"])
        )
        means = {}
        for scorer in (*COMPONENT_SCORERS, OVERALL_SCORER):
            payload = json.loads(
                scorer_cache_path(sweep_root, generation, scorer).read_text()
            )
            means[scorer] = float(
                np.mean([float(row["score"]) for row in payload["rows"]])
            )
        summaries.append({**configuration, **means})
    require_nonzero_selection_metric(
        summaries,
        LAMBDA_SELECTION_SCORER,
        context="MGSM H-infinity lambda calibration",
    )
    selected = sorted(
        summaries,
        key=lambda row: (-row[LAMBDA_SELECTION_SCORER], row["lambda"]),
    )[0]
    selected_root = _lambda_candidate_root(
        model_key, calibration_id, str(selected["configuration_id"])
    )
    canonical_base = root / "base"
    canonical_base.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_root / "base/controller.pt", canonical_base / "controller.pt")
    source_diagnostics = diagnostic_root(selected_root / "base/controller.pt")
    destination_diagnostics = diagnostic_root(canonical_base / "controller.pt")
    if destination_diagnostics.exists():
        shutil.rmtree(destination_diagnostics)
    shutil.copytree(source_diagnostics, destination_diagnostics)
    payload = {
        "schema_version": 1,
        "model": [MODELS[model_key].model_id, MODELS[model_key].revision],
        "benchmark": BENCHMARK,
        "calibration_id": calibration_id,
        "protocol": {
            "enabled": True,
            "selection_metric": LAMBDA_SWEEP.selection_metric,
            "values": list(LAMBDA_SWEEP.values),
            "fixed_q_over_r": LAMBDA_SWEEP.fixed_q_over_r,
            "fixed_q_final_over_r": LAMBDA_SWEEP.fixed_q_final_over_r,
            "fixed_r": LAMBDA_SWEEP.fixed_r,
            "tuning_samples": len(
                artifacts.prepare(model_key)["calibration"]["tuning"]
            ),
            "tuning_repetitions": 1,
            "evaluated_model_kv_cache": False,
        },
        "selected": selected,
        "candidates": summaries,
    }
    _write_json(_lambda_selection_path(model_key, calibration_id), payload)
    return payload


def synthesize_grid(model_key: str, device: str, calibration_id: str) -> None:
    root = _root(model_key, calibration_id)
    multiplier = selected_multiplier(model_key, calibration_id)
    grid_root = _q_grid_root(model_key, calibration_id)
    bundle = torch.load(
        diagnostic_root(root / "base/controller.pt") / "input.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    problem = FiniteHorizonControlProblem(**bundle["problem"])
    options = HInfinityOptions(**bundle["options"])
    settings = bundle["calibration"]["settings"]
    for configuration in grid(multiplier):
        destination = grid_root / "controllers" / f"{configuration['grid_id']}.pt"
        if destination.exists():
            continue
        candidate = FiniteHorizonControlProblem(
            dynamics=problem.dynamics,
            control_channels=problem.control_channels,
            disturbance_channels=problem.disturbance_channels,
            state_costs=problem.state_costs
            * (float(configuration["q"]) / float(settings["q"])),
            control_costs=problem.control_costs
            * (float(configuration["r"]) / float(settings["r"])),
            terminal_cost=problem.terminal_cost
            * (float(configuration["q_final"]) / float(settings["q_final"])),
        )
        solution = HInfinityController.synthesize(
            candidate, device=device, options=options
        ).solution()
        if not solution.feasible or solution.gamma_star is None:
            raise ValueError(f"Infeasible MGSM H-infinity point: {configuration}")
        _write_torch(
            destination,
            {
                "identity": {
                    "model": [MODELS[model_key].model_id, MODELS[model_key].revision],
                    "benchmark": BENCHMARK,
                    "calibration_id": calibration_id,
                    "configuration_id": configuration["grid_id"],
                    "parameters": {
                        key: float(configuration[key])
                        for key in ("lambda", "q", "r", "q_final")
                    },
                },
                "gains": solution.gains.cpu(),
                "feasible": solution.feasible,
                "gamma_star": solution.gamma_star,
                "diagnostics": solution.diagnostics,
            },
        )


def generate_worker(
    model_key: str,
    device: str,
    shard_index: int,
    shard_count: int,
    calibration_id: str,
    generation_batch_size: int | None,
) -> None:
    root = _root(model_key, calibration_id)
    multiplier = selected_multiplier(model_key, calibration_id)
    grid_root = _q_grid_root(model_key, calibration_id)
    data = artifacts.prepare(model_key)
    model, tokenizer = artifacts.load_model(model_key, device)
    base_payload = torch.load(
        root / "base/controller.pt", map_location="cpu", weights_only=True, mmap=True
    )
    base = ControllerArtifact(**base_payload["artifact"])
    prompts = [
        runtime.format_calibration_prompt(tokenizer, str(row["text"]))
        for row in data["calibration"]["tuning"]
    ]
    batch_size = generation_batch_size or MODELS[model_key].activation_batch_size
    for configuration in grid(multiplier)[shard_index::shard_count]:
        grid_id = str(configuration["grid_id"])
        destination = grid_root / "generations" / f"{grid_id}.json"
        if runtime.generation_complete(destination):
            continue
        controller = torch.load(
            grid_root / "controllers" / f"{grid_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        artifact = replace(
            base,
            hinf_gains=controller["gains"],
            hinf_feasible=bool(controller["feasible"]),
            gamma_star=float(controller["gamma_star"]),
            hinf_diagnostics=controller["diagnostics"],
        )
        policy = build_policy("hinf", artifact, kp=0.0, ki=0.0, kd=0.0)
        completions, generated = runtime.generate_completions(
            model,
            tokenizer,
            prompts,
            policy=policy,
            use_cache=False,
            batch_size=batch_size,
            max_new_tokens=256,
        )
        rows = [
            {
                "prompt_id": row["prompt_id"],
                "grid_id": grid_id,
                "text": row["text"],
                "completion": completion,
                "concept": artifacts.CONCEPT,
                "generated_tokens": count,
            }
            for row, completion, count in zip(
                data["calibration"]["tuning"], completions, generated, strict=True
            )
        ]
        _write_json(
            destination,
            {
                "identity": {
                    "model": [MODELS[model_key].model_id, MODELS[model_key].revision],
                    "configuration": configuration,
                    "calibration_id": calibration_id,
                    "evaluated_model_kv_cache": False,
                    "generation_batch_size": batch_size,
                },
                "status": "complete",
                "repetitions": [
                    {"repetition": 0, "sample_count": len(rows), "rows": rows}
                ],
            },
        )


def score_grid(
    model_key: str,
    calibration_id: str,
    *,
    api_concurrency: int,
    api_batch_size: int,
) -> None:
    root = _q_grid_root(model_key, calibration_id)
    generations = sorted((root / "generations").glob("*.json"))
    if len(generations) != len(grid(selected_multiplier(model_key, calibration_id))):
        raise ValueError("MGSM H-infinity grid generations are incomplete")
    _score_axbench_generations(
        generations,
        root,
        api_concurrency=api_concurrency,
        api_batch_size=api_batch_size,
    )
    _score_exact_generations(model_key, generations, root)


def _freeze_selected_diagnostics(
    model_key: str,
    calibration_id: str,
    parameters: dict[str, float],
    controller: dict,
) -> Path:
    root = _root(model_key, calibration_id)
    source = diagnostic_root(root / "base/controller.pt") / "input.pt"
    bundle = torch.load(source, map_location="cpu", weights_only=True)
    problem = dict(bundle["problem"])
    horizon = int(problem["dynamics"].shape[0])
    dimension = int(problem["dynamics"].shape[1])
    identity = torch.eye(dimension, dtype=problem["dynamics"].dtype)
    problem["state_costs"] = parameters["q"] * identity.unsqueeze(0).repeat(horizon, 1, 1)
    problem["control_costs"] = parameters["r"] * identity.unsqueeze(0).repeat(horizon, 1, 1)
    problem["terminal_cost"] = parameters["q_final"] * identity
    bundle["problem"] = problem
    bundle["calibration"]["settings"].update(
        {key: parameters[key] for key in ("q", "r", "q_final")}
    )
    fingerprint = configuration_hash(
        {
            "base_calibration_fingerprint": bundle["record"]["calibration_fingerprint"],
            "selected_parameters": parameters,
        }
    )
    bundle["record"].update(
        {"run_id": "calibration-" + fingerprint[:20], "calibration_fingerprint": fingerprint}
    )
    destination = root / "controller_diagnostics/input.pt"
    _write_torch(destination, bundle)
    solution = HInfinityController(
        gains=controller["gains"],
        control_channels=problem["control_channels"],
        feasible=bool(controller["feasible"]),
        gamma_star=float(controller["gamma_star"]),
        diagnostics=controller["diagnostics"],
    ).solution()
    return freeze_diagnostic_score(
        destination, "cpu", cache_root=destination.parent, solution=solution
    )


def select(model_key: str, calibration_id: str) -> dict:
    root = _root(model_key, calibration_id)
    multiplier = selected_multiplier(model_key, calibration_id)
    grid_root = _q_grid_root(model_key, calibration_id)
    quality_config = CALIBRATION_CONFIG[QUALITY_SCORER]
    quality_weights = (
        float(quality_config["mgsm_exact_match_weight"]),
        float(quality_config["axbench_overall_weight"]),
    )
    if not np.isclose(sum(quality_weights), 1.0):
        raise ValueError("MGSM-quality calibration weights must sum to one")
    normalizer = float(quality_config["axbench_score_normalizer"])
    if normalizer <= 0:
        raise ValueError("AXBench calibration score normalizer must be positive")
    summaries = []
    for configuration in grid(multiplier):
        generation = grid_root / "generations" / f"{configuration['grid_id']}.json"
        means = {}
        for scorer in (*COMPONENT_SCORERS, OVERALL_SCORER):
            payload = json.loads(
                scorer_cache_path(grid_root, generation, scorer).read_text()
            )
            means[scorer] = float(np.mean([float(row["score"]) for row in payload["rows"]]))
        exact_rows = json.loads(
            scorer_cache_path(grid_root, generation, EXACT_SCORER).read_text()
        )["rows"]
        overall_rows = json.loads(
            scorer_cache_path(grid_root, generation, OVERALL_SCORER).read_text()
        )["rows"]
        exact_by_prompt = {
            str(row["prompt_id"]): float(row["score"]) for row in exact_rows
        }
        overall_by_prompt = {
            str(row["prompt_id"]): float(row["score"]) for row in overall_rows
        }
        if exact_by_prompt.keys() != overall_by_prompt.keys():
            raise ValueError(
                f"MGSM calibration scorer alignment mismatch for {configuration['grid_id']}"
            )
        response_scores = [
            weighted_harmonic_mean(
                (
                    exact_by_prompt[prompt_id],
                    float(np.clip(overall_by_prompt[prompt_id] / normalizer, 0.0, 1.0)),
                ),
                quality_weights,
            )
            for prompt_id in exact_by_prompt
        ]
        means[EXACT_SCORER] = float(np.mean(list(exact_by_prompt.values())))
        means[QUALITY_SCORER] = float(np.mean(response_scores))
        summaries.append({**configuration, **means})
    require_nonzero_selection_metric(
        summaries,
        GRID_SELECTION_SCORER,
        context="MGSM H-infinity Q/Qf calibration",
    )
    if COMPOSITION.calibration.selection_metric == QUALITY_SCORER:
        rank_key = lambda row: (
            -row[QUALITY_SCORER],
            -row[EXACT_SCORER],
            -row[OVERALL_SCORER],
            row["q"],
            row["q_final"],
        )
        selection_description = (
            "mean per-response weighted harmonic mean of exact-answer accuracy "
            "and normalized AXBench Overall"
        )
        selection_source = (
            "MGSM exact-accuracy/AXBench-Overall weighted-harmonic calibration argmax"
        )
        metric_configuration = quality_config
    else:
        rank_key = lambda row: (
            -row[OVERALL_SCORER],
            row["q"],
            row["q_final"],
        )
        selection_description = "mean per-response AXBench three-judge harmonic mean"
        selection_source = "AXBench three-judge harmonic-mean calibration argmax"
        metric_configuration = None
    selected = sorted(summaries, key=rank_key)[0]
    parameters = {key: float(selected[key]) for key in ("lambda", "q", "r", "q_final")}
    controller = torch.load(
        grid_root / "controllers" / f"{selected['grid_id']}.pt",
        map_location="cpu",
        weights_only=True,
    )
    _write_torch(root / "controller.pt", controller)
    diagnostic = _freeze_selected_diagnostics(model_key, calibration_id, parameters, controller)
    payload = {
        "schema_version": 1,
        "model": [MODELS[model_key].model_id, MODELS[model_key].revision],
        "benchmark": BENCHMARK,
        "calibration_id": calibration_id,
        "protocol": {
            "selection_strategy": "grid",
            "selection_metric": COMPOSITION.calibration.selection_metric,
            "selection_metric_description": selection_description,
            "metric_configuration": metric_configuration,
            "tuning_samples": 50,
            "tuning_repetitions": 1,
            "evaluated_model_kv_cache": False,
            "q_over_r": list(Q_OVER_R),
            "q_final_over_r": list(Q_FINAL_OVER_R),
            "fixed_r": FIXED_R,
            "fixed_setpoint_multiplier": multiplier,
            "lambda_sweep_enabled": LAMBDA_SWEEP.enabled,
            "lambda_selection": (
                str(_lambda_selection_path(model_key, calibration_id).relative_to(root))
                if LAMBDA_SWEEP.enabled
                else None
            ),
        },
        "selected": {
            **selected,
            "configuration_id": selected["grid_id"],
            "parameters": parameters,
            "gamma_star": float(controller["gamma_star"]),
            "source": selection_source,
        },
        "grid": summaries,
        "diagnostic_bundle": str(diagnostic.relative_to(root)),
    }
    _write_json(root / "selection.json", payload)
    return payload


def select_fixed(
    model_key: str,
    device: str,
    calibration_id: str,
    *,
    q_over_r: float,
    q_final_over_r: float,
    r: float,
) -> dict:
    if min(q_over_r, q_final_over_r, r) <= 0:
        raise ValueError("Fixed H-infinity Q/R, Qf/R, and R must be positive")
    q, q_final = q_over_r * r, q_final_over_r * r
    fit_base(model_key, device, calibration_id, q=q, r=r, q_final=q_final)
    base = torch.load(
        _root(model_key, calibration_id) / "base/controller.pt",
        map_location="cpu",
        weights_only=True,
    )
    artifact = ControllerArtifact(**base["artifact"])
    parameters = {
        "lambda": SETPOINT_MULTIPLIER,
        "q": float(q),
        "r": float(r),
        "q_final": float(q_final),
    }
    controller = {
        "identity": {"configuration_id": "fixed", "parameters": parameters},
        "gains": artifact.hinf_gains,
        "feasible": artifact.hinf_feasible,
        "gamma_star": artifact.gamma_star,
        "diagnostics": artifact.hinf_diagnostics,
    }
    _write_torch(_root(model_key, calibration_id) / "controller.pt", controller)
    diagnostic = diagnostic_root(_root(model_key, calibration_id) / "base/controller.pt")
    payload = {
        "schema_version": 1,
        "model": [MODELS[model_key].model_id, MODELS[model_key].revision],
        "benchmark": BENCHMARK,
        "calibration_id": calibration_id,
        "protocol": {"selection_strategy": "fixed", "evaluated_model_kv_cache": False},
        "selected": {
            "configuration_id": "fixed",
            "q_over_r": float(q_over_r),
            "q_final_over_r": float(q_final_over_r),
            "parameters": parameters,
            "gamma_star": artifact.gamma_star,
            "source": "fixed configuration supplied at calibration launch",
        },
        "diagnostic_bundle": str(diagnostic.relative_to(_root(model_key, calibration_id))),
    }
    _write_json(_root(model_key, calibration_id) / "selection.json", payload)
    return payload


def calibrate(
    model_key: str,
    devices: list[str],
    log_root: Path,
    calibration_id: str,
    generation_batch_size: int | None,
    *,
    api_concurrency: int,
    api_batch_size: int,
    fixed_parameters: dict[str, float] | None,
) -> None:
    root = _root(model_key, calibration_id)
    if fixed_parameters is not None:
        if (root / "selection.json").exists() and (root / "controller.pt").exists():
            return
        select_fixed(model_key, devices[0], calibration_id, **fixed_parameters)
        return
    if (root / "selection.json").exists() and (root / "controller.pt").exists():
        saved = json.loads((root / "selection.json").read_text())
        protocol = saved.get("protocol", {})
        if (
            protocol.get("selection_metric")
            == COMPOSITION.calibration.selection_metric
            and bool(protocol.get("lambda_sweep_enabled", False))
            == LAMBDA_SWEEP.enabled
        ):
            return
    if LAMBDA_SWEEP.enabled:
        lambda_selection = _lambda_selection_path(model_key, calibration_id)
        if lambda_selection.exists():
            saved_lambda = json.loads(lambda_selection.read_text())
            expected_protocol = {
                "selection_metric": LAMBDA_SWEEP.selection_metric,
                "values": list(LAMBDA_SWEEP.values),
                "fixed_q_over_r": LAMBDA_SWEEP.fixed_q_over_r,
                "fixed_q_final_over_r": LAMBDA_SWEEP.fixed_q_final_over_r,
                "fixed_r": LAMBDA_SWEEP.fixed_r,
            }
            actual_protocol = saved_lambda.get("protocol", {})
            actual = {
                key: actual_protocol.get(key) for key in expected_protocol
            }
            if actual != expected_protocol:
                raise ValueError(
                    f"Calibration ID {calibration_id!r} already contains a different "
                    "lambda sweep; use a new calibration ID"
                )
            if not (root / "base/controller.pt").exists():
                select_lambda(model_key, calibration_id)
        else:
            lambda_jobs = [
                (
                    f"hinf-lambda-{index:02d}",
                    [
                        sys.executable,
                        "-m",
                        "robust_steerability.benchmarks.mgsm_calibration",
                        "--stage",
                        "lambda-worker",
                        "--model",
                        model_key,
                        "--device",
                        "{device}",
                        "--candidate-index",
                        str(index),
                        "--calibration-id",
                        calibration_id,
                        *(
                            ["--generation-batch-size", str(generation_batch_size)]
                            if generation_batch_size is not None
                            else []
                        ),
                    ],
                )
                for index in range(len(lambda_candidates()))
            ]
            run_jobs(lambda_jobs, devices, log_root / "hinf-lambda-generation")
            score_lambda_sweep(
                model_key,
                calibration_id,
                api_concurrency=api_concurrency,
                api_batch_size=api_batch_size,
            )
            select_lambda(model_key, calibration_id)
    else:
        fit_base(model_key, devices[0], calibration_id)
        release_cuda_memory(devices[0])
    synthesize_grid(model_key, devices[0], calibration_id)
    jobs = [
        (
            f"hinf-grid-{index:02d}",
            [
                sys.executable,
                "-m",
                "robust_steerability.benchmarks.mgsm_calibration",
                "--stage",
                "generate-worker",
                "--model",
                model_key,
                "--device",
                "{device}",
                "--shard-index",
                str(index),
                "--shard-count",
                str(len(devices)),
                "--calibration-id",
                calibration_id,
                *(
                    ["--generation-batch-size", str(generation_batch_size)]
                    if generation_batch_size is not None
                    else []
                ),
            ],
        )
        for index in range(len(devices))
    ]
    run_jobs(jobs, devices, log_root / "hinf-grid-generation")
    score_grid(
        model_key,
        calibration_id,
        api_concurrency=api_concurrency,
        api_batch_size=api_batch_size,
    )
    select(model_key, calibration_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "base",
            "lambda-worker",
            "score-lambda",
            "select-lambda",
            "synthesize",
            "generate-worker",
            "score-grid",
            "select",
        ),
        required=True,
    )
    parser.add_argument("--model", choices=artifacts.MODEL_KEYS, required=True)
    parser.add_argument("--device")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--calibration-id", default="selected")
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument("--api-concurrency", type=int, default=500)
    parser.add_argument("--api-batch-size", type=int, default=20)
    arguments = parser.parse_args()
    if arguments.stage == "base":
        fit_base(arguments.model, arguments.device, arguments.calibration_id)
    elif arguments.stage == "lambda-worker":
        generate_lambda_candidate(
            arguments.model,
            arguments.device,
            arguments.candidate_index,
            arguments.calibration_id,
            arguments.generation_batch_size,
        )
    elif arguments.stage == "score-lambda":
        score_lambda_sweep(
            arguments.model,
            arguments.calibration_id,
            api_concurrency=arguments.api_concurrency,
            api_batch_size=arguments.api_batch_size,
        )
    elif arguments.stage == "select-lambda":
        select_lambda(arguments.model, arguments.calibration_id)
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
        score_grid(
            arguments.model,
            arguments.calibration_id,
            api_concurrency=arguments.api_concurrency,
            api_batch_size=arguments.api_batch_size,
        )
    else:
        select(arguments.model, arguments.calibration_id)


if __name__ == "__main__":
    main()
