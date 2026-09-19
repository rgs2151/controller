"""H-infinity calibration for the AXBench-steered L-CiteEval benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from robust_steerability.artifacts import configuration_hash
from robust_steerability.benchmarks import lciteeval_artifacts as artifacts
from robust_steerability.benchmarks import lciteeval_runtime as runtime
from robust_steerability.benchmarks.calibration import require_nonzero_selection_metric
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
from robust_steerability.judges.specs import scorer_cache_path
from robust_steerability.modeling.interventions import register_generation_policy_hooks
from robust_steerability.modeling.huggingface import cuda_device_index, release_cuda_memory
from robust_steerability.source_methods.id_benchmark import runtime_provenance


BENCHMARK = artifacts.BENCHMARK
CALIBRATION_ID_DEFAULT = "selected"
Q_OVER_R = (0.01, 0.1, 1.0, 10.0)
Q_FINAL_OVER_R = (0.01, 0.1, 10**-0.5)
FIXED_R = 1.0
SETPOINT_MULTIPLIER = 1.5
GRID_SELECTION_SOURCE = "AXBench three-judge harmonic-mean calibration argmax"
FIXED_SELECTION_SOURCE = "fixed configuration supplied at calibration launch"
API_SCORERS = (
    "axbench_concept_relevance",
    "axbench_instruction_relevance",
    "axbench_fluency",
)


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _grid() -> list[dict[str, float | str]]:
    return [
        {
            "grid_id": f"q_{q_index:02d}_qf_{qf_index:02d}",
            "lambda": SETPOINT_MULTIPLIER,
            "q": float(q_ratio * FIXED_R),
            "r": FIXED_R,
            "q_final": float(qf_ratio * FIXED_R),
            "q_over_r": float(q_ratio),
            "q_final_over_r": float(qf_ratio),
        }
        for q_index, q_ratio in enumerate(Q_OVER_R)
        for qf_index, qf_ratio in enumerate(Q_FINAL_OVER_R)
    ]


def _settings(model_key: str, q: float, r: float, q_final: float) -> dict[str, object]:
    model = MODELS[model_key]
    return {
        "behavior": BENCHMARK,
        "seed": 42,
        "fit_prompts_per_class": 72,
        "disturbance_prompts": 200,
        "calibration_max_length": 1024,
        "activation_batch_size": model.activation_batch_size,
        "jacobian_prompts": artifacts.JACOBIAN_PROMPTS,
        "jacobian_max_length": artifacts.JACOBIAN_MAX_LENGTH,
        "jacobian_vjp_chunk_size": model.jacobian_vjp_chunk_size,
        "state_rank": 8,
        "numerical_floor": 1e-4,
        "alqr_setpoint_multiplier": SETPOINT_MULTIPLIER,
        "spid_setpoint_multiplier": SETPOINT_MULTIPLIER,
        "hinf_setpoint_multiplier": SETPOINT_MULTIPLIER,
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
            "direction": "AXBench concept 499",
            "tuning": "AlpacaEval frozen 50",
            "evaluation": "L-CiteEval-Length HotpotQA",
        },
    }
    _artifact, metadata = calibrate_controller(
        model,
        tokenizer,
        model_label=MODELS[model_key].label,
        model_id=MODELS[model_key].model_id,
        cache_path=_root(model_key, calibration_id) / "base/controller.pt",
        nominal_dynamics_path=artifact_root(BENCHMARK, model_key) / "dynamics.pt",
        calibration_data=calibration_data,
        settings=_settings(model_key, q, r, q_final),
        controller_device=device,
        semantic_calibration={
            "contrast": setpoint["contrast"],
            "feature_norm": setpoint["feature_norm"],
        },
    )
    return metadata


def synthesize_grid(model_key: str, device: str, calibration_id: str) -> None:
    root = _root(model_key, calibration_id)
    base_path = root / "base/controller.pt"
    bundle = torch.load(
        diagnostic_root(base_path) / "input.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    problem = FiniteHorizonControlProblem(**bundle["problem"])
    options = HInfinityOptions(**bundle["options"])
    settings = bundle["calibration"]["settings"]
    for configuration in _grid():
        destination = root / "grid/controllers" / f"{configuration['grid_id']}.pt"
        if destination.exists():
            continue
        candidate = FiniteHorizonControlProblem(
            dynamics=problem.dynamics,
            control_channels=problem.control_channels,
            disturbance_channels=problem.disturbance_channels,
            state_costs=problem.state_costs * (float(configuration["q"]) / float(settings["q"])),
            control_costs=problem.control_costs * (float(configuration["r"]) / float(settings["r"])),
            terminal_cost=problem.terminal_cost * (
                float(configuration["q_final"]) / float(settings["q_final"])
            ),
        )
        solution = HInfinityController.synthesize(candidate, device=device, options=options).solution()
        if not solution.feasible:
            raise ValueError(f"Infeasible L-CiteEval H-infinity point: {configuration}")
        _write_torch(
            destination,
            {
                "identity": {
                    "model": [MODELS[model_key].model_id, MODELS[model_key].revision],
                    "benchmark": BENCHMARK,
                    "calibration_id": calibration_id,
                    "parameters": {
                        key: float(configuration[key])
                        for key in ("lambda", "q", "r", "q_final")
                    },
                    "configuration_id": configuration["grid_id"],
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
    data = artifacts.prepare(model_key)
    model, tokenizer = artifacts.load_model(model_key, device)
    base_payload = torch.load(
        root / "base/controller.pt", map_location="cpu", weights_only=True, mmap=True
    )
    base = ControllerArtifact(**base_payload["artifact"])
    prompts = [
        runtime.format_short_instruction(tokenizer, str(row["text"]))
        for row in data["calibration"]["tuning"]
    ]
    batch_size = generation_batch_size or MODELS[model_key].activation_batch_size
    for configuration in _grid()[shard_index::shard_count]:
        grid_id = str(configuration["grid_id"])
        destination = root / "grid/generations" / f"{grid_id}.json"
        if runtime.generation_complete(destination):
            continue
        controller = torch.load(
            root / "grid/controllers" / f"{grid_id}.pt",
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
        started_at = _utc_now()
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(cuda_device_index(device))
        completions, generated = runtime.generate_completions(
            model,
            tokenizer,
            prompts,
            policy=policy,
            use_cache=False,
            batch_size=batch_size,
            max_new_tokens=128,
        )
        rows = [
            {
                "prompt_id": row["prompt_id"],
                "grid_id": grid_id,
                "text": row["text"],
                "concept": artifacts.AXBENCH_CONCEPT,
                "completion": completion,
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
                "attempts": [
                    {
                        "status": "complete",
                        "started_at_utc": started_at,
                        "finished_at_utc": _utc_now(),
                        "elapsed_seconds": time.perf_counter() - started,
                        "runtime": runtime_provenance(device),
                        "gpu_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(
                            cuda_device_index(device)
                        ),
                        "gpu_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(
                            cuda_device_index(device)
                        ),
                    }
                ],
                "repetitions": [{"repetition": 0, "sample_count": 50, "rows": rows}],
            },
        )


def score_grid(
    model_key: str,
    calibration_id: str,
    *,
    api_concurrency: int,
    api_batch_size: int,
) -> None:
    grid_root = _root(model_key, calibration_id) / "grid"
    generations = sorted((grid_root / "generations").glob("*.json"))
    if len(generations) != len(_grid()):
        raise ValueError("L-CiteEval H-infinity grid generations are incomplete")
    openai_scoring.score_generations(
        generations,
        grid_root,
        list(API_SCORERS),
        concurrency=api_concurrency,
        batch_size=api_batch_size,
    )
    for generation in generations:
        maps = []
        for scorer in API_SCORERS:
            payload = json.loads(scorer_cache_path(grid_root, generation, scorer).read_text())
            maps.append({row["prompt_id"]: float(row["score"]) for row in payload["rows"]})
        rows = [
            {
                "prompt_id": prompt_id,
                "score": harmonic_mean([mapping[prompt_id] for mapping in maps]),
            }
            for prompt_id in maps[0]
        ]
        _write_json(
            scorer_cache_path(grid_root, generation, "axbench_overall"),
            {"status": "complete", "rows": rows},
        )


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
    problem["state_costs"] = (parameters["q"] * identity).unsqueeze(0).repeat(horizon, 1, 1)
    problem["control_costs"] = (parameters["r"] * identity).unsqueeze(0).repeat(horizon, 1, 1)
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
        {
            "run_id": "calibration-" + fingerprint[:20],
            "calibration_fingerprint": fingerprint,
            "protocol_id": "full-reduced-state-"
            + configuration_hash(
                {
                    key: value
                    for key, value in bundle["calibration"]["settings"].items()
                    if key != "model_loading"
                }
            )[:16],
        }
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
        destination,
        "cpu",
        cache_root=destination.parent,
        solution=solution,
    )


def select(model_key: str, calibration_id: str) -> dict:
    root = _root(model_key, calibration_id)
    summaries = []
    for configuration in _grid():
        generation = root / "grid/generations" / f"{configuration['grid_id']}.json"
        means = {}
        for scorer in (*API_SCORERS, "axbench_overall"):
            payload = json.loads(
                scorer_cache_path(root / "grid", generation, scorer).read_text()
            )
            means[scorer] = float(np.mean([float(row["score"]) for row in payload["rows"]]))
        summaries.append({**configuration, **means})
    require_nonzero_selection_metric(
        summaries,
        "axbench_overall",
        context="L-CiteEval H-infinity calibration",
    )
    selected = sorted(
        summaries,
        key=lambda row: (
            -row["axbench_overall"],
            -row["axbench_instruction_relevance"],
            -row["axbench_fluency"],
            row["q"],
            row["q_final"],
        ),
    )[0]
    parameters = {
        key: float(selected[key]) for key in ("lambda", "q", "r", "q_final")
    }
    controller = torch.load(
        root / "grid/controllers" / f"{selected['grid_id']}.pt",
        map_location="cpu",
        weights_only=True,
    )
    _write_torch(root / "controller.pt", controller)
    diagnostic = _freeze_selected_diagnostics(
        model_key, calibration_id, parameters, controller
    )
    payload = {
        "schema_version": 1,
        "model": [MODELS[model_key].model_id, MODELS[model_key].revision],
        "benchmark": BENCHMARK,
        "calibration_id": calibration_id,
        "protocol": {
            "selection_strategy": "grid",
            "selection_metric": "mean per-response AXBench three-judge harmonic mean",
            "tuning_samples": 50,
            "tuning_repetitions": 1,
            "evaluated_model_kv_cache": False,
            "q_over_r": list(Q_OVER_R),
            "q_final_over_r": list(Q_FINAL_OVER_R),
            "fixed_r": FIXED_R,
            "fixed_setpoint_multiplier": SETPOINT_MULTIPLIER,
        },
        "selected": {
            **selected,
            "configuration_id": selected["grid_id"],
            "parameters": parameters,
            "gamma_star": float(controller["gamma_star"]),
            "source": GRID_SELECTION_SOURCE,
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
    metadata = fit_base(
        model_key, device, calibration_id, q=q, r=r, q_final=q_final
    )
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
    diagnostic = (
        diagnostic_root(_root(model_key, calibration_id) / "base/controller.pt")
        / "runs"
        / ("calibration-" + str(metadata["fingerprint"])[:20])
    )
    if not diagnostic.exists():
        raise FileNotFoundError(f"Missing Hannah diagnostic bundle: {diagnostic}")
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
            "source": FIXED_SELECTION_SOURCE,
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
    if (root / "selection.json").exists() and (root / "controller.pt").exists():
        saved = json.loads((root / "selection.json").read_text())
        diagnostic = root / saved["diagnostic_bundle"]
        if not diagnostic.exists():
            raise FileNotFoundError(
                "Selected H-infinity calibration is missing Hannah's diagnostic "
                f"bundle: {diagnostic}"
            )
        if fixed_parameters is not None:
            parameters = saved["selected"]["parameters"]
            actual = {
                "q_over_r": float(parameters["q"] / parameters["r"]),
                "q_final_over_r": float(parameters["q_final"] / parameters["r"]),
                "r": float(parameters["r"]),
            }
            expected = {key: float(value) for key, value in fixed_parameters.items()}
            if actual != expected:
                raise ValueError(
                    f"Calibration ID {calibration_id!r} contains {actual}, not {expected}"
                )
        return
    if fixed_parameters is not None:
        select_fixed(model_key, devices[0], calibration_id, **fixed_parameters)
        return
    fit_base(model_key, devices[0], calibration_id)
    release_cuda_memory(devices[0])
    synthesize_grid(model_key, devices[0], calibration_id)
    jobs = [
        (
            f"hinf-grid-{index:02d}",
            [
                sys.executable,
                "-m",
                "robust_steerability.benchmarks.lciteeval_calibration",
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
        choices=("base", "synthesize", "generate-worker", "score-grid", "select"),
        required=True,
    )
    parser.add_argument(
        "--model",
        choices=("qwen25_3b_instruct", "llama31_8b_instruct"),
        required=True,
    )
    parser.add_argument("--device")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--calibration-id", default=CALIBRATION_ID_DEFAULT)
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument("--api-concurrency", type=int, default=500)
    parser.add_argument("--api-batch-size", type=int, default=20)
    arguments = parser.parse_args()
    if arguments.stage == "base":
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
