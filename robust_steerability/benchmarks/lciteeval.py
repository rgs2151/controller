"""Portable L-CiteEval benchmark: artifacts, calibration, generation, scoring."""

from __future__ import annotations

import argparse
import json
import sys

from robust_steerability.benchmarks import lciteeval_artifacts as artifacts
from robust_steerability.benchmarks import lciteeval_calibration as calibration
from robust_steerability.benchmarks import lciteeval_runtime as runtime
from robust_steerability.benchmarks.composition import (
    load_composition,
    requested_scorers,
    validate_requested_scorers,
)
from robust_steerability.benchmarks.execution import default_run_id, tracked_stage
from robust_steerability.benchmarks.launcher import run_data_shards
from robust_steerability.benchmarks.layout import calibration_root
from robust_steerability.benchmarks.specs import MODELS
from robust_steerability.experiments.resources import resolve_cuda_devices
from robust_steerability.judges import openai as openai_scoring
from robust_steerability.judges.specs import scorer_spec


COMPOSITION = load_composition("lciteeval_spanish")
DATASETS = COMPOSITION.dataset_keys
DEFAULT_DATASETS = COMPOSITION.default_datasets
METHODS = COMPOSITION.available_methods
DEFAULT_METHODS = COMPOSITION.default_methods


def _names(value: str, allowed: tuple[str, ...]) -> list[str]:
    names = list(allowed) if value == "all" else [item.strip() for item in value.split(",")]
    unknown = set(names) - set(allowed)
    if unknown or not names:
        raise ValueError(f"Unsupported names: {sorted(unknown)}")
    return names


def _write_selection(model_key: str, method: str, calibration_id: str) -> None:
    if method == "spid":
        parameters = {"lambda": 1.5, "kp": 0.5, "ki": 0.5, "kd": 0.01}
        source = "frozen S-PID concept-steering configuration"
    elif method == "alqr":
        parameters = {"lambda": 1.5, "q": 0.1, "r": 1.0, "q_final": 0.1}
        source = "frozen upstream A-LQR concept-steering configuration"
    else:
        return
    destination = calibration_root(
        COMPOSITION.benchmark, model_key, method, calibration_id
    ) / "selection.json"
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": [MODELS[model_key].model_id, MODELS[model_key].revision],
                "benchmark": COMPOSITION.benchmark,
                "method": method,
                "calibration_id": calibration_id,
                "source": source,
                "parameters": parameters,
            },
            indent=2,
        )
        + "\n"
    )


def artifact_stage(model_key: str, devices: list[str], log_root) -> None:
    artifacts.prepare(model_key)
    artifacts.fit_setpoint(model_key, devices[0])
    artifacts.fit_jacobians(model_key, devices, log_root=log_root / "jacobians")
    artifacts.write_manifest(model_key)


def calibration_stage(
    model_key: str,
    methods: list[str],
    devices: list[str],
    log_root,
    calibration_id: str,
    generation_batch_size: int | None,
    api_concurrency: int,
    api_batch_size: int,
    h_infinity_parameters: dict[str, float] | None,
) -> None:
    for method in methods:
        _write_selection(model_key, method, calibration_id)
    if "h_infinity" in methods:
        calibration.calibrate(
            model_key,
            devices,
            log_root,
            calibration_id,
            generation_batch_size,
            api_concurrency=api_concurrency,
            api_batch_size=api_batch_size,
            fixed_parameters=h_infinity_parameters,
        )


def evaluation_stage(
    model_key: str,
    methods: list[str],
    datasets: list[str],
    devices: list[str],
    log_root,
    calibration_id: str,
    generation_batch_size: int | None,
    use_cache: bool,
) -> None:
    artifacts.prepare(model_key)
    ordered_methods = [method for method in METHODS if method in methods]
    for method in ordered_methods:
        for condition in datasets:
            destination = runtime.generation_path(
                model_key, condition, method, use_cache=use_cache
            )
            if runtime.generation_complete(destination):
                continue
            command = [
                sys.executable,
                "-m",
                "robust_steerability.benchmarks.lciteeval_runtime",
                "--stage",
                "generate-shard",
                "--model",
                model_key,
                "--condition",
                condition,
                "--method",
                method,
                "--device",
                "{device}",
                "--calibration-id",
                calibration_id,
                "--kv-cache",
                "on" if use_cache else "off",
                *(
                    ["--generation-batch-size", str(generation_batch_size)]
                    if generation_batch_size is not None
                    else []
                ),
            ]
            run_data_shards(
                f"generate-{condition}-{method}",
                command,
                devices,
                log_root / "generation" / method / condition,
            )
            runtime.merge_shards(
                model_key,
                condition,
                method,
                len(devices),
                use_cache=use_cache,
            )


def score_stage(
    model_key: str,
    methods: list[str],
    datasets: list[str],
    scorers: tuple[str, ...] | None,
    api_concurrency: int,
    api_batch_size: int,
    use_cache: bool,
) -> None:
    generation_paths = []
    requested_by_dataset = {}
    for condition in datasets:
        selected = requested_scorers(COMPOSITION.dataset(condition), scorers)
        requested_by_dataset[condition] = selected
        for method in methods:
            generation = runtime.generation_path(
                model_key, condition, method, use_cache=use_cache
            )
            if not runtime.generation_complete(generation):
                raise ValueError(f"Missing completed generation: {generation}")
            generation_paths.append((condition, method, generation))
    for condition, _method, generation in generation_paths:
        selected = requested_by_dataset[condition]
        if "axbench_rule_spanish" in selected:
            runtime.score_spanish_adherence(model_key, generation, use_cache=use_cache)

    bilingual_scorers = sorted(
        {
            scorer
            for selected in requested_by_dataset.values()
            for scorer in selected
            if scorer_spec(scorer).backend == "openai_lcite_bilingual"
        }
    )
    if bilingual_scorers:
        from robust_steerability.judges import lciteeval_openai

        lciteeval_openai.score_generations(
            [generation for _condition, _method, generation in generation_paths],
            runtime.cache_root(model_key, use_cache),
            runtime.dataset_map(model_key),
            bilingual_scorers,
            concurrency=api_concurrency,
            batch_size=api_batch_size,
        )

    api_scorers = sorted(
        {
            scorer
            for selected in requested_by_dataset.values()
            for scorer in selected
            if scorer_spec(scorer).backend == "openai_0_2"
        }
    )
    if api_scorers:
        openai_scoring.score_generations(
            [generation for _condition, _method, generation in generation_paths],
            runtime.cache_root(model_key, use_cache),
            api_scorers,
            concurrency=api_concurrency,
            batch_size=api_batch_size,
        )
    for condition, _method, generation in generation_paths:
        if "axbench_spanish_overall" in requested_by_dataset[condition]:
            runtime.score_axbench_overall(model_key, generation, use_cache=use_cache)
    for condition, method, _generation in generation_paths:
        runtime.summarize(
            model_key,
            condition,
            method,
            requested_by_dataset[condition],
            use_cache=use_cache,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("artifacts", "calibrate", "evaluate", "score"))
    parser.add_argument("--model", choices=COMPOSITION.models, required=True)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--scorers", default="default")
    parser.add_argument("--kv-cache", choices=("off", "on"), default="off")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--calibration-id", default="selected")
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument("--h-infinity-q-over-r", type=float)
    parser.add_argument("--h-infinity-q-final-over-r", type=float)
    parser.add_argument("--h-infinity-r", type=float)
    parser.add_argument("--api-concurrency", type=int, default=500)
    parser.add_argument("--api-batch-size", type=int, default=20)
    parser.add_argument("--run-id")
    arguments = parser.parse_args()
    methods = _names(arguments.methods, METHODS)
    datasets = _names(arguments.datasets, DATASETS)
    scorers = (
        None
        if arguments.scorers in {"default", "all"}
        else tuple(item.strip() for item in arguments.scorers.split(",") if item.strip())
    )
    if scorers is not None:
        for scorer in scorers:
            scorer_spec(scorer)
    validate_requested_scorers(COMPOSITION, datasets, scorers)
    fixed_values = (
        arguments.h_infinity_q_over_r,
        arguments.h_infinity_q_final_over_r,
        arguments.h_infinity_r,
    )
    if any(value is not None for value in fixed_values) and not all(
        value is not None for value in fixed_values
    ):
        raise ValueError("Fixed H-infinity selection requires Q/R, Qf/R, and R")
    fixed = (
        {
            "q_over_r": float(fixed_values[0]),
            "q_final_over_r": float(fixed_values[1]),
            "r": float(fixed_values[2]),
        }
        if all(value is not None for value in fixed_values)
        else None
    )
    use_cache = arguments.kv_cache == "on"
    selected_score_keys = (
        {
            scorer
            for dataset in datasets
            for scorer in requested_scorers(COMPOSITION.dataset(dataset), scorers)
        }
        if arguments.stage == "score"
        else set()
    )
    gpu_required = arguments.stage != "score"
    devices = resolve_cuda_devices(arguments.devices) if gpu_required else []
    run_id = arguments.run_id or default_run_id(
        COMPOSITION.benchmark, arguments.model, arguments.stage
    )
    with tracked_stage(
        run_id=run_id,
        benchmark=COMPOSITION.benchmark,
        model=arguments.model,
        stage=arguments.stage,
        methods=methods,
        datasets=datasets,
        devices=arguments.devices if devices else "none",
        use_cache=use_cache,
        calibration_id=arguments.calibration_id,
        parameters={
            "generation_batch_size": arguments.generation_batch_size,
            "scorers": sorted(selected_score_keys),
            "api_concurrency": arguments.api_concurrency,
            "api_batch_size": arguments.api_batch_size,
            "h_infinity_fixed_parameters": fixed,
        },
    ) as log_root:
        if arguments.stage == "artifacts":
            artifact_stage(arguments.model, devices, log_root)
        elif arguments.stage == "calibrate":
            calibration_stage(
                arguments.model,
                methods,
                devices,
                log_root,
                arguments.calibration_id,
                arguments.generation_batch_size,
                arguments.api_concurrency,
                arguments.api_batch_size,
                fixed,
            )
        elif arguments.stage == "evaluate":
            evaluation_stage(
                arguments.model,
                methods,
                datasets,
                devices,
                log_root,
                arguments.calibration_id,
                arguments.generation_batch_size,
                use_cache,
            )
        else:
            score_stage(
                arguments.model,
                methods,
                datasets,
                scorers,
                arguments.api_concurrency,
                arguments.api_batch_size,
                use_cache,
            )


if __name__ == "__main__":
    main()
