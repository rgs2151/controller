"""Portable HarmBench artifacts, calibration, generation, and scoring."""

from __future__ import annotations

import argparse
import json
import sys

from robust_steerability.benchmarks import harmful_artifacts as artifacts
from robust_steerability.benchmarks import harmful_calibration as calibration
from robust_steerability.benchmarks import harmful_runtime as runtime
from robust_steerability.benchmarks import multiple_choice
from robust_steerability.benchmarks.composition import (
    load_composition,
    requested_scorers,
    validate_requested_scorers,
)
from robust_steerability.benchmarks.execution import default_run_id, tracked_stage
from robust_steerability.benchmarks.launcher import run_data_shards
from robust_steerability.benchmarks.layout import calibration_root
from robust_steerability.benchmarks.specs import MODELS
from robust_steerability.experiments.resources import (
    group_cuda_workers,
    primary_cuda_device,
    resolve_cuda_devices,
)
from robust_steerability.judges import openai as openai_scoring
from robust_steerability.judges.harmbench import score_generations
from robust_steerability.judges.specs import scorer_spec
from robust_steerability.modeling.huggingface import load_access_token


COMPOSITION = load_composition("harmful")
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


def _write_alqr_selection(model_key: str, calibration_id: str) -> None:
    destination = calibration_root(
        "harmful", model_key, "alqr", calibration_id
    ) / "selection.json"
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": [MODELS[model_key].model_id, MODELS[model_key].revision],
                "benchmark": "harmful",
                "method": "alqr",
                "calibration_id": calibration_id,
                "source": "frozen project concept-steering configuration",
                "parameters": {
                    "lambda": 1.5,
                    "q": 0.1,
                    "r": 1.0,
                    "q_final": 0.1,
                },
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
    classifier_batch_size: int,
    api_concurrency: int,
    api_batch_size: int,
    h_infinity_parameters: dict[str, float] | None,
) -> None:
    if "alqr" in methods:
        _write_alqr_selection(model_key, calibration_id)
    if "h_infinity" in methods:
        calibration.calibrate(
            model_key,
            devices,
            log_root,
            calibration_id,
            generation_batch_size,
            classifier_batch_size=classifier_batch_size,
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
    evaluation_behaviors: int | None,
    max_new_tokens: int,
) -> None:
    artifacts.prepare(model_key)
    native = [key for key in datasets if COMPOSITION.dataset(key).runtime == "harmful"]
    capability = [
        key for key in datasets if COMPOSITION.dataset(key).runtime == "multiple_choice"
    ]
    ordered_methods = [method for method in METHODS if method in methods]
    for method in ordered_methods:
        for condition in native:
            destination = runtime.generation_path(
                model_key, condition, method, use_cache=use_cache
            )
            if runtime.generation_complete(destination):
                continue
            command = [
                sys.executable,
                "-m",
                "robust_steerability.benchmarks.harmful_runtime",
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
                *(
                    ["--evaluation-behaviors", str(evaluation_behaviors)]
                    if evaluation_behaviors is not None else []
                ),
                "--max-new-tokens", str(max_new_tokens),
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
                behavior_count=evaluation_behaviors,
            )
    for method in ordered_methods:
        for dataset_key in capability:
            dataset = COMPOSITION.dataset(dataset_key)
            multiple_choice.prepare("harmful", model_key, dataset)
            destination = multiple_choice.generation_path(
                "harmful", model_key, dataset, method, use_cache=use_cache
            )
            if multiple_choice.generation_complete(destination):
                continue
            multiple_choice.launch_generation(
                "harmful",
                model_key,
                dataset,
                method,
                devices,
                calibration_id,
                generation_batch_size,
                use_cache,
                log_root / "generation" / method / dataset_key,
            )


def score_stage(
    model_key: str,
    methods: list[str],
    datasets: list[str],
    scorers: tuple[str, ...] | None,
    devices: list[str],
    classifier_batch_size: int,
    api_concurrency: int,
    api_batch_size: int,
    use_cache: bool,
) -> None:
    native = [key for key in datasets if COMPOSITION.dataset(key).runtime == "harmful"]
    capability = [
        key for key in datasets if COMPOSITION.dataset(key).runtime == "multiple_choice"
    ]
    selected_by_dataset = {
        condition: requested_scorers(COMPOSITION.dataset(condition), scorers)
        for condition in native
    }
    generations = [
        (
            condition,
            method,
            runtime.generation_path(model_key, condition, method, use_cache=use_cache),
        )
        for condition in native
        for method in methods
    ]
    for _condition, _method, generation in generations:
        if not runtime.generation_complete(generation):
            raise ValueError(f"Missing completed generation: {generation}")
    classifier_paths = [
        generation
        for condition, _method, generation in generations
        if "harmbench_test_success" in selected_by_dataset[condition]
    ]
    if classifier_paths:
        score_generations(
            classifier_paths,
            runtime.cache_root(model_key, use_cache),
            "harmbench_test_success",
            primary_cuda_device(devices[0]),
            load_access_token(artifacts.REPO_ROOT),
            batch_size=classifier_batch_size,
        )
    api_scorers = sorted(
        {
            scorer
            for selected in selected_by_dataset.values()
            for scorer in selected
            if scorer_spec(scorer).backend == "openai_0_2"
        }
    )
    if api_scorers:
        openai_scoring.score_generations(
            [generation for _condition, _method, generation in generations],
            runtime.cache_root(model_key, use_cache),
            api_scorers,
            concurrency=api_concurrency,
            batch_size=api_batch_size,
        )
    for condition, _method, generation in generations:
        if "axbench_overall" in selected_by_dataset[condition]:
            runtime.score_axbench_overall(model_key, generation, use_cache=use_cache)
    for condition, method, _generation in generations:
        runtime.summarize(
            model_key,
            condition,
            method,
            selected_by_dataset[condition],
            use_cache=use_cache,
        )
    for dataset_key in capability:
        dataset = COMPOSITION.dataset(dataset_key)
        selected = requested_scorers(dataset, scorers)
        if selected and selected != ("mmlu_accuracy",):
            raise ValueError(f"Unsupported scorers for {dataset_key}: {selected}")
        if not selected:
            continue
        for method in methods:
            multiple_choice.score_and_summarize(
                "harmful", model_key, dataset, method, use_cache=use_cache
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
    parser.add_argument("--evaluation-behaviors", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=runtime.MAX_NEW_TOKENS)
    parser.add_argument("--classifier-batch-size", type=int, default=8)
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
    physical_devices = resolve_cuda_devices(arguments.devices)
    devices = group_cuda_workers(
        physical_devices, MODELS[arguments.model].devices_per_worker
    )
    run_id = arguments.run_id or default_run_id("harmful", arguments.model, arguments.stage)
    with tracked_stage(
        run_id=run_id,
        benchmark="harmful",
        model=arguments.model,
        stage=arguments.stage,
        methods=methods,
        datasets=datasets,
        devices=arguments.devices,
        use_cache=use_cache,
        calibration_id=arguments.calibration_id,
        parameters={
            "generation_batch_size": arguments.generation_batch_size,
            "evaluation_behaviors": arguments.evaluation_behaviors,
            "max_new_tokens": arguments.max_new_tokens,
            "classifier_batch_size": arguments.classifier_batch_size,
            "scorers": list(scorers) if scorers is not None else "default",
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
                arguments.classifier_batch_size,
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
                arguments.evaluation_behaviors,
                arguments.max_new_tokens,
            )
        else:
            score_stage(
                arguments.model,
                methods,
                datasets,
                scorers,
                devices,
                arguments.classifier_batch_size,
                arguments.api_concurrency,
                arguments.api_batch_size,
                use_cache,
            )


if __name__ == "__main__":
    main()
