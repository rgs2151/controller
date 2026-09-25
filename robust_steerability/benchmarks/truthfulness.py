"""Portable TruthfulQA artifacts, calibration, generation, and scoring."""

from __future__ import annotations

import argparse
import json
import sys

from robust_steerability.benchmarks import artifacts
from robust_steerability.benchmarks import truthfulness_calibration
from robust_steerability.benchmarks import multiple_choice
from robust_steerability.benchmarks import truthfulness_runtime as runtime
from robust_steerability.benchmarks.composition import (
    load_composition,
    requested_scorers,
    validate_requested_scorers,
)
from robust_steerability.benchmarks.execution import default_run_id, tracked_stage
from robust_steerability.benchmarks.launcher import run_data_shards, run_jobs
from robust_steerability.benchmarks.layout import calibration_root
from robust_steerability.benchmarks.methods import method_spec
from robust_steerability.benchmarks.specs import MODELS
from robust_steerability.experiments.resources import resolve_cuda_devices
from robust_steerability.judges import scorer_cache_path, scorer_spec
from robust_steerability.judges import openai as openai_scoring
from robust_steerability.judges.exact import harmonic_mean
from robust_steerability.source_methods.protocol import (
    alqr_setting_source,
    paper_alqr_setting,
)


COMPOSITION = load_composition("truthfulness")
DATASETS = COMPOSITION.dataset_keys
METHODS = COMPOSITION.available_methods
DEFAULT_METHODS = COMPOSITION.default_methods


def _names(value: str, allowed: tuple[str, ...]) -> list[str]:
    names = list(allowed) if value == "all" else [item.strip() for item in value.split(",")]
    unknown = set(names) - set(allowed)
    if unknown or not names:
        raise ValueError(f"Unsupported names: {sorted(unknown)}")
    return names


def _write_axbench_overall(generation_path) -> None:
    mappings = []
    for scorer in (
        "axbench_concept_relevance",
        "axbench_instruction_relevance",
        "axbench_fluency",
    ):
        payload = json.loads(
            scorer_cache_path(runtime.CACHE_ROOT, generation_path, scorer).read_text()
        )
        mappings.append(
            {str(row["prompt_id"]): float(row["score"]) for row in payload["rows"]}
        )
    rows = [
        {
            "prompt_id": prompt_id,
            "score": harmonic_mean([mapping[prompt_id] for mapping in mappings]),
        }
        for prompt_id in mappings[0]
    ]
    destination = scorer_cache_path(
        runtime.CACHE_ROOT, generation_path, "axbench_overall"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps({"status": "complete", "rows": rows}, indent=2) + "\n")
    temporary.replace(destination)


def _write_alqr_selection(model_key: str, calibration_id: str) -> None:
    model = MODELS[model_key]
    setting = paper_alqr_setting("truthfulness", model.model_id)
    destination = calibration_root(
        "truthfulness", model_key, "alqr", calibration_id
    ) / "selection.json"
    payload = {
        "schema_version": 1,
        "model": [model.model_id, model.revision],
        "behavior": "truthfulness",
        "method": "alqr",
        "calibration_id": calibration_id,
        "source": alqr_setting_source("truthfulness", model.model_id),
        "parameters": {
            "lambda": setting.multiplier,
            "q": setting.q,
            "r": setting.r,
            "q_final": setting.q_final,
        },
    }
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")


def artifact_stage(model_key: str, devices: list[str], log_root) -> None:
    artifacts.prepare(model_key, "truthfulness")
    artifacts.fit_setpoint(model_key, "truthfulness", devices[0])
    artifacts.fit_jacobians(
        model_key, "truthfulness", devices, log_root=log_root / "jacobians"
    )
    artifacts.write_manifest(model_key, "truthfulness")


def calibration_stage(
    model_key: str,
    methods: list[str],
    devices: list[str],
    log_root,
    calibration_id: str,
    generation_batch_size: int | None,
    api_concurrency: int,
    api_batch_size: int,
    selection_metric: str,
    h_infinity_parameters: dict[str, float] | None,
    h_infinity_lambda_sweep: bool,
) -> None:
    if "alqr" in methods:
        _write_alqr_selection(model_key, calibration_id)
    if "h_infinity" in methods:
        truthfulness_calibration.calibrate(
            model_key,
            devices,
            log_root,
            calibration_id,
            generation_batch_size,
            api_concurrency=api_concurrency,
            api_batch_size=api_batch_size,
            selection_metric=selection_metric,
            fixed_parameters=h_infinity_parameters,
            lambda_sweep=h_infinity_lambda_sweep,
        )
    runtime._configure_runtime(model_key, calibration_id)
    runtime.prepare("truthfulness", "id")
    jobs = []
    for method in methods:
        if not method_spec(method).requires_source_fit:
            continue
        jobs.append(
            (
                f"fit-{method}",
                [
                    sys.executable,
                    "-m",
                    "robust_steerability.benchmarks.truthfulness_runtime",
                    "--stage",
                    "fit-source",
                    "--model",
                    model_key,
                    "--method",
                    method,
                    "--behavior",
                    "truthfulness",
                    "--distribution",
                    "id",
                    "--calibration-id",
                    calibration_id,
                    "--device",
                    "{device}",
                ],
            )
        )
    if jobs:
        run_jobs(jobs, devices, log_root / "source-method-fits")


def evaluation_stage(
    model_key: str,
    methods: list[str],
    datasets: list[str],
    devices: list[str],
    log_root,
    calibration_id: str,
    generation_batch_size: int | None,
    use_cache: bool,
    evaluation_samples: int | None,
    evaluation_repetitions: int | None,
) -> None:
    """Generate and cache model responses without running any judge."""

    runtime._configure_runtime(
        model_key,
        calibration_id,
        generation_batch_size,
        use_cache=use_cache,
        evaluation_samples=evaluation_samples,
        evaluation_repetitions=evaluation_repetitions,
    )
    native_datasets = [
        dataset for dataset in datasets
        if COMPOSITION.dataset(dataset).runtime == "truthfulness"
    ]
    multiple_choice_datasets = [
        dataset for dataset in datasets
        if COMPOSITION.dataset(dataset).runtime == "multiple_choice"
    ]
    for dataset in native_datasets:
        runtime.prepare("truthfulness", dataset)
    ordered_methods = [method for method in METHODS if method in methods]
    for method in ordered_methods:
        for dataset in native_datasets:
            existing = runtime._generation_files(
                model_key, method, "truthfulness", dataset
            )
            if len(existing) == 1 and multiple_choice.generation_complete(existing[0]):
                continue
            if method_spec(method).runtime == "h_infinity":
                runtime.launch_hinf_generation(
                    model_key,
                    "truthfulness",
                    dataset,
                    devices,
                    log_root / "generation" / method / dataset,
                )
                continue
            command = [
                sys.executable,
                "-m",
                "robust_steerability.benchmarks.truthfulness_runtime",
                "--stage", "generate",
                "--model", model_key,
                "--method", method,
                "--behavior", "truthfulness",
                "--distribution", dataset,
                "--calibration-id", calibration_id,
                "--kv-cache", "on" if use_cache else "off",
                "--device", "{device}",
                *(
                    ["--generation-batch-size", str(generation_batch_size)]
                    if generation_batch_size is not None
                    else []
                ),
                *(
                    ["--evaluation-samples", str(evaluation_samples)]
                    if evaluation_samples is not None else []
                ),
                *(
                    ["--evaluation-repetitions", str(evaluation_repetitions)]
                    if evaluation_repetitions is not None else []
                ),
            ]
            run_data_shards(
                f"generate-{dataset}-{method}",
                command,
                devices,
                log_root / "generation" / method / dataset,
            )
            runtime.merge_source_generation(
                model_key, method, "truthfulness", dataset, len(devices)
            )

    for method in ordered_methods:
        for dataset_key in multiple_choice_datasets:
            dataset = COMPOSITION.dataset(dataset_key)
            multiple_choice.prepare("truthfulness", model_key, dataset)
            destination = multiple_choice.generation_path(
                "truthfulness", model_key, dataset, method, use_cache=use_cache
            )
            if multiple_choice.generation_complete(destination):
                continue
            multiple_choice.launch_generation(
                "truthfulness",
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
    log_root,
    calibration_id: str,
    api_concurrency: int,
    api_batch_size: int,
    use_cache: bool,
    evaluation_samples: int | None,
    evaluation_repetitions: int | None,
) -> None:
    """Run selected scorers against existing generations, then summarize them."""

    runtime._configure_runtime(
        model_key,
        calibration_id,
        use_cache=use_cache,
        evaluation_samples=evaluation_samples,
        evaluation_repetitions=evaluation_repetitions,
    )
    native_datasets = [
        dataset for dataset in datasets
        if COMPOSITION.dataset(dataset).runtime == "truthfulness"
    ]
    multiple_choice_datasets = [
        dataset for dataset in datasets
        if COMPOSITION.dataset(dataset).runtime == "multiple_choice"
    ]
    generation_paths = {}
    native_scorers = {}
    for dataset in native_datasets:
        selected = requested_scorers(COMPOSITION.dataset(dataset), scorers)
        native_scorers[dataset] = selected
        if not selected:
            continue
        generation_paths[dataset] = []
        for method in methods:
            files = runtime._generation_files(model_key, method, "truthfulness", dataset)
            if len(files) != 1:
                raise ValueError(
                    f"Expected one {dataset}/{method} generation cache; found {len(files)}"
                )
            generation_paths[dataset].append(files[0])

    jobs = []
    for dataset in native_datasets:
        if not native_scorers[dataset]:
            continue
        for method in methods:
            for scorer in native_scorers[dataset]:
                if scorer_spec(scorer).backend != "huggingface_binary":
                    continue
                jobs.append(
                    (
                        f"score-{dataset}-{method}-{scorer}",
                        [
                            sys.executable,
                            "-m",
                            "robust_steerability.benchmarks.truthfulness_runtime",
                            "--stage",
                            "score-scorer",
                            "--model",
                            model_key,
                            "--method",
                            method,
                            "--behavior",
                            "truthfulness",
                            "--distribution",
                            dataset,
                            "--calibration-id",
                            calibration_id,
                            "--kv-cache",
                            "on" if use_cache else "off",
                            "--scorer",
                            scorer,
                            "--device",
                            "{device}",
                        ],
                    )
                )
    if jobs:
        run_jobs(jobs, devices, log_root / "scoring")
    for dataset in native_datasets:
        api_scorers = [
            key for key in native_scorers[dataset]
            if scorer_spec(key).backend == "openai_0_2"
        ]
        if "axbench_overall" in native_scorers[dataset]:
            api_scorers = list(dict.fromkeys([
                *api_scorers,
                "axbench_concept_relevance",
                "axbench_instruction_relevance",
                "axbench_fluency",
            ]))
        if api_scorers:
            openai_scoring.score_generations(
                generation_paths[dataset],
                runtime.CACHE_ROOT,
                api_scorers,
                concurrency=api_concurrency,
                batch_size=api_batch_size,
                row_defaults={"concept": runtime.TRUTHFULNESS_CONCEPT},
            )
        if "axbench_overall" in native_scorers[dataset]:
            for generation_path in generation_paths[dataset]:
                _write_axbench_overall(generation_path)
    for dataset in native_datasets:
        if not native_scorers[dataset]:
            continue
        for method in methods:
            generation_path = runtime._generation_files(
                model_key, method, "truthfulness", dataset
            )[0]
            for key in native_scorers[dataset]:
                path = scorer_cache_path(runtime.CACHE_ROOT, generation_path, key)
                if not path.exists() or json.loads(path.read_text()).get("status") != "complete":
                    raise ValueError(f"Requested scorer did not complete: {path}")
            runtime.summarize(
                model_key,
                method,
                "truthfulness",
                dataset,
                native_scorers[dataset],
            )
    for dataset_key in multiple_choice_datasets:
        dataset = COMPOSITION.dataset(dataset_key)
        selected = requested_scorers(dataset, scorers)
        if not selected:
            continue
        if selected != ("mmlu_accuracy",):
            raise ValueError(f"Unsupported scorers for {dataset_key}: {selected}")
        for method in methods:
            multiple_choice.score_and_summarize(
                "truthfulness", model_key, dataset, method, use_cache=use_cache
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("artifacts", "calibrate", "evaluate", "score"))
    parser.add_argument("--model", choices=COMPOSITION.models, required=True)
    parser.add_argument(
        "--run-profile",
        choices=("default", *(profile.name for profile in COMPOSITION.run_profiles)),
        default="default",
    )
    parser.add_argument("--methods")
    parser.add_argument("--datasets")
    parser.add_argument("--scorers", default="default")
    parser.add_argument("--kv-cache", choices=("off", "on"), default="off")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--calibration-id", default="selected")
    parser.add_argument(
        "--selection-metric",
        choices=COMPOSITION.calibration.available_selection_metrics,
        default=COMPOSITION.calibration.selection_metric,
    )
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument("--evaluation-samples", type=int)
    parser.add_argument("--evaluation-repetitions", type=int)
    parser.add_argument("--h-infinity-q-over-r", type=float)
    parser.add_argument("--h-infinity-q-final-over-r", type=float)
    parser.add_argument("--h-infinity-r", type=float)
    parser.add_argument(
        "--h-infinity-lambda-sweep",
        action="store_true",
        help=(
            "Run the configured H-infinity setpoint multiplier sweep before "
            "the Q/R and Qf/R calibration grid"
        ),
    )
    parser.add_argument(
        "--api-concurrency", type=int, default=openai_scoring.DEFAULT_CONCURRENCY
    )
    parser.add_argument(
        "--api-batch-size", type=int, default=openai_scoring.DEFAULT_BATCH_SIZE
    )
    parser.add_argument("--run-id")
    arguments = parser.parse_args()
    profile = (
        None
        if arguments.run_profile == "default"
        else COMPOSITION.run_profile(arguments.run_profile)
    )
    if profile is not None and arguments.model not in profile.models:
        raise ValueError(
            f"Run profile {profile.name!r} does not include model {arguments.model!r}"
        )
    methods_value = arguments.methods or ",".join(
        profile.methods if profile is not None else DEFAULT_METHODS
    )
    datasets_value = arguments.datasets or ",".join(
        profile.datasets if profile is not None else DATASETS
    )
    if arguments.evaluation_repetitions is None and profile is not None:
        arguments.evaluation_repetitions = profile.evaluation_repetitions
    methods = _names(methods_value, METHODS)
    datasets = _names(datasets_value, DATASETS)
    scorers = (
        None
        if arguments.scorers in {"default", "all"}
        else tuple(item.strip() for item in arguments.scorers.split(",") if item.strip())
    )
    if scorers is not None:
        for scorer in scorers:
            scorer_spec(scorer)
    validate_requested_scorers(COMPOSITION, datasets, scorers)
    h_infinity_values = (
        arguments.h_infinity_q_over_r,
        arguments.h_infinity_q_final_over_r,
        arguments.h_infinity_r,
    )
    if any(value is not None for value in h_infinity_values) and not all(
        value is not None for value in h_infinity_values
    ):
        raise ValueError(
            "Fixed H-infinity selection requires Q/R, Qf/R, and R together"
        )
    h_infinity_parameters = (
        {
            "q_over_r": float(arguments.h_infinity_q_over_r),
            "q_final_over_r": float(arguments.h_infinity_q_final_over_r),
            "r": float(arguments.h_infinity_r),
        }
        if all(value is not None for value in h_infinity_values)
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
    gpu_required = arguments.stage != "score" or any(
        scorer_spec(key).backend == "huggingface_binary" for key in selected_score_keys
    )
    devices = resolve_cuda_devices(arguments.devices) if gpu_required else []
    run_id = arguments.run_id or default_run_id(
        "truthfulness", arguments.model, arguments.stage
    )
    with tracked_stage(
        run_id=run_id,
        benchmark="truthfulness",
        model=arguments.model,
        stage=arguments.stage,
        methods=methods,
        datasets=datasets,
        devices=arguments.devices if devices else "none",
        use_cache=use_cache,
        calibration_id=arguments.calibration_id,
        parameters={
            "run_profile": arguments.run_profile,
            "generation_batch_size": arguments.generation_batch_size,
            "evaluation_samples": arguments.evaluation_samples,
            "evaluation_repetitions": arguments.evaluation_repetitions,
            "scorers": sorted(selected_score_keys) if arguments.stage == "score" else [],
            "api_concurrency": arguments.api_concurrency,
            "api_batch_size": arguments.api_batch_size,
            "selection_metric": arguments.selection_metric,
            "h_infinity_fixed_parameters": h_infinity_parameters,
            "h_infinity_lambda_sweep": arguments.h_infinity_lambda_sweep,
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
                arguments.selection_metric,
                h_infinity_parameters,
                arguments.h_infinity_lambda_sweep,
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
                arguments.evaluation_samples,
                arguments.evaluation_repetitions,
            )
        else:
            score_stage(
                arguments.model,
                methods,
                datasets,
                scorers,
                devices,
                log_root,
                arguments.calibration_id,
                arguments.api_concurrency,
                arguments.api_batch_size,
                use_cache,
                arguments.evaluation_samples,
                arguments.evaluation_repetitions,
            )


if __name__ == "__main__":
    main()
