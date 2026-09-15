"""Portable TruthfulQA artifacts, calibration, generation, and scoring."""

from __future__ import annotations

import argparse
import json
import sys

from robust_steerability.benchmarks import artifacts
from robust_steerability.benchmarks import truthfulness_calibration
from robust_steerability.benchmarks import truthfulness_runtime as runtime
from robust_steerability.benchmarks.execution import default_run_id, tracked_stage
from robust_steerability.benchmarks.launcher import run_jobs
from robust_steerability.benchmarks.layout import calibration_root
from robust_steerability.benchmarks.specs import METHODS, MODELS
from robust_steerability.experiments.resources import resolve_cuda_devices
from robust_steerability.judges import (
    TRUTHFULNESS_JUDGES,
    judge_cache_path,
    judge_spec,
)
from robust_steerability.judges import openai as openai_judges
from robust_steerability.modeling.huggingface import load_access_token
from robust_steerability.source_methods.id_benchmark import fit_source_method_calibration
from robust_steerability.source_methods.protocol import paper_alqr_setting


DATASETS = ("id", "spanish")
DEFAULT_METHODS = ("original", "alqr", "h_infinity")


def _names(value: str, allowed: tuple[str, ...]) -> list[str]:
    names = list(allowed) if value == "all" else [item.strip() for item in value.split(",")]
    unknown = set(names) - set(allowed)
    if unknown or not names:
        raise ValueError(f"Unsupported names: {sorted(unknown)}")
    return names


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
        "source": "published A-LQR configuration",
        "parameters": {
            "lambda": setting.multiplier,
            "q": setting.q,
            "r": setting.r,
            "q_final": setting.q_final,
        },
    }
    if destination.exists() and json.loads(destination.read_text()) != payload:
        raise ValueError(f"A-LQR selection changed: {destination}")
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
        )
    model = MODELS[model_key]
    runtime._configure_runtime(model_key, calibration_id)
    runtime.prepare("truthfulness", "id")
    data_path = runtime._data_path("truthfulness")
    for method in methods:
        if method in {"original", "alqr", "h_infinity"}:
            continue
        root = calibration_root("truthfulness", model_key, method, calibration_id)
        selected_parameters = None
        if method in {"iti", "spid"}:
            selection_path = root / "selection.json"
            if not selection_path.exists():
                raise FileNotFoundError(
                    f"Record the development-grid selection before calibrating "
                    f"{method}: {selection_path}"
                )
            selected_parameters = json.loads(selection_path.read_text())["parameters"]
        fit_source_method_calibration(
            behavior="truthfulness",
            model_id=model.model_id,
            revision=model.revision,
            method=method,
            device=devices[0],
            token=load_access_token(runtime.REPO),
            calibration_root=root,
            calibration_data_path=data_path,
            selected_parameters=selected_parameters,
        )


def evaluation_stage(
    model_key: str,
    methods: list[str],
    datasets: list[str],
    devices: list[str],
    log_root,
    calibration_id: str,
    generation_batch_size: int | None,
) -> None:
    """Generate and cache model responses without running any judge."""

    runtime._configure_runtime(model_key, calibration_id, generation_batch_size)
    for dataset in datasets:
        runtime.prepare("truthfulness", dataset)
    source_methods = [method for method in methods if method != "h_infinity"]
    jobs = []
    for dataset in datasets:
        for method in source_methods:
            jobs.append(
                (
                    f"generate-{dataset}-{method}",
                    [
                        sys.executable,
                        "-m",
                        "robust_steerability.benchmarks.truthfulness_runtime",
                        "--stage",
                        "generate",
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
                        "--device",
                        "{device}",
                        *(
                            ["--generation-batch-size", str(generation_batch_size)]
                            if generation_batch_size is not None
                            else []
                        ),
                    ],
                )
            )
    if jobs:
        run_jobs(jobs, devices, log_root / "generation")
    if "h_infinity" in methods:
        hinf_devices = devices[: runtime.EVALUATION_REPETITIONS]
        for dataset in datasets:
            runtime.launch_hinf_generation(
                model_key,
                "truthfulness",
                dataset,
                hinf_devices,
                log_root / "generation",
            )


def score_stage(
    model_key: str,
    methods: list[str],
    datasets: list[str],
    judges: list[str],
    devices: list[str],
    log_root,
    calibration_id: str,
    api_concurrency: int,
    api_batch_size: int,
) -> None:
    """Run selected judges against existing generations, then summarize them."""

    runtime._configure_runtime(model_key, calibration_id)
    generation_paths = []
    for dataset in datasets:
        for method in methods:
            files = runtime._generation_files(model_key, method, "truthfulness", dataset)
            if len(files) != 1:
                raise ValueError(
                    f"Expected one {dataset}/{method} generation cache; found {len(files)}"
                )
            generation_paths.append(files[0])

    gpu_judges = [key for key in judges if judge_spec(key).backend == "huggingface_binary"]
    api_judges = [key for key in judges if judge_spec(key).backend == "openai_0_2"]
    jobs = []
    for dataset in datasets:
        for method in methods:
            for judge in gpu_judges:
                jobs.append(
                    (
                        f"score-{dataset}-{method}-{judge}",
                        [
                            sys.executable,
                            "-m",
                            "robust_steerability.benchmarks.truthfulness_runtime",
                            "--stage",
                            "score-judge",
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
                            "--judge",
                            judge,
                            "--device",
                            "{device}",
                        ],
                    )
                )
    if jobs:
        run_jobs(jobs, devices, log_root / "judges")
    if api_judges:
        openai_judges.score_generations(
            generation_paths,
            runtime.CACHE_ROOT,
            api_judges,
            concurrency=api_concurrency,
            batch_size=api_batch_size,
        )
    for dataset in datasets:
        for method in methods:
            generation_path = runtime._generation_files(
                model_key, method, "truthfulness", dataset
            )[0]
            available = tuple(
                key
                for key in TRUTHFULNESS_JUDGES
                if judge_cache_path(runtime.CACHE_ROOT, generation_path, key).exists()
                and json.loads(
                    judge_cache_path(runtime.CACHE_ROOT, generation_path, key).read_text()
                ).get("status") == "complete"
            )
            missing = set(judges) - set(available)
            if missing:
                raise ValueError(f"Requested judges did not complete: {sorted(missing)}")
            runtime.summarize(
                model_key,
                method,
                "truthfulness",
                dataset,
                available,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("artifacts", "calibrate", "evaluate", "score"))
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--judges", default=",".join(TRUTHFULNESS_JUDGES))
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--calibration-id", default="selected")
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument(
        "--api-concurrency", type=int, default=openai_judges.DEFAULT_CONCURRENCY
    )
    parser.add_argument(
        "--api-batch-size", type=int, default=openai_judges.DEFAULT_BATCH_SIZE
    )
    parser.add_argument("--run-id")
    arguments = parser.parse_args()
    methods = _names(arguments.methods, METHODS)
    datasets = _names(arguments.datasets, DATASETS)
    judges = _names(arguments.judges, TRUTHFULNESS_JUDGES)
    gpu_required = arguments.stage != "score" or any(
        judge_spec(key).backend == "huggingface_binary" for key in judges
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
        use_cache=False,
        calibration_id=arguments.calibration_id,
        parameters={
            "generation_batch_size": arguments.generation_batch_size,
            "judges": judges if arguments.stage == "score" else [],
            "api_concurrency": arguments.api_concurrency,
            "api_batch_size": arguments.api_batch_size,
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
            )
        else:
            score_stage(
                arguments.model,
                methods,
                datasets,
                judges,
                devices,
                log_root,
                arguments.calibration_id,
                arguments.api_concurrency,
                arguments.api_batch_size,
            )


if __name__ == "__main__":
    main()
