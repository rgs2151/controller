"""Portable TruthfulQA pipeline with cache-only AXBench quality judging."""

from __future__ import annotations

import argparse
import json
import sys

from robust_steerability.benchmarks import artifacts
from robust_steerability.benchmarks import axbench_quality
from robust_steerability.benchmarks import truthfulness_calibration
from robust_steerability.benchmarks.execution import default_run_id, tracked_stage
from robust_steerability.benchmarks.launcher import run_jobs
from robust_steerability.benchmarks.layout import calibration_root
from robust_steerability.benchmarks.specs import METHODS, MODELS
from robust_steerability.benchmarks import truthfulness_runtime as runtime
from robust_steerability.experiments.resources import resolve_cuda_devices
from robust_steerability.modeling.huggingface import load_access_token
from robust_steerability.source_methods.id_benchmark import (
    fit_source_method_calibration,
)
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
    """Materialize only explicitly selected, benchmark-owned calibrations."""

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
    runtime._configure_runtime("off", model_key, calibration_id)
    runtime.prepare("truthfulness", "id")
    data_path = runtime._data_path("truthfulness")
    for method in methods:
        if method in {"original", "alqr", "h_infinity"}:
            continue
        root = calibration_root(
            "truthfulness", model_key, method, calibration_id
        )
        selected_parameters = None
        if method in {"iti", "spid"}:
            selection_path = root / "selection.json"
            if not selection_path.exists():
                raise FileNotFoundError(
                    f"Record the preserved development-grid selection before "
                    f"calibrating {method}: {selection_path}"
                )
            selection = json.loads(selection_path.read_text())
            selected_parameters = selection["parameters"]
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
    use_cache: bool,
    log_root,
    calibration_id: str,
    generation_batch_size: int | None,
    quality_workers: int,
) -> None:
    cache_value = "on" if use_cache else "off"
    runtime._configure_runtime(
        cache_value, model_key, calibration_id, generation_batch_size
    )
    for dataset in datasets:
        runtime.prepare("truthfulness", dataset)
    source_methods = [method for method in methods if method != "h_infinity"]
    generation_jobs = []
    for dataset in datasets:
        for method in source_methods:
            generation_jobs.append((
                f"generate-{dataset}-{method}",
                [
                    sys.executable, "-m",
                    "robust_steerability.benchmarks.truthfulness_runtime",
                    "--stage", "generate", "--model", model_key,
                    "--method", method, "--behavior", "truthfulness",
                    "--distribution", dataset, "--kv-cache", cache_value,
                    "--calibration-id", calibration_id,
                    "--device", "{device}",
                    *(
                        ["--generation-batch-size", str(generation_batch_size)]
                        if generation_batch_size is not None
                        else []
                    ),
                ],
            ))
    if generation_jobs:
        run_jobs(generation_jobs, devices, log_root / "generation")
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
    score_jobs = []
    for dataset in datasets:
        for method in methods:
            if method == "h_infinity":
                for judge in ("true", "helpful"):
                    score_jobs.append((
                        f"score-{dataset}-{method}-{judge}",
                        [
                            sys.executable, "-m",
                            "robust_steerability.benchmarks.truthfulness_runtime",
                            "--stage", "score-judge", "--model", model_key,
                            "--method", method, "--behavior", "truthfulness",
                            "--distribution", dataset, "--kv-cache", cache_value,
                            "--calibration-id", calibration_id,
                            "--judge", judge, "--device", "{device}",
                        ],
                    ))
                continue
            score_jobs.append((
                f"score-{dataset}-{method}",
                [
                    sys.executable, "-m",
                    "robust_steerability.benchmarks.truthfulness_runtime",
                    "--stage", "score", "--model", model_key,
                    "--method", method, "--behavior", "truthfulness",
                    "--distribution", dataset, "--kv-cache", cache_value,
                    "--calibration-id", calibration_id,
                    "--device", "{device}",
                ],
            ))
    run_jobs(score_jobs, devices, log_root / "scoring")
    if "h_infinity" in methods:
        for dataset in datasets:
            files = runtime._generation_files(
                model_key, "h_infinity", "truthfulness", dataset
            )
            if len(files) != 1:
                raise ValueError(
                    f"Expected one H-infinity generation cache; found {len(files)}"
                )
            runtime._merge_truth_judge_shards(files[0])
    quality_stage(
        model_key,
        methods,
        datasets,
        use_cache,
        calibration_id,
        quality_workers,
    )


def quality_stage(
    model_key: str,
    methods: list[str],
    datasets: list[str],
    use_cache: bool,
    calibration_id: str,
    quality_workers: int,
) -> None:
    """Score cached generations with the AXBench relevance and fluency rubrics."""

    cache_value = "on" if use_cache else "off"
    runtime._configure_runtime(cache_value, model_key, calibration_id)
    for dataset in datasets:
        for method in methods:
            files = runtime._generation_files(
                model_key, method, "truthfulness", dataset
            )
            if len(files) != 1:
                raise ValueError(
                    f"Expected one {method} generation cache; found {len(files)}"
                )
            generation_path = files[0]
            quality_path = axbench_quality.score_generation(
                generation_path,
                runtime.CACHE_ROOT,
                workers=quality_workers,
            )
            runtime.summarize(model_key, method, "truthfulness", dataset)
            axbench_quality.add_quality_to_summary(
                model_key=model_key,
                method=method,
                distribution=dataset,
                use_cache=use_cache,
                generation_path=generation_path,
                quality_path=quality_path,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("artifacts", "calibrate", "evaluate", "quality", "all")
    )
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--kv-cache", choices=("on", "off"), default="off")
    parser.add_argument("--calibration-id", default="selected")
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument(
        "--quality-workers", type=int, default=axbench_quality.DEFAULT_WORKERS
    )
    parser.add_argument("--run-id")
    arguments = parser.parse_args()
    methods = _names(arguments.methods, METHODS)
    datasets = _names(arguments.datasets, DATASETS)
    devices = (
        []
        if arguments.stage == "quality"
        else resolve_cuda_devices(arguments.devices)
    )
    run_id = arguments.run_id or default_run_id(
        "truthfulness", arguments.model, arguments.stage
    )
    with tracked_stage(
        run_id=run_id, benchmark="truthfulness", model=arguments.model,
        stage=arguments.stage, methods=methods, datasets=datasets,
        devices=arguments.devices, use_cache=arguments.kv_cache == "on",
        calibration_id=arguments.calibration_id,
        parameters={
            "generation_batch_size": arguments.generation_batch_size,
            "axbench_quality_workers": arguments.quality_workers,
            "axbench_quality_batch_size": axbench_quality.BATCH_SIZE,
        },
    ) as log_root:
        if arguments.stage in {"artifacts", "all"}:
            artifact_stage(arguments.model, devices, log_root)
        if arguments.stage in {"calibrate", "all"}:
            calibration_stage(
                arguments.model,
                methods,
                devices,
                log_root,
                arguments.calibration_id,
                arguments.generation_batch_size,
            )
        if arguments.stage == "quality":
            quality_stage(
                arguments.model,
                methods,
                datasets,
                arguments.kv_cache == "on",
                arguments.calibration_id,
                arguments.quality_workers,
            )
        if arguments.stage in {"evaluate", "all"}:
            evaluation_stage(
                arguments.model, methods, datasets, devices,
                arguments.kv_cache == "on", log_root,
                arguments.calibration_id,
                arguments.generation_batch_size,
                arguments.quality_workers,
            )


if __name__ == "__main__":
    main()
