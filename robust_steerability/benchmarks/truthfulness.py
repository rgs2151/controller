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
    TRUTHFULNESS_SCORERS,
    scorer_cache_path,
    scorer_spec,
)
from robust_steerability.judges import openai as openai_scoring
from robust_steerability.source_methods.protocol import (
    SPID_SOURCE_GRIDS,
    paper_alqr_setting,
)


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
    h_infinity_parameters: dict[str, float] | None,
    spid_lambda: float | None,
    iti_top_heads: int | None,
    iti_alpha: float | None,
) -> None:
    required_source_selections = {
        "iti": iti_top_heads is not None and iti_alpha is not None,
        "spid": spid_lambda is not None,
    }
    for method, supplied in required_source_selections.items():
        selection_path = calibration_root(
            "truthfulness", model_key, method, calibration_id
        ) / "selection.json"
        if method in methods and not supplied and not selection_path.exists():
            raise FileNotFoundError(
                f"Pass an explicit fixed selection for {method}; no final Gemma "
                f"selection is preserved by the source: {selection_path}"
            )
    if "alqr" in methods:
        _write_alqr_selection(model_key, calibration_id)
    if "h_infinity" in methods:
        truthfulness_calibration.calibrate(
            model_key,
            devices,
            log_root,
            calibration_id,
            generation_batch_size,
            fixed_parameters=h_infinity_parameters,
        )
    runtime._configure_runtime(model_key, calibration_id)
    runtime.prepare("truthfulness", "id")
    jobs = []
    for method in methods:
        if method in {"original", "alqr", "h_infinity"}:
            continue
        root = calibration_root("truthfulness", model_key, method, calibration_id)
        selected_parameters = None
        if method == "spid" and spid_lambda is not None:
            gains = SPID_SOURCE_GRIDS["truthfulness"][model_key]
            selected_parameters = {
                "lambda": float(spid_lambda),
                "kp": gains.kp,
                "ki": gains.ki,
                "kd": gains.kd,
            }
        elif method == "iti" and iti_top_heads is not None and iti_alpha is not None:
            selected_parameters = {
                "top_heads": int(iti_top_heads),
                "alpha": float(iti_alpha),
            }
        elif method in {"iti", "spid"}:
            selection_path = root / "selection.json"
            if not selection_path.exists():
                raise FileNotFoundError(
                    f"Pass an explicit fixed selection for {method}; no final Gemma "
                    f"selection is preserved by the source: {selection_path}"
                )
            selected_parameters = json.loads(selection_path.read_text())["parameters"]
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
                    *(
                        ["--selected-parameters", json.dumps(selected_parameters)]
                        if selected_parameters is not None
                        else []
                    ),
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
) -> None:
    """Generate and cache model responses without running any judge."""

    runtime._configure_runtime(
        model_key, calibration_id, generation_batch_size, use_cache=use_cache
    )
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
                        "--kv-cache",
                        "on" if use_cache else "off",
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
    scorers: list[str],
    devices: list[str],
    log_root,
    calibration_id: str,
    api_concurrency: int,
    api_batch_size: int,
    use_cache: bool,
) -> None:
    """Run selected scorers against existing generations, then summarize them."""

    runtime._configure_runtime(model_key, calibration_id, use_cache=use_cache)
    generation_paths = []
    for dataset in datasets:
        for method in methods:
            files = runtime._generation_files(model_key, method, "truthfulness", dataset)
            if len(files) != 1:
                raise ValueError(
                    f"Expected one {dataset}/{method} generation cache; found {len(files)}"
                )
            generation_paths.append(files[0])

    gpu_scorers = [
        key for key in scorers if scorer_spec(key).backend == "huggingface_binary"
    ]
    api_scorers = [key for key in scorers if scorer_spec(key).backend == "openai_0_2"]
    jobs = []
    for dataset in datasets:
        for method in methods:
            for scorer in gpu_scorers:
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
    if api_scorers:
        openai_scoring.score_generations(
            generation_paths,
            runtime.CACHE_ROOT,
            api_scorers,
            concurrency=api_concurrency,
            batch_size=api_batch_size,
        )
    for dataset in datasets:
        for method in methods:
            generation_path = runtime._generation_files(
                model_key, method, "truthfulness", dataset
            )[0]
            for key in scorers:
                path = scorer_cache_path(runtime.CACHE_ROOT, generation_path, key)
                if not path.exists() or json.loads(path.read_text()).get("status") != "complete":
                    raise ValueError(f"Requested scorer did not complete: {path}")
            runtime.summarize(
                model_key,
                method,
                "truthfulness",
                dataset,
                tuple(scorers),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("artifacts", "calibrate", "evaluate", "score"))
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--scorers", default=",".join(TRUTHFULNESS_SCORERS))
    parser.add_argument("--kv-cache", choices=("off", "on"), default="off")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--calibration-id", default="selected")
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument("--h-infinity-q-over-r", type=float)
    parser.add_argument("--h-infinity-q-final-over-r", type=float)
    parser.add_argument("--h-infinity-r", type=float)
    parser.add_argument("--spid-lambda", type=float)
    parser.add_argument("--iti-top-heads", type=int)
    parser.add_argument("--iti-alpha", type=float)
    parser.add_argument(
        "--api-concurrency", type=int, default=openai_scoring.DEFAULT_CONCURRENCY
    )
    parser.add_argument(
        "--api-batch-size", type=int, default=openai_scoring.DEFAULT_BATCH_SIZE
    )
    parser.add_argument("--run-id")
    arguments = parser.parse_args()
    methods = _names(arguments.methods, METHODS)
    datasets = _names(arguments.datasets, DATASETS)
    scorers = _names(arguments.scorers, TRUTHFULNESS_SCORERS)
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
    if (arguments.iti_top_heads is None) != (arguments.iti_alpha is None):
        raise ValueError("Fixed ITI selection requires top-heads and alpha together")
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
    gpu_required = arguments.stage != "score" or any(
        scorer_spec(key).backend == "huggingface_binary" for key in scorers
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
            "generation_batch_size": arguments.generation_batch_size,
            "scorers": scorers if arguments.stage == "score" else [],
            "api_concurrency": arguments.api_concurrency,
            "api_batch_size": arguments.api_batch_size,
            "h_infinity_fixed_parameters": h_infinity_parameters,
            "spid_lambda": arguments.spid_lambda,
            "iti_top_heads": arguments.iti_top_heads,
            "iti_alpha": arguments.iti_alpha,
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
                h_infinity_parameters,
                arguments.spid_lambda,
                arguments.iti_top_heads,
                arguments.iti_alpha,
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
                devices,
                log_root,
                arguments.calibration_id,
                arguments.api_concurrency,
                arguments.api_batch_size,
                use_cache,
            )


if __name__ == "__main__":
    main()
