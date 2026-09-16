"""Portable RealToxicityPrompts artifacts, calibration, evaluation, and scoring."""

from __future__ import annotations

import argparse
import json

from robust_steerability.benchmarks import artifacts
from robust_steerability.benchmarks import multiple_choice
from robust_steerability.benchmarks.execution import default_run_id, tracked_stage
from robust_steerability.benchmarks.composition import (
    load_composition,
    requested_scorers,
    validate_requested_scorers,
)
from robust_steerability.benchmarks.layout import calibration_root
from robust_steerability.benchmarks.specs import MODELS
from robust_steerability.benchmarks import toxicity_runtime as runtime
from robust_steerability.experiments.resources import resolve_cuda_devices
from robust_steerability.judges.specs import scorer_spec
from robust_steerability.judges import openai as openai_scoring
from robust_steerability.source_methods.protocol import (
    SPID_SOURCE_GRIDS,
    paper_alqr_setting,
)


COMPOSITION = load_composition("toxicity")
METHODS = COMPOSITION.available_methods
DEFAULT_METHODS = COMPOSITION.default_methods
DATASETS = COMPOSITION.dataset_keys
DEFAULT_DATASETS = COMPOSITION.default_datasets


def _names(value: str, allowed: tuple[str, ...]) -> list[str]:
    names = list(allowed) if value == "all" else [item.strip() for item in value.split(",")]
    unknown = set(names) - set(allowed)
    if unknown or not names:
        raise ValueError(f"Unsupported names: {sorted(unknown)}")
    return names


def _write_alqr_selection(model_key: str, calibration_id: str) -> None:
    model = MODELS[model_key]
    setting = paper_alqr_setting("toxicity", model.model_id)
    destination = calibration_root(
        "toxicity", model_key, "alqr", calibration_id
    ) / "selection.json"
    payload = {
        "schema_version": 1,
        "model": [model.model_id, model.revision],
        "benchmark": "toxicity",
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


def _write_spid_selection(model_key: str, calibration_id: str) -> None:
    model = MODELS[model_key]
    grid = SPID_SOURCE_GRIDS["toxicity"][model_key]
    destination = calibration_root(
        "toxicity", model_key, "spid", calibration_id
    ) / "selection.json"
    payload = {
        "schema_version": 1,
        "model": [model.model_id, model.revision],
        "benchmark": "toxicity",
        "method": "spid",
        "calibration_id": calibration_id,
        "source": "fixed source-method setting; S-PID is not swept",
        "kv_cache": False,
        "parameters": {
            "lambda": 1.0,
            "kp": grid.kp,
            "ki": grid.ki,
            "kd": grid.kd,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("artifacts", "calibrate", "evaluate", "score"))
    parser.add_argument("--model", choices=COMPOSITION.models, required=True)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--calibration-id", default="selected")
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument("--scorers", default="default")
    parser.add_argument(
        "--api-concurrency", type=int, default=openai_scoring.DEFAULT_CONCURRENCY
    )
    parser.add_argument(
        "--api-batch-size", type=int, default=openai_scoring.DEFAULT_BATCH_SIZE
    )
    parser.add_argument("--kv-cache", choices=("off", "on"), default="off")
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
        scorer_spec(key).backend.startswith("huggingface")
        for key in selected_score_keys
    )
    devices = resolve_cuda_devices(arguments.devices) if gpu_required else ["cpu"]
    runtime.configure_model(
        arguments.model,
        arguments.calibration_id,
        arguments.generation_batch_size,
        use_cache=use_cache,
    )
    run_id = arguments.run_id or default_run_id(
        "toxicity", arguments.model, arguments.stage
    )
    with tracked_stage(
        run_id=run_id, benchmark="toxicity", model=arguments.model,
        stage=arguments.stage, methods=methods, datasets=datasets,
        devices=arguments.devices if gpu_required else "none", use_cache=use_cache,
        calibration_id=arguments.calibration_id,
        parameters={
            "generation_batch_size": arguments.generation_batch_size,
            "scorers": sorted(selected_score_keys) if arguments.stage == "score" else [],
            "api_concurrency": arguments.api_concurrency,
            "api_batch_size": arguments.api_batch_size,
        },
    ) as log_root:
        if arguments.stage == "artifacts":
            artifacts.prepare(arguments.model, "toxicity")
            artifacts.fit_setpoint(arguments.model, "toxicity", devices[0])
            artifacts.fit_jacobians(
                arguments.model,
                "toxicity",
                devices,
                log_root=log_root / "jacobians",
            )
            artifacts.write_manifest(arguments.model, "toxicity")
        elif arguments.stage == "calibrate":
            if "alqr" in methods:
                _write_alqr_selection(arguments.model, arguments.calibration_id)
            if "spid" in methods:
                _write_spid_selection(arguments.model, arguments.calibration_id)
            if "h_infinity" in methods:
                runtime.calibrate(
                    devices,
                    log_root=log_root / "calibration",
                    api_concurrency=arguments.api_concurrency,
                    api_batch_size=arguments.api_batch_size,
                )
        elif arguments.stage == "evaluate":
            native_datasets = [
                dataset for dataset in datasets
                if COMPOSITION.dataset(dataset).runtime == "toxicity"
            ]
            multiple_choice_datasets = [
                dataset for dataset in datasets
                if COMPOSITION.dataset(dataset).runtime == "multiple_choice"
            ]
            native_pending = any(
                not multiple_choice.generation_complete(
                    runtime._generation_path(dataset, method)
                )
                for dataset in native_datasets
                for method in methods
            )
            if native_pending:
                runtime.evaluate(
                    devices,
                    methods=methods,
                    distributions=native_datasets,
                    log_root=log_root / "evaluation",
                )
            ordered_methods = [method for method in METHODS if method in methods]
            for method in ordered_methods:
                for dataset_key in multiple_choice_datasets:
                    dataset = COMPOSITION.dataset(dataset_key)
                    multiple_choice.prepare("toxicity", arguments.model, dataset)
                    destination = multiple_choice.generation_path(
                        "toxicity", arguments.model, dataset, method, use_cache=use_cache
                    )
                    if multiple_choice.generation_complete(destination):
                        continue
                    multiple_choice.launch_generation(
                        "toxicity",
                        arguments.model,
                        dataset,
                        method,
                        devices,
                        arguments.calibration_id,
                        arguments.generation_batch_size,
                        use_cache,
                        log_root / "evaluation" / method / dataset_key,
                    )
        else:
            native_datasets = [
                dataset for dataset in datasets
                if COMPOSITION.dataset(dataset).runtime == "toxicity"
            ]
            multiple_choice_datasets = [
                dataset for dataset in datasets
                if COMPOSITION.dataset(dataset).runtime == "multiple_choice"
            ]
            for dataset in native_datasets:
                selected = requested_scorers(COMPOSITION.dataset(dataset), scorers)
                if not selected:
                    continue
                runtime.score(
                    devices,
                    methods=methods,
                    distributions=[dataset],
                    scorers=list(selected),
                    log_root=log_root / "scoring",
                    api_concurrency=arguments.api_concurrency,
                    api_batch_size=arguments.api_batch_size,
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
                        "toxicity",
                        arguments.model,
                        dataset,
                        method,
                        use_cache=use_cache,
                    )


if __name__ == "__main__":
    main()
