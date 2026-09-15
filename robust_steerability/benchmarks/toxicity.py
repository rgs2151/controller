"""Portable RTP-to-Jigsaw artifacts, calibration, evaluation, and scoring."""

from __future__ import annotations

import argparse
import json

from robust_steerability.benchmarks import artifacts
from robust_steerability.benchmarks.execution import default_run_id, tracked_stage
from robust_steerability.benchmarks.layout import calibration_root
from robust_steerability.benchmarks.specs import MODELS
from robust_steerability.benchmarks import toxicity_runtime as runtime
from robust_steerability.experiments.resources import resolve_cuda_devices
from robust_steerability.source_methods.protocol import paper_alqr_setting


METHODS = ("original", "spid", "alqr", "h_infinity")
DATASETS = ("rtp", "jigsaw")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("artifacts", "calibrate", "evaluate", "score"))
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--methods", default="all")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--calibration-id", default="selected")
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument("--scorers", default="all")
    parser.add_argument("--kv-cache", choices=("off", "on"), default="off")
    parser.add_argument("--run-id")
    arguments = parser.parse_args()
    methods = _names(arguments.methods, METHODS)
    datasets = _names(arguments.datasets, DATASETS)
    scorers = _names(arguments.scorers, runtime.SCORERS)
    use_cache = arguments.kv_cache == "on"
    gpu_required = arguments.stage != "score" or bool(
        {"toxicity_classifier", "perplexity"} & set(scorers)
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
            "scorers": scorers if arguments.stage == "score" else [],
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
            if {"spid", "h_infinity"} & set(methods):
                runtime.calibrate(devices, log_root=log_root / "calibration")
        elif arguments.stage == "evaluate":
            runtime.evaluate(
                devices,
                methods=methods,
                distributions=datasets,
                log_root=log_root / "evaluation",
            )
        else:
            runtime.score(
                devices,
                methods=methods,
                distributions=datasets,
                scorers=scorers,
                log_root=log_root / "scoring",
            )


if __name__ == "__main__":
    main()
