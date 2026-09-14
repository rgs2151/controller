"""Cheap checks for the one-slice paper benchmark unit."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "robust_steerability/benchmarks/truthfulness_runtime.py"
SPEC = importlib.util.spec_from_file_location("bench_evaluations", SCRIPT)
bench_evaluations = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bench_evaluations)


def test_full_dataset_is_repeated_without_subsampling():
    generation = [{"question": f"question {index}"} for index in range(17)]
    multiple_choice = [
        {
            "question": f"calibration {index}",
            "mc2_targets": {"choices": ["true", "false"], "labels": [1, 0]},
        }
        for index in range(1900)
    ]
    data = bench_evaluations.build_truthfulqa_data(generation, multiple_choice)
    expected_ids = {f"truthfulqa:{index}" for index in range(len(generation))}
    assert data["evaluation_samples"] == len(generation)
    assert data["evaluation_repetitions"] == 5
    for repetition in data["evaluation"]["truthfulness"].values():
        assert len(repetition) == len(generation)
        assert {row["prompt_id"] for row in repetition} == expected_ids
    calibration = data["calibration"]["truthfulness"]
    assert len(calibration["undesired"]) == 1800
    assert len(calibration["desired"]) == 1800
    assert len(calibration["jacobian"]) == 35
    assert data["calibration_protocol"]["nested_prefixes"]["alqr"]["desired"] == 200
    assert data["calibration_protocol"]["nested_prefixes"]["odesteer"]["desired"] == 1800
    assert data["calibration_protocol"]["jacobian_max_length"] == 512
