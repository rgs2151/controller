"""Cheap checks for the one-slice paper benchmark unit."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "parking/bench_evaluations/bench_evaluations.py"
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


def test_toxicity_data_uses_five_full_rtp_samples_and_shared_mmlu():
    toxic = [
        {"prompt_id": f"rtp:toxic:{index}", "text": f"toxic {index}", "toxicity": 0.9}
        for index in range(5100)
    ]
    nontoxic = [
        {
            "prompt_id": f"rtp:neutral:{index}",
            "text": f"neutral {index}",
            "toxicity": 0.1,
        }
        for index in range(5100)
    ]
    mmlu = [
        {
            "prompt_id": f"mmlu:subject:{index}",
            "prompt": f"Question: {index}\nA. a\nB. b\nC. c\nD. d\nAnswer:",
            "answer_index": index % 4,
            "subject": "subject",
        }
        for index in range(1000)
    ]
    data = bench_evaluations.build_toxicity_data(
        toxic + nontoxic,
        toxic,
        nontoxic,
        mmlu,
    )
    assert data["evaluation_samples"] == 1000
    assert data["evaluation_repetitions"] == 5
    assert len(data["calibration"]["toxicity"]["undesired"]) == 5000
    assert len(data["calibration"]["toxicity"]["desired"]) == 5000
    assert len(data["calibration"]["toxicity"]["jacobian"]) == 50
    assert len(data["capability_evaluation"]["mmlu"]) == 1000
    for repetition in data["evaluation"]["toxicity"].values():
        assert len(repetition) == 1000
        assert len({row["prompt_id"] for row in repetition}) == 1000
