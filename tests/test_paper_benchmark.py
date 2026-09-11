"""Cheap checks for the one-slice paper benchmark unit."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "parking/paper_benchmark/paper_benchmark.py"
SPEC = importlib.util.spec_from_file_location("paper_benchmark", SCRIPT)
paper_benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(paper_benchmark)


def test_full_dataset_is_repeated_without_subsampling():
    generation = [{"question": f"question {index}"} for index in range(17)]
    multiple_choice = [
        {
            "question": f"calibration {index}",
            "mc2_targets": {"choices": ["true", "false"], "labels": [1, 0]},
        }
        for index in range(20)
    ]
    data = paper_benchmark.build_truthfulqa_data(generation, multiple_choice)
    expected_ids = {f"truthfulqa:{index}" for index in range(len(generation))}
    assert data["evaluation_samples"] == len(generation)
    assert data["evaluation_repetitions"] == 5
    for repetition in data["evaluation"]["truthfulness"].values():
        assert len(repetition) == len(generation)
        assert {row["prompt_id"] for row in repetition} == expected_ids
    calibration = data["calibration"]["truthfulness"]
    assert len(calibration["undesired"]) == 12
    assert len(calibration["desired"]) == 12
    assert len(calibration["jacobian"]) == 1


def test_empty_table_is_explicitly_tbd():
    table = paper_benchmark.markdown_table({})
    assert "| Gemma-2-2B | Original | TBD |" in table
    assert "| Gemma-2-2B | A-LQR | TBD |" in table
    assert "five complete 817-question repetitions" in table


def test_completed_id_result_fills_only_id_components():
    result = {
        "metrics": {
            "truth_x_info": {"mean": 67.81, "standard_error": 0.38},
            "truth": {"mean": 73.17, "standard_error": 0.5},
            "info": {"mean": 92.68, "standard_error": 0.4},
        }
    }
    table = paper_benchmark.markdown_table({("gemma2b", "alqr"): result})
    row = next(line for line in table.splitlines() if "| Gemma-2-2B | A-LQR |" in line)
    assert "67.81 ± 0.38" in row
    assert "73.17 ± 0.50" in row
    assert "92.68 ± 0.40" in row
    assert row.count("TBD") == 4
