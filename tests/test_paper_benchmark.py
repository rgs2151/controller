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
        for index in range(1900)
    ]
    data = paper_benchmark.build_truthfulqa_data(generation, multiple_choice)
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


def test_empty_table_is_explicitly_tbd():
    table = paper_benchmark.truthfulness_markdown_table({})
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
    table = paper_benchmark.truthfulness_markdown_table({("gemma2b", "alqr"): result})
    row = next(line for line in table.splitlines() if "| Gemma-2-2B | A-LQR |" in line)
    assert "67.81 ± 0.38" in row
    assert "73.17 ± 0.50" in row
    assert "92.68 ± 0.40" in row
    assert row.count("TBD") == 4


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
    data = paper_benchmark.build_toxicity_data(
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


def test_toxicity_table_fills_only_completed_source_metrics():
    result = {
        "metrics": {
            "toxicity": {"mean": 0.18, "standard_error": 0.08},
            "dist_2": {"mean": 0.68, "standard_error": 0.004},
            "mmlu": {"mean": 53.56, "standard_error": 1.62},
            "perplexity": {"mean": 12.26, "standard_error": 0.08},
        }
    }
    table = paper_benchmark.toxicity_markdown_table({("gemma2b", "alqr"): result})
    row = next(line for line in table.splitlines() if "| Gemma-2-2B | A-LQR |" in line)
    assert "0.18 ± 0.08" in row
    assert "0.68 ± 0.00" in row
    assert "53.56 ± 1.62" in row
    assert "12.26 ± 0.08" in row
