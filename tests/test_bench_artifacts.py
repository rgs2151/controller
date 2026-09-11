"""Cheap checks for the calibration-only benchmark unit."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "parking/bench_artifacts/bench_artifacts.py"
SPEC = importlib.util.spec_from_file_location("calibration_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


def test_truthfulness_calibration_selection_is_frozen():
    rows = [
        {
            "question": f"question {index}",
            "mc2_targets": {"choices": ["true", "false"], "labels": [1, 0]},
        }
        for index in range(300)
    ]
    data = benchmark.build_calibration_data(rows)
    calibration = data["calibration"]
    assert {key: len(value) for key, value in calibration.items()} == {
        "undesired": 200,
        "desired": 200,
        "jacobian": 35,
    }
    assert data["selection"]["jacobian"]["max_length"] == 512
    assert data["selection"]["seeds"] == {"undesired": 42, "desired": 43, "jacobian": 44}


def test_jacobian_sharding_and_raw_aggregation(tmp_path):
    records = [{"prompt_id": str(index)} for index in range(5)]
    assert [row["prompt_id"] for row in benchmark._partition_records(records, 0, 2)] == [
        "0",
        "2",
        "4",
    ]
    assert [row["prompt_id"] for row in benchmark._partition_records(records, 1, 2)] == [
        "1",
        "3",
    ]

    shards = [tmp_path / "shard_00", tmp_path / "shard_01"]
    values = [1.0, 3.0, 5.0]
    for index, value in enumerate(values):
        shard = shards[index % 2]
        prompt = shard / f"prompt_{index}"
        prompt.mkdir(parents=True)
        torch.save(torch.full((2, 2), value), prompt / "layer_000.pt")
        torch.save(torch.full((2, 2), value + 1), prompt / "layer_001.pt")
    averaged = benchmark.aggregate_raw_jacobians(shards, [2, 1])
    assert averaged.shape == (2, 2, 2)
    assert torch.equal(averaged[0], torch.full((2, 2), 3.0))
    assert torch.equal(averaged[1], torch.full((2, 2), 4.0))
