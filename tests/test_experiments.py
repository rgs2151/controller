from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from robust_steerability.control import PIDController, PIDGains
from robust_steerability.experiments.manifest import load_manifest
from robust_steerability.runtime.policy import ReducedStateSetpointPolicy


def test_manifest_rejects_obsolete_method_name(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "id_toxicity",
                "revisions": {"dataset": "dataset-commit"},
                "models": [
                    {
                        "label": "tiny",
                        "model_id": "tiny",
                        "revision": "model-commit",
                        "dtype": "float32",
                        "quantized": False,
                    }
                ],
                "methods": ["new_method"],
            }
        )
    )
    with pytest.raises(ValueError, match="Unknown method"):
        load_manifest(path)


def test_manifest_rejects_moving_model_revision(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "calibration",
                "revisions": {"dataset": "dataset-commit"},
                "models": [
                    {
                        "label": "tiny",
                        "model_id": "tiny",
                        "revision": "main",
                        "dtype": "float32",
                        "quantized": False,
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="immutable"):
        load_manifest(path)


def test_current_reference_manifests_are_complete() -> None:
    root = Path(__file__).resolve().parents[1] / "ref/paper_benchmark_50"
    for name in ("toxicity.json", "truthfulness.json"):
        manifest = load_manifest(root / name)
        assert manifest.payload["sample_count"] == 50


def test_reduced_policy_decodes_next_layer_control_coordinates() -> None:
    identity = torch.eye(2).unsqueeze(0)
    policy = ReducedStateSetpointPolicy(
        controller=PIDController(PIDGains(1.0, 0.0, 0.0), identity),
        means=torch.zeros(1, 3),
        encoders=torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]]),
        decoders=torch.tensor([[[2.0, 0.0], [0.0, 3.0], [0.0, 0.0]]]),
        feature_unit=torch.tensor([[1.0, 0.0]]),
        setpoints=torch.tensor([2.0]),
    )
    policy.prepare(torch.device("cpu"), torch.float32)
    policy.reset()
    delta = policy.activation_delta(0, torch.tensor([[1.0, 4.0, 7.0]]))
    torch.testing.assert_close(delta, torch.tensor([[2.0, -12.0, 0.0]]))
