from __future__ import annotations

import pytest
import torch

from robust_steerability.calibration import nominal_artifact


def test_nominal_dynamics_cache_is_shared_and_strict(tmp_path, monkeypatch) -> None:
    calls = []

    def fit(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.eye(3).repeat(2, 1, 1)

    monkeypatch.setattr(nominal_artifact, "average_prompt_jacobians", fit)
    records = [{"prompt_id": "p0", "text": "first"}]
    arguments = dict(
        artifact_path=tmp_path / "nominal" / "dynamics.pt",
        behavior="truthfulness",
        model_id="model",
        model_revision="revision",
        max_length=32,
        vjp_chunk_size=4,
        runtime={"requested_device": "cpu"},
    )
    first = nominal_artifact.load_or_fit_nominal_dynamics(
        object(), object(), records, **arguments
    )
    second = nominal_artifact.load_or_fit_nominal_dynamics(
        object(), object(), records, **arguments
    )
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert len(calls) == 1

    changed = [{"prompt_id": "p0", "text": "changed"}]
    reused = nominal_artifact.reuse_or_fit_nominal_dynamics(
        object(), object(), changed, **arguments
    )
    torch.testing.assert_close(reused, first, rtol=0, atol=0)
    assert len(calls) == 1

    with pytest.raises(ValueError, match="identity mismatch"):
        nominal_artifact.load_or_fit_nominal_dynamics(
            object(), object(), changed, **arguments
        )
