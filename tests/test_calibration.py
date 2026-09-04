from __future__ import annotations

import hashlib
import json
import unittest

import numpy as np
import torch

from robust_steerability.artifacts import configuration_hash
from robust_steerability.benchmarks.base import validate_disjoint_prompt_ids
from robust_steerability.calibration.disturbances import fit_disturbance_geometry
from robust_steerability.calibration.normalization import fit_whitening
from robust_steerability.calibration.residuals import residual_metrics
from robust_steerability.calibration.targets import build_contrastive_target


def legacy_residual_metrics(
    states: torch.Tensor,
    controls: torch.Tensor,
    controller: dict[str, torch.Tensor],
    epsilon: float,
) -> dict[str, object]:
    nominal = controller["nominal"].float()
    dynamics = controller["jacobians"].float()
    gains = controller["gains"].float()
    feature_unit = controller["feature_unit"].float()
    feature_norm = controller["feature_norm"].float()
    beta = controller["beta"].float()
    last_states = states[:, -1, :]
    last_controls = controls[:, -1, :]
    deviations = last_states[:-1] - nominal[:-1]
    predicted = nominal[1:] + torch.einsum("lij,lj->li", dynamics, deviations) + last_controls
    residuals = last_states[1:] - predicted
    magnitude = float(torch.linalg.vector_norm(residuals).item())
    response = torch.zeros_like(last_states[0])
    for layer_index in range(dynamics.shape[0]):
        gain_feature = gains[layer_index] @ feature_unit[layer_index]
        response = (
            dynamics[layer_index] @ response
            - gain_feature * torch.dot(feature_unit[layer_index], response)
            + residuals[layer_index]
        )
    scale = float(feature_norm[-1].clamp_min(epsilon).item())
    effect = abs(float(torch.dot(feature_unit[-1], response).item())) / scale
    failure = abs(float((beta[-1] - torch.dot(feature_unit[-1], last_states[-1])).item())) / scale
    profile = torch.linalg.vector_norm(residuals, dim=1) / torch.linalg.vector_norm(
        last_states[1:], dim=1
    ).clamp_min(epsilon)
    return {
        "residual_magnitude": magnitude,
        "directional_effect": effect,
        "amplification": effect / max(magnitude, epsilon),
        "failure": failure,
        "layer_relative_residual": profile.numpy(),
    }


class CalibrationTests(unittest.TestCase):
    def test_split_validation_rejects_overlap(self) -> None:
        with self.assertRaises(ValueError):
            validate_disjoint_prompt_ids({"shared"}, set(), {"shared"})

    def test_configuration_hash_is_unchanged(self) -> None:
        configuration = {"b": 2, "a": [1, 3]}
        expected = hashlib.sha256(
            json.dumps(configuration, sort_keys=True).encode()
        ).hexdigest()
        self.assertEqual(configuration_hash(configuration), expected)

    def test_contrastive_target(self) -> None:
        target = torch.tensor([[3.0, 4.0], [2.0, 0.0]])
        opposite = torch.zeros_like(target)
        result = build_contrastive_target(target, opposite, 2.5)
        torch.testing.assert_close(result["feature"], target)
        torch.testing.assert_close(result["feature_norm"], torch.tensor([5.0, 2.0]))
        torch.testing.assert_close(result["beta"], torch.tensor([12.5, 5.0]))

    def test_residual_metrics_match_previous_unit_formula(self) -> None:
        torch.manual_seed(4)
        states = torch.randn(4, 3, 2)
        controls = torch.randn(3, 3, 2)
        feature = torch.nn.functional.normalize(torch.randn(4, 2), dim=1)
        controller = {
            "nominal": torch.randn(4, 2),
            "jacobians": torch.randn(3, 2, 2),
            "gains": torch.randn(3, 2, 2),
            "feature_unit": feature,
            "feature_norm": torch.rand(4) + 0.5,
            "beta": torch.rand(4),
        }
        expected = legacy_residual_metrics(states, controls, controller, 1e-12)
        actual = residual_metrics(states, controls, controller, 1e-12)
        for name in ["residual_magnitude", "directional_effect", "amplification", "failure"]:
            self.assertEqual(actual[name], expected[name])
        np.testing.assert_array_equal(
            actual["layer_relative_residual"],
            expected["layer_relative_residual"],
        )

    def test_disturbance_geometry_has_common_padded_rank(self) -> None:
        generator = torch.Generator().manual_seed(2)
        coefficients = torch.randn(20, 2, 1, generator=generator)
        directions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        residuals = coefficients * directions.unsqueeze(0)
        geometry = fit_disturbance_geometry(residuals, 0.95)
        self.assertEqual(geometry.channels.shape, (2, 3, 1))
        torch.testing.assert_close(geometry.retained_ranks, torch.ones(2, dtype=torch.long))
        self.assertTrue(torch.all(geometry.explained_variance >= 0.95))

    def test_whitening_uses_calibration_statistics(self) -> None:
        values = torch.tensor(
            [[-1.0, -2.0], [1.0, 2.0], [-1.0, 2.0], [1.0, -2.0]]
        )
        transform = fit_whitening(values)
        whitened = transform.apply(values)
        covariance = whitened.T @ whitened / (len(whitened) - 1)
        torch.testing.assert_close(covariance, torch.eye(2), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
