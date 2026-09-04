"""Representation-dynamics residual measurements."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def residual_metrics(
    states: torch.Tensor,
    controls: torch.Tensor,
    controller: Mapping[str, object],
    epsilon: float = 1e-12,
) -> dict[str, object]:
    """Compute the residual-check unit's controller-aware prompt metrics."""

    nominal = controller["nominal"].float()
    dynamics = controller["jacobians"].float()
    gains = controller["gains"].float()
    feature_unit = controller["feature_unit"].float()
    feature_norm = controller["feature_norm"].float()
    beta = controller["beta"].float()
    last_states = states[:, -1, :]
    last_controls = controls[:, -1, :]
    deviations = last_states[:-1] - nominal[:-1]
    predicted = (
        nominal[1:]
        + torch.einsum("lij,lj->li", dynamics, deviations)
        + last_controls
    )
    residuals = last_states[1:] - predicted
    residual_magnitude = float(torch.linalg.vector_norm(residuals).item())

    residual_response = torch.zeros_like(last_states[0])
    for layer_index in range(dynamics.shape[0]):
        gain_feature = gains[layer_index] @ feature_unit[layer_index]
        residual_response = (
            dynamics[layer_index] @ residual_response
            - gain_feature
            * torch.dot(feature_unit[layer_index], residual_response)
            + residuals[layer_index]
        )
    final_scale = float(feature_norm[-1].clamp_min(epsilon).item())
    directional_effect = (
        abs(float(torch.dot(feature_unit[-1], residual_response).item()))
        / final_scale
    )
    amplification = directional_effect / max(residual_magnitude, epsilon)
    failure = (
        abs(float((beta[-1] - torch.dot(feature_unit[-1], last_states[-1])).item()))
        / final_scale
    )
    layer_relative_residual = torch.linalg.vector_norm(
        residuals,
        dim=1,
    ) / torch.linalg.vector_norm(last_states[1:], dim=1).clamp_min(epsilon)
    return {
        "residual_magnitude": residual_magnitude,
        "directional_effect": directional_effect,
        "amplification": amplification,
        "failure": failure,
        "layer_relative_residual": layer_relative_residual.numpy(),
    }
