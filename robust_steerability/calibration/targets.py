"""Semantic target construction from calibrated activations."""

from __future__ import annotations

import torch


def build_contrastive_target(
    target_mean: torch.Tensor,
    opposite_mean: torch.Tensor,
    setpoint_multiplier: float,
    epsilon: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Build the layer-wise linear-feature setpoint used by Activation-LQR."""

    feature = target_mean - opposite_mean
    feature_norm = torch.linalg.vector_norm(feature, dim=1)
    feature_unit = feature / feature_norm.clamp_min(epsilon).unsqueeze(1)
    return {
        "nominal": target_mean,
        "opposite_mean": opposite_mean,
        "feature": feature,
        "feature_norm": feature_norm,
        "feature_unit": feature_unit,
        "beta": setpoint_multiplier * feature_norm,
    }
