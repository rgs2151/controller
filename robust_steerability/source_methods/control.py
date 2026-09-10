"""Source-faithful A-LQR, S-PID, and ActAddLFS construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from robust_steerability.control import LQRController, PIDController, PIDGains, solve_identity_input_lqr
from robust_steerability.runtime.policy import SemanticSetpointPolicy


@dataclass(frozen=True)
class SetpointCalibration:
    contrast: torch.Tensor
    feature_norm: torch.Tensor

    def setpoints(self, multiplier: float) -> torch.Tensor:
        return multiplier * self.feature_norm

    def unit_features(self, zero_threshold: float) -> torch.Tensor:
        """Normalize with the method's own source threshold."""

        unit = torch.zeros_like(self.contrast)
        nonzero = self.feature_norm != 0 if zero_threshold == 0 else self.feature_norm >= zero_threshold
        unit[nonzero] = self.contrast[nonzero] / self.feature_norm[nonzero, None]
        return unit


def fit_setpoint_calibration(negative_mean: torch.Tensor, positive_mean: torch.Tensor) -> SetpointCalibration:
    """Fit the paper's positive-minus-negative linear feature at every depth."""

    if negative_mean.shape != positive_mean.shape or negative_mean.ndim != 2:
        raise ValueError("means must have matching shape (layers + 1, hidden)")
    contrast = positive_mean - negative_mean
    norms = torch.linalg.vector_norm(contrast, dim=1)
    return SetpointCalibration(contrast=contrast, feature_norm=norms)


def build_alqr_policy(
    dynamics: torch.Tensor,
    calibration: SetpointCalibration,
    *,
    multiplier: float,
    q: float,
    r: float,
    q_final: float,
    device: str | torch.device,
) -> SemanticSetpointPolicy:
    """Synthesize and construct the exact full-state A-LQR tracking policy."""

    gains = solve_identity_input_lqr(dynamics, device, q, r, q_final)
    controller = LQRController.from_tracking_gains(gains)
    return SemanticSetpointPolicy(
        controller,
        calibration.unit_features(0.0),
        calibration.setpoints(multiplier),
    )


def build_spid_policy(
    calibration: SetpointCalibration,
    *,
    multiplier: float,
    kp: float,
    ki: float,
    kd: float,
) -> SemanticSetpointPolicy:
    """Construct the source S-PID policy, including its ten-layer reset rule."""

    controller = PIDController(PIDGains(kp, ki, kd))
    return SemanticSetpointPolicy(
        controller,
        calibration.unit_features(1e-6),
        calibration.setpoints(multiplier),
    )


class LayerSelectedSetpointPolicy(SemanticSetpointPolicy):
    """ActAddLFS: proportional setpoint tracking at selected layers only."""

    def __init__(self, calibration: SetpointCalibration, multiplier: float, layer_indices: tuple[int, ...]):
        super().__init__(
            PIDController(PIDGains(1.0, 0.0, 0.0)),
            calibration.unit_features(1e-6),
            calibration.setpoints(multiplier),
        )
        self.layer_indices = layer_indices

    def activation_delta(self, layer_index: int, activation: torch.Tensor) -> torch.Tensor:
        if layer_index not in self.layer_indices:
            return torch.zeros_like(activation)
        return super().activation_delta(layer_index, activation)
