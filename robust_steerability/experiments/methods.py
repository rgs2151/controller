"""Canonical controller names and reduced-state policy construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from robust_steerability.control import (
    HInfinityController,
    LQRController,
    PIDController,
    PIDGains,
)
from robust_steerability.runtime.policy import ReducedStateSetpointPolicy


METHOD_LABELS = {
    "original": "Original",
    "alqr": "A-LQR",
    "spid": "S-PID",
    "hinf": "H-infinity",
}


@dataclass
class ReducedControllerArtifact:
    """Calibrated coordinates and controller solutions for one model."""

    means: torch.Tensor
    encoders: torch.Tensor
    decoders: torch.Tensor
    feature_unit: torch.Tensor
    setpoints: torch.Tensor
    control_channels: torch.Tensor
    lqr_gains: torch.Tensor
    hinf_gains: torch.Tensor
    hinf_feasible: bool
    gamma_star: float | None
    hinf_diagnostics: dict[str, object]


def build_policy(
    method: str,
    artifact: ReducedControllerArtifact,
    *,
    kp: float,
    ki: float,
    kd: float,
):
    """Build the online policy for one canonical method name."""

    if method == "original":
        return None
    if method == "alqr":
        controller = LQRController.from_tracking_gains(
            artifact.lqr_gains,
            control_channels=artifact.control_channels,
        )
    elif method == "spid":
        controller = PIDController(
            PIDGains(proportional=kp, integral=ki, derivative=kd),
            control_channels=artifact.control_channels,
        )
    elif method == "hinf":
        controller = HInfinityController(
            gains=artifact.hinf_gains,
            control_channels=artifact.control_channels,
            feasible=artifact.hinf_feasible,
            gamma_star=artifact.gamma_star,
            diagnostics=artifact.hinf_diagnostics,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return ReducedStateSetpointPolicy(
        controller=controller,
        means=artifact.means,
        encoders=artifact.encoders,
        decoders=artifact.decoders,
        feature_unit=artifact.feature_unit,
        setpoints=artifact.setpoints,
    )
