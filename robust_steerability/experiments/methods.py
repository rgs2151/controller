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
from robust_steerability.runtime.diagnostics import ReducedTrajectoryRecorder


METHOD_LABELS = {
    "original": "Original",
    "alqr": "A-LQR",
    "spid": "S-PID",
    "hinf": "H-infinity",
    "iti": "ITI", "actadd": "ActAdd", "mean_act": "Mean-AcT",
    "linear_act": "Linear-AcT", "pid_act": "PID-AcT", "odesteer": "ODESteer",
}


@dataclass
class ReducedControllerArtifact:
    """Calibrated coordinates and controller solutions for one model."""

    means: torch.Tensor
    encoders: torch.Tensor
    decoders: torch.Tensor
    feature_unit: torch.Tensor
    setpoints: torch.Tensor
    reference_controls: torch.Tensor
    control_channels: torch.Tensor
    lqr_gains: torch.Tensor
    hinf_gains: torch.Tensor
    hinf_feasible: bool
    gamma_star: float | None
    hinf_diagnostics: dict[str, object]
    baselines: dict


def build_policy(
    method: str,
    artifact: ReducedControllerArtifact,
    *,
    kp: float,
    ki: float,
    kd: float,
    record: bool = False,
):
    """Build the online policy for one canonical method name."""

    if method == "original":
        return None
    if method in {"iti", "actadd", "mean_act", "linear_act", "pid_act", "odesteer"}:
        from robust_steerability.experiments.baselines import BaselinePolicy
        return BaselinePolicy(method, artifact.baselines, strength=artifact.baselines["strengths"][method], record=record)
    if method == "alqr":
        controller = LQRController(
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
        reference_controls=artifact.reference_controls,
        recorder=ReducedTrajectoryRecorder() if record else None,
    )
