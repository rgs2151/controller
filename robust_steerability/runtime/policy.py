"""Policies that translate controller outputs into activation interventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from robust_steerability.control.pid import PIDController, PIDGains
from robust_steerability.control.types import ControllerSolution


class ActivationPolicy(Protocol):
    """The only policy surface required by a model adapter."""

    def reset(self) -> None: ...

    def activation_delta(
        self,
        layer_index: int,
        activation: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass
class SetpointLQRPolicy:
    """Activation-LQR feedback around a layer-wise semantic setpoint."""

    tracking_gains: torch.Tensor
    feature_unit: torch.Tensor
    setpoints: torch.Tensor

    @classmethod
    def from_artifact(cls, artifact: dict[str, object]) -> SetpointLQRPolicy:
        return cls(
            tracking_gains=artifact["gains"],
            feature_unit=artifact["feature_unit"],
            setpoints=artifact["beta"],
        )

    def reset(self) -> None:
        return None

    def activation_delta(
        self,
        layer_index: int,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        feature = self.feature_unit[layer_index].to(device=activation.device)
        gain = self.tracking_gains[layer_index].to(device=activation.device)
        setpoint = self.setpoints[layer_index].to(device=activation.device)
        scalar_error = setpoint - activation @ feature
        vector_error = scalar_error.unsqueeze(-1) * feature
        return vector_error @ gain.T


@dataclass
class StateFeedbackPolicy:
    """Apply a synthesized state-feedback solution around reference states."""

    solution: ControllerSolution
    control_channels: torch.Tensor
    reference_states: torch.Tensor

    def reset(self) -> None:
        return None

    def activation_delta(
        self,
        layer_index: int,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        reference = self.reference_states[layer_index].to(
            device=activation.device,
            dtype=activation.dtype,
        )
        gain = self.solution.gains[layer_index].to(
            device=activation.device,
            dtype=activation.dtype,
        )
        channel = self.control_channels[layer_index].to(
            device=activation.device,
            dtype=activation.dtype,
        )
        deviation = activation - reference
        control = deviation @ gain.T
        return control @ channel.T


class PIDSetpointPolicy:
    """PID feedback on the same semantic setpoint used by Activation-LQR."""

    def __init__(
        self,
        feature_unit: torch.Tensor,
        setpoints: torch.Tensor,
        gains: PIDGains,
    ):
        self.feature_unit = feature_unit
        self.setpoints = setpoints
        self.controller = PIDController(gains)

    def reset(self) -> None:
        self.controller.reset()

    def activation_delta(
        self,
        layer_index: int,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        feature = self.feature_unit[layer_index].to(
            device=activation.device,
            dtype=activation.dtype,
        )
        setpoint = self.setpoints[layer_index].to(
            device=activation.device,
            dtype=activation.dtype,
        )
        scalar_error = setpoint - activation @ feature
        vector_error = scalar_error.unsqueeze(-1) * feature
        return self.controller.control(vector_error)


@dataclass
class ActivationAdditionPolicy:
    """Apply fixed layer-wise activation directions."""

    directions: torch.Tensor
    strength: float

    def reset(self) -> None:
        return None

    def activation_delta(
        self,
        layer_index: int,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        direction = self.directions[layer_index].to(
            device=activation.device,
            dtype=activation.dtype,
        )
        return self.strength * direction.expand_as(activation)


@dataclass
class SumPolicy:
    """Compose multiple policies, including multi-concept steering."""

    policies: tuple[ActivationPolicy, ...]

    def reset(self) -> None:
        for policy in self.policies:
            policy.reset()

    def activation_delta(
        self,
        layer_index: int,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        deltas = [
            policy.activation_delta(layer_index, activation)
            for policy in self.policies
        ]
        return torch.stack(deltas).sum(dim=0)
