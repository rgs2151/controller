"""Model-independent adapters from activations to controller feedback inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from robust_steerability.control.base import Controller


class ActivationPolicy(Protocol):
    """The only runtime surface required by a model adapter."""

    def prepare(self, device: torch.device, dtype: torch.dtype) -> None: ...

    def reset(self) -> None: ...

    def activation_delta(
        self,
        layer_index: int,
        activation: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass
class DirectControllerPolicy:
    """Pass model activations directly to a controller as feedback inputs."""

    controller: Controller

    def prepare(self, device: torch.device, dtype: torch.dtype) -> None:
        self.controller.to(device=device, dtype=dtype)

    def reset(self) -> None:
        self.controller.reset()

    def activation_delta(
        self,
        layer_index: int,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        return self.controller.intervention(layer_index, activation)


@dataclass
class SemanticSetpointPolicy:
    """Build a full-state deviation from a layer-wise semantic setpoint."""

    controller: Controller
    feature_unit: torch.Tensor
    setpoints: torch.Tensor

    def prepare(self, device: torch.device, dtype: torch.dtype) -> None:
        self.controller.to(device=device, dtype=dtype)
        self.feature_unit = self.feature_unit.to(device=device, dtype=dtype)
        self.setpoints = self.setpoints.to(device=device, dtype=dtype)

    def reset(self) -> None:
        self.controller.reset()

    def activation_delta(
        self,
        layer_index: int,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        feature = self.feature_unit[layer_index]
        setpoint = self.setpoints[layer_index]
        scalar_deviation = activation @ feature - setpoint
        state_deviation = scalar_deviation.unsqueeze(-1) * feature
        return self.controller.intervention(layer_index, state_deviation)


@dataclass
class ReferenceStatePolicy:
    """Build controller feedback from full layer-wise reference states."""

    controller: Controller
    reference_states: torch.Tensor

    def prepare(self, device: torch.device, dtype: torch.dtype) -> None:
        self.controller.to(device=device, dtype=dtype)
        self.reference_states = self.reference_states.to(device=device, dtype=dtype)

    def reset(self) -> None:
        self.controller.reset()

    def activation_delta(
        self,
        layer_index: int,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        state_deviation = activation - self.reference_states[layer_index]
        return self.controller.intervention(layer_index, state_deviation)


@dataclass
class SumPolicy:
    """Compose multiple policies, including multi-concept steering."""

    policies: tuple[ActivationPolicy, ...]

    def __post_init__(self) -> None:
        if not self.policies:
            raise ValueError("SumPolicy requires at least one policy")

    def prepare(self, device: torch.device, dtype: torch.dtype) -> None:
        for policy in self.policies:
            policy.prepare(device, dtype)

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
