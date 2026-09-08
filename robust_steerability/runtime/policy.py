"""Model-independent adapters from activations to controller feedback inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol

import torch

from robust_steerability.control.base import Controller
from robust_steerability.runtime.diagnostics import ReducedTrajectoryRecorder


class ActivationPolicy(Protocol):
    """The only runtime surface required by a model adapter."""

    site: ClassVar[str] = "block_input"

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

    site: ClassVar[str] = "block_input"
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

    site: ClassVar[str] = "block_input"
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
class ReducedStateSetpointPolicy:
    """Apply a controller in calibrated reduced coordinates.

    Layer inputs are centered and projected with ``encoders``. The controller
    returns an intervention in the next layer's standardized coordinates, and
    ``decoders`` map it back to the model hidden space.
    """

    site: ClassVar[str] = "block_input"
    controller: Controller
    means: torch.Tensor
    encoders: torch.Tensor
    decoders: torch.Tensor
    feature_unit: torch.Tensor
    setpoints: torch.Tensor
    reference_controls: torch.Tensor
    recorder: ReducedTrajectoryRecorder | None = None

    def prepare(self, device: torch.device, dtype: torch.dtype) -> None:
        self.controller.to(device=device, dtype=torch.float32)
        self.means = self.means.to(device=device, dtype=torch.float32)
        self.encoders = self.encoders.to(device=device, dtype=torch.float32)
        self.decoders = self.decoders.to(device=device, dtype=torch.float32)
        self.feature_unit = self.feature_unit.to(device=device, dtype=torch.float32)
        self.setpoints = self.setpoints.to(device=device, dtype=torch.float32)
        self.reference_controls = self.reference_controls.to(device=device, dtype=torch.float32)

    def reset(self) -> None:
        self.controller.reset()
        if self.recorder is not None:
            self.recorder.reset()

    def activation_delta(
        self,
        layer_index: int,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        activation_float = activation.float()
        reduced = (
            activation_float - self.means[layer_index]
        ) @ self.encoders[layer_index]
        feature = self.feature_unit[layer_index]
        state_deviation = reduced - self.setpoints[layer_index] * feature
        deviation_control = self.controller.control(layer_index, state_deviation)
        control = self.reference_controls[layer_index] + deviation_control
        channels = self.controller.control_channels
        reduced_delta = control if channels is None else control @ channels[layer_index].T
        hidden_delta = reduced_delta @ self.decoders[layer_index].T
        hidden_delta = hidden_delta.to(dtype=activation.dtype)
        if self.recorder is not None:
            self.recorder.append(layer_index, state=reduced, feedback=state_deviation,
                                 control=control, deviation_control=deviation_control, reduced_intervention=reduced_delta,
                                 hidden_delta=hidden_delta)
        return hidden_delta


@dataclass
class ReferenceStatePolicy:
    """Build controller feedback from full layer-wise reference states."""

    site: ClassVar[str] = "block_input"
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

    site: ClassVar[str] = "block_input"
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
