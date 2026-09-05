"""Pluggable steering-method registry for AppliedControler experiments.

This module exposes ready adapters for methods already implemented in-repo and
placeholder entries for methods that will be wired from external GitHub code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from robust_steerability.control import (
    ActivationAdditionController,
    HInfinityController,
    LQRController,
    PIDController,
    PIDGains,
)
from robust_steerability.runtime.policy import (
    ActivationPolicy,
    DirectControllerPolicy,
    SemanticSetpointPolicy,
)


@dataclass
class ControllerArtifact:
    """Serialized tensors needed to instantiate an online steering policy."""

    feature_unit: torch.Tensor | None = None
    setpoints: torch.Tensor | None = None
    gains: torch.Tensor | None = None
    directions: torch.Tensor | None = None
    control_channels: torch.Tensor | None = None
    feasible: bool | None = None
    gamma_star: float | None = None
    diagnostics: dict[str, object] | None = None


class MethodAdapter(Protocol):
    """Factory for model-agnostic activation policies."""

    def build_policy(
        self,
        artifact: ControllerArtifact,
        **kwargs: object,
    ) -> ActivationPolicy:
        """Build and return a runtime policy for one method."""


class ALQRAdapter:
    """Build a SemanticSetpointPolicy backed by Activation-LQR gains."""

    def build_policy(
        self,
        artifact: ControllerArtifact,
        **kwargs: object,
    ) -> ActivationPolicy:
        del kwargs
        if artifact.gains is None:
            raise ValueError("ALQRAdapter requires artifact.gains")
        if artifact.feature_unit is None or artifact.setpoints is None:
            raise ValueError("ALQRAdapter requires feature_unit and setpoints")
        return SemanticSetpointPolicy(
            controller=LQRController.from_tracking_gains(
                artifact.gains,
                control_channels=artifact.control_channels,
            ),
            feature_unit=artifact.feature_unit,
            setpoints=artifact.setpoints,
        )


class SPIDAdapter:
    """Build a SemanticSetpointPolicy backed by a PID controller."""

    def build_policy(
        self,
        artifact: ControllerArtifact,
        **kwargs: object,
    ) -> ActivationPolicy:
        if artifact.feature_unit is None or artifact.setpoints is None:
            raise ValueError("SPIDAdapter requires feature_unit and setpoints")
        gains = PIDGains(
            proportional=float(kwargs.get("kp", 1.0)),
            integral=float(kwargs.get("ki", 0.0)),
            derivative=float(kwargs.get("kd", 0.0)),
        )
        controller = PIDController(gains=gains, control_channels=artifact.control_channels)
        return SemanticSetpointPolicy(
            controller=controller,
            feature_unit=artifact.feature_unit,
            setpoints=artifact.setpoints,
        )


class ActAddAdapter:
    """Build a direct activation-addition baseline policy."""

    def build_policy(
        self,
        artifact: ControllerArtifact,
        **kwargs: object,
    ) -> ActivationPolicy:
        if artifact.directions is None:
            raise ValueError("ActAddAdapter requires artifact.directions")
        strength = float(kwargs.get("strength", 1.0))
        return DirectControllerPolicy(
            controller=ActivationAdditionController(
                directions=artifact.directions,
                strength=strength,
            )
        )


class HInfinityAdapter:
    """Build a SemanticSetpointPolicy backed by an H-infinity controller."""

    def build_policy(
        self,
        artifact: ControllerArtifact,
        **kwargs: object,
    ) -> ActivationPolicy:
        del kwargs
        if artifact.gains is None:
            raise ValueError("HInfinityAdapter requires artifact.gains")
        if artifact.control_channels is None:
            raise ValueError("HInfinityAdapter requires artifact.control_channels")
        if artifact.feature_unit is None or artifact.setpoints is None:
            raise ValueError("HInfinityAdapter requires feature_unit and setpoints")
        feasible = True if artifact.feasible is None else bool(artifact.feasible)
        controller = HInfinityController(
            gains=artifact.gains,
            control_channels=artifact.control_channels,
            feasible=feasible,
            gamma_star=artifact.gamma_star,
            diagnostics={} if artifact.diagnostics is None else dict(artifact.diagnostics),
        )
        return SemanticSetpointPolicy(
            controller=controller,
            feature_unit=artifact.feature_unit,
            setpoints=artifact.setpoints,
        )


class DeferredAdapter:
    """Placeholder for methods that will be imported from external repositories."""

    def __init__(self, method_name: str):
        self._method_name = method_name

    def build_policy(
        self,
        artifact: ControllerArtifact,
        **kwargs: object,
    ) -> ActivationPolicy:
        del artifact, kwargs
        raise NotImplementedError(
            f"Method '{self._method_name}' is not wired yet. "
            "Attach the external implementation and return an ActivationPolicy."
        )


def default_registry() -> dict[str, MethodAdapter]:
    """Return built-in and deferred method adapters keyed by method name."""

    return {
        "alqr": ALQRAdapter(),
        "spid": SPIDAdapter(),
        "actadd": ActAddAdapter(),
        "hinf": HInfinityAdapter(),
        "external_method_a": DeferredAdapter("external_method_a"),
        "external_method_b": DeferredAdapter("external_method_b"),
        "new_method": HInfinityAdapter(),
    }
