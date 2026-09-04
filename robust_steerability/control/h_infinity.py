"""Offline synthesis and online execution for finite-horizon H-infinity."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from robust_steerability.control.base import SynthesizedController
from robust_steerability.control.types import (
    ControllerSolution,
    FiniteHorizonControlProblem,
)
from robust_steerability.control.validation import (
    validate_control_problem,
    validate_controller_solution,
)


@dataclass(frozen=True)
class HInfinityOptions:
    """Numerical search settings for finite-horizon H-infinity synthesis."""

    gamma_lower: float = 0.0
    gamma_upper: float = 1000.0
    tolerance: float = 1e-5
    max_iterations: int = 100


class HInfinityController(SynthesizedController):
    """Finite-horizon H-infinity controller.

    This file is the complete controller-theory extension point. The
    collaborator implements offline synthesis in :meth:`synthesize`. Online,
    :meth:`control` computes ``u[k] = K[k] x[k]`` and the inherited
    :meth:`intervention` method returns ``B[k] u[k]`` for the model runtime.
    """

    def __init__(
        self,
        gains: torch.Tensor,
        control_channels: torch.Tensor,
        *,
        feasible: bool,
        gamma_star: float | None,
        diagnostics: dict[str, object] | None = None,
    ):
        super().__init__(control_channels)
        if gains.ndim != 3:
            raise ValueError("gains must have shape (horizon, control, state)")
        if (
            control_channels.ndim != 3
            or control_channels.shape[0] != gains.shape[0]
            or control_channels.shape[1] != gains.shape[2]
            or control_channels.shape[2] != gains.shape[1]
        ):
            raise ValueError(
                "control_channels must have shape (horizon, state, control)"
            )
        if not torch.isfinite(gains).all():
            raise ValueError("gains contain non-finite values")
        self.register_buffer("gains", gains)
        self.feasible = feasible
        self.gamma_star = gamma_star
        self.diagnostics = diagnostics or {}

    # Offline synthesis and construction.

    @classmethod
    def synthesize(
        cls,
        problem: FiniteHorizonControlProblem,
        *,
        device: str | torch.device | None = None,
        options: HInfinityOptions = HInfinityOptions(),
    ) -> HInfinityController:
        """Synthesize and return a ready finite-horizon H-infinity controller.

        TODO(controller collaborator): implement the verified finite-horizon
        H-infinity recursion, its saddle-point feasibility checks, and the
        gamma search configured by ``options``, performing the tensor work on
        ``device`` when it is supplied. Return ``cls.from_solution`` with gains
        under the direct convention ``u[k] = K[k] x[k]`` and include
        feasibility, ``gamma_star``, and useful numerical diagnostics in the
        resulting :class:`ControllerSolution`.
        """

        validate_control_problem(problem)
        if problem.disturbance_channels is None:
            raise ValueError("H-infinity synthesis requires disturbance_channels")
        del device, options
        raise NotImplementedError(
            "Implement the finite-horizon H-infinity recursion and gamma search here"
        )

    @classmethod
    def from_solution(
        cls,
        problem: FiniteHorizonControlProblem,
        solution: ControllerSolution,
    ) -> HInfinityController:
        """Construct online H-infinity behavior from synthesized outputs."""

        validate_control_problem(problem)
        if problem.disturbance_channels is None:
            raise ValueError("H-infinity requires disturbance_channels")
        validate_controller_solution(problem, solution)
        if solution.controller != "h_infinity":
            raise ValueError("solution.controller must be 'h_infinity'")
        if solution.gamma_star is None:
            raise ValueError("H-infinity solution must report gamma_star")
        return cls(
            gains=solution.gains,
            control_channels=problem.control_channels.to(solution.gains.device),
            feasible=solution.feasible,
            gamma_star=solution.gamma_star,
            diagnostics=solution.diagnostics,
        )

    # Online control.

    def control(
        self,
        layer_index: int,
        feedback_input: torch.Tensor,
    ) -> torch.Tensor:
        """Return the layer control coordinates for a state deviation."""

        return feedback_input @ self.gains[layer_index].T

    # Serialization.

    def solution(self) -> ControllerSolution:
        """Return a serializable copy of the synthesized H-infinity result."""

        return ControllerSolution(
            controller="h_infinity",
            gains=self.gains.detach().cpu(),
            feasible=self.feasible,
            gamma_star=self.gamma_star,
            diagnostics=dict(self.diagnostics),
        )
