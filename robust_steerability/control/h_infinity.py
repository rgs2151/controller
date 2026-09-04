"""Collaborator-owned finite-horizon H-infinity synthesis entry point."""

from __future__ import annotations

from dataclasses import dataclass

from robust_steerability.control.types import (
    ControllerSolution,
    FiniteHorizonControlProblem,
)
from robust_steerability.control.validation import validate_control_problem


@dataclass(frozen=True)
class HInfinityOptions:
    """Numerical search settings for finite-horizon H-infinity synthesis."""

    gamma_lower: float = 0.0
    gamma_upper: float = 1000.0
    tolerance: float = 1e-5
    max_iterations: int = 100


def synthesize_h_infinity(
    problem: FiniteHorizonControlProblem,
    options: HInfinityOptions = HInfinityOptions(),
) -> ControllerSolution:
    """Synthesize finite-horizon H-infinity feedback.

    This is the controller-theory extension point. ``problem`` is already in
    deviation coordinates and contains every numerical object required by the
    controller. The implementation must return gains following
    ``u[k] = K[k] x[k]``, report feasibility, and set ``gamma_star``.
    """

    validate_control_problem(problem)
    if problem.disturbance_channels is None:
        raise ValueError("H-infinity synthesis requires disturbance_channels")
    raise NotImplementedError(
        "Implement the verified finite-horizon H-infinity recursion and gamma bisection here"
    )
