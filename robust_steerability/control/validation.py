"""Validation for the controller/LLM boundary."""

from __future__ import annotations

import torch

from robust_steerability.control.types import (
    ControllerSolution,
    FiniteHorizonControlProblem,
)


def validate_control_problem(problem: FiniteHorizonControlProblem) -> None:
    """Validate the dimensions and finite values of a control problem."""

    dynamics = problem.dynamics
    control_channels = problem.control_channels
    state_costs = problem.state_costs
    control_costs = problem.control_costs
    terminal_cost = problem.terminal_cost

    if dynamics.ndim != 3 or dynamics.shape[1] != dynamics.shape[2]:
        raise ValueError("dynamics must have shape (horizon, state, state)")
    horizon, state_dimension, _ = dynamics.shape
    if control_channels.ndim != 3 or control_channels.shape[:2] != (horizon, state_dimension):
        raise ValueError("control_channels must have shape (horizon, state, control)")
    control_dimension = control_channels.shape[2]
    if state_costs.shape != (horizon, state_dimension, state_dimension):
        raise ValueError("state_costs must have shape (horizon, state, state)")
    if control_costs.shape != (horizon, control_dimension, control_dimension):
        raise ValueError("control_costs must have shape (horizon, control, control)")
    if terminal_cost.shape != (state_dimension, state_dimension):
        raise ValueError("terminal_cost must have shape (state, state)")
    if problem.disturbance_channels is not None:
        disturbance_channels = problem.disturbance_channels
        if disturbance_channels.ndim != 3 or disturbance_channels.shape[:2] != (
            horizon,
            state_dimension,
        ):
            raise ValueError(
                "disturbance_channels must have shape (horizon, state, disturbance)"
            )
    tensors = [
        dynamics,
        control_channels,
        state_costs,
        control_costs,
        terminal_cost,
    ]
    if problem.disturbance_channels is not None:
        tensors.append(problem.disturbance_channels)
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("control problem contains non-finite values")


def validate_controller_solution(
    problem: FiniteHorizonControlProblem,
    solution: ControllerSolution,
) -> None:
    """Validate controller gain shape and numerical values."""

    expected = (
        problem.horizon,
        problem.control_dimension,
        problem.state_dimension,
    )
    if solution.gains.shape != expected:
        raise ValueError(f"controller gains must have shape {expected}")
    if not torch.isfinite(solution.gains).all():
        raise ValueError("controller gains contain non-finite values")
