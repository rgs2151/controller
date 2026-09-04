"""Finite-horizon discrete-time LQR synthesis."""

from __future__ import annotations

import torch

from robust_steerability.control.types import (
    ControllerSolution,
    FiniteHorizonControlProblem,
)
from robust_steerability.control.validation import (
    validate_control_problem,
    validate_controller_solution,
)


def synthesize_lqr(
    problem: FiniteHorizonControlProblem,
    device: str | torch.device | None = None,
) -> ControllerSolution:
    """Solve finite-horizon LQR and return gains for ``u[k] = K[k] x[k]``."""

    validate_control_problem(problem)
    target_device = device if device is not None else problem.dynamics.device
    dynamics = problem.dynamics.to(device=target_device, dtype=torch.float32)
    control_channels = problem.control_channels.to(device=target_device, dtype=torch.float32)
    state_costs = problem.state_costs.to(device=target_device, dtype=torch.float32)
    control_costs = problem.control_costs.to(device=target_device, dtype=torch.float32)
    value = problem.terminal_cost.to(device=target_device, dtype=torch.float32)
    gains = torch.empty(
        problem.horizon,
        problem.control_dimension,
        problem.state_dimension,
        device=target_device,
        dtype=torch.float32,
    )

    for layer_index in reversed(range(problem.horizon)):
        layer_dynamics = dynamics[layer_index]
        control_channel = control_channels[layer_index]
        control_hessian = (
            control_costs[layer_index]
            + control_channel.T @ value @ control_channel
        )
        cross_term = control_channel.T @ value @ layer_dynamics
        positive_gain = torch.linalg.solve(control_hessian, cross_term)
        gains[layer_index] = -positive_gain
        value = (
            state_costs[layer_index]
            + layer_dynamics.T @ value @ layer_dynamics
            - cross_term.T @ positive_gain
        )
        value = 0.5 * (value + value.T)

    solution = ControllerSolution(
        controller="lqr",
        gains=gains.cpu(),
        diagnostics={"feedback_convention": "u[k] = K[k] x[k]"},
    )
    validate_controller_solution(problem, solution)
    return solution


def solve_identity_input_lqr(
    dynamics: torch.Tensor,
    device: str | torch.device,
    state_cost: float,
    control_cost: float,
    terminal_cost: float,
) -> torch.Tensor:
    """Return positive tracking-error gains used by Activation-LQR.

    Activation-LQR applies these gains to ``reference_error = reference -
    state``. They are therefore the negative of the direct state-feedback
    gains returned by :func:`synthesize_lqr`.
    """

    layer_count, state_dimension, _ = dynamics.shape
    identity = torch.eye(state_dimension, dtype=torch.float32)
    problem = FiniteHorizonControlProblem(
        dynamics=dynamics.float(),
        control_channels=identity.unsqueeze(0).repeat(layer_count, 1, 1),
        state_costs=(state_cost * identity).unsqueeze(0).repeat(layer_count, 1, 1),
        control_costs=(control_cost * identity).unsqueeze(0).repeat(layer_count, 1, 1),
        terminal_cost=terminal_cost * identity,
    )
    return -synthesize_lqr(problem, device=device).gains
