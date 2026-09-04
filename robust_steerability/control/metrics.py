"""Independent numerical checks for synthesized controllers."""

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


def _positive_semidefinite_square_root(matrix: torch.Tensor) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (matrix + matrix.T))
    if float(eigenvalues.min().item()) < -1e-6:
        raise ValueError("performance costs must be positive semidefinite")
    return eigenvectors @ torch.diag(eigenvalues.clamp_min(0).sqrt()) @ eigenvectors.T


def closed_loop_disturbance_gain(
    problem: FiniteHorizonControlProblem,
    solution: ControllerSolution,
) -> float:
    """Compute the exact finite-horizon induced gain for a small problem.

    This explicit operator construction is intended for controller tests and
    independent verification, not full hidden-state-scale experiments.
    Stage cost is ``x[k]^T Q[k] x[k] + u[k]^T R[k] u[k]`` and terminal cost
    is ``x[T]^T Q_terminal x[T]`` with zero initial state.
    """

    validate_control_problem(problem)
    validate_controller_solution(problem, solution)
    if problem.disturbance_channels is None:
        raise ValueError("disturbance gain requires disturbance_channels")

    dtype = torch.float64
    dynamics = problem.dynamics.to(dtype=dtype, device="cpu")
    controls = problem.control_channels.to(dtype=dtype, device="cpu")
    disturbances = problem.disturbance_channels.to(dtype=dtype, device="cpu")
    state_costs = problem.state_costs.to(dtype=dtype, device="cpu")
    control_costs = problem.control_costs.to(dtype=dtype, device="cpu")
    terminal_cost = problem.terminal_cost.to(dtype=dtype, device="cpu")
    gains = solution.gains.to(dtype=dtype, device="cpu")
    disturbance_dimension = disturbances.shape[2]
    input_dimension = problem.horizon * disturbance_dimension
    output_dimension = (
        problem.horizon * (problem.state_dimension + problem.control_dimension)
        + problem.state_dimension
    )
    operator = torch.zeros(output_dimension, input_dimension, dtype=dtype)
    state_roots = [
        _positive_semidefinite_square_root(state_costs[index])
        for index in range(problem.horizon)
    ]
    control_roots = [
        _positive_semidefinite_square_root(control_costs[index])
        for index in range(problem.horizon)
    ]
    terminal_root = _positive_semidefinite_square_root(terminal_cost)

    for input_index in range(input_dimension):
        disturbance_sequence = torch.zeros(
            problem.horizon,
            disturbance_dimension,
            dtype=dtype,
        )
        disturbance_sequence.reshape(-1)[input_index] = 1.0
        state = torch.zeros(problem.state_dimension, dtype=dtype)
        outputs = []
        for layer_index in range(problem.horizon):
            control = gains[layer_index] @ state
            outputs.append(state_roots[layer_index] @ state)
            outputs.append(control_roots[layer_index] @ control)
            state = (
                dynamics[layer_index] @ state
                + controls[layer_index] @ control
                + disturbances[layer_index] @ disturbance_sequence[layer_index]
            )
        outputs.append(terminal_root @ state)
        operator[:, input_index] = torch.cat(outputs)
    return float(torch.linalg.svdvals(operator)[0].item())
