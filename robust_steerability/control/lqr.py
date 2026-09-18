"""Offline synthesis and online execution for finite-horizon LQR."""

from __future__ import annotations

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


def _solve_lqr(
    problem: FiniteHorizonControlProblem,
    device: str | torch.device | None = None,
) -> ControllerSolution:
    """Solve finite-horizon LQR for direct gains ``u[k] = K[k] x[k]``."""

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


class LQRController(SynthesizedController):
    """Finite-horizon LQR controller with offline and online behavior.

    Online inputs are state deviations ``x[k] = state[k] - reference[k]``.
    ``control`` returns ``u[k] = K[k] x[k]`` and the inherited
    ``intervention`` method returns ``B[k] u[k]``.
    """

    def __init__(
        self,
        gains: torch.Tensor,
        control_channels: torch.Tensor | None = None,
        *,
        feasible: bool = True,
        diagnostics: dict[str, object] | None = None,
    ):
        super().__init__(control_channels)
        if gains.ndim != 3:
            raise ValueError("gains must have shape (horizon, control, state)")
        if control_channels is not None and (
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
        self.diagnostics = diagnostics or {}

    # Offline synthesis and construction.

    @classmethod
    def synthesize(
        cls,
        problem: FiniteHorizonControlProblem,
        *,
        device: str | torch.device | None = None,
    ) -> LQRController:
        """Run the offline Riccati recursion and return a ready controller."""

        solution = _solve_lqr(problem, device=device)
        controller = cls(
            gains=solution.gains,
            control_channels=problem.control_channels.detach().cpu(),
            feasible=solution.feasible,
            diagnostics=solution.diagnostics,
        )
        target_device = device if device is not None else problem.dynamics.device
        return controller.to(target_device)

    @classmethod
    def from_solution(
        cls,
        problem: FiniteHorizonControlProblem,
        solution: ControllerSolution,
    ) -> LQRController:
        """Construct online LQR behavior from an already synthesized solution."""

        validate_control_problem(problem)
        validate_controller_solution(problem, solution)
        return cls(
            gains=solution.gains,
            control_channels=problem.control_channels.to(solution.gains.device),
            feasible=solution.feasible,
            diagnostics=solution.diagnostics,
        )

    @classmethod
    def from_tracking_gains(
        cls,
        tracking_gains: torch.Tensor,
        control_channels: torch.Tensor | None = None,
    ) -> LQRController:
        """Load positive gains stored in existing Activation-LQR artifacts.

        Historical artifacts multiply their gains by ``reference - state``.
        The controller interface instead receives ``state - reference``, so
        those stored gains are negated once at construction.
        """

        return cls(-tracking_gains, control_channels)

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
        """Return a serializable copy of the synthesized LQR result."""

        return ControllerSolution(
            controller="lqr",
            gains=self.gains.detach().cpu(),
            feasible=self.feasible,
            diagnostics=dict(self.diagnostics),
        )


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
    gains used by :class:`LQRController`.
    """

    if dynamics.ndim != 3 or dynamics.shape[1] != dynamics.shape[2]:
        raise ValueError("Expected square layer dynamics")
    layer_count, state_dimension, _ = dynamics.shape
    if state_cost < 0 or control_cost <= 0 or terminal_cost < 0:
        raise ValueError("LQR requires nonnegative state costs and a positive control cost")
    identity = torch.eye(state_dimension, dtype=torch.float32, device=device)
    value = terminal_cost * identity
    gains = torch.empty(layer_count, state_dimension, state_dimension, dtype=torch.float32)
    # Stream one full-order layer at a time; do not allocate horizon-sized B/Q/R.
    for layer_index in reversed(range(layer_count)):
        a = dynamics[layer_index].to(device=device, dtype=torch.float32)
        cross = value @ a
        gain = torch.linalg.solve(value + control_cost * identity, cross)
        value = state_cost * identity + a.T @ cross - cross.T @ gain
        value = 0.5 * (value + value.T)
        if not torch.isfinite(gain).all() or not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite LQR recursion at layer {layer_index}")
        gains[layer_index] = gain.cpu()
    return gains
