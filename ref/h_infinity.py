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
    numerical_tolerance: float = 1e-7
    max_gamma: float = 1e12
    deployment_margin: float = 0.0


def _empty_diagnostics() -> dict[str, object]:
    """Return the complete diagnostic surface for one gamma solve."""

    return {
        "min_disturbance_margin": float("inf"),
        "min_control_margin": float("inf"),
        "max_condition_number_M": 0.0,
        "max_condition_number_H": 0.0,
        "max_norm_S": 0.0,
        "max_norm_K": 0.0,
        "failed_layer": None,
        "failure_reason": None,
    }


def _solve_for_gamma(
    problem: FiniteHorizonControlProblem,
    gamma: float,
    *,
    device: str | torch.device,
    numerical_tolerance: float,
) -> tuple[bool, torch.Tensor, list[torch.Tensor], dict[str, object]]:
    """Run the finite-horizon H-infinity recursion for one fixed gamma."""

    dynamics = problem.dynamics.to(device=device, dtype=torch.float32)
    control_channels = problem.control_channels.to(device=device, dtype=torch.float32)
    disturbance_channels = problem.disturbance_channels
    assert disturbance_channels is not None
    disturbance_channels = disturbance_channels.to(device=device, dtype=torch.float32)
    state_costs = problem.state_costs.to(device=device, dtype=torch.float32)
    control_costs = problem.control_costs.to(device=device, dtype=torch.float32)
    value = problem.terminal_cost.to(device=device, dtype=torch.float32)
    gains = torch.zeros(
        problem.horizon,
        problem.control_dimension,
        problem.state_dimension,
        device=device,
        dtype=torch.float32,
    )
    values: list[torch.Tensor] = [value.new_empty(0) for _ in range(problem.horizon)]
    values.append(value)
    diagnostics = _empty_diagnostics()
    diagnostics["max_norm_S"] = float(torch.linalg.matrix_norm(value).item())

    disturbance_identity = torch.eye(
        disturbance_channels.shape[2], device=device, dtype=torch.float32
    )
    for layer_index in reversed(range(problem.horizon)):
        layer_dynamics = dynamics[layer_index]
        control_channel = control_channels[layer_index]
        disturbance_channel = disturbance_channels[layer_index]

        disturbance_hessian = (
            gamma**2 * disturbance_identity
            - disturbance_channel.T @ value @ disturbance_channel
        )
        disturbance_hessian = 0.5 * (
            disturbance_hessian + disturbance_hessian.T
        )
        disturbance_eigenvalues = torch.linalg.eigvalsh(disturbance_hessian)
        disturbance_margin = float(disturbance_eigenvalues[0].item())
        diagnostics["min_disturbance_margin"] = min(
            float(diagnostics["min_disturbance_margin"]), disturbance_margin
        )
        if disturbance_margin <= numerical_tolerance:
            diagnostics["failed_layer"] = layer_index
            diagnostics["failure_reason"] = (
                "disturbance Hessian is not positive definite"
            )
            return False, gains, values, diagnostics

        diagnostics["max_condition_number_M"] = max(
            float(diagnostics["max_condition_number_M"]),
            float(torch.linalg.cond(disturbance_hessian).item()),
        )
        disturbance_response = torch.linalg.solve(
            disturbance_hessian, disturbance_channel.T @ value
        )
        robust_value = value + value @ disturbance_channel @ disturbance_response
        robust_value = 0.5 * (robust_value + robust_value.T)

        control_hessian = (
            control_costs[layer_index]
            + control_channel.T @ robust_value @ control_channel
        )
        control_hessian = 0.5 * (control_hessian + control_hessian.T)
        control_eigenvalues = torch.linalg.eigvalsh(control_hessian)
        control_margin = float(control_eigenvalues[0].item())
        diagnostics["min_control_margin"] = min(
            float(diagnostics["min_control_margin"]), control_margin
        )
        if control_margin <= numerical_tolerance:
            diagnostics["failed_layer"] = layer_index
            diagnostics["failure_reason"] = (
                "control Hessian is not positive definite"
            )
            return False, gains, values, diagnostics

        diagnostics["max_condition_number_H"] = max(
            float(diagnostics["max_condition_number_H"]),
            float(torch.linalg.cond(control_hessian).item()),
        )
        cross_term = control_channel.T @ robust_value @ layer_dynamics
        positive_gain = torch.linalg.solve(control_hessian, cross_term)
        gains[layer_index] = -positive_gain
        value = (
            state_costs[layer_index]
            + layer_dynamics.T @ robust_value @ layer_dynamics
            - cross_term.T @ positive_gain
        )
        value = 0.5 * (value + value.T)
        values[layer_index] = value
        diagnostics["max_norm_S"] = max(
            float(diagnostics["max_norm_S"]),
            float(torch.linalg.matrix_norm(value).item()),
        )
        diagnostics["max_norm_K"] = max(
            float(diagnostics["max_norm_K"]),
            float(torch.linalg.matrix_norm(gains[layer_index]).item()),
        )

    return True, gains, values, diagnostics


class HInfinityController(SynthesizedController):
    """Finite-horizon H-infinity controller.

    Offline synthesis computes the layer gains and attenuation boundary.
    Online, :meth:`control` computes ``u[k] = K[k] x[k]`` and the inherited
    :meth:`intervention` method returns ``B[k] u[k]``.
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
        """Synthesize the smallest feasible finite-horizon attenuation level."""

        validate_control_problem(problem)
        if problem.disturbance_channels is None:
            raise ValueError("H-infinity synthesis requires disturbance_channels")
        if options.gamma_lower < 0.0:
            raise ValueError("gamma_lower must be non-negative")
        if options.gamma_upper <= options.gamma_lower:
            raise ValueError("gamma_upper must be greater than gamma_lower")
        if options.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if options.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if options.numerical_tolerance <= 0.0:
            raise ValueError("numerical_tolerance must be positive")
        if options.max_gamma < options.gamma_upper:
            raise ValueError("max_gamma must be at least gamma_upper")
        if options.deployment_margin < 0.0:
            raise ValueError("deployment_margin must be non-negative")

        target_device = device if device is not None else problem.dynamics.device

        lower = options.gamma_lower
        lower_feasible, _, _, _ = _solve_for_gamma(
            problem,
            lower,
            device=target_device,
            numerical_tolerance=options.numerical_tolerance,
        )
        while lower_feasible and lower > 0.0:
            lower *= 0.5
            lower_feasible, _, _, _ = _solve_for_gamma(
                problem,
                lower,
                device=target_device,
                numerical_tolerance=options.numerical_tolerance,
            )

        upper = options.gamma_upper
        upper_feasible, upper_gains, _, upper_diagnostics = _solve_for_gamma(
            problem,
            upper,
            device=target_device,
            numerical_tolerance=options.numerical_tolerance,
        )
        upper_bound_expansions = 0
        while not upper_feasible and upper < options.max_gamma:
            upper = min(2.0 * upper, options.max_gamma)
            upper_feasible, upper_gains, _, upper_diagnostics = _solve_for_gamma(
                problem,
                upper,
                device=target_device,
                numerical_tolerance=options.numerical_tolerance,
            )
            upper_bound_expansions += 1

        if not upper_feasible:
            diagnostics = dict(upper_diagnostics)
            diagnostics.update(
                {
                    "gamma_star": None,
                    "gamma_used": None,
                    "gamma_lower_final": lower,
                    "gamma_upper_final": upper,
                    "bisection_iterations": 0,
                    "bisection_converged": False,
                    "upper_bound_expansions": upper_bound_expansions,
                    "feedback_convention": "u[k] = K[k] x[k]",
                }
            )
            return cls(
                gains=upper_gains.detach().cpu(),
                control_channels=problem.control_channels.detach().cpu(),
                feasible=False,
                gamma_star=None,
                diagnostics=diagnostics,
            ).to(target_device)

        bisection_iterations = 0
        while (
            upper - lower >= options.tolerance
            and bisection_iterations < options.max_iterations
        ):
            midpoint = 0.5 * (lower + upper)
            midpoint_feasible, midpoint_gains, _, midpoint_diagnostics = (
                _solve_for_gamma(
                    problem,
                    midpoint,
                    device=target_device,
                    numerical_tolerance=options.numerical_tolerance,
                )
            )
            if midpoint_feasible:
                upper = midpoint
                upper_gains = midpoint_gains
                upper_diagnostics = midpoint_diagnostics
            else:
                lower = midpoint
            bisection_iterations += 1

        gamma_star = upper
        gamma_used = (1.0 + options.deployment_margin) * gamma_star
        feasible, gains, _, diagnostics = _solve_for_gamma(
            problem,
            gamma_used,
            device=target_device,
            numerical_tolerance=options.numerical_tolerance,
        )
        diagnostics.update(
            {
                "gamma_star": gamma_star,
                "gamma_used": gamma_used,
                "robust_steerability": 1.0 / gamma_star,
                "gamma_lower_final": lower,
                "gamma_upper_final": upper,
                "bisection_iterations": bisection_iterations,
                "bisection_converged": upper - lower < options.tolerance,
                "upper_bound_expansions": upper_bound_expansions,
                "feedback_convention": "u[k] = K[k] x[k]",
            }
        )
        solution = ControllerSolution(
            controller="h_infinity",
            gains=gains.detach().cpu(),
            feasible=feasible,
            gamma_star=gamma_star,
            diagnostics=diagnostics,
        )
        validate_controller_solution(problem, solution)
        return cls.from_solution(problem, solution).to(target_device)

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
        if solution.feasible and solution.gamma_star is None:
            raise ValueError("feasible H-infinity solution must report gamma_star")
        return cls(
            gains=solution.gains,
            control_channels=problem.control_channels.to(solution.gains.device),
            feasible=solution.feasible,
            gamma_star=solution.gamma_star,
            diagnostics=solution.diagnostics,
        )

    # Online control.

    def reset(self) -> None:
        """Reset online state; finite-horizon state feedback is stateless."""

        return None

    def control(
        self,
        layer_index: int,
        feedback_input: torch.Tensor,
    ) -> torch.Tensor:
        """Return the layer control coordinates for a state deviation."""

        if not self.feasible:
            raise RuntimeError("cannot use an infeasible H-infinity controller")
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
