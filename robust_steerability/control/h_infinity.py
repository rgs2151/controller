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


@dataclass(frozen=True)
class _GammaResult:
    """Internal result of one fixed-gamma backward game recursion."""

    feasible: bool
    gains: torch.Tensor | None
    minimum_disturbance_margin: float
    minimum_control_margin: float
    maximum_disturbance_condition: float
    maximum_control_condition: float
    maximum_value_norm: float
    failure_layer: int | None = None
    failure_reason: str | None = None


def _validate_options(options: HInfinityOptions) -> None:
    if options.gamma_lower < 0.0:
        raise ValueError("gamma_lower must be non-negative")
    if options.gamma_upper <= options.gamma_lower:
        raise ValueError("gamma_upper must be greater than gamma_lower")
    if options.tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if options.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")


def _strict_positive_margin(matrix: torch.Tensor) -> tuple[float, float]:
    """Return minimum eigenvalue and condition number for a symmetric matrix."""

    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    minimum = float(eigenvalues[0].item())
    maximum = float(eigenvalues[-1].item())
    if minimum <= 0.0:
        return minimum, float("inf")
    return minimum, maximum / minimum


def _fixed_gamma_recursion(
    problem: FiniteHorizonControlProblem,
    gamma: float,
    *,
    device: str | torch.device,
) -> _GammaResult:
    """Solve the finite-horizon minimax game for one attenuation level.

    The game is

        min_u max_w sum_k x'Qx + u'Ru - gamma^2 w'w + x_T'Q_f x_T

    subject to ``x[k+1] = A[k]x[k] + B[k]u[k] + D[k]w[k]``.
    Gains use the package convention ``u[k] = K[k] x[k]``.

    At each layer the adversarial disturbance is eliminated first. If
    ``P = S[k+1]`` is the next value matrix, finite worst-case disturbance
    energy requires

        gamma^2 I - D' P D > 0.

    Eliminating the disturbance gives the robustified value matrix

        P_bar = P + P D (gamma^2 I - D' P D)^-1 D' P,

    after which the minimizing control step has the usual LQR form with
    ``P_bar`` in place of ``P``.
    """

    dtype = torch.float64
    dynamics = problem.dynamics.to(device=device, dtype=dtype)
    control_channels = problem.control_channels.to(device=device, dtype=dtype)
    disturbance_channels = problem.disturbance_channels
    if disturbance_channels is None:
        raise ValueError("H-infinity synthesis requires disturbance_channels")
    disturbance_channels = disturbance_channels.to(device=device, dtype=dtype)
    state_costs = problem.state_costs.to(device=device, dtype=dtype)
    control_costs = problem.control_costs.to(device=device, dtype=dtype)
    value = problem.terminal_cost.to(device=device, dtype=dtype)

    gains = torch.empty(
        problem.horizon,
        problem.control_dimension,
        problem.state_dimension,
        device=device,
        dtype=dtype,
    )

    minimum_disturbance_margin = float("inf")
    minimum_control_margin = float("inf")
    maximum_disturbance_condition = 0.0
    maximum_control_condition = 0.0
    maximum_value_norm = float(torch.linalg.matrix_norm(value, ord=2).item())

    disturbance_dimension = disturbance_channels.shape[2]
    disturbance_identity = torch.eye(
        disturbance_dimension,
        device=device,
        dtype=dtype,
    )

    for layer_index in reversed(range(problem.horizon)):
        layer_dynamics = dynamics[layer_index]
        control_channel = control_channels[layer_index]
        disturbance_channel = disturbance_channels[layer_index]

        disturbance_hessian = (
            gamma * gamma * disturbance_identity
            - disturbance_channel.T @ value @ disturbance_channel
        )
        disturbance_margin, disturbance_condition = _strict_positive_margin(
            disturbance_hessian
        )
        minimum_disturbance_margin = min(
            minimum_disturbance_margin,
            disturbance_margin,
        )
        maximum_disturbance_condition = max(
            maximum_disturbance_condition,
            disturbance_condition,
        )
        if disturbance_margin <= 0.0:
            return _GammaResult(
                feasible=False,
                gains=None,
                minimum_disturbance_margin=minimum_disturbance_margin,
                minimum_control_margin=minimum_control_margin,
                maximum_disturbance_condition=maximum_disturbance_condition,
                maximum_control_condition=maximum_control_condition,
                maximum_value_norm=maximum_value_norm,
                failure_layer=layer_index,
                failure_reason="disturbance saddle-point matrix is not positive definite",
            )

        value_times_disturbance = value @ disturbance_channel
        robust_value = value + value_times_disturbance @ torch.linalg.solve(
            disturbance_hessian,
            value_times_disturbance.T,
        )
        robust_value = 0.5 * (robust_value + robust_value.T)

        control_hessian = (
            control_costs[layer_index]
            + control_channel.T @ robust_value @ control_channel
        )
        control_margin, control_condition = _strict_positive_margin(control_hessian)
        minimum_control_margin = min(minimum_control_margin, control_margin)
        maximum_control_condition = max(maximum_control_condition, control_condition)
        if control_margin <= 0.0:
            return _GammaResult(
                feasible=False,
                gains=None,
                minimum_disturbance_margin=minimum_disturbance_margin,
                minimum_control_margin=minimum_control_margin,
                maximum_disturbance_condition=maximum_disturbance_condition,
                maximum_control_condition=maximum_control_condition,
                maximum_value_norm=maximum_value_norm,
                failure_layer=layer_index,
                failure_reason="control minimizing Hessian is not positive definite",
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
        if not torch.isfinite(value).all():
            return _GammaResult(
                feasible=False,
                gains=None,
                minimum_disturbance_margin=minimum_disturbance_margin,
                minimum_control_margin=minimum_control_margin,
                maximum_disturbance_condition=maximum_disturbance_condition,
                maximum_control_condition=maximum_control_condition,
                maximum_value_norm=float("inf"),
                failure_layer=layer_index,
                failure_reason="value recursion produced non-finite entries",
            )
        maximum_value_norm = max(
            maximum_value_norm,
            float(torch.linalg.matrix_norm(value, ord=2).item()),
        )

    return _GammaResult(
        feasible=True,
        gains=gains,
        minimum_disturbance_margin=minimum_disturbance_margin,
        minimum_control_margin=minimum_control_margin,
        maximum_disturbance_condition=maximum_disturbance_condition,
        maximum_control_condition=maximum_control_condition,
        maximum_value_norm=maximum_value_norm,
    )


def _infeasible_solution(
    problem: FiniteHorizonControlProblem,
    *,
    diagnostics: dict[str, object],
) -> ControllerSolution:
    return ControllerSolution(
        controller="h_infinity",
        gains=torch.zeros(
            problem.horizon,
            problem.control_dimension,
            problem.state_dimension,
            dtype=torch.float32,
        ),
        feasible=False,
        gamma_star=None,
        diagnostics=diagnostics,
    )


class HInfinityController(SynthesizedController):
    """Finite-horizon H-infinity controller.

    Offline synthesis computes the smallest feasible disturbance attenuation
    level in the configured interval and the corresponding layer-wise gains.
    Online execution is memoryless full-state feedback: ``u[k] = K[k] x[k]``.
    The inherited :meth:`intervention` method returns ``B[k] u[k]``.
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
        if feasible and gamma_star is None:
            raise ValueError("a feasible H-infinity controller requires gamma_star")
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

        The search returns the smallest feasible ``gamma`` resolved to
        ``options.tolerance`` inside ``[gamma_lower, gamma_upper]``. Feasibility
        is determined by the finite-horizon saddle-point conditions at every
        layer. If the configured upper bound is itself infeasible, synthesis

        returns an explicitly infeasible controller; online use of that object
        raises rather than silently falling back to another controller.
        """

        validate_control_problem(problem)
        if problem.disturbance_channels is None:
            raise ValueError("H-infinity synthesis requires disturbance_channels")
        _validate_options(options)

        target_device = device if device is not None else problem.dynamics.device
        lower = float(options.gamma_lower)
        upper = float(options.gamma_upper)

        upper_result = _fixed_gamma_recursion(
            problem,
            upper,
            device=target_device,
        )
        evaluations = 1
        if not upper_result.feasible:
            diagnostics: dict[str, object] = {
                "feedback_convention": "u[k] = K[k] x[k]",
                "gamma_search_iterations": 0,
                "gamma_evaluations": evaluations,
                "gamma_lower": lower,
                "gamma_upper": upper,
                "gamma_tolerance": options.tolerance,
                "failure_layer": upper_result.failure_layer,
                "failure_reason": upper_result.failure_reason,
                "minimum_disturbance_margin": upper_result.minimum_disturbance_margin,
                "minimum_control_margin": upper_result.minimum_control_margin,
                "maximum_disturbance_condition": upper_result.maximum_disturbance_condition,
                "maximum_control_condition": upper_result.maximum_control_condition,
            }
            solution = _infeasible_solution(problem, diagnostics=diagnostics)
            controller = cls(
                gains=solution.gains,
                control_channels=problem.control_channels.detach().cpu().float(),
                feasible=False,
                gamma_star=None,
                diagnostics=solution.diagnostics,
            )
            return controller.to(target_device)

        lower_result = _fixed_gamma_recursion(
            problem,
            lower,
            device=target_device,
        )
        evaluations += 1

        iterations = 0
        best_result = upper_result
        if lower_result.feasible:
            upper = lower
            best_result = lower_result
        else:
            while (
                upper - lower > options.tolerance
                and iterations < options.max_iterations
            ):
                candidate = 0.5 * (lower + upper)
                candidate_result = _fixed_gamma_recursion(
                    problem,
                    candidate,
                    device=target_device,
                )
                evaluations += 1
                iterations += 1
                if candidate_result.feasible:
                    upper = candidate
                    best_result = candidate_result
                else:
                    lower = candidate

        gamma_star = upper
        if best_result.gains is None:
            raise RuntimeError("feasible H-infinity recursion did not return gains")

        diagnostics = {
            "feedback_convention": "u[k] = K[k] x[k]",
            "gamma_search_iterations": iterations,
            "gamma_evaluations": evaluations,
            "gamma_star_bracket_lower": lower,
            "gamma_star_bracket_upper": upper,
            "gamma_bracket_width": upper - lower,
            "gamma_tolerance": options.tolerance,
            "search_converged": upper - lower <= options.tolerance,
            "lower_bound_feasible": lower_result.feasible,
            "minimum_disturbance_margin": best_result.minimum_disturbance_margin,
            "minimum_control_margin": best_result.minimum_control_margin,
            "maximum_disturbance_condition": best_result.maximum_disturbance_condition,
            "maximum_control_condition": best_result.maximum_control_condition,
            "maximum_value_norm": best_result.maximum_value_norm,
            "synthesis_dtype": "float64",
        }
        solution = ControllerSolution(
            controller="h_infinity",
            gains=best_result.gains.to(dtype=torch.float32, device="cpu"),
            feasible=True,
            gamma_star=gamma_star,
            diagnostics=diagnostics,
        )
        validate_controller_solution(problem, solution)
        controller = cls.from_solution(problem, solution)
        return controller.to(target_device)

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
            raise ValueError("a feasible H-infinity solution must report gamma_star")
        return cls(
            gains=solution.gains,
            control_channels=problem.control_channels.to(
                device=solution.gains.device,
                dtype=solution.gains.dtype,
            ),
            feasible=solution.feasible,
            gamma_star=solution.gamma_star,
            diagnostics=solution.diagnostics,
        )

    # Online control.

    def reset(self) -> None:
        """Reset controller state; finite-horizon H-infinity is memoryless here."""

        return None

    def control(
        self,
        layer_index: int,
        feedback_input: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``u[k] = K[k] x[k]`` for the current state deviation."""

        if not self.feasible:
            raise RuntimeError("cannot execute an infeasible H-infinity controller")
        return feedback_input @ self.gains[layer_index].T

    def intervention(
        self,
        layer_index: int,
        feedback_input: torch.Tensor,
    ) -> torch.Tensor:
        """Return the final activation intervention ``B[k] u[k]``."""

        control = self.control(layer_index, feedback_input)
        return control @ self.control_channels[layer_index].T

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
