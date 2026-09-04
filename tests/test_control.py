from __future__ import annotations

import unittest

import torch

from robust_steerability.control.h_infinity import synthesize_h_infinity
from robust_steerability.control.lqr import (
    solve_identity_input_lqr,
    synthesize_lqr,
)
from robust_steerability.control.metrics import closed_loop_disturbance_gain
from robust_steerability.control.pid import PIDController, PIDGains
from robust_steerability.control.types import (
    ControllerSolution,
    FiniteHorizonControlProblem,
)
from robust_steerability.runtime.policy import SetpointLQRPolicy


def legacy_identity_lqr(
    dynamics: torch.Tensor,
    state_cost: float,
    control_cost: float,
    terminal_cost: float,
) -> torch.Tensor:
    layer_count, dimension, _ = dynamics.shape
    identity = torch.eye(dimension)
    value = terminal_cost * identity
    gains = torch.empty_like(dynamics)
    for layer_index in reversed(range(layer_count)):
        layer_dynamics = dynamics[layer_index]
        lhs = value + control_cost * identity
        rhs = value @ layer_dynamics
        gain = torch.linalg.solve(lhs, rhs)
        next_value = (
            state_cost * identity
            + layer_dynamics.T @ value @ layer_dynamics
            - rhs.T @ gain
        )
        value = 0.5 * (next_value + next_value.T)
        gains[layer_index] = gain
    return gains


class LQRTests(unittest.TestCase):
    def test_identity_input_solver_matches_residual_unit_recursion(self) -> None:
        dynamics = torch.tensor(
            [
                [[0.8, 0.1], [0.0, 0.7]],
                [[0.9, 0.0], [0.2, 0.6]],
                [[0.7, -0.1], [0.1, 0.8]],
            ],
            dtype=torch.float32,
        )
        expected = legacy_identity_lqr(dynamics, 0.1, 1.0, 1.0)
        actual = solve_identity_input_lqr(dynamics, "cpu", 0.1, 1.0, 1.0)
        torch.testing.assert_close(actual, expected)

    def test_general_solver_uses_direct_feedback_sign(self) -> None:
        dynamics = torch.tensor([[[1.0]]])
        channels = torch.tensor([[[1.0]]])
        problem = FiniteHorizonControlProblem(
            dynamics=dynamics,
            control_channels=channels,
            state_costs=torch.tensor([[[1.0]]]),
            control_costs=torch.tensor([[[1.0]]]),
            terminal_cost=torch.tensor([[1.0]]),
        )
        solution = synthesize_lqr(problem)
        self.assertLess(float(solution.gains[0, 0, 0]), 0.0)

    def test_h_infinity_extension_point_requires_implementation(self) -> None:
        problem = FiniteHorizonControlProblem(
            dynamics=torch.tensor([[[1.0]]]),
            control_channels=torch.tensor([[[1.0]]]),
            disturbance_channels=torch.tensor([[[1.0]]]),
            state_costs=torch.tensor([[[1.0]]]),
            control_costs=torch.tensor([[[1.0]]]),
            terminal_cost=torch.tensor([[1.0]]),
        )
        with self.assertRaises(NotImplementedError):
            synthesize_h_infinity(problem)

    def test_independent_disturbance_gain(self) -> None:
        problem = FiniteHorizonControlProblem(
            dynamics=torch.tensor([[[0.0]]]),
            control_channels=torch.tensor([[[0.0]]]),
            disturbance_channels=torch.tensor([[[2.0]]]),
            state_costs=torch.tensor([[[0.0]]]),
            control_costs=torch.tensor([[[1.0]]]),
            terminal_cost=torch.tensor([[1.0]]),
        )
        solution = ControllerSolution(controller="zero", gains=torch.zeros(1, 1, 1))
        self.assertAlmostEqual(closed_loop_disturbance_gain(problem, solution), 2.0)


class PolicyTests(unittest.TestCase):
    def test_setpoint_policy_matches_activation_lqr_formula(self) -> None:
        gains = torch.tensor([[[2.0, 0.0], [0.0, 3.0]]])
        feature = torch.tensor([[1.0, 0.0]])
        setpoints = torch.tensor([4.0])
        activation = torch.tensor([[1.5, 7.0]])
        policy = SetpointLQRPolicy(gains, feature, setpoints)
        expected_error = torch.tensor([[2.5, 0.0]])
        expected = expected_error @ gains[0].T
        torch.testing.assert_close(policy.activation_delta(0, activation), expected)

    def test_pid_reset_removes_history(self) -> None:
        controller = PIDController(PIDGains(1.0, 0.5, 0.25))
        error = torch.tensor([2.0])
        first = controller.control(error)
        controller.control(torch.tensor([1.0]))
        controller.reset()
        torch.testing.assert_close(controller.control(error), first)


if __name__ == "__main__":
    unittest.main()
