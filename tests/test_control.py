from __future__ import annotations

import unittest

import torch

from robust_steerability.control.activation_addition import (
    ActivationAdditionController,
)
from robust_steerability.control.base import Controller
from robust_steerability.control.h_infinity import HInfinityController, HInfinityOptions
from robust_steerability.control.lqr import (
    LQRController,
    solve_identity_input_lqr,
)
from robust_steerability.control.metrics import closed_loop_disturbance_gain
from robust_steerability.control.pid import PIDController, PIDGains
from robust_steerability.control.types import (
    ControllerSolution,
    FiniteHorizonControlProblem,
)
from robust_steerability.runtime.policy import SemanticSetpointPolicy


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


def scalar_problem(*, disturbance: bool = False) -> FiniteHorizonControlProblem:
    return FiniteHorizonControlProblem(
        dynamics=torch.tensor([[[1.0]]]),
        control_channels=torch.tensor([[[2.0]]]),
        disturbance_channels=torch.tensor([[[1.0]]]) if disturbance else None,
        state_costs=torch.tensor([[[1.0]]]),
        control_costs=torch.tensor([[[1.0]]]),
        terminal_cost=torch.tensor([[1.0]]),
    )


class ControllerInterfaceTests(unittest.TestCase):
    def test_base_controller_requires_control_implementation(self) -> None:
        with self.assertRaises(TypeError):
            Controller()

    def test_activation_addition_uses_common_intervention_surface(self) -> None:
        controller = ActivationAdditionController(
            directions=torch.tensor([[1.0, -2.0]]),
            strength=0.5,
        )
        activation = torch.zeros(3, 2)
        expected = torch.tensor([[0.5, -1.0]]).repeat(3, 1)
        torch.testing.assert_close(controller.intervention(0, activation), expected)


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

    def test_synthesized_controller_uses_direct_feedback_and_channel(self) -> None:
        controller = LQRController.synthesize(scalar_problem())
        state_deviation = torch.tensor([[3.0]])
        control = controller.control(0, state_deviation)
        self.assertLess(float(control[0, 0]), 0.0)
        torch.testing.assert_close(
            controller.intervention(0, state_deviation),
            2.0 * control,
        )

    def test_tracking_artifact_policy_preserves_activation_lqr_formula(self) -> None:
        gains = torch.tensor([[[2.0, 0.0], [0.0, 3.0]]])
        feature = torch.tensor([[1.0, 0.0]])
        setpoints = torch.tensor([4.0])
        activation = torch.tensor([[1.5, 7.0]])
        policy = SemanticSetpointPolicy(
            controller=LQRController.from_tracking_gains(gains),
            feature_unit=feature,
            setpoints=setpoints,
        )
        policy.prepare(torch.device("cpu"), torch.float32)
        self.assertEqual(policy.controller.gains.device.type, "cpu")
        self.assertEqual(policy.controller.gains.dtype, torch.float32)
        expected_error = torch.tensor([[2.5, 0.0]])
        expected = expected_error @ gains[0].T
        torch.testing.assert_close(policy.activation_delta(0, activation), expected)


class HInfinityTests(unittest.TestCase):
    def test_synthesize_returns_feasible_controller(self) -> None:
        controller = HInfinityController.synthesize(
            scalar_problem(disturbance=True),
            options=HInfinityOptions(gamma_lower=0.01, gamma_upper=10.0, tolerance=1e-4),
        )
        self.assertTrue(controller.feasible)
        self.assertIsNotNone(controller.gamma_star)
        state_deviation = torch.tensor([[2.0]])
        control = controller.control(0, state_deviation)
        self.assertEqual(tuple(control.shape), (1, 1))

    def test_infeasible_controller_raises_on_control(self) -> None:
        controller = HInfinityController.synthesize(
            scalar_problem(disturbance=True),
            options=HInfinityOptions(gamma_lower=0.01, gamma_upper=0.5, tolerance=1e-4),
        )
        self.assertFalse(controller.feasible)
        self.assertIsNone(controller.gamma_star)
        with self.assertRaises(RuntimeError):
            controller.control(0, torch.tensor([[1.0]]))

    def test_synthesized_solution_has_complete_online_behavior(self) -> None:
        problem = scalar_problem(disturbance=True)
        solution = ControllerSolution(
            controller="h_infinity",
            gains=torch.tensor([[[-2.0]]]),
            feasible=True,
            gamma_star=1.25,
            diagnostics={"iterations": 7},
        )
        controller = HInfinityController.from_solution(problem, solution)
        state_deviation = torch.tensor([[4.0]])
        torch.testing.assert_close(
            controller.control(0, state_deviation),
            torch.tensor([[-8.0]]),
        )
        torch.testing.assert_close(
            controller.intervention(0, state_deviation),
            torch.tensor([[-16.0]]),
        )
        serialized = controller.solution()
        self.assertEqual(serialized.controller, solution.controller)
        torch.testing.assert_close(serialized.gains, solution.gains)
        self.assertEqual(serialized.feasible, solution.feasible)
        self.assertEqual(serialized.gamma_star, solution.gamma_star)
        self.assertEqual(serialized.diagnostics, solution.diagnostics)

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


class PIDTests(unittest.TestCase):
    def test_reset_removes_history(self) -> None:
        controller = PIDController(PIDGains(1.0, 0.5, 0.25))
        state_deviation = torch.tensor([-2.0])
        first = controller.control(0, state_deviation)
        controller.control(1, torch.tensor([-1.0]))
        controller.reset()
        torch.testing.assert_close(controller.control(0, state_deviation), first)


if __name__ == "__main__":
    unittest.main()
