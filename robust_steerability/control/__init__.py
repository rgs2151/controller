"""Controller abstractions, synthesis, and numerical contracts."""

from robust_steerability.control.activation_addition import ActivationAdditionController
from robust_steerability.control.base import Controller, SynthesizedController
from robust_steerability.control.h_infinity import (
    HInfinityController,
    HInfinityOptions,
)
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

__all__ = [
    "ActivationAdditionController",
    "Controller",
    "ControllerSolution",
    "FiniteHorizonControlProblem",
    "HInfinityController",
    "HInfinityOptions",
    "LQRController",
    "PIDController",
    "PIDGains",
    "SynthesizedController",
    "closed_loop_disturbance_gain",
    "solve_identity_input_lqr",
]
