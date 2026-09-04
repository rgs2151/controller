"""Controller synthesis and controller-only numerical contracts."""

from robust_steerability.control.activation_addition import activation_delta
from robust_steerability.control.h_infinity import (
    HInfinityOptions,
    synthesize_h_infinity,
)
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

__all__ = [
    "ControllerSolution",
    "FiniteHorizonControlProblem",
    "HInfinityOptions",
    "PIDController",
    "PIDGains",
    "activation_delta",
    "closed_loop_disturbance_gain",
    "solve_identity_input_lqr",
    "synthesize_h_infinity",
    "synthesize_lqr",
]
