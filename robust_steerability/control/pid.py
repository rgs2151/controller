"""Model-agnostic PID control for vector tracking errors."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from robust_steerability.control.base import Controller


@dataclass(frozen=True)
class PIDGains:
    proportional: float
    integral: float
    derivative: float


class PIDController(Controller):
    """Stateful PID controller over layer-wise state deviations."""

    def __init__(
        self,
        gains: PIDGains,
        control_channels: torch.Tensor | None = None,
    ):
        super().__init__(control_channels)
        if control_channels is not None and (
            control_channels.ndim != 3
            or control_channels.shape[1] != control_channels.shape[2]
        ):
            raise ValueError(
                "PID control_channels must have shape (horizon, state, state)"
            )
        self.gains = gains
        self.integral_error: torch.Tensor | None = None
        self.previous_error: torch.Tensor | None = None

    def reset(self) -> None:
        self.integral_error = None
        self.previous_error = None

    def control(
        self,
        layer_index: int,
        feedback_input: torch.Tensor,
    ) -> torch.Tensor:
        """Return PID control for ``reference - state`` tracking error."""

        del layer_index
        error = -feedback_input
        if self.integral_error is None:
            self.integral_error = torch.zeros_like(error)
            derivative = torch.zeros_like(error)
        else:
            derivative = error - self.previous_error
        self.integral_error = self.integral_error + error
        self.previous_error = error
        return (
            self.gains.proportional * error
            + self.gains.integral * self.integral_error
            + self.gains.derivative * derivative
        )
