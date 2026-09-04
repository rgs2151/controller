"""Model-agnostic PID control for vector tracking errors."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PIDGains:
    proportional: float
    integral: float
    derivative: float


class PIDController:
    """Stateful PID controller over an arbitrary tensor-valued error signal."""

    def __init__(self, gains: PIDGains):
        self.gains = gains
        self.integral_error: torch.Tensor | None = None
        self.previous_error: torch.Tensor | None = None

    def reset(self) -> None:
        self.integral_error = None
        self.previous_error = None

    def control(self, error: torch.Tensor) -> torch.Tensor:
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
