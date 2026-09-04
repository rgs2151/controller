"""Abstract interfaces shared by every controller implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import torch

from robust_steerability.control.types import (
    ControllerSolution,
    FiniteHorizonControlProblem,
)


class Controller(torch.nn.Module, ABC):
    """Online controller that maps a layer-wise feedback input to an intervention.

    ``control`` returns control coordinates ``u[k]``. ``intervention`` maps
    those coordinates through ``B[k]`` and returns the vector that the model
    runtime adds to the layer activation. A missing control channel denotes
    identity input, so the control and intervention are the same vector.
    """

    def __init__(self, control_channels: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("control_channels", control_channels)

    def reset(self) -> None:
        """Reset online controller state before an independent rollout."""

        return None

    @abstractmethod
    def control(
        self,
        layer_index: int,
        feedback_input: torch.Tensor,
    ) -> torch.Tensor:
        """Return control coordinates ``u[k]`` for one layer."""

    def intervention(
        self,
        layer_index: int,
        feedback_input: torch.Tensor,
    ) -> torch.Tensor:
        """Return the ready-to-apply activation intervention ``B[k] u[k]``."""

        control = self.control(layer_index, feedback_input)
        if self.control_channels is None:
            return control
        channel = self.control_channels[layer_index]
        return control @ channel.T


class SynthesizedController(Controller, ABC):
    """Controller constructed offline from a finite-horizon problem."""

    @classmethod
    @abstractmethod
    def synthesize(
        cls,
        problem: FiniteHorizonControlProblem,
        **kwargs: object,
    ) -> Self:
        """Run offline synthesis and return a ready-to-use controller."""

    @abstractmethod
    def solution(self) -> ControllerSolution:
        """Return the controller's serializable synthesis result."""
