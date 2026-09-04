"""Static activation-addition baseline."""

from __future__ import annotations

import torch

from robust_steerability.control.base import Controller


class ActivationAdditionController(Controller):
    """Fixed layer-wise activation addition under the controller interface."""

    def __init__(self, directions: torch.Tensor, strength: float):
        super().__init__()
        if directions.ndim != 2:
            raise ValueError("directions must have shape (horizon, state)")
        if not torch.isfinite(directions).all():
            raise ValueError("directions contain non-finite values")
        self.register_buffer("directions", directions)
        self.register_buffer(
            "strength",
            torch.as_tensor(strength, dtype=directions.dtype),
        )

    def control(
        self,
        layer_index: int,
        feedback_input: torch.Tensor,
    ) -> torch.Tensor:
        """Return the fixed intervention, ignoring the feedback value."""

        return (self.strength * self.directions[layer_index]).expand_as(feedback_input)
