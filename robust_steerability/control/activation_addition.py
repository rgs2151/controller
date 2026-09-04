"""Static activation-addition baseline."""

from __future__ import annotations

import torch


def activation_delta(direction: torch.Tensor, strength: float) -> torch.Tensor:
    """Return the additive intervention for a direction and scalar strength."""

    return strength * direction
