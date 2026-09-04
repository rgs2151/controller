"""Nominal layer-dynamics aggregation."""

from __future__ import annotations

import torch


def mean_layer_dynamics(jacobians: torch.Tensor) -> torch.Tensor:
    """Average prompt-wise layer Jacobians with shape (records, layers, d, d)."""

    if jacobians.ndim != 4:
        raise ValueError("jacobians must have shape (records, layers, state, state)")
    return jacobians.float().mean(dim=0)
