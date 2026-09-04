"""Numerical contracts shared by controller implementations.

Controller modules operate only on the tensors defined here. They do not load
language models, tokenize text, collect activations, or register model hooks.

The canonical finite-horizon convention is

    x[k + 1] = A[k] x[k] + B[k] u[k] + D[k] w[k]
    u[k] = K[k] x[k]

so ``ControllerSolution.gains`` are applied directly, without an implicit
minus sign.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class FiniteHorizonControlProblem:
    """A controller-ready, model-agnostic finite-horizon problem."""

    dynamics: torch.Tensor
    control_channels: torch.Tensor
    state_costs: torch.Tensor
    control_costs: torch.Tensor
    terminal_cost: torch.Tensor
    disturbance_channels: torch.Tensor | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        return int(self.dynamics.shape[0])

    @property
    def state_dimension(self) -> int:
        return int(self.dynamics.shape[1])

    @property
    def control_dimension(self) -> int:
        return int(self.control_channels.shape[2])

    @property
    def disturbance_dimension(self) -> int | None:
        if self.disturbance_channels is None:
            return None
        return int(self.disturbance_channels.shape[2])


@dataclass(frozen=True)
class ControllerSolution:
    """Layer-wise feedback gains and synthesis diagnostics."""

    controller: str
    gains: torch.Tensor
    feasible: bool = True
    gamma_star: float | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)
