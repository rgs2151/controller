"""Mean-AcT, Linear-AcT, and PID-AcT operators from their source code."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import torch


@dataclass(frozen=True)
class MeanTransport:
    source_mean: torch.Tensor
    target_mean: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True)
class LinearTransport:
    slope: torch.Tensor
    intercept: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor


@dataclass(frozen=True)
class PIDTransport:
    difference: torch.Tensor
    mask: torch.Tensor


def fit_mean_transport(responses: torch.Tensor, source_labels: torch.Tensor) -> MeanTransport:
    """Fit source-to-target mean transport in float64, as in ml-act."""

    values = responses.to(torch.float64)
    labels = source_labels.to(torch.bool)
    source, target = values[labels], values[~labels]
    if len(source) == 0 or len(source) != len(target):
        raise ValueError("source and target response sets must be nonempty and balanced")
    source_mean, target_mean = source.mean(dim=0), target.mean(dim=0)
    mask = (source.std(dim=0) > 1e-4) & (target.std(dim=0) > 1e-4)
    return MeanTransport(source_mean.float(), target_mean.float(), mask)


def fit_linear_transport(
    responses: torch.Tensor,
    source_labels: torch.Tensor,
    *,
    random_state: np.random.RandomState,
) -> LinearTransport:
    """Fit ml-act's sorted one-dimensional OLS map and q_0_100 mask."""

    values = responses.float()
    labels = source_labels.to(torch.bool)
    source = values[labels].sort(dim=0).values.numpy().astype(np.float64)
    target = values[~labels].sort(dim=0).values.numpy().astype(np.float64)
    if len(source) == 0 or len(source) != len(target):
        raise ValueError("source and target response sets must be nonempty and balanced")
    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    noisy_source = source + 1e-8 * random_state.randn(*source.shape)
    centered_source = noisy_source - source_mean
    centered_target = target - target_mean
    slope = (centered_source * centered_target).sum(axis=0) / (centered_source**2).sum(axis=0)
    intercept = target_mean.squeeze(0) - slope * source_mean.squeeze(0)
    return LinearTransport(
        torch.from_numpy(slope).float(),
        torch.from_numpy(intercept).float(),
        torch.from_numpy(source.min(axis=0)).float(),
        torch.from_numpy(source.max(axis=0)).float(),
    )


def fit_pid_transport(
    responses: torch.Tensor,
    source_labels: torch.Tensor,
    previous_differences: tuple[torch.Tensor, ...],
) -> PIDTransport:
    """Fit the published PID-AcT mean shift, including its 0.005 history term."""

    fitted = fit_mean_transport(responses, source_labels)
    difference = fitted.target_mean - fitted.source_mean
    if previous_differences:
        history = torch.cat(previous_differences).mean(dim=0)
        difference = difference + 0.005 * (history + difference)
    return PIDTransport(difference, fitted.mask)


def apply_mean_transport(output: torch.Tensor, fitted: MeanTransport, strength: float) -> torch.Tensor:
    changed = output.float().clone()
    mask = fitted.mask.to(output.device)
    delta = (fitted.target_mean - fitted.source_mean).to(output.device)
    flat = changed.reshape(-1, changed.shape[-1])
    selected = mask.unsqueeze(0) & (flat > -1e6) & (flat < 1e6)
    flat[:] = torch.where(selected, flat + strength * delta, flat)
    return changed.to(output.dtype)


def apply_linear_transport(output: torch.Tensor, fitted: LinearTransport, strength: float) -> torch.Tensor:
    changed = output.float().clone()
    flat = changed.reshape(-1, changed.shape[-1])
    slope = fitted.slope.to(output.device)
    intercept = fitted.intercept.to(output.device)
    lower, upper = fitted.lower.to(output.device), fitted.upper.to(output.device)
    transported = flat * slope + intercept
    selected = (lower < flat) & (flat < upper)
    flat[:] = torch.where(selected, strength * transported + (1 - strength) * flat, flat)
    return changed.to(output.dtype)


def apply_pid_transport(output: torch.Tensor, fitted: PIDTransport, strength: float) -> torch.Tensor:
    changed = output.float().clone()
    mask = fitted.mask.to(output.device)
    flat = changed.reshape(-1, changed.shape[-1])
    delta = 0.7 * strength * fitted.difference.to(output.device)
    selected = mask.unsqueeze(0) & (flat > -1e6) & (flat < 1e6)
    flat[:] = torch.where(selected, flat + delta, flat)
    return changed.to(output.dtype)


def matching_module_names(model: torch.nn.Module, patterns: tuple[str, ...]) -> tuple[str, ...]:
    """Use ml-act's full regular-expression matching in named-module order."""

    return tuple(
        name for name, _module in model.named_modules()
        if name and any(re.fullmatch(pattern, name) is not None for pattern in patterns)
    )


def register_transport_hooks(
    model: torch.nn.Module,
    fits: dict[str, MeanTransport | LinearTransport | PIDTransport],
    *,
    strength: float,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Apply each fitted AcT operator to all output tokens of its source module."""

    modules = dict(model.named_modules())
    if tuple(fits) != tuple(name for name in modules if name in fits):
        raise ValueError("transport fits must follow model.named_modules() order")
    handles = []
    for name, fitted in fits.items():
        if name not in modules:
            raise ValueError(f"Missing fitted module {name!r}")

        def make_hook(parameters):
            def hook(_module, _inputs, output):
                if not torch.is_tensor(output):
                    raise ValueError("AcT source operators require tensor module outputs")
                if isinstance(parameters, MeanTransport):
                    return apply_mean_transport(output, parameters, strength)
                if isinstance(parameters, LinearTransport):
                    return apply_linear_transport(output, parameters, strength)
                return apply_pid_transport(output, parameters, strength)
            return hook

        handles.append(modules[name].register_forward_hook(make_hook(fitted)))
    return handles
