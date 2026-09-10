"""Exact ODESteer degree-two normalized count-sketch operator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class ODESteerFit:
    indices: torch.Tensor
    signs: torch.Tensor
    coefficient: torch.Tensor
    intercept: torch.Tensor
    gamma: float = 0.1
    coef0: float = 1.0
    steps: int = 10

    @property
    def components(self) -> int:
        return self.coefficient.numel()


def polynomial_features(values: torch.Tensor, fitted: ODESteerFit) -> torch.Tensor:
    normalized = values / (values.norm(dim=-1, keepdim=True) + 1e-12)
    extended = torch.cat([
        normalized * fitted.gamma ** 0.5,
        normalized.new_full((len(normalized), 1), fitted.coef0 ** 0.5),
    ], dim=1)
    sketches = []
    indices = fitted.indices.to(values.device)
    signs = fitted.signs.to(values.device)
    for degree_index in range(fitted.indices.shape[0]):
        sketch = extended.new_zeros(len(extended), fitted.components)
        sketch.scatter_add_(
            1,
            indices[degree_index].unsqueeze(0).expand(len(extended), -1),
            extended * signs[degree_index].to(extended.dtype),
        )
        sketches.append(sketch)
    spectra = torch.fft.rfft(torch.stack(sketches, dim=1), dim=-1)
    return torch.fft.irfft(spectra.prod(dim=1), n=fitted.components, dim=-1)


def vector_field(values: torch.Tensor, fitted: ODESteerFit) -> torch.Tensor:
    """Return the normalized analytical gradient of the source classifier logit."""

    normalized = values / (values.norm(dim=-1, keepdim=True) + 1e-12)
    extended = torch.cat([
        normalized * fitted.gamma ** 0.5,
        normalized.new_full((len(normalized), 1), fitted.coef0 ** 0.5),
    ], dim=1)
    indices = fitted.indices.to(values.device)
    signs = fitted.signs.to(values.device)
    degree, components = indices.shape[0], fitted.components
    sketches = []
    for degree_index in range(degree):
        sketch = extended.new_zeros(len(extended), components)
        sketch.scatter_add_(
            1,
            indices[degree_index].unsqueeze(0).expand(len(extended), -1),
            extended * signs[degree_index].to(extended.dtype),
        )
        sketches.append(sketch)
    spectra = torch.fft.rfft(torch.stack(sketches, dim=1), dim=-1)
    prefix = torch.empty_like(spectra)
    suffix = torch.empty_like(spectra)
    prefix[:, 0] = 1
    suffix[:, -1] = 1
    if degree > 1:
        prefix[:, 1:] = torch.cumprod(spectra[:, :-1], dim=1)
        suffix[:, :-1] = torch.flip(
            torch.cumprod(torch.flip(spectra[:, 1:], dims=[1]), dim=1),
            dims=[1],
        )
    other_products = prefix * suffix
    other_sketches = torch.fft.irfft(other_products, n=components, dim=-1).to(values.dtype)
    coefficient_spectrum = torch.fft.rfft(fitted.coefficient.to(values.device, values.dtype))
    correlations = torch.fft.irfft(
        torch.conj(torch.fft.rfft(other_sketches, dim=-1))
        * coefficient_spectrum.view(1, 1, -1),
        n=components,
        dim=-1,
    ).real
    feature_count = values.shape[-1]
    indices = indices[:, :feature_count]
    signs = signs[:, :feature_count].to(values.dtype)
    gathered = correlations.gather(
        2, indices.unsqueeze(0).expand(len(values), -1, -1)
    )
    gradient_normalized = (gathered * signs.unsqueeze(0)).sum(dim=1)
    projection = (normalized * gradient_normalized).sum(dim=1, keepdim=True)
    gradient = (gradient_normalized - normalized * projection) / (
        values.norm(dim=-1, keepdim=True) + 1e-12
    )
    return gradient / (gradient.norm(dim=-1, keepdim=True) + 1e-10)


def fit_odesteer(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    components: int = 8000,
    degree: int = 2,
    gamma: float = 0.1,
    coef0: float = 1.0,
    steps: int = 10,
) -> ODESteerFit:
    """Fit using the source run's current global PyTorch RNG state."""

    if positive.ndim != 2 or positive.shape[1:] != negative.shape[1:]:
        raise ValueError("positive and negative activations must have shape (samples, hidden)")
    indices = torch.randint(
        components,
        (degree, positive.shape[1] + (1 if coef0 != 0 else 0)),
        device=positive.device,
    )
    signs = torch.randint(
        2,
        indices.shape,
        device=positive.device,
        dtype=torch.int8,
    ) * 2 - 1
    empty = ODESteerFit(indices, signs, torch.empty(components, device=positive.device), torch.empty(1), gamma, coef0, steps)
    values = torch.cat([positive, negative]).float()
    labels = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
    features = polynomial_features(values, empty)
    classifier = LogisticRegression(max_iter=1000).fit(features.cpu().numpy(), labels)
    return ODESteerFit(
        indices.cpu(),
        signs.cpu(),
        torch.from_numpy(classifier.coef_.ravel()).to(values.dtype),
        torch.from_numpy(classifier.intercept_.ravel()).to(values.dtype),
        gamma,
        coef0,
        steps,
    )


def steer(values: torch.Tensor, fitted: ODESteerFit, *, time: float) -> torch.Tensor:
    """Integrate the source autonomous field with its ten-step Euler solver."""

    if time == 0:
        return values
    state = values
    step_size = time / fitted.steps
    for _ in range(fitted.steps):
        state = state + step_size * vector_field(state, fitted)
    return state


def register_odesteer_hook(model, fitted: ODESteerFit, *, layer_index: int, time: float):
    """Apply ODESteer at one block output and at the last token only."""

    from robust_steerability.modeling.interventions import _decoder_layers

    layers = _decoder_layers(model)
    if not 0 <= layer_index < len(layers):
        raise ValueError(f"layer_index must be in [0, {len(layers) - 1}]")

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        changed = hidden.clone()
        changed[:, -1, :] = steer(changed[:, -1, :].float(), fitted, time=time).to(hidden.dtype)
        if isinstance(output, tuple):
            return (changed,) + output[1:]
        return changed

    return [layers[layer_index].register_forward_hook(hook)]
