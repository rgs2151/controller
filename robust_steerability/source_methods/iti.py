"""Inference-Time Intervention fitting and attention-head intervention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from robust_steerability.modeling.interventions import _decoder_layers


@dataclass(frozen=True)
class ITIFit:
    head_order: tuple[tuple[int, int], ...]
    directions: torch.Tensor
    validation_accuracy: torch.Tensor

    def selected_heads(self, top_heads: int) -> tuple[tuple[int, int], ...]:
        if top_heads < 1:
            raise ValueError("top_heads must be positive")
        return self.head_order[:top_heads]


def fit_iti(
    activations: torch.Tensor,
    labels: np.ndarray,
    *,
    seed: int = 42,
) -> ITIFit:
    """Fit and rank every per-head probe once, as in honest_llama."""

    if activations.ndim != 4:
        raise ValueError("activations must have shape (samples, layers, heads, head_dim)")
    if len(activations) != len(labels) or set(np.asarray(labels).tolist()) != {0, 1}:
        raise ValueError("binary labels must match activations")
    x_train, x_validation, y_train, y_validation = train_test_split(
        activations.numpy(), np.asarray(labels), test_size=0.2, random_state=seed, stratify=labels
    )
    layer_count, head_count = activations.shape[1:3]
    accuracies = []
    head_directions = []
    for layer_index in range(layer_count):
        for head_index in range(head_count):
            probe = LogisticRegression(random_state=seed, max_iter=1000).fit(
                x_train[:, layer_index, head_index], y_train
            )
            accuracies.append(float(probe.score(x_validation[:, layer_index, head_index], y_validation)))
            direction = probe.coef_.reshape(-1)
            direction /= np.linalg.norm(direction)
            scale = np.std(x_train[:, layer_index, head_index] @ direction)
            head_directions.append(torch.from_numpy(direction * scale).float())
    order = np.argsort(np.asarray(accuracies))[::-1]
    ranked = tuple((int(index) // head_count, int(index) % head_count) for index in order)
    dense = torch.zeros(layer_count, head_count, activations.shape[-1])
    for flat_index in range(layer_count * head_count):
        dense[int(flat_index) // head_count, int(flat_index) % head_count] = head_directions[int(flat_index)]
    return ITIFit(ranked, dense, torch.tensor(accuracies).reshape(layer_count, head_count))


def register_iti_hooks(
    model,
    fitted: ITIFit,
    *,
    top_heads: int,
    alpha: float,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Add selected directions at each attention output projection input, last token only."""

    layers = _decoder_layers(model)
    if len(layers) != fitted.directions.shape[0]:
        raise ValueError("ITI fit layer count does not match model")
    selected = set(fitted.selected_heads(top_heads))
    handles = []
    for layer_index, layer in enumerate(layers):
        direction = fitted.directions[layer_index].clone()
        for head_index in range(direction.shape[0]):
            if (layer_index, head_index) not in selected:
                direction[head_index].zero_()
        direction = direction.reshape(-1)
        if not torch.any(direction):
            continue
        projection = layer.self_attn.o_proj

        def make_hook(value):
            def hook(_module, inputs):
                changed = inputs[0].clone()
                changed[:, -1, :] -= alpha * value.to(changed.device, changed.dtype)
                return (changed,) + inputs[1:]
            return hook

        handles.append(projection.register_forward_pre_hook(make_hook(direction)))
    return handles
