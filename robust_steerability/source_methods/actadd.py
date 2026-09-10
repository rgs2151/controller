"""Activation Addition as implemented by the A-LQR comparison code."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from robust_steerability.modeling.interventions import _decoder_layers


def fit_actadd_direction(negative_mean: torch.Tensor, positive_mean: torch.Tensor) -> torch.Tensor:
    """Return the adapter's positive-minus-negative sequence-valued direction."""

    if (
        negative_mean.ndim != 3
        or positive_mean.ndim != 3
        or negative_mean.shape[0] != positive_mean.shape[0]
        or negative_mean.shape[2] != positive_mean.shape[2]
    ):
        raise ValueError("ActAdd means must match on layers and hidden size")
    maximum = max(negative_mean.shape[1], positive_mean.shape[1])
    if negative_mean.shape[1] < maximum:
        negative_mean = torch.nn.functional.pad(
            negative_mean, (0, 0, 0, maximum - negative_mean.shape[1], 0, 0)
        )
    if positive_mean.shape[1] < maximum:
        positive_mean = torch.nn.functional.pad(
            positive_mean, (0, 0, 0, maximum - positive_mean.shape[1], 0, 0)
        )
    return positive_mean - negative_mean


def positionwise_mean(activations: list[torch.Tensor], attention_masks: list[torch.Tensor]) -> torch.Tensor:
    """Average each token position using the adapter's padding-mask rule."""

    if not activations or len(activations) != len(attention_masks):
        raise ValueError("matching activation and mask batches are required")
    layer_count = activations[0].shape[0]
    hidden_size = activations[0].shape[-1]
    maximum = max(batch.shape[2] for batch in activations)
    total = activations[0].new_zeros(layer_count, maximum, hidden_size)
    count = activations[0].new_zeros(maximum)
    for batch, mask in zip(activations, attention_masks, strict=True):
        if batch.ndim != 4 or batch.shape[0] != layer_count or batch.shape[-1] != hidden_size:
            raise ValueError("activations must have shape (layers, batch, sequence, hidden)")
        length = batch.shape[2]
        selected = mask[:, :length].to(device=batch.device, dtype=batch.dtype)
        total[:, :length] += (batch * selected[None, :, :, None]).sum(dim=1)
        count[:length] += selected.sum(dim=0)
    return total / count.clamp_min(1.0)[None, :, None]


def collect_positionwise_mean(model, tokenizer, texts: list[str], *, batch_size: int) -> torch.Tensor:
    """Collect the exact sequence-valued ActAdd calibration statistic."""

    if not texts:
        raise ValueError("ActAdd calibration requires prompts")
    if tokenizer.padding_side != "left":
        raise ValueError("the source ActAdd adapter requires left padding")
    layers = _decoder_layers(model)
    device = next(model.parameters()).device
    activation_batches = []
    masks = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start:start + batch_size], return_tensors="pt", padding=True, truncation=True
        ).to(device)
        captured: list[torch.Tensor | None] = [None] * len(layers)
        handles = []

        def make_hook(layer_index):
            def hook(_module, inputs):
                captured[layer_index] = inputs[0].detach()
            return hook

        for layer_index, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(make_hook(layer_index)))
        with torch.inference_mode():
            model(**encoded, use_cache=False, return_dict=True)
        for handle in handles:
            handle.remove()
        if any(value is None for value in captured):
            raise RuntimeError("failed to capture every ActAdd layer input")
        activation_batches.append(torch.stack([value for value in captured if value is not None]).cpu())
        masks.append(encoded["attention_mask"].cpu())
    return positionwise_mean(activation_batches, masks)


@dataclass
class ActAddSteerer:
    """Add one position-wise contrast at one layer on the first model call."""

    direction: torch.Tensor
    layer_index: int
    strength: float

    def __post_init__(self) -> None:
        if self.direction.ndim != 2:
            raise ValueError("direction must have shape (sequence, hidden)")
        self._applied = False

    def reset(self) -> None:
        self._applied = False

    def register(self, model) -> list[torch.utils.hooks.RemovableHandle]:
        layers = _decoder_layers(model)
        if not 0 <= self.layer_index < len(layers):
            raise ValueError(f"layer_index must be in [0, {len(layers) - 1}]")

        def hook(_module, inputs):
            if self._applied:
                return inputs
            hidden = inputs[0]
            apply_length = min(hidden.shape[-2], self.direction.shape[0])
            changed = hidden.clone()
            changed[..., :apply_length, :] += (
                self.strength * self.direction[:apply_length].to(hidden.device, hidden.dtype)
            )
            self._applied = True
            return (changed,) + inputs[1:]

        return [layers[self.layer_index].register_forward_pre_hook(hook)]
