"""Generic transformer hooks for activation policies."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM

from robust_steerability.runtime.policy import ActivationPolicy


def _model_device_and_dtype(
    model: AutoModelForCausalLM,
) -> tuple[torch.device, torch.dtype]:
    """Locate the device and floating dtype used by the first decoder layer."""

    first_layer = model.model.layers[0]
    tensors = list(first_layer.parameters()) + list(first_layer.buffers())
    if not tensors:
        raise ValueError("the first decoder layer has no parameters or buffers")
    device = tensors[0].device
    model_dtype = getattr(model, "dtype", None)
    if isinstance(model_dtype, torch.dtype) and model_dtype.is_floating_point:
        dtype = model_dtype
    else:
        dtype = next(
            (tensor.dtype for tensor in tensors if tensor.is_floating_point()),
            torch.float32,
        )
    return device, dtype


def forward_with_policy(
    model: AutoModelForCausalLM,
    encoded: dict[str, torch.Tensor],
    policy: ActivationPolicy | None,
    steer_start_index: int | None = None,
) -> tuple[object, torch.Tensor, torch.Tensor]:
    """Run a forward pass while recording raw states and applied deltas."""

    layer_count = len(model.model.layers)
    sequence_length = int(encoded["input_ids"].shape[1])
    hidden_size = model.config.hidden_size
    controls = torch.zeros(layer_count, sequence_length, hidden_size, dtype=torch.float32)
    states: list[torch.Tensor | None] = [None] * (layer_count + 1)
    handles = []

    def make_input_hook(layer_index: int):
        def hook(_module, args):
            states[layer_index] = args[0][0].detach().cpu().float()

        return hook

    for layer_index, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_pre_hook(make_input_hook(layer_index)))

    if policy is not None:
        device, dtype = _model_device_and_dtype(model)
        policy.prepare(device, dtype)
        policy.reset()

        def make_policy_hook(layer_index: int):
            def hook(_module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                changed = hidden.clone()
                if steer_start_index is None:
                    activation = args[0][:, -1, :]
                    delta = policy.activation_delta(layer_index, activation)
                    changed[:, -1, :] = changed[:, -1, :] + delta.to(changed.dtype)
                    controls[layer_index, -1, :] = delta[0].detach().cpu().float()
                else:
                    activation = args[0][:, steer_start_index:, :]
                    delta = policy.activation_delta(layer_index, activation)
                    changed[:, steer_start_index:, :] = (
                        changed[:, steer_start_index:, :] + delta.to(changed.dtype)
                    )
                    controls[layer_index, steer_start_index:, :] = (
                        delta[0].detach().cpu().float()
                    )
                if layer_index == layer_count - 1:
                    states[layer_count] = changed[0].detach().cpu().float()
                if isinstance(output, tuple):
                    return (changed,) + output[1:]
                return changed

            return hook

        for layer_index, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_hook(make_policy_hook(layer_index)))
    else:

        def final_output_hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            states[layer_count] = hidden[0].detach().cpu().float()

        handles.append(model.model.layers[-1].register_forward_hook(final_output_hook))

    with torch.no_grad():
        output = model(**encoded, use_cache=False, return_dict=True)
    for handle in handles:
        handle.remove()
    if any(state is None for state in states):
        raise RuntimeError("Failed to cache every raw transformer state")
    return output, torch.stack([state for state in states if state is not None]), controls


def register_generation_policy_hooks(
    model: AutoModelForCausalLM,
    policy: ActivationPolicy,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Apply one policy to the last token processed at every decoder layer."""

    device, dtype = _model_device_and_dtype(model)
    policy.prepare(device, dtype)
    policy.reset()
    handles = []

    def make_hook(layer_index: int):
        def hook(_module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            activation = args[0][:, -1, :]
            delta = policy.activation_delta(layer_index, activation)
            changed = hidden.clone()
            changed[:, -1, :] = changed[:, -1, :] + delta.to(changed.dtype)
            if isinstance(output, tuple):
                return (changed,) + output[1:]
            return changed

        return hook

    for layer_index, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_hook(layer_index)))
    return handles
