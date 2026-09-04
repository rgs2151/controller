"""Layer input capture and last-token transformer Jacobians."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM


def capture_layer_inputs(
    model: AutoModelForCausalLM,
    encoded: dict[str, torch.Tensor],
) -> list[tuple[torch.Tensor, dict[str, object]]]:
    captured: list[tuple[torch.Tensor, dict[str, object]] | None] = [
        None
    ] * len(model.model.layers)
    handles = []

    def make_hook(layer_index: int):
        def hook(_module, args, kwargs):
            kept_kwargs: dict[str, object] = {}
            for name in [
                "attention_mask",
                "position_ids",
                "cache_position",
                "position_embeddings",
            ]:
                value = kwargs.get(name)
                if isinstance(value, tuple):
                    kept_kwargs[name] = tuple(item.detach() for item in value)
                elif torch.is_tensor(value):
                    kept_kwargs[name] = value.detach()
                elif value is not None:
                    kept_kwargs[name] = value
            kept_kwargs["use_cache"] = False
            captured[layer_index] = (args[0].detach(), kept_kwargs)

        return hook

    for layer_index, layer in enumerate(model.model.layers):
        handles.append(
            layer.register_forward_pre_hook(make_hook(layer_index), with_kwargs=True)
        )
    with torch.no_grad():
        model(**encoded, use_cache=False, return_dict=True)
    for handle in handles:
        handle.remove()
    if any(item is None for item in captured):
        raise RuntimeError("Failed to capture all transformer layer inputs")
    return [item for item in captured if item is not None]


def layer_last_token_jacobian(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    layer_kwargs: dict[str, object],
    vjp_chunk_size: int,
) -> torch.Tensor:
    """Compute one decoder block's last-token state Jacobian in row chunks."""

    prefix = hidden_states[:, :-1, :].detach()
    last_state = hidden_states[0, -1, :].detach().requires_grad_(True)

    def block_last(state: torch.Tensor) -> torch.Tensor:
        full_state = torch.cat([prefix, state.view(1, 1, -1)], dim=1)
        output = layer(full_state, **layer_kwargs)
        if isinstance(output, tuple):
            output = output[0]
        return output[0, -1, :]

    output = block_last(last_state)
    dimension = output.numel()
    rows = []
    for start in range(0, dimension, vjp_chunk_size):
        stop = min(start + vjp_chunk_size, dimension)
        grad_outputs = torch.zeros(
            stop - start,
            dimension,
            device=output.device,
            dtype=output.dtype,
        )
        grad_outputs[
            torch.arange(stop - start, device=output.device),
            torch.arange(start, stop, device=output.device),
        ] = 1
        gradient = torch.autograd.grad(
            output,
            last_state,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=stop < dimension,
        )[0]
        rows.append(gradient.detach().cpu())
    return torch.cat(rows, dim=0)
