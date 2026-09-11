"""Method-specific calibration for the frozen source implementations."""

from __future__ import annotations

import numpy as np
import torch

from robust_steerability.modeling.interventions import _decoder_layers
from robust_steerability.source_methods.actadd import collect_positionwise_mean, fit_actadd_direction
from robust_steerability.source_methods.control import SetpointCalibration, fit_setpoint_calibration
from robust_steerability.source_methods.iti import ITIFit, fit_iti
from robust_steerability.source_methods.odesteer import ODESteerFit, fit_odesteer
from robust_steerability.source_methods.protocol import (
    ACT_ADAPTER_MODULE_LIMIT,
    ACT_FIT_SAMPLES_PER_CLASS,
    ACTADD_FIT_SAMPLES_PER_CLASS,
    ALQR_CALIBRATION_COUNTS,
    ITI_FIT_SAMPLES_PER_CLASS,
    ODESTEER_FIT_SAMPLES_PER_CLASS,
    ODESTEER_PARAMETERS,
)
from robust_steerability.source_methods.transport import (
    LinearTransport,
    MeanTransport,
    PIDTransport,
    fit_linear_transport,
    fit_mean_transport,
    fit_pid_transport,
    matching_module_names,
    register_transport_hooks,
)


def fit_actadd_calibration(
    model,
    tokenizer,
    *,
    undesired_texts: list[str],
    desired_texts: list[str],
    batch_size: int,
) -> torch.Tensor:
    """Fit the source's 100-per-class, position-wise ActAdd direction."""

    required = ACTADD_FIT_SAMPLES_PER_CLASS
    if len(undesired_texts) != required or len(desired_texts) != required:
        raise ValueError(f"ActAdd requires exactly {required} prompts per class")
    undesired = collect_positionwise_mean(model, tokenizer, undesired_texts, batch_size=batch_size)
    desired = collect_positionwise_mean(model, tokenizer, desired_texts, batch_size=batch_size)
    return fit_actadd_direction(undesired, desired)


def collect_decoder_states(
    model,
    tokenizer,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int | None = None,
) -> torch.Tensor:
    """Collect source decoder inputs and the final block output at the last token."""

    if tokenizer.padding_side != "left":
        raise ValueError("source last-token calibration requires left padding")
    layers = _decoder_layers(model)
    device = next(model.parameters()).device
    collected = []
    for start in range(0, len(texts), batch_size):
        tokenize_arguments = {"return_tensors": "pt", "padding": True, "truncation": True}
        if max_length is not None:
            tokenize_arguments["max_length"] = max_length
        encoded = tokenizer(texts[start:start + batch_size], **tokenize_arguments).to(device)
        states: list[torch.Tensor | None] = [None] * (len(layers) + 1)
        handles = []

        def make_input_hook(layer_index):
            def hook(_module, inputs):
                states[layer_index] = inputs[0][:, -1, :].detach().float()
            return hook

        def terminal_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            states[-1] = hidden[:, -1, :].detach().float()

        for layer_index, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(make_input_hook(layer_index)))
        handles.append(layers[-1].register_forward_hook(terminal_hook))
        with torch.inference_mode():
            model(**encoded, use_cache=False, return_dict=True)
        for handle in handles:
            handle.remove()
        if any(state is None for state in states):
            raise RuntimeError("failed to capture every decoder state")
        collected.append(torch.stack([state for state in states if state is not None], dim=1).cpu())
    return torch.cat(collected)


def fit_setpoint_from_records(
    model,
    tokenizer,
    *,
    behavior: str,
    negative_records: list[dict],
    positive_records: list[dict],
    activation_batch_size: int,
) -> SetpointCalibration:
    """Fit the shared A-LQR/S-PID setpoint from exact source counts."""

    counts = ALQR_CALIBRATION_COUNTS[behavior]
    if len(negative_records) != counts.undesired or len(positive_records) != counts.desired:
        raise ValueError(
            f"{behavior} setpoint requires exactly {counts.undesired} undesired "
            f"and {counts.desired} desired prompts"
        )
    negative_states = collect_decoder_states(
        model, tokenizer, [row["text"] for row in negative_records], batch_size=activation_batch_size
    )
    positive_states = collect_decoder_states(
        model, tokenizer, [row["text"] for row in positive_records], batch_size=activation_batch_size
    )
    return fit_setpoint_calibration(negative_states.mean(dim=0), positive_states.mean(dim=0))


def collect_attention_head_activations(
    model,
    tokenizer,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    """Collect last-token pre-o_proj activations in (sample, layer, head, dim)."""

    if tokenizer.padding_side != "left":
        raise ValueError("ITI last-token calibration requires left padding")
    layers = _decoder_layers(model)
    head_count = int(model.config.num_attention_heads)
    device = next(model.parameters()).device
    batches = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start:start + batch_size], return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(device)
        captured: list[torch.Tensor | None] = [None] * len(layers)
        handles = []

        def make_hook(layer_index):
            def hook(_module, inputs):
                captured[layer_index] = inputs[0][:, -1, :].detach().float().cpu()
            return hook

        for layer_index, layer in enumerate(layers):
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(make_hook(layer_index)))
        with torch.inference_mode():
            model(**encoded, use_cache=False, return_dict=True)
        for handle in handles:
            handle.remove()
        if any(value is None for value in captured):
            raise RuntimeError("failed to capture every attention-head activation")
        flat = torch.stack([value for value in captured if value is not None], dim=1)
        batches.append(flat.reshape(len(flat), len(layers), head_count, -1))
    return torch.cat(batches)


def fit_iti_calibration(
    model,
    tokenizer,
    *,
    undesired_texts: list[str],
    desired_texts: list[str],
    batch_size: int,
    max_length: int = 50,
    seed: int = 42,
) -> ITIFit:
    """Fit the source's 80-per-class probe bank and head ranking once."""

    required = ITI_FIT_SAMPLES_PER_CLASS
    if len(undesired_texts) != required or len(desired_texts) != required:
        raise ValueError(f"ITI requires exactly {required} prompts per class")
    activations = collect_attention_head_activations(
        model,
        tokenizer,
        undesired_texts + desired_texts,
        batch_size=batch_size,
        max_length=max_length,
    )
    labels = np.asarray([1] * required + [0] * required)
    return fit_iti(activations, labels, seed=seed)


def collect_block_output_activations(
    model,
    tokenizer,
    texts: list[str],
    *,
    layer_index: int,
    batch_size: int,
    max_length: int = 256,
) -> torch.Tensor:
    """Collect the source ODESteer layer-output activation at the last token."""

    if tokenizer.padding_side != "left":
        raise ValueError("ODESteer last-token calibration requires left padding")
    layers = _decoder_layers(model)
    if not 0 <= layer_index < len(layers):
        raise ValueError(f"layer_index must be in [0, {len(layers) - 1}]")
    device = next(model.parameters()).device
    batches = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start:start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        current = None

        def hook(_module, _inputs, output):
            nonlocal current
            hidden = output[0] if isinstance(output, tuple) else output
            current = hidden[:, -1, :].detach().float().cpu()

        handle = layers[layer_index].register_forward_hook(hook)
        with torch.inference_mode():
            model(**encoded, use_cache=False, return_dict=True)
        handle.remove()
        if current is None:
            raise RuntimeError(f"failed to capture ODESteer layer {layer_index}")
        batches.append(current)
    return torch.cat(batches)


def fit_odesteer_calibration(
    model,
    tokenizer,
    *,
    behavior: str,
    layer_index: int,
    undesired_texts: list[str],
    desired_texts: list[str],
    batch_size: int,
) -> ODESteerFit:
    """Fit one source ODESteer layer with the behavior-specific full fit size."""

    if behavior not in ODESTEER_FIT_SAMPLES_PER_CLASS:
        raise ValueError(f"Unsupported behavior {behavior!r}")
    required = ODESTEER_FIT_SAMPLES_PER_CLASS[behavior]
    if len(undesired_texts) != required or len(desired_texts) != required:
        raise ValueError(f"ODESteer {behavior} requires exactly {required} prompts per class")
    desired = collect_block_output_activations(
        model,
        tokenizer,
        desired_texts,
        layer_index=layer_index,
        batch_size=batch_size,
    )
    undesired = collect_block_output_activations(
        model,
        tokenizer,
        undesired_texts,
        layer_index=layer_index,
        batch_size=batch_size,
    )
    return fit_odesteer(
        desired,
        undesired,
        components=int(ODESTEER_PARAMETERS["n_components"]),
        degree=int(ODESTEER_PARAMETERS["degree"]),
        gamma=float(ODESTEER_PARAMETERS["gamma"]),
        coef0=float(ODESTEER_PARAMETERS["coef0"]),
        steps=int(ODESTEER_PARAMETERS["steps"]),
    )


def _collect_module_responses(
    model,
    tokenizer,
    texts: list[str],
    *,
    module_name: str,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    module = dict(model.named_modules())[module_name]
    device = next(model.parameters()).device
    responses = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start:start + batch_size], return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(device)
        current = None

        def hook(_module, _inputs, output):
            nonlocal current
            if not torch.is_tensor(output):
                raise ValueError("AcT calibration requires tensor module outputs")
            mask = encoded["attention_mask"].bool()
            masked = output.detach().float().masked_fill(~mask[..., None], torch.nan)
            current = torch.nanmean(masked, dim=1).cpu()

        handle = module.register_forward_hook(hook)
        with torch.inference_mode():
            model(**encoded, use_cache=False, return_dict=True)
        handle.remove()
        if current is None:
            raise RuntimeError(f"failed to capture {module_name}")
        responses.append(current)
    return torch.cat(responses)


def fit_transport_stack(
    model,
    tokenizer,
    *,
    source_texts: list[str],
    target_texts: list[str],
    module_patterns: tuple[str, ...],
    behavior: str,
    method: str,
    batch_size: int,
    max_length: int = 128,
    seed: int = 42,
    module_limit: int = ACT_ADAPTER_MODULE_LIMIT,
) -> dict[str, MeanTransport | LinearTransport | PIDTransport]:
    """Fit AcT modules incrementally with all preceding source hooks active."""

    if behavior not in ACT_FIT_SAMPLES_PER_CLASS:
        raise ValueError(f"Unsupported behavior {behavior!r}")
    required = ACT_FIT_SAMPLES_PER_CLASS[behavior]
    if len(source_texts) != required or len(target_texts) != required:
        raise ValueError(f"AcT {behavior} requires exactly {required} prompts per class")
    module_names = matching_module_names(model, module_patterns)[:module_limit]
    if not module_names:
        raise ValueError("no modules match the source AcT patterns")
    texts = source_texts + target_texts
    source_labels = torch.tensor([True] * len(source_texts) + [False] * len(target_texts))
    fits: dict[str, MeanTransport | LinearTransport | PIDTransport] = {}
    raw_pid_differences: list[torch.Tensor] = []
    linear_random_state = np.random.RandomState(seed)
    for module_name in module_names:
        handles = register_transport_hooks(model, fits, strength=1.0)
        responses = _collect_module_responses(
            model, tokenizer, texts, module_name=module_name,
            batch_size=batch_size, max_length=max_length,
        )
        for handle in handles:
            handle.remove()
        if method == "mean_act":
            fitted = fit_mean_transport(responses, source_labels)
        elif method == "linear_act":
            fitted = fit_linear_transport(responses, source_labels, random_state=linear_random_state)
        elif method == "pid_act":
            mean_fit = fit_mean_transport(responses, source_labels)
            fitted = fit_pid_transport(responses, source_labels, tuple(raw_pid_differences))
            raw_pid_differences.append(mean_fit.target_mean - mean_fit.source_mean)
        else:
            raise ValueError(f"Unsupported AcT method {method!r}")
        fits[module_name] = fitted
    return fits
