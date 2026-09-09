"""Actual transformer-Jacobian identification for reference Activation-LQR."""

from pathlib import Path

import torch

from robust_steerability.artifacts import configuration_hash
from robust_steerability.modeling.interventions import _decoder_layers
from robust_steerability.modeling.jacobians import capture_layer_inputs, layer_last_token_jacobian


def average_prompt_jacobians(model, tokenizer, records, *, cache_dir: Path,
                             max_length: int, vjp_chunk_size: int,
                             model_revision: str) -> torch.Tensor:
    """Average last-token block derivatives with prefix states held fixed.

    Every prompt/layer derivative is saved independently in its original dtype.
    This retains expensive calculations without retaining autograd graphs or
    materializing all prompts' Jacobians on a GPU at once.
    """
    if not records:
        raise ValueError("Jacobian identification requires fitting prompts")
    cache_dir.mkdir(parents=True, exist_ok=True)
    layers = _decoder_layers(model)
    device = next(model.parameters()).device
    source_files = [Path(__file__), Path(capture_layer_inputs.__code__.co_filename)]
    source = configuration_hash({path.name: configuration_hash({"bytes": path.read_bytes().hex()})
                                 for path in source_files})
    total = None
    for index, record in enumerate(records):
        encoded = tokenizer(record["text"], return_tensors="pt", truncation=True,
                            max_length=max_length).to(device)
        key = configuration_hash({"prompt_id": record["prompt_id"], "tokens": encoded["input_ids"].tolist(),
                                  "implementation": source, "model": model.config._name_or_path,
                                  "model_revision": model_revision,
                                  "model_config": model.config.to_dict()})
        folder = cache_dir / key
        folder.mkdir(exist_ok=True)
        paths = [folder / f"layer_{layer_index:03d}.pt" for layer_index in range(len(layers))]
        captured = None
        if any(not path.exists() for path in paths):
            captured = capture_layer_inputs(model, encoded)
        for layer_index, path in enumerate(paths):
            if path.exists():
                derivative = torch.load(path, map_location="cpu", weights_only=True)
            else:
                hidden, kwargs = captured[layer_index]
                derivative = layer_last_token_jacobian(layers[layer_index], hidden, kwargs, vjp_chunk_size)
                if not torch.isfinite(derivative).all():
                    raise ValueError(f"Nonfinite Jacobian for {record['prompt_id']} layer {layer_index}")
                temporary = path.with_suffix(".pt.tmp")
                torch.save(derivative, temporary)
                temporary.replace(path)
            if total is None:
                total = torch.zeros(len(layers), *derivative.shape, dtype=torch.float64)
            total[layer_index].add_(derivative)
        print(f"Jacobian {index + 1}/{len(records)}: {record['prompt_id']}", flush=True)
        del captured
    return (total / len(records)).float()


def project_dynamics(jacobians: torch.Tensor, encoders: torch.Tensor,
                     decoders: torch.Tensor) -> torch.Tensor:
    """Transform derivatives covariantly: E[k+1]' J[k] D[k]."""
    if len(encoders) != len(jacobians) + 1 or encoders.shape != decoders.shape:
        raise ValueError("Coordinate maps must include all decoder inputs and the terminal output")
    return torch.stack([encoders[k + 1].T @ jacobians[k].to(encoders.device) @ decoders[k]
                        for k in range(len(jacobians))])
