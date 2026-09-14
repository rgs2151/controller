"""Actual transformer-Jacobian identification for reference Activation-LQR."""

import json
from pathlib import Path

import torch

from robust_steerability.artifacts import configuration_hash
from robust_steerability.modeling.interventions import _decoder_layers
from robust_steerability.modeling.jacobians import capture_layer_inputs, layer_last_token_jacobian


def average_prompt_jacobians(model, tokenizer, records, *, cache_dir: Path,
                             max_length: int, vjp_chunk_size: int,
                             model_revision: str) -> torch.Tensor:
    """Average last-token block derivatives with prefix states held fixed.

    A resumable float64 sum and count are checkpointed per worker. Individual
    prompt Jacobians are never retained. This preserves the estimator while
    bounding storage by one accumulator per worker rather than one full tensor
    per prompt.
    """
    if not records:
        raise ValueError("Jacobian identification requires fitting prompts")
    cache_dir.mkdir(parents=True, exist_ok=True)
    layers = _decoder_layers(model)
    device = next(model.parameters()).device
    source_files = [Path(__file__), Path(capture_layer_inputs.__code__.co_filename)]
    source = configuration_hash({
        path.name: configuration_hash({"bytes": path.read_bytes().hex()})
        for path in source_files
    })
    identity = {
        "schema_version": 1,
        "model": str(model.config._name_or_path),
        "model_revision": model_revision,
        "model_config": model.config.to_dict(),
        "max_length": max_length,
        "vjp_chunk_size": vjp_chunk_size,
        "records": [
            {"prompt_id": str(record["prompt_id"]), "text": str(record["text"])}
            for record in records
        ],
        "implementation": source,
    }
    checkpoint_path = cache_dir / "partial.pt"
    metadata_path = cache_dir / "partial.json"
    total = None
    processed = 0
    if checkpoint_path.exists() or metadata_path.exists():
        if not checkpoint_path.exists() or not metadata_path.exists():
            raise ValueError(f"Incomplete Jacobian accumulator: {cache_dir}")
        metadata = json.loads(metadata_path.read_text())
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if metadata.get("identity") != identity or payload.get("identity") != identity:
            raise ValueError(f"Jacobian accumulator identity mismatch: {cache_dir}")
        processed = int(payload["count"])
        total = payload["sum"]
        if total.dtype != torch.float64 or not 0 <= processed <= len(records):
            raise ValueError(f"Invalid Jacobian accumulator: {cache_dir}")

    def checkpoint() -> None:
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save({"identity": identity, "count": processed, "sum": total}, temporary)
        temporary.replace(checkpoint_path)
        temporary_metadata = metadata_path.with_suffix(".json.tmp")
        temporary_metadata.write_text(json.dumps({
            "identity": identity,
            "status": "complete" if processed == len(records) else "partial",
            "count": processed,
        }, indent=2) + "\n")
        temporary_metadata.replace(metadata_path)

    for index, record in enumerate(records[processed:], start=processed):
        encoded = tokenizer(record["text"], return_tensors="pt", truncation=True,
                            max_length=max_length).to(device)
        captured = capture_layer_inputs(model, encoded)
        for layer_index in range(len(layers)):
            hidden, kwargs = captured[layer_index]
            derivative = layer_last_token_jacobian(
                layers[layer_index], hidden, kwargs, vjp_chunk_size
            )
            if not torch.isfinite(derivative).all():
                raise ValueError(
                    f"Nonfinite Jacobian for {record['prompt_id']} layer {layer_index}"
                )
            if total is None:
                total = torch.zeros(len(layers), *derivative.shape, dtype=torch.float64)
            total[layer_index].add_(derivative)
        processed = index + 1
        if processed % 4 == 0 or processed == len(records):
            checkpoint()
        print(f"Jacobian {index + 1}/{len(records)}: {record['prompt_id']}", flush=True)
        del captured
    if total is None or processed != len(records):
        raise RuntimeError("Jacobian accumulator did not reach the expected count")
    return (total / len(records)).float()


def project_dynamics(jacobians: torch.Tensor, encoders: torch.Tensor,
                     decoders: torch.Tensor) -> torch.Tensor:
    """Transform derivatives covariantly: E[k+1]' J[k] D[k]."""
    if len(encoders) != len(jacobians) + 1 or encoders.shape != decoders.shape:
        raise ValueError("Coordinate maps must include all decoder inputs and the terminal output")
    return torch.stack([encoders[k + 1].T @ jacobians[k].to(encoders.device) @ decoders[k]
                        for k in range(len(jacobians))])
