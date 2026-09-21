"""Actual transformer-Jacobian identification for reference Activation-LQR."""

import json
from pathlib import Path

import torch

from robust_steerability.modeling.interventions import _decoder_layers
from robust_steerability.modeling.jacobians import capture_layer_inputs, layer_last_token_jacobian
from robust_steerability.modeling.tokenization import (
    RAW_TEXT_TOKENIZATION,
    tokenize_prompts,
)


def average_prompt_jacobians(model, tokenizer, records, *, cache_dir: Path,
                             max_length: int, vjp_chunk_size: int,
                             model_revision: str,
                             prompt_tokenization: str = RAW_TEXT_TOKENIZATION) -> torch.Tensor:
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
    }
    if prompt_tokenization != RAW_TEXT_TOKENIZATION:
        identity["prompt_tokenization"] = prompt_tokenization
    checkpoint_path = cache_dir / "partial.pt"
    metadata_path = cache_dir / "partial.json"
    total = None
    processed = 0
    if checkpoint_path.exists():
        if not metadata_path.exists():
            raise ValueError(f"Jacobian checkpoint is missing its identity: {cache_dir}")
        saved_identity = json.loads(metadata_path.read_text()).get("identity", {})
        saved_mode = saved_identity.pop(
            "prompt_tokenization", RAW_TEXT_TOKENIZATION
        )
        expected_identity = dict(identity)
        expected_mode = expected_identity.pop(
            "prompt_tokenization", RAW_TEXT_TOKENIZATION
        )
        if saved_identity != expected_identity or saved_mode != expected_mode:
            raise ValueError(
                "Jacobian checkpoint has a different immutable prompt/tokenization identity: "
                f"{cache_dir}"
            )
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        processed = int(payload["count"])
        if processed < 0 or processed > len(records):
            raise ValueError(f"Invalid Jacobian checkpoint count in {cache_dir}")
        total = payload["sum"]

    def checkpoint() -> None:
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save({"count": processed, "sum": total}, temporary)
        temporary.replace(checkpoint_path)
        temporary_metadata = metadata_path.with_suffix(".json.tmp")
        temporary_metadata.write_text(json.dumps({
            "identity": identity,
            "status": "complete" if processed == len(records) else "partial",
            "count": processed,
        }, indent=2) + "\n")
        temporary_metadata.replace(metadata_path)

    for index, record in enumerate(records[processed:], start=processed):
        encoded = tokenize_prompts(
            tokenizer,
            record["text"],
            prompt_tokenization=prompt_tokenization,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)
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
