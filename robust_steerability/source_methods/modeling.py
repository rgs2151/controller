"""Exact model-loading choices used by each comparison implementation."""

from __future__ import annotations

from robust_steerability.modeling.huggingface import CausalModelLoadSpec, load_causal_model
from robust_steerability.source_methods.protocol import METHOD_MODEL_LOADING, model_key


def source_model_spec(
    method: str,
    behavior: str,
    model_id: str,
    checkpoint_revision: str,
) -> CausalModelLoadSpec:
    """Build a pinned spec without imposing the project's H∞ loading choices."""

    model_key(model_id)
    if method == "actadd_lfs":
        method = "alqr"
    if method not in METHOD_MODEL_LOADING:
        raise ValueError(f"Unsupported source method {method!r}")
    if behavior not in {"toxicity", "truthfulness"}:
        raise ValueError(f"Unsupported behavior {behavior!r}")
    settings = METHOD_MODEL_LOADING[method]
    compute_dtype = settings.get("compute_dtype")
    if method in {"original", "alqr", "spid"}:
        compute_dtype = "float32" if behavior == "truthfulness" else "float16"
    return CausalModelLoadSpec(
        model_id=model_id,
        revision=checkpoint_revision,
        quantized=settings["quantization"] == "nf4",
        dtype=str(settings["model_dtype"]),
        attention_implementation=None,
        quantization_compute_dtype=str(compute_dtype or "float16"),
    )


def load_source_model(method: str, behavior: str, model_id: str, checkpoint_revision: str, device: str, token: str):
    """Load one method's source-defined model and left-padding tokenizer."""

    return load_causal_model(
        source_model_spec(method, behavior, model_id, checkpoint_revision),
        device,
        token,
    )
