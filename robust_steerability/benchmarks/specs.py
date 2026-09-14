"""Pinned model and benchmark definitions shared by every pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    model_id: str
    revision: str
    activation_batch_size: int
    jacobian_vjp_chunk_size: int


MODELS = {
    "gemma2b": ModelSpec(
        "gemma2b", "Gemma-2-2B", "google/gemma-2-2b",
        "c5ebcd40d208330abc697524c919956e692655cf", 16, 32,
    ),
    "llama8b": ModelSpec(
        "llama8b", "Llama-3-8B", "meta-llama/Meta-Llama-3-8B",
        "8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920", 8, 16,
    ),
    "qwen14b": ModelSpec(
        "qwen14b", "Qwen-2.5-14B", "Qwen/Qwen2.5-14B",
        "97e1e76335b7017d8f67c08a19d103c0504298c9", 4, 8,
    ),
}

METHODS = (
    "original", "iti", "actadd", "mean_act", "linear_act", "pid_act",
    "odesteer", "spid", "alqr", "h_infinity",
)


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    fit_dataset: str
    transfer_datasets: tuple[str, ...]
    evaluation_samples: int
    repetitions: int
    jacobian_prompts: int


BENCHMARK_SPECS = {
    "truthfulness": BenchmarkSpec(
        "truthfulness", "truthfulqa", ("truthfulqa_spanish",), 817, 5, 35
    ),
    "toxicity": BenchmarkSpec(
        "toxicity", "realtoxicityprompts", ("jigsaw",), 1000, 5, 50
    ),
}
