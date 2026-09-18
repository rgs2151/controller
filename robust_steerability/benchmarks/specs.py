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
    "llama32_1b_instruct": ModelSpec(
        "llama32_1b_instruct",
        "Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "9213176726f574b556790deb65791e0c5aa438b6",
        32,
        32,
    ),
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
    "qwen25_3b_instruct": ModelSpec(
        "qwen25_3b_instruct",
        "Qwen-2.5-3B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        4,
        8,
    ),
    "llama31_8b_instruct": ModelSpec(
        "llama31_8b_instruct",
        "Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "0e9e39f249a16976918f6564b8830bc894c89659",
        2,
        8,
    ),
    "qwen3_4b": ModelSpec(
        "qwen3_4b",
        "Qwen3-4B",
        "Qwen/Qwen3-4B",
        "1cfa9a7208912126459214e8b04321603b3df60c",
        8,
        16,
    ),
    "qwen3_8b": ModelSpec(
        "qwen3_8b",
        "Qwen3-8B",
        "Qwen/Qwen3-8B",
        "b968826d9c46dd6066d109eabc6255188de91218",
        4,
        8,
    ),
}
