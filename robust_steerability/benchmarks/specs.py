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
    devices_per_worker: int = 1


MODELS = {
    "gpt2_xl": ModelSpec(
        "gpt2_xl",
        "GPT-2 XL",
        "openai-community/gpt2-xl",
        "15ea56dee5df4983c59b2538573817e1667135e2",
        32,
        32,
    ),
    "qwen25_05b": ModelSpec(
        "qwen25_05b",
        "Qwen-2.5-0.5B",
        "Qwen/Qwen2.5-0.5B",
        "060db6499f32faf8b98477b0a26969ef7d8b9987",
        64,
        64,
    ),
    "llama32_1b_instruct": ModelSpec(
        "llama32_1b_instruct",
        "Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "9213176726f574b556790deb65791e0c5aa438b6",
        32,
        32,
    ),
    "llama32_3b_instruct": ModelSpec(
        "llama32_3b_instruct",
        "Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "0cb88a4f764b7a12671c53f0838cd831a0843b95",
        16,
        16,
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
    "qwen32b": ModelSpec(
        "qwen32b", "Qwen-2.5-32B", "Qwen/Qwen2.5-32B",
        "1818d35814b8319459f4bd55ed1ac8709630f003", 2, 4,
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
    "llama31_70b_instruct": ModelSpec(
        "llama31_70b_instruct",
        "Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "1605565b47bb9346c5515c34102e054115b4f98b",
        1,
        16,
        2,
    ),
    "qwen3_4b": ModelSpec(
        "qwen3_4b",
        "Qwen3-4B",
        "Qwen/Qwen3-4B",
        "1cfa9a7208912126459214e8b04321603b3df60c",
        8,
        16,
    ),
    "gemma3_4b_it": ModelSpec(
        "gemma3_4b_it",
        "Gemma-3-4B-Instruct",
        "google/gemma-3-4b-it",
        "093f9f388b31de276ce2de164bdc2081324b9767",
        8,
        16,
    ),
    "phi4_mini_instruct": ModelSpec(
        "phi4_mini_instruct",
        "Phi-4-mini-instruct",
        "microsoft/Phi-4-mini-instruct",
        "cfbefacb99257ffa30c83adab238a50856ac3083",
        8,
        16,
    ),
    "granite33_2b_instruct": ModelSpec(
        "granite33_2b_instruct",
        "Granite-3.3-2B-Instruct",
        "ibm-granite/granite-3.3-2b-instruct",
        "707f574c62054322f6b5b04b6d075f0a8f05e0f0",
        8,
        16,
    ),
}
