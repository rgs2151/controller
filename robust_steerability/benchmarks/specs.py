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
    "pythia_14m": ModelSpec(
        "pythia_14m",
        "Pythia-14M",
        "EleutherAI/pythia-14m",
        "cf967c0a9a04383db6f7b1108d86b2962634b4ac",
        64,
        64,
    ),
    "pythia_31m": ModelSpec(
        "pythia_31m",
        "Pythia-31M",
        "EleutherAI/pythia-31m",
        "e556ace21b489575e94e9d50b6dad2fcc7419679",
        64,
        64,
    ),
    "distilgpt2": ModelSpec(
        "distilgpt2",
        "DistilGPT-2",
        "distilbert/distilgpt2",
        "2290a62682d06624634c1f46a6ad5be0f47f38aa",
        64,
        64,
    ),
    "gpt2_small": ModelSpec(
        "gpt2_small",
        "GPT-2 Small",
        "openai-community/gpt2",
        "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        64,
        64,
    ),
    "smollm2_135m": ModelSpec(
        "smollm2_135m",
        "SmolLM2-135M",
        "HuggingFaceTB/SmolLM2-135M",
        "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        64,
        64,
    ),
    "pythia_160m": ModelSpec(
        "pythia_160m",
        "Pythia-160M",
        "EleutherAI/pythia-160m",
        "50f5173d932e8e61f858120bcb800b97af589f46",
        64,
        64,
    ),
    "gpt2_medium": ModelSpec(
        "gpt2_medium",
        "GPT-2 Medium",
        "openai-community/gpt2-medium",
        "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
        64,
        64,
    ),
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
    "gpt2_large": ModelSpec(
        "gpt2_large",
        "GPT-2 Large",
        "openai-community/gpt2-large",
        "32b71b12589c2f8d625668d2335a01cac3249519",
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
    "olmo2_32b_instruct": ModelSpec(
        "olmo2_32b_instruct",
        "OLMo-2-0325-32B-Instruct",
        "allenai/OLMo-2-0325-32B-Instruct",
        "b96024342a77a69aa0dda815c3454a671f477463",
        2,
        4,
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
