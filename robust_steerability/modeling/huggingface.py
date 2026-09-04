"""The only package module that knows how project models are loaded."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def load_access_token(repo_root: Path) -> str:
    """Load the Hugging Face token from the environment or repository .env."""

    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = repo_root / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip().strip("\"'")
                if token:
                    return token
    raise RuntimeError(f"HF_TOKEN is not set and was not found in {env_path}")


def cuda_device_index(device: str) -> int:
    if not device.startswith("cuda:"):
        raise ValueError("The quantized model requires a CUDA device such as cuda:0")
    return int(device.split(":", 1)[1])


def load_quantized_causal_model(
    model_id: str,
    revision: str,
    device: str,
    token: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the pinned 4-bit causal model configuration used by the project."""

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        token=token,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        token=token,
        quantization_config=quantization,
        dtype=torch.float32,
        device_map={"": cuda_device_index(device)},
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def load_sequence_classifier(
    model_id: str,
    revision: str,
    device: str,
    token: str,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load a pinned Hugging Face sequence classifier."""

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        token=token,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        revision=revision,
        token=token,
        dtype=torch.float32,
    ).to(device)
    model.eval()
    return model, tokenizer


def tokenize_text(
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device,
    max_length: int,
) -> dict[str, torch.Tensor]:
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)
