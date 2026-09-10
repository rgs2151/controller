"""The only package module that knows how project models are loaded."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)


@dataclass(frozen=True)
class CausalModelLoadSpec:
    """Explicit loading choices for one causal-language-model experiment."""

    model_id: str
    revision: str = "main"
    quantized: bool = False
    dtype: str = "bfloat16"
    attention_implementation: str | None = "eager"
    quantization_compute_dtype: str = "float16"


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


def load_causal_model(
    spec: CausalModelLoadSpec,
    device: str,
    token: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal model on one explicit device using a reproducible spec."""

    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if spec.dtype not in dtype_by_name:
        raise ValueError(f"Unsupported model dtype: {spec.dtype}")
    if spec.quantization_compute_dtype not in dtype_by_name:
        raise ValueError(f"Unsupported quantization compute dtype: {spec.quantization_compute_dtype}")
    dtype = dtype_by_name[spec.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        token=token or None,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs: dict[str, object] = {
        "revision": spec.revision,
        "token": token or None,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if spec.attention_implementation is not None:
        model_kwargs["attn_implementation"] = spec.attention_implementation
    if spec.quantized:
        if not device.startswith("cuda:"):
            raise ValueError("4-bit model loading requires an explicit CUDA device")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype_by_name[spec.quantization_compute_dtype],
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": cuda_device_index(device)}
        model = AutoModelForCausalLM.from_pretrained(spec.model_id, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(spec.model_id, **model_kwargs).to(
            device
        )
    model.eval()
    return model, tokenizer


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
