"""Small-model residual exploration for Qwen-0.5B on SST-2-like sentences.

This script follows the repo's residual logic: it captures the model's layer
inputs, approximates each layer's local Jacobian, and compares the observed next
hidden state to the one-step linear prediction. The final output is a compact
layer-wise residual plot and a JSON summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from robust_steerability.modeling.jacobians import capture_layer_inputs, layer_last_token_jacobian

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(__file__).resolve().parent / "cache"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"

CACHE_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Residual dynamics for a small Qwen model.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="sst-2")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vjp_chunk_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def get_sentences(dataset_name: str, num_samples: int) -> tuple[list[str], list[int]]:
    """Load a small sample set from a common benchmark."""
    dataset_name_lower = dataset_name.lower().replace("_", "-")
    if dataset_name_lower in {"sst-2", "sst2"}:
        dataset = load_dataset("glue", "sst2", split=f"train[:{num_samples}]")
    else:
        dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]")

    texts: list[str] = []
    labels: list[int] = []
    for row in dataset:
        text = row.get("sentence") or row.get("text") or row.get("prompt")
        if text is None:
            continue
        texts.append(str(text))
        labels.append(int(row.get("label", 0)))
    if not texts:
        raise ValueError(f"No text rows were found in dataset {dataset_name!r}.")
    return texts, labels


def collect_hidden_states(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, text: str, max_length: int, device: str) -> tuple[torch.Tensor, list[tuple[torch.Tensor, dict[str, object]]]]:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True, return_dict=True, use_cache=False)
    hidden_states = outputs.hidden_states
    layer_inputs = capture_layer_inputs(model, encoded)
    return hidden_states, layer_inputs


def compute_residuals_for_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    max_length: int,
    device: str,
    vjp_chunk_size: int,
) -> list[dict[str, float]]:
    hidden_states, layer_inputs = collect_hidden_states(model, tokenizer, text, max_length, device)

    residuals: list[dict[str, float]] = []
    num_layers = len(model.model.layers)

    for layer_idx in range(num_layers):
        current_hidden = hidden_states[layer_idx][0, -1, :]
        next_hidden = hidden_states[layer_idx + 1][0, -1, :]

        layer_input, layer_kwargs = layer_inputs[layer_idx]
        last_input = layer_input[0, -1, :]
        jacobian = layer_last_token_jacobian(
            model.model.layers[layer_idx],
            layer_input,
            layer_kwargs,
            vjp_chunk_size=vjp_chunk_size,
        ).to(device)
        last_input = last_input.to(device)
        predicted = jacobian @ last_input

        residual_vec = next_hidden.detach().cpu() - predicted.detach().cpu()
        residual_norm = float(torch.linalg.norm(residual_vec))
        state_norm = float(torch.linalg.norm(next_hidden.detach().cpu()))
        rel_residual = residual_norm / max(state_norm, 1e-12)

        residuals.append(
            {
                "layer": layer_idx,
                "residual_norm": residual_norm,
                "relative_residual": rel_residual,
                "state_norm": state_norm,
                "current_hidden_norm": float(torch.linalg.norm(current_hidden.detach().cpu())),
            }
        )

    return residuals


def summarize_layers(samples: list[dict[str, float]], num_layers: int) -> list[dict[str, float]]:
    summary: list[dict[str, float]] = []
    for layer_idx in range(num_layers):
        values = [sample[layer_idx]["relative_residual"] for sample in samples]
        summary.append(
            {
                "layer": layer_idx,
                "mean_relative_residual": float(np.mean(values)),
                "median_relative_residual": float(np.median(values)),
                "std_relative_residual": float(np.std(values)),
                "min_relative_residual": float(np.min(values)),
                "max_relative_residual": float(np.max(values)),
            }
        )
    return summary


def plot_summary(summary: list[dict[str, float]], labels: list[int], output_path: Path) -> None:
    layers = [row["layer"] for row in summary]
    means = [row["mean_relative_residual"] for row in summary]
    plt.figure(figsize=(10, 5))
    plt.plot(layers, means, marker="o", linewidth=2)
    plt.xlabel("Layer index")
    plt.ylabel("Mean relative residual")
    plt.title(f"Residual dynamics across layers ({len(labels)} samples)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_name = args.model_name
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )
    model.to(args.device)
    model.eval()

    texts, labels = get_sentences(args.dataset_name, args.num_samples)
    layer_samples: list[list[dict[str, float]]] = []

    for text, label in zip(texts, labels):
        residuals = compute_residuals_for_text(
            model=model,
            tokenizer=tokenizer,
            text=text,
            max_length=args.max_length,
            device=args.device,
            vjp_chunk_size=args.vjp_chunk_size,
        )
        layer_samples.append([{**entry, "label": label} for entry in residuals])

    num_layers = len(layer_samples[0])
    summary = summarize_layers(layer_samples, num_layers)

    json_path = CACHE_DIR / "qwen_residual_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plot_path = PLOTS_DIR / "qwen_layer_residuals.png"
    plot_summary(summary, labels, plot_path)

    print(f"Saved summary to: {json_path}")
    print(f"Saved plot to: {plot_path}")
    print("Top 5 layers by mean relative residual:")
    for row in sorted(summary, key=lambda x: x["mean_relative_residual"], reverse=True)[:5]:
        print(row)


if __name__ == "__main__":
    main()
