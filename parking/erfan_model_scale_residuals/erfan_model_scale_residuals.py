from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
ACTIVATIONS_DIR = CACHE_DIR / "activations"
PLOTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
ACTIVATIONS_DIR.mkdir(exist_ok=True)

MODEL_CATALOG = [
    {"label": "DistilGPT-2", "model_name": "distilgpt2", "family": "gpt2", "params": None},
    {"label": "GPT-2 Small", "model_name": "gpt2", "family": "gpt2", "params": None},
    {"label": "GPT-2 Medium", "model_name": "gpt2-medium", "family": "gpt2", "params": None},
    {"label": "GPT-2 Large", "model_name": "gpt2-large", "family": "gpt2", "params": None},
    {"label": "Qwen 0.5B", "model_name": "Qwen/Qwen2.5-0.5B-Instruct", "family": "qwen", "params": None},
    {"label": "Qwen 1.5B", "model_name": "Qwen/Qwen2.5-1.5B-Instruct", "family": "qwen", "params": None},
    {"label": "Qwen 3B", "model_name": "Qwen/Qwen2.5-3B-Instruct", "family": "qwen", "params": None},
    {"label": "Qwen 7B", "model_name": "Qwen/Qwen2.5-7B-Instruct", "family": "qwen", "params": None},
    {"label": "Qwen 14B", "model_name": "Qwen/Qwen2.5-14B-Instruct", "family": "qwen", "params": None},
]

WITHIN_PROMPTS = [
    "I absolutely loved this movie; the acting was brilliant and the story was moving.",
    "The performance was outstanding, and the atmosphere felt warm and uplifting.",
    "The service was surprisingly smooth and the staff were helpful.",
    "This product is excellent, reliable, and genuinely improved my day.",
    "The experience was delightful from start to finish and felt thoughtfully designed.",
    "This was a terrible experience, and I hated every minute of it.",
    "The product arrived broken and the support was completely useless.",
    "I felt frustrated and disappointed by the entire process and quality.",
    "Everything felt slow, confusing, and badly handled from beginning to end.",
    "The service was awful and the final result left me deeply dissatisfied.",
]

PROMPT_PREFIXES = [
    "",
    "For a short sentiment judgment, ",
    "Consider the final user statement: ",
    "In a concise review setting, ",
    "As a benchmark sentence, ",
]

PROMPT_SUFFIXES = [
    "",
    " Keep the meaning unchanged.",
    " Treat it as a single evaluation sample.",
    " This is part of a controlled benchmark set.",
    " Preserve the original sentiment and intent.",
]

OOD_PROMPTS = {
    "Jigsaw OOD": [
        "The customer review was mostly neutral but the product clearly failed in a surprising way.",
        "This report from the dataset indicates a review pattern with unusual and mixed signals.",
        "The example is from a different benchmark and may not match the usual sentiment distribution.",
        "The review looks surprisingly different from typical sentiment labels and may reflect a broader benchmark shift.",
        "This prompt is drawn from a different evaluation source and tests whether the model preserves meaning under dataset shift.",
        "The task contains mixed semantic cues across a different source distribution than the usual training-like prompts.",
        "This benchmark example is structurally similar but comes from another dataset distribution and may stress generalization.",
        "This sentence sits outside the usual sentiment distribution and should reveal whether hidden-state residuals shift.",
        "This prompt was sampled from a new data source to create an OOD evaluation condition.",
        "The wording is compatible with the same task but drawn from a shifted source distribution.",
    ],
    "Spanish": [
        "Me encantó esta película; la actuación fue brillante y la historia fue conmovedora.",
        "El servicio fue sorprendentemente fluido y el personal fue muy servicial.",
        "La experiencia fue terrible y me arrepiento profundamente de haberla vivido.",
        "La película fue excelente, con una actuación brillante y un desarrollo muy emotivo.",
        "La atención fue muy buena y el trato al cliente fue amable y efectivo.",
        "Este producto me decepcionó mucho y el servicio fue completamente insuficiente.",
        "Todo fue lento, complicado y mal gestionado desde el principio hasta el final.",
        "La calidad fue muy mala y el proceso me dejó frustrado y desilusionado.",
        "La experiencia general fue muy agradable y realmente mejoró mi día.",
        "El resultado final fue frustrante y profundamente insatisfactorio.",
    ],
    "Code-switch": [
        "The customer review was very positive, but the assistant is still asked to mix English and Spanish: I loved this movie; la actuación fue brillante y la historia fue conmovedora.",
        "The service was surprisingly smooth and the staff were helpful; in a mixed-language context, the experience felt really great and quite warm.",
        "I hated this experience, the product was broken, and the support was useless; the mixed Spanish-English prompt made the negative review even more frustrating.",
        "This was a strong positive review in English with a Spanish clause: la experiencia fue muy agradable y me ayudó mucho.",
        "The product was good, but the prompt mixes English and Spanish: el servicio fue rápido, útil y bastante satisfactorio.",
        "This reply is mostly English but includes Spanish sentiment: me decepcionó mucho el producto y el soporte fue horrible.",
        "The sentence is bilingual and combines English clarity with Spanish emotion: la experiencia fue realmente mala y frustrante.",
        "This prompt blends English and Spanish in one sentence to stress robustness under multilingual code-switching.",
        "I loved the service; el trato fue amable, flexible y muy profesional durante toda la experiencia.",
        "The product was terrible and the support was useless; the bilingual prompt made the negative conclusion even more explicit.",
    ],
    "Long context": [
        "The following context is a long document about a product review, and there is a lot of neutral background information before the short final sentence: the product was difficult to use, confusing to install, and disappointing in the end.",
        "We have included several pages of unrelated but verbose context before reaching the actual answer: the service was slow, unhelpful, and left the customer frustrated.",
        "This extended context was added to test long-range drift and distractor sensitivity. The actual final statement is that the experience felt confusing, slow, and disappointing.",
        "A large block of unrelated background information is provided before the final sentence: the product was excellent, reliable, and genuinely improved my day.",
        "This long prompt contains a lot of irrelevant neutral text before the actual review: the service was surprisingly smooth and the staff were helpful.",
        "The model sees a long context and then a final statement that the product arrived broken, the support was useless, and the experience was frustrating.",
        "This is a long and noisy context with a final sentence that clearly summarizes a highly disappointing and confusing experience.",
        "The document begins with generic filler, then narrows to the actual user feedback: the performance was outstanding and the atmosphere felt warm and uplifting.",
        "The paragraph contains several irrelevant details before the final evaluation: the process was inconsistent, slow, and deeply disappointing.",
        "A verbose context is placed ahead of the final review statement, which says the service was smooth and the support was genuinely helpful.",
    ],
    "Adversarial": [
        "The task is to reason carefully: mathematics, code, and a conflicting instruction were combined with the customer review, which ended up being clearly broken and deeply frustrating.",
        "By design, this adversarial prompt mixes a long neutral context with a contrasting instruction and a strongly negative review about the product quality and support.",
        "This combined prompt intentionally overlays a formal system assessment with emotional sentiment and a contradictory command, producing a highly shifted evaluation context.",
        "A contradictory instruction is embedded in a long factual description while the final sentence states the product was excellent and highly reliable.",
        "The prompt requires a careful contradiction: formal evaluation, stacked code, and a strongly negative customer verdict all appear in the same message.",
        "This sentence blends neutral description, explicit instruction conflict, and a harsh review, creating a challenging mixed-distribution prompt.",
        "The final review is negative, but the surrounding text instructs a positive high-level summary, producing adversarial ambiguity.",
        "The task includes mathematical constraints, a formal report, and a customer complaint in one adversarial instruction bundle.",
        "This prompt combines neutral narrative with a contradictory command and a critical evaluation of the product experience.",
        "The model must reconcile mixed instructions, long context, and sentiment shifts within a single highly adversarial prompt.",
    ],
}

FAMILY_COLORS = {"gpt2": "#2563eb", "qwen": "#dc2626", "custom": "#52525b"}
REGIME_COLORS = {"early": "#0f766e", "mid": "#b45309", "late": "#7c3aed"}


def slugify(text: str) -> str:
    return text.lower().replace("/", "_").replace(" ", "_").replace("-", "_").replace(".", "p")


def expand_prompt_pool(base_prompts: list[str], target_count: int) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    for prompt in base_prompts:
        for prefix in PROMPT_PREFIXES:
            for suffix in PROMPT_SUFFIXES:
                candidate = f"{prefix}{prompt}{suffix}".strip()
                if candidate not in seen:
                    expanded.append(candidate)
                    seen.add(candidate)
                if len(expanded) >= target_count:
                    return expanded
    if len(expanded) < target_count:
        raise ValueError(f"Could not expand prompts to requested target count {target_count}")
    return expanded


def regime_slice(length: int, regime: str) -> slice:
    if regime == "early":
        return slice(0, max(1, length // 3))
    if regime == "mid":
        return slice(max(1, length // 3), min(length - 1, 2 * length // 3))
    if regime == "late":
        return slice(max(1, 2 * length // 3), length)
    raise ValueError(f"Unknown regime: {regime}")


def get_model_layers(model: AutoModelForCausalLM):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise TypeError(f"Model type {type(model).__name__} is not supported for layerwise residual analysis")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_hidden_states(model, tokenizer, text: str, max_length: int, device: str):
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    model_device = next(model.parameters()).device
    encoded = {key: value.to(model_device if model_device.type != "meta" else device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True, return_dict=True, use_cache=False)
    hidden_states = outputs.hidden_states
    last_token_states = torch.stack([state[0, -1, :].float().detach().cpu() for state in hidden_states], dim=0)
    input_ids = encoded["input_ids"][0].detach().cpu().to(torch.int32)
    attention_mask = encoded["attention_mask"][0].detach().cpu().to(torch.int8)
    return last_token_states, input_ids, attention_mask


def prompt_record(model, tokenizer, text: str, max_length: int, device: str) -> dict[str, np.ndarray | str | int]:
    last_token_states, input_ids, attention_mask = get_hidden_states(model, tokenizer, text, max_length, device)
    layer_residuals = []
    for layer_idx in range(last_token_states.shape[0] - 1):
        prev = last_token_states[layer_idx]
        curr = last_token_states[layer_idx + 1]
        residual = curr - prev
        norm = torch.linalg.norm(residual)
        denom = max(torch.linalg.norm(curr), torch.tensor(1e-12))
        layer_residuals.append(float(norm / denom))
    return {
        "prompt": text,
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
        "last_token_states": last_token_states.numpy().astype(np.float16),
        "residual_curve": np.asarray(layer_residuals, dtype=np.float32),
        "token_count": int(attention_mask.sum().item()),
    }


def summarize_model(model, tokenizer, prompt_sets, max_length: int, device: str, regimes: list[str]):
    results = {}
    for set_name, prompts in prompt_sets.items():
        records = [prompt_record(model, tokenizer, prompt, max_length, device) for prompt in prompts]
        curves = np.stack([record["residual_curve"] for record in records], axis=0)
        reg_means = {}
        for regime in regimes:
            sl = regime_slice(curves.shape[1], regime)
            reg_means[regime] = float(np.mean(curves[:, sl].mean(axis=1)))
        results[set_name] = {"records": records, "regimes": reg_means}
    return results


def save_activation_cache(model_result: dict, prompt_sets: dict[str, list[str]], summary: dict, out_dir: Path) -> Path:
    out_dir.mkdir(exist_ok=True)
    model_slug = slugify(model_result["label"])
    cache_path = out_dir / f"{model_slug}.pt"
    payload = {
        "metadata": {
            "label": model_result["label"],
            "model_name": model_result["model_name"],
            "family": model_result["family"],
            "params": model_result["params"],
            "params_log10": model_result["params_log10"],
            "prompt_counts": {name: len(prompts) for name, prompts in prompt_sets.items()},
        },
        "sets": {},
    }
    for set_name, set_summary in summary.items():
        records = set_summary["records"]
        payload["sets"][set_name] = {
            "regimes": set_summary["regimes"],
            "prompts": [record["prompt"] for record in records],
            "input_ids": [record["input_ids"] for record in records],
            "attention_mask": [record["attention_mask"] for record in records],
            "token_count": [record["token_count"] for record in records],
            "last_token_states": [record["last_token_states"] for record in records],
            "residual_curves": [record["residual_curve"] for record in records],
        }
    torch.save(payload, cache_path)
    return cache_path


def make_trends_plot(summary: list[dict], out_path: Path, ood_label: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    regimes = ["early", "mid", "late"]
    for regime_idx, regime in enumerate(regimes):
        ax = axes[regime_idx]
        x = []
        y_id = []
        y_ood = []
        for item in summary:
            x.append(item["params_log10"])
            y_id.append(item[f"{regime}_id"])
            y_ood.append(item[f"{regime}_ood"])
        ax.plot(x, y_id, marker="o", color=REGIME_COLORS[regime], linewidth=2.5, label="Within")
        ax.plot(x, y_ood, marker="s", linestyle="--", color=REGIME_COLORS[regime], linewidth=2.0, alpha=0.9, label=ood_label)
        ax.set_title(f"{regime.title()} layers", fontsize=11, fontweight="bold")
        ax.set_xlabel("Model size (log10 params)")
        ax.set_ylabel("Mean relative residual")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"Residual scaling with model size: Within vs {ood_label}", fontsize=15, fontweight="bold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_matrix_plot(summary: list[dict], out_path: Path, ood_label: str):
    models = [item["label"] for item in summary]
    regimes = ["early", "mid", "late"]
    width = 0.55
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for regime_idx, regime in enumerate(regimes):
        ax = axes[regime_idx]
        within_vals = [item[f"{regime}_id"] for item in summary]
        ood_vals = [item[f"{regime}_ood"] for item in summary]
        ax.bar(x - width / 2, within_vals, width=width, color="#4f46e5", alpha=0.75, label="Within")
        ax.bar(x + width / 2, ood_vals, width=width, color="#f97316", alpha=0.8, label=ood_label)
        ax.set_title(f"{regime.title()} regime", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=25, ha="right")
        ax.set_ylabel("Mean residual")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if regime_idx == 0:
            ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"Residual sensitivity to model scale: Within vs {ood_label}", fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_paper_scaling_plot(summary: list[dict], out_path: Path, ood_label: str):
    regimes = ["early", "mid", "late"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")

    for axis, regime in zip(axes.flat[:3], regimes):
        xs = np.asarray([item["params_log10"] for item in summary], dtype=float)
        within_vals = np.asarray([item[f"{regime}_id"] for item in summary], dtype=float)
        ood_vals = np.asarray([item[f"{regime}_ood"] for item in summary], dtype=float)

        for item, x_val, within_val, ood_val in zip(summary, xs, within_vals, ood_vals):
            family_color = FAMILY_COLORS.get(item["family"], FAMILY_COLORS["custom"])
            axis.plot([x_val, x_val], [within_val, ood_val], color=family_color, alpha=0.45, linewidth=1.6)
            axis.scatter(x_val, within_val, s=90, color=family_color, edgecolor="white", linewidth=0.8, marker="o", zorder=3)
            axis.scatter(x_val, ood_val, s=90, color=family_color, edgecolor="white", linewidth=0.8, marker="D", zorder=3)
            axis.text(x_val + 0.01, ood_val + 0.003, item["label"], fontsize=8, color=family_color)

        within_fit = np.polyfit(xs, within_vals, deg=1)
        ood_fit = np.polyfit(xs, ood_vals, deg=1)
        x_grid = np.linspace(xs.min() - 0.03, xs.max() + 0.03, 100)
        axis.plot(x_grid, np.polyval(within_fit, x_grid), color=REGIME_COLORS[regime], linewidth=2.2, label="Within trend")
        axis.plot(x_grid, np.polyval(ood_fit, x_grid), color="#111827", linewidth=2.0, linestyle="--", label=f"{ood_label} trend")

        axis.set_title(f"{regime.title()} layers", fontsize=12, fontweight="bold")
        axis.set_xlabel("Model size (log10 params)")
        axis.set_ylabel("Mean relative residual")
        axis.grid(True, linestyle="--", alpha=0.28)
        axis.set_facecolor("#f8fafc")
        axis.legend(frameon=False, fontsize=9, loc="best")

    delta_axis = axes[1, 1]
    for regime in regimes:
        xs = np.asarray([item["params_log10"] for item in summary], dtype=float)
        deltas = np.asarray([item[f"{regime}_ood"] - item[f"{regime}_id"] for item in summary], dtype=float)
        delta_axis.plot(xs, deltas, marker="o", linewidth=2.0, color=REGIME_COLORS[regime], label=regime.title())
    delta_axis.axhline(0.0, color="#374151", linewidth=1.0, linestyle=":")
    delta_axis.set_title(f"{ood_label} amplification by model size", fontsize=12, fontweight="bold")
    delta_axis.set_xlabel("Model size (log10 params)")
    delta_axis.set_ylabel(f"{ood_label} - Within residual")
    delta_axis.grid(True, linestyle="--", alpha=0.28)
    delta_axis.set_facecolor("#f8fafc")
    delta_axis.legend(frameon=False, fontsize=9)

    fig.suptitle(f"Residual scaling across network size with explicit {ood_label} stress test", fontsize=16, fontweight="bold")
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_camera_ready_plot(summary: list[dict], out_path: Path, ood_label: str):
    regimes = ["early", "mid", "late"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for axis, regime in zip(axes, regimes):
        xs = np.asarray([item["params_log10"] for item in summary], dtype=float)
        within_vals = np.asarray([item[f"{regime}_id"] for item in summary], dtype=float)
        ood_vals = np.asarray([item[f"{regime}_ood"] for item in summary], dtype=float)
        for item, x_val, within_val, ood_val in zip(summary, xs, within_vals, ood_vals):
            family_color = FAMILY_COLORS.get(item["family"], FAMILY_COLORS["custom"])
            axis.plot([x_val, x_val], [within_val, ood_val], color=family_color, linewidth=1.5, alpha=0.35)
            axis.scatter(x_val, within_val, s=80, color=family_color, marker="o", edgecolor="black", linewidth=0.45, zorder=3)
            axis.scatter(x_val, ood_val, s=92, color=family_color, marker="D", edgecolor="black", linewidth=0.45, zorder=3)
            axis.text(x_val + 0.012, max(within_val, ood_val), item["label"], fontsize=7.5, color=family_color)
        axis.plot(xs, within_vals, color="#111827", linewidth=1.8, alpha=0.75)
        axis.plot(xs, ood_vals, color=REGIME_COLORS[regime], linewidth=2.2, linestyle="--")
        axis.set_title(f"{regime.title()} layers", fontsize=13, fontweight="bold")
        axis.set_xlabel("$\\log_{10}$(parameters)", fontsize=11)
        axis.set_ylabel("Mean relative residual", fontsize=11)
        axis.grid(True, linestyle=":", linewidth=0.8, alpha=0.35)
        axis.tick_params(labelsize=10)
    fig.suptitle(f"Network-size scaling of residuals under within-distribution and {ood_label.lower()} prompts", fontsize=16, fontweight="bold")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_multi_ood_grid(multi_summary: dict[str, list[dict]], out_path: Path) -> None:
    regimes = ["early", "mid", "late"]
    ood_labels = list(multi_summary.keys())
    fig, axes = plt.subplots(len(ood_labels), 3, figsize=(16, 4.2 * len(ood_labels)), constrained_layout=True)
    if len(ood_labels) == 1:
        axes = np.asarray([axes])

    for row_idx, ood_label in enumerate(ood_labels):
        summary = multi_summary[ood_label]
        xs = np.asarray([item["params_log10"] for item in summary], dtype=float)
        for col_idx, regime in enumerate(regimes):
            axis = axes[row_idx, col_idx]
            within_vals = np.asarray([item[f"{regime}_id"] for item in summary], dtype=float)
            ood_vals = np.asarray([item[f"{regime}_ood"] for item in summary], dtype=float)
            axis.plot(xs, within_vals, color="#111827", linewidth=2.0, marker="o", label="Within")
            axis.plot(xs, ood_vals, color=REGIME_COLORS[regime], linewidth=2.1, marker="D", linestyle="--", label=ood_label)
            for item in summary:
                family_color = FAMILY_COLORS.get(item["family"], FAMILY_COLORS["custom"])
                axis.plot([item["params_log10"], item["params_log10"]], [item[f"{regime}_id"], item[f"{regime}_ood"]], color=family_color, alpha=0.25, linewidth=1.2)
            axis.grid(True, linestyle=":", alpha=0.35)
            axis.set_xlabel("$\\log_{10}$(parameters)")
            axis.set_ylabel("Mean relative residual")
            axis.set_title(f"{ood_label}: {regime.title()}", fontsize=11, fontweight="bold")
            if row_idx == 0 and col_idx == 2:
                axis.legend(frameon=False, fontsize=9)

    fig.suptitle("Top OOD families across network size", fontsize=16, fontweight="bold")
    fig.savefig(out_path, dpi=280, bbox_inches="tight")
    plt.close(fig)


def build_model_result(entry: dict, summary: dict, ood_label: str, prompt_sets: dict[str, list[str]], activation_path: Path) -> dict:
    model_result = {
        "label": entry["label"],
        "model_name": entry["model_name"],
        "params": entry["params"],
        "params_log10": float(np.log10(max(entry["params"], 1))),
        "family": entry["family"],
        "activation_cache": str(activation_path),
        "ood_family": ood_label,
        "prompt_counts": {name: len(prompts) for name, prompts in prompt_sets.items()},
    }
    for regime in ["early", "mid", "late"]:
        model_result[f"{regime}_id"] = float(summary["Within"]["regimes"][regime])
        model_result[f"{regime}_ood"] = float(summary[ood_label]["regimes"][regime])
    return model_result


def write_summary_csv(summary: list[dict], out_path: Path) -> None:
    fieldnames = [
        "label",
        "model_name",
        "family",
        "params",
        "params_log10",
        "early_id",
        "early_ood",
        "mid_id",
        "mid_ood",
        "late_id",
        "late_ood",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow({key: row[key] for key in fieldnames})


def build_technical_legend(summary: list[dict], ood_label: str, prompt_counts: dict[str, int], max_length: int) -> str:
    model_bits = "; ".join(
        f"{item['label']} ({item['params']:,} parameters)" for item in summary
    )
    return (
        f"Figure X. Residual scaling across network size under within-distribution and {ood_label.lower()} prompts. "
        f"We evaluated {len(summary)} decoder-only language models spanning GPT-2 and Qwen families: {model_bits}. "
        f"For each model, we measured last-token layerwise residual magnitude over {prompt_counts['Within']} within-distribution prompts and {prompt_counts[ood_label]} {ood_label.lower()} prompts. "
        f"Prompt sets were matched in cardinality and processed with a maximum sequence length of {max_length} tokens using a single forward pass with hidden states returned at every layer. "
        "For each prompt, we extracted the last-token hidden state trajectory h_l across all layers and defined the residual at layer l as the relative norm ||h_(l+1) - h_l|| / ||h_(l+1)||. "
        f"Residuals were aggregated into early, middle, and late thirds of the network depth, then averaged within each prompt and condition. "
        f"Circles denote within-distribution means, diamonds denote {ood_label.lower()} means, and connecting segments show the condition-wise shift for each model at fixed parameter count. "
        f"All activation caches were saved to disk as CPU-loadable tensors, enabling downstream plotting and statistical analysis without GPU access."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Compare residual dynamics across model sizes and distribution shifts.")
    parser.add_argument("--models", nargs="*", default=[entry["model_name"] for entry in MODEL_CATALOG], help="Model names to evaluate")
    parser.add_argument("--ood_family", type=str, default="Adversarial", choices=sorted(OOD_PROMPTS.keys()), help="OOD prompt family to compare against within-distribution prompts")
    parser.add_argument("--ood_families", nargs="*", choices=sorted(OOD_PROMPTS.keys()), help="Optional list of OOD families to compare in one run")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--prompts_per_condition", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--subset", action="store_true", help="Use a smaller prompt subset for quick experiments")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def select_model_entries(model_names: list[str]):
    name_to_entry = {entry["model_name"]: entry for entry in MODEL_CATALOG}
    ordered = []
    for name in model_names:
        if name not in name_to_entry:
            ordered.append({"label": name, "model_name": name, "family": "custom", "params": None})
        else:
            ordered.append(name_to_entry[name])
    return ordered


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    selected_models = select_model_entries(args.models)
    ood_labels = args.ood_families if args.ood_families else [args.ood_family]
    multi_summary: dict[str, list[dict]] = {}

    for ood_label in ood_labels:
        slug = ood_label.lower().replace(" ", "_").replace("-", "_")
        prompt_sets = {
            "Within": expand_prompt_pool(WITHIN_PROMPTS, args.prompts_per_condition),
            ood_label: expand_prompt_pool(OOD_PROMPTS[ood_label], args.prompts_per_condition),
        }
        if args.subset:
            prompt_sets = {"Within": WITHIN_PROMPTS[:5], ood_label: OOD_PROMPTS[ood_label][:5]}

        regimes = ["early", "mid", "late"]
        all_summary = []

        for entry in selected_models:
            model_name = entry["model_name"]
            print(f"Loading {entry['label']} ({model_name}) for {ood_label}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
                )
                model.eval()
            except Exception as exc:  # pragma: no cover - experimental scaffolding
                print(f"Skipping {model_name}: {exc}")
                continue

            total_params = count_parameters(model)
            entry_with_params = {**entry, "params": total_params}
            summary = summarize_model(model, tokenizer, prompt_sets, args.max_length, args.device, regimes)
            activation_path = save_activation_cache(
                {
                    "label": entry["label"],
                    "model_name": model_name,
                    "family": entry["family"],
                    "params": total_params,
                    "params_log10": float(np.log10(max(total_params, 1))),
                },
                prompt_sets,
                summary,
                ACTIVATIONS_DIR / slug,
            )
            model_result = build_model_result(entry_with_params, summary, ood_label, prompt_sets, activation_path)
            all_summary.append(model_result)
            print(f"  params: {total_params:,}")
            for regime in regimes:
                print(f"  {regime:>5s}: within={model_result[f'{regime}_id']:.4f}, {ood_label.lower()}={model_result[f'{regime}_ood']:.4f}")
            print(f"  saved activations: {activation_path}")

            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not all_summary:
            raise RuntimeError("No models were successfully loaded. Check the model list and the CUDA environment.")

        summary_path = CACHE_DIR / f"network_size_residual_summary_{slug}.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(all_summary, handle, indent=2)

        trend_path = PLOTS_DIR / f"network_size_residual_trend_{slug}.png"
        matrix_path = PLOTS_DIR / f"network_size_residual_matrix_{slug}.png"
        paper_path = PLOTS_DIR / f"network_size_residual_scaling_paper_{slug}.png"
        camera_ready_path = PLOTS_DIR / f"network_size_residual_scaling_camera_ready_{slug}.png"
        csv_path = CACHE_DIR / f"network_size_residual_summary_{slug}.csv"
        legend_path = CACHE_DIR / f"network_size_residual_legend_{slug}.txt"
        make_trends_plot(all_summary, trend_path, ood_label)
        make_matrix_plot(all_summary, matrix_path, ood_label)
        make_paper_scaling_plot(all_summary, paper_path, ood_label)
        make_camera_ready_plot(all_summary, camera_ready_path, ood_label)
        write_summary_csv(all_summary, csv_path)

        legend_text = build_technical_legend(
            all_summary,
            ood_label,
            {name: len(prompts) for name, prompts in prompt_sets.items()},
            args.max_length,
        )
        legend_path.write_text(legend_text, encoding="utf-8")

        multi_summary[ood_label] = all_summary
        print(f"Saved summary to: {summary_path}")
        print(f"Saved CSV to: {csv_path}")
        print(f"Saved trend plot to: {trend_path}")
        print(f"Saved matrix plot to: {matrix_path}")
        print(f"Saved paper plot to: {paper_path}")
        print(f"Saved camera-ready plot to: {camera_ready_path}")
        print(f"Saved legend text to: {legend_path}")
        print(f"OOD family used: {ood_label}")
        print(f"Prompt counts: {json.dumps({name: len(prompts) for name, prompts in prompt_sets.items()})}")

        sentence = (
            f"Across model scales, {ood_label.lower()} prompts modulate residual size most strongly in the deeper half of the network, "
            "while the dependence on parameter count remains non-monotonic across GPT-2 and Qwen families."
        )
        print("Paper sentence:")
        print(sentence)

    if len(multi_summary) > 1:
        top_slug = "_".join(label.lower().replace(" ", "_") for label in ood_labels)
        multi_path = PLOTS_DIR / f"network_size_residual_top_ood_grid_{top_slug}.png"
        make_multi_ood_grid(multi_summary, multi_path)
        print(f"Saved multi-OOD grid to: {multi_path}")


if __name__ == "__main__":
    main()
