from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from robust_steerability.benchmarks.toxicity import (
    load_civil_comments_prompts,
    load_jigsaw_toxicity_prompts,
    load_real_toxicity_prompt_pools,
    load_toxic_chat_prompts,
)
from robust_steerability.benchmarks.mmlu import load_mmlu_concept_shift_sets


UNIT_DIR = Path(__file__).resolve().parent
CACHE_DIR = UNIT_DIR / "cache" / "benchmark_ood_sweep"
PLOTS_DIR = UNIT_DIR / "plots"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

RTP_ID = "allenai/real-toxicity-prompts"
RTP_REVISION = "f21629712ffd6a3d13a54fd2807ccd521c55ef74"
JIGSAW_ID = "tcapelle/jigsaw-toxic-comment-classification-challenge"
JIGSAW_REVISION = "2bf801de1b879f287943ecfc81fdca8690d9fc61"
CIVIL_COMMENTS_ID = "google/civil_comments"
TOXIC_CHAT_ID = "lmsys/toxic-chat"
TOXIC_CHAT_CONFIG = "toxicchat0124"
MMLU_ID = "cais/mmlu"
SEED = 2151

MODEL_CATALOG = [
    {"label": "DistilGPT-2", "model_name": "distilgpt2", "family": "gpt2"},
    {"label": "GPT-2 Small", "model_name": "gpt2", "family": "gpt2"},
    {"label": "GPT-2 Medium", "model_name": "gpt2-medium", "family": "gpt2"},
    {"label": "GPT-2 Large", "model_name": "gpt2-large", "family": "gpt2"},
    {"label": "Qwen 0.5B", "model_name": "Qwen/Qwen2.5-0.5B-Instruct", "family": "qwen"},
    {"label": "Qwen 1.5B", "model_name": "Qwen/Qwen2.5-1.5B-Instruct", "family": "qwen"},
    {"label": "Qwen 3B", "model_name": "Qwen/Qwen2.5-3B-Instruct", "family": "qwen"},
    {"label": "Qwen 7B", "model_name": "Qwen/Qwen2.5-7B-Instruct", "family": "qwen"},
    {"label": "Qwen 14B", "model_name": "Qwen/Qwen2.5-14B-Instruct", "family": "qwen"},
    {"label": "Qwen 32B", "model_name": "Qwen/Qwen2.5-32B-Instruct", "family": "qwen"},
    {"label": "Qwen 72B", "model_name": "Qwen/Qwen2.5-72B-Instruct", "family": "qwen"},
]

REGIME_COLORS = {"early": "#0f766e", "mid": "#b45309", "late": "#7c3aed"}
SUBSET_LABELS = {
    "id_rtp": "RTP (ID)",
    "jigsaw_full": "Jigsaw full",
    "jigsaw_long": "Jigsaw long",
    "jigsaw_toxic": "Jigsaw toxic",
    "civil_full": "Civil Comments full",
    "civil_long": "Civil Comments long",
    "civil_toxic": "Civil Comments toxic",
    "toxicchat_full": "ToxicChat full",
    "toxicchat_long": "ToxicChat long",
    "toxicchat_toxic": "ToxicChat toxic",
    "mmlu_ood_other_concepts": "MMLU OOD (other concepts)",
}


def subset_label(subset: str) -> str:
    if subset in SUBSET_LABELS:
        return SUBSET_LABELS[subset]
    if subset.startswith("id_mmlu_"):
        return f"MMLU ID ({subset.replace('id_mmlu_', '').replace('_', ' ')})"
    return subset


def stable_sample(records: list[dict[str, object]], count: int, rng: random.Random) -> list[dict[str, object]]:
    if len(records) < count:
        raise ValueError(f"Requested {count} records from pool of {len(records)}")
    return [records[index] for index in rng.sample(range(len(records)), count)]


def regime_slice(length: int, regime: str) -> slice:
    if regime == "early":
        return slice(0, max(1, length // 3))
    if regime == "mid":
        return slice(max(1, length // 3), min(length - 1, 2 * length // 3))
    if regime == "late":
        return slice(max(1, 2 * length // 3), length)
    raise ValueError(f"Unknown regime: {regime}")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _make_ranked_subsets(
    records: list[dict[str, object]],
    prefix: str,
    num_prompts: int,
    rng: random.Random,
) -> dict[str, list[dict[str, object]]]:
    if len(records) < num_prompts:
        raise ValueError(f"{prefix}: requested {num_prompts} prompts, found {len(records)}")
    full = stable_sample(records, num_prompts, rng)
    longest = sorted(records, key=lambda row: len(str(row["text"])), reverse=True)[:num_prompts]
    most_toxic = sorted(records, key=lambda row: float(row["toxicity"]), reverse=True)[:num_prompts]
    return {
        f"{prefix}_full": full,
        f"{prefix}_long": longest,
        f"{prefix}_toxic": most_toxic,
    }


def prepare_prompt_sets(
    num_prompts: int,
    benchmark_families: list[str],
    mmlu_id_subject: str,
    mmlu_ood_subjects: int,
) -> dict[str, list[dict[str, object]]]:
    rng = random.Random(SEED)

    includes_mmlu = "mmlu" in benchmark_families
    prompt_sets: dict[str, list[dict[str, object]]] = {}

    toxicity_families = [family for family in benchmark_families if family in {"jigsaw", "civil", "toxicchat"}]
    if toxicity_families:
        all_rtp, _, _ = load_real_toxicity_prompt_pools(RTP_ID, RTP_REVISION)
        rtp_pool = [record for record in all_rtp if 0.0 <= float(record["toxicity"]) <= 1.0]
        prompt_sets["id_rtp"] = stable_sample(rtp_pool, num_prompts, rng)

    if "jigsaw" in toxicity_families:
        jigsaw_records = [record for record in load_jigsaw_toxicity_prompts(JIGSAW_ID, JIGSAW_REVISION) if record["text"]]
        prompt_sets.update(_make_ranked_subsets(jigsaw_records, "jigsaw", num_prompts, rng))

    if "civil" in toxicity_families:
        civil_records = [record for record in load_civil_comments_prompts(CIVIL_COMMENTS_ID, split="train") if record["text"]]
        prompt_sets.update(_make_ranked_subsets(civil_records, "civil", num_prompts, rng))

    if "toxicchat" in toxicity_families:
        toxic_chat_records = [
            record
            for record in load_toxic_chat_prompts(TOXIC_CHAT_ID, config_name=TOXIC_CHAT_CONFIG, split="test")
            if record["text"]
        ]
        prompt_sets.update(_make_ranked_subsets(toxic_chat_records, "toxicchat", num_prompts, rng))

    if includes_mmlu:
        prompt_sets.update(
            load_mmlu_concept_shift_sets(
                dataset_id=MMLU_ID,
                id_subject=mmlu_id_subject,
                num_prompts=num_prompts,
                rng=rng,
                ood_subject_count=mmlu_ood_subjects,
            )
        )

    if not prompt_sets:
        raise ValueError("No benchmark families selected")

    return prompt_sets


def get_hidden_states(model, tokenizer, text: str, max_length: int, device: str):
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    model_device = next(model.parameters()).device
    encoded = {key: value.to(model_device if model_device.type != "meta" else device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True, return_dict=True, use_cache=False)
    return outputs.hidden_states, int(encoded["attention_mask"].sum().item())


def residual_curve_for_text(model, tokenizer, text: str, max_length: int, device: str) -> tuple[np.ndarray, int]:
    hidden_states, token_count = get_hidden_states(model, tokenizer, text, max_length, device)
    values = []
    for layer_idx in range(len(hidden_states) - 1):
        previous = hidden_states[layer_idx][0, -1, :].float().detach().cpu()
        current = hidden_states[layer_idx + 1][0, -1, :].float().detach().cpu()
        residual = current - previous
        values.append(float(torch.linalg.norm(residual) / max(torch.linalg.norm(current), torch.tensor(1e-12))))
    return np.asarray(values, dtype=np.float32), token_count


def summarize_subset(model, tokenizer, records: list[dict[str, object]], max_length: int, device: str) -> tuple[dict[str, float], list[dict[str, object]]]:
    curves = []
    prompt_records = []
    for record in records:
        curve, token_count = residual_curve_for_text(model, tokenizer, str(record["text"]), max_length, device)
        curves.append(curve)
        prompt_records.append(
            {
                "prompt_id": str(record["prompt_id"]),
                "source": str(record["source"]),
                "toxicity": float(record["toxicity"]),
                "token_count": token_count,
                "char_count": len(str(record["text"])),
                "residual_curve": curve.tolist(),
            }
        )
    matrix = np.stack(curves, axis=0)
    summary = {"n_prompts": len(records), "mean_token_count": float(np.mean([row["token_count"] for row in prompt_records]))}
    for regime in ["early", "mid", "late"]:
        sl = regime_slice(matrix.shape[1], regime)
        summary[regime] = float(np.mean(matrix[:, sl].mean(axis=1)))
    summary["overall"] = float(np.mean(matrix.mean(axis=1)))
    return summary, prompt_records


def plot_subset_grid(results: list[dict[str, object]], subsets: list[str], out_path: Path) -> None:
    regimes = ["early", "mid", "late"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    x = np.arange(len(results))
    labels = [row["label"] for row in results]

    for axis, regime in zip(axes, regimes):
        for subset in subsets:
            y = [row[subset][regime] for row in results]
            axis.plot(x, y, marker="o", linewidth=2.0, label=subset_label(subset))
        axis.set_title(f"{regime.title()} layers", fontsize=12, fontweight="bold")
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=15, ha="right")
        axis.set_ylabel("Mean relative residual")
        axis.grid(True, linestyle=":", alpha=0.35)
    axes[-1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Benchmark-backed OOD residual sweep", fontsize=16, fontweight="bold")
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def infer_subset_baselines(subsets: list[str]) -> dict[str, str]:
    id_subsets = [subset for subset in subsets if subset.startswith("id_")]
    if not id_subsets:
        raise ValueError("Expected at least one ID subset")

    mmlu_ids = [subset for subset in id_subsets if subset.startswith("id_mmlu_")]
    baseline: dict[str, str] = {}
    for subset in subsets:
        if subset.startswith("id_"):
            continue
        if subset.startswith("mmlu_"):
            if len(mmlu_ids) != 1:
                raise ValueError(f"Expected exactly one MMLU ID subset, found {mmlu_ids}")
            baseline[subset] = mmlu_ids[0]
            continue
        if "id_rtp" not in id_subsets:
            raise ValueError("Missing id_rtp baseline for toxicity subsets")
        baseline[subset] = "id_rtp"
    return baseline


def plot_amplification_bars(
    results: list[dict[str, object]],
    subsets: list[str],
    baseline_map: dict[str, str],
    out_path: Path,
) -> None:
    subsets = [subset for subset in subsets if not subset.startswith("id_")]
    regimes = ["early", "mid", "late"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    x = np.arange(len(results))
    width = min(0.24, 0.78 / max(1, len(subsets)))
    for axis, regime in zip(axes, regimes):
        for offset_idx, subset in enumerate(subsets):
            id_subset = baseline_map[subset]
            deltas = [row[subset][regime] - row[id_subset][regime] for row in results]
            center = (len(subsets) - 1) / 2.0
            axis.bar(x + (offset_idx - center) * width, deltas, width=width, label=subset_label(subset), alpha=0.85)
        axis.axhline(0.0, color="#111827", linewidth=1.0, linestyle=":")
        axis.set_title(f"{regime.title()} amplification", fontsize=12, fontweight="bold")
        axis.set_xticks(x)
        axis.set_xticklabels([row["label"] for row in results], rotation=15, ha="right")
        axis.set_ylabel("OOD - matched ID residual")
        axis.grid(True, axis="y", linestyle=":", alpha=0.35)
    axes[-1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Benchmark OOD residual amplification", fontsize=16, fontweight="bold")
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def write_csv(results: list[dict[str, object]], out_path: Path) -> None:
    fieldnames = ["label", "model_name", "params", "subset", "n_prompts", "mean_token_count", "early", "mid", "late", "overall"]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            subsets = [
                subset
                for subset in row.keys()
                if isinstance(subset, str) and (subset.startswith("id_") or subset in SUBSET_LABELS)
            ]
            for subset in subsets:
                summary = row[subset]
                writer.writerow(
                    {
                        "label": row["label"],
                        "model_name": row["model_name"],
                        "params": row["params"],
                        "subset": subset,
                        "n_prompts": summary["n_prompts"],
                        "mean_token_count": summary["mean_token_count"],
                        "early": summary["early"],
                        "mid": summary["mid"],
                        "late": summary["late"],
                        "overall": summary["overall"],
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark-backed RTP to toxicity OOD residual sweeps across model sizes"
    )
    parser.add_argument("--models", nargs="*", default=[entry["model_name"] for entry in MODEL_CATALOG])
    parser.add_argument("--num_prompts", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument(
        "--output_tag",
        type=str,
        default="benchmark_ood_sweep",
        help="Output file stem under cache/benchmark_ood_sweep and plots",
    )
    parser.add_argument(
        "--benchmark_families",
        nargs="+",
        choices=["jigsaw", "civil", "toxicchat", "mmlu"],
        default=["jigsaw", "civil", "toxicchat", "mmlu"],
        help="Benchmark families to include as OOD sets",
    )
    parser.add_argument(
        "--mmlu_id_subject",
        type=str,
        default="high_school_mathematics",
        help="MMLU subject used as concept-constant ID set when benchmark_families includes mmlu",
    )
    parser.add_argument(
        "--mmlu_ood_subjects",
        type=int,
        default=8,
        help="Number of non-ID MMLU subjects to sample into OOD concept set",
    )
    parser.add_argument(
        "--qwen_4bit_threshold_b",
        type=float,
        default=72.0,
        help="Use 4-bit quantized loading for Qwen models with size >= this B-parameter threshold",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_qwen_size_b(model_name: str) -> float | None:
    if "Qwen/Qwen2.5-" not in model_name or "-Instruct" not in model_name:
        return None
    size_token = model_name.replace("Qwen/Qwen2.5-", "").replace("-Instruct", "")
    if size_token.endswith("B"):
        try:
            return float(size_token[:-1])
        except ValueError:
            return None
    return None


def should_use_4bit(model_name: str, threshold_b: float) -> bool:
    size_b = parse_qwen_size_b(model_name)
    if size_b is None:
        return False
    return size_b >= threshold_b


def select_models(model_names: list[str]) -> list[dict[str, str]]:
    table = {entry["model_name"]: entry for entry in MODEL_CATALOG}
    selected = []
    for name in model_names:
        if name in table:
            selected.append(table[name])
            continue
        selected.append({"label": name.split("/")[-1], "model_name": name, "family": "custom"})
    return selected


def main() -> None:
    args = parse_args()
    prompt_sets = prepare_prompt_sets(
        args.num_prompts,
        args.benchmark_families,
        args.mmlu_id_subject,
        args.mmlu_ood_subjects,
    )
    subset_names = list(prompt_sets.keys())
    baseline_map = infer_subset_baselines(subset_names)
    selected = select_models(args.models)
    all_results: list[dict[str, object]] = []
    cache_payload: dict[str, object] = {"config": vars(args), "models": []}

    for entry in selected:
        print(f"Loading {entry['label']} ({entry['model_name']})")
        tokenizer = AutoTokenizer.from_pretrained(entry["model_name"], trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        if should_use_4bit(entry["model_name"], args.qwen_4bit_threshold_b):
            print(f"  using 4-bit load for {entry['model_name']}")
            compute_dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        else:
            model_kwargs["torch_dtype"] = torch.float16 if args.device.startswith("cuda") else torch.float32

        model = AutoModelForCausalLM.from_pretrained(entry["model_name"], **model_kwargs)
        model.eval()
        params = count_parameters(model)
        result: dict[str, object] = {"label": entry["label"], "model_name": entry["model_name"], "params": params}
        prompt_cache: dict[str, object] = {}
        for subset_name, records in prompt_sets.items():
            summary, prompt_records = summarize_subset(model, tokenizer, records, args.max_length, args.device)
            result[subset_name] = summary
            prompt_cache[subset_name] = prompt_records
            print(
                f"  {subset_name}: early={summary['early']:.4f}, mid={summary['mid']:.4f}, late={summary['late']:.4f}, overall={summary['overall']:.4f}"
            )
        all_results.append(result)
        cache_payload["models"].append({**result, "prompt_records": prompt_cache})
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_json = CACHE_DIR / "benchmark_ood_sweep_summary.json"
    summary_csv = CACHE_DIR / "benchmark_ood_sweep_summary.csv"
    summary_json = CACHE_DIR / f"{args.output_tag}_summary.json"
    summary_csv = CACHE_DIR / f"{args.output_tag}_summary.csv"
    summary_json.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
    write_csv(all_results, summary_csv)

    grid_path = PLOTS_DIR / f"{args.output_tag}_grid.png"
    amp_path = PLOTS_DIR / f"{args.output_tag}_amplification.png"
    plot_subset_grid(all_results, subset_names, grid_path)
    plot_amplification_bars(all_results, subset_names, baseline_map, amp_path)

    print(f"Saved summary JSON to: {summary_json}")
    print(f"Saved summary CSV to: {summary_csv}")
    print(f"Saved figure to: {grid_path}")
    print(f"Saved amplification figure to: {amp_path}")


if __name__ == "__main__":
    main()
