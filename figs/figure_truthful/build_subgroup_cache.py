from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset


UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[1]
OUTPUT = UNIT / "cache" / "truthfulqa_category_txi.csv"
DATASET_ARROW = (
    Path.home()
    / ".cache/huggingface/datasets/truthful_qa/generation/0.0.0"
    / "741b8276f2d1982aa3d5b832d3ee81ed3b896490"
    / "truthful_qa-validation.arrow"
)

METHOD_KEYS = {
    "Original": "original",
    "ITI": "iti",
    "ActAdd": "actadd",
    "Mean-AcT": "mean_act",
    "Linear-AcT": "linear_act",
    "PID-AcT": "pid_act",
    "ODESteer": "odesteer",
    "S-PID": "spid",
    "A-LQR": "alqr",
    "H∞ (ours)": "h_infinity",
}

MODELS = {
    "GPT-2 XL": "gpt2_xl",
    "Llama-3-8B": "llama8b",
    "Qwen-2.5-14B": "qwen14b",
    "OLMo-2-32B": "olmo2_32b_instruct",
}


def score_root(model: str, method: str, split: str) -> Path:
    cache = REPO / "benchmarks/truthfulness/cache"
    if model == "GPT-2 XL":
        suffix = (
            "txi95_fluency05_n200_r1_hinf_seed000"
            if method == "H∞ (ours)" and split == "ID"
            else "txi95_fluency05_n200_r1"
        )
        return cache / "gpt2_xl/evaluations/kv_cache_off/calibrations" / suffix
    if model == "Llama-3-8B" and method == "H∞ (ours)":
        return (
            cache
            / "llama8b/evaluations/kv_cache_off/calibrations/txi_fluency_95_05"
        )
    if model == "OLMo-2-32B":
        return (
            cache
            / "olmo2_32b_instruct/evaluations/kv_cache_off/calibrations"
            / "txi95_fluency05_n200_r1"
        )
    return cache / MODELS[model] / "evaluations/kv_cache_off"


def prompt_scores(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in payload["rows"]:
        if row.get("score") is not None:
            grouped[str(row["prompt_id"])].append(float(row["score"]))
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def main() -> None:
    dataset = Dataset.from_file(str(DATASET_ARROW))
    categories = {
        f"truthfulqa:{index}": str(category)
        for index, category in enumerate(dataset["category"])
    }
    counts = Counter(categories.values())
    selected_categories = [name for name, _ in counts.most_common(10)]

    rows: list[dict[str, object]] = []
    for split, evaluation_key in [
        ("ID", "truthfulness"),
        ("OOD", "truthfulness_spanish"),
    ]:
        for model, model_key in MODELS.items():
            for method, method_key in METHOD_KEYS.items():
                root = score_root(model, method, split)
                truth_path = (
                    root
                    / "scores/truthfulqa_true"
                    / evaluation_key
                    / method_key
                    / "final.json"
                )
                info_path = (
                    root
                    / "scores/truthfulqa_informative"
                    / evaluation_key
                    / method_key
                    / "final.json"
                )
                if not truth_path.exists() or not info_path.exists():
                    continue
                truth = prompt_scores(truth_path)
                info = prompt_scores(info_path)
                common = set(truth) & set(info)
                for category in selected_categories:
                    prompt_ids = [
                        prompt_id
                        for prompt_id in common
                        if categories.get(prompt_id) == category
                    ]
                    if not prompt_ids:
                        continue
                    true_pct = 100.0 * sum(truth[p] for p in prompt_ids) / len(prompt_ids)
                    info_pct = 100.0 * sum(info[p] for p in prompt_ids) / len(prompt_ids)
                    rows.append(
                        {
                            "split": split,
                            "model": model,
                            "method": method,
                            "category": category,
                            "n_questions": len(prompt_ids),
                            "true_pct": true_pct,
                            "informative_pct": info_pct,
                            "txi_pct": true_pct * info_pct / 100.0,
                            "truth_source": truth_path.relative_to(REPO),
                            "informative_source": info_path.relative_to(REPO),
                        }
                    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
