"""TruthfulQA prompt preparation and descriptive metrics."""

from __future__ import annotations

import random

import numpy as np
from datasets import load_dataset


TRUTHFULQA_ID = "truthful_qa"
TRUTHFULQA_REVISION = "741b8276f2d1982aa3d5b832d3ee81ed3b896490"


def load_truthfulqa_prompts(seed: int, count: int, excluded_prompt_ids: set[str]) -> list[dict[str, object]]:
    """Sample TruthfulQA generation questions with the paper prompt format."""

    dataset = load_dataset(
        TRUTHFULQA_ID,
        "generation",
        split="validation",
        revision=TRUTHFULQA_REVISION,
    )
    rows = [
        {
            "prompt_id": f"truthfulqa:{index}",
            "question": str(row["question"]).strip(),
            "prompt": f"Q: {str(row['question']).strip()} A:",
        }
        for index, row in enumerate(dataset)
        if str(row.get("question", "")).strip() and f"truthfulqa:{index}" not in excluded_prompt_ids
    ]
    rng = random.Random(seed)
    if count < 1 or len(rows) < count:
        raise ValueError(f"Requested {count} TruthfulQA prompts; available: {len(rows)}")
    return [rows[i] for i in rng.sample(range(len(rows)), count)]
def completion_after_prompt(text: str, prompt: str) -> str:
    completion = text[len(prompt) :] if text.startswith(prompt) else text
    new_question = completion.find("Q:")
    if new_question > 0:
        completion = completion[:new_question]
    return completion.strip()
def bernoulli_percent(scores: list[float]) -> tuple[float, float]:
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return 0.0, 0.0
    probability = float(values.mean())
    standard_error = (probability * (1.0 - probability) / values.size) ** 0.5
    return 100.0 * probability, 100.0 * standard_error


def product_percent(
    first_mean: float,
    first_standard_error: float,
    second_mean: float,
    second_standard_error: float,
) -> tuple[float, float]:
    mean = first_mean * second_mean / 100.0
    variance = (
        (second_mean / 100.0) ** 2 * first_standard_error**2
        + (first_mean / 100.0) ** 2 * second_standard_error**2
    )
    return mean, variance**0.5
