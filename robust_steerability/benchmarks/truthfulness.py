"""TruthfulQA and MMLU prompt preparation and descriptive metrics."""

from __future__ import annotations

import random

import numpy as np
from datasets import load_dataset


LETTERS = "ABCD"
TRUTHFULQA_ID = "truthful_qa"
TRUTHFULQA_REVISION = "741b8276f2d1982aa3d5b832d3ee81ed3b896490"
MMLU_ID = "cais/mmlu"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"


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


def _format_mmlu_question(row: dict[str, object], include_answer: bool) -> str:
    lines = ["Question: " + str(row["question"]).strip()]
    for index, choice in enumerate(row["choices"][: len(LETTERS)]):
        lines.append(f"{LETTERS[index]}. {str(choice).strip()}")
    lines.append(
        f"Answer: {LETTERS[int(row['answer'])]}" if include_answer else "Answer:"
    )
    return "\n".join(lines)


def load_mmlu_five_shot_prompts(
    seed: int,
    count: int,
    shots: int = 5,
) -> list[dict[str, object]]:
    """Sample MMLU test questions with same-subject dev demonstrations."""

    test = load_dataset(MMLU_ID, "all", split="test", revision=MMLU_REVISION)
    dev = load_dataset(MMLU_ID, "all", split="dev", revision=MMLU_REVISION)
    dev_by_subject: dict[str, list[dict[str, object]]] = {}
    for row in dev:
        dev_by_subject.setdefault(str(row["subject"]), []).append(dict(row))
    rng = random.Random(seed + 17)
    if count < 1 or len(test) < count:
        raise ValueError(f"Requested {count} MMLU prompts; available: {len(test)}")
    test_by_subject = {}
    for row_index, row in enumerate(test):
        test_by_subject.setdefault(str(row["subject"]), []).append(row_index)
    subjects = sorted(test_by_subject)
    indices = []
    while len(indices) < count:
        row_index = rng.choice(test_by_subject[rng.choice(subjects)])
        if row_index not in indices:
            indices.append(row_index)
    output = []
    for row_index in indices:
        row = dict(test[row_index])
        answer = int(row["answer"])
        if answer < 0 or answer >= len(LETTERS):
            raise ValueError(f"Invalid MMLU answer for test row {row_index}")
        subject = str(row["subject"])
        examples = rng.sample(dev_by_subject[subject], shots)
        if len(examples) != shots:
            raise ValueError(f"Missing {shots}-shot demonstrations for {subject}")
        demonstrations = "\n\n".join(
            _format_mmlu_question(example, True) for example in examples
        )
        prompt = demonstrations + "\n\n" if demonstrations else ""
        prompt += _format_mmlu_question(row, False)
        output.append(
            {
                "prompt_id": f"mmlu:{subject}:{row_index}",
                "prompt": prompt,
                "answer_index": answer,
                "subject": subject,
                "demonstrations": examples,
            }
        )
    return output


def completion_after_prompt(text: str, prompt: str) -> str:
    completion = text[len(prompt) :] if text.startswith(prompt) else text
    new_question = completion.find("Q:")
    if new_question > 0:
        completion = completion[:new_question]
    return completion.strip()


def parse_mmlu_letter(text: str) -> int | None:
    """Accept one answer letter, never a letter embedded in a word or sentence."""
    answer = text.strip().upper()
    return LETTERS.index(answer) if answer in tuple(LETTERS) else None


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
