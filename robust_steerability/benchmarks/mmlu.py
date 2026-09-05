"""MMLU prompt loaders for concept-shift experiments."""

from __future__ import annotations

import random
from typing import Iterable

from datasets import get_dataset_config_names, load_dataset


def _format_mmlu_prompt(question: str, choices: Iterable[object]) -> str:
    letters = ["A", "B", "C", "D", "E", "F"]
    normalized = [str(choice).strip() for choice in choices]
    option_lines = []
    for index, choice in enumerate(normalized):
        if not choice:
            continue
        label = letters[index] if index < len(letters) else f"Option {index + 1}"
        option_lines.append(f"{label}. {choice}")
    options_text = "\n".join(option_lines)
    return f"Question: {question.strip()}\n{options_text}\nAnswer:"


def load_mmlu_subject_prompts(
    dataset_id: str,
    subject: str,
    split: str = "test",
    revision: str | None = None,
) -> list[dict[str, object]]:
    """Load formatted prompts for one MMLU subject."""

    kwargs = {"path": dataset_id, "name": subject}
    if revision:
        kwargs["revision"] = revision

    try_splits = [split, "validation", "dev", "test"]
    dataset = None
    last_error: Exception | None = None
    for candidate in try_splits:
        try:
            dataset = load_dataset(**kwargs)[candidate]
            break
        except Exception as exc:  # pragma: no cover - remote dataset variants
            last_error = exc
            continue
    if dataset is None:
        raise RuntimeError(
            f"Failed to load MMLU subject '{subject}' from {dataset_id}"
        ) from last_error

    records: list[dict[str, object]] = []
    for row_index, row in enumerate(dataset):
        question = row.get("question")
        choices = row.get("choices")
        if question is None or choices is None:
            continue
        question_text = str(question).strip()
        if not question_text:
            continue
        prompt_text = _format_mmlu_prompt(question_text, choices)
        if len(prompt_text.strip()) == 0:
            continue
        records.append(
            {
                "prompt_id": f"mmlu:{subject}:{row_index}",
                "text": prompt_text,
                "source": "mmlu",
                "toxicity": 0.0,
                "subject": subject,
            }
        )
    return records


def load_mmlu_concept_shift_sets(
    dataset_id: str,
    id_subject: str,
    num_prompts: int,
    rng: random.Random,
    ood_subject_count: int = 8,
    revision: str | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Build ID/OOD prompt pools where OOD uses different MMLU subjects."""

    subjects = [name for name in get_dataset_config_names(dataset_id) if name != "all"]
    if id_subject not in subjects:
        available = ", ".join(subjects[:12])
        raise ValueError(
            f"MMLU ID subject '{id_subject}' not found in {dataset_id}. "
            f"Available examples: {available}"
        )

    id_records = load_mmlu_subject_prompts(dataset_id, id_subject, revision=revision)
    if len(id_records) < num_prompts:
        raise ValueError(
            f"MMLU ID subject {id_subject} has {len(id_records)} prompts, needs {num_prompts}"
        )

    other_subjects = [name for name in subjects if name != id_subject]
    take_subjects = min(len(other_subjects), max(1, ood_subject_count))
    chosen_subjects = rng.sample(other_subjects, take_subjects)

    ood_pool: list[dict[str, object]] = []
    for subject in chosen_subjects:
        ood_pool.extend(load_mmlu_subject_prompts(dataset_id, subject, revision=revision))
    if len(ood_pool) < num_prompts:
        raise ValueError(
            f"MMLU OOD pool has {len(ood_pool)} prompts, needs {num_prompts}"
        )

    id_sample = [id_records[index] for index in rng.sample(range(len(id_records)), num_prompts)]
    ood_sample = [ood_pool[index] for index in rng.sample(range(len(ood_pool)), num_prompts)]

    return {
        f"id_mmlu_{id_subject}": id_sample,
        "mmlu_ood_other_concepts": ood_sample,
    }
