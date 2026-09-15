"""Pinned MMLU loaders and exact-answer parsing."""

from __future__ import annotations

import random
from typing import Iterable

from datasets import get_dataset_config_names, load_dataset


LETTERS = "ABCD"
MMLU_ID = "cais/mmlu"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"


def _format_five_shot_question(row: dict[str, object], include_answer: bool) -> str:
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
    test_by_subject: dict[str, list[int]] = {}
    for row_index, row in enumerate(test):
        test_by_subject.setdefault(str(row["subject"]), []).append(row_index)
    subjects = sorted(test_by_subject)
    indices: list[int] = []
    selected: set[int] = set()
    while len(indices) < count:
        row_index = rng.choice(test_by_subject[rng.choice(subjects)])
        if row_index not in selected:
            indices.append(row_index)
            selected.add(row_index)
    output = []
    for row_index in indices:
        row = dict(test[row_index])
        answer = int(row["answer"])
        if answer < 0 or answer >= len(LETTERS):
            raise ValueError(f"Invalid MMLU answer for test row {row_index}")
        subject = str(row["subject"])
        examples = rng.sample(dev_by_subject[subject], shots)
        demonstrations = "\n\n".join(
            _format_five_shot_question(example, True) for example in examples
        )
        prompt = demonstrations + "\n\n" if demonstrations else ""
        prompt += _format_five_shot_question(row, False)
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


def parse_mmlu_letter(text: str) -> int | None:
    """Accept one answer letter, never a letter embedded in a word or sentence."""

    answer = text.strip().upper()
    return LETTERS.index(answer) if answer in tuple(LETTERS) else None


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
