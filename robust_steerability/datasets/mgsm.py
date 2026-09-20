"""Pinned MGSM inputs for multilingual Spanish-steering transfer."""

from __future__ import annotations

import random
import json
from pathlib import Path

from datasets import load_dataset


MGSM_ID = "juletxara/mgsm"
MGSM_REVISION = "b2f13d426afe3be8d69a7e739b36724db8b66bbc"
GSM8K_ID = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
LANGUAGES = ("en", "es", "bn", "zh", "fr", "de", "ja", "ru", "sw", "te", "th")
TRANSFER_LANGUAGES = tuple(language for language in LANGUAGES if language not in {"en", "es"})
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "bn": "Bengali",
    "zh": "Chinese",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "ru": "Russian",
    "sw": "Swahili",
    "te": "Telugu",
    "th": "Thai",
}
MULTILINGUAL_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "data/mgsm/h_infinity_multilingual_calibration.json"
)
ANSWER_LABELS = {
    "en": "Step-by-Step Answer:",
    "es": "Respuesta paso a paso:",
    "bn": "ধাপে ধাপে উত্তর:",
    "zh": "逐步解答：",
    "fr": "Réponse étape par étape :",
    "de": "Schritt-für-Schritt-Antwort:",
    "ja": "ステップごとの答え:",
    "ru": "Пошаговоерешение:",
    "sw": "Jibu la Hatua kwa Hatua:",
    "te": "దశలవారీగా సమాధానం:",
    "th": "คำตอบทีละขั้นตอน:",
}


def load_mgsm(language: str, split: str) -> list[dict]:
    """Load one pinned MGSM language configuration."""

    if language not in LANGUAGES:
        raise ValueError(f"Unsupported MGSM language {language!r}")
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported MGSM split {split!r}")
    return list(
        load_dataset(
            MGSM_ID,
            language,
            split=split,
            revision=MGSM_REVISION,
        )
    )


def aligned_direction_pairs() -> list[dict]:
    """Return all 250 answer-aligned English/Spanish question pairs."""

    english = load_mgsm("en", "test")
    spanish = load_mgsm("es", "test")
    if len(english) != 250 or len(spanish) != 250:
        raise ValueError("Pinned MGSM English/Spanish test sets are not 250 rows")
    pairs = []
    for index, (negative, positive) in enumerate(zip(english, spanish, strict=True)):
        if int(negative["answer_number"]) != int(positive["answer_number"]):
            raise ValueError(f"MGSM answer alignment failed at row {index}")
        pairs.append(
            {
                "pair_id": f"mgsm-pair-{index:03d}",
                "problem_index": index,
                "answer_number": int(negative["answer_number"]),
                "english_question": str(negative["question"]),
                "spanish_question": str(positive["question"]),
            }
        )
    return pairs


def native_cot_prompt(language: str, exemplars: list[dict], question: str) -> str:
    """Reproduce MGSM's native-language eight-shot CoT text format."""

    if len(exemplars) != 8:
        raise ValueError(f"MGSM native CoT requires eight {language} exemplars")
    demonstrations = [
        f"{str(row['question']).strip()}\n{str(row['answer']).strip()}"
        for row in exemplars
    ]
    demonstrations.append(f"{question.strip()}\n{ANSWER_LABELS[language]}")
    return "\n\n".join(demonstrations)


def materialize_evaluation(tokenizer) -> dict:
    """Build the same 250 problem identities in all eleven input languages."""

    evaluation: dict[str, list[dict]] = {}
    expected_answers: list[int] | None = None
    for language in LANGUAGES:
        exemplars = load_mgsm(language, "train")
        rows = load_mgsm(language, "test")
        if len(exemplars) != 8 or len(rows) != 250:
            raise ValueError(f"Pinned MGSM {language} split sizes changed")
        answers = [int(row["answer_number"]) for row in rows]
        if expected_answers is None:
            expected_answers = answers
        elif answers != expected_answers:
            raise ValueError(f"MGSM problem order is not aligned for {language}")
        records = []
        for index, row in enumerate(rows):
            task_text = native_cot_prompt(language, exemplars, str(row["question"]))
            model_input = tokenizer.apply_chat_template(
                [{"role": "user", "content": task_text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            records.append(
                {
                    "prompt_id": f"mgsm-{index:03d}-{language}",
                    "problem_index": index,
                    "language": language,
                    "language_name": LANGUAGE_NAMES[language],
                    "question": str(row["question"]),
                    "text": str(row["question"]),
                    "model_input": model_input,
                    "input_tokens": len(
                        tokenizer(model_input, add_special_tokens=False)["input_ids"]
                    ),
                    "answer_number": int(row["answer_number"]),
                }
            )
        evaluation[language] = records
    return {
        "schema_version": 1,
        "source": {
            "dataset": MGSM_ID,
            "revision": MGSM_REVISION,
            "train_examples_per_language": 8,
            "test_examples_per_language": 250,
        },
        "prompt": {
            "style": "MGSM native-language eight-shot chain of thought",
            "thinking_mode": "disabled",
        },
        "evaluation": evaluation,
    }


def gsm8k_calibration_splits(seed: int = 42) -> dict[str, list[dict]]:
    """Freeze 200 disturbance and 50 disjoint H-infinity tuning questions."""

    rows = list(
        load_dataset(
            GSM8K_ID,
            "main",
            split="train",
            revision=GSM8K_REVISION,
        )
    )
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    disturbance_indices = indices[:200]
    tuning_indices = indices[200:250]

    def records(prefix: str, selected: list[int]) -> list[dict]:
        return [
            {
                "prompt_id": f"gsm8k-{prefix}-{index:04d}",
                "source_index": index,
                "text": (
                    "Solve this math problem. Show the reasoning, then give the final "
                    "Arabic-numeral answer.\n\n" + str(rows[index]["question"])
                ),
                "answer": str(rows[index]["answer"]),
            }
            for index in selected
        ]

    return {
        "disturbance": records("disturbance", disturbance_indices),
        "tuning": records("tuning", tuning_indices),
    }


def multilingual_calibration_splits() -> dict[str, list[dict]]:
    """Load the frozen four-language H-infinity calibration prompts."""

    payload = json.loads(MULTILINGUAL_CALIBRATION_PATH.read_text())
    splits = payload["splits"]
    if len(splits["disturbance"]) != 200 or len(splits["tuning"]) != 50:
        raise ValueError("Frozen multilingual MGSM calibration counts changed")
    return splits
