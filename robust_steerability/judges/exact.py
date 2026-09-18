"""Deterministic scorers that do not require a judge model."""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
import re
import string

from robust_steerability.datasets.mmlu import parse_mmlu_letter
from robust_steerability.judges.specs import scorer_cache_path, scorer_spec


def remove_citations(text: str) -> str:
    """Apply the citation removal used by L-CiteEval's answer scorer."""

    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", text)).replace(
        " |", ""
    ).replace("]", "")


def normalize_lcite_answer(text: str) -> str:
    """Normalize an answer exactly along L-CiteEval's HotpotQA path."""

    lowered = text.lower()
    unpunctuated = "".join(character for character in lowered if character not in string.punctuation)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", unpunctuated)
    return " ".join(without_articles.split())


def lcite_answer_overlap(prediction: str, answer: str | int | list[str]) -> dict[str, float]:
    """Return the released L-CiteEval token-overlap metrics for one answer."""

    prediction = remove_citations(prediction.strip().split("\n", 1)[0])
    gold_answers = answer if isinstance(answer, list) else [str(answer)]
    best = {"answer_precision": 0.0, "answer_recall": 0.0, "answer_f1": 0.0}
    for gold in gold_answers:
        predicted_tokens = normalize_lcite_answer(prediction).split()
        gold_tokens = normalize_lcite_answer(str(gold)).split()
        common = Counter(predicted_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        precision = overlap / len(predicted_tokens) if predicted_tokens else 0.0
        recall = overlap / len(gold_tokens) if gold_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        best["answer_precision"] = max(best["answer_precision"], precision)
        best["answer_recall"] = max(best["answer_recall"], recall)
        best["answer_f1"] = max(best["answer_f1"], f1)
    return best


def harmonic_mean(scores: list[float]) -> float:
    """AXBench aggregation: zero if any component is zero."""

    if not scores or any(score == 0 for score in scores):
        return 0.0
    return len(scores) / sum(1.0 / score for score in scores)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def score_multiple_choice_generation(
    generation_path: Path,
    root: Path,
    scorer_key: str = "mmlu_accuracy",
) -> Path:
    spec = scorer_spec(scorer_key)
    if spec.backend != "exact_multiple_choice":
        raise ValueError(f"{scorer_key!r} is not an exact multiple-choice scorer")
    generation = json.loads(generation_path.read_text())
    if generation.get("status") != "complete":
        raise ValueError(f"Generation is incomplete: {generation_path}")
    destination = scorer_cache_path(root, generation_path, scorer_key)
    if destination.exists() and json.loads(destination.read_text()).get("status") == "complete":
        return destination
    rows = []
    for repetition in generation["repetitions"]:
        for row in repetition["rows"]:
            prediction = parse_mmlu_letter(str(row["completion"]))
            answer = int(row["answer_index"])
            rows.append(
                {
                    "prompt_id": str(row["prompt_id"]),
                    "repetition": int(repetition["repetition"]),
                    "prediction_index": prediction,
                    "answer_index": answer,
                    "score": float(prediction == answer),
                    "valid": prediction is not None,
                }
            )
    _write_json(
        destination,
        {
            "scorer": {
                "schema_version": 1,
                "scorer_key": scorer_key,
                "backend": spec.backend,
                "metric": spec.metric,
                "rubric": spec.rubric,
            },
            "status": "complete",
            "rows": rows,
        },
    )
    return destination
