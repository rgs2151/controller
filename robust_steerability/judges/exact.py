"""Deterministic scorers that do not require a judge model."""

from __future__ import annotations

import json
from pathlib import Path

from robust_steerability.datasets.mmlu import parse_mmlu_letter
from robust_steerability.judges.specs import scorer_cache_path, scorer_spec


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
