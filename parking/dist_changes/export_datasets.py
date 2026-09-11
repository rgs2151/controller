"""Export the current frozen 50-prompt datasets into unit-local CSV files."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


UNIT = Path(__file__).resolve().parent
CACHE = UNIT / "cache"
DATASETS = CACHE / "datasets"
MODEL_ID = "google/gemma-2-2b"
MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"

SET_STATUS = {
    "id": "evaluated",
    "spanish": "evaluated_translation_requires_review",
    "japanese_romaji": "proposed_not_evaluated",
    "long_context_end": "proposed_not_evaluated",
    "long_context_start": "proposed_not_evaluated",
    "corrupting_words": "proposed_not_evaluated",
    "bos_mix": "proposed_not_evaluated",
    "lciteeval_complexity": "candidate_requires_lciteeval_evaluator",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sets() -> dict[str, list[dict[str, object]]]:
    prepared = _load_json(CACHE / "prepared.json")
    translations = _load_json(CACHE / "translations.json")
    japanese = _load_json(CACHE / "japanese_romaji.json")
    long_context = _load_json(CACHE / "long_context.json")
    attacks = _load_json(CACHE / "prompt_attacks.json")
    lciteeval = _load_json(CACHE / "lciteeval_complexity.json")
    result = {
        "id": prepared["prompt_sets"]["id"],
        "spanish": translations["rows"],
        "japanese_romaji": japanese["rows"],
        "long_context_end": long_context["sets"]["long_context_end"],
        "long_context_start": long_context["sets"]["long_context_start"],
        "corrupting_words": attacks["sets"]["corrupting_words"],
        "bos_mix": attacks["sets"]["bos_mix"],
        "lciteeval_complexity": lciteeval["records"],
    }
    if translations.get("status") != "complete":
        raise ValueError("Spanish translations are incomplete")
    if japanese.get("status") != "complete":
        raise ValueError("Japanese romaji translations are incomplete")
    if long_context["identity"].get("status") != "proposed_not_evaluated":
        raise ValueError("Unexpected long-context status")
    if attacks["identity"].get("status") != "proposed_not_evaluated":
        raise ValueError("Unexpected template-attack status")
    if lciteeval["identity"].get("status") != "candidate_requires_lciteeval_evaluator":
        raise ValueError("Unexpected L-CiteEval status")
    for name, records in result.items():
        if len(records) != 50:
            raise ValueError(f"{name} must contain exactly 50 prompts")
    paired_names = tuple(name for name in result if name != "lciteeval_complexity")
    anchors = {
        name: [str(record["source_prompt_id"]) for record in records]
        for name, records in result.items()
        if name in paired_names
    }
    if any(ids != anchors["id"] for ids in anchors.values()):
        raise ValueError("Distribution sets do not contain the same ordered questions")
    return result


def main() -> None:
    if DATASETS.exists():
        raise FileExistsError(
            f"Frozen dataset bundle already exists: {DATASETS}; evaluation reads it directly"
        )
    sets = _sets()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    DATASETS.mkdir(parents=True)
    manifest_sets = {}
    fieldnames = (
        "row",
        "set",
        "status",
        "source_prompt_id",
        "prompt_id",
        "source_question",
        "input_tokens",
        "prompt_sha256",
        "construction",
        "marker_repeats",
        "bos_token",
        "bos_token_id",
        "insertion_gaps",
        "paired_with_truthfulqa",
        "source_dataset",
        "source_config",
        "source_task",
        "hardness",
        "source_length",
        "document_count",
        "reference_answer_json",
        "prompt",
        "source_spans_json",
    )
    for set_name, records in sets.items():
        path = DATASETS / f"{set_name}.csv"
        hashes = []
        lengths = []
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, record in enumerate(records):
                prompt = str(record["prompt"])
                prompt_hash = _sha256_text(prompt)
                input_tokens = len(tokenizer.encode(prompt, add_special_tokens=True))
                if set_name in {
                    "long_context_end",
                    "long_context_start",
                    "lciteeval_complexity",
                }:
                    recorded_tokens = int(record["input_tokens"])
                    if input_tokens != recorded_tokens:
                        raise ValueError("Long-context token count changed during export")
                writer.writerow(
                    {
                        "row": index,
                        "set": set_name,
                        "status": SET_STATUS[set_name],
                        "source_prompt_id": record["source_prompt_id"],
                        "prompt_id": record["prompt_id"],
                        "source_question": record["question"],
                        "input_tokens": input_tokens,
                        "prompt_sha256": prompt_hash,
                        "construction": record.get("construction", "unchanged ID prompt"),
                        "marker_repeats": record.get("marker_repeats", ""),
                        "bos_token": record.get("bos_token", ""),
                        "bos_token_id": record.get("bos_token_id", ""),
                        "insertion_gaps": record.get("insertion_gaps", ""),
                        "paired_with_truthfulqa": record.get(
                            "paired_with_truthfulqa",
                            set_name != "lciteeval_complexity",
                        ),
                        "source_dataset": record.get(
                            "source_dataset", "truthful_qa"
                        ),
                        "source_config": record.get("source_config", "generation"),
                        "source_task": record.get("source_task", "truthfulness"),
                        "hardness": record.get("hardness", ""),
                        "source_length": record.get("source_length", ""),
                        "document_count": record.get("document_count", ""),
                        "reference_answer_json": json.dumps(
                            record.get("reference_answer"), ensure_ascii=False
                        )
                        if "reference_answer" in record
                        else "",
                        "prompt": prompt,
                        "source_spans_json": json.dumps(
                            record.get("source_spans", []),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }
                )
                hashes.append(prompt_hash)
                lengths.append(input_tokens)
        manifest_sets[set_name] = {
            "path": str(path.relative_to(UNIT)),
            "rows": len(records),
            "status": SET_STATUS[set_name],
            "minimum_input_tokens": min(lengths),
            "maximum_input_tokens": max(lengths),
            "ordered_prompt_hash": _sha256_text("\n".join(hashes)),
            "paired_with_truthfulqa": set_name != "lciteeval_complexity",
        }

    manifest = {
        "schema_version": 4,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "source_questions": (
            "Seven sets share the same ordered 50 held-out TruthfulQA questions; "
            "lciteeval_complexity contains 50 separate L-CiteEval questions"
        ),
        "long_context_formula": {
            "model_context_tokens": 8192,
            "target_fraction": 0.875,
            "target_input_tokens": 7168,
            "formula": (
                "floor(0.875 * model_context_tokens); one deterministic document "
                "prefix is cut so both question-at-end and question-at-start prompts fit"
            ),
        },
        "lciteeval_complexity": {
            "paired_with_truthfulqa": False,
            "evaluation": "L-CiteEval correctness and citation metrics",
        },
        "sets": manifest_sets,
    }
    (DATASETS / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
