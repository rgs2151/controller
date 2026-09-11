"""Export the seven current 50-prompt distribution sets into unit-local CSV files."""

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
    "long_context": "proposed_not_evaluated",
    "d2": "proposed_not_evaluated",
    "d3": "proposed_not_evaluated",
    "d6": "proposed_not_evaluated",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sets() -> dict[str, list[dict[str, object]]]:
    prepared = _load_json(CACHE / "prepared.json")
    translations = _load_json(CACHE / "translations.json")
    japanese = _load_json(CACHE / "japanese_romaji.json")
    long_context = _load_json(CACHE / "long_context_v2.json")
    attacks = _load_json(CACHE / "template_attacks.json")
    result = {
        "id": prepared["prompt_sets"]["id"],
        "spanish": translations["rows"],
        "japanese_romaji": japanese["rows"],
        "long_context": long_context["records"],
        "d2": attacks["sets"]["d2"],
        "d3": attacks["sets"]["d3"],
        "d6": attacks["sets"]["d6"],
    }
    if translations.get("status") != "complete":
        raise ValueError("Spanish translations are incomplete")
    if japanese.get("status") != "complete":
        raise ValueError("Japanese romaji translations are incomplete")
    if long_context["identity"].get("status") != "proposed_not_evaluated":
        raise ValueError("Unexpected long-context status")
    if attacks["identity"].get("status") != "proposed_not_evaluated":
        raise ValueError("Unexpected template-attack status")
    for name, records in result.items():
        if len(records) != 50:
            raise ValueError(f"{name} must contain exactly 50 prompts")
    anchors = {
        name: [str(record["source_prompt_id"]) for record in records]
        for name, records in result.items()
    }
    if any(ids != anchors["id"] for ids in anchors.values()):
        raise ValueError("Distribution sets do not contain the same ordered questions")
    return result


def main() -> None:
    sets = _sets()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    DATASETS.mkdir(parents=True, exist_ok=True)
    retired = DATASETS / "adversarial.csv"
    if retired.exists():
        retired.unlink()
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
                if set_name == "long_context":
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
        }

    manifest = {
        "schema_version": 2,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "source_questions": (
            "the same ordered 50 held-out TruthfulQA questions in every set"
        ),
        "long_context_formula": {
            "model_context_tokens": 8192,
            "target_fraction": 0.875,
            "target_input_tokens": 7168,
            "formula": (
                "floor(0.875 * model_context_tokens); deterministic document text is "
                "cut with the model tokenizer so the unchanged Q/A suffix remains last"
            ),
        },
        "sets": manifest_sets,
    }
    (DATASETS / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
