"""Transfer the completed D2, D3, and D6 recipes onto TruthfulQA anchors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[1]
CACHE = UNIT / "cache"
SOURCE = REPO / "parking" / "ood_adversarial" / "cache" / "deep_search"
OUTPUT = CACHE / "template_attacks.json"
SEED = 2151
EXPECTED_SUFFIX_HASHES = {
    "text_only_transfer": "974c878f9b0d6da3a906f43ee71723e3244e1bbac2a6afdb08e383dc5b5f4aa6",
    "shared_text_transfer": "3221da102e105bb48f3c6775971c795274c45b60cd9353e2d9558edb431bd7c3",
}


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _suffix(tag: str) -> str:
    path = SOURCE / tag / "config.json"
    payload = json.loads(path.read_text())
    suffix = str(payload["seeds"][0])
    if _sha256(suffix) != EXPECTED_SUFFIX_HASHES[tag]:
        raise ValueError(f"{tag} suffix does not match the completed attack")
    return suffix


def _text_attack(
    records: list[dict[str, object]], tag: str, prefix: str
) -> list[dict[str, object]]:
    suffix = _suffix(tag)
    return [
        {
            **record,
            "prompt_id": f"{prefix}:" + str(record["prompt_id"]),
            "prompt": str(record["prompt"]) + suffix,
            "text": str(record["prompt"]) + suffix,
            "construction": f"exact Llama-3.2-1B {tag} suffix transferred from ood_adversarial",
        }
        for record in records
    ]


def _d6(records: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered_ids = sorted(str(record["source_prompt_id"]) for record in records)
    permutation = np.random.default_rng(SEED).permutation(len(ordered_ids))
    low = {ordered_ids[index] for index in permutation[: len(ordered_ids) // 2]}
    output = []
    for record in records:
        repeats = 16 if str(record["source_prompt_id"]) in low else 64
        prompt = str(record["prompt"]) + "<|begin_of_text|>" * repeats
        output.append(
            {
                **record,
                "prompt_id": "d6:" + str(record["prompt_id"]),
                "prompt": prompt,
                "text": prompt,
                "marker_repeats": repeats,
                "construction": "D6 seeded 16/64 mix of the exact Llama begin-of-text marker string",
            }
        )
    return output


def main() -> None:
    prepared = json.loads((CACHE / "prepared.json").read_text())
    records = prepared["prompt_sets"]["id"]
    if len(records) != 50:
        raise ValueError("Expected exactly 50 TruthfulQA anchors")
    attacks = {
        "d2": _text_attack(records, "text_only_transfer", "d2"),
        "d3": _text_attack(records, "shared_text_transfer", "d3"),
        "d6": _d6(records),
    }
    OUTPUT.write_text(
        json.dumps(
            {
                "identity": {
                    "schema_version": 1,
                    "status": "proposed_not_evaluated",
                    "seed": SEED,
                    "source": "parking/ood_adversarial completed D2, D3, and D6 recipes",
                    "suffix_sha256": EXPECTED_SUFFIX_HASHES,
                },
                "sets": attacks,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
