"""Transfer the completed D2, D3, and D6 recipes onto TruthfulQA anchors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[1]
CACHE = UNIT / "cache"
SOURCE = REPO / "parking" / "ood_adversarial" / "cache" / "deep_search"
OUTPUT = CACHE / "template_attacks.json"
SEED = 2151
MODEL_ID = "google/gemma-2-2b"
MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
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


def _mix_bos(prompt: str, repeats: int, tokenizer) -> tuple[str, int]:
    words = prompt.split()
    gap_count = len(words) - 1
    allocations = [0] * gap_count
    for marker_index in range(repeats):
        gap_index = min(
            gap_count - 1,
            int((marker_index + 0.5) * gap_count / repeats),
        )
        allocations[gap_index] += 1
    pieces = [words[0]]
    for word, count in zip(words[1:], allocations, strict=True):
        pieces.extend([tokenizer.bos_token] * count)
        pieces.append(word)
    mixed = " ".join(pieces)
    actual = tokenizer.encode(mixed, add_special_tokens=False).count(
        tokenizer.bos_token_id
    )
    if actual != repeats:
        raise ValueError(f"Expected {repeats} Gemma BOS tokens; found {actual}")
    return mixed, gap_count


def _d6(records: list[dict[str, object]], tokenizer) -> list[dict[str, object]]:
    ordered_ids = sorted(str(record["source_prompt_id"]) for record in records)
    permutation = np.random.default_rng(SEED).permutation(len(ordered_ids))
    low = {ordered_ids[index] for index in permutation[: len(ordered_ids) // 2]}
    output = []
    for record in records:
        repeats = 16 if str(record["source_prompt_id"]) in low else 64
        prompt, gap_count = _mix_bos(str(record["prompt"]), repeats, tokenizer)
        output.append(
            {
                **record,
                "prompt_id": "d6:" + str(record["prompt_id"]),
                "prompt": prompt,
                "text": prompt,
                "marker_repeats": repeats,
                "bos_token": tokenizer.bos_token,
                "bos_token_id": tokenizer.bos_token_id,
                "insertion_gaps": gap_count,
                "construction": "D6 seeded 16/64 Gemma BOS-token mix distributed evenly across the prompt",
            }
        )
    return output


def main() -> None:
    prepared = json.loads((CACHE / "prepared.json").read_text())
    records = prepared["prompt_sets"]["id"]
    if len(records) != 50:
        raise ValueError("Expected exactly 50 TruthfulQA anchors")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    if tokenizer.bos_token != "<bos>" or tokenizer.bos_token_id != 2:
        raise ValueError("Pinned Gemma BOS token changed")
    attacks = {
        "d2": _text_attack(records, "text_only_transfer", "d2"),
        "d3": _text_attack(records, "shared_text_transfer", "d3"),
        "d6": _d6(records, tokenizer),
    }
    payload = {
        "identity": {
            "schema_version": 2,
            "status": "proposed_not_evaluated",
            "seed": SEED,
            "model": [MODEL_ID, MODEL_REVISION],
            "d6_bos_token": tokenizer.bos_token,
            "d6_bos_token_id": tokenizer.bos_token_id,
            "source": "parking/ood_adversarial completed D2, D3, and D6 recipes",
            "suffix_sha256": EXPECTED_SUFFIX_HASHES,
        },
        "sets": attacks,
    }
    if OUTPUT.exists():
        if json.loads(OUTPUT.read_text()) != payload:
            raise ValueError("Frozen template-attack cache differs from this construction")
        return
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
