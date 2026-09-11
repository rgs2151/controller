"""Build the proposed long-context replacement from pinned public-domain books."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re

import requests
from transformers import AutoTokenizer


UNIT = Path(__file__).resolve().parent
CACHE = UNIT / "cache"
PREPARED = CACHE / "prepared.json"
OUTPUT = CACHE / "long_context_v2.json"
SOURCE_DIR = CACHE / "long_context_sources"

MODEL_ID = "google/gemma-2-2b"
MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
TARGET_INPUT_TOKENS = 7168
SEED = 2151
DOCUMENTS_PER_PROMPT = 7
TOKENS_PER_DOCUMENT = 1100

SOURCES = (
    (2701, "Moby-Dick", "907420db6c4b68c70e2988cd2ad9c8cf79138667a01b63376d18dd17fef1a18b"),
    (2600, "War and Peace", "2d5bb2ad5f422765e714617e21fa31bbaf8958aa79682c86fca6660fcc5d1b2b"),
    (1342, "Pride and Prejudice", "3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75"),
    (84, "Frankenstein", "7810cd483cffcf2cc8a1d8f0d5807931e69d4f48cd14149b8c76f88af82fead3"),
    (1661, "The Adventures of Sherlock Holmes", "922e2a12ccb43a4c9544c260b2166c6ad2097aeb5957faeee113f173bb857cd0"),
    (1228, "On the Origin of Species", "ededa9c0bf8761efed092c303b46c1c92de956838cba6249a33bedfd6d7363b4"),
    (1497, "The Republic", "917c1cb469e1a8eba6083808764d7131da8d79140b575b4214c9d02a73ec4528"),
    (98, "A Tale of Two Cities", "d54c2b80d40a40b982cd88852c6180bb944d95acdb028af3d0e01a1750681784"),
    (35, "The Time Machine", "2892e919000e17c83e1dac51b30f4675db50536b644d7579fe8a89bb399a9bdc"),
    (205, "Walden", "2d9a76a2e3e8195c69430516ebd33c4d0757a53ad432ff6186b7b794e6fe99f9"),
    (23, "Narrative of the Life of Frederick Douglass", "234c15348a66919bad1d534cdd48ee8ddf91f50115a706cbefaa8edd049672fd"),
    (1404, "The Federalist Papers", "0b92fadc9d7afd767eefd27805a20e44417ff26d1fb2f964229f39dcc7b2a65d"),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_url(ebook_id: int) -> str:
    return f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"


def _source_text(ebook_id: int, expected_sha256: str) -> str:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    path = SOURCE_DIR / f"pg{ebook_id}.txt"
    if path.exists():
        raw = path.read_bytes()
    else:
        response = requests.get(
            _source_url(ebook_id),
            headers={
                "User-Agent": (
                    "RobustSteerabilityResearch/0.1 "
                    "(research dataset construction)"
                )
            },
            timeout=60,
        )
        response.raise_for_status()
        raw = response.content
        path.write_bytes(raw)
    actual_sha256 = _sha256(raw)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Project Gutenberg source {ebook_id} changed: {actual_sha256}"
        )
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text)
    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text)
    if start is None or end is None or end.start() <= start.end():
        raise ValueError(f"Could not isolate Project Gutenberg text {ebook_id}")
    body = text[start.end() : end.start()].strip()
    return re.sub(r"\n{3,}", "\n\n", body)


def _fit_prompt(tokenizer, context: str, suffix: str) -> tuple[str, int]:
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    low, high = 0, len(context_ids)
    best_prompt = suffix.lstrip()
    best_count = len(tokenizer.encode(best_prompt, add_special_tokens=True))
    while low <= high:
        midpoint = (low + high) // 2
        candidate = tokenizer.decode(context_ids[:midpoint], skip_special_tokens=True).rstrip()
        prompt = candidate + suffix
        count = len(tokenizer.encode(prompt, add_special_tokens=True))
        if count <= TARGET_INPUT_TOKENS:
            best_prompt, best_count = prompt, count
            low = midpoint + 1
        else:
            high = midpoint - 1
    if best_count < TARGET_INPUT_TOKENS - 8:
        raise ValueError(f"Long prompt is unexpectedly short: {best_count} tokens")
    return best_prompt, best_count


def main() -> None:
    prepared = json.loads(PREPARED.read_text())
    id_records = prepared["prompt_sets"]["id"]
    if len(id_records) != 50:
        raise ValueError("Expected exactly 50 frozen TruthfulQA questions")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    source_tokens = {}
    for ebook_id, title, expected_sha256 in SOURCES:
        text = _source_text(ebook_id, expected_sha256)
        source_tokens[ebook_id] = tokenizer.encode(text, add_special_tokens=False)
        if len(source_tokens[ebook_id]) < 2 * TOKENS_PER_DOCUMENT:
            raise ValueError(f"Source is too short for sampling: {title}")

    records = []
    for index, record in enumerate(id_records):
        rng = random.Random(f"{SEED}:{record['source_prompt_id']}")
        selected = rng.sample(list(SOURCES), DOCUMENTS_PER_PROMPT)
        pieces = []
        source_spans = []
        for document_index, (ebook_id, title, expected_sha256) in enumerate(
            selected, start=1
        ):
            tokens = source_tokens[ebook_id]
            start = rng.randrange(0, len(tokens) - TOKENS_PER_DOCUMENT)
            excerpt = tokenizer.decode(
                tokens[start : start + TOKENS_PER_DOCUMENT],
                skip_special_tokens=True,
            ).strip()
            pieces.append(
                f"DOCUMENT {document_index}: {title}\n\n{excerpt}\n\n"
                f"END DOCUMENT {document_index}\n\n"
            )
            source_spans.append(
                {
                    "ebook_id": ebook_id,
                    "title": title,
                    "url": _source_url(ebook_id),
                    "sha256": expected_sha256,
                    "start_token": start,
                    "sampled_tokens": TOKENS_PER_DOCUMENT,
                }
            )
        suffix = f"\n\nQ: {record['question']} A:"
        prompt, token_count = _fit_prompt(tokenizer, "".join(pieces), suffix)
        records.append(
            {
                **record,
                "prompt_id": "long-v2:" + str(record["prompt_id"]),
                "prompt": prompt,
                "text": prompt,
                "construction": (
                    "7168-token multi-document public-domain distractor context"
                ),
                "input_tokens": token_count,
                "source_spans": source_spans,
            }
        )
        print(f"long context: {index + 1}/50 ({token_count} tokens)", flush=True)

    payload = {
        "identity": {
            "schema_version": 2,
            "status": "proposed_not_evaluated",
            "model": [MODEL_ID, MODEL_REVISION],
            "target_input_tokens": TARGET_INPUT_TOKENS,
            "seed": SEED,
            "documents_per_prompt": DOCUMENTS_PER_PROMPT,
            "sampled_tokens_per_document": TOKENS_PER_DOCUMENT,
            "source_license": "Project Gutenberg public-domain ebooks",
            "sources": [
                {
                    "ebook_id": ebook_id,
                    "title": title,
                    "url": _source_url(ebook_id),
                    "sha256": expected_sha256,
                }
                for ebook_id, title, expected_sha256 in SOURCES
            ],
        },
        "records": records,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
