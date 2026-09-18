"""Freeze 50 Gemma-compatible L-CiteEval questions spanning three hardness levels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import pandas as pd
import requests
from transformers import AutoTokenizer


UNIT = Path(__file__).resolve().parent
CACHE = UNIT / "cache"
SOURCE_DIR = CACHE / "lciteeval_source"
OUTPUT = CACHE / "lciteeval_complexity.json"

MODEL_ID = "google/gemma-2-2b"
MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
DATASET_ID = "Jonaszky123/L-CiteEval"
DATASET_REVISION = "c79c928529593f478e6573c969cf73d22f0cf0f9"
SEED = 2151
MAX_INPUT_TOKENS = 8000

INSTRUCTION = (
    "Write an accurate and concise answer to the question using only the provided "
    "passages. Every sentence must end with citations to the supporting passage "
    "numbers, such as [1] or [1][2]."
)

SOURCES = {
    "narrativeqa": {
        "url": (
            "https://huggingface.co/datasets/Jonaszky123/L-CiteEval/resolve/"
            "refs%2Fconvert%2Fparquet/L-CiteEval-Hardness_narrativeqa/"
            "test/0000.parquet"
        ),
        "sha256": "14e577e28f599310b8ba275f15347ea4200060c91bc9770fd1876083674d7244",
        "quotas": {"easy": 8, "medium": 9, "hard": 8},
    },
    "locomo": {
        "url": (
            "https://huggingface.co/datasets/Jonaszky123/L-CiteEval/resolve/"
            "refs%2Fconvert%2Fparquet/L-CiteEval-Hardness_locomo/"
            "test/0000.parquet"
        ),
        "sha256": "099399c690f443a045e01df653625aa8f302dc2f37c3bf598061d9eaee9359af",
        "quotas": {"easy": 9, "medium": 7, "hard": 9},
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(task: str) -> Path:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    path = SOURCE_DIR / f"{task}.parquet"
    specification = SOURCES[task]
    if not path.exists():
        response = requests.get(str(specification["url"]), timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    actual = _sha256(path)
    if actual != specification["sha256"]:
        raise ValueError(f"L-CiteEval source changed for {task}: {actual}")
    return path


def _prompt(docs: list[str], question: str) -> str:
    passages = "\n".join(
        f"Passage [{index}]: {str(document).strip()}"
        for index, document in enumerate(docs, start=1)
    )
    return f"{passages}\n\n{INSTRUCTION}\n\nQuestion: {question}\nAnswer:"


def _answer(value):
    return value.tolist() if hasattr(value, "tolist") else value


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    selected = []
    for task, specification in SOURCES.items():
        frame = pd.read_parquet(_source(task))
        candidates = {hardness: [] for hardness in specification["quotas"]}
        for row in frame.to_dict(orient="records"):
            question = str(row["question"]).strip()
            docs = list(row["docs"])
            prompt = _prompt(docs, question)
            input_tokens = len(tokenizer.encode(prompt, add_special_tokens=True))
            hardness = str(row["hardness"])
            if input_tokens > MAX_INPUT_TOKENS:
                continue
            source_id = f"lciteeval:{task}:{int(row['id'])}"
            candidates[hardness].append(
                {
                    "prompt_id": source_id,
                    "source_prompt_id": source_id,
                    "question": question,
                    "prompt": prompt,
                    "text": prompt,
                    "input_tokens": input_tokens,
                    "construction": (
                        f"L-CiteEval-Hardness {task} {hardness} example with all "
                        "provided passages"
                    ),
                    "source_dataset": DATASET_ID,
                    "source_config": f"L-CiteEval-Hardness_{task}",
                    "source_task": task,
                    "hardness": hardness,
                    "source_length": int(row["length"]),
                    "document_count": len(docs),
                    "reference_answer": _answer(row["answer"]),
                    "paired_with_truthfulqa": False,
                }
            )
        for hardness, quota in specification["quotas"].items():
            pool = sorted(candidates[hardness], key=lambda record: record["prompt_id"])
            if len(pool) < quota:
                raise ValueError(
                    f"Only {len(pool)} Gemma-compatible {task}/{hardness} rows; "
                    f"need {quota}"
                )
            selected.extend(
                random.Random(f"{SEED}:{task}:{hardness}").sample(pool, quota)
            )

    random.Random(SEED).shuffle(selected)
    if len(selected) != 50:
        raise ValueError(f"Expected 50 L-CiteEval rows; found {len(selected)}")
    payload = {
        "identity": {
            "schema_version": 1,
            "status": "proposed_not_evaluated",
            "seed": SEED,
            "model": [MODEL_ID, MODEL_REVISION],
            "dataset": [DATASET_ID, DATASET_REVISION],
            "source_sha256": {
                task: specification["sha256"]
                for task, specification in SOURCES.items()
            },
            "maximum_input_tokens": MAX_INPUT_TOKENS,
            "selection": {
                task: specification["quotas"]
                for task, specification in SOURCES.items()
            },
            "paired_with_truthfulqa": False,
            "evaluation": "pinned TruthfulQA Truth and Info judges",
        },
        "records": selected,
    }
    if OUTPUT.exists():
        if json.loads(OUTPUT.read_text()) != payload:
            raise ValueError("Frozen L-CiteEval cache differs from this construction")
        return
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
