"""Pinned ten-question L-CiteEval-Length and AXBench concept inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from robust_steerability.benchmarks.layout import REPO_ROOT


LCITE_REPOSITORY = "Jonaszky123/L-CiteEval"
LCITE_REVISION = "c79c928529593f478e6573c969cf73d22f0cf0f9"
LCITE_FILE = "L-CiteEval-Length/hotpotqa.json"
AXBENCH_REPOSITORY = "pyvene/axbench-concept500"
AXBENCH_REVISION = "ad8a5d60c4616b599c24dd6689f05f696ec610f3"
AXBENCH_VARIANT = "prod_9b_l20_v1"
AXBENCH_CONCEPT_ID = 499
AXBENCH_CONCEPT = "positive sentiments and descriptions of enjoyable experiences"
AXBENCH_SHA256 = "5fb3042cc484a3e194f1043aab047bc1cfbe4efe884131d0113c75e9d2da599a"
ALPACA_REPOSITORY = "tatsu-lab/alpaca_eval"
ALPACA_REVISION = "2edc6fad8be6b14ea7230aabfd08188da6b8b814"
ALPACA_FILE = "alpaca_eval.json"
ALPACA_SHA256 = "d92b92c51e8f1962a21193abe74e6f727c2bc8286035f4041505ff38a7c3ae51"
TUNING_SELECTION_SHA256 = "2991b504931d0a2f74539b0db43a775366b4c4d2dc6773b59b52ba380258ed5d"
CONDITIONS = ("8k", "32k")
EVALUATION_QUESTIONS = 10
MAX_NEW_TOKENS = 128


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_file(repository: str, revision: str, filename: str) -> Path:
    return Path(
        hf_hub_download(repo_id=repository, repo_type="dataset", revision=revision, filename=filename)
    )


def load_length_rows() -> list[dict]:
    rows = json.loads(_source_file(LCITE_REPOSITORY, LCITE_REVISION, LCITE_FILE).read_text())
    if len(rows) != 120 or len({str(row["question"]) for row in rows}) != 40:
        raise ValueError("Pinned L-CiteEval HotpotQA slice is not the expected 40 x 3 set")
    return rows


def condition_for_length(length: int) -> str:
    if length < 8_000:
        return "8k"
    if length < 16_000:
        return "16k"
    return "32k"


def _document_text(document: str | dict) -> str:
    if isinstance(document, str):
        return document
    title = str(document.get("title", ""))
    text = str(document["text"]).strip("\n")
    if title and text.startswith(title + "\n"):
        text = text[len(title) + 1 :]
    return text


def _demo(item: dict, template: dict, *, test: bool) -> str:
    documents = "".join(
        template["doc_prompt"].replace("{P}", _document_text(document)).replace("{ID}", str(index + 1))
        for index, document in enumerate(item["docs"][:1500])
    )
    rendered = (
        template["demo_prompt"]
        .replace("{D}", documents)
        .replace("{INST}", template["instruction"])
        .replace("{Q}", str(item["question"]))
        .replace("{A}", "")
        .rstrip()
    )
    if test:
        return rendered
    answer = item["answer"]
    answer_text = "\n" + "\n".join(answer) if isinstance(answer, list) else str(answer)
    return rendered + answer_text


def official_hotpot_prompt(item: dict, tokenizer, template: dict) -> str:
    """Reproduce the released one-shot prompt and target-model chat template."""

    content = _demo(template["demos"][0], template, test=False)
    content += str(template["demo_sep"])
    content += _demo(item, template, test=True)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def load_official_template() -> dict:
    return json.loads(
        (REPO_ROOT / "benchmarks/lciteeval/hotpotqa_prompt.json").read_text()
    )


def materialize_evaluation(tokenizer, *, context_window: int) -> dict:
    template = load_official_template()
    rows = load_length_rows()
    question_order: dict[str, int] = {}
    records = {condition: [] for condition in CONDITIONS}
    for row in rows:
        question = str(row["question"])
        if question not in question_order:
            question_order[question] = len(question_order)
        matched_index = question_order[question]
        condition = condition_for_length(int(row["length"]))
        if condition not in CONDITIONS or matched_index >= EVALUATION_QUESTIONS:
            continue
        model_input = official_hotpot_prompt(row, tokenizer, template)
        input_tokens = len(tokenizer(model_input, add_special_tokens=False)["input_ids"])
        if input_tokens + MAX_NEW_TOKENS > context_window:
            raise ValueError(
                f"L-CiteEval {row['id']} requires {input_tokens + MAX_NEW_TOKENS} tokens, "
                f"exceeding the configured {context_window} token window"
            )
        records[condition].append(
            {
                "prompt_id": f"lcite-hotpot-{matched_index:02d}-{condition}",
                "source_id": int(row["id"]),
                "matched_question_index": matched_index,
                "condition": condition,
                "reported_length": int(row["length"]),
                "input_tokens": input_tokens,
                "question": question,
                "instruction": str(template["instruction"]),
                "text": f"{template['instruction']}\n\nQuestion: {question}",
                "model_input": model_input,
                "answer": row["answer"],
                "docs": row["docs"],
            }
        )
    expected_indices = list(range(EVALUATION_QUESTIONS))
    for condition in CONDITIONS:
        indices = [int(row["matched_question_index"]) for row in records[condition]]
        if indices != expected_indices:
            raise ValueError(f"L-CiteEval matched question order changed for {condition}")
    return {
        "schema_version": 1,
        "source": {"dataset": LCITE_REPOSITORY, "revision": LCITE_REVISION, "file": LCITE_FILE},
        "prompt": {"source": "L-CITEEVAL prompt/hotpotqa_default.json", "shot": 1},
        "evaluation": records,
    }


def _axbench_parquet() -> Path:
    local = (
        REPO_ROOT
        / "ref/axbench/axbench/concept500"
        / AXBENCH_VARIANT
        / "generate/train_data.parquet"
    )
    if not local.exists():
        filename = f"{AXBENCH_VARIANT}/generate/train_data.parquet"
        local = _source_file(AXBENCH_REPOSITORY, AXBENCH_REVISION, filename)
    if _sha256(local) != AXBENCH_SHA256:
        raise ValueError("AXBench concept parquet does not match its frozen SHA-256")
    return local


def axbench_direction_rows() -> dict[str, list[dict]]:
    frame = pd.read_parquet(_axbench_parquet()).reset_index(names="source_row")
    desired = frame[(frame["concept_id"] == AXBENCH_CONCEPT_ID) & (frame["category"] == "positive")]
    genres = desired["concept_genre"].unique().tolist()
    if len(desired) != 72 or genres != ["text"]:
        raise ValueError("AXBench concept 499 does not match the frozen 72-row text concept")
    undesired = frame[
        (frame["concept_id"] == -1)
        & (frame["category"] == "negative")
        & (frame["concept_genre"] == genres[0])
    ]
    if len(undesired) != 72:
        raise ValueError("AXBench genre-matched negative set is not the expected 72 rows")

    def records(label: str, selected: pd.DataFrame) -> list[dict]:
        return [
            {
                "prompt_id": f"axbench-499-{label}-{int(row.source_row):04d}",
                "prompt": str(row.input),
                "response": str(row.output),
                "source_row": int(row.source_row),
            }
            for row in selected.itertuples(index=False)
        ]

    return {"desired": records("desired", desired), "undesired": records("undesired", undesired)}


def _alpaca_frame() -> pd.DataFrame:
    path = _source_file(ALPACA_REPOSITORY, ALPACA_REVISION, ALPACA_FILE)
    if _sha256(path) != ALPACA_SHA256:
        raise ValueError("Pinned AlpacaEval source does not match its frozen SHA-256")
    return pd.DataFrame(json.loads(path.read_text())).reset_index(names="source_row")


def h_infinity_prompt_splits() -> dict[str, list[dict]]:
    """Return 200 disturbance and the frozen 50 tuning prompts without overlap."""

    frame = _alpaca_frame()
    tuning_frame = frame.sample(n=50, replace=False, random_state=499)
    canonical = [
        {"source_index": int(row.source_row), "instruction": str(row.instruction)}
        for row in tuning_frame.itertuples(index=False)
    ]
    serialized = json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode()
    if hashlib.sha256(serialized).hexdigest() != TUNING_SELECTION_SHA256:
        raise ValueError("Frozen 50-prompt H-infinity tuning selection changed")
    disturbance_frame = frame.drop(index=tuning_frame.index).sample(
        n=200, replace=False, random_state=500
    )

    def records(prefix: str, selected: pd.DataFrame) -> list[dict]:
        return [
            {
                "prompt_id": f"alpaca-eval-{prefix}-{int(row.source_row):04d}",
                "text": str(row.instruction),
                "source_index": int(row.source_row),
            }
            for row in selected.itertuples(index=False)
        ]

    return {
        "disturbance": records("disturbance", disturbance_frame),
        "tuning": records("tuning", tuning_frame),
    }
