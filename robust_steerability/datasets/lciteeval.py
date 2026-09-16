"""Pinned inputs for the Spanish-transfer L-CiteEval benchmark."""

from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from robust_steerability.benchmarks.layout import REPO_ROOT
from robust_steerability.datasets.mgsm import aligned_direction_pairs


LCITE_REPOSITORY = "Jonaszky123/L-CiteEval"
LCITE_REVISION = "c79c928529593f478e6573c969cf73d22f0cf0f9"
HOTPOT_FILE = "L-CiteEval-Length/hotpotqa.json"
TWOWIKI_FILE = "L-CiteEval-Data/2wikimultihopqa.json"
TWOWIKI_REPOSITORY = "framolfese/2WikiMultihopQA"
TWOWIKI_REVISION = "fe713bfbd1afbca1a65246741a75890405d56a3a"
SPANISH_CONCEPT = "respond only in Spanish, and no other language is allowed"
CONDITIONS = ("8k", "16k", "32k")
DIRECTION_PAIRS = 250
DISTURBANCE_PROMPTS = 200
TUNING_PROMPTS = 40
DISTURBANCE_TARGET_TOKENS = 7_500


def _source_file(filename: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=LCITE_REPOSITORY,
            repo_type="dataset",
            revision=LCITE_REVISION,
            filename=filename,
        )
    )


def load_length_rows() -> list[dict]:
    rows = json.loads(_source_file(HOTPOT_FILE).read_text())
    if len(rows) != 120 or len({str(row["question"]) for row in rows}) != 40:
        raise ValueError("Pinned L-CiteEval HotpotQA slice is not the expected 40 x 3 set")
    return rows


def load_2wiki_selection_rows() -> list[dict]:
    """Select the shortest official L-CiteEval context for each of 40 questions."""

    rows = json.loads(_source_file(TWOWIKI_FILE).read_text())
    grouped: dict[str, list[dict]] = {}
    order: list[str] = []
    for row in rows:
        question = str(row["question"])
        if question not in grouped:
            grouped[question] = []
            order.append(question)
        grouped[question].append(row)
    if len(rows) != 240 or len(grouped) != TUNING_PROMPTS:
        raise ValueError("Pinned L-CiteEval 2Wiki set is not the expected 40 x 6 set")
    return [min(grouped[question], key=lambda row: int(row["length"])) for question in order]


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


def official_prompt(item: dict, tokenizer, template: dict) -> str:
    content = _demo(template["demos"][0], template, test=False)
    content += str(template["demo_sep"])
    content += _demo(item, template, test=True)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def load_official_template(task: str) -> dict:
    filename = {
        "hotpotqa": "benchmarks/lciteeval/hotpotqa_prompt.json",
        "2wiki": "ref/L-CITEEVAL/prompt/2WikiMultihopQA_default.json",
    }[task]
    return json.loads((REPO_ROOT / filename).read_text())


def materialize_evaluation(tokenizer, *, context_window: int) -> dict:
    template = load_official_template("hotpotqa")
    rows = load_length_rows()
    question_order: dict[str, int] = {}
    records = {condition: [] for condition in CONDITIONS}
    for row in rows:
        question = str(row["question"])
        if question not in question_order:
            question_order[question] = len(question_order)
        matched_index = question_order[question]
        condition = condition_for_length(int(row["length"]))
        model_input = official_prompt(row, tokenizer, template)
        input_tokens = len(tokenizer(model_input, add_special_tokens=False)["input_ids"])
        if input_tokens + 200 > context_window:
            raise ValueError(
                f"L-CiteEval {row['id']} requires {input_tokens + 200} tokens, "
                f"exceeding the configured {context_window} token window"
            )
        records[condition].append(
            {
                "prompt_id": f"lcite-hotpot-{matched_index:02d}-{condition}",
                "source_id": str(row["id"]),
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
    for condition in CONDITIONS:
        indices = [int(row["matched_question_index"]) for row in records[condition]]
        if indices != list(range(40)):
            raise ValueError(f"L-CiteEval matched question order changed for {condition}")
    return {
        "schema_version": 1,
        "source": {"dataset": LCITE_REPOSITORY, "revision": LCITE_REVISION, "file": HOTPOT_FILE},
        "prompt": {"source": "L-CITEEVAL prompt/hotpotqa_default.json", "shot": 1},
        "evaluation": records,
    }


def _chat_question(tokenizer, question: str) -> str:
    options = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], enable_thinking=False, **options
        )
    except TypeError:
        return tokenizer.apply_chat_template([{"role": "user", "content": question}], **options)


def materialize_direction(tokenizer) -> dict[str, list[dict]]:
    pairs = aligned_direction_pairs()
    if len(pairs) != DIRECTION_PAIRS:
        raise ValueError("MGSM did not provide the frozen 250 direction pairs")
    return {
        "undesired": [
            {"prompt_id": f"{pair['pair_id']}-en", "pair_id": pair["pair_id"], "text": _chat_question(tokenizer, str(pair["english_question"]))}
            for pair in pairs
        ],
        "desired": [
            {"prompt_id": f"{pair['pair_id']}-es", "pair_id": pair["pair_id"], "text": _chat_question(tokenizer, str(pair["spanish_question"]))}
            for pair in pairs
        ],
    }


def _context_documents(row: dict) -> list[dict]:
    context = row.get("context")
    if isinstance(context, dict):
        titles = list(context.get("title", []))
        sentences = list(context.get("sentences", []))
        return [
            {"title": str(title), "text": f"{title}\n{' '.join(map(str, body))}"}
            for title, body in zip(titles, sentences, strict=True)
        ]
    documents = []
    for item in context or row.get("docs", []):
        if isinstance(item, dict):
            documents.append({"title": str(item.get("title", "")), "text": _document_text(item)})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            title, body = item
            text = " ".join(map(str, body)) if isinstance(body, list) else str(body)
            documents.append({"title": str(title), "text": f"{title}\n{text}"})
    if not documents:
        raise ValueError("Unsupported upstream 2WikiMultihopQA context schema")
    return documents


def _upstream_2wiki_rows() -> list[dict]:
    stream = load_dataset(
        TWOWIKI_REPOSITORY,
        split="train",
        revision=TWOWIKI_REVISION,
        streaming=True,
    ).shuffle(seed=42, buffer_size=10_000)
    rows = []
    for row in stream:
        rows.append(dict(row))
        if len(rows) == DISTURBANCE_PROMPTS * 2:
            break
    if len(rows) != DISTURBANCE_PROMPTS * 2:
        raise ValueError("Upstream 2WikiMultihopQA train stream is unexpectedly short")
    return rows


def _long_disturbance_record(source: dict, distractor_pool: list[tuple[dict, int]], tokenizer, template: dict, index: int, *, context_window: int) -> dict:
    docs = _context_documents(source)
    base_item = {"question": str(source["question"]), "answer": str(source.get("answer", "")), "docs": docs}
    base_tokens = len(tokenizer(official_prompt(base_item, tokenizer, template), add_special_tokens=False)["input_ids"])
    estimated = base_tokens
    pool = list(distractor_pool)
    random.Random(10_000 + index).shuffle(pool)
    for document, document_tokens in pool:
        if estimated >= DISTURBANCE_TARGET_TOKENS:
            break
        docs.append(document)
        estimated += document_tokens
    item = {**base_item, "docs": docs}
    model_input = official_prompt(item, tokenizer, template)
    input_tokens = len(tokenizer(model_input, add_special_tokens=False)["input_ids"])
    if input_tokens + 200 > context_window:
        raise ValueError(f"2Wiki disturbance prompt exceeds the {context_window}-token model window")
    return {
        "prompt_id": f"2wiki-train-disturbance-{index:03d}",
        "source_index": index,
        "question": str(source["question"]),
        "text": f"{template['instruction']}\n\nQuestion: {source['question']}",
        "model_input": model_input,
        "input_tokens": input_tokens,
    }


def materialize_h_infinity_splits(tokenizer, *, context_window: int) -> dict[str, list[dict]]:
    template = load_official_template("2wiki")
    upstream = _upstream_2wiki_rows()
    distractors = upstream[DISTURBANCE_PROMPTS:]
    distractor_pool = [
        (
            document,
            len(tokenizer(_document_text(document), add_special_tokens=False)["input_ids"]) + 8,
        )
        for row in distractors
        for document in _context_documents(row)
    ]
    disturbance = [
        _long_disturbance_record(row, distractor_pool, tokenizer, template, index, context_window=context_window)
        for index, row in enumerate(upstream[:DISTURBANCE_PROMPTS])
    ]
    tuning = []
    for index, row in enumerate(load_2wiki_selection_rows()):
        model_input = official_prompt(row, tokenizer, template)
        input_tokens = len(tokenizer(model_input, add_special_tokens=False)["input_ids"])
        if input_tokens + 200 > context_window:
            raise ValueError("Official 2Wiki tuning prompt exceeds the model context window")
        tuning.append(
            {
                "prompt_id": f"lcite-2wiki-tuning-{index:02d}",
                "source_id": str(row["id"]),
                "question": str(row["question"]),
                "text": f"{template['instruction']}\n\nQuestion: {row['question']}",
                "model_input": model_input,
                "input_tokens": input_tokens,
                "answer": row["answer"],
                "docs": row["docs"],
            }
        )
    return {"disturbance": disturbance, "tuning": tuning}
