"""Pinned inputs for the ten-question Spanish L-CiteEval Small benchmark."""

from __future__ import annotations

from datasets import load_dataset

from robust_steerability.datasets.lciteeval_spanish import (
    CONDITIONS,
    LCITE_REPOSITORY,
    LCITE_REVISION,
    HOTPOT_FILE as LCITE_FILE,
    SPANISH_CONCEPT,
    TWOWIKI_REPOSITORY,
    TWOWIKI_REVISION,
    _context_documents,
    _document_text,
    _long_disturbance_record,
    load_2wiki_selection_rows,
    load_official_template,
    materialize_direction,
    materialize_evaluation as materialize_full_evaluation,
    official_prompt,
)


EVALUATION_QUESTIONS = 10
MAX_NEW_TOKENS = 128
JACOBIAN_PROMPTS = 50
DISTURBANCE_PROMPTS = 200
TUNING_PROMPTS = 10
UPSTREAM_DISTRACTOR_ROWS = 200


def materialize_evaluation(tokenizer, *, context_window: int) -> dict:
    """Keep ten matched HotpotQA identities at every configurable length."""

    payload = materialize_full_evaluation(tokenizer, context_window=context_window)
    payload["evaluation"] = {
        condition: rows[:EVALUATION_QUESTIONS]
        for condition, rows in payload["evaluation"].items()
    }
    for condition, rows in payload["evaluation"].items():
        expected = list(range(EVALUATION_QUESTIONS))
        actual = [int(row["matched_question_index"]) for row in rows]
        if actual != expected:
            raise ValueError(f"L-CiteEval Small matched identities changed for {condition}")
    payload["schema_version"] = 2
    payload["sample_policy"] = "first 10 matched identities from the pinned 40 x 3 release"
    return payload


def _upstream_rows() -> list[dict]:
    required = JACOBIAN_PROMPTS + DISTURBANCE_PROMPTS + UPSTREAM_DISTRACTOR_ROWS
    stream = load_dataset(
        TWOWIKI_REPOSITORY,
        split="train",
        revision=TWOWIKI_REVISION,
        streaming=True,
    ).shuffle(seed=42, buffer_size=10_000)
    rows = []
    for row in stream:
        rows.append(dict(row))
        if len(rows) == required:
            break
    if len(rows) != required:
        raise ValueError("Upstream 2WikiMultihopQA train stream is unexpectedly short")
    return rows


def materialize_h_infinity_splits(tokenizer, *, context_window: int) -> dict[str, list[dict]]:
    """Build disjoint 8K Jacobian, disturbance, and tuning splits."""

    template = load_official_template("2wiki")
    upstream = _upstream_rows()
    jacobian_sources = upstream[:JACOBIAN_PROMPTS]
    disturbance_sources = upstream[
        JACOBIAN_PROMPTS : JACOBIAN_PROMPTS + DISTURBANCE_PROMPTS
    ]
    distractor_sources = upstream[JACOBIAN_PROMPTS + DISTURBANCE_PROMPTS :]
    distractor_pool = [
        (
            document,
            len(tokenizer(_document_text(document), add_special_tokens=False)["input_ids"]) + 8,
        )
        for row in distractor_sources
        for document in _context_documents(row)
    ]

    def long_records(sources: list[dict], prefix: str, seed_offset: int) -> list[dict]:
        records = []
        for index, source in enumerate(sources):
            record = _long_disturbance_record(
                source,
                distractor_pool,
                tokenizer,
                template,
                seed_offset + index,
                context_window=context_window,
            )
            record["prompt_id"] = f"2wiki-train-{prefix}-{index:03d}"
            record["source_index"] = seed_offset + index
            record["text"] = record["model_input"]
            records.append(record)
        return records

    tuning = []
    for index, row in enumerate(load_2wiki_selection_rows()[:TUNING_PROMPTS]):
        model_input = official_prompt(row, tokenizer, template)
        input_tokens = len(tokenizer(model_input, add_special_tokens=False)["input_ids"])
        if input_tokens + MAX_NEW_TOKENS > context_window:
            raise ValueError("Official 2Wiki tuning prompt exceeds the model context window")
        tuning.append(
            {
                "prompt_id": f"lcite-2wiki-tuning-{index:02d}",
                "source_id": str(row["id"]),
                "question": str(row["question"]),
                "instruction": str(template["instruction"]),
                "text": f"{template['instruction']}\n\nQuestion: {row['question']}",
                "model_input": model_input,
                "input_tokens": input_tokens,
                "answer": row["answer"],
                "docs": row["docs"],
            }
        )

    return {
        "jacobian": long_records(jacobian_sources, "jacobian", 0),
        "disturbance": long_records(disturbance_sources, "disturbance", JACOBIAN_PROMPTS),
        "tuning": tuning,
    }


__all__ = (
    "CONDITIONS",
    "DISTURBANCE_PROMPTS",
    "EVALUATION_QUESTIONS",
    "JACOBIAN_PROMPTS",
    "LCITE_FILE",
    "LCITE_REPOSITORY",
    "LCITE_REVISION",
    "MAX_NEW_TOKENS",
    "SPANISH_CONCEPT",
    "TUNING_PROMPTS",
    "materialize_direction",
    "materialize_evaluation",
    "materialize_h_infinity_splits",
)
