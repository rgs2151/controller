"""Cached completion-only translation used by multilingual task scorers."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from robust_steerability.judges.openai import ENDPOINT, RETRYABLE_STATUS, _api_key


MODEL = "gpt-4o-mini-2024-07-18"
CITATION_PATTERN = re.compile(r"\[\d+\]")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def translation_path(root: Path, generation: Path) -> Path:
    return root / "normalizations/lcite_english" / generation.relative_to(root)


def _flatten(payload: dict) -> list[dict]:
    return [row for repetition in payload["repetitions"] for row in repetition["rows"]]


def _response_format(count: int) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "completion_translation_batch",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "translations": {
                        "type": "array",
                        "minItems": count,
                        "maxItems": count,
                        "items": {
                            "type": "object",
                            "properties": {
                                "item_index": {"type": "integer"},
                                "english": {"type": "string"},
                            },
                            "required": ["item_index", "english"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["translations"],
                "additionalProperties": False,
            },
        },
    }


async def _request(session, semaphore, key: str, indexed_rows: list[tuple[int, dict]]) -> list[dict]:
    source = [
        {"item_index": index, "completion": str(row["completion"])}
        for index, row in indexed_rows
    ]
    body = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Translate each model completion into English for evaluation. Preserve meaning, "
                    "sentence boundaries, numbers, and every citation marker such as [3] exactly. "
                    "Do not answer, correct, summarize, explain, or add text."
                ),
            },
            {"role": "user", "content": json.dumps(source, ensure_ascii=False)},
        ],
        "response_format": _response_format(len(source)),
        "temperature": 0,
        "max_completion_tokens": 8192,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(8):
        retry_after = None
        try:
            async with semaphore:
                async with session.post(ENDPOINT, headers=headers, json=body) as response:
                    text = await response.text()
                    if response.status == 200:
                        payload = json.loads(text)
                        returned = json.loads(payload["choices"][0]["message"]["content"])["translations"]
                        by_index = {int(row["item_index"]): row for row in returned}
                        expected = [index for index, _row in indexed_rows]
                        if sorted(by_index) != expected:
                            raise ValueError("Translation API changed item indices")
                        result = []
                        for index, row in indexed_rows:
                            english = str(by_index[index]["english"])
                            original_citations = CITATION_PATTERN.findall(str(row["completion"]))
                            translated_citations = CITATION_PATTERN.findall(english)
                            if original_citations != translated_citations:
                                raise ValueError(
                                    f"Translation changed citations for prompt {row['prompt_id']}"
                                )
                            result.append(
                                {
                                    "item_index": index,
                                    "prompt_id": str(row["prompt_id"]),
                                    "completion_english": english,
                                }
                            )
                        return result
                    if response.status not in RETRYABLE_STATUS or attempt == 7:
                        raise RuntimeError(f"OpenAI translation error {response.status}: {text}")
                    retry_after = response.headers.get("Retry-After")
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt == 7:
                raise
        await asyncio.sleep(float(retry_after) if retry_after else min(2**attempt, 30))
    raise RuntimeError("OpenAI translation retry loop terminated unexpectedly")


async def _translate_async(generations: list[Path], root: Path, concurrency: int, batch_size: int) -> list[Path]:
    jobs = []
    destinations = []
    for generation in generations:
        destination = translation_path(root, generation)
        destinations.append(destination)
        if destination.exists() and json.loads(destination.read_text()).get("status") == "complete":
            continue
        payload = json.loads(generation.read_text())
        if payload.get("status") != "complete":
            raise ValueError(f"Generation is incomplete: {generation}")
        rows = _flatten(payload)
        indexed = list(enumerate(rows))
        for start in range(0, len(indexed), batch_size):
            jobs.append((destination, indexed[start : start + batch_size]))

    if not jobs:
        return destinations
    grouped: dict[Path, list[dict]] = {destination: [] for destination, _rows in jobs}
    timeout = aiohttp.ClientTimeout(total=300)
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [
            asyncio.create_task(_request(session, semaphore, _api_key(), rows))
            for _destination, rows in jobs
        ]
        for (destination, _rows), task in zip(jobs, tasks, strict=True):
            grouped[destination].extend(await task)
    for destination, rows in grouped.items():
        rows.sort(key=lambda row: int(row["item_index"]))
        _write_json(
            destination,
            {
                "schema_version": 1,
                "status": "complete",
                "model": MODEL,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "rows": rows,
            },
        )
    return destinations


def translate_generations(generations: list[Path], root: Path, *, concurrency: int, batch_size: int) -> list[Path]:
    if concurrency < 1 or batch_size < 1:
        raise ValueError("concurrency and batch_size must be positive")
    return asyncio.run(_translate_async(generations, root, concurrency, batch_size))


def translated_completions(root: Path, generation: Path) -> dict[str, str]:
    path = translation_path(root, generation)
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"Translation cache is incomplete: {path}")
    return {str(row["prompt_id"]): str(row["completion_english"]) for row in payload["rows"]}
