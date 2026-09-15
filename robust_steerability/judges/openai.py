"""Concurrent OpenAI scoring for independent AXBench-style judges."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from robust_steerability.benchmarks.layout import REPO_ROOT
from robust_steerability.judges.specs import scorer_cache_path, scorer_spec


ENDPOINT = "https://api.openai.com/v1/chat/completions"
DEFAULT_CONCURRENCY = 500
DEFAULT_BATCH_SIZE = 20
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _api_key() -> str:
    value = os.environ.get("OPENAI_API_KEY")
    if value:
        return value
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                value = line.split("=", 1)[1].strip().strip("\"'")
                if value:
                    return value
    raise RuntimeError(f"OPENAI_API_KEY is not set and was not found in {env_path}")


def _flatten(payload: dict) -> list[dict]:
    return [row for repetition in payload["repetitions"] for row in repetition["rows"]]


def _scorer_record(scorer_key: str, batch_size: int) -> dict[str, object]:
    spec = scorer_spec(scorer_key)
    return {
        "schema_version": 1,
        "scorer_key": scorer_key,
        "model": spec.model_id,
        "temperature": 0,
        "rating_range": [spec.minimum, spec.maximum],
        "rubric": spec.rubric,
        "batch_size": batch_size,
        "response_format": "strict JSON schema",
    }


def _response_format(count: int) -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "independent_judge_batch",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "minItems": count,
                        "maxItems": count,
                        "items": {
                            "type": "object",
                            "properties": {
                                "item_index": {"type": "integer"},
                                "score": {"type": "integer", "enum": [0, 1, 2]},
                                "explanation": {"type": "string"},
                            },
                            "required": ["item_index", "score", "explanation"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["results"],
                "additionalProperties": False,
            },
        },
    }


def _request_rows(scorer_key: str, indexed_rows: list[tuple[int, dict]]) -> list[dict]:
    rows = []
    for item_index, row in indexed_rows:
        item = {
            "item_index": item_index,
            "instruction": str(row["text"]),
            "response": str(row["completion"]),
        }
        if scorer_key == "axbench_concept_relevance":
            if "concept" not in row:
                raise ValueError("axbench_concept_relevance requires a concept on every row")
            item["concept"] = str(row["concept"])
        rows.append(item)
    return rows


async def _request(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_key: str,
    scorer_key: str,
    batch_index: int,
    indexed_rows: list[tuple[int, dict]],
) -> dict:
    request_rows = _request_rows(scorer_key, indexed_rows)
    spec = scorer_spec(scorer_key)
    body = {
        "model": spec.model_id,
        "messages": [
            {"role": "system", "content": spec.rubric},
            {"role": "user", "content": json.dumps(request_rows, ensure_ascii=False)},
        ],
        "response_format": _response_format(len(request_rows)),
        "temperature": 0,
        "max_completion_tokens": 4096,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(8):
        async with semaphore:
            async with session.post(ENDPOINT, headers=headers, json=body) as response:
                response_text = await response.text()
                if response.status == 200:
                    payload = json.loads(response_text)
                    choice = payload["choices"][0]
                    if choice.get("finish_reason") != "stop":
                        raise RuntimeError(
                            f"OpenAI scorer batch {batch_index} did not finish cleanly"
                        )
                    returned = json.loads(choice["message"]["content"])["results"]
                    expected = [row["item_index"] for row in request_rows]
                    by_index = {row["item_index"]: row for row in returned}
                    if sorted(by_index) != expected or len(by_index) != len(returned):
                        raise ValueError(
                            f"OpenAI scorer batch {batch_index} changed item indices"
                        )
                    return {
                        "batch_index": batch_index,
                        "results": [by_index[index] for index in expected],
                        "api": {
                            "request_id": payload.get("id"),
                            "returned_model": payload.get("model"),
                            "system_fingerprint": payload.get("system_fingerprint"),
                            "usage": payload.get("usage", {}),
                        },
                    }
                if response.status not in RETRYABLE_STATUS or attempt == 7:
                    raise RuntimeError(
                        f"OpenAI API error {response.status} in {scorer_key} batch "
                        f"{batch_index}: {response_text}"
                    )
                retry_after = response.headers.get("Retry-After")
        await asyncio.sleep(float(retry_after) if retry_after else min(2**attempt, 30))
    raise RuntimeError("OpenAI API retry loop terminated unexpectedly")


async def _score_async(
    generation_paths: list[Path],
    root: Path,
    scorer_keys: list[str],
    concurrency: int,
    batch_size: int,
) -> list[Path]:
    states: dict[Path, dict] = {}
    generation_rows: dict[Path, list[dict]] = {}
    jobs = []
    destinations = []
    for generation_path in generation_paths:
        generation = json.loads(generation_path.read_text())
        if generation.get("status") != "complete":
            raise ValueError(f"Generation is incomplete: {generation_path}")
        rows = _flatten(generation)
        for scorer_key in scorer_keys:
            spec = scorer_spec(scorer_key)
            if spec.backend != "openai_0_2":
                raise ValueError(f"{scorer_key} is not an OpenAI scorer")
            destination = scorer_cache_path(root, generation_path, scorer_key)
            saved = {
                "scorer": _scorer_record(scorer_key, batch_size),
                "status": "partial",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "batches": [],
                "rows": [],
            }
            if destination.exists():
                saved = json.loads(destination.read_text())
                if saved.get("status") == "complete":
                    destinations.append(destination)
                    continue
            completed = {int(batch["batch_index"]) for batch in saved["batches"]}
            states[destination] = saved
            generation_rows[destination] = rows
            destinations.append(destination)
            indexed = list(enumerate(rows))
            for start in range(0, len(indexed), batch_size):
                batch_index = start // batch_size
                if batch_index not in completed:
                    jobs.append((destination, scorer_key, batch_index, indexed[start:start + batch_size]))
            _write_json(destination, saved)

    if jobs:
        timeout = aiohttp.ClientTimeout(total=240)
        connector = aiohttp.TCPConnector(limit=concurrency)
        semaphore = asyncio.Semaphore(concurrency)
        key = _api_key()
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            async def run_job(destination, scorer_key, batch_index, rows):
                result = await _request(
                    session, semaphore, key, scorer_key, batch_index, rows
                )
                return destination, result

            tasks = [
                asyncio.create_task(
                    run_job(destination, scorer_key, batch_index, rows)
                )
                for destination, scorer_key, batch_index, rows in jobs
            ]
            pending_writes: dict[Path, int] = {path: 0 for path in states}
            for future in asyncio.as_completed(tasks):
                destination, result = await future
                states[destination]["batches"].append(result)
                states[destination]["batches"].sort(key=lambda row: row["batch_index"])
                pending_writes[destination] += 1
                if pending_writes[destination] >= 10:
                    _write_json(destination, states[destination])
                    pending_writes[destination] = 0

    for destination, saved in states.items():
        rows = generation_rows[destination]
        ordered = [
            result
            for batch in sorted(saved["batches"], key=lambda row: row["batch_index"])
            for result in batch["results"]
        ]
        if [row["item_index"] for row in ordered] != list(range(len(rows))):
            raise ValueError(f"Scorer batches are incomplete or misordered: {destination}")
        saved["rows"] = [
            {
                "item_index": result["item_index"],
                "prompt_id": str(source["prompt_id"]),
                "score": float(result["score"]),
                "explanation": result["explanation"],
            }
            for result, source in zip(ordered, rows, strict=True)
        ]
        saved["status"] = "complete"
        saved["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(destination, saved)
    return destinations


def score_generations(
    generation_paths: list[Path],
    root: Path,
    scorer_keys: list[str],
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[Path]:
    if concurrency < 1 or batch_size < 1:
        raise ValueError("concurrency and batch_size must be positive")
    return asyncio.run(
        _score_async(generation_paths, root, scorer_keys, concurrency, batch_size)
    )
