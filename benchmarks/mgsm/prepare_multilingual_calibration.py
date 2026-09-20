"""Freeze multilingual MGSM-format prompts for H-infinity calibration."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from robust_steerability.benchmarks.layout import REPO_ROOT
from robust_steerability.datasets.mgsm import (
    GSM8K_ID,
    GSM8K_REVISION,
    LANGUAGE_NAMES,
    gsm8k_calibration_splits,
    load_mgsm,
    native_cot_prompt,
)


MODEL = "gpt-4o-mini-2024-07-18"
ENDPOINT = "https://api.openai.com/v1/chat/completions"
LANGUAGES = ("bn", "de", "ru", "th")
DESTINATION = REPO_ROOT / "data/mgsm/h_infinity_multilingual_calibration.json"


def _api_key() -> str:
    value = os.environ.get("OPENAI_API_KEY")
    if value:
        return value
    for line in (REPO_ROOT / ".env").read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError("OPENAI_API_KEY is unavailable")


def _language_assignment(split: str, count: int) -> list[str]:
    if split == "disturbance" and count == 200:
        return list(LANGUAGES) * 50
    if split == "tuning" and count == 50:
        return list(LANGUAGES) * 12 + ["bn", "de"]
    raise ValueError(f"Unexpected MGSM calibration split size: {split}={count}")


def _question(record: dict) -> str:
    return str(record["text"]).split("\n\n", 1)[1]


def _numbers(text: str) -> Counter[str]:
    return Counter(re.findall(r"\d+(?:[.,]\d+)?", text))


async def _translate(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    question: str,
    language: str,
) -> str:
    target = LANGUAGE_NAMES[language]
    payload = {
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"Translate the English math word problem into {target}. Preserve "
                    "its exact meaning, all proper nouns, units, and every Arabic-numeral "
                    "token exactly. Do not solve it, explain it, or add instructions."
                ),
            },
            {"role": "user", "content": question},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "math_problem_translation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"translation": {"type": "string"}},
                    "required": ["translation"],
                    "additionalProperties": False,
                },
            },
        },
    }
    async with semaphore:
        for attempt in range(8):
            async with session.post(ENDPOINT, json=payload) as response:
                if response.status == 200:
                    body = await response.json()
                    content = body["choices"][0]["message"]["content"]
                    translated = str(json.loads(content)["translation"]).strip()
                    if _numbers(question) != _numbers(translated):
                        raise ValueError(
                            f"Translation changed numeric tokens for {language}: {question!r}"
                        )
                    return translated
                if response.status not in {408, 409, 429, 500, 502, 503, 504}:
                    raise RuntimeError(
                        f"Translation request failed ({response.status}): "
                        f"{(await response.text())[:500]}"
                    )
            await asyncio.sleep(2**attempt)
    raise RuntimeError(f"Translation retries exhausted for {language}")


async def _prepare(concurrency: int) -> dict:
    source = gsm8k_calibration_splits()
    exemplars = {language: load_mgsm(language, "train") for language in LANGUAGES}
    jobs = []
    identities = []
    for split in ("disturbance", "tuning"):
        records = source[split]
        languages = _language_assignment(split, len(records))
        for record, language in zip(records, languages, strict=True):
            identities.append((split, record, language))
            jobs.append((_question(record), language))

    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }
    timeout = aiohttp.ClientTimeout(total=300)
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        translations = await asyncio.gather(
            *[
                _translate(session, semaphore, question, language)
                for question, language in jobs
            ]
        )

    splits = {"disturbance": [], "tuning": []}
    for (split, source_record, language), translation in zip(
        identities, translations, strict=True
    ):
        splits[split].append(
            {
                "prompt_id": (
                    f"gsm8k-multilingual-{split}-"
                    f"{int(source_record['source_index']):04d}-{language}"
                ),
                "source_prompt_id": str(source_record["prompt_id"]),
                "source_index": int(source_record["source_index"]),
                "language": language,
                "language_name": LANGUAGE_NAMES[language],
                "source_question": _question(source_record),
                "translated_question": translation,
                "text": native_cot_prompt(language, exemplars[language], translation),
                "answer": str(source_record["answer"]),
            }
        )

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": GSM8K_ID,
            "revision": GSM8K_REVISION,
            "seed": 42,
            "translator": MODEL,
            "translation_temperature": 0,
        },
        "construction": {
            "languages": list(LANGUAGES),
            "disturbance_allocation": {language: 50 for language in LANGUAGES},
            "tuning_allocation": {"bn": 13, "de": 13, "ru": 12, "th": 12},
            "prompt_style": "native MGSM eight-shot chain of thought",
            "final_evaluation_languages_excluded": ["zh", "fr", "ja", "sw", "te"],
        },
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=100)
    arguments = parser.parse_args()
    if DESTINATION.exists():
        raise FileExistsError(f"Frozen calibration dataset already exists: {DESTINATION}")
    payload = asyncio.run(_prepare(arguments.concurrency))
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = DESTINATION.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(DESTINATION)


if __name__ == "__main__":
    main()
