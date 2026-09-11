"""Write five complete, matched prompt examples for each current distribution."""

import csv
from pathlib import Path
import random


UNIT = Path(__file__).resolve().parent
CACHE = UNIT / "cache"
PLOTS = UNIT / "plots"
SEED = 2151
EXAMPLE_COUNT = 5

SECTIONS = (
    (
        "ID",
        "id",
        "The original held-out TruthfulQA question with no distribution change.",
    ),
    (
        "Spanish",
        "spanish",
        "Llama-3.2-3B-Instruct deterministically translated each English question into Spanish, then the prompt asks for an English answer.",
    ),
    (
        "Japanese (romaji)",
        "japanese_romaji",
        "Meta-Llama-3.1-8B-Instruct deterministically translated each English question into Japanese, pykakasi converted it to Hepburn romaji, and the prompt asks for an English answer.",
    ),
    (
        "Long context",
        "long_context",
        "Seven deterministic public-domain book excerpts are token-trimmed to about 7,168 Gemma tokens and placed before the unchanged question.",
    ),
    (
        "D2",
        "d2",
        "The exact fixed text-only overshoot suffix found against Llama-3.2-1B is appended unchanged to every Gemma question.",
    ),
    (
        "D3",
        "d3",
        "The exact shared suffix refined on the four hardest Llama-3.2-1B D2 cases is appended unchanged to every Gemma question.",
    ),
    (
        "D6",
        "d6",
        "A seeded split appends the exact Llama `<|begin_of_text|>` string 16 times to 25 Gemma questions and 64 times to the other 25.",
    ),
)


def _load_rows() -> dict[str, list[dict[str, object]]]:
    rows = {}
    for _, condition, _ in SECTIONS:
        with (CACHE / "datasets" / f"{condition}.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows[condition] = list(csv.DictReader(handle))
        if len(rows[condition]) != 50:
            raise ValueError(f"{condition} does not contain 50 prompts")
    return rows


def main() -> None:
    rows = _load_rows()
    id_rows = rows["id"]
    selected_ids = random.Random(SEED).sample(
        [str(row["source_prompt_id"]) for row in id_rows], EXAMPLE_COUNT
    )
    by_condition = {
        condition: {str(row["source_prompt_id"]): row for row in records}
        for condition, records in rows.items()
    }

    lines: list[str] = []
    for title, condition, meaning in SECTIONS:
        lines.extend(
            [
                f"# {title}",
                "",
                meaning,
                "",
            ]
        )
        for source_id in selected_ids:
            row = by_condition[condition][source_id]
            prompt = "\n".join(
                line.rstrip()
                for line in str(row["prompt"]).splitlines()
            ).rstrip()
            lines.extend(
                [
                    "```text",
                    prompt,
                    "```",
                    "",
                ]
            )
    (PLOTS / "ood_examples.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
