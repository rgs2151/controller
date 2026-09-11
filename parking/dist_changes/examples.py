"""Write five complete, matched prompt examples for each current distribution."""

import csv
from pathlib import Path
import random


UNIT = Path(__file__).resolve().parent
CACHE = UNIT / "cache"
PLOTS = UNIT / "plots"
OUTPUT = PLOTS / "ood_examples"
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
        "D2 uses gradient search to find one text-only suffix that maximized A-LQR overshoot on a single Llama-3.2-1B prompt, then appends that exact frozen suffix to every TruthfulQA prompt.",
    ),
    (
        "D3",
        "d3",
        "D3 starts from D2 and jointly refines one text-only suffix against the four Llama-3.2-1B prompts where D2 transferred weakest, then appends that exact frozen suffix to every TruthfulQA prompt.",
    ),
    (
        "D6",
        "d6",
        "D6 distributes Gemma-2-2B's actual `<bos>` token evenly through each Q/A prompt, using seed 2151 to assign 16 insertions to 25 questions and 64 insertions to the other 25 without consulting outcomes.",
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

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for title, condition, meaning in SECTIONS:
        lines = [f"# {title}", "", meaning, ""]
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
        (OUTPUT / f"{condition}.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
