"""Write five complete prompt examples for each current dataset."""

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
        "Long Context End",
        "long_context_end",
        "Seven fixed public-domain book excerpts fill about 7,168 Gemma tokens, with the unchanged TruthfulQA question placed at the end.",
    ),
    (
        "Long Context Start",
        "long_context_start",
        "The exact documents and question from Long Context End are retained, but the question and answer cue are placed before the documents.",
    ),
    (
        "Corrupting Words",
        "corrupting_words",
        "One frozen text-only suffix found by gradient search to maximize A-LQR overshoot on a Llama-3.2-1B prompt is appended to every TruthfulQA prompt.",
    ),
    (
        "BOS Mix",
        "bos_mix",
        "Gemma-2-2B's actual `<bos>` token is distributed through each prompt, with a fixed seed assigning 16 insertions to 25 questions and 64 to the other 25.",
    ),
    (
        "L-CiteEval Complexity",
        "lciteeval_complexity",
        "Fifty separate NarrativeQA and LoCoMo questions from L-CiteEval span easy, medium, and hard long-context examples that fit Gemma-2-2B.",
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
        condition_ids = (
            random.Random(f"{SEED}:{condition}").sample(
                list(by_condition[condition]), EXAMPLE_COUNT
            )
            if condition == "lciteeval_complexity"
            else selected_ids
        )
        for source_id in condition_ids:
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
