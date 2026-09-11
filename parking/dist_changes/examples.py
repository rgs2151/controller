"""Write five complete, matched prompt examples for each displayed distribution."""

import json
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
        (
            "A Spanish rendering of the original question followed by an instruction "
            "to answer in English. These are the translator's actual outputs, including "
            "cases where it changed or added content."
        ),
    ),
    (
        "Long context",
        "long_context",
        (
            "The current long-context condition repeats one neutral archive sentence "
            "25 times before the original question, then tells the model to ignore it. "
            "The repeated block is compacted below so the examples remain readable."
        ),
    ),
    (
        "Adversarial",
        "adversarial",
        (
            "The current adversarial condition places 32 copies of an irrelevant "
            "two-sentence block after the answer cue. This disrupts the normal "
            "question-answer format. The repeated block is compacted below."
        ),
    ),
)

LONG_BLOCK = (
    "The archive contains routine descriptions of rooms, furniture, weather, "
    "and ordinary daily events. "
)
ADVERSARIAL_BLOCK = (
    "The archive records routine schedules, inventory counts, weather summaries, "
    "and ordinary procedural reminders. This background is unrelated to the "
    "question under analysis. "
)


def _load_rows() -> dict[str, list[dict[str, object]]]:
    payload = json.loads((CACHE / "generations" / "alqr.json").read_text())
    if payload.get("status") != "complete":
        raise ValueError("A-LQR generation cache is incomplete")
    selected = json.loads((PLOTS / "summary.json").read_text())["selected_adversarial"]
    rows = dict(payload["conditions"])
    rows["adversarial"] = rows[selected]
    return rows


def _readable_prompt(condition: str, prompt: str) -> str:
    if condition == "long_context":
        prefix = LONG_BLOCK * 25
        if not prompt.startswith(prefix):
            raise ValueError("Long-context prompt does not match its frozen construction")
        remainder = prompt[len(prefix) :].lstrip()
        return (
            '[Repeated 25 times: "'
            + LONG_BLOCK.strip()
            + '"]\n\n'
            + remainder
        )
    if condition == "adversarial":
        suffix = ADVERSARIAL_BLOCK * 32
        if not prompt.endswith(suffix):
            raise ValueError("Adversarial prompt does not match its frozen construction")
        question = prompt[: -len(suffix)].rstrip()
        return (
            question
            + '\n\n[Repeated 32 times: "'
            + ADVERSARIAL_BLOCK.strip()
            + '"]'
        )
    return prompt


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
        for index, source_id in enumerate(selected_ids, start=1):
            row = by_condition[condition][source_id]
            prompt = "\n".join(
                line.rstrip()
                for line in _readable_prompt(condition, str(row["prompt"])).splitlines()
            ).rstrip()
            lines.extend(
                [
                    f"**Example {index}**",
                    "",
                    "```text",
                    prompt,
                    "```",
                    "",
                ]
            )
    (PLOTS / "ood_examples.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
