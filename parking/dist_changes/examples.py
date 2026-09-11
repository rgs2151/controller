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
        "Q: {original question} A:",
    ),
    (
        "Spanish",
        "spanish",
        (
            "A Spanish rendering of the original question followed by an instruction "
            "to answer in English. These are the translator's actual outputs, including "
            "cases where it changed or added content."
        ),
        "Pregunta: {translated question}\nResponde en inglés.\nRespuesta:",
    ),
    (
        "Long context",
        "long_context",
        (
            "The original question placed after a long repeated neutral archive and "
            "an instruction to ignore that archive."
        ),
        (
            "{neutral archive sentence repeated 25 times}\n"
            "Ignore the unrelated archive and answer this question.\n"
            "Q: {original question} A:"
        ),
    ),
    (
        "Adversarial",
        "adversarial",
        (
            "The original question followed, after its answer cue, by 32 copies of "
            "irrelevant context. This moves the model's continuation point away from "
            "the question and disrupts the expected question-answer format."
        ),
        (
            "Q: {original question} A:\n\n"
            "{irrelevant context sentence repeated 32 times}"
        ),
    ),
)


def _load_rows() -> dict[str, list[dict[str, object]]]:
    payload = json.loads((CACHE / "generations" / "alqr.json").read_text())
    if payload.get("status") != "complete":
        raise ValueError("A-LQR generation cache is incomplete")
    selected = json.loads((PLOTS / "summary.json").read_text())["selected_adversarial"]
    rows = dict(payload["conditions"])
    rows["adversarial"] = rows[selected]
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
    for title, condition, meaning, prompt_format in SECTIONS:
        lines.extend(
            [
                f"# {title}",
                "",
                meaning,
                "",
                "## Format",
                "",
                "```text",
                prompt_format,
                "```",
                "",
                "## Five actual evaluated prompts",
                "",
            ]
        )
        for index, source_id in enumerate(selected_ids, start=1):
            row = by_condition[condition][source_id]
            prompt = "\n".join(
                line.rstrip() for line in str(row["prompt"]).splitlines()
            ).rstrip()
            lines.extend(
                [
                    f"### {index}. `{source_id}`",
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
