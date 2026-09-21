#!/usr/bin/env python3
"""Refresh plotting CSVs from the synchronized benchmark result tables."""

from __future__ import annotations

import csv
import re
from pathlib import Path


UNIT = Path(__file__).resolve().parents[1]
REPO = UNIT.parents[1]
CACHE = UNIT / "cache"
TABLES = REPO / "figs" / "bench_table"

METHOD_NAMES = {
    "H∞ (ours)": "H-infinity",
    "A-LQR": "A-LQR",
    "S-PID": "S-PID",
    "PID-AcT": "PID-AcT",
    "Linear-AcT": "Linear-AcT",
    "Mean-AcT": "Mean-AcT",
    "ActAdd": "ActAdd",
    "ITI": "ITI",
    "Original": "Original",
}


def markdown_rows(path: Path) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    started = False
    for line in lines:
        if not line.startswith("|"):
            if started:
                break
            continue
        started = True
        if line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and cells[0] not in {"Model", "Context", "Template"}:
            rows.append(cells)
    return rows


def mean_and_error(cell: str) -> tuple[float, float]:
    match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*±\s*(\d+(?:\.\d+)?)\s*", cell)
    if match is None:
        raise ValueError(f"Expected a mean ± error cell, got {cell!r}")
    return float(match.group(1)), float(match.group(2))


def write_csv(name: str, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    destination = CACHE / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def refresh_truthfulqa() -> None:
    output = []
    sources = (
        ("ID", TABLES / "truthfulness" / "truthfulqa.md"),
        ("OOD", TABLES / "truthfulness" / "truthfulqa_spanish.md"),
    )
    for split, path in sources:
        for model, method, true_cell, _info_cell, relevance_cell, fluency_cell in markdown_rows(path):
            if method == "ODESteer":
                continue
            true, true_se = mean_and_error(true_cell)
            relevance, _ = mean_and_error(relevance_cell)
            fluency, _ = mean_and_error(fluency_cell)
            output.append(
                {
                    "split": split,
                    "model": model,
                    "method": METHOD_NAMES[method],
                    "true": true,
                    "true_se": true_se,
                    "relevance": relevance,
                    "fluency": fluency,
                }
            )
    write_csv(
        "truthfulqa_figure_data.csv",
        ["split", "model", "method", "true", "true_se", "relevance", "fluency"],
        output,
    )


def refresh_mgsm() -> None:
    output = []
    path = TABLES / "mgsm" / "mgsm_full.md"
    for model, language, method, accuracy_cell, relevance_cell, instruction_cell, fluency_cell in markdown_rows(path):
        accuracy, _ = mean_and_error(accuracy_cell)
        relevance, _ = mean_and_error(relevance_cell)
        instruction, _ = mean_and_error(instruction_cell)
        fluency, _ = mean_and_error(fluency_cell)
        output.append(
            {
                "model": model,
                "language": language,
                "method": METHOD_NAMES[method],
                "accuracy": accuracy,
                "target_relevance": relevance,
                "instruction_relevance": instruction,
                "fluency": fluency,
            }
        )
    write_csv(
        "mgsm_transfer_results.csv",
        [
            "model",
            "language",
            "method",
            "accuracy",
            "target_relevance",
            "instruction_relevance",
            "fluency",
        ],
        output,
    )


def refresh_lciteeval() -> None:
    output = []
    path = TABLES / "lciteeval" / "lciteeval_summary.md"
    for model, context, method, recall_cell, citation_cell, steering_cell in markdown_rows(path):
        recall, _ = mean_and_error(recall_cell)
        citation, _ = mean_and_error(citation_cell)
        steering, _ = mean_and_error(steering_cell)
        output.append(
            {
                "model": model,
                "context_k": int(context.removesuffix("K")),
                "method": METHOD_NAMES[method],
                "answer_recall": recall,
                "citation_f1": citation,
                "steering_quality": steering,
            }
        )
    write_csv(
        "lciteeval_context_results.csv",
        ["model", "context_k", "method", "answer_recall", "citation_f1", "steering_quality"],
        output,
    )


def refresh_harmbench() -> None:
    output = []
    path = TABLES / "harmful" / "harmbench_full.md"
    for template, model, method, asr_cell, safe_cell, instruction_cell, fluency_cell, overall_cell in markdown_rows(path):
        asr, _ = mean_and_error(asr_cell)
        safe, _ = mean_and_error(safe_cell)
        instruction, _ = mean_and_error(instruction_cell)
        fluency, _ = mean_and_error(fluency_cell)
        overall, _ = mean_and_error(overall_cell)
        output.append(
            {
                "model": model,
                "template": template,
                "method": METHOD_NAMES[method],
                "asr": asr,
                "safe_concept_relevance": safe,
                "instruction_relevance": instruction,
                "fluency": fluency,
                "overall_steering": overall,
            }
        )
    write_csv(
        "harmbench_refusal_results.csv",
        [
            "model",
            "template",
            "method",
            "asr",
            "safe_concept_relevance",
            "instruction_relevance",
            "fluency",
            "overall_steering",
        ],
        output,
    )


def main() -> None:
    refresh_truthfulqa()
    refresh_mgsm()
    refresh_lciteeval()
    refresh_harmbench()


if __name__ == "__main__":
    main()
