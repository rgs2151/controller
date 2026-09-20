"""Compute matched-question jackknife uncertainty for L-CiteEval reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO / "benchmarks/lciteeval/cache/qwen25_3b_instruct"
DEFAULT_OUTPUT = Path(__file__).with_name("lciteeval_jackknife.json")
CONDITIONS = ("8k", "16k", "32k")
METHODS = ("original", "spid", "alqr", "h_infinity")
SCORERS = {
    "lcite_answer_overlap.answer_recall": ("lcite_answer_overlap", "answer_recall", 100.0),
    "lcite_citation_nli.citation_f1": ("lcite_citation_nli", "citation_f1", 100.0),
    "axbench_concept_relevance.score": ("axbench_concept_relevance", "score", 1.0),
    "axbench_instruction_relevance.score": ("axbench_instruction_relevance", "score", 1.0),
    "axbench_fluency.score": ("axbench_fluency", "score", 1.0),
    "axbench_overall.score": ("axbench_overall", "score", 1.0),
}
GROUP_COUNT = 10


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _groups(question_ids: set[str]) -> list[list[str]]:
    ordered = sorted(
        question_ids,
        key=lambda item: hashlib.sha256(f"lciteeval-jackknife-v1:{item}".encode()).hexdigest(),
    )
    groups = [[] for _ in range(GROUP_COUNT)]
    for index, question_id in enumerate(ordered):
        groups[index % GROUP_COUNT].append(question_id)
    if [len(group) for group in groups] != [4] * GROUP_COUNT:
        raise RuntimeError("Expected ten L-CiteEval groups of four matched questions")
    return [sorted(group) for group in groups]


def _score_map(
    root: Path,
    condition: str,
    method: str,
    scorer_directory: str,
    value_key: str,
    prompt_to_question: dict[str, str],
) -> dict[str, float]:
    path = (
        root
        / "evaluations/kv_cache_off/scores"
        / scorer_directory
        / f"hotpotqa_{condition}"
        / method
        / "final.json"
    )
    rows = json.loads(path.read_text())["rows"]
    scores = {
        prompt_to_question[row["prompt_id"]]: float(row[value_key])
        for row in rows
        if row.get(value_key) is not None
    }
    if len(scores) != len(rows):
        raise RuntimeError(f"Invalid or duplicate rows in {path}")
    return scores


def _jackknife(values: dict[str, float], groups: list[list[str]]) -> dict[str, object]:
    full_mean = sum(values.values()) / len(values)
    estimates = []
    for group in groups:
        excluded = set(group)
        retained = [value for question_id, value in values.items() if question_id not in excluded]
        estimates.append(sum(retained) / len(retained))
    center = sum(estimates) / len(estimates)
    standard_error = math.sqrt(
        (len(groups) - 1)
        / len(groups)
        * sum((estimate - center) ** 2 for estimate in estimates)
    )
    return {
        "full_sample_mean": full_mean,
        "jackknife_standard_error": standard_error,
        "leave_group_out_means": estimates,
    }


def main() -> None:
    args = _arguments()
    root = args.cache_root.expanduser().resolve()
    dataset = json.loads((root / "datasets/lciteeval.json").read_text())
    question_ids = {
        str(row["matched_question_index"])
        for row in dataset["evaluation"]["8k"]
    }
    groups = _groups(question_ids)
    report: dict[str, object] = {
        "schema_version": 1,
        "method": "ten-group delete-one-group matched-question jackknife",
        "group_count": GROUP_COUNT,
        "question_count": len(question_ids),
        "group_sizes": [len(group) for group in groups],
        "groups": groups,
        "conditions": {},
    }
    for condition in CONDITIONS:
        rows = dataset["evaluation"][condition]
        prompt_to_question = {
            row["prompt_id"]: str(row["matched_question_index"]) for row in rows
        }
        if set(prompt_to_question.values()) != question_ids:
            raise RuntimeError(f"Question identities differ at {condition}")
        condition_report: dict[str, object] = {}
        for method in METHODS:
            method_report: dict[str, object] = {}
            for metric_key, (scorer, value_key, scale) in SCORERS.items():
                scores = _score_map(
                    root,
                    condition,
                    method,
                    scorer,
                    value_key,
                    prompt_to_question,
                )
                if set(scores) != question_ids:
                    raise RuntimeError(
                        f"Question identities differ for {condition}/{method}/{metric_key}"
                    )
                method_report[metric_key] = _jackknife(
                    {question_id: scale * value for question_id, value in scores.items()},
                    groups,
                )
            condition_report[method] = method_report
        report["conditions"][condition] = condition_report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
