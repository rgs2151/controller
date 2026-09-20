"""Add five-group question-jackknife uncertainty to Qwen-2.5-32B results."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

import pyarrow.parquet as pq


METHODS = ("original", "spid", "alqr", "h_infinity")
DISTRIBUTIONS = ("truthfulness", "truthfulness_spanish")
METRICS = {
    "truth": ("truthfulqa_true", 100.0),
    "info": ("truthfulqa_informative", 100.0),
    "instruction_relevance": ("axbench_instruction_relevance", 1.0),
    "fluency": ("axbench_fluency", 1.0),
}
GROUP_COUNT = 5
PARTITION_SEED = 20260920


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n")


def _finite_json(value):
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _category_map(dataset_path: Path) -> dict[str, str]:
    records = pq.read_table(dataset_path, columns=["category"]).to_pylist()
    return {
        f"truthfulqa:{index}": str(record["category"])
        for index, record in enumerate(records)
    }


def _balanced_groups(
    prompt_ids: list[str], categories: dict[str, str]
) -> list[list[str]]:
    by_category: dict[str, list[str]] = {}
    for prompt_id in prompt_ids:
        by_category.setdefault(categories[prompt_id], []).append(prompt_id)
    rng = random.Random(PARTITION_SEED)
    groups: list[list[str]] = [[] for _ in range(GROUP_COUNT)]
    for category in sorted(by_category, key=lambda key: (-len(by_category[key]), key)):
        members = sorted(by_category[category])
        rng.shuffle(members)
        start = min(range(GROUP_COUNT), key=lambda index: (len(groups[index]), index))
        for offset, prompt_id in enumerate(members):
            groups[(start + offset) % GROUP_COUNT].append(prompt_id)
    return [sorted(group) for group in groups]


def _scores(path: Path) -> dict[str, float]:
    rows = _read_json(path)["rows"]
    scores = {
        str(row["prompt_id"]): float(row["score"])
        for row in rows
        if row.get("valid", True)
    }
    if len(scores) != len(rows):
        raise ValueError(f"Invalid or duplicate score rows in {path}")
    return scores


def _jackknife(values: dict[str, float], groups: list[list[str]]) -> tuple[float, list[float]]:
    all_ids = set(values)
    leave_group_out = []
    for group in groups:
        retained = all_ids.difference(group)
        leave_group_out.append(sum(values[prompt_id] for prompt_id in retained) / len(retained))
    center = sum(leave_group_out) / len(leave_group_out)
    standard_error = math.sqrt(
        (len(groups) - 1)
        / len(groups)
        * sum((estimate - center) ** 2 for estimate in leave_group_out)
    )
    return standard_error, leave_group_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--truthfulqa-parquet", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()

    reference_path = (
        arguments.scores_root
        / "truthfulqa_true/truthfulness/original/final.json"
    )
    prompt_ids = [str(row["prompt_id"]) for row in _read_json(reference_path)["rows"]]
    if len(prompt_ids) != 409 or len(set(prompt_ids)) != 409:
        raise ValueError("Expected exactly 409 unique Qwen-2.5-32B question identities")
    categories = _category_map(arguments.truthfulqa_parquet)
    groups = _balanced_groups(prompt_ids, categories)
    if sorted(prompt_id for group in groups for prompt_id in group) != sorted(prompt_ids):
        raise ValueError("Jackknife groups do not partition the evaluated questions")

    report = {
        "schema_version": 1,
        "model": "Qwen/Qwen2.5-32B",
        "method": "five-group delete-one-group jackknife",
        "partition_seed": PARTITION_SEED,
        "stratification": "TruthfulQA category-balanced deterministic partition",
        "sample_count": len(prompt_ids),
        "group_sizes": [len(group) for group in groups],
        "group_prompt_ids": groups,
        "partition_sha256": hashlib.sha256(
            json.dumps(groups, separators=(",", ":")).encode()
        ).hexdigest(),
        "interpretation": "Question-sampling variability only; not decoding-run variability.",
        "results": {},
    }

    expected_ids = set(prompt_ids)
    for distribution in DISTRIBUTIONS:
        report["results"][distribution] = {}
        for method in METHODS:
            result_path = arguments.results_root / distribution / f"{method}.json"
            result = _finite_json(_read_json(result_path))
            method_report = {}
            for result_metric, (scorer, scale) in METRICS.items():
                score_path = (
                    arguments.scores_root
                    / scorer
                    / distribution
                    / method
                    / "final.json"
                )
                values = _scores(score_path)
                if set(values) != expected_ids:
                    raise ValueError(f"Question identities differ in {score_path}")
                values = {key: value * scale for key, value in values.items()}
                full_mean = sum(values.values()) / len(values)
                saved_mean = float(result["metrics"][result_metric]["mean"])
                if not math.isclose(full_mean, saved_mean, abs_tol=1e-10):
                    raise ValueError(
                        f"Saved mean mismatch for {distribution}/{method}/{result_metric}: "
                        f"{saved_mean} != {full_mean}"
                    )
                standard_error, leave_group_out = _jackknife(values, groups)
                result["metrics"][result_metric]["standard_error"] = standard_error
                method_report[result_metric] = {
                    "full_mean": full_mean,
                    "jackknife_standard_error": standard_error,
                    "leave_group_out_estimates": leave_group_out,
                }
            result["uncertainty"] = {
                "method": "five-group delete-one-group jackknife",
                "group_count": GROUP_COUNT,
                "partition_seed": PARTITION_SEED,
                "partition_sha256": report["partition_sha256"],
                "stratification": report["stratification"],
                "captures": report["interpretation"],
            }
            _write_json(result_path, result)
            report["results"][distribution][method] = method_report
    _write_json(arguments.report, report)


if __name__ == "__main__":
    main()
