"""Compute behavior-clustered jackknife uncertainty for HarmBench reports.

The direct request and all five jailbreak-template prompts derived from one
HarmBench behavior are one sampling unit.  The ten fixed groups are balanced
over the joint functional/semantic behavior categories as closely as the
finite category counts allow.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = Path(__file__).with_name("harmbench_jackknife.json")
MODELS = ("llama32_1b_instruct", "llama32_3b_instruct", "llama31_8b_instruct")
METHODS = ("original", "alqr", "h_infinity")
SCORERS = (
    "harmbench_test_success",
    "axbench_concept_relevance",
    "axbench_instruction_relevance",
    "axbench_fluency",
    "axbench_overall",
)
GROUP_COUNT = 10


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help="Override a model cache directory (repeatable).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _cache_roots(overrides: list[str]) -> dict[str, Path]:
    roots = {
        model: REPO / "benchmarks/harmful/cache" / model for model in MODELS
    }
    for override in overrides:
        model, separator, raw_path = override.partition("=")
        if not separator or model not in roots:
            raise ValueError(f"Expected MODEL=PATH for one of {MODELS}: {override}")
        roots[model] = Path(raw_path).expanduser().resolve()
    return roots


def _stable_rank(label: str) -> int:
    return int(hashlib.sha256(f"harmbench-jackknife-v1:{label}".encode()).hexdigest(), 16)


def _balanced_groups(direct_rows: list[dict]) -> list[list[str]]:
    """Make ten deterministic, approximately category-balanced groups."""

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in direct_rows:
        stratum = (row["functional_category"], row["semantic_category"])
        strata[stratum].append(row["behavior_id"])

    groups: list[list[str]] = [[] for _ in range(GROUP_COUNT)]
    stratum_counts: list[dict[tuple[str, str], int]] = [
        defaultdict(int) for _ in range(GROUP_COUNT)
    ]
    for stratum, behavior_ids in sorted(
        strata.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        offset = _stable_rank("/".join(stratum)) % GROUP_COUNT
        order = sorted(range(GROUP_COUNT), key=lambda index: (index - offset) % GROUP_COUNT)
        rank = {group_index: index for index, group_index in enumerate(order)}
        for behavior_id in sorted(behavior_ids, key=_stable_rank):
            group_index = min(
                range(GROUP_COUNT),
                key=lambda index: (
                    stratum_counts[index][stratum],
                    len(groups[index]),
                    rank[index],
                ),
            )
            groups[group_index].append(behavior_id)
            stratum_counts[group_index][stratum] += 1

    if max(map(len, groups)) - min(map(len, groups)) > 1:
        raise RuntimeError(f"Unbalanced group sizes: {[len(group) for group in groups]}")
    return [sorted(group) for group in groups]


def _score_map(cache_root: Path, scorer: str, condition: str, method: str) -> dict[str, float]:
    path = (
        cache_root
        / "evaluations/kv_cache_off/scores"
        / scorer
        / condition
        / method
        / "final.json"
    )
    payload = json.loads(path.read_text())
    rows = payload["rows"]
    scores = {
        row["prompt_id"]: float(row["score"])
        for row in rows
        if row.get("valid", True) and row.get("score") is not None
    }
    if len(scores) != len(rows):
        raise RuntimeError(f"Invalid or duplicate scorer rows in {path}")
    return scores


def _jackknife(
    values_by_behavior: dict[str, list[float]], groups: list[list[str]]
) -> dict[str, object]:
    all_values = [value for values in values_by_behavior.values() for value in values]
    full_mean = sum(all_values) / len(all_values)
    estimates = []
    for group in groups:
        excluded = set(group)
        retained = [
            value
            for behavior_id, values in values_by_behavior.items()
            if behavior_id not in excluded
            for value in values
        ]
        estimates.append(sum(retained) / len(retained))
    estimate_mean = sum(estimates) / len(estimates)
    standard_error = math.sqrt(
        (len(groups) - 1)
        / len(groups)
        * sum((estimate - estimate_mean) ** 2 for estimate in estimates)
    )
    return {
        "full_sample_mean": full_mean,
        "jackknife_standard_error": standard_error,
        "leave_group_out_means": estimates,
    }


def _condition_maps(dataset: dict) -> dict[str, dict[str, str]]:
    maps = {
        "direct": {
            row["behavior_id"]: row["prompt_id"]
            for row in dataset["evaluation"]["direct"]
        }
    }
    for template_index in range(5):
        maps[str(template_index)] = {
            row["behavior_id"]: row["prompt_id"]
            for row in dataset["evaluation"]["human_jailbreak"]
            if int(row["template_index"]) == template_index
        }
    return maps


def main() -> None:
    args = _arguments()
    roots = _cache_roots(args.cache_root)
    report: dict[str, object] = {
        "schema_version": 1,
        "method": "ten-group delete-one-group behavior-clustered jackknife",
        "group_count": GROUP_COUNT,
        "models": {},
    }

    for model, cache_root in roots.items():
        dataset_path = cache_root / "datasets/harmbench.json"
        dataset = json.loads(dataset_path.read_text())
        evaluated_direct_ids = set(
            _score_map(
                cache_root,
                "harmbench_test_success",
                "harmbench_direct",
                "original",
            )
        )
        direct_rows = [
            row
            for row in dataset["evaluation"]["direct"]
            if row["prompt_id"] in evaluated_direct_ids
        ]
        groups = _balanced_groups(direct_rows)
        condition_maps = _condition_maps(dataset)
        evaluated_behaviors = {row["behavior_id"] for row in direct_rows}
        condition_maps = {
            condition: {
                behavior_id: prompt_id
                for behavior_id, prompt_id in prompt_ids.items()
                if behavior_id in evaluated_behaviors
            }
            for condition, prompt_ids in condition_maps.items()
        }
        if any(
            len(prompt_ids) != len(evaluated_behaviors)
            for prompt_ids in condition_maps.values()
        ):
            raise RuntimeError(f"Templates do not match evaluated behaviors for {model}")
        model_report: dict[str, object] = {
            "behavior_count": len(direct_rows),
            "group_sizes": [len(group) for group in groups],
            "groups": groups,
            "conditions": {},
            "human_jailbreak_aggregate": {},
        }

        for condition_key, prompt_ids in condition_maps.items():
            score_condition = (
                "harmbench_direct"
                if condition_key == "direct"
                else "harmbench_human_jailbreak"
            )
            condition_report: dict[str, object] = {}
            for method in METHODS:
                method_report: dict[str, object] = {}
                for scorer in SCORERS:
                    scores = _score_map(cache_root, scorer, score_condition, method)
                    scale = 100.0 if scorer == "harmbench_test_success" else 1.0
                    values = {
                        behavior_id: [scale * scores[prompt_id]]
                        for behavior_id, prompt_id in prompt_ids.items()
                    }
                    method_report[scorer] = _jackknife(values, groups)
                condition_report[method] = method_report
            model_report["conditions"][condition_key] = condition_report

        # Aggregate the five jailbreak variants while deleting all variants of
        # a behavior together.  This preserves the repeated-template cluster.
        for method in METHODS:
            scores = _score_map(
                cache_root,
                "harmbench_test_success",
                "harmbench_human_jailbreak",
                method,
            )
            values_by_behavior: dict[str, list[float]] = defaultdict(list)
            for condition_key in map(str, range(5)):
                for behavior_id, prompt_id in condition_maps[condition_key].items():
                    values_by_behavior[behavior_id].append(100.0 * scores[prompt_id])
            model_report["human_jailbreak_aggregate"][method] = _jackknife(
                values_by_behavior, groups
            )

        report["models"][model] = model_report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
