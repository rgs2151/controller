"""Compute question-clustered jackknife uncertainty for MGSM reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = Path(__file__).with_name("mgsm_jackknife.json")
MODELS = ("qwen3_4b", "llama32_3b_instruct")
LANGUAGES = ("zh", "fr", "ja", "sw", "te")
METHODS = ("original", "alqr", "h_infinity")
SCORERS = (
    "mgsm_exact_match",
    "axbench_rule_spanish",
    "axbench_instruction_relevance",
    "axbench_fluency",
)
GROUP_COUNT = 10


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _roots(overrides: list[str]) -> dict[str, Path]:
    roots = {model: REPO / "benchmarks/mgsm/cache" / model for model in MODELS}
    for override in overrides:
        model, separator, raw_path = override.partition("=")
        if not separator or model not in roots:
            raise ValueError(f"Expected MODEL=PATH for one of {MODELS}: {override}")
        roots[model] = Path(raw_path).expanduser().resolve()
    return roots


def _question_id(prompt_id: str) -> str:
    # mgsm-001-zh -> mgsm-001; language is the repeated variant.
    return prompt_id.rsplit("-", 1)[0]


def _groups(question_ids: set[str]) -> list[list[str]]:
    ordered = sorted(
        question_ids,
        key=lambda item: hashlib.sha256(f"mgsm-jackknife-v1:{item}".encode()).hexdigest(),
    )
    groups = [[] for _ in range(GROUP_COUNT)]
    for index, question_id in enumerate(ordered):
        groups[index % GROUP_COUNT].append(question_id)
    if max(map(len, groups)) - min(map(len, groups)) > 1:
        raise RuntimeError("MGSM jackknife groups are unbalanced")
    return [sorted(group) for group in groups]


def _score_map(root: Path, scorer: str, language: str, method: str) -> dict[str, float] | None:
    path = (
        root
        / "evaluations/kv_cache_off/scores"
        / scorer
        / f"mgsm_{language}"
        / method
        / "final.json"
    )
    if not path.exists():
        return None
    rows = json.loads(path.read_text())["rows"]
    scores = {
        _question_id(row["prompt_id"]): float(row["score"])
        for row in rows
        if row.get("score") is not None
    }
    if len(scores) != len(rows):
        raise RuntimeError(f"Invalid or duplicate rows in {path}")
    return scores


def _jackknife(values: dict[str, list[float]], groups: list[list[str]]) -> dict[str, object]:
    all_values = [value for question_values in values.values() for value in question_values]
    full_mean = sum(all_values) / len(all_values)
    estimates = []
    for group in groups:
        excluded = set(group)
        retained = [
            value
            for question_id, question_values in values.items()
            if question_id not in excluded
            for value in question_values
        ]
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
    report: dict[str, object] = {
        "schema_version": 1,
        "method": "ten-group delete-one-group matched-question jackknife",
        "group_count": GROUP_COUNT,
        "models": {},
    }
    for model, root in _roots(args.cache_root).items():
        reference = _score_map(root, "mgsm_exact_match", "zh", "original")
        if reference is None:
            raise FileNotFoundError(f"Missing MGSM reference scores for {model}")
        groups = _groups(set(reference))
        model_report: dict[str, object] = {
            "question_count": len(reference),
            "group_sizes": [len(group) for group in groups],
            "groups": groups,
            "languages": {},
            "summary": {},
            "missing": [],
        }
        for language in LANGUAGES:
            language_report: dict[str, object] = {}
            for method in METHODS:
                method_report: dict[str, object] = {}
                for scorer in SCORERS:
                    scores = _score_map(root, scorer, language, method)
                    if scores is None:
                        model_report["missing"].append(
                            {"language": language, "method": method, "scorer": scorer}
                        )
                        continue
                    if set(scores) != set(reference):
                        raise RuntimeError(
                            f"Question identities differ for {model}/{language}/{method}/{scorer}"
                        )
                    scale = 100.0 if scorer == "mgsm_exact_match" else 1.0
                    method_report[scorer] = _jackknife(
                        {question_id: [scale * value] for question_id, value in scores.items()},
                        groups,
                    )
                language_report[method] = method_report
            model_report["languages"][language] = language_report

        # The reported summary macro-average is recomputed inside each deletion,
        # with the same question identities removed from all five languages.
        for method in METHODS:
            method_report: dict[str, object] = {}
            for scorer in SCORERS:
                per_language = [
                    _score_map(root, scorer, language, method) for language in LANGUAGES
                ]
                if any(scores is None for scores in per_language):
                    continue
                scale = 100.0 if scorer == "mgsm_exact_match" else 1.0
                values = {
                    question_id: [
                        scale * scores[question_id] for scores in per_language if scores is not None
                    ]
                    for question_id in reference
                }
                method_report[scorer] = _jackknife(values, groups)
            model_report["summary"][method] = method_report
        report["models"][model] = model_report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
