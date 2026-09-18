"""Pinned HarmBench text behaviors, direction records, and jailbreak prompts."""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import random
import urllib.request


HARM_BENCH_REVISION = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"
SOURCE_ROOT = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
    + HARM_BENCH_REVISION
)
SOURCE_FILES = {
    "validation_behaviors": (
        "data/behavior_datasets/harmbench_behaviors_text_val.csv",
        "7bfaaa5e2b9b95002c89e2d33c019f58e9e8496e2fc65d368f1eeb8179e69534",
    ),
    "test_behaviors": (
        "data/behavior_datasets/harmbench_behaviors_text_test.csv",
        "75d257b3e7428c52eb7b0154318f455af3e01b09a3794b5e2f3d36054f3c0e29",
    ),
    "classifier_validation": (
        "data/classifier_val_sets/text_behaviors_val_set.json",
        "e740727b6874e492063e097122a9916833d57f9bd2ce3e86713ec188741a41c8",
    ),
    "human_jailbreaks": (
        "baselines/human_jailbreaks/jailbreaks.py",
        "62e53558de1da3e67d1aec15c0dff59c848c0b374fd262ffad8f723fda0cd099",
    ),
}
SAFE_CONCEPT = (
    "the response safely declines or redirects rather than carrying out the "
    "requested harmful behavior"
)


def _source_bytes(key: str) -> bytes:
    relative, expected = SOURCE_FILES[key]
    with urllib.request.urlopen(f"{SOURCE_ROOT}/{relative}") as response:
        content = response.read()
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise ValueError(f"Pinned HarmBench source hash changed for {relative}")
    return content


def _behavior_rows(split: str) -> list[dict[str, str]]:
    key = {"validation": "validation_behaviors", "test": "test_behaviors"}[split]
    text = _source_bytes(key).decode("utf-8")
    rows = [dict(row) for row in csv.DictReader(io.StringIO(text))]
    return [row for row in rows if row["FunctionalCategory"] != "copyright"]


def validation_behaviors() -> list[dict[str, str]]:
    rows = _behavior_rows("validation")
    if len(rows) != 60:
        raise ValueError("Pinned HarmBench validation set is not the expected 60 rows")
    return rows


def test_behaviors() -> list[dict[str, str]]:
    rows = _behavior_rows("test")
    if len(rows) != 240:
        raise ValueError("Pinned HarmBench test set is not the expected 240 rows")
    return rows


def _stratified_sample(
    rows: list[dict], count: int, *, category: str, seed: int
) -> list[dict]:
    """Deterministically preserve category proportions with largest remainders."""

    if count > len(rows):
        raise ValueError("Cannot sample more HarmBench rows than are available")
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row[category]), []).append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    exact = {key: count * len(values) / len(rows) for key, values in groups.items()}
    allocation = {key: int(value) for key, value in exact.items()}
    remaining = count - sum(allocation.values())
    order = sorted(groups, key=lambda key: (-(exact[key] - allocation[key]), key))
    for key in order[:remaining]:
        allocation[key] += 1
    selected = [
        row
        for key in sorted(groups)
        for row in groups[key][: allocation[key]]
    ]
    rng.shuffle(selected)
    return selected


def direction_rows() -> dict[str, list[dict]]:
    """Return 50 human-majority non-exhibiting and 50 exhibiting completions."""

    behavior_by_id = {
        row["BehaviorID"]: row for row in validation_behaviors()
    }
    source = json.loads(_source_bytes("classifier_validation").decode("utf-8"))
    candidates = {"desired": [], "undesired": []}
    for behavior_id, entries in source.items():
        if behavior_id not in behavior_by_id:
            continue
        behavior = behavior_by_id[behavior_id]
        for source_index, entry in enumerate(entries):
            majority = int(
                sum(int(entry[f"human_{annotator}"]) for annotator in range(3)) >= 2
            )
            label = "undesired" if majority else "desired"
            candidates[label].append(
                {
                    "prompt_id": f"harmbench-direction-{label}-{behavior_id}-{source_index}",
                    "behavior_id": behavior_id,
                    "behavior": behavior["Behavior"],
                    "context": behavior["ContextString"],
                    "functional_category": behavior["FunctionalCategory"],
                    "semantic_category": behavior["SemanticCategory"],
                    "test_case": str(entry["test_case"]),
                    "response": str(entry["generation"]),
                    "human_labels": [int(entry[f"human_{index}"]) for index in range(3)],
                    "source_model": str(entry["model"]),
                    "source_method": str(entry["method"]),
                }
            )
    if len(candidates["desired"]) != 66 or len(candidates["undesired"]) != 52:
        raise ValueError("Pinned HarmBench classifier-validation counts changed")
    return {
        "desired": _stratified_sample(
            candidates["desired"], 50, category="semantic_category", seed=2151
        ),
        "undesired": _stratified_sample(
            candidates["undesired"], 50, category="semantic_category", seed=2152
        ),
    }


def tuning_behaviors() -> list[dict]:
    """Return the frozen 50 direct validation behaviors used by H-infinity."""

    return _stratified_sample(
        validation_behaviors(), 50, category="SemanticCategory", seed=2153
    )


def _jailbreak_templates() -> list[str]:
    tree = ast.parse(_source_bytes("human_jailbreaks").decode("utf-8"))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "JAILBREAKS" for target in node.targets)
    ]
    if len(assignments) != 1:
        raise ValueError("Could not locate the pinned HarmBench JAILBREAKS list")
    templates = ast.literal_eval(assignments[0].value)
    if not isinstance(templates, list) or len(templates) != 114:
        raise ValueError("Pinned HarmBench jailbreak list is not the expected 114 templates")
    return [
        str(template)
        .encode("utf-16-le", errors="surrogatepass")
        .decode("utf-16-le")
        for template in templates
    ]


def direct_request(row: dict[str, str]) -> str:
    context = str(row["ContextString"])
    behavior = str(row["Behavior"])
    return f"{context}\n\n---\n\n{behavior}" if context else behavior


def materialize_evaluation(tokenizer) -> dict:
    """Build direct and official random-subset-five test cases once."""

    behaviors = test_behaviors()
    templates = _jailbreak_templates()
    rng = random.Random(1)
    direct = []
    jailbreak = []
    for row in behaviors:
        behavior_id = str(row["BehaviorID"])
        request = direct_request(row)
        common = {
            "behavior_id": behavior_id,
            "behavior": str(row["Behavior"]),
            "context": str(row["ContextString"]),
            "functional_category": str(row["FunctionalCategory"]),
            "semantic_category": str(row["SemanticCategory"]),
        }
        direct_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": request}],
            tokenize=False,
            add_generation_prompt=True,
        )
        direct.append(
            {
                **common,
                "prompt_id": f"harmbench-{behavior_id}-direct",
                "condition": "direct",
                "text": request,
                "model_input": direct_prompt,
            }
        )
        rng.shuffle(templates)
        for template_index, template in enumerate(templates[:5]):
            assembled = f"{template}\n\n{request}"
            jailbreak.append(
                {
                    **common,
                    "prompt_id": f"harmbench-{behavior_id}-human-{template_index}",
                    "condition": "human_jailbreak",
                    "template_index": template_index,
                    "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                    "text": assembled,
                    "model_input": tokenizer.apply_chat_template(
                        [{"role": "user", "content": assembled}],
                        tokenize=False,
                        add_generation_prompt=True,
                    ),
                }
            )
    if len(direct) != 240 or len(jailbreak) != 1_200:
        raise ValueError("Materialized HarmBench evaluation counts changed")
    return {
        "schema_version": 1,
        "source": {
            "repository": "centerforaisafety/HarmBench",
            "revision": HARM_BENCH_REVISION,
            "human_jailbreak_configuration": {"random_subset": 5, "seed": 1},
        },
        "evaluation": {"direct": direct, "human_jailbreak": jailbreak},
    }
