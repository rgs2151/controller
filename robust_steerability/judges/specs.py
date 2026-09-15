"""Pinned judge definitions shared by benchmark scoring stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JudgeSpec:
    key: str
    backend: str
    metric: str
    model_id: str
    revision: str | None
    minimum: float
    maximum: float
    prompt_label: str | None = None


ALL_JUDGES = {
    "true": JudgeSpec(
        "true",
        "huggingface_binary",
        "truth",
        "allenai/truthfulqa-truth-judge-llama2-7B",
        "8f718cb4e68bc30a488727a0e37190835a7c129e",
        0.0,
        1.0,
        "True",
    ),
    "informative": JudgeSpec(
        "informative",
        "huggingface_binary",
        "info",
        "allenai/truthfulqa-info-judge-llama2-7B",
        "72601eff294cbf7e8d7188ae82397e19cb9bb9bd",
        0.0,
        1.0,
        "Helpful",
    ),
    "concept_relevance": JudgeSpec(
        "concept_relevance",
        "openai_0_2",
        "concept_relevance",
        "gpt-4o-mini",
        None,
        0.0,
        2.0,
    ),
    "instruction_relevance": JudgeSpec(
        "instruction_relevance",
        "openai_0_2",
        "instruction_relevance",
        "gpt-4o-mini",
        None,
        0.0,
        2.0,
    ),
    "fluency": JudgeSpec(
        "fluency",
        "openai_0_2",
        "fluency",
        "gpt-4o-mini",
        None,
        0.0,
        2.0,
    ),
}

TRUTHFULNESS_JUDGES = (
    "true",
    "informative",
    "instruction_relevance",
    "fluency",
)


def judge_spec(key: str) -> JudgeSpec:
    if key not in ALL_JUDGES:
        raise ValueError(f"Unknown judge {key!r}")
    return ALL_JUDGES[key]


def judge_cache_path(root: Path, generation_path: Path, judge_key: str) -> Path:
    relative = generation_path.relative_to(root / "generations")
    return root / "judges" / judge_key / relative
