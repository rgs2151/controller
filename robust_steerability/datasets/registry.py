"""Dataset adapters used by composable benchmark evaluation branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from robust_steerability.datasets.mmlu import (
    MMLU_ID,
    MMLU_REVISION,
    load_mmlu_five_shot_prompts,
)


@dataclass(frozen=True)
class DatasetAdapter:
    key: str
    dataset_id: str
    revision: str
    task: str
    prepare: Callable[[int, int], list[dict[str, object]]]


ADAPTERS = {
    "mmlu": DatasetAdapter(
        key="mmlu",
        dataset_id=MMLU_ID,
        revision=MMLU_REVISION,
        task="multiple_choice",
        prepare=lambda seed, count: load_mmlu_five_shot_prompts(seed, count, shots=5),
    ),
}


def dataset_adapter(key: str) -> DatasetAdapter:
    if key not in ADAPTERS:
        raise ValueError(f"No evaluation adapter is registered for dataset {key!r}")
    return ADAPTERS[key]
