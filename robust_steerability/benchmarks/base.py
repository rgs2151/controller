"""Records shared by behavior-specific benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchmarkRecord:
    prompt_id: str
    text: str
    split: str
    condition: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationRecord:
    prompt_id: str
    condition: str
    controller: str
    completion: str
    generated_tokens: int


def validate_disjoint_prompt_ids(
    fit_ids: set[str],
    calibration_ids: set[str],
    test_ids: set[str],
) -> None:
    """Enforce the project-wide fit/calibration/test isolation rule."""

    if fit_ids & calibration_ids:
        raise ValueError("fit and calibration prompt IDs overlap")
    if fit_ids & test_ids:
        raise ValueError("fit and test prompt IDs overlap")
    if calibration_ids & test_ids:
        raise ValueError("calibration and test prompt IDs overlap")
