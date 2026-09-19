"""Shared calibration-selection invariants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math


def require_nonzero_selection_metric(
    candidates: Sequence[Mapping[str, object]],
    metric: str,
    *,
    context: str,
) -> None:
    """Refuse to choose arbitrarily when calibration has no metric signal."""

    if not candidates:
        raise RuntimeError(f"{context} produced no calibration candidates")
    values = [float(candidate[metric]) for candidate in candidates]
    if all(value == 0.0 for value in values):
        raise RuntimeError(
            f"{context} failed: every candidate received {metric}=0; "
            "refusing to select an arbitrary configuration"
        )


def weighted_harmonic_mean(
    values: Sequence[float],
    weights: Sequence[float],
) -> float:
    """Return a strict weighted harmonic mean for normalized calibration scores."""

    if not values or len(values) != len(weights):
        raise ValueError("Weighted harmonic mean requires matching non-empty inputs")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Weighted harmonic inputs must be finite values in [0, 1]")
    if any(not math.isfinite(weight) or weight <= 0.0 for weight in weights):
        raise ValueError("Weighted harmonic weights must be finite and positive")
    if any(value == 0.0 for value in values):
        return 0.0
    return float(sum(weights) / sum(
        weight / value for value, weight in zip(values, weights, strict=True)
    ))
