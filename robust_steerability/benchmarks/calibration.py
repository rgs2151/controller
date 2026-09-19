"""Shared calibration-selection invariants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


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
