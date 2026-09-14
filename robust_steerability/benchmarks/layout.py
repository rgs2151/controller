"""Filesystem ownership for portable benchmark stages."""

from __future__ import annotations

from pathlib import Path


BENCHMARKS = ("truthfulness", "toxicity")
KV_CACHE_CONDITIONS = ("kv_cache_off", "kv_cache_on")
REPO_ROOT = Path(__file__).resolve().parents[2]


def benchmark_root(benchmark: str) -> Path:
    """Return the repository unit that owns one benchmark."""

    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {benchmark!r}")
    return REPO_ROOT / "benchmarks" / benchmark


def model_root(benchmark: str, model_key: str) -> Path:
    """Return all machine-local state for one model and benchmark."""

    if not model_key or "/" in model_key:
        raise ValueError(f"Invalid model key {model_key!r}")
    return benchmark_root(benchmark) / "cache" / model_key


def artifact_root(benchmark: str, model_key: str) -> Path:
    """Return controller-neutral fit artifacts (setpoint and averaged A)."""

    return model_root(benchmark, model_key) / "artifacts"


def calibration_root(
    benchmark: str,
    model_key: str,
    method: str,
    calibration_id: str = "selected",
) -> Path:
    """Return one named, immutable method-calibration directory."""

    if not method or not calibration_id or "/" in method or "/" in calibration_id:
        raise ValueError("method and calibration_id must be simple names")
    return model_root(benchmark, model_key) / "calibrations" / method / calibration_id


def kv_cache_condition(use_cache: bool) -> str:
    """Return the explicit directory name for one decoding-cache policy."""

    return "kv_cache_on" if use_cache else "kv_cache_off"


def evaluation_root(
    benchmark: str,
    model_key: str,
    *,
    use_cache: bool,
) -> Path:
    """Return evaluations isolated by model and evaluated-model cache policy."""

    return model_root(benchmark, model_key) / "evaluations" / kv_cache_condition(use_cache)


def results_root(benchmark: str) -> Path:
    """Return small, Git-tracked benchmark summaries."""

    return benchmark_root(benchmark) / "results"
