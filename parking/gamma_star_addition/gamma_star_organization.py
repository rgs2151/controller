#!/usr/bin/env python3
"""Build compact, provenance-preserving gamma-star analysis tables.

The source benchmark caches are read-only.  This script extracts scalar metadata
from JSON and small torch ZIP archives without importing torch, then writes only
inside this analysis unit's ``plots`` directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import pickle
import re
import statistics
import subprocess
import sys
import zipfile
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[1]
PLOTS = UNIT / "plots"
FIGURE_BENCHMARKS = frozenset({"harmful", "truthfulness", "mgsm"})


PERFORMANCE_SOURCES = {
    "harmful": ROOT / "figs/benchmark_figures/cache/harmbench_refusal_results.csv",
    "truthfulness": ROOT / "figs/benchmark_figures/cache/truthfulqa_figure_data.csv",
    "mgsm": ROOT / "figs/benchmark_figures/cache/mgsm_transfer_results.csv",
    "lciteeval": ROOT / "figs/benchmark_figures/cache/lciteeval_context_results.csv",
}


CONTROLLER_FIELDS = [
    "benchmark", "model_key", "model_id", "model_revision", "model_label",
    "model_family", "parameter_count", "calibration_id", "calibration_state",
    "configuration_id", "is_selected", "feasible", "gamma_star", "gamma_used",
    "s_rob", "negative_log_gamma_star", "setpoint_multiplier", "q", "r",
    "q_final", "q_over_r", "q_final_over_r", "selection_metric",
    "tuning_samples", "tuning_repetitions", "normalization_protocol_id",
    "protocol_id", "coordinates", "min_disturbance_margin", "min_control_margin",
    "max_condition_number_m", "max_condition_number_h", "max_norm_s", "max_norm_k",
    "bisection_iterations", "bisection_converged", "controller_path",
    "controller_sha256", "selection_path",
]

SELECTION_FIELDS = [
    "benchmark", "model_key", "model_id", "model_revision", "model_label",
    "model_family", "parameter_count", "calibration_id", "calibration_state",
    "selected_configuration_id", "gamma_star", "gamma_used", "s_rob",
    "negative_log_gamma_star", "setpoint_multiplier", "q", "r", "q_final",
    "q_over_r", "q_final_over_r", "selection_metric", "selection_metric_description",
    "tuning_samples", "tuning_repetitions", "fit_count", "calibration_count",
    "normalization_protocol_id", "protocol_id", "coordinates", "feasible",
    "zero_disturbance_channel", "covariance_error_mean", "covariance_error_median",
    "covariance_error_max", "calibration_metrics_json", "selection_path",
    "score_path", "selection_sha256", "score_sha256",
]

PERFORMANCE_FIELDS = [
    "benchmark", "model_key", "model_label", "condition_axis", "condition",
    "distribution", "method", "metric", "value", "standard_error",
    "higher_is_better", "unit", "source_path", "source_sha256",
]

PAIR_FIELDS = [
    "benchmark", "model_key", "model_label", "model_family", "parameter_count",
    "calibration_id", "calibration_state", "calibration_mapping_status",
    "condition_axis", "condition", "distribution", "outcome_metric",
    "ood_reliability", "best_non_hinf_reliability", "hinf_advantage",
    "gamma_star", "s_rob", "negative_log_gamma_star", "selection_metric",
    "performance_source", "selection_source",
]

FINAL_POINT_FIELDS = [
    "benchmark", "model_key", "model_label", "model_family", "parameter_count",
    "calibration_id", "condition_axis", "condition", "outcome_metric",
    "ood_reliability", "gamma_star", "s_rob", "selection_metric",
    "performance_source", "selection_source",
]

FINAL_CALIBRATIONS = {
    ("harmful", "llama32_1b"): "selected",
    ("harmful", "llama32_3b"): "selected",
    ("harmful", "llama31_8b"): "selected",
    ("truthfulness", "gpt2_xl"): "txi95_fluency05_n200_r1",
    ("truthfulness", "llama31_8b"): "txi_fluency_95_05",
    ("truthfulness", "qwen25_14b"): "selected",
    ("truthfulness", "olmo2_32b"): "txi95_fluency05_n200_r1",
    ("mgsm", "qwen3_4b"): "selected",
    ("mgsm", "phi4_mini"): "selected",
    ("mgsm", "granite33_2b"): "selected",
    ("lciteeval", "qwen25_3b"): "selected",
    ("lciteeval", "llama32_1b"): "selected",
}

def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return None


class TorchMetadataUnpickler(pickle.Unpickler):
    """Read metadata from trusted local torch ZIPs while discarding tensors."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch._utils" and name.startswith("_rebuild_tensor"):
            return lambda *args: {"_tensor_omitted": True}
        if module == "torch" and name.endswith("Storage"):
            return type(name, (), {})
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        raise pickle.UnpicklingError(f"unsupported global {module}.{name}")

    def persistent_load(self, pid: Any) -> Any:
        return {"_storage_omitted": True, "descriptor": repr(pid)}


def load_torch_metadata(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(members) != 1:
            raise ValueError(f"expected one data.pkl in {path}, found {len(members)}")
        result = TorchMetadataUnpickler(io.BytesIO(archive.read(members[0]))).load()
    if not isinstance(result, dict):
        raise TypeError(f"controller payload is not a dictionary: {path}")
    return result


def model_key(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", text.lower())
    aliases = [
        ("llama328b", "llama32_8b"),
        ("llama318b", "llama31_8b"),
        ("llama323b", "llama32_3b"),
        ("llama321b", "llama32_1b"),
        ("llama38b", "llama31_8b"),
        ("gemma22b", "gemma2_2b"),
        ("gemma34b", "gemma3_4b"),
        ("gpt2xl", "gpt2_xl"),
        ("gpt2large", "gpt2_large"),
        ("gpt2medium", "gpt2_medium"),
        ("distilgpt2", "distilgpt2"),
        ("pythia160m", "pythia_160m"),
        ("pythia31m", "pythia_31m"),
        ("pythia14m", "pythia_14m"),
        ("smollm2135m", "smollm2_135m"),
        ("qwen2505b", "qwen25_05b"),
        ("qwen2532b", "qwen25_32b"),
        ("qwen2514b", "qwen25_14b"),
        ("qwen253b", "qwen25_3b"),
        ("qwen34b", "qwen3_4b"),
        ("olmo2032532b", "olmo2_32b"),
        ("olmo232b", "olmo2_32b"),
        ("phi4mini", "phi4_mini"),
        ("granite332b", "granite33_2b"),
        ("openaicommunitygpt2", "gpt2_small"),
        ("gpt2small", "gpt2_small"),
    ]
    for token, key in aliases:
        if token in normalized:
            return key
    return normalized


def calibration_state(path: Path) -> str:
    lowered = path.as_posix().lower()
    if "superseded" in lowered:
        return "superseded"
    if "selection_history" in lowered or "/history/" in lowered:
        return "historical"
    return "current_candidate"


def score_candidates(calibration_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    found: list[tuple[Path, dict[str, Any]]] = []
    run_root = calibration_dir / "controller_diagnostics/runs"
    for path in sorted(run_root.glob("*/score.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        found.append((path, payload))
    return found


def matching_score(
    candidates: list[tuple[Path, dict[str, Any]]], gamma_star: float | None
) -> tuple[Path | None, dict[str, Any]]:
    if not candidates:
        return None, {}
    if gamma_star is not None:
        finite = [
            item for item in candidates
            if isinstance(item[1].get("gamma_star"), (int, float))
        ]
        if finite:
            return min(finite, key=lambda item: abs(float(item[1]["gamma_star"]) - gamma_star))
    return candidates[-1]


def mean_or_none(values: Iterable[Any]) -> float | None:
    numbers = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return statistics.fmean(numbers) if numbers else None


def median_or_none(values: Iterable[Any]) -> float | None:
    numbers = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return statistics.median(numbers) if numbers else None


def max_or_none(values: Iterable[Any]) -> float | None:
    numbers = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return max(numbers) if numbers else None


def extract_calibrations() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    controller_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    pattern = "*/cache/*/calibrations/h_infinity/**/selection.json"
    for selection_path in sorted((ROOT / "benchmarks").glob(pattern)):
        payload = json.loads(selection_path.read_text())
        calibration_dir = selection_path.parent
        benchmark = str(payload.get("benchmark") or selection_path.parts[-7])
        identity = payload.get("model") or ["", ""]
        mid = str(identity[0]) if identity else ""
        revision = str(identity[1]) if len(identity) > 1 else ""
        selected = payload.get("selected") or {}
        selected_parameters = selected.get("parameters") or {}
        protocol = payload.get("protocol") or {}
        selected_id = str(
            selected.get("configuration_id")
            or selected.get("grid_id")
            or selected.get("candidate_id")
            or selected_parameters.get("grid_id")
            or ""
        )
        selected_controller = calibration_dir / "controller.pt"
        selected_meta: dict[str, Any] = {}
        if selected_controller.exists():
            try:
                selected_meta = load_torch_metadata(selected_controller)
            except Exception as exc:  # preserve the inventory and report the unreadable source
                warnings.append(f"{relative(selected_controller)}: {type(exc).__name__}: {exc}")
        selected_artifact = selected_meta.get("artifact") or {}
        gamma = selected.get("gamma_star")
        if not isinstance(gamma, (int, float)):
            gamma = selected_meta.get("gamma_star")
        if not isinstance(gamma, (int, float)):
            gamma = selected_artifact.get("gamma_star")
        gamma = float(gamma) if isinstance(gamma, (int, float)) else None
        score_path, score = matching_score(score_candidates(calibration_dir), gamma)
        label = str(score.get("model_label") or mid.rsplit("/", 1)[-1])
        family = str(score.get("model_family") or "")
        params = score.get("parameter_count")
        diagnostics = (
            selected_meta.get("diagnostics")
            or selected_artifact.get("hinf_diagnostics")
            or score.get("diagnostics")
            or {}
        )
        gamma_used = diagnostics.get("gamma_used")
        s_rob = (1.0 / gamma) if gamma is not None and gamma > 0 else None
        covariance_errors = score.get("covariance_relative_error_by_layer") or []
        selected_metrics = {
            key: value for key, value in selected.items()
            if key not in {
                "grid_id", "configuration_id", "parameters", "source", "lambda",
                "q", "r", "q_final", "q_over_r", "q_final_over_r", "gamma_star",
            } and scalar(value) is not None
        }
        state = calibration_state(selection_path)
        selection_rows.append({
            "benchmark": benchmark,
            "model_key": model_key(mid),
            "model_id": mid,
            "model_revision": revision,
            "model_label": label,
            "model_family": family,
            "parameter_count": params,
            "calibration_id": payload.get("calibration_id") or calibration_dir.name,
            "calibration_state": state,
            "selected_configuration_id": selected_id,
            "gamma_star": gamma,
            "gamma_used": gamma_used,
            "s_rob": s_rob,
            "negative_log_gamma_star": -math.log(gamma) if gamma and gamma > 0 else None,
            "setpoint_multiplier": selected.get("lambda", selected_parameters.get("lambda")),
            "q": selected.get("q", selected_parameters.get("q")),
            "r": selected.get("r", selected_parameters.get("r")),
            "q_final": selected.get("q_final", selected_parameters.get("q_final")),
            "q_over_r": selected.get("q_over_r", selected_parameters.get("q_over_r")),
            "q_final_over_r": selected.get("q_final_over_r", selected_parameters.get("q_final_over_r")),
            "selection_metric": protocol.get("selection_metric"),
            "selection_metric_description": protocol.get("selection_metric_description"),
            "tuning_samples": protocol.get("tuning_samples"),
            "tuning_repetitions": protocol.get("tuning_repetitions"),
            "fit_count": score.get("fit_count"),
            "calibration_count": score.get("calibration_count"),
            "normalization_protocol_id": score.get("normalization_protocol_id"),
            "protocol_id": score.get("protocol_id"),
            "coordinates": score.get("coordinates"),
            "feasible": selected_meta.get("feasible", selected_artifact.get("hinf_feasible", score.get("feasible"))),
            "zero_disturbance_channel": score.get("zero_disturbance_channel"),
            "covariance_error_mean": mean_or_none(covariance_errors),
            "covariance_error_median": median_or_none(covariance_errors),
            "covariance_error_max": max_or_none(covariance_errors),
            "calibration_metrics_json": json.dumps(selected_metrics, sort_keys=True, separators=(",", ":")),
            "selection_path": relative(selection_path),
            "score_path": relative(score_path) if score_path else "",
            "selection_sha256": sha256(selection_path),
            "score_sha256": sha256(score_path) if score_path else "",
        })

        controller_paths = sorted((calibration_dir / "grid/controllers").glob("*.pt"))
        if not controller_paths and selected_controller.exists():
            controller_paths = [selected_controller]
        for controller_path in controller_paths:
            try:
                controller = load_torch_metadata(controller_path)
            except Exception as exc:
                warnings.append(f"{relative(controller_path)}: {type(exc).__name__}: {exc}")
                continue
            controller_identity = controller.get("identity") or {}
            controller_params = controller_identity.get("parameters") or {}
            controller_artifact = controller.get("artifact") or {}
            controller_diag = controller.get("diagnostics") or controller_artifact.get("hinf_diagnostics") or {}
            config_id = str(
                controller_identity.get("configuration_id")
                or controller_identity.get("selection_candidate")
                or controller_params.get("grid_id")
                or controller_path.stem
            )
            controller_gamma = controller.get("gamma_star", controller_artifact.get("gamma_star"))
            controller_gamma = float(controller_gamma) if isinstance(controller_gamma, (int, float)) else None
            controller_rows.append({
                "benchmark": benchmark,
                "model_key": model_key(mid),
                "model_id": mid,
                "model_revision": revision,
                "model_label": label,
                "model_family": family,
                "parameter_count": params,
                "calibration_id": payload.get("calibration_id") or calibration_dir.name,
                "calibration_state": state,
                "configuration_id": config_id,
                "is_selected": config_id == selected_id,
                "feasible": controller.get("feasible", controller_artifact.get("hinf_feasible")),
                "gamma_star": controller_gamma,
                "gamma_used": controller_diag.get("gamma_used"),
                "s_rob": 1.0 / controller_gamma if controller_gamma and controller_gamma > 0 else None,
                "negative_log_gamma_star": -math.log(controller_gamma) if controller_gamma and controller_gamma > 0 else None,
                "setpoint_multiplier": controller_params.get("lambda"),
                "q": controller_params.get("q"),
                "r": controller_params.get("r"),
                "q_final": controller_params.get("q_final"),
                "q_over_r": (
                    controller_params.get("q") / controller_params.get("r")
                    if controller_params.get("r") else None
                ),
                "q_final_over_r": (
                    controller_params.get("q_final") / controller_params.get("r")
                    if controller_params.get("r") else None
                ),
                "selection_metric": protocol.get("selection_metric"),
                "tuning_samples": protocol.get("tuning_samples"),
                "tuning_repetitions": protocol.get("tuning_repetitions"),
                "normalization_protocol_id": score.get("normalization_protocol_id"),
                "protocol_id": score.get("protocol_id"),
                "coordinates": score.get("coordinates"),
                "min_disturbance_margin": controller_diag.get("min_disturbance_margin"),
                "min_control_margin": controller_diag.get("min_control_margin"),
                "max_condition_number_m": controller_diag.get("max_condition_number_M"),
                "max_condition_number_h": controller_diag.get("max_condition_number_H"),
                "max_norm_s": controller_diag.get("max_norm_S"),
                "max_norm_k": controller_diag.get("max_norm_K"),
                "bisection_iterations": controller_diag.get("bisection_iterations"),
                "bisection_converged": controller_diag.get("bisection_converged"),
                "controller_path": relative(controller_path),
                "controller_sha256": sha256(controller_path),
                "selection_path": relative(selection_path),
            })
    return controller_rows, selection_rows, warnings


def append_metrics(
    output: list[dict[str, Any]], benchmark: str, model: str, axis: str,
    condition: str, distribution: str, method: str, raw: dict[str, str],
    metrics: dict[str, tuple[bool, str]], source: Path,
) -> None:
    for metric, (higher, unit) in metrics.items():
        text = raw.get(metric, "")
        if text == "":
            continue
        output.append({
            "benchmark": benchmark,
            "model_key": model_key(model),
            "model_label": model,
            "condition_axis": axis,
            "condition": condition,
            "distribution": distribution,
            "method": raw.get("method", method),
            "metric": metric,
            "value": float(text),
            "standard_error": float(raw[f"{metric}_se"]) if raw.get(f"{metric}_se", "") else None,
            "higher_is_better": higher,
            "unit": unit,
            "source_path": relative(source),
            "source_sha256": sha256(source),
        })


def extract_performance() -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for benchmark, source in PERFORMANCE_SOURCES.items():
        if not source.exists():
            warnings.append(f"missing canonical performance source: {relative(source)}")
            continue
        with source.open(newline="") as handle:
            records = list(csv.DictReader(handle))
        for raw in records:
            model = raw["model"]
            method = raw["method"]
            if benchmark == "harmful":
                condition = raw["template"]
                distribution = "id" if condition.lower() == "direct" else "ood"
                metrics = {
                    "asr": (False, "percent"),
                    "safe_concept_relevance": (True, "judge_score_0_2"),
                    "instruction_relevance": (True, "judge_score_0_2"),
                    "fluency": (True, "judge_score_0_2"),
                    "overall_steering": (True, "judge_score_0_2"),
                }
                append_metrics(rows, benchmark, model, "template", condition, distribution, method, raw, metrics, source)
            elif benchmark == "truthfulness":
                condition = raw["split"]
                distribution = "id" if condition.upper() == "ID" else "ood"
                metrics = {
                    "true": (True, "percent"),
                    "relevance": (True, "judge_score_0_2"),
                    "fluency": (True, "judge_score_0_2"),
                }
                append_metrics(rows, benchmark, model, "language_split", condition, distribution, method, raw, metrics, source)
            elif benchmark == "mgsm":
                condition = raw["language"]
                metrics = {
                    "accuracy": (True, "percent"),
                    "target_relevance": (True, "judge_score_0_2"),
                    "instruction_relevance": (True, "judge_score_0_2"),
                    "fluency": (True, "judge_score_0_2"),
                }
                append_metrics(rows, benchmark, model, "language", condition, "ood", method, raw, metrics, source)
            elif benchmark == "lciteeval":
                condition = f"{raw['context_k']}K"
                distribution = "id" if raw["context_k"] == "8" else "ood"
                metrics = {
                    "answer_recall": (True, "percent"),
                    "citation_f1": (True, "percent"),
                    "steering_quality": (True, "judge_score_0_2"),
                }
                append_metrics(rows, benchmark, model, "context_length", condition, distribution, method, raw, metrics, source)

    toxicity_methods = {
        "original": "Original", "spid": "S-PID", "alqr": "A-LQR", "h_infinity": "H-infinity"
    }
    for source in sorted((ROOT / "benchmarks/toxicity/cache").glob(
        "*/evaluations/kv_cache_off/results/rtp/*.json"
    )):
        payload = json.loads(source.read_text())
        identity = payload.get("identity") or {}
        model_identity = identity.get("model") or [source.parts[-7]]
        model = str(model_identity[0])
        method_key = str(identity.get("method") or source.stem)
        method = toxicity_methods.get(method_key, method_key)
        for metric, values in (payload.get("metrics") or {}).items():
            if not isinstance(values, dict) or not isinstance(values.get("mean"), (int, float)):
                continue
            higher = metric not in {"toxicity", "perplexity"}
            unit = "percent" if metric == "toxicity" else ("ratio" if metric == "dist_2" else "score")
            rows.append({
                "benchmark": "toxicity",
                "model_key": model_key(model),
                "model_label": model.rsplit("/", 1)[-1],
                "condition_axis": "dataset",
                "condition": "RealToxicityPrompts",
                "distribution": "ood",
                "method": method,
                "metric": metric,
                "value": float(values["mean"]),
                "standard_error": values.get("standard_error"),
                "higher_is_better": higher,
                "unit": unit,
                "source_path": relative(source),
                "source_sha256": sha256(source),
            })

    spanish_metric_map = {
        "lcite_answer_bilingual.score": ("answer_recall", 100.0, "percent", True),
        "lcite_citation_bilingual.citation_f1": ("citation_f1", 100.0, "percent", True),
        "lcite_citation_bilingual.citation_precision": ("citation_precision", 100.0, "percent", True),
        "lcite_citation_bilingual.citation_recall": ("citation_recall", 100.0, "percent", True),
        "axbench_rule_spanish.score": ("spanish_adherence", 1.0, "judge_score_0_2", True),
        "axbench_instruction_relevance.score": ("instruction_relevance", 1.0, "judge_score_0_2", True),
        "axbench_fluency.score": ("fluency", 1.0, "judge_score_0_2", True),
        "axbench_spanish_overall.score": ("overall_steering", 1.0, "judge_score_0_2", True),
    }
    for source in sorted((ROOT / "benchmarks/lciteeval_spanish/cache").glob(
        "*/evaluations/kv_cache_off/summaries/hotpotqa_*/*.json"
    )):
        payload = json.loads(source.read_text())
        identity = payload.get("identity") or {}
        model_identity = payload.get("model") or identity.get("model") or [source.parts[-7]]
        model = str(model_identity[0])
        condition = str(payload.get("condition") or identity.get("condition") or source.parent.name).upper()
        method_key = str(payload.get("method") or identity.get("method") or source.stem)
        method = toxicity_methods.get(method_key, method_key)
        distribution = "id" if condition == "8K" else "ood"
        for raw_metric, (metric, factor, unit, higher) in spanish_metric_map.items():
            value = (payload.get("metrics") or {}).get(raw_metric)
            if not isinstance(value, (int, float)):
                continue
            rows.append({
                "benchmark": "lciteeval_spanish",
                "model_key": model_key(model),
                "model_label": model.rsplit("/", 1)[-1],
                "condition_axis": "context_length",
                "condition": condition,
                "distribution": distribution,
                "method": method,
                "metric": metric,
                "value": float(value) * factor,
                "standard_error": None,
                "higher_is_better": higher,
                "unit": unit,
                "source_path": relative(source),
                "source_sha256": sha256(source),
            })
    return rows, warnings


def method_is_hinf(method: str) -> bool:
    token = re.sub(r"[^a-z]", "", method.lower())
    return token in {"hinfinity", "hinfinityours", "hinf"}


def reliability(metric: str, value: float) -> float | None:
    if metric in {"asr", "toxicity"}:
        return 100.0 - value
    if metric in {"true", "accuracy", "answer_recall", "citation_f1"}:
        return value
    return None


def build_pairs(
    selections: list[dict[str, Any]], performance: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    active_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in selections:
        if row["calibration_state"] == "current_candidate":
            active_counts[(row["benchmark"], row["model_key"])] += 1
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in performance:
        if reliability(row["metric"], float(row["value"])) is not None:
            grouped[(row["benchmark"], row["model_key"], row["condition"], row["metric"])].append(row)
    pairs: list[dict[str, Any]] = []
    for selection in selections:
        if selection["calibration_state"] != "current_candidate":
            continue
        key_prefix = (selection["benchmark"], selection["model_key"])
        mapping = "unique_current_candidate" if active_counts[key_prefix] == 1 else "ambiguous_multiple_current_candidates"
        for (benchmark, mkey, condition, metric), candidates in grouped.items():
            if (benchmark, mkey) != key_prefix:
                continue
            ours = [row for row in candidates if method_is_hinf(str(row["method"]))]
            baselines = [row for row in candidates if not method_is_hinf(str(row["method"]))]
            if len(ours) != 1:
                continue
            ours_value = reliability(metric, float(ours[0]["value"]))
            baseline_values = [reliability(metric, float(row["value"])) for row in baselines]
            baseline_values = [value for value in baseline_values if value is not None]
            best = max(baseline_values) if baseline_values else None
            pairs.append({
                "benchmark": benchmark,
                "model_key": mkey,
                "model_label": ours[0]["model_label"],
                "model_family": selection["model_family"],
                "parameter_count": selection["parameter_count"],
                "calibration_id": selection["calibration_id"],
                "calibration_state": selection["calibration_state"],
                "calibration_mapping_status": mapping,
                "condition_axis": ours[0]["condition_axis"],
                "condition": condition,
                "distribution": ours[0]["distribution"],
                "outcome_metric": metric,
                "ood_reliability": ours_value,
                "best_non_hinf_reliability": best,
                "hinf_advantage": ours_value - best if ours_value is not None and best is not None else None,
                "gamma_star": selection["gamma_star"],
                "s_rob": selection["s_rob"],
                "negative_log_gamma_star": selection["negative_log_gamma_star"],
                "selection_metric": selection["selection_metric"],
                "performance_source": ours[0]["source_path"],
                "selection_source": selection["selection_path"],
            })
    return pairs


def markdown_rows(path: Path) -> list[dict[str, str]]:
    lines = path.read_text().splitlines()
    header_index = next(
        index for index, line in enumerate(lines)
        if line.startswith("|") and index + 1 < len(lines) and lines[index + 1].startswith("|---")
    )
    headers = [cell.strip() for cell in lines[header_index].strip("|").split("|")]
    output: list[dict[str, str]] = []
    for line in lines[header_index + 2:]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == len(headers):
            output.append(dict(zip(headers, cells)))
    return output


def mean_from_cell(cell: str) -> float:
    return float(cell.split("±", 1)[0].strip())


def build_final_report_points(selections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_index = {
        (row["benchmark"], row["model_key"], row["calibration_id"]): row
        for row in selections
        if row["calibration_state"] == "current_candidate"
    }
    raw_points: list[dict[str, Any]] = []

    truth_path = ROOT / "figs/bench_table/truthfulness/truthfulqa_spanish.md"
    for row in markdown_rows(truth_path):
        if row["Method"] == "H∞ (ours)":
            raw_points.append({
                "benchmark": "truthfulness", "model_label": row["Model"],
                "condition_axis": "language_split", "condition": "Spanish",
                "outcome_metric": "true", "ood_reliability": mean_from_cell(row["True (%) ↑"]),
                "performance_source": relative(truth_path),
            })

    harm_path = ROOT / "figs/bench_table/harmful/harmbench_full.md"
    for row in markdown_rows(harm_path):
        if row["Method"] == "H∞ (ours)" and row["Template"] != "Direct":
            raw_points.append({
                "benchmark": "harmful", "model_label": row["Model"],
                "condition_axis": "template", "condition": row["Template"],
                "outcome_metric": "safe_response_rate",
                "ood_reliability": 100.0 - mean_from_cell(row["ASR (%) ↓"]),
                "performance_source": relative(harm_path),
            })

    mgsm_path = ROOT / "figs/bench_table/mgsm/mgsm_full.md"
    for row in markdown_rows(mgsm_path):
        if row["Method"] == "H∞ (ours)":
            raw_points.append({
                "benchmark": "mgsm", "model_label": row["Model"],
                "condition_axis": "language", "condition": row["Language"],
                "outcome_metric": "accuracy",
                "ood_reliability": mean_from_cell(row["Accuracy (%) ↑"]),
                "performance_source": relative(mgsm_path),
            })

    lcite_path = ROOT / "figs/bench_table/lciteeval/lciteeval_full.md"
    for row in markdown_rows(lcite_path):
        if row["Method"] != "H∞ (ours)" or row["Context"] != "16K":
            continue
        for column, metric in (
            ("Answer recall (%) ↑", "answer_recall"),
            ("Citation F1 (%) ↑", "citation_f1"),
        ):
            raw_points.append({
                "benchmark": "lciteeval", "model_label": row["Model"],
                "condition_axis": "context_length", "condition": "16K",
                "outcome_metric": metric, "ood_reliability": mean_from_cell(row[column]),
                "performance_source": relative(lcite_path),
            })

    final: list[dict[str, Any]] = []
    for point in raw_points:
        mkey = model_key(point["model_label"])
        calibration_id = FINAL_CALIBRATIONS.get((point["benchmark"], mkey))
        if calibration_id is None:
            raise KeyError(f"final calibration is not registered for {(point['benchmark'], mkey)}")
        selection = selected_index.get((point["benchmark"], mkey, calibration_id))
        if selection is None:
            raise KeyError(f"registered final calibration is unavailable for {(point['benchmark'], mkey, calibration_id)}")
        final.append({
            **point,
            "model_key": mkey,
            "model_family": selection["model_family"],
            "parameter_count": selection["parameter_count"],
            "calibration_id": calibration_id,
            "gamma_star": selection["gamma_star"],
            "s_rob": selection["s_rob"],
            "selection_metric": selection["selection_metric"],
            "selection_source": selection["selection_path"],
        })
    return final


def is_final_calibration(row: dict[str, Any]) -> bool:
    """Return true only for the calibration explicitly bound to a final paper row."""
    expected = FINAL_CALIBRATIONS.get((row["benchmark"], row["model_key"]))
    return (
        expected is not None
        and row["calibration_id"] == expected
        and row["calibration_state"] == "current_candidate"
    )


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def render_reliability_figure() -> None:
    renderer = UNIT / "render_srob_figure.py"
    candidates = [Path("/home/dev/miniconda3/bin/python3.13"), Path(sys.executable)]
    plotting_python: Path | None = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        probe = subprocess.run(
            [str(candidate), "-c", "import matplotlib,numpy"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if probe.returncode == 0:
            plotting_python = candidate
            break
    if plotting_python is None:
        raise RuntimeError("no Python environment with matplotlib and numpy was found")
    subprocess.run(
        [
            str(plotting_python), str(renderer),
            "--input", str(PLOTS / "final_report_points.csv"),
            "--output-prefix", str(PLOTS / "srob_ood_reliability"),
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate sources and report counts without writing")
    args = parser.parse_args()
    controllers, selections, calibration_warnings = extract_calibrations()
    final_selections = [row for row in selections if is_final_calibration(row)]
    final_controllers = [row for row in controllers if is_final_calibration(row)]
    final_points = [
        row for row in build_final_report_points(final_selections)
        if row["benchmark"] in FIGURE_BENCHMARKS
    ]
    expected_selections = len(FINAL_CALIBRATIONS)
    if len(final_selections) != expected_selections:
        raise RuntimeError(
            f"expected {expected_selections} final calibrations, found {len(final_selections)}"
        )
    warnings = calibration_warnings
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "definitions": {
            "gamma_star": "minimum feasible H-infinity attenuation boundary saved by synthesis",
            "s_rob": "1 / gamma_star in the saved controller coordinates",
            "negative_log_gamma_star": "-log(gamma_star)",
            "ood_reliability": "benchmark-native higher-is-better H-infinity outcome; ASR is transformed as 100-ASR",
            "hinf_advantage": "H-infinity reliability minus the best non-H-infinity method in the same model/condition/metric",
        },
        "caveat": (
            "Raw gamma_star and s_rob retain coordinate, cost, model-scale, and calibration-protocol dependence. "
            "Do not pool them as commensurate without a declared reporting normalization or grouped analysis."
        ),
        "counts": {
            "final_controller_configurations": len(final_controllers),
            "final_calibration_selections": len(final_selections),
            "final_report_points": len(final_points),
        },
        "final_figure_benchmarks": sorted({row["benchmark"] for row in final_points}),
        "warnings": warnings,
        "outputs": {
            "final_controller_grid": "plots/final_controller_grid.csv",
            "final_selected_calibrations": "plots/final_selected_calibrations.csv",
            "final_report_points": "plots/final_report_points.csv",
        },
    }
    if args.check:
        print(json.dumps(summary, indent=2))
        return
    PLOTS.mkdir(parents=True, exist_ok=True)
    for legacy_name in (
        "controller_grid.csv", "selected_calibrations.csv", "performance_long.csv",
        "analysis_candidates.csv", "srob_ood_reliability_stats.json",
    ):
        (PLOTS / legacy_name).unlink(missing_ok=True)
    write_csv(PLOTS / "final_controller_grid.csv", CONTROLLER_FIELDS, final_controllers)
    write_csv(PLOTS / "final_selected_calibrations.csv", SELECTION_FIELDS, final_selections)
    write_csv(PLOTS / "final_report_points.csv", FINAL_POINT_FIELDS, final_points)
    render_reliability_figure()
    summary["outputs"].update({
        "srob_ood_reliability_pdf": "plots/srob_ood_reliability.pdf",
        "srob_ood_reliability_png": "plots/srob_ood_reliability.png",
    })
    (PLOTS / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
