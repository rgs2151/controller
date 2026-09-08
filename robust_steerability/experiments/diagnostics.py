"""Portable H-infinity analysis bundles, adapted from Hannah's diagnostic exporter.

No Hugging Face dependency: readers need only PyTorch/NumPy and this package.
The frozen reference remains in ref/h_infinity_optimization.py.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import shutil
import tempfile
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from robust_steerability.control import HInfinityController, HInfinityOptions
from robust_steerability.control.types import ControllerSolution, FiniteHorizonControlProblem
from robust_steerability.control.validation import validate_control_problem, validate_controller_solution

PREDICTORS = (
    "log_parameter_count", "probe_accuracy", "semantic_snr", "linearization_error",
    "jacobian_subspace_similarity", "gramian_metric", "nominal_lqr_objective",
    "minimum_nominal_control_energy", "s_rob", "negative_log_gamma_star",
)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_tensors(value):
    """Keep portable bundles restricted to ordinary containers and CPU tensors."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {key: cpu_tensors(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tensors(item) for item in value)
    return value


def json_safe(value):
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def verify_run(directory: Path) -> None:
    manifest = read_json(directory / "manifest.json")
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported diagnostic schema")
    for name, expected in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("Frozen artifact paths must be local filenames")
        if sha256(directory / name) != expected:
            raise ValueError(f"Frozen score artifact changed: {directory / name}")


def safe_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("IDs must start with a letter/digit and contain only letters, digits, _, ., -")
    return value


def prompt_ids(values: list[str]) -> set[str]:
    if not values or any(not isinstance(v, str) or not v for v in values):
        raise ValueError("Prompt IDs must be nonempty dataset-qualified strings")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate prompt IDs")
    return set(values)


def score(bundle_path: Path, device: str, *, cache_root: Path,
          solution: ControllerSolution | None = None) -> Path:
    """Freeze one model-behavior score without reading evaluation outcomes."""
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    required = {"problem", "record", "splits", "normalization", "calibration", "predictors"}
    if set(bundle) - (required | {"options"}) or not required <= set(bundle):
        raise ValueError(f"Bundle must contain {sorted(required)} and optionally options")
    record = bundle["record"]
    for key in ("run_id", "model_id", "model_revision", "model_family", "behavior",
                "intervention_channel", "protocol_id", "parameter_count", "synthetic"):
        if key not in record:
            raise ValueError(f"Missing record field: {key}")
    run_id = safe_id(record["run_id"])
    if type(record["synthetic"]) is not bool or not math.isfinite(float(record["parameter_count"])) or record["parameter_count"] <= 0:
        raise ValueError("synthetic must be boolean; parameter_count must be positive")
    splits = bundle["splits"]
    if set(splits) != {"fit", "calibration"}:
        raise ValueError("Score input splits must contain fit and calibration only")
    fit_ids, cal_ids = prompt_ids(splits["fit"]), prompt_ids(splits["calibration"])
    if fit_ids & cal_ids:
        raise ValueError("Fit and calibration prompts overlap")
    normalization = bundle["normalization"]
    for key in ("protocol_id", "coordinates", "state_whitening", "control_std",
                "semantic_output_std", "depth_increment", "stage_costs_depth_weighted"):
        if key not in normalization:
            raise ValueError(f"Missing normalization field: {key}")
    if normalization["coordinates"] not in {"normalized", "raw"}:
        raise ValueError("coordinates must be normalized or raw")
    problem = FiniteHorizonControlProblem(**bundle["problem"])
    validate_control_problem(problem)
    if problem.disturbance_channels is None or min(
        problem.horizon, problem.state_dimension, problem.control_dimension,
        problem.disturbance_dimension,
    ) < 1:
        raise ValueError("All problem dimensions must be positive and D is required")
    if normalization["coordinates"] == "normalized":
        whitening = normalization["state_whitening"]
        if whitening.shape != (problem.horizon + 1, problem.state_dimension, problem.state_dimension):
            raise ValueError("state_whitening must have shape (T+1,n,n)")
        if not torch.isfinite(whitening).all() or torch.linalg.svdvals(whitening.double()).min() <= 0:
            raise ValueError("State whitening must be finite and nonsingular")
        for key in ("control_std", "semantic_output_std", "depth_increment"):
            values = normalization[key]
            if not isinstance(values, torch.Tensor) or not torch.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f"{key} must be a positive finite tensor")
        if normalization["control_std"].shape != (problem.horizon, problem.control_dimension):
            raise ValueError("control_std must have shape (T,m)")
        depth = normalization["depth_increment"]
        if depth.shape != (problem.horizon,) or not torch.isclose(depth.sum(), torch.tensor(1.0, dtype=depth.dtype)):
            raise ValueError("depth_increment must have length T and sum to one")
        if normalization["stage_costs_depth_weighted"] is not True:
            raise ValueError("Normalized cross-model scores require depth-weighted stage costs")
    for key in ("state_costs", "terminal_cost", "control_costs"):
        cost = getattr(problem, key).double()
        if not torch.allclose(cost, cost.transpose(-1, -2), atol=1e-7, rtol=1e-6):
            raise ValueError(f"{key} must be symmetric")
        minimum = torch.linalg.eigvalsh(cost).min().item()
        if minimum < -1e-8 or (key == "control_costs" and minimum <= 0):
            raise ValueError(f"{key} must be PSD (strictly PD for control costs)")
    calibration = bundle["calibration"]
    for key in ("residuals", "state_basis", "target_readouts", "protected_readouts",
                "reference_states", "reference_controls", "disturbance_construction"):
        if key not in calibration:
            raise ValueError(f"Missing calibration field: {key}")
    target = calibration["target_readouts"]
    if not isinstance(target, torch.Tensor) or target.ndim != 3 or target.shape[0] != problem.horizon + 1 or target.shape[2] != problem.state_dimension:
        raise ValueError("target_readouts must have shape (T+1,p,n)")
    if not torch.isfinite(target).all():
        raise ValueError("target_readouts must be finite")
    if normalization["coordinates"] == "normalized" and normalization["semantic_output_std"].shape != target.shape[:2]:
        raise ValueError("semantic_output_std must have shape (T+1,p)")
    residuals = calibration["residuals"]
    expected = (len(cal_ids), problem.horizon, problem.state_dimension)
    if residuals.shape != expected or not torch.isfinite(residuals).all() or len(cal_ids) < 2:
        raise ValueError(f"Calibration residuals must be finite and have shape {expected}, N>=2")
    # Baselines are provided by their owning fit/calibration analyses with definitions.
    predictors = {name: None for name in PREDICTORS}
    predictors["log_parameter_count"] = math.log(float(record["parameter_count"]))
    for name, entry in bundle["predictors"].items():
        if name not in PREDICTORS[1:-2]:
            raise ValueError(f"Unknown or computed predictor: {name}")
        if entry["split"] not in {"fit", "calibration"} or not entry["definition"]:
            raise ValueError(f"Predictor {name} needs a fit/calibration source and definition")
        if entry["value"] is not None and not math.isfinite(float(entry["value"])):
            raise ValueError(f"Nonfinite predictor: {name}")
        predictors[name] = entry["value"]
    options = HInfinityOptions(**bundle.get("options", {}))
    option_values = asdict(options)
    if not all(math.isfinite(float(v)) for v in option_values.values()):
        raise ValueError("Synthesis options must be finite")
    destination = cache_root / "runs" / run_id
    fingerprint = {"input_sha256": sha256(bundle_path),
                   "controller_sha256": sha256(Path(inspect.getfile(HInfinityController))),
                   "exporter_sha256": sha256(Path(__file__)), "device": device}
    if solution is not None:
        solution_digest = hashlib.sha256(solution.gains.detach().cpu().contiguous().numpy().tobytes())
        solution_digest.update(json.dumps(json_safe({"gamma_star": solution.gamma_star,
                               "feasible": solution.feasible, "diagnostics": solution.diagnostics}),
                               sort_keys=True).encode())
        fingerprint["solution_sha256"] = solution_digest.hexdigest()
    if destination.exists():
        manifest = destination / "manifest.json"
        if not manifest.exists() or read_json(manifest)["fingerprint"] != fingerprint:
            raise ValueError("Run ID already exists with different or incomplete content; use a new run_id")
        verify_run(destination)
        return destination
    if solution is None:
        with torch.no_grad():
            solution = HInfinityController.synthesize(problem, device=device, options=options).solution()
    validate_controller_solution(problem, solution)
    if solution.controller != "h_infinity":
        raise ValueError("Expected the H-infinity solution")
    converged = bool(solution.diagnostics.get("bisection_converged", False))
    gamma = solution.gamma_star
    zero_disturbance = bool(torch.count_nonzero(problem.disturbance_channels) == 0)
    valid_score = solution.feasible and converged and gamma is not None and math.isfinite(gamma) and gamma > 0 and not zero_disturbance
    if valid_score:
        predictors["s_rob"] = 1.0 / gamma
        predictors["negative_log_gamma_star"] = -math.log(gamma)
    centered = residuals.double() - residuals.double().mean(dim=0, keepdim=True)
    covariance = torch.einsum("nti,ntj->tij", centered, centered) / (len(cal_ids) - 1)
    d = problem.disturbance_channels.double()
    covariance_error = torch.linalg.matrix_norm(covariance - d @ d.transpose(-1, -2), dim=(-2, -1))
    covariance_norm = torch.linalg.matrix_norm(covariance, dim=(-2, -1))
    covariance_relative_error = torch.where(covariance_norm > 0, covariance_error / covariance_norm, torch.full_like(covariance_norm, float("nan")))
    metadata = {**record, "gamma_star": gamma, "gamma_used": solution.diagnostics.get("gamma_used"),
                "s_rob": predictors["s_rob"], "negative_log_gamma_star": predictors["negative_log_gamma_star"],
                "feasible": solution.feasible, "bisection_converged": converged,
                "score_available": valid_score, "predictors": predictors,
                "zero_disturbance_channel": zero_disturbance,
                "score_status": "available" if valid_score else "zero disturbance gives an infinite ideal score" if zero_disturbance else "infeasible or unconverged",
                "normalization_protocol_id": normalization["protocol_id"],
                "coordinates": normalization["coordinates"],
                "stage_costs_depth_weighted": normalization["stage_costs_depth_weighted"],
                "score_definition": "1 / numerical feasible upper boundary gamma_star",
                "diagnostics": solution.diagnostics,
                "fit_count": len(fit_ids), "calibration_count": len(cal_ids),
                "covariance_relative_error_by_layer": covariance_relative_error}
    destination.parent.mkdir(parents=True, exist_ok=True)
    final_destination = destination
    destination = Path(tempfile.mkdtemp(prefix=".freeze-", dir=destination.parent))
    # Keep the exact input bytes, including bases, normalizers, residuals and provenance.
    (destination / "calibration_input.pt").write_bytes(bundle_path.read_bytes())
    (destination / "controller_source.py").write_bytes(Path(inspect.getfile(HInfinityController)).read_bytes())
    (destination / "exporter_source.py").write_bytes(Path(__file__).read_bytes())
    torch.save(cpu_tensors({"controller": solution.controller, "gains": solution.gains,
                "feasible": solution.feasible, "gamma_star": gamma,
                "diagnostics": solution.diagnostics,
                "control_channels": problem.control_channels.cpu(),
                "residual_covariance": covariance,
                "covariance_relative_error_by_layer": covariance_relative_error}), destination / "controller.pt")
    write_json(destination / "score.json", metadata)
    write_json(destination / "manifest.json", {"schema_version": 1, "fingerprint": fingerprint,
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "torch_version": str(torch.__version__), "options": option_values,
               "files": {p.name: sha256(p) for p in destination.iterdir() if p.is_file()}})
    destination.rename(final_destination)
    return final_destination


def evaluate(path: Path, *, cache_root: Path) -> Path:
    """Attach prompt-level held-out records to an already frozen score."""
    payload = read_json(path)
    run_id, evaluation_id = safe_id(payload["run_id"]), safe_id(payload["evaluation_id"])
    run = cache_root / "runs" / run_id
    verify_run(run)
    manifest = read_json(run / "manifest.json")
    if payload["score_manifest_sha256"] != sha256(run / "manifest.json"):
        raise ValueError("Evaluation must reference the frozen score manifest hash")
    if datetime.fromisoformat(payload["evaluation_started_at_utc"]) <= datetime.fromisoformat(manifest["created_at_utc"]):
        raise ValueError("Evaluation must start after the score was frozen")
    bundle = torch.load(run / "calibration_input.pt", map_location="cpu", weights_only=True)
    excluded = set(bundle["splits"]["fit"]) | set(bundle["splits"]["calibration"])
    score_record = read_json(run / "score.json")
    for field in ("controller", "shift", "protocol_id", "success_definition",
                  "matching_rule", "generation_config", "evaluator"):
        if field not in payload or payload[field] in (None, "", {}):
            raise ValueError(f"Missing evaluation field: {field}")
    if payload["protocol_id"] != score_record["protocol_id"]:
        raise ValueError("Evaluation protocol differs from score protocol")
    observations = payload["observations"]
    if not observations:
        raise ValueError("No held-out observations")
    keys = set()
    for row in observations:
        for field in ("prompt_id", "seed", "success", "raw_score", "control_energy", "collateral_metrics"):
            if field not in row:
                raise ValueError(f"Missing observation field: {field}")
        if not isinstance(row["prompt_id"], str) or not row["prompt_id"]:
            raise ValueError("Prompt IDs must be nonempty dataset-qualified strings")
        key = (row["prompt_id"], row["seed"])
        if key in keys or row["prompt_id"] in excluded:
            raise ValueError("Duplicate prompt/seed or held-out prompt overlaps fit/calibration")
        if type(row["success"]) is not bool:
            raise ValueError("success must be Boolean using a predefined behavioral threshold")
        if not all(math.isfinite(float(row[v])) for v in ("raw_score", "control_energy")) or row["control_energy"] < 0:
            raise ValueError("Scores and nonnegative control energies must be finite")
        keys.add(key)
    target = run / "evaluations" / evaluation_id
    if target.exists():
        if (target / "observations.json").read_bytes() != path.read_bytes():
            raise ValueError("Evaluation ID exists with different content; use a new evaluation_id")
        load_run(run)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    final_target = target
    target = Path(tempfile.mkdtemp(prefix=".evaluation-", dir=target.parent))
    (target / "observations.json").write_bytes(path.read_bytes())
    by_prompt = {}
    for row in observations:
        by_prompt.setdefault(row["prompt_id"], []).append(float(row["success"]))
    reliability = float(np.mean([np.mean(v) for v in by_prompt.values()]))
    write_json(target / "summary.json", {"run_id": run_id, "evaluation_id": evaluation_id,
               "controller": payload["controller"], "shift": payload["shift"],
               "protocol_id": payload["protocol_id"], "n_prompts": len(by_prompt),
               "n_generations": len(observations), "reliability": reliability,
               "reliability_percent": 100 * reliability, "source_sha256": sha256(path),
               "success_definition": payload["success_definition"], "matching_rule": payload["matching_rule"]})
    write_json(target / "manifest.json", {"files": {
        "observations.json": sha256(target / "observations.json"),
        "summary.json": sha256(target / "summary.json"),
    }})
    target.rename(final_target)
    return final_target


def copy_run(source: Path, cache_root: Path) -> Path:
    """Import a frozen calibration without writing to another unit's cache."""
    verify_run(source)
    target = cache_root / "runs" / safe_id(source.name)
    if target.exists():
        verify_run(target)
        if sha256(target / "manifest.json") != sha256(source / "manifest.json"):
            raise ValueError("Diagnostic run ID collision")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".import-", dir=target.parent))
    for name in [*read_json(source / "manifest.json")["files"], "manifest.json"]:
        shutil.copyfile(source / name, temporary / name)
    temporary.rename(target)
    return target


def load_run(directory: Path) -> dict:
    """Read an exported run on CPU; no model, token, GPU, or pickle objects needed."""
    verify_run(directory)
    benchmark = directory / "benchmark.json"
    if benchmark.exists():
        for name, expected in read_json(benchmark)["source_hashes"].items():
            if Path(name).name != name or sha256(directory / name) != expected:
                raise ValueError("Runtime source snapshot changed")
    evaluations = {}
    for path in sorted((directory / "evaluations").glob("*/manifest.json")):
        for name, expected in read_json(path)["files"].items():
            if Path(name).name != name or sha256(path.parent / name) != expected:
                raise ValueError(f"Evaluation artifact changed: {path.parent / name}")
        payload = read_json(path.parent / "observations.json")
        if payload["score_manifest_sha256"] != sha256(directory / "manifest.json"):
            raise ValueError("Evaluation refers to a different score")
        if "benchmark_config_sha256" in payload and payload["benchmark_config_sha256"] != sha256(benchmark):
            raise ValueError("Benchmark configuration changed")
        for row in payload["observations"]:
            if "trace_file" in row:
                trace = directory / row["trace_file"]
                if not trace.resolve().is_relative_to(directory.resolve()) or sha256(trace) != row["trace_sha256"]:
                    raise ValueError("Online trace changed or leaves the bundle")
        evaluations[path.parent.name] = payload
    return {"score": read_json(directory / "score.json"),
            "manifest": read_json(directory / "manifest.json"),
            "calibration": torch.load(directory / "calibration_input.pt", map_location="cpu", weights_only=True),
            "controller": torch.load(directory / "controller.pt", map_location="cpu", weights_only=True),
            "evaluations": evaluations}


def share_report(directory: Path, output: Path) -> Path:
    """Small Git-friendly inventory; never copies prompt text or tensor contents."""
    loaded = load_run(directory)
    files = {str(path.relative_to(directory)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted(directory.rglob("*")) if path.is_file()}
    report = {"schema_version": 1, "score": loaded["score"],
              "manifest_sha256": sha256(directory / "manifest.json"),
              "total_bytes": sum(entry["bytes"] for entry in files.values()), "files": files,
              "evaluations": [read_json(path) for path in sorted((directory / "evaluations").glob("*/summary.json"))],
              "sharing": "Metadata only. Full tensors and prompts remain in the ignored cache; share the ZIP separately."}
    write_json(output, report)
    return output


def pack_run(directory: Path, output: Path) -> Path:
    """Produce a self-contained ZIP for file transfer, without uploading anything."""
    load_run(directory)
    if output.resolve().is_relative_to(directory.resolve()):
        raise ValueError("ZIP output must be outside the run directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, Path(directory.name) / path.relative_to(directory))
    return output
