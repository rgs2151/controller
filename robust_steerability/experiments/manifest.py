"""Validated experiment-manifest schema."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from robust_steerability.artifacts import configuration_hash


ALLOWED_KINDS = {"truthfulness", "id_toxicity", "ood_toxicity", "calibration"}
ALLOWED_METHODS = {"original", "alqr", "spid", "hinf", "iti", "actadd", "mean_act", "linear_act", "pid_act", "odesteer"}
REQUIRED_CONTROLLER_FIELDS = {
    "seed", "fit_prompts_per_class", "disturbance_prompts", "calibration_max_length",
    "activation_batch_size", "jacobian_prompts", "jacobian_max_length",
    "jacobian_vjp_chunk_size", "state_rank", "numerical_floor",
    "disturbance_variance", "disturbance_coverage", "alqr_setpoint_multiplier",
    "spid_setpoint_multiplier", "hinf_setpoint_multiplier",
    "q", "r", "q_final", "alqr_q", "alqr_r", "alqr_q_final", "kp", "ki", "kd",
    "gamma_lower", "gamma_upper", "gamma_tolerance", "gamma_max_iterations",
    "gamma_deployment_margin", "behavior",
    "baseline_strengths",
}
OBSOLETE_CONTROLLER_FIELDS = {"ridge", "whitening_floor"}


@dataclass(frozen=True)
class ExperimentManifest:
    """One complete model-by-benchmark execution matrix."""

    path: Path
    payload: dict[str, object]
    fingerprint: str

    @property
    def kind(self) -> str:
        return str(self.payload["kind"])

    @property
    def unit_dir(self) -> Path:
        return self.path.parent

    @property
    def models(self) -> list[dict[str, object]]:
        return list(self.payload["models"])

    @property
    def methods(self) -> list[str]:
        return [str(value) for value in self.payload.get("methods", [])]


def load_manifest(path: str | Path) -> ExperimentManifest:
    """Read and validate a version-one benchmark manifest."""

    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    kind = str(payload.get("kind", ""))
    if kind not in ALLOWED_KINDS:
        raise ValueError(f"Unsupported experiment kind: {kind}")
    revisions = payload.get("revisions")
    if not isinstance(revisions, dict) or not revisions:
        raise ValueError("manifest revisions must pin every external dependency")
    for dependency, revision in revisions.items():
        if not isinstance(revision, str) or not revision or revision in {"main", "master"}:
            raise ValueError(
                f"dependency {dependency!r} requires an immutable revision"
            )
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("manifest models must be a non-empty list")
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("every model entry must be an object")
        missing = {"label", "model_id", "revision", "dtype", "quantized"} - set(model)
        if missing:
            raise ValueError(f"model entry is missing {sorted(missing)}")
        if str(model["revision"]) in {"main", "master"}:
            raise ValueError("model revisions must be immutable")
    methods = [str(value) for value in payload.get("methods", [])]
    if kind != "calibration":
        if not methods:
            raise ValueError("benchmark manifests require methods")
        unknown = set(methods) - ALLOWED_METHODS
        if unknown:
            raise ValueError(f"Unknown method names: {sorted(unknown)}")
        if len(methods) != len(set(methods)):
            raise ValueError("manifest methods must be unique")
        if int(payload.get("sample_count", 0)) < 1:
            raise ValueError("benchmark manifests require a positive sample_count")
        controller = payload.get("controller")
        if not isinstance(controller, dict):
            raise ValueError("benchmark manifests require controller settings")
        missing_controller = REQUIRED_CONTROLLER_FIELDS - set(controller)
        if missing_controller:
            raise ValueError(f"controller settings are missing {sorted(missing_controller)}")
        obsolete = OBSOLETE_CONTROLLER_FIELDS & set(controller)
        if obsolete:
            raise ValueError(f"obsolete controller settings are forbidden: {sorted(obsolete)}")
        if int(controller["jacobian_prompts"]) != int(controller["fit_prompts_per_class"]):
            raise ValueError("the 50-sample reference pilot requires every positive fit prompt for Jacobians")
        if int(controller["jacobian_max_length"]) != 24:
            raise ValueError("reference A-LQR Jacobians require a 24-token context cap")
        expected_baselines = {"iti", "actadd", "mean_act", "linear_act", "pid_act", "odesteer"}
        if set(controller["baseline_strengths"]) != expected_baselines:
            raise ValueError("baseline_strengths must specify every adapted baseline exactly once")
        if any(float(value) <= 0 for value in controller["baseline_strengths"].values()):
            raise ValueError("baseline strengths must be positive")
        mmlu = payload.get("subset_generation", {}).get("mmlu", {})
        if mmlu.get("do_sample") is not False or int(mmlu.get("max_new_tokens", 0)) != 1:
            raise ValueError("reference MMLU requires one greedy generated token")
    return ExperimentManifest(
        path=manifest_path,
        payload=payload,
        fingerprint=configuration_hash(payload),
    )
