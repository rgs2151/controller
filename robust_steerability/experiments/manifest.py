"""Validated experiment-manifest schema."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from robust_steerability.artifacts import configuration_hash


ALLOWED_KINDS = {"truthfulness", "id_toxicity", "ood_toxicity", "calibration"}
ALLOWED_METHODS = {"original", "alqr", "spid", "hinf", "iti", "actadd", "mean_act", "linear_act", "pid_act", "odesteer"}


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
    return ExperimentManifest(
        path=manifest_path,
        payload=payload,
        fingerprint=configuration_hash(payload),
    )
