"""Strict cache contract for nominal transformer dynamics shared by controllers."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from robust_steerability.artifacts import configuration_hash
from robust_steerability.calibration.nominal import average_prompt_jacobians
from robust_steerability.modeling import interventions, jacobians


SCHEMA_VERSION = 1


def nominal_dynamics_cache_path(
    cache_root: Path,
    *,
    behavior: str,
    model_id: str,
) -> Path:
    """Return the controller-neutral location used by A-LQR and H-infinity."""

    model_key = "".join(
        character.lower() if character.isalnum() else "_" for character in model_id
    ).strip("_")
    return cache_root / "nominal_dynamics" / behavior / model_key / "dynamics.pt"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _save_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def nominal_dynamics_identity(
    *,
    behavior: str,
    model_id: str,
    model_revision: str,
    records: list[dict],
    max_length: int,
    vjp_chunk_size: int,
) -> dict[str, object]:
    """Describe every scientific input to one averaged Jacobian artifact."""

    if not records:
        raise ValueError("Nominal dynamics requires at least one Jacobian prompt")
    normalized_records = []
    for record in records:
        if "prompt_id" not in record or "text" not in record:
            raise ValueError("Every Jacobian record requires prompt_id and text")
        normalized_records.append(
            {"prompt_id": str(record["prompt_id"]), "text": str(record["text"])}
        )
    implementation_paths = (
        Path(average_prompt_jacobians.__code__.co_filename),
        Path(jacobians.__file__),
        Path(interventions.__file__),
    )
    implementation = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in implementation_paths
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "averaged_last_token_transformer_jacobians",
        "behavior": behavior,
        "model_id": model_id,
        "model_revision": model_revision,
        "records": normalized_records,
        "record_count": len(normalized_records),
        "max_length": max_length,
        "vjp_chunk_size": vjp_chunk_size,
        "implementation_sha256": implementation,
        "fingerprint": configuration_hash(
            {
                "behavior": behavior,
                "model_id": model_id,
                "model_revision": model_revision,
                "records": normalized_records,
                "max_length": max_length,
                "vjp_chunk_size": vjp_chunk_size,
                "implementation_sha256": implementation,
            }
        ),
    }


def _validate_dynamics(dynamics: torch.Tensor, path: Path) -> None:
    if dynamics.ndim != 3 or dynamics.shape[-1] != dynamics.shape[-2]:
        raise ValueError(f"Nominal dynamics must have shape (layers, hidden, hidden): {path}")
    if not torch.isfinite(dynamics).all():
        raise ValueError(f"Nominal dynamics contains nonfinite values: {path}")


def load_nominal_dynamics(
    artifact_path: Path,
    identity: dict[str, object],
) -> torch.Tensor:
    """Load one complete artifact, rejecting any identity or integrity mismatch."""

    metadata_path = artifact_path.with_suffix(".json")
    if not artifact_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Incomplete nominal-dynamics cache: {artifact_path}")
    record = json.loads(metadata_path.read_text())
    if record.get("identity") != identity:
        raise ValueError(f"Nominal-dynamics cache identity mismatch: {artifact_path}")
    if record.get("status") != "complete":
        raise ValueError(f"Nominal-dynamics cache is not complete: {artifact_path}")
    if record.get("artifact_sha256") != _sha256(artifact_path):
        raise ValueError(f"Nominal-dynamics cache checksum mismatch: {artifact_path}")
    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "identity",
        "dynamics",
    }:
        raise ValueError(f"Invalid nominal-dynamics artifact structure: {artifact_path}")
    if payload["schema_version"] != SCHEMA_VERSION or payload["identity"] != identity:
        raise ValueError(f"Invalid nominal-dynamics artifact identity: {artifact_path}")
    dynamics = payload["dynamics"]
    _validate_dynamics(dynamics, artifact_path)
    return dynamics


def load_shared_nominal_dynamics(
    artifact_path: Path,
    *,
    behavior: str,
    model_id: str,
    model_revision: str,
) -> torch.Tensor:
    """Load an A-LQR-produced A matrix as the authoritative nominal model."""

    metadata_path = artifact_path.with_suffix(".json")
    if not artifact_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Incomplete nominal-dynamics cache: {artifact_path}")
    record = json.loads(metadata_path.read_text())
    identity = record.get("identity", {})
    expected = {
        "behavior": behavior,
        "model_id": model_id,
        "model_revision": model_revision,
    }
    actual = {key: identity.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"Nominal-dynamics source mismatch: {artifact_path}")
    return load_nominal_dynamics(artifact_path, identity)


def nominal_dynamics_signature(artifact_path: Path) -> dict[str, object]:
    """Return the verified identity and content hash used by downstream caches."""

    metadata_path = artifact_path.with_suffix(".json")
    if not artifact_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Incomplete nominal-dynamics cache: {artifact_path}")
    record = json.loads(metadata_path.read_text())
    if record.get("status") != "complete":
        raise ValueError(f"Nominal-dynamics cache is not complete: {artifact_path}")
    artifact_sha256 = _sha256(artifact_path)
    if record.get("artifact_sha256") != artifact_sha256:
        raise ValueError(f"Nominal-dynamics cache checksum mismatch: {artifact_path}")
    if "identity" not in record:
        raise ValueError(f"Nominal-dynamics cache has no identity: {artifact_path}")
    return {"identity": record["identity"], "artifact_sha256": artifact_sha256}


def save_nominal_dynamics(
    artifact_path: Path,
    identity: dict[str, object],
    dynamics: torch.Tensor,
    *,
    attempts: list[dict[str, object]],
) -> None:
    """Write the canonical shared artifact and its integrity metadata."""

    _validate_dynamics(dynamics, artifact_path)
    _save_torch(
        artifact_path,
        {
            "schema_version": SCHEMA_VERSION,
            "identity": identity,
            "dynamics": dynamics.cpu(),
        },
    )
    _write_json(
        artifact_path.with_suffix(".json"),
        {
            "identity": identity,
            "status": "complete",
            "attempts": attempts,
            "artifact_sha256": _sha256(artifact_path),
        },
    )


def load_or_fit_nominal_dynamics(
    model,
    tokenizer,
    records: list[dict],
    *,
    artifact_path: Path,
    behavior: str,
    model_id: str,
    model_revision: str,
    max_length: int,
    vjp_chunk_size: int,
    runtime: dict[str, object],
) -> torch.Tensor:
    """Reuse a matching A-LQR artifact, or compute it with the shared estimator."""

    identity = nominal_dynamics_identity(
        behavior=behavior,
        model_id=model_id,
        model_revision=model_revision,
        records=records,
        max_length=max_length,
        vjp_chunk_size=vjp_chunk_size,
    )
    metadata_path = artifact_path.with_suffix(".json")
    if artifact_path.exists():
        return load_nominal_dynamics(artifact_path, identity)
    if metadata_path.exists():
        record = json.loads(metadata_path.read_text())
        if record.get("identity") != identity or record.get("status") != "partial":
            raise ValueError(f"Nominal-dynamics cache identity mismatch: {artifact_path}")
    else:
        record = {"identity": identity, "status": "partial", "attempts": []}

    started = time.perf_counter()
    attempt = {
        "started_at_utc": _utc_now(),
        "status": "running",
        "runtime": runtime,
    }
    record["attempts"].append(attempt)
    _write_json(metadata_path, record)
    dynamics = average_prompt_jacobians(
        model,
        tokenizer,
        records,
        cache_dir=artifact_path.parent / "jacobians",
        max_length=max_length,
        vjp_chunk_size=vjp_chunk_size,
        model_revision=model_revision,
    )
    attempt["finished_at_utc"] = _utc_now()
    attempt["elapsed_seconds"] = time.perf_counter() - started
    attempt["status"] = "complete"
    save_nominal_dynamics(
        artifact_path,
        identity,
        dynamics,
        attempts=record["attempts"],
    )
    return dynamics


def reuse_or_fit_nominal_dynamics(
    model,
    tokenizer,
    records: list[dict],
    *,
    artifact_path: Path,
    behavior: str,
    model_id: str,
    model_revision: str,
    max_length: int,
    vjp_chunk_size: int,
    runtime: dict[str, object],
) -> torch.Tensor:
    """Reuse an authoritative shared A, computing it only when none exists."""

    if artifact_path.exists():
        return load_shared_nominal_dynamics(
            artifact_path,
            behavior=behavior,
            model_id=model_id,
            model_revision=model_revision,
        )
    return load_or_fit_nominal_dynamics(
        model,
        tokenizer,
        records,
        artifact_path=artifact_path,
        behavior=behavior,
        model_id=model_id,
        model_revision=model_revision,
        max_length=max_length,
        vjp_chunk_size=vjp_chunk_size,
        runtime=runtime,
    )
