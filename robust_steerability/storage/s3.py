"""Explicitly publish or fetch large model-benchmark artifacts through S3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from robust_steerability.benchmarks.layout import (
    artifact_root,
    calibration_root,
)


def _client():
    try:
        import boto3
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("Install the project dependencies to use S3 transport") from error
    endpoint = os.environ.get("ROBUST_STEERING_S3_ENDPOINT")
    return boto3.client("s3", endpoint_url=endpoint)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage_root(
    benchmark: str, model: str, stage: str, method: str | None,
    calibration_id: str,
) -> Path:
    if stage == "artifacts":
        return artifact_root(benchmark, model)
    if stage == "calibration":
        if method is None:
            raise ValueError("calibration transport requires --method")
        return calibration_root(benchmark, model, method, calibration_id)
    raise ValueError("Only large artifacts and calibrations belong in S3")


def _files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(path for path in root.rglob("*") if path.is_file())


def _base_key(
    prefix: str,
    benchmark: str,
    model: str,
    stage: str,
    method: str | None,
    calibration_id: str,
) -> str:
    parts = [prefix, benchmark, model, stage]
    if stage == "calibration":
        parts.extend([str(method), calibration_id])
    return "/".join(part.strip("/") for part in parts if part)


def push(
    *, bucket: str, prefix: str, benchmark: str, model: str, stage: str,
    method: str | None, calibration_id: str,
) -> dict[str, object]:
    root = _stage_root(benchmark, model, stage, method, calibration_id)
    client = _client()
    objects = []
    base_key = _base_key(
        prefix, benchmark, model, stage, method, calibration_id
    )
    for path in _files(root):
        relative = path.relative_to(root).as_posix()
        key = f"{base_key}/{relative}"
        client.upload_file(str(path), bucket, key)
        objects.append({
            "key": key, "bytes": path.stat().st_size, "sha256": _sha256(path)
        })
    manifest = {
        "schema_version": 1, "benchmark": benchmark, "model": model,
        "stage": stage, "method": method, "calibration_id": calibration_id,
        "objects": objects,
    }
    client.put_object(
        Bucket=bucket,
        Key=f"{base_key}/_manifest.json",
        Body=(json.dumps(manifest, indent=2) + "\n").encode(),
        ContentType="application/json",
    )
    return manifest


def pull(
    *, bucket: str, prefix: str, benchmark: str, model: str, stage: str,
    method: str | None, calibration_id: str,
) -> dict[str, object]:
    root = _stage_root(benchmark, model, stage, method, calibration_id)
    client = _client()
    base_key = _base_key(
        prefix, benchmark, model, stage, method, calibration_id
    )
    response = client.get_object(Bucket=bucket, Key=f"{base_key}/_manifest.json")
    manifest = json.loads(response["Body"].read())
    if {
        "benchmark": manifest.get("benchmark"), "model": manifest.get("model"),
        "stage": manifest.get("stage"), "method": manifest.get("method"),
        "calibration_id": manifest.get("calibration_id"),
    } != {
        "benchmark": benchmark, "model": model, "stage": stage,
        "method": method, "calibration_id": calibration_id,
    }:
        raise ValueError("S3 artifact manifest does not match the requested object")
    for item in manifest["objects"]:
        relative = Path(str(item["key"])).relative_to(base_key)
        destination = root / relative
        if destination.exists():
            if _sha256(destination) != item["sha256"]:
                raise FileExistsError(
                    f"Refusing to overwrite a different local artifact: {destination}"
                )
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        client.download_file(bucket, item["key"], str(temporary))
        if _sha256(temporary) != item["sha256"]:
            temporary.unlink()
            raise ValueError(f"Downloaded checksum mismatch: {item['key']}")
        temporary.replace(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("push", "pull"))
    parser.add_argument("--bucket", default=os.environ.get("ROBUST_STEERING_S3_BUCKET"))
    parser.add_argument("--prefix", default=os.environ.get("ROBUST_STEERING_S3_PREFIX", "robust-steering"))
    parser.add_argument("--benchmark", choices=("truthfulness", "toxicity"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stage", choices=("artifacts", "calibration"), required=True)
    parser.add_argument("--method")
    parser.add_argument("--calibration-id", default="selected")
    arguments = parser.parse_args()
    if not arguments.bucket:
        raise ValueError("Set ROBUST_STEERING_S3_BUCKET or pass --bucket")
    operation = push if arguments.action == "push" else pull
    manifest = operation(
        bucket=arguments.bucket, prefix=arguments.prefix,
        benchmark=arguments.benchmark, model=arguments.model,
        stage=arguments.stage, method=arguments.method,
        calibration_id=arguments.calibration_id,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
