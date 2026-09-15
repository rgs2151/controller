"""Machine-local run records for directly launched benchmark stages."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import torch

from robust_steerability.benchmarks.layout import REPO_ROOT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _git(command: list[str]) -> str:
    return subprocess.check_output(command, cwd=REPO_ROOT, text=True).strip()


def machine_provenance() -> dict[str, object]:
    packages: dict[str, str | None] = {}
    for name in ("accelerate", "bitsandbytes", "datasets", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda_devices = []
    if torch.cuda.is_available():
        cuda_devices = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    return {
        "hostname": socket.gethostname(),
        "lightning_cloudspace_id": os.environ.get("LIGHTNING_CLOUDSPACE_ID"),
        "lightning_cloudspace_host": os.environ.get("LIGHTNING_CLOUDSPACE_HOST"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_devices": cuda_devices,
        "packages": packages,
    }


def default_run_id(benchmark: str, model: str, stage: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{benchmark}-{model}-{stage}-{stamp}-{socket.gethostname()}"


@contextmanager
def tracked_stage(
    *, run_id: str, benchmark: str, model: str, stage: str,
    methods: list[str], datasets: list[str], devices: str, use_cache: bool,
    calibration_id: str | None, parameters: dict[str, object],
) -> Iterator[Path]:
    """Record stage ownership and completion without controlling another host."""

    run_root = REPO_ROOT / "logs" / run_id
    record_path = run_root / "run.json"
    if record_path.exists():
        raise FileExistsError(f"Run id {run_id!r} already exists")
    record = {
        "schema_version": 1, "run_id": run_id, "status": "running",
        "started_at_utc": utc_now(), "benchmark": benchmark, "model": model,
        "stage": stage, "methods": methods, "datasets": datasets,
        "evaluated_model_kv_cache": use_cache, "devices": devices,
        "calibration_id": calibration_id, "parameters": parameters,
        "command": list(sys.argv), "working_directory": str(Path.cwd()),
        "git": {
            "commit": _git(["git", "rev-parse", "HEAD"]),
            "dirty": bool(_git(["git", "status", "--porcelain"])),
            "branch": _git(["git", "branch", "--show-current"]),
        },
        "machine": machine_provenance(), "published_s3_objects": [],
    }
    started = time.perf_counter()
    _write_json(record_path, record)
    try:
        yield run_root
    except BaseException as error:
        record.update({
            "status": "failed", "finished_at_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "error": {"type": type(error).__name__, "message": str(error)},
        })
        _write_json(record_path, record)
        raise
    else:
        record.update({
            "status": "complete",
            "finished_at_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
        })
        _write_json(record_path, record)
