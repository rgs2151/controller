"""Local multi-GPU subprocess scheduling used inside one SSH host."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from robust_steerability.benchmarks.layout import REPO_ROOT


def run_jobs(
    jobs: list[tuple[str, list[str]]],
    devices: list[str],
    log_root: Path,
) -> None:
    """Run at most one job per requested device and fail on any worker error."""

    if not devices or len(devices) != len(set(devices)):
        raise ValueError("Devices must be a nonempty list of distinct CUDA devices")
    log_root.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    active: list[tuple[str, str, subprocess.Popen, object, Path]] = []
    available = list(devices)
    failures = []
    while pending or active:
        while pending and available:
            label, command = pending.pop(0)
            device = available.pop(0)
            command = [device if token == "{device}" else token for token in command]
            log_path = log_root / f"{label}.log"
            handle = log_path.open("a")
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active.append((label, device, process, handle, log_path))
        completed_index = next(
            (
                index
                for index, (_label, _device, process, _handle, _path) in enumerate(active)
                if process.poll() is not None
            ),
            None,
        )
        if completed_index is None:
            time.sleep(0.5)
            continue
        label, device, process, handle, log_path = active.pop(completed_index)
        return_code = process.returncode
        handle.close()
        available.append(device)
        if return_code:
            failures.append({
                "job": label, "return_code": return_code, "log": str(log_path)
            })
            for _label, _device, running, running_handle, _path in active:
                running.terminate()
                running.wait()
                running_handle.close()
            break
    if failures:
        raise RuntimeError(f"Benchmark workers failed: {failures}")


def run_data_shards(
    label: str,
    command: list[str],
    devices: list[str],
    log_root: Path,
) -> None:
    """Run one method on every GPU by assigning one data shard per device."""

    shard_count = len(devices)
    jobs = [
        (
            f"{label}-shard-{shard_index:02d}",
            [
                *command,
                "--shard-index", str(shard_index),
                "--shard-count", str(shard_count),
            ],
        )
        for shard_index in range(shard_count)
    ]
    run_jobs(jobs, devices, log_root)
