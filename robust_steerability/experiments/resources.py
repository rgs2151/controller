"""Runtime resource selection for benchmark entry points."""

from __future__ import annotations

import torch


def resolve_cuda_devices(value: str) -> list[str]:
    """Resolve ``auto`` or a comma-separated list of visible CUDA devices."""

    if value == "auto":
        devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    else:
        devices = [item.strip() for item in value.split(",") if item.strip()]
    if not devices:
        raise RuntimeError("No CUDA devices are available")
    if len(devices) != len(set(devices)):
        raise ValueError("CUDA devices must be distinct")
    if any(not device.startswith("cuda:") for device in devices):
        raise ValueError("CUDA devices must use cuda:<index> names")
    return devices
