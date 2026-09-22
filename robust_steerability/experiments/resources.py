"""Runtime resource selection for benchmark entry points."""

from __future__ import annotations

import torch


DEVICE_GROUP_SEPARATOR = "+"


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


def cuda_device_group(devices: list[str]) -> str:
    """Encode multiple physical CUDA devices as one model-parallel worker."""

    if not devices or len(devices) != len(set(devices)):
        raise ValueError("A CUDA device group must contain distinct devices")
    if any(not device.startswith("cuda:") for device in devices):
        raise ValueError("CUDA device groups require cuda:<index> names")
    return DEVICE_GROUP_SEPARATOR.join(devices)


def cuda_devices_in_group(device: str) -> list[str]:
    """Decode one single-device or model-parallel worker specification."""

    devices = device.split(DEVICE_GROUP_SEPARATOR)
    if not devices or len(devices) != len(set(devices)):
        raise ValueError(f"Invalid CUDA device group: {device!r}")
    if any(not item.startswith("cuda:") for item in devices):
        raise ValueError(f"Invalid CUDA device group: {device!r}")
    return devices


def primary_cuda_device(device: str) -> str:
    """Return the first CUDA device assigned to one worker."""

    return cuda_devices_in_group(device)[0]


def group_cuda_workers(devices: list[str], devices_per_worker: int) -> list[str]:
    """Partition physical GPUs into equal model-parallel worker groups."""

    if devices_per_worker < 1:
        raise ValueError("devices_per_worker must be positive")
    if len(devices) % devices_per_worker:
        raise ValueError(
            f"{len(devices)} CUDA devices cannot be divided into workers of "
            f"{devices_per_worker}"
        )
    return [
        cuda_device_group(devices[start : start + devices_per_worker])
        for start in range(0, len(devices), devices_per_worker)
    ]
