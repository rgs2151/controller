"""Reusable control and language-model integration for robust steerability."""

from __future__ import annotations

import os


def _configure_lightning_node_local_caches() -> None:
    """Select ephemeral node-local caches before importing Hugging Face."""

    if not (
        os.environ.get("LIGHTNING_CLOUD_SPACE_ID")
        or os.environ.get("LIGHTNING_CLOUDSPACE_ID")
        or os.environ.get("LIGHTNING_RESOURCE_TYPE") == "cloudspace"
    ):
        return
    defaults = {
        "HF_HOME": "/tmp/robust-steerability",
        "HF_HUB_CACHE": "/tmp/robust-steerability/huggingface",
        "HF_DATASETS_CACHE": "/tmp/robust-steerability/datasets",
        "XDG_CACHE_HOME": "/tmp/robust-steerability/xdg",
    }
    for name, default in defaults.items():
        configured = os.environ.get(name)
        if not configured or configured.startswith("/teamspace/"):
            os.environ[name] = default
    transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
    if transformers_cache and transformers_cache.startswith("/teamspace/"):
        os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HUB_CACHE"]


_configure_lightning_node_local_caches()

from robust_steerability.control import (
    Controller,
    ControllerSolution,
    FiniteHorizonControlProblem,
    HInfinityController,
    LQRController,
)

__all__ = [
    "Controller",
    "ControllerSolution",
    "FiniteHorizonControlProblem",
    "HInfinityController",
    "LQRController",
]
