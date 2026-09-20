"""Node-local Hugging Face model staging for remote benchmark runs."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_LIGHTNING_HF_CACHE = Path("/tmp/robust-steerability/huggingface")


def _is_lightning_studio() -> bool:
    return bool(
        os.environ.get("LIGHTNING_CLOUD_SPACE_ID")
        or os.environ.get("LIGHTNING_CLOUDSPACE_ID")
        or os.environ.get("LIGHTNING_RESOURCE_TYPE") == "cloudspace"
    )


def configure_node_local_hf_cache() -> Path | None:
    """Use ephemeral local storage for model weights on Lightning Studios."""

    configured = os.environ.get("HF_HUB_CACHE")
    if configured:
        return Path(configured)
    if not _is_lightning_studio():
        return None
    os.environ["HF_HUB_CACHE"] = str(DEFAULT_LIGHTNING_HF_CACHE)
    return DEFAULT_LIGHTNING_HF_CACHE


def _optional_access_token(repo_root: Path) -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = repo_root / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            token = line.split("=", 1)[1].strip().strip("\"'")
            if token:
                return token
    return None


def stage_model_snapshot(
    *, model_id: str, revision: str, repo_root: Path
) -> Path | None:
    """Download one pinned model snapshot once before GPU workers are launched."""

    cache_dir = configure_node_local_hf_cache()
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            token=_optional_access_token(repo_root),
            cache_dir=cache_dir,
        )
    )
