"""Small helpers for versioned experiment artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


def configuration_hash(configuration: Mapping[str, object]) -> str:
    """Return the stable hash used to validate cached experiment artifacts."""

    payload = json.dumps(dict(configuration), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def implementation_hash(unit_dir: Path | None = None) -> str:
    """Bind caches to actual source bytes, not a manually maintained version tag."""
    package = Path(__file__).resolve().parent
    sources = {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(package.rglob("*.py"))}
    if unit_dir is not None:
        sources.update({"unit/" + path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted(unit_dir.glob("*.py"))})
    return configuration_hash(sources)
