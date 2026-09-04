"""Small helpers for versioned experiment artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def configuration_hash(configuration: Mapping[str, object]) -> str:
    """Return the stable hash used to validate cached experiment artifacts."""

    payload = json.dumps(dict(configuration), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()
