#!/usr/bin/env bash
set -euo pipefail

# Build the Python environment on the node's fast ephemeral disk. Benchmark
# artifacts and the repository remain in persistent Teamspace storage.
repo=${1:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
venv=${ROBUST_STEERING_NODE_VENV:-/tmp/robust-steerability-venv}
uv_cache=${UV_CACHE_DIR:-/tmp/robust-steerability-uv-cache}

rm -rf "$venv"
mkdir -p "$uv_cache"
uv venv --python /usr/bin/python3 "$venv"
UV_CACHE_DIR="$uv_cache" uv pip install --python "$venv/bin/python" -e "$repo"
printf '%s\n' "$venv/bin/python"
