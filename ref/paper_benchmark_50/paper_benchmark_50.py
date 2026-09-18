"""Run the complete, fixed 50-prompt benchmark matrix on separate GPUs."""

import argparse
from pathlib import Path

from prepare_data import prepare
from score_quality import score
from robust_steerability.experiments.manifest import load_manifest
from robust_steerability.experiments.runner import run_manifest


UNIT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    args = parser.parse_args()
    devices = args.devices.split(",")
    if not devices or any(not device.startswith("cuda:") for device in devices):
        raise ValueError("This benchmark requires explicit CUDA devices")
    for name in ("toxicity", "truthfulness"):
        manifest = load_manifest(UNIT / f"{name}.json")
        if manifest.payload["sample_count"] != 50:
            raise ValueError("Use a separate analysis unit for a different sample count")
    prepare(devices[0])
    for name in ("toxicity", "truthfulness"):
        run_manifest(UNIT / f"{name}.json", devices)
    score(devices[0])


if __name__ == "__main__":
    main()
