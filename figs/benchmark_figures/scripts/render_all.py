#!/usr/bin/env python3
"""Refresh benchmark data and render every figure in this unit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
PLOTS = SCRIPTS.parent / "plots"


def main() -> None:
    subprocess.run([sys.executable, SCRIPTS / "refresh_data.py"], check=True)
    for path in sorted(PLOTS.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
    for name in (
        "plot_truth_quality_frontier.py",
        "plot_truthfulqa_model_summary.py",
        "plot_truthfulqa_gain_matrix.py",
        "plot_transfer_results.py",
        "plot_harmbench_results.py",
    ):
        subprocess.run([sys.executable, SCRIPTS / name], check=True)


if __name__ == "__main__":
    main()
