#!/usr/bin/env python3
"""Recompute S_rob = 1 / gamma_star from the bundled final selections."""
import csv
from pathlib import Path

root = Path(__file__).resolve().parent
with (root / "final_selected_calibrations.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
fields = ["benchmark", "model_key", "model_label", "calibration_id", "gamma_star", "s_rob_recomputed"]
with (root / "derived_srob.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        gamma = float(row["gamma_star"])
        writer.writerow({
            "benchmark": row["benchmark"], "model_key": row["model_key"],
            "model_label": row["model_label"], "calibration_id": row["calibration_id"],
            "gamma_star": gamma, "s_rob_recomputed": 1.0 / gamma,
        })
print(f"wrote {len(rows)} rows to derived_srob.csv")
