"""Export a paper-style OOD toxicity table from a run_steering results CSV.

Input rows (per method): label, subset, baseline_toxicity_mean,
steered_toxicity_mean, run_status. Emits one Original row per
(label, subset) plus one row per method, with toxicity_percent and
percent_change vs Original, tagged with a provenance run_status.

Usage:
  python AppliedControler/export_ood_paper_style.py \
    --results results_reports/ood_runs/steering_results_ood_global3_distilgpt2_smoke.csv \
    --output results_reports/paper_style_table_ood_global3_smoke.csv \
    --provenance smoke_test
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "results_reports")

METHOD_LABELS = {
    "alqr": "A-LQR",
    "spid": "S-PID",
    "new_method": "H-infinity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default=os.path.join(
            REPORTS, "ood_runs", "steering_results_ood_global3_distilgpt2_smoke.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(REPORTS, "paper_style_table_ood_global3_smoke.csv"),
    )
    parser.add_argument("--provenance", default="smoke_test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.results)

    rows: list[dict[str, object]] = []
    for (label, subset), group in raw.groupby(["label", "subset"], sort=False):
        baseline = float(group["baseline_toxicity_mean"].iloc[0])
        rows.append(
            {
                "label": label,
                "subset": subset,
                "method": "Original",
                "toxicity_percent": baseline * 100.0,
                "percent_change": 0.0,
                "eval_prompts": int(group["eval_prompts"].iloc[0]),
                "run_status": args.provenance,
            }
        )
        for _, row in group.iterrows():
            method = METHOD_LABELS.get(str(row["method"]))
            if method is None:
                continue
            steered = float(row["steered_toxicity_mean"])
            status = str(row["run_status"])
            provenance = (
                args.provenance if status == "ok" else f"{args.provenance}_{status}"
            )
            rows.append(
                {
                    "label": label,
                    "subset": subset,
                    "method": method,
                    "toxicity_percent": steered * 100.0,
                    "percent_change": (steered - baseline) / baseline * 100.0
                    if baseline > 0
                    else float("nan"),
                    "eval_prompts": int(row["eval_prompts"]),
                    "run_status": provenance,
                }
            )

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(out)} rows)")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
