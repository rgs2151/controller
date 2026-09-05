"""Render the OOD steering summary figure (analogue of make_final_summary_figure.py).

One panel per global top-3 OOD benchmark (from residual analysis); each panel
groups models on the x-axis with one bar per method; y = mean continuation
toxicity (%) on that OOD prompt set (log scale). Missing model/subset data
renders as red rotated "N/A".

Input (results_reports/): paper_style_table_ood_global3[_smoke].csv
Outputs: final_summary_figure_ood[_smoke].png / .pdf
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "results_reports")

try:
    from AppliedControler.figure_config import (
        METHODS,
        METHOD_COLORS,
        NA_COLOR,
        OOD_SUBSETS,
        OOD_SUBSET_LABELS,
        TOX_MODEL_ORDER as MODEL_ORDER,
        TOX_MODEL_LABELS as MODEL_LABELS,
        OOD_FULLY_RUN as FULLY_RUN,
        model_label_color,
    )
except ImportError:
    from figure_config import (
        METHODS,
        METHOD_COLORS,
        NA_COLOR,
        OOD_SUBSETS,
        OOD_SUBSET_LABELS,
        TOX_MODEL_ORDER as MODEL_ORDER,
        TOX_MODEL_LABELS as MODEL_LABELS,
        OOD_FULLY_RUN as FULLY_RUN,
        model_label_color,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        default=os.path.join(REPORTS, "paper_style_table_ood_global3_smoke.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.table)
    smoke = "smoke" in os.path.basename(args.table)
    suffix = "_smoke" if smoke else ""

    fig, axes = plt.subplots(1, len(OOD_SUBSETS), figsize=(5.2 * len(OOD_SUBSETS), 4.6))

    n_methods = len(METHODS)
    width = 0.19
    x = np.arange(len(MODEL_ORDER))
    for p, subset in enumerate(OOD_SUBSETS):
        ax = axes[p]
        sub = df[df["subset"] == subset]
        for j, method in enumerate(METHODS):
            means, missing = [], []
            for model in MODEL_ORDER:
                row = sub[(sub["label"] == model) & (sub["method"] == method)]
                if row.empty or math.isnan(float(row["toxicity_percent"].iloc[0])):
                    means.append(0.0)
                    missing.append(True)
                else:
                    means.append(float(row["toxicity_percent"].iloc[0]))
                    missing.append(False)
            offs = x + (j - (n_methods - 1) / 2) * width
            bars = ax.bar(
                offs,
                means,
                width,
                label=method if p == 0 else None,
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.6,
            )
            for off, bar, mean, miss in zip(offs, bars, means, missing):
                if miss:
                    ax.text(
                        off,
                        1.5e-3,
                        "N/A",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90,
                        color=NA_COLOR,
                    )
                else:
                    ax.text(
                        off,
                        mean * 1.3,
                        f"{mean:.3g}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                    )
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 100)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=8)
        for tick, model in zip(ax.get_xticklabels(), MODEL_ORDER):
            tick.set_color(model_label_color(model, FULLY_RUN))
        ax.set_title(f"{chr(65 + p)}. {OOD_SUBSET_LABELS[subset]}")
        ax.spines[["top", "right"]].set_visible(False)
        if p == 0:
            ax.set_ylabel(r"OOD toxicity (%)  $\downarrow$  (log)")
            ax.legend(frameon=False, fontsize=9)

    tag = "  [SMOKE TEST - reduced samples]" if smoke else ""
    fig.suptitle(
        "Control-theoretic steering on top-3 OOD benchmarks "
        f"(residual analysis){tag}",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(REPORTS, f"final_summary_figure_ood{suffix}.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
