"""Render the model-dissected OOD figures (analogues of
make_final_summary_model_dissected*.py).

Two figures, same structure (rows: models, columns: the 3 global top-3 OOD
benchmarks from residual analysis, bars: methods):
  1) final_summary_model_dissected_ood[_smoke]           - OOD toxicity % (log)
  2) final_summary_model_dissected_ood_reduction[_smoke] - reduction vs Original %

Input (results_reports/): paper_style_table_ood_global3[_smoke].csv
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def render(
    df: pd.DataFrame,
    value_col_transform,
    log_scale: bool,
    col_header_note: str,
    title: str,
    out_stem: str,
) -> None:
    n_rows, n_cols = len(MODEL_ORDER), len(OOD_SUBSETS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.2 * n_cols, 2.6 * n_rows), sharex=True
    )

    for i, model in enumerate(MODEL_ORDER):
        for k, subset in enumerate(OOD_SUBSETS):
            ax = axes[i, k]
            sub = df[(df["label"] == model) & (df["subset"] == subset)]
            means, missing = [], []
            for method in METHODS:
                row = sub[sub["method"] == method]
                value = (
                    value_col_transform(row.iloc[0]) if not row.empty else float("nan")
                )
                if isinstance(value, float) and math.isnan(value):
                    means.append(0.0)
                    missing.append(True)
                else:
                    means.append(float(value))
                    missing.append(False)

            bars = ax.bar(
                METHODS,
                means,
                color=[METHOD_COLORS[m] for m in METHODS],
                edgecolor="black",
                linewidth=0.6,
            )

            if log_scale:
                ax.set_yscale("log")
                ax.set_ylim(1e-3, 100)
                na_y = 1.5e-3
                for bar, mean, miss in zip(bars, means, missing):
                    if not miss:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            mean * 1.3,
                            f"{mean:.3g}",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                        )
            else:
                ax.set_ylim(0, 115)
                na_y = 2.0
                for bar, mean, miss in zip(bars, means, missing):
                    if not miss:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            mean + 2 if mean >= 0 else 2,
                            f"{mean:.1f}",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                        )

            for bar, miss in zip(bars, missing):
                if miss:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        na_y,
                        "N/A",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90,
                        color=NA_COLOR,
                    )

            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(
                    f"{OOD_SUBSET_LABELS[subset]}\n{col_header_note}", fontsize=10
                )
            if k == 0:
                ax.set_ylabel(
                    MODEL_LABELS[model],
                    fontsize=10,
                    fontweight="bold",
                    color=model_label_color(model, FULLY_RUN),
                )
            if i == n_rows - 1:
                ax.tick_params(axis="x", rotation=30)

    fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        out = os.path.join(REPORTS, f"{out_stem}.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.table)
    smoke = "smoke" in os.path.basename(args.table)
    suffix = "_smoke" if smoke else ""
    tag = "  [SMOKE TEST - reduced samples]" if smoke else ""

    render(
        df,
        value_col_transform=lambda row: float(row["toxicity_percent"]),
        log_scale=True,
        col_header_note=r"toxicity (%)  $\downarrow$  (log)",
        title=(
            "Per-model OOD toxicity breakdown "
            f"(rows: models, columns: OOD benchmarks, bars: methods){tag}"
        ),
        out_stem=f"final_summary_model_dissected_ood{suffix}",
    )
    render(
        df,
        value_col_transform=lambda row: -float(row["percent_change"]),
        log_scale=False,
        col_header_note=r"reduction vs Original (%)  $\uparrow$",
        title=(
            "Per-model OOD toxicity reduction "
            f"(rows: models, columns: OOD benchmarks, bars: methods){tag}"
        ),
        out_stem=f"final_summary_model_dissected_ood_reduction{suffix}",
    )


if __name__ == "__main__":
    main()
