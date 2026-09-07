"""Render the model-dissected summary figure.

One row per model; each row breaks that model down across the four
truthfulness metrics (T*I, True, Info, MMLU) with one bar per method.
Complements make_final_summary_figure.py (which groups models side by side).

Input (results_reports/):
  paper_style_table_truthfulness_all_models_methods.csv

Outputs (results_reports/):
  final_summary_model_dissected.png / final_summary_model_dissected.pdf
"""

from __future__ import annotations

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
        TRUTH_MODEL_ORDER as MODEL_ORDER,
        TRUTH_MODEL_LABELS as MODEL_LABELS,
        TRUTH_FULLY_RUN as FULLY_RUN,
        model_label_color,
    )
except ImportError:
    from figure_config import (
        METHODS,
        METHOD_COLORS,
        TRUTH_MODEL_ORDER as MODEL_ORDER,
        TRUTH_MODEL_LABELS as MODEL_LABELS,
        TRUTH_FULLY_RUN as FULLY_RUN,
        model_label_color,
    )

METRICS = [
    ("ti_mean", "ti_std", r"T$\cdot$I (%)  $\uparrow$"),
    ("true_mean", "true_std", r"True (%)  $\uparrow$"),
    ("info_mean", "info_std", r"Info (%)  $\uparrow$"),
    ("mmlu_mean", "mmlu_std", r"MMLU (%)  $\uparrow$"),
]


def main() -> None:
    truth = pd.read_csv(
        os.path.join(REPORTS, "paper_style_table_truthfulness_all_models_methods.csv")
    )
    truth = truth.drop_duplicates(subset=["model", "method"], keep="last")

    models = [m for m in MODEL_ORDER if (truth["model"] == m).any()]
    n_rows, n_cols = len(models), len(METRICS)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.1 * n_cols, 2.6 * n_rows), sharex=True
    )

    for i, model in enumerate(models):
        sub = truth[truth["model"] == model].set_index("method")
        for k, (mean_col, std_col, metric_label) in enumerate(METRICS):
            ax = axes[i, k]
            means, stds, missing = [], [], []
            for method in METHODS:
                if method not in sub.index or math.isnan(float(sub.loc[method, mean_col])):
                    means.append(0.0)
                    stds.append(0.0)
                    missing.append(True)
                else:
                    means.append(float(sub.loc[method, mean_col]))
                    stds.append(float(sub.loc[method, std_col]))
                    missing.append(False)
            bars = ax.bar(
                METHODS,
                means,
                yerr=stds,
                capsize=3,
                color=[METHOD_COLORS[m] for m in METHODS],
                edgecolor="black",
                linewidth=0.6,
            )
            for bar, miss in zip(bars, missing):
                if miss:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        2.0,
                        "N/A",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90,
                        color="#b30000",
                    )
            ax.set_ylim(0, 105)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(metric_label, fontsize=10)
            if k == 0:
                ax.set_ylabel(
                    MODEL_LABELS[model],
                    fontsize=10,
                    fontweight="bold",
                    color=model_label_color(model, FULLY_RUN),
                )
            if i == n_rows - 1:
                ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        "Control-theoretic steering of LLMs: per-model breakdown "
        "(rows: models, columns: metrics, bars: methods)",
        fontsize=12,
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        out = os.path.join(REPORTS, f"final_summary_model_dissected.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
