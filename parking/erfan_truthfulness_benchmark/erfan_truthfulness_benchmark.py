"""Plot archived Erfan results; fresh execution lives in paper_benchmark_50."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


UNIT_DIR = Path(__file__).resolve().parent
MANIFEST = UNIT_DIR / "manifest.json"
RESULTS = UNIT_DIR / "plots" / "results.csv"
METHODS = ["Original", "A-LQR", "S-PID", "H-infinity"]
METHOD_TICK_LABELS = ["Orig.", "A-LQR", "S-PID", r"$H_\infty$"]
COLORS = {
    "Original": "#737373",
    "A-LQR": "midnightblue",
    "S-PID": "darkorange",
    "H-infinity": "darkgreen",
}


def setup_style() -> None:
    sns.set_theme(context="talk", style="ticks", palette="dark")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["lines.linewidth"] = 1
    plt.rcParams["patch.linewidth"] = 0
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300


def plot() -> None:
    setup_style()
    table = pd.read_csv(RESULTS)
    models = list(dict.fromkeys(table["model"].tolist()))
    metrics = [
        ("ti_mean", "ti_se", r"Truthful $\times$ informative (\%)"),
        ("true_mean", "true_se", r"Truthful (\%)"),
        ("info_mean", "info_se", r"Informative (\%)"),
        ("mmlu_mean", "mmlu_se", r"MMLU accuracy (\%)"),
    ]
    fig, axes = plt.subplots(
        len(models),
        len(metrics),
        figsize=(3.1 * len(metrics), 2.5 * len(models)),
        squeeze=False,
    )
    for row_index, model in enumerate(models):
        model_rows = table[table["model"] == model].set_index("method_label")
        for column_index, (mean_key, error_key, title) in enumerate(metrics):
            ax = axes[row_index, column_index]
            values = [float(model_rows.loc[method, mean_key]) for method in METHODS]
            errors = [float(model_rows.loc[method, error_key]) for method in METHODS]
            ax.bar(
                np.arange(len(METHODS)),
                values,
                yerr=errors,
                capsize=2,
                color=[COLORS[method] for method in METHODS],
            )
            ax.set_ylim(-5, 105)
            ax.set_yticks([0, 100])
            ax.set_xticks(np.arange(len(METHODS)))
            ax.set_xticklabels(METHOD_TICK_LABELS if row_index == len(models) - 1 else [])
            if row_index == len(models) - 1:
                ax.tick_params(axis="x", labelsize=7)
            if row_index == 0:
                ax.set_title(title, fontsize=10)
            if column_index == 0:
                ax.set_ylabel(model, fontsize=10)
            ax.set_box_aspect(1)
            sns.despine(ax=ax, trim=True, offset=10)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.08, top=0.95, wspace=0.5, hspace=0.55)
    for extension in ("pdf", "png"):
        fig.savefig(
            UNIT_DIR / "plots" / f"truthfulness_model_metrics.{extension}",
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["plot"])
    args = parser.parse_args()
    plot()


if __name__ == "__main__":
    main()
