"""Plot archived Erfan results; the superseded pilot lives in ref."""

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
    plt.rcParams["patch.linewidth"] = 0
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300


def plot() -> None:
    setup_style()
    table = pd.read_csv(RESULTS)
    models = list(dict.fromkeys(table["model"].tolist()))
    fig, axes = plt.subplots(len(models), 2, figsize=(7.4, 2.45 * len(models)), squeeze=False)
    for row_index, model in enumerate(models):
        rows = table[table["model"] == model].set_index("method_label")
        toxicity = [float(rows.loc[method, "toxicity_percent"]) for method in METHODS]
        reduction = [-float(rows.loc[method, "percent_change"]) for method in METHODS]
        for column_index, (values, title) in enumerate(
            [(toxicity, r"Continuation toxicity (\%)"), (reduction, r"Reduction from Original (\%)")]
        ):
            ax = axes[row_index, column_index]
            ax.bar(np.arange(len(METHODS)), values, color=[COLORS[method] for method in METHODS])
            upper = max(1.0, max(values) * 1.1)
            lower = min(0.0, min(values) * 1.1)
            ax.set_ylim(lower, upper)
            ax.set_yticks([lower, upper])
            ax.set_xticks(np.arange(len(METHODS)))
            ax.set_xticklabels(METHOD_TICK_LABELS if row_index == len(models) - 1 else [])
            if row_index == len(models) - 1:
                ax.tick_params(axis="x", labelsize=7)
            if row_index == 0:
                ax.set_title(title, fontsize=11)
            if column_index == 0:
                ax.set_ylabel(model, fontsize=10)
            ax.set_box_aspect(1)
            sns.despine(ax=ax, trim=True, offset=10)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.09, top=0.95, wspace=0.45, hspace=0.55)
    for extension in ("pdf", "png"):
        fig.savefig(UNIT_DIR / "plots" / f"id_toxicity_model_metrics.{extension}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["plot"])
    args = parser.parse_args()
    plot()


if __name__ == "__main__":
    main()
