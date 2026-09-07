"""Combine the fresh Erfan truthfulness and ID-toxicity benchmark outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


UNIT_DIR = Path(__file__).resolve().parent
ROOT = UNIT_DIR.parents[1]
TRUTH = ROOT / "parking" / "erfan_truthfulness_benchmark" / "plots" / "results.csv"
TOXICITY = ROOT / "parking" / "erfan_id_toxicity_benchmark" / "plots" / "results.csv"
METHODS = ["Original", "A-LQR", "S-PID", "H-infinity"]
METHOD_LEGEND_LABELS = ["Original", "A-LQR", "S-PID", r"$H_\infty$"]
COLORS = {
    "Original": "#737373",
    "A-LQR": "midnightblue",
    "S-PID": "darkorange",
    "H-infinity": "darkgreen",
}
SHORT_MODEL_LABELS = {
    "DistilGPT-2": "D",
    "Qwen-2.5-0.5B": "0.5",
    "Qwen-2.5-1.5B": "1.5",
    "Qwen-2.5-7B": "7",
    "Qwen-2.5-14B": "14",
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


def main() -> None:
    setup_style()
    truth = pd.read_csv(TRUTH)
    toxicity = pd.read_csv(TOXICITY)
    models = list(dict.fromkeys(truth["model"].tolist()))
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    width = 0.18
    x = np.arange(len(models))
    for method_index, method in enumerate(METHODS):
        positions = x + (method_index - 1.5) * width
        truth_values = [
            float(truth[(truth["model"] == model) & (truth["method_label"] == method)]["ti_mean"].iloc[0])
            for model in models
        ]
        toxicity_values = [
            float(toxicity[(toxicity["model"] == model) & (toxicity["method_label"] == method)]["toxicity_percent"].iloc[0])
            for model in models
        ]
        axes[0].bar(
            positions,
            truth_values,
            width=width,
            color=COLORS[method],
            label=METHOD_LEGEND_LABELS[method_index],
        )
        axes[1].bar(positions, toxicity_values, width=width, color=COLORS[method])
    for ax, label, upper in [
        (axes[0], r"Truthful $\times$ informative (\%)", 100.0),
        (axes[1], r"RTP continuation toxicity (\%)", max(1.0, float(toxicity["toxicity_percent"].max()) * 1.1)),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(
            [SHORT_MODEL_LABELS[model] for model in models],
            rotation=30,
            ha="right",
            fontsize=6,
        )
        ax.set_ylim(0, upper)
        ax.set_yticks([0, upper])
        ax.set_ylabel(label)
        ax.set_box_aspect(1)
        ax.set_xlabel("Qwen scale (B); D = DistilGPT-2", fontsize=6)
        sns.despine(ax=ax, trim=True, offset=10)
    axes[0].set_title("Truthfulness")
    axes[1].set_title("In-distribution toxicity")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.20, top=0.92, wspace=0.45)
    for extension in ("pdf", "png"):
        fig.savefig(UNIT_DIR / "plots" / f"benchmark_summary.{extension}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
