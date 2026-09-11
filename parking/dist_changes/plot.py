"""Render separate Truth and Info bars from cached aggregate scores."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


UNIT = Path(__file__).resolve().parent
PLOTS = UNIT / "plots"
CONDITIONS = ("id", "spanish", "long_context", "adversarial")
METHODS = ("alqr", "hinf")
LABELS = {
    "id": "ID",
    "spanish": "Spanish",
    "long_context": "Long\ncontext",
    "adversarial": "Adversarial",
}
COLORS = {"alqr": "black", "hinf": "#d62728"}


def main() -> None:
    frame = pd.read_csv(PLOTS / "distribution_scores.csv")
    expected_rows = len(CONDITIONS) * len(METHODS)
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} aggregate rows; found {len(frame)}")

    sns.set_theme(context="talk", style="ticks", palette="dark")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )

    panels = (
        ("truth_percent", "Truthfulness", "Truth score (%)"),
        ("info_percent", "Informativeness", "Info score (%)"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=False)
    base_positions = np.arange(len(CONDITIONS), dtype=float)
    width = 0.34

    for panel_index, (column, title, ylabel) in enumerate(panels):
        ax = axes[panel_index]
        for method_index, method in enumerate(METHODS):
            group = frame[frame["method"] == method].set_index("display_condition")
            values = np.asarray(
                [group.loc[condition, column] for condition in CONDITIONS], dtype=float
            )
            positions = base_positions + (method_index - 0.5) * width
            bars = ax.bar(
                positions,
                values,
                width=width,
                color=COLORS[method],
            )
            for bar, value in zip(bars, values, strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(value + 2.0, 102.0),
                    f"{value:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=COLORS[method],
                )
        ax.set_title(title, fontsize=16)
        ax.set_xticks(base_positions, [LABELS[item] for item in CONDITIONS])
        ax.tick_params(axis="x", labelsize=10)
        ax.set_xlim(-0.55, len(CONDITIONS) - 0.45)
        ax.set_ylim(0, 105)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        sns.despine(ax=ax, trim=True)

    fig.legend(
        handles=(
            Patch(facecolor=COLORS["alqr"], label="A-LQR"),
            Patch(facecolor=COLORS["hinf"], label=r"$H_\infty$"),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(PLOTS / "distribution_shift_truth_info.pdf", bbox_inches="tight")
    fig.savefig(PLOTS / "distribution_shift_truth_info.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
