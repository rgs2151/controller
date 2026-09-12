"""Render Truth, Info, dynamics mismatch, and residual amplification."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


UNIT = Path(__file__).resolve().parent
PLOTS = UNIT / "plots"
CONDITIONS = (
    "id",
    "spanish",
    "japanese_romaji",
    "long_context_end",
    "long_context_start",
    "corrupting_words",
    "bos_mix",
    "lciteeval_complexity",
)
METHODS = ("alqr", "hinf")
LABELS = {
    "id": "ID",
    "spanish": "Spanish",
    "japanese_romaji": "Japanese (romaji)",
    "long_context_end": "Long context (end)",
    "long_context_start": "Long context (start)",
    "corrupting_words": "Corrupting words",
    "bos_mix": "BOS mix",
    "lciteeval_complexity": "L-CiteEval complexity",
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
        ("truth_percent", "Truthfulness", "Truth score (%)", ".0f", 105.0),
        ("info_percent", "Informativeness", "Info score (%)", ".0f", 105.0),
        (
            "mean_layer_relative_residual_percent",
            "Dynamics mismatch",
            "Mean layer-relative residual (%) ↓",
            ".1f",
            65.0,
        ),
        (
            "residual_to_performance_gain",
            "Residual amplification",
            "Residual-to-performance gain ↓",
            ".3f",
            0.07,
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 9.5), sharey=False)
    base_positions = np.arange(len(CONDITIONS), dtype=float)
    width = 0.34

    for panel_index, (column, title, ylabel, value_format, y_maximum) in enumerate(panels):
        ax = axes.flat[panel_index]
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
                label_offset = 0.025 * y_maximum
                if panel_index == 2 and method_index == 1:
                    label_offset += 0.035 * y_maximum
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(value + label_offset, 0.975 * y_maximum),
                    format(value, value_format),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=COLORS[method],
                )
        ax.set_title(title, fontsize=16)
        ax.set_xticks(base_positions, [LABELS[item] for item in CONDITIONS])
        ax.tick_params(axis="x", labelsize=8.5, pad=4, rotation=28)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        ax.set_xlim(-0.55, len(CONDITIONS) - 0.45)
        ax.set_ylim(0, y_maximum)
        ax.set_yticks(
            np.arange(0, 101, 20) if y_maximum == 105.0 else [0, y_maximum]
        )
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        sns.despine(ax=ax, trim=True, offset=6)

    fig.legend(
        handles=(
            Patch(facecolor=COLORS["alqr"], label="A-LQR"),
            Patch(facecolor=COLORS["hinf"], label=r"$H_\infty$"),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.2, w_pad=1.8)
    for previous in (
        PLOTS / "distribution_shift_truth_info.pdf",
        PLOTS / "distribution_shift_truth_info.png",
    ):
        if previous.exists():
            previous.unlink()
    fig.savefig(
        PLOTS / "distribution_shift_truth_info_residuals.pdf", bbox_inches="tight"
    )
    fig.savefig(
        PLOTS / "distribution_shift_truth_info_residuals.png", bbox_inches="tight"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
