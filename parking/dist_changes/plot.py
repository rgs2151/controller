"""Render separate Truth and Info bootstrap box plots from cached judge outputs."""

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
BOOTSTRAP_SAMPLES = 10_000
SEED = 2151


def _bootstrap_percent(values: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    return 100.0 * values[indices].mean(axis=1)


def main() -> None:
    frame = pd.read_csv(PLOTS / "generations.csv")
    expected_rows = len(CONDITIONS) * len(METHODS) * 50
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} displayed rows; found {len(frame)}")

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
        ("truth_score", "Truthfulness", "Truth score (%)"),
        ("info_score", "Informativeness", "Info score (%)"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=False)
    base_positions = np.arange(len(CONDITIONS), dtype=float)
    offsets = {"alqr": -0.19, "hinf": 0.19}

    for panel_index, (column, title, ylabel) in enumerate(panels):
        ax = axes[panel_index]
        for condition_index, condition in enumerate(CONDITIONS):
            for method_index, method in enumerate(METHODS):
                group = frame[
                    (frame["display_condition"] == condition)
                    & (frame["method"] == method)
                ]
                if len(group) != 50:
                    raise ValueError(f"Expected 50 rows for {method}/{condition}")
                values = group[column].to_numpy(dtype=float)
                samples = _bootstrap_percent(
                    values,
                    SEED + 1000 * panel_index + 100 * condition_index + method_index,
                )
                position = base_positions[condition_index] + offsets[method]
                ax.boxplot(
                    [samples],
                    positions=[position],
                    widths=0.32,
                    whis=(2.5, 97.5),
                    showfliers=False,
                    patch_artist=True,
                    boxprops={
                        "facecolor": COLORS[method],
                        "edgecolor": COLORS[method],
                        "linewidth": 1.1,
                    },
                    medianprops={"color": "white", "linewidth": 1.5},
                    whiskerprops={"color": COLORS[method], "linewidth": 1.1},
                    capprops={"color": COLORS[method], "linewidth": 1.1},
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
