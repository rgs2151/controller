#!/usr/bin/env python3
"""Recreate the single retained 4x4 brain/GPT-2 diagnostic figure."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REFERENCE_RESULTS = HERE / "ref" / "original_notebook_rerun" / "results"
PLOTS = HERE / "plots"
ID_COLOR = "#8A8F94"
OOD_COLOR = "#8B1E1E"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
    "pdf.fonttype": 42,
})


def clean_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#D9DDE1", linewidth=0.65, alpha=0.75)
    ax.set_axisbelow(True)


def grouped_bars(ax, table, horizon, value, title, ylabel, absolute=False):
    pivot = table.pivot(index=horizon, columns="split", values=value).sort_index()
    if absolute:
        pivot = pivot.abs()
    x = np.arange(len(pivot))
    width = 0.36
    ax.bar(x - width / 2, pivot["ID"], width, color=ID_COLOR)
    ax.bar(x + width / 2, pivot["OOD"], width, color=OOD_COLOR)
    ax.set_xticks(x, [str(v) for v in pivot.index])
    ax.set_xlabel("Horizon (ms)" if horizon == "horizon_ms" else "Horizon (tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, weight="bold", loc="left")
    clean_axis(ax)


def behavior_scatter(ax, table, metric, title, xlabel, ylabel):
    for split, color in (("ID", ID_COLOR), ("OOD", OOD_COLOR)):
        subset = table[table["split"] == split]
        ax.scatter(subset["behavior"], subset[metric], s=28, color=color,
                   alpha=0.52, edgecolor="none")
        if len(subset) >= 3 and np.ptp(subset["behavior"]) > 0:
            fit = linregress(subset["behavior"], subset[metric])
            x = np.linspace(subset["behavior"].min(), subset["behavior"].max(), 80)
            ax.plot(x, fit.intercept + fit.slope * x, color=color, linewidth=2)
    ax.set_title(title, weight="bold", loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    clean_axis(ax)


def aggregate_tables():
    brain = pd.read_csv(REFERENCE_RESULTS / "final_panelD_error_shrinkage_LVF_theta.csv")
    model = pd.read_csv(REFERENCE_RESULTS / "final_llmfit_panelD_error_shrinkage.csv")
    conflict = brain[brain["axis"] == "conflict"].copy()
    stimulation = brain[brain["axis"] == "stim-context"].copy()
    spanish = model[model["split"].isin(["ID", "OOD-es"])].copy()
    spanish["split"] = spanish["split"].replace({"OOD-es": "OOD"})
    long_context = model[model["split"].isin(["ID", "OOD-ctx"])].copy()
    long_context["split"] = long_context["split"].replace({"OOD-ctx": "OOD"})
    return conflict, stimulation, spanish, long_context


def main():
    behavior = (
        pd.read_csv(RESULTS / "brain_conflict_behavior.csv"),
        pd.read_csv(RESULTS / "brain_stim_context_behavior.csv"),
        pd.read_csv(RESULTS / "gpt2_spanish_paired_behavior.csv"),
        pd.read_csv(RESULTS / "gpt2_long_context_paired_behavior.csv"),
    )
    for table in behavior:
        table["abs_correction_benefit"] = table["correction_benefit"].abs()
    aggregate = aggregate_tables()
    fig, axes = plt.subplots(
        4, 4, figsize=(14.5, 12.8),
        gridspec_kw={"height_ratios": [0.88, 1.12, 0.88, 1.12]},
    )

    brain_titles = ("Conflict shift", "Stimulation-context shift")
    model_titles = ("Spanish shift", "Long-context shift")
    for column, (table, title) in enumerate(zip(aggregate[:2], brain_titles)):
        grouped_bars(axes[0, column], table, "horizon_ms", "reduction", title,
                     r"$|g(x)|$ RMS reduction", absolute=True)
        grouped_bars(axes[0, column + 2], table, "horizon_ms", "med_lin", title,
                     "Linear residual RMS (z)")
    for column, (table, title) in enumerate(zip(aggregate[2:], model_titles)):
        grouped_bars(axes[2, column], table, "horizon_tok", "reduction", title,
                     r"$|g(x)|$ RMS reduction", absolute=True)
        grouped_bars(axes[2, column + 2], table, "horizon_tok", "med_lin", title,
                     "Linear residual RMS (z)")

    for column, (table, title) in enumerate(zip(behavior[:2], brain_titles)):
        behavior_scatter(axes[1, column], table, "abs_correction_benefit", title,
                         "Response time (ms)", r"$|g(x)|$ correction magnitude")
        behavior_scatter(axes[1, column + 2], table, "linear_residual_rms", title,
                         "Response time (ms)", "Linear residual RMS (z)")
    for column, (table, title) in enumerate(zip(behavior[2:], model_titles)):
        behavior_scatter(axes[3, column], table, "abs_correction_benefit", title,
                         "Correct-label NLL", r"$|g(x)|$ correction magnitude")
        behavior_scatter(axes[3, column + 2], table, "linear_residual_rms", title,
                         "Correct-label NLL", "Linear residual RMS (z)")

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none", color=ID_COLOR,
                   label="ID", markersize=8),
        plt.Line2D([0], [0], marker="o", linestyle="none", color=OOD_COLOR,
                   label="OOD", markersize=8),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.055, top=0.94,
                        wspace=0.5, hspace=0.68)
    PLOTS.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(PLOTS / f"residual_metric_behavior_grid_4x4.{extension}",
                    dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
