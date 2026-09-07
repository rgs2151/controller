"""Run and plot Erfan's three-condition multimodel OOD steering benchmark."""

from __future__ import annotations

import argparse
import subprocess
import sys
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
METHOD_LEGEND_LABELS = ["Original", "A-LQR", "S-PID", r"$H_\infty$"]
METHOD_TICK_LABELS = ["Orig.", "A-LQR", "S-PID", r"$H_\infty$"]
SUBSETS = ["jigsaw_long", "toxicchat_long", "mmlu_ood_other_concepts"]
SUBSET_LABELS = {
    "jigsaw_long": "Jigsaw long",
    "toxicchat_long": "ToxicChat long",
    "mmlu_ood_other_concepts": "MMLU concept shift",
}
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


def run(devices: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "robust_steerability.experiments", "run", "--manifest", str(MANIFEST), "--devices", devices],
        cwd=UNIT_DIR.parents[1],
        check=True,
    )


def _model_by_condition(table: pd.DataFrame) -> None:
    models = list(dict.fromkeys(table["model"].tolist()))
    fig, axes = plt.subplots(1, len(SUBSETS), figsize=(11.4, 3.5), squeeze=False)
    x = np.arange(len(models))
    width = 0.18
    for panel, subset in enumerate(SUBSETS):
        ax = axes[0, panel]
        rows = table[table["subset"] == subset]
        for method_index, method in enumerate(METHODS):
            values = [
                float(rows[(rows["model"] == model) & (rows["method_label"] == method)]["toxicity_percent"].iloc[0])
                for model in models
            ]
            positions = x + (method_index - 1.5) * width
            ax.bar(
                positions,
                values,
                width=width,
                color=COLORS[method],
                label=METHOD_LEGEND_LABELS[method_index] if panel == 0 else None,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [SHORT_MODEL_LABELS[model] for model in models],
            rotation=30,
            ha="right",
            fontsize=6,
        )
        upper = max(1.0, float(rows["toxicity_percent"].max()) * 1.1)
        ax.set_ylim(0, upper)
        ax.set_yticks([0, upper])
        ax.set_title(SUBSET_LABELS[subset], fontsize=11)
        if panel == 0:
            ax.set_ylabel(r"Continuation toxicity (\%)")
            ax.legend(loc="upper left", fontsize=8)
        ax.set_box_aspect(1)
        ax.set_xlabel("Qwen scale (B); D = DistilGPT-2", fontsize=6)
        sns.despine(ax=ax, trim=True, offset=10)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.20, top=0.92, wspace=0.45)
    for extension in ("pdf", "png"):
        fig.savefig(UNIT_DIR / "plots" / f"ood_toxicity_across_models.{extension}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _model_grid(table: pd.DataFrame, value_column: str, stem: str, reduction: bool) -> None:
    models = list(dict.fromkeys(table["model"].tolist()))
    fig, axes = plt.subplots(len(models), len(SUBSETS), figsize=(3.2 * len(SUBSETS), 2.4 * len(models)), squeeze=False)
    for row_index, model in enumerate(models):
        for column_index, subset in enumerate(SUBSETS):
            ax = axes[row_index, column_index]
            rows = table[(table["model"] == model) & (table["subset"] == subset)].set_index("method_label")
            values = [float(rows.loc[method, value_column]) for method in METHODS]
            if reduction:
                values = [-value for value in values]
            ax.bar(np.arange(len(METHODS)), values, color=[COLORS[method] for method in METHODS])
            lower = min(0.0, min(values) * 1.1)
            upper = max(1.0, max(values) * 1.1)
            ax.set_ylim(lower, upper)
            ax.set_yticks([lower, upper])
            ax.set_xticks(np.arange(len(METHODS)))
            ax.set_xticklabels(METHOD_TICK_LABELS if row_index == len(models) - 1 else [])
            if row_index == len(models) - 1:
                ax.tick_params(axis="x", labelsize=7)
            if row_index == 0:
                ax.set_title(SUBSET_LABELS[subset], fontsize=10)
            if column_index == 0:
                ax.set_ylabel(model, fontsize=10)
            ax.set_box_aspect(1)
            sns.despine(ax=ax, trim=True, offset=10)
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.08, top=0.96, wspace=0.55, hspace=0.55)
    for extension in ("pdf", "png"):
        fig.savefig(UNIT_DIR / "plots" / f"{stem}.{extension}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot() -> None:
    setup_style()
    table = pd.read_csv(RESULTS)
    _model_by_condition(table)
    _model_grid(table, "toxicity_percent", "ood_toxicity_model_metrics", False)
    _model_grid(table, "percent_change", "ood_reduction_model_metrics", True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["run", "plot", "all"])
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    args = parser.parse_args()
    if args.stage in {"run", "all"}:
        run(args.devices)
    if args.stage in {"plot", "all"}:
        plot()


if __name__ == "__main__":
    main()
