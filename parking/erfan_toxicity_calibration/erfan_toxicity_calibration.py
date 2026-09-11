"""Plot archived Erfan results; the superseded pilot lives in ref."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


UNIT_DIR = Path(__file__).resolve().parent
MANIFEST = UNIT_DIR / "manifest.json"
COLORS = {"gamma": "darkgreen", "rank": "midnightblue"}
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


def plot() -> None:
    setup_style()
    manifest = json.loads(MANIFEST.read_text())
    labels = [str(model["label"]) for model in manifest["models"]]
    gamma = []
    median_rank = []
    for label in labels:
        slug = "".join(character.lower() if character.isalnum() else "_" for character in label).strip("_")
        payload = torch.load(UNIT_DIR / "cache" / "controllers" / f"{slug}.pt", map_location="cpu", weights_only=False)
        metadata = payload["metadata"]
        gamma.append(float(metadata["gamma_star"]))
        median_rank.append(float(np.median(metadata["disturbance_ranks"])))
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].bar(x, gamma, color=COLORS["gamma"])
    axes[0].set_ylabel(r"Minimum feasible $\gamma^\star$")
    axes[1].bar(x, median_rank, color=COLORS["rank"])
    axes[1].set_ylabel("Median disturbance rank")
    for ax, values in zip(axes, (gamma, median_rank), strict=True):
        upper = max(values) * 1.1
        ax.set_ylim(0, upper)
        ax.set_yticks([0, upper])
        ax.set_xticks(x)
        ax.set_xticklabels(
            [SHORT_MODEL_LABELS[label] for label in labels],
            rotation=30,
            ha="right",
            fontsize=6,
        )
        ax.set_box_aspect(1)
        ax.set_xlabel("Qwen scale (B); D = DistilGPT-2", fontsize=6)
        sns.despine(ax=ax, trim=True, offset=10)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.22, top=0.95, wspace=0.48)
    for extension in ("pdf", "png"):
        fig.savefig(UNIT_DIR / "plots" / f"controller_calibration.{extension}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["plot"])
    args = parser.parse_args()
    plot()


if __name__ == "__main__":
    main()
