#!/usr/bin/env python3
"""Render the robust-steerability versus OOD-reliability figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


STYLES = {
    "harmful": ("HarmBench", "#5B6573", "o"),
    "truthfulness": ("Truthfulness", "#687DA3", "s"),
    "mgsm": ("MGSM", "#C27D5A", "^"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["benchmark"] in STYLES and row["s_rob"] and row["ood_reliability"]]

    coincident: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        coincident[(
            row["benchmark"], row["s_rob"], row["ood_reliability"],
            row["outcome_metric"],
        )].append(row)
    for group in coincident.values():
        for index, row in enumerate(group):
            offset = (index - (len(group) - 1) / 2.0) * 0.018
            row["_display_s_rob"] = str(float(row["s_rob"]) * (10.0 ** offset))

    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 10.5,
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 8.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    })
    fig, ax = plt.subplots(figsize=(5.45, 4.55))
    for benchmark, (label, color, marker) in STYLES.items():
        selected = [row for row in rows if row["benchmark"] == benchmark]
        if selected:
            ax.scatter(
                [float(row["_display_s_rob"]) for row in selected],
                [float(row["ood_reliability"]) for row in selected],
                s=49 if marker != "*" else 76,
                marker=marker,
                facecolors=color,
                edgecolors="white",
                linewidths=0.6,
                alpha=0.9,
                zorder=3,
                label=label,
            )
    x_all = [float(row["s_rob"]) for row in rows]
    ax.set_xscale("log")
    ax.set_xlim(min(x_all) / 1.18, max(x_all) * 1.18)
    ax.set_ylim(0, 104)
    ax.set_yticks(list(range(0, 101, 20)))
    ax.set_xlabel(r"Robust steerability $S_{\mathrm{rob}}=1/\gamma^\star$")
    ax.set_ylabel("Held-out OOD reliability (%)")
    ax.set_title(r"$S_{\mathrm{rob}}$ and OOD reliability", fontsize=13, fontweight="bold", pad=9)
    ax.grid(True, which="major", color="#D9DDE1", linewidth=0.65, alpha=0.72)
    ax.grid(False, which="minor")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.05)
    ax.tick_params(width=0.9, length=4.5)
    represented = {row["benchmark"] for row in rows}
    handles = [
        Line2D(
            [0], [0], marker=marker, linestyle="none", markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=0.6, markersize=7.2,
            label=label,
        )
        for benchmark, (label, color, marker) in STYLES.items()
        if benchmark in represented
    ]
    labels = [handle.get_label() for handle in handles]
    ax.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=4,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.25,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.165, right=0.985, top=0.985, bottom=0.285)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)



if __name__ == "__main__":
    main()
