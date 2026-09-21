#!/usr/bin/env python3
"""Create scalable TruthfulQA gain matrices for the paper.

The figure reports two complementary changes relative to each model's
unsteered output:

    Delta True = True(method) - True(Original)
    Delta QAT  = QAT(method)  - QAT(Original)

where QAT = True(%) * min(relevance / 2, fluency / 2).  Rows are models and
columns are steering methods.  Add future models by appending rows to the CSV;
the layout grows automatically.  ODESteer is excluded because its current
implementation is invalid.
"""

from __future__ import annotations

import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "cache" / "truthfulqa_figure_data.csv"
PDF_DIR = ROOT / "plots" / "pdf"
PNG_DIR = ROOT / "plots" / "figures"

METHOD_ORDER = [
    "ITI",
    "ActAdd",
    "Mean-AcT",
    "Linear-AcT",
    "PID-AcT",
    "S-PID",
    "A-LQR",
    "H-infinity",
]

DISPLAY_NAMES = {
    "ITI": "ITI",
    "ActAdd": "ActAdd",
    "Mean-AcT": "Mean-AcT",
    "Linear-AcT": "Linear-AcT",
    "PID-AcT": "PID-AcT",
    "S-PID": "S-PID",
    "A-LQR": "A-LQR",
    "H-infinity": r"$H_\infty$",
}

RED = "#A61B1B"
BLUE = "#2B5C9A"
INK = "#22252A"
MUTED = "#62676F"
GRID = "#D8DADD"
GAIN_CMAP = LinearSegmentedColormap.from_list(
    "paper_gain", [BLUE, "#F8F8F7", RED], N=256
)


@dataclass(frozen=True)
class Record:
    split: str
    model: str
    method: str
    true: float
    true_se: float
    relevance: float
    fluency: float

    @property
    def qat(self) -> float:
        return self.true * min(self.relevance / 2.0, self.fluency / 2.0)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.titlesize": 10.2,
            "axes.titleweight": "semibold",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_data(path: Path) -> list[Record]:
    records: list[Record] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            method = row["method"].strip()
            if method == "ODESteer":
                continue
            records.append(
                Record(
                    split=row["split"].strip().upper(),
                    model=row["model"].strip(),
                    method=method,
                    true=float(row["true"]),
                    true_se=float(row["true_se"]),
                    relevance=float(row["relevance"]),
                    fluency=float(row["fluency"]),
                )
            )
    return records


def group_split(
    records: list[Record], split: str
) -> OrderedDict[str, dict[str, Record]]:
    grouped: OrderedDict[str, dict[str, Record]] = OrderedDict()
    for record in records:
        if record.split == split:
            grouped.setdefault(record.model, {})[record.method] = record
    for model, rows in grouped.items():
        if "Original" not in rows:
            raise ValueError(f"{split}/{model} is missing Original")
    return grouped


def gain_matrices(
    grouped: OrderedDict[str, dict[str, Record]],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    models = list(grouped)
    true_gain = np.full((len(models), len(METHOD_ORDER)), np.nan)
    qat_gain = np.full_like(true_gain, np.nan)
    for i, model in enumerate(models):
        rows = grouped[model]
        original = rows["Original"]
        for j, method in enumerate(METHOD_ORDER):
            if method not in rows:
                continue
            true_gain[i, j] = rows[method].true - original.true
            qat_gain[i, j] = rows[method].qat - original.qat
    return models, true_gain, qat_gain


def symmetric_limit(matrix: np.ndarray) -> float:
    finite = np.abs(matrix[np.isfinite(matrix)])
    return max(float(finite.max(initial=1.0)), 1.0)


def text_color(value: float, limit: float) -> str:
    return "white" if abs(value) / limit > 0.56 else INK


def draw_matrix(
    ax: plt.Axes,
    matrix: np.ndarray,
    models: list[str],
    title: str,
    colorbar_label: str,
    show_ylabels: bool,
) -> None:
    limit = symmetric_limit(matrix)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, cmap=GAIN_CMAP, norm=norm, aspect="auto")

    ax.set_title(title, loc="left", pad=9)
    ax.set_xticks(np.arange(len(METHOD_ORDER)))
    ax.set_xticklabels(
        [DISPLAY_NAMES[method] for method in METHOD_ORDER],
        rotation=39,
        ha="left",
        rotation_mode="anchor",
    )
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=4, labelsize=7.3)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models if show_ylabels else [""] * len(models))
    ax.tick_params(axis="y", length=0, pad=7, labelsize=8.2)

    ax.set_xticks(np.arange(-0.5, len(METHOD_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(matrix.shape[0]):
        finite_row = np.flatnonzero(np.isfinite(matrix[i]))
        winner = finite_row[np.argmax(matrix[i, finite_row])] if finite_row.size else -1
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                ax.text(j, i, "--", ha="center", va="center", color="#9A9DA2")
                continue
            ax.text(
                j,
                i,
                f"{value:+.1f}",
                ha="center",
                va="center",
                color=text_color(value, limit),
                fontsize=7.25,
                fontweight="bold" if j == winner else "normal",
            )
            if j == winner:
                ax.scatter(
                    j + 0.34,
                    i - 0.30,
                    marker="*",
                    s=20,
                    color="#E5B84B",
                    edgecolor=INK,
                    linewidth=0.25,
                    zorder=5,
                    clip_on=False,
                )

    hinf_col = METHOD_ORDER.index("H-infinity")
    ax.add_patch(
        Rectangle(
            (hinf_col - 0.5, -0.5),
            1,
            len(models),
            fill=False,
            edgecolor=RED,
            linewidth=1.8,
            clip_on=False,
        )
    )
    ax.get_xticklabels()[hinf_col].set_color(RED)
    ax.get_xticklabels()[hinf_col].set_fontweight("bold")

    cbar = ax.figure.colorbar(
        image,
        ax=ax,
        orientation="horizontal",
        fraction=0.075,
        pad=0.12,
        aspect=34,
    )
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=6.6, length=2, pad=1)
    cbar.set_label(colorbar_label, fontsize=7.2, labelpad=2)


def draw_margin_panel(
    ax: plt.Axes,
    models: list[str],
    qat_gain: np.ndarray,
) -> None:
    hinf_col = METHOD_ORDER.index("H-infinity")
    margins: list[float] = []
    baselines: list[str] = []
    for row in qat_gain:
        hinf = row[hinf_col]
        candidates = [
            (METHOD_ORDER[j], row[j])
            for j in range(len(METHOD_ORDER))
            if j != hinf_col and np.isfinite(row[j])
        ]
        if not np.isfinite(hinf) or not candidates:
            margins.append(np.nan)
            baselines.append("pending")
            continue
        best_method, best_value = max(candidates, key=lambda pair: pair[1])
        margins.append(float(hinf - best_value))
        baselines.append(best_method)

    finite = np.array([value for value in margins if np.isfinite(value)])
    xmax = max(float(finite.max(initial=1.0)) * 1.58, 1.0)
    xmin = min(float(finite.min(initial=0.0)) * 1.25, 0.0)
    ax.set_xlim(xmin - 0.04 * xmax, xmax)
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.axvline(0, color="#AEB2B7", linewidth=0.8, zorder=0)

    for i, (margin, baseline) in enumerate(zip(margins, baselines, strict=True)):
        if not np.isfinite(margin):
            ax.text(0, i, "pending", va="center", color=MUTED, fontsize=7.0)
            continue
        color = RED if margin >= 0 else BLUE
        ax.plot([0, margin], [i, i], color=color, linewidth=3.0, solid_capstyle="round")
        ax.scatter(
            margin,
            i,
            marker="D",
            s=27,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        offset = 0.045 * (xmax - xmin)
        ax.text(
            margin + (offset if margin >= 0 else -offset),
            i - 0.08,
            f"{margin:+.2f}",
            ha="left" if margin >= 0 else "right",
            va="center",
            color=color,
            fontsize=8.0,
            fontweight="bold",
        )
        ax.text(
            margin + (offset if margin >= 0 else -offset),
            i + 0.19,
            f"vs. {DISPLAY_NAMES[baseline]}",
            ha="left" if margin >= 0 else "right",
            va="center",
            color=MUTED,
            fontsize=6.6,
        )

    ax.set_title(r"(c) $H_\infty$ advantage", loc="left", pad=9)
    ax.set_xlabel("QAT points over best baseline", fontsize=7.2, labelpad=5)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=6.6, length=2.5)
    ax.grid(axis="x", color="#ECEDEF", linewidth=0.7, zorder=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def create_figure(
    split: str,
    grouped: OrderedDict[str, dict[str, Record]],
    output_pdf: Path,
    output_png: Path,
) -> None:
    models, true_gain, qat_gain = gain_matrices(grouped)
    if not models:
        raise ValueError(f"No models found for {split}")

    # Reserve a stable header band above the method labels.  The height grows
    # with the number of models, so the same design remains legible when the
    # remaining model rows are appended to the CSV.
    height = max(3.85, 2.65 + 0.44 * len(models))
    fig = plt.figure(figsize=(7.35, height))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 0.58],
        left=0.105,
        right=0.985,
        top=0.65,
        bottom=0.22,
        wspace=0.19,
    )
    ax_true = fig.add_subplot(grid[0, 0])
    ax_qat = fig.add_subplot(grid[0, 1])
    ax_margin = fig.add_subplot(grid[0, 2])

    draw_matrix(
        ax_true,
        true_gain,
        models,
        "(a) Truthfulness gain",
        r"$\Delta$ True (percentage points)",
        show_ylabels=True,
    )
    draw_matrix(
        ax_qat,
        qat_gain,
        models,
        "(b) Quality-adjusted gain",
        r"$\Delta$ QAT (points)",
        show_ylabels=False,
    )
    draw_margin_panel(ax_margin, models, qat_gain)

    split_name = "In-distribution TruthfulQA" if split == "ID" else "Spanish-to-English OOD transfer"
    fig.suptitle(
        split_name,
        x=0.105,
        y=0.972,
        ha="left",
        fontsize=13.0,
        fontweight="bold",
    )
    fig.text(
        0.105,
        0.895,
        "Changes are relative to the same model without steering;  "
        r"QAT $=$ True $\times \min(\mathrm{relevance}/2,\,\mathrm{fluency}/2)$.",
        ha="left",
        va="top",
        color=MUTED,
        fontsize=7.8,
    )
    fig.text(
        0.105,
        0.045,
        r"$\star$ marks the best method in each row. The red outline identifies $H_\infty$. "
        "Values use reported means; ODESteer is excluded because its implementation is invalid.",
        ha="left",
        va="bottom",
        color=MUTED,
        fontsize=6.8,
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output_png, dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    configure_style()
    records = load_data(DATA_PATH)
    for split in ("ID", "OOD"):
        slug = split.lower()
        create_figure(
            split,
            group_split(records, split),
            PDF_DIR / f"truthfulqa-{slug}-gain-matrix.pdf",
            PNG_DIR / f"truthfulqa-{slug}-gain-matrix.png",
        )


if __name__ == "__main__":
    main()
