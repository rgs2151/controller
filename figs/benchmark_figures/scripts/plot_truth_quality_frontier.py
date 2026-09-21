#!/usr/bin/env python3
"""Create scalable ID and OOD TruthfulQA frontier figures.

The data source is ``cache/truthfulqa_figure_data.csv``. All models are overlaid
in one panel per split. Foundation-model identity is encoded by color and
controller identity by marker shape, with eight model colors reserved. Adding
the six remaining models requires only appending rows to the CSV. ODESteer is
excluded.
"""

from __future__ import annotations

import csv
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "cache" / "truthfulqa_figure_data.csv"
PDF_DIR = ROOT / "plots" / "pdf"
PNG_DIR = ROOT / "plots" / "figures"

METHOD_ORDER = [
    "H-infinity",
    "A-LQR",
    "S-PID",
    "ITI",
    "ActAdd",
    "Mean-AcT",
    "Linear-AcT",
    "PID-AcT",
    "Original",
]

DISPLAY_NAMES = {
    "H-infinity": r"$H_\infty$ (ours)",
    "A-LQR": "A-LQR",
    "S-PID": "S-PID",
    "ITI": "ITI",
    "ActAdd": "ActAdd",
    "Mean-AcT": "Mean-AcT",
    "Linear-AcT": "Linear-AcT",
    "PID-AcT": "PID-AcT",
    "Original": "Original",
}

# Foundation-model identity is encoded by color; controller identity by shape.
# Eight model colors are reserved so future results do not alter the mapping.
MODEL_COLORS = [
    "#6F4CC3",  # Gemma
    "#0866D9",  # Llama
    "#D97706",
    "#16865C",
    "#C13C72",
    "#8A6D3B",
    "#4F6D7A",
    "#C49A00",
]

METHOD_STYLES = {
    "H-infinity": ("D", 68, 9),
    "A-LQR": ("s", 52, 8),
    "S-PID": ("^", 50, 7),
    "ITI": ("P", 44, 6),
    "ActAdd": ("X", 44, 6),
    "Mean-AcT": ("v", 40, 5),
    "Linear-AcT": ("<", 40, 5),
    "PID-AcT": (">", 40, 5),
    "Original": ("o", 44, 4),
}

INK = "#25282D"
MUTED = "#666C75"
GRID = "#E1E4E8"
SPINE = "#AEB4BC"
QAT_COLOR = "#AEB6C0"
OURS_RING = "#D08A00"


@dataclass(frozen=True)
class Result:
    split: str
    model: str
    method: str
    true: float
    true_se: float
    relevance: float
    fluency: float

    @property
    def quality(self) -> float:
        return min(self.relevance / 2.0, self.fluency / 2.0)

    @property
    def qat(self) -> float:
        return self.true * self.quality


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11.5,
            "axes.labelsize": 15.0,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_data(path: Path) -> list[Result]:
    records: list[Result] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            method = row["method"].strip()
            if method == "ODESteer":
                continue
            records.append(
                Result(
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
    records: list[Result], split: str
) -> OrderedDict[str, list[Result]]:
    grouped: OrderedDict[str, list[Result]] = OrderedDict()
    for record in records:
        if record.split == split:
            grouped.setdefault(record.model, []).append(record)
    for model, rows in grouped.items():
        present = {row.method for row in rows}
        unknown = present - set(METHOD_ORDER)
        if unknown:
            raise ValueError(f"Unknown methods in {split}/{model}: {sorted(unknown)}")
    return grouped


def shared_limits(
    grouped: OrderedDict[str, list[Result]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    rows = [row for model_rows in grouped.values() for row in model_rows]
    x = np.array([row.quality for row in rows])
    y_low = np.array([row.true - row.true_se for row in rows])
    y_high = np.array([row.true + row.true_se for row in rows])
    x_pad = max(0.025, 0.08 * float(np.ptp(x)))
    y_pad = max(3.0, 0.08 * float(y_high.max() - y_low.min()))
    x_limits = (
        math.floor((float(x.min()) - x_pad) / 0.05) * 0.05,
        math.ceil((float(x.max()) + x_pad) / 0.05) * 0.05,
    )
    y_limits = (
        math.floor((float(y_low.min()) - y_pad) / 5.0) * 5.0,
        math.ceil((float(y_high.max()) + y_pad) / 5.0) * 5.0,
    )
    return x_limits, y_limits


def draw_qat_contours(
    ax: plt.Axes,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    quality_grid = np.linspace(max(x_limits[0], 0.01), x_limits[1], 400)
    low = x_limits[0] * y_limits[0]
    high = x_limits[1] * y_limits[1]
    step = 5.0 if high - low < 35 else 10.0
    first = math.ceil(low / step) * step
    for level in np.arange(first, high + 0.1, step):
        if np.isclose(level, 60.0):
            continue
        truth = level / quality_grid
        visible = (truth >= y_limits[0]) & (truth <= y_limits[1])
        if visible.any():
            ax.plot(
                quality_grid[visible],
                truth[visible],
                color=QAT_COLOR,
                linewidth=1.0,
                linestyle=(0, (3.4, 2.6)),
                zorder=0,
            )
            indices = np.flatnonzero(visible)
            label_fraction = 0.36 if 25.0 <= level <= 35.0 else 0.82
            label_index = indices[int(label_fraction * (len(indices) - 1))]
            ax.text(
                quality_grid[label_index],
                truth[label_index],
                f"QAT={level:.0f}",
                color="#7B838D",
                fontsize=9.5,
                rotation=-18,
                ha="left",
                va="bottom",
                bbox={"fc": "white", "ec": "none", "pad": 0.35, "alpha": 0.78},
                zorder=0.2,
            )


def draw_panel(
    ax: plt.Axes,
    grouped: OrderedDict[str, list[Result]],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.55)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.7)

    draw_qat_contours(ax, x_limits, y_limits)
    for model_index, (_, rows) in enumerate(grouped.items()):
        model_color = MODEL_COLORS[model_index % len(MODEL_COLORS)]
        for row in rows:
            marker, size, zorder = METHOD_STYLES[row.method]
            if row.method == "H-infinity":
                ax.scatter(
                    row.quality,
                    row.true,
                    s=size * 1.55,
                    marker=marker,
                    facecolor="none",
                    edgecolor=OURS_RING,
                    linewidth=1.55,
                    zorder=zorder + 0.05,
                )
            ax.scatter(
                row.quality,
                row.true,
                s=size,
                marker=marker,
                color=model_color,
                edgecolor="white",
                linewidth=0.80 if row.method == "H-infinity" else 0.60,
                zorder=zorder + 0.1,
            )

    ax.set_xlabel(r"Quality bottleneck  $\min(\mathrm{IR}/2,\,F/2)$  $\rightarrow$")
    ax.set_ylabel(r"TruthfulQA True (\%)  $\rightarrow$")
    ax.set_xticks(np.arange(x_limits[0], x_limits[1] + 0.001, 0.05))
    ax.set_yticks(np.arange(y_limits[0], y_limits[1] + 0.001, 10.0))
    ax.tick_params(axis="both", labelsize=13.5, length=4.2, width=0.8, pad=5.0)


def draw_vector_model_mark(
    ax: plt.Axes, model: str, x: float, y: float, color: str
) -> None:
    """Draw resolution-independent model marks in axes coordinates."""

    if model.startswith("Gemma"):
        x_radius = 0.047
        y_radius = 0.038
        vertices = [
            (x, y + y_radius),
            (x + x_radius * 0.28, y + y_radius * 0.28),
            (x + x_radius, y),
            (x + x_radius * 0.28, y - y_radius * 0.28),
            (x, y - y_radius),
            (x - x_radius * 0.28, y - y_radius * 0.28),
            (x - x_radius, y),
            (x - x_radius * 0.28, y + y_radius * 0.28),
        ]
        ax.add_patch(
            Polygon(
                vertices,
                closed=True,
                facecolor=color,
                edgecolor="white",
                linewidth=0.55,
                transform=ax.transAxes,
            )
        )
        ax.add_patch(
            Polygon(
                [(x, y + 0.017), (x + 0.022, y), (x, y - 0.017), (x - 0.022, y)],
                closed=True,
                facecolor="white",
                edgecolor="none",
                transform=ax.transAxes,
            )
        )
    elif model.startswith("Llama"):
        t = np.linspace(0.0, 2.0 * np.pi, 240)
        xx = x + 0.052 * np.sin(t)
        yy = y + 0.026 * np.sin(2.0 * t)
        ax.plot(
            xx,
            yy,
            color=color,
            linewidth=2.2,
            solid_capstyle="round",
            transform=ax.transAxes,
            clip_on=False,
        )
    else:
        ax.add_patch(
            Circle(
                (x, y),
                0.028,
                facecolor=color,
                edgecolor="white",
                linewidth=0.6,
                transform=ax.transAxes,
            )
        )
        ax.text(
            x,
            y,
            model[:1],
            ha="center",
            va="center",
            color="white",
            fontsize=5.5,
            fontweight="bold",
            transform=ax.transAxes,
        )


def add_model_legend(
    fig: plt.Figure,
    grouped: OrderedDict[str, list[Result]],
) -> None:
    """Add an upper-right model legend scalable to seven families."""

    legend_ax = fig.add_axes([0.785, 0.49, 0.205, 0.44])
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")

    legend_ax.text(
        0.03, 0.985, "Models", ha="left", va="top", fontsize=14.0,
        fontweight="bold", color=INK, transform=legend_ax.transAxes
    )
    model_step = 0.090
    model_start = 0.860
    for index, model in enumerate(grouped):
        y = model_start - index * model_step
        color = MODEL_COLORS[index % len(MODEL_COLORS)]
        draw_vector_model_mark(legend_ax, model, 0.085, y, color)
        legend_ax.text(
            0.16, y, model, ha="left", va="center", fontsize=12.5,
            color=INK, transform=legend_ax.transAxes
        )



def method_legend_handles() -> list[Line2D]:
    handles: list[Line2D] = []
    for method in METHOD_ORDER:
        marker, _, _ = METHOD_STYLES[method]
        handles.append(
            Line2D(
                [0], [0], marker=marker, linestyle="none",
                markerfacecolor="#555B63",
                markeredgecolor=OURS_RING if method == "H-infinity" else "white",
                markeredgewidth=1.5 if method == "H-infinity" else 0.55,
                markersize=7.2 if method == "H-infinity" else 5.7,
                label=DISPLAY_NAMES[method],
            )
        )
    return handles


def build_figure(
    split: str,
    grouped: OrderedDict[str, list[Result]],
) -> plt.Figure:
    if not grouped:
        raise ValueError(f"No models found for split {split}")
    if len(grouped) > len(MODEL_COLORS):
        raise ValueError("At most eight model colors are currently reserved")
    fig, ax = plt.subplots(figsize=(7.9, 4.8))
    x_limits, y_limits = shared_limits(grouped)
    draw_panel(ax, grouped, x_limits, y_limits)
    add_model_legend(fig, grouped)
    method_legend = fig.legend(
        handles=method_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=len(METHOD_ORDER),
        frameon=False,
        fontsize=11.0,
        handlelength=0.78,
        handletextpad=0.16,
        columnspacing=0.30,
        borderaxespad=0.0,
    )
    method_legend.get_texts()[0].set_fontweight("bold")
    fig.subplots_adjust(
        left=0.090,
        right=0.755,
        top=0.955,
        bottom=0.220,
    )
    return fig


def save_split(split: str, grouped: OrderedDict[str, list[Result]]) -> None:
    fig = build_figure(split, grouped)
    slug = split.lower()
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        PDF_DIR / f"truthfulqa-{slug}-quality-frontier.pdf",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    fig.savefig(
        PNG_DIR / f"truthfulqa-{slug}-quality-frontier.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)


def main() -> None:
    configure_style()
    records = load_data(DATA_PATH)
    for split in ("ID", "OOD"):
        save_split(split, group_split(records, split))


if __name__ == "__main__":
    main()
