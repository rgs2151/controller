#!/usr/bin/env python3
"""Render model-specific L-CiteEval context-length transfer figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import MaxNLocator
from PIL import Image


UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[1]
SOURCE = REPO / "figs/bench_table/lciteeval/lciteeval_full.md"
LOGOS = REPO / "figs/logos"
PLOTS = UNIT / "plots"

TEAL = "#398197"
INK = "#25282D"
MUTED = "#687078"
GRID = "#E2E5E8"
SPINE = "#AEB4BA"
ID_SHADE = "#F3F5F6"
OOD_SHADE = "#FBF4EC"

CONTEXTS = ("8K", "16K")
MODEL_SPECS = {
    "Qwen2.5-3B-Instruct": {
        "label": "Qwen2.5-3B-Instruct",
        "slug": "qwen2_5_3b",
        "logo": LOGOS / "qwen_transparent.png",
        "logo_zoom": 0.032,
        "methods": ("Original", "S-PID", "A-LQR", "H-infinity"),
    },
    "Llama-3.2-1B-Instruct": {
        "label": "Llama-3.2-1B-Instruct",
        "slug": "llama3_2_1b",
        "logo": LOGOS / "llama_transparent.png",
        "logo_zoom": 0.032,
        "methods": ("Original", "A-LQR", "H-infinity"),
    },
}

METHOD_LABELS = {
    "Original": "Original",
    "S-PID": "S-PID",
    "A-LQR": "A-LQR",
    "H-infinity": r"$H_\infty$ (ours)",
}
METHOD_COLORS = {
    "Original": "#59636D",
    "S-PID": "#B4775D",
    "A-LQR": "#687DA3",
    "H-infinity": TEAL,
}
METHOD_MARKERS = {
    "Original": "o",
    "S-PID": "^",
    "A-LQR": "s",
    "H-infinity": "D",
}
METHOD_STYLES = {
    "Original": (0, (1.0, 1.5)),
    "S-PID": (0, (5.0, 2.0, 1.0, 2.0)),
    "A-LQR": (0, (4.0, 2.0)),
    "H-infinity": "-",
}


@dataclass(frozen=True)
class Result:
    model: str
    context: str
    method: str
    answer_recall: float
    citation_f1: float


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 10.5,
            "axes.titlesize": 13.5,
            "axes.titleweight": "bold",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def mean(value: str) -> float:
    return float(value.split("±", maxsplit=1)[0].strip())


def read_results() -> list[Result]:
    results: list[Result] = []
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "±" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 8:
            continue
        model, context, method, answer_recall, citation_f1, *_ = cells
        if model not in MODEL_SPECS or context not in CONTEXTS:
            continue
        canonical_method = "H-infinity" if method.startswith("H∞") else method
        if canonical_method not in MODEL_SPECS[model]["methods"]:
            continue
        results.append(
            Result(
                model=model,
                context=context,
                method=canonical_method,
                answer_recall=mean(answer_recall),
                citation_f1=mean(citation_f1),
            )
        )

    expected = sum(len(spec["methods"]) * len(CONTEXTS) for spec in MODEL_SPECS.values())
    if len(results) != expected:
        raise ValueError(f"Expected {expected} final-table rows, found {len(results)}")
    return results


def tight_limits(values: list[float]) -> tuple[float, float]:
    """Return compact limits with enough padding to keep markers unclipped."""

    low = min(values)
    high = max(values)
    span = max(high - low, 1.0)
    padding = 0.10 * span
    return low - padding, high + padding


def style_axis(ax: plt.Axes, *, ylim: tuple[float, float]) -> None:
    ax.axvspan(-0.5, 0.5, color=ID_SHADE, zorder=0)
    ax.axvspan(0.5, 1.5, color=OOD_SHADE, zorder=0)
    ax.grid(True, axis="y", color=GRID, linewidth=0.75, zorder=1)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.20, 1.20)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.set_xticks([0, 1], ["8K\nmatched", "16K\nlong-context"])
    ax.tick_params(length=3.5, width=0.8, labelsize=10.0)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)


def add_model_heading(fig: plt.Figure, spec: dict[str, object]) -> None:
    heading = fig.add_axes([0.32, 0.865, 0.36, 0.10])
    heading.axis("off")
    logo = Image.open(Path(spec["logo"])).convert("RGBA")
    heading.add_artist(
        AnnotationBbox(
            OffsetImage(logo, zoom=float(spec["logo_zoom"]), interpolation="lanczos"),
            (0.08, 0.50),
            xycoords="axes fraction",
            frameon=False,
            pad=0,
        )
    )
    heading.text(
        0.18,
        0.50,
        str(spec["label"]),
        ha="left",
        va="center",
        fontsize=13.0,
        fontweight="bold",
        color=INK,
    )


def render_model(model: str, results: list[Result]) -> None:
    spec = MODEL_SPECS[model]
    methods = tuple(spec["methods"])
    lookup = {(row.context, row.method): row for row in results if row.model == model}

    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.55), gridspec_kw={"wspace": 0.25})
    metrics = (
        ("answer_recall", "Answer recall (%) ↑"),
        ("citation_f1", "Citation F1 (%) ↑"),
    )

    for ax, (field, title) in zip(axes, metrics, strict=True):
        all_values = [
            getattr(lookup[(context, method)], field)
            for method in methods
            for context in CONTEXTS
        ]
        style_axis(ax, ylim=tight_limits(all_values))
        ax.set_title(title, pad=10)
        for method in methods:
            values = [getattr(lookup[(context, method)], field) for context in CONTEXTS]
            ours = method == "H-infinity"
            ax.plot(
                [0, 1],
                values,
                color=METHOD_COLORS[method],
                linestyle=METHOD_STYLES[method],
                linewidth=3.0 if ours else 2.0,
                marker=METHOD_MARKERS[method],
                markersize=8.0 if ours else 6.5,
                markerfacecolor=METHOD_COLORS[method],
                markeredgecolor="#D08A00" if ours else "white",
                markeredgewidth=1.35 if ours else 0.8,
                solid_capstyle="round",
                zorder=5 if ours else 3,
            )

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linestyle=METHOD_STYLES[method],
            linewidth=3.0 if method == "H-infinity" else 2.0,
            marker=METHOD_MARKERS[method],
            markersize=7.0,
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="#D08A00" if method == "H-infinity" else "white",
            markeredgewidth=1.1 if method == "H-infinity" else 0.7,
            label=METHOD_LABELS[method],
        )
        for method in methods
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=len(methods),
        frameon=False,
        handlelength=2.3,
        columnspacing=1.35,
        handletextpad=0.55,
        fontsize=10.3,
    )
    add_model_heading(fig, spec)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.25, top=0.79)

    PLOTS.mkdir(parents=True, exist_ok=True)
    stem = f"figure_long_context_{spec['slug']}"
    fig.savefig(PLOTS / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(PLOTS / f"{stem}.png", dpi=450, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def render() -> None:
    setup_style()
    results = read_results()
    for model in MODEL_SPECS:
        render_model(model, results)


if __name__ == "__main__":
    render()
