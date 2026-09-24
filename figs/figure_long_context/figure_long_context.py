#!/usr/bin/env python3
"""Render model-specific L-CiteEval context-length transfer figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import MaxNLocator
from PIL import Image


UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[1]
SOURCE = REPO / "figs/bench_table/lciteeval/lciteeval_full.md"
LOGOS = REPO / "figs/logos"
PLOTS = UNIT / "plots"

TEAL = "#007C7C"
INK = "#25282D"
MUTED = "#687078"
GRID = "#E2E5E8"
SPINE = "#AEB4BA"

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
LOGO_METHOD_COLORS = {
    "Original": "#697680",
    "S-PID": "#BF7558",
    "A-LQR": "#756FA6",
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


def tinted_logo(path: Path, color: str) -> np.ndarray:
    """Tint a transparent company logo while retaining antialiased edges."""

    image = Image.open(path).convert("RGBA")
    array = np.asarray(image, dtype=np.float32) / 255.0
    alpha = array[..., 3]
    luminance = 0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
    mask = alpha * np.clip(1.25 - luminance, 0.30, 1.0)
    output = np.empty_like(array)
    output[..., :3] = np.asarray(mpl.colors.to_rgb(color), dtype=np.float32)
    output[..., 3] = mask
    return output


def add_logo_marker(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    model: str,
    method: str,
) -> None:
    spec = MODEL_SPECS[model]
    zoom = 0.024 if model.startswith("Qwen") else 0.026
    ax.add_artist(
        AnnotationBbox(
            OffsetImage(
                tinted_logo(Path(spec["logo"]), LOGO_METHOD_COLORS[method]),
                zoom=zoom,
                interpolation="lanczos",
            ),
            (x, y),
            frameon=False,
            pad=0,
            zorder=6 if method == "H-infinity" else 5,
        )
    )


def add_logo_legend(fig: plt.Figure, model: str, methods: tuple[str, ...]) -> None:
    """Draw a compact legend with the same tinted logos used in the panels."""

    legend_ax = fig.add_axes([0.16, 0.005, 0.68, 0.09])
    legend_ax.axis("off")
    xs = np.linspace(0.08, 0.92, len(methods))
    spec = MODEL_SPECS[model]
    for x, method in zip(xs, methods, strict=True):
        zoom = 0.020 if model.startswith("Qwen") else 0.022
        legend_ax.add_artist(
            AnnotationBbox(
                OffsetImage(
                    tinted_logo(Path(spec["logo"]), LOGO_METHOD_COLORS[method]),
                    zoom=zoom,
                    interpolation="lanczos",
                ),
                (x - 0.045, 0.50),
                xycoords="axes fraction",
                frameon=False,
                pad=0,
            )
        )
        legend_ax.text(
            x,
            0.50,
            METHOD_LABELS[method],
            transform=legend_ax.transAxes,
            ha="left",
            va="center",
            fontsize=10.3,
            color=INK,
        )


def render_model(model: str, results: list[Result]) -> None:
    spec = MODEL_SPECS[model]
    methods = tuple(spec["methods"])
    lookup = {(row.context, row.method): row for row in results if row.model == model}

    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.55), gridspec_kw={"wspace": 0.25})
    metrics = (
        ("answer_recall", "Answer recall (%)"),
        ("citation_f1", "Citation F1 (%)"),
    )

    for ax, (field, title) in zip(axes, metrics, strict=True):
        all_values = [
            getattr(lookup[(context, method)], field)
            for method in methods
            for context in CONTEXTS
        ]
        style_axis(ax, ylim=tight_limits(all_values))
        ax.set_ylabel(title, fontsize=13.5, fontweight="bold", labelpad=10)
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


def render_model_logo_variant(model: str, results: list[Result]) -> None:
    """Render the alternate design with tinted model logos as data marks."""

    spec = MODEL_SPECS[model]
    methods = tuple(spec["methods"])
    lookup = {(row.context, row.method): row for row in results if row.model == model}

    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.55), gridspec_kw={"wspace": 0.25})
    metrics = (
        ("answer_recall", "Answer recall (%)"),
        ("citation_f1", "Citation F1 (%)"),
    )

    for ax, (field, label) in zip(axes, metrics, strict=True):
        all_values = [
            getattr(lookup[(context, method)], field)
            for method in methods
            for context in CONTEXTS
        ]
        style_axis(ax, ylim=tight_limits(all_values))
        ax.set_ylabel(label, fontsize=13.5, fontweight="bold", labelpad=10)
        for method in methods:
            values = [getattr(lookup[(context, method)], field) for context in CONTEXTS]
            ours = method == "H-infinity"
            ax.plot(
                [0, 1],
                values,
                color=LOGO_METHOD_COLORS[method],
                linestyle=METHOD_STYLES[method],
                linewidth=3.2 if ours else 2.0,
                solid_capstyle="round",
                zorder=4 if ours else 3,
            )
            for x, value in enumerate(values):
                add_logo_marker(ax, x=float(x), y=value, model=model, method=method)

    add_logo_legend(fig, model, methods)
    add_model_heading(fig, spec)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.25, top=0.79)

    PLOTS.mkdir(parents=True, exist_ok=True)
    stem = f"figure_long_context_{spec['slug']}_logos"
    fig.savefig(PLOTS / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(PLOTS / f"{stem}.png", dpi=450, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def render() -> None:
    setup_style()
    results = read_results()
    for model in MODEL_SPECS:
        render_model(model, results)
        render_model_logo_variant(model, results)


if __name__ == "__main__":
    render()
