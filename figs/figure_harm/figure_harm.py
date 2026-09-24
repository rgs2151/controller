#!/usr/bin/env python3
"""Render the three-panel HarmBench summary figure."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    AnnotationBbox,
    HPacker,
    OffsetImage,
    TextArea,
)
from PIL import Image


UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[1]
SOURCE = REPO / "figs/benchmark_figures/cache/harmbench_refusal_results.csv"
PLOTS = UNIT / "plots"
LLAMA_LOGO = REPO / "figs/logos/llama_transparent.png"

TEAL = "#398197"  # preserved only for the original figure variant
PROJECT_TEAL = "#007C7C"
INK = "#202124"
MUTED = "#6E757B"
GRID = "#E2E5E7"
MODEL_LABELS = {
    "Llama-3.2-1B-Instruct": "Llama-3.2-1B",
    "Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
}
MODEL_SIZES = {
    "Llama-3.2-1B-Instruct": 1.0,
    "Llama-3.2-3B-Instruct": 3.0,
    "Llama-3.1-8B-Instruct": 8.0,
}
MODEL_MARKERS = {
    "Llama-3.2-1B-Instruct": "o",
    "Llama-3.2-3B-Instruct": "^",
    "Llama-3.1-8B-Instruct": "s",
}
METHODS = ("Original", "A-LQR", "H-infinity")
METHOD_COLORS = {"Original": "#BCC1C5", "A-LQR": "#777F86", "H-infinity": TEAL}
LOGO_METHOD_COLORS = {
    "Original": "#737B80",
    "A-LQR": "#6558A6",
    "H-infinity": PROJECT_TEAL,
}
METHOD_LABELS = {"Original": "Original", "A-LQR": "A-LQR", "H-infinity": r"$H_\infty$ (ours)"}
TEMPLATES = (
    "Direct",
    "John persona",
    "YOJA/Nona roleplay",
    "DNE nonresponse",
    "APM programmer",
    "Jailbreak Bot",
)
TEMPLATE_LABELS = {
    "Direct": "Direct",
    "John persona": "John\npersona",
    "YOJA/Nona roleplay": "YOJA/Nona\nroleplay",
    "DNE nonresponse": "DNE\nnonresponse",
    "APM programmer": "APM\nprogrammer",
    "Jailbreak Bot": "Jailbreak\nBot",
}


def setup_style() -> None:
    sns.set_theme(context="talk", style="ticks", palette="dark")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_rows() -> list[dict[str, str | float]]:
    numeric = {"asr", "safe_concept_relevance", "instruction_relevance", "fluency", "overall_steering"}
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) if key in numeric else value.strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def marker_size(model: str, base: float = 62.0) -> float:
    """Encode model scale in marker area without letting 8B dominate."""

    return base + 34.0 * math.log2(MODEL_SIZES[model])


def smooth_density(values: list[float], low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(low, high, 240)
    spread = float(np.std(values))
    bandwidth = max(0.035 * (high - low), 0.50 * spread)
    density = np.zeros_like(grid)
    for value in values:
        density += np.exp(-0.5 * ((grid - value) / bandwidth) ** 2)
    density /= max(float(density.max()), 1e-9)
    return grid, density


def aggregates(rows: list[dict[str, str | float]]) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[dict[str, str | float]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["method"]))].append(row)
    result: dict[tuple[str, str], dict[str, float]] = {}
    for key, subset in grouped.items():
        asr = np.asarray([float(row["asr"]) for row in subset])
        result[key] = {
            "mean_asr": float(asr.mean()),
            "worst_asr": float(asr.max()),
            "mean_rejection": float(100.0 - asr.mean()),
            "worst_rejection": float(100.0 - asr.max()),
            "safe": float(np.mean([float(row["safe_concept_relevance"]) for row in subset])),
        }
    return result


def style_main_axis(
    ax: plt.Axes, *, grid_axis: str = "both", font_scale: float = 1.0
) -> None:
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.75, zorder=0)
    ax.tick_params(labelsize=9.5 * font_scale, colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    sns.despine(ax=ax, trim=True, offset=6)


def tinted_logo(color: str) -> np.ndarray:
    image = np.asarray(Image.open(LLAMA_LOGO).convert("RGBA"), dtype=np.uint8)
    output = np.zeros_like(image)
    output[..., :3] = np.round(
        255 * np.asarray(mpl.colors.to_rgb(color))
    ).astype(np.uint8)
    output[..., 3] = image[..., 3]
    return output


def logo_zoom(model: str, *, scale: float = 1.0) -> float:
    """Log-scale 1B/3B/8B marks without allowing the 8B logo to dominate."""
    return scale * (0.016 + 0.004 * math.log2(MODEL_SIZES[model]))


def add_model_mark(
    ax: plt.Axes,
    x: float,
    y: float,
    *,
    model: str,
    method: str,
    size_factor: float,
    logo_images: dict[str, np.ndarray] | None,
    colors: dict[str, str],
) -> None:
    is_ours = method == "H-infinity"
    if logo_images is None:
        ax.scatter(
            x,
            y,
            s=size_factor * marker_size(model),
            marker=MODEL_MARKERS[model],
            facecolor=colors[method],
            edgecolor=INK if is_ours else "white",
            linewidth=1.1 if is_ours else 0.7,
            alpha=1.0 if is_ours else 0.88,
            zorder=6 if is_ours else 5,
        )
        return
    image = OffsetImage(
        logo_images[method],
        zoom=logo_zoom(model, scale=1.10 if is_ours else 1.0),
        resample=True,
    )
    image.set_alpha(
        1.0 if is_ours else (0.95 if method == "A-LQR" else 0.72)
    )
    ax.add_artist(
        AnnotationBbox(
            image,
            (x, y),
            frameon=False,
            pad=0,
            zorder=7 if is_ours else 5,
            annotation_clip=True,
        )
    )


def draw_rejection_panel(
    fig: plt.Figure,
    spec,
    summary: dict[tuple[str, str], dict[str, float]],
    *,
    logo_images: dict[str, np.ndarray] | None = None,
    colors: dict[str, str] = METHOD_COLORS,
    font_scale: float = 1.0,
) -> None:
    nested = spec.subgridspec(2, 2, width_ratios=(1, 0.17), height_ratios=(0.18, 1), hspace=0.05, wspace=0.06)
    top = fig.add_subplot(nested[0, 0])
    ax = fig.add_subplot(nested[1, 0])
    right = fig.add_subplot(nested[1, 1])

    x_limits = (83.5, 100.0)
    y_limits = (75.0, 100.0)
    for method in METHODS:
        xs = [summary[(model, method)]["mean_rejection"] for model in MODEL_SIZES]
        ys = [summary[(model, method)]["worst_rejection"] for model in MODEL_SIZES]
        gx, dx = smooth_density(xs, *x_limits)
        gy, dy = smooth_density(ys, *y_limits)
        top.plot(gx, dx, color=colors[method], lw=1.5, ls="-" if method == "H-infinity" else (0, (3, 2)), alpha=0.95)
        top.fill_between(gx, 0, dx, color=colors[method], alpha=0.10)
        right.plot(dy, gy, color=colors[method], lw=1.5, ls="-" if method == "H-infinity" else (0, (3, 2)), alpha=0.95)
        right.fill_betweenx(gy, 0, dy, color=colors[method], alpha=0.10)
        for model, x, y in zip(MODEL_SIZES, xs, ys, strict=True):
            add_model_mark(
                ax, x, y, model=model, method=method, size_factor=1.0,
                logo_images=logo_images, colors=colors,
            )

    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_xlabel("Mean attack rejection across conditions (%)", fontsize=10.5 * font_scale)
    ax.set_ylabel("Worst-case attack rejection (%)", fontsize=10.5 * font_scale)
    style_main_axis(ax, font_scale=font_scale)
    for marginal in (top, right):
        marginal.set_xticks([])
        marginal.set_yticks([])
        marginal.grid(False)
        for spine in marginal.spines.values():
            spine.set_visible(False)
    top.set_xlim(*x_limits)
    right.set_ylim(*y_limits)


def draw_profile_panel(
    ax: plt.Axes,
    rows: list[dict[str, str | float]],
    *,
    logo_images: dict[str, np.ndarray] | None = None,
    colors: dict[str, str] = METHOD_COLORS,
    font_scale: float = 1.0,
    bottom_spine_at_zero: bool = True,
) -> None:
    indexed = {(str(row["model"]), str(row["template"]), str(row["method"])): float(row["asr"]) for row in rows}
    x = np.arange(len(TEMPLATES))
    displayed_values: list[float] = []
    for model in MODEL_SIZES:
        for method in METHODS:
            values = np.asarray([indexed[(model, template, method)] for template in TEMPLATES])
            displayed_values.extend(values.tolist())
            for template_x, value in zip(x, values, strict=True):
                add_model_mark(
                    ax, float(template_x), float(value), model=model,
                    method=method, size_factor=0.64, logo_images=logo_images,
                    colors=colors,
                )
    ax.set_xticks(x)
    ax.set_xticklabels([TEMPLATE_LABELS[t] for t in TEMPLATES], fontsize=7.7 * font_scale)
    ax.set_ylabel("Attack success rate (%)", fontsize=10.5 * font_scale)
    ax.set_xlabel(
        "Jailbreak template",
        fontsize=10.5 * font_scale,
        labelpad=14,
    )
    # Keep the visible baseline at zero while reserving enough internal room for
    # markers centered exactly on zero to render without clipping.
    ax.set_ylim(-2.0, 1.10 * max(displayed_values))
    style_main_axis(ax, grid_axis="y", font_scale=font_scale)
    ax.tick_params(axis="x", pad=8)
    if bottom_spine_at_zero:
        ax.spines["bottom"].set_position(("data", 0))


def draw_frontier_panel(
    fig: plt.Figure,
    spec,
    summary: dict[tuple[str, str], dict[str, float]],
    *,
    logo_images: dict[str, np.ndarray] | None = None,
    colors: dict[str, str] = METHOD_COLORS,
    font_scale: float = 1.0,
) -> None:
    nested = spec.subgridspec(2, 2, width_ratios=(1, 0.17), height_ratios=(0.18, 1), hspace=0.05, wspace=0.06)
    top = fig.add_subplot(nested[0, 0])
    ax = fig.add_subplot(nested[1, 0])
    right = fig.add_subplot(nested[1, 1])

    all_x = [summary[(model, method)]["mean_asr"] for model in MODEL_SIZES for method in METHODS]
    all_y = [summary[(model, method)]["safe"] for model in MODEL_SIZES for method in METHODS]
    x_limits = (0.0, max(all_x) * 1.10)
    y_limits = (min(all_y) - 0.012, max(all_y) + 0.008)
    for model in MODEL_SIZES:
        target = summary[(model, "H-infinity")]
        for method in ("Original", "A-LQR"):
            source = summary[(model, method)]
            ax.plot(
                [source["mean_asr"], target["mean_asr"]],
                [source["safe"], target["safe"]],
                color="#B7BDC1",
                lw=0.9,
                ls=(0, (3, 2)),
                alpha=0.55,
            )
        for method in METHODS:
            point = summary[(model, method)]
            add_model_mark(
                ax, point["mean_asr"], point["safe"], model=model,
                method=method, size_factor=0.70, logo_images=logo_images,
                colors=colors,
            )

    for method in METHODS:
        method_x = [summary[(model, method)]["mean_asr"] for model in MODEL_SIZES]
        method_y = [summary[(model, method)]["safe"] for model in MODEL_SIZES]
        gx, dx = smooth_density(method_x, *x_limits)
        gy, dy = smooth_density(method_y, *y_limits)
        linewidth = 2.0 if method == "H-infinity" else 1.25
        fill_alpha = 0.16 if method == "H-infinity" else 0.08
        top.plot(gx, dx, color=colors[method], lw=linewidth)
        top.fill_between(gx, 0, dx, color=colors[method], alpha=fill_alpha)
        right.plot(dy, gy, color=colors[method], lw=linewidth)
        right.fill_betweenx(gy, 0, dy, color=colors[method], alpha=fill_alpha)

    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_xlabel("Mean attack success rate (%)", fontsize=10.5 * font_scale)
    ax.set_ylabel("Mean safe-concept relevance (0–2)", fontsize=10.5 * font_scale)
    style_main_axis(ax, font_scale=font_scale)
    for marginal in (top, right):
        marginal.set_xticks([])
        marginal.set_yticks([])
        marginal.grid(False)
        for spine in marginal.spines.values():
            spine.set_visible(False)
    top.set_xlim(*x_limits)
    right.set_ylim(*y_limits)


def create_figure(rows: list[dict[str, str | float]]) -> plt.Figure:
    summary = aggregates(rows)
    fig = plt.figure(figsize=(15.4, 4.9))
    outer = fig.add_gridspec(1, 3, width_ratios=(1.03, 1.18, 0.98), wspace=0.36)
    draw_rejection_panel(fig, outer[0], summary)
    draw_profile_panel(fig.add_subplot(outer[1]), rows)
    draw_frontier_panel(fig, outer[2], summary)

    model_handles = [
        Line2D(
            [0],
            [0],
            marker=MODEL_MARKERS[model],
            linestyle="none",
            markersize=10.5 + 1.3 * math.log2(MODEL_SIZES[model]),
            markerfacecolor="#62696F",
            markeredgecolor="#62696F",
            label=MODEL_LABELS[model],
        )
        for model in MODEL_SIZES
    ]
    method_handles = [
        Line2D([0], [0], color=METHOD_COLORS[method], lw=2.2, ls="-" if method == "H-infinity" else (0, (3, 2)), label=METHOD_LABELS[method])
        for method in METHODS
    ]
    first = fig.legend(handles=model_handles, loc="upper center", bbox_to_anchor=(0.50, 1.035), ncol=3, fontsize=14.0, handletextpad=0.45, columnspacing=1.35)
    fig.add_artist(first)
    fig.legend(handles=method_handles, loc="lower center", bbox_to_anchor=(0.50, -0.015), ncol=3, fontsize=9.5, handlelength=2.1, columnspacing=1.3)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.87, bottom=0.20)
    return fig


def draw_logo_size_legend(
    fig: plt.Figure,
    *,
    bounds: tuple[float, float, float, float] = (0.285, 0.885, 0.43, 0.09),
    fontsize: float = 14.0,
    separation: float = 24.0,
) -> None:
    ax = fig.add_axes(bounds)
    ax.set_axis_off()
    native = np.asarray(Image.open(LLAMA_LOGO).convert("RGBA"))
    groups = []
    legend_zoom = {
        "Llama-3.2-1B-Instruct": 0.018,
        "Llama-3.2-3B-Instruct": 0.024,
        "Llama-3.1-8B-Instruct": 0.031,
    }
    for model in MODEL_SIZES:
        logo = OffsetImage(native, zoom=legend_zoom[model], resample=True)
        text = TextArea(
            MODEL_LABELS[model],
            textprops={"fontsize": fontsize, "fontfamily": "Arial", "color": "black"},
        )
        groups.append(HPacker(children=[logo, text], align="center", pad=0, sep=4))
    packed = HPacker(children=groups, align="center", pad=0, sep=separation)
    ax.add_artist(
        AnchoredOffsetbox(
            loc="center",
            child=packed,
            pad=0,
            frameon=False,
            bbox_to_anchor=(0.5, 0.5),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
    )


def create_logo_figure(rows: list[dict[str, str | float]]) -> plt.Figure:
    """Logo-based HarmBench variant with publication-scale typography."""
    summary = aggregates(rows)
    logo_images = {
        method: tinted_logo(LOGO_METHOD_COLORS[method]) for method in METHODS
    }
    font_scale = 1.28
    fig = plt.figure(figsize=(17.2, 5.35))
    outer = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.03, 1.50, 0.98),
        wspace=0.31,
    )
    draw_rejection_panel(
        fig,
        outer[0],
        summary,
        logo_images=logo_images,
        colors=LOGO_METHOD_COLORS,
        font_scale=font_scale,
    )
    draw_profile_panel(
        fig.add_subplot(outer[1]),
        rows,
        logo_images=logo_images,
        colors=LOGO_METHOD_COLORS,
        font_scale=font_scale,
    )
    draw_frontier_panel(
        fig,
        outer[2],
        summary,
        logo_images=logo_images,
        colors=LOGO_METHOD_COLORS,
        font_scale=font_scale,
    )
    draw_logo_size_legend(fig)
    method_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=10.0 if method == "H-infinity" else 8.0,
            markerfacecolor=LOGO_METHOD_COLORS[method],
            markeredgecolor="none",
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    legend = fig.legend(
        handles=method_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=len(METHODS),
        fontsize=12.0,
        handlelength=0.8,
        handletextpad=0.35,
        columnspacing=2.0,
        labelcolor="black",
    )
    for text in legend.get_texts():
        if "ours" in text.get_text():
            text.set_fontweight("bold")
    fig.subplots_adjust(left=0.060, right=0.985, top=0.82, bottom=0.22)
    return fig


def logo_method_handles() -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=10.0 if method == "H-infinity" else 8.0,
            markerfacecolor=LOGO_METHOD_COLORS[method],
            markeredgecolor="none",
            alpha=1.0 if method != "Original" else 0.78,
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]


def add_logo_method_legend(
    fig: plt.Figure,
    *,
    y: float,
    fontsize: float = 12.0,
) -> None:
    legend = fig.legend(
        handles=logo_method_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=len(METHODS),
        fontsize=fontsize,
        handlelength=0.8,
        handletextpad=0.35,
        columnspacing=2.0,
        labelcolor="black",
    )
    for text in legend.get_texts():
        if "ours" in text.get_text():
            text.set_fontweight("bold")


def create_logo_main_figure(rows: list[dict[str, str | float]]) -> plt.Figure:
    summary = aggregates(rows)
    logo_images = {
        method: tinted_logo(LOGO_METHOD_COLORS[method]) for method in METHODS
    }
    font_scale = 1.28
    fig = plt.figure(figsize=(12.6, 5.35))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.48, 1.0),
        wspace=0.27,
    )
    profile_grid = outer[0].subgridspec(
        2,
        1,
        height_ratios=(0.18, 1.0),
        hspace=0.05,
    )
    spacer = fig.add_subplot(profile_grid[0])
    spacer.set_axis_off()
    profile_ax = fig.add_subplot(profile_grid[1])
    draw_profile_panel(
        profile_ax,
        rows,
        logo_images=logo_images,
        colors=LOGO_METHOD_COLORS,
        font_scale=font_scale,
        bottom_spine_at_zero=False,
    )
    draw_frontier_panel(
        fig,
        outer[1],
        summary,
        logo_images=logo_images,
        colors=LOGO_METHOD_COLORS,
        font_scale=font_scale,
    )
    draw_logo_size_legend(
        fig,
        bounds=(0.25, 0.885, 0.50, 0.09),
        fontsize=14.0,
        separation=24.0,
    )
    add_logo_method_legend(fig, y=-0.01)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.82, bottom=0.27)
    return fig


def create_logo_rejection_figure(
    rows: list[dict[str, str | float]],
) -> plt.Figure:
    summary = aggregates(rows)
    logo_images = {
        method: tinted_logo(LOGO_METHOD_COLORS[method]) for method in METHODS
    }
    fig = plt.figure(figsize=(6.2, 5.35))
    outer = fig.add_gridspec(1, 1)
    draw_rejection_panel(
        fig,
        outer[0],
        summary,
        logo_images=logo_images,
        colors=LOGO_METHOD_COLORS,
        font_scale=1.22,
    )
    draw_logo_size_legend(
        fig,
        bounds=(0.06, 0.885, 0.88, 0.09),
        fontsize=10.5,
        separation=10.0,
    )
    add_logo_method_legend(fig, y=-0.01, fontsize=10.5)
    fig.subplots_adjust(left=0.15, right=0.96, top=0.82, bottom=0.25)
    return fig


def render_logo_variant() -> None:
    setup_style()
    PLOTS.mkdir(parents=True, exist_ok=True)
    figure = create_logo_figure(read_rows())
    figure.savefig(
        PLOTS / "figure_harm_logos.pdf",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    figure.savefig(
        PLOTS / "figure_harm_logos.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(figure)
    for stem, creator in [
        ("figure_harm_main", create_logo_main_figure),
        ("figure_harm_rejection", create_logo_rejection_figure),
    ]:
        figure = creator(read_rows())
        figure.savefig(
            PLOTS / f"{stem}.pdf",
            bbox_inches="tight",
            pad_inches=0.04,
        )
        figure.savefig(
            PLOTS / f"{stem}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.04,
        )
        plt.close(figure)


def main() -> None:
    setup_style()
    PLOTS.mkdir(parents=True, exist_ok=True)
    figure = create_figure(read_rows())
    figure.savefig(PLOTS / "figure_harm.pdf", bbox_inches="tight", pad_inches=0.04)
    figure.savefig(PLOTS / "figure_harm.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    render_logo_variant()


if __name__ == "__main__":
    main()
