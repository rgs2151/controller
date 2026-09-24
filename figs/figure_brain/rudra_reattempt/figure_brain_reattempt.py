#!/usr/bin/env python3
"""Native recreation of the full brain/LLM residual-analysis figure."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch


UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[2]
RESULTS = REPO / "NeuroAnalysis" / "results"
PLOTS = UNIT / "plots"
PANELS = PLOTS / "panels"

INK = "#17191C"
MUTED = "#626970"
LIGHT = "#F4F5F6"
GRID = "#DDE1E4"
SPINE = "#AAB0B5"
ID_COLOR = "#7895B4"
OOD_COLOR = "#B77967"
BLUE = "#2E78C7"
GREEN = "#269D66"
PURPLE = "#7651A8"
ORANGE = "#E59A34"
RED = "#C83D42"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panel_title(ax: Axes, letter: str, title: str) -> None:
    ax.text(
        -0.02,
        1.04,
        letter,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.02,
        1.04,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )


def style_data_axis(ax: Axes, *, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.75, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_color(SPINE)
        ax.spines[name].set_linewidth(0.9)
    ax.tick_params(labelsize=8.2, length=3.2, width=0.8, pad=2)


def arrow(ax: Axes, start: tuple[float, float], end: tuple[float, float], **kwargs) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.0,
            color=MUTED,
            shrinkA=2,
            shrinkB=2,
            **kwargs,
        )
    )


def box(
    ax: Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = LIGHT,
    edgecolor: str = SPINE,
    fontsize: float = 7.4,
    weight: str = "normal",
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
    )


def draw_a_seeg_state(ax: Axes) -> None:
    panel_title(ax, "A", "Intracranial neural state")
    ax.set_axis_off()
    # Native schematic placeholder: no raw MAT/anatomical export is embedded.
    ax.add_patch(Ellipse((0.34, 0.62), 0.54, 0.47, facecolor="none", edgecolor=MUTED, lw=1.2))
    ax.plot([0.17, 0.30, 0.44, 0.55], [0.55, 0.72, 0.69, 0.52], color=GRID, lw=1.0)
    ax.plot([0.20, 0.32, 0.45, 0.50], [0.67, 0.52, 0.58, 0.73], color=GRID, lw=1.0)
    leads = [
        ((0.24, 0.74), (0.36, 0.51), BLUE),
        ((0.36, 0.78), (0.44, 0.48), GREEN),
        ((0.49, 0.69), (0.55, 0.55), PURPLE),
    ]
    for start, end, color in leads:
        xs = np.linspace(start[0], end[0], 7)
        ys = np.linspace(start[1], end[1], 7)
        ax.plot(xs, ys, color=color, lw=1.2)
        ax.scatter(xs, ys, s=16, color=color, edgecolor="white", linewidth=0.35, zorder=3)
    t = np.linspace(0, 1, 150)
    for index, (y, color, phase) in enumerate(((0.31, GREEN, 0.0), (0.20, ORANGE, 0.7), (0.09, PURPLE, 1.4))):
        trace = y + 0.022 * np.sin(2 * np.pi * (8 + index) * t + phase) + 0.008 * np.sin(2 * np.pi * 21 * t)
        ax.plot(0.12 + 0.72 * t, trace, color=color, lw=1.0)
    ax.text(0.85, 0.20, "simultaneous\nchannel traces", ha="right", va="center", fontsize=7.0, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def draw_b_msit(ax: Axes) -> None:
    panel_title(ax, "B", "MSIT task and held-out shifts")
    ax.set_axis_off()
    box(ax, (0.04, 0.60), 0.19, 0.18, "Fixation\n0.5 s")
    box(ax, (0.30, 0.60), 0.22, 0.18, "Stimulus /\ndecision")
    box(ax, (0.59, 0.60), 0.19, 0.18, "Response")
    arrow(ax, (0.23, 0.69), (0.30, 0.69))
    arrow(ax, (0.52, 0.69), (0.59, 0.69))
    ax.text(0.04, 0.43, "Conflict", fontsize=7.5, fontweight="bold")
    box(ax, (0.19, 0.36), 0.23, 0.14, "low (ID)", facecolor="#EAF1F7", edgecolor=ID_COLOR)
    box(ax, (0.60, 0.36), 0.25, 0.14, "high (OOD)", facecolor="#F6ECE8", edgecolor=OOD_COLOR)
    arrow(ax, (0.42, 0.43), (0.60, 0.43))
    ax.text(0.04, 0.20, "Context", fontsize=7.5, fontweight="bold")
    box(ax, (0.19, 0.13), 0.23, 0.14, "NS1 (ID)", facecolor="#EAF1F7", edgecolor=ID_COLOR)
    box(ax, (0.60, 0.13), 0.25, 0.14, "NS2 (OOD)", facecolor="#F6ECE8", edgecolor=OOD_COLOR)
    arrow(ax, (0.42, 0.20), (0.60, 0.20))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def draw_c_cortical_model(ax: Axes) -> None:
    panel_title(ax, "C", "Distributed cortical state model")
    ax.set_axis_off()
    nodes = {
        "PPC": ((0.08, 0.73), PURPLE),
        "TP": ((0.08, 0.32), ORANGE),
        "VLPFC": ((0.29, 0.50), "#5B8BD0"),
        "DLPFC": ((0.47, 0.70), BLUE),
        "PT": ((0.58, 0.29), RED),
        "SMA/PreM": ((0.73, 0.57), GREEN),
        "M1/S1": ((0.91, 0.57), "#83BD3F"),
    }
    for source, target in (("PPC", "DLPFC"), ("TP", "VLPFC"), ("VLPFC", "DLPFC"), ("DLPFC", "SMA/PreM"), ("PT", "SMA/PreM"), ("SMA/PreM", "M1/S1")):
        arrow(ax, nodes[source][0], nodes[target][0], connectionstyle="arc3,rad=0.08")
    for label, (position, color) in nodes.items():
        ax.add_patch(Circle(position, 0.038, facecolor=color, edgecolor="white", linewidth=0.8, zorder=3))
        ax.text(position[0], position[1] + 0.065, label, ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax.text(0.50, 0.10, r"$x_{t+\Delta}=A_\Delta x_t+b_\Delta+g_\Delta(x_t)$", ha="center", fontsize=11.5)
    ax.text(0.39, 0.025, "local linear dynamics", ha="center", fontsize=6.6, color=MUTED)
    ax.text(0.72, 0.025, "structured residual", ha="center", fontsize=6.6, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def draw_state_scatter(ax: Axes, data: pd.DataFrame) -> None:
    panel_title(ax, "D", "Residual magnitude across leads")
    selected = data[data["area"].notna()].copy()
    colors = mpl.colormaps["viridis"](np.linspace(0.08, 0.88, len(selected)))
    for (_, row), color in zip(selected.iterrows(), colors, strict=True):
        ax.scatter(row["n_ch"], row["med_resid"], s=50, color=color, edgecolor="white", linewidth=0.5, zorder=3)
        ax.annotate(str(row.iloc[0]), (row["n_ch"], row["med_resid"]), xytext=(3, 3), textcoords="offset points", fontsize=6.3)
    ax.set_xlabel("Channels in lead")
    ax.set_ylabel("Median residual RMS (z)")
    style_data_axis(ax, grid_axis="both")


def draw_error_bars(
    ax: Axes,
    data: pd.DataFrame,
    *,
    letter: str,
    title: str,
    condition: str | None,
    ood_key: str,
    horizon: str,
    xlabel: str,
) -> None:
    panel_title(ax, letter, title)
    if condition is not None and "axis" in data.columns:
        selected = data[data["axis"] == condition]
        ood_label = "OOD"
    else:
        selected = data[data["split"].isin(["ID", ood_key])]
        ood_label = ood_key
    table = selected.pivot(index=horizon, columns="split", values="reduction").sort_index()
    x = np.arange(len(table))
    width = 0.34
    ax.bar(x - width / 2, table["ID"], width, color=ID_COLOR, label="ID", zorder=3)
    ax.bar(x + width / 2, table[ood_label], width, color=OOD_COLOR, label="OOD", zorder=3)
    ax.axhline(0, color=INK, linewidth=0.9, zorder=4)
    ax.set_xticks(x, [str(v) for v in table.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Residual-RMS reduction (z)")
    style_data_axis(ax)


def draw_g_gpt_architecture(ax: Axes) -> None:
    panel_title(ax, "G", "GPT-2 Small residual-stream architecture")
    ax.set_axis_off()
    box(ax, (0.02, 0.42), 0.15, 0.20, "Tokens", facecolor="#F4F4F4")
    box(ax, (0.22, 0.42), 0.18, 0.20, "Embedding\n+ position", facecolor="#EEF3F8", edgecolor=ID_COLOR)
    arrow(ax, (0.17, 0.52), (0.22, 0.52))
    block_x = np.linspace(0.47, 0.83, 5)
    for index, xpos in enumerate(block_x):
        label = str(index + 1) if index < 2 else (r"$\cdots$" if index == 2 else str(8 + index))
        face = "#EAE4F2" if index % 2 == 0 else "#E5EFF8"
        edge = PURPLE if index % 2 == 0 else BLUE
        box(ax, (xpos, 0.38), 0.055, 0.28, label, facecolor=face, edgecolor=edge, fontsize=7.0, weight="bold")
        if index == 0:
            arrow(ax, (0.40, 0.52), (xpos, 0.52))
        else:
            arrow(ax, (block_x[index - 1] + 0.055, 0.52), (xpos, 0.52))
    box(ax, (0.91, 0.42), 0.08, 0.20, "LN\nlogits", facecolor="#F4F4F4", fontsize=6.8)
    arrow(ax, (0.885, 0.52), (0.91, 0.52))
    ax.plot([0.47, 0.885], [0.73, 0.73], color=MUTED, lw=0.9)
    ax.plot([0.47, 0.47], [0.70, 0.73], color=MUTED, lw=0.9)
    ax.plot([0.885, 0.885], [0.70, 0.73], color=MUTED, lw=0.9)
    ax.text(0.677, 0.77, "12 transformer blocks", ha="center", fontsize=7.6, fontweight="bold")
    ax.text(0.50, 0.20, r"final-token state  $h_k = H_k[T,:]$", ha="center", fontsize=9.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def draw_h_prompt_shifts(ax: Axes) -> None:
    panel_title(ax, "H", "Prompt distributions")
    ax.set_axis_off()
    box(ax, (0.05, 0.56), 0.32, 0.23, 'English ID\n"The movie was\nsurprisingly moving."', facecolor="#EAF1F7", edgecolor=ID_COLOR, fontsize=6.6)
    box(ax, (0.63, 0.64), 0.32, 0.20, 'Spanish OOD\n"La película fue\nsorprendentemente..."', facecolor="#F6ECE8", edgecolor=OOD_COLOR, fontsize=6.5)
    box(ax, (0.63, 0.28), 0.32, 0.20, "Long-context OOD\nreview embedded in\nextended context", facecolor="#F6ECE8", edgecolor=OOD_COLOR, fontsize=6.5)
    arrow(ax, (0.37, 0.67), (0.63, 0.74))
    arrow(ax, (0.37, 0.67), (0.63, 0.38))
    ax.text(0.49, 0.77, "language shift", ha="center", fontsize=6.4, color=MUTED)
    ax.text(0.49, 0.42, "context shift", ha="center", fontsize=6.4, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def draw_i_layerwise_linearization(ax: Axes) -> None:
    panel_title(ax, "I", "Layer-wise local linearization")
    ax.set_axis_off()
    xs = np.linspace(0.05, 0.91, 12)
    colors = mpl.colormaps["turbo"](np.linspace(0.08, 0.92, 12))
    for index, (xpos, color) in enumerate(zip(xs, colors, strict=True), start=1):
        ax.add_patch(FancyBboxPatch((xpos, 0.52), 0.047, 0.18, boxstyle="round,pad=0.006", facecolor="white", edgecolor=color, linewidth=1.6))
        ax.text(xpos + 0.0235, 0.61, str(index), ha="center", va="center", fontsize=5.8, fontweight="bold", color=color)
        if index < 12:
            arrow(ax, (xpos + 0.047, 0.61), (xs[index], 0.61))
    ax.text(0.50, 0.82, r"$h_1 \rightarrow h_2 \rightarrow \cdots \rightarrow h_{13}$", ha="center", fontsize=9.5)
    ax.text(0.50, 0.35, r"$x_{k+1}=A_kx_k+B_ku_k+D_kw_k,\quad k=1,\ldots,12$", ha="center", fontsize=10.2)
    ax.text(0.50, 0.20, "one fitted local transition and disturbance geometry per block", ha="center", fontsize=7.2, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def draw_j_layers(ax: Axes, data: pd.DataFrame) -> None:
    panel_title(ax, "J", "Residual magnitude across depth")
    selected = data[data["med_resid"].notna()].copy()
    layers = pd.to_numeric(selected.iloc[:, 0])
    colors = mpl.colormaps["turbo"](np.linspace(0.05, 0.95, len(selected)))
    ax.plot(layers, selected["med_resid"], color=GRID, lw=1.0, zorder=1)
    for layer, value, color in zip(layers, selected["med_resid"], colors, strict=True):
        ax.scatter(layer, value, s=48, color=color, edgecolor="white", linewidth=0.5, zorder=3)
        ax.annotate(f"L{int(layer)}", (layer, value), xytext=(2, 3), textcoords="offset points", fontsize=5.9)
    ax.set_xlabel("Transformer layer (128 dims)")
    ax.set_ylabel("Median residual RMS (z)")
    ax.set_xlim(0.5, 12.8)
    style_data_axis(ax, grid_axis="both")


PanelDrawer = Callable[[Axes], None]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(RESULTS / "final_panelD_error_shrinkage_LVF_theta.csv"),
        pd.read_csv(RESULTS / "final_panelF_state_size_theta.csv"),
        pd.read_csv(RESULTS / "final_llmfit_panelD_error_shrinkage.csv"),
        pd.read_csv(RESULTS / "final_llmfit_panelF_state_size.csv"),
    )


def panel_drawers() -> list[tuple[str, PanelDrawer]]:
    brain_error, brain_state, llm_error, llm_state = load_data()
    return [
        ("a_seeg_state", draw_a_seeg_state),
        ("c_cortical_model", draw_c_cortical_model),
        ("d_brain_state_size", lambda ax: draw_state_scatter(ax, brain_state)),
        ("b_msit_shifts", draw_b_msit),
        (
            "e_brain_conflict",
            lambda ax: draw_error_bars(
                ax,
                brain_error,
                letter="E",
                title="Conflict shift",
                condition="conflict",
                ood_key="OOD",
                horizon="horizon_ms",
                xlabel=r"Horizon $\Delta$ (ms)",
            ),
        ),
        (
            "f_brain_stim_context",
            lambda ax: draw_error_bars(
                ax,
                brain_error,
                letter="F",
                title="Stimulation-context shift",
                condition="stim-context",
                ood_key="OOD",
                horizon="horizon_ms",
                xlabel=r"Horizon $\Delta$ (ms)",
            ),
        ),
        ("g_gpt2_architecture", draw_g_gpt_architecture),
        ("i_layerwise_linearization", draw_i_layerwise_linearization),
        ("j_llm_layers", lambda ax: draw_j_layers(ax, llm_state)),
        ("h_prompt_shifts", draw_h_prompt_shifts),
        (
            "k_llm_spanish",
            lambda ax: draw_error_bars(
                ax,
                llm_error,
                letter="K",
                title="Spanish shift",
                condition=None,
                ood_key="OOD-es",
                horizon="horizon_tok",
                xlabel=r"Horizon $\Delta$ (tokens)",
            ),
        ),
        (
            "l_llm_long_context",
            lambda ax: draw_error_bars(
                ax,
                llm_error,
                letter="L",
                title="Long-context shift",
                condition=None,
                ood_key="OOD-ctx",
                horizon="horizon_tok",
                xlabel=r"Horizon $\Delta$ (tokens)",
            ),
        ),
    ]


def create_composite() -> plt.Figure:
    fig = plt.figure(figsize=(13.2, 10.7))
    grid = fig.add_gridspec(
        4,
        3,
        width_ratios=(1.18, 1.06, 1.0),
        height_ratios=(1.10, 0.90, 1.08, 0.92),
        left=0.055,
        right=0.985,
        top=0.94,
        bottom=0.07,
        wspace=0.32,
        hspace=0.50,
    )
    drawers = panel_drawers()
    axes: list[Axes] = []
    for index, (_, drawer) in enumerate(drawers):
        ax = fig.add_subplot(grid[index // 3, index % 3])
        drawer(ax)
        axes.append(ax)
    fig.text(0.012, 0.755, "BIOLOGICAL\nNETWORK", rotation=90, ha="center", va="center", fontsize=10.5, fontweight="bold", color=MUTED)
    fig.text(0.012, 0.285, "ARTIFICIAL\nNETWORK", rotation=90, ha="center", va="center", fontsize=10.5, fontweight="bold", color=MUTED)
    # Subtle row separator; it denotes systems, not a data axis.
    fig.add_artist(mpl.lines.Line2D([0.045, 0.985], [0.505, 0.505], transform=fig.transFigure, color="#D5D9DC", lw=1.0))
    return fig


def save_figure(fig: plt.Figure, stem: Path, *, dpi: int = 300) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def render() -> None:
    setup_style()
    save_figure(create_composite(), PLOTS / "figure_brain_reattempt", dpi=320)
    PANELS.mkdir(parents=True, exist_ok=True)
    for stem, drawer in panel_drawers():
        fig, ax = plt.subplots(figsize=(4.1, 3.0))
        drawer(ax)
        fig.subplots_adjust(left=0.18, right=0.97, top=0.84, bottom=0.20)
        save_figure(fig, PANELS / stem, dpi=300)


if __name__ == "__main__":
    render()
