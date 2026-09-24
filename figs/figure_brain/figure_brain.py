#!/usr/bin/env python3
"""Recreate the recoverable brain/LLM linearization panels as separate figures."""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[1]
SOURCE = REPO / "NeuroAnalysis/results"
SOURCE_FIGURES = REPO / "NeuroAnalysis/figures"
PLOTS = UNIT / "plots"

INK = "#25282D"
MUTED = "#687078"
GRID = "#E2E5E8"
SPINE = "#AEB4BA"
ID_COLOR = "#7895B4"
OOD_COLOR = "#B77967"
LINEAR_COLOR = "#2858A5"

AREA_ORDER = (
    "dorsolateral prefrontal",
    "ventrolateral prefrontal",
    "premotor / dorsomedial frontal",
    "sensorimotor",
    "temporal / peri-insular",
    "posterior temporal",
    "parieto-occipital",
)
AREA_COLORS = dict(
    zip(
        AREA_ORDER,
        ("#1F4EA1", "#5B8BD0", "#2E9C5C", "#8BC34A", "#E0A13C", "#C1272D", "#7B4FA6"),
        strict=True,
    )
)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 10.5,
            "axes.labelsize": 12.0,
            "axes.titlesize": 12.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)
    ax.tick_params(length=3.5, width=0.8)


def save(fig: plt.Figure, stem: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(PLOTS / f"{stem}.png", dpi=450, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def stage_existing_electrode_map() -> None:
    """Keep the surviving source export visible inside this analysis unit.

    The raw neural caches needed to recompute this anatomy panel are absent, so
    this is deliberately a byte-for-byte staging operation rather than a
    misleading reconstruction.
    """
    PLOTS.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        shutil.copy2(
            SOURCE_FIGURES / f"electrode_map_P12.{suffix}",
            PLOTS / f"brain_electrode_map_existing_export.{suffix}",
        )


def plot_error_shrinkage(
    data: pd.DataFrame,
    *,
    condition: str,
    stem: str,
    title: str,
    horizon_column: str,
    ood_key: str,
    xlabel: str,
) -> None:
    if "axis" in data.columns:
        selected = data[data["axis"] == condition]
        ood_name = "OOD"
    else:
        selected = data[data["split"].isin(["ID", ood_key])]
        ood_name = ood_key
    table = selected.pivot(index=horizon_column, columns="split", values="reduction").sort_index()
    x = np.arange(len(table), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(4.25, 3.15))
    ax.bar(x - width / 2, table["ID"], width, color=ID_COLOR, label="ID", zorder=3)
    ax.bar(x + width / 2, table[ood_name], width, color=OOD_COLOR, label="OOD", zorder=3)
    ax.axhline(0, color=INK, linewidth=1.2, zorder=4)
    ax.set_xticks(x, [str(value) for value in table.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Residual-RMS reduction (z)")
    ax.set_title(title, loc="left", fontweight="bold", pad=8)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    style_axis(ax)
    save(fig, stem)


def plot_brain_state_across_leads(data: pd.DataFrame) -> None:
    leads = data[data["area"].notna()].copy()
    fig, ax = plt.subplots(figsize=(4.4, 3.25))
    for _, row in leads.iterrows():
        ax.scatter(
            row["n_ch"],
            row["med_resid"],
            s=105,
            color=AREA_COLORS[str(row["area"])],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.annotate(
            str(row.iloc[0]),
            (row["n_ch"], row["med_resid"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8.5,
        )
    ax.set_xlabel("Channels in lead")
    ax.set_ylabel("Median residual RMS (z)")
    style_axis(ax, grid_axis="both")
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=color, label=area, markersize=6)
        for area, color in AREA_COLORS.items()
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=6.5,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.22),
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.subplots_adjust(bottom=0.34)
    save(fig, "brain_state_size_across_leads")


def plot_brain_state_lvf_subsets(data: pd.DataFrame) -> None:
    subsets = data[data["mean"].notna()].copy()
    k = pd.to_numeric(subsets.iloc[:, 0])
    yerr = subsets["std"].fillna(0.0)
    fig, ax = plt.subplots(figsize=(4.25, 3.15))
    ax.errorbar(
        k,
        subsets["mean"],
        yerr=yerr,
        fmt="o-",
        color=LINEAR_COLOR,
        linewidth=2.3,
        markersize=7,
        capsize=5,
        capthick=1.5,
        zorder=3,
    )
    ax.set_xlabel(r"State dimension $k$ (LVF)")
    ax.set_ylabel("Median residual RMS (z)")
    style_axis(ax, grid_axis="both")
    save(fig, "brain_state_size_lvf_subsets")


def plot_llm_state_across_layers(data: pd.DataFrame) -> None:
    layers = data[data["med_resid"].notna()].copy()
    layer = pd.to_numeric(layers.iloc[:, 0])
    colors = mpl.colormaps["turbo"](np.linspace(0.05, 0.95, len(layers)))
    fig, ax = plt.subplots(figsize=(4.25, 3.15))
    for x, y, color in zip(layer, layers["med_resid"], colors, strict=True):
        ax.scatter(x, y, s=105, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        ax.annotate(f"L{int(x)}", (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8.5)
    ax.set_xlabel("Layer (fixed 128 dimensions)")
    ax.set_ylabel("Median residual RMS (z)")
    ax.set_xlim(0, 13.5)
    style_axis(ax, grid_axis="both")
    save(fig, "llm_state_size_across_layers")


def plot_llm_state_hidden_dimensions(data: pd.DataFrame) -> None:
    subsets = data[data["mean"].notna()].copy()
    k = pd.to_numeric(subsets.iloc[:, 0])
    yerr = subsets["std"].fillna(0.0)
    fig, ax = plt.subplots(figsize=(4.25, 3.15))
    ax.errorbar(
        k,
        subsets["mean"],
        yerr=yerr,
        fmt="o-",
        color=LINEAR_COLOR,
        linewidth=2.3,
        markersize=7,
        capsize=5,
        capthick=1.5,
        zorder=3,
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"State dimension $k$ (layer 6)")
    ax.set_ylabel("Median residual RMS (z)")
    style_axis(ax, grid_axis="both")
    save(fig, "llm_state_size_hidden_dimensions")


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=12,
            linewidth=1.2,
            color=MUTED,
            connectionstyle="arc3,rad=0.05",
        )
    )


def plot_information_flow_schematic() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.set_axis_off()
    nodes = {
        "PPC": ((0.08, 0.76), "#7B4FA6"),
        "TP": ((0.08, 0.34), "#E0A13C"),
        "VLPFC": ((0.28, 0.52), "#5B8BD0"),
        "DLPFC": ((0.45, 0.72), "#1F4EA1"),
        "PT": ((0.56, 0.30), "#C1272D"),
        "SMA/PreM": ((0.68, 0.60), "#2E9C5C"),
        "M1/S1": ((0.88, 0.60), "#8BC34A"),
    }
    edges = (
        ("PPC", "DLPFC"),
        ("TP", "VLPFC"),
        ("VLPFC", "DLPFC"),
        ("DLPFC", "SMA/PreM"),
        ("PT", "SMA/PreM"),
        ("SMA/PreM", "M1/S1"),
    )
    for source, target in edges:
        arrow(ax, nodes[source][0], nodes[target][0])
    for label, (position, color) in nodes.items():
        ax.scatter(*position, s=260, color=color, edgecolor="white", linewidth=1.0, zorder=3)
        ax.text(position[0], position[1] + 0.075, label, ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.text(
        0.50,
        0.095,
        r"$x_{t+\Delta} = A_\Delta x_t + b_\Delta + g_\Delta(x_t)$",
        ha="center",
        va="center",
        fontsize=14,
    )
    ax.text(0.43, 0.015, "local linear dynamics", ha="center", va="center", fontsize=8.5, color=MUTED)
    ax.text(0.68, 0.015, "nonlinear residual", ha="center", va="center", fontsize=8.5, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.04, 0.95)
    save(fig, "brain_information_flow_schematic")


def render() -> None:
    setup_style()
    stage_existing_electrode_map()
    brain_error = pd.read_csv(SOURCE / "final_panelD_error_shrinkage_LVF_theta.csv")
    brain_state = pd.read_csv(SOURCE / "final_panelF_state_size_theta.csv")
    llm_error = pd.read_csv(SOURCE / "final_llmfit_panelD_error_shrinkage.csv")
    llm_state = pd.read_csv(SOURCE / "final_llmfit_panelF_state_size.csv")

    plot_error_shrinkage(
        brain_error,
        condition="conflict",
        stem="brain_error_shrinkage_conflict",
        title="Conflict shift (low → high)",
        horizon_column="horizon_ms",
        ood_key="OOD",
        xlabel=r"Horizon $\Delta$ (ms)",
    )
    plot_error_shrinkage(
        brain_error,
        condition="stim-context",
        stem="brain_error_shrinkage_stim_context",
        title="Stimulation context (NS1 → NS2)",
        horizon_column="horizon_ms",
        ood_key="OOD",
        xlabel=r"Horizon $\Delta$ (ms)",
    )
    plot_brain_state_across_leads(brain_state)
    plot_brain_state_lvf_subsets(brain_state)

    plot_error_shrinkage(
        llm_error,
        condition="spanish",
        stem="llm_error_shrinkage_spanish",
        title="Spanish shift",
        horizon_column="horizon_tok",
        ood_key="OOD-es",
        xlabel=r"Horizon $\Delta$ (tokens)",
    )
    plot_error_shrinkage(
        llm_error,
        condition="long-context",
        stem="llm_error_shrinkage_long_context",
        title="Long-context shift",
        horizon_column="horizon_tok",
        ood_key="OOD-ctx",
        xlabel=r"Horizon $\Delta$ (tokens)",
    )
    plot_llm_state_across_layers(llm_state)
    plot_llm_state_hidden_dimensions(llm_state)
    plot_information_flow_schematic()


if __name__ == "__main__":
    render()
