#!/usr/bin/env python3
"""Create extensible MGSM and L-CiteEval transfer figures.

Both inputs use long-form CSV files. Adding a model, language, context length,
or method requires adding rows rather than changing the plotting logic.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, OffsetImage, TextArea
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
MGSM_DATA = ROOT / "cache" / "mgsm_transfer_results.csv"
LCITE_DATA = ROOT / "cache" / "lciteeval_context_results.csv"
PDF_DIR = ROOT / "plots" / "pdf"
PNG_DIR = ROOT / "plots" / "figures"
QWEN_LOGO = ROOT.parent / "logos" / "qwen.png"
LLAMA_LOGO = ROOT.parent / "logos" / "llama.png"

METHOD_ORDER = ["H-infinity", "A-LQR", "S-PID", "Original"]
METHOD_LABEL = {
    "H-infinity": r"$H_\infty$ (ours)",
    "A-LQR": "A-LQR",
    "S-PID": "S-PID",
    "Original": "Original",
}
METHOD_MARKER = {
    "H-infinity": "D",
    "A-LQR": "s",
    "S-PID": "^",
    "Original": "o",
}
METHOD_COLOR = {
    "H-infinity": "#A51C30",
    "A-LQR": "#203A78",
    "S-PID": "#16794A",
    "Original": "#777D86",
}
METHOD_LINESTYLE = {
    "H-infinity": "-",
    "A-LQR": "--",
    "S-PID": "-.",
    "Original": ":",
}
MODEL_COLORS = [
    "#5B4CC4",
    "#0875D1",
    "#D97706",
    "#16865C",
    "#C13C72",
    "#4F6D7A",
    "#8A6D3B",
    "#C49A00",
]
LANGUAGE_ORDER = ["Chinese", "French", "Japanese", "Swahili", "Telugu"]

INK = "#20242A"
MUTED = "#68717D"
GRID = "#E2E6EA"
SPINE = "#AEB5BD"
QAA_COLOR = "#B3BBC5"
OURS_RING = "#D08A00"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11.2,
            "axes.labelsize": 15.8,
            "axes.titlesize": 16.4,
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


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric_columns = {
        "accuracy",
        "target_relevance",
        "instruction_relevance",
        "fluency",
        "context_k",
        "answer_recall",
        "citation_f1",
        "steering_quality",
    }
    parsed: list[dict[str, object]] = []
    for row in rows:
        clean: dict[str, object] = {}
        for key, value in row.items():
            clean[key] = float(value) if key in numeric_columns else value.strip()
        parsed.append(clean)
    return parsed


def model_color_map(models: list[str]) -> dict[str, str]:
    if len(models) > len(MODEL_COLORS):
        raise ValueError("Add more MODEL_COLORS before plotting additional models")
    return {model: MODEL_COLORS[index] for index, model in enumerate(models)}


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.75)
    ax.tick_params(labelsize=10.6, length=4.0, width=0.8)


def add_model_logo_legend(
    fig: plt.Figure,
    models: list[str],
    colors: dict[str, str],
    *,
    center_y: float,
    fontsize: float,
) -> None:
    """Draw company marks rather than generic dots in the model legend."""

    entries = []
    for model in models:
        normalized = model.lower()
        if normalized.startswith("qwen"):
            logo_path, zoom = QWEN_LOGO, 0.023
        elif normalized.startswith("llama"):
            logo_path, zoom = LLAMA_LOGO, 0.0135
        else:
            raise ValueError(f"No model logo registered for {model!r}")
        if not logo_path.exists():
            raise FileNotFoundError(f"Missing model logo: {logo_path}")
        entries.append(
            HPacker(
                children=[
                    OffsetImage(plt.imread(logo_path), zoom=zoom),
                    TextArea(
                        model,
                        textprops={
                            "fontsize": fontsize,
                            "fontweight": "bold",
                            "color": colors[model],
                        },
                    ),
                ],
                align="center",
                pad=0,
                sep=4,
            )
        )
    row = HPacker(children=entries, align="center", pad=0, sep=18)
    legend = AnchoredOffsetbox(
        loc="center",
        child=row,
        frameon=False,
        pad=0,
        borderpad=0,
        bbox_to_anchor=(0.5, center_y),
        bbox_transform=fig.transFigure,
    )
    fig.add_artist(legend)


def save_figure(fig: plt.Figure, stem: str) -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.045)
    fig.savefig(
        PNG_DIR / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.045,
    )
    plt.close(fig)


def mgsm_macro(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["method"]))].append(row)
    summary: dict[tuple[str, str], dict[str, float]] = {}
    metrics = ["accuracy", "target_relevance", "instruction_relevance", "fluency"]
    for key, values in grouped.items():
        summary[key] = {
            metric: float(np.mean([float(value[metric]) for value in values]))
            for metric in metrics
        }
    return summary


def quality_bottleneck(record: dict[str, float]) -> float:
    return min(
        record["target_relevance"],
        record["instruction_relevance"],
        record["fluency"],
    ) / 2.0


def method_handles() -> list[Line2D]:
    handles: list[Line2D] = []
    for method in METHOD_ORDER:
        handles.append(
            Line2D(
                [0],
                [0],
                marker=METHOD_MARKER[method],
                linestyle="none",
                markerfacecolor="#555B63",
                markeredgecolor=OURS_RING if method == "H-infinity" else "white",
                markeredgewidth=1.5 if method == "H-infinity" else 0.6,
                markersize=7.5 if method == "H-infinity" else 6.3,
                label=METHOD_LABEL[method],
            )
        )
    return handles


def draw_qaa_contours(ax: plt.Axes) -> None:
    x = np.linspace(0.045, 0.72, 500)
    for level in (5, 10, 15, 20):
        y = level / x
        visible = (y >= 0) & (y <= 45)
        ax.plot(
            x[visible],
            y[visible],
            color=QAA_COLOR,
            linewidth=0.9,
            linestyle=(0, (3.5, 2.8)),
            zorder=0.2,
        )
        indices = np.flatnonzero(visible)
        if indices.size:
            label_fraction = 0.32 if level == 20 else 0.78
            index = indices[int(label_fraction * (len(indices) - 1))]
            ax.text(
                x[index],
                y[index],
                f"QAA={level}",
                color="#78818C",
                fontsize=8.6,
                rotation=-18,
                ha="left",
                va="bottom",
                bbox={"fc": "white", "ec": "none", "pad": 0.3, "alpha": 0.82},
            )


def create_mgsm_figure(rows: list[dict[str, object]]) -> plt.Figure:
    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    colors = model_color_map(models)
    macro = mgsm_macro(rows)

    fig, (ax_pareto, ax_language) = plt.subplots(
        1,
        2,
        figsize=(10.8, 5.35),
        gridspec_kw={"width_ratios": [1.17, 1.0]},
    )
    style_axis(ax_pareto)
    style_axis(ax_language)

    draw_qaa_contours(ax_pareto)
    label_offsets = {
        "Original": (7, -2),
        "S-PID": (7, 7),
        "A-LQR": (8, -8),
        "H-infinity": (8, 8),
    }
    for model_index, model in enumerate(models):
        color = colors[model]
        for method in METHOD_ORDER:
            record = macro.get((model, method))
            if record is None:
                continue
            x = quality_bottleneck(record)
            y = record["accuracy"]
            marker = METHOD_MARKER[method]
            if method == "H-infinity":
                ax_pareto.scatter(
                    x,
                    y,
                    s=170,
                    marker=marker,
                    facecolor="none",
                    edgecolor=OURS_RING,
                    linewidth=2.0,
                    zorder=5,
                )
            ax_pareto.scatter(
                x,
                y,
                s=82 if method == "H-infinity" else 66,
                marker=marker,
                facecolor=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=6,
            )
            if len(models) == 1:
                dx, dy = label_offsets[method]
                label = METHOD_LABEL[method].replace(" (ours)", "")
                label = label.replace("$", "") if method != "H-infinity" else r"$H_\infty$"
                ax_pareto.annotate(
                    f"{label}  {y:.1f}",
                    (x, y),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=9.2,
                    fontweight="bold" if method == "H-infinity" else "normal",
                    color=INK,
                    ha="left",
                    va="center",
                )

    ax_pareto.set_xlim(-0.025, 0.73)
    ax_pareto.set_ylim(-2.5, 45.0)
    ax_pareto.set_xlabel(
        r"Transfer quality  $\min(T/2,\,IR/2,\,F/2)$  $\rightarrow$"
    )
    ax_pareto.set_ylabel("MGSM accuracy (%)  $\\rightarrow$")
    ax_pareto.set_title("A   Capability-steering trade-off", loc="left", fontweight="bold")

    # Per-language comparison. Method is encoded by marker/vertical lane and
    # foundation model by color, so additional models remain distinguishable.
    y_base = np.arange(len(LANGUAGE_ORDER))[::-1]
    compared_methods = ["H-infinity", "A-LQR", "S-PID"]
    method_offsets = {"H-infinity": 0.22, "A-LQR": 0.0, "S-PID": -0.22}
    model_offsets = np.linspace(-0.055, 0.055, max(len(models), 1))
    for model_index, model in enumerate(models):
        color = colors[model]
        model_rows = [row for row in rows if row["model"] == model]
        by_language_method = {
            (str(row["language"]), str(row["method"])): float(row["accuracy"])
            for row in model_rows
        }
        for language_index, language in enumerate(LANGUAGE_ORDER):
            for method in compared_methods:
                value = by_language_method.get((language, method))
                if value is None:
                    continue
                y = (
                    y_base[language_index]
                    + method_offsets[method]
                    + model_offsets[model_index]
                )
                if method == "H-infinity":
                    ax_language.scatter(
                        value,
                        y,
                        s=115,
                        marker=METHOD_MARKER[method],
                        facecolor="none",
                        edgecolor=OURS_RING,
                        linewidth=1.65,
                        zorder=4,
                    )
                ax_language.scatter(
                    value,
                    y,
                    s=62 if method == "H-infinity" else 52,
                    marker=METHOD_MARKER[method],
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=5,
                )
                if method == "H-infinity" and len(models) == 1:
                    ax_language.text(
                        value + 2.3,
                        y,
                        f"{value:.0f}",
                        ha="left",
                        va="center",
                        fontsize=9.0,
                        fontweight="bold",
                        color=INK,
                    )

    ax_language.set_xlim(-3, 78)
    ax_language.set_yticks(y_base)
    ax_language.set_yticklabels(LANGUAGE_ORDER)
    ax_language.set_xlabel("MGSM accuracy (%)  $\\rightarrow$")
    ax_language.set_title("B   Accuracy across transfer languages", loc="left", fontweight="bold")

    add_model_logo_legend(
        fig,
        models,
        colors,
        center_y=0.965,
        fontsize=15.5,
    )
    fig.legend(
        handles=method_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
        fontsize=15.3,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=1.05,
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.865, bottom=0.205, wspace=0.28)
    return fig


def context_metric_limits(metric: str, values: list[float]) -> tuple[float, float]:
    if metric == "steering_quality":
        return -0.012, max(0.20, max(values) + 0.035)
    padding = max(0.8, 0.16 * (max(values) - min(values)))
    return min(values) - padding, max(values) + padding


def create_lcite_figure(
    rows: list[dict[str, object]],
    context_subset: tuple[int, ...] | None = None,
) -> plt.Figure:
    if context_subset is not None:
        allowed_contexts = set(context_subset)
        rows = [
            row
            for row in rows
            if int(float(row["context_k"])) in allowed_contexts
        ]
        if not rows:
            raise ValueError("No L-CiteEval rows remain after context filtering")

    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    colors = model_color_map(models)
    compared_methods = ["H-infinity", "A-LQR", "S-PID"]
    metrics = [
        ("answer_recall", "A   Answer recall (%)"),
        ("citation_f1", "B   Citation F1 (%)"),
        ("steering_quality", "C   Overall steering (0-2)"),
    ]
    context_ticks = sorted({int(float(row["context_k"])) for row in rows})
    nrows = len(models)
    fig, axes = plt.subplots(
        nrows,
        3,
        figsize=(12.4, 4.75 + 2.95 * (nrows - 1)),
        squeeze=False,
        sharex=True,
    )
    # fig.suptitle(
    #     r"Robustness advantage emerges under context-length shift",
    #     x=0.505,
    #     y=1.0,
    #     fontsize=14.2,
    #     fontweight="bold",
    # )

    for model_index, model in enumerate(models):
        model_color = colors[model]
        model_rows = [row for row in rows if row["model"] == model]
        available_contexts = sorted({int(float(row["context_k"])) for row in model_rows})
        pending_contexts = [context for context in context_ticks if context not in available_contexts]
        for metric_index, (metric, title) in enumerate(metrics):
            ax = axes[model_index, metric_index]
            style_axis(ax)
            metric_values = [float(row[metric]) for row in model_rows]
            y_limits = context_metric_limits(metric, metric_values)
            if metric == "answer_recall":
                y_limits = (y_limits[0], y_limits[1] + 1.2)
            ax.set_ylim(*y_limits)
            ax.set_xscale("log", base=2)
            ax.set_xlim(
                7.0,
                38.0 if max(context_ticks) >= 32 else 19.0,
            )
            ax.set_xticks(context_ticks)
            context_labels = {
                8: "8K\nmatched",
                16: "16K\nlong-context",
                32: "32K\nlong-context",
            }
            ax.set_xticklabels(
                [context_labels.get(context, f"{context}K") for context in context_ticks]
            )
            if model_index == 0:
                ax.set_title(title, loc="left", fontweight="bold")
            if metric_index == 0:
                ax.text(
                    -0.24,
                    0.5,
                    model,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    fontweight="bold",
                    color=model_color,
                )

            # Context regimes are shown as background bands. The labels make
            # the matched-to-shifted comparison explicit without asserting a
            # residual magnitude that has not yet been plotted directly.
            ax.axvspan(7.0, 8.0 * np.sqrt(2.0), color="#F7F8FA", zorder=-1)
            ax.axvspan(
                8.0 * np.sqrt(2.0),
                16.0 * np.sqrt(2.0),
                color="#FFF5EE",
                zorder=-1,
            )
            for pending_context in pending_contexts:
                lower = pending_context / np.sqrt(2.0)
                upper = pending_context * np.sqrt(2.0)
                ax.axvspan(lower, upper, color="#EEF1F4", zorder=0)

            for method in METHOD_ORDER:
                method_rows = sorted(
                    [row for row in model_rows if row["method"] == method],
                    key=lambda row: float(row["context_k"]),
                )
                if not method_rows:
                    continue
                x = np.array([float(row["context_k"]) for row in method_rows])
                y = np.array([float(row[metric]) for row in method_rows])
                line_color = MUTED if method == "Original" else model_color
                ax.plot(
                    x,
                    y,
                    color=line_color,
                    linewidth=3.0 if method == "H-infinity" else 2.0,
                    linestyle=METHOD_LINESTYLE[method],
                    marker=METHOD_MARKER[method],
                    markersize=8.2 if method == "H-infinity" else 6.8,
                    markeredgecolor="white",
                    markeredgewidth=0.65,
                    zorder=4 if method == "H-infinity" else 3,
                )
                if method == "H-infinity":
                    ax.scatter(
                        x,
                        y,
                        s=105,
                        marker=METHOD_MARKER[method],
                        facecolor="none",
                        edgecolor=OURS_RING,
                        linewidth=1.6,
                        zorder=3.8,
                    )

            # Name the controller winner at each observed endpoint. This is
            # the visual thesis: nominal leaders at 8K are overtaken by the
            # robust controller after the context-length shift.
            controller_rows = [
                row for row in model_rows if row["method"] in compared_methods
            ]
            for context, descriptor in ((min(available_contexts), "leads"),
                                        (max(available_contexts), "best")):
                candidates = [
                    row for row in controller_rows
                    if int(float(row["context_k"])) == context
                ]
                if not candidates:
                    continue
                winner = max(candidates, key=lambda row: float(row[metric]))
                winner_method = str(winner["method"])
                winner_value = float(winner[metric])
                label = (
                    r"$H_\infty$ best"
                    if winner_method == "H-infinity"
                    else f"{METHOD_LABEL[winner_method]} {descriptor}"
                )
                is_short_context = context == min(available_contexts)
                if is_short_context and metric == "citation_f1":
                    annotation_offset = (48, -42)
                    annotation_va = "top"
                else:
                    annotation_offset = (5, 9)
                    annotation_va = "bottom"
                ax.annotate(
                    label,
                    (context, winner_value),
                    xytext=annotation_offset,
                    textcoords="offset points",
                    ha="left",
                    va=annotation_va,
                    fontsize=9.1,
                    fontweight="bold" if winner_method == "H-infinity" else "normal",
                    color=(
                        METHOD_COLOR["H-infinity"]
                        if winner_method == "H-infinity"
                        else MUTED
                    ),
                )

    line_handles = [
        Line2D(
            [0],
            [0],
            color="#555B63",
            linewidth=2.7 if method == "H-infinity" else 1.9,
            linestyle=METHOD_LINESTYLE[method],
            marker=METHOD_MARKER[method],
            markerfacecolor="#555B63",
            markeredgecolor=OURS_RING if method == "H-infinity" else "white",
            markeredgewidth=1.3 if method == "H-infinity" else 0.6,
            markersize=7.8,
            label=METHOD_LABEL[method],
        )
        for method in METHOD_ORDER
    ]
    fig.legend(
        handles=line_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.022),
        ncol=4,
        frameon=False,
        fontsize=15.5,
        handlelength=2.2,
        handletextpad=0.45,
        columnspacing=1.15,
    )
    add_model_logo_legend(
        fig,
        models,
        colors,
        center_y=0.925,
        fontsize=13.5,
    )
    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        top=0.82,
        bottom=0.275,
        wspace=0.27,
        hspace=0.30,
    )
    return fig


def main() -> None:
    configure_style()
    save_figure(create_mgsm_figure(read_csv(MGSM_DATA)), "mgsm-transfer-overview")
    lcite_rows = read_csv(LCITE_DATA)
    save_figure(
        create_lcite_figure(lcite_rows),
        "lciteeval-context-robustness",
    )
    save_figure(
        create_lcite_figure(lcite_rows, context_subset=(8, 16)),
        "lciteeval-context-robustness-drop-32k",
    )


if __name__ == "__main__":
    main()
