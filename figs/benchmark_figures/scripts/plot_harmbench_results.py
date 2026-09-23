#!/usr/bin/env python3
"""Create a publication-ready HarmBench robust-refusal figure."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, OffsetImage, TextArea
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "cache" / "harmbench_refusal_results.csv"
PDF_DIR = ROOT / "plots" / "pdf"
PNG_DIR = ROOT / "plots" / "figures"
LLAMA_LOGO = ROOT.parent / "logos" / "llama.png"

TEMPLATES = [
    "Direct",
    "John persona",
    "DNE nonresponse",
    "Jailbreak Bot",
    "YOJA/Nona roleplay",
    "APM programmer",
]
METHODS = ["H-infinity", "A-LQR", "Original"]
LABELS = {
    "H-infinity": r"$H_\infty$ (ours)",
    "A-LQR": "A-LQR",
    "Original": "Original",
}
MARKERS = {"H-infinity": "D", "A-LQR": "s", "Original": "o"}
COLORS = {
    "H-infinity": "#5B4CC4",
    "A-LQR": "#5B4CC4",
    "Original": "#777D86",
}
OURS_RING = "#D08A00"
INK = "#20242A"
MUTED = "#68717D"
GRID = "#E2E6EA"
SPINE = "#AEB5BD"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11.2,
            "axes.labelsize": 12.8,
            "axes.titlesize": 13.4,
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


def read_results() -> list[dict[str, object]]:
    with DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric = {
        "asr",
        "safe_concept_relevance",
        "instruction_relevance",
        "fluency",
        "overall_steering",
    }
    parsed: list[dict[str, object]] = []
    for row in rows:
        parsed.append(
            {
                key: float(value) if key in numeric else value.strip()
                for key, value in row.items()
            }
        )
    return parsed


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="x", color=GRID, linewidth=0.75, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)
    ax.tick_params(labelsize=10.7, length=4.0, width=0.8)


def add_llama_model_legend(
    fig: plt.Figure,
    models: list[str],
    colors: dict[str, str],
    *,
    center_y: float,
    fontsize: float,
) -> None:
    """Draw the official Meta mark for every Llama model legend entry."""

    if not LLAMA_LOGO.exists():
        raise FileNotFoundError(f"Missing model logo: {LLAMA_LOGO}")
    logo = plt.imread(LLAMA_LOGO)
    entries = [
        HPacker(
            children=[
                OffsetImage(logo, zoom=0.0135),
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
        for model in models
    ]
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


def create_figure(rows: list[dict[str, object]]) -> plt.Figure:
    by_template_method = {
        (str(row["template"]), str(row["method"])): float(row["asr"])
        for row in rows
    }
    model = str(rows[0]["model"])
    fig, (ax_raw, ax_frontier) = plt.subplots(
        1,
        2,
        figsize=(11.2, 5.8),
        gridspec_kw={"width_ratios": [1.30, 0.92]},
    )
    style_axis(ax_raw)
    style_axis(ax_frontier)

    ordered_templates = sorted(
        TEMPLATES,
        key=lambda template: min(
            by_template_method[(template, "Original")],
            by_template_method[(template, "A-LQR")],
        ),
        reverse=True,
    )
    display_names = {
        "Direct": "Direct",
        "YOJA/Nona roleplay": "YOJA/Nona\nroleplay",
        "John persona": "John\npersona",
        "DNE nonresponse": "DNE\nnonresponse",
        "Jailbreak Bot": "Jailbreak\nBot",
        "APM programmer": "APM\nprogrammer",
    }
    x = np.arange(len(ordered_templates))
    profiles = {
        method: np.array(
            [by_template_method[(template, method)] for template in ordered_templates]
        )
        for method in METHODS
    }
    strongest_profile = np.minimum(profiles["Original"], profiles["A-LQR"])
    raw_ymax = max(10.8, 1.10 * max(float(values.max()) for values in profiles.values()))

    ax_raw.grid(False)
    ax_raw.grid(True, axis="y", color=GRID, linewidth=0.75, zorder=0)
    ax_raw.fill_between(
        x,
        profiles["H-infinity"],
        strongest_profile,
        color="#E4BD6A",
        alpha=0.20,
        zorder=1,
    )
    ax_raw.plot(
        x,
        profiles["Original"],
        color=COLORS["Original"],
        linewidth=2.0,
        linestyle=":",
        marker=MARKERS["Original"],
        markersize=7.2,
        markeredgecolor="white",
        markeredgewidth=0.7,
        zorder=3,
    )
    ax_raw.plot(
        x,
        profiles["A-LQR"],
        color=COLORS["A-LQR"],
        linewidth=2.2,
        linestyle="--",
        marker=MARKERS["A-LQR"],
        markersize=7.0,
        markeredgecolor="white",
        markeredgewidth=0.7,
        zorder=4,
    )
    ax_raw.plot(
        x,
        profiles["H-infinity"],
        color=COLORS["H-infinity"],
        linewidth=3.0,
        linestyle="-",
        marker=MARKERS["H-infinity"],
        markersize=8.0,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=5,
    )
    ax_raw.scatter(
        x,
        profiles["H-infinity"],
        s=145,
        marker=MARKERS["H-infinity"],
        facecolor="none",
        edgecolor=OURS_RING,
        linewidth=1.9,
        zorder=4.8,
    )
    for index, value in enumerate(profiles["H-infinity"]):
        ax_raw.annotate(
            f"{value:.2f}",
            (index, value),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.9,
            fontweight="bold",
            color=INK,
        )
    ax_raw.text(
        2.65,
        max(profiles["H-infinity"]) + 0.08 * raw_ymax,
        rf"$H_\infty$ remains below {math.ceil(max(profiles['H-infinity']))}%",
        ha="center",
        va="center",
        fontsize=10.1,
        fontweight="bold",
        color="#A51C30",
        bbox={"fc": "white", "ec": "none", "pad": 0.45, "alpha": 0.88},
    )
    ax_raw.text(
        1.65,
        0.55 * raw_ymax,
        "robustness margin",
        ha="center",
        va="center",
        fontsize=9.1,
        color="#8A6724",
        rotation=-7,
    )
    ax_raw.set_xlim(-0.35, len(ordered_templates) - 0.65)
    ax_raw.set_ylim(0, raw_ymax)
    ax_raw.set_xticks(x)
    ax_raw.set_xticklabels([display_names[template] for template in ordered_templates])
    ax_raw.set_ylabel(r"Attack success rate (%)  $\downarrow$")
    ax_raw.set_xlabel("Jailbreak templates ordered by baseline difficulty")
    ax_raw.set_title("A   Robustness profile across jailbreaks", loc="left", fontweight="bold")

    summary: dict[str, dict[str, float]] = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        summary[method] = {
            "asr": float(np.mean([float(row["asr"]) for row in method_rows])),
            "safe": float(
                np.mean([float(row["safe_concept_relevance"]) for row in method_rows])
            ),
        }

    h_mean = summary["H-infinity"]["asr"]
    baseline_mean = min(summary["Original"]["asr"], summary["A-LQR"]["asr"])
    aggregate_reduction = 100.0 * (baseline_mean - h_mean) / baseline_mean
    frontier_xmax = max(7.2, 1.18 * max(point["asr"] for point in summary.values()))
    frontier_ymin = min(1.945, min(point["safe"] for point in summary.values()) - 0.015)
    frontier_ymax = max(1.985, max(point["safe"] for point in summary.values()) + 0.008)

    ax_frontier.add_patch(
        Rectangle(
            (0, 1.97),
            frontier_xmax,
            frontier_ymax - 1.97,
            facecolor="#FBF5E9",
            edgecolor="none",
            alpha=0.85,
            zorder=0,
        )
    )
    ax_frontier.text(
        0.04 * frontier_xmax,
        frontier_ymax - 0.002,
        "safer and better aligned",
        ha="left",
        va="top",
        fontsize=9.3,
        color="#8A6724",
        fontweight="bold",
    )

    for source_method, arc in (("Original", 0.12), ("A-LQR", -0.12)):
        source = summary[source_method]
        target = summary["H-infinity"]
        ax_frontier.add_patch(
            FancyArrowPatch(
                (source["asr"], source["safe"]),
                (target["asr"] + 0.13, target["safe"] - 0.0005),
                arrowstyle="-|>",
                mutation_scale=12,
                connectionstyle=f"arc3,rad={arc}",
                linewidth=1.25,
                linestyle=(0, (3, 2.2)),
                color="#B4BAC2",
                zorder=2,
            )
        )

    frontier_offsets = {
        "Original": (8, -12),
        "A-LQR": (8, 10),
        "H-infinity": (9, 9),
    }
    for method in METHODS:
        point = summary[method]
        if method == "H-infinity":
            ax_frontier.scatter(
                point["asr"],
                point["safe"],
                s=250,
                marker=MARKERS[method],
                facecolor="none",
                edgecolor=OURS_RING,
                linewidth=2.1,
                zorder=5,
            )
        ax_frontier.scatter(
            point["asr"],
            point["safe"],
            s=120 if method == "H-infinity" else 92,
            marker=MARKERS[method],
            facecolor=COLORS[method],
            edgecolor="white",
            linewidth=0.8,
            zorder=6,
        )
        dx, dy = frontier_offsets[method]
        point_label = LABELS[method].replace(" (ours)", "")
        if method == "H-infinity":
            point_label = r"$H_\infty$"
        ax_frontier.annotate(
            f"{point_label}  {point['asr']:.2f}%",
            (point["asr"], point["safe"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold" if method == "H-infinity" else "normal",
            color=INK,
        )

    ax_frontier.annotate(
        f"{aggregate_reduction:.0f}% lower mean ASR",
        xy=(summary["H-infinity"]["asr"], summary["H-infinity"]["safe"]),
        xytext=(0.46 * frontier_xmax, frontier_ymax - 0.008),
        textcoords="data",
        arrowprops={
            "arrowstyle": "->",
            "color": "#A51C30",
            "linewidth": 1.4,
            "connectionstyle": "arc3,rad=-0.16",
        },
        fontsize=10.0,
        fontweight="bold",
        color="#A51C30",
        ha="left",
        va="center",
    )
    ax_frontier.set_xlim(0, frontier_xmax)
    ax_frontier.set_ylim(frontier_ymin, frontier_ymax)
    ax_frontier.set_xlabel(r"Mean attack success rate (%)   lower is better $\leftarrow$")
    ax_frontier.set_ylabel("Mean safe-concept relevance (0-2)  $\\rightarrow$")
    ax_frontier.set_title("B   Safety Pareto frontier", loc="left", fontweight="bold")

    add_llama_model_legend(
        fig,
        [model],
        {model: "#5B4CC4"},
        center_y=0.965,
        fontsize=11.0,
    )
    method_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[method],
            linestyle="none",
            markerfacecolor="#555B63",
            markeredgecolor=OURS_RING if method == "H-infinity" else "white",
            markeredgewidth=1.5 if method == "H-infinity" else 0.7,
            markersize=8.0 if method == "H-infinity" else 7.0,
            label=LABELS[method],
        )
        for method in METHODS
    ]
    fig.legend(
        handles=method_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        frameon=False,
        fontsize=10.5,
        handletextpad=0.4,
        columnspacing=1.25,
    )
    fig.subplots_adjust(
        left=0.155,
        right=0.985,
        top=0.865,
        bottom=0.255,
        wspace=0.28,
    )
    return fig


def create_combined_figure(rows: list[dict[str, object]]) -> plt.Figure:
    """Combine all models in a clean two-panel summary."""

    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    model_colors = {
        model: color
        for model, color in zip(models, ("#5B4CC4", "#1976C9", "#D97706"))
    }
    indexed = {
        (str(row["model"]), str(row["template"]), str(row["method"])): row
        for row in rows
    }

    baseline_difficulty = {
        template: float(
            np.mean(
                [
                    min(
                        float(indexed[(model, template, "Original")]["asr"]),
                        float(indexed[(model, template, "A-LQR")]["asr"]),
                    )
                    for model in models
                ]
            )
        )
        for template in TEMPLATES
    }
    ordered_templates = sorted(
        TEMPLATES,
        key=lambda template: baseline_difficulty[template],
        reverse=True,
    )
    display_names = {
        "Direct": "Direct",
        "YOJA/Nona roleplay": "YOJA/Nona\nroleplay",
        "John persona": "John\npersona",
        "DNE nonresponse": "DNE\nnonresponse",
        "Jailbreak Bot": "Jailbreak\nBot",
        "APM programmer": "APM\nprogrammer",
    }
    x = np.arange(len(ordered_templates))

    summaries: dict[str, dict[str, dict[str, float]]] = {}
    displayed_profile_values: list[float] = []
    for model in models:
        summaries[model] = {}
        for template in ordered_templates:
            displayed_profile_values.extend(
                [
                    float(indexed[(model, template, "H-infinity")]["asr"]),
                    min(
                        float(indexed[(model, template, "Original")]["asr"]),
                        float(indexed[(model, template, "A-LQR")]["asr"]),
                    ),
                ]
            )
        for method in METHODS:
            method_rows = [
                indexed[(model, template, method)] for template in TEMPLATES
            ]
            summaries[model][method] = {
                "asr": float(np.mean([float(row["asr"]) for row in method_rows])),
                "safe": float(
                    np.mean(
                        [float(row["safe_concept_relevance"]) for row in method_rows]
                    )
                ),
            }

    fig, (ax_raw, ax_frontier) = plt.subplots(
        1,
        2,
        figsize=(12.4, 6.0),
        gridspec_kw={"width_ratios": [1.26, 0.94]},
    )
    style_axis(ax_raw)
    style_axis(ax_frontier)
    ax_raw.grid(False)
    ax_raw.grid(True, axis="y", color=GRID, linewidth=0.75, zorder=0)

    for model in models:
        color = model_colors[model]
        h_profile = np.array(
            [
                float(indexed[(model, template, "H-infinity")]["asr"])
                for template in ordered_templates
            ]
        )
        best_baseline = np.array(
            [
                min(
                    float(indexed[(model, template, "Original")]["asr"]),
                    float(indexed[(model, template, "A-LQR")]["asr"]),
                )
                for template in ordered_templates
            ]
        )
        ax_raw.plot(
            x,
            best_baseline,
            color=color,
            linewidth=1.7,
            linestyle=(0, (3, 2.2)),
            alpha=0.58,
            zorder=2,
        )
        ax_raw.plot(
            x,
            h_profile,
            color=color,
            linewidth=2.8,
            linestyle="-",
            marker="D",
            markersize=7.2,
            markerfacecolor=color,
            markeredgecolor=OURS_RING,
            markeredgewidth=1.45,
            zorder=4,
        )

    ax_raw.set_xlim(-0.35, len(ordered_templates) - 0.65)
    ax_raw.set_ylim(0, 1.08 * max(displayed_profile_values))
    ax_raw.set_xticks(x)
    ax_raw.set_xticklabels(
        [display_names[template] for template in ordered_templates],
        fontsize=9.2,
    )
    ax_raw.set_ylabel(r"Attack success rate (%)  $\downarrow$")
    ax_raw.set_xlabel("Jailbreak template")
    ax_raw.set_title(
        "A   Robustness profile across jailbreaks",
        loc="left",
        fontweight="bold",
    )
    ax_raw.text(
        0.985,
        0.965,
        r"solid: $H_\infty$     dashed: min ASR(Original, A-LQR)",
        transform=ax_raw.transAxes,
        ha="right",
        va="top",
        fontsize=9.2,
        color=MUTED,
    )

    all_safe = [float(row["safe_concept_relevance"]) for row in rows]
    safe_ymin = min(all_safe) - 0.012
    safe_ymax = max(1.985, max(all_safe) + 0.006)
    frontier_xmax = 1.15 * max(
        point["asr"]
        for model_summary in summaries.values()
        for point in model_summary.values()
    )
    ax_frontier.add_patch(
        Rectangle(
            (0, 1.97),
            frontier_xmax,
            safe_ymax - 1.97,
            facecolor="#FBF5E9",
            edgecolor="none",
            alpha=0.80,
            zorder=0,
        )
    )
    ax_frontier.text(
        0.97,
        0.98,
        "safer and better aligned",
        transform=ax_frontier.transAxes,
        ha="right",
        va="top",
        fontsize=9.3,
        color="#8A6724",
        fontweight="bold",
    )

    reduction_offsets = {
        models[0]: (10, -16),
        models[1]: (10, -15),
        models[2]: (10, -12),
    }
    for model in models:
        color = model_colors[model]
        summary = summaries[model]
        target = summary["H-infinity"]
        for source_method in ("Original", "A-LQR"):
            source = summary[source_method]
            ax_frontier.plot(
                [source["asr"], target["asr"]],
                [source["safe"], target["safe"]],
                color=color,
                linewidth=1.1,
                linestyle=(0, (3, 2.2)),
                alpha=0.30,
                zorder=2,
            )
        for method in METHODS:
            point = summary[method]
            ax_frontier.scatter(
                point["asr"],
                point["safe"],
                s=112 if method == "H-infinity" else 76,
                marker=MARKERS[method],
                facecolor=color,
                edgecolor=OURS_RING if method == "H-infinity" else "white",
                linewidth=1.65 if method == "H-infinity" else 0.7,
                zorder=5 if method == "H-infinity" else 4,
            )
        baseline_mean = min(summary["Original"]["asr"], summary["A-LQR"]["asr"])
        reduction = 100.0 * (baseline_mean - target["asr"]) / baseline_mean
        ax_frontier.annotate(
            rf"$\downarrow${reduction:.0f}%",
            (target["asr"], target["safe"]),
            xytext=reduction_offsets[model],
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color=color,
        )

    ax_frontier.set_xlim(0, frontier_xmax)
    ax_frontier.set_ylim(safe_ymin, safe_ymax)
    ax_frontier.set_xlabel(
        r"Mean attack success rate (%)   lower is better $\leftarrow$"
    )
    ax_frontier.set_ylabel("Mean safe-concept relevance (0-2)  $\\rightarrow$")
    ax_frontier.set_title(
        "B   Safety Pareto frontier",
        loc="left",
        fontweight="bold",
    )

    method_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[method],
            linestyle="none",
            markerfacecolor="#555B63",
            markeredgecolor=OURS_RING if method == "H-infinity" else "white",
            markeredgewidth=1.5 if method == "H-infinity" else 0.7,
            markersize=8.0 if method == "H-infinity" else 7.0,
            label=LABELS[method],
        )
        for method in METHODS
    ]
    add_llama_model_legend(
        fig,
        models,
        model_colors,
        center_y=0.955,
        fontsize=10.8,
    )
    fig.legend(
        handles=method_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        frameon=False,
        fontsize=10.8,
        handletextpad=0.4,
        columnspacing=1.25,
    )
    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        top=0.85,
        bottom=0.22,
        wspace=0.28,
    )
    return fig


def main() -> None:
    configure_style()
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_results()
    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        slug = model.lower().replace("-instruct", "").replace(".", "_").replace("-", "_")
        stem = f"harmbench-robust-refusal-{slug}"
        fig = create_figure(model_rows)
        fig.savefig(PDF_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05)
        fig.savefig(
            PNG_DIR / f"{stem}.png",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.close(fig)

    combined = create_combined_figure(rows)
    combined_stem = "harmbench-robust-refusal-combined"
    combined.savefig(
        PDF_DIR / f"{combined_stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    combined.savefig(
        PNG_DIR / f"{combined_stem}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(combined)


if __name__ == "__main__":
    main()
