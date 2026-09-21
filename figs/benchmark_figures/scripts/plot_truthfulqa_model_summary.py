#!/usr/bin/env python3
"""Create expandable ID and OOD model-summary figures for TruthfulQA.

Each model occupies one row. All valid steering methods are shown as markers,
while H-infinity is emphasized and compared with the strongest non-H-infinity
baseline. Add future results by appending rows to
``cache/truthfulqa_figure_data.csv``; no plotting-code changes are required.

The plotted quantity is the change in quality-adjusted truthfulness (QAT)
relative to the unsteered Original model:

    QAT = True(%) * min(instruction_relevance / 2, fluency / 2).

ODESteer is intentionally excluded because its implementation is invalid.
QAT uncertainty is not shown because per-example paired estimates are not yet
available.
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


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
    "H-infinity": r"$H_\infty$",
    "A-LQR": "A-LQR",
    "S-PID": "S-PID",
    "ITI": "ITI",
    "ActAdd": "ActAdd",
    "Mean-AcT": "Mean-AcT",
    "Linear-AcT": "Linear-AcT",
    "PID-AcT": "PID-AcT",
}

COLORS = {
    "H-infinity": "#A61B1B",
    "A-LQR": "#243B8F",
    "S-PID": "#17733A",
    "other": "#8B9098",
}

MARKERS = {
    "H-infinity": "D",
    "A-LQR": "s",
    "S-PID": "^",
    "ITI": "P",
    "ActAdd": "X",
    "Mean-AcT": "v",
    "Linear-AcT": "<",
    "PID-AcT": ">",
}


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
            "axes.labelsize": 8.6,
            "axes.titleweight": "semibold",
            "axes.labelcolor": "#303338",
            "text.color": "#222428",
            "xtick.color": "#555A62",
            "ytick.color": "#303338",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def load_data(path: Path) -> list[Record]:
    records: list[Record] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "split", "model", "method", "true", "true_se", "relevance", "fluency"
        }
        if set(reader.fieldnames or []) != required:
            raise ValueError(f"CSV columns must be exactly {sorted(required)}")
        for row in reader:
            if row["method"] == "ODESteer":
                continue
            records.append(
                Record(
                    split=row["split"].strip().upper(),
                    model=row["model"].strip(),
                    method=row["method"].strip(),
                    true=float(row["true"]),
                    true_se=float(row["true_se"]),
                    relevance=float(row["relevance"]),
                    fluency=float(row["fluency"]),
                )
            )
    return records


def group_models(records: list[Record], split: str) -> OrderedDict[str, list[Record]]:
    models: OrderedDict[str, list[Record]] = OrderedDict()
    for record in records:
        if record.split != split:
            continue
        models.setdefault(record.model, []).append(record)

    for model, rows in models.items():
        methods = {row.method for row in rows}
        if "Original" not in methods:
            raise ValueError(f"{split}/{model} is missing Original")
        unknown = methods - set(METHOD_ORDER) - {"Original"}
        if unknown:
            raise ValueError(f"Unknown methods in {split}/{model}: {sorted(unknown)}")
    return models


def method_style(method: str) -> tuple[str, str, float, int]:
    color = COLORS.get(method, COLORS["other"])
    marker = MARKERS[method]
    if method == "H-infinity":
        return color, marker, 56.0, 6
    if method == "A-LQR":
        return color, marker, 42.0, 5
    if method == "S-PID":
        return color, marker, 43.0, 4
    return color, marker, 31.0, 3


def compute_model_deltas(rows: list[Record]) -> dict[str, float]:
    baseline = next(row.qat for row in rows if row.method == "Original")
    return {row.method: row.qat - baseline for row in rows if row.method != "Original"}


def x_limits(all_deltas: list[float]) -> tuple[float, float]:
    values = np.array([0.0, *all_deltas], dtype=float)
    span = max(float(np.ptp(values)), 1.0)
    lower = float(values.min() - 0.13 * span)
    upper = float(values.max() + 0.20 * span)
    step = 1.0 if span < 12 else 2.0
    return np.floor(lower / step) * step, np.ceil(upper / step) * step


def create_figure(
    split: str,
    models: OrderedDict[str, list[Record]],
    output_pdf: Path,
    output_png: Path,
) -> None:
    if not models:
        raise ValueError(f"No complete models available for {split}")

    model_deltas = {model: compute_model_deltas(rows) for model, rows in models.items()}
    all_deltas = [value for deltas in model_deltas.values() for value in deltas.values()]
    xmin, xmax = x_limits(all_deltas)

    model_names = list(models)
    n_models = len(model_names)
    figure_height = max(3.25, 1.75 + 0.68 * n_models)
    fig, ax = plt.subplots(figsize=(7.25, figure_height))

    y_centers = np.arange(n_models - 1, -1, -1, dtype=float)
    jitter = dict(zip(METHOD_ORDER, np.linspace(-0.22, 0.22, len(METHOD_ORDER)), strict=True))

    for row_index, (model, y_center) in enumerate(zip(model_names, y_centers, strict=True)):
        if row_index % 2 == 0:
            ax.axhspan(y_center - 0.39, y_center + 0.39, color="#F5F6F7", zorder=0)

        deltas = model_deltas[model]
        for method in METHOD_ORDER:
            if method not in deltas:
                continue
            color, marker, size, zorder = method_style(method)
            value = deltas[method]
            ax.scatter(
                value,
                y_center + jitter[method],
                s=size,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.55,
                zorder=zorder,
            )

        if "H-infinity" not in deltas:
            continue
        hinf_value = deltas["H-infinity"]
        baseline_candidates = {
            method: value for method, value in deltas.items() if method != "H-infinity"
        }
        if baseline_candidates:
            best_method, best_value = max(baseline_candidates.items(), key=lambda item: item[1])
            callout_y = y_center + 0.32
            ax.plot(
                [best_value, hinf_value],
                [callout_y, callout_y],
                color=COLORS["H-infinity"],
                linewidth=1.45,
                solid_capstyle="round",
                zorder=2,
            )
            ax.scatter(
                [best_value, hinf_value],
                [callout_y, callout_y],
                s=10,
                color=COLORS["H-infinity"],
                zorder=2.2,
            )
            margin = hinf_value - best_value
            ax.annotate(
                f"{margin:+.2f} vs {DISPLAY_NAMES[best_method]}",
                xy=((best_value + hinf_value) / 2.0, callout_y),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.1,
                fontweight="semibold",
                color=COLORS["H-infinity"],
            )

        ax.annotate(
            f"{hinf_value:+.2f}",
            xy=(hinf_value, y_center + jitter["H-infinity"]),
            xytext=(7, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.6,
            fontweight="bold",
            color=COLORS["H-infinity"],
            zorder=8,
        )

    ax.axvline(0.0, color="#303338", linewidth=1.05, zorder=1)
    ax.text(
        0.0,
        y_centers.max() + 0.50,
        "Original",
        ha="center",
        va="bottom",
        fontsize=7.1,
        color="#555A62",
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.52, y_centers.max() + 0.60)
    ax.set_yticks(y_centers, model_names)
    ax.tick_params(axis="y", labelsize=8.5, length=0, pad=8)
    ax.tick_params(axis="x", labelsize=7.4, length=3, width=0.6)
    ax.grid(axis="x", color="#D9DCE1", linewidth=0.65, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#AEB3BA")
    ax.spines["bottom"].set_linewidth(0.65)
    ax.set_xlabel(r"Change in quality-adjusted truthfulness from Original, $\Delta$QAT $\rightarrow$")

    if split == "ID":
        title = "In-distribution TruthfulQA"
    else:
        title = "Spanish-input, English-output TruthfulQA (OOD)"
    ax.set_title(title, loc="left", pad=18, fontsize=11.5)
    ax.text(
        0.0,
        1.02,
        r"QAT = True(\%) $\times$ min(IR/2, Fluency/2); higher is better",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="#666B73",
    )

    legend_handles = []
    for method in METHOD_ORDER:
        color, marker, _, _ = method_style(method)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.55,
                markersize=5.6,
                label=DISPLAY_NAMES[method],
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.055),
        ncol=4,
        frameon=False,
        fontsize=7.2,
        handletextpad=0.45,
        columnspacing=1.15,
    )
    fig.text(
        0.5,
        0.012,
        "Values are computed from reported means. ODESteer is excluded; QAT uncertainty requires paired bootstrap estimates.",
        ha="center",
        va="bottom",
        fontsize=6.6,
        color="#696E76",
    )
    fig.subplots_adjust(left=0.18, right=0.975, top=0.82, bottom=0.28)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_png, format="png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root / "cache/truthfulqa_figure_data.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "plots",
    )
    return parser.parse_args()


def main() -> None:
    configure_style()
    args = parse_args()
    records = load_data(args.data)

    outputs = {
        "ID": (
            args.output_root / "pdf/truthfulqa-id-model-summary.pdf",
            args.output_root / "figures/truthfulqa-id-model-summary.png",
        ),
        "OOD": (
            args.output_root / "pdf/truthfulqa-ood-model-summary.pdf",
            args.output_root / "figures/truthfulqa-ood-model-summary.png",
        ),
    }

    for split, (pdf_path, png_path) in outputs.items():
        models = group_models(records, split)
        create_figure(split, models, pdf_path, png_path)
        print(f"{split} PDF: {pdf_path}")
        print(f"{split} PNG: {png_path}")


if __name__ == "__main__":
    main()
