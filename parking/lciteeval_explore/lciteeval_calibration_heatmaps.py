"""Render the four behavioral L-CiteEval H-infinity calibration heatmaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


UNIT = Path(__file__).resolve().parent
DEFAULT_SELECTION = UNIT / "cache/selection.json"
PLOTS = UNIT / "plots"

PANELS = (
    ("Concept relevance", "axbench_concept_relevance", 2),
    ("Instruction relevance", "axbench_instruction_relevance", 2),
    ("Fluency", "axbench_fluency", 2),
    ("Overall steering", "axbench_overall", 3),
)


def _label(value: float) -> str:
    return f"{value:.3g}"


def render(selection_path: Path) -> None:
    selection = json.loads(selection_path.read_text())
    grid = selection["grid"]
    q_values = sorted({float(row["q_over_r"]) for row in grid})
    q_final_values = sorted({float(row["q_final_over_r"]) for row in grid})
    q_index = {value: index for index, value in enumerate(q_values)}
    q_final_index = {
        value: index for index, value in enumerate(q_final_values)
    }
    shape = (len(q_values), len(q_final_values))

    selected = selection["selected"]
    selected_row = q_index[float(selected["q_over_r"])]
    selected_column = q_final_index[float(selected["q_final_over_r"])]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "figure.titlesize": 17,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "image.interpolation": "none",
        }
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.5, 10.5),
        constrained_layout=True,
    )
    image = None
    for axis, (title, metric, decimals) in zip(
        axes.flat,
        PANELS,
        strict=True,
    ):
        values = np.full(shape, np.nan)
        for row in grid:
            values[
                q_index[float(row["q_over_r"])],
                q_final_index[float(row["q_final_over_r"])],
            ] = float(row[metric])
        if np.isnan(values).any():
            raise ValueError(f"Incomplete calibration grid for {metric}")

        image = axis.imshow(
            values,
            cmap="viridis",
            vmin=0.0,
            vmax=2.0,
            aspect="auto",
        )
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel(r"Terminal-cost ratio $Q_f/R$")
        axis.set_ylabel(r"State-cost ratio $Q/R$")
        axis.set_xticks(
            range(len(q_final_values)),
            [_label(value) for value in q_final_values],
        )
        axis.set_yticks(
            range(len(q_values)),
            [_label(value) for value in q_values],
        )

        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                is_selected = (
                    row_index == selected_row
                    and column_index == selected_column
                )
                marker = "$\\star$\n" if is_selected else ""
                axis.text(
                    column_index,
                    row_index,
                    f"{marker}{value:.{decimals}f}",
                    ha="center",
                    va="center",
                    color="white" if value < 1.0 else "black",
                    fontweight="bold" if is_selected else "normal",
                    fontsize=12,
                )

        axis.add_patch(
            Rectangle(
                (selected_column - 0.48, selected_row - 0.48),
                0.96,
                0.96,
                fill=False,
                edgecolor="#ff3b30",
                linewidth=4,
            )
        )

    colorbar = figure.colorbar(
        image,
        ax=axes,
        location="right",
        shrink=0.86,
        pad=0.02,
    )
    colorbar.set_label("AXBench score (0–2)")
    figure.suptitle(
        r"L-CiteEval $H_\infty$ calibration grid"
        "\n"
        rf"Selected: $Q/R={_label(float(selected['q_over_r']))}$, "
        rf"$Q_f/R={_label(float(selected['q_final_over_r']))}$, "
        rf"$R={_label(float(selected['r']))}$, "
        rf"$\lambda={_label(float(selected['lambda']))}$, "
        rf"$\gamma^\star={float(selected['gamma_star']):.4f}$",
        fontweight="bold",
    )

    PLOTS.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(
            PLOTS / f"lciteeval_calibration_heatmaps.{suffix}",
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    args = parser.parse_args()
    render(args.selection)


if __name__ == "__main__":
    main()
