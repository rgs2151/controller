"""Render cached Gemma-2-2B H-infinity toxicity calibration sweeps."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns
import torch


UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[1]
CALIBRATION = (
    REPO
    / "benchmarks/toxicity/cache/gemma2b/calibrations/h_infinity/selected"
)
SELECTION = CALIBRATION / "selection.json"
CONTROLLERS = CALIBRATION / "grid/controllers"
PLOTS = UNIT / "plots"


def _label(value: float) -> str:
    return f"{value:.3g}"


def _render(
    matrix: np.ndarray,
    *,
    filename: str,
    title: str,
    colorbar_label: str,
    annotation_format: str,
    q_labels: list[str],
    q_final_labels: list[str],
    selected_row: int,
    selected_column: int,
    cmap: str,
) -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.titlesize": 18,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    figure, axis = plt.subplots(figsize=(6.2, 8.2))
    sns.heatmap(
        matrix,
        ax=axis,
        cmap=cmap,
        annot=True,
        fmt=annotation_format,
        linewidths=0.4,
        linecolor="0.88",
        xticklabels=q_final_labels,
        yticklabels=q_labels,
        cbar_kws={"label": colorbar_label, "shrink": 0.78},
    )
    axis.add_patch(
        Rectangle(
            (selected_column, selected_row),
            1,
            1,
            fill=False,
            edgecolor="#b00000",
            linewidth=3,
        )
    )
    axis.set_title(title, pad=14)
    axis.set_xlabel(r"Terminal-state cost ratio $Q_f/R$")
    axis.set_ylabel(r"Running-state cost ratio $Q/R$")
    axis.tick_params(axis="x", rotation=35)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(PLOTS / f"{filename}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    selection = json.loads(SELECTION.read_text())
    rows = selection["grid"]
    q_values = sorted({row["parameters"]["q_over_r"] for row in rows})
    q_final_values = sorted(
        {row["parameters"]["q_final_over_r"] for row in rows}
    )
    q_index = {value: index for index, value in enumerate(q_values)}
    q_final_index = {value: index for index, value in enumerate(q_final_values)}
    shape = (len(q_values), len(q_final_values))
    toxicity = np.full(shape, np.nan)
    perplexity = np.full(shape, np.nan)
    gamma_star = np.full(shape, np.nan)

    for row in rows:
        parameters = row["parameters"]
        i = q_index[parameters["q_over_r"]]
        j = q_final_index[parameters["q_final_over_r"]]
        toxicity[i, j] = row["toxicity"]
        perplexity[i, j] = row["perplexity"]
        controller = torch.load(
            CONTROLLERS / f"{row['candidate_id']}.pt",
            map_location="cpu",
            weights_only=True,
        )
        gamma_star[i, j] = float(controller["gamma_star"])

    if any(np.isnan(matrix).any() for matrix in (toxicity, perplexity, gamma_star)):
        raise ValueError("Toxicity calibration grid is incomplete")

    selected = selection["selected"]
    selected_parameters = selected["parameters"]
    selected_row = q_index[selected_parameters["q_over_r"]]
    selected_column = q_final_index[selected_parameters["q_final_over_r"]]
    common = {
        "q_labels": [_label(value) for value in q_values],
        "q_final_labels": [_label(value) for value in q_final_values],
        "selected_row": selected_row,
        "selected_column": selected_column,
    }

    _render(
        toxicity,
        filename="h_infinity_toxicity_heatmap",
        title=r"$H_\infty$ toxicity calibration ($n=100\times5$)",
        colorbar_label="Classifier toxicity (%) ↓",
        annotation_format=".1f",
        cmap="Greys_r",
        **common,
    )
    _render(
        perplexity,
        filename="h_infinity_perplexity_heatmap",
        title=r"$H_\infty$ calibration perplexity ($n=100\times5$)",
        colorbar_label="Perplexity ↓",
        annotation_format=".1f",
        cmap="Greys_r",
        **common,
    )
    _render(
        gamma_star,
        filename="h_infinity_gamma_star_heatmap",
        title=r"$H_\infty$ attenuation boundary",
        colorbar_label=r"Minimum feasible $\gamma^\star$",
        annotation_format=".1f",
        cmap="Greys",
        **common,
    )


if __name__ == "__main__":
    main()
