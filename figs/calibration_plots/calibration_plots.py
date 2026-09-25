#!/usr/bin/env python3
"""Render the four appendix calibration sweeps as reproducible heatmap figures.

The benchmark selection JSON files provide every scored grid component.  The
diagnostic bundles provide the frozen control problem, from which gamma-star is
re-synthesized at every (Q/R, Q_f/R) point.  No benchmark is rerun.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[1]
PLOTS = UNIT / "plots"
CACHE = UNIT / "cache"
TEAL = "#007C7C"
INK = "#17191C"


@dataclass(frozen=True)
class Metric:
    key: str
    title: str
    direction: str = "up"
    scale: float = 1.0
    decimals: int = 2


@dataclass(frozen=True)
class ModelSweep:
    name: str
    selection: Path
    metrics: tuple[Metric, ...]


@dataclass(frozen=True)
class Benchmark:
    slug: str
    title: str
    models: tuple[ModelSweep, ...]


def p(relative: str) -> Path:
    return REPO / relative


BENCHMARKS = (
    Benchmark(
        "truthfulqa",
        "TruthfulQA calibration",
        (
            ModelSweep(
                "GPT-2 XL",
                p("benchmarks/truthfulness/cache/gpt2_xl/calibrations/h_infinity/txi95_fluency05_n200_r1/selection.json"),
                (
                    Metric("truth", "True (%)", decimals=1),
                    Metric("axbench_fluency", "Fluency", decimals=2),
                    Metric("truthfulqa_txi_fluency_composite", "Calibration score", decimals=3),
                    Metric("gamma_star", r"$\gamma^\star$", "down", decimals=2),
                ),
            ),
            ModelSweep(
                "Llama-3-8B",
                p("benchmarks/truthfulness/cache/llama8b/calibrations/h_infinity/txi_fluency_95_05/selection.json"),
                (
                    Metric("truth", "True (%)", decimals=1),
                    Metric("axbench_fluency", "Fluency", decimals=2),
                    Metric("truthfulqa_txi_fluency_composite", "Calibration score", decimals=3),
                    Metric("gamma_star", r"$\gamma^\star$", "down", decimals=2),
                ),
            ),
            ModelSweep(
                "Qwen2.5-14B",
                p("benchmarks/truthfulness/cache/qwen14b/calibrations/h_infinity/selected/selection.json"),
                (
                    Metric("truth", "True (%)", decimals=1),
                    Metric("axbench_fluency", "Fluency", decimals=2),
                    Metric("truthfulness_quality_composite", "Calibration score", decimals=3),
                    Metric("gamma_star", r"$\gamma^\star$", "down", decimals=2),
                ),
            ),
            ModelSweep(
                "OLMo-2-32B",
                p("benchmarks/truthfulness/cache/olmo2_32b_instruct/calibrations/h_infinity/txi95_fluency05_n200_r1/selection.json"),
                (
                    Metric("truth", "True (%)", decimals=1),
                    Metric("axbench_fluency", "Fluency", decimals=2),
                    Metric("truthfulqa_txi_fluency_composite", "Calibration score", decimals=3),
                    Metric("gamma_star", r"$\gamma^\star$", "down", decimals=2),
                ),
            ),
        ),
    ),
    Benchmark(
        "harmbench",
        "HarmBench calibration",
        tuple(
            ModelSweep(
                name,
                p(path),
                (
                    Metric("harmbench_validation_success", "Safe validation (%)", scale=100.0, decimals=0),
                    Metric("axbench_fluency", "Fluency", decimals=2),
                    Metric("axbench_overall", "Calibration score", decimals=2),
                    Metric("gamma_star", r"$\gamma^\star$", "down", decimals=2),
                ),
            )
            for name, path in (
                ("Llama-3.2-1B", "benchmarks/harmful/cache/llama32_1b_instruct/calibrations/h_infinity/selected/selection.json"),
                ("Llama-3.2-3B", "benchmarks/harmful/cache/llama32_3b_instruct/calibrations/h_infinity/selected/selection.json"),
                ("Llama-3.1-8B", "benchmarks/harmful/cache/llama31_8b_instruct/calibrations/h_infinity/selected/selection.json"),
            )
        ),
    ),
    Benchmark(
        "mgsm",
        "MGSM calibration",
        (
            ModelSweep(
                "Qwen3-4B",
                p("benchmarks/mgsm/cache/qwen3_4b/calibrations/h_infinity/selected/selection.json"),
                (
                    Metric("axbench_fluency", "Fluency", decimals=2),
                    Metric("mgsm_axbench_overall", "Calibration score", decimals=2),
                    Metric("gamma_star", r"$\gamma^\star$", "down", decimals=2),
                ),
            ),
            *tuple(
                ModelSweep(
                    name,
                    p(path),
                    (
                        Metric("axbench_fluency", "Fluency", decimals=2),
                        Metric("mgsm_quality_composite", "Calibration score", decimals=3),
                        Metric("gamma_star", r"$\gamma^\star$", "down", decimals=2),
                    ),
                )
                for name, path in (
                    ("Phi-4-mini-instruct", "benchmarks/mgsm/cache/phi4_mini_instruct/calibrations/h_infinity/selected/selection.json"),
                    ("Granite-3.3-2B-Instruct", "benchmarks/mgsm/cache/granite33_2b_instruct/calibrations/h_infinity/selected/selection.json"),
                )
            ),
        ),
    ),
    Benchmark(
        "lciteeval",
        "L-CiteEval calibration",
        (
            ModelSweep(
                "Qwen2.5-3B-Instruct",
                p("benchmarks/lciteeval/cache/qwen25_3b_instruct/calibrations/h_infinity/selected/selection.json"),
                (
                    Metric("axbench_fluency", "Fluency", decimals=2),
                    Metric("axbench_overall", "Calibration score", decimals=2),
                    Metric("gamma_star", r"$\gamma^\star$", "down", decimals=2),
                ),
            ),
            ModelSweep(
                "Llama-3.2-1B-Instruct",
                p("benchmarks/lciteeval/cache/llama32_1b_instruct/calibrations/h_infinity/selected/selection.json"),
                (
                    Metric("fluency_normalized", "Fluency", decimals=2),
                    Metric("weighted_task_quality_and_steering_score", "Calibration score", decimals=3),
                    Metric("gamma_star", r"$\gamma^\star$", "down", decimals=2),
                ),
            ),
        ),
    ),
)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.titleweight": "bold",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def short_number(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0, abs_tol=1e-10):
        return str(int(round(value)))
    if value < 0.1:
        return f"{value:.3g}"
    return f"{value:.3g}"


def make_problem(bundle: dict, q: float, r: float, qf: float):
    sys.path.insert(0, str(REPO))
    from robust_steerability.control import FiniteHorizonControlProblem

    raw = bundle["problem"]
    horizon, state_dimension, _ = raw["dynamics"].shape
    control_dimension = raw["control_channels"].shape[-1]
    return FiniteHorizonControlProblem(
        dynamics=raw["dynamics"],
        control_channels=raw["control_channels"],
        disturbance_channels=raw["disturbance_channels"],
        state_costs=q * torch.eye(state_dimension).repeat(horizon, 1, 1),
        control_costs=r * torch.eye(control_dimension).repeat(horizon, 1, 1),
        terminal_cost=qf * torch.eye(state_dimension),
        metadata=raw.get("metadata", {}),
    )


def gamma_grid(model: ModelSweep, data: dict, refresh: bool) -> dict[str, float]:
    """Load or recompute gamma-star for each saved grid configuration."""

    CACHE.mkdir(parents=True, exist_ok=True)
    model_slug = "".join(character.lower() if character.isalnum() else "_" for character in model.name).strip("_")
    cache_path = CACHE / f"{model_slug}_gamma.json"
    fingerprint = {
        "selection": str(model.selection.relative_to(REPO)),
        "diagnostic_bundle": data["diagnostic_bundle"],
        "points": [[row["grid_id"], row["q"], row["r"], row["q_final"]] for row in data["grid"]],
    }
    if cache_path.exists() and not refresh:
        cached = load_json(cache_path)
        if cached.get("fingerprint") == fingerprint:
            return {key: float(value) for key, value in cached["gamma_star"].items()}

    sys.path.insert(0, str(REPO))
    from robust_steerability.control import HInfinityController, HInfinityOptions

    run_dir = model.selection.parent / data["diagnostic_bundle"]
    bundle_path = run_dir / "calibration_input.pt"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing diagnostic input: {bundle_path}")
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    options = HInfinityOptions(**bundle["options"])
    values: dict[str, float] = {}
    for row in data["grid"]:
        problem = make_problem(bundle, float(row["q"]), float(row["r"]), float(row["q_final"]))
        controller = HInfinityController.synthesize(problem, device="cpu", options=options)
        if controller.gamma_star is None:
            raise RuntimeError(f"No feasible gamma for {model.name} {row['grid_id']}")
        values[row["grid_id"]] = float(controller.gamma_star)

    selected_id = data["selected"]["grid_id"]
    score_path = run_dir / "score.json"
    if score_path.exists():
        expected = float(load_json(score_path)["gamma_star"])
        # The archived score can differ by one final bisection step across
        # PyTorch builds while remaining inside the configured 1e-5 tolerance.
        if not math.isclose(values[selected_id], expected, rel_tol=0, abs_tol=1e-5):
            raise ValueError(
                f"Gamma reconstruction mismatch for {model.name}: "
                f"{values[selected_id]} != {expected}"
            )
    payload = {"fingerprint": fingerprint, "gamma_star": values}
    cache_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return values


def matrix(data: dict, metric: Metric, gammas: dict[str, float]) -> tuple[np.ndarray, list[float], list[float]]:
    qs = [float(value) for value in data["protocol"]["q_over_r"]]
    qfs = [float(value) for value in data["protocol"]["q_final_over_r"]]
    rows = {(float(row["q_over_r"]), float(row["q_final_over_r"])): row for row in data["grid"]}
    values = np.full((len(qs), len(qfs)), np.nan, dtype=float)
    for i, q in enumerate(qs):
        for j, qf in enumerate(qfs):
            row = rows[(q, qf)]
            raw = gammas[row["grid_id"]] if metric.key == "gamma_star" else row[metric.key]
            values[i, j] = float(raw) * metric.scale
    return values, qs, qfs


def annotation(value: float, decimals: int) -> str:
    if decimals == 0:
        return f"{value:.0f}"
    if decimals == 1:
        return f"{value:.1f}"
    if decimals == 2:
        if abs(value) >= 100:
            return f"{value:.0f}"
        return f"{value:.2f}"
    return f"{value:.3f}"


def draw_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    qs: list[float],
    qfs: list[float],
    metric: Metric,
    selected: tuple[float, float],
    *,
    show_y: bool,
) -> None:
    finite = values[np.isfinite(values)]
    low, high = float(finite.min()), float(finite.max())
    if math.isclose(low, high):
        low -= 0.5
        high += 0.5
    norm = Normalize(vmin=low, vmax=high)
    cmap = mpl.colormaps["Greys"] if metric.direction == "up" else mpl.colormaps["Greys_r"]
    # A fixed cell aspect keeps every grid comparable without the severe
    # horizontal stretching produced by ``aspect='auto'``.
    ax.imshow(values, cmap=cmap, norm=norm, aspect=0.72, interpolation="nearest")
    ax.set_title(metric.title + (" ↑" if metric.direction == "up" else " ↓"), pad=5)
    ax.set_xticks(range(len(qfs)), [short_number(value) for value in qfs])
    ax.set_yticks(range(len(qs)), [short_number(value) for value in qs] if show_y else [])
    ax.set_xlabel(r"$Q_f/R$", labelpad=3)
    if show_y:
        ax.set_ylabel(r"$Q/R$", labelpad=3)
    ax.tick_params(axis="both", length=0, labelsize=7.3, pad=2)
    ax.set_xticks(np.arange(-0.5, len(qfs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(qs), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.05)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INK)
        spine.set_linewidth(0.9)

    for i in range(len(qs)):
        for j in range(len(qfs)):
            value = values[i, j]
            tone = norm(value)
            if metric.direction == "down":
                tone = 1.0 - tone
            color = "white" if tone > 0.62 else INK
            ax.text(j, i, annotation(value, metric.decimals), ha="center", va="center", fontsize=6.6, color=color)

    selected_q, selected_qf = selected
    i = min(range(len(qs)), key=lambda index: abs(qs[index] - selected_q))
    j = min(range(len(qfs)), key=lambda index: abs(qfs[index] - selected_qf))
    ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, edgecolor=TEAL, linewidth=2.3, zorder=4))
    ax.scatter(
        [j + 0.29],
        [i - 0.27],
        marker="*",
        s=34,
        facecolor=TEAL,
        edgecolor="white",
        linewidth=0.35,
        zorder=5,
        clip_on=False,
    )


def render_benchmark(benchmark: Benchmark, refresh_gamma: bool) -> None:
    rows = len(benchmark.models)
    # TruthfulQA uses a 2x2 model layout because its four eight-row sweeps
    # would otherwise produce an anomalously tall figure. Other suites keep
    # one model per row and use a canvas sized to their actual panel count.
    if benchmark.slug == "truthfulqa":
        fig = plt.figure(figsize=(10.2, 4.75), constrained_layout=False)
        outer = fig.add_gridspec(
            2,
            2,
            left=0.055,
            right=0.995,
            bottom=0.075,
            top=0.985,
            hspace=0.17,
            wspace=0.20,
        )
    else:
        columns = len(benchmark.models[0].metrics)
        fig = plt.figure(
            figsize=(1.43 * columns + 1.00, 1.82 * rows + 0.12),
            constrained_layout=False,
        )
        outer = fig.add_gridspec(
            rows,
            1,
            left=0.13,
            right=0.995,
            bottom=0.07,
            top=0.985,
            hspace=0.40,
        )

    for row_index, model in enumerate(benchmark.models):
        data = load_json(model.selection)
        gammas = gamma_grid(model, data, refresh_gamma)
        selected = (float(data["selected"]["q_over_r"]), float(data["selected"]["q_final_over_r"]))
        if benchmark.slug == "truthfulqa":
            slot = outer[row_index // 2, row_index % 2]
            inner = slot.subgridspec(1, len(model.metrics), wspace=0.055)
        else:
            inner = outer[row_index].subgridspec(1, len(model.metrics), wspace=0.065)
        axes = []
        for column, metric in enumerate(model.metrics):
            ax = fig.add_subplot(inner[0, column])
            values, qs, qfs = matrix(data, metric, gammas)
            draw_heatmap(ax, values, qs, qfs, metric, selected, show_y=column == 0)
            axes.append(ax)
        if benchmark.slug == "truthfulqa":
            axes[0].set_ylabel(model.name + "\n" + r"$Q/R$", labelpad=5, fontweight="bold")
            axes[0].yaxis.set_label_coords(-0.27, 0.5)
            continue
        label_axis = axes[0]
        label_x = -0.38
        label_axis.text(
            label_x,
            0.5,
            model.name,
            transform=label_axis.transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontsize=8.8,
            fontweight="bold",
            color=INK,
        )

    PLOTS.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            PLOTS / f"{benchmark.slug}_calibration.{suffix}",
            bbox_inches="tight",
            pad_inches=0.025,
        )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=[item.slug for item in BENCHMARKS], action="append")
    parser.add_argument("--refresh-gamma", action="store_true", help="Recompute cached gamma-star grids")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()
    selected = set(args.benchmark or [item.slug for item in BENCHMARKS])
    for benchmark in BENCHMARKS:
        if benchmark.slug in selected:
            render_benchmark(benchmark, args.refresh_gamma)
            print(f"rendered {benchmark.slug}")


if __name__ == "__main__":
    main()
