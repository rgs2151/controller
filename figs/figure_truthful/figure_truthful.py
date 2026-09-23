from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch
from PIL import Image


UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[1]
PLOTS = UNIT / "plots"
LOGOS = REPO / "figs/logos"
CATEGORY_CACHE = UNIT / "cache/truthfulqa_category_txi.csv"

TEAL = "#398197"
GRAY = "#A7ADB2"
INK = "#202124"
GRID = "#E3E6E8"

MODEL_SIZES = {
    "GPT-2 XL": 1.5,
    "Llama-3-8B": 8.0,
    "Qwen-2.5-14B": 14.0,
    "OLMo-2-32B": 32.0,
}
MODEL_FAMILIES = {
    "GPT-2 XL": "GPT",
    "Llama-3-8B": "LLaMA",
    "Qwen-2.5-14B": "Qwen",
    "OLMo-2-32B": "OLMo",
}
LOGO_FILES = {
    "GPT": LOGOS / "openai_transparent.png",
    "LLaMA": LOGOS / "llama_transparent.png",
    "Qwen": LOGOS / "qwen_transparent.png",
    "OLMo": LOGOS / "ai2_transparent.png",
}
LOGO_SCALE = {"GPT": 1.00, "LLaMA": 1.00, "Qwen": 0.92, "OLMo": 1.00}

METHOD_ORDER = [
    "Original",
    "ITI",
    "ActAdd",
    "Mean-AcT",
    "Linear-AcT",
    "PID-AcT",
    "ODESteer",
    "S-PID",
    "A-LQR",
    "H∞ (ours)",
]
METHOD_COLORS = {
    "Original": "#59636D",
    "ITI": "#8E88A4",
    "ActAdd": "#B88898",
    "Mean-AcT": "#9DA6AD",
    "Linear-AcT": "#B4A06F",
    "PID-AcT": "#9A8678",
    "ODESteer": "#7C9C8D",
    "S-PID": "#B4775D",
    "A-LQR": "#687DA3",
    "H∞ (ours)": TEAL,
}
METHOD_STYLES = {
    "Original": (0, (1.0, 1.2)),
    "ITI": (0, (1.2, 1.5)),
    "ActAdd": (0, (4.0, 1.6, 1.0, 1.6)),
    "Mean-AcT": (0, (7.0, 2.2)),
    "Linear-AcT": (0, (2.0, 1.3)),
    "PID-AcT": (0, (5.5, 1.5, 1.2, 1.5)),
    "ODESteer": (0, (3.5, 1.5)),
    "S-PID": (0, (5.0, 2.0)),
    "A-LQR": "-",
    "H∞ (ours)": "-",
}


@dataclass(frozen=True)
class Result:
    split: str
    model: str
    method: str
    true: float
    informative: float
    txi: float


def setup_style() -> None:
    sns.set_theme(context="talk", style="ticks", palette="dark")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
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


def numeric(cell: str) -> float:
    match = re.match(r"\s*(-?\d+(?:\.\d+)?)", cell)
    if match is None:
        raise ValueError(f"No numeric prefix in {cell!r}")
    return float(match.group(1))


def read_table(path: Path, split: str) -> list[Result]:
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| Model |"))
    header = [cell.strip() for cell in lines[start].strip("|").split("|")]
    records: list[Result] = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        row = dict(
            zip(header, [cell.strip() for cell in line.strip("|").split("|")], strict=True)
        )
        if row["Model"] not in MODEL_SIZES:
            continue
        records.append(
            Result(
                split=split,
                model=row["Model"],
                method=row["Method"],
                true=numeric(row["True (%) ↑"]),
                informative=numeric(row["Informative (%) ↑"]),
                txi=numeric(row["T×I (%) ↑"]),
            )
        )
    return records


def square_logo(path: Path, side: int = 256) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    pixels = np.asarray(image)
    visible = (pixels[..., 3] > 10) & (np.min(pixels[..., :3], axis=2) < 246)
    if np.any(visible):
        ys, xs = np.where(visible)
        image = image.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    image.thumbnail((side - 24, side - 24), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (side, side), (255, 255, 255, 0))
    canvas.alpha_composite(image, ((side - image.width) // 2, (side - image.height) // 2))
    return np.asarray(canvas)


def logo_zoom(size_b: float, base: float = 0.040) -> float:
    return base + 0.010 * math.log2(max(size_b, 1.0))


def tinted_logo(image: np.ndarray, color: str) -> np.ndarray:
    tinted = np.zeros_like(image)
    tinted[..., :3] = np.round(255 * np.asarray(mpl.colors.to_rgb(color))).astype(
        np.uint8
    )
    tinted[..., 3] = image[..., 3]
    return tinted


def add_logo(
    ax: plt.Axes,
    images: dict[str, np.ndarray],
    model: str,
    x: float,
    y: float,
    *,
    zoom: float | None = None,
    method: str | None = None,
    coordinates: str | mpl.transforms.Transform = "data",
) -> None:
    family = MODEL_FAMILIES[model]
    if zoom is None:
        zoom = logo_zoom(MODEL_SIZES[model])
    image_key = family if method is None else f"{family}::{method}"
    image = OffsetImage(
        images[image_key], zoom=zoom * LOGO_SCALE[family], resample=True
    )
    image.set_alpha(1.0 if method in {None, "H∞ (ours)"} else 0.70)
    transform_kwargs = {} if coordinates == "data" else {"xycoords": coordinates}
    ax.add_artist(
        AnnotationBbox(
            image,
            (x, y),
            frameon=False,
            pad=0,
            zorder=12 if method == "H∞ (ours)" else 7,
            annotation_clip=False,
            **transform_kwargs,
        )
    )


def weighted(points: list[Result], value: str) -> float:
    return sum(getattr(row, value) * MODEL_SIZES[row.model] for row in points) / sum(
        MODEL_SIZES[row.model] for row in points
    )


def draw_box_panel(
    ax: plt.Axes,
    records: list[Result],
    images: dict[str, np.ndarray],
) -> None:
    centers = np.array([0.0, 3.55])
    for center, split in zip(centers, ["ID", "OOD"], strict=True):
        split_rows = [row for row in records if row.split == split]
        by_model: dict[str, list[Result]] = defaultdict(list)
        for row in split_rows:
            by_model[row.model].append(row)
        baseline = [
            max((row for row in rows if row.method != "H∞ (ours)"), key=lambda row: row.true)
            for rows in by_model.values()
        ]
        ours = [next(row for row in rows if row.method == "H∞ (ours)") for rows in by_model.values()]
        positions = [center - 0.60, center + 0.60]
        for position, points, color in zip(
            positions, [baseline, ours], [GRAY, TEAL], strict=True
        ):
            scores = [point.true for point in points]
            ax.boxplot(
                [scores],
                positions=[position],
                widths=0.72,
                patch_artist=True,
                showfliers=False,
                whis=(0, 100),
                manage_ticks=False,
                boxprops={
                    "facecolor": color,
                    "edgecolor": color,
                    "alpha": 0.30,
                    "linewidth": 1.15,
                },
                medianprops={"color": INK, "linewidth": 1.25},
                whiskerprops={"color": color, "linewidth": 1.15, "alpha": 0.8},
                capprops={"color": color, "linewidth": 1.15, "alpha": 0.8},
                zorder=2,
            )
            offsets = np.linspace(-0.22, 0.22, len(points))
            for point, offset in zip(points, offsets, strict=True):
                add_logo(ax, images, point.model, position + float(offset), point.true)

    ax.set_xlim(-1.0, centers[-1] + 1.0)
    ax.set_ylim(-5, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xticks(centers, ["English", "Spanish"], fontsize=8.0)
    ax.set_ylabel("True (%)", fontsize=10.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", length=0, pad=7, labelsize=7.0)
    ax.spines["left"].set_bounds(0, 100)
    ax.spines["left"].set_position(("outward", 4))
    ax.spines["bottom"].set_position(("outward", 4))
    ax.axhline(0, color=INK, linewidth=0.8, zorder=3)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)


def smooth_density(values: list[float], grid: np.ndarray, bandwidth: float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    scaled = (grid[:, None] - array[None, :]) / bandwidth
    density = np.exp(-0.5 * scaled**2).sum(axis=1)
    peak = float(density.max(initial=0.0))
    return density / peak if peak else density


def draw_txi_field(ax: plt.Axes) -> None:
    grid = np.linspace(0, 100, 301)
    xx, yy = np.meshgrid(grid, grid)
    field = xx * yy / 100.0
    ax.contourf(
        xx,
        yy,
        field,
        levels=np.arange(0, 101, 10),
        cmap="Greys",
        alpha=0.17,
        antialiased=True,
        zorder=-3,
    )
    contours = ax.contour(
        xx,
        yy,
        field,
        levels=[20, 40, 60, 80],
        colors="#777B80",
        linewidths=0.65,
        linestyles="--",
        alpha=0.75,
        zorder=-1,
    )
    ax.clabel(contours, fmt=lambda value: f"T×I={value:.0f}", fontsize=6.8, inline=True)


def draw_frontier(
    main_ax: plt.Axes,
    top_ax: plt.Axes,
    right_ax: plt.Axes,
    records: list[Result],
    split: str,
    images: dict[str, np.ndarray],
) -> None:
    rows = [row for row in records if row.split == split]
    draw_txi_field(main_ax)
    for method in METHOD_ORDER:
        for row in rows:
            if row.method == method:
                add_logo(
                    main_ax,
                    images,
                    row.model,
                    row.informative,
                    row.true,
                    method=row.method,
                )

    main_ax.set_xlim(0, 100)
    main_ax.set_ylim(0, 100)
    main_ax.set_xticks(np.arange(0, 101, 20))
    main_ax.set_yticks(np.arange(0, 101, 20))
    main_ax.set_xlabel("Informative (%)", fontsize=10.5)
    main_ax.set_ylabel("True (%)", fontsize=10.5)
    main_ax.tick_params(labelsize=8)
    main_ax.grid(True, color=GRID, linewidth=0.55, zorder=-2)
    for spine in main_ax.spines.values():
        spine.set_color("#AEB4BC")
        spine.set_linewidth(0.7)

    x_grid = np.linspace(0, 100, 300)
    y_grid = np.linspace(0, 100, 300)
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row.method == method]
        if not method_rows:
            continue
        color = METHOD_COLORS[method]
        primary = method == "H∞ (ours)"
        alpha = 1.0 if primary else 0.46
        linewidth = 1.9 if primary else 0.9
        x_density = smooth_density(
            [row.informative for row in method_rows], x_grid, 5.0
        )
        y_density = smooth_density([row.true for row in method_rows], y_grid, 5.0)
        top_ax.plot(
            x_grid,
            x_density,
            color=color,
            linestyle=METHOD_STYLES[method],
            linewidth=linewidth,
            alpha=alpha,
        )
        right_ax.plot(
            y_density,
            y_grid,
            color=color,
            linestyle=METHOD_STYLES[method],
            linewidth=linewidth,
            alpha=alpha,
        )
        if primary:
            top_ax.fill_between(x_grid, 0, x_density, color=color, alpha=0.10)
            right_ax.fill_betweenx(y_grid, 0, y_density, color=color, alpha=0.10)

    top_ax.set_xlim(0, 100)
    top_ax.set_ylim(0, 1.08)
    top_ax.set_xticks([])
    top_ax.set_yticks([])
    top_ax.spines[["top", "left", "right"]].set_visible(False)
    top_ax.spines["bottom"].set_color("#AEB4BC")

    right_ax.set_xlim(0, 1.08)
    right_ax.set_ylim(0, 100)
    right_ax.set_xticks([])
    right_ax.set_yticks([])
    right_ax.spines[["top", "right", "bottom"]].set_visible(False)
    right_ax.spines["left"].set_color("#AEB4BC")


def category_radar_values() -> tuple[
    list[str], dict[str, tuple[list[float], list[float]]]
]:
    rows = list(csv.DictReader(CATEGORY_CACHE.open()))
    categories = list(dict.fromkeys(row["category"] for row in rows))
    ours_by_split: dict[tuple[str, str], float] = {}
    best_by_split: dict[tuple[str, str], float] = {}
    for split in ["ID", "OOD"]:
        for category in categories:
            relevant = [
                row for row in rows if row["split"] == split and row["category"] == category
            ]
            by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in relevant:
                by_model[row["model"]].append(row)
            ours_points: list[float] = []
            best_points: list[float] = []
            for model, model_rows in by_model.items():
                ours = next(row for row in model_rows if row["method"] == "H∞ (ours)")
                best = max(
                    (row for row in model_rows if row["method"] != "H∞ (ours)"),
                    key=lambda row: float(row["txi_pct"]),
                )
                ours_points.append(float(ours["txi_pct"]))
                best_points.append(float(best["txi_pct"]))

            ours_by_split[(split, category)] = float(np.mean(ours_points))
            best_by_split[(split, category)] = float(np.mean(best_points))

    values = {
        split: (
            [ours_by_split[(split, category)] for category in categories],
            [best_by_split[(split, category)] for category in categories],
        )
        for split in ["ID", "OOD"]
    }
    return categories, values


def draw_radar(ax: plt.Axes, split: str) -> None:
    categories, values = category_radar_values()
    ours, best = values[split]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    angles = np.r_[angles, angles[0]]
    ours_closed = np.r_[ours, ours[0]]
    best_closed = np.r_[best, best[0]]
    ax.plot(angles, best_closed, color=GRAY, linewidth=1.7, linestyle="--")
    ax.fill(angles, best_closed, color=GRAY, alpha=0.08)
    ax.plot(angles, ours_closed, color=TEAL, linewidth=2.2)
    ax.fill(angles, ours_closed, color=TEAL, alpha=0.12)
    ax.scatter(angles[:-1], best, s=14, color=GRAY, zorder=4)
    ax.scatter(angles[:-1], ours, s=18, color=TEAL, zorder=5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8.6)
    ax.tick_params(axis="x", pad=10)
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels([])
    ax.grid(color="#C9CED3", linewidth=0.6, linestyle=":")
    ax.spines["polar"].set_color("#AEB4BC")


def draw_model_legend(ax: plt.Axes, images: dict[str, np.ndarray]) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    models = list(MODEL_SIZES)
    labels = ["GPT-2 XL", "LLaMA-3-8B", "Qwen-2.5-14B", "OLMo-2-32B"]
    for model, label, x in zip(
        models, labels, np.linspace(0.17, 0.83, len(models)), strict=True
    ):
        add_logo(
            ax,
            images,
            model,
            float(x) - 0.040,
            0.5,
            zoom=0.054,
            coordinates=ax.transAxes,
        )
        ax.text(float(x) + 0.010, 0.5, label, ha="left", va="center", fontsize=8.0)


def method_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linestyle=METHOD_STYLES[method],
            linewidth=2.0 if method == "H∞ (ours)" else 1.2,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=METHOD_COLORS[method],
            markeredgewidth=1.4 if method == "H∞ (ours)" else 1.0,
            markersize=6,
            alpha=1.0 if method == "H∞ (ours)" else 0.72,
            label=method,
        )
        for method in METHOD_ORDER
    ]


def main() -> None:
    setup_style()
    PLOTS.mkdir(parents=True, exist_ok=True)
    records = read_table(
        REPO / "figs/bench_table/truthfulness/truthfulqa.md", "ID"
    ) + read_table(
        REPO / "figs/bench_table/truthfulness/truthfulqa_spanish.md", "OOD"
    )
    images = {family: square_logo(path) for family, path in LOGO_FILES.items()}
    for family in LOGO_FILES:
        for method in METHOD_ORDER:
            images[f"{family}::{method}"] = tinted_logo(
                images[family], METHOD_COLORS[method]
            )

    fig = plt.figure(figsize=(15.4, 4.8))
    outer = fig.add_gridspec(
        1,
        9,
        width_ratios=[1.68, 0.38, 3.00, 0.22, 3.00, 0.46, 2.10, 1.18, 2.10],
        left=0.035,
        right=0.985,
        bottom=0.24,
        top=0.86,
        wspace=0.12,
    )
    bar_ax = fig.add_subplot(outer[0])
    draw_box_panel(bar_ax, records, images)

    frontier_axes: list[tuple[plt.Axes, plt.Axes, plt.Axes]] = []
    for index in [2, 4]:
        nested = outer[index].subgridspec(
            2,
            2,
            width_ratios=[8.0, 1.15],
            height_ratios=[1.10, 7.0],
            hspace=0.04,
            wspace=0.04,
        )
        top_ax = fig.add_subplot(nested[0, 0])
        main_ax = fig.add_subplot(nested[1, 0])
        right_ax = fig.add_subplot(nested[1, 1])
        frontier_axes.append((main_ax, top_ax, right_ax))

    draw_frontier(*frontier_axes[0], records, "ID", images)
    draw_frontier(*frontier_axes[1], records, "OOD", images)
    model_ax = fig.add_axes([0.245, 0.885, 0.370, 0.075])
    draw_model_legend(model_ax, images)
    radar_id_ax = fig.add_subplot(outer[6], projection="polar")
    radar_ood_ax = fig.add_subplot(outer[8], projection="polar")
    draw_radar(radar_id_ax, "ID")
    draw_radar(radar_ood_ax, "OOD")

    bar_ax.legend(
        handles=[
            Patch(facecolor=GRAY, label="Best competitor"),
            Patch(facecolor=TEAL, label=r"H$\infty$ (ours)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.21),
        ncol=1,
        fontsize=8.0,
        handlelength=1.4,
        labelspacing=0.45,
    )
    fig.legend(
        handles=method_handles(),
        loc="lower center",
        bbox_to_anchor=(0.405, 0.025),
        ncol=5,
        fontsize=8.4,
        handlelength=1.6,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color=TEAL, linewidth=2.2, label=r"H$\infty$ (ours)"),
            Line2D(
                [0],
                [0],
                color=GRAY,
                linewidth=1.7,
                linestyle="--",
                label="Best competitor",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.825, 0.965),
        ncol=2,
        fontsize=9.0,
        handlelength=1.8,
        columnspacing=1.4,
    )

    for suffix in ["pdf", "png"]:
        fig.savefig(
            PLOTS / f"figure_truthful.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
