from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch
from PIL import Image


UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[1]
PLOTS = UNIT / "plots"
LOGOS = REPO / "figs" / "logos"

TEAL = "#398197"
GRAY = "#A7ADB2"
INK = "#000000"

MODEL_SIZES = {
    "GPT-2 XL": 1.5,
    "Llama-3-8B": 8.0,
    "Qwen-2.5-14B": 14.0,
    "OLMo-2-32B": 32.0,
    "Llama-3.2-1B-Instruct": 1.0,
    "Llama-3.2-3B-Instruct": 3.0,
    "Llama-3.1-8B-Instruct": 8.0,
    "Qwen3-4B": 4.0,
    "Phi-4-mini": 3.8,
    "Granite-3.3-2B": 2.0,
    "Qwen2.5-3B-Instruct": 3.0,
    "Llama-3.2-1B-Instruct": 1.0,
}

MODEL_FAMILIES = {
    "GPT-2 XL": "GPT",
    "Llama-3-8B": "LLaMA",
    "Qwen-2.5-14B": "Qwen",
    "OLMo-2-32B": "OLMo",
    "Llama-3.2-1B-Instruct": "LLaMA",
    "Llama-3.2-3B-Instruct": "LLaMA",
    "Llama-3.1-8B-Instruct": "LLaMA",
    "Qwen3-4B": "Qwen",
    "Phi-4-mini": "Phi",
    "Granite-3.3-2B": "Granite",
    "Qwen2.5-3B-Instruct": "Qwen",
}

LOGO_FILES = {
    "GPT": LOGOS / "openai_transparent.png",
    "LLaMA": LOGOS / "llama_transparent.png",
    "Qwen": LOGOS / "qwen_transparent.png",
    "Phi": LOGOS / "microsoft_transparent.png",
    "Granite": LOGOS / "ibm_transparent.png",
    "OLMo": LOGOS / "ai2_transparent.png",
}

LOGO_VISUAL_SCALE = {
    "GPT": 1.00,
    "LLaMA": 1.00,
    "Qwen": 0.92,
    "Phi": 0.92,
    "Granite": 1.12,
    "OLMo": 1.00,
}


def setup_style() -> None:
    sns.set_theme(context="talk", style="ticks", palette="dark")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1,
            "patch.linewidth": 0,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def read_markdown_table(path: Path) -> list[dict[str, str]]:
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| Model |"))
    header = [cell.strip() for cell in lines[start].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append(dict(zip(header, cells, strict=True)))
    return rows


def numeric(cell: str) -> float:
    match = re.match(r"\s*(-?\d+(?:\.\d+)?)", cell)
    if match is None:
        raise ValueError(f"No numeric prefix in {cell!r}")
    return float(match.group(1))


def select_scores(
    rows: list[dict[str, str]],
    metric: str,
    condition_key: str | None = None,
    condition_value: str | None = None,
    invert_percent: bool = False,
    exclude_competitors: tuple[str, ...] = (),
) -> dict[str, object]:
    selected = [
        row
        for row in rows
        if condition_key is None or row[condition_key] == condition_value
    ]
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        by_model[row["Model"]].append(row)

    baseline: list[dict[str, object]] = []
    ours: list[dict[str, object]] = []
    for model, model_rows in by_model.items():
        if model not in MODEL_SIZES:
            raise KeyError(f"Missing parameter count for {model}")

        def score(row: dict[str, str]) -> float:
            value = numeric(row[metric])
            return 100.0 - value if invert_percent else value

        competitor = max(
            (
                row
                for row in model_rows
                if row["Method"] != "H∞ (ours)"
                and row["Method"] not in exclude_competitors
            ),
            key=score,
        )
        hinf = next(row for row in model_rows if row["Method"] == "H∞ (ours)")
        baseline.append(
            {
                "model": model,
                "family": MODEL_FAMILIES[model],
                "size_b": MODEL_SIZES[model],
                "method": competitor["Method"],
                "score": score(competitor),
            }
        )
        ours.append(
            {
                "model": model,
                "family": MODEL_FAMILIES[model],
                "size_b": MODEL_SIZES[model],
                "method": hinf["Method"],
                "score": score(hinf),
            }
        )

    def weighted(points: list[dict[str, object]]) -> float:
        numerator = sum(float(point["score"]) * float(point["size_b"]) for point in points)
        denominator = sum(float(point["size_b"]) for point in points)
        return numerator / denominator

    return {
        "baseline": baseline,
        "ours": ours,
        "baseline_mean": weighted(baseline),
        "ours_mean": weighted(ours),
    }


def build_results() -> dict[str, dict[str, dict[str, object]]]:
    truth_id = read_markdown_table(
        REPO / "figs/bench_table/truthfulness/truthfulqa.md"
    )
    truth_ood = read_markdown_table(
        REPO / "figs/bench_table/truthfulness/truthfulqa_spanish.md"
    )
    harmful = read_markdown_table(
        REPO / "figs/bench_table/harmful/harmbench_summary.md"
    )
    mgsm = read_markdown_table(REPO / "figs/bench_table/mgsm/mgsm_full.md")
    lcite = read_markdown_table(
        REPO / "figs/bench_table/lciteeval/lciteeval_summary.md"
    )

    return {
        "Truthfulness shift": {
            "ID": select_scores(truth_id, "True (%) ↑"),
            "OOD": select_scores(truth_ood, "True (%) ↑"),
        },
        "Adversarial shift": {
            "ID": select_scores(harmful, "Direct ASR (%) ↓", invert_percent=True),
            "OOD": select_scores(
                harmful,
                "Human-jailbreak average ASR (%) ↓",
                invert_percent=True,
            ),
        },
        "Language shift": {
            language: select_scores(
                mgsm,
                "Accuracy (%) ↑",
                condition_key="Language",
                condition_value=language,
                exclude_competitors=("Original",),
            )
            for language in ["Chinese", "French", "Japanese", "Swahili", "Telugu"]
        },
        "Context shift": {
            "ID": select_scores(
                lcite,
                "Citation F1 (%) ↑",
                condition_key="Context",
                condition_value="8K",
            ),
            "OOD": select_scores(
                lcite,
                "Citation F1 (%) ↑",
                condition_key="Context",
                condition_value="16K",
            ),
        },
    }


def square_logo(path: Path, side: int = 256) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    pixels = np.asarray(image)
    visible = (pixels[..., 3] > 10) & (np.min(pixels[..., :3], axis=2) < 246)
    if np.any(visible):
        ys, xs = np.where(visible)
        image = image.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    image.thumbnail((side - 24, side - 24), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (side, side), (255, 255, 255, 0))
    position = ((side - image.width) // 2, (side - image.height) // 2)
    canvas.alpha_composite(image, position)
    return np.asarray(canvas)


def logo_zoom(size_b: float) -> float:
    return 0.048 + 0.012 * math.log2(max(size_b, 1.0))


def add_logo(
    ax: plt.Axes,
    logo_images: dict[str, np.ndarray],
    family: str,
    x: float,
    y: float,
    zoom: float,
    frame: bool = False,
) -> None:
    image = OffsetImage(
        logo_images[family],
        zoom=zoom * LOGO_VISUAL_SCALE[family],
        resample=True,
    )
    box = AnnotationBbox(
        image,
        (x, y),
        frameon=frame,
        pad=0,
        bboxprops={
            "boxstyle": "round,pad=0.06",
            "facecolor": "white",
            "edgecolor": "#5F6368",
            "linewidth": 0.45,
        },
        zorder=8,
        annotation_clip=False,
    )
    ax.add_artist(box)


def draw_panel(
    ax: plt.Axes,
    title: str,
    conditions: dict[str, dict[str, object]],
    ylabel: str,
    ymax: float,
    logo_images: dict[str, np.ndarray],
    condition_labels: list[str],
    group_gap: float | None = None,
) -> None:
    if group_gap is None:
        group_gap = 3.15 if len(conditions) <= 2 else 2.70
    width = 0.92
    centers = np.arange(len(conditions), dtype=float) * group_gap
    for center, (condition, values) in zip(centers, conditions.items(), strict=True):
        positions = [center - 0.53, center + 0.53]
        heights = [float(values["baseline_mean"]), float(values["ours_mean"])]
        ax.bar(positions, heights, width=width, color=[GRAY, TEAL], zorder=2)

        for position, height, key in zip(
            positions, heights, ["baseline", "ours"], strict=True
        ):
            points = list(values[key])
            offsets = np.linspace(-0.30, 0.30, len(points)) if len(points) > 1 else [0.0]
            for point, offset in zip(points, offsets, strict=True):
                add_logo(
                    ax,
                    logo_images,
                    str(point["family"]),
                    position + float(offset),
                    float(point["score"]),
                    logo_zoom(float(point["size_b"])),
                )
    left = centers[0] - 0.9
    right = centers[-1] + 0.9
    ax.set_xlim(left, right)
    ax.set_ylim(-0.05 * ymax, ymax)
    ax.set_yticks(np.linspace(0, ymax, 6))
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(centers)
    ax.set_xticklabels(condition_labels, fontsize=9.5)
    ax.set_title(title, fontsize=14, fontweight="semibold", pad=20)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.spines["left"].set_bounds(0, ymax)
    ax.spines["left"].set_position(("outward", 4))
    ax.spines["bottom"].set_position(("outward", 4))
    ax.axhline(0, color=INK, linewidth=0.8, zorder=3)
    ax.yaxis.grid(True, color="#E5E7E9", linewidth=0.6, zorder=0)


def write_values(results: dict[str, dict[str, dict[str, object]]]) -> None:
    path = PLOTS / "figure_0_overall_values.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            lineterminator="\n",
            fieldnames=[
                "shift",
                "condition",
                "series",
                "model",
                "family",
                "parameters_b",
                "selected_method",
                "score",
                "weighted_mean",
            ],
        )
        writer.writeheader()
        for shift, conditions in results.items():
            for condition, values in conditions.items():
                for series in ["baseline", "ours"]:
                    mean = float(values[f"{series}_mean"])
                    for point in values[series]:
                        writer.writerow(
                            {
                                "shift": shift,
                                "condition": condition,
                                "series": series,
                                "model": point["model"],
                                "family": point["family"],
                                "parameters_b": point["size_b"],
                                "selected_method": point["method"],
                                "score": point["score"],
                                "weighted_mean": mean,
                            }
                        )


def draw_model_legend(ax: plt.Axes, logo_images: dict[str, np.ndarray]) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 1.03, "Model family", ha="center", va="bottom", fontsize=11, fontweight="semibold")
    families = ["GPT", "LLaMA", "Qwen", "Phi", "Granite", "OLMo"]
    y_positions = np.linspace(0.86, 0.41, len(families))
    for family, y in zip(families, y_positions, strict=True):
        add_logo(ax, logo_images, family, 0.24, float(y), 0.078)
        ax.text(0.39, y, family, ha="left", va="center", fontsize=9.5)


def main() -> None:
    setup_style()
    PLOTS.mkdir(exist_ok=True)
    results = build_results()
    write_values(results)
    logo_images = {family: square_logo(path) for family, path in LOGO_FILES.items()}

    # Preserve all established physical sizes while giving the five-condition
    # language panel 30% more room.
    fig = plt.figure(figsize=(18.6, 5.53))
    grid = fig.add_gridspec(
        1,
        6,
        width_ratios=[1.35, 1.35, 0.58, 3.84, 1.35, 0.82],
        left=0.045,
        right=0.985,
        bottom=0.19,
        top=0.84,
        wspace=0.44,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in [0, 1, 3, 4, 5]]

    draw_panel(
        axes[0],
        "Truthfulness shift",
        results["Truthfulness shift"],
        "Truthful responses (%)",
        100,
        logo_images,
        ["English", "Spanish"],
    )
    draw_panel(
        axes[1],
        "Adversarial shift",
        results["Adversarial shift"],
        "Safe responses (%)",
        100,
        logo_images,
        ["Direct", "Adversaries"],
    )
    draw_panel(
        axes[2],
        "Language shift",
        results["Language shift"],
        "Accuracy (%)",
        80,
        logo_images,
        ["Chinese", "French", "Japanese", "Swahili", "Telugu"],
        group_gap=3.10,
    )
    draw_panel(
        axes[3],
        "Context shift",
        results["Context shift"],
        "Citation F1 (%)",
        10,
        logo_images,
        ["8K", "16K"],
    )
    draw_model_legend(axes[4], logo_images)

    fig.legend(
        handles=[
            Patch(facecolor=GRAY, label="Best competitor"),
            Patch(facecolor=TEAL, label=r"H$\infty$ (ours)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        fontsize=13,
        handlelength=1.8,
        handleheight=0.9,
        columnspacing=2.2,
        labelspacing=0.0,
    )

    for suffix in ["pdf", "png"]:
        fig.savefig(
            PLOTS / f"figure_0_overall.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
