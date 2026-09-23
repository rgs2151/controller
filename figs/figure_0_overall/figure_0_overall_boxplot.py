from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

import figure_0_overall as base


UNIT = Path(__file__).resolve().parent
PLOTS = UNIT / "plots"

BOX_LINEWIDTH = 3.45
MEDIAN_LINEWIDTH = 3.75


def recolor_logo(image: np.ndarray, color: str) -> np.ndarray:
    """Recolor one transparent logo without changing its silhouette."""

    result = np.empty_like(image)
    result[..., :3] = np.asarray(to_rgba(color)[:3]) * 255
    result[..., 3] = image[..., 3]
    return result.astype(np.uint8)


def draw_box_panel(
    ax: plt.Axes,
    title: str,
    conditions: dict[str, dict[str, object]],
    ylabel: str,
    ymax: float,
    logo_images: dict[str, np.ndarray],
    condition_labels: list[str],
    group_gap: float | None = None,
    ymin: float = 0.0,
) -> None:
    if group_gap is None:
        group_gap = 3.15 if len(conditions) <= 2 else 2.70
    width = 0.72
    centers = np.arange(len(conditions), dtype=float) * group_gap

    for center, (_, values) in zip(centers, conditions.items(), strict=True):
        positions = [center - 0.60, center + 0.60]
        for position, key, color in zip(
            positions,
            ["baseline", "ours"],
            [base.GRAY, base.TEAL],
            strict=True,
        ):
            points = list(values[key])
            scores = [float(point["score"]) for point in points]
            ax.boxplot(
                [scores],
                positions=[position],
                widths=width,
                patch_artist=True,
                showfliers=False,
                whis=(0, 100),
                manage_ticks=False,
                boxprops={
                    "facecolor": to_rgba(color, 0.30),
                    "edgecolor": color,
                    "linewidth": BOX_LINEWIDTH,
                },
                medianprops={"color": color, "linewidth": MEDIAN_LINEWIDTH},
                whiskerprops={"color": color, "linewidth": BOX_LINEWIDTH},
                capprops={"color": color, "linewidth": BOX_LINEWIDTH},
                zorder=2,
            )

            offsets = (
                np.linspace(-0.22, 0.22, len(points))
                if len(points) > 1
                else [0.0]
            )
            for point, offset in zip(points, offsets, strict=True):
                base.add_logo(
                    ax,
                    {
                        family: recolor_logo(image, color)
                        for family, image in logo_images.items()
                    },
                    str(point["family"]),
                    position + float(offset),
                    float(point["score"]),
                    base.logo_zoom(float(point["size_b"])),
                )

    ax.set_xlim(centers[0] - 1.0, centers[-1] + 1.0)
    displayed_range = ymax - ymin
    lower_limit = ymin - 0.05 * displayed_range
    ax.set_ylim(lower_limit, ymax)
    ax.set_yticks(
        np.linspace(0, ymax, 6)
        if ymin == 0
        else np.arange(ymin, ymax + 0.1, 10)
    )
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(centers)
    ax.set_xticklabels(condition_labels, fontsize=9.5)
    ax.set_title(title, fontsize=14, fontweight="semibold", pad=20)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.spines["left"].set_bounds(ymin, ymax)
    ax.spines["left"].set_position(("outward", 4))
    ax.spines["bottom"].set_position(("outward", 4))
    if ymin == 0:
        ax.axhline(0, color=base.INK, linewidth=0.8, zorder=3)
    ax.yaxis.grid(True, color="#E5E7E9", linewidth=0.6, zorder=0)


def main() -> None:
    base.setup_style()
    PLOTS.mkdir(exist_ok=True)
    results = base.build_results()
    logo_images = {
        family: base.square_logo(path) for family, path in base.LOGO_FILES.items()
    }

    # Reduce the canvas in lockstep with the dedicated spacer so the other
    # panels retain their established physical sizes.
    fig = plt.figure(figsize=(18.02, 5.53))
    grid = fig.add_gridspec(
        1,
        6,
        width_ratios=[1.35, 1.35, 0.29, 3.84, 1.35, 0.82],
        left=0.045,
        right=0.985,
        bottom=0.19,
        top=0.84,
        wspace=0.44,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in [0, 1, 3, 4, 5]]

    draw_box_panel(
        axes[0],
        "Truthfulness shift",
        results["Truthfulness shift"],
        "Truthful responses (%)",
        100,
        logo_images,
        ["English", "Spanish"],
    )
    draw_box_panel(
        axes[1],
        "Adversarial shift",
        results["Adversarial shift"],
        "Safe responses (%)",
        100,
        logo_images,
        ["Direct", "Adversaries"],
        ymin=70,
    )
    draw_box_panel(
        axes[2],
        "Language shift",
        results["Language shift"],
        "Accuracy (%)",
        60,
        logo_images,
        ["Chinese", "French", "Japanese", "Swahili", "Telugu"],
        group_gap=3.10,
    )
    draw_box_panel(
        axes[3],
        "Context shift",
        results["Context shift"],
        "Citation F1 (%)",
        10,
        logo_images,
        ["8K", "16K"],
    )
    base.draw_model_legend(axes[4], logo_images)

    fig.legend(
        handles=[
            Patch(
                facecolor=to_rgba(base.GRAY, 0.30),
                edgecolor=base.GRAY,
                linewidth=BOX_LINEWIDTH,
                label="Best competitor",
            ),
            Patch(
                facecolor=to_rgba(base.TEAL, 0.30),
                edgecolor=base.TEAL,
                linewidth=BOX_LINEWIDTH,
                label=r"H$\infty$ (ours)",
            ),
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
            PLOTS / f"figure_0_overall_boxplot.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
