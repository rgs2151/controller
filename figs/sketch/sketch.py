"""Generate paper-figure layout mockups with explicitly synthetic values."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


UNIT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = UNIT_DIR / "plots"

METHOD_COLORS = {
    "Original": "0.55",
    "A-LQR": "midnightblue",
    "S-PID": "darkorange",
    r"$H_\infty$": "darkgreen",
}
SHIFT_COLORS = {
    "ID": "midnightblue",
    "Natural OOD": "darkorange",
    "Adversarial": "darkred",
}


def setup_style() -> None:
    sns.set_theme(context="paper", style="ticks", palette="dark")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["figure.titlesize"] = 16
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["lines.linewidth"] = 1.5
    plt.rcParams["patch.linewidth"] = 0
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.dpi"] = 180
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.transparent"] = False


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.14,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def finish_axis(ax: mpl.axes.Axes, *, offset: int = 7) -> None:
    sns.despine(ax=ax, trim=True, offset=offset)


def save_figure(fig: mpl.figure.Figure, stem: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.text(
        0.995,
        0.006,
        "LAYOUT MOCKUP — PLACEHOLDER VALUES",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="0.45",
        fontweight="bold",
    )
    for suffix in ("pdf", "png"):
        fig.savefig(
            PLOTS_DIR / f"{stem}.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
    plt.close(fig)


def draw_box(
    ax: mpl.axes.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 10,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.4,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )


def draw_arrow(
    ax: mpl.axes.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    connectionstyle: str = "arc3",
    linewidth: float = 2.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            color=color,
            linewidth=linewidth,
            connectionstyle=connectionstyle,
        )
    )


def make_fig1_framing() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    fig.suptitle("Closed-loop stimulation: biological and language-model systems", y=1.02)

    ax = axes[0]
    ax.set_title("Biological stimulation", fontsize=13)
    draw_box(ax, (0.08, 0.42), 0.23, 0.18, "Neural state\n" + r"$h_t$", facecolor="#d9f0f0", edgecolor="teal")
    draw_box(ax, (0.39, 0.70), 0.24, 0.16, "Controller", facecolor="#f2d9de", edgecolor="darkred")
    draw_box(ax, (0.69, 0.42), 0.23, 0.18, "Stimulation\n" + r"$u_t$", facecolor="#f8e4e4", edgecolor="darkred")
    draw_box(ax, (0.39, 0.10), 0.24, 0.16, "Behavior", facecolor="#eeeeee", edgecolor="0.3")
    draw_arrow(ax, (0.31, 0.52), (0.42, 0.70), color="teal")
    draw_arrow(ax, (0.63, 0.78), (0.78, 0.60), color="darkred")
    draw_arrow(ax, (0.69, 0.46), (0.60, 0.26), color="darkred")
    draw_arrow(ax, (0.39, 0.18), (0.22, 0.42), color="teal", connectionstyle="arc3,rad=-0.32")
    ax.text(0.50, 0.91, "observe → decide → stimulate", ha="center", fontsize=9, color="0.35")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "A")

    ax = axes[1]
    ax.set_title("Activation steering", fontsize=13)
    draw_box(ax, (0.04, 0.42), 0.19, 0.18, "Prompt /\ncontext", facecolor="#eeeeee", edgecolor="0.3")
    draw_box(ax, (0.30, 0.42), 0.21, 0.18, "Hidden state\n" + r"$h_\ell$", facecolor="#d9f0f0", edgecolor="teal")
    draw_box(ax, (0.58, 0.70), 0.27, 0.16, r"$H_\infty$ controller", facecolor="#f2d9de", edgecolor="darkred")
    draw_box(ax, (0.61, 0.40), 0.25, 0.18, "Intervention\n" + r"$u_\ell$", facecolor="#f8e4e4", edgecolor="darkred")
    draw_box(ax, (0.58, 0.10), 0.27, 0.16, "Next layers /\nbehavior", facecolor="#e4edfa", edgecolor="midnightblue")
    draw_arrow(ax, (0.23, 0.51), (0.30, 0.51), color="0.25")
    draw_arrow(ax, (0.51, 0.54), (0.62, 0.70), color="teal")
    draw_arrow(ax, (0.72, 0.70), (0.73, 0.58), color="darkred")
    draw_arrow(ax, (0.73, 0.40), (0.72, 0.26), color="darkred")
    draw_arrow(ax, (0.58, 0.18), (0.40, 0.42), color="teal", connectionstyle="arc3,rad=-0.30")
    ax.text(0.50, 0.91, "state feedback across transformer depth", ha="center", fontsize=9, color="0.35")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "B")

    fig.text(
        0.5,
        0.03,
        "Shared abstraction: observe the evolving state, compute feedback, and apply a targeted intervention.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95), w_pad=2.0)
    save_figure(fig, "fig1_framing")


def make_fig2_dynamics_and_residuals() -> None:
    rng = np.random.default_rng(2151)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4))
    fig.suptitle("Representation dynamics and residual failure modes", y=1.01)

    ax = axes[0, 0]
    depth = np.linspace(0, 1, 17)
    id_x = -1.1 + 2.6 * depth
    id_y = 0.25 * np.sin(2.2 * np.pi * depth)
    ood_x = id_x + 0.18 * depth
    ood_y = id_y + 0.65 * depth**1.7
    steered_y = id_y + 0.18 * depth
    ax.plot(id_x, id_y, color="0.45", marker="o", markersize=3, label="ID trajectory")
    ax.plot(ood_x, ood_y, color="darkred", marker="o", markersize=3, label="OOD trajectory")
    ax.plot(id_x, steered_y, color="midnightblue", linestyle="--", label="Controlled trajectory")
    ax.scatter([id_x[-1]], [steered_y[-1]], marker="*", s=100, color="black", label="Target")
    ax.set_xlabel("Reduced state coordinate 1")
    ax.set_ylabel("Reduced state coordinate 2")
    ax.set_title("Representation trajectories")
    ax.legend(fontsize=7, loc="best")
    panel_label(ax, "A")
    finish_axis(ax)

    ax = axes[0, 1]
    id_mean = 0.20 + 0.04 * np.sin(2 * np.pi * depth)
    natural = id_mean + 0.08 * depth + 0.012 * np.cos(3 * np.pi * depth)
    adversarial = id_mean + 0.15 * depth**1.5 + 0.018 * np.sin(4 * np.pi * depth)
    for name, values in (("ID", id_mean), ("Natural OOD", natural), ("Adversarial", adversarial)):
        color = SHIFT_COLORS[name]
        ax.plot(depth, values, color=color, label=name)
        ax.fill_between(depth, values - 0.015, values + 0.015, color=color, alpha=0.12)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 1])
    ax.set_xlabel("Normalized transformer depth")
    ax.set_ylabel("Mean relative residual")
    ax.set_title("Residual growth across depth")
    ax.legend(fontsize=8, loc="upper left")
    panel_label(ax, "B")
    finish_axis(ax)

    ax = axes[1, 0]
    conditions = ["ID", "Translation", "Long context", "Adversarial"]
    centers = [30, 36, 43, 51]
    distributions = [np.clip(rng.normal(center, 5 + 2 * index, 50), 0, 100) for index, center in enumerate(centers)]
    bp = ax.boxplot(
        distributions,
        tick_labels=conditions,
        patch_artist=True,
        widths=0.62,
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], ["midnightblue", "darkred", "darkred", "black"], strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.88)
    for index, values in enumerate(distributions, start=1):
        jitter = rng.normal(index, 0.045, len(values))
        ax.scatter(jitter, values, s=7, color="0.25", alpha=0.23)
    ax.axhline(100, color="0.35", linestyle="--", linewidth=1)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 100])
    ax.set_ylabel("Remaining target error (% of baseline)")
    ax.set_title("Frozen-controller failure by shift")
    ax.tick_params(axis="x", rotation=20)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    panel_label(ax, "C")
    finish_axis(ax)

    ax = axes[1, 1]
    model_size = np.array([0.08, 0.12, 0.35, 0.5, 0.77, 1.5, 3.0, 7.0, 14.0])
    early = np.array([0.04, 0.05, 0.03, 0.06, 0.02, 0.05, 0.04, 0.06, 0.04])
    middle = np.array([0.12, 0.08, 0.05, 0.09, 0.04, 0.07, 0.06, 0.05, 0.07])
    late = np.array([0.08, 0.03, 0.06, 0.05, 0.22, 0.10, 0.16, 0.08, 0.11])
    ax.plot(model_size, early, marker="o", color="teal", label="Early")
    ax.plot(model_size, middle, marker="o", color="darkorange", label="Middle")
    ax.plot(model_size, late, marker="o", color="purple", label="Late")
    ax.axhline(0, color="0.3", linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("Model parameters (billions)")
    ax.set_ylabel("OOD − ID residual")
    ax.set_title("Residual amplification across scale")
    ax.legend(fontsize=8, loc="best")
    panel_label(ax, "D")
    finish_axis(ax)

    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.96), h_pad=2.2, w_pad=2.2)
    save_figure(fig, "fig2_dynamics_and_residuals")


def make_fig3_robust_steerability() -> None:
    rng = np.random.default_rng(2151)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.5))
    fig.suptitle(r"Robust steerability as a prospective predictor", y=1.01)

    ax = axes[0, 0]
    steps = [
        (0.03, "Fit / calibration\nactivations", "#eeeeee", "0.35"),
        (0.28, "Identify\n" + r"$A_k,B_k,D_k$", "#d9f0f0", "teal"),
        (0.54, "Solve " + r"$H_\infty$" + "\nfor " + r"$\gamma^\star$", "#f2d9de", "darkred"),
        (0.80, r"$S_{\rm rob}=1/\gamma^\star$", "#e4edfa", "midnightblue"),
    ]
    for x, text, face, edge in steps:
        draw_box(ax, (x, 0.39), 0.17, 0.22, text, facecolor=face, edgecolor=edge, fontsize=7.5)
    for left, right in zip(steps[:-1], steps[1:], strict=True):
        draw_arrow(ax, (left[0] + 0.17, 0.50), (right[0], 0.50), color="0.25", linewidth=1.4)
    ax.text(0.50, 0.20, "No held-out steering outcomes used", ha="center", color="darkred", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Calibration-only computation")
    panel_label(ax, "A")

    ax = axes[0, 1]
    families = np.repeat(["GPT-2", "Qwen", "Llama"], 9)
    behaviors = np.tile(np.repeat(["Toxicity", "Truthfulness", "Refusal"], 3), 3)
    srob = np.linspace(0.12, 0.92, 27) + rng.normal(0, 0.035, 27)
    reliability = 22 + 68 * srob + rng.normal(0, 8, 27)
    family_colors = {"GPT-2": "midnightblue", "Qwen": "darkred", "Llama": "darkgreen"}
    behavior_markers = {"Toxicity": "o", "Truthfulness": "s", "Refusal": "^"}
    for family in family_colors:
        for behavior in behavior_markers:
            mask = (families == family) & (behaviors == behavior)
            ax.scatter(
                srob[mask],
                reliability[mask],
                color=family_colors[family],
                marker=behavior_markers[behavior],
                s=38,
                alpha=0.82,
            )
    fit = np.polyfit(srob, reliability, 1)
    xx = np.linspace(0.08, 0.98, 100)
    ax.plot(xx, np.polyval(fit, xx), color="black", linestyle="--")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=color, label=family, markersize=7) for family, color in family_colors.items()]
    handles += [Line2D([0], [0], marker=marker, color="0.25", linestyle="None", label=behavior, markersize=7) for behavior, marker in behavior_markers.items()]
    ax.legend(handles=handles, fontsize=7, ncol=2, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 100])
    ax.set_xlabel(r"Robust steerability $S_{\rm rob}=1/\gamma^\star$")
    ax.set_ylabel("Held-out OOD steering reliability")
    ax.set_title("The money plot")
    panel_label(ax, "B")
    finish_axis(ax)

    ax = axes[1, 0]
    predictors = ["Scale", "Probe", "Residual", "LQR cost", r"$S_{\rm rob}$"]
    scores = [0.09, 0.18, 0.23, 0.27, 0.52]
    colors = ["0.65", "0.65", "0.65", "0.65", "darkred"]
    ax.bar(np.arange(len(predictors)), scores, color=colors)
    ax.axhline(0, color="0.2", linewidth=0.8)
    ax.set_ylim(0, 0.60)
    ax.set_yticks([0, 0.60])
    ax.set_xticks(np.arange(len(predictors)), predictors, rotation=25, ha="right")
    ax.set_ylabel(r"Cross-validated $R^2$")
    ax.set_title("Prediction beyond simpler quantities")
    panel_label(ax, "C")
    finish_axis(ax)

    ax = axes[1, 1]
    observed = np.linspace(15, 90, 18) + rng.normal(0, 3, 18)
    predicted = observed + rng.normal(0, 10, 18)
    group = np.repeat(["Held-out GPT-2", "Held-out Qwen", "Held-out Llama"], 6)
    group_colors = {
        "Held-out GPT-2": "midnightblue",
        "Held-out Qwen": "darkred",
        "Held-out Llama": "darkgreen",
    }
    for name, color in group_colors.items():
        mask = group == name
        ax.scatter(observed[mask], predicted[mask], color=color, label=name, s=34)
    ax.plot([0, 100], [0, 100], color="black", linestyle="--")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 100])
    ax.set_yticks([0, 100])
    ax.set_xlabel("Observed OOD reliability")
    ax.set_ylabel("Predicted OOD reliability")
    ax.set_title("Leave-one-family-out prediction")
    ax.legend(fontsize=7, loc="upper left")
    panel_label(ax, "D")
    finish_axis(ax)

    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.96), h_pad=2.2, w_pad=2.2)
    save_figure(fig, "fig3_robust_steerability")


def grouped_bars(
    ax: mpl.axes.Axes,
    model_labels: list[str],
    values: dict[str, list[float]],
    *,
    ylabel: str,
    title: str,
) -> None:
    x = np.arange(len(model_labels))
    width = 0.19
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(values))
    for offset, (method, method_values) in zip(offsets, values.items(), strict=True):
        ax.bar(x + offset, method_values, width=width, color=METHOD_COLORS[method], label=method)
    ax.set_xticks(x, model_labels, rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 100])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    finish_axis(ax)


def make_fig4_id_controller_benchmark() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4))
    fig.suptitle("In-distribution controller benchmark across model scale", y=1.01)
    models = ["DistilGPT-2", "Qwen-0.5B", "Qwen-1.5B", "Qwen-3B"]

    truthfulness = {
        "Original": [32, 38, 43, 48],
        "A-LQR": [40, 45, 52, 58],
        "S-PID": [44, 49, 55, 61],
        r"$H_\infty$": [41, 47, 56, 64],
    }
    grouped_bars(axes[0, 0], models, truthfulness, ylabel="Truthfulness × informativeness", title="Truthfulness")
    panel_label(axes[0, 0], "A")

    toxicity = {
        "Original": [0, 0, 0, 0],
        "A-LQR": [38, 45, 52, 56],
        "S-PID": [30, 39, 44, 49],
        r"$H_\infty$": [40, 48, 57, 62],
    }
    grouped_bars(axes[0, 1], models, toxicity, ylabel="Toxicity reduction (%)", title="ID toxicity reduction")
    panel_label(axes[0, 1], "B")

    utility = {
        "Original": [100, 100, 100, 100],
        "A-LQR": [86, 88, 91, 93],
        "S-PID": [94, 95, 96, 97],
        r"$H_\infty$": [90, 92, 94, 95],
    }
    grouped_bars(axes[1, 0], models, utility, ylabel="Utility retained (%)", title="Collateral capability retention")
    panel_label(axes[1, 0], "C")

    ax = axes[1, 1]
    pareto = {
        "Original": (0.0, 20),
        "A-LQR": (0.72, 62),
        "S-PID": (0.50, 55),
        r"$H_\infty$": (0.58, 68),
    }
    for method, (energy, reliability) in pareto.items():
        ax.scatter(energy, reliability, color=METHOD_COLORS[method], s=75, label=method)
    ax.set_xlim(-0.05, 0.85)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 0.8])
    ax.set_yticks([0, 100])
    ax.set_xlabel("Normalized intervention energy")
    ax.set_ylabel("ID steering reliability")
    ax.set_title("Performance–intervention tradeoff")
    panel_label(ax, "D")
    finish_axis(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4, fontsize=9)
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.91), h_pad=2.2, w_pad=2.2)
    save_figure(fig, "fig4_id_controller_benchmark")


def make_fig5_ood_controller_benchmark() -> None:
    rng = np.random.default_rng(2151)
    fig, axes = plt.subplots(2, 2, figsize=(10.9, 7.5))
    fig.suptitle(r"OOD controller benchmark and robust-control crossover", y=1.01)

    ax = axes[0, 0]
    conditions = ["ID", "Dataset", "Translate", "Long ctx", "Adversarial"]
    x = np.arange(len(conditions))
    curves = {
        "Original": [20, 18, 16, 15, 14],
        "A-LQR": [74, 67, 57, 43, 35],
        "S-PID": [70, 65, 58, 49, 42],
        r"$H_\infty$": [72, 69, 65, 61, 57],
    }
    for method, values in curves.items():
        ax.plot(x, values, marker="o", color=METHOD_COLORS[method], label=method)
    ax.set_xticks(x, conditions, rotation=22, ha="right")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 100])
    ax.set_ylabel("Steering reliability")
    ax.set_title("ID-to-OOD degradation")
    ax.legend(fontsize=8, loc="lower left")
    panel_label(ax, "A")
    finish_axis(ax)

    ax = axes[0, 1]
    model_labels = ["DistilGPT-2", "Qwen-0.5B", "Qwen-1.5B", "Qwen-3B"]
    heat = np.array(
        [
            [64, 57, 48, 39],
            [69, 63, 56, 50],
            [72, 68, 63, 58],
            [76, 72, 68, 64],
        ]
    )
    hm = sns.heatmap(
        heat,
        ax=ax,
        cmap="Greys",
        vmin=0,
        vmax=100,
        xticklabels=conditions[1:],
        yticklabels=model_labels,
        cbar_kws={"shrink": 0.75, "ticks": [0, 100]},
    )
    hm.collections[0].colorbar.ax.set_ylabel(r"$H_\infty$ reliability", rotation=270, labelpad=15)
    ax.set_xlabel("OOD condition")
    ax.set_ylabel("")
    ax.set_title(r"Complete $H_\infty$ benchmark matrix")
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.tick_params(axis="y", labelsize=7)
    for label in ax.get_yticklabels():
        label.set_rotation(0)
        label.set_ha("right")
        label.set_va("center")
    panel_label(ax, "B")

    ax = axes[1, 0]
    mismatch = np.linspace(0, 1, 14)
    lqr = 78 - 52 * mismatch + rng.normal(0, 2, len(mismatch))
    hinf = 76 - 25 * mismatch + rng.normal(0, 2, len(mismatch))
    ax.plot(mismatch, lqr, color="midnightblue", marker="o", markersize=3, label="A-LQR")
    ax.plot(mismatch, hinf, color="darkgreen", marker="o", markersize=3, label=r"$H_\infty$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 100])
    ax.set_xlabel("Measured dynamics mismatch")
    ax.set_ylabel("Steering reliability")
    ax.set_title("Robust-control crossover")
    ax.legend(fontsize=8, loc="lower left")
    panel_label(ax, "C")
    finish_axis(ax)

    ax = axes[1, 1]
    gamma_star = np.linspace(0.5, 3.0, 22)
    advantage = -3 + 10 * gamma_star + rng.normal(0, 4, len(gamma_star))
    behavior = np.tile(["Toxicity", "Truthfulness", "Refusal"], 8)[: len(gamma_star)]
    marker_map = {"Toxicity": "o", "Truthfulness": "s", "Refusal": "^"}
    for name, marker in marker_map.items():
        mask = behavior == name
        ax.scatter(gamma_star[mask], advantage[mask], marker=marker, s=38, label=name, color="darkred", alpha=0.75)
    fit = np.polyfit(gamma_star, advantage, 1)
    ax.plot(gamma_star, np.polyval(fit, gamma_star), color="black", linestyle="--")
    ax.axhline(0, color="0.4", linestyle=":")
    ax.set_xlabel(r"Fragility $\gamma^\star$ (higher = worse)")
    ax.set_ylabel(r"$H_\infty$ − A-LQR reliability")
    ax.set_title("Does predicted fragility identify benefit?")
    ax.legend(fontsize=7, loc="upper left")
    panel_label(ax, "D")
    finish_axis(ax)

    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.96), h_pad=2.2, w_pad=2.2)
    save_figure(fig, "fig5_ood_controller_benchmark")


def make_fig6_long_context_prediction() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.65))
    fig.suptitle(r"Long-context stress test for robust activation control", y=1.05)
    context = np.array([128, 512, 2048, 8192, 32768])
    labels = ["128", "512", "2k", "8k", "32k"]

    ax = axes[0]
    lqr_error = [28, 32, 43, 61, 78]
    hinf_error = [30, 32, 37, 43, 50]
    ax.plot(context, lqr_error, marker="o", color="midnightblue", label="A-LQR")
    ax.plot(context, hinf_error, marker="o", color="darkgreen", label=r"$H_\infty$")
    ax.set_xscale("log")
    ax.set_xticks(context, labels)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 100])
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Remaining target error (%)")
    ax.set_title("Steering degradation")
    ax.legend(fontsize=8, loc="upper left")
    panel_label(ax, "A")
    finish_axis(ax)

    ax = axes[1]
    lqr_utility = [96, 94, 89, 77, 61]
    hinf_utility = [95, 95, 93, 90, 86]
    ax.plot(context, lqr_utility, marker="o", color="midnightblue")
    ax.plot(context, hinf_utility, marker="o", color="darkgreen")
    ax.set_xscale("log")
    ax.set_xticks(context, labels)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 100])
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Utility retained (%)")
    ax.set_title("Collateral behavior")
    panel_label(ax, "B")
    finish_axis(ax)

    ax = axes[2]
    advantage = np.array(lqr_error) - np.array(hinf_error)
    ax.plot(context, advantage, marker="o", color="darkred")
    ax.axhline(0, color="0.4", linestyle=":")
    ax.fill_between(context, 0, advantage, where=advantage >= 0, color="darkred", alpha=0.12)
    ax.set_xscale("log")
    ax.set_xticks(context, labels)
    bound = max(abs(float(advantage.min())), abs(float(advantage.max()))) + 3
    ax.set_ylim(-bound, bound)
    ax.set_yticks([-round(bound), round(bound)])
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel(r"A-LQR error − $H_\infty$ error")
    ax.set_title(r"$H_\infty$ advantage")
    panel_label(ax, "C")
    finish_axis(ax)

    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.92), w_pad=2.0)
    save_figure(fig, "fig6_long_context_prediction")


def make_figs1_hinf_validation() -> None:
    rng = np.random.default_rng(2151)
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.3))
    fig.suptitle(r"Validation of finite-horizon $H_\infty$ synthesis", y=1.01)

    ax = axes[0, 0]
    gamma = np.linspace(0.5, 2.0, 120)
    margin = gamma**2 - 1.0
    ax.plot(gamma, margin, color="darkred")
    ax.axhline(0, color="0.3", linestyle=":")
    ax.axvline(1.0, color="black", linestyle="--", label=r"$\gamma^\star$")
    ax.fill_between(gamma, 0, margin, where=margin >= 0, color="darkgreen", alpha=0.13)
    ax.set_xlabel(r"Candidate attenuation $\gamma$")
    ax.set_ylabel("Minimum feasibility margin")
    ax.set_title("Saddle-point feasibility")
    ax.legend(fontsize=8)
    panel_label(ax, "A")
    finish_axis(ax)

    ax = axes[0, 1]
    exact = np.linspace(0.4, 2.8, 18)
    synthesized = exact + rng.normal(0, 0.07, len(exact))
    ax.scatter(exact, synthesized, color="midnightblue", s=35)
    ax.plot([0, 3], [0, 3], color="black", linestyle="--")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xticks([0, 3])
    ax.set_yticks([0, 3])
    ax.set_xlabel("Exact finite-horizon gain")
    ax.set_ylabel(r"Synthesized $\gamma^\star$")
    ax.set_title("Independent numerical agreement")
    panel_label(ax, "B")
    finish_axis(ax)

    ax = axes[1, 0]
    disturbance_energy = np.linspace(0.05, 1.0, 25)
    gamma_bound = 1.35
    output_energy = (gamma_bound**2) * disturbance_energy * rng.uniform(0.35, 0.90, len(disturbance_energy))
    ax.scatter(disturbance_energy, output_energy, color="darkgreen", s=30)
    ax.plot(disturbance_energy, (gamma_bound**2) * disturbance_energy, color="black", linestyle="--", label=r"$\gamma^{\star 2}\|w\|^2$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 2])
    ax.set_xlabel("Disturbance energy")
    ax.set_ylabel("Performance-output energy")
    ax.set_title("Induced-gain bound")
    ax.legend(fontsize=8)
    panel_label(ax, "C")
    finish_axis(ax)

    ax = axes[1, 1]
    certified = np.linspace(0.1, 1.0, 30)
    observed = certified * rng.uniform(0.45, 0.98, len(certified))
    ax.scatter(certified, observed, color="darkred", s=28)
    ax.plot([0, 1.05], [0, 1.05], color="black", linestyle="--")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel("Certified tracking-error bound")
    ax.set_ylabel("Observed tracking error")
    ax.set_title("Certificate coverage")
    panel_label(ax, "D")
    finish_axis(ax)

    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.96), h_pad=2.2, w_pad=2.2)
    save_figure(fig, "figs1_hinf_validation")


def make_figs2_nonlinearity_validation() -> None:
    rng = np.random.default_rng(2151)
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.3))
    fig.suptitle("Local linearity and nonlinear excursion in representation space", y=1.01)

    ax = axes[0, 0]
    excursion = np.linspace(0, 1, 80)
    residual = 0.02 + 0.55 * excursion**2 + rng.normal(0, 0.018, len(excursion))
    ax.scatter(excursion, residual, s=12, color="darkred", alpha=0.55)
    ax.plot(excursion, 0.02 + 0.55 * excursion**2, color="black", linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 1])
    ax.set_xlabel("Distance from nominal trajectory")
    ax.set_ylabel("Linearization residual")
    ax.set_title("Error grows with nonlinear excursion")
    panel_label(ax, "A")
    finish_axis(ax)

    ax = axes[0, 1]
    predicted = np.linspace(-1, 1, 90)
    noise_scale = 0.05 + 0.18 * np.abs(predicted)
    measured = predicted + rng.normal(0, noise_scale)
    ax.scatter(predicted, measured, s=12, color="midnightblue", alpha=0.55)
    ax.plot([-1.2, 1.2], [-1.2, 1.2], color="black", linestyle="--")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xticks([-1, 1])
    ax.set_yticks([-1, 1])
    ax.set_xlabel("Linear-model prediction")
    ax.set_ylabel("Observed state displacement")
    ax.set_title("Predicted versus observed dynamics")
    panel_label(ax, "B")
    finish_axis(ax)

    ax = axes[1, 0]
    depth = np.linspace(0, 1, 17)
    strengths = [0.25, 0.5, 1.0, 2.0]
    cmap = mpl.colors.ListedColormap(mpl.colormaps["RdYlBu_r"](np.linspace(0.08, 0.92, len(strengths))))
    norm = mpl.colors.Normalize(vmin=min(strengths), vmax=max(strengths))
    for strength in strengths:
        profile = 0.03 + 0.04 * depth + 0.10 * strength**1.5 * depth**2
        ax.plot(depth, profile, color=cmap(norm(strength)))
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 1])
    ax.set_xlabel("Normalized transformer depth")
    ax.set_ylabel("Relative residual")
    ax.set_title("Residual by intervention strength")
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.70, ticks=[min(strengths), max(strengths)])
    cbar.set_label("Steering strength", rotation=270, labelpad=14)
    panel_label(ax, "C")
    finish_axis(ax)

    ax = axes[1, 1]
    coverage = 100 - 45 * excursion**1.7
    corrected = 100 - 18 * excursion**1.5
    ax.plot(excursion, coverage, color="midnightblue", label="Linear certificate")
    ax.plot(excursion, corrected, color="darkgreen", label="Nonlinear correction")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 100])
    ax.set_xlabel("Distance from nominal trajectory")
    ax.set_ylabel("Rollouts satisfying bound (%)")
    ax.set_title("Certificate validity region")
    ax.legend(fontsize=8, loc="lower left")
    panel_label(ax, "D")
    finish_axis(ax)

    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.96), h_pad=2.2, w_pad=2.2)
    save_figure(fig, "figs2_nonlinearity_validation")


def make_figs3_tuning_and_ablations() -> None:
    rng = np.random.default_rng(2151)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.5))
    fig.suptitle("Gain selection and robustness ablations", y=1.01)

    gain = np.array([0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 2.0])

    ax = axes[0, 0]
    target_lqr = np.array([18, 35, 53, 68, 77, 82, 84, 85])
    target_hinf = np.array([18, 32, 49, 63, 73, 79, 82, 84])
    ax.plot(gain, target_lqr, marker="o", color="midnightblue", label="A-LQR")
    ax.plot(gain, target_hinf, marker="o", color="darkgreen", label=r"$H_\infty$")
    ax.axvspan(0.55, 0.90, color="0.85", alpha=0.65, label="selected range")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 100])
    ax.set_xlabel("Gain multiplier")
    ax.set_ylabel("Target steering success (%)")
    ax.set_title("Gain sweep: target response")
    ax.legend(fontsize=7, loc="lower right")
    panel_label(ax, "A")
    finish_axis(ax)

    ax = axes[0, 1]
    utility_lqr = np.array([100, 98, 94, 86, 70, 48, 28, 8])
    utility_hinf = np.array([100, 99, 97, 92, 84, 72, 59, 34])
    energy_lqr = np.array([0, 4, 11, 23, 38, 55, 72, 100])
    energy_hinf = np.array([0, 3, 9, 18, 30, 43, 57, 82])
    ax.plot(gain, utility_lqr, marker="o", color="midnightblue", label="A-LQR utility")
    ax.plot(gain, utility_hinf, marker="o", color="darkgreen", label=r"$H_\infty$ utility")
    ax.plot(gain, energy_lqr, color="midnightblue", linestyle="--", label="A-LQR energy")
    ax.plot(gain, energy_hinf, color="darkgreen", linestyle="--", label=r"$H_\infty$ energy")
    ax.axvspan(0.55, 0.90, color="0.85", alpha=0.65)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 100])
    ax.set_xlabel("Gain multiplier")
    ax.set_ylabel("Utility retained / normalized energy (%)")
    ax.set_title("Oversteering and intervention cost")
    ax.legend(fontsize=6.8, loc="center left")
    panel_label(ax, "B")
    finish_axis(ax)

    ax = axes[1, 0]
    n_cal = np.array([16, 32, 64, 128, 256])
    gamma_error = np.array([34, 25, 17, 11, 7])
    predictive_r2 = np.array([0.10, 0.22, 0.36, 0.48, 0.55])
    line1 = ax.plot(n_cal, gamma_error, marker="o", color="darkred", label=r"$\gamma^\star$ error")
    ax.set_xscale("log", base=2)
    ax.set_xlim(14, 290)
    ax.set_ylim(0, 40)
    ax.set_xticks(n_cal, [str(value) for value in n_cal])
    ax.set_yticks([0, 40])
    ax.set_xlabel("Calibration prompts")
    ax.set_ylabel(r"Relative error in $\gamma^\star$ (%)", color="darkred")
    ax.tick_params(axis="y", colors="darkred")
    twin = ax.twinx()
    line2 = twin.plot(n_cal, predictive_r2, marker="s", color="darkgreen", label=r"OOD prediction $R^2$")
    twin.set_ylim(0, 0.60)
    twin.set_yticks([0, 0.60])
    twin.set_ylabel(r"Held-out prediction $R^2$", color="darkgreen")
    twin.tick_params(axis="y", colors="darkgreen")
    twin.spines["top"].set_visible(False)
    ax.set_title("Calibration-size ablation")
    ax.legend(line1 + line2, [line.get_label() for line in line1 + line2], fontsize=7, loc="center right")
    panel_label(ax, "C")
    finish_axis(ax)

    ax = axes[1, 1]
    geometries = ["Isotropic\n" + r"$D=I$", "Learned\nresidual", "Target-\naligned", "Worst-\ncase"]
    centers = np.array([2, 8, 15, 24])
    samples = [rng.normal(center, 4 + idx, 45) for idx, center in enumerate(centers)]
    bp = ax.boxplot(samples, tick_labels=geometries, patch_artist=True, widths=0.62, showfliers=False)
    for patch, color in zip(bp["boxes"], ["0.45", "darkorange", "darkred", "black"], strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.88)
    for index, values in enumerate(samples, start=1):
        jitter = rng.normal(index, 0.045, len(values))
        ax.scatter(jitter, values, s=7, color="0.25", alpha=0.20)
    ax.axhline(0, color="0.4", linestyle=":")
    ax.set_ylabel(r"$H_\infty$ − A-LQR reliability")
    ax.set_title("Disturbance-geometry ablation")
    panel_label(ax, "D")
    finish_axis(ax)

    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.96), h_pad=2.2, w_pad=2.8)
    save_figure(fig, "figs3_tuning_and_ablations")


def make_overview() -> None:
    stems = [
        ("fig1_framing", "Figure 1 — Framing"),
        ("fig2_dynamics_and_residuals", "Figure 2 — Dynamics and residuals"),
        ("fig3_robust_steerability", "Figure 3 — Robust steerability"),
        ("fig4_id_controller_benchmark", "Figure 4 — ID benchmark"),
        ("fig5_ood_controller_benchmark", "Figure 5 — OOD benchmark"),
        ("fig6_long_context_prediction", "Figure 6 — Long context"),
        ("figs1_hinf_validation", "Figure S1 — H∞ validation"),
        ("figs2_nonlinearity_validation", "Figure S2 — Nonlinearity"),
        ("figs3_tuning_and_ablations", "Figure S3 — Tuning and ablations"),
    ]
    fig = plt.figure(figsize=(12, 20))
    grid = fig.add_gridspec(5, 2)
    fig.suptitle("Proposed paper figure arrangement", y=0.995, fontsize=20)
    axes = [fig.add_subplot(grid[row, column]) for row in range(4) for column in range(2)]
    axes.append(fig.add_subplot(grid[4, :]))
    for ax, (stem, title) in zip(axes, stems, strict=True):
        image = plt.imread(PLOTS_DIR / f"{stem}.png")
        ax.imshow(image)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=1.6)
    save_figure(fig, "figure_arrangement_overview")


def main() -> None:
    setup_style()
    make_fig1_framing()
    make_fig2_dynamics_and_residuals()
    make_fig3_robust_steerability()
    make_fig4_id_controller_benchmark()
    make_fig5_ood_controller_benchmark()
    make_fig6_long_context_prediction()
    make_figs1_hinf_validation()
    make_figs2_nonlinearity_validation()
    make_figs3_tuning_and_ablations()
    make_overview()


if __name__ == "__main__":
    main()
