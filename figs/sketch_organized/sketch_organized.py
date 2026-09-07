"""Assemble the paper sketch with real results wherever they are available."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


UNIT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UNIT_DIR.parents[1]
PLOTS_DIR = UNIT_DIR / "plots"

TRUTH_RESULTS = PROJECT_ROOT / "parking/erfan_truthfulness_benchmark/plots/results.csv"
ID_RESULTS = PROJECT_ROOT / "parking/erfan_id_toxicity_benchmark/plots/results.csv"
OOD_RESULTS = PROJECT_ROOT / "parking/erfan_ood_steering_benchmark/plots/results.csv"
OOD_FAILURE_RESULTS = PROJECT_ROOT / "parking/ood_explore/plots/lqr_ood_failure_metrics.csv"
ADVERSARIAL_RESULTS = (
    PROJECT_ROOT / "parking/ood_adversarial/plots/adversarial_attempt_metrics.csv"
)
DISTIL_OOD_RESIDUALS = (
    PROJECT_ROOT
    / "parking/erfan_ood_target_selection/plots/selected_ood_targets_global3_distilgpt2.csv"
)
QWEN05_OOD_RESIDUALS = (
    PROJECT_ROOT
    / "parking/erfan_ood_target_selection/plots/selected_ood_targets_global3_qwen05b.csv"
)
HINF_EQUIVALENCE = PROJECT_ROOT / "parking/h_infinity_optimization/plots/equivalence.csv"
HINF_BENCHMARK = PROJECT_ROOT / "parking/h_infinity_optimization/plots/benchmark_summary.csv"

PANEL_STATUS = [
    ("fig1_framing", "A-B", "conceptual", "No experimental values required"),
    ("fig2_dynamics_and_residuals", "A-D", "real", "Existing dynamics, residual, and OOD analyses"),
    ("fig3_robust_steerability", "A", "conceptual", "Calibration-only workflow"),
    ("fig3_robust_steerability", "B-D", "placeholder", "Prospective prediction experiment missing"),
    ("fig4_id_controller_benchmark", "A-C", "real", "Current 50-record benchmark"),
    ("fig4_id_controller_benchmark", "D", "placeholder", "Intervention-energy comparison missing"),
    ("fig5_ood_controller_benchmark", "A-B", "real", "Five-model, 50-prompt OOD benchmark"),
    ("fig5_ood_controller_benchmark", "C-D", "real_preliminary", "Toxicity-only crossover; final experiment missing"),
    ("fig6_long_context_prediction", "A-C", "placeholder", "Context-length sweep missing"),
    ("figs1_hinf_validation", "A-D", "real", "H-infinity parity and optimization diagnostics"),
    ("figs2_nonlinearity_validation", "A", "placeholder", "Excursion analysis missing"),
    ("figs2_nonlinearity_validation", "B", "real", "Existing actual-versus-linearized layer trajectory"),
    ("figs2_nonlinearity_validation", "C-D", "placeholder", "Strength and coverage analyses missing"),
    ("figs3_tuning_and_ablations", "A-D", "placeholder", "Gain, calibration-size, and geometry sweeps missing"),
]

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
    for suffix in ("pdf", "png"):
        fig.savefig(
            PLOTS_DIR / f"{stem}.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
    plt.close(fig)


def status_badge(ax: mpl.axes.Axes, text: str, *, real: bool) -> None:
    ax.text(
        0.99,
        0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="darkgreen" if real else "0.35",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
        zorder=20,
    )


def source_image_panel(ax: mpl.axes.Axes, path: Path, label: str) -> None:
    ax.imshow(plt.imread(path))
    ax.axis("off")
    panel_label(ax, label)
    status_badge(ax, "REAL DATA", real=True)


def mark_placeholder(ax: mpl.axes.Axes) -> None:
    status_badge(ax, "PLACEHOLDER — EXPERIMENT MISSING", real=False)


def write_panel_status() -> None:
    table = pd.DataFrame(PANEL_STATUS, columns=["figure", "panels", "status", "basis"])
    table.to_csv(PLOTS_DIR / "panel_status.csv", index=False, lineterminator="\n")


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

    source_image_panel(
        axes[0, 0],
        PROJECT_ROOT / "parking/erfan_linearization_error/plots/paper_hidden_state_trajectories.png",
        "A",
    )
    source_image_panel(
        axes[0, 1],
        PROJECT_ROOT / "parking/residual_checks/plots/layer_residuals.png",
        "B",
    )

    ax = axes[1, 0]
    failure = pd.read_csv(OOD_FAILURE_RESULTS)
    adversarial = pd.read_csv(ADVERSARIAL_RESULTS)
    conditions = ["id", "translation", "long_context_switch"]
    labels = ["ID", "Spanish", "Long context", "Adversarial"]
    distributions = [
        failure.loc[failure["condition"] == condition, "remaining_error_pct"].to_numpy()
        for condition in conditions
    ]
    distributions.append(
        adversarial.loc[
            adversarial["attempt"] == "positive_dispersion",
            "remaining_error_pct",
        ].to_numpy()
    )
    bp = ax.boxplot(
        distributions,
        tick_labels=labels,
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
    status_badge(ax, "REAL DATA · 50 PROMPTS", real=True)
    finish_axis(ax)

    source_image_panel(
        axes[1, 1],
        PROJECT_ROOT
        / "parking/erfan_model_scale_residuals/plots/"
        "network_size_residual_scaling_camera_ready_adversarial.png",
        "D",
    )

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
    mark_placeholder(ax)
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
    mark_placeholder(ax)
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
    mark_placeholder(ax)
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
    upper: float = 100,
) -> None:
    x = np.arange(len(model_labels))
    width = 0.19
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(values))
    for offset, (method, method_values) in zip(offsets, values.items(), strict=True):
        ax.bar(x + offset, method_values, width=width, color=METHOD_COLORS[method], label=method)
    ax.set_xticks(x, model_labels, rotation=20, ha="right")
    ax.set_ylim(0, upper)
    ax.set_yticks([0, 100])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    finish_axis(ax)


def make_fig4_id_controller_benchmark() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4))
    fig.suptitle("In-distribution controller benchmark across model scale", y=1.01)
    truth = pd.read_csv(TRUTH_RESULTS)
    toxicity = pd.read_csv(ID_RESULTS)
    model_names = list(dict.fromkeys(truth["model"].tolist()))
    models = ["D", "0.5", "1.5", "7", "14"]
    methods = [
        ("Original", "Original"),
        ("A-LQR", "A-LQR"),
        ("S-PID", "S-PID"),
        ("H-infinity", r"$H_\infty$"),
    ]

    truthfulness = {
        display: [
            float(
                truth.loc[
                    (truth["model"] == model) & (truth["method_label"] == source),
                    "ti_mean",
                ].iloc[0]
            )
            for model in model_names
        ]
        for source, display in methods
    }
    grouped_bars(
        axes[0, 0],
        models,
        truthfulness,
        ylabel=r"Truthful $\times$ informative (\%)",
        title="Truthfulness",
    )
    panel_label(axes[0, 0], "A")
    status_badge(axes[0, 0], "REAL DATA · 50 PROMPTS", real=True)

    toxicity_reduction = {
        display: [
            -float(
                toxicity.loc[
                    (toxicity["model"] == model)
                    & (toxicity["method_label"] == source),
                    "percent_change",
                ].iloc[0]
            )
            for model in model_names
        ]
        for source, display in methods
    }
    grouped_bars(
        axes[0, 1],
        models,
        toxicity_reduction,
        ylabel=r"Toxicity reduction (\%)",
        title="ID toxicity reduction",
    )
    panel_label(axes[0, 1], "B")
    status_badge(axes[0, 1], "REAL DATA · 50 PROMPTS", real=True)

    original_mmlu = {
        model: float(
            truth.loc[
                (truth["model"] == model) & (truth["method_label"] == "Original"),
                "mmlu_mean",
            ].iloc[0]
        )
        for model in model_names
    }
    utility = {
        display: [
            100
            * float(
                truth.loc[
                    (truth["model"] == model) & (truth["method_label"] == source),
                    "mmlu_mean",
                ].iloc[0]
            )
            / original_mmlu[model]
            for model in model_names
        ]
        for source, display in methods
    }
    grouped_bars(
        axes[1, 0],
        models,
        utility,
        ylabel=r"MMLU retained (\%)",
        title="Collateral capability retention",
        upper=110,
    )
    panel_label(axes[1, 0], "C")
    status_badge(axes[1, 0], "REAL DATA · 50 QUESTIONS", real=True)

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
    mark_placeholder(ax)
    finish_axis(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4, fontsize=9)
    fig.text(0.5, 0.012, "Qwen scale (B); D = DistilGPT-2", ha="center", fontsize=8)
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.91), h_pad=2.2, w_pad=2.2)
    save_figure(fig, "fig4_id_controller_benchmark")


def make_fig5_ood_controller_benchmark() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.9, 7.5))
    fig.suptitle(r"OOD controller benchmark and robust-control crossover", y=1.01)
    ood = pd.read_csv(OOD_RESULTS)
    id_results = pd.read_csv(ID_RESULTS)
    id_results = id_results.assign(subset="rtp_random")
    combined = pd.concat([id_results, ood], ignore_index=True, sort=False)
    conditions = ["rtp_random", "jigsaw_long", "toxicchat_long", "mmlu_ood_other_concepts"]
    condition_labels = ["ID RTP", "Jigsaw", "ToxicChat", "MMLU shift"]
    methods = [
        ("Original", "Original"),
        ("A-LQR", "A-LQR"),
        ("S-PID", "S-PID"),
        ("H-infinity", r"$H_\infty$"),
    ]

    ax = axes[0, 0]
    x = np.arange(len(conditions))
    for source, display in methods:
        ratios_by_condition = []
        for condition in conditions:
            condition_rows = combined[combined["subset"] == condition]
            ratios = []
            for model in dict.fromkeys(condition_rows["model"].tolist()):
                original = float(
                    condition_rows.loc[
                        (condition_rows["model"] == model)
                        & (condition_rows["method_label"] == "Original"),
                        "toxicity_mean",
                    ].iloc[0]
                )
                steered = float(
                    condition_rows.loc[
                        (condition_rows["model"] == model)
                        & (condition_rows["method_label"] == source),
                        "toxicity_mean",
                    ].iloc[0]
                )
                ratios.append(steered / original)
            ratios_by_condition.append(np.asarray(ratios))
        medians = np.asarray([np.median(values) for values in ratios_by_condition])
        lower = np.asarray([np.quantile(values, 0.25) for values in ratios_by_condition])
        upper = np.asarray([np.quantile(values, 0.75) for values in ratios_by_condition])
        color = METHOD_COLORS[display]
        ax.plot(x, medians, marker="o", color=color, label=display)
        ax.fill_between(x, lower, upper, color=color, alpha=0.10)
        for index, values in enumerate(ratios_by_condition):
            ax.scatter(
                np.full(len(values), index),
                values,
                color=color,
                s=10,
                alpha=0.28,
            )
    ax.axhline(1, color="0.35", linestyle=":")
    ax.set_yscale("log")
    ax.set_xticks(x, condition_labels, rotation=22, ha="right")
    ax.set_ylabel("Post-steering / Original toxicity")
    ax.set_title("ID-to-OOD transfer")
    ax.legend(fontsize=8, loc="lower left")
    panel_label(ax, "A")
    status_badge(ax, "REAL DATA · 50/CELL", real=True)
    finish_axis(ax)

    ax = axes[0, 1]
    model_names = list(dict.fromkeys(ood["model"].tolist()))
    model_labels = ["D", "0.5", "1.5", "7", "14"]
    ood_conditions = conditions[1:]
    heat = np.empty((len(model_names), len(ood_conditions)))
    annotations = np.empty_like(heat, dtype=object)
    for row_index, model in enumerate(model_names):
        for column_index, condition in enumerate(ood_conditions):
            rows = ood[(ood["model"] == model) & (ood["subset"] == condition)]
            original = float(rows.loc[rows["method_label"] == "Original", "toxicity_mean"].iloc[0])
            hinf = float(rows.loc[rows["method_label"] == "H-infinity", "toxicity_mean"].iloc[0])
            ratio = hinf / original
            heat[row_index, column_index] = np.log10(ratio)
            annotations[row_index, column_index] = f"{ratio:.2g}×"
    hm = sns.heatmap(
        heat,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-3,
        vmax=3,
        annot=annotations,
        fmt="",
        annot_kws={"fontsize": 7},
        xticklabels=condition_labels[1:],
        yticklabels=model_labels,
        cbar_kws={"shrink": 0.75, "ticks": [-3, 0, 3]},
    )
    hm.collections[0].colorbar.ax.set_ylabel(
        r"$\log_{10}(H_\infty / \mathrm{Original})$ toxicity",
        rotation=270,
        labelpad=15,
    )
    ax.set_xlabel("OOD condition")
    ax.set_ylabel("Qwen scale (B); D = DistilGPT-2")
    ax.set_title(r"$H_\infty$ transfer matrix")
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.tick_params(axis="y", labelsize=7)
    for label in ax.get_yticklabels():
        label.set_rotation(0)
        label.set_ha("right")
        label.set_va("center")
    panel_label(ax, "B")
    status_badge(ax, "REAL DATA · 50/CELL", real=True)

    ax = axes[1, 0]
    residuals = pd.concat(
        [pd.read_csv(DISTIL_OOD_RESIDUALS), pd.read_csv(QWEN05_OOD_RESIDUALS)],
        ignore_index=True,
    )
    for source, display in [("A-LQR", "A-LQR"), ("H-infinity", r"$H_\infty$")]:
        xs = []
        ys = []
        for row in residuals.itertuples(index=False):
            rows = ood[(ood["model"] == row.label) & (ood["subset"] == row.subset)]
            original = float(rows.loc[rows["method_label"] == "Original", "toxicity_mean"].iloc[0])
            steered = float(rows.loc[rows["method_label"] == source, "toxicity_mean"].iloc[0])
            xs.append(float(row.overall))
            ys.append(np.log10(steered / original))
        ax.scatter(xs, ys, color=METHOD_COLORS[display], label=display, s=42, alpha=0.82)
    ax.axhline(0, color="0.4", linestyle=":")
    ax.set_xlabel("Mean OOD relative residual")
    ax.set_ylabel(r"$\log_{10}$(toxicity / Original)")
    ax.set_title("Residual magnitude versus transfer")
    ax.legend(fontsize=8, loc="best")
    panel_label(ax, "C")
    status_badge(ax, "REAL · 2 MODELS × 3 SHIFTS", real=True)
    finish_axis(ax)

    ax = axes[1, 1]
    marker_map = {"jigsaw_long": "o", "toxicchat_long": "s", "mmlu_ood_other_concepts": "^"}
    marker_labels = {
        "jigsaw_long": "Jigsaw",
        "toxicchat_long": "ToxicChat",
        "mmlu_ood_other_concepts": "MMLU shift",
    }
    for condition, marker in marker_map.items():
        xs = []
        ys = []
        for model in model_names:
            rows = ood[(ood["model"] == model) & (ood["subset"] == condition)]
            alqr = float(rows.loc[rows["method_label"] == "A-LQR", "toxicity_mean"].iloc[0])
            hinf = float(rows.loc[rows["method_label"] == "H-infinity", "toxicity_mean"].iloc[0])
            gamma = float(rows.loc[rows["method_label"] == "H-infinity", "gamma_star"].iloc[0])
            xs.append(gamma)
            ys.append(np.log10(hinf / alqr))
        ax.scatter(
            xs,
            ys,
            marker=marker,
            s=42,
            label=marker_labels[condition],
            color="darkred",
            alpha=0.72,
        )
    ax.axhline(0, color="0.4", linestyle=":")
    ax.set_xlabel(r"Fragility $\gamma^\star$ (higher = worse)")
    ax.set_ylabel(r"$\log_{10}(H_\infty / \mathrm{A\!-\!LQR})$ toxicity")
    ax.set_title("Does predicted fragility identify benefit?")
    ax.legend(fontsize=7, loc="upper left")
    panel_label(ax, "D")
    status_badge(ax, "REAL PRELIMINARY · TOXICITY", real=True)
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
    mark_placeholder(ax)
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
    mark_placeholder(ax)
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
    mark_placeholder(ax)
    finish_axis(ax)

    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.92), w_pad=2.0)
    save_figure(fig, "fig6_long_context_prediction")


def make_figs1_hinf_validation() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.3))
    fig.suptitle(r"Validation and optimization of finite-horizon $H_\infty$ synthesis", y=1.01)
    equivalence = pd.read_csv(HINF_EQUIVALENCE)
    benchmark = pd.read_csv(HINF_BENCHMARK)
    case_order = [
        "scalar analytic",
        "square time-varying",
        "rectangular channels",
        "conditioned costs",
        "capped infeasible",
    ]
    case_labels = ["S", "TV", "R", "C", "I"]

    ax = axes[0, 0]
    normalized_errors = {
        r"$\gamma^\star$": ("gamma_absolute_error", 1e-5, "black", "o"),
        "gain": ("gain_relative_error", 5e-5, "darkred", "o"),
        "intervention": ("intervention_relative_error", 5e-5, "midnightblue", "o"),
        "diagnostics": ("diagnostic_relative_error", 1e-3, "darkgreen", "x"),
    }
    x = np.arange(len(case_order))
    for label, (column, tolerance, color, marker) in normalized_errors.items():
        values = []
        for case in case_order:
            value = equivalence.loc[equivalence["case"] == case, column].max() / tolerance
            values.append(max(float(value), 1e-8))
        ax.scatter(x, values, color=color, marker=marker, s=34, label=label)
    ax.axhline(1, color="0.35", linestyle="--")
    ax.set_yscale("log")
    ax.set_xticks(x, case_labels)
    ax.set_ylim(1e-8, 2)
    ax.set_yticks([1e-8, 1])
    ax.set_xlabel("Validation case")
    ax.set_ylabel("Error / tolerance")
    ax.set_title("Hannah-oracle equivalence")
    ax.legend(fontsize=7, ncol=2, loc="lower center")
    panel_label(ax, "A")
    status_badge(ax, "REAL DATA · 10/10 PASS", real=True)
    finish_axis(ax)

    ax = axes[0, 1]
    gpu_equivalence = equivalence[
        equivalence["device"].str.startswith("cuda") & equivalence["feasible"]
    ]
    ratios = []
    bound_labels = []
    for case, short_label in zip(case_order, case_labels, strict=True):
        row = gpu_equivalence[gpu_equivalence["case"] == case]
        if row.empty:
            continue
        ratios.append(float(row["induced_gain_optimized"].iloc[0] / row["gamma_used"].iloc[0]))
        bound_labels.append(short_label)
    ax.scatter(np.arange(len(ratios)), ratios, color="midnightblue", s=38)
    ax.axhline(1, color="black", linestyle="--")
    ax.set_xticks(np.arange(len(ratios)), bound_labels)
    ax.set_ylim(min(ratios) - 0.0002, 1.00005)
    ax.set_xlabel("GPU validation case")
    ax.set_ylabel(r"Induced gain / deployed $\gamma$")
    ax.set_title("Independent attenuation bound")
    panel_label(ax, "B")
    status_badge(ax, "REAL DATA · ALL BELOW 1", real=True)
    finish_axis(ax)

    ax = axes[1, 0]
    gpu_optimized = benchmark[
        benchmark["device"].str.startswith("cuda") & (benchmark["method"] == "Optimized")
    ].sort_values("state_dimension")
    ax.plot(
        gpu_optimized["state_dimension"],
        gpu_optimized["speedup"],
        marker="o",
        color="darkgreen",
    )
    ax.axhline(1, color="0.35", linestyle="--")
    ax.set_xticks([64, 768])
    ax.set_xlabel("State dimension")
    ax.set_ylabel("Reference / optimized time")
    ax.set_title("GPU synthesis speedup")
    panel_label(ax, "C")
    status_badge(ax, "REAL DATA · RTX 5090", real=True)
    finish_axis(ax)

    ax = axes[1, 1]
    gpu_memory = benchmark[benchmark["device"].str.startswith("cuda")]
    for method, color in [("Hannah reference", "darkred"), ("Optimized", "midnightblue")]:
        rows = gpu_memory[gpu_memory["method"] == method].sort_values("state_dimension")
        ax.plot(
            rows["state_dimension"],
            rows["median_peak_memory_mib"],
            marker="o",
            color=color,
            label=method,
        )
    ax.set_yscale("log")
    ax.set_xticks([64, 768])
    ax.set_xlabel("State dimension")
    ax.set_ylabel("Peak allocated memory (MiB)")
    ax.set_title("GPU memory")
    ax.legend(fontsize=7, loc="upper left")
    panel_label(ax, "D")
    status_badge(ax, "REAL DATA · RTX 5090", real=True)
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
    mark_placeholder(ax)
    finish_axis(ax)

    source_image_panel(
        axes[0, 1],
        PROJECT_ROOT
        / "parking/erfan_linearization_error/plots/actual_vs_linearized_hidden_dynamics.png",
        "B",
    )

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
    mark_placeholder(ax)
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
    mark_placeholder(ax)
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
    mark_placeholder(ax)
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
    mark_placeholder(ax)
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
    mark_placeholder(ax)
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
    mark_placeholder(ax)
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
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    write_panel_status()
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
