from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from robust_steerability.control.h_infinity import (
    HInfinityController,
    HInfinityOptions,
    _solve_for_gamma,
)
from robust_steerability.control.lqr import LQRController
from robust_steerability.control.types import FiniteHorizonControlProblem


UNIT_DIR = Path(__file__).resolve().parent
CACHE_DIR = UNIT_DIR / "cache"
PLOTS_DIR = UNIT_DIR / "plots"
SWEEP_PATH = CACHE_DIR / "gamma_sweep.csv"
SUMMARY_PATH = PLOTS_DIR / "h_infinity_sanity_summary.json"

HORIZON = 12
NUMERICAL_TOLERANCE = 1e-7
GAMMA_TOLERANCE = 1e-6
COLORS = {
    "h_infinity": "darkred",
    "lqr": "midnightblue",
    "feasible": "darkgreen",
    "infeasible": "darkred",
}


def setup_style() -> None:
    sns.set_theme(context="talk", style="ticks", palette="dark")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["lines.linewidth"] = 1
    plt.rcParams["patch.linewidth"] = 0
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.transparent"] = False


def synthetic_problem(disturbance_scale: float = 1.0) -> FiniteHorizonControlProblem:
    layers = torch.arange(HORIZON, dtype=torch.float32)
    dynamics = torch.zeros(HORIZON, 2, 2)
    dynamics[:, 0, 0] = 0.92 + 0.02 * torch.sin(layers / 3.0)
    dynamics[:, 0, 1] = 0.12
    dynamics[:, 1, 0] = -0.04
    dynamics[:, 1, 1] = 0.86 + 0.02 * torch.cos(layers / 4.0)

    control_channels = torch.tensor(
        [[0.35, 0.05], [0.18, 0.28]], dtype=torch.float32
    ).unsqueeze(0).repeat(HORIZON, 1, 1)

    disturbance_channels = disturbance_scale * torch.tensor(
        [[0.22, 0.04], [0.06, 0.30]], dtype=torch.float32
    ).unsqueeze(0).repeat(HORIZON, 1, 1)

    state_cost = torch.diag(torch.tensor([1.0, 0.4]))
    state_costs = state_cost.unsqueeze(0).repeat(HORIZON, 1, 1)
    control_costs = torch.diag(torch.tensor([0.3, 0.45])).unsqueeze(0).repeat(HORIZON, 1, 1)
    terminal_cost = 2.0 * state_cost
    return FiniteHorizonControlProblem(
        dynamics=dynamics,
        control_channels=control_channels,
        disturbance_channels=disturbance_channels,
        state_costs=state_costs,
        control_costs=control_costs,
        terminal_cost=terminal_cost,
        metadata={"system": "two-state synthetic sanity check"},
    )


def gamma_sweep(
    problem: FiniteHorizonControlProblem,
    gamma_star: float,
) -> pd.DataFrame:
    gamma_values = np.geomspace(0.45 * gamma_star, 20.0 * gamma_star, 80)
    records = []
    for gamma in gamma_values:
        feasible, _, _, diagnostics = _solve_for_gamma(
            problem,
            float(gamma),
            device="cpu",
            numerical_tolerance=NUMERICAL_TOLERANCE,
        )
        records.append(
            {
                "gamma": float(gamma),
                "feasible": feasible,
                "min_disturbance_margin": diagnostics["min_disturbance_margin"],
                "min_control_margin": diagnostics["min_control_margin"],
                "max_condition_number_M": diagnostics["max_condition_number_M"],
                "max_condition_number_H": diagnostics["max_condition_number_H"],
                "max_norm_S": diagnostics["max_norm_S"],
                "max_norm_K": diagnostics["max_norm_K"],
                "failed_layer": diagnostics["failed_layer"],
                "failure_reason": diagnostics["failure_reason"],
            }
        )
    frame = pd.DataFrame(records)
    frame.to_csv(SWEEP_PATH, index=False)
    return frame


def plot_gamma_diagnostics(frame: pd.DataFrame, gamma_star: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.2))
    feasible = frame[frame["feasible"]]
    colors = frame["feasible"].map(
        {True: COLORS["feasible"], False: COLORS["infeasible"]}
    )

    axes[0, 0].scatter(frame["gamma"], frame["feasible"].astype(int), c=colors, s=16)
    axes[0, 0].axvline(gamma_star, color="black", linestyle="--")
    axes[0, 0].set_ylabel("Feasible")
    axes[0, 0].set_yticks([0, 1], labels=["No", "Yes"])
    axes[0, 0].set_title(r"Feasibility boundary $\gamma^\star$")

    axes[0, 1].plot(
        feasible["gamma"],
        feasible["min_disturbance_margin"],
        color=COLORS["h_infinity"],
        label=r"$\lambda_{\min}(M_k)$",
    )
    axes[0, 1].plot(
        feasible["gamma"],
        feasible["min_control_margin"],
        color=COLORS["lqr"],
        label=r"$\lambda_{\min}(H_k)$",
    )
    axes[0, 1].axhline(NUMERICAL_TOLERANCE, color="black", linestyle=":")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("Minimum eigenvalue")
    axes[0, 1].set_title("Feasibility margins")
    axes[0, 1].legend(fontsize=8, loc="best")

    axes[1, 0].plot(
        feasible["gamma"],
        feasible["max_condition_number_M"],
        color=COLORS["h_infinity"],
        label=r"$M_k$",
    )
    axes[1, 0].plot(
        feasible["gamma"],
        feasible["max_condition_number_H"],
        color=COLORS["lqr"],
        label=r"$H_k$",
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("Maximum condition number")
    axes[1, 0].set_title("Linear-solve conditioning")
    axes[1, 0].legend(fontsize=8, loc="best")

    axes[1, 1].plot(
        feasible["gamma"],
        feasible["max_norm_S"],
        color=COLORS["h_infinity"],
        label=r"$\max_k\|S_k\|_F$",
    )
    axes[1, 1].plot(
        feasible["gamma"],
        feasible["max_norm_K"],
        color=COLORS["lqr"],
        label=r"$\max_k\|K_k\|_F$",
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("Maximum Frobenius norm")
    axes[1, 1].set_title("Riccati and gain norms")
    axes[1, 1].legend(fontsize=8, loc="best")

    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xlabel(r"Attenuation $\gamma$")
        ax.set_xticks([frame["gamma"].min(), frame["gamma"].max()])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_box_aspect(1)
        sns.despine(ax=ax, trim=True, offset=10)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.10, top=0.92, hspace=0.65, wspace=0.65)
    for extension in ["pdf", "png"]:
        fig.savefig(PLOTS_DIR / f"gamma_diagnostics.{extension}", bbox_inches="tight")
    plt.close(fig)


def lqr_recovery(problem: FiniteHorizonControlProblem, gamma_star: float) -> pd.DataFrame:
    lqr_gains = LQRController.synthesize(problem).gains
    gamma_values = np.geomspace(1.01 * gamma_star, max(1e4, 1000 * gamma_star), 60)
    records = []
    for gamma in gamma_values:
        feasible, gains, _, _ = _solve_for_gamma(
            problem,
            float(gamma),
            device="cpu",
            numerical_tolerance=NUMERICAL_TOLERANCE,
        )
        relative_error = torch.linalg.vector_norm(gains - lqr_gains) / torch.linalg.vector_norm(lqr_gains)
        records.append(
            {
                "gamma": float(gamma),
                "feasible": feasible,
                "relative_gain_error": float(relative_error.item()),
            }
        )

    zero_disturbance = synthetic_problem(disturbance_scale=0.0)
    zero_h_infinity, zero_gains, _, _ = _solve_for_gamma(
        zero_disturbance,
        1.0,
        device="cpu",
        numerical_tolerance=NUMERICAL_TOLERANCE,
    )
    zero_lqr_gains = LQRController.synthesize(zero_disturbance).gains
    zero_error = torch.linalg.vector_norm(zero_gains - zero_lqr_gains) / torch.linalg.vector_norm(zero_lqr_gains)
    frame = pd.DataFrame(records)
    frame.attrs["zero_disturbance_feasible"] = zero_h_infinity
    frame.attrs["zero_disturbance_relative_gain_error"] = float(zero_error.item())
    return frame


def plot_lqr_recovery(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    displayed_error = frame["relative_gain_error"].clip(lower=1e-9)
    ax.plot(
        frame["gamma"],
        displayed_error,
        color=COLORS["h_infinity"],
    )
    ax.text(
        0.97,
        0.95,
        rf"$D_k=0$: error $={frame.attrs['zero_disturbance_relative_gain_error']:.1e}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=COLORS["lqr"],
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Attenuation $\gamma$")
    ax.set_ylabel("Relative gain error")
    ax.set_title("Recovery of LQR gains")
    ax.set_xticks([frame["gamma"].min(), frame["gamma"].max()])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_box_aspect(1)
    sns.despine(ax=ax, trim=True, offset=10)
    for extension in ["pdf", "png"]:
        fig.savefig(PLOTS_DIR / f"lqr_recovery.{extension}", bbox_inches="tight")
    plt.close(fig)


def disturbed_rollout(
    problem: FiniteHorizonControlProblem,
    h_infinity: HInfinityController,
    lqr: LQRController,
) -> pd.DataFrame:
    records = []
    disturbance = torch.tensor(
        [
            [
                0.65 * np.sin(0.7 * layer) + 0.35 * np.cos(1.9 * layer),
                0.45 * np.cos(0.5 * layer) - 0.25 * np.sin(1.3 * layer),
            ]
            for layer in range(HORIZON)
        ],
        dtype=torch.float32,
    )
    for name, controller in [("H-infinity", h_infinity), ("LQR", lqr)]:
        state = torch.tensor([1.0, -0.7], dtype=torch.float32)
        records.append({"controller": name, "layer": 0, "state_norm": float(torch.linalg.vector_norm(state))})
        for layer in range(HORIZON):
            control = controller.control(layer, state)
            state = (
                problem.dynamics[layer] @ state
                + problem.control_channels[layer] @ control
                + problem.disturbance_channels[layer] @ disturbance[layer]
            )
            records.append(
                {
                    "controller": name,
                    "layer": layer + 1,
                    "state_norm": float(torch.linalg.vector_norm(state).item()),
                }
            )
    return pd.DataFrame(records)


def plot_disturbed_rollout(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    for controller, color in [("H-infinity", COLORS["h_infinity"]), ("LQR", COLORS["lqr"])]:
        rows = frame[frame["controller"] == controller]
        ax.plot(rows["layer"], rows["state_norm"], color=color, label=controller)
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"State norm $\|x_k\|_2$")
    ax.set_title("Matched disturbed rollout")
    ax.set_xlim(0, HORIZON)
    ax.set_xticks([0, HORIZON])
    ymax = float(frame["state_norm"].max())
    ax.set_ylim(0, 1.05 * ymax)
    ax.set_yticks([0, round(ymax, 1)])
    ax.legend(fontsize=8, loc="best")
    ax.set_box_aspect(1)
    sns.despine(ax=ax, trim=True, offset=10)
    for extension in ["pdf", "png"]:
        fig.savefig(PLOTS_DIR / f"disturbed_rollout.{extension}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for path in PLOTS_DIR.glob("*"):
        if path.is_file():
            path.unlink()
    setup_style()

    problem = synthetic_problem()
    h_infinity = HInfinityController.synthesize(
        problem,
        options=HInfinityOptions(
            gamma_lower=0.0,
            gamma_upper=1.0,
            tolerance=GAMMA_TOLERANCE,
            numerical_tolerance=NUMERICAL_TOLERANCE,
            deployment_margin=0.01,
        ),
    )
    if not h_infinity.feasible or h_infinity.gamma_star is None:
        raise RuntimeError(f"Synthetic H-infinity synthesis failed: {h_infinity.diagnostics}")

    sweep = gamma_sweep(problem, h_infinity.gamma_star)
    recovery = lqr_recovery(problem, h_infinity.gamma_star)
    lqr = LQRController.synthesize(problem)
    rollout = disturbed_rollout(problem, h_infinity, lqr)

    plot_gamma_diagnostics(sweep, h_infinity.gamma_star)
    plot_lqr_recovery(recovery)
    plot_disturbed_rollout(rollout)

    largest_gamma_error = float(recovery.iloc[-1]["relative_gain_error"])
    summary = {
        "gamma_star": h_infinity.gamma_star,
        "gamma_used": h_infinity.diagnostics["gamma_used"],
        "bisection_converged": h_infinity.diagnostics["bisection_converged"],
        "min_disturbance_margin_at_gamma_used": h_infinity.diagnostics["min_disturbance_margin"],
        "min_control_margin_at_gamma_used": h_infinity.diagnostics["min_control_margin"],
        "zero_disturbance_relative_gain_error": recovery.attrs["zero_disturbance_relative_gain_error"],
        "largest_gamma": float(recovery.iloc[-1]["gamma"]),
        "largest_gamma_relative_gain_error": largest_gamma_error,
        "h_infinity_final_state_norm": float(rollout[rollout["controller"] == "H-infinity"].iloc[-1]["state_norm"]),
        "lqr_final_state_norm": float(rollout[rollout["controller"] == "LQR"].iloc[-1]["state_norm"]),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
