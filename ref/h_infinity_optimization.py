from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from robust_steerability.control.validation import validate_control_problem

from robust_steerability.control.h_infinity import (
    HInfinityController,
    HInfinityOptions,
)
from robust_steerability.control.metrics import closed_loop_disturbance_gain
from robust_steerability.control.types import FiniteHorizonControlProblem


REPO_ROOT = Path(__file__).resolve().parents[2]
UNIT_DIR = Path(__file__).resolve().parent
CACHE_DIR = UNIT_DIR / "cache"
PLOTS_DIR = UNIT_DIR / "plots"
REFERENCE_PATH = REPO_ROOT / "ref" / "h_infinity.py"
REFERENCE_SHA256 = "293c7f45a54c6ffb6caf2a13cec9b0e09396fd4921973bccc019ba09e19f96f0"
BENCHMARK_PROTOCOL_VERSION = 2
SEED = 2151
GAMMA_ATOL = 1e-5
GAIN_RTOL = 5e-5
INTERVENTION_RTOL = 5e-5
DIAGNOSTIC_RTOL = 1e-3
METHOD_COLORS = {"Hannah reference": "darkred", "Optimized": "midnightblue"}
CASE_LABELS = {
    "scalar analytic": "S",
    "square time-varying": "TV",
    "rectangular channels": "R",
    "conditioned costs": "C",
    "capped infeasible": "I",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference() -> ModuleType:
    actual_hash = file_sha256(REFERENCE_PATH)
    if actual_hash != REFERENCE_SHA256:
        raise ValueError(
            f"Reference hash changed: expected {REFERENCE_SHA256}, got {actual_hash}"
        )
    module_name = "hannah_h_infinity_reference"
    specification = importlib.util.spec_from_file_location(module_name, REFERENCE_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load {REFERENCE_PATH}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def random_problem(
    *,
    horizon: int,
    state_dimension: int,
    control_dimension: int,
    disturbance_dimension: int,
    seed: int,
    condition_scale: float = 1.0,
) -> FiniteHorizonControlProblem:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    identity = torch.eye(state_dimension, dtype=torch.float32)
    dynamics = 0.65 * identity.unsqueeze(0).repeat(horizon, 1, 1)
    dynamics += (
        0.06
        / math.sqrt(state_dimension)
        * torch.randn(
            horizon,
            state_dimension,
            state_dimension,
            generator=generator,
        )
    )
    control_channels = (
        0.35
        / math.sqrt(control_dimension)
        * torch.randn(
            horizon,
            state_dimension,
            control_dimension,
            generator=generator,
        )
    )
    disturbance_channels = (
        0.25
        / math.sqrt(disturbance_dimension)
        * torch.randn(
            horizon,
            state_dimension,
            disturbance_dimension,
            generator=generator,
        )
    )
    state_diagonal = torch.logspace(
        -math.log10(condition_scale),
        0.0,
        state_dimension,
        dtype=torch.float32,
    )
    control_diagonal = torch.logspace(
        -0.5 * math.log10(condition_scale),
        0.0,
        control_dimension,
        dtype=torch.float32,
    )
    state_costs = torch.diag(0.1 * state_diagonal).repeat(horizon, 1, 1)
    control_costs = torch.diag(control_diagonal).repeat(horizon, 1, 1)
    terminal_cost = torch.diag(state_diagonal)
    return FiniteHorizonControlProblem(
        dynamics=dynamics,
        control_channels=control_channels,
        disturbance_channels=disturbance_channels,
        state_costs=state_costs,
        control_costs=control_costs,
        terminal_cost=terminal_cost,
    )


def scalar_analytic_problem() -> FiniteHorizonControlProblem:
    return FiniteHorizonControlProblem(
        dynamics=torch.tensor([[[0.8]]], dtype=torch.float32),
        control_channels=torch.tensor([[[0.6]]], dtype=torch.float32),
        disturbance_channels=torch.tensor([[[0.4]]], dtype=torch.float32),
        state_costs=torch.tensor([[[0.1]]], dtype=torch.float32),
        control_costs=torch.tensor([[[1.0]]], dtype=torch.float32),
        terminal_cost=torch.tensor([[1.2]], dtype=torch.float32),
    )


def benchmark_problem(
    state_dimension: int,
    *,
    horizon: int = 6,
    seed: int = SEED,
) -> FiniteHorizonControlProblem:
    generator = torch.Generator(device="cpu").manual_seed(seed + state_dimension)
    identity = torch.eye(state_dimension, dtype=torch.float32)
    dynamics = 0.72 * identity.unsqueeze(0).repeat(horizon, 1, 1)
    dynamics += (
        0.04
        / math.sqrt(state_dimension)
        * torch.randn(
            horizon,
            state_dimension,
            state_dimension,
            generator=generator,
        )
    )
    channels = identity.unsqueeze(0).repeat(horizon, 1, 1)
    return FiniteHorizonControlProblem(
        dynamics=dynamics,
        control_channels=channels,
        disturbance_channels=channels.clone(),
        state_costs=(0.1 * identity).unsqueeze(0).repeat(horizon, 1, 1),
        control_costs=identity.unsqueeze(0).repeat(horizon, 1, 1),
        terminal_cost=identity,
    )


def reference_options(reference: ModuleType, options: HInfinityOptions) -> object:
    return reference.HInfinityOptions(**asdict(options))


def relative_tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = float(torch.linalg.vector_norm(actual - expected).item())
    denominator = max(float(torch.linalg.vector_norm(expected).item()), 1e-12)
    return numerator / denominator


def numeric_diagnostic_error(
    actual: dict[str, object],
    expected: dict[str, object],
) -> float:
    errors = []
    for key in sorted(set(actual).intersection(expected)):
        actual_value = actual[key]
        expected_value = expected[key]
        if not isinstance(actual_value, (int, float)) or isinstance(actual_value, bool):
            continue
        if not isinstance(expected_value, (int, float)) or isinstance(expected_value, bool):
            continue
        if not math.isfinite(float(actual_value)) or not math.isfinite(float(expected_value)):
            continue
        denominator = max(abs(float(expected_value)), 1e-12)
        errors.append(abs(float(actual_value) - float(expected_value)) / denominator)
    return max(errors, default=0.0)


def parity_cases() -> list[tuple[str, FiniteHorizonControlProblem, HInfinityOptions]]:
    options = HInfinityOptions(
        gamma_lower=0.01,
        gamma_upper=10.0,
        tolerance=GAMMA_ATOL,
        max_iterations=60,
        numerical_tolerance=1e-7,
        deployment_margin=1e-3,
    )
    return [
        ("scalar analytic", scalar_analytic_problem(), options),
        (
            "square time-varying",
            random_problem(
                horizon=5,
                state_dimension=8,
                control_dimension=8,
                disturbance_dimension=8,
                seed=SEED,
            ),
            options,
        ),
        (
            "rectangular channels",
            random_problem(
                horizon=5,
                state_dimension=10,
                control_dimension=4,
                disturbance_dimension=3,
                seed=SEED + 1,
            ),
            options,
        ),
        (
            "conditioned costs",
            random_problem(
                horizon=4,
                state_dimension=10,
                control_dimension=4,
                disturbance_dimension=3,
                seed=SEED + 2,
                condition_scale=100.0,
            ),
            options,
        ),
        (
            "capped infeasible",
            scalar_analytic_problem(),
            HInfinityOptions(
                gamma_lower=0.01,
                gamma_upper=0.1,
                tolerance=GAMMA_ATOL,
                max_iterations=60,
                numerical_tolerance=1e-7,
                max_gamma=0.1,
                deployment_margin=0.0,
            ),
        ),
    ]


def run_parity(reference: ModuleType, devices: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for device in devices:
        for case_index, (case_name, problem, options) in enumerate(parity_cases()):
            reference_controller = reference.HInfinityController.synthesize(
                problem,
                device=device,
                options=reference_options(reference, options),
            )
            optimized_controller = HInfinityController.synthesize(
                problem,
                device=device,
                options=options,
            )
            reference_solution = reference_controller.solution()
            optimized_solution = optimized_controller.solution()
            case_cache = CACHE_DIR / "parity_solutions" / device.replace(":", "_") / str(case_index)
            case_cache.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"case": case_name, "synthetic": True, "problem": asdict(problem),
                 "options": asdict(options), "reference_solution": asdict(reference_solution),
                 "optimized_solution": asdict(optimized_solution)},
                case_cache / "synthesis.pt",
            )
            feasibility_match = (
                reference_controller.feasible == optimized_controller.feasible
            )
            gamma_error = 0.0
            if reference_controller.gamma_star is not None:
                assert optimized_controller.gamma_star is not None
                gamma_error = abs(
                    reference_controller.gamma_star - optimized_controller.gamma_star
                )
            gain_error = relative_tensor_error(
                optimized_solution.gains,
                reference_solution.gains,
            )
            intervention_error = 0.0
            induced_gain_reference = float("nan")
            induced_gain_optimized = float("nan")
            gamma_used = float("nan")
            if reference_controller.feasible:
                feedback_generator = torch.Generator(device="cpu").manual_seed(
                    SEED + case_index
                )
                feedback = torch.randn(
                    7,
                    problem.state_dimension,
                    generator=feedback_generator,
                ).to(device)
                reference_intervention = reference_controller.intervention(0, feedback)
                optimized_intervention = optimized_controller.intervention(0, feedback)
                intervention_error = relative_tensor_error(
                    optimized_intervention.cpu(),
                    reference_intervention.cpu(),
                )
                induced_gain_reference = closed_loop_disturbance_gain(
                    problem,
                    reference_solution,
                )
                induced_gain_optimized = closed_loop_disturbance_gain(
                    problem,
                    optimized_solution,
                )
                gamma_used = float(reference_solution.diagnostics["gamma_used"])
            diagnostic_error = numeric_diagnostic_error(
                optimized_solution.diagnostics,
                reference_solution.diagnostics,
            )
            gamma_match = gamma_error <= options.tolerance
            gain_match = gain_error <= GAIN_RTOL
            intervention_match = intervention_error <= INTERVENTION_RTOL
            diagnostic_match = diagnostic_error <= DIAGNOSTIC_RTOL
            analytic_gamma = float("nan")
            analytic_gamma_error = float("nan")
            if case_name == "scalar analytic":
                analytic_gamma = math.sqrt(0.4**2 * 1.2 + options.numerical_tolerance)
                assert reference_controller.gamma_star is not None
                analytic_gamma_error = abs(
                    reference_controller.gamma_star - analytic_gamma
                )
            records.append(
                {
                    "device": device,
                    "case": case_name,
                    "feasible": reference_controller.feasible,
                    "feasibility_match": feasibility_match,
                    "gamma_reference": reference_controller.gamma_star,
                    "gamma_optimized": optimized_controller.gamma_star,
                    "s_rob_reference": 1.0 / reference_controller.gamma_star if reference_controller.feasible and reference_controller.gamma_star else None,
                    "s_rob_optimized": 1.0 / optimized_controller.gamma_star if optimized_controller.feasible and optimized_controller.gamma_star else None,
                    "synthetic": True,
                    "gamma_absolute_error": gamma_error,
                    "gain_relative_error": gain_error,
                    "intervention_relative_error": intervention_error,
                    "diagnostic_relative_error": diagnostic_error,
                    "induced_gain_reference": induced_gain_reference,
                    "induced_gain_optimized": induced_gain_optimized,
                    "gamma_used": gamma_used,
                    "analytic_gamma": analytic_gamma,
                    "analytic_gamma_error": analytic_gamma_error,
                    "gamma_match": gamma_match,
                    "gain_match": gain_match,
                    "intervention_match": intervention_match,
                    "diagnostic_match": diagnostic_match,
                    "parity_pass": feasibility_match
                    and gamma_match
                    and gain_match
                    and intervention_match
                    and diagnostic_match,
                }
            )
            print(f"parity {device} {case_name}: {records[-1]['parity_pass']}")
    return pd.DataFrame.from_records(records)


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def timed_synthesis(
    *,
    controller_class: type,
    options: object,
    problem: FiniteHorizonControlProblem,
    device: str,
) -> tuple[float, float, bool]:
    if device.startswith("cuda"):
        cuda_device = torch.device(device)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(cuda_device)
        baseline_memory = torch.cuda.memory_allocated(cuda_device)
    else:
        baseline_memory = 0
    synchronize(device)
    start = time.perf_counter()
    controller = controller_class.synthesize(
        problem,
        device=device,
        options=options,
    )
    synchronize(device)
    elapsed = time.perf_counter() - start
    if device.startswith("cuda"):
        peak_memory = torch.cuda.max_memory_allocated(torch.device(device))
        allocated = max(0, peak_memory - baseline_memory) / (1024**2)
    else:
        allocated = float("nan")
    output_on_requested_device = controller.gains.device == torch.device(device)
    del controller
    return elapsed, allocated, output_on_requested_device


def warm_up(reference: ModuleType, device: str, options: HInfinityOptions) -> None:
    problem = benchmark_problem(16, horizon=2)
    for controller_class, candidate_options in [
        (reference.HInfinityController, reference_options(reference, options)),
        (HInfinityController, options),
    ]:
        timed_synthesis(
            controller_class=controller_class,
            options=candidate_options,
            problem=problem,
            device=device,
        )


def run_benchmark(
    reference: ModuleType,
    *,
    gpu_device: str,
    dimensions: list[int],
    repeats: int,
) -> pd.DataFrame:
    options = HInfinityOptions(
        gamma_lower=0.01,
        gamma_upper=10.0,
        tolerance=1e-4,
        max_iterations=60,
        numerical_tolerance=1e-7,
        deployment_margin=1e-3,
    )
    devices = ["cpu", gpu_device]
    for device in devices:
        warm_up(reference, device, options)

    records: list[dict[str, object]] = []
    for device in devices:
        device_dimensions = dimensions if device != "cpu" else [n for n in dimensions if n <= 128]
        for state_dimension in device_dimensions:
            problem = benchmark_problem(state_dimension)
            run_repeats = repeats if state_dimension <= 256 else max(2, repeats - 1)
            methods = [
                (
                    "Hannah reference",
                    reference.HInfinityController,
                    reference_options(reference, options),
                ),
                ("Optimized", HInfinityController, options),
            ]
            for repeat in range(run_repeats):
                for method, controller_class, candidate_options in methods:
                    elapsed, peak_memory, output_on_device = timed_synthesis(
                        controller_class=controller_class,
                        options=candidate_options,
                        problem=problem,
                        device=device,
                    )
                    records.append(
                        {
                            "device": device,
                            "state_dimension": state_dimension,
                            "horizon": problem.horizon,
                            "method": method,
                            "repeat": repeat,
                            "seconds": elapsed,
                            "peak_memory_mib": peak_memory,
                            "output_on_requested_device": output_on_device,
                        }
                    )
                    print(
                        f"timing {device} n={state_dimension} {method} "
                        f"repeat={repeat + 1}/{run_repeats}: {elapsed:.3f}s"
                    )
    return pd.DataFrame.from_records(records)


def summarize_benchmark(timings: pd.DataFrame) -> pd.DataFrame:
    summary = (
        timings.groupby(["device", "state_dimension", "horizon", "method"], as_index=False)
        .agg(
            median_seconds=("seconds", "median"),
            minimum_seconds=("seconds", "min"),
            maximum_seconds=("seconds", "max"),
            median_peak_memory_mib=("peak_memory_mib", "median"),
            repeats=("repeat", "count"),
            output_on_requested_device=("output_on_requested_device", "all"),
        )
        .sort_values(["device", "state_dimension", "method"])
    )
    runtime_wide = summary.pivot(
        index=["device", "state_dimension"],
        columns="method",
        values="median_seconds",
    )
    runtime_wide["speedup"] = (
        runtime_wide["Hannah reference"] / runtime_wide["Optimized"]
    )
    speedups = runtime_wide["speedup"].rename("speedup").reset_index()
    return summary.merge(speedups, on=["device", "state_dimension"], how="left")


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


def plot_diagnostics(
    parity: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    gpu_name: str,
) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5))

    error_specs = [
        ("gamma_absolute_error", GAMMA_ATOL, r"$\gamma^\star$"),
        ("gain_relative_error", GAIN_RTOL, "gain"),
        ("intervention_relative_error", INTERVENTION_RTOL, "intervention"),
        ("diagnostic_relative_error", DIAGNOSTIC_RTOL, "diagnostics"),
    ]
    case_order = list(parity["case"].drop_duplicates())
    parity_colors = ["black", "darkred", "midnightblue", "darkgreen"]
    for (column, tolerance, label), color in zip(error_specs, parity_colors, strict=True):
        for use_gpu, offset, marker in [(False, -0.12, "o"), (True, 0.12, "x")]:
            device_mask = parity["device"].str.startswith("cuda") == use_gpu
            rows = parity.loc[device_mask].set_index("case").loc[case_order]
            normalized_error = np.maximum(
                rows[column].to_numpy(dtype=float) / tolerance,
                1e-8,
            )
            axes[0, 0].scatter(
                np.arange(len(case_order)) + offset,
                normalized_error,
                s=25,
                marker=marker,
                color=color,
            )
    axes[0, 0].axhline(1.0, color="0.5", linestyle="--")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xticks(np.arange(len(case_order)))
    axes[0, 0].set_xticklabels(
        [CASE_LABELS[name] for name in case_order],
        fontsize=8,
    )
    axes[0, 0].set_xlabel("Validation case")
    axes[0, 0].set_ylabel("Error / tolerance")
    axes[0, 0].set_title("Oracle equivalence")
    metric_handles = [
        Line2D([], [], color=color, marker="o", linestyle="None", label=label)
        for (_, _, label), color in zip(error_specs, parity_colors, strict=True)
    ]
    device_handles = [
        Line2D([], [], color="0.4", marker="o", linestyle="None", label="CPU"),
        Line2D([], [], color="0.4", marker="x", linestyle="None", label="GPU"),
    ]
    axes[0, 0].legend(
        handles=metric_handles + device_handles,
        fontsize=7,
        loc="upper left",
        ncol=2,
    )

    feasible = parity.loc[
        parity["feasible"] & parity["device"].str.startswith("cuda")
    ].copy()
    feasible["label"] = feasible["case"].map(CASE_LABELS)
    bound_positions = np.arange(len(feasible))
    for offset, (column, label) in zip(
        [-0.12, 0.12],
        [
            ("induced_gain_reference", "Hannah reference"),
            ("induced_gain_optimized", "Optimized"),
        ],
        strict=True,
    ):
        axes[0, 1].scatter(
            bound_positions + offset,
            feasible[column] / feasible["gamma_used"],
            s=30,
            color=METHOD_COLORS[label],
            label=label,
        )
    axes[0, 1].axhline(1.0, color="black", linestyle="--")
    axes[0, 1].set_xticks(bound_positions)
    axes[0, 1].set_xticklabels(feasible["label"], fontsize=8)
    axes[0, 1].set_xlabel("GPU validation case")
    axes[0, 1].set_ylabel(r"Gain / $\gamma_{used}$")
    axes[0, 1].set_title("Independent gain bound")
    axes[0, 1].legend(fontsize=7, loc="lower right")

    for panel, device, title in [
        (axes[0, 2], "cpu", "CPU synthesis"),
        (axes[1, 0], "cuda", "GPU synthesis"),
    ]:
        data = benchmark.loc[
            benchmark["device"].str.startswith(device)
        ]
        for method in ["Hannah reference", "Optimized"]:
            method_data = data.loc[data["method"] == method]
            panel.plot(
                method_data["state_dimension"],
                method_data["median_seconds"],
                marker="o",
                color=METHOD_COLORS[method],
                label=method,
            )
        panel.set_xscale("log", base=2)
        dimensions = method_data["state_dimension"].astype(int).tolist()
        ticks = dimensions if len(dimensions) <= 3 else [dimensions[0], dimensions[len(dimensions) // 2], dimensions[-1]]
        panel.set_xticks(ticks)
        panel.set_xticklabels(ticks)
        panel.set_yscale("log")
        panel.set_xlabel("State dimension")
        panel.set_ylabel("Median seconds")
        panel.set_title(title)
        panel.legend(fontsize=7, loc="upper left")

    gpu_benchmark = benchmark.loc[
        (benchmark["device"].str.startswith("cuda"))
        & (benchmark["method"] == "Optimized")
    ]
    axes[1, 1].plot(
        gpu_benchmark["state_dimension"],
        gpu_benchmark["speedup"],
        marker="o",
        color="darkgreen",
    )
    axes[1, 1].axhline(1.0, color="black", linestyle="--")
    axes[1, 1].set_xscale("log", base=2)
    gpu_dimensions = gpu_benchmark["state_dimension"].astype(int).tolist()
    gpu_ticks = [gpu_dimensions[0], gpu_dimensions[len(gpu_dimensions) // 2], gpu_dimensions[-1]]
    axes[1, 1].set_xticks(gpu_ticks)
    axes[1, 1].set_xticklabels(gpu_ticks)
    axes[1, 1].set_xlabel("State dimension")
    axes[1, 1].set_ylabel("Reference / optimized time")
    axes[1, 1].set_title("GPU speedup")

    gpu_memory = benchmark.loc[benchmark["device"].str.startswith("cuda")]
    for method in ["Hannah reference", "Optimized"]:
        method_data = gpu_memory.loc[gpu_memory["method"] == method]
        axes[1, 2].plot(
            method_data["state_dimension"],
            method_data["median_peak_memory_mib"],
            marker="o",
            color=METHOD_COLORS[method],
            label=method,
        )
    axes[1, 2].set_xscale("log", base=2)
    axes[1, 2].set_xticks(gpu_ticks)
    axes[1, 2].set_xticklabels(gpu_ticks)
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_xlabel("State dimension")
    axes[1, 2].set_ylabel("Peak allocated MiB")
    axes[1, 2].set_title("GPU memory")
    axes[1, 2].legend(fontsize=7, loc="upper left")

    for ax in axes.flat:
        sns.despine(ax=ax, trim=True, offset=6)
    fig.suptitle(f"H∞ optimization diagnostics — {gpu_name}", fontsize=15)
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.16,
        top=0.90,
        hspace=0.62,
        wspace=0.45,
    )
    fig.savefig(
        PLOTS_DIR / "h_infinity_optimization.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(PLOTS_DIR / "h_infinity_optimization.png", bbox_inches="tight")
    plt.close(fig)


def write_summary(
    parity: pd.DataFrame,
    benchmark: pd.DataFrame,
    timings: pd.DataFrame,
    *,
    gpu_device: str,
) -> dict[str, object]:
    gpu_rows = benchmark.loc[
        (benchmark["device"] == gpu_device)
        & (benchmark["method"] == "Optimized")
    ].sort_values("state_dimension")
    pipeline_row = gpu_rows.iloc[-1]
    feasible = parity.loc[parity["feasible"]]
    bound_ratios = pd.concat(
        [
            feasible["induced_gain_reference"] / feasible["gamma_used"],
            feasible["induced_gain_optimized"] / feasible["gamma_used"],
        ],
        ignore_index=True,
    )
    analytic_error = parity["analytic_gamma_error"].dropna()
    payload: dict[str, object] = {
        "reference_path": str(REFERENCE_PATH.relative_to(REPO_ROOT)),
        "reference_sha256": file_sha256(REFERENCE_PATH),
        "optimized_path": "robust_steerability/control/h_infinity.py",
        "optimized_sha256": file_sha256(
            REPO_ROOT / "robust_steerability" / "control" / "h_infinity.py"
        ),
        "seed": SEED,
        "torch_version": torch.__version__,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cpu_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_device": gpu_device,
        "gpu_name": torch.cuda.get_device_name(torch.device(gpu_device)),
        "all_outputs_on_requested_device": bool(
            timings["output_on_requested_device"].all()
        ),
        "parity_cases": int(len(parity)),
        "parity_passed": int(parity["parity_pass"].sum()),
        "all_parity_passed": bool(parity["parity_pass"].all()),
        "max_gamma_absolute_error": float(parity["gamma_absolute_error"].max()),
        "max_gain_relative_error": float(parity["gain_relative_error"].max()),
        "max_intervention_relative_error": float(
            parity["intervention_relative_error"].max()
        ),
        "max_diagnostic_relative_error": float(
            parity["diagnostic_relative_error"].max()
        ),
        "max_induced_gain_over_gamma_used": float(bound_ratios.max()),
        "max_scalar_analytic_gamma_error": float(analytic_error.max()),
        "largest_benchmark_state_dimension": int(pipeline_row["state_dimension"]),
        "largest_dimension_gpu_speedup": float(pipeline_row["speedup"]),
        "largest_dimension_optimized_seconds": float(
            pipeline_row["median_seconds"]
        ),
        "largest_dimension_optimized_peak_memory_mib": float(
            pipeline_row["median_peak_memory_mib"]
        ),
    }
    (PLOTS_DIR / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (CACHE_DIR / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


# Calibration score caching and prospective prediction exports.

UNIT = Path(__file__).resolve().parent
CACHE = UNIT / "cache" / "robust_steerability"
PREDICTORS = (
    "log_parameter_count", "probe_accuracy", "semantic_snr", "linearization_error",
    "jacobian_subspace_similarity", "gramian_metric", "nominal_lqr_objective",
    "minimum_nominal_control_energy", "s_rob", "negative_log_gamma_star",
)
PROBLEM_KEYS = (
    "dynamics", "control_channels", "disturbance_channels", "state_costs",
    "control_costs", "terminal_cost",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def read_json(path: Path):
    return json.loads(path.read_text())


def verify_run(directory: Path) -> None:
    for name, expected in read_json(directory / "manifest.json")["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Frozen score artifact changed: {directory / name}")


def safe_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("IDs must start with a letter/digit and contain only letters, digits, _, ., -")
    return value


def prompt_ids(values: list[str]) -> set[str]:
    if not values or any(not isinstance(v, str) or not v for v in values):
        raise ValueError("Prompt IDs must be nonempty dataset-qualified strings")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate prompt IDs")
    return set(values)


def score(bundle_path: Path, device: str) -> Path:
    """Freeze one model-behavior score without reading evaluation outcomes."""
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    required = {"problem", "record", "splits", "normalization", "calibration", "predictors"}
    if set(bundle) - (required | {"options"}) or not required <= set(bundle):
        raise ValueError(f"Bundle must contain {sorted(required)} and optionally options")
    record = bundle["record"]
    for key in ("run_id", "model_id", "model_revision", "model_family", "behavior",
                "intervention_channel", "protocol_id", "parameter_count", "synthetic"):
        if key not in record:
            raise ValueError(f"Missing record field: {key}")
    run_id = safe_id(record["run_id"])
    if type(record["synthetic"]) is not bool or not math.isfinite(float(record["parameter_count"])) or record["parameter_count"] <= 0:
        raise ValueError("synthetic must be boolean; parameter_count must be positive")
    splits = bundle["splits"]
    if set(splits) != {"fit", "calibration"}:
        raise ValueError("Score input splits must contain fit and calibration only")
    fit_ids, cal_ids = prompt_ids(splits["fit"]), prompt_ids(splits["calibration"])
    if fit_ids & cal_ids:
        raise ValueError("Fit and calibration prompts overlap")
    normalization = bundle["normalization"]
    for key in ("protocol_id", "coordinates", "state_whitening", "control_std",
                "semantic_output_std", "depth_increment", "stage_costs_depth_weighted"):
        if key not in normalization:
            raise ValueError(f"Missing normalization field: {key}")
    if normalization["coordinates"] not in {"normalized", "raw"}:
        raise ValueError("coordinates must be normalized or raw")
    problem = FiniteHorizonControlProblem(**bundle["problem"])
    validate_control_problem(problem)
    if problem.disturbance_channels is None or min(
        problem.horizon, problem.state_dimension, problem.control_dimension,
        problem.disturbance_dimension,
    ) < 1:
        raise ValueError("All problem dimensions must be positive and D is required")
    if normalization["coordinates"] == "normalized":
        whitening = normalization["state_whitening"]
        if whitening.shape != (problem.horizon + 1, problem.state_dimension, problem.state_dimension):
            raise ValueError("state_whitening must have shape (T+1,n,n)")
        if not torch.isfinite(whitening).all() or torch.linalg.svdvals(whitening.double()).min() <= 0:
            raise ValueError("State whitening must be finite and nonsingular")
        for key in ("control_std", "semantic_output_std", "depth_increment"):
            values = normalization[key]
            if not isinstance(values, torch.Tensor) or not torch.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f"{key} must be a positive finite tensor")
        if normalization["control_std"].shape != (problem.horizon, problem.control_dimension):
            raise ValueError("control_std must have shape (T,m)")
        depth = normalization["depth_increment"]
        if depth.shape != (problem.horizon,) or not torch.isclose(depth.sum(), torch.tensor(1.0, dtype=depth.dtype)):
            raise ValueError("depth_increment must have length T and sum to one")
        if normalization["stage_costs_depth_weighted"] is not True:
            raise ValueError("Normalized cross-model scores require depth-weighted stage costs")
    for key in ("state_costs", "terminal_cost", "control_costs"):
        cost = getattr(problem, key).double()
        if not torch.allclose(cost, cost.transpose(-1, -2), atol=1e-7, rtol=1e-6):
            raise ValueError(f"{key} must be symmetric")
        minimum = torch.linalg.eigvalsh(cost).min().item()
        if minimum < -1e-8 or (key == "control_costs" and minimum <= 0):
            raise ValueError(f"{key} must be PSD (strictly PD for control costs)")
    calibration = bundle["calibration"]
    for key in ("residuals", "state_basis", "target_readouts", "protected_readouts",
                "reference_states", "reference_controls", "disturbance_construction"):
        if key not in calibration:
            raise ValueError(f"Missing calibration field: {key}")
    target = calibration["target_readouts"]
    if not isinstance(target, torch.Tensor) or target.ndim != 3 or target.shape[0] != problem.horizon + 1 or target.shape[2] != problem.state_dimension:
        raise ValueError("target_readouts must have shape (T+1,p,n)")
    if not torch.isfinite(target).all():
        raise ValueError("target_readouts must be finite")
    if normalization["coordinates"] == "normalized" and normalization["semantic_output_std"].shape != target.shape[:2]:
        raise ValueError("semantic_output_std must have shape (T+1,p)")
    residuals = calibration["residuals"]
    expected = (len(cal_ids), problem.horizon, problem.state_dimension)
    if residuals.shape != expected or not torch.isfinite(residuals).all() or len(cal_ids) < 2:
        raise ValueError(f"Calibration residuals must be finite and have shape {expected}, N>=2")
    # Baselines are provided by their owning fit/calibration analyses with definitions.
    predictors = {name: None for name in PREDICTORS}
    predictors["log_parameter_count"] = math.log(float(record["parameter_count"]))
    for name, entry in bundle["predictors"].items():
        if name not in PREDICTORS[1:-2]:
            raise ValueError(f"Unknown or computed predictor: {name}")
        if entry["split"] not in {"fit", "calibration"} or not entry["definition"]:
            raise ValueError(f"Predictor {name} needs a fit/calibration source and definition")
        if entry["value"] is not None and not math.isfinite(float(entry["value"])):
            raise ValueError(f"Nonfinite predictor: {name}")
        predictors[name] = entry["value"]
    options = HInfinityOptions(**bundle.get("options", {}))
    option_values = asdict(options)
    if not all(math.isfinite(float(v)) for v in option_values.values()):
        raise ValueError("Synthesis options must be finite")
    destination = CACHE / "runs" / run_id
    fingerprint = {"input_sha256": sha256(bundle_path),
                   "controller_sha256": sha256(Path(inspect.getfile(HInfinityController))),
                   "exporter_sha256": sha256(Path(__file__)), "device": device}
    if destination.exists():
        manifest = destination / "manifest.json"
        if not manifest.exists() or read_json(manifest)["fingerprint"] != fingerprint:
            raise ValueError("Run ID already exists with different or incomplete content; use a new run_id")
        verify_run(destination)
        return destination
    with torch.no_grad():
        controller = HInfinityController.synthesize(problem, device=device, options=options)
    solution = controller.solution()
    converged = bool(solution.diagnostics.get("bisection_converged", False))
    gamma = solution.gamma_star
    zero_disturbance = bool(torch.count_nonzero(problem.disturbance_channels) == 0)
    valid_score = solution.feasible and converged and gamma is not None and math.isfinite(gamma) and gamma > 0 and not zero_disturbance
    if valid_score:
        predictors["s_rob"] = 1.0 / gamma
        predictors["negative_log_gamma_star"] = -math.log(gamma)
    centered = residuals.double() - residuals.double().mean(dim=0, keepdim=True)
    covariance = torch.einsum("nti,ntj->tij", centered, centered) / (len(cal_ids) - 1)
    d = problem.disturbance_channels.double()
    covariance_error = torch.linalg.matrix_norm(covariance - d @ d.transpose(-1, -2), dim=(-2, -1))
    covariance_norm = torch.linalg.matrix_norm(covariance, dim=(-2, -1))
    covariance_relative_error = torch.where(covariance_norm > 0, covariance_error / covariance_norm, torch.full_like(covariance_norm, float("nan")))
    metadata = {**record, "gamma_star": gamma, "gamma_used": solution.diagnostics.get("gamma_used"),
                "s_rob": predictors["s_rob"], "negative_log_gamma_star": predictors["negative_log_gamma_star"],
                "feasible": solution.feasible, "bisection_converged": converged,
                "score_available": valid_score, "predictors": predictors,
                "zero_disturbance_channel": zero_disturbance,
                "score_status": "available" if valid_score else "zero disturbance gives an infinite ideal score" if zero_disturbance else "infeasible or unconverged",
                "normalization_protocol_id": normalization["protocol_id"],
                "coordinates": normalization["coordinates"],
                "stage_costs_depth_weighted": normalization["stage_costs_depth_weighted"],
                "score_definition": "1 / numerical feasible upper boundary gamma_star",
                "diagnostics": solution.diagnostics,
                "fit_count": len(fit_ids), "calibration_count": len(cal_ids),
                "covariance_relative_error_by_layer": covariance_relative_error}
    destination.mkdir(parents=True)
    # Keep the exact input bytes, including bases, normalizers, residuals and provenance.
    (destination / "calibration_input.pt").write_bytes(bundle_path.read_bytes())
    (destination / "controller_source.py").write_bytes(Path(inspect.getfile(HInfinityController)).read_bytes())
    (destination / "exporter_source.py").write_bytes(Path(__file__).read_bytes())
    torch.save({"controller": solution.controller, "gains": solution.gains,
                "feasible": solution.feasible, "gamma_star": gamma,
                "diagnostics": solution.diagnostics,
                "control_channels": problem.control_channels.cpu(),
                "residual_covariance": covariance,
                "covariance_relative_error_by_layer": covariance_relative_error}, destination / "controller.pt")
    write_json(destination / "score.json", metadata)
    write_json(destination / "manifest.json", {"schema_version": 1, "fingerprint": fingerprint,
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "torch_version": str(torch.__version__), "options": option_values,
               "files": {p.name: sha256(p) for p in destination.iterdir() if p.is_file()}})
    return destination


def evaluate(path: Path) -> Path:
    """Attach prompt-level held-out records to an already frozen score."""
    payload = read_json(path)
    run_id, evaluation_id = safe_id(payload["run_id"]), safe_id(payload["evaluation_id"])
    run = CACHE / "runs" / run_id
    verify_run(run)
    manifest = read_json(run / "manifest.json")
    if payload["score_manifest_sha256"] != sha256(run / "manifest.json"):
        raise ValueError("Evaluation must reference the frozen score manifest hash")
    if datetime.fromisoformat(payload["evaluation_started_at_utc"]) <= datetime.fromisoformat(manifest["created_at_utc"]):
        raise ValueError("Evaluation must start after the score was frozen")
    bundle = torch.load(run / "calibration_input.pt", map_location="cpu", weights_only=True)
    excluded = set(bundle["splits"]["fit"]) | set(bundle["splits"]["calibration"])
    score_record = read_json(run / "score.json")
    for field in ("controller", "shift", "protocol_id", "success_definition",
                  "matching_rule", "generation_config", "evaluator"):
        if field not in payload or payload[field] in (None, "", {}):
            raise ValueError(f"Missing evaluation field: {field}")
    if payload["protocol_id"] != score_record["protocol_id"]:
        raise ValueError("Evaluation protocol differs from score protocol")
    observations = payload["observations"]
    if not observations:
        raise ValueError("No held-out observations")
    keys = set()
    for row in observations:
        for field in ("prompt_id", "seed", "success", "raw_score", "control_energy", "collateral_metrics"):
            if field not in row:
                raise ValueError(f"Missing observation field: {field}")
        if not isinstance(row["prompt_id"], str) or not row["prompt_id"]:
            raise ValueError("Prompt IDs must be nonempty dataset-qualified strings")
        key = (row["prompt_id"], row["seed"])
        if key in keys or row["prompt_id"] in excluded:
            raise ValueError("Duplicate prompt/seed or held-out prompt overlaps fit/calibration")
        if type(row["success"]) is not bool:
            raise ValueError("success must be Boolean using a predefined behavioral threshold")
        if not all(math.isfinite(float(row[v])) for v in ("raw_score", "control_energy")) or row["control_energy"] < 0:
            raise ValueError("Scores and nonnegative control energies must be finite")
        keys.add(key)
    target = run / "evaluations" / evaluation_id
    if target.exists():
        if (target / "observations.json").read_bytes() != path.read_bytes():
            raise ValueError("Evaluation ID exists with different content; use a new evaluation_id")
        return target
    target.mkdir(parents=True)
    (target / "observations.json").write_bytes(path.read_bytes())
    by_prompt = {}
    for row in observations:
        by_prompt.setdefault(row["prompt_id"], []).append(float(row["success"]))
    reliability = float(np.mean([np.mean(v) for v in by_prompt.values()]))
    write_json(target / "summary.json", {"run_id": run_id, "evaluation_id": evaluation_id,
               "controller": payload["controller"], "shift": payload["shift"],
               "protocol_id": payload["protocol_id"], "n_prompts": len(by_prompt),
               "n_generations": len(observations), "reliability": reliability,
               "reliability_percent": 100 * reliability, "source_sha256": sha256(path),
               "success_definition": payload["success_definition"], "matching_rule": payload["matching_rule"]})
    return target


def rho(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def r_squared(y, prediction):
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    return None if denominator == 0 else 1 - float(np.sum((y - prediction) ** 2)) / denominator


def prepare_panels(analysis_id: str, controller: str, shift: str, protocol: str,
                   normalization: str, include_synthetic: bool) -> Path:
    """Save panel-ready rows and held-out regression predictions, without rerunning synthesis."""
    destination = CACHE / "analyses" / safe_id(analysis_id)
    if destination.exists():
        raise ValueError("Analysis ID exists; use a new analysis_id to preserve prior results")
    rows, exclusions, sources = [], [], {}
    for path in sorted((CACHE / "runs").glob("*/score.json")):
        verify_run(path.parent)
        record = read_json(path)
        if record["protocol_id"] != protocol or record["normalization_protocol_id"] != normalization:
            continue
        sources[str(path.relative_to(CACHE))] = sha256(path)
        reason = None
        if not record["score_available"]:
            reason = "infeasible or unconverged synthesis"
        elif record["synthetic"] != include_synthetic:
            reason = "synthetic/empirical cohort mismatch"
        elif record["coordinates"] != "normalized" or record["stage_costs_depth_weighted"] is not True:
            reason = "cross-model normalized coordinates required"
        evaluations = []
        for candidate in sorted((path.parent / "evaluations").glob("*/summary.json")):
            result = read_json(candidate)
            if result["controller"] == controller and result["shift"] == shift:
                evaluations.append(result)
                sources[str(candidate.relative_to(CACHE))] = sha256(candidate)
        if len(evaluations) != 1:
            reason = "exactly one matching evaluation per run is required"
        if reason:
            exclusions.append({"run_id": record["run_id"], "reason": reason})
            continue
        evaluation = evaluations[0]
        rows.append({**{k: record[k] for k in ("run_id", "model_id", "model_family", "behavior", "parameter_count")},
                     **record["predictors"], **evaluation})
    pairs = [(row["model_id"], row["behavior"]) for row in rows]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Multiple runs for a model-behavior pair; select a unique predefined protocol")
    matching_rules = {json.dumps(row["matching_rule"], sort_keys=True) for row in rows}
    if len(matching_rules) > 1:
        raise ValueError("Controller comparison budgets/matching rules differ across pairs")
    predictions, metrics, correlations, descriptive_lines = [], [], [], []
    skipped = []
    y = np.array([row["reliability"] for row in rows])
    rng = np.random.default_rng(2151)
    for predictor in PREDICTORS:
        if len(rows) < 3 or any(row[predictor] is None for row in rows):
            skipped.append({"predictor": predictor, "reason": "fewer than 3 pairs or missing predictor values"})
            continue
        x = np.array([row[predictor] for row in rows], dtype=float)
        models = np.array([row["model_id"] for row in rows])
        bootstrap_values = []
        unique_models = np.unique(models)
        for _ in range(500):
            drawn = rng.choice(unique_models, size=len(unique_models), replace=True)
            indices = np.concatenate([np.flatnonzero(models == m) for m in drawn])
            value = rho(x[indices], y[indices])
            if value is not None:
                bootstrap_values.append(value)
        interval = np.quantile(bootstrap_values, [0.025, 0.975]).tolist() if len(bootstrap_values) >= 100 else [None, None]
        correlations.append({"predictor": predictor, "spearman_rho": rho(x, y),
                             "cluster_bootstrap_ci95": interval, "valid_bootstrap_samples": len(bootstrap_values)})
        coefficients = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
        descriptive_lines.append({"predictor": predictor, "intercept": coefficients[0], "slope": coefficients[1],
                                  "x_min": x.min(), "x_max": x.max(), "purpose": "descriptive in-sample line only"})
        for scheme, group_key in (("leave_model_out", "model_id"), ("leave_family_out", "model_family"),
                                  ("leave_behavior_out", "behavior"), ("leave_scale_out", "parameter_count")):
            groups = np.array([str(row[group_key]) for row in rows])
            if len(np.unique(groups)) < 2:
                skipped.append({"predictor": predictor, "scheme": scheme, "reason": "fewer than 2 groups"})
                continue
            predicted = np.full(len(y), np.nan)
            for held_out in np.unique(groups):
                train, test = groups != held_out, groups == held_out
                if train.sum() < 3:
                    continue
                mean, std = float(x[train].mean()), float(x[train].std())
                # Constant training predictors reduce to a training-mean model.
                scale = std if std > 0 else 1.0
                train_x = np.column_stack([np.ones(train.sum()), (x[train] - mean) / scale])
                beta = np.linalg.lstsq(train_x, y[train], rcond=None)[0]
                predicted[test] = beta[0] + beta[1] * (x[test] - mean) / scale
                for index in np.flatnonzero(test):
                    predictions.append({"run_id": rows[index]["run_id"], "model_family": rows[index]["model_family"],
                                        "behavior": rows[index]["behavior"], "predictor": predictor,
                                        "scheme": scheme, "held_out": held_out, "observed": y[index],
                                        "predicted": predicted[index], "training_run_ids": [rows[i]["run_id"] for i in np.flatnonzero(train)],
                                        "training_x_mean": mean, "training_x_scale": scale,
                                        "intercept_standardized": beta[0], "slope_standardized": beta[1]})
            complete = bool(np.isfinite(predicted).all())
            metrics.append({"predictor": predictor, "scheme": scheme, "n_pairs": len(y),
                            "n_predicted": int(np.isfinite(predicted).sum()), "complete": complete,
                            "r_squared": r_squared(y, predicted) if complete else None,
                            "spearman_rho": rho(predicted, y) if complete else None})
    destination.mkdir(parents=True)
    for name, contents in (("pair_table", rows), ("excluded_runs", exclusions),
                           ("cv_predictions", predictions), ("cv_metrics", metrics),
                           ("correlations", correlations), ("descriptive_lines", descriptive_lines),
                           ("unavailable_analyses", skipped)):
        write_json(destination / f"{name}.json", contents)
    write_json(destination / "panel_manifest.json", {
        "analysis_id": analysis_id, "controller": controller, "shift": shift,
        "protocol_id": protocol, "normalization_protocol_id": normalization,
        "synthetic": include_synthetic, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_hashes": sources, "exporter_sha256": sha256(Path(__file__)),
        "seed": 2151, "bootstrap_repetitions": 500,
        "panel_A": "runs/*/calibration_input.pt, manifest.json, score.json",
        "panel_B": "pair_table.json, correlations.json, descriptive_lines.json",
        "panel_C": "cv_metrics.json; leave_model_out, shared cohort, univariate OLS",
        "panel_D": "cv_predictions.json; leave_family_out, s_rob or negative_log_gamma_star",
        "response": "prompt-averaged binary success fraction; multiply by 100 for percent",
        "limitations": ["No held-out outcome is used by score()", "S_rob is not restricted to [0,1]",
                        "Point estimates use the current solver's numerical feasibility boundary",
                        "This is linear binary-success prediction, not the draft's mixed-effects model",
                        "No inference can verify the declared provenance of upstream tensors"]})
    return destination


def score_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--bundle", type=Path, required=True)
    score_parser.add_argument("--device", default="cpu")
    evaluation_parser = commands.add_parser("evaluate")
    evaluation_parser.add_argument("--input", type=Path, required=True)
    panel_parser = commands.add_parser("prepare-panels")
    panel_parser.add_argument("--analysis-id", required=True)
    panel_parser.add_argument("--controller", required=True)
    panel_parser.add_argument("--shift", required=True)
    panel_parser.add_argument("--protocol-id", required=True)
    panel_parser.add_argument("--normalization-id", required=True)
    panel_parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "score":
        result = score(args.bundle, args.device)
    elif args.command == "evaluate":
        result = evaluate(args.input)
    else:
        result = prepare_panels(args.analysis_id, args.controller, args.shift, args.protocol_id,
                                args.normalization_id, args.synthetic)
    print(result)



def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in {"score", "evaluate", "prepare-panels"}:
        score_main(sys.argv[1:])
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 768],
    )
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    if not args.device.startswith("cuda"):
        raise ValueError("--device must select a CUDA device for this diagnostic")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the H-infinity optimization unit")
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2")
    if min(args.dimensions) <= 0:
        raise ValueError("all benchmark dimensions must be positive")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    reference = load_reference()
    dimensions = sorted(set(args.dimensions))
    cache_config = {
        "device": args.device,
        "dimensions": dimensions,
        "repeats": args.repeats,
        "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "reference_sha256": REFERENCE_SHA256,
        "optimized_sha256": file_sha256(
            REPO_ROOT / "robust_steerability" / "control" / "h_infinity.py"
        ),
    }
    cache_config_path = CACHE_DIR / "config.json"
    parity_cache_path = CACHE_DIR / "equivalence.csv"
    timings_cache_path = CACHE_DIR / "timings.csv"
    cache_exists = all(
        path.exists()
        for path in [cache_config_path, parity_cache_path, timings_cache_path]
    )
    if args.recompute or not cache_exists:
        devices = ["cpu", args.device]
        parity = run_parity(reference, devices)
        timings = run_benchmark(
            reference,
            gpu_device=args.device,
            dimensions=dimensions,
            repeats=args.repeats,
        )
        parity.to_csv(parity_cache_path, index=False)
        timings.to_csv(timings_cache_path, index=False)
        cache_config_path.write_text(json.dumps(cache_config, indent=2) + "\n")
    else:
        saved_config = json.loads(cache_config_path.read_text())
        if saved_config != cache_config:
            raise ValueError("Cached benchmark configuration differs; rerun with --recompute")
        parity = pd.read_csv(parity_cache_path)
        timings = pd.read_csv(timings_cache_path)
    benchmark = summarize_benchmark(timings)
    benchmark.to_csv(CACHE_DIR / "benchmark_summary.csv", index=False)
    for path in [
        PLOTS_DIR / "equivalence.csv",
        PLOTS_DIR / "benchmark_summary.csv",
        PLOTS_DIR / "h_infinity_optimization.pdf",
        PLOTS_DIR / "h_infinity_optimization.png",
        PLOTS_DIR / "summary.json",
    ]:
        if path.exists():
            path.unlink()
    parity.to_csv(PLOTS_DIR / "equivalence.csv", index=False)
    benchmark.to_csv(PLOTS_DIR / "benchmark_summary.csv", index=False)
    gpu_name = torch.cuda.get_device_name(torch.device(args.device))
    plot_diagnostics(parity, benchmark, gpu_name=gpu_name)
    summary = write_summary(
        parity,
        benchmark,
        timings,
        gpu_device=args.device,
    )
    print(json.dumps(summary, indent=2))
    if not summary["all_parity_passed"]:
        raise RuntimeError("H-infinity reference parity failed")


if __name__ == "__main__":
    main()
