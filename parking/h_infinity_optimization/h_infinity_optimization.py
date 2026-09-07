from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.lines import Line2D

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
BENCHMARK_PROTOCOL_VERSION = 1
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
    return payload


def main() -> None:
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
