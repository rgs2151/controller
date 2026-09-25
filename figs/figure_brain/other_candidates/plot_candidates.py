#!/usr/bin/env python3
"""Flip-through exploration of closed-loop candidates with P10 held fixed."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
UNIT = HERE.parent
DATA = UNIT.parents[1] / "data" / "brain"
RESULTS = UNIT / "results"
ANALYSIS_VERSION = "artifact-free-stimulation-context-candidate-screen-v1"


def load_parent():
    path = UNIT / "stimulation_context_illustration.py"
    spec = importlib.util.spec_from_file_location("stim_context", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


parent = load_parent()


CLOSED_CASES = (
    # LMF is P14's marginal domain winner but only four clean channels overlap
    # the stimulation montage; LAT is the highest-ranked valid matched lead.
    dict(subject=14, baseline_session=1, stim_session=1, lead="LAT", policy="Closed-loop"),
    dict(subject=15, baseline_session=1, stim_session=1, lead="LAT", policy="Closed-loop"),
    dict(subject=17, baseline_session=1, stim_session=1, lead="LPF", policy="Closed-loop"),
)


def candidate_residuals() -> pd.DataFrame:
    path = HERE / "closed_loop_context_residuals.csv"
    if path.exists():
        frame = pd.read_csv(path)
        complete = (set(frame.subject_number) == {14, 15, 17}
                    and set(frame.horizon_ms) == set(parent.HORIZONS_MS)
                    and "candidate_analysis_version" in frame
                    and set(frame.candidate_analysis_version) == {ANALYSIS_VERSION})
        if complete:
            print(f"reuse candidate residual cache: {path}", flush=True)
            return frame
    rows = []
    for case in CLOSED_CASES:
        print(f"compute P{case['subject']} closed-loop context", flush=True)
        case_rows = parent.compute_context_case(case, False)
        for row in case_rows:
            row["candidate_analysis_version"] = ANALYSIS_VERSION
        rows.extend(case_rows)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def candidate_accuracy() -> pd.DataFrame:
    rows = []
    all_cases = (parent.CASES[0],) + CLOSED_CASES
    for case in all_cases:
        subject = case["subject"]
        baseline = sorted(DATA.glob(
            f"BIPOLFieldTripFormat_AlignedToImagePresent_P{subject}_NoStim*.mat"))
        stimulation = sorted(DATA.glob(
            f"BIPOLFieldTripFormat_AlignedToImagePresent_P{subject}_Stim*.mat"))
        rows.append(parent.accuracy_row(subject, "No stimulation", baseline))
        rows.append(parent.accuracy_row(subject, case["policy"], stimulation))
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "candidate_accuracy.csv", index=False)
    return frame


def accuracy_panel(axis, accuracy: pd.DataFrame, candidate: int):
    ordered = pd.concat([
        accuracy[(accuracy.subject == "P10") & (accuracy.condition == "No stimulation")],
        accuracy[(accuracy.subject == "P10") & (accuracy.condition == "Open-loop")],
        accuracy[(accuracy.subject == f"P{candidate}") & (accuracy.condition == "No stimulation")],
        accuracy[(accuracy.subject == f"P{candidate}") & (accuracy.condition == "Closed-loop")],
    ], ignore_index=True)
    labels = ["P10\nNo stim", "P10\nOpen-loop",
              f"P{candidate}\nNo stim", f"P{candidate}\nClosed-loop"]
    colors = [parent.ID_COLOR, parent.OPEN_COLOR, parent.ID_COLOR, parent.CLOSED_COLOR]
    x = np.arange(len(ordered)); y = ordered.accuracy_pct.to_numpy()
    errors = np.vstack([y - ordered.ci_low_pct.to_numpy(),
                        ordered.ci_high_pct.to_numpy() - y])
    axis.bar(x, y, color=colors, width=0.68)
    axis.errorbar(x, y, yerr=errors, fmt="none", ecolor="black", capsize=3, lw=1.2)
    axis.set_xticks(x, labels, fontsize=9)
    axis.set_ylim(70, 101)
    axis.set_ylabel("Response accuracy (%)", fontsize=11)
    axis.set_title("Behavioral performance", fontsize=13, weight="bold")
    axis.grid(axis="y", color="#D9DDE1", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    for xi, yi in zip(x, y):
        axis.text(xi, yi + 0.65, f"{yi:.1f}", ha="center", va="bottom", fontsize=9)


def render(domain: pd.DataFrame, open_loop: pd.DataFrame, closed: pd.DataFrame,
           accuracy: pd.DataFrame):
    global_max = 1.14 * max(domain.linear_residual_rms.max(),
                            open_loop.linear_residual_rms.max(),
                            closed.linear_residual_rms.max())
    for case in CLOSED_CASES:
        candidate = case["subject"]
        candidate_frame = closed[closed.subject_number == candidate]
        fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.25),
                                 gridspec_kw={"width_ratios": [1.15, 1, 1, 0.9]})
        parent.grouped_bars(axes[0], domain, "Domain shift\nP10 · LAT · no stimulation")
        parent.grouped_bars(axes[1], open_loop,
                            "Stimulation-context shift\nP10 · open-loop")
        parent.grouped_bars(axes[2], candidate_frame,
                            f"Stimulation-context shift\nP{candidate} · closed-loop")
        for axis in axes[:3]:
            axis.set_ylim(0, global_max)
        accuracy_panel(axes[3], accuracy, candidate)
        handles = [plt.Rectangle((0, 0), 1, 1, color=parent.ID_COLOR, label="ID"),
                   plt.Rectangle((0, 0), 1, 1, color=parent.OOD_COLOR, label="OOD")]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.39, 1.04),
                   ncol=2, frameon=False, fontsize=11)
        fig.subplots_adjust(left=0.06, right=0.99, bottom=0.18, top=0.78, wspace=0.38)
        for extension in ("png", "pdf"):
            fig.savefig(HERE / f"p10_open_vs_p{candidate}_closed.{extension}",
                        dpi=220, bbox_inches="tight")
        plt.close(fig)


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    domain = pd.read_csv(RESULTS / "all_subject_conflict_linear_residual.csv")
    domain = domain[domain.subject_number == 10]
    current = pd.read_csv(RESULTS / "stimulation_context_residual_rms.csv")
    open_loop = current[current.subject_number == 10]
    closed = candidate_residuals()
    accuracy = candidate_accuracy()
    render(domain, open_loop, closed, accuracy)
    summary = (closed.pivot_table(index=["subject", "subject_number", "lead", "horizon_ms"],
                                  columns="split", values="linear_residual_rms")
               .reset_index())
    summary["ood_over_id"] = summary.OOD / summary.ID
    ranked = (summary.groupby(["subject", "subject_number", "lead"], as_index=False)
              .agg(mean_id=("ID", "mean"), mean_ood=("OOD", "mean"),
                   mean_ood_over_id=("ood_over_id", "mean"))
              .sort_values("mean_ood_over_id"))
    ranked.to_csv(HERE / "candidate_summary.csv", index=False)
    print(ranked.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
