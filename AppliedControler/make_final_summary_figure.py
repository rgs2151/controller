"""Render the final cross-benchmark summary figure.

Panel A: TruthfulQA T*I (%) per method, grouped by model (with std error bars).
Panel B: RealToxicityPrompts toxicity (%) per method for DistilGPT-2 (log scale).

Inputs (results_reports/):
  paper_style_table_truthfulness_ours_plus_qwen14b_methods.csv  (DistilGPT-2 + Qwen-14B rows)
  paper_style_table_truthfulness_qwen7b_*_rerun.csv             (optional Qwen-7B rows)
  paper_style_table_paper_like_calibrated.csv                   (toxicity rows)

Outputs (results_reports/):
  final_summary_figure.png / final_summary_figure.pdf
"""

from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "results_reports")

METHODS = ["Original", "A-LQR", "S-PID", "H-infinity"]
METHOD_COLORS = {
    "Original": "#8c8c8c",
    "A-LQR": "#4c72b0",
    "S-PID": "#dd8452",
    "H-infinity": "#55a868",
}
MODEL_ORDER = ["DistilGPT-2-ours", "Qwen-2.5-7B-ours", "Qwen-2.5-14B-ours"]
MODEL_LABELS = {
    "DistilGPT-2-ours": "DistilGPT-2",
    "Qwen-2.5-7B-ours": "Qwen-2.5-7B",
    "Qwen-2.5-14B-ours": "Qwen-2.5-14B",
}


def load_truthfulness() -> pd.DataFrame:
    frames = []
    canonical = os.path.join(
        REPORTS, "paper_style_table_truthfulness_ours_plus_qwen14b_methods.csv"
    )
    frames.append(pd.read_csv(canonical))
    for stage in ("original", "alqr", "spid", "hinf"):
        path = os.path.join(
            REPORTS, f"paper_style_table_truthfulness_qwen7b_{stage}_rerun.csv"
        )
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["model", "method"], keep="last")
    return df


def main() -> None:
    truth = load_truthfulness()
    tox = pd.read_csv(os.path.join(REPORTS, "paper_style_table_paper_like_calibrated.csv"))

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.65, 1.0]}
    )

    # ---------------- Panel A: truthfulness ----------------
    models = [m for m in MODEL_ORDER if (truth["model"] == m).any()]
    n_methods = len(METHODS)
    width = 0.19
    x = np.arange(len(models))
    for j, method in enumerate(METHODS):
        means, stds, missing = [], [], []
        for model in models:
            row = truth[(truth["model"] == model) & (truth["method"] == method)]
            if row.empty or math.isnan(float(row["ti_mean"].iloc[0])):
                means.append(0.0)
                stds.append(0.0)
                missing.append(True)
            else:
                means.append(float(row["ti_mean"].iloc[0]))
                stds.append(float(row["ti_std"].iloc[0]))
                missing.append(False)
        offs = x + (j - (n_methods - 1) / 2) * width
        bars = ax_a.bar(
            offs,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=method,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.6,
        )
        for off, bar, miss in zip(offs, bars, missing):
            if miss:
                ax_a.text(
                    off,
                    1.0,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                    color="#b30000",
                )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax_a.set_ylabel(r"TruthfulQA  T$\cdot$I (%)  $\uparrow$")
    ax_a.set_title("A. Truthfulness across model scales")
    ax_a.legend(frameon=False, fontsize=9)
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.set_ylim(bottom=0)

    # ---------------- Panel B: toxicity ----------------
    tox_d = tox[tox["label"] == "DistilGPT-2"].set_index("method")
    vals = [float(tox_d.loc[m, "toxicity_percent"]) for m in METHODS]
    bars = ax_b.bar(
        METHODS,
        vals,
        color=[METHOD_COLORS[m] for m in METHODS],
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, v in zip(bars, vals):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2,
            v * 1.15,
            f"{v:.3g}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax_b.set_yscale("log")
    ax_b.set_ylabel(r"RTP toxicity (%)  $\downarrow$  (log)")
    ax_b.set_title("B. Toxicity steering (DistilGPT-2)")
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.tick_params(axis="x", rotation=15)

    fig.suptitle(
        "Control-theoretic steering of LLMs: truthfulness and toxicity summary",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(REPORTS, f"final_summary_figure.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
