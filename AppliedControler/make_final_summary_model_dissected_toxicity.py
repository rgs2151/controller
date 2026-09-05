"""Render the model-dissected toxicity summary figure.

Same structure as make_final_summary_model_dissected.py (rows: models,
columns: metrics, bars: methods), but for RealToxicityPrompts toxicity.

Inputs (results_reports/):
  paper_style_table_paper_like_calibrated.csv          (toxicity % + % change)
  final_table_method_summary_paper_like_calibrated.csv (mean toxicity score)

Outputs (results_reports/):
  final_summary_model_dissected_toxicity.png / .pdf
"""

from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
MODEL_ORDER = [
    "DistilGPT-2",
    "Qwen-2.5-1.5B",
    "Qwen-2.5-7B",
    "Qwen-2.5-14B",
]
MODEL_LABELS = {
    "DistilGPT-2": "DistilGPT-2",
    "Qwen-2.5-1.5B": "Qwen-2.5-1.5B",
    "Qwen-2.5-7B": "Qwen-2.5-7B",
    "Qwen-2.5-14B": "Qwen-2.5-14B",
}
# (key, label, log_scale)
METRICS = [
    ("tox_rate", r"RTP toxicity (%)  $\downarrow$  (log)", True),
    ("tox_reduction", r"Toxicity reduction vs Original (%)  $\uparrow$", False),
]


def load_toxicity() -> dict[str, dict[str, dict[str, float]]]:
    """Return {model: {method: {metric: value}}}."""
    rate = pd.read_csv(
        os.path.join(REPORTS, "paper_style_table_paper_like_calibrated.csv")
    )

    data: dict[str, dict[str, dict[str, float]]] = {}
    for _, row in rate.iterrows():
        model = str(row["label"])
        method = str(row["method"])
        entry = data.setdefault(model, {}).setdefault(method, {})
        entry["tox_rate"] = float(row["toxicity_percent"])
        entry["tox_reduction"] = -float(row["percent_change"])
    return data


def main() -> None:
    data = load_toxicity()

    n_rows, n_cols = len(MODEL_ORDER), len(METRICS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.2 * n_cols, 2.6 * n_rows), sharex=True
    )

    for i, model in enumerate(MODEL_ORDER):
        sub = data.get(model, {})
        for k, (metric_key, metric_label, log_scale) in enumerate(METRICS):
            ax = axes[i, k]
            means, missing = [], []
            for method in METHODS:
                value = sub.get(method, {}).get(metric_key)
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    means.append(0.0)
                    missing.append(True)
                else:
                    means.append(float(value))
                    missing.append(False)

            bars = ax.bar(
                METHODS,
                means,
                color=[METHOD_COLORS[m] for m in METHODS],
                edgecolor="black",
                linewidth=0.6,
            )

            if log_scale:
                ax.set_yscale("log")
                ax.set_ylim(1e-3, 20)
                na_y = 1.5e-3
                for bar, mean, miss in zip(bars, means, missing):
                    if not miss:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            mean * 1.3,
                            f"{mean:.3g}",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                        )
            else:
                ax.set_ylim(0, 105)
                na_y = 2.0
                for bar, mean, miss in zip(bars, means, missing):
                    if not miss and mean > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            mean + 2,
                            f"{mean:.1f}",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                        )

            for bar, miss in zip(bars, missing):
                if miss:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        na_y,
                        "N/A",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90,
                        color="#b30000",
                    )

            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(metric_label, fontsize=10)
            if k == 0:
                ax.set_ylabel(MODEL_LABELS[model], fontsize=10, fontweight="bold")
            if i == n_rows - 1:
                ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        "Control-theoretic steering of LLMs: per-model toxicity breakdown "
        "(rows: models, columns: metrics, bars: methods)",
        fontsize=12,
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        out = os.path.join(REPORTS, f"final_summary_model_dissected_toxicity.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
