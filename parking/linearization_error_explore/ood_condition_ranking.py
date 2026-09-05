from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
SUMMARY_PATH = Path(__file__).resolve().parent.parent / "ood_explore" / "plots" / "lqr_ood_failure_summary.json"
OUTPUT_PATH = PLOTS_DIR / "ood_condition_ranking.png"

with SUMMARY_PATH.open("r", encoding="utf-8") as fh:
    summary = json.load(fh)

conditions = summary["conditions"]
items = []
for name, stats in conditions.items():
    items.append(
        {
            "name": name,
            "label": {
                "id": "ID",
                "dataset_shift": "Dataset",
                "translation": "Spanish",
                "code_switching": "Code-switch",
                "pragmatic_reversal": "Pragmatic",
                "surface_corruption": "Corrupted",
                "domain_topic_shift": "Domain",
                "long_context_switch": "Long ctx",
                "concept_collision": "Collision",
                "adversarial_ood": "Adversarial",
            }.get(name, name),
            "median": float(stats["median"]),
            "q1": float(stats["q1"]),
            "q3": float(stats["q3"]),
        }
    )

# Sort by median to expose the strongest OOD shifts.
items = sorted(items, key=lambda x: x["median"], reverse=True)

fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
fig.patch.set_facecolor("#f8f8f8")
ax.set_facecolor("#f8f8f8")

positions = np.arange(len(items))
colors = [
    "#7c4dff" if item["name"] == "id" else
    "#4f87d9" if item["name"] in {"dataset_shift", "code_switching", "pragmatic_reversal", "concept_collision"} else
    "#f59e0b" if item["name"] in {"translation", "domain_topic_shift", "long_context_switch"} else
    "#ef4444"
    for item in items
]

bar_container = ax.bar(
    positions,
    [item["median"] for item in items],
    yerr=[
        [item["median"] - item["q1"] for item in items],
        [item["q3"] - item["median"] for item in items],
    ],
    capsize=5,
    color=colors,
    edgecolor="black",
    linewidth=1.2,
    alpha=0.9,
)

for i, item in enumerate(items):
    ax.text(
        i,
        item["median"] + 1.7,
        f"{item['median']:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    if item["name"] in {"long_context_switch", "translation", "domain_topic_shift", "adversarial_ood"}:
        ax.scatter(i, item["median"] + 4.0, marker="*", s=120, color="#111111", zorder=5)

ax.set_xticks(positions)
ax.set_xticklabels([item["label"] for item in items], rotation=35, ha="right")
ax.set_ylabel("Median remaining target error (% of unsteered error)")
ax.set_title("OOD family ranking by median remaining error")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.set_ylim(0, max(50, max(item["median"] for item in items) * 1.25))

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
print(f"Saved: {OUTPUT_PATH}")
print("Ordered ranking:")
for idx, item in enumerate(items, start=1):
    print(f"{idx}. {item['label']}: median {item['median']:.2f}%")
