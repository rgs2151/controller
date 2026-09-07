from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
SUMMARY_PATH = Path(__file__).resolve().parent.parent / "ood_explore" / "plots" / "lqr_ood_failure_summary.json"


def setup_style() -> None:
    sns.set_theme(context="talk", style="ticks", palette="dark")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["patch.linewidth"] = 0
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300


setup_style()

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

fig, ax = plt.subplots(figsize=(7.0, 3.8))

positions = np.arange(len(items))
colors = [
    "#737373" if item["name"] == "id" else
    "midnightblue" if item["name"] in {"dataset_shift", "code_switching", "pragmatic_reversal", "concept_collision"} else
    "darkorange" if item["name"] in {"translation", "domain_topic_shift", "long_context_switch"} else
    "darkred"
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
)

for i, item in enumerate(items):
    ax.text(
        i,
        item["median"] + 0.8,
        f"{item['median']:.0f}",
        ha="center",
        va="bottom",
        fontsize=7,
    )
    if item["name"] in {"long_context_switch", "translation", "domain_topic_shift", "adversarial_ood"}:
        ax.scatter(i, item["median"] + 2.4, marker="*", s=65, facecolor="none", edgecolor="black", zorder=5)

ax.set_xticks(positions)
ax.set_xticklabels([item["label"] for item in items], rotation=35, ha="right")
ax.set_ylabel(r"Remaining target error (\%)")
upper = max(50, max(item["median"] for item in items) * 1.25)
ax.set_ylim(0, upper)
ax.set_yticks([0, upper])
sns.despine(ax=ax, trim=True, offset=10)
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.31, top=0.96)
for extension in ("pdf", "png"):
    output_path = PLOTS_DIR / f"ood_condition_ranking.{extension}"
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
print("Ordered ranking:")
for idx, item in enumerate(items, start=1):
    print(f"{idx}. {item['label']}: median {item['median']:.2f}%")
