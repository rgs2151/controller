from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
PLOTS_DIR = ROOT / "plots"
ACT_DIR = CACHE_DIR / "activations" / "adversarial"
SUMMARY_CSV = CACHE_DIR / "network_size_residual_summary_adversarial.csv"
SUMMARY_JSON = CACHE_DIR / "network_size_residual_summary_adversarial.json"

REGIME_COLORS = {"early": "#0f766e", "mid": "#b45309", "late": "#7c3aed"}
FAMILY_COLORS = {"gpt2": "#2563eb", "qwen": "#dc2626", "custom": "#52525b"}


def load_table() -> pd.DataFrame:
    return pd.read_csv(SUMMARY_CSV)


def load_activation_stats() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for path in sorted(ACT_DIR.glob("*.pt")):
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        meta = bundle["metadata"]
        within = bundle["sets"]["Within"]
        adversarial = bundle["sets"]["Adversarial"]
        layer_count = len(within["last_token_states"][0]) - 1
        rows.append(
            {
                "label": meta["label"],
                "model_name": meta["model_name"],
                "family": meta["family"],
                "params": int(meta["params"]),
                "layers": int(layer_count),
                "within_prompts": int(len(within["prompts"])),
                "adversarial_prompts": int(len(adversarial["prompts"])),
                "within_mean_tokens": float(np.mean(within["token_count"])),
                "adversarial_mean_tokens": float(np.mean(adversarial["token_count"])),
                "within_max_tokens": int(np.max(within["token_count"])),
                "adversarial_max_tokens": int(np.max(adversarial["token_count"])),
            }
        )
    return rows


def write_supplement_stats(rows: list[dict[str, float | int | str]]) -> tuple[Path, Path]:
    csv_path = CACHE_DIR / "network_size_residual_supplement_stats.csv"
    tex_path = CACHE_DIR / "network_size_residual_supplement_stats.tex"
    fieldnames = [
        "label",
        "model_name",
        "family",
        "params",
        "layers",
        "within_prompts",
        "adversarial_prompts",
        "within_mean_tokens",
        "adversarial_mean_tokens",
        "within_max_tokens",
        "adversarial_max_tokens",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & Params (B) & Layers & ID prompts & Adv. prompts & Mean ID tokens & Mean Adv. tokens \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['label']} & {row['params'] / 1e9:.3f} & {row['layers']} & {row['within_prompts']} & {row['adversarial_prompts']} & {row['within_mean_tokens']:.1f} & {row['adversarial_mean_tokens']:.1f} \\\""
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Supplementary model and prompt statistics for the adversarial network-size sweep. Layer counts correspond to decoder block depth; token counts report the mean prompt length after tokenization and truncation.}",
            r"\label{tab:network_size_residual_supplement_stats}",
            r"\end{table}",
        ]
    )
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, tex_path


def write_paper_latex(table: pd.DataFrame) -> Path:
    out_path = CACHE_DIR / "network_size_residual_summary_adversarial_paper.tex"
    deltas = {
        "early": table["early_ood"] - table["early_id"],
        "mid": table["mid_ood"] - table["mid_id"],
        "late": table["late_ood"] - table["late_id"],
    }
    max_idx = {regime: int(np.argmax(delta.to_numpy())) for regime, delta in deltas.items()}

    def fmt(value: float, bold: bool = False) -> str:
        text = f"{value:.3f}"
        return rf"\textbf{{{text}}}" if bold else text

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{Early} & \multicolumn{3}{c}{Mid} & \multicolumn{3}{c}{Late} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r"Model & Params (B) & ID & Adv. & $\Delta$ & ID & Adv. & $\Delta$ & ID & Adv. & $\Delta$ \\",
        r"\midrule",
    ]
    for idx, row in table.iterrows():
        early_delta = row["early_ood"] - row["early_id"]
        mid_delta = row["mid_ood"] - row["mid_id"]
        late_delta = row["late_ood"] - row["late_id"]
        lines.append(
            "{} & {:.3f} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                row["label"],
                row["params"] / 1e9,
                fmt(row["early_id"]),
                fmt(row["early_ood"], idx == max_idx["early"]),
                fmt(early_delta, idx == max_idx["early"]),
                fmt(row["mid_id"]),
                fmt(row["mid_ood"], idx == max_idx["mid"]),
                fmt(mid_delta, idx == max_idx["mid"]),
                fmt(row["late_id"]),
                fmt(row["late_ood"], idx == max_idx["late"]),
                fmt(late_delta, idx == max_idx["late"]),
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Residual summary for the 50-prompt within-distribution versus adversarial network-size sweep. $\Delta$ denotes adversarial minus within-distribution residual magnitude. Bold entries mark the strongest adversarial amplification within each layer regime.}",
            r"\label{tab:network_size_residual_adversarial_paper}",
            r"\end{table*}",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def make_amplification_plot(table: pd.DataFrame) -> Path:
    out_path = PLOTS_DIR / "network_size_residual_amplification_adversarial.png"
    regimes = ["early", "mid", "late"]
    fig, ax = plt.subplots(figsize=(10.8, 5.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    xs = table["params_log10"].to_numpy()
    for regime in regimes:
        delta = table[f"{regime}_ood"].to_numpy() - table[f"{regime}_id"].to_numpy()
        ax.plot(xs, delta, marker="o", linewidth=2.2, color=REGIME_COLORS[regime], label=f"{regime.title()} layers")
        for _, row in table.iterrows():
            family_color = FAMILY_COLORS.get(row["family"], FAMILY_COLORS["custom"])
            y = row[f"{regime}_ood"] - row[f"{regime}_id"]
            ax.scatter(row["params_log10"], y, s=70, color=family_color, edgecolor="black", linewidth=0.4, zorder=3)
    for _, row in table.iterrows():
        y = row["mid_ood"] - row["mid_id"]
        ax.text(row["params_log10"] + 0.015, y, row["label"], fontsize=7.5)
    ax.axhline(0.0, color="#111827", linewidth=1.1, linestyle=":")
    ax.set_xlabel(r"$\log_{10}$(parameters)", fontsize=12)
    ax.set_ylabel("Adversarial - within residual", fontsize=12)
    ax.set_title("Adversarial residual amplification across network size", fontsize=15, fontweight="bold")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.35)
    ax.legend(frameon=False, fontsize=10)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    table = load_table()
    supplement_rows = load_activation_stats()
    with SUMMARY_JSON.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    order = {row["label"]: idx for idx, row in enumerate(summary)}
    supplement_rows.sort(key=lambda row: order[row["label"]])

    paper_table_path = write_paper_latex(table)
    supplement_csv_path, supplement_tex_path = write_supplement_stats(supplement_rows)
    amplification_plot_path = make_amplification_plot(table)

    print(f"Saved paper table to: {paper_table_path}")
    print(f"Saved supplement CSV to: {supplement_csv_path}")
    print(f"Saved supplement TeX to: {supplement_tex_path}")
    print(f"Saved amplification plot to: {amplification_plot_path}")


if __name__ == "__main__":
    main()
