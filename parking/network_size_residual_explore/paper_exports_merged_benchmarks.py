from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

UNIT_DIR = Path(__file__).resolve().parent
CACHE_DIR = UNIT_DIR / "cache" / "benchmark_ood_sweep"
PLOTS_DIR = UNIT_DIR / "plots"
SUMMARY_CSV = CACHE_DIR / "benchmark_ood_sweep_summary.csv"


def baseline_for_subset(subset: str, mmlu_id_subset: str | None) -> str:
    if subset.startswith("id_"):
        return subset
    if subset.startswith("mmlu_"):
        if mmlu_id_subset is None:
            raise ValueError("MMLU subset present without MMLU ID baseline")
        return mmlu_id_subset
    return "id_rtp"


def subset_display(subset: str) -> str:
    if subset == "id_rtp":
        return "RTP (ID)"
    if subset.startswith("id_mmlu_"):
        return "MMLU ID: " + subset.replace("id_mmlu_", "").replace("_", " ")
    names = {
        "jigsaw_full": "Jigsaw full",
        "jigsaw_long": "Jigsaw long",
        "jigsaw_toxic": "Jigsaw toxic",
        "civil_full": "Civil full",
        "civil_long": "Civil long",
        "civil_toxic": "Civil toxic",
        "toxicchat_full": "ToxicChat full",
        "toxicchat_long": "ToxicChat long",
        "toxicchat_toxic": "ToxicChat toxic",
        "mmlu_ood_other_concepts": "MMLU OOD other concepts",
    }
    return names.get(subset, subset)


def add_delta_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label in df["label"].unique():
        block = df[df["label"] == label].copy()
        subset_to_row = {row["subset"]: row for _, row in block.iterrows()}
        mmlu_id_subset = next((s for s in subset_to_row if s.startswith("id_mmlu_")), None)
        for subset, row in subset_to_row.items():
            baseline_subset = baseline_for_subset(subset, mmlu_id_subset)
            baseline = subset_to_row[baseline_subset]
            out = row.to_dict()
            out["baseline_subset"] = baseline_subset
            out["delta_early"] = float(row["early"] - baseline["early"])
            out["delta_mid"] = float(row["mid"] - baseline["mid"])
            out["delta_late"] = float(row["late"] - baseline["late"])
            out["delta_overall"] = float(row["overall"] - baseline["overall"])
            rows.append(out)
    return pd.DataFrame(rows)


def write_main_table(df_delta: pd.DataFrame) -> tuple[Path, Path]:
    ood = df_delta[~df_delta["subset"].str.startswith("id_")].copy()
    rows = []
    for label in ood["label"].unique():
        block = ood[ood["label"] == label]
        top_overall = block.loc[block["delta_overall"].idxmax()]
        top_late = block.loc[block["delta_late"].idxmax()]
        mmlu = block[block["subset"] == "mmlu_ood_other_concepts"]
        mmlu_delta = float(mmlu.iloc[0]["delta_overall"]) if len(mmlu) else np.nan
        rows.append(
            {
                "label": label,
                "params_b": float(block.iloc[0]["params"]) / 1e9,
                "top_overall_subset": str(top_overall["subset"]),
                "top_overall_delta": float(top_overall["delta_overall"]),
                "top_late_subset": str(top_late["subset"]),
                "top_late_delta": float(top_late["delta_late"]),
                "mmlu_delta_overall": mmlu_delta,
            }
        )

    out_df = pd.DataFrame(rows).sort_values("params_b").reset_index(drop=True)
    csv_path = CACHE_DIR / "benchmark_network_size_condensed_table.csv"
    tex_path = CACHE_DIR / "benchmark_network_size_condensed_table.tex"
    out_df.to_csv(csv_path, index=False)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & Params (B) & Best overall OOD ($\Delta$) & Best late OOD ($\Delta$) & MMLU $\Delta_{overall}$ \\",
        r"\midrule",
    ]
    for _, row in out_df.iterrows():
        lines.append(
            f"{row['label']} & {row['params_b']:.3f} & {subset_display(row['top_overall_subset'])} ({row['top_overall_delta']:+.3f}) & {subset_display(row['top_late_subset'])} ({row['top_late_delta']:+.3f}) & {row['mmlu_delta_overall']:+.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Condensed network-size benchmark summary. Deltas are computed against matched ID baselines (RTP for toxicity subsets, MMLU-ID for MMLU OOD).}",
            r"\label{tab:benchmark_network_size_condensed}",
            r"\end{table*}",
        ]
    )
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, tex_path


def write_regime_winner_table(df_delta: pd.DataFrame) -> tuple[Path, Path]:
    ood = df_delta[~df_delta["subset"].str.startswith("id_")].copy()
    rows = []
    for label in ood["label"].unique():
        block = ood[ood["label"] == label]
        early = block.loc[block["delta_early"].idxmax()]
        mid = block.loc[block["delta_mid"].idxmax()]
        late = block.loc[block["delta_late"].idxmax()]
        rows.append(
            {
                "label": label,
                "params_b": float(block.iloc[0]["params"]) / 1e9,
                "early_winner": str(early["subset"]),
                "early_delta": float(early["delta_early"]),
                "mid_winner": str(mid["subset"]),
                "mid_delta": float(mid["delta_mid"]),
                "late_winner": str(late["subset"]),
                "late_delta": float(late["delta_late"]),
            }
        )

    out_df = pd.DataFrame(rows).sort_values("params_b").reset_index(drop=True)
    csv_path = CACHE_DIR / "benchmark_network_size_regime_winners.csv"
    tex_path = CACHE_DIR / "benchmark_network_size_regime_winners.tex"
    out_df.to_csv(csv_path, index=False)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & Params (B) & Early winner ($\Delta$) & Mid winner ($\Delta$) & Late winner ($\Delta$) \\",
        r"\midrule",
    ]
    for _, row in out_df.iterrows():
        lines.append(
            f"{row['label']} & {row['params_b']:.3f} & {subset_display(row['early_winner'])} ({row['early_delta']:+.3f}) & {subset_display(row['mid_winner'])} ({row['mid_delta']:+.3f}) & {subset_display(row['late_winner'])} ({row['late_delta']:+.3f}) \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Per-regime OOD winners across network size. Each delta is relative to the subset's matched ID baseline.}",
            r"\label{tab:benchmark_network_size_regime_winners}",
            r"\end{table*}",
        ]
    )
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, tex_path


def make_strongest_delta_plot(df_delta: pd.DataFrame) -> Path:
    ood = df_delta[~df_delta["subset"].str.startswith("id_")].copy()
    rows = []
    for label in ood["label"].unique():
        block = ood[ood["label"] == label]
        params_b = float(block.iloc[0]["params"]) / 1e9
        rows.append(
            {
                "label": label,
                "params_b": params_b,
                "best_early": float(block["delta_early"].max()),
                "best_mid": float(block["delta_mid"].max()),
                "best_late": float(block["delta_late"].max()),
            }
        )

    plot_df = pd.DataFrame(rows).sort_values("params_b").reset_index(drop=True)
    x = np.arange(len(plot_df))
    out_path = PLOTS_DIR / "benchmark_network_size_strongest_ood_deltas.png"

    fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    ax.plot(x, plot_df["best_early"], marker="o", linewidth=2.2, color="#0f766e", label="Early best delta")
    ax.plot(x, plot_df["best_mid"], marker="o", linewidth=2.2, color="#b45309", label="Mid best delta")
    ax.plot(x, plot_df["best_late"], marker="o", linewidth=2.2, color="#7c3aed", label="Late best delta")
    ax.axhline(0.0, color="#111827", linestyle=":", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=25, ha="right")
    ax.set_ylabel("Max OOD - matched ID residual")
    ax.set_title("Strongest OOD residual amplification by network size", fontsize=15, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.legend(frameon=False)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_caption_block(df_delta: pd.DataFrame) -> Path:
    ood = df_delta[~df_delta["subset"].str.startswith("id_")].copy()

    def mean_delta(subset: str, col: str) -> float:
        block = ood[ood["subset"] == subset]
        if len(block) == 0:
            return float("nan")
        return float(block[col].mean())

    jigsaw_long_late = mean_delta("jigsaw_long", "delta_late")
    toxicchat_long_late = mean_delta("toxicchat_long", "delta_late")
    mmlu_overall = mean_delta("mmlu_ood_other_concepts", "delta_overall")

    text = (
        "Across 9 model scales (DistilGPT-2 to Qwen 14B), benchmark-conditioned OOD prompts show "
        "non-uniform residual amplification with strongest effects in mid/late layers. "
        f"The average late-layer amplification is highest for Jigsaw-long (delta={jigsaw_long_late:+.3f}) "
        f"and remains positive for ToxicChat-long (delta={toxicchat_long_late:+.3f}), while MMLU concept-shift "
        f"shows smaller but measurable overall change (delta={mmlu_overall:+.3f}) relative to its matched ID subject baseline."
    )
    out_path = CACHE_DIR / "benchmark_network_size_caption_block.txt"
    out_path.write_text(text + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    df = pd.read_csv(SUMMARY_CSV)
    df_delta = add_delta_columns(df)

    main_csv, main_tex = write_main_table(df_delta)
    winners_csv, winners_tex = write_regime_winner_table(df_delta)
    plot_path = make_strongest_delta_plot(df_delta)
    caption_path = write_caption_block(df_delta)

    print(f"Saved condensed table CSV to: {main_csv}")
    print(f"Saved condensed table TeX to: {main_tex}")
    print(f"Saved regime winner CSV to: {winners_csv}")
    print(f"Saved regime winner TeX to: {winners_tex}")
    print(f"Saved strongest-delta figure to: {plot_path}")
    print(f"Saved caption block to: {caption_path}")


if __name__ == "__main__":
    main()
