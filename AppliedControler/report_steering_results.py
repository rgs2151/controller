"""Generate final steering tables and meaningful plots from run_steering outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


METHOD_LABELS = {
    "alqr": "A-LQR",
    "spid": "S-PID",
    "new_method": "H-infinity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", default="summary", help="Suffix label for output files")
    return parser.parse_args()


def _safe_float(value: object) -> float:
    return float(value)


def load_rows(path: Path) -> list[dict[str, object]]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError("Input JSON must be a list of rows")
    cleaned: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        method = str(row.get("method", ""))
        if method not in {"alqr", "spid", "new_method"}:
            continue
        cleaned.append(row)
    if not cleaned:
        raise ValueError("No rows found for methods alqr/spid/new_method")
    return cleaned


def split_rows_by_status(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    counts: dict[str, int] = defaultdict(int)
    ok_rows: list[dict[str, object]] = []
    for row in rows:
        status = str(row.get("run_status", "ok"))
        counts[status] += 1
        if status == "ok":
            ok_rows.append(row)
    return ok_rows, counts


def select_paired_ok_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    methods = {"alqr", "spid", "new_method"}
    by_target: dict[tuple[str, str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["label"]), str(row["subset"]), str(row["model_name"]))
        by_target[key][str(row["method"])] = row

    valid_targets = {
        key
        for key, grouped in by_target.items()
        if all(
            method in grouped and str(grouped[method].get("run_status", "ok")) == "ok"
            for method in methods
        )
    }
    return [
        row
        for row in rows
        if (str(row["label"]), str(row["subset"]), str(row["model_name"])) in valid_targets
        and str(row["method"]) in methods
        and str(row.get("run_status", "ok")) == "ok"
    ]


def build_method_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    grouped_pct_change: dict[str, list[float]] = defaultdict(list)
    baseline: dict[str, list[float]] = defaultdict(list)
    steered: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        method = str(row["method"])
        delta = _safe_float(row["toxicity_delta"])
        base = _safe_float(row["baseline_toxicity_mean"])
        steer = _safe_float(row["steered_toxicity_mean"])
        grouped[method].append(delta)
        baseline[method].append(base)
        steered[method].append(steer)
        if abs(base) > 1e-12:
            grouped_pct_change[method].append((delta / base) * 100.0)

    summary: list[dict[str, object]] = []
    for method in sorted(grouped.keys()):
        deltas = grouped[method]
        n = len(deltas)
        mean_delta = statistics.fmean(deltas)
        std_delta = statistics.stdev(deltas) if n > 1 else 0.0
        stderr = std_delta / math.sqrt(n) if n > 1 else 0.0
        ci95 = 1.96 * stderr
        pct_changes = grouped_pct_change[method]
        mean_pct_change = statistics.fmean(pct_changes) if pct_changes else 0.0
        std_pct_change = statistics.stdev(pct_changes) if len(pct_changes) > 1 else 0.0
        stderr_pct_change = std_pct_change / math.sqrt(len(pct_changes)) if len(pct_changes) > 1 else 0.0
        ci95_pct_change = 1.96 * stderr_pct_change
        summary.append(
            {
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "n": n,
                "baseline_mean": statistics.fmean(baseline[method]),
                "steered_mean": statistics.fmean(steered[method]),
                "mean_delta": mean_delta,
                "median_delta": statistics.median(deltas),
                "std_delta": std_delta,
                "min_delta": min(deltas),
                "max_delta": max(deltas),
                "improvement_rate": sum(1 for value in deltas if value < 0.0) / n,
                "ci95": ci95,
                "mean_pct_change": mean_pct_change,
                "std_pct_change": std_pct_change,
                "ci95_pct_change": ci95_pct_change,
            }
        )
    summary.sort(key=lambda item: item["mean_delta"])
    return summary


def build_target_winners(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_target: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["label"]), str(row["subset"]), str(row["model_name"]))
        by_target[key].append(row)

    winners: list[dict[str, object]] = []
    for (label, subset, model_name), target_rows in sorted(by_target.items()):
        ranked = sorted(target_rows, key=lambda item: _safe_float(item["toxicity_delta"]))
        best = ranked[0]
        winners.append(
            {
                "label": label,
                "model_name": model_name,
                "subset": subset,
                "winner_method": str(best["method"]),
                "winner_method_label": METHOD_LABELS.get(str(best["method"]), str(best["method"])),
                "winner_delta": _safe_float(best["toxicity_delta"]),
            }
        )
    return winners


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def plot_method_means(summary: list[dict[str, object]], path: Path) -> None:
    labels = [str(row["method_label"]) for row in summary]
    means = [float(row["mean_delta"]) for row in summary]
    errors = [float(row["ci95"]) for row in summary]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    colors = ["#1f77b4" if value <= 0 else "#d62728" for value in means]
    ax.bar(labels, means, yerr=errors, capsize=5, color=colors, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Mean toxicity delta (steered - baseline)")
    ax.set_title("Method Effect Size with 95% CI")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_method_percent_change(summary: list[dict[str, object]], path: Path) -> None:
    labels = [f"{row['method_label']}\n(n={row['n']})" for row in summary]
    means = [float(row["mean_pct_change"]) for row in summary]
    errors = [float(row["ci95_pct_change"]) for row in summary]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    colors = ["#1f77b4" if value <= 0 else "#d62728" for value in means]
    ax.bar(labels, means, yerr=errors, capsize=5, color=colors, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Mean toxicity change (%)")
    ax.set_title("Relative Toxicity Change vs Baseline (95% CI)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_actual_toxicity_percent(summary: list[dict[str, object]], path: Path) -> None:
    labels = [f"{row['method_label']}\n(n={row['n']})" for row in summary]
    baseline = [100.0 * float(row["baseline_mean"]) for row in summary]
    steered = [100.0 * float(row["steered_mean"]) for row in summary]

    x = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar([v - width / 2 for v in x], baseline, width=width, label="Baseline", color="#9e9e9e")
    ax.bar([v + width / 2 for v in x], steered, width=width, label="Steered", color="#4e79a7")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean toxicity (%)")
    ax.set_title("Actual Toxicity Level: Baseline vs Steered")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_mean_toxicity(summary: list[dict[str, object]], path: Path) -> None:
    labels = [f"{row['method_label']}\n(n={row['n']})" for row in summary]
    values = [100.0 * float(row["steered_mean"]) for row in summary]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]
    ax.bar(labels, values, color=colors[: len(labels)], alpha=0.9)
    ax.set_ylabel("Mean steered toxicity (%)")
    ax.set_title("Mean Toxicity by Method")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_method_distributions(rows: list[dict[str, object]], path: Path) -> None:
    methods = ["alqr", "spid", "new_method"]
    labels = [METHOD_LABELS[m] for m in methods]
    data = [
        [_safe_float(row["toxicity_delta"]) for row in rows if str(row["method"]) == method]
        for method in methods
    ]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    box = ax.boxplot(data, labels=labels, patch_artist=True)
    palette = ["#4e79a7", "#f28e2b", "#59a14f"]
    for patch, color in zip(box["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Toxicity delta")
    ax.set_title("Distribution of Target-Level Deltas")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_target_heatmap(rows: list[dict[str, object]], path: Path) -> None:
    methods = ["alqr", "spid", "new_method"]
    target_keys = sorted({(str(row["label"]), str(row["subset"])) for row in rows})
    target_labels = [f"{label}\n{subset}" for (label, subset) in target_keys]

    matrix: list[list[float]] = []
    for method in methods:
        by_target = {
            (str(row["label"]), str(row["subset"])): _safe_float(row["toxicity_delta"])
            for row in rows
            if str(row["method"]) == method
        }
        matrix.append([by_target.get(key, float("nan")) for key in target_keys])

    fig, ax = plt.subplots(figsize=(max(10, len(target_keys) * 0.7), 3.8))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
    ax.set_yticks(range(len(methods)), [METHOD_LABELS[m] for m in methods])
    ax.set_xticks(range(len(target_labels)), target_labels, rotation=55, ha="right")
    ax.set_title("Per-Target Toxicity Delta Heatmap")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Delta (steered - baseline)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_markdown_summary(
    path: Path,
    summary: list[dict[str, object]],
    winners: list[dict[str, object]],
    status_counts: dict[str, int],
) -> None:
    lines = [
        "# Steering Results Summary",
        "",
        "## Method Ranking (lower mean delta is better)",
        "",
        "| Rank | Method | N | Mean Delta | 95% CI | Improvement Rate |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(summary, start=1):
        lines.append(
            "| "
            f"{idx} | {row['method_label']} | {row['n']} | {row['mean_delta']:.6f} | "
            f"{row['ci95']:.6f} | {row['improvement_rate']:.1%} |"
        )

    winner_counts: dict[str, int] = defaultdict(int)
    for row in winners:
        winner_counts[str(row["winner_method"])] += 1

    lines.extend([
        "",
        "## Target Winner Counts",
        "",
        "| Method | Wins |",
        "|---|---:|",
    ])
    for method in ["alqr", "spid", "new_method"]:
        lines.append(f"| {METHOD_LABELS[method]} | {winner_counts.get(method, 0)} |")

    lines.extend([
        "",
        "## Row Status Counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ])
    for status in sorted(status_counts.keys()):
        lines.append(f"| {status} | {status_counts[status]} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_paper_style_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_target_method: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for row in rows:
        key = (
            str(row["label"]),
            str(row["model_name"]),
            str(row["subset"]),
            str(row["method"]),
        )
        by_target_method[key] = row

    target_keys = sorted({(k[0], k[1], k[2]) for k in by_target_method.keys()})
    output: list[dict[str, object]] = []
    for label, model_name, subset in target_keys:
        method_rows = {
            method: by_target_method.get((label, model_name, subset, method))
            for method in ["alqr", "spid", "new_method"]
        }
        baseline_values = [
            _safe_float(row["baseline_toxicity_mean"]) for row in method_rows.values() if row is not None
        ]
        baseline_mean = statistics.fmean(baseline_values) if baseline_values else float("nan")
        output.append(
            {
                "label": label,
                "model_name": model_name,
                "subset": subset,
                "method": "Original",
                "toxicity_percent": 100.0 * baseline_mean,
                "percent_change": 0.0,
                "status": "ok",
            }
        )
        for method in ["alqr", "spid", "new_method"]:
            row = method_rows.get(method)
            if row is None:
                output.append(
                    {
                        "label": label,
                        "model_name": model_name,
                        "subset": subset,
                        "method": METHOD_LABELS[method],
                        "toxicity_percent": float("nan"),
                        "percent_change": float("nan"),
                        "status": "missing",
                    }
                )
                continue
            status = str(row.get("run_status", "ok"))
            steered = 100.0 * _safe_float(row["steered_toxicity_mean"])
            baseline = _safe_float(row["baseline_toxicity_mean"])
            delta = _safe_float(row["toxicity_delta"])
            pct_change = (delta / baseline) * 100.0 if abs(baseline) > 1e-12 else float("nan")
            output.append(
                {
                    "label": label,
                    "model_name": model_name,
                    "subset": subset,
                    "method": METHOD_LABELS[method],
                    "toxicity_percent": steered,
                    "percent_change": pct_change,
                    "status": status,
                }
            )
    return output


def write_paper_style_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Paper-Style Toxicity Table",
        "",
        "Lower toxicity is better. Percent change is relative to the Original baseline.",
    ]
    group_keys = sorted({(str(r["label"]), str(r["subset"])) for r in rows})
    for label, subset in group_keys:
        lines.extend([
            "",
            f"## {label} - {subset}",
            "",
            "| Method | Toxicity (%) | Change vs Original (%) | Status |",
            "|---|---:|---:|---|",
        ])
        group_rows = [r for r in rows if str(r["label"]) == label and str(r["subset"]) == subset]
        method_order = ["Original", "A-LQR", "S-PID", "H-infinity"]
        by_method = {str(r["method"]): r for r in group_rows}
        for method in method_order:
            row = by_method.get(method)
            if row is None:
                continue
            tox = row["toxicity_percent"]
            pct = row["percent_change"]
            tox_text = f"{tox:.2f}" if isinstance(tox, float) and not math.isnan(tox) else "N/A"
            pct_text = f"{pct:+.2f}" if isinstance(pct, float) and not math.isnan(pct) else "N/A"
            lines.append(f"| {method} | {tox_text} | {pct_text} | {row['status']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_escape(value: str) -> str:
    escaped = value
    replacements = {
        "\\": r"\\textbackslash{}",
        "&": r"\\&",
        "%": r"\\%",
        "$": r"\\$",
        "#": r"\\#",
        "_": r"\\_",
        "{": r"\\{",
        "}": r"\\}",
        "~": r"\\textasciitilde{}",
        "^": r"\\textasciicircum{}",
    }
    for key, rep in replacements.items():
        escaped = escaped.replace(key, rep)
    return escaped


def write_paper_style_latex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\\documentclass[10pt]{article}",
        r"\\usepackage[margin=0.7in]{geometry}",
        r"\\usepackage{booktabs}",
        r"\\usepackage{longtable}",
        r"\\usepackage{array}",
        r"\\begin{document}",
        r"\\small",
        r"\\begin{center}",
        r"\\textbf{Paper-Style Toxicity Table}",
        r"\\end{center}",
        r"\\vspace{4pt}",
        r"\\noindent Lower toxicity is better. Percent change is relative to the Original baseline.",
        r"\\vspace{6pt}",
        r"\\begin{longtable}{p{3.2cm}p{3.2cm}p{2.6cm}p{2.2cm}r r l}",
        r"\\toprule",
        r"Model & Subset & Method & Status & Toxicity (\\%) & Change (\\%) & Notes \\",
        r"\\midrule",
        r"\\endfirsthead",
        r"\\toprule",
        r"Model & Subset & Method & Status & Toxicity (\\%) & Change (\\%) & Notes \\",
        r"\\midrule",
        r"\\endhead",
        r"\\bottomrule",
        r"\\endfoot",
    ]

    method_order = ["Original", "A-LQR", "S-PID", "H-infinity"]
    groups = sorted({(str(r["label"]), str(r["subset"])) for r in rows})
    for label, subset in groups:
        group_rows = [r for r in rows if str(r["label"]) == label and str(r["subset"]) == subset]
        by_method = {str(r["method"]): r for r in group_rows}
        first_line = True
        for method in method_order:
            row = by_method.get(method)
            if row is None:
                continue
            model_col = _latex_escape(label) if first_line else ""
            subset_col = _latex_escape(subset) if first_line else ""
            method_col = _latex_escape(method)
            status_col = _latex_escape(str(row["status"]))
            tox = row["toxicity_percent"]
            pct = row["percent_change"]
            tox_text = f"{tox:.2f}" if isinstance(tox, float) and not math.isnan(tox) else "N/A"
            pct_text = f"{pct:+.2f}" if isinstance(pct, float) and not math.isnan(pct) else "N/A"
            note = ""
            if str(row["status"]) == "infeasible":
                note = "H-infinity infeasible"
            lines.append(
                f"{model_col} & {subset_col} & {method_col} & {status_col} & {tox_text} & {pct_text} & {_latex_escape(note)} \\\\"
            )
            first_line = False
        lines.append(r"\\midrule")

    lines.extend([
        r"\\end{longtable}",
        r"\\end{document}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paper_style_pdf(path: Path, rows: list[dict[str, object]]) -> None:
    groups = sorted({(str(r["label"]), str(r["subset"])) for r in rows})
    method_order = ["Original", "A-LQR", "S-PID", "H-infinity"]

    with PdfPages(path) as pdf:
        # Cover page.
        fig = plt.figure(figsize=(11.0, 8.5))
        fig.suptitle("Paper-Style Toxicity Table", fontsize=16, fontweight="bold", y=0.95)
        fig.text(
            0.06,
            0.88,
            "Lower toxicity is better. Percent change is relative to the Original baseline.",
            fontsize=11,
        )
        fig.text(0.06, 0.83, "Methods included: Original, A-LQR, S-PID, H-infinity", fontsize=11)
        fig.text(0.06, 0.78, "Status marks infeasible rows explicitly when present.", fontsize=11)
        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # One table page per target group for readability.
        for label, subset in groups:
            group_rows = [r for r in rows if str(r["label"]) == label and str(r["subset"]) == subset]
            by_method = {str(r["method"]): r for r in group_rows}
            table_rows = []
            for method in method_order:
                row = by_method.get(method)
                if row is None:
                    continue
                tox = row["toxicity_percent"]
                pct = row["percent_change"]
                tox_text = f"{tox:.2f}" if isinstance(tox, float) and not math.isnan(tox) else "N/A"
                pct_text = f"{pct:+.2f}" if isinstance(pct, float) and not math.isnan(pct) else "N/A"
                table_rows.append([
                    method,
                    tox_text,
                    pct_text,
                    str(row["status"]),
                ])

            fig, ax = plt.subplots(figsize=(11.0, 8.5))
            ax.axis("off")
            ax.set_title(f"{label} - {subset}", fontsize=14, pad=16)
            table = ax.table(
                cellText=table_rows,
                colLabels=["Method", "Toxicity (%)", "Change vs Original (%)", "Status"],
                loc="center",
                cellLoc="center",
                colLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.2, 1.6)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.input_json)
    ok_rows, status_counts = split_rows_by_status(rows)
    if not ok_rows:
        raise ValueError("No rows with run_status=ok available for summary/plots")
    summary = build_method_summary(ok_rows)
    winners = build_target_winners(ok_rows)

    summary_csv = output_dir / f"final_table_method_summary_{args.tag}.csv"
    paired_summary_csv = output_dir / f"final_table_method_summary_paired_{args.tag}.csv"
    winners_csv = output_dir / f"final_table_target_winners_{args.tag}.csv"
    summary_md = output_dir / f"final_report_{args.tag}.md"
    paper_rows_csv = output_dir / f"paper_style_table_{args.tag}.csv"
    paper_rows_md = output_dir / f"paper_style_table_{args.tag}.md"
    paper_rows_tex = output_dir / f"paper_style_table_{args.tag}.tex"
    paper_rows_pdf = output_dir / f"paper_style_table_{args.tag}.pdf"

    write_csv(
        summary_csv,
        summary,
        [
            "method",
            "method_label",
            "n",
            "baseline_mean",
            "steered_mean",
            "mean_delta",
            "median_delta",
            "std_delta",
            "min_delta",
            "max_delta",
            "improvement_rate",
            "ci95",
            "mean_pct_change",
            "std_pct_change",
            "ci95_pct_change",
        ],
    )

    paired_ok_rows = select_paired_ok_rows(rows)
    paired_summary = build_method_summary(paired_ok_rows) if paired_ok_rows else []
    if paired_summary:
        write_csv(
            paired_summary_csv,
            paired_summary,
            [
                "method",
                "method_label",
                "n",
                "baseline_mean",
                "steered_mean",
                "mean_delta",
                "median_delta",
                "std_delta",
                "min_delta",
                "max_delta",
                "improvement_rate",
                "ci95",
                "mean_pct_change",
                "std_pct_change",
                "ci95_pct_change",
            ],
        )
    write_csv(
        winners_csv,
        winners,
        [
            "label",
            "model_name",
            "subset",
            "winner_method",
            "winner_method_label",
            "winner_delta",
        ],
    )
    write_markdown_summary(summary_md, summary, winners, status_counts)

    paper_rows = build_paper_style_rows(rows)
    write_csv(
        paper_rows_csv,
        paper_rows,
        [
            "label",
            "model_name",
            "subset",
            "method",
            "toxicity_percent",
            "percent_change",
            "status",
        ],
    )
    write_paper_style_markdown(paper_rows_md, paper_rows)
    write_paper_style_latex(paper_rows_tex, paper_rows)
    write_paper_style_pdf(paper_rows_pdf, paper_rows)

    plot_method_means(summary, output_dir / f"plot_method_mean_delta_{args.tag}.png")
    plot_method_percent_change(summary, output_dir / f"plot_method_percent_change_{args.tag}.png")
    plot_actual_toxicity_percent(summary, output_dir / f"plot_actual_toxicity_percent_{args.tag}.png")
    plot_mean_toxicity(summary, output_dir / f"plot_mean_toxicity_{args.tag}.png")
    plot_method_distributions(ok_rows, output_dir / f"plot_method_distribution_{args.tag}.png")
    plot_target_heatmap(ok_rows, output_dir / f"plot_target_heatmap_{args.tag}.png")
    if paired_summary:
        plot_method_means(
            paired_summary,
            output_dir / f"plot_method_mean_delta_paired_{args.tag}.png",
        )
        plot_method_percent_change(
            paired_summary,
            output_dir / f"plot_method_percent_change_paired_{args.tag}.png",
        )
        plot_actual_toxicity_percent(
            paired_summary,
            output_dir / f"plot_actual_toxicity_percent_paired_{args.tag}.png",
        )
        plot_mean_toxicity(
            paired_summary,
            output_dir / f"plot_mean_toxicity_paired_{args.tag}.png",
        )
        plot_method_distributions(
            paired_ok_rows,
            output_dir / f"plot_method_distribution_paired_{args.tag}.png",
        )
        plot_target_heatmap(
            paired_ok_rows,
            output_dir / f"plot_target_heatmap_paired_{args.tag}.png",
        )

    print(f"Wrote tables and plots to {output_dir}")


if __name__ == "__main__":
    main()
