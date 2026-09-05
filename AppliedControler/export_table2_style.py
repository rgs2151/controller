"""Export a Table-2-style summary table (LaTeX + PDF) from paper-style toxicity CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


METHOD_ORDER = [
    "Original",
    "ITI",
    "ActAdd",
    "Mean-AcT",
    "Linear-AcT",
    "PID-AcT",
    "ODESteer",
    "S-PID",
    "A-LQR",
    "H-infinity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--title", default="Table 2. Summary of toxicity evaluations.")
    parser.add_argument(
        "--uncertainty-placeholder",
        type=float,
        default=0.00,
        help="Placeholder uncertainty to display as ± value when true uncertainty is unavailable.",
    )
    return parser.parse_args()


def _ordered_group(group: pd.DataFrame) -> pd.DataFrame:
    rank = {name: idx for idx, name in enumerate(METHOD_ORDER)}
    return (
        group.assign(_order=group["method"].map(lambda x: rank.get(str(x), 999)))
        .sort_values("_order")
        .drop(columns=["_order"])
    )


def _fmt_tox_with_pm(value: float | None, pm: float) -> str:
    if value is None:
        return "-- ± --"
    return f"{value:.2f} ± {pm:.2f}"


def _build_block_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = list(df.groupby(["label", "subset"], sort=True))
    for block_index, ((label, subset), group) in enumerate(grouped):
        group = _ordered_group(group)
        row_map = {str(r["method"]): r for _, r in group.iterrows()}
        steer_values = [
            float(r["toxicity_percent"])
            for m, r in row_map.items()
            if m != "Original" and str(r.get("status", "ok")) == "ok"
        ]
        best_value = min(steer_values) if steer_values else None

        methods = [m for m in METHOD_ORDER if (m in row_map or m in {"ITI", "ActAdd", "Mean-AcT", "Linear-AcT", "PID-AcT", "ODESteer"})]
        # Remove duplicate if METHOD_ORDER already contains all desired values.
        seen: set[str] = set()
        methods = [m for m in methods if not (m in seen or seen.add(m))]

        block_model = f"{label}-{subset}"
        block_start = len(rows)
        for method in methods:
            source = row_map.get(method)
            tox = float(source["toxicity_percent"]) if source is not None else None
            status = str(source["status"]) if source is not None else "missing"
            rows.append(
                {
                    "block_index": block_index,
                    "block_model": block_model,
                    "method": method,
                    "toxicity_percent": tox,
                    "status": status,
                    "is_best": tox is not None and best_value is not None and abs(tox - best_value) < 1e-12,
                }
            )

        # Mark one row to carry vertical model label in PDF.
        block_end = len(rows) - 1
        mid = (block_start + block_end) // 2
        rows[mid]["show_model_label"] = True
    return rows


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\\textbackslash{}")
        .replace("&", r"\\&")
        .replace("%", r"\\%")
        .replace("_", r"\\_")
        .replace("#", r"\\#")
        .replace("{", r"\\{")
        .replace("}", r"\\}")
    )


def write_latex(df: pd.DataFrame, path: Path, title: str, pm: float) -> None:
    block_rows = _build_block_rows(df)
    block_sizes: dict[str, int] = {}
    for row in block_rows:
        block_sizes[row["block_model"]] = block_sizes.get(row["block_model"], 0) + 1

    lines = [
        r"\\documentclass[10pt]{article}",
        r"\\usepackage[margin=0.7in]{geometry}",
        r"\\usepackage{booktabs}",
        r"\\usepackage[table]{xcolor}",
        r"\\usepackage{multirow}",
        r"\\usepackage{graphicx}",
        r"\\usepackage{newtxtext,newtxmath}",
        r"\\begin{document}",
        r"\\begin{table}[t]",
        r"\\centering",
        rf"\\caption{{\\textit{{{title}}}}}",
        r"\\renewcommand{\\arraystretch}{1.24}",
        r"\\setlength{\\tabcolsep}{8pt}",
        r"\\begin{tabular}{llcc}",
        r"\\toprule",
        r"Model & Method & \\cellcolor{blue!12} Tox (\\%) (\\downarrow) & Status \\",
        r"\\midrule",
    ]

    for idx, row in enumerate(block_rows):
        model_tex = ""
        is_block_start = idx == 0 or block_rows[idx - 1]["block_model"] != row["block_model"]
        if is_block_start:
            nrows = block_sizes[row["block_model"]]
            model_tex = (
                rf"\\multirow{{{nrows}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{_latex_escape(row['block_model'])}}}}}"
            )

        tox_text = _fmt_tox_with_pm(row["toxicity_percent"], pm)
        if row["is_best"]:
            tox_text = rf"\\textbf{{{tox_text}}}"

        lines.append(
            f"{model_tex} & {_latex_escape(row['method'])} & \\cellcolor{{blue!12}} {tox_text} & {_latex_escape(row['status'])} \\\\"
        )

        next_block = None
        if idx < len(block_rows) - 1:
            next_block = block_rows[idx + 1]["block_model"]
        if idx < len(block_rows) - 1 and next_block != row["block_model"]:
            lines.append(r"\\midrule")

    lines.extend([
        r"\\bottomrule",
        r"\\end{tabular}",
        r"\\end{table}",
        r"\\end{document}",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pdf(df: pd.DataFrame, path: Path, title: str, pm: float) -> None:
    block_rows = _build_block_rows(df)
    rows: list[list[str]] = []
    for row in block_rows:
        model_col = row["block_model"] if row.get("show_model_label") else ""
        method = row["method"]
        tox_text = _fmt_tox_with_pm(row["toxicity_percent"], pm)
        if row["is_best"]:
            tox_text = f"**{tox_text}**"
        rows.append([model_col, method, tox_text, row["status"]])

    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        plt.rcParams["font.family"] = "serif"
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.set_title(title, fontsize=21, fontstyle="italic", pad=18)

        table = ax.table(
            cellText=rows,
            colLabels=["Model", "Method", "Tox (%) (down)", "Status"],
            loc="upper center",
            cellLoc="center",
            colLoc="center",
            bbox=[0.02, 0.02, 0.96, 0.93],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(13)

        # Header styling.
        for c in range(4):
            cell = table[(0, c)]
            cell.set_facecolor("#f7f7f7")
            cell.set_linewidth(0.9)
            cell.get_text().set_weight("bold")

        # Body styling, highlight toxicity column and rotate model labels.
        nrows = len(rows)
        for r in range(1, nrows + 1):
            for c in range(4):
                cell = table[(r, c)]
                cell.set_linewidth(0.45)
                if c == 2:
                    cell.set_facecolor("#dbe8f5")
                else:
                    cell.set_facecolor("white")

                txt = cell.get_text().get_text()
                if txt.startswith("**") and txt.endswith("**"):
                    cell.get_text().set_text(txt[2:-2])
                    cell.get_text().set_weight("bold")
                if c == 0 and txt:
                    cell.get_text().set_rotation(90)
                    cell.get_text().set_va("center")
                    cell.get_text().set_ha("center")

        # Increase row height for paper-like spacing.
        table.scale(1.0, 1.46)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    required = {"label", "subset", "method", "toxicity_percent", "status"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    write_latex(df, args.output_tex, args.title, args.uncertainty_placeholder)
    write_pdf(df, args.output_pdf, args.title, args.uncertainty_placeholder)
    print(f"Wrote LaTeX: {args.output_tex}")
    print(f"Wrote PDF: {args.output_pdf}")


if __name__ == "__main__":
    main()
