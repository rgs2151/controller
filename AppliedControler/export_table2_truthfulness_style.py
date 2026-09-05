"""Export a paper-style truthfulness table (LaTeX + PDF)."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument(
        "--title",
        default="Table 2. Summary of results for truthfulness evaluations.",
    )
    return parser.parse_args()


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


def _fmt_latex(mean: float, std: float) -> str:
    if math.isnan(mean) or math.isnan(std):
        return "-- \\pm --"
    return f"{mean:.2f} \\pm {std:.2f}"


def _fmt_pdf(mean: float, std: float) -> str:
    if math.isnan(mean) or math.isnan(std):
        return "-- +/- --"
    return f"{mean:.2f} +/- {std:.2f}"


def _validate(df: pd.DataFrame) -> None:
    required = {
        "model",
        "method",
        "ti_mean",
        "ti_std",
        "true_mean",
        "true_std",
        "info_mean",
        "info_std",
        "mmlu_mean",
        "mmlu_std",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")


def write_latex(df: pd.DataFrame, path: Path, title: str) -> None:
    lines: list[str] = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.6in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage[table]{xcolor}",
        r"\usepackage{multirow}",
        r"\usepackage{graphicx}",
        r"\usepackage{newtxtext,newtxmath}",
        r"\begin{document}",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{\textit{{{title}}}}}",
        r"\renewcommand{\arraystretch}{1.22}",
        r"\setlength{\tabcolsep}{6.5pt}",
        r"\begin{tabular}{llccc|c}",
        r"\toprule",
        r"Model & Method & \cellcolor{blue!16} T\!\cdot\!I (\%) (\uparrow) & True (\%) (\uparrow) & Info (\%) (\uparrow) & MMLU (\%) (\uparrow) \\",
        r"\midrule",
    ]

    for model, block in df.groupby("model", sort=False):
        nrows = len(block)
        best_ti = block[block["method"] != "Original"]["ti_mean"].max()
        for i, (_, row) in enumerate(block.iterrows()):
            model_tex = ""
            if i == 0:
                model_tex = (
                    rf"\multirow{{{nrows}}}{{*}}{{\rotatebox[origin=c]{{90}}{{{_latex_escape(str(model))}}}}}"
                )

            ti = _fmt_latex(float(row["ti_mean"]), float(row["ti_std"]))
            if row["method"] != "Original" and abs(float(row["ti_mean"]) - float(best_ti)) < 1e-12:
                ti = rf"\textbf{{{ti}}}"

            method = _latex_escape(str(row["method"]))
            true_txt = _fmt_latex(float(row["true_mean"]), float(row["true_std"]))
            info_txt = _fmt_latex(float(row["info_mean"]), float(row["info_std"]))
            mmlu_txt = _fmt_latex(float(row["mmlu_mean"]), float(row["mmlu_std"]))
            lines.append(
                f"{model_tex} & {method} & \\cellcolor{{blue!16}} {ti} & {true_txt} & {info_txt} & {mmlu_txt} \\\\"
            )

            if i == 0 and nrows > 1:
                lines.append(r"\midrule")

        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines.extend([
        r"\end{tabular}",
        r"\end{table}",
        r"\end{document}",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pdf(df: pd.DataFrame, path: Path, title: str) -> None:
    rows: list[list[str]] = []
    for model, block in df.groupby("model", sort=False):
        best_ti = block[block["method"] != "Original"]["ti_mean"].max()
        for i, (_, row) in enumerate(block.iterrows()):
            model_txt = str(model) if i == 0 else ""
            ti = _fmt_pdf(float(row["ti_mean"]), float(row["ti_std"]))
            if row["method"] != "Original" and abs(float(row["ti_mean"]) - float(best_ti)) < 1e-12:
                ti = f"**{ti}**"
            rows.append(
                [
                    model_txt,
                    str(row["method"]),
                    ti,
                    _fmt_pdf(float(row["true_mean"]), float(row["true_std"])),
                    _fmt_pdf(float(row["info_mean"]), float(row["info_std"])),
                    _fmt_pdf(float(row["mmlu_mean"]), float(row["mmlu_std"])),
                ]
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        plt.rcParams["font.family"] = "serif"
        fig, ax = plt.subplots(figsize=(8.7, 12.2))
        ax.axis("off")
        ax.set_title(title, fontsize=21, fontstyle="italic", pad=14)

        table = ax.table(
            cellText=rows,
            colLabels=["Model", "Method", "T.I (%) (up)", "True (%) (up)", "Info (%) (up)", "MMLU (%) (up)"],
            loc="upper center",
            cellLoc="center",
            colLoc="center",
            bbox=[0.01, 0.01, 0.98, 0.95],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12.0)

        for c in range(6):
            cell = table[(0, c)]
            cell.set_facecolor("#f7f7f7")
            cell.set_linewidth(0.9)
            cell.get_text().set_weight("bold")

        nrows = len(rows)
        for r in range(1, nrows + 1):
            for c in range(6):
                cell = table[(r, c)]
                cell.set_linewidth(0.45)
                cell.set_edgecolor("#333333")
                cell.set_facecolor("#dbe8f5" if c == 2 else "white")

                txt = cell.get_text().get_text()
                if txt.startswith("**") and txt.endswith("**"):
                    cell.get_text().set_text(txt[2:-2])
                    cell.get_text().set_weight("bold")

                if c == 0 and txt:
                    cell.get_text().set_rotation(90)
                    cell.get_text().set_va("center")
                    cell.get_text().set_ha("center")

        table.scale(1.0, 1.36)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    _validate(df)
    write_latex(df, args.output_tex, args.title)
    write_pdf(df, args.output_pdf, args.title)
    print(f"Wrote LaTeX: {args.output_tex}")
    print(f"Wrote PDF: {args.output_pdf}")


if __name__ == "__main__":
    main()