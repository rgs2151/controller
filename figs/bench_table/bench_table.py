"""Render the paper benchmark tables to synchronized Markdown and TeX."""

from __future__ import annotations

import json
from pathlib import Path


UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[1]
RESULTS = REPO / "parking/bench_evaluations/cache/results"
PLOTS = UNIT / "plots"

MODELS = (
    ("gemma2b", "Gemma-2-2B"),
    ("llama8b", "Llama-3-8B"),
    ("qwen14b", "Qwen-2.5-14B"),
)
METHODS = (
    ("original", "Original", "Original"),
    ("iti", "ITI", "ITI"),
    ("actadd", "ActAdd", "ActAdd"),
    ("mean_act", "Mean-AcT", "Mean-AcT"),
    ("linear_act", "Linear-AcT", "Linear-AcT"),
    ("pid_act", "PID-AcT", "PID-AcT"),
    ("odesteer", "ODESteer", "ODESteer"),
    ("spid", "S-PID", "S-PID"),
    ("alqr", "A-LQR", "A-LQR"),
    ("h_infinity", "H∞ (ours)", "$\\mathbf{H_\\infty}$ (ours)"),
)


def _load_results(behavior: str) -> dict[tuple[str, str], dict]:
    results = {}
    for model_key, _model_label in MODELS:
        for method, _markdown_label, _tex_label in METHODS:
            path = RESULTS / behavior / model_key / f"{method}.json"
            if path.exists():
                results[(model_key, method)] = json.loads(path.read_text())
    return results


def _metric(result: dict | None, name: str, *, required: bool = False):
    if result is None:
        return None
    value = result.get("metrics", {}).get(name)
    if value is None:
        if required:
            raise ValueError(f"Completed result is missing metric {name}")
        return None
    return float(value["mean"]), float(value["standard_error"])


def _rows(behavior: str, results: dict[tuple[str, str], dict]) -> list[dict]:
    rows = []
    for model_key, model_label in MODELS:
        for method, markdown_label, tex_label in METHODS:
            result = results.get((model_key, method))
            if behavior == "toxicity":
                values = [
                    _metric(result, "toxicity", required=True),
                    _metric(result, "dist_2", required=True),
                    _metric(result, "mmlu", required=True),
                    _metric(result, "perplexity", required=True),
                ]
            else:
                values = [
                    _metric(result, "truth_x_info", required=True),
                    _metric(result, "spanish"),
                    _metric(result, "adversarial"),
                    _metric(result, "long"),
                    _metric(result, "truth", required=True),
                    _metric(result, "info", required=True),
                    _metric(result, "mmlu"),
                ]
            rows.append(
                {
                    "model": model_label,
                    "method_markdown": markdown_label,
                    "method_tex": tex_label,
                    "values": values,
                }
            )
    return rows


def _markdown_cell(value) -> str:
    if value is None:
        return "TBD"
    return f"{value[0]:.2f} ± {value[1]:.2f}"


def _tex_cell(value) -> str:
    if value is None:
        return r"\textbf{TBD}"
    return f"${value[0]:.2f}\\,\\pm\\,{value[1]:.2f}$"


def _markdown_table(title: str, headers: tuple[str, ...], rows: list[dict]) -> list[str]:
    lines = [f"## {title}", "", "| Model | Method | " + " | ".join(headers) + " |"]
    lines.append("|---|---|" + "---:|" * len(headers))
    for row in rows:
        values = " | ".join(_markdown_cell(value) for value in row["values"])
        lines.append(f"| {row['model']} | {row['method_markdown']} | {values} |")
    return lines


def render_markdown(toxicity_rows: list[dict], truthfulness_rows: list[dict]) -> str:
    lines = [
        "# Benchmark tables",
        "",
        "Generated from completed summaries in `parking/bench_evaluations/cache/results/`. "
        "The TeX table is generated from the same rows.",
        "",
    ]
    lines.extend(
        _markdown_table(
            "Toxicity benchmark",
            ("CLS Tox. (%) ↓", "Dist 2 ↑", "MMLU (%) ↑", "PPL ↓"),
            toxicity_rows,
        )
    )
    lines.extend(
        [
            "",
            "Toxicity, Dist-2, and PPL are mean ± SE across five complete 1,000-prompt RTP repetitions. "
            "MMLU is accuracy ± prompt-level SE on one shared 1,000-question 5-shot set.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            "Truthfulness benchmark",
            (
                "TruthfulQA–ID T×I ↑",
                "Spanish",
                "Adversarial",
                "Long",
                "True (%) ↑",
                "Info (%) ↑",
                "MMLU (%) ↑",
            ),
            truthfulness_rows,
        )
    )
    lines.extend(
        [
            "",
            "Truthfulness values are mean ± SE across five complete 817-question repetitions. "
            "TBD cells have not been run.",
            "",
        ]
    )
    return "\n".join(lines)


def _tex_rows(rows: list[dict]) -> list[str]:
    lines = []
    methods_per_model = len(METHODS)
    for index, row in enumerate(rows):
        model = (
            f"\\multirow{{{methods_per_model}}}{{*}}{{{row['model']}}}"
            if index % methods_per_model == 0
            else ""
        )
        values = " & ".join(_tex_cell(value) for value in row["values"])
        lines.append(f"{model} & {row['method_tex']} & {values} \\\\")
        if index % methods_per_model == methods_per_model - 1 and index != len(rows) - 1:
            lines.append(r"\midrule")
    return lines


def render_tex(toxicity_rows: list[dict], truthfulness_rows: list[dict]) -> str:
    lines = [
        "% Generated by figs/bench_table/bench_table.py. Do not edit by hand.",
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\caption{Source-comparable toxicity benchmark. Toxicity, Dist-2, and PPL are mean $\pm$ SE across five complete 1,000-prompt RealToxicityPrompts repetitions. MMLU is accuracy $\pm$ prompt-level SE on one shared 1,000-question five-shot set. TBD marks cells that have not been run.}",
        r"\label{tab:toxicity-benchmark}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Model & Method & CLS Tox. (\%) $\downarrow$ & Dist-2 $\uparrow$ & MMLU (\%) $\uparrow$ & PPL $\downarrow$ \\",
        r"\midrule",
        *_tex_rows(toxicity_rows),
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
        "",
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\caption{Source-comparable truthfulness benchmark. $\mathrm{T{\cdot}I}$ is truthful-times-informative performance. Values are mean $\pm$ SE across five complete 817-question TruthfulQA repetitions. TBD marks cells that have not been run.}",
        r"\label{tab:truthfulness-benchmark}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.7pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"Model & Method & TruthfulQA--ID T$\times$I $\uparrow$ & Spanish & Adversarial & Long & True (\%) $\uparrow$ & Info (\%) $\uparrow$ & MMLU (\%) $\uparrow$ \\",
        r"\midrule",
        *_tex_rows(truthfulness_rows),
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    toxicity_rows = _rows("toxicity", _load_results("toxicity"))
    truthfulness_rows = _rows("truthfulness", _load_results("truthfulness"))
    PLOTS.mkdir(parents=True, exist_ok=True)
    (PLOTS / "bench_table.md").write_text(
        render_markdown(toxicity_rows, truthfulness_rows)
    )
    (PLOTS / "bench_table.tex").write_text(render_tex(toxicity_rows, truthfulness_rows))


if __name__ == "__main__":
    main()
