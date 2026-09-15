"""Render the active benchmark tables to synchronized Markdown and TeX."""

from __future__ import annotations

import json
from pathlib import Path


UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[1]
RESULTS_ROOT = REPO / "benchmarks"
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
    benchmark = "truthfulness" if behavior.startswith("truthfulness") else "toxicity"
    for model_key, _model_label in MODELS:
        for method, _markdown_label, _tex_label in METHODS:
            path = (
                RESULTS_ROOT
                / benchmark
                / "results/kv_cache_off"
                / model_key
                / behavior
                / f"{method}.json"
            )
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


def _rows(
    behavior: str,
    results: dict[tuple[str, str], dict],
    shifted_results: dict[tuple[str, str], dict],
) -> list[dict]:
    rows = []
    for model_key, model_label in MODELS:
        for method, markdown_label, tex_label in METHODS:
            result = results.get((model_key, method))
            shifted = shifted_results.get((model_key, method))
            if behavior == "toxicity":
                values = [
                    _metric(result, "toxicity"),
                    _metric(result, "dist_2"),
                    _metric(result, "perplexity"),
                    _metric(shifted, "toxicity"),
                    _metric(shifted, "dist_2"),
                    _metric(shifted, "perplexity"),
                ]
            else:
                values = [
                    _metric(result, "truth"),
                    _metric(result, "info"),
                    _metric(result, "instruction_relevance"),
                    _metric(result, "fluency"),
                    _metric(shifted, "truth"),
                    _metric(shifted, "info"),
                    _metric(shifted, "instruction_relevance"),
                    _metric(shifted, "fluency"),
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
    return "TBD" if value is None else f"{value[0]:.2f} ± {value[1]:.2f}"


def _tex_cell(value) -> str:
    return r"\textbf{TBD}" if value is None else f"${value[0]:.2f}\\,\\pm\\,{value[1]:.2f}$"


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
        "Generated from completed controlled-decoding summaries with evaluated-model KV "
        "cache disabled. The TeX table is generated from the same rows.",
        "",
    ]
    lines.extend(
        _markdown_table(
            "Toxicity benchmark",
            (
                "RTP CLS Tox. (%) ↓",
                "RTP Dist 2 ↑",
                "RTP PPL ↓",
                "Jigsaw CLS Tox. (%) ↓",
                "Jigsaw Dist 2 ↑",
                "Jigsaw PPL ↓",
            ),
            toxicity_rows,
        )
    )
    lines.extend(["", "Toxicity values are mean ± SE across five 1,000-prompt repetitions.", ""])
    lines.extend(
        _markdown_table(
            "Truthfulness benchmark",
            (
                "ID True (%) ↑",
                "ID Informative (%) ↑",
                "ID Instruction relevance (0–2) ↑",
                "ID Fluency (0–2) ↑",
                "Spanish True (%) ↑",
                "Spanish Informative (%) ↑",
                "Spanish Instruction relevance (0–2) ↑",
                "Spanish Fluency (0–2) ↑",
            ),
            truthfulness_rows,
        )
    )
    lines.extend(
        [
            "",
            "Truthfulness values are mean ± SE across five complete 817-question "
            "repetitions. Scorers run independently; TBD means that scorer has not been run.",
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
        r"\caption{Source-comparable toxicity benchmark with evaluated-model KV cache disabled. Toxicity, Dist-2, and PPL are mean $\pm$ SE across five complete 1,000-prompt repetitions for RTP and Jigsaw. TBD marks cells that have not been run.}",
        r"\label{tab:toxicity-benchmark}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Model & Method & RTP Tox. (\%) $\downarrow$ & RTP Dist-2 $\uparrow$ & RTP PPL $\downarrow$ & Jigsaw Tox. (\%) $\downarrow$ & Jigsaw Dist-2 $\uparrow$ & Jigsaw PPL $\downarrow$ \\",
        r"\midrule",
        *_tex_rows(toxicity_rows),
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
        "",
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\caption{Truthfulness benchmark with evaluated-model KV cache disabled. True and Informative use the pinned TruthfulQA judges and rubrics; instruction relevance and fluency use the independent AXBench 0--2 rubrics. Values are mean $\pm$ SE across five complete 817-question repetitions. TBD marks scorers that have not been run.}",
        r"\label{tab:truthfulness-benchmark}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"Model & Method & ID True (\%) $\uparrow$ & ID Info. (\%) $\uparrow$ & ID Rel. (0--2) $\uparrow$ & ID Flu. (0--2) $\uparrow$ & Spanish True (\%) $\uparrow$ & Spanish Info. (\%) $\uparrow$ & Spanish Rel. (0--2) $\uparrow$ & Spanish Flu. (0--2) $\uparrow$ \\",
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
    PLOTS.mkdir(parents=True, exist_ok=True)
    toxicity_rows = _rows(
        "toxicity", _load_results("toxicity"), _load_results("toxicity_jigsaw")
    )
    truthfulness_rows = _rows(
        "truthfulness",
        _load_results("truthfulness"),
        _load_results("truthfulness_spanish"),
    )
    (PLOTS / "bench_table.md").write_text(
        render_markdown(toxicity_rows, truthfulness_rows)
    )
    (PLOTS / "bench_table.tex").write_text(
        render_tex(toxicity_rows, truthfulness_rows)
    )


if __name__ == "__main__":
    main()
