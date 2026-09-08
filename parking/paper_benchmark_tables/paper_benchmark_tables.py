"""Export manuscript tables exclusively from the current 50-prompt benchmark."""

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from robust_steerability.experiments.diagnostics import read_json, sha256, write_json
from robust_steerability.experiments.manifest import load_manifest
from robust_steerability.experiments.runner import _job_fingerprint

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[1]
SOURCE = ROOT / "parking/paper_benchmark_50"
PLOTS = UNIT / "plots"
METHODS = [
    ("original", "Original"), ("iti", "ITI"), ("actadd", "ActAdd"),
    ("mean_act", "Mean-AcT"), ("linear_act", "Linear-AcT"),
    ("pid_act", "PID-AcT"), ("odesteer", "ODESteer"),
    ("spid", "S-PID"), ("alqr", "A-LQR"),
    ("hinf", r"$\boldsymbol{H_\infty}$ (ours)"),
]
MODELS = ["DistilGPT-2", "Qwen-2.5-0.5B", "Qwen-2.5-1.5B", "Qwen-2.5-7B", "Qwen-2.5-14B"]



def number(mean, se=None, *, digits=2):
    if se is None:
        return "$" + f"{mean:.{digits}f}" + "$"
    return "$" + rf"{mean:.{digits}f}\,\pm\,{se:.{digits}f}" + "$"


def export():
    lookup, completions, provenance = {}, {}, []
    for behavior in ("toxicity", "truthfulness"):
        manifest = load_manifest(SOURCE / (behavior + ".json"))
        for model in manifest.models:
            slug = "".join(c.lower() if c.isalnum() else "_" for c in model["label"]).strip("_")
            path = SOURCE / "cache/jobs" / behavior / slug / "result.json"
            if not path.exists():
                continue
            payload = read_json(path)
            if payload["fingerprint"] != _job_fingerprint(manifest, model):
                raise ValueError(f"Result protocol mismatch: {path}")
            for row in payload["rows"]:
                if row["sample_count"] != 50 or row["mmlu_sample_count"] != 50:
                    raise ValueError("Only the current 50-prompt evaluation belongs in these tables")
                lookup[(behavior, row["model"], row["subset"], row["method"])] = row
            source = path.parent / "completions.json"
            completions[(behavior, model["label"])] = read_json(source)["subsets"]
            provenance.extend({"path": str(p.relative_to(ROOT)), "sha256": sha256(p)}
                              for p in (path, source))
    quality_path = SOURCE / "cache/quality/summary.json"
    quality = {}
    if quality_path.exists():
        for row in read_json(quality_path)["rows"]:
            path = SOURCE / row["source"]
            if row["source_sha256"] != sha256(path):
                raise ValueError("Quality results do not match the current completions")
            quality[(row["model"], row["method"])] = row
        provenance.append({"path": str(quality_path.relative_to(ROOT)), "sha256": sha256(quality_path)})
    coverage = []

    def value(table, model, method, metric):
        behavior = "toxicity" if table == 1 else "truthfulness"
        subset = metric if metric in {"spanish", "adversarial", "long", "jigsaw", "rtp_id"} else (
            "rtp_id" if table == 1 else "truthfulqa_id")
        row = lookup.get((behavior, model, subset, method))
        result, count, status = r"\TBD", 0, "not_yet_measured"
        if row is not None:
            count = 50
            if table == 1 and metric in {"rtp_id", "spanish", "adversarial", "jigsaw", "long"}:
                result = number(row["toxicity_percent"], row["toxicity_se"])
            elif metric == "mmlu":
                result = number(row["mmlu_mean"], row["mmlu_se"])
            elif table == 2:
                field = "ti" if metric in {"ti", "spanish", "adversarial", "long"} else metric
                result = number(row[field + "_mean"], row[field + "_se"])
            elif metric == "dist2":
                pairs = []
                for record in completions[("toxicity", model)]["rtp_id"][method]:
                    words = record["completion"].lower().split()
                    pairs.extend(zip(words[:-1], words[1:]))
                if pairs:
                    result = number(len(set(pairs)) / len(pairs), digits=3)
                else:
                    status = "undefined_no_bigrams"
            elif metric == "ppl" and (model, method) in quality:
                q = quality[(model, method)]
                count = q["ppl_count"]
                if q["ppl_mean"] is not None:
                    result = number(q["ppl_mean"], q["ppl_se"])
                    if count != 50:
                        result += rf" \scriptsize$(n={count})$"
                else:
                    status = "undefined_empty_completions"
        if result != r"\TBD":
            status = "measured"
        coverage.append({"table": table, "model": model, "method": method, "metric": metric,
                         "sample_count": count, "status": status, "latex": result})
        return result

    captions = {
        1: r"""Toxicity steering with 50 prompts per condition and fixed settings.
Toxicity is mean toxic-class probability (\%), not thresholded frequency.
Values show mean $\pm$ prompt-level SE; Dist-2 is pooled ID bigram diversity,
without cross-completion bigrams. PPL scores nonempty ID continuations under
a fixed unsteered Mistral-7B; a smaller valid count is shown explicitly.
MMLU uses 50 context-fitting, intact five-shot questions.
Spanish requests English output; Adversarial transfers D6 literal markers;
Jigsaw/Long use longest Jigsaw/ToxicChat prompts.
Baseline operators use our shared fit/intervention setup, not the original
papers' complete protocols. Red TBD cells are unmeasured or undefined.""",
        2: r"""Truthfulness steering with separate truthfulness calibration,
fixed settings, and 50 questions per condition.
$\mathrm{T{\cdot}I}$ is the product of marginal True and Info rates (\%);
its SE includes their within-question covariance. True, Info, and MMLU
show percentage mean $\pm$ Bernoulli SE.
Spanish requests English output; Adversarial transfers D6 literal markers;
Long adds unrelated archive text. MMLU uses the same intact five-shot questions.
Baseline operators use our shared fit/intervention setup.
Red TBD cells are unmeasured. These are single-run prompt-level errors,
not variation across repeated runs.""",
    }
    metrics = {
        1: ["rtp_id", "spanish", "adversarial", "jigsaw", "long", "dist2", "mmlu", "ppl"],
        2: ["ti", "spanish", "adversarial", "long", "true", "info", "mmlu"],
    }
    headers = {
        1: r"""& & \multicolumn{5}{c|}{\textbf{Continuation toxicity (\%) $\downarrow$}}
& \multicolumn{3}{c}{\textbf{Quality and capability}}\\
\cmidrule(lr){3-7}\cmidrule(lr){8-10}
Model & Method & RTP--ID & Spanish & Adversarial & Jigsaw & Long
& Dist-2 $\uparrow$ & MMLU (\%) $\uparrow$ & PPL $\downarrow$\\""",
        2: r"""& & \multicolumn{4}{c|}{\textbf{Truthfulness$\boldsymbol{\times}$Informativeness $\uparrow$}}
& \multicolumn{3}{c}{\textbf{ID components and capability}}\\
\cmidrule(lr){3-6}\cmidrule(lr){7-9}
Model & Method & TruthfulQA--ID & Spanish & Adversarial & Long
& True (\%) $\uparrow$ & Info (\%) $\uparrow$ & MMLU (\%) $\uparrow$\\""",
    }
    previous = [PLOTS / name for name in ("table1.tex", "table2.tex", "cell_coverage.csv", "provenance.json", "toxicity_summary.csv", "benchmark_tables.pdf")
                if (PLOTS / name).exists()]
    if previous:
        (UNIT / "cache").mkdir(parents=True, exist_ok=True)
        archive = Path(tempfile.mkdtemp(prefix="table_export_", dir=UNIT / "cache"))
        for path in previous:
            path.rename(archive / path.name)
        print(f"Previous table iteration retained in {archive}", flush=True)
    for table in (1, 2):
        label = "toxicity" if table == 1 else "truthfulness"
        columns = "llccccc|ccc" if table == 1 else "llcccc|ccc"
        lines = [r"\begin{table*}[!htbp]", r"\centering", r"\caption{" + captions[table] + "}",
                 rf"\label{{tab:{label}-benchmark}}", r"\scriptsize",
                 r"\setlength{\tabcolsep}{2.7pt}", r"\renewcommand{\arraystretch}{1.02}",
                 r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{{columns}}}",
                 r"\toprule", headers[table], r"\midrule"]
        for model_index, model in enumerate(MODELS):
            name = model.replace("Qwen-2.5", "Qwen2.5")
            for method_index, (method, display) in enumerate(METHODS):
                model_cell = rf"\multirow{{10}}{{*}}{{{name}}}" if method_index == 0 else ""
                cells = [value(table, model, method, metric) for metric in metrics[table]]
                lines.append(" & ".join([model_cell, display, *cells]) + r"\\")
            if model_index < len(MODELS) - 1:
                lines.append(r"\midrule")
        lines.extend([r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}", ""])
        (PLOTS / f"table{table}.tex").write_text("\n".join(lines))
    pd.DataFrame(coverage).to_csv(PLOTS / "cell_coverage.csv", index=False)
    write_json(PLOTS / "provenance.json", {
        "sources": provenance,
        "populated_cells": {str(t): sum(r["table"] == t and r["status"] == "measured" for r in coverage)
                            for t in (1, 2)},
        "expected_cells": {"1": 400, "2": 350},
        "exclusions": ["All historical Erfan result caches", "All earlier-paper result values"],
    })
    print(pd.DataFrame(coverage).groupby(["table", "status"]).size().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["export"])
    parser.parse_args()
    PLOTS.mkdir(parents=True, exist_ok=True)
    export()
