#!/usr/bin/env python3
"""Build the final-only, ID+OOD Gamma Star collaborator bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path


UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[1]
PLOTS = UNIT / "plots"
EXPORTS = UNIT / "exports"
BUNDLE = EXPORTS / "gamma_star_collaborator_bundle"

FINAL_MODELS = {
    "truthfulness": {"gpt2_xl", "llama31_8b", "qwen25_14b", "olmo2_32b"},
    "harmful": {"llama32_1b", "llama32_3b", "llama31_8b"},
    "mgsm": {"qwen3_4b", "phi4_mini", "granite33_2b"},
    "lciteeval": {"qwen25_3b", "llama32_1b"},
}

METRICS = {
    "truthfulness": {
        "True (%) ↑": ("true", True, "percent"),
        "Informative (%) ↑": ("informative", True, "percent"),
        "T×I (%) ↑": ("true_times_informative", True, "percent"),
        "Instruction relevance (0–2) ↑": ("instruction_relevance", True, "score_0_2"),
        "Fluency (0–2) ↑": ("fluency", True, "score_0_2"),
    },
    "harmful": {
        "ASR (%) ↓": ("attack_success_rate", False, "percent"),
        "Safe-concept relevance (0–2) ↑": ("safe_concept_relevance", True, "score_0_2"),
        "Instruction relevance (0–2) ↑": ("instruction_relevance", True, "score_0_2"),
        "Fluency (0–2) ↑": ("fluency", True, "score_0_2"),
        "Overall steering (0–2) ↑": ("overall_steering", True, "score_0_2"),
    },
    "mgsm": {
        "Accuracy (%) ↑": ("accuracy", True, "percent"),
        "Spanish relevance (0–2) ↑": ("spanish_relevance", True, "score_0_2"),
        "Instruction relevance (0–2) ↑": ("instruction_relevance", True, "score_0_2"),
        "Fluency (0–2) ↑": ("fluency", True, "score_0_2"),
    },
    "lciteeval": {
        "Answer recall (%) ↑": ("answer_recall", True, "percent"),
        "Citation F1 (%) ↑": ("citation_f1", True, "percent"),
        "Concept relevance (0–2) ↑": ("concept_relevance", True, "score_0_2"),
        "Instruction relevance (0–2) ↑": ("instruction_relevance", True, "score_0_2"),
        "Fluency (0–2) ↑": ("fluency", True, "score_0_2"),
    },
}


def model_key(text: str) -> str:
    import re
    normalized = re.sub(r"[^a-z0-9]+", "", text.lower())
    aliases = [
        ("llama318b", "llama31_8b"), ("llama38b", "llama31_8b"),
        ("llama323b", "llama32_3b"), ("llama321b", "llama32_1b"),
        ("gpt2xl", "gpt2_xl"), ("qwen2514b", "qwen25_14b"),
        ("qwen253b", "qwen25_3b"), ("qwen34b", "qwen3_4b"),
        ("olmo232b", "olmo2_32b"), ("phi4mini", "phi4_mini"),
        ("granite332b", "granite33_2b"),
    ]
    for token, key in aliases:
        if token in normalized:
            return key
    return normalized


def markdown_rows(path: Path) -> list[dict[str, str]]:
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("|---"))
    headers = [cell.strip() for cell in lines[start].strip("|").split("|")]
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))
    return rows


def mean_se(cell: str) -> tuple[float, float | None]:
    parts = [part.strip() for part in cell.split("±", 1)]
    return float(parts[0]), float(parts[1]) if len(parts) == 2 else None


def add_source(output: list[dict[str, object]], benchmark: str, path: Path, distribution_fn, condition_axis: str, condition_column: str) -> None:
    for source_row in markdown_rows(path):
        key = model_key(source_row["Model"])
        if key not in FINAL_MODELS[benchmark]:
            continue
        condition = distribution_fn(source_row)[1]
        distribution = distribution_fn(source_row)[0]
        for column, (metric, higher, unit) in METRICS[benchmark].items():
            value, se = mean_se(source_row[column])
            output.append({
                "benchmark": benchmark,
                "distribution": distribution,
                "condition_axis": condition_axis,
                "condition": condition,
                "model_key": key,
                "model_label": source_row["Model"],
                "method": source_row["Method"],
                "metric": metric,
                "value": value,
                "standard_error": se,
                "higher_is_better": str(higher).lower(),
                "unit": unit,
                "source_path": path.relative_to(ROOT).as_posix(),
            })


def final_performance() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    add_source(rows, "truthfulness", ROOT / "figs/bench_table/truthfulness/truthfulqa.md", lambda _: ("id", "English"), "language", "")
    add_source(rows, "truthfulness", ROOT / "figs/bench_table/truthfulness/truthfulqa_spanish.md", lambda _: ("ood", "Spanish"), "language", "")
    add_source(rows, "harmful", ROOT / "figs/bench_table/harmful/harmbench_full.md", lambda r: (("id" if r["Template"] == "Direct" else "ood"), r["Template"]), "attack_template", "Template")
    add_source(rows, "mgsm", ROOT / "figs/bench_table/mgsm/mgsm_full.md", lambda r: ("ood", r["Language"]), "language", "Language")
    add_source(rows, "lciteeval", ROOT / "figs/bench_table/lciteeval/lciteeval_full.md", lambda r: (("id" if r["Context"] == "8K" else "ood"), r["Context"]), "context_length", "Context")
    keys = [(r["benchmark"], r["distribution"], r["condition"], r["model_key"], r["method"], r["metric"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate final-performance rows detected")
    observed = defaultdict(set)
    for row in rows:
        observed[row["benchmark"]].add(row["model_key"])
    if dict(observed) != FINAL_MODELS:
        raise RuntimeError(f"final model matrix mismatch: {dict(observed)}")
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def hinf_analysis(performance: list[dict[str, object]], selections: list[dict[str, str]]) -> list[dict[str, object]]:
    selection_index = {(r["benchmark"], r["model_key"]): r for r in selections}
    primary = {
        "truthfulness": {"true"},
        "harmful": {"attack_success_rate"},
        "mgsm": {"accuracy"},
        "lciteeval": {"answer_recall", "citation_f1"},
    }
    groups = defaultdict(list)
    for row in performance:
        if row["metric"] in primary[row["benchmark"]]:
            groups[(row["benchmark"], row["distribution"], row["condition_axis"], row["condition"], row["model_key"], row["model_label"], row["metric"])].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        benchmark, distribution, axis, condition, mkey, label, metric = key
        ours = next(r for r in rows if r["method"] == "H∞ (ours)")
        competitors = [r for r in rows if r["method"] != "H∞ (ours)"]
        if metric == "attack_success_rate":
            ours_reliability = 100.0 - float(ours["value"])
            best_row = min(competitors, key=lambda r: float(r["value"]))
            best_reliability = 100.0 - float(best_row["value"])
            outcome_metric = "safe_response_rate"
        else:
            ours_reliability = float(ours["value"])
            best_row = max(competitors, key=lambda r: float(r["value"]))
            best_reliability = float(best_row["value"])
            outcome_metric = metric
        selected = selection_index[(benchmark, mkey)]
        output.append({
            "benchmark": benchmark, "distribution": distribution,
            "condition_axis": axis, "condition": condition,
            "model_key": mkey, "model_label": label,
            "calibration_id": selected["calibration_id"],
            "outcome_metric": outcome_metric,
            "hinf_reliability": ours_reliability,
            "best_comparator_method": best_row["method"],
            "best_comparator_reliability": best_reliability,
            "hinf_differential": ours_reliability - best_reliability,
            "gamma_star": selected["gamma_star"],
            "s_rob": selected["s_rob"],
            "selection_metric": selected["selection_metric"],
            "performance_source": ours["source_path"],
            "selection_source": selected["selection_path"],
        })
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    performance = final_performance()
    selections = read_csv(PLOTS / "final_selected_calibrations.csv")
    controllers = read_csv(PLOTS / "final_controller_grid.csv")
    analysis = hinf_analysis(performance, selections)

    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)
    write_csv(BUNDLE / "final_performance_id_ood.csv", performance)
    write_csv(BUNDLE / "final_hinf_srob_analysis.csv", analysis)
    shutil.copy2(PLOTS / "final_selected_calibrations.csv", BUNDLE / "final_selected_calibrations.csv")
    shutil.copy2(PLOTS / "final_controller_grid.csv", BUNDLE / "final_controller_grid.csv")
    derive_script = '''#!/usr/bin/env python3
"""Recompute S_rob = 1 / gamma_star from the bundled final selections."""
import csv
from pathlib import Path

root = Path(__file__).resolve().parent
with (root / "final_selected_calibrations.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
fields = ["benchmark", "model_key", "model_label", "calibration_id", "gamma_star", "s_rob_recomputed"]
with (root / "derived_srob.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\\n")
    writer.writeheader()
    for row in rows:
        gamma = float(row["gamma_star"])
        writer.writerow({
            "benchmark": row["benchmark"], "model_key": row["model_key"],
            "model_label": row["model_label"], "calibration_id": row["calibration_id"],
            "gamma_star": gamma, "s_rob_recomputed": 1.0 / gamma,
        })
print(f"wrote {len(rows)} rows to derived_srob.csv")
'''
    (BUNDLE / "derive_srob.py").write_text(derive_script)
    readme = """# Gamma Star collaborator bundle

This bundle contains only final-paper benchmark/model pairs. It excludes toxicity, Spanish L-CiteEval, L-CiteEval Small, superseded calibrations, and unreported models.

- `final_performance_id_ood.csv`: all methods and metrics from the final tables, with ID/OOD labels.
- `final_hinf_srob_analysis.csv`: H-infinity primary outcomes joined to final gamma_star/S_rob, the best reported comparator, and the signed H-infinity differential. Repeated conditions from one model share the same controller and are not independent gamma estimates.
- `final_selected_calibrations.csv`: one explicit final H-infinity calibration for each of 12 benchmark/model pairs.
- `final_controller_grid.csv`: only controller-grid configurations belonging to those 12 final calibrations.
- `derive_srob.py`: standalone standard-library script that recomputes `S_rob = 1 / gamma_star` from the final selections.

MGSM has no reported ID evaluation in the final table, so its rows are OOD only. HarmBench Direct is ID and the five jailbreak templates are OOD. Truthfulness English is ID and Spanish is OOD. L-CiteEval 8K is ID and 16K is OOD.

Definitions: `S_rob = 1 / gamma_star`. `hinf_differential = hinf_reliability - best_comparator_reliability`, with HarmBench first transformed to safe-response rate (`100 - ASR`) so positive is always better for H-infinity.

Run `python3 derive_srob.py` inside the extracted directory to create `derived_srob.csv` independently.
"""
    (BUNDLE / "README.md").write_text(readme)
    manifest = {
        "schema_version": 1,
        "exclusions": ["toxicity", "lciteeval_spanish", "lciteeval_small", "superseded_calibrations", "unreported_models"],
        "allowed_models": {key: sorted(value) for key, value in FINAL_MODELS.items()},
        "counts": {
            "performance_rows": len(performance),
            "hinf_analysis_rows": len(analysis),
            "selected_calibrations": len(selections),
            "controller_configurations": len(controllers),
        },
    }
    files = sorted(path for path in BUNDLE.iterdir() if path.name != "manifest.json")
    manifest["files"] = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in files}
    (BUNDLE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    archive = EXPORTS / "gamma_star_collaborator_bundle.zip"
    archive.unlink(missing_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(BUNDLE.iterdir()):
            handle.write(path, arcname=f"gamma_star_collaborator_bundle/{path.name}")
    print(json.dumps({"bundle": str(BUNDLE), "zip": str(archive), **manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
