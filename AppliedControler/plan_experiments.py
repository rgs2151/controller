"""Generate AppliedControler experiment plans from selected OOD target tables."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS_TOP1 = Path(__file__).resolve().parent / "selected_ood_targets_top1.csv"
DEFAULT_TARGETS_TOP3 = Path(__file__).resolve().parent / "selected_ood_targets_top3.csv"
DEFAULT_SUMMARY = (
    ROOT
    / "parking"
    / "network_size_residual_explore"
    / "cache"
    / "benchmark_ood_sweep"
    / "benchmark_ood_sweep_summary.csv"
)


@dataclass
class MethodSpec:
    """Method declaration used by the comparison runner."""

    name: str
    params: dict[str, float]


@dataclass
class ExperimentSpec:
    """One model-subset-method run target."""

    label: str
    model_name: str
    subset: str
    rank: int
    overall: float
    early: float
    mid: float
    late: float
    method: str
    method_params: dict[str, float]
    artifact_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        type=Path,
        default=DEFAULT_TARGETS_TOP1,
        help="CSV file with selected OOD targets (top1 or top3).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Benchmark summary CSV used to map labels to huggingface model names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "experiment_plan_top1.json",
        help="Output JSON file for the generated plan.",
    )
    parser.add_argument(
        "--include-deferred",
        action="store_true",
        help="Include deferred placeholders for two external methods and one new method.",
    )
    return parser.parse_args()


def default_methods(include_deferred: bool) -> list[MethodSpec]:
    hinf_params = {
        "gamma_lower": 0.01,
        "gamma_upper": 10.0,
        "gamma_tolerance": 1e-4,
        "gamma_max_iterations": 60.0,
    }
    methods = [
        MethodSpec("alqr", {}),
        MethodSpec("spid", {"kp": 1.0, "ki": 0.0, "kd": 0.0}),
        MethodSpec("new_method", dict(hinf_params)),
    ]
    if include_deferred:
        methods.extend(
            [
                MethodSpec("external_method_a", {}),
                MethodSpec("external_method_b", {}),
            ]
        )
    return methods


def build_label_to_model_name(summary: pd.DataFrame) -> dict[str, str]:
    pairs = summary[["label", "model_name"]].drop_duplicates()
    return dict(zip(pairs["label"], pairs["model_name"], strict=False))


def extract_ranked_rows(targets: pd.DataFrame) -> list[tuple[int, pd.Series]]:
    records: list[tuple[int, pd.Series]] = []
    for _label, group in targets.groupby("label", sort=False):
        ranked = group.sort_values("overall", ascending=False).reset_index(drop=True)
        for offset, row in ranked.iterrows():
            records.append((offset + 1, row))
    return records


def plan_specs(
    targets: pd.DataFrame,
    summary: pd.DataFrame,
    methods: list[MethodSpec],
) -> list[ExperimentSpec]:
    label_to_model_name = build_label_to_model_name(summary)
    ranked_rows = extract_ranked_rows(targets)
    specs: list[ExperimentSpec] = []
    for rank, row in ranked_rows:
        label = str(row["label"])
        if label not in label_to_model_name:
            raise KeyError(f"Missing model_name mapping for label '{label}'")
        model_name = label_to_model_name[label]
        subset = str(row["subset"])
        for method in methods:
            specs.append(
                ExperimentSpec(
                    label=label,
                    model_name=model_name,
                    subset=subset,
                    rank=rank,
                    overall=float(row["overall"]),
                    early=float(row["early"]),
                    mid=float(row["mid"]),
                    late=float(row["late"]),
                    method=method.name,
                    method_params=dict(method.params),
                    artifact_path=(
                        "parking/residual_checks/cache/"
                        f"{label_to_model_name[label].replace('/', '__')}_{subset}_{method.name}.pt"
                    ),
                )
            )
    return specs


def write_plan(path: Path, methods: list[MethodSpec], specs: list[ExperimentSpec]) -> None:
    payload = {
        "methods": [asdict(method) for method in methods],
        "count": len(specs),
        "experiments": [asdict(spec) for spec in specs],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    targets = pd.read_csv(args.targets)
    summary = pd.read_csv(args.summary)
    methods = default_methods(args.include_deferred)
    specs = plan_specs(targets, summary, methods)
    write_plan(args.output, methods, specs)
    print(f"Wrote {len(specs)} experiment specs to {args.output}")


if __name__ == "__main__":
    main()
