"""Hannah's cached-data analysis commands; no model loading or text generation."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from robust_steerability.experiments.diagnostics import (
    PREDICTORS, evaluate, load_run, pack_run, read_json, safe_id, score,
    sha256, share_report, verify_run, write_json,
)

def rho(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)

def r_squared(y, prediction):
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    return None if denominator == 0 else 1 - float(np.sum((y - prediction) ** 2)) / denominator

def prepare_panels(analysis_id: str, controller: str, shift: str, protocol: str,
                   normalization: str, include_synthetic: bool, *, cache_root: Path) -> Path:
    """Save panel-ready rows and held-out regression predictions, without rerunning synthesis."""
    destination = cache_root / "analyses" / safe_id(analysis_id)
    if destination.exists():
        raise ValueError("Analysis ID exists; use a new analysis_id to preserve prior results")
    rows, exclusions, sources = [], [], {}
    for path in sorted((cache_root / "runs").glob("*/score.json")):
        load_run(path.parent)
        record = read_json(path)
        if record["protocol_id"] != protocol or record["normalization_protocol_id"] != normalization:
            continue
        sources[str(path.relative_to(cache_root))] = sha256(path)
        reason = None
        if not record["score_available"]:
            reason = "infeasible or unconverged synthesis"
        elif record["synthetic"] != include_synthetic:
            reason = "synthetic/empirical cohort mismatch"
        elif record["coordinates"] != "normalized" or record["stage_costs_depth_weighted"] is not True:
            reason = "cross-model normalized coordinates required"
        evaluations = []
        for candidate in sorted((path.parent / "evaluations").glob("*/summary.json")):
            result = read_json(candidate)
            if result["controller"] == controller and result["shift"] == shift:
                evaluations.append(result)
                sources[str(candidate.relative_to(cache_root))] = sha256(candidate)
        if len(evaluations) != 1:
            reason = "exactly one matching evaluation per run is required"
        if reason:
            exclusions.append({"run_id": record["run_id"], "reason": reason})
            continue
        evaluation = evaluations[0]
        rows.append({**{k: record[k] for k in ("run_id", "model_id", "model_family", "behavior", "parameter_count")},
                     **record["predictors"], **evaluation})
    pairs = [(row["model_id"], row["behavior"]) for row in rows]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Multiple runs for a model-behavior pair; select a unique predefined protocol")
    matching_rules = {json.dumps(row["matching_rule"], sort_keys=True) for row in rows}
    if len(matching_rules) > 1:
        raise ValueError("Controller comparison budgets/matching rules differ across pairs")
    predictions, metrics, correlations, descriptive_lines = [], [], [], []
    skipped = []
    y = np.array([row["reliability"] for row in rows])
    rng = np.random.default_rng(2151)
    for predictor in PREDICTORS:
        if len(rows) < 3 or any(row[predictor] is None for row in rows):
            skipped.append({"predictor": predictor, "reason": "fewer than 3 pairs or missing predictor values"})
            continue
        x = np.array([row[predictor] for row in rows], dtype=float)
        models = np.array([row["model_id"] for row in rows])
        bootstrap_values = []
        unique_models = np.unique(models)
        for _ in range(500):
            drawn = rng.choice(unique_models, size=len(unique_models), replace=True)
            indices = np.concatenate([np.flatnonzero(models == m) for m in drawn])
            value = rho(x[indices], y[indices])
            if value is not None:
                bootstrap_values.append(value)
        interval = np.quantile(bootstrap_values, [0.025, 0.975]).tolist() if len(bootstrap_values) >= 100 else [None, None]
        correlations.append({"predictor": predictor, "spearman_rho": rho(x, y),
                             "cluster_bootstrap_ci95": interval, "valid_bootstrap_samples": len(bootstrap_values)})
        coefficients = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
        descriptive_lines.append({"predictor": predictor, "intercept": coefficients[0], "slope": coefficients[1],
                                  "x_min": x.min(), "x_max": x.max(), "purpose": "descriptive in-sample line only"})
        for scheme, group_key in (("leave_model_out", "model_id"), ("leave_family_out", "model_family"),
                                  ("leave_behavior_out", "behavior"), ("leave_scale_out", "parameter_count")):
            groups = np.array([str(row[group_key]) for row in rows])
            if len(np.unique(groups)) < 2:
                skipped.append({"predictor": predictor, "scheme": scheme, "reason": "fewer than 2 groups"})
                continue
            predicted = np.full(len(y), np.nan)
            for held_out in np.unique(groups):
                train, test = groups != held_out, groups == held_out
                if train.sum() < 3:
                    continue
                mean, std = float(x[train].mean()), float(x[train].std())
                # Constant training predictors reduce to a training-mean model.
                scale = std if std > 0 else 1.0
                train_x = np.column_stack([np.ones(train.sum()), (x[train] - mean) / scale])
                beta = np.linalg.lstsq(train_x, y[train], rcond=None)[0]
                predicted[test] = beta[0] + beta[1] * (x[test] - mean) / scale
                for index in np.flatnonzero(test):
                    predictions.append({"run_id": rows[index]["run_id"], "model_family": rows[index]["model_family"],
                                        "behavior": rows[index]["behavior"], "predictor": predictor,
                                        "scheme": scheme, "held_out": held_out, "observed": y[index],
                                        "predicted": predicted[index], "training_run_ids": [rows[i]["run_id"] for i in np.flatnonzero(train)],
                                        "training_x_mean": mean, "training_x_scale": scale,
                                        "intercept_standardized": beta[0], "slope_standardized": beta[1]})
            complete = bool(np.isfinite(predicted).all())
            metrics.append({"predictor": predictor, "scheme": scheme, "n_pairs": len(y),
                            "n_predicted": int(np.isfinite(predicted).sum()), "complete": complete,
                            "r_squared": r_squared(y, predicted) if complete else None,
                            "spearman_rho": rho(predicted, y) if complete else None})
    destination.mkdir(parents=True)
    for name, contents in (("pair_table", rows), ("excluded_runs", exclusions),
                           ("cv_predictions", predictions), ("cv_metrics", metrics),
                           ("correlations", correlations), ("descriptive_lines", descriptive_lines),
                           ("unavailable_analyses", skipped)):
        write_json(destination / f"{name}.json", contents)
    write_json(destination / "panel_manifest.json", {
        "analysis_id": analysis_id, "controller": controller, "shift": shift,
        "protocol_id": protocol, "normalization_protocol_id": normalization,
        "synthetic": include_synthetic, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_hashes": sources, "exporter_sha256": sha256(Path(__file__)),
        "seed": 2151, "bootstrap_repetitions": 500,
        "panel_A": "runs/*/calibration_input.pt, manifest.json, score.json",
        "panel_B": "pair_table.json, correlations.json, descriptive_lines.json",
        "panel_C": "cv_metrics.json; leave_model_out, shared cohort, univariate OLS",
        "panel_D": "cv_predictions.json; leave_family_out, s_rob or negative_log_gamma_star",
        "response": "prompt-averaged binary success fraction; multiply by 100 for percent",
        "limitations": ["No held-out outcome is used by score()", "S_rob is not restricted to [0,1]",
                        "Point estimates use the current solver's numerical feasibility boundary",
                        "This is linear binary-success prediction, not the draft's mixed-effects model",
                        "No inference can verify the declared provenance of upstream tensors"]})
    return destination

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--bundle", type=Path, required=True)
    score_parser.add_argument("--device", default="cuda:0")
    score_parser.add_argument("--cache-root", type=Path, required=True)
    evaluation_parser = commands.add_parser("evaluate")
    evaluation_parser.add_argument("--input", type=Path, required=True)
    evaluation_parser.add_argument("--cache-root", type=Path, required=True)
    panel_parser = commands.add_parser("prepare-panels")
    panel_parser.add_argument("--analysis-id", required=True)
    panel_parser.add_argument("--controller", required=True)
    panel_parser.add_argument("--shift", required=True)
    panel_parser.add_argument("--protocol-id", required=True)
    panel_parser.add_argument("--normalization-id", required=True)
    panel_parser.add_argument("--synthetic", action="store_true")
    panel_parser.add_argument("--cache-root", type=Path, required=True)
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("--run", type=Path, required=True)
    inspect_parser.add_argument("--report", type=Path)
    pack_parser = commands.add_parser("pack")
    pack_parser.add_argument("--run", type=Path, required=True)
    pack_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "score":
        result = score(args.bundle, args.device, cache_root=args.cache_root)
    elif args.command == "evaluate":
        result = evaluate(args.input, cache_root=args.cache_root)
    elif args.command == "prepare-panels":
        result = prepare_panels(args.analysis_id, args.controller, args.shift, args.protocol_id,
                                args.normalization_id, args.synthetic, cache_root=args.cache_root)
    elif args.command == "inspect":
        loaded = load_run(args.run)
        print(json.dumps(loaded["score"], indent=2))
        result = share_report(args.run, args.report) if args.report else args.run
    else:
        result = pack_run(args.run, args.output)
    print(result)


if __name__ == "__main__":
    main()
