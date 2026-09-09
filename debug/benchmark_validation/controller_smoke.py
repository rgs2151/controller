"""Fit production controller artifacts, then inspect small online generations."""

import argparse
import gc
from pathlib import Path

import torch

from robust_steerability.artifacts import implementation_hash
from robust_steerability.experiments.diagnostics import write_json
from robust_steerability.experiments.generation import generate_completions
from robust_steerability.experiments.manifest import load_manifest
from robust_steerability.experiments.methods import build_policy
from robust_steerability.experiments.runner import (
    _controller_arguments, _job_fingerprint, _load_controller, _model_spec,
    _prompt_sets, _token,
)
from robust_steerability.modeling.huggingface import load_causal_model


UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-index", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--records", choices=("calibration", "heldout"), default="calibration")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    model_entry = manifest.models[args.model_index]
    model, tokenizer = load_causal_model(
        _model_spec(model_entry), args.device, _token(ROOT))
    artifact, metadata = _load_controller(
        manifest, model_entry, model, tokenizer, args.device)
    excluded = (set(metadata["fit_prompt_ids"]) |
                set(metadata["source_fit_prompt_ids"]) |
                set(metadata["calibration_prompt_ids"]))
    prompt_sets = _prompt_sets(manifest, excluded)
    id_subset = "rtp_id" if manifest.kind == "id_toxicity" else "truthfulqa_id"
    records = (metadata["tuning_records"] if args.records == "calibration"
               else prompt_sets[id_subset])[:3]
    output = {
        "implementation_sha256": implementation_hash(),
        "job_fingerprint": _job_fingerprint(manifest, model_entry),
        "manifest": str(manifest.path.relative_to(ROOT)),
        "model": model_entry,
        "controller": metadata,
        "record_source": args.records,
        "records": [],
    }
    for method in manifest.methods:
        policy = build_policy(
            method, artifact, kp=float(manifest.payload["controller"]["kp"]),
            ki=float(manifest.payload["controller"]["ki"]),
            kd=float(manifest.payload["controller"]["kd"]), record=True)
        rows = generate_completions(
            model, tokenizer, records, policy=policy,
            seed=int(manifest.payload["seed"]), max_length=192,
            max_new_tokens=30, do_sample=False, top_p=1.0,
            temperature=1.0, repetition_penalty=1.0,
            trace_directory=UNIT / "cache/smoke_traces" / manifest.path.stem /
                            str(args.model_index) / method)
        output["records"].append({
            "method": method,
            "completions": [{"prompt_id": row["prompt_id"],
                             "completion": row["completion"],
                             "control_energy": row["control_energy"],
                             "hidden_delta_energy": row["hidden_delta_energy"]}
                            for row in rows],
        })
        print(f"Smoke {model_entry['label']} {method}: 3/3", flush=True)
    destination = UNIT / "cache" / f"smoke_{manifest.path.stem}_{args.model_index}_{args.records}.json"
    write_json(destination, output)
    print(f"Online controller smoke complete: {destination}", flush=True)
    del model, tokenizer, artifact
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
