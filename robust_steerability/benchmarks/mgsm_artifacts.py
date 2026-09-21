"""MGSM English-to-Spanish DiffMean and shared nominal dynamics artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from robust_steerability.benchmarks.artifacts import (
    _partition_records,
    aggregate_jacobian_partials,
)
from robust_steerability.benchmarks.layout import REPO_ROOT, artifact_root, dataset_root
from robust_steerability.benchmarks.specs import MODELS
from robust_steerability.calibration.nominal import average_prompt_jacobians
from robust_steerability.calibration.nominal_artifact import (
    nominal_dynamics_identity,
    save_nominal_dynamics,
)
from robust_steerability.datasets.mgsm import (
    aligned_direction_pairs,
    gsm8k_calibration_splits,
    materialize_evaluation,
)
from robust_steerability.modeling.huggingface import (
    CausalModelLoadSpec,
    load_access_token,
    load_causal_model,
    release_cuda_memory,
)
from robust_steerability.modeling.interventions import _decoder_layers
from robust_steerability.source_methods.id_benchmark import runtime_provenance


BENCHMARK = "mgsm"
CONCEPT = "respond only in Spanish, and no other language is allowed"
DIRECTION_RECORDS_PER_CLASS = 250
JACOBIAN_PROMPTS = 50
MAX_CALIBRATION_LENGTH = 512
MODEL_KEYS = ("qwen3_4b", "llama32_3b_instruct", "phi4_mini_instruct")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _save_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def model_load_spec(model_key: str) -> CausalModelLoadSpec:
    if model_key not in MODEL_KEYS:
        raise ValueError(f"{model_key!r} is not an MGSM model")
    model = MODELS[model_key]
    return CausalModelLoadSpec(
        model_id=model.model_id,
        revision=model.revision,
        quantized=False,
        dtype="bfloat16",
        attention_implementation="sdpa",
    )


def load_model(model_key: str, device: str):
    return load_causal_model(
        model_load_spec(model_key), device, load_access_token(REPO_ROOT)
    )


def _chat_text(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _prefix_length(tokenizer) -> int:
    left = tokenizer.apply_chat_template(
        [{"role": "user", "content": "1"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    right = tokenizer.apply_chat_template(
        [{"role": "user", "content": "2"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=True)):
        if left_token != right_token:
            return index
    raise ValueError("Could not identify the Qwen3 chat prefix")


def _suffix_length(tokenizer) -> int:
    left = tokenizer.apply_chat_template(
        [{"role": "user", "content": "1"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    right = tokenizer.apply_chat_template(
        [{"role": "user", "content": "2"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    for index, (left_token, right_token) in enumerate(zip(reversed(left), reversed(right))):
        if left_token != right_token:
            return index
    return 0


def prepare(model_key: str, tokenizer=None) -> dict:
    """Freeze direction, disturbance, tuning, and evaluation records once."""

    artifact_path = artifact_root(BENCHMARK, model_key) / "data.json"
    evaluation_path = dataset_root(BENCHMARK, model_key) / "mgsm.json"
    if artifact_path.exists() and evaluation_path.exists():
        return json.loads(artifact_path.read_text())
    model = MODELS[model_key]
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model.model_id,
            revision=model.revision,
            token=load_access_token(REPO_ROOT),
            padding_side="left",
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    pairs = aligned_direction_pairs()
    undesired = [
        {
            "prompt_id": f"{pair['pair_id']}-en",
            "pair_id": pair["pair_id"],
            "problem_index": pair["problem_index"],
            "text": _chat_text(tokenizer, str(pair["english_question"])),
        }
        for pair in pairs
    ]
    desired = [
        {
            "prompt_id": f"{pair['pair_id']}-es",
            "pair_id": pair["pair_id"],
            "problem_index": pair["problem_index"],
            "text": _chat_text(tokenizer, str(pair["spanish_question"])),
        }
        for pair in pairs
    ]
    auxiliary = gsm8k_calibration_splits()
    payload = {
        "schema_version": 1,
        "benchmark": BENCHMARK,
        "model": [model.model_id, model.revision],
        "concept": CONCEPT,
        "direction_estimator": "paired English/Spanish question DiffMean over valid tokens",
        "calibration": {
            "undesired": undesired,
            "desired": desired,
            "jacobian": desired[:JACOBIAN_PROMPTS],
            "disturbance": auxiliary["disturbance"],
            "tuning": auxiliary["tuning"],
        },
    }
    _write_json(artifact_path, payload)
    _write_json(evaluation_path, materialize_evaluation(tokenizer))
    return payload


def _class_token_mean(model, tokenizer, texts: list[str], batch_size: int) -> torch.Tensor:
    layers = _decoder_layers(model)
    device = next(model.parameters()).device
    prefix = _prefix_length(tokenizer)
    suffix = _suffix_length(tokenizer)
    total = None
    count = 0
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_CALIBRATION_LENGTH,
        ).to(device)
        captured: list[torch.Tensor | None] = [None] * (len(layers) + 1)
        handles = []

        def input_hook(index: int):
            def capture(_module, args):
                captured[index] = args[0].detach()

            return capture

        def terminal_hook(_module, _args, output):
            captured[-1] = (output[0] if isinstance(output, tuple) else output).detach()

        for index, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(input_hook(index)))
        handles.append(layers[-1].register_forward_hook(terminal_hook))
        try:
            with torch.inference_mode():
                model(**encoded, use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        if any(value is None for value in captured):
            raise RuntimeError("Failed to capture MGSM direction states")
        positions = encoded["attention_mask"].long().cumsum(dim=1) - 1
        lengths = encoded["attention_mask"].sum(dim=1, keepdim=True)
        valid = encoded["attention_mask"].bool() & (positions >= prefix)
        if suffix:
            valid &= positions < (lengths - suffix)
        batch_count = int(valid.sum())
        if batch_count == 0:
            raise ValueError("MGSM direction batch has no valid question tokens")
        batch_sum = torch.stack(
            [value.float()[valid].sum(dim=0).double().cpu() for value in captured]
        )
        if total is None:
            total = torch.zeros_like(batch_sum)
        total.add_(batch_sum)
        count += batch_count
    if total is None or count == 0:
        raise ValueError("MGSM direction class is empty")
    return (total / count).float()


def fit_setpoint(model_key: str, device: str) -> None:
    destination = artifact_root(BENCHMARK, model_key) / "setpoint.pt"
    if destination.exists():
        return
    model, tokenizer = load_model(model_key, device)
    data = prepare(model_key, tokenizer)
    started = time.perf_counter()
    batch_size = MODELS[model_key].activation_batch_size
    negative = _class_token_mean(
        model, tokenizer, [row["text"] for row in data["calibration"]["undesired"]], batch_size
    )
    positive = _class_token_mean(
        model, tokenizer, [row["text"] for row in data["calibration"]["desired"]], batch_size
    )
    contrast = positive - negative
    feature_norm = torch.linalg.vector_norm(contrast, dim=1)
    _save_torch(
        destination,
        {
            "identity": {
                "model": [MODELS[model_key].model_id, MODELS[model_key].revision],
                "concept": CONCEPT,
                "estimator": "paired MGSM question DiffMean",
                "desired_records": DIRECTION_RECORDS_PER_CLASS,
                "undesired_records": DIRECTION_RECORDS_PER_CLASS,
            },
            "contrast": contrast,
            "feature_norm": feature_norm,
        },
    )
    _write_json(
        artifact_root(BENCHMARK, model_key) / "runs/setpoint.json",
        {
            "status": "complete",
            "elapsed_seconds": time.perf_counter() - started,
            "runtime": runtime_provenance(device),
        },
    )
    del model, tokenizer
    release_cuda_memory(device)


def fit_jacobian_shard(
    model_key: str, device: str, shard_index: int, shard_count: int
) -> None:
    root = artifact_root(BENCHMARK, model_key)
    records = _partition_records(
        prepare(model_key)["calibration"]["jacobian"], shard_index, shard_count
    )
    partial = root / "jacobian_partials" / f"shard_{shard_index:02d}"
    model, tokenizer = load_model(model_key, device)
    started = time.perf_counter()
    mean = average_prompt_jacobians(
        model,
        tokenizer,
        records,
        cache_dir=partial,
        max_length=MAX_CALIBRATION_LENGTH,
        vjp_chunk_size=MODELS[model_key].jacobian_vjp_chunk_size,
        model_revision=MODELS[model_key].revision,
    )
    _write_json(
        root / "runs" / f"jacobian_shard_{shard_index:02d}.json",
        {
            "status": "complete",
            "record_count": len(records),
            "matrix_shape": list(mean.shape),
            "elapsed_seconds": time.perf_counter() - started,
            "runtime": runtime_provenance(device),
        },
    )


def fit_jacobians(
    model_key: str, devices: list[str], *, log_root: Path | None = None
) -> None:
    root = artifact_root(BENCHMARK, model_key)
    destination = root / "dynamics.pt"
    if destination.exists():
        return
    records = prepare(model_key)["calibration"]["jacobian"]
    logs = log_root or root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    for shard_index, device in enumerate(devices):
        path = logs / f"jacobian_shard_{shard_index:02d}.log"
        handle = path.open("a")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "robust_steerability.benchmarks.mgsm_artifacts",
                "--stage",
                "jacobian-shard",
                "--model",
                model_key,
                "--device",
                device,
                "--shard-index",
                str(shard_index),
                "--shard-count",
                str(len(devices)),
            ],
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, handle, path))
    failures = []
    for process, handle, path in processes:
        process.wait()
        handle.close()
        if process.returncode:
            failures.append(str(path))
    if failures:
        raise RuntimeError(f"MGSM Jacobian workers failed: {failures}")
    partials = [
        root / "jacobian_partials" / f"shard_{index:02d}"
        for index in range(len(devices))
    ]
    counts = [
        len(_partition_records(records, index, len(devices)))
        for index in range(len(devices))
    ]
    dynamics = aggregate_jacobian_partials(partials, counts)
    identity = nominal_dynamics_identity(
        behavior=CONCEPT,
        model_id=MODELS[model_key].model_id,
        model_revision=MODELS[model_key].revision,
        records=records,
        max_length=MAX_CALIBRATION_LENGTH,
        vjp_chunk_size=MODELS[model_key].jacobian_vjp_chunk_size,
    )
    save_nominal_dynamics(
        destination,
        identity,
        dynamics,
        attempts=[{"status": "complete", "created_at_utc": _utc_now()}],
    )
    for partial in partials:
        shutil.rmtree(partial)


def write_manifest(model_key: str) -> None:
    root = artifact_root(BENCHMARK, model_key)
    required = ("data.json", "setpoint.pt", "dynamics.pt")
    _write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "model": [MODELS[model_key].model_id, MODELS[model_key].revision],
            "concept": CONCEPT,
            "shared_by": ["spid", "alqr", "h_infinity"],
            "status": {
                name: "complete" if (root / name).exists() else "missing"
                for name in required
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("prepare", "setpoint", "jacobian-shard", "jacobians", "manifest"),
        required=True,
    )
    parser.add_argument("--model", choices=MODEL_KEYS, required=True)
    parser.add_argument("--device")
    parser.add_argument("--devices", nargs="*")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    arguments = parser.parse_args()
    if arguments.stage == "prepare":
        prepare(arguments.model)
    elif arguments.stage == "setpoint":
        fit_setpoint(arguments.model, arguments.device)
    elif arguments.stage == "jacobian-shard":
        fit_jacobian_shard(
            arguments.model,
            arguments.device,
            arguments.shard_index,
            arguments.shard_count,
        )
    elif arguments.stage == "jacobians":
        fit_jacobians(arguments.model, arguments.devices)
    else:
        write_manifest(arguments.model)


if __name__ == "__main__":
    main()
