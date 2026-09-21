"""Spanish DiffMean setpoint and 8K shared dynamics for L-CiteEval Small."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
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
from robust_steerability.datasets.lciteeval_small import (
    JACOBIAN_PROMPTS,
    SPANISH_CONCEPT,
    materialize_direction,
    materialize_evaluation,
    materialize_h_infinity_splits,
)
from robust_steerability.modeling.huggingface import (
    CausalModelLoadSpec,
    cuda_device_index,
    load_access_token,
    load_causal_model,
    release_cuda_memory,
)
from robust_steerability.modeling.interventions import _decoder_layers
from robust_steerability.modeling.tokenization import (
    PREFORMATTED_CHAT_TOKENIZATION,
    tokenize_prompts,
)
from robust_steerability.source_methods.id_benchmark import runtime_provenance


BENCHMARK = "lciteeval_small"
AXBENCH_CONCEPT = SPANISH_CONCEPT
JACOBIAN_MAX_LENGTH = 10_000
CONTEXT_WINDOW = 131_072
PROMPT_TOKENIZATION = PREFORMATTED_CHAT_TOKENIZATION


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
    model = MODELS[model_key]
    if model_key != "llama32_1b_instruct":
        raise ValueError(f"{model_key!r} is not the L-CiteEval Small model")
    return CausalModelLoadSpec(
        model_id=model.model_id,
        revision=model.revision,
        quantized=False,
        dtype="bfloat16",
        attention_implementation="sdpa",
        rope_scaling=None,
    )


def load_model(model_key: str, device: str):
    return load_causal_model(
        model_load_spec(model_key), device, load_access_token(REPO_ROOT)
    )


def _suffix_length(tokenizer) -> int:
    first = tokenizer.apply_chat_template(
        [{"role": "user", "content": "1"}], tokenize=True
    )
    second = tokenizer.apply_chat_template(
        [{"role": "user", "content": "2"}], tokenize=True
    )
    for index, (left, right) in enumerate(zip(reversed(first), reversed(second))):
        if left != right:
            return index
    return 0


def _prefix_length(tokenizer) -> int:
    first = tokenizer.apply_chat_template(
        [{"role": "user", "content": "1"}], tokenize=True
    )
    second = tokenizer.apply_chat_template(
        [{"role": "user", "content": "2"}], tokenize=True
    )
    for index, (left, right) in enumerate(zip(first, second)):
        if left != right:
            return index
    raise ValueError("Could not identify the AXBench chat prefix")


def _axbench_text(tokenizer, model_key: str, prompt: str, response: str) -> str:
    messages = []
    if model_key == "llama32_1b_instruct":
        messages.append({"role": "system", "content": "You are a helpful assistant."})
    messages.extend(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    )
    tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    suffix = _suffix_length(tokenizer)
    stop = -suffix if suffix else None
    tokens = tokens[1:stop]
    return tokenizer.decode(tokens)


def _instruction_text(tokenizer, instruction: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )


def prepare(model_key: str, tokenizer=None) -> dict:
    model = MODELS[model_key]
    artifact_path = artifact_root(BENCHMARK, model_key) / "data.json"
    evaluation_path = dataset_root(BENCHMARK, model_key) / "lciteeval.json"
    if artifact_path.exists() and evaluation_path.exists():
        payload = json.loads(artifact_path.read_text())
        if payload.get("prompt_tokenization") != PROMPT_TOKENIZATION:
            payload["prompt_tokenization"] = PROMPT_TOKENIZATION
            _write_json(artifact_path, payload)
        return payload
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
    direction = materialize_direction(tokenizer)
    prompt_splits = materialize_h_infinity_splits(
        tokenizer, context_window=CONTEXT_WINDOW
    )
    evaluation = materialize_evaluation(tokenizer, context_window=CONTEXT_WINDOW)
    payload = {
        "schema_version": 1,
        "benchmark": BENCHMARK,
        "model": [model.model_id, model.revision],
        "concept": SPANISH_CONCEPT,
        "prompt_tokenization": PROMPT_TOKENIZATION,
        "direction_estimator": "paired MGSM Spanish-minus-English question DiffMean",
        "calibration": {
            **direction,
            "jacobian": prompt_splits["jacobian"],
            "disturbance": prompt_splits["disturbance"],
            "tuning": prompt_splits["tuning"],
        },
    }
    _write_json(artifact_path, payload)
    _write_json(
        evaluation_path,
        evaluation,
    )
    return payload


def _class_token_mean(model, tokenizer, texts: list[str], batch_size: int) -> torch.Tensor:
    layers = _decoder_layers(model)
    device = next(model.parameters()).device
    prefix = _prefix_length(tokenizer)
    total = None
    count = 0
    for start in range(0, len(texts), batch_size):
        encoded = tokenize_prompts(
            tokenizer,
            texts[start : start + batch_size],
            prompt_tokenization=PROMPT_TOKENIZATION,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=JACOBIAN_MAX_LENGTH,
        ).to(device)
        captured: list[torch.Tensor | None] = [None] * (len(layers) + 1)
        handles = []

        def input_hook(index: int):
            def capture(_module, args):
                captured[index] = args[0].detach()

            return capture

        def terminal_hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[-1] = hidden.detach()

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
            raise RuntimeError("Failed to capture AXBench DiffMean states")
        positions = encoded["attention_mask"].long().cumsum(dim=1) - 1
        valid = encoded["attention_mask"].bool() & (positions >= prefix)
        batch_count = int(valid.sum())
        if batch_count == 0:
            raise ValueError("AXBench batch has no non-prefix tokens")
        batch_sum = torch.stack(
            [value.float()[valid].sum(dim=0).double().cpu() for value in captured]
        )
        if total is None:
            total = torch.zeros_like(batch_sum)
        total.add_(batch_sum)
        count += batch_count
    if total is None or count == 0:
        raise ValueError("AXBench direction class is empty")
    return (total / count).float()


def fit_setpoint(model_key: str, device: str) -> None:
    destination = artifact_root(BENCHMARK, model_key) / "setpoint.pt"
    if destination.exists():
        payload = torch.load(destination, map_location="cpu", weights_only=True)
        if payload.get("identity", {}).get("prompt_tokenization") != PROMPT_TOKENIZATION:
            raise ValueError(
                "The existing L-CiteEval Small setpoint used the obsolete "
                "duplicate-BOS tokenization. Remove this derived artifact before rerunning."
            )
        return
    model, tokenizer = load_model(model_key, device)
    data = prepare(model_key, tokenizer)
    batch_size = MODELS[model_key].activation_batch_size
    started = time.perf_counter()
    negative = _class_token_mean(
        model,
        tokenizer,
        [str(row["text"]) for row in data["calibration"]["undesired"]],
        batch_size,
    )
    positive = _class_token_mean(
        model,
        tokenizer,
        [str(row["text"]) for row in data["calibration"]["desired"]],
        batch_size,
    )
    contrast = positive - negative
    feature_norm = torch.linalg.vector_norm(contrast, dim=1)
    _save_torch(
        destination,
        {
            "identity": {
                "model": [MODELS[model_key].model_id, MODELS[model_key].revision],
                "concept": AXBENCH_CONCEPT,
                "estimator": "paired MGSM Spanish-minus-English question DiffMean",
                "desired_records": 250,
                "undesired_records": 250,
                "prompt_tokenization": PROMPT_TOKENIZATION,
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
    cache = artifact_root(BENCHMARK, model_key)
    data = prepare(model_key)
    records = _partition_records(
        data["calibration"]["jacobian"], shard_index, shard_count
    )
    partial = cache / "jacobian_partials" / f"shard_{shard_index:02d}"
    model, tokenizer = load_model(model_key, device)
    started = time.perf_counter()
    mean = average_prompt_jacobians(
        model,
        tokenizer,
        records,
        cache_dir=partial,
        max_length=JACOBIAN_MAX_LENGTH,
        vjp_chunk_size=MODELS[model_key].jacobian_vjp_chunk_size,
        model_revision=MODELS[model_key].revision,
        prompt_tokenization=PROMPT_TOKENIZATION,
    )
    _write_json(
        cache / "runs" / f"jacobian_shard_{shard_index:02d}.json",
        {
            "status": "complete",
            "record_count": len(records),
            "layer_count": int(mean.shape[0]),
            "hidden_size": int(mean.shape[-1]),
            "elapsed_seconds": time.perf_counter() - started,
            "runtime": runtime_provenance(device),
        },
    )


def fit_jacobians(
    model_key: str, devices: list[str], *, log_root: Path | None = None
) -> None:
    cache = artifact_root(BENCHMARK, model_key)
    destination = cache / "dynamics.pt"
    if destination.exists():
        metadata_path = destination.with_suffix(".json")
        identity = (
            json.loads(metadata_path.read_text()).get("identity", {})
            if metadata_path.exists()
            else {}
        )
        if identity.get("prompt_tokenization") != PROMPT_TOKENIZATION:
            raise ValueError(
                "The existing L-CiteEval Small dynamics used the obsolete "
                "duplicate-BOS tokenization. Remove this derived artifact and its "
                "Jacobian partials before rerunning."
            )
        return
    data = prepare(model_key)
    records = data["calibration"]["jacobian"]
    logs = log_root or cache / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    for shard_index, device in enumerate(devices):
        path = logs / f"jacobian_shard_{shard_index:02d}.log"
        handle = path.open("a")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "robust_steerability.benchmarks.lciteeval_small_artifacts",
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
        raise RuntimeError(f"L-CiteEval Jacobian workers failed: {failures}")
    partials = [
        cache / "jacobian_partials" / f"shard_{index:02d}"
        for index in range(len(devices))
    ]
    counts = [
        len(_partition_records(records, index, len(devices)))
        for index in range(len(devices))
    ]
    dynamics = aggregate_jacobian_partials(partials, counts)
    identity = nominal_dynamics_identity(
        behavior=AXBENCH_CONCEPT,
        model_id=MODELS[model_key].model_id,
        model_revision=MODELS[model_key].revision,
        records=records,
        max_length=JACOBIAN_MAX_LENGTH,
        vjp_chunk_size=MODELS[model_key].jacobian_vjp_chunk_size,
        prompt_tokenization=PROMPT_TOKENIZATION,
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
            "concept": AXBENCH_CONCEPT,
            "prompt_tokenization": PROMPT_TOKENIZATION,
            "shared_by": ["alqr", "h_infinity"],
            "status": {
                name: "complete" if (root / name).exists() else "missing"
                for name in required
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "setpoint", "jacobian-shard"), required=True
    )
    parser.add_argument(
        "--model", choices=("llama32_1b_instruct",), required=True
    )
    parser.add_argument("--device")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    arguments = parser.parse_args()
    if arguments.stage == "prepare":
        prepare(arguments.model)
    elif arguments.stage == "setpoint":
        fit_setpoint(arguments.model, arguments.device)
    else:
        fit_jacobian_shard(
            arguments.model,
            arguments.device,
            arguments.shard_index,
            arguments.shard_count,
        )


if __name__ == "__main__":
    main()
