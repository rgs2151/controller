"""Compare cached and uncached Spanish TruthfulQA generation on 100 matched prompts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from robust_steerability.benchmarks.metrics import judge_label, truth_judge_prompt
from robust_steerability.experiments.methods import ControllerArtifact, build_policy
from robust_steerability.modeling.huggingface import load_access_token
from robust_steerability.modeling.interventions import register_generation_policy_hooks
from robust_steerability.source_methods.control import build_alqr_policy
from robust_steerability.source_methods.generation import generate_batched
from robust_steerability.source_methods.id_benchmark import (
    load_frozen_alqr_artifacts,
    runtime_provenance,
)
from robust_steerability.source_methods.modeling import (
    load_source_model,
)
from robust_steerability.source_methods.protocol import (
    SOURCE_RANDOM_SEED,
    paper_alqr_setting,
)


UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[1]
DATA = REPO / "parking/truthfulqa_spanish/data/truthfulqa_spanish.json"
BENCH = REPO / "benchmarks/truthfulness/cache/gemma2b/evaluations"
ALQR_ROOT = REPO / "benchmarks/truthfulness/cache/gemma2b/artifacts"
HINF_SOURCE = REPO / "parking/dist_changes/cache/hinf_controller.pt"
HINF_SELECTED = ALQR_ROOT / "h_infinity_truthfulness/controller.pt"
MODEL_ID = "google/gemma-2-2b"
MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
TRUTH_JUDGE = "allenai/truthfulqa-truth-judge-llama2-7B"
TRUTH_JUDGE_REVISION = "8f718cb4e68bc30a488727a0e37190835a7c129e"
INFO_JUDGE = "allenai/truthfulqa-info-judge-llama2-7B"
INFO_JUDGE_REVISION = "72601eff294cbf7e8d7188ae82397e19cb9bb9bd"
METHODS = ("original", "alqr", "h_infinity")
SAMPLE_COUNT = 100
BATCH_SIZE = 8
JUDGE_BATCH_SIZE = 16
JUDGE_MAX_NEW_TOKENS = 20
REFERENCE = UNIT / "cache/cache_on_reference.json"
RESULTS = UNIT / "cache/results.json"
TABLE = UNIT / "plots/kv_cache_investigation.md"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _records() -> list[dict]:
    payload = json.loads(DATA.read_text())
    rows = payload["evaluation"]["truthfulness_spanish"]["0"][:SAMPLE_COUNT]
    if (
        payload.get("status") != "quality_checked"
        or payload.get("quality_audit", {}).get("final_failed") != 0
        or len(rows) != SAMPLE_COUNT
        or any("Responde en inglés.\nRespuesta:" not in row["text"] for row in rows)
    ):
        raise ValueError("Frozen Spanish dataset does not match this investigation")
    return rows


def _single_json(root: Path) -> Path:
    paths = sorted(root.glob("*.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one JSON cache under {root}; found {len(paths)}")
    return paths[0]


def prepare_reference() -> None:
    records = _records()
    identity = {
        "schema_version": 1,
        "sample_count": SAMPLE_COUNT,
        "selection": "first 100 prompts of Spanish repetition 0",
        "prompt_ids": [row["prompt_id"] for row in records],
        "spanish_data_sha256": _sha(DATA),
        "generation_use_cache": True,
    }
    methods = {}
    source_hashes = {}
    for method in METHODS:
        generation_path = _single_json(
            BENCH / "generations/truthfulness_spanish/gemma2b" / method
        )
        score_path = (
            BENCH
            / "scores/truthfulness_spanish/gemma2b"
            / method
            / generation_path.name
        )
        generation = json.loads(generation_path.read_text())
        scores = json.loads(score_path.read_text())
        if generation.get("status") != "complete" or scores.get("status") != "complete":
            raise ValueError(f"Incomplete cache-on benchmark data for {method}")
        generation_rows = generation["repetitions"][0]["rows"][:SAMPLE_COUNT]
        true_rows = scores["true"][:SAMPLE_COUNT]
        info_rows = scores["helpful"][:SAMPLE_COUNT]
        expected_ids = [row["prompt_id"] for row in records]
        if not (
            [row["prompt_id"] for row in generation_rows] == expected_ids
            == [row["prompt_id"] for row in true_rows]
            == [row["prompt_id"] for row in info_rows]
        ):
            raise ValueError(f"Cache-on rows do not align for {method}")
        methods[method] = [
            {
                "prompt_id": source["prompt_id"],
                "question": source["question"],
                "text": source["text"],
                "completion": generated["completion"],
                "true": float(true["score"]),
                "info": float(info["score"]),
            }
            for source, generated, true, info in zip(
                records, generation_rows, true_rows, info_rows, strict=True
            )
        ]
        source_hashes[method] = {
            "generation": _sha(generation_path),
            "scores": _sha(score_path),
        }
    payload = {"identity": identity, "source_hashes": source_hashes, "methods": methods}
    if REFERENCE.exists() and json.loads(REFERENCE.read_text()) != payload:
        raise ValueError(f"Cache-on reference changed: {REFERENCE}")
    if not REFERENCE.exists():
        _write_json(REFERENCE, payload)


def _alqr_policy(device: str):
    setting = paper_alqr_setting("truthfulness", MODEL_ID)
    setpoint, dynamics, hashes, _selection = load_frozen_alqr_artifacts(
        artifact_root=ALQR_ROOT,
        behavior="truthfulness",
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        parameters={
            "lambda": setting.multiplier,
            "q": setting.q,
            "r": setting.r,
            "q_final": setting.q_final,
        },
    )
    policy = build_alqr_policy(
        dynamics,
        setpoint,
        multiplier=setting.multiplier,
        q=setting.q,
        r=setting.r,
        q_final=setting.q_final,
        device=device,
    )
    return policy, hashes, asdict(setting)


def _hinf_policy():
    source = torch.load(HINF_SOURCE, map_location="cpu", weights_only=True, mmap=True)
    selected = torch.load(
        HINF_SELECTED, map_location="cpu", weights_only=True, mmap=True
    )
    if not bool(selected.get("feasible")):
        raise ValueError("Frozen H-infinity controller is infeasible")
    base = ControllerArtifact(**source["artifact"])
    artifact = replace(
        base,
        hinf_gains=selected["gains"],
        hinf_feasible=True,
        gamma_star=float(selected["gamma_star"]),
        hinf_diagnostics=selected["diagnostics"],
    )
    policy = build_policy("hinf", artifact, kp=0.0, ki=0.0, kd=0.0)
    return policy, {
        "source": _sha(HINF_SOURCE),
        "selected": _sha(HINF_SELECTED),
    }, selected["identity"]["parameters"]


def _generation_path(method: str) -> Path:
    return UNIT / "cache/generations" / f"{method}.json"


def generate(method: str, device: str) -> None:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    records = _records()
    destination = _generation_path(method)
    parameters = {}
    artifact_hashes = {}
    if method == "alqr":
        _policy, artifact_hashes, parameters = _alqr_policy(device)
        del _policy
    elif method == "h_infinity":
        _policy, artifact_hashes, parameters = _hinf_policy()
        del _policy
    identity = {
        "schema_version": 1,
        "method": method,
        "model": [MODEL_ID, MODEL_REVISION],
        "data_sha256": _sha(DATA),
        "prompt_ids": [row["prompt_id"] for row in records],
        "sample_count": SAMPLE_COUNT,
        "seed": SOURCE_RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "generation_use_cache": False,
        "parameters": parameters,
        "controller_artifacts_sha256": artifact_hashes,
        "implementation_sha256": _sha(Path(__file__)),
    }
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved.get("identity") != identity or saved.get("status") != "complete":
            raise ValueError(f"Cache-off generation identity changed: {destination}")
        return

    started_at = _utc_now()
    started = time.perf_counter()
    token = load_access_token(REPO)
    model_method = "alqr" if method in {"alqr", "h_infinity"} else "original"
    model, tokenizer = load_source_model(
        model_method,
        "truthfulness",
        MODEL_ID,
        MODEL_REVISION,
        device,
        token,
    )
    policy = None
    if method == "alqr":
        policy, _hashes, _parameters = _alqr_policy(device)
    elif method == "h_infinity":
        policy, _hashes, _parameters = _hinf_policy()
    completions = generate_batched(
        model,
        tokenizer,
        [row["text"] for row in records],
        behavior="truthfulness",
        batch_size=BATCH_SIZE,
        seed=SOURCE_RANDOM_SEED,
        use_cache=False,
        register_hooks=(
            None
            if policy is None
            else lambda: register_generation_policy_hooks(model, policy)
        ),
    )
    payload = {
        "identity": identity,
        "status": "complete",
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": runtime_provenance(device),
        "rows": [
            {
                "prompt_id": row["prompt_id"],
                "question": row["question"],
                "text": row["text"],
                "completion": completion,
            }
            for row, completion in zip(records, completions, strict=True)
        ],
    }
    _write_json(destination, payload)
    del policy, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def _load_judge(model_id: str, revision: str, device: str, token: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        token=token,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        token=token,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
        dtype=torch.float32,
        device_map={"": int(device.split(":", 1)[1])},
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def score_judge(judge: str, device: str) -> None:
    specifications = {
        "true": ("True", TRUTH_JUDGE, TRUTH_JUDGE_REVISION),
        "info": ("Helpful", INFO_JUDGE, INFO_JUDGE_REVISION),
    }
    label, model_id, revision = specifications[judge]
    generations = {method: json.loads(_generation_path(method).read_text()) for method in METHODS}
    identity = {
        "schema_version": 1,
        "judge": judge,
        "judge_model": [model_id, revision],
        "generation_sha256": {
            method: _sha(_generation_path(method)) for method in METHODS
        },
        "batch_size": JUDGE_BATCH_SIZE,
        "max_new_tokens": JUDGE_MAX_NEW_TOKENS,
        "generation_use_cache": True,
        "rubric": f"Q: {{question}}\\nA: {{answer}}\\n{label}:",
        "parser": "exact case-insensitive yes scores 1; every other output scores 0",
    }
    destination = UNIT / "cache/scores" / f"{judge}.json"
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved.get("identity") != identity or saved.get("status") != "complete":
            raise ValueError(f"Judge cache identity changed: {destination}")
        return
    joined = [
        (method, row)
        for method in METHODS
        for row in generations[method]["rows"]
    ]
    token = load_access_token(REPO)
    started = time.perf_counter()
    model, tokenizer = _load_judge(model_id, revision, device, token)
    output = {method: [] for method in METHODS}
    for start in range(0, len(joined), JUDGE_BATCH_SIZE):
        batch = joined[start : start + JUDGE_BATCH_SIZE]
        prompts = [
            truth_judge_prompt(row["question"], row["completion"], label)
            for _method, row in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=JUDGE_MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        token_rows = generated.sequences[:, encoded["input_ids"].shape[1] :].cpu().tolist()
        answers = tokenizer.batch_decode(token_rows, skip_special_tokens=True)
        for (method, row), prompt, answer, token_ids in zip(
            batch, prompts, answers, token_rows, strict=True
        ):
            raw_answer = answer.strip()
            score, valid = judge_label(raw_answer)
            output[method].append(
                {
                    "prompt_id": row["prompt_id"],
                    "judge_prompt": prompt,
                    "raw_answer": raw_answer,
                    "generated_token_ids": token_ids,
                    "score": score,
                    "valid": valid,
                }
            )
    _write_json(
        destination,
        {
            "identity": identity,
            "status": "complete",
            "elapsed_seconds": time.perf_counter() - started,
            "runtime": runtime_provenance(device),
            "methods": output,
        },
    )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def _metrics(rows: list[dict]) -> dict[str, float]:
    truth = 100.0 * sum(row["true"] for row in rows) / len(rows)
    info = 100.0 * sum(row["info"] for row in rows) / len(rows)
    return {"truth_x_info": truth * info / 100.0, "truth": truth, "info": info}


def summarize() -> None:
    reference = json.loads(REFERENCE.read_text())
    true = json.loads((UNIT / "cache/scores/true.json").read_text())
    info = json.loads((UNIT / "cache/scores/info.json").read_text())
    summaries = []
    matched_diagnostics = []
    for method in METHODS:
        on_rows = reference["methods"][method]
        off_generation = json.loads(_generation_path(method).read_text())["rows"]
        true_rows = true["methods"][method]
        info_rows = info["methods"][method]
        expected_ids = [row["prompt_id"] for row in on_rows]
        if not (
            [row["prompt_id"] for row in off_generation] == expected_ids
            == [row["prompt_id"] for row in true_rows]
            == [row["prompt_id"] for row in info_rows]
        ):
            raise ValueError(f"Cache comparison rows do not align for {method}")
        off_rows = [
            {"true": float(t["score"]), "info": float(i["score"])}
            for t, i in zip(true_rows, info_rows, strict=True)
        ]
        matched_diagnostics.append(
            {
                "method": method,
                "exact_completion_matches": sum(
                    on["completion"] == off["completion"]
                    for on, off in zip(on_rows, off_generation, strict=True)
                ),
                "truth_label_flips": sum(
                    float(on["true"]) != float(off["score"])
                    for on, off in zip(on_rows, true_rows, strict=True)
                ),
                "info_label_flips": sum(
                    float(on["info"]) != float(off["score"])
                    for on, off in zip(on_rows, info_rows, strict=True)
                ),
            }
        )
        summaries.extend(
            (
                {"method": method, "kv_cache": "on", "metrics": _metrics(on_rows)},
                {"method": method, "kv_cache": "off", "metrics": _metrics(off_rows)},
            )
        )
    result = {
        "schema_version": 1,
        "sample_count": SAMPLE_COUNT,
        "repetitions": 1,
        "selection": "first 100 prompts of Spanish repetition 0",
        "summaries": summaries,
        "matched_diagnostics": matched_diagnostics,
    }
    _write_json(RESULTS, result)
    labels = {"original": "Original", "alqr": "A-LQR", "h_infinity": "H∞"}
    lines = [
        "# KV cache investigation",
        "",
        "## Method",
        "",
        "Compare generation-time KV cache on versus off on the same 100 Spanish "
        "TruthfulQA prompts for Original, A-LQR, and H∞. The cache-on condition "
        "comes from repetition 0 of the completed benchmark; the cache-off condition "
        "is regenerated with every other setting held fixed. Both conditions use "
        "the same pinned True and Info judges with judge-side KV caching enabled.",
        "",
        "## Variables",
        "",
        "- Independent variable: generation-time KV-cache state.",
        "- Groups: Original, A-LQR, and H∞.",
        "- Outcomes: T×I, True percentage, Info percentage, exact answer matches, "
        "and matched judge-label flips.",
        "",
        "## Statistics",
        "",
        "Values are percentages from one matched 100-prompt run. There are no "
        "repetitions, standard errors, confidence intervals, or inferential tests.",
        "",
        "| Method | KV cache | T×I ↑ | True (%) ↑ | Info (%) ↑ |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summaries:
        metrics = row["metrics"]
        lines.append(
            f"| {labels[row['method']]} | {row['kv_cache']} | "
            f"{metrics['truth_x_info']:.2f} | {metrics['truth']:.2f} | "
            f"{metrics['info']:.2f} |"
        )
    lines.extend(
        (
            "",
            "Matched answer-level diagnostics:",
            "",
            "| Method | Exact answers (of 100) | True-label flips | Info-label flips |",
            "|---|---:|---:|---:|",
        )
    )
    for row in matched_diagnostics:
        lines.append(
            f"| {labels[row['method']]} | {row['exact_completion_matches']} | "
            f"{row['truth_label_flips']} | {row['info_label_flips']} |"
        )
    lines.extend(
        (
            "",
            "## Legends",
            "",
            "The output is tabular: rows group steering method and KV-cache state; "
            "higher T×I, True, and Info are better.",
            "",
            "## Interpretation",
            "",
            "Original is nearly invariant: 96/100 generated answers are exact matches. "
            "A-LQR and H∞ each have 0/100 exact matches, accompanied by substantial "
            "judge-label movement. In this hooked feedback pipeline, disabling KV cache "
            "therefore changes controller execution and is not merely a speed toggle.",
            "",
            "## Notes",
            "",
            "This diagnostic does not choose which cache state is mathematically correct. "
            "It establishes that the current controller hooks are cache-sensitive. It "
            "does not alter benchmark caches, benchmark results, or the benchmark table.",
            "",
            "## References",
            "",
            "- `benchmarks/truthfulness/`",
            "- `parking/truthfulqa_spanish/`",
        )
    )
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines) + "\n")


def _launch(arguments: list[str], log_name: str) -> tuple[subprocess.Popen, object]:
    log_path = UNIT / "cache/logs" / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a")
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), *arguments],
        cwd=REPO,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    return process, handle


def _wait(jobs: list[tuple[str, subprocess.Popen, object]]) -> None:
    failures = []
    for name, process, handle in jobs:
        return_code = process.wait()
        handle.close()
        if return_code:
            failures.append(name)
    if failures:
        raise RuntimeError(f"Investigation subprocesses failed: {failures}")


def run() -> None:
    prepare_reference()
    jobs = []
    for method, device in (("original", "cuda:0"), ("alqr", "cuda:1")):
        process, handle = _launch(
            ["--stage", "generate", "--method", method, "--device", device],
            f"generate_{method}.log",
        )
        jobs.append((method, process, handle))
    _wait(jobs)
    process, handle = _launch(
        [
            "--stage",
            "generate",
            "--method",
            "h_infinity",
            "--device",
            "cuda:0",
        ],
        "generate_h_infinity.log",
    )
    _wait([("h_infinity", process, handle)])
    jobs = []
    for judge, device in (("true", "cuda:0"), ("info", "cuda:1")):
        process, handle = _launch(
            ["--stage", "score-judge", "--judge", judge, "--device", device],
            f"score_{judge}.log",
        )
        jobs.append((judge, process, handle))
    _wait(jobs)
    summarize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("prepare", "generate", "score-judge", "summarize", "run"),
        required=True,
    )
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--judge", choices=("true", "info"))
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"))
    arguments = parser.parse_args()
    if arguments.stage == "prepare":
        prepare_reference()
    elif arguments.stage == "generate":
        if arguments.method is None or arguments.device is None:
            raise ValueError("generate requires --method and --device")
        generate(arguments.method, arguments.device)
    elif arguments.stage == "score-judge":
        if arguments.judge is None or arguments.device is None:
            raise ValueError("score-judge requires --judge and --device")
        score_judge(arguments.judge, arguments.device)
    elif arguments.stage == "summarize":
        summarize()
    else:
        run()


if __name__ == "__main__":
    main()
