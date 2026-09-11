"""Run the paper benchmark one model, method, and dataset at a time."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from robust_steerability.benchmarks.metrics import judge_label, truth_judge_prompt
from robust_steerability.modeling.huggingface import cuda_device_index, load_access_token
from robust_steerability.source_methods.id_benchmark import run_generation_job
from robust_steerability.source_methods.protocol import (
    ALQR_SWEEPS,
    CALIBRATION_COUNTS,
    SOURCE_RANDOM_SEED,
)


UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[1]
MODEL_KEY = "gemma2b"
MODEL_ID = "google/gemma-2-2b"
MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
TRUTHFULQA_ID = "truthful_qa"
TRUTHFULQA_REVISION = "741b8276f2d1982aa3d5b832d3ee81ed3b896490"
TRUTH_JUDGE = "allenai/truthfulqa-truth-judge-llama2-7B"
TRUTH_JUDGE_REVISION = "8f718cb4e68bc30a488727a0e37190835a7c129e"
INFO_JUDGE = "allenai/truthfulqa-info-judge-llama2-7B"
INFO_JUDGE_REVISION = "72601eff294cbf7e8d7188ae82397e19cb9bb9bd"
EVALUATION_REPETITIONS = 5
EVALUATION_SAMPLES = 817
JUDGE_BATCH_SIZE = 16
JUDGE_MAX_NEW_TOKENS = 20
METHODS = (
    "original",
    "iti",
    "actadd",
    "mean_act",
    "linear_act",
    "pid_act",
    "odesteer",
    "spid",
    "alqr",
    "h_infinity",
)
METHOD_LABELS = {
    "original": "Original",
    "iti": "ITI",
    "actadd": "ActAdd",
    "mean_act": "Mean-AcT",
    "linear_act": "Linear-AcT",
    "pid_act": "PID-AcT",
    "odesteer": "ODESteer",
    "spid": "S-PID",
    "alqr": "A-LQR",
    "h_infinity": "H∞ (ours)",
}
TABLE_MODELS = (
    ("gemma2b", "Gemma-2-2B"),
    ("llama8b", "Llama-3-8B"),
    ("qwen14b", "Qwen-2.5-14B"),
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample(records: list[dict], count: int, seed: int) -> list[dict]:
    if len(records) < count:
        raise ValueError(f"Requested {count} records from a pool of {len(records)}")
    indices = random.Random(seed).sample(range(len(records)), count)
    return [records[index] for index in indices]


def build_truthfulqa_data(generation_rows: list[dict], multiple_choice_rows: list[dict]) -> dict:
    """Create pinned calibration selections and five full evaluation permutations."""

    true_records: list[dict] = []
    false_records: list[dict] = []
    for row_index, row in enumerate(multiple_choice_rows):
        question = str(row["question"]).strip()
        targets = row["mc2_targets"]
        for answer_index, (answer, label) in enumerate(
            zip(targets["choices"], targets["labels"], strict=True)
        ):
            record = {
                "prompt_id": f"truthfulqa:mc2:{row_index}:{answer_index}",
                "question_id": f"truthfulqa:{row_index}",
                "text": f"Q: {question} A: {str(answer).strip()}",
            }
            (true_records if int(label) == 1 else false_records).append(record)

    evaluation = [
        {
            "prompt_id": f"truthfulqa:{row_index}",
            "question": str(row["question"]).strip(),
            "text": f"Q: {str(row['question']).strip()} A:",
        }
        for row_index, row in enumerate(generation_rows)
    ]
    counts = CALIBRATION_COUNTS["truthfulness"]
    payload = {
        "schema_version": 1,
        "seed": SOURCE_RANDOM_SEED,
        "evaluation_repetitions": EVALUATION_REPETITIONS,
        "evaluation_samples": len(evaluation),
        "datasets": {
            "truthfulness": {
                "id": TRUTHFULQA_ID,
                "revision": TRUTHFULQA_REVISION,
                "split": "validation",
            }
        },
        "calibration": {
            "truthfulness": {
                "undesired": _sample(false_records, counts.negative, SOURCE_RANDOM_SEED),
                "desired": _sample(true_records, counts.positive, SOURCE_RANDOM_SEED + 1),
                "jacobian": _sample(true_records, counts.jacobian, SOURCE_RANDOM_SEED + 2),
            }
        },
        "evaluation": {
            "truthfulness": {
                str(repetition): _sample(
                    evaluation,
                    len(evaluation),
                    SOURCE_RANDOM_SEED + 100_000 * repetition,
                )
                for repetition in range(EVALUATION_REPETITIONS)
            }
        },
    }
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def prepare() -> None:
    destination = UNIT / "cache/data.json"
    if destination.exists():
        saved = json.loads(destination.read_text())
        if (
            saved.get("schema_version") != 1
            or saved.get("evaluation_samples") != EVALUATION_SAMPLES
            or saved.get("evaluation_repetitions") != EVALUATION_REPETITIONS
        ):
            raise ValueError(f"Dataset cache does not match the benchmark protocol: {destination}")
        return
    generation = list(
        load_dataset(
            TRUTHFULQA_ID,
            "generation",
            split="validation",
            revision=TRUTHFULQA_REVISION,
        )
    )
    multiple_choice = list(
        load_dataset(
            TRUTHFULQA_ID,
            "multiple_choice",
            split="validation",
            revision=TRUTHFULQA_REVISION,
        )
    )
    if len(generation) != EVALUATION_SAMPLES:
        raise ValueError(f"Expected {EVALUATION_SAMPLES} TruthfulQA questions; found {len(generation)}")
    _write_json(destination, build_truthfulqa_data(generation, multiple_choice))


def generate(method: str, device: str) -> None:
    if method not in {"original", "alqr"}:
        raise ValueError("The first benchmark slice permits only original or alqr")
    if not (UNIT / "cache/data.json").exists():
        raise ValueError("Run the prepare stage before generation")
    run_generation_job(
        unit=UNIT,
        behavior="truthfulness",
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        method=method,
        device=device,
        token=load_access_token(REPO),
    )


def _generation_files(method: str) -> list[Path]:
    root = UNIT / "cache/generations/truthfulness" / MODEL_KEY / method
    return sorted(path for path in root.glob("*.json") if path.name != "unsupported.json")


def _score_path(generation_path: Path) -> Path:
    relative = generation_path.relative_to(UNIT / "cache/generations")
    return UNIT / "cache/scores" / relative


def _flatten_generation(payload: dict) -> list[dict]:
    return [row for repetition in payload["repetitions"] for row in repetition["rows"]]


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
        device_map={"": cuda_device_index(device)},
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def _score_identity(generation_path: Path) -> dict:
    return {
        "schema_version": 1,
        "generation_sha256": _sha(generation_path),
        "truth_judge": [TRUTH_JUDGE, TRUTH_JUDGE_REVISION],
        "info_judge": [INFO_JUDGE, INFO_JUDGE_REVISION],
        "compute_dtype": "bfloat16",
        "batch_size": JUDGE_BATCH_SIZE,
        "max_new_tokens": JUDGE_MAX_NEW_TOKENS,
        "answer_extraction": "generated_token_suffix",
    }


def _judge_batch(model, tokenizer, prompts: list[str], device: str) -> list[dict]:
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
    token_rows = generated.sequences[:, encoded["input_ids"].shape[1]:].cpu().tolist()
    answers = tokenizer.batch_decode(token_rows, skip_special_tokens=True)
    output = []
    for answer, token_ids in zip(answers, token_rows, strict=True):
        raw_answer = answer.strip()
        value, valid = judge_label(raw_answer)
        output.append(
            {
                "raw_answer": raw_answer,
                "generated_token_ids": token_ids,
                "score": value,
                "valid": valid,
            }
        )
    return output


def _score_generation(generation_path: Path, device: str, token: str) -> None:
    destination = _score_path(generation_path)
    identity = _score_identity(generation_path)
    generation_payload = json.loads(generation_path.read_text())
    if generation_payload["status"] != "complete":
        raise ValueError(f"Generation is incomplete: {generation_path}")
    generation_rows = _flatten_generation(generation_payload)
    total = len(generation_rows)
    saved = {
        "identity": identity,
        "status": "partial",
        "true": [],
        "helpful": [],
    }
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved["identity"] != identity:
            raise ValueError(f"Judge cache mismatch: {destination}")
        if saved["status"] == "complete":
            if len(saved["true"]) != total or len(saved["helpful"]) != total:
                raise ValueError(f"Incomplete judge cache marked complete: {destination}")
            return
        if saved["status"] != "partial":
            raise ValueError(f"Unknown judge cache status: {destination}")

    for key, label, judge_id, revision in (
        ("true", "True", TRUTH_JUDGE, TRUTH_JUDGE_REVISION),
        ("helpful", "Helpful", INFO_JUDGE, INFO_JUDGE_REVISION),
    ):
        if len(saved[key]) > total:
            raise ValueError(f"Judge cache has too many rows: {destination}")
        model, tokenizer = _load_judge(judge_id, revision, device, token)
        for start in range(len(saved[key]), total, JUDGE_BATCH_SIZE):
            batch_rows = generation_rows[start:start + JUDGE_BATCH_SIZE]
            prompts = [
                truth_judge_prompt(str(row["question"]), str(row["completion"]), label)
                for row in batch_rows
            ]
            batch_scores = _judge_batch(model, tokenizer, prompts, device)
            saved[key].extend(
                {
                    "prompt_id": row["prompt_id"],
                    "judge_prompt": prompt,
                    **score,
                }
                for row, prompt, score in zip(batch_rows, prompts, batch_scores, strict=True)
            )
            if len(saved[key]) % (20 * JUDGE_BATCH_SIZE) == 0 or len(saved[key]) == total:
                _write_json(destination, saved)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    saved["status"] = "complete"
    _write_json(destination, saved)


def score(method: str, device: str) -> None:
    files = _generation_files(method)
    if len(files) != 1:
        raise ValueError(f"Expected one {method} generation cache; found {len(files)}")
    _score_generation(files[0], device, load_access_token(REPO))


def _mean_se(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1) / math.sqrt(len(array)))


def summarize(method: str) -> dict:
    files = _generation_files(method)
    if len(files) != 1:
        raise ValueError(f"Expected one {method} generation cache; found {len(files)}")
    generation_path = files[0]
    score_path = _score_path(generation_path)
    if not score_path.exists():
        raise ValueError(f"Missing judge cache: {score_path}")
    generation = json.loads(generation_path.read_text())
    scores = json.loads(score_path.read_text())
    if generation["status"] != "complete" or scores["status"] != "complete":
        raise ValueError(f"Cannot summarize incomplete caches for {method}")
    generation_rows = _flatten_generation(generation)
    for key in ("true", "helpful"):
        if [row["prompt_id"] for row in scores[key]] != [row["prompt_id"] for row in generation_rows]:
            raise ValueError(f"{key} judge rows do not align with generation rows")
        invalid = [row for row in scores[key] if not row["valid"]]
        if invalid:
            raise ValueError(f"{len(invalid)} malformed {key} judge answers require inspection")

    offset = 0
    per_repetition = []
    for repetition in generation["repetitions"]:
        count = len(repetition["rows"])
        truth = 100.0 * float(np.mean([row["score"] for row in scores["true"][offset:offset + count]]))
        info = 100.0 * float(np.mean([row["score"] for row in scores["helpful"][offset:offset + count]]))
        per_repetition.append(
            {
                "repetition": repetition["repetition"],
                "truth": truth,
                "info": info,
                "truth_x_info": truth * info / 100.0,
            }
        )
        offset += count
    metrics = {}
    for output_name, key in (("truth_x_info", "truth_x_info"), ("truth", "truth"), ("info", "info")):
        mean, standard_error = _mean_se([row[key] for row in per_repetition])
        metrics[output_name] = {"mean": mean, "standard_error": standard_error}
    result = {
        "identity": {
            "generation_sha256": _sha(generation_path),
            "scores_sha256": _sha(score_path),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "method": method,
            "dataset": [TRUTHFULQA_ID, TRUTHFULQA_REVISION],
        },
        "evaluation_samples_per_repetition": EVALUATION_SAMPLES,
        "evaluation_repetitions": EVALUATION_REPETITIONS,
        "per_repetition": per_repetition,
        "metrics": metrics,
    }
    _write_json(UNIT / "cache/results/truthfulness" / MODEL_KEY / f"{method}.json", result)
    return result


def _cell(result: dict | None, metric: str) -> str:
    if result is None:
        return "TBD"
    value = result["metrics"][metric]
    return f"{value['mean']:.2f} ± {value['standard_error']:.2f}"


def markdown_table(results: dict[tuple[str, str], dict]) -> str:
    lines = [
        "# Truthfulness benchmark",
        "",
        "| Model | Method | TruthfulQA–ID T×I ↑ | Spanish | Adversarial | Long | True (%) ↑ | Info (%) ↑ | MMLU (%) ↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_key, model_label in TABLE_MODELS:
        for method in METHODS:
            result = results.get((model_key, method))
            lines.append(
                f"| {model_label} | {METHOD_LABELS[method]} | {_cell(result, 'truth_x_info')} | "
                f"TBD | TBD | TBD | {_cell(result, 'truth')} | {_cell(result, 'info')} | TBD |"
            )
    lines.extend(
        [
            "",
            "Values are mean ± SE across five complete 817-question repetitions. TBD cells have not been run.",
            "",
        ]
    )
    return "\n".join(lines)


def render() -> None:
    results = {}
    root = UNIT / "cache/results/truthfulness"
    for model_key, _model_label in TABLE_MODELS:
        for method in METHODS:
            path = root / model_key / f"{method}.json"
            if path.exists():
                results[(model_key, method)] = json.loads(path.read_text())
    destination = UNIT / "plots/benchmark_table.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markdown_table(results))


def launch_pair(stage: str) -> None:
    log_root = UNIT / "cache/logs"
    log_root.mkdir(parents=True, exist_ok=True)
    running = []
    for method, device in (("original", "cuda:0"), ("alqr", "cuda:1")):
        log_path = log_root / f"{stage}_{MODEL_KEY}_{method}.log"
        handle = log_path.open("a")
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--stage",
                stage,
                "--method",
                method,
                "--device",
                device,
            ],
            cwd=REPO,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        running.append((method, process, handle))
    failures = []
    for method, process, handle in running:
        return_code = process.wait()
        handle.close()
        if return_code:
            failures.append((method, return_code))
    if failures:
        raise RuntimeError(f"Pair stage failed: {failures}")


def smoke() -> None:
    counts = CALIBRATION_COUNTS["truthfulness"]
    sweep = ALQR_SWEEPS["truthfulness"][MODEL_KEY]
    if counts.__dict__ != {
        "negative": 12,
        "positive": 12,
        "jacobian": 1,
        "jacobian_class": "positive",
        "jacobian_max_length": 24,
    }:
        raise ValueError("Truthfulness calibration no longer matches the official refactored source")
    if sweep.__dict__ != {"lambdas": (3.0,), "q": 0.1, "r": 1.0, "q_final": 0.3}:
        raise ValueError("Gemma-2-2B A-LQR parameters no longer match the official refactored source")
    if truth_judge_prompt("Question?", "Answer.", "True") != "Q: Question?\nA: Answer.\nTrue:":
        raise ValueError("Truth judge rubric changed")
    if truth_judge_prompt("Question?", "Answer.", "Helpful") != "Q: Question?\nA: Answer.\nHelpful:":
        raise ValueError("Info judge rubric changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("prepare", "generate", "score", "summarize", "render", "generate-pair", "score-pair", "smoke"),
        required=True,
    )
    parser.add_argument("--method", choices=("original", "alqr"))
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"))
    arguments = parser.parse_args()
    if arguments.stage == "prepare":
        prepare()
    elif arguments.stage in {"generate", "score"}:
        if arguments.method is None or arguments.device is None:
            raise ValueError(f"{arguments.stage} requires --method and --device")
        (generate if arguments.stage == "generate" else score)(arguments.method, arguments.device)
    elif arguments.stage == "summarize":
        if arguments.method is None:
            raise ValueError("summarize requires --method")
        summarize(arguments.method)
    elif arguments.stage == "render":
        render()
    elif arguments.stage in {"generate-pair", "score-pair"}:
        launch_pair(arguments.stage.removesuffix("-pair"))
    else:
        smoke()


if __name__ == "__main__":
    main()
