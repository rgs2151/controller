"""Independent local scoring with the pinned TruthfulQA judges."""

from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from robust_steerability.benchmarks.metrics import judge_label, truth_judge_prompt
from robust_steerability.judges.specs import scorer_cache_path, scorer_spec
from robust_steerability.modeling.huggingface import cuda_device_index
from robust_steerability.source_methods.id_benchmark import runtime_provenance


BATCH_SIZE = 16
MAX_NEW_TOKENS = 20
INPUT_MAX_LENGTH = 1024


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def load_scorer(model_id: str, revision: str, device: str, token: str):
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


def score_batch(model, tokenizer, prompts: list[str], device: str) -> list[dict]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=INPUT_MAX_LENGTH,
    ).to(device)
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
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


def _scorer_record(scorer_key: str) -> dict:
    spec = scorer_spec(scorer_key)
    if spec.backend != "huggingface_binary" or spec.revision is None:
        raise ValueError(f"{scorer_key} is not a local binary scorer")
    return {
        "schema_version": 1,
        "scorer_key": scorer_key,
        "model": [spec.model_id, spec.revision],
        "compute_dtype": "bfloat16",
        "batch_size": BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "input_max_length": INPUT_MAX_LENGTH,
        "generation": {"do_sample": False, "use_cache": True},
        "quantization": {
            "format": "nf4",
            "double_quantization": True,
            "model_dtype": "float32",
            "compute_dtype": "bfloat16",
        },
        "rubric": spec.rubric,
        "answer_extraction": "generated_token_suffix",
        "answer_parser": (
            "strip and lowercase; exact yes scores 1, every other output scores 0; "
            "exact yes/no validity is retained for audit"
        ),
    }


def score_generation(
    generation_path: Path,
    root: Path,
    scorer_key: str,
    device: str,
    token: str,
) -> Path:
    spec = scorer_spec(scorer_key)
    if spec.backend != "huggingface_binary" or spec.revision is None:
        raise ValueError(f"{scorer_key} is not a local binary scorer")
    generation = json.loads(generation_path.read_text())
    if generation.get("status") != "complete":
        raise ValueError(f"Generation is incomplete: {generation_path}")
    rows = [
        row
        for repetition in generation["repetitions"]
        for row in repetition["rows"]
    ]
    destination = scorer_cache_path(root, generation_path, scorer_key)
    saved = {
        "scorer": _scorer_record(scorer_key),
        "status": "partial",
        "attempts": [],
        "rows": [],
    }
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved.get("status") == "complete":
            return destination

    attempt = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "starting_row": len(saved["rows"]),
        "runtime": runtime_provenance(device),
    }
    saved["attempts"].append(attempt)
    _write_json(destination, saved)
    started = time.perf_counter()
    model, tokenizer = load_scorer(
        spec.model_id, spec.revision, device, token
    )
    for start in range(len(saved["rows"]), len(rows), BATCH_SIZE):
        batch_rows = rows[start:start + BATCH_SIZE]
        prompts = [
            truth_judge_prompt(
                str(row["question"]),
                str(row["completion"]),
                str(spec.prompt_label),
            )
            for row in batch_rows
        ]
        scores = score_batch(model, tokenizer, prompts, device)
        saved["rows"].extend(
            {
                "prompt_id": row["prompt_id"],
                "judge_prompt": prompt,
                **score,
            }
            for row, prompt, score in zip(batch_rows, prompts, scores, strict=True)
        )
        if len(saved["rows"]) % (20 * BATCH_SIZE) == 0 or len(saved["rows"]) == len(rows):
            _write_json(destination, saved)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    attempt["status"] = "complete"
    attempt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    attempt["elapsed_seconds"] = time.perf_counter() - started
    saved["status"] = "complete"
    _write_json(destination, saved)
    return destination
