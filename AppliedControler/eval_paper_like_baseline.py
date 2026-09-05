"""Estimate Original (unsteered) toxicity under paper-like RTP sampling."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from robust_steerability.benchmarks.toxicity import (
    load_real_toxicity_prompt_pools,
    toxicity_probabilities,
)
from robust_steerability.modeling.huggingface import load_access_token, load_sequence_classifier

RTP_ID = "allenai/real-toxicity-prompts"
RTP_REVISION = "f21629712ffd6a3d13a54fd2807ccd521c55ef74"
TOX_CLASSIFIER_ID = "s-nlp/roberta_toxicity_classifier"
TOX_CLASSIFIER_REV = "main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["distilgpt2", "gpt2", "gpt2-medium"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-prompts", type=int, default=80)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2151)
    parser.add_argument(
        "--prompt-pool",
        choices=["rtp_random", "rtp_nontoxic"],
        default="rtp_random",
        help="rtp_random matches paper setup; rtp_nontoxic is a fallback calibration pool.",
    )
    parser.add_argument(
        "--prompt-toxicity-min",
        type=float,
        default=0.0,
        help="Minimum RTP prompt toxicity filter (applies to rtp_random).",
    )
    parser.add_argument(
        "--prompt-toxicity-max",
        type=float,
        default=1.0,
        help="Maximum RTP prompt toxicity filter (applies to rtp_random).",
    )
    parser.add_argument("--output-csv", type=Path, default=Path("AppliedControler/baseline_calibration.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("AppliedControler/baseline_calibration.json"))
    parser.add_argument("--hf-token", default="")
    return parser.parse_args()


def sample_prompts(
    pool_name: str,
    count: int,
    seed: int,
    toxicity_min: float,
    toxicity_max: float,
) -> list[dict[str, object]]:
    all_records, _, nontoxic_records = load_real_toxicity_prompt_pools(RTP_ID, RTP_REVISION)
    if pool_name == "rtp_random":
        pool = [
            row
            for row in all_records
            if toxicity_min <= float(row["toxicity"]) <= toxicity_max
        ]
    else:
        pool = nontoxic_records
    if not pool:
        raise ValueError(
            f"No prompts in selected pool={pool_name} for toxicity range "
            f"[{toxicity_min}, {toxicity_max}]"
        )
    rng = random.Random(seed)
    if len(pool) <= count:
        return pool
    return [pool[idx] for idx in rng.sample(range(len(pool)), count)]


def generate_texts(
    model,
    tokenizer,
    prompts: list[str],
    max_length: int,
    max_new_tokens: int,
) -> list[str]:
    out: list[str] = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        encoded = {k: v.to(next(model.parameters()).device) for k, v in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=True,
                top_p=0.3,
                temperature=1.0,
                repetition_penalty=1.2,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        out.append(tokenizer.decode(generated[0], skip_special_tokens=True))
    return out


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    hf_token = args.hf_token.strip()
    if not hf_token:
        try:
            hf_token = load_access_token(repo_root)
        except RuntimeError:
            hf_token = ""

    prompts = sample_prompts(
        args.prompt_pool,
        args.eval_prompts,
        args.seed,
        args.prompt_toxicity_min,
        args.prompt_toxicity_max,
    )
    prompt_texts = [str(row["text"]) for row in prompts]

    classifier, classifier_tokenizer = load_sequence_classifier(
        model_id=TOX_CLASSIFIER_ID,
        revision=TOX_CLASSIFIER_REV,
        device=args.device,
        token=hf_token,
    )

    rows: list[dict[str, object]] = []
    for model_name in args.models:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token=hf_token)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, token=hf_token).to(args.device)
        model.eval()

        texts = generate_texts(
            model=model,
            tokenizer=tokenizer,
            prompts=prompt_texts,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
        )
        probs = toxicity_probabilities(
            texts,
            model=classifier,
            tokenizer=classifier_tokenizer,
            device=args.device,
        )
        mean_tox = float(np.mean(probs))
        std_tox = float(np.std(probs))
        rows.append(
            {
                "model_name": model_name,
                "prompt_pool": args.prompt_pool,
                "eval_prompts": len(prompt_texts),
                "max_new_tokens": args.max_new_tokens,
                "prompt_toxicity_min": args.prompt_toxicity_min,
                "prompt_toxicity_max": args.prompt_toxicity_max,
                "baseline_toxicity_mean": mean_tox,
                "baseline_toxicity_percent": 100.0 * mean_tox,
                "baseline_toxicity_std": std_tox,
            }
        )

        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_name",
                "prompt_pool",
                "eval_prompts",
                "max_new_tokens",
                "prompt_toxicity_min",
                "prompt_toxicity_max",
                "baseline_toxicity_mean",
                "baseline_toxicity_percent",
                "baseline_toxicity_std",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    args.output_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    for row in rows:
        print(
            f"{row['model_name']}: baseline={row['baseline_toxicity_percent']:.2f}% "
            f"(pool={row['prompt_pool']}, prompts={row['eval_prompts']}, "
            f"range=[{row['prompt_toxicity_min']}, {row['prompt_toxicity_max']}])"
        )


if __name__ == "__main__":
    main()
