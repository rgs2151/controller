"""Cache ID continuation perplexity under a fixed unsteered external model."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from robust_steerability.artifacts import configuration_hash
from robust_steerability.experiments.diagnostics import read_json, sha256, write_json
from robust_steerability.modeling.huggingface import CausalModelLoadSpec, load_access_token, load_causal_model


UNIT = Path(__file__).resolve().parent
SCORER = "mistralai/Mistral-7B-v0.1"
REVISION = "27d67f1b5f57dc0953326b2601d68371d40ea8da"


def distinct_two(texts):
    pairs = []
    for text in texts:
        words = text.lower().split()
        pairs.extend(zip(words[:-1], words[1:]))
    return len(set(pairs)) / len(pairs) if pairs else None


def continuation_input(tokenizer, prompt, completion, capacity):
    """Keep every completion token; only truncate prompt context from the left."""
    # Separate tokenization makes the prompt/continuation boundary unambiguous.
    context = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    target = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if not target:
        return context[-capacity:], 0
    if len(target) >= capacity:
        raise ValueError("Completion alone exceeds perplexity scorer context")
    context = context[-(capacity - len(target)):]
    if not context:
        raise ValueError("Conditional perplexity requires at least one context token")
    return context + target, len(target)


def score(device):
    manifest = read_json(UNIT / "toxicity.json")
    expected = ["".join(c.lower() if c.isalnum() else "_" for c in row["label"]).strip("_")
                for row in manifest["models"]]
    inputs = [UNIT / "cache/jobs/toxicity" / model / "completions.json" for model in expected]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"Benchmark generation must finish first: {path}")
        payload = read_json(path)
        if set(payload["subsets"]["rtp_id"]) != set(manifest["methods"]):
            raise ValueError(f"Incomplete ID methods: {path}")
    model = tokenizer = None
    summary = []
    for path in inputs:
        payload = read_json(path)
        for method, records in payload["subsets"]["rtp_id"].items():
            if len(records) != 50:
                raise ValueError("This unit requires exactly 50 evaluation prompts")
            folder = UNIT / "cache/quality" / path.parent.name / method
            folder.mkdir(parents=True, exist_ok=True)
            values = []
            for row in records:
                key = configuration_hash({"prompt": row["prompt"], "completion": row["completion"],
                                          "model": SCORER, "revision": REVISION,
                                          "protocol": "separate-tokenization-conditional-v1"})
                checkpoint = folder / (key + ".json")
                if checkpoint.exists():
                    values.append(read_json(checkpoint))
                    continue
                if model is None:
                    model, tokenizer = load_causal_model(
                        CausalModelLoadSpec(SCORER, REVISION, quantized=True), device,
                        load_access_token(UNIT.parents[1]))
                tokens, target_count = continuation_input(
                    tokenizer, row["prompt"], row["completion"], int(model.config.max_position_embeddings))
                loss_values = []
                if target_count:
                    encoded = torch.tensor([tokens], device=device)
                    with torch.inference_mode():
                        logits = model(input_ids=encoded, attention_mask=torch.ones_like(encoded), use_cache=False).logits
                        losses = F.cross_entropy(logits[:, :-1].float().flatten(0, 1), encoded[:, 1:].flatten(), reduction="none")
                    loss_values = losses[-target_count:].cpu().tolist()
                mean_nll = float(np.mean(loss_values)) if loss_values else None
                record = {"prompt_id": row["prompt_id"], "key": key, "source_trace_sha256": row["trace_sha256"],
                          "input_token_ids": tokens, "target_token_count": target_count,
                          "token_nll": loss_values, "mean_nll": mean_nll,
                          "perplexity": None if mean_nll is None else math.exp(mean_nll),
                          "undefined_reason": "empty completion" if mean_nll is None else None,
                          "scorer": SCORER, "revision": REVISION, "quantized": True}
                write_json(checkpoint, record)
                values.append(record)
            ppls = [value["perplexity"] for value in values if value["perplexity"] is not None]
            summary.append({"model": payload["model"]["label"], "method": method, "sample_count": 50,
                            "ppl_count": len(ppls), "ppl_mean": float(np.mean(ppls)) if ppls else None,
                            "ppl_se": float(np.std(ppls, ddof=1) / len(ppls) ** 0.5) if len(ppls) > 1 else None,
                            "dist2": distinct_two([row["completion"] for row in records]),
                            "source": str(path.relative_to(UNIT)), "source_sha256": sha256(path)})
            write_json(UNIT / "cache/quality/summary.json", {"scorer": SCORER, "revision": REVISION, "rows": summary})
            print(f"Quality cached: {payload['model']['label']} {method}, {len(ppls)}/50 nonempty completions", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    score(parser.parse_args().device)
