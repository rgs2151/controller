"""Disjoint behavior-specific fit and calibration records."""

import random

from datasets import load_dataset

from robust_steerability.benchmarks.ood import RTP_ID, RTP_REVISION, stable_sample
from robust_steerability.benchmarks.toxicity import load_real_toxicity_prompt_pools
from robust_steerability.benchmarks.truthfulness import TRUTHFULQA_ID, TRUTHFULQA_REVISION


def calibration_records(behavior: str, count: int, scale_count: int, seed: int):
    rng = random.Random(seed)
    if behavior == "toxicity_mitigation":
        pool, negative, positive = load_real_toxicity_prompt_pools(RTP_ID, RTP_REVISION)
        negative = stable_sample(negative, count, rng)
        positive = stable_sample(positive, count, rng)
        excluded = {row["prompt_id"] for row in negative + positive}
        scale = stable_sample([row for row in pool if row["prompt_id"] not in excluded], scale_count, rng)
        return negative, positive, scale, {"id": RTP_ID, "revision": RTP_REVISION}
    if behavior != "truthfulness":
        raise ValueError(f"Unknown calibration behavior: {behavior}")
    dataset = load_dataset(TRUTHFULQA_ID, "generation", split="validation", revision=TRUTHFULQA_REVISION)
    indices = rng.sample(range(len(dataset)), count + scale_count)
    negative, positive = [], []
    for index in indices[:count]:
        row = dataset[index]
        if not row["incorrect_answers"] or not row["correct_answers"]:
            raise ValueError(f"TruthfulQA row {index} has no contrastive answers")
        for group, answer, label in (
            (negative, row["incorrect_answers"][0], "incorrect"),
            (positive, row["best_answer"], "correct"),
        ):
            group.append({"prompt_id": f"truthfulqa:{index}:{label}",
                          "source_prompt_id": f"truthfulqa:{index}",
                          "text": f"Q: {row['question']} A: {answer}"})
    scale = [{"prompt_id": f"truthfulqa:{index}", "text": f"Q: {dataset[index]['question']} A:"}
             for index in indices[count:]]
    return negative, positive, scale, {"id": TRUTHFULQA_ID, "revision": TRUTHFULQA_REVISION}
