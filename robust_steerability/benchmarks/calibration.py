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
    multiple_choice = load_dataset(TRUTHFULQA_ID, "multiple_choice", split="validation", revision=TRUTHFULQA_REVISION)
    question_indices = {row["question"]: index for index, row in enumerate(dataset)}
    pools = {0: [], 1: []}
    unmatched_questions = []
    for row in multiple_choice:
        if row["question"] not in question_indices:
            unmatched_questions.append(row["question"])
            continue
        index = question_indices[row["question"]]
        targets = row["mc2_targets"]
        for answer_index, (answer, label) in enumerate(zip(targets["choices"], targets["labels"], strict=True)):
            pools[label].append({"prompt_id": f"truthfulqa:{index}:mc2:{answer_index}",
                                 "source_prompt_id": f"truthfulqa:{index}",
                                 "text": f"Q: {row['question']} A: {answer}"})
    negative, positive = rng.sample(pools[0], count), rng.sample(pools[1], count)
    excluded = {row["source_prompt_id"] for row in negative + positive}
    indices = rng.sample([index for index in range(len(dataset))
                          if f"truthfulqa:{index}" not in excluded], scale_count)
    scale = [{"prompt_id": f"truthfulqa:{index}", "text": f"Q: {dataset[index]['question']} A:"}
             for index in indices]
    return negative, positive, scale, {
        "id": TRUTHFULQA_ID, "revision": TRUTHFULQA_REVISION,
        "alignment": "multiple_choice joined to generation by exact question text",
        "unmatched_multiple_choice_questions": unmatched_questions,
    }
