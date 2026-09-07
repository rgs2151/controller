"""Reusable, deterministic prompt construction for steering benchmarks."""

from __future__ import annotations

import random

from robust_steerability.benchmarks.mmlu import load_mmlu_concept_shift_sets
from robust_steerability.benchmarks.toxicity import (
    load_civil_comments_prompts,
    load_jigsaw_toxicity_prompts,
    load_real_toxicity_prompt_pools,
    load_toxic_chat_prompts,
)


RTP_ID = "allenai/real-toxicity-prompts"
RTP_REVISION = "f21629712ffd6a3d13a54fd2807ccd521c55ef74"
JIGSAW_ID = "tcapelle/jigsaw-toxic-comment-classification-challenge"
JIGSAW_REVISION = "2bf801de1b879f287943ecfc81fdca8690d9fc61"
CIVIL_COMMENTS_ID = "google/civil_comments"
TOXIC_CHAT_ID = "lmsys/toxic-chat"
TOXIC_CHAT_REVISION = "29df8e4dba60e1f4af4b4075c0705c5b313548a8"
TOXIC_CHAT_CONFIG = "toxicchat0124"
MMLU_ID = "cais/mmlu"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"


def stable_sample(
    records: list[dict[str, object]],
    count: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    """Return a fixed-size sample without replacement."""

    if len(records) < count:
        raise ValueError(f"Requested {count} records from a pool of {len(records)}")
    return [records[index] for index in rng.sample(range(len(records)), count)]


def _ranked_subsets(
    records: list[dict[str, object]],
    prefix: str,
    count: int,
    rng: random.Random,
) -> dict[str, list[dict[str, object]]]:
    valid = [record for record in records if str(record["text"]).strip()]
    return {
        f"{prefix}_full": stable_sample(valid, count, rng),
        f"{prefix}_long": sorted(
            valid, key=lambda row: len(str(row["text"])), reverse=True
        )[:count],
        f"{prefix}_toxic": sorted(
            valid, key=lambda row: float(row["toxicity"]), reverse=True
        )[:count],
    }


def load_ood_prompt_sets(
    count: int,
    seed: int,
    families: tuple[str, ...] = ("jigsaw", "toxicchat", "mmlu"),
    mmlu_id_subject: str = "high_school_mathematics",
    mmlu_ood_subjects: int = 8,
) -> dict[str, list[dict[str, object]]]:
    """Build the ID and OOD prompt sets used by Erfan's benchmark matrix."""

    rng = random.Random(seed)
    output: dict[str, list[dict[str, object]]] = {}
    if any(name in families for name in ("jigsaw", "civil", "toxicchat")):
        all_rtp, _, _ = load_real_toxicity_prompt_pools(RTP_ID, RTP_REVISION)
        output["id_rtp"] = stable_sample(all_rtp, count, rng)
    if "jigsaw" in families:
        output.update(
            _ranked_subsets(
                load_jigsaw_toxicity_prompts(JIGSAW_ID, JIGSAW_REVISION),
                "jigsaw",
                count,
                rng,
            )
        )
    if "civil" in families:
        output.update(
            _ranked_subsets(
                load_civil_comments_prompts(CIVIL_COMMENTS_ID, split="train"),
                "civil",
                count,
                rng,
            )
        )
    if "toxicchat" in families:
        output.update(
            _ranked_subsets(
                load_toxic_chat_prompts(
                    TOXIC_CHAT_ID,
                    revision=TOXIC_CHAT_REVISION,
                    config_name=TOXIC_CHAT_CONFIG,
                    split="test",
                ),
                "toxicchat",
                count,
                rng,
            )
        )
    if "mmlu" in families:
        output.update(
            load_mmlu_concept_shift_sets(
                dataset_id=MMLU_ID,
                revision=MMLU_REVISION,
                id_subject=mmlu_id_subject,
                num_prompts=count,
                rng=rng,
                ood_subject_count=mmlu_ood_subjects,
            )
        )
    if not output:
        raise ValueError("At least one benchmark family is required")
    return output
