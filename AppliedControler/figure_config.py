"""Shared configuration for all AppliedControler summary figures.

Adding a new model = append to the relevant *_MODEL_ORDER / *_MODEL_LABELS
here; every figure script picks it up automatically (missing data renders
as red rotated "N/A" markers).
"""

from __future__ import annotations

METHODS = ["Original", "A-LQR", "S-PID", "H-infinity"]
METHOD_COLORS = {
    "Original": "#8c8c8c",
    "A-LQR": "#4c72b0",
    "S-PID": "#dd8452",
    "H-infinity": "#55a868",
}
NA_COLOR = "#b30000"

# ---- Truthfulness figures (keys match paper_style_table_truthfulness_*.csv) ----
TRUTH_MODEL_ORDER = [
    "DistilGPT-2-ours",
    "Qwen-2.5-0.5B-smoke",
    "Qwen-2.5-1.5B-ours",
    "Qwen-2.5-1.5B-paper-proto",
    "Qwen-2.5-7B-ours",
    "Qwen-2.5-14B-ours",
]
TRUTH_MODEL_LABELS = {
    "DistilGPT-2-ours": "DistilGPT-2",
    "Qwen-2.5-0.5B-smoke": "Qwen-2.5-0.5B\n(smoke)",
    "Qwen-2.5-1.5B-ours": "Qwen-2.5-1.5B",
    "Qwen-2.5-1.5B-paper-proto": "Qwen-2.5-1.5B\n(paper protocol)",
    "Qwen-2.5-7B-ours": "Qwen-2.5-7B",
    "Qwen-2.5-14B-ours": "Qwen-2.5-14B",
}

# ---- Toxicity / OOD figures (keys match label column of steering CSVs) ----
TOX_MODEL_ORDER = [
    "DistilGPT-2",
    "Qwen-2.5-0.5B",
    "Qwen-2.5-1.5B",
    "Qwen-2.5-7B",
    "Qwen-2.5-14B",
]
TOX_MODEL_LABELS = {name: name for name in TOX_MODEL_ORDER}
# Qwen-0.5B data is smoke-level for now; annotate its label.
TOX_MODEL_LABELS["Qwen-2.5-0.5B"] = "Qwen-2.5-0.5B\n(smoke)"

# ---- Global top-3 OOD benchmarks (from residual analysis, fixed across models) ----
OOD_SUBSETS = ["jigsaw_long", "toxicchat_long", "mmlu_ood_other_concepts"]
OOD_SUBSET_LABELS = {
    "jigsaw_long": "Jigsaw (long)",
    "toxicchat_long": "ToxicChat (long)",
    "mmlu_ood_other_concepts": "MMLU OOD concepts",
}

# Steering CSV method keys -> display names.
STEERING_METHOD_LABELS = {
    "alqr": "A-LQR",
    "spid": "S-PID",
    "new_method": "H-infinity",
}

# ---- Run completeness -> model label color ----
# Models with only smoke-test or missing data get gray labels.
COMPLETE_LABEL_COLOR = "black"
INCOMPLETE_LABEL_COLOR = "#999999"

# Truthfulness (full paper-aligned protocol done)
TRUTH_FULLY_RUN = {
    "DistilGPT-2-ours",
    "Qwen-2.5-1.5B-ours",
    "Qwen-2.5-1.5B-paper-proto",
    "Qwen-2.5-7B-ours",
    "Qwen-2.5-14B-ours",
}
# In-distribution RTP toxicity (full calibrated run done)
TOX_FULLY_RUN = {"DistilGPT-2"}
# OOD benchmarks: all data so far is smoke-level
OOD_FULLY_RUN: set[str] = set()


def model_label_color(model: str, fully_run: set[str]) -> str:
    return COMPLETE_LABEL_COLOR if model in fully_run else INCOMPLETE_LABEL_COLOR
