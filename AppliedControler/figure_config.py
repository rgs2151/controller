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
    "Qwen-2.5-1.5B-ours",
    "Qwen-2.5-1.5B-paper-proto",
    "Qwen-2.5-7B-ours",
    "Qwen-2.5-14B-ours",
]
TRUTH_MODEL_LABELS = {
    "DistilGPT-2-ours": "DistilGPT-2",
    "Qwen-2.5-1.5B-ours": "Qwen-2.5-1.5B",
    "Qwen-2.5-1.5B-paper-proto": "Qwen-2.5-1.5B\n(paper protocol)",
    "Qwen-2.5-7B-ours": "Qwen-2.5-7B",
    "Qwen-2.5-14B-ours": "Qwen-2.5-14B",
}

# ---- Toxicity / OOD figures (keys match label column of steering CSVs) ----
TOX_MODEL_ORDER = [
    "DistilGPT-2",
    "Qwen-2.5-1.5B",
    "Qwen-2.5-7B",
    "Qwen-2.5-14B",
]
TOX_MODEL_LABELS = {name: name for name in TOX_MODEL_ORDER}

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
