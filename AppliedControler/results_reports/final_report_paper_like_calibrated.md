# Steering Results Summary

## Method Ranking (lower mean delta is better)

| Rank | Method | N | Mean Delta | 95% CI | Improvement Rate |
|---:|---|---:|---:|---:|---:|
| 1 | H-infinity | 1 | -0.044349 | 0.000000 | 100.0% |
| 2 | S-PID | 1 | -0.044132 | 0.000000 | 100.0% |
| 3 | A-LQR | 1 | -0.043196 | 0.000000 | 100.0% |

## Target Winner Counts

| Method | Wins |
|---|---:|
| A-LQR | 0 |
| S-PID | 0 |
| H-infinity | 1 |

## Row Status Counts

| Status | Count |
|---|---:|
| ok | 3 |

## Truthfulness Artifact Status

- Canonical truthfulness CSV: [paper_style_table_truthfulness_ours_plus_qwen14b_methods.csv](paper_style_table_truthfulness_ours_plus_qwen14b_methods.csv)
- Canonical truthfulness LaTeX: [table2_style_truthfulness_ours_plus_qwen14b_methods.tex](table2_style_truthfulness_ours_plus_qwen14b_methods.tex)
- Canonical truthfulness PDF: [table2_style_truthfulness_ours_plus_qwen14b_methods.pdf](table2_style_truthfulness_ours_plus_qwen14b_methods.pdf)

All artifacts were regenerated from scratch (2026-09-05) via the checkpointed pipeline [AppliedControler/rerun_truthfulness_pipeline.py](../rerun_truthfulness_pipeline.py) after replacing the H-infinity core in [robust_steerability/control/h_infinity.py](../../robust_steerability/control/h_infinity.py). Per-stage logs and status JSON are in [truthfulness_rerun_runs/](truthfulness_rerun_runs/).

Stage outcomes (7/8 completed):
- DistilGPT-2: Original, A-LQR, S-PID, H-infinity — all completed (the new H-infinity core synthesizes and steers successfully at this scale).
- Qwen-2.5-14B (4-bit NF4): Original, A-LQR, S-PID — completed with CPU controller synthesis and reduced budgets.
- Qwen-2.5-14B H-infinity — reported as "--" in the canonical table. The finite-horizon gain recursion over the 5120-dim hidden state exceeds available host RAM (>84 GB observed) even at horizon 24 with CPU synthesis; runs were bounded by a 60 GB memory cap and 900 s timeout and could not complete. This is a hardware limit, not a code failure (the same core succeeds on DistilGPT-2).
