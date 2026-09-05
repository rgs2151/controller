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

- Canonical truthfulness CSV (all models): [paper_style_table_truthfulness_all_models_methods.csv](paper_style_table_truthfulness_all_models_methods.csv)
- Canonical truthfulness LaTeX: [table2_style_truthfulness_all_models_methods.tex](table2_style_truthfulness_all_models_methods.tex)
- Canonical truthfulness PDF: [table2_style_truthfulness_all_models_methods.pdf](table2_style_truthfulness_all_models_methods.pdf)
- Final summary figure: [final_summary_figure.png](final_summary_figure.png) / [final_summary_figure.pdf](final_summary_figure.pdf) (Panel A: T*I across model scales; Panel B: RTP toxicity per method, log scale)
- Previous two-model table (superseded): [paper_style_table_truthfulness_ours_plus_qwen14b_methods.csv](paper_style_table_truthfulness_ours_plus_qwen14b_methods.csv)

Run-status notes (see `run_status` column in the canonical CSV):
- DistilGPT-2 (all 4 methods) and Qwen-2.5-7B (Original/A-LQR/S-PID): `completed`, generated with bfloat16-safe loading.
- Qwen-2.5-7B and 14B H-infinity: `infeasible_ram` — the finite-horizon gain recursion exceeds host RAM/time budgets (>84 GB at 14B; 7B timed out at 30 min under a 60 GB cap). Hardware limit, not a code failure: the same core succeeds on DistilGPT-2.
- Qwen-2.5-14B (Original/A-LQR/S-PID): `completed_fp16_caveat` — these rows were produced before a float16-overflow bug in the quantized loader was found (Qwen2.5 in fp16 emits degenerate text). The loader now uses bfloat16; these rows should be regenerated before publication.
