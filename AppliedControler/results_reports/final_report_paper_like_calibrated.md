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
- DistilGPT-2 (all 4 methods), Qwen-2.5-1.5B (all 4 methods), and Qwen-2.5-7B (Original/A-LQR/S-PID): `completed`, generated with bfloat16-safe loading.
- Qwen-2.5-1.5B is the first model beyond DistilGPT-2 with a complete 4-method column: H-infinity synthesis is feasible at 1536-dim (modest RAM), though steering is heavy-handed at this gain (T*I 0.0, Info 6.25). S-PID again improves over baseline (31.25 vs 25.0 T*I).
- Qwen-2.5-1.5B-paper-proto: `completed_paper_aligned` — evaluated with the A-LQR paper's exact protocol (base non-Instruct Qwen2.5-1.5B in 4-bit NF4; "Q: ... A:" prompts; 437 TruthfulQA samples; max_new_tokens=50, top_p=0.3, repetition_penalty=1.2; True/Info judged by allenai/truthfulqa-{truth,info}-judge-llama2-7B; MMLU 5-shot, 200 questions). Original T*I 36.5 +/- 2.2 now matches the paper's baseline range (Llama-3.2-1B: 40.2, Qwen-2.5-3B: 41.6); MMLU 49.5%. S-PID improves T*I to 54.4 with no MMLU cost; A-LQR/H-infinity over-steer at these gains (Info collapses to ~3.5%). Earlier `-ours` rows use a stricter lexical heuristic and small samples and are not comparable in absolute terms to the paper.
- Qwen-2.5-7B and 14B H-infinity: `infeasible_ram` — the finite-horizon gain recursion exceeds host RAM/time budgets (>84 GB at 14B; 7B timed out at 30 min under a 60 GB cap). Hardware limit, not a code failure: the same core succeeds on DistilGPT-2 and Qwen-1.5B.
- Qwen-2.5-14B (Original/A-LQR/S-PID): `completed_fp16_caveat` — these rows were produced before a float16-overflow bug in the quantized loader was found (Qwen2.5 in fp16 emits degenerate text). The loader now uses bfloat16; these rows should be regenerated before publication.
