# Truthfulness benchmark

This benchmark uses the same four-stage interface as every benchmark:

1. `artifacts` freezes the fit data, semantic setpoint, and average of 35 Jacobians.
2. `calibrate` records published A-LQR parameters, fits each baseline's required
   one-time artifact, and either selects H∞ on a 50-question development set by
   the configured calibration objective or synthesizes one explicitly supplied
   H∞ configuration without a sweep.
3. `evaluate` generates responses only; it never runs a scorer.
4. `score` applies any requested subset of independent scorers to existing responses.

The benchmark's default scorers are:

- `truthfulqa_true`: pinned TruthfulQA True judge and rubric; binary percentage.
- `truthfulqa_informative`: pinned TruthfulQA Helpful judge and rubric; binary percentage.
- `axbench_concept_relevance`: AXBench 0–2 relevance of truthful, factually
  accurate, non-misleading answers through `gpt-4o-mini`.
- `axbench_instruction_relevance`: AXBench 0–2 rubric through `gpt-4o-mini`.
- `axbench_fluency`: AXBench 0–2 rubric through `gpt-4o-mini`.
- `axbench_overall`: per-response harmonic mean of the three AXBench measures.
- `mmlu_accuracy`: exact A/B/C/D accuracy on 200 frozen five-shot MMLU
  questions; no learned judge.

The default H∞ objective is `truthfulqa_txi_fluency_composite`: the mean
per-response weighted sum of TruthfulQA True-times-Informative (weight .95) and
normalized AXBench fluency (weight .05). A response contributes to the primary
term only when both binary TruthfulQA judges pass. Fluency is divided by two so
both terms are on `[0, 1]`; the additive form prevents a zero fluency score from
collapsing the TruthfulQA signal. The configuration retains the historical
`truthfulqa_true_mean_percentage`, `mean_axbench_overall`, and
`truthfulness_quality_composite` objectives; `--selection-metric` selects among
them. Concept relevance and instruction relevance remain reported outcomes but
do not enter the new default objective.
H∞ always uses the same model-specific published setpoint multiplier as A-LQR;
there is no Truthfulness lambda sweep. Calibration sweeps only `Q/R` and `Qf/R`
with `R=1`.

```bash
python -m robust_steerability.benchmarks.truthfulness artifacts --model llama8b --devices auto
python -m robust_steerability.benchmarks.truthfulness calibrate --model llama8b --methods alqr,h_infinity --devices auto
python -m robust_steerability.benchmarks.truthfulness evaluate --model llama8b --methods original,alqr,h_infinity --datasets id,spanish,mmlu --devices auto
python -m robust_steerability.benchmarks.truthfulness score --model llama8b --methods original,alqr,h_infinity --datasets id,spanish,mmlu --scorers default --devices auto
```

To freeze an already chosen H∞ configuration without running its grid, supply
all three cost arguments during `calibrate`:

```bash
python -m robust_steerability.benchmarks.truthfulness calibrate --model gemma2b --methods h_infinity --calibration-id fixed_q0p1_qf0p31622777_r1 --h-infinity-q-over-r 0.1 --h-infinity-q-final-over-r 0.31622776601683794 --h-infinity-r 1 --devices auto
```

This still fits the task/model disturbance geometry and synthesizes H∞ once;
it skips candidate generation and judging. The resulting calibration directory
contains the exact selected gains, gamma, and Hannah diagnostic bundle.

ITI and S-PID have preserved source grids but no preserved final Gemma choice.
The project therefore freezes Gemma truthfulness best guesses from those grids:
ITI uses 32 heads and alpha 10; S-PID uses lambda 1, Kp=.7, Ki=.01, and Kd=.1.
Neither method is swept.

For Llama-3-8B, the frozen truthfulness choices are ITI with 32 heads and alpha
10, S-PID with lambda 1, Kp=.1, Ki=.1, and Kd=0, and ODESteer at layer 19 with
time 25. The ITI and S-PID values are central source-grid choices. The
truthfulness adapter does not preserve a final Llama ODESteer selection, so its
preserved same-model comparison setting is carried across without a sweep.

Evaluated-model KV cache defaults to off. `--kv-cache on` remains available for
an explicit appendix comparison and writes to a different directory. API scoring
defaults to concurrency 500 and batch size 20.

`benchmark.toml` is the composition surface. TruthfulQA owns fitting and
calibration; Spanish and MMLU consume the selected controllers without fitting
or selecting anything again. Every dataset has an independent cache namespace,
so adding MMLU leaves complete TruthfulQA and Spanish generations untouched.
A named `--calibration-id` also isolates its evaluation generations, scores,
and compact results. The historical `selected` outputs remain at their original
paths, so a new H∞ attempt cannot overwrite or silently reuse the published row.

The Gemma-2-2B and Llama-3-8B T×I rerun uses calibration ID
`txi_fluency_95_05`, H∞ only, and one complete pass over the same 817 English
questions plus the same 817 Spanish-transfer questions. Existing Original and
A-LQR rows stay untouched. Five-group question jackknife uncertainty is added
after scoring; it measures question-sampling variability, not decoding-run
variability. Gemma runs first, followed by Llama.

For each model, the rerun is the same three-stage sequence; do not launch the
Llama sequence until the Gemma score stage completes:

```bash
python -m robust_steerability.benchmarks.truthfulness calibrate --model gemma2b --methods h_infinity --calibration-id txi_fluency_95_05 --selection-metric truthfulqa_txi_fluency_composite --devices auto
python -m robust_steerability.benchmarks.truthfulness evaluate --model gemma2b --methods h_infinity --datasets id,spanish --calibration-id txi_fluency_95_05 --evaluation-repetitions 1 --devices auto
python -m robust_steerability.benchmarks.truthfulness score --model gemma2b --methods h_infinity --datasets id,spanish --calibration-id txi_fluency_95_05 --evaluation-repetitions 1 --scorers default --devices auto
```

Repeat those commands with `--model llama8b`. The one-pass score stage writes
the five-group question-jackknife standard errors automatically.

```text
cache/<model>/artifacts/
cache/<model>/datasets/
cache/<model>/calibrations/<method>/<calibration-id>/
cache/<model>/evaluations/<kv-cache-condition>/generations/<dataset>/<method>/
cache/<model>/evaluations/<kv-cache-condition>/calibrations/<calibration-id>/generations/<dataset>/<method>/
cache/<model>/evaluations/<kv-cache-condition>/calibrations/<calibration-id>/scores/<scorer>/<dataset>/<method>/
cache/<model>/evaluations/<kv-cache-condition>/calibrations/<calibration-id>/results/<dataset>/<method>.json
results/<kv-cache-condition>/<model>/<dataset>/<method>.json
results/<kv-cache-condition>/calibrations/<calibration-id>/<model>/<dataset>/<method>.json
```

On Lightning, this complete cache hierarchy is written directly to the
producing Studio's Teamspace Drive at
`~/robust-steering-cache/truthfulness/<model>/`. Small summaries and stage logs
are committed to Git.
