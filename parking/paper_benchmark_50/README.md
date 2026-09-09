# prepare_data

## Method

- Freeze 50 held-out RealToxicityPrompts and 50 TruthfulQA questions after excluding every fit and disturbance-scaling source question.
- Draw RTP from the complete test pool; do not restrict evaluation to a narrow input-toxicity interval.
- Create Spanish, frozen D6 adversarial, dataset-shift, and long-context conditions. Spanish prompts explicitly request English output for the fixed English evaluators.
- Draw MMLU questions uniformly by subject, sample five demonstrations, use the reference `Question`/`Answer` format, and retain the first 50 seeded candidates that fit every benchmark model intact.
- Bind the prompt cache to the preparation source files and pinned dataset/model revisions.

## Variables

- Data/input: RealToxicityPrompts, TruthfulQA generation and multiple-choice configurations, MMLU, Jigsaw, and ToxicChat.
- Sessions/groups: 50 records per condition; identical prompt order across all methods and both behavior manifests for MMLU.
- Labels/targets: RTP classifier targets, TruthfulQA questions, MMLU answer indices, and source IDs for leakage checks.
- Signals/features/measures: prompt text, source toxicity, translations, adversarial marker count, demonstrations, and correct answer.
- Parameters/thresholds: seed 2151, five MMLU demonstrations, 1,022-token common MMLU input cap.
- Outputs: `cache/toxicity_prompts.json`, `cache/truthfulness_prompts.json`, `cache/data_manifest.json`, and translation checkpoints.

## Statistics

- Tests/models: no statistical test; selection is deterministic under the recorded seed.
- Null hypothesis: not applicable.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: every condition must contain exactly 50 unique, nonleaking records; MMLU prompts must fit without truncation.
- What the statistic means: dataset source scores are selection metadata, not benchmark outcomes.
- Why this statistic is appropriate here: this script freezes inputs and does not estimate an effect.

## Legends

- X axis: not applicable.
- Y axis: not applicable.
- Color/value: not applicable.
- Grouping: condition name.
- Ordering/sorting: deterministic sampled order fixes matched generation seeds.
- Lines/markers/labels: `prompt_id` identifies the condition record; `source_prompt_id` identifies its parent.
- Panels: not applicable.

## Interpretation

- Spanish is a bilingual instruction condition, not a translation-only causal contrast.
- The adversarial set transfers the already frozen D6 recipe; it does not search against these benchmark outcomes.

## Notes

- Run `python parking/paper_benchmark_50/prepare_data.py --device cuda:0`.
- A source or revision mismatch is an error; this unit does not reconstruct or adapt old prompt caches.

## References

- `robust_steerability/benchmarks/{calibration,ood,toxicity,truthfulness}.py`.
- `parking/ood_adversarial/` for the frozen D6 exploration.

# paper_benchmark_50

## Method

- Fit behavior targets from 50 negative and 50 positive examples and compute one actual last-token transformer-block Jacobian for every positive fit prompt, using the reference 24-token context cap. Prefix-token states are held fixed.
- Average full Jacobians for the published A-LQR recursion. Apply A-LQR and S-PID to the same context-updated semantic setpoint error used by the preserved implementation.
- Project the Jacobians into an eight-dimensional target-preserving orthonormal basis for H-infinity. State coordinates are not whitened.
- Define the H-infinity feedback state as the current semantic setpoint error in that basis. The nearest point on the semantic hyperplane is recomputed for every forward call, nominal feedforward is zero, and the next-layer orthonormal basis decodes the intervention.
- Fit the unscaled disturbance covariance factor from centered fit residuals, retain 95% variance, and scale it by the 95th percentile trajectory energy on 50 disjoint calibration prompts.
- Standardize target/protected performance readouts, use orthonormal control channels, and divide stage costs by transformer depth. Run Hannah's unchanged H-infinity recursion and cache her complete diagnostic bundle before evaluation.
- Fit ITI, ActAdd, Mean-AcT, Linear-AcT, PID-AcT, and ODESteer only from the same fit records. Generate every method on the same 50 prompts and matched per-prompt seeds.
- Score toxicity as the percentage assigned the toxic classifier label, TruthfulQA with the pinned True and Helpful judges, and MMLU from exactly one greedy answer token.

## Variables

- Data/input: the prompt files produced by `prepare_data.py`; 50 fit records per class and 50 disjoint disturbance-scaling records per behavior.
- Sessions/groups: five pinned models, two behaviors, ten methods, and the condition sets declared in each manifest.
- Labels/targets: negative/positive behavior classes, toxic/neutral classifier labels, True/Helpful judge labels, and MMLU answers A--D.
- Signals/features/measures: raw decoder activations, attention-head inputs, full and reduced Jacobians, semantic tracking errors, residuals, disturbance channels, gains, online interventions, generations, and evaluator outputs.
- Parameters/thresholds: rank 8, residual variance 0.95, disturbance coverage 0.95, H-infinity absolute bisection tolerance `1e-5`, and deployment margin 1%.
- Outputs: controller/Jacobian caches, Hannah diagnostic runs, per-prompt generations/traces/evaluator records, and `plots/toxicity_results.csv` plus `plots/truthfulness_results.csv`. H-infinity keeps the complete controller-coordinate tensors required by Hannah's handoff. Original is scalar-only; the six baselines, A-LQR, and S-PID keep per-step norm/energy trajectories plus controller artifacts and generations, avoiding redundant model-width tensor copies at 7B and 14B scale.

## Statistics

- Tests/models: descriptive prompt-level means and standard errors; logistic regression is used only for ITI head ranking and ODESteer fitting.
- Null hypothesis: no inferential null is tested in this 50-prompt pilot.
- Alternative hypothesis: not applicable until the planned full repeated evaluation.
- Thresholds/decision rule: toxic probability greater than 0.5 is the toxic class; True/Helpful outputs must parse exactly as yes/no; MMLU chance performance is 25%.
- What the statistic means: toxicity is a Bernoulli toxic-label rate; True, Helpful, and MMLU are Bernoulli success rates; their standard errors quantify prompt sampling variability within this pilot.
- Why this statistic is appropriate here: every reported outcome is a per-prompt binary classification or answer event under a fixed 50-prompt protocol.
- H-infinity feasibility is a numerical Riccati condition, not a significance test. Gamma star is the final feasible bisection boundary and robust steerability is its inverse.

## Legends

- X axis: not applicable; the script writes tabular rows.
- Y axis: not applicable.
- Color/value: not applicable.
- Grouping: model, method, behavior, and condition.
- Ordering/sorting: manifest model/method order and frozen prompt order.
- Lines/markers/labels: method labels are Original, ITI, ActAdd, Mean-AcT, Linear-AcT, PID-AcT, ODESteer, S-PID, A-LQR, and H-infinity.
- Panels: paper table placement is owned by `parking/paper_benchmark_tables/`.

## Interpretation

- This is one 50-prompt validation run per cell, not five repeated 1,000-prompt trials.
- Qwen-2.5-14B A-LQR/S-PID uses the preserved paper parameters. The additional model scales are declared extensions, not claimed reproductions of unreported source checkpoints.
- Baseline operators are transparent local adaptations; exact original-paper equivalence is not claimed where an upstream implementation does not define this common model/behavior protocol.

## Notes

- Run `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python parking/paper_benchmark_50/paper_benchmark_50.py --devices cuda:0,cuda:1`.
- Toxicity continuations use at most 100 new tokens; truthfulness uses 50. Sampling uses temperature 1, top-p 0.3, top-k 50, and repetition penalty 1.2. MMLU uses one greedy token.
- Every cache is bound to manifest, prompt, model revision, and source hashes. Mismatches fail; there is no old-cache reconstruction path or smoke/full flag.
- H-infinity handoff data includes exact problem tensors, fitted/calibration activations and IDs, residuals, disturbance construction, gains, solver diagnostics, online traces, evaluator observations, and source hashes.
- Baseline coefficients are frozen before table evaluation: ITI 0.25, ActAdd 0.1, Mean-AcT/Linear-AcT/PID-AcT 0.5, and ODESteer 1. The calibration-only online preflight rejected ITI strength 1 on DistilGPT-2 because it produced repetitive text and a large intervention/state norm ratio.

## References

- `ref/lqr-activation-steering/` for A-LQR/S-PID behavior and evaluation protocols.
- `ref/h_infinity.py` and `ref/h_infinity_optimization.py` for Hannah's solver and diagnostic exporter.
- `H_inf_Implementation_Kasra_version.pdf` for the robust recursion and limiting checks.

# score_quality

## Method

- Read only fresh RTP-ID generations whose job fingerprint matches the current source and manifest.
- Concatenate each prompt and continuation, tokenize jointly with the pinned external Mistral scorer, retain the first 128 tokens, and teacher-force every adjacent token exactly as in the reference scorer.
- Compute per-sequence perplexity from mean token negative log likelihood. Empty completions remain included because the prompt still defines a joint sequence.
- Compute pooled lowercase alphanumeric Dist-2, allowing bigrams across concatenated completion boundaries as in the reference implementation.

## Variables

- Data/input: `cache/jobs/toxicity/<model>/completions.json`, RTP-ID, all methods.
- Sessions/groups: 50 sequences per model and method.
- Labels/targets: observed next tokens under the external language model.
- Signals/features/measures: joint token IDs, token negative log likelihood, sequence perplexity, and pooled bigram diversity.
- Parameters/thresholds: Mistral-7B-v0.1 revision `27d67...`, NF4, float16 quantized compute, float32 nonquantized modules, right truncation at 128 tokens.
- Outputs: per-sequence quality checkpoints and `cache/quality/summary.json`.

## Statistics

- Tests/models: fixed external autoregressive language model; descriptive arithmetic mean and sample standard error across 50 per-sequence perplexities.
- Null hypothesis: no inferential null is tested.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: a sequence needs at least two joint tokens for perplexity; otherwise its value is explicitly undefined.
- What the statistic means: lower perplexity means greater likelihood under the fixed external scorer; higher Dist-2 means a larger fraction of unique pooled bigrams.
- Why this statistic is appropriate here: it reproduces the source paper's prompt-inclusive fluency/diversity evaluation for direct comparability.

## Legends

- X axis: not applicable.
- Y axis: not applicable.
- Color/value: not applicable.
- Grouping: model and method.
- Ordering/sorting: benchmark manifest order.
- Lines/markers/labels: `ppl_mean`, `ppl_se`, and `dist2` name the table measures.
- Panels: Table 1 quality columns.

## Interpretation

- Perplexity is prompt-inclusive and first-128-token truncated; it is not conditional continuation-only perplexity.
- Dist-2 is one pooled estimate and has no repeated-run error bar.

## Notes

- Run after generation with `python parking/paper_benchmark_50/score_quality.py --device cuda:0`.
- Checkpoints bind the exact prompt, completion, scorer revision, protocol, and implementation source.

## References

- `ref/lqr-activation-steering/steer/benchmarks/ppl_from_file.py`.
- `robust_steerability/benchmarks/metrics.py`.
