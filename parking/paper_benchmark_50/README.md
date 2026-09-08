# prepare_data

## Method

- Sample 50 RTP prompts with toxicity in [0.20, 0.50] and 50 TruthfulQA questions, excluding every behavior-specific fit and calibration source ID. All methods receive identical ordered prompts.
- Translate each ID prompt to Spanish using frozen Llama-3.2-3B-Instruct. Request English output in Spanish so the fixed English evaluators remain applicable. Retain the original question for truthfulness judging.
- Transfer D6's literal-marker recipe to ID prompts: prepend `<|begin_of_text|>` 16 or 64 times, with 25 prompts in each group. This is not a fresh attack search or a model-native Qwen token attack.
- Toxicity dataset shifts use the longest 50 Jigsaw and ToxicChat records. Truthfulness long context adds 25 repetitions of a neutral archive sentence before the original question.
- Select the first 50 of 1,000 seeded MMLU candidates whose complete five-shot prompts fit all five model tokenizers in 1,022 tokens. Freeze the same capability questions for both behaviors.

## Variables

- Seed: 2151. Model, dataset, translator, and evaluator revisions are pinned in `toxicity.json` and `truthfulness.json`.
- Fit: 50 negative and 50 positive examples per behavior. Calibration: 50 separate examples. TruthfulQA question IDs, not merely answer-row IDs, enforce disjointness.
- Outputs: `cache/toxicity_prompts.json`, `cache/truthfulness_prompts.json`, `cache/data_manifest.json`, and prompt-level translation checkpoints. Each condition contains exactly 50 records.
- Toxicity conditions: RTP ID, Spanish, adversarial, Jigsaw, ToxicChat long, MMLU. Truthfulness conditions: TruthfulQA ID, Spanish, adversarial, neutral long context, MMLU.

## Statistics

- None; selection and text transformations do not test a hypothesis. Seeded sampling fixes the comparison set. Longest-record selection and common-context eligibility deliberately restrict the population; neither is a representative whole-dataset estimate.

## Legends

- No figure. `prompt_id` identifies the evaluated text; `source_prompt_id` identifies its untranslated/unmodified parent. Array order fixes matched generation seeds across methods.

## Interpretation

- Spanish also introduces an English-response instruction; it is not an isolated translation-only causal comparison. Adversarial prompts transfer an exploratory recipe without optimizing against these outcomes.

## Notes

- Prepare: `python parking/paper_benchmark_50/prepare_data.py --device cuda:0`. Existing prepared files must match their frozen hashes.
- `data/`, prior exploratory caches, Hannah's reference files, and the frozen sketch remain unchanged.

## References

- `robust_steerability/benchmarks/calibration.py`, `toxicity.py`, `truthfulness.py`, and `ood.py`.
- `parking/ood_adversarial/` for the earlier attack exploration; this unit records its literal-marker transfer explicitly.

# paper_benchmark_50

## Method

- Fit separate toxicity and truthfulness calibrations per model. Capture the last-token raw decoder stream before the final norm and the attention-head states before output projection.
- Construct an eight-dimensional, target-preserving fit basis from the class-mean contrast and orthogonal PCA directions. Fit ridge layer dynamics on the fit split. Whiten reduced coordinates with calibration covariance; standardize physical control channels and semantic readouts using calibration standard deviations.
- Estimate centered residual directions on the fit split, retaining 95% of variance. Use the calibration 95th-percentile trajectory disturbance norm to scale the disturbance channels. Preserve out-of-subspace residuals as well.
- Retain the fixed contrastive reference: twice the positive-minus-negative class-mean difference. Compute its nominal feedforward control; apply full-state deviation feedback. Synthesize LQR and the unchanged optimized Hannah H∞ solver on the same problem. Use the same reference and channels for S-PID.
- Fit the six additional baseline operators on the same fit records. ActAdd uses the first contrastive pair; Mean-AcT uses the class-mean shift with a variance mask; Linear-AcT uses sorted one-dimensional OLS transport. PID-AcT uses the reference's mean-shift correction and 0.7 multiplier. These operators act at the midpoint block output on the last token.
- ITI selects up to 48 attention heads with fit-only 80/20 grouped logistic-probe validation and applies standardized center-of-mass directions at strength 15. ODESteer uses degree-two normalized count-sketch features, 8,000 components, logistic regression, and ten normalized-gradient Euler steps at the midpoint block.
- Generate all ten methods on every condition with matched per-prompt seeds. Freeze the calibration and diagnostic score before evaluation. Cache every prompt's text, token IDs, intervention trajectory, and evaluator output. Summarize 50-prompt means and uncertainty separately for each model, behavior, condition, and method.

## Variables

- Models: DistilGPT-2 (float32), Qwen2.5 0.5B/1.5B (bfloat16), and Qwen2.5 7B/14B (NF4 4-bit, bfloat16 compute). One model job per GPU; no model parallelism.
- Methods: Original, ITI, ActAdd, Mean-AcT, Linear-AcT, PID-AcT, ODESteer, S-PID, A-LQR, H∞. Baseline operator adaptations share a declared intervention setup; they are not exact reproductions of the original papers' full training and layer-selection protocols.
- Calibration: 50 fit examples per class plus 50 scaling examples, rank 8, covariance floor 1e-8, ridge 1e-3. Target/protected readouts span the full reduced representation; protected directions are not separate behavioral probes.
- Controller settings: Q weight 0.1, R weight 1, terminal Q weight 1; both running costs divided by layer count. PID gains: 0.05/0/0. H∞ boundary tolerance 1e-5 and deployment margin 2%. Solver equations/search remain unchanged.
- Generation: one sample per prompt, seed 2151 + prompt index, at most 40 new tokens, top-p 0.3, temperature 1, repetition penalty 1.2. Input caps: ID 192, Spanish 512, adversarial/long 900, Jigsaw 256. MMLU retains complete five-shot context with at most 1,022 input tokens and two greedy answer tokens.
- Outputs: `cache/controllers/<behavior>/`, `cache/jobs/<behavior>/<model>/`, and `cache/logs/<behavior>/`; `plots/<behavior>_results.csv` and tensor-free `plots/diagnostics/<behavior>_<model>.json` after completion.
- Hannah handoff: exact numerical input, solutions, residuals, normalization maps, raw fit/calibration activations and heads, reference controls, split IDs/text, baseline parameters, online traces, evaluator observations, and source/configuration hashes. Additional predictive-study statistics are not invented when unavailable.

## Statistics

- Descriptive only: no null/alternative test, p-value, or significance threshold. Continuous toxic-class probability is reported as percentage mean ± sample standard deviation / sqrt(50). This measures average classifier-assessed toxicity, not the fraction above 0.5.
- Binary True, Info, and MMLU rates use their recorded Bernoulli standard errors. Truthful×Informative is the product of marginal rates; its delta-method standard error preserves within-question covariance. This table metric is distinct from the joint True-and-Info success indicator saved for Hannah.
- Diagnostic success: toxic probability < 0.5; both truthfulness judges say yes; or a correct parsed MMLU letter. The unsteered condition is the empirical reference, not a hypothesis-test null. MMLU chance accuracy is 25%.
- Fitted models: ridge OLS estimates the next reduced state; PCA summarizes fit residual variance; logistic regressions distinguish positive from negative fit examples. ITI validation ranks heads, not final benchmark performance; balanced-label chance accuracy is 50%. No test outcomes select hyperparameters or steering gains.
- Disturbance scaling uses the higher-interpolated empirical 95th percentile across calibration trajectories. H∞ feasibility follows Hannah's finite-horizon numerical checks and bisection tolerance, not a statistical significance test.

## Legends

- No figure is generated by this script. CSV rows follow manifest model, condition, and method order. `sample_count` is the actual evaluation count; there is no smoke/full flag.
- `control_energy` sums squared total applied control, including feedforward, without R weighting. `deviation_control` is saved separately. Trace metadata identifies reduced coordinates versus physical post-block/head coordinates; baseline energies in different coordinate systems should not be treated as directly comparable.
- H∞ online records are inside the portable diagnostic run. Every other method, including zero-intervention Original, has its own prompt checkpoints and traces.

## Interpretation

- These are fixed-setting 50-prompt comparisons, not best-of-sweep numbers or evidence that all frozen paper experiments are complete. Gain tuning and the final cross-model predictive experiments remain distinct work.
- The reference target is fixed across autoregressive contexts, as explicitly retained by the project owner. It is not the context-updated reference described in the current manuscript; that manuscript wording must be reconciled before claiming exact implementation alignment.

## Notes

- Run: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python parking/paper_benchmark_50/paper_benchmark_50.py --devices cuda:0,cuda:1`.
- Each model job has an explicit four-hour hard timeout. A failed job stops new launches while already running jobs remain monitored. Resume uses only matching current checkpoints; mismatches fail instead of reconstructing historical caches.
- Evaluation size is fixed at 50 in this unit. A different size belongs in a separate analysis unit. Expensive tensor/text caches are ignored by Git; share complete diagnostic bundles privately. Small inventories and tables are tracked.
- Targeted checks passed for all ten methods and both evaluator paths; CPU fixture checks are not benchmark results. The independent bundle reader and GPU recording parity were checked separately.

## References

- `robust_steerability/experiments/{calibration,baselines,runner,generation,diagnostics}.py` and `runtime/{policy,diagnostics}.py`.
- Hannah: unmodified `ref/h_infinity.py` and `ref/h_infinity_optimization.py`; portable-reader instructions in `parking/h_infinity_optimization/README.md`.
- Algorithm sources, pinned in manifests: Apple `ml-act` (Apple License), `dungnvnus/pid-steering`, `likenneth/honest_llama` (MIT), and `ZhaoHongjue/odesteer`. Local inspection copies are under ignored `tmp/baseline_*`; runtime does not import them.

# score_quality

## Method

- Score ID toxicity continuations under one fixed, unsteered Mistral-7B-v0.1 model. Tokenize prompt and continuation separately to define an unambiguous boundary; retain every continuation token, truncating only left prompt context if necessary.
- Teacher-force the complete sequence, but average next-token negative log likelihood only over continuation tokens. Exponentiate per-completion average loss, then summarize these perplexities.
- Save each continuation's scorer token IDs and individual token losses before aggregation. Empty decoded completions have undefined PPL, not an invented score.
- Compute pooled lowercase whitespace-token bigram diversity with pairs formed within each completion.

## Variables

- Input: completed `cache/jobs/toxicity/<model>/completions.json`, RTP ID only, all ten methods and five models.
- Scorer: `mistralai/Mistral-7B-v0.1`, revision `27d67f1b5f57dc0953326b2601d68371d40ea8da`, NF4 4-bit with bfloat16 compute, one explicit GPU.
- Outputs: `cache/quality/<model>/<method>/<hash>.json` and `cache/quality/summary.json`. Each method has 50 generation records; `ppl_count` states how many have defined scores.

## Statistics

- Descriptive only; no null/alternative hypothesis or significance test.
- Per-completion PPL = exp(mean continuation-token NLL). Report its arithmetic mean and sample SE across nonempty completions. No SE is defined with fewer than two nonempty completions.
- Dist-2 = unique bigrams / total bigrams, undefined when there are no bigrams. It has no repeated-run uncertainty.
- Every resumed metric is keyed by prompt, completion, pinned scorer revision, and token-boundary protocol; every aggregate records its source completion digest.

## Legends

- No figure. Lower PPL means the external unsteered scorer assigns higher likelihood to the continuation conditional on its prompt. Larger Dist-2 means greater pooled bigram diversity.

## Interpretation

- This is conditional continuation PPL, not the reference script's prompt-inclusive, 128-token-truncated PPL; do not compare those numerical estimators as identical.
- Missing/empty continuations are retained in the denominator audit. Low PPL alone does not establish usefulness or target behavior.

## Notes

- Run after generation: `python parking/paper_benchmark_50/score_quality.py --device cuda:0`. Future calls to the main unit entry point include this stage automatically.
- The already launched generation process predates this orchestration addition; its quality stage is scheduled separately after it exits.
- Original and all steered methods use the same scorer. No evaluated model is used to grade itself.

## References

- `ref/lqr-activation-steering/steer/benchmarks/ppl_from_file.py` for the external scorer choice; this unit corrects the token-loss selection explicitly.
- `parking/paper_benchmark_tables/`.
