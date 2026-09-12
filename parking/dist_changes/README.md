# distribution_shift_truth_info_residuals

## Method

- Use the pinned `google/gemma-2-2b` checkpoint and the clean TruthfulQA A-LQR artifacts in `parking/bench_artifacts`.
- Reuse A-LQR's averaged 35-prompt Jacobian matrix unchanged when fitting the rank-8 full-state H∞ controller. Fit both semantic targets from the same 200 false and 200 true MC2 answer prompts; fit H∞ disturbance geometry on 200 separate generation questions.
- Evaluate A-LQR and H∞ on eight frozen 50-prompt sets: ID, Spanish, Japanese-romaji, Long Context End, Long Context Start, Corrupting Words, BOS Mix, and L-CiteEval Complexity.
- Generate with the A-LQR paper protocol and score each completion with the pinned binary TruthfulQA truth and informativeness judges.
- During evaluation, cache the last-input-token state at every decoder depth and the actual post-block hidden intervention. Compute the raw one-step residual as `h[k+1] - mean[k+1] - A[k] @ (h[k] - mean[k]) - u[k]`; divide its norm by the next-state norm and average across layers and prompts.
- Project the observed trajectory into the shared rank-8 coordinates. Starting from zero deviation, drive each controller's closed-loop depth dynamics with its observed reduced residual sequence and measure the common Q/R/Qf performance energy. Report the pooled square root of total performance energy divided by total residual energy.
- Write a 2×2 grouped-bar figure containing Truth, Info, full-state dynamics mismatch, and rank-8 residual-to-performance gain.

The x-axis distributions are:

| Label | Construction |
| --- | --- |
| ID | The unchanged held-out TruthfulQA prompt `Q: question A:`. |
| Spanish | Llama-3.2-3B-Instruct deterministically translates the question into Spanish, followed by a request to answer in English. |
| Japanese (romaji) | Meta-Llama-3.1-8B-Instruct deterministically translates the question into Japanese, pykakasi converts it to Hepburn romaji, and the prompt asks for an English answer. |
| Long Context End | Seven fixed public-domain excerpts fill about 7,168 Gemma tokens; the unchanged question is at the end. |
| Long Context Start | The same documents and question are used, but the question and `A:` cue come before the documents. |
| Corrupting Words | A frozen text-only suffix found by gradient search to maximize A-LQR overshoot on one Llama-3.2-1B prompt is appended to every TruthfulQA prompt. |
| BOS Mix | Gemma-2-2B's actual `<bos>` token is distributed through each prompt; a fixed seed assigns 16 insertions to 25 questions and 64 to the other 25. |
| L-CiteEval Complexity | Fifty NarrativeQA and LoCoMo long-context questions spanning easy, medium, and hard examples. |

## Variables

- Data/input: pinned TruthfulQA generation and multiple-choice validation splits plus 50 L-CiteEval NarrativeQA/LoCoMo questions; 50 prompts per condition.
- Sessions/groups: A-LQR and corrected full-state H∞ on all eight conditions.
- Labels/targets: positive-minus-negative TruthfulQA MC2 semantic target; binary `True` and `Helpful` judge decisions.
- Signals/features/measures: generated completion, Truth percentage, Info percentage, full 2304-dimensional hidden states and interventions, full-state relative residual, rank-8 residual energy, and rank-8 Q/R/Qf performance energy.
- Parameters/thresholds: seed 2151; 200 false plus 200 true fit prompts; 35 shared Jacobians; 200 disturbance prompts; rank 8; 50 evaluations per method and condition; 50 generated tokens.
- Outputs: `plots/distribution_shift_truth_info_residuals.pdf`, `plots/distribution_shift_truth_info_residuals.png`, `plots/distribution_scores.csv`, `plots/summary.json`, and one five-prompt inspection file per dataset under `plots/ood_examples/`.

## Statistics

- Tests/models: descriptive binary judge percentages, mean layer-relative residual, and pooled residual-to-performance energy gain; no inferential test or confidence interval is shown.
- Null hypothesis: not tested in this exploratory unit.
- Alternative hypothesis: not tested; the practical comparison is whether H∞ retains benchmark quality and attenuates observed residual disturbances more than A-LQR.
- Thresholds/decision rule: exact judge `yes` maps to 1 and exact `no` maps to 0; malformed outputs are rejected. Lower is better for both residual measures.
- What the statistics mean: Truth and Info are fractions of positive judge decisions; dynamics mismatch is the average residual norm relative to next-state norm; residual-to-performance gain is the pooled closed-loop cost induced per unit of observed reduced residual energy.
- Why these statistics are appropriate here: the benchmark panels test task behavior, while the two residual panels separate disturbance size from the controller's response to that disturbance.

## Legends

- X axis: ID, Spanish, Japanese (romaji), Long Context End, Long Context Start, Corrupting Words, BOS Mix, and L-CiteEval Complexity.
- Y axes: top-left is Truth percentage; top-right is Info percentage; bottom-left is mean layer-relative residual percentage; bottom-right is residual-to-performance gain.
- Color/value: black is A-LQR and red is H∞; bar height and the printed number give the same aggregate value.
- Grouping: two bars per distribution, each based on the same 50 prompts; the first seven sets share TruthfulQA anchors and L-CiteEval supplies its own 50 questions.
- Ordering/sorting: distributions use the fixed conceptual order above; methods always appear A-LQR then H∞.
- Lines/markers/labels: bars have no error bars or confidence intervals. Higher is better in the top row and lower is better in the bottom row.
- Panels: Truthfulness and Informativeness are the top row; Dynamics mismatch and Residual amplification are the bottom row.

## Interpretation

- H∞ does not outperform A-LQR on Truth in this 50-prompt run, although it has higher Info on six of the eight conditions.
- The two methods encounter similar full-state mismatch: H∞ is 0.15 percentage points lower on Spanish and 0.05–2.49 points higher on the other conditions.
- H∞ has lower residual-to-performance gain in all eight conditions, with reductions of about 24–31% relative to A-LQR. The intended robustness mechanism is therefore visible even though it does not yet translate into better Truth scores.
- These results separate two claims: H∞ attenuates the downstream cost of measured residuals more strongly, but the present controller/evaluation setup does not establish better benchmark performance under these shifts.

## Notes

- `cache/residual_rollouts/{alqr,hinf}/` stores one strict manifest and one file per condition. Every condition file contains prompt IDs, input-token counts, full prompt-end states, actual hidden controls, reduced states and controls, raw and reduced residuals, and per-prompt energy/gain values.
- The residual stage is part of `--stage all`, is resumable by complete condition, and never writes outside this unit. Once cached, plot changes and further residual summaries require no model forward pass.
- `plots/ood_examples/` contains one inspection file per dataset, each with five complete literal prompts.
- `cache/datasets/` contains eight immutable 50-row CSVs. A later full-size run must use a separate frozen bundle.
- The automatic Spanish translations require replacement or manual validation. The romaji and BOS-mix outputs expose judge failures, and the TruthfulQA judges do not receive the source passages or references needed to validate L-CiteEval answers; those benchmark bars are diagnostic rather than final task-valid scores.
- Every artifact produced by this analysis remains under `parking/dist_changes/`. The unit only reads frozen A-LQR inputs from `parking/bench_artifacts/`.

## References

- `parking/bench_artifacts/`
- `parking/ood_explore/`
- `parking/ood_adversarial/`
- [L-CiteEval dataset](https://huggingface.co/datasets/Jonaszky123/L-CiteEval)
- `ref/paper_benchmark_50/prepare_data.py`
- `robust_steerability/experiments/calibration.py`
- `robust_steerability/modeling/interventions.py`
- `robust_steerability/source_methods/`
