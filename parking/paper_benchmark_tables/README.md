# table1

## Method

- Read completed current toxicity jobs from `parking/paper_benchmark_50/`, requiring matching manifest and prompt-source fingerprints. No historical Erfan results or earlier-paper numbers enter this export.
- Join model, condition, and method rows. Retain toxic-class probability percentage means and prompt-level SEs. Compute pooled ID Dist-2 with bigrams formed within, never across, completions.
- Use the same toxicity-steered MMLU evaluations for capability retention. Join separately cached conditional Mistral perplexity by model/method and verify source completion hashes.
- Keep all five model blocks and ten methods. Unmeasured or undefined cells remain red TBD; export each cell's status and denominator.

## Variables

- Script: `paper_benchmark_tables.py export`.
- Inputs: `../paper_benchmark_50/cache/jobs/toxicity/<model>/{result,completions}.json`; quality input `../paper_benchmark_50/cache/quality/summary.json`.
- Conditions: RTP ID, Spanish with English-response instruction, transferred D6 literal markers, longest Jigsaw prompts, and longest ToxicChat prompts. Exact splits and token caps are in the source unit README/manifests.
- Outputs: `plots/table1.tex`, `plots/cell_coverage.csv`, `plots/provenance.json`. There are 400 planned cells in Table 1.
- Methods: Original, ITI, ActAdd, Mean-AcT, Linear-AcT, PID-AcT, ODESteer, S-PID, A-LQR, H∞. Baselines use the source unit's declared adaptations, not exact published full protocols.

## Statistics

- Descriptive only; no null/alternative hypothesis, p-value, or winner significance threshold.
- Toxicity: mean classifier probability × 100; SE = sample SD / sqrt(50). MMLU: cached Bernoulli mean/SE over exactly 50 intact five-shot questions.
- Dist-2: unique within-completion bigrams divided by all within-completion bigrams; undefined if none exist. It has one pooled estimate, not a repeated-run SE.
- PPL: arithmetic mean of per-completion conditional perplexities and sample SE among nonempty completions; valid counts below 50 are shown explicitly. Empty completions remain visible, not fabricated as EOS-based scores.
- These summaries compare matched prompts under fixed settings. Prompt-level SE is not run-to-run variation. All source rows must have 50 evaluation prompts; cache hashes must match.

## Legends

- Rows: five model blocks, with ten method rows per block.
- Columns: RTP-ID, Spanish, Adversarial, Jigsaw, Long, Dist-2, MMLU, PPL.
- Black values are measurements; red TBD cells are missing or undefined. Lower is better for toxicity/PPL; higher is better for Dist-2/MMLU. No bold winner is selected.

## Interpretation

- A low toxicity score alone does not establish useful steering. Inspect diversity, capability retention, empty completions, and perplexity together.
- These are fixed-setting, 50-prompt results; no test-set-selected gain sweep is represented.

## Notes

- Generate/scoring belongs to the source benchmark unit; this exporter only reads caches.
- Before exporting a new iteration, archive this unit's existing plot outputs. Do not delete source benchmark caches.
- Manuscript table replacement and PDF compilation follow verification of the exported files; generating TeX alone does not update the manuscript PDF.

## References

- `parking/paper_benchmark_50/README.md`, `score_quality.py`, and manifests.
- `robust_steerability/experiments/runner.py`.

# table2

## Method

- Read completed, fingerprint-matching truthfulness jobs from the current 50-prompt unit.
- Populate ID and OOD True-times-Info plus ID True, Info, and capability retention under the separately truthfulness-calibrated policies.
- Keep all planned model/method rows. Missing jobs remain unmeasured; never substitute toxicity-calibrated TruthfulQA results.

## Variables

- Input: `../paper_benchmark_50/cache/jobs/truthfulness/<model>/result.json`.
- Outputs: `plots/table2.tex` and shared cell-coverage/provenance files; 350 planned cells.
- Conditions: TruthfulQA ID, Spanish with English-response instruction, transferred D6 markers, neutral archive context; same five-shot MMLU questions as the toxicity unit.
- Judge revisions and generated answer tokens are saved in each job's `judge_scores.json`.

## Statistics

- Descriptive only; no hypothesis test or significance threshold. Source sample count and MMLU count must both equal 50.
- True and Info: binary judge percentages with Bernoulli SEs. T·I: product of marginal rates, in percent, with a paired delta-method SE including within-question covariance.
- T·I is not the mean of promptwise joint True-and-Info success. Hannah's diagnostic handoff retains that distinct joint-success endpoint.
- Paired prompt-level uncertainty describes this one fixed run, not variation across independent training runs.

## Legends

- Rows: the same five model blocks and ten methods as Table 1.
- Columns: ID T·I, Spanish, Adversarial, Long, ID True, ID Info, and MMLU, all percentages.
- Black: measured mean ± SE. Red TBD: not measured. Higher values indicate higher rates; no winner or significance marks.

## Interpretation

- Truthfulness and informativeness must be considered together; consistently uninformative responses should not be presented as successful truthfulness steering.
- A completed table does not complete the paper's distinct predictive and ablation experiments.

## Notes

- Historical outputs and frozen figure sketches are not inputs to this exporter.
- The manuscript has unrelated unfinished sections and reference warnings; table exports do not resolve those issues.

## References

- `parking/paper_benchmark_50/README.md` and `truthfulness.json`.
- `robust_steerability/benchmarks/calibration.py` and `experiments/runner.py`.
