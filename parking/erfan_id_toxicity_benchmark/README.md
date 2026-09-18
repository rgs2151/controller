# id_toxicity_model_metrics

## Method

- Select 50 held-out RealToxicityPrompts records with prompt toxicity in `[0.20, 0.50]`, excluding every controller fit and disturbance-calibration record.
- Generate 40-token continuations under Original, A-LQR, S-PID, and final H∞ with matched prompt seeds.
- Score completions with the pinned RoBERTa toxic-class classifier and compute mean toxicity and percentage reduction from Original.
- Plot both measurements for all five models.

## Variables

- Data/input: pinned RealToxicityPrompts and pinned RoBERTa toxicity classifier.
- Sessions/groups: five model sizes × four steering methods.
- Labels/targets: in-distribution toxicity control.
- Signals/features/measures: toxic-class probability and relative reduction from the Original mean.
- Parameters/thresholds: 50 prompts; 40 generated tokens; seed 2151; prompt-toxicity range `[0.20, 0.50]`.
- Outputs: `plots/results.csv`, `plots/id_toxicity_model_metrics.pdf`, and `.png`.

## Statistics

- Tests/models: descriptive arithmetic means and percentage changes.
- Null hypothesis: no inferential test is performed.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: lower mean toxic-class probability is better; positive plotted reduction means lower toxicity than Original.
- What the statistic means: toxicity is the average classifier probability across matched prompts.
- Why this statistic is appropriate here: the outcome is continuous and bounded, and all methods use the same prompt sample and seeds.

## Legends

- X axis: steering method.
- Y axis: continuation toxicity percentage or reduction from Original.
- Color/value: gray Original, midnight blue A-LQR, dark orange S-PID, dark green H-infinity.
- Grouping: rows are models; columns are metrics.
- Ordering/sorting: increasing Qwen scale after DistilGPT-2.
- Lines/markers/labels: bars only.
- Panels: toxicity and relative reduction.

## Interpretation

- H∞ reduces mean continuation toxicity by 82.3% for DistilGPT-2 and by more
  than 99% for every Qwen model. A-LQR is similarly strong; S-PID has almost no
  effect for DistilGPT-2 but reduces toxicity by 80.1% to 99.2% across Qwen
  scales.

## Notes

- These are archived results under their recorded original calibration. This unit now exposes plotting only; the superseded 50-prompt runner is retained at `ref/paper_benchmark_50/`. Historical caches are not upgraded or used as current benchmark results.

- Cache files contain prompt-level completions and make reruns resumable.

## References

- Historical implementation: `ref/erfan_applied_controller/`.
- Controller calibration: `parking/erfan_toxicity_calibration/`.
