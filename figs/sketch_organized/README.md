# fig1_framing

## Method

- Draw the conceptual feedback-loop analogy between biological stimulation and transformer activation steering.
- Expose only the controller-facing sequence: observe state, compute feedback, intervene, and continue the system.

## Variables

- Data/input: none.
- Labels/targets: neural or hidden state, controller, intervention, and downstream behavior.
- Outputs: `plots/fig1_framing.{pdf,png}`.

## Statistics

- None; this output is conceptual.
- Null and alternative hypotheses: not applicable.
- Decision rule: no experimental values are required.

## Legends

- Teal: observed state; red: controller and intervention; gray: context or behavior; blue: downstream computation.
- Panels: A is biological stimulation and B is activation steering.

## Interpretation

- Both systems can be described using the same feedback abstraction without claiming biological equivalence.

## Notes

- This panel is complete unless manuscript terminology changes.

## References

- `figs/sketch/plots/fig1_framing.pdf`
- `/home/dev/controller/paper/iclr2026_conference.tex`

# fig2_dynamics_and_residuals

## Method

- Panel A embeds Erfan's measured PCA trajectories across positive and negative prompt sets.
- Panel B embeds the 50-prompt RTP-versus-Jigsaw layer-residual comparison under baseline and A-LQR.
- Panel C selects ID, Spanish translation, and long-context records from the OOD exploration table, adds the all-worse dispersion attack from the adversarial unit, and plots prompt-level remaining target error.
- Panel D embeds Erfan's model-scale residual comparison for within-distribution and adversarial prompts.

## Variables

- Data/input: Erfan linearization outputs, `parking/residual_checks/plots/layer_residuals.png`, the OOD failure table, and `parking/ood_adversarial/plots/adversarial_attempt_metrics.csv`.
- Signals/features/measures: PCA state coordinates, relative one-step residual, remaining target error as a percentage of the unsteered baseline, and relative residual-block update.
- Parameters/thresholds: 50 prompts per OOD-failure condition; 100% remaining error means no A-LQR benefit.
- Outputs: `plots/fig2_dynamics_and_residuals.{pdf,png}`.

## Statistics

- Tests/models: descriptive trajectories, prompt means, and box-plot distributions; source analyses contain their own inferential checks.
- Null hypothesis: none tested in this assembled figure.
- Alternative hypothesis: none tested in this assembled figure.
- Decision rule: larger relative residual or remaining error indicates worse linear prediction or steering.
- Why appropriate: the figure summarizes already-computed measurements without recomputing or pooling incompatible units.

## Legends

- Panel A: color identifies prompt set and black endpoints mark final states.
- Panel B: blue is RTP ID, red is Jigsaw OOD, dotted is baseline, and solid is A-LQR.
- Panel C: blue is ID, red is natural OOD, and black is adversarial OOD; boxes show quartiles and medians, whiskers use 1.5 IQR, and points are prompts.
- Panel D: circles are within-distribution means, diamonds are adversarial means, and subpanels are early, middle, and late depth.

## Interpretation

- Existing results establish structured hidden-state trajectories, larger OOD residuals, measurable A-LQR degradation across shifts, and model-dependent residual scaling.

## Notes

- All four panels contain real existing results; panel D is preserved from Erfan's controller-independent analysis.

## References

- `parking/erfan_linearization_error/`
- `parking/residual_checks/`
- `parking/ood_explore/`
- `parking/erfan_model_scale_residuals/`

# fig3_robust_steerability

## Method

- Retain the calibration-only computation diagram for identifying dynamics, solving H∞, and computing robust steerability.
- Retain synthetic layouts for the prospective predictor, predictor comparison, and leave-one-family-out validation.

## Variables

- Data/input: no completed prediction dataset yet.
- Labels/targets: calibration-only robust steerability and held-out OOD steering reliability.
- Parameters/thresholds: model-behavior pairs must define the outer validation split.
- Outputs: `plots/fig3_robust_steerability.{pdf,png}`.

## Statistics

- Tests/models: intended cross-validated regression, bootstrap confidence intervals, and leave-one-model-family-out evaluation.
- Null hypothesis: calibration-only robust steerability does not predict held-out OOD reliability beyond simpler predictors.
- Alternative hypothesis: robust steerability improves prospective prediction.
- Decision rule: no result is claimed until the held-out predictive experiment is run.
- Why appropriate: model-family-level validation prevents prompts from the same fitted pair leaking across folds.

## Legends

- Panel A: conceptual calibration flow.
- Panels B-D: colored synthetic points and bars are visibly labeled as placeholders.
- Dashed diagonal in panel D: perfect prediction.

## Interpretation

- The money-plot experiment remains missing; the existing five-model calibration values alone do not complete it.

## Notes

- Panels B-D are placeholders and must not be cited as results.

## References

- `figs/sketch_organized/TODO.md`
- `/home/dev/controller/paper/iclr2026_conference.tex`

# fig4_id_controller_benchmark

## Method

- Read the current five-model truthfulness and ID-toxicity result tables produced with Original, A-LQR, S-PID, and final H∞.
- Plot TruthfulQA truthful-times-informative percentage, toxicity reduction relative to Original, and MMLU accuracy retained relative to each model's Original score.
- Retain the intervention-energy tradeoff as a synthetic placeholder because intervention energy was not exported by the benchmark.

## Variables

- Data/input: `parking/erfan_truthfulness_benchmark/plots/results.csv` and `parking/erfan_id_toxicity_benchmark/plots/results.csv`.
- Sessions/groups: DistilGPT-2 and Qwen-2.5 at 0.5B, 1.5B, 7B, and 14B; four methods.
- Signals/features/measures: truthful-times-informative percentage, percentage toxicity reduction, and MMLU percentage retained.
- Parameters/thresholds: 50 TruthfulQA prompts, 50 ID toxicity prompts, and 50 MMLU questions per model-method cell.
- Outputs: `plots/fig4_id_controller_benchmark.{pdf,png}`.

## Statistics

- Tests/models: descriptive Bernoulli means for TruthfulQA/MMLU and arithmetic mean classifier probability for toxicity.
- Null hypothesis: no inferential test is performed in this assembled figure.
- Alternative hypothesis: not applicable.
- Decision rule: higher is better for all three real panels; 100% MMLU retained matches Original.
- Why appropriate: all methods use matched records and seeds inside each model.

## Legends

- X axis: D for DistilGPT-2, followed by Qwen parameter scale in billions.
- Y axes: metric named in each panel.
- Color/value: gray Original, midnight blue A-LQR, dark orange S-PID, and dark green H∞.
- Panels: A truthfulness, B ID toxicity reduction, C MMLU retention, and D intervention-energy placeholder.

## Interpretation

- The current run supplies real benchmark values for panels A-C. H∞ sharply lowers ID toxicity across all five models while truthfulness and MMLU effects remain model-dependent.

## Notes

- Panel D is a placeholder and requires matched intervention-energy measurements.
- These are the requested 50-record runs, not a larger paper-final sample.

## References

- `parking/erfan_truthfulness_benchmark/README.md`
- `parking/erfan_id_toxicity_benchmark/README.md`

# fig5_ood_controller_benchmark

## Method

- Combine the ID RTP and three OOD toxicity result tables and divide each steered mean toxicity by the matched Original mean.
- Panel A plots each model ratio plus the cross-model median and interquartile range for every method and condition.
- Panel B plots log10 H∞-to-Original toxicity ratio for every model-condition cell.
- Panel C joins Erfan's OOD relative residual summaries for DistilGPT-2 and Qwen-0.5B to A-LQR and H∞ toxicity ratios.
- Panel D compares calibration gamma-star with log10 H∞-to-A-LQR toxicity for five models and three OOD shifts.

## Variables

- Data/input: ID and OOD benchmark result CSVs plus Erfan's two selected-target residual CSVs.
- Sessions/groups: five models, four methods, ID RTP, Jigsaw, ToxicChat, and MMLU concept shift.
- Signals/features/measures: toxicity ratio, log toxicity ratio, mean OOD relative residual, and gamma-star.
- Parameters/thresholds: 50 prompts per benchmark cell; ratio 1 or log10 ratio 0 means no change.
- Outputs: `plots/fig5_ood_controller_benchmark.{pdf,png}`.

## Statistics

- Tests/models: descriptive cell means, cross-model median and IQR, and direct joins; no fitted association or inferential test.
- Null hypothesis: no inferential test is performed.
- Alternative hypothesis: not applicable.
- Decision rule: ratios below 1 and log ratios below 0 mean lower toxicity.
- Why appropriate: ratios normalize large baseline-toxicity differences while the untransformed cell values remain available in the source table.

## Legends

- Panel A: lines are cross-model medians, shading is the model IQR, and faint points are individual models.
- Panel B: blue cells reduce toxicity and red cells increase it; annotations show H∞/Original fold change.
- Panel C: midnight blue is A-LQR and dark green is H∞.
- Panel D: marker shape identifies Jigsaw, ToxicChat, or MMLU shift; negative values favor H∞ over A-LQR.

## Interpretation

- OOD transfer is sharply model-condition dependent. H∞ succeeds on several Qwen cells but amplifies ToxicChat toxicity for DistilGPT-2 and Qwen-7B.
- Panels C-D are real but preliminary: they cover toxicity only and do not complete the preregistered cross-behavior crossover claim.

## Notes

- Tiny Original toxicity in some cells can create large ratios; consult the source absolute-toxicity table before interpreting fold changes.

## References

- `parking/erfan_ood_steering_benchmark/README.md`
- `parking/erfan_ood_target_selection/README.md`

# fig6_long_context_prediction

## Method

- Retain the planned context-length sweep layout for A-LQR and H∞ target error, utility, and controller advantage.

## Variables

- Data/input: no completed multi-length benchmark.
- Labels/targets: context length, remaining target error, collateral utility, and H∞ advantage.
- Outputs: `plots/fig6_long_context_prediction.{pdf,png}`.

## Statistics

- Tests/models: intended prompt means with uncertainty across matched prompts.
- Null hypothesis: H∞ advantage does not increase with context length.
- Alternative hypothesis: H∞ advantage increases as accumulated mismatch grows.
- Decision rule: no result is claimed from the synthetic curves.
- Why appropriate: matched prompts across token budgets isolate length from prompt composition.

## Legends

- Midnight blue is A-LQR, dark green is H∞, and the gray line denotes equal error.
- All three panels are labeled placeholders.

## Interpretation

- Long prompts exist in the current OOD suite, but context length was not swept; Figure 6 therefore remains missing.

## Notes

- Do not substitute the single long-prompt condition for a length-response experiment.

## References

- `figs/sketch_organized/TODO.md`

# figs1_hinf_validation

## Method

- Read deterministic parity and benchmark tables from the H∞ optimization unit.
- Normalize gamma-star, gain, intervention, and diagnostic differences by their validation tolerances.
- Plot the independently computed induced gain divided by deployed gamma, GPU speedup, and peak CUDA allocation across state dimension.

## Variables

- Data/input: `parking/h_infinity_optimization/plots/equivalence.csv` and `benchmark_summary.csv`.
- Sessions/groups: Hannah reference versus optimized package implementation; CPU and RTX 5090 GPU cases.
- Signals/features/measures: error/tolerance, induced-gain ratio, reference/optimized time, and peak MiB.
- Parameters/thresholds: parity passes below 1; induced-gain certificate passes at or below 1; state dimensions 64-768.
- Outputs: `plots/figs1_hinf_validation.{pdf,png}`.

## Statistics

- Tests/models: deterministic numerical thresholds and median synchronized runtime across repetitions.
- Null hypothesis: none; this is numerical validation rather than population inference.
- Alternative hypothesis: none.
- Decision rule: all parity metrics must meet tolerance and induced gain must remain below deployed gamma.
- Why appropriate: both implementations receive identical tensors, so direct numerical comparison tests equivalence.

## Legends

- Panel A: normalized parity errors by validation case; dashed line is the tolerance boundary.
- Panel B: induced-gain/deployed-gamma ratio; dashed line is the certificate boundary.
- Panel C: dark green is GPU speedup and dashed line is no speedup.
- Panel D: dark red is Hannah reference and midnight blue is optimized memory.

## Interpretation

- All 10 CPU/GPU parity cases pass, every independent gain ratio is below 1, and the optimized implementation reaches about 3.86x speedup at dimension 768 with lower peak memory.

## Notes

- Runtime and memory are hardware-specific; mathematical parity is portable.

## References

- `parking/h_infinity_optimization/README.md`

# figs2_nonlinearity_validation

## Method

- Embed Erfan's measured actual-versus-Jacobian-predicted hidden-state norm trajectory in panel B.
- Retain the planned nonlinear-excursion, steering-strength, and certificate-coverage panels as synthetic placeholders.

## Variables

- Data/input: `parking/erfan_linearization_error/plots/actual_vs_linearized_hidden_dynamics.png` for panel B.
- Labels/targets: trajectory distance, linearization residual, steering strength, and certificate coverage.
- Outputs: `plots/figs2_nonlinearity_validation.{pdf,png}`.

## Statistics

- Tests/models: descriptive measured trajectory in panel B; intended matched prompt summaries in the missing panels.
- Null hypothesis: no inferential test is currently performed.
- Alternative hypothesis: not applicable.
- Decision rule: only panel B is treated as data.
- Why appropriate: the available layer trajectory checks linear prediction but does not supply the missing excursion or coverage experiments.

## Legends

- Panel B: blue is observed hidden-state norm and dashed red is Jacobian-predicted norm.
- Panels A, C, and D are explicitly labeled placeholders.

## Interpretation

- Existing evidence shows layerwise prediction divergence, especially at the last layer, but does not establish a nonlinear validity region.

## Notes

- A nonlinear correction is not claimed or implemented here.

## References

- `parking/erfan_linearization_error/README.md`

# figs3_tuning_and_ablations

## Method

- Retain layouts for gain-response, utility-energy, calibration-size, and disturbance-geometry experiments.

## Variables

- Data/input: no completed sweep tables.
- Labels/targets: gain multiplier, steering success, MMLU utility, intervention energy, gamma-star error, predictive R-squared, and disturbance geometry.
- Outputs: `plots/figs3_tuning_and_ablations.{pdf,png}`.

## Statistics

- Tests/models: intended matched-prompt gain sweep and repeated calibration subsampling.
- Null hypothesis: controller choice or calibration size does not change the named outcome.
- Alternative hypothesis: an operating gain and stable calibration regime exist, and disturbance geometry changes robust-control advantage.
- Decision rule: no result is claimed from these synthetic curves or boxes.
- Why appropriate: matched conditions are required to separate gain, sample size, and disturbance geometry effects.

## Legends

- Midnight blue is A-LQR and dark green is H∞; dashed lines denote energy in panel B.
- All four panels are labeled placeholders.

## Interpretation

- Gain selection and the two lower-priority ablations remain missing.

## Notes

- The selected gain must be fixed without consulting headline OOD outcomes.

## References

- `figs/sketch_organized/TODO.md`

# figure_arrangement_overview

## Method

- Assemble the nine generated figure images in proposed manuscript order without changing their panel contents.

## Variables

- Data/input: `plots/fig1_framing.png` through `plots/figs3_tuning_and_ablations.png`.
- Outputs: `plots/figure_arrangement_overview.{pdf,png}`.

## Statistics

- None; this output is a visual index.
- Null and alternative hypotheses: not applicable.
- Decision rule: panel badges define whether content is real, preliminary, conceptual, or placeholder.

## Legends

- Layout order: Figures 1-6 followed by Figures S1-S3.
- Legends and axes are inherited from each source figure.

## Interpretation

- The overview exposes exactly which parts of the planned paper story are already supported by data and which experiments remain.

## Notes

- Run `python figs/sketch_organized/sketch_organized.py` from the repository root to regenerate the unit from existing result artifacts.

## References

- `plots/panel_status.csv`
- `figs/sketch/`

# panel_status

## Method

- Record one row per figure or contiguous panel group with its evidence status and source basis.

## Variables

- Data/input: the explicit `PANEL_STATUS` mapping in `sketch_organized.py`.
- Labels/targets: `conceptual`, `real`, `real_preliminary`, or `placeholder`.
- Outputs: `plots/panel_status.csv`.

## Statistics

- None; this output is a descriptive inventory.
- Null and alternative hypotheses: not applicable.
- Decision rule: `real` requires an existing measured artifact; `real_preliminary` is measured but does not complete the frozen claim.

## Legends

- Columns: figure stem, panel letters, status, and evidence basis.

## Interpretation

- The table is the machine-readable answer to which parts of the sketch are currently complete.

## Notes

- Update this table whenever a placeholder is replaced by a completed experiment.

## References

- `figs/sketch_organized/TODO.md`
