# brain_electrode_map_existing_export

## Method

- Stage the surviving P12 electrode-map PDF and PNG from `NeuroAnalysis/figures/` inside this unit.
- Preserve the files byte-for-byte; do not redraw or imply that the absent raw neural cache was rerun.

## Variables

- Data/input: existing `NeuroAnalysis/figures/electrode_map_P12.pdf` and `.png`.
- Sessions/groups: participant P12 electrode coverage and representative signals.
- Labels/targets: lead locations and example neural traces from the original export.
- Signals/features/measures: visual anatomy and representative traces; no new measurement is computed here.
- Parameters/thresholds: inherited from the original neuro analysis.
- Outputs: `plots/brain_electrode_map_existing_export.pdf` and `.png`.

## Statistics

- None; this is a staged source export.
- Null hypothesis: not applicable.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: not applicable.
- What the statistic means: no statistic is newly computed.
- Why this statistic is appropriate here: not applicable.

## Legends

- X axis: inherited from the original export.
- Y axis: inherited from the original export.
- Color/value: inherited from the original export.
- Grouping: inherited from the original export.
- Ordering/sorting: inherited from the original export.
- Lines/markers/labels: inherited from the original export.
- Panels: combined anatomy and representative-signal source panel.

## Interpretation

- This preserves the available anatomical context while making the recomputation boundary explicit.

## Notes

- Exact regeneration is blocked because the ignored neural cache and raw `.mat` inputs are absent from this checkout.

## References

- `NeuroAnalysis/brain_local_linearization.ipynb`.

# brain_information_flow_schematic

## Method

- Reconstruct the coarse cortical information-flow abstraction from the supplied figure reference and the neuro analysis specification.
- Draw labeled cortical modules and directed conceptual connections, followed by the local-linear dynamics plus nonlinear-residual equation.
- Write standalone PDF and PNG outputs.

## Variables

- Data/input: conceptual labels in `NeuroAnalysis/ANALYSIS_SPEC.md` and the supplied composition reference.
- Sessions/groups: not applicable.
- Labels/targets: PPC, TP, VLPFC, DLPFC, PT, SMA/PreM, and M1/S1.
- Signals/features/measures: conceptual directed information flow and the decomposition of future state into local-linear and residual terms.
- Parameters/thresholds: none.
- Outputs: `plots/brain_information_flow_schematic.pdf` and `.png`.

## Statistics

- None; this output is a conceptual schematic.
- Null hypothesis: not applicable.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: not applicable.
- What the statistic means: no statistic is shown.
- Why this statistic is appropriate here: the panel communicates the model abstraction, not an empirical estimate.

## Legends

- X axis: none.
- Y axis: none.
- Color/value: color distinguishes coarse cortical modules; it does not encode a measured magnitude.
- Grouping: modules are arranged from association regions toward motor/sensorimotor regions.
- Ordering/sorting: conceptual cortical flow.
- Lines/markers/labels: circles are modules and arrows are hypothesized computational flow.
- Panels: one standalone schematic.

## Interpretation

- The panel introduces the local-linear-plus-residual decomposition used by the quantitative brain analyses.

## Notes

- Connections are a computational abstraction and must not be described as proven monosynaptic pathways.

## References

- `NeuroAnalysis/ANALYSIS_SPEC.md`.

# brain_error_shrinkage_conflict

## Method

- Load the frozen P12 LVF theta-band error summary for conflict shift.
- Compare the median per-token residual RMS of the linear model with the linear-plus-residual model.
- Plot their difference, `median RMS(linear) - median RMS(linear + g)`, for ID low-conflict and OOD high-conflict trials at each horizon.

## Variables

- Data/input: `NeuroAnalysis/results/final_panelD_error_shrinkage_LVF_theta.csv`.
- Sessions/groups: P12, LVF lead, theta power, low- versus high-conflict trials.
- Labels/targets: ID and OOD.
- Signals/features/measures: reduction in median residual RMS, in training-normalized z units.
- Parameters/thresholds: horizons 20, 40, 100, and 200 ms; positive values mean the learned residual reduces error.
- Outputs: `plots/brain_error_shrinkage_conflict.pdf` and `.png`.

## Statistics

- Tests/models: descriptive medians from frozen ridge-plus-residual models; no inferential test.
- Null hypothesis: reduction equals zero.
- Alternative hypothesis: positive reduction indicates improved median prediction error.
- Thresholds/decision rule: the zero line separates improvement from worsening.
- What the statistic means: change in typical per-token state-prediction error attributable to `g(x)`.
- Why this statistic is appropriate here: it measures the residual model's direct contribution in the same normalized units across horizons.

## Legends

- X axis: prediction horizon in milliseconds.
- Y axis: median residual-RMS reduction in z units.
- Color/value: blue is ID; terracotta is OOD.
- Grouping: paired bars by horizon.
- Ordering/sorting: increasing horizon.
- Lines/markers/labels: horizontal zero reference.
- Panels: one standalone conflict-shift plot.

## Interpretation

- Conflict shift shows no positive OOD rescue by the learned residual over the tested horizons.

## Notes

- Single participant and single lead.

## References

- `NeuroAnalysis/final_figure.ipynb`.

# brain_error_shrinkage_stim_context

## Method

- Use the same frozen P12 LVF theta-band error summary for NS1-to-NS2 stimulation-context transfer.
- Plot error reduction from `g(x)` for held-out NS1 ID trials and non-stimulated NS2 OOD trials at each horizon.

## Variables

- Data/input: `NeuroAnalysis/results/final_panelD_error_shrinkage_LVF_theta.csv`.
- Sessions/groups: NS1 ID and NS2 OOD.
- Labels/targets: ID and OOD.
- Signals/features/measures: median residual-RMS reduction in z units.
- Parameters/thresholds: horizons 20, 40, 100, and 200 ms; positive is improvement.
- Outputs: `plots/brain_error_shrinkage_stim_context.pdf` and `.png`.

## Statistics

- Tests/models: descriptive medians from frozen models; no inferential test.
- Null hypothesis: reduction equals zero.
- Alternative hypothesis: positive reduction indicates improved prediction.
- Thresholds/decision rule: zero separates improvement from worsening.
- What the statistic means: typical-error reduction attributable to the nonlinear residual correction.
- Why this statistic is appropriate here: it directly tests whether the ID-fitted correction transfers under session-context shift.

## Legends

- X axis: prediction horizon in milliseconds.
- Y axis: median residual-RMS reduction in z units.
- Color/value: blue is ID; terracotta is OOD.
- Grouping: paired bars by horizon.
- Ordering/sorting: increasing horizon.
- Lines/markers/labels: horizontal zero reference.
- Panels: one standalone stimulation-context plot.

## Interpretation

- The main positive effect is OOD error shrinkage at 200 ms.

## Notes

- Single participant and single lead.

## References

- `NeuroAnalysis/final_figure.ipynb`.

# brain_state_size_across_leads

## Method

- Load per-lead P12 theta-band residual summaries at a 200-ms horizon.
- Plot each clean lead's channel count against median linear-model residual RMS.
- Color leads by coarse coordinate-derived anatomical area.

## Variables

- Data/input: `NeuroAnalysis/results/final_panelF_state_size_theta.csv`.
- Sessions/groups: 12 clean P12 leads with at least five matched channels.
- Labels/targets: lead identity and coarse anatomical area.
- Signals/features/measures: channel count and median residual RMS in z units.
- Parameters/thresholds: 200-ms horizon.
- Outputs: `plots/brain_state_size_across_leads.pdf` and `.png`.

## Statistics

- None; this is a descriptive scatter of lead-level summaries.
- Null hypothesis: not tested.
- Alternative hypothesis: not tested.
- Thresholds/decision rule: none.
- What the statistic means: absolute typical prediction error for each sampled neural population.
- Why this statistic is appropriate here: it exposes both population size and anatomical heterogeneity without collapsing leads.

## Legends

- X axis: channels in the lead.
- Y axis: median residual RMS in z units.
- Color/value: coarse anatomical area.
- Grouping: one point per lead.
- Ordering/sorting: numeric channel count.
- Lines/markers/labels: lead abbreviations annotate points.
- Panels: one standalone scatter plot.

## Interpretation

- Across-lead differences reflect anatomy as well as state size and are observational rather than causal.

## Notes

- Area labels are coordinate-derived coarse groupings, not clinical parcellations.

## References

- `NeuroAnalysis/final_figure.ipynb`.

# brain_state_size_lvf_subsets

## Method

- Load repeated random LVF channel-subset summaries at a 200-ms horizon.
- Plot mean median residual RMS by subset dimension with between-subset standard deviation.

## Variables

- Data/input: `NeuroAnalysis/results/final_panelF_state_size_theta.csv`.
- Sessions/groups: random channel subsets within LVF.
- Labels/targets: state dimension k.
- Signals/features/measures: mean median residual RMS and subset-to-subset standard deviation.
- Parameters/thresholds: k = 2, 4, 6, 8, 10, 12, and 15; ten subsets except the full 15-channel state.
- Outputs: `plots/brain_state_size_lvf_subsets.pdf` and `.png`.

## Statistics

- Tests/models: descriptive mean and standard deviation across subset refits.
- Null hypothesis: not tested.
- Alternative hypothesis: not tested.
- Thresholds/decision rule: none.
- What the statistic means: absolute prediction error as the observed state dimension changes within one lead.
- Why this statistic is appropriate here: within-lead subsampling isolates state size from anatomical location.

## Legends

- X axis: LVF state dimension k.
- Y axis: median residual RMS in z units.
- Color/value: one blue series.
- Grouping: one summary per state dimension.
- Ordering/sorting: increasing k.
- Lines/markers/labels: points are means; capped bars are standard deviations.
- Panels: one standalone line plot.

## Interpretation

- Residual magnitude rises with included channels while subset variability narrows.

## Notes

- The full-state point has one realization and therefore no standard-deviation bar.

## References

- `NeuroAnalysis/final_figure.ipynb`.

# llm_error_shrinkage_spanish

## Method

- Load the frozen GPT-2 fitted-linearization summary.
- Compare ID prompts with Spanish-shift OOD prompts at token horizons 1, 2, 4, and 8.
- Plot median residual-RMS reduction contributed by `g(x)`.

## Variables

- Data/input: `NeuroAnalysis/results/final_llmfit_panelD_error_shrinkage.csv`.
- Sessions/groups: GPT-2 layer 6; ID and Spanish OOD prompt banks.
- Labels/targets: ID and OOD.
- Signals/features/measures: median residual-RMS reduction in ID-normalized z units.
- Parameters/thresholds: token horizons 1, 2, 4, and 8; positive is improvement.
- Outputs: `plots/llm_error_shrinkage_spanish.pdf` and `.png`.

## Statistics

- Tests/models: descriptive medians from frozen ridge-plus-residual models; no inferential test.
- Null hypothesis: reduction equals zero.
- Alternative hypothesis: positive reduction indicates improved median prediction.
- Thresholds/decision rule: zero separates improvement from worsening.
- What the statistic means: contribution of the learned residual to hidden-state prediction.
- Why this statistic is appropriate here: it is the direct LLM analog of the neural error-shrinkage measure.

## Legends

- X axis: token prediction horizon.
- Y axis: median residual-RMS reduction in z units.
- Color/value: blue is ID; terracotta is OOD.
- Grouping: paired bars by horizon.
- Ordering/sorting: increasing horizon.
- Lines/markers/labels: horizontal zero reference.
- Panels: one standalone Spanish-shift plot.

## Interpretation

- OOD benefit disappears and reverses at longer horizons despite positive ID reduction.

## Notes

- Single model and primary layer; 50 prompts per bank in the source analysis.

## References

- `NeuroAnalysis/final_figure_llm_fitted.ipynb`.

# llm_error_shrinkage_long_context

## Method

- Repeat the fitted GPT-2 error-shrinkage comparison using long-context OOD prompts.

## Variables

- Data/input: `NeuroAnalysis/results/final_llmfit_panelD_error_shrinkage.csv`.
- Sessions/groups: GPT-2 layer 6; ID and long-context OOD prompt banks.
- Labels/targets: ID and OOD.
- Signals/features/measures: median residual-RMS reduction in ID-normalized z units.
- Parameters/thresholds: token horizons 1, 2, 4, and 8; positive is improvement.
- Outputs: `plots/llm_error_shrinkage_long_context.pdf` and `.png`.

## Statistics

- Tests/models: descriptive medians from frozen models; no inferential test.
- Null hypothesis: reduction equals zero.
- Alternative hypothesis: positive reduction indicates improved prediction.
- Thresholds/decision rule: zero separates improvement from worsening.
- What the statistic means: change in typical hidden-state prediction error attributable to `g(x)`.
- Why this statistic is appropriate here: it matches the brain transfer panel while changing only the OOD prompt family.

## Legends

- X axis: token prediction horizon.
- Y axis: median residual-RMS reduction in z units.
- Color/value: blue is ID; terracotta is OOD.
- Grouping: paired bars by horizon.
- Ordering/sorting: increasing horizon.
- Lines/markers/labels: horizontal zero reference.
- Panels: one standalone long-context plot.

## Interpretation

- The residual correction helps OOD only at short horizons and hurts at 4–8 tokens.

## Notes

- Single model and primary layer.

## References

- `NeuroAnalysis/final_figure_llm_fitted.ipynb`.

# llm_state_size_across_layers

## Method

- Load fixed-128-dimension GPT-2 state summaries at a four-token horizon.
- Plot median linear-model residual RMS separately for layers 1–12.

## Variables

- Data/input: `NeuroAnalysis/results/final_llmfit_panelF_state_size.csv`.
- Sessions/groups: GPT-2 layers 1–12.
- Labels/targets: layer identity.
- Signals/features/measures: median residual RMS in ID-normalized z units.
- Parameters/thresholds: 128 hidden dimensions and four-token horizon.
- Outputs: `plots/llm_state_size_across_layers.pdf` and `.png`.

## Statistics

- None; this is a descriptive layer-level scatter.
- Null hypothesis: not tested.
- Alternative hypothesis: not tested.
- Thresholds/decision rule: none.
- What the statistic means: typical linearization error at a fixed observed dimension across depth.
- Why this statistic is appropriate here: fixing dimension isolates layer/depth differences.

## Legends

- X axis: transformer layer.
- Y axis: median residual RMS in z units.
- Color/value: ordered layer color.
- Grouping: one point per layer.
- Ordering/sorting: layer 1 through 12.
- Lines/markers/labels: each point is annotated with its layer.
- Panels: one standalone scatter plot.

## Interpretation

- Later layers generally exhibit larger fitted-linearization residuals, with local non-monotonicity.

## Notes

- Layer color is ordinal and does not encode a second measurement.

## References

- `NeuroAnalysis/final_figure_llm_fitted.ipynb`.

# llm_state_size_hidden_dimensions

## Method

- Load repeated hidden-dimension subset summaries at GPT-2 layer 6 and a four-token horizon.
- Plot mean median residual RMS by observed state dimension with subset standard deviation.

## Variables

- Data/input: `NeuroAnalysis/results/final_llmfit_panelF_state_size.csv`.
- Sessions/groups: random hidden-dimension subsets at layer 6.
- Labels/targets: state dimension k.
- Signals/features/measures: mean median residual RMS and subset-to-subset standard deviation.
- Parameters/thresholds: k = 8 through 768 on a base-2 logarithmic axis; ten subsets except full dimension.
- Outputs: `plots/llm_state_size_hidden_dimensions.pdf` and `.png`.

## Statistics

- Tests/models: descriptive mean and standard deviation across subset refits.
- Null hypothesis: not tested.
- Alternative hypothesis: not tested.
- Thresholds/decision rule: none.
- What the statistic means: typical prediction error as more hidden-state dimensions are observed.
- Why this statistic is appropriate here: within-layer subsampling isolates observation dimension from network depth.

## Legends

- X axis: observed state dimension k on a base-2 logarithmic scale.
- Y axis: median residual RMS in z units.
- Color/value: one blue series.
- Grouping: one summary per dimension.
- Ordering/sorting: increasing k.
- Lines/markers/labels: points are means; capped bars are standard deviations.
- Panels: one standalone line plot.

## Interpretation

- Residual RMS falls sharply as the observed hidden state approaches full dimension and then saturates.

## Notes

- The full 768-dimensional state has one realization and no standard-deviation bar.

## References

- `NeuroAnalysis/final_figure_llm_fitted.ipynb`.
