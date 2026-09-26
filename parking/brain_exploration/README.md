# all_subject_conflict_linear_residual

## Method

- Load every `BIPOLFieldTripFormat_AlignedToImagePresent_P*_NoStim*.mat` file
  from `data/brain/` and group files by participant and no-stimulation session.
- For each participant, select the session with the most trials; break ties in
  favor of the later numbered session. Exclude channels marked ictal and retain
  anatomical leads containing at least five clean bipolar channels.
- Band-pass each neural epoch from 4–8 Hz, compute Hilbert power, and average
  power in causal 40 ms windows sampled every 20 ms from 40 to 1500 ms after
  image presentation. Normalize each channel with baseline power estimated
  from −500 to 0 ms using the training trials only.
- Define low-conflict trials (`Conflict <= 1`) as the source distribution and
  high-conflict trials (`Conflict >= 2`) as the shifted distribution. Split the
  low-conflict trials deterministically into 60% training, 20% validation, and
  20% held-out ID test trials with random seed 0.
- Fit a separate affine ridge model predicting future theta-power state from
  current state for every participant, lead, and horizon. Choose the ridge
  penalty from `0.001, 0.01, 0.1, 1, 10, 100` by maximum validation population
  R².
- Measure held-out prediction error as the median channel-root-mean-square
  residual in baseline z units for ID low-conflict trials and OOD high-conflict
  trials at 20, 40, 100, 200, 300, 400, 500, and 700 ms horizons.
- Select one displayed lead per participant by the largest mean proportional
  OOD excess, `(OOD / ID) − 1`, across all horizons; break remaining ties by
  absolute OOD-minus-ID separation and then lead name.
- Write the complete all-lead results, lead ranking, selected participant
  results, participant ranking, and the cross-participant grid in PNG and PDF.

## Variables

- Data/input: no-stimulation aligned bipolar FieldTrip MAT files in
  `data/brain/`.
- Sessions/groups: one deterministically selected no-stimulation session for
  each participant with an eligible lead; P13 is absent because the available
  release contains only a stimulation session.
- Labels/targets: ID is held-out low conflict; OOD is high conflict.
- Signals/features/measures: 4–8 Hz Hilbert power, causal 40 ms windows,
  20 ms stride, affine future-state prediction, and median linear residual RMS.
- Parameters/thresholds: at least five clean bipolar channels per lead;
  horizons `20, 40, 100, 200, 300, 400, 500, 700` ms; deterministic seed 0;
  60/20/20 low-conflict split.
- Outputs: `results/all_subject_all_leads_conflict_linear_residual.csv`,
  `results/all_subject_lead_ranking.csv`,
  `results/all_subject_conflict_linear_residual.csv`,
  `results/all_subject_conflict_ranking.csv`, and
  `plots/all_subject_conflict_linear_residual.{png,pdf}`.

## Statistics

- Tests/models: descriptive affine ridge prediction models are fitted
  separately for every participant, lead, and horizon; ridge strength is chosen
  on low-conflict validation R².
- Null hypothesis: no inferential null hypothesis is tested in this exploratory
  screen.
- Alternative hypothesis: no formal alternative hypothesis or significance
  threshold is used.
- Thresholds/decision rule: the displayed lead maximizes mean proportional OOD
  residual excess across the eight horizons.
- What the statistic means: residual RMS quantifies the typical multichannel
  future-state prediction mismatch left by the locally linear model; values are
  expressed in baseline-standardized theta-power units.
- Why this statistic is appropriate here: it directly compares nominal-model
  predictability between held-out source trials and conflict-shifted trials
  while avoiding domination by a few large residual samples.

## Legends

- X axis: future prediction horizon in milliseconds.
- Y axis: median linear residual RMS in baseline z units; every participant
  panel has an independent y range.
- Color/value: gray `#8A8F94` is held-out low-conflict ID; red `#8B1E1E` is
  high-conflict OOD.
- Grouping: paired ID and OOD bars at each horizon within each participant.
- Ordering/sorting: participants are ordered numerically in a five-column grid;
  horizons increase from 20 to 700 ms.
- Lines/markers/labels: each title reports participant, selected lead, channel
  count, and mean proportional OOD excess.
- Panels: one panel per eligible participant; unused cells at the end of the
  rectangular grid are blank.

## Interpretation

- The grid is an exploratory cross-subject screen for participants and leads
  whose locally linear theta-state prediction error separates high-conflict
  trials from matched held-out low-conflict trials.
- Lead selection is outcome-dependent and therefore descriptive. Any selected
  participant or lead requires confirmation in a separately specified analysis
  before supporting an inferential neuroscience claim.

## Notes

- `brain_exploration.py` is cache-first: cosmetic replots and ranking changes
  can reuse the complete saved all-lead result table without refitting models.
- The `cache/conflict_screen/` copy preserves the no-stimulation theta states
  used for the existing results and future extensions of this exploration.
- This parking unit is independent of the finalized `figs/figure_brain/` unit.
  Work here must not overwrite or regenerate Figure Brain outputs.

## References

- Finalized source figure unit: `figs/figure_brain/`.
- Original collaborator notebook rerun:
  `figs/figure_brain/ref/original_notebook_rerun/`.
