# Calibration Plots

This unit replaces the appendix calibration tables with one heatmap figure per
benchmark.  The authoritative manuscript index is
`paper/ICRL2027/iclr2027_conference.tex`, section **Datasets, Evaluation
Protocols, and Full Results**.  `rudra.tex` is intentionally not used.

## Outputs

- `plots/truthfulqa_calibration.{png,pdf}`
- `plots/harmbench_calibration.{png,pdf}`
- `plots/mgsm_calibration.{png,pdf}`
- `plots/lciteeval_calibration.{png,pdf}`

Every model row shows a compact, benchmark-consistent subset: fluency, the
final calibration score, and the attenuation boundary `gamma_star`, plus the
benchmark's primary validation outcome where it was recorded consistently.
Rows are `Q/R`; columns are `Q_f/R`; `R=1` in every sweep.
Grayscale intensity is normalized independently within a panel, so exact cell
annotations—not cross-panel shade—carry numerical meaning. Darker always means
better: higher for scored outcomes and lower for `gamma_star`. The selected
configuration is outlined and starred in the repository teal (`#007C7C`).
Outputs are tightly cropped to their contents rather than forced onto a shared
canvas. Model names are vertical, and the four-model TruthfulQA suite uses a
2-by-2 model layout so it does not become an anomalously tall appendix figure.

## Provenance

Metric surfaces come directly from each benchmark/model `selection.json`.
`gamma_star` is recomputed for every grid point from that selection's frozen
`controller_diagnostics/.../calibration_input.pt`. The script asserts that the
reconstructed selected-cell value matches the diagnostic `score.json` whenever
the latter is present. Cached reconstructions live in `cache/`; no language
model, generation, judge, or benchmark evaluation is rerun.

## Rebuild

From `code/` with the repository environment active:

```bash
python figs/calibration_plots/calibration_plots.py
```

Use `--refresh-gamma` only when the frozen diagnostic inputs or controller
synthesis implementation intentionally change.
