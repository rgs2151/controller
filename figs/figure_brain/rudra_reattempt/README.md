# Figure Brain — Rudra reattempt

This directory is an independent reconstruction of the full brain/LLM
comparison figure. It does not modify or depend on the legacy composite
exports in `../plots/`.

## Construction

- `figure_brain_reattempt.py` contains one native Matplotlib draw function per
  panel.
- The complete 4-by-3 composition and every standalone panel call the same
  draw functions.
- No rendered plot, PDF, or screenshot is embedded in the composite.
- Quantitative panels read only the frozen CSV summaries in
  `NeuroAnalysis/results/`.
- The electrode and architecture panels are native vector schematics in this
  first layout pass. Raw MAT files are deliberately not read while they are
  being transferred.

## Organization

| Row | Left | Middle | Right |
|---|---|---|---|
| Brain 1 | A: sEEG state | C: cortical state model | D: residual vs channels |
| Brain 2 | B: MSIT shifts | E: conflict shift | F: stimulation-context shift |
| LLM 1 | G: GPT-2 architecture | I: 12 layer-wise local models | J: residual vs layer |
| LLM 2 | H: prompt shifts | K: Spanish shift | L: long-context shift |

## Outputs

- `plots/figure_brain_reattempt.pdf`
- `plots/figure_brain_reattempt.png`
- `plots/panels/*.pdf` and `*.png`

