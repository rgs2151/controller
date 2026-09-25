# Recreation validation

The empirical P12 and GPT-2 Small calculations were rerun locally on
2026-09-24 from the canonical collaborator notebooks preserved under `source/`.

## Inputs

- Brain: P12 NoStim2 and Stim1 MAT sessions from `data/brain/`.
- GPT-2: `gpt2` (GPT-2 Small), the exact 50 Within, 50 Spanish, and 50
  long-context prompts reconstructed from Erfan's historical generator at
  commit `8cd7a02`.
- Seed: 0.
- Splits, normalization, horizons, ridge grid, residual MLP, layer, and state
  dimensions are unchanged from the source notebooks.

## Result agreement

Fresh result tables were compared elementwise with the previously tracked
`NeuroAnalysis/results/` exports. Maximum absolute numeric differences were:

| Result | Maximum absolute difference |
|---|---:|
| Brain error shrinkage | 1.95e-7 |
| Brain context-length/horizon | 1.30e-10 |
| Brain state-size | 3.33e-16 |
| GPT-2 error shrinkage | 2.84e-3 |
| GPT-2 context-length/horizon | 5.25e-7 |
| GPT-2 state-size | 1.05e-6 |

The only visible numerical drift is the residual MLP in GPT-2 Panel D; its
qualitative signs and conclusions are unchanged. Ridge-only panels reproduce to
floating-point precision. The drift is consistent with numerical/library-level
variation in a seeded neural-network fit, rather than a changed dataset or
analysis definition.

## Durable cache boundary

The decoded neural sessions, theta tokens, exact prompts, and all GPT-2 hidden
trajectories are cached under `cache/`. Completed executed notebooks retain the
full cell-level logs under `executed/`. Final numerical outputs and standalone
panels are in `results/` and `plots/`.
