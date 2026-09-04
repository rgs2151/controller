# Data

`data/` contains organized, analysis-ready prompt, activation, intervention, and evaluation records for the robust-steerability experiments. Large arrays remain local and are ignored by git; this README is the tracked inventory and schema contract.

## Structure

```text
data/
  representation_dynamics/
    residual_trajectories.npz
  prompts/
    fit.jsonl
    calibration.jsonl
    test.jsonl
  evaluations/
    steering_generations.parquet
```

- `prompts/`: disjoint prompt records used to fit semantic directions and nominal dynamics, calibrate disturbance geometry, and evaluate held-out steering.
- `representation_dynamics/`: aligned hidden-state deviations, one-step dynamics residuals, prompt splits, and shift-condition labels.
- `evaluations/`: generated-text outcomes and behavior-specific target, collateral, and intervention-energy measurements.

## Residual-Trajectory Schema

`representation_dynamics/residual_trajectories.npz` contains one model--behavior dataset with:

- `residuals`: float array shaped `(records, layers, state_dimensions)` containing `x[k+1] - A[k] x[k] - B[k] u[k]`.
- `state_deviations`: float array with the same shape containing deviation states aligned to each residual.
- `prompt_split`: string array shaped `(records,)` with values `fit`, `calibration`, or `test`.
- `shift_condition`: string array shaped `(records,)` with values `id`, `paraphrase`, `ood`, or `adversarial`.
- `normalized_depth`: float array shaped `(layers,)`, ordered from the earliest analyzed layer to the latest and scaled to `[0, 1]`.
- `model_id`: scalar string containing the exact model repository and checkpoint identifier.
- `behavior`: scalar string naming the steering target.

## Record Hierarchy

- A prompt record is uniquely identified by `prompt_id` within a dataset version.
- Each activation record is keyed by `prompt_id`, exact `model_id`, `behavior`, token position, generation step, and layer index.
- Each evaluation record is keyed by the same identifiers plus controller and intervention setting.

## Alignment Rules

- Preserve prompt order through tokenization, activation extraction, controller rollout, and evaluation; join records by `prompt_id`, never by row position alone.
- Record the exact model revision, tokenizer revision, prompt text, tokenized length, steered token position, generation step, and layer index with every extracted trajectory.
- Align residuals and deviation states so residual at layer `k` predicts the stored deviation at layer `k + 1` under the same prompt, token position, and intervention.
- Never use test prompts to estimate semantic directions, nominal dynamics, disturbance bases, normalization statistics, controller weights, or thresholds.
- Represent layer position as `layer_index / number_of_analyzed_layers` for cross-model plots while retaining the original integer layer index.

## Missing-Data Rules

- Do not impute missing activations, residuals, generations, or behavior labels.
- Exclude only the affected aligned trajectory or comparison and record the exclusion reason in the owning compact-unit README.
- Do not silently replace failed generations or non-finite measurements with zero.

## Inventory

| name | records | notes |
| --- | ---: | --- |
| `prompts/fit.jsonl` | 0 | Fit split; not assembled yet. |
| `prompts/calibration.jsonl` | 0 | Disturbance-calibration split; not assembled yet. |
| `prompts/test.jsonl` | 0 | Held-out evaluation split; not assembled yet. |
| `evaluations/steering_generations.parquet` | 0 | Controller evaluation table; not generated yet. |
