# benchmark_validation

## Method

- Compare the optimized H-infinity recursion with Hannah's unchanged `ref/h_infinity.py` implementation on the same seeded finite-horizon problem.
- Check the three limiting cases required by Kasra's implementation guide: zero disturbance recovers LQR, a large attenuation level recovers LQR, and feasibility is monotone as gamma increases.
- Load pinned DistilGPT-2 and Qwen-2.5-0.5B checkpoints on separate GPUs and compute first- and last-block last-token Jacobians.
- Compare the row-chunked package Jacobian against the preserved A-LQR definition: differentiate with respect to the complete block input and retain only the last-token input columns.
- Write one JSON validation record per model to `cache/`.

## Variables

- Data/input: one fixed validation sentence and one seeded synthetic control problem.
- Sessions/groups: DistilGPT-2 on one GPU and Qwen-2.5-0.5B on the other.
- Labels/targets: the unchanged Hannah H-infinity solver, the preserved A-LQR Jacobian calculation, and nominal LQR gains.
- Signals/features/measures: gamma star, controller gains, fixed-gamma feasibility, maximum absolute Jacobian difference, runtime, and peak allocated GPU memory.
- Parameters/thresholds: relative tolerance `2e-4`, absolute tolerance `2e-5`, H-infinity bisection tolerance `1e-5`, and Jacobian row chunk size 32.
- Outputs: `cache/distil.json` and `cache/qwen.json`.

## Statistics

- Tests/models: deterministic numerical equivalence checks; no fitted statistical model.
- Null hypothesis: the optimized and reference calculations differ beyond the declared floating-point tolerance.
- Alternative hypothesis: every checked output agrees within the declared tolerance and every required limiting behavior holds.
- Thresholds/decision rule: the script raises immediately on any failed equality, non-monotone feasibility sequence, or failed limiting check.
- What the statistic means: maximum absolute difference records the largest elementwise disagreement between two independently invoked implementations.
- Why this statistic is appropriate here: the unit validates numerical implementation parity, not population-level uncertainty.

## Legends

- X axis: not applicable; this unit does not produce a plot.
- Y axis: not applicable.
- Color/value: not applicable.
- Grouping: one JSON record per pinned model and GPU.
- Ordering/sorting: controller gamma values increase; transformer layers are first then last.
- Lines/markers/labels: not applicable.
- Panels: not applicable.

## Interpretation

- `passed: true` means the controller and Jacobian paths satisfy all declared reference checks on that GPU and checkpoint.
- These checks validate the numerical machinery; they do not choose the scientific definition of the H-infinity nominal reference trajectory.

## Notes

- The script reads but never modifies either reference implementation.
- The cache binds results to the package, script, model revisions, and Hannah reference hashes.
- This is a focused implementation diagnostic, not a benchmark result.

## References

- `ref/h_infinity.py`: Hannah's preserved implementation.
- `ref/lqr-activation-steering/steer/lqr_utils.py`: preserved A-LQR Jacobian definition.
- `H_inf_Implementation_Kasra_version.pdf`: solver equations and mandatory limiting checks supplied by the project owner.

# controller_smoke

## Method

- Load one production benchmark manifest and its pinned model on an explicit GPU.
- Fit or verify the exact production controller artifact, including 50 prompt-level Jacobians, A-LQR, S-PID, H-infinity, and all six adapted baseline parameters.
- Generate deterministic 30-token continuations for three held-out ID prompts under every table method.
- Save text and both controller-coordinate and physical hidden-delta energies for manual oversteering inspection.

## Variables

- Data/input: the first three held-out ID records after fit/calibration exclusion.
- Sessions/groups: one model and behavior per invocation.
- Labels/targets: all ten manifest methods.
- Signals/features/measures: decoded continuation, sum of squared controller coordinates, and sum of squared hidden-state additions.
- Parameters/thresholds: 50 fit prompts per class, 50 Jacobian prompts, greedy generation, and 30 output tokens.
- Outputs: `cache/smoke_<behavior>_<model-index>.json` and prompt-level trace tensors.

## Statistics

- Tests/models: deterministic manual diagnostic; no fitted statistical test.
- Null hypothesis: not applicable.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: any empty, repetitive, nonsensical, or generic continuation is inspected before the full run; large energies are interpreted relative to Original and the other methods, not with an invented universal cutoff.
- What the statistic means: energy quantifies the magnitude of the exact applied intervention over forward calls and layers.
- Why this statistic is appropriate here: it catches gross online wiring and oversteering failures before thousands of generations are launched.

## Legends

- X axis: not applicable.
- Y axis: not applicable.
- Color/value: not applicable.
- Grouping: records are grouped by method.
- Ordering/sorting: manifest method order, then fixed prompt order.
- Lines/markers/labels: method and prompt IDs label each record.
- Panels: not applicable.

## Interpretation

- This smoke validates online behavior qualitatively and reuses the resulting controller cache; it is not entered into the paper tables.
- Passing numerical parity does not guarantee sensible generated text, so both diagnostics are required before the full evaluation.

## Notes

- The smoke never chooses parameters from held-out benchmark outcomes.
- The full runner rejects the cache if any production source, manifest, model revision, or controller setting changes.

## References

- `ref/paper_benchmark_50/{toxicity,truthfulness}.json`.
- `robust_steerability/experiments/{calibration,generation,methods}.py`.
