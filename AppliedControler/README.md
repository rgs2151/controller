# AppliedControler

This folder is the staging area for controller-application experiments on the merged 10-model benchmark set.

## Ready Assets

- Top OOD target per model: `selected_ood_targets_top1.csv`
- Top-3 OOD targets per model: `selected_ood_targets_top3.csv`
- Method adapter scaffold: `method_registry.py`
- Experiment-plan generator: `plan_experiments.py`

## Existing Controller Implementations Found In Repo

- A-LQR implementation core:
  - `robust_steerability/control/lqr.py`
  - `robust_steerability/control/__init__.py`
- PID steering implementation core:
  - `robust_steerability/control/pid.py`
  - `robust_steerability/control/__init__.py`
- H-infinity implementation core:
  - `robust_steerability/control/h_infinity.py`
  - `robust_steerability/control/__init__.py`
- Activation addition baseline:
  - `robust_steerability/control/activation_addition.py`
- Controller tests:
  - `tests/test_control.py`

## Reference Material Found

- Paper snapshot: `ref/2604.19018v1.pdf`
- Reference note: `ref/README.md`

Note: `ref/README.md` mentions an upstream checkout (`lqr-activation-steering`) but that directory is not currently present in this workspace.

## Next Step Inputs Needed

To implement and compare methods here:

1. GitHub URL(s) for the two standard methods you want matched.
2. Your new method specification:
   - state/control definition
   - update rule/equations
   - required hyperparameters
   - any offline fit stage and online runtime stage
3. Desired evaluation protocol:
   - prompt counts per benchmark
   - decoding settings
   - metric priority (overall/early/mid/late or custom)

Once you share those, we can build a unified runner in this folder and evaluate on the selected OOD targets.

## Quick Start (Ready Now)

Generate a Top-1 plan with built-in methods (A-LQR, S-PID, new_method=H-infinity):

```bash
python AppliedControler/plan_experiments.py
```

Generate a Top-3 plan and include deferred placeholders for your two standard methods + new method:

```bash
python AppliedControler/plan_experiments.py \
  --targets AppliedControler/selected_ood_targets_top3.csv \
  --output AppliedControler/experiment_plan_top3.json \
  --include-deferred
```

Note: `new_method` is the in-repo H-infinity implementation. The default plan
now compares exactly three methods: A-LQR, S-PID, and H-infinity (`new_method`).

## Run Steering (3-Method Comparison)

Run a quick preflight first (checks plan triplets and torch runtime):

```bash
python AppliedControler/run_steering.py \
  --plan AppliedControler/experiment_plan_top1.json \
  --preflight
```

Run a first steering pass on Top-1 targets:

```bash
python -m AppliedControler.run_steering \
  --plan AppliedControler/experiment_plan_top1.json \
  --output-csv AppliedControler/steering_results_top1.csv \
  --output-json AppliedControler/steering_results_top1.json \
  --quantized \
  --limit-experiments 3 \
  --eval-prompts 12 \
  --calibration-prompts 16 \
  --jacobian-prompts 4
```

This runs method comparison for: `alqr`, `spid`, and `new_method` (H-infinity).

## Final Table + Plots

Generate a final summary table bundle and three plots from a steering JSON output:

```bash
python -m AppliedControler.report_steering_results \
  --input-json AppliedControler/steering_results_top1_cpu.json \
  --output-dir AppliedControler/results_reports \
  --tag top1_cpu
```

Outputs include:
- method summary table CSV
- target winner table CSV
- ranked markdown report
- paper-style per-model table (CSV + Markdown)
- mean-delta bar chart with confidence intervals
- percent-change bar chart with confidence intervals
- actual toxicity percentage chart (baseline vs steered)
- per-method delta distribution plot
- per-target heatmap across methods
