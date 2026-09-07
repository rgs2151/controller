# Control-Theory Collaborator Workplan

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-07
- Verification Status: UNVERIFIED
- Version Label: code_plan_v4

## Fixed rule

No figure or experiment has two owners. The collaborator never writes or debugs Hugging Face, tokenizer, model-hook, prompt-loading, or generation code.

The existing H∞ implementation is the fixed controller under test. Ordinary validation and benchmark runs do not require editing `robust_steerability/control/h_infinity.py`.

The figure specification in `figs/sketch/` is frozen at commit `415c45b`. Real result units may follow it but must not modify it.

## Required before the handoff: a turnkey benchmark pack

The LLM-side pipeline must first be wrapped in one config-driven command. The wrapper—not the collaborator—contains all model, prompt, method, condition, calibration-size, gain, seed, caching, GPU, and plotting loops.

The implemented manifest interface is:

```bash
python -m robust_steerability.experiments run --manifest parking/erfan_toxicity_calibration/manifest.json --devices cuda:0,cuda:1
python -m robust_steerability.experiments run --manifest parking/erfan_truthfulness_benchmark/manifest.json --devices cuda:0,cuda:1
python -m robust_steerability.experiments run --manifest parking/erfan_id_toxicity_benchmark/manifest.json --devices cuda:0,cuda:1
python -m robust_steerability.experiments run --manifest parking/erfan_ood_steering_benchmark/manifest.json --devices cuda:0,cuda:1
```

Erfan's original implementation and outputs are archived unchanged under
`ref/erfan_applied_controller/`. The maintained runner now lives in
`robust_steerability.experiments`; new figure suites should be added as manifests
instead of reviving the archived package.

The pack must provide:

- one frozen experiment manifest listing models, datasets, conditions, methods, seeds, and sweep values;
- a validated one-model smoke run;
- resumable jobs with one status/log file per job;
- separate collaborator cache and result directories;
- automatic execution of Original, A-LQR, S-PID, and H∞ when a figure needs them;
- automatic metrics and plot generation;
- no source-code edits for ordinary runs.

Connecting the existing H∞ controller to this pack is an LLM-pipeline infrastructure task and must be finished before the handoff. It is not assigned to the collaborator.

The user shows her the smoke command once. After that, running the matrix is operational work, not Hugging Face development.

## Collaborator ownership

### 1. H∞ validation — no routine controller edits

She owns:

- H∞ tests in `tests/test_control.py`;
- independent numerical validation code and Figure S1;
- checking the mathematical validity of gamma-star, feasibility, gains, and certificates produced by the existing controller.

Tasks:

1. Treat `robust_steerability/control/h_infinity.py` as fixed while auditing the finite-horizon minimax recursion, feedback sign, terminal cost, and feasibility conditions.
2. Validate scalar, MIMO, rectangular-channel, and time-varying systems.
3. Compare gamma-star with the independent induced-gain calculation on small systems.
4. Add near-singular, infeasible, serialization, dtype, and CPU/GPU tests.
5. If validation exposes a defect, first record a minimal failing case; edit the controller only to repair that confirmed defect.

### 2. H∞ optimization — separate implementation task

After the fixed implementation is validated, she may optimize `robust_steerability/control/h_infinity.py` for memory and runtime while preserving its numerical outputs and public interface.

This is the only planned task that inherently requires her to edit the H∞ implementation. She also produces before/after runtime and peak-memory benchmarks.

### 3. Figure S1 — fully hers, no LLM required

She creates the real H∞ validation unit:

- feasibility margin versus candidate gamma;
- synthesized versus independently computed finite-horizon gain;
- performance-output energy versus disturbance energy and its bound;
- observed tracking error versus certified bound.

### 4. Figure S3 — fully hers through the runner

She defines the mathematically valid controller sweep, then launches the prepared S3 suite:

- gain/weight sweep versus target steering success;
- oversteering, utility retention, and intervention energy;
- calibration-size ablation;
- energy-matched disturbance-geometry ablation.

For H∞, she should resynthesize across valid Q/R/S or other controller parameters rather than blindly scaling a certified gain.

### 5. Figure 3 — fully hers through the runner

She launches the prepared Figure 3 suite, checks job completion, and generates the final figure:

- calibration-only gamma-star and S_rob;
- held-out OOD steering reliability;
- competing predictors;
- cross-validated comparison;
- leave-one-model-family-out prediction.

The manifest and runner enforce the calibration/test split. She does not choose prompts or implement model evaluation.

### 6. Figure 5 — fully hers through the runner

She launches the complete OOD suite and produces every panel. The runner automatically evaluates Original, A-LQR, S-PID, and H∞ under the same settings.

She owns checking that all configured jobs completed, rerunning interrupted jobs with `--resume`, and producing the final OOD plot bundle.

### 7. Figure 6 — fully hers through the runner

She launches the long-context suite and produces target error, utility retention, and H∞-versus-A-LQR advantage across the frozen context lengths.

She does not construct prompts or debug attention/tokenization behavior; those are fixed inside the benchmark pack.

## User ownership

The user owns Figures 1, 2, 4, and S2:

- Figure 1: conceptual framing.
- Figure 2: residual analysis and OOD/adversarial A-LQR failure exploration.
- Figure 4: the complete ID benchmark and intervention-energy tradeoff.
- Figure S2: nonlinear representation dynamics and local-linearity analysis.

The user also owns all underlying LLM infrastructure:

- Hugging Face model/tokenizer loading;
- prompt datasets and split definitions;
- activation and Jacobian hooks;
- residual extraction and disturbance-geometry plumbing;
- generation and behavioral metrics;
- fixes for model-specific failures in the turnkey runner.

## Compute separation

- User runs on GPU 0.
- Collaborator runs the turnkey suites on GPU 1.
- Their caches, logs, status files, and result directories are separate.
- The collaborator never waits for the user's experiment outputs and never imports values from the user's figures.

## What the collaborator actually does with LLM experiments

1. Pull the repository.
2. Run the independent H∞ validation and create Figure S1 without changing the controller during a passing run.
3. Run the demonstrated smoke command.
4. Run the four prepared suite commands on GPU 1.
5. Monitor status files and resume interrupted jobs.
6. Run the plotting command.
7. Investigate controller/numerical failures herself.
8. Send model/tokenizer/hook failures back to the LLM owner with the job log.

That is enough for a software engineer who does not know LLM internals: she operates a tested experiment product instead of becoming responsible for the product's Hugging Face implementation.

## Definition of done

- The collaborator has completed H∞ verification; controller edits were made only for a confirmed defect or the separate optimization task.
- Figures 3, 5, 6, S1, and S3 contain real values and regenerate from her commands.
- Every baseline required by those figures was launched automatically by her suites.
- The user did not manually prepare intermediate results or finish any of her panels.
- The collaborator did not need to modify Hugging Face or model-hook code.
