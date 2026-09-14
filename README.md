# Robust Steerability

Research code for measuring and controlling robust steerability in language-model representation dynamics under distribution shift.

## Setup

Create the environment from the repo root:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate robust-steerability
```

Install the package in editable mode if the environment was created without the pip step:

```bash
python -m pip install -e .
```

## Running Units

Compact units live in `parking/` while they are being explored and in `figs/` after graduation.

Run the cache-first A-LQR residual and generated-toxicity smoke test from the repo root:

```bash
python parking/residual_checks/residual_checks.py all
```

Run the benchmark as three explicit stages on any visible GPU set:

```bash
python -m robust_steerability.benchmarks.truthfulness artifacts --model gemma2b --devices auto
python -m robust_steerability.benchmarks.truthfulness calibrate --model gemma2b --devices auto
python -m robust_steerability.benchmarks.truthfulness evaluate --model gemma2b --devices auto
```

Use `--generation-batch-size <n>` only when a larger remote GPU has been
validated for that batch. The chosen value is recorded and becomes part of the
generation cache identity.

Render the synchronized Markdown and TeX benchmark tables:

```bash
python figs/bench_table/bench_table.py
```

Each method checkpoints complete repetitions inside the owning unit's ignored
`cache/` directory. Repeating a stage resumes only an identity-matched cache.

Each unit owns its own `cache/` and `plots/` folders. Existing caches are reused by default. To recompute a unit, delete that unit's relevant cache or run the unit with `--recompute` when supported.

## Package Architecture

`robust_steerability/` separates controller mathematics from language-model integration:

- `control/`: the common controller interface, finite-horizon problems, and each
  controller's offline and online logic for LQR, PID, activation addition, and
  H-infinity.
- `modeling/`: Hugging Face loading, activation capture, Jacobians, and transformer intervention hooks.
- `calibration/`: semantic targets, nominal dynamics, residual measurements, disturbance geometry, and calibration-only normalization.
- `runtime/`: policies that translate controller outputs into activation interventions.
- `datasets/`: pinned dataset loaders and prompt construction.
- `benchmarks/`: portable artifact, calibration, evaluation, and scoring pipelines.
- `experiments/`: reusable controller-calibration and diagnostic internals.

Controller implementations consume only finite-horizon tensors and expose a
common `control`/`intervention` interface. They do not import Transformers or
interact with model hooks. See `robust_steerability/control/README.md` for the
collaborator guide and the H-infinity extension point.

## Data

Place organized analysis-ready data under:

```text
data/representation_dynamics/
```

See `data/README.md` for the required arrays, record hierarchy, variable meanings, split rules, alignment rules, and current inventory.

## Multi-machine Runs

Enter each remote machine directly with SSH, pull the desired Git commit, open a
named GNU Screen, and invoke the owning benchmark entry point there. Benchmark
code discovers and uses the visible GPUs; `server/` contains no machine names or
scientific run plans.

Git carries code, logs, tables, and plots. Remote S3 carries only large reusable
artifacts such as averaged dynamics and controller `.pt` caches; raw per-prompt
Jacobians are not retained. The local workstation never
connects to S3. See `server/README.md` for the exact operating workflow.

## Project Docs

- `AGENTS.md`: working rules for Codex agents.
- `ORGANIZATION.md`: folder layout, compact-unit pattern, cache rules, and README template.
- `STYLE.md`: figure styling standards.
- `DECISIONS.md`: project-wide scientific and analytical decisions.
- `skills/`: repo-local Rudra workflow skills.
