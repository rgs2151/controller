# Robust Steerability

Research code for measuring and controlling robust steerability in language-model representation dynamics under distribution shift.

## Experiment Compute Map

| Benchmark | Run owner | Current state |
|---|---|---|
| Truthfulness | Remote 1 | Completed runs and reusable cache live there. |
| L-CiteEval | Remote 1 | Completed runs and reusable cache live there. |
| MGSM | Remote 2 | Completed runs and reusable cache live there. |
| Toxicity | Local 1 | Gemma-2-2B and Llama-3-8B runs and reusable caches live here. |
| HarmBench | Local 1 | Completed run and reusable cache live here. |

The run owner is the machine that retains each benchmark's ignored scientific
cache. Git synchronizes code, logs, result tables, and plots, but does not move
these machine-owned caches.

## Setup

Create the environment from the repo root:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate robust-steerability
python -m nltk.downloader punkt_tab
```

Install the package in editable mode if the environment was created without the pip step:

```bash
python -m pip install -e .
```

`punkt_tab` is the pinned L-CiteEval sentence-segmentation resource used before
AutoAIS citation scoring; install it once in every new environment.

## Running Units

Compact units live in `parking/` while they are being explored and in `figs/` after graduation.

Run the cache-first A-LQR residual and generated-toxicity smoke test from the repo root:

```bash
python parking/residual_checks/residual_checks.py all
```

Run the truthfulness benchmark as four explicit stages on any visible GPU set:

```bash
python -m robust_steerability.benchmarks.truthfulness artifacts --model gemma2b --devices auto
python -m robust_steerability.benchmarks.truthfulness calibrate --model gemma2b --devices auto
python -m robust_steerability.benchmarks.truthfulness evaluate --model gemma2b --devices auto
python -m robust_steerability.benchmarks.truthfulness score --model gemma2b --scorers default --devices auto
```

L-CiteEval uses the same four-stage interface for Qwen2.5-3B-Instruct and
Llama-3.1-8B-Instruct:

```bash
python -m robust_steerability.benchmarks.lciteeval artifacts --model qwen25_3b_instruct --devices auto
python -m robust_steerability.benchmarks.lciteeval calibrate --model qwen25_3b_instruct --devices auto
python -m robust_steerability.benchmarks.lciteeval evaluate --model qwen25_3b_instruct --devices auto
python -m robust_steerability.benchmarks.lciteeval score --model qwen25_3b_instruct --scorers default --devices auto
```

MGSM language transfer uses the same interface for the completed Qwen3-4B run
and the planned Gemma-3-4B-Instruct run. The Gemma run evaluates only S-PID,
A-LQR, and H-infinity:

```bash
python -m robust_steerability.benchmarks.mgsm artifacts --model gemma3_4b_it --methods spid,alqr,h_infinity --devices auto
python -m robust_steerability.benchmarks.mgsm calibrate --model gemma3_4b_it --methods spid,alqr,h_infinity --devices auto
python -m robust_steerability.benchmarks.mgsm evaluate --model gemma3_4b_it --methods spid,alqr,h_infinity --devices auto
python -m robust_steerability.benchmarks.mgsm score --model gemma3_4b_it --methods spid,alqr,h_infinity --scorers default --devices auto
```

HarmBench robust refusal initially uses Llama-3.2-1B-Instruct:

```bash
python -m robust_steerability.benchmarks.harmful artifacts --model llama32_1b_instruct --devices auto
python -m robust_steerability.benchmarks.harmful calibrate --model llama32_1b_instruct --devices auto
python -m robust_steerability.benchmarks.harmful evaluate --model llama32_1b_instruct --devices auto
python -m robust_steerability.benchmarks.harmful score --model llama32_1b_instruct --scorers default --devices auto
```

Use `--generation-batch-size <n>` only when a larger remote GPU has been
validated for that batch. The chosen value is recorded in the stage log.

Render the synchronized Markdown and TeX benchmark tables:

```bash
python figs/bench_table/bench_table.py
```

Each method checkpoints complete repetitions inside the owning benchmark's
ignored `cache/` directory. The directory hierarchy is the run index. Delete
the exact indexed directory when a run must be recomputed.

## Package Architecture

`robust_steerability/` separates controller mathematics from language-model integration:

- `control/`: the common controller interface, finite-horizon problems, and each
  controller's offline and online logic for LQR, PID, activation addition, and
  H-infinity.
- `modeling/`: Hugging Face loading, activation capture, Jacobians, and transformer intervention hooks.
- `calibration/`: semantic targets, nominal dynamics, residual measurements, disturbance geometry, and calibration-only normalization.
- `runtime/`: policies that translate controller outputs into activation interventions.
- `datasets/`: pinned dataset loaders and prompt construction.
- `benchmarks/`: the universal artifact, calibration, generation-only evaluation,
  and independent scoring pipeline.
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

Git carries code, logs, tables, and plots. On Lightning, every ignored benchmark
cache—datasets, averaged dynamics, calibrations, generations, scorer outputs,
and controller diagnostics—lives in the producing Studio's Teamspace Drive
storage under `~/robust-steering-cache/`. Other Studios in the Teamspace can
read those files through the Drive. The local workstation continues to use the
benchmark unit's local ignored cache. See `server/README.md` for the exact
operating workflow.

## Project Docs

- `AGENTS.md`: working rules for Codex agents.
- `ORGANIZATION.md`: folder layout, compact-unit pattern, cache rules, and README template.
- `STYLE.md`: figure styling standards.
- `DECISIONS.md`: project-wide scientific and analytical decisions.
- `skills/`: repo-local Rudra workflow skills.
