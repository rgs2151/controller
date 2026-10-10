# Robust Steering

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white) ![PyTorch 2.9.1](https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?logo=pytorch&logoColor=white) ![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-007C7C)

Closed-loop control for steering language-model representations under distribution shift.

Robust Steering implements finite-horizon H∞ and LQR controllers, model activation interventions, and benchmark pipelines for evaluating steering effectiveness and capability preservation. Controllers are synthesized from identified layer-to-layer dynamics and applied during generation using the model's current representation. The repository contains the installable `robust_steerability` package, benchmark configurations and results, and scripts for reproducing the accompanying figures and analyses.

[Installation](#installation) · [Running benchmarks](#running-benchmarks) · [Controller API](#controller-api) · [Results and figures](#results-and-figures) · [Repository layout](#repository-layout)

## Installation

### Install from source

Use Python 3.11 or newer. Clone the repository and install the package in your Python environment:

```bash
git clone https://github.com/rgs2151/controller.git
cd controller
python -m pip install -e .
```

The distribution name is `robust-steerability`; the Python import is `robust_steerability`. The installation specifies PyTorch 2.9.1 and Transformers 4.57.1 in [pyproject.toml](pyproject.toml).

Alternatively, create the supplied Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate robust-steerability
```

For Jupyter support in an existing environment:

```bash
python -m pip install -e '.[notebook]'
```

### Prepare model and scorer access

Controller synthesis can run on CPU. Benchmark artifact fitting, calibration, and generation require CUDA GPUs with sufficient memory for the selected model. Some scoring stages also load local judge models and require GPU memory.

For gated models, obtain access to the corresponding Hugging Face model repository and authenticate:

```bash
hf auth login
```

OpenAI-based scoring requires an API key in the environment. Set it before running any stage that uses an API judge, including calibration when its selection metric requires one:

```bash
export OPENAI_API_KEY="your-api-key"
```

API judging incurs usage costs. Judge models and scoring settings are specified in each benchmark's `benchmark.toml`. For L-CiteEval citation scoring, install the sentence-segmentation resource:

```bash
python -m nltk.downloader punkt_tab
```

## Running benchmarks

Run the commands below from the repository root. Each benchmark has the same four-stage interface:

| Stage | What it does |
|---|---|
| `artifacts` | Prepare datasets and fit the semantic target, nominal dynamics, and shared controller artifacts. |
| `calibrate` | Select method-specific controller settings using the benchmark's calibration protocol. |
| `evaluate` | Generate and cache responses using the selected settings. |
| `score` | Score saved responses and write result summaries without repeating generation. |

The `original` method is the unsteered model, `alqr` is A-LQR, and `h_infinity` is the H∞ controller. Additional baseline methods are benchmark-specific. Calibration produces settings that are reused during evaluation; changing an evaluation condition does not itself recalibrate the controller.

### Example: TruthfulQA

This example compares the original model, A-LQR, and H∞ on English prompts and Spanish-input transfer, with English output:

```bash
python -m robust_steerability.benchmarks.truthfulness artifacts --model llama8b --devices auto
python -m robust_steerability.benchmarks.truthfulness calibrate --model llama8b --methods alqr,h_infinity --devices auto
python -m robust_steerability.benchmarks.truthfulness evaluate --model llama8b --methods original,alqr,h_infinity --datasets id,spanish --devices auto
python -m robust_steerability.benchmarks.truthfulness score --model llama8b --methods original,alqr,h_infinity --datasets id,spanish --scorers default --devices auto
```

### Choose a benchmark

Use the module key and model key from the following table. Dataset keys are the configured default evaluation conditions; the linked protocols document the full model lists, baselines, metrics, and experiment settings.

| Benchmark | Module key | Example model key | Default dataset keys | Steering and evaluation |
|---|---|---|---|---|
| [TruthfulQA](benchmarks/truthfulness/README.md) | `truthfulness` | `llama8b` | `id,spanish` | Truthfulness under English-to-Spanish prompt-language shift, with English output. |
| [HarmBench](benchmarks/harmful/README.md) | `harmful` | `llama32_1b_instruct` | `direct,human_jailbreak` | Safe refusal on direct harmful requests and jailbreak variants. |
| [MGSM](benchmarks/mgsm/README.md) | `mgsm` | `qwen3_4b` | `zh,fr,ja,sw,te` | Spanish-language output steering while measuring mathematical accuracy across input languages. |
| [L-CiteEval](benchmarks/lciteeval/README.md) | `lciteeval` | `qwen25_3b_instruct` | `8k,16k` | Positive-sentiment steering while measuring answer and citation quality in long contexts. |
| [Toxicity](benchmarks/toxicity/README.md) | `toxicity` | `gemma2b` | `rtp` | Reduced-toxicity steering on RealToxicityPrompts. |

For example, run the full HarmBench pipeline with its configured default methods and datasets:

```bash
BENCHMARK=harmful
MODEL=llama32_1b_instruct

python -m robust_steerability.benchmarks.${BENCHMARK} artifacts --model "$MODEL" --devices auto
python -m robust_steerability.benchmarks.${BENCHMARK} calibrate --model "$MODEL" --devices auto
python -m robust_steerability.benchmarks.${BENCHMARK} evaluate --model "$MODEL" --devices auto
python -m robust_steerability.benchmarks.${BENCHMARK} score --model "$MODEL" --scorers default --devices auto
```

To run MGSM, L-CiteEval, or Toxicity, change `BENCHMARK` and `MODEL` to the corresponding keys above. Omitting `--methods` and `--datasets` uses that benchmark's defaults, not every registered method or condition.

### Configure a run

Inspect the command-line options before launching an experiment:

```bash
python -m robust_steerability.benchmarks.harmful --help
```

| Option | Usage |
|---|---|
| `--model` | Select a registered model key; required for every stage. |
| `--methods` | Select comma-separated methods, such as `original,alqr,h_infinity`. |
| `--datasets` | Select comma-separated evaluation conditions from the benchmark configuration. |
| `--devices` | Use `auto` for visible GPUs, `cuda:0` for one GPU, or `cuda:0,cuda:1` for multiple GPUs. |
| `--calibration-id` | Select a named calibration, such as `selected`; use the same ID for calibration, evaluation, and scoring. |
| `--kv-cache` | Select `off` or `on`; the default is `off`, with separate evaluation and result namespaces. |
| `--scorers` | On the scoring stage, use `default` or a comma-separated list of supported scorer keys. |
| `--generation-batch-size` | Set the generation batch size to suit available GPU memory. |

GPU scheduling and model-parallel support depend on the benchmark and model. Sample limits, controller overrides, and API concurrency options are listed in each module's `--help`. The [benchmark configurations](benchmarks/) specify the protocols; [DECISIONS.md](DECISIONS.md) records shared reproducibility conventions and benchmark-specific choices.

## Controller API

The controller API can be used independently of language-model loading or benchmark execution. This CPU example synthesizes an H∞ controller for a four-step, two-dimensional system:

```python
import torch
from robust_steerability import FiniteHorizonControlProblem, HInfinityController

horizon, dimension = 4, 2
identity = torch.eye(dimension)
layers = identity.repeat(horizon, 1, 1)

problem = FiniteHorizonControlProblem(
    dynamics=0.9 * layers,
    control_channels=layers,
    disturbance_channels=0.1 * layers,
    state_costs=layers,
    control_costs=layers,
    terminal_cost=identity,
)
controller = HInfinityController.synthesize(problem, device="cpu")
solution = controller.solution()

state_deviation = torch.tensor([[1.0, -0.5]])
delta = controller.intervention(0, state_deviation)
print("gamma_star:", solution.gamma_star)
print("activation intervention:", delta)
```

`control(layer_index, state_deviation)` returns control coordinates. `intervention(...)` maps those coordinates through the control channel to produce the activation intervention. The synthesized gains already include their sign and should be applied directly. For model generation, the runtime policies connect the controller to model activations and semantic setpoints.

See the [controller documentation](robust_steerability/control/README.md) for mathematical contracts and synthesis diagnostics, and the [package documentation](robust_steerability/README.md) for model adapters and runtime integration.

## Results and figures

### Locate experiment outputs

The benchmark pipelines separate scientific caches from result summaries. On a local checkout, the main locations are:

| Location | Contents |
|---|---|
| `benchmarks/<benchmark>/benchmark.toml` | Registered models, methods, evaluation conditions, and protocol settings. |
| `benchmarks/<benchmark>/cache/<model>/datasets/` | Materialized dataset inputs. |
| `benchmarks/<benchmark>/cache/<model>/artifacts/` | Fitted dynamics and shared controller artifacts. |
| `benchmarks/<benchmark>/cache/<model>/calibrations/` | Method-specific calibration artifacts. |
| `benchmarks/<benchmark>/cache/<model>/evaluations/kv_cache_off/` | Cached generations and scoring artifacts for KV-cache-off runs. |
| `benchmarks/<benchmark>/results/kv_cache_off/` | Exported benchmark result summaries. |
| `logs/<run-id>/` | Run manifests, parameters, completion status, and execution records. |

KV-cache-on runs use the corresponding `kv_cache_on` directories. Existing compatible artifacts and completed outputs are reused by the pipelines. Keep the caches if you plan to rescore generations or revisit an experiment. Hugging Face model downloads are stored separately from fitted dynamics, controller artifacts, and response caches.

On Lightning environments, benchmark caches use persistent `robust-steering-cache` storage instead of the local cache directory. `LIGHTNING_ARTIFACTS_DIR` overrides its base directory; each run manifest records the resolved storage paths.

### Reproduce tables and plots

Figure and table implementations live in `figs/`, alongside their documentation and outputs. The benchmark-table script produces Markdown, LaTeX, and PDF files and requires a LaTeX installation with `latexmk`. Regenerate the tables from available result summaries with:

```bash
python figs/bench_table/bench_table.py
```

For individual plots, use the rendering command and input paths documented in the relevant figure folder. Rendering from existing result summaries does not require rerunning model generation. Analyses that depend on activations or other compute artifacts require their corresponding caches.

Model weights, raw datasets, and large compute caches are not included in the source repository. Consult the benchmark protocols and [data inventory](data/README.md) for the inputs required by a particular experiment.

## Repository layout

`robust_steerability/` is the Python package. The remaining folders contain benchmark specifications, reproducibility materials, and supplementary research; they are not required to import the controller API.

| Folder | Purpose |
|---|---|
| `robust_steerability/` | Controllers, model adapters, calibration, runtime policies, datasets, benchmarks, and judges. |
| `benchmarks/` | Benchmark configurations, protocol documentation, local caches, and result summaries. |
| `figs/` | Figure and table scripts, documentation, and rendered outputs. |
| `parking/` | Supplementary analyses and exploratory experiments. |
| `debug/` | Diagnostic analyses and investigations. |
| `ref/` | Reference implementations and historical material. |
| `data/` | Local datasets and data inventory. |
| `docs/` | Technical notes and experiment designs. |
| `logs/` | Experiment manifests and execution records. |
| `server/` | Remote environment setup and execution support. |
| `tmp/` | Temporary local artifacts. |

Within the package, `control/` implements controller synthesis, `modeling/` handles activation capture and intervention hooks, `calibration/` fits semantic targets and dynamics, and `runtime/` applies closed-loop control during generation. `benchmarks/`, `datasets/`, `judges/`, and `experiments/` provide the evaluation pipelines and shared experiment utilities.

## Attribution

The [source-method documentation](robust_steerability/source_methods/README.md) records the provenance of incorporated baseline implementations. Reference code retains its upstream attribution and applicable licenses. Benchmark documentation identifies the datasets, evaluation methods, and scoring models used in each protocol.
