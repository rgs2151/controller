# Robust Steering

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![PyTorch 2.9.1](https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?logo=pytorch&logoColor=white)
![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-007C7C)

**Feedback control for steering language-model representations under distribution shift.**

Robust Steering provides finite-horizon H-infinity and LQR controllers, activation
interventions, and reproducible benchmark pipelines. It identifies local
layer-to-layer dynamics from model activations, estimates disturbance geometry
from prediction residuals, and synthesizes feedback controllers before deployment.
The Python distribution and import names are `robust-steerability` and
`robust_steerability`, respectively.

## Installation

Python 3.11 or newer is required. From a source checkout:

```bash
git clone https://github.com/rgs2151/controller.git
cd controller
python -m pip install -e .
```

For the notebook environment:

```bash
conda env create -f environment.yml
conda activate robust-steerability
```

Controller synthesis can run on CPU. Language-model experiments generally require
a compatible CUDA GPU and sufficient memory for the selected model. Gated model
repositories require Hugging Face access; OpenAI-based scorers require an
`OPENAI_API_KEY`. L-CiteEval citation scoring also needs the sentence-segmentation
resource:

```bash
python -m nltk.downloader punkt_tab
```

## Quick start

Synthesize a controller for a small finite-horizon system without loading a
language model:

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
print(solution.gamma_star, delta)
```

`control(layer_index, state_deviation)` returns control coordinates;
`intervention(...)` maps them through the control channel into an activation
delta. Returned gains already include the feedback sign and are applied directly.
See the [controller guide](robust_steerability/control/README.md) for the numerical
contracts, supported controllers, and synthesis diagnostics.

## Package architecture

| Module | Purpose |
|---|---|
| `control/` | Model-independent H-infinity, LQR, PID, and activation-addition controllers |
| `modeling/` | Model loading, activation capture, Jacobians, and intervention hooks |
| `calibration/` | Semantic targets, nominal dynamics, residuals, and disturbance geometry |
| `runtime/` | Controller policies and activation intervention during generation |
| `datasets/` | Dataset loading and prompt construction |
| `benchmarks/` | Artifact fitting, calibration, generation, and scoring |
| `judges/` | Task metrics and response-quality scorers |
| `experiments/` | Shared calibration, diagnostics, and execution helpers |

Controller mathematics is independent of Transformers and benchmark-specific
logic. The [package guide](robust_steerability/README.md) describes integration
with model adapters and experiment pipelines.

## Benchmarks

Each benchmark exposes four stages. Generation and scoring are separate, allowing
saved responses to be rescored without another model-generation pass.

```bash
python -m robust_steerability.benchmarks.truthfulness artifacts --model llama8b --devices auto
python -m robust_steerability.benchmarks.truthfulness calibrate --model llama8b --methods alqr,h_infinity --devices auto
python -m robust_steerability.benchmarks.truthfulness evaluate --model llama8b --methods original,alqr,h_infinity --datasets id,spanish --devices auto
python -m robust_steerability.benchmarks.truthfulness score --model llama8b --methods original,alqr,h_infinity --datasets id,spanish --scorers default --devices auto
```

| Benchmark | Steering target | Evaluation shift | Protocol |
|---|---|---|---|
| TruthfulQA | Truthfulness | English to Spanish prompts; English output | [Truthfulness](benchmarks/truthfulness/README.md) |
| HarmBench | Safe refusal | Direct harmful requests to jailbreak wrappers | [HarmBench](benchmarks/harmful/README.md) |
| MGSM | Spanish-language output | Held-out input languages | [MGSM](benchmarks/mgsm/README.md) |
| L-CiteEval | Positive sentiment | Matched 8K to 16K context | [L-CiteEval](benchmarks/lciteeval/README.md) |
| Toxicity | Reduced toxicity | RealToxicityPrompts; optional MMLU capability check | [Toxicity](benchmarks/toxicity/README.md) |

Each benchmark's `benchmark.toml` records its composition and settings. Model keys,
available conditions, calibration choices, and metrics are documented in the
linked protocols. [Reproducibility conventions](DECISIONS.md) describe the shared
coordinate system, decoding settings, and baseline choices.

## Reproducing results and figures

Tracked result summaries live in `benchmarks/<benchmark>/results/`. Scientific
caches contain datasets, dynamics, calibrated controllers, generations, and
scorer outputs; they are stored separately and excluded from Git. Local benchmark
caches live in `benchmarks/<benchmark>/cache/`. Existing compatible caches are
reused by the pipeline.

Figure and table scripts live beside their outputs and documentation in `figs/`.
For example, regenerate benchmark tables from available results with:

```bash
python figs/bench_table/bench_table.py
```

The [figure index](figs/README.md) and individual figure READMEs describe required
inputs and rendering commands. Raw data, model weights, and large compute caches
are not distributed with the package; see the [data inventory](data/README.md).

## Repository layout

The package is accompanied by benchmark specifications and research artifacts:

| Folder | Contents |
|---|---|
| `robust_steerability/` | Installable Python package |
| `benchmarks/` | Benchmark specifications and result summaries |
| `figs/` | Figure and table implementations and outputs |
| `parking/` | Supplementary analyses and explorations |
| `debug/` | Diagnostics and investigations |
| `ref/` | Reference implementations and historical material |
| `data/` | Local datasets and their documented inventory |
| `docs/` | Experiment designs and technical notes |
| `logs/` | Recorded experiment manifests and execution logs |
| `server/` | Remote environment setup and execution support |
| `tmp/` | Local temporary artifacts |

## References and attribution

Source-method implementations and their provenance are described in
[source methods](robust_steerability/source_methods/README.md). Reference code
retains its upstream attribution and applicable licenses. Benchmark protocols
link to the datasets and scoring methods used in each experiment.
