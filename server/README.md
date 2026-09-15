# Remote execution

Remote machines are operated directly over SSH. `server/` does not contain a
machine registry, experiment plan, remote launcher, or benchmark configuration.

## Run workflow

From the local workstation, enter the selected machine:

```bash
ssh <ssh-host>
```

On that machine, inspect the checkout, update it, and start a named Screen:

```bash
cd ~/controller
git status --short
git pull --ff-only
screen -S <run-name>
source .venv/bin/activate
python -m robust_steerability.benchmarks.truthfulness artifacts --model llama8b --devices auto
python -m robust_steerability.benchmarks.truthfulness calibrate --model llama8b --devices auto
python -m robust_steerability.benchmarks.truthfulness evaluate --model llama8b --devices auto
python -m robust_steerability.benchmarks.truthfulness score --model llama8b --scorers truthfulqa_true,truthfulqa_informative,axbench_instruction_relevance,axbench_fluency --devices auto
```

Supplying `--h-infinity-q-over-r`, `--h-infinity-q-final-over-r`, and
`--h-infinity-r` to `calibrate` skips only the H-infinity candidate sweep. Run
`artifacts` first on a new model/task, then `calibrate`; the latter still fits
every requested source method, estimates H-infinity disturbance geometry,
synthesizes the fixed controller, and saves its diagnostics. Use `--methods
all` when the run should contain every comparison method.

Create the persistent Studio environment once per checkout before starting a
run:

```bash
cd ~/controller
uv venv --python /usr/bin/python3 .venv
uv pip install --python .venv/bin/python -e .
```

The local workstation continues to use the `robust-steerability` Conda
environment; Lightning Studios use the checkout-local `.venv` above.

If the remote GPU has been checked with a smoke run and supports a larger
generation batch, pass `--generation-batch-size <n>` to the calibration or
evaluation command. This is a run parameter, not a machine configuration.

Detach with `Ctrl-a d`. Later, SSH into the same machine and inspect the actual
process and files there:

```bash
screen -ls
screen -r <run-name>
nvidia-smi
git status --short
```

The benchmark entry point—not the server folder—owns models, datasets, methods,
GPU discovery, sharding, cache paths, logs, tables, and plots. Every stage log
records UTC start/end, elapsed time, requested devices, visible GPUs, host,
command, Git state, methods, datasets, cache condition, and stage parameters.
Use `python -m robust_steerability.benchmarks.toxicity ...` for the RTP-to-Jigsaw
pipeline. Use the same commands on every machine; no host name is encoded in code.

## Storage boundary

- Git carries code, run logs, result tables, and plots.
- On Lightning, the complete ignored benchmark cache is written directly to
  `~/robust-steering-cache/<benchmark>/<model>/`. A Studio home is persistent
  Teamspace Drive storage, so no AWS transport or copy step exists.
- Each model/benchmark remains owned by the Studio that ran it. Peer Studios can
  inspect its cache at
  `/teamspace/studios/<producer-studio>/robust-steering-cache/`.
- The local workstation uses `benchmarks/<benchmark>/cache/<model>/` and does
  not access remote Teamspace storage.

After a run finishes, inspect its outputs on the owning machine, commit only its
logs/results/plots, and push them. Teamspace cache directories stay out of Git.
Continue a model/benchmark on its owning Studio so new files remain writable;
peer-Studio views are for inspection or explicit read-only reuse.
