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
```

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
GPU discovery, sharding, checkpointing, cache paths, logs, tables, and plots.
Use `python -m robust_steerability.benchmarks.toxicity ...` for the RTP-to-Jigsaw
pipeline. Use the same commands on every machine; no host name is encoded in code.

## Storage boundary

- Git carries code, run logs, result tables, and plots.
- A Studio's persistent disk carries its checkout and active working cache.
- The remote S3 connection carries only large reusable artifacts needed across
  remote machines: averaged dynamics, setpoints, and calibrated controller bundles.
- The local workstation never connects to S3.

After a run finishes, inspect its outputs on the owning machine, commit only its
logs/results/plots, and push them. Cache directories stay out of Git.

Publish or fetch large objects explicitly after the run is verified:

```bash
python -m robust_steerability.storage.s3 push --bucket robust-steering --benchmark truthfulness --model llama8b --stage artifacts
python -m robust_steerability.storage.s3 pull --bucket robust-steering --benchmark truthfulness --model llama8b --stage artifacts
```
