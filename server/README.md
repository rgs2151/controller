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
conda activate robust-steerability
python <owning-unit-or-package-entrypoint> --devices auto
```

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

## Storage boundary

- Git carries code, run logs, result tables, and plots.
- A Studio's persistent disk carries its checkout and active working cache.
- The remote S3 connection carries only large reusable artifacts needed across
  remote machines, such as Jacobians and `.pt` controller caches.
- The local workstation never connects to S3.

After a run finishes, inspect its outputs on the owning machine, commit only its
logs/results/plots, and push them. Cache directories stay out of Git.
