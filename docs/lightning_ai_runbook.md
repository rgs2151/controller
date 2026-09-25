# Lightning AI runbook

Use this procedure for every benchmark launched on a Lightning Studio. The
benchmark CLI remains the source of truth; this document only defines where the
code, environment, model, and benchmark state live while a job is running.

## Storage rule

| Item | While the job runs | Durable copy |
|---|---|---|
| Code checkout | `/tmp/controller-run` | GitHub and `~/controller` |
| Python environment | `/tmp/robust-steerability-venv` | None; recreate it |
| Hugging Face models | `/tmp/robust-steerability/huggingface` | None; redownload pinned revisions |
| Active benchmark cache | `/tmp/<run>/robust-steering-cache/<benchmark>/<model>` | `~/robust-steering-cache/<benchmark>/<model>` |
| Logs and compact results | Node-local checkout while running | Git |
| Figures and tables | Git checkout | Git |

Do not execute repeatedly against Teamspace files. Teamspace is the durable
store, not the hot working directory. Never place a virtual environment or a
Hugging Face model cache there.

## 1. Before allocating GPUs

1. Identify the exact benchmark, model, methods, datasets, cache condition,
   calibration ID, and stage to run.
2. Inspect the durable cache and determine the earliest missing stage. Do not
   rerun a complete earlier stage merely because a later stage is missing.
3. Confirm that the requested code commit is pushed.
4. Decide the generation batch size from both memory and shard count. For `N`
   GPUs, the dataset must produce at least `N` batches; an oversized batch can
   leave GPUs idle even when it fits in memory.

## 2. Create node-local code and environment

Fast-forward `~/controller` when Teamspace Git is responsive, but run from a
fresh node-local checkout:

```bash
rm -rf /tmp/controller-run
git clone <repository-url> /tmp/controller-run
cd /tmp/controller-run
git checkout <commit-or-branch>
cp ~/controller/.env .env
chmod 600 .env
PYTHON=$(server/bootstrap_node_env.sh)
```

The bootstrap always deletes and recreates
`/tmp/robust-steerability-venv`. Run it once per newly provisioned container,
not once per benchmark stage.

If a Teamspace `git pull` stalls, do not wait while GPUs burn. Clone directly
from GitHub. If remote authentication is unavailable, transfer a Git bundle
from the coordinator and clone that bundle into `/tmp/controller-run`. Verify
the node-local commit with `git rev-parse --short HEAD` before launching.

Never print `.env` values. It is sufficient to verify that the required key
names exist.

## 3. Select fresh or continued state

Set one node-local active root for the entire job:

```bash
export LIGHTNING_ARTIFACTS_DIR=/tmp/<run>
export HF_HOME=/tmp/robust-steerability
export HF_HUB_CACHE=/tmp/robust-steerability/huggingface
export HF_DATASETS_CACHE=/tmp/robust-steerability/datasets
export ROBUST_STEERING_SKIP_GIT_STATUS=1
```

The cache produced beneath this root is
`$LIGHTNING_ARTIFACTS_DIR/robust-steering-cache/...`.

### Fresh model × benchmark run

Start with an empty active benchmark directory. Do not copy an unrelated model,
calibration, or evaluation cache. Run the required stages in order:

```text
artifacts -> calibrate -> evaluate -> score
```

The pinned evaluated-model revision is downloaded directly from Hugging Face
to the node-local cache. Redownloading is normally faster and safer than
loading model weights from Teamspace.

### Continued run

Copy only the durable prerequisites for the first stage being resumed:

| Stage to run | Copy from the durable model cache |
|---|---|
| `artifacts` | Frozen dataset plus only genuinely resumable artifact shards |
| `calibrate` | Frozen dataset and completed controller-neutral artifacts |
| `evaluate` | Frozen dataset, required method calibration, and controller artifacts required by that method |
| `score` | Completed generation files for the requested methods and datasets |

Examples of state worth staging are the frozen dataset JSON, averaged dynamics
matrix, setpoint, selected controller, method fit, and completed/compatible
generation shards. Do not copy raw per-prompt Jacobians when the saved averaged
dynamics matrix is the artifact consumed downstream.

Copy the exact hierarchy. Cache identity is scoped by benchmark, model,
calibration ID, KV-cache condition, dataset namespace, and method. Never place
an artifact under a different identity merely to make the loader find it.

For a small prerequisite:

```bash
mkdir -p /tmp/<run>/robust-steering-cache/<benchmark>/<model>/datasets
cp ~/robust-steering-cache/<benchmark>/<model>/datasets/<dataset>.json \
  /tmp/<run>/robust-steering-cache/<benchmark>/<model>/datasets/
```

Use `rsync -a --partial` for larger unique experiment artifacts. Models and the
Python environment are exceptions: recreate or redownload them instead of
copying them from Teamspace.

## 4. Interrupted and changed runs

- Resume partial shards only when the scientific identity and execution
  identity are unchanged, including batch size and shard count.
- If batch size, shard count, method parameters, calibration ID, dataset, or
  cache condition changes, remove only the affected **node-local** partial
  shard directory and restart that slice.
- Preserve completed durable results. A new attempt or calibration belongs in
  its existing explicit namespace or in a newly named calibration namespace;
  it must not silently overwrite a scientifically different result.
- Do not delete controller artifacts just because generation or scoring failed.
- Do not use `rsync --delete` when syncing results back to Teamspace.

## 5. Launch and monitor

Run the stages in a named detached Screen. Use one fresh environment and one
active cache root for the full chain:

```bash
screen -L -Logfile /tmp/<run>.log -dmS <run> bash -lc '
  set -euo pipefail
  cd /tmp/controller-run
  export LIGHTNING_ARTIFACTS_DIR=/tmp/<run>
  export HF_HOME=/tmp/robust-steerability
  export HF_HUB_CACHE=/tmp/robust-steerability/huggingface
  export HF_DATASETS_CACHE=/tmp/robust-steerability/datasets
  export ROBUST_STEERING_SKIP_GIT_STATUS=1
  PYTHON=/tmp/robust-steerability-venv/bin/python
  "$PYTHON" -m robust_steerability.benchmarks.<benchmark> <stage> ...
'
```

Before leaving the job unattended, verify:

- the Screen is alive;
- the run record points to the requested commit and node-local cache;
- the pinned model snapshot finished downloading;
- the expected worker count exists;
- every allocated GPU receives a shard;
- shard files are advancing; and
- logs contain no traceback or out-of-memory error.

Long multi-model launchers must persist and verify each completed model before
starting the next one. Do not defer every result copy to the end of the full
queue: a Lightning compute replacement destroys node-local `/tmp` even when the
detached job completed successfully.

Do not interpret idle GPUs during a deliberately single-worker fit as failure.
Do investigate idle GPUs during sharded generation.

## 6. Finish and synchronize

After all GPU work and scoring finish, merge the node-local cache into the
durable cache without deleting existing results:

```bash
rsync -a --partial \
  /tmp/<run>/robust-steering-cache/<benchmark>/<model>/ \
  ~/robust-steering-cache/<benchmark>/<model>/
```

Then copy or commit the run logs, compact result JSON, tables, and figures from
the node-local checkout. Pull/fetch the current Git tip before pushing so a
different machine's completed result is not lost.

Only after both cache synchronization and Git synchronization are verified may
the Studio be moved back to CPU or shut down. The node-local environment,
model download, execution checkout, and active cache can then be discarded.
