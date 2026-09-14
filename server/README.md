# Multi-machine execution

`server/` owns remote-machine registration, shared-S3 paths, job partitioning,
dispatch, status, and logs. Scientific code and result interpretation remain in
their owning `parking/` or `figs/` units.

## Architecture

- Git is the source of code and job plans. Every machine in one wave must use
  the same clean commit.
- S3 is the source of reusable artifacts, datasets, checkpoints, and generated
  results. Machines do not pass result folders directly to one another.
- Each plan is one concurrent wave. Jobs may share read-only inputs but must own
  disjoint local output paths, S3 output prefixes, and physical GPU slots.
- Before a job runs, its declared inputs are synchronized from S3. After it
  finishes—or fails after producing partial files—its declared outputs are
  synchronized back to S3.
- Remote process metadata and logs stay outside the checkout under the machine's
  `job_root`. Local launch provenance is written to ignored `server/cache/`.

## Files

- `machines.json`: SSH host, persistent checkout, remote job root, and private
  environment-file location for every machine.
- `storage.json`: shared object-store implementation and the environment variable
  that holds its root URI. It contains no credentials or bucket name.
- `plans/`: versioned execution waves. The example partitions toxicity generation
  by controller and GPU.
- `manage_jobs.py`: validation, code synchronization, dispatch, monitoring, and
  log access.

## Machine environment

Every machine must have the AWS CLI, credentials or an attached IAM role, and the
same S3 root URI. Store the URI outside Git in the machine's `environment_file`:

```bash
ROBUST_STEERING_S3_URI=s3://YOUR_BUCKET/YOUR_PREFIX
```

The environment file may also contain machine-local secrets when IAM is not
available. Keep it mode `600`; never add it to the repository.

Machine 1 is `robust-steering-1`. Its persistent checkout is
`/teamspace/studios/this_studio/controller`, its job metadata lives under
`/teamspace/studios/this_studio/.robust_steering_jobs`, and its private environment
file is `/teamspace/studios/this_studio/.robust_steering_env`.

## Plan contract

Plans use schema version 2. Each job declares:

- `machine_id` and physical `gpu_ids`;
- one shell `command` executed from the repository root;
- zero or more `inputs`, each mapping a repository-relative cache directory to a
  prefix relative to the configured S3 root;
- one or more `outputs` using the same mapping.

The manager rejects duplicate jobs, unknown machines, overlapping GPU slots,
overlapping concurrent outputs, output/input races between jobs, dirty checkouts,
different commits within one wave, unavailable GPUs, and inaccessible S3.

When a job is restricted to physical GPU 3, that process sees it as `cuda:0`, so
commands should normally request `cuda:0`.

## Commands

Run from the repository root:

```bash
python server/manage_jobs.py inventory
python server/manage_jobs.py sync-code
python server/manage_jobs.py validate --plan server/plans/toxicity_generation.example.json
python server/manage_jobs.py launch --plan server/plans/toxicity_generation.example.json
python server/manage_jobs.py status --plan server/plans/toxicity_generation.example.json
python server/manage_jobs.py logs --plan server/plans/toxicity_generation.example.json --job gemma2b_h_infinity
```

Use a new `plan_id` for a deliberate rerun. Reusing an existing remote
`plan_id/job_id` is rejected so earlier logs are preserved.
