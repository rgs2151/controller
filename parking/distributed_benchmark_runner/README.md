# distributed_benchmark_runner

## Method

- Load a versioned machine registry and one versioned concurrent-wave job plan.
- Validate unique job IDs, known machine IDs, non-overlapping physical GPU slots, non-overlapping repository-relative cache ownership, a clean remote Git checkout, and availability of every requested GPU before dispatch.
- Launch each command through its configured SSH target inside the machine's persistent repository checkout, restricting the process to its declared physical GPUs with `CUDA_VISIBLE_DEVICES`.
- Record the command, PID, UTC start and finish times, exit code, and combined standard-output/error log in a job directory outside the remote repository.
- Query remote job state or log tails without changing a run, and collect declared output directories into this unit's ignored local cache for inspection before integration.

## Variables

- Data/input: `machines.json` and a schema-version-1 plan in `plans/`.
- Sessions/groups: one plan is one concurrently launched wave; every job is one independently logged process.
- Labels/targets: `machine_id`, `job_id`, physical `gpu_ids`, exact shell `command`, and `output_owners` identify each partition.
- Signals/features/measures: remote PID liveness, process exit code, UTC timestamps, Git revision, GPU inventory, and job log text.
- Parameters/thresholds: duplicate job IDs, overlapping output owners, unknown machines, dirty remote checkouts, unavailable GPUs, and concurrent reuse of one machine/GPU slot are rejected.
- Outputs: remote metadata under each machine's `job_root`; collected artifacts under `cache/<plan_id>/collected/`; local launch provenance under `cache/<plan_id>/launch.json`.

## Statistics

- None; this unit performs deterministic validation, dispatch, monitoring, and file transfer.
- Null hypothesis: not applicable.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: a plan launches only after all ownership and GPU-allocation checks pass; exit code zero is complete and any other recorded exit code is failed.
- What the statistic means: reported states describe process execution, not scientific benchmark performance.
- Why this statistic is appropriate here: the unit manages computation and preserves provenance but does not analyze model outputs.

## Legends

- X axis: none.
- Y axis: none.
- Color/value: none.
- Grouping: jobs are grouped by `plan_id`; outputs are grouped by job and declared cache owner.
- Ordering/sorting: jobs retain their order in the plan file.
- Lines/markers/labels: status output is `job_id`, `machine_id`, and state.
- Panels: none.

## Interpretation

- A valid plan guarantees that concurrently dispatched jobs do not share a physical GPU or claim overlapping cache paths; launch additionally requires a clean remote commit and all requested GPUs.
- Scientific results remain owned by their original compact units; collection into this manager's cache is a review step, not publication integration.

## Notes

- Machine 1 is SSH target `robust-steering-1`, with persistent checkout `/teamspace/studios/this_studio/controller`.
- The Studio may remain on CPU while idle; `inventory` reports zero GPUs until a GPU machine is attached.
- GPU IDs are physical on the remote machine. A job restricted to physical GPU 3 sees that device internally as `cuda:0`.
- Commands: `inventory`; `validate --plan <path>`; `launch --plan <path>`; `status --plan <path>`; `logs --plan <path> --job <id>`; and `collect --plan <path>`.
- Reusing a remote `plan_id/job_id` is rejected. Use a new plan ID for a deliberate rerun so earlier logs and exit codes remain intact.

## References

- `AGENTS.md`.
- `ORGANIZATION.md`.
- `parking/bench_artifacts/`.
- `parking/bench_evaluations/`.
