# Remote run logs

Benchmark entry points write Git-versioned execution records under a named run
directory here. Every stage uses the same `run.json` schema.

```text
logs/<run_id>/
  run.json
  ...worker logs...
```

`run.json` records status, UTC start/end, elapsed seconds, host, command, Git
state, benchmark, model, stage, methods, datasets, cache mode, requested devices,
visible GPU inventory, calibration ID, and stage parameters.

The local coordinator enters the machine over SSH and reads its live Screen and
log files there. Logs are not mirrored into a separate local cache. After
completion, commit them with the selected tables and plots.

Large tensors, Jacobians, model caches, and `.pt` artifacts do not belong here.
On Lightning they live in the producing Studio's Teamspace Drive cache under
`~/robust-steering-cache/`.
