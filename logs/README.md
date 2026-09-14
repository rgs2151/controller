# Remote run logs

Benchmark entry points write Git-versioned execution records under a named run
directory here. Each owning benchmark defines its exact files; there is no
server-level run schema.

```text
logs/<run_id>/
  ...benchmark-owned logs and run metadata...
```

The local coordinator enters the machine over SSH and reads its live Screen and
log files there. Logs are not mirrored into a separate local cache and are never
stored in S3. After completion, commit them with the selected tables and plots.

Large tensors, Jacobians, model caches, and `.pt` artifacts do not belong here.
