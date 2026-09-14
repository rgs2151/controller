"""Dispatch non-overlapping benchmark jobs across SSH machines with shared S3."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


SERVER = Path(__file__).resolve().parent
CACHE = SERVER / "cache"
DEFAULT_MACHINES = SERVER / "machines.json"
DEFAULT_STORAGE = SERVER / "storage.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _local_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(SERVER.parent), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _machine_map(path: Path) -> dict[str, dict]:
    payload = _read_json(path)
    if payload.get("schema_version") != 1:
        raise ValueError("Machine registry must use schema_version 1")
    machines = payload.get("machines")
    if not isinstance(machines, list) or not machines:
        raise ValueError("Machine registry contains no machines")
    required = {"machine_id", "ssh_host", "repo", "job_root", "environment_file"}
    result: dict[str, dict] = {}
    for machine in machines:
        if not isinstance(machine, dict) or not required.issubset(machine):
            raise ValueError(f"Each machine must define {sorted(required)}")
        machine_id = machine["machine_id"]
        if not isinstance(machine_id, str) or not machine_id:
            raise ValueError("machine_id must be a non-empty string")
        if machine_id in result:
            raise ValueError(f"Duplicate machine_id: {machine_id}")
        result[machine_id] = machine
    return result


def _storage_config(path: Path) -> dict:
    storage = _read_json(path)
    if storage.get("schema_version") != 1 or storage.get("backend") != "s3":
        raise ValueError("Storage registry must use schema_version 1 and backend s3")
    env_name = storage.get("root_uri_environment_variable")
    cli = storage.get("cli")
    if not isinstance(env_name, str) or not env_name.isidentifier():
        raise ValueError("root_uri_environment_variable must be a shell-safe name")
    if cli != "aws":
        raise ValueError("Only the aws CLI storage implementation is supported")
    return storage


def _safe_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or not value.replace("_", "").isalnum():
        raise ValueError(f"{label} must contain only letters, numbers, and underscores")
    return value


def _local_cache_path(value: object) -> PurePosixPath:
    if not isinstance(value, str):
        raise ValueError("local_path must be a string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "cache" not in path.parts:
        raise ValueError(f"local_path must be a repository-relative cache path: {value}")
    return path


def _s3_prefix(value: object) -> PurePosixPath:
    if not isinstance(value, str):
        raise ValueError("s3_prefix must be a string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.startswith("s3://"):
        raise ValueError(f"s3_prefix must be relative to the configured S3 root: {value}")
    if not path.parts or str(path) in {"", "."}:
        raise ValueError("s3_prefix must not be empty")
    return path


def _bindings(job: dict, key: str, *, required: bool) -> list[dict[str, str]]:
    values = job.get(key, [])
    if not isinstance(values, list) or (required and not values):
        qualifier = "at least one" if required else "a list of"
        raise ValueError(f"Job {job.get('job_id')} must declare {qualifier} {key} binding")
    normalized = []
    for value in values:
        if not isinstance(value, dict) or set(value) != {"local_path", "s3_prefix"}:
            raise ValueError(
                f"Each {key} binding must contain only local_path and s3_prefix"
            )
        normalized.append(
            {
                "local_path": str(_local_cache_path(value["local_path"])),
                "s3_prefix": str(_s3_prefix(value["s3_prefix"])),
            }
        )
    return normalized


def _overlap(first: PurePosixPath, second: PurePosixPath) -> bool:
    return first == second or first in second.parents or second in first.parents


def validate_plan(plan_path: Path, machines_path: Path) -> tuple[dict, dict[str, dict]]:
    machines = _machine_map(machines_path)
    plan = _read_json(plan_path)
    if plan.get("schema_version") != 2:
        raise ValueError("Job plan must use schema_version 2")
    plan_id = _safe_identifier(plan.get("plan_id"), "plan_id")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Job plan contains no jobs")

    job_ids: set[str] = set()
    gpu_slots: set[tuple[str, int]] = set()
    all_inputs: list[tuple[str, PurePosixPath, PurePosixPath]] = []
    all_outputs: list[tuple[str, PurePosixPath, PurePosixPath]] = []
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("Each job must be a JSON object")
        job_id = _safe_identifier(job.get("job_id"), "job_id")
        if job_id in job_ids:
            raise ValueError(f"Duplicate job_id: {job_id}")
        job_ids.add(job_id)

        machine_id = job.get("machine_id")
        if machine_id not in machines:
            raise ValueError(f"Unknown machine_id for {job_id}: {machine_id}")
        command = job.get("command")
        if not isinstance(command, str) or not command.strip() or "\n" in command:
            raise ValueError(f"Job {job_id} must have one non-empty command line")

        gpu_ids = job.get("gpu_ids")
        if not isinstance(gpu_ids, list) or any(
            not isinstance(gpu_id, int) or gpu_id < 0 for gpu_id in gpu_ids
        ):
            raise ValueError(f"Job {job_id} has invalid gpu_ids")
        if len(gpu_ids) != len(set(gpu_ids)):
            raise ValueError(f"Job {job_id} repeats a GPU ID")
        for gpu_id in gpu_ids:
            slot = (machine_id, gpu_id)
            if slot in gpu_slots:
                raise ValueError(
                    f"Concurrent GPU slot assigned more than once: {machine_id}/GPU {gpu_id}"
                )
            gpu_slots.add(slot)

        job["inputs"] = _bindings(job, "inputs", required=False)
        job["outputs"] = _bindings(job, "outputs", required=True)
        all_inputs.extend(
            (job_id, PurePosixPath(item["local_path"]), PurePosixPath(item["s3_prefix"]))
            for item in job["inputs"]
        )
        all_outputs.extend(
            (job_id, PurePosixPath(item["local_path"]), PurePosixPath(item["s3_prefix"]))
            for item in job["outputs"]
        )

    for index, (job_id, local_path, s3_prefix) in enumerate(all_outputs):
        for other_job, other_local, other_s3 in all_outputs[index + 1 :]:
            if _overlap(local_path, other_local) or _overlap(s3_prefix, other_s3):
                raise ValueError(
                    "Concurrent outputs overlap: "
                    f"{job_id} ({local_path}, {s3_prefix}) and "
                    f"{other_job} ({other_local}, {other_s3})"
                )
        for input_job, input_local, input_s3 in all_inputs:
            if input_job != job_id and (
                _overlap(local_path, input_local) or _overlap(s3_prefix, input_s3)
            ):
                raise ValueError(
                    f"Output for {job_id} overlaps concurrent input for {input_job}"
                )

    plan["plan_id"] = plan_id
    return plan, machines


def _ssh(machine: dict, script: str, *, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", machine["ssh_host"], "bash", "-s"],
        input=script,
        text=True,
        check=True,
        capture_output=capture,
    )


def _environment_setup(machine: dict) -> str:
    environment_file = shlex.quote(machine["environment_file"])
    return f"""environment_file={environment_file}
if [ -f \"$environment_file\" ]; then
  set -a
  source \"$environment_file\"
  set +a
fi"""


def _remote_job_dir(machine: dict, plan_id: str, job_id: str) -> str:
    return str(PurePosixPath(machine["job_root"]) / plan_id / job_id)


def _remote_state(machine: dict, storage: dict) -> dict:
    repo = shlex.quote(machine["repo"])
    env_name = storage["root_uri_environment_variable"]
    script = f"""set -euo pipefail
{_environment_setup(machine)}
echo commit=$(git -C {repo} rev-parse HEAD)
echo dirty=$(git -C {repo} status --porcelain | wc -l)
if command -v nvidia-smi >/dev/null 2>&1; then
  echo gpu_ids=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | paste -sd, -)
else
  echo gpu_ids=
fi
if command -v aws >/dev/null 2>&1; then echo aws_cli=1; else echo aws_cli=0; fi
if [ -n \"${{{env_name}:-}}\" ]; then echo s3_configured=1; else echo s3_configured=0; fi
if command -v aws >/dev/null 2>&1 && [ -n \"${{{env_name}:-}}\" ] && \
   aws s3 ls \"${{{env_name}}}\" >/dev/null 2>&1; then
  echo s3_ready=1
else
  echo s3_ready=0
fi
"""
    rows = dict(line.split("=", 1) for line in _ssh(machine, script).stdout.splitlines())
    return {
        "git_commit": rows["commit"],
        "dirty_count": int(rows["dirty"].strip()),
        "gpu_ids": {int(value) for value in rows["gpu_ids"].split(",") if value.strip()},
        "aws_cli": rows["aws_cli"] == "1",
        "s3_configured": rows["s3_configured"] == "1",
        "s3_ready": rows["s3_ready"] == "1",
    }


def inventory(machines_path: Path, storage_path: Path) -> None:
    storage = _storage_config(storage_path)
    for machine_id, machine in _machine_map(machines_path).items():
        state = _remote_state(machine, storage)
        gpu_ids = ",".join(str(value) for value in sorted(state["gpu_ids"])) or "none"
        s3 = "ready" if state["s3_ready"] else "unavailable"
        print(
            f"{machine_id}\tssh={machine['ssh_host']}\tcommit={state['git_commit'][:12]}"
            f"\tdirty={state['dirty_count']}\tgpus={gpu_ids}\tshared_s3={s3}"
        )


def sync_code(machines_path: Path) -> None:
    for machine_id, machine in _machine_map(machines_path).items():
        repo = shlex.quote(machine["repo"])
        script = f"""set -euo pipefail
test -z \"$(git -C {repo} status --porcelain)\"
git -C {repo} checkout main >/dev/null
git -C {repo} pull --ff-only
git -C {repo} rev-parse HEAD
"""
        commit = _ssh(machine, script).stdout.strip().splitlines()[-1]
        print(f"{machine_id}\t{commit}")


def _sync_in_commands(job: dict, env_name: str) -> str:
    commands = []
    for index, binding in enumerate(job["inputs"]):
        local_path = shlex.quote(binding["local_path"])
        prefix = shlex.quote(binding["s3_prefix"].rstrip("/") + "/")
        commands.append(
            f"""input_uri=\"${{{env_name}%/}}/\"{prefix}
if aws s3 ls \"$input_uri\" --recursive > \"$job_dir/input_{index}.objects\" && \
   test -s \"$job_dir/input_{index}.objects\"; then
  mkdir -p \"$repo\"/{local_path}
  aws s3 sync \"$input_uri\" \"$repo\"/{local_path}/ --only-show-errors || prepare_failed=1
else
  printf 'missing or unreadable declared input: %s\n' \"$input_uri\"
  prepare_failed=1
fi"""
        )
    return "\n".join(commands)


def _sync_out_commands(job: dict, env_name: str) -> str:
    commands = []
    for binding in job["outputs"]:
        local_path = shlex.quote(binding["local_path"])
        prefix = shlex.quote(binding["s3_prefix"].rstrip("/") + "/")
        commands.append(
            f"""local_output=\"$repo\"/{local_path}
output_uri=\"${{{env_name}%/}}/\"{prefix}
if [ -d \"$local_output\" ]; then
  aws s3 sync \"$local_output/\" \"$output_uri\" --only-show-errors || upload_failed=1
elif [ -f \"$local_output\" ]; then
  aws s3 cp \"$local_output\" \"$output_uri\" --only-show-errors || upload_failed=1
else
  printf 'missing declared output: %s\n' \"$local_output\" >> \"$job_dir/stdout.log\"
  upload_failed=1
fi"""
        )
    return "\n".join(commands)


def launch(plan_path: Path, machines_path: Path, storage_path: Path) -> None:
    storage = _storage_config(storage_path)
    plan, machines = validate_plan(plan_path, machines_path)
    target_ids = {job["machine_id"] for job in plan["jobs"]}
    machine_states = {
        machine_id: _remote_state(machines[machine_id], storage)
        for machine_id in target_ids
    }
    commits = {state["git_commit"] for state in machine_states.values()}
    if len(commits) != 1:
        raise ValueError("All machines in one wave must use the same Git commit")
    if commits != {_local_commit()}:
        raise ValueError("Remote machines must match the local Git commit; run sync-code")
    for machine_id, state in machine_states.items():
        if state["dirty_count"]:
            raise ValueError(f"Remote repository is dirty on {machine_id}")
        if not state["s3_ready"]:
            raise ValueError(f"Shared S3 is not configured and accessible on {machine_id}")
    for job in plan["jobs"]:
        missing = set(job["gpu_ids"]) - machine_states[job["machine_id"]]["gpu_ids"]
        if missing:
            raise ValueError(
                f"Job {job['job_id']} requests unavailable GPUs on "
                f"{job['machine_id']}: {sorted(missing)}"
            )

    launch_rows = []
    env_name = storage["root_uri_environment_variable"]
    for job in plan["jobs"]:
        machine = machines[job["machine_id"]]
        job_dir = _remote_job_dir(machine, plan["plan_id"], job["job_id"])
        command_b64 = base64.b64encode(job["command"].encode()).decode()
        cuda_devices = ",".join(str(value) for value in job["gpu_ids"])
        cuda_line = (
            f"export CUDA_VISIBLE_DEVICES={shlex.quote(cuda_devices)}"
            if cuda_devices
            else "unset CUDA_VISIBLE_DEVICES"
        )
        input_sync = _sync_in_commands(job, env_name)
        output_sync = _sync_out_commands(job, env_name)
        script = f"""set -euo pipefail
job_dir={shlex.quote(job_dir)}
test ! -e \"$job_dir\"
mkdir -p \"$job_dir\"
printf '%s' {shlex.quote(command_b64)} > \"$job_dir/command.b64\"
cat > \"$job_dir/run.sh\" <<'RUN_SCRIPT'
#!/usr/bin/env bash
set -uo pipefail
job_dir={shlex.quote(job_dir)}
repo={shlex.quote(machine['repo'])}
{_environment_setup(machine)}
{cuda_line}
exec >> \"$job_dir/stdout.log\" 2>&1
date -u +%Y-%m-%dT%H:%M:%SZ > \"$job_dir/started_at_utc\"
command=$(base64 --decode \"$job_dir/command.b64\")
cd \"$repo\"
prepare_failed=0
{input_sync}
if [ \"$prepare_failed\" -eq 0 ]; then
  bash -lc \"$command\"
  exit_code=$?
else
  exit_code=73
fi
upload_failed=0
{output_sync}
if [ \"$upload_failed\" -ne 0 ] && [ \"$exit_code\" -eq 0 ]; then exit_code=74; fi
printf '%s\n' \"$exit_code\" > \"$job_dir/exit_code\"
date -u +%Y-%m-%dT%H:%M:%SZ > \"$job_dir/finished_at_utc\"
exit \"$exit_code\"
RUN_SCRIPT
chmod 700 \"$job_dir/run.sh\"
nohup \"$job_dir/run.sh\" >/dev/null 2>&1 &
printf '%s\n' \"$!\" > \"$job_dir/pid\"
cat \"$job_dir/pid\"
"""
        pid = _ssh(machine, script).stdout.strip()
        launch_rows.append(
            {
                "job_id": job["job_id"],
                "machine_id": job["machine_id"],
                "gpu_ids": job["gpu_ids"],
                "git_commit": machine_states[job["machine_id"]]["git_commit"],
                "remote_pid": int(pid),
                "remote_job_dir": job_dir,
                "inputs": job["inputs"],
                "outputs": job["outputs"],
            }
        )
        print(f"launched {job['job_id']} on {job['machine_id']} pid={pid}")

    destination = CACHE / plan["plan_id"] / "launch.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "plan_id": plan["plan_id"],
                "launched_at_utc": _utc_now(),
                "plan_sha256": _sha256(plan_path),
                "machines_sha256": _sha256(machines_path),
                "storage_sha256": _sha256(storage_path),
                "jobs": launch_rows,
            },
            indent=2,
        )
        + "\n"
    )


def status(plan_path: Path, machines_path: Path) -> None:
    plan, machines = validate_plan(plan_path, machines_path)
    for job in plan["jobs"]:
        machine = machines[job["machine_id"]]
        job_dir = _remote_job_dir(machine, plan["plan_id"], job["job_id"])
        script = f"""set -u
job_dir={shlex.quote(job_dir)}
if [ -f \"$job_dir/exit_code\" ]; then
  code=$(cat \"$job_dir/exit_code\")
  if [ \"$code\" = 0 ]; then echo complete; else echo failed:$code; fi
elif [ -f \"$job_dir/pid\" ] && kill -0 \"$(cat \"$job_dir/pid\")\" 2>/dev/null; then
  echo running
elif [ -f \"$job_dir/pid\" ]; then
  echo lost
else
  echo not_launched
fi
"""
        state = _ssh(machine, script).stdout.strip()
        print(f"{job['job_id']}\t{job['machine_id']}\t{state}")


def logs(plan_path: Path, machines_path: Path, job_id: str, lines: int) -> None:
    plan, machines = validate_plan(plan_path, machines_path)
    matches = [job for job in plan["jobs"] if job["job_id"] == job_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown job_id: {job_id}")
    job = matches[0]
    machine = machines[job["machine_id"]]
    job_dir = _remote_job_dir(machine, plan["plan_id"], job_id)
    script = f"tail -n {int(lines)} {shlex.quote(job_dir + '/stdout.log')}"
    print(_ssh(machine, script).stdout, end="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machines", type=Path, default=DEFAULT_MACHINES)
    parser.add_argument("--storage", type=Path, default=DEFAULT_STORAGE)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("inventory")
    subparsers.add_parser("sync-code")
    for action in ("validate", "launch", "status"):
        command = subparsers.add_parser(action)
        command.add_argument("--plan", type=Path, required=True)
    log_parser = subparsers.add_parser("logs")
    log_parser.add_argument("--plan", type=Path, required=True)
    log_parser.add_argument("--job", required=True)
    log_parser.add_argument("--lines", type=int, default=50)
    arguments = parser.parse_args()

    if arguments.action == "inventory":
        inventory(arguments.machines, arguments.storage)
    elif arguments.action == "sync-code":
        sync_code(arguments.machines)
    elif arguments.action == "validate":
        _storage_config(arguments.storage)
        plan, _ = validate_plan(arguments.plan, arguments.machines)
        print(f"valid plan: {plan['plan_id']} ({len(plan['jobs'])} jobs)")
    elif arguments.action == "launch":
        launch(arguments.plan, arguments.machines, arguments.storage)
    elif arguments.action == "status":
        status(arguments.plan, arguments.machines)
    else:
        logs(arguments.plan, arguments.machines, arguments.job, arguments.lines)


if __name__ == "__main__":
    main()
