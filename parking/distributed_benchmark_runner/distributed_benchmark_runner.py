"""Dispatch non-overlapping benchmark jobs across SSH-accessible machines."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


UNIT = Path(__file__).resolve().parent
CACHE = UNIT / "cache"
DEFAULT_MACHINES = UNIT / "machines.json"


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


def _machine_map(path: Path) -> dict[str, dict]:
    payload = _read_json(path)
    if payload.get("schema_version") != 1:
        raise ValueError("Machine registry must use schema_version 1")
    machines = payload.get("machines")
    if not isinstance(machines, list) or not machines:
        raise ValueError("Machine registry contains no machines")
    result = {machine["machine_id"]: machine for machine in machines}
    if len(result) != len(machines):
        raise ValueError("Machine IDs must be unique")
    return result


def _validate_output_owner(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "cache" not in path.parts:
        raise ValueError(f"Output owner must be a repository-relative cache path: {value}")
    return str(path)


def validate_plan(plan_path: Path, machines_path: Path) -> tuple[dict, dict[str, dict]]:
    machines = _machine_map(machines_path)
    plan = _read_json(plan_path)
    if plan.get("schema_version") != 1:
        raise ValueError("Job plan must use schema_version 1")
    plan_id = plan.get("plan_id")
    if not isinstance(plan_id, str) or not plan_id or not plan_id.replace("_", "").isalnum():
        raise ValueError("plan_id must contain only letters, numbers, and underscores")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Job plan contains no jobs")

    job_ids: set[str] = set()
    output_owners: list[PurePosixPath] = []
    gpu_slots: set[tuple[str, int]] = set()
    for job in jobs:
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or not job_id or not job_id.replace("_", "").isalnum():
            raise ValueError("job_id must contain only letters, numbers, and underscores")
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

        owners = job.get("output_owners")
        if not isinstance(owners, list) or not owners:
            raise ValueError(f"Job {job_id} must declare output_owners")
        for owner_value in owners:
            owner = PurePosixPath(_validate_output_owner(owner_value))
            for existing in output_owners:
                if owner == existing or owner in existing.parents or existing in owner.parents:
                    raise ValueError(
                        f"Overlapping output ownership: {existing} and {owner}"
                    )
            output_owners.append(owner)
    return plan, machines


def _ssh(machine: dict, script: str, *, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", machine["ssh_host"], "bash", "-s"],
        input=script,
        text=True,
        check=True,
        capture_output=capture,
    )


def inventory(machines_path: Path) -> None:
    for machine_id, machine in _machine_map(machines_path).items():
        repo = shlex.quote(machine["repo"])
        script = f"""set -euo pipefail
echo machine_id={shlex.quote(machine_id)}
echo ssh_host={shlex.quote(machine['ssh_host'])}
echo hostname=$(hostname)
echo repo={repo}
git -C {repo} rev-parse --abbrev-ref HEAD
git -C {repo} rev-parse HEAD
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
  echo GPUs=0
fi
"""
        print(_ssh(machine, script).stdout.rstrip())


def _remote_job_dir(machine: dict, plan_id: str, job_id: str) -> str:
    return str(PurePosixPath(machine["job_root"]) / plan_id / job_id)


def _remote_state(machine: dict) -> dict:
    repo = shlex.quote(machine["repo"])
    script = f"""set -euo pipefail
echo commit=$(git -C {repo} rev-parse HEAD)
echo dirty=$(git -C {repo} status --porcelain | wc -l)
if command -v nvidia-smi >/dev/null 2>&1; then
  echo gpu_ids=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | paste -sd, -)
else
  echo gpu_ids=
fi
"""
    rows = dict(
        line.split("=", 1) for line in _ssh(machine, script).stdout.splitlines()
    )
    return {
        "git_commit": rows["commit"],
        "dirty_count": int(rows["dirty"].strip()),
        "gpu_ids": {
            int(value) for value in rows["gpu_ids"].split(",") if value.strip()
        },
    }


def launch(plan_path: Path, machines_path: Path) -> None:
    plan, machines = validate_plan(plan_path, machines_path)
    machine_states = {
        machine_id: _remote_state(machines[machine_id])
        for machine_id in {job["machine_id"] for job in plan["jobs"]}
    }
    for machine_id, state in machine_states.items():
        if state["dirty_count"]:
            raise ValueError(f"Remote repository is dirty on {machine_id}")
    for job in plan["jobs"]:
        missing = set(job["gpu_ids"]) - machine_states[job["machine_id"]]["gpu_ids"]
        if missing:
            raise ValueError(
                f"Job {job['job_id']} requests unavailable GPUs on "
                f"{job['machine_id']}: {sorted(missing)}"
            )
    launch_rows = []
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
        script = f"""set -euo pipefail
job_dir={shlex.quote(job_dir)}
repo={shlex.quote(machine['repo'])}
test ! -e "$job_dir"
mkdir -p "$job_dir"
printf '%s' {shlex.quote(command_b64)} > "$job_dir/command.b64"
cat > "$job_dir/run.sh" <<'RUN_SCRIPT'
#!/usr/bin/env bash
set -uo pipefail
job_dir={shlex.quote(job_dir)}
repo={shlex.quote(machine['repo'])}
{cuda_line}
date -u +%Y-%m-%dT%H:%M:%SZ > "$job_dir/started_at_utc"
command=$(base64 --decode "$job_dir/command.b64")
cd "$repo"
set +e
bash -lc "$command" > "$job_dir/stdout.log" 2>&1
exit_code=$?
set -e
printf '%s\n' "$exit_code" > "$job_dir/exit_code"
date -u +%Y-%m-%dT%H:%M:%SZ > "$job_dir/finished_at_utc"
exit "$exit_code"
RUN_SCRIPT
chmod 700 "$job_dir/run.sh"
nohup "$job_dir/run.sh" >/dev/null 2>&1 &
printf '%s\n' "$!" > "$job_dir/pid"
cat "$job_dir/pid"
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
            }
        )
        print(f"launched {job['job_id']} on {job['machine_id']} pid={pid}")

    destination = CACHE / plan["plan_id"] / "launch.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "plan_id": plan["plan_id"],
                "launched_at_utc": _utc_now(),
                "plan_sha256": _sha256(plan_path),
                "machines_sha256": _sha256(machines_path),
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
if [ -f "$job_dir/exit_code" ]; then
  code=$(cat "$job_dir/exit_code")
  if [ "$code" = 0 ]; then echo complete; else echo failed:$code; fi
elif [ -f "$job_dir/pid" ] && kill -0 "$(cat "$job_dir/pid")" 2>/dev/null; then
  echo running
elif [ -f "$job_dir/pid" ]; then
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


def collect(plan_path: Path, machines_path: Path) -> None:
    plan, machines = validate_plan(plan_path, machines_path)
    for job in plan["jobs"]:
        machine = machines[job["machine_id"]]
        for owner in job["output_owners"]:
            source = f"{machine['ssh_host']}:{machine['repo'].rstrip('/')}/{owner.rstrip('/')}/"
            destination = CACHE / plan["plan_id"] / "collected" / job["job_id"] / owner
            destination.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["rsync", "-a", "--info=progress2", source, str(destination) + "/"],
                check=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machines", type=Path, default=DEFAULT_MACHINES)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("inventory")
    for action in ("validate", "launch", "status", "collect"):
        command = subparsers.add_parser(action)
        command.add_argument("--plan", type=Path, required=True)
    log_parser = subparsers.add_parser("logs")
    log_parser.add_argument("--plan", type=Path, required=True)
    log_parser.add_argument("--job", required=True)
    log_parser.add_argument("--lines", type=int, default=50)
    arguments = parser.parse_args()

    if arguments.action == "inventory":
        inventory(arguments.machines)
    elif arguments.action == "validate":
        plan, _ = validate_plan(arguments.plan, arguments.machines)
        print(f"valid plan: {plan['plan_id']} ({len(plan['jobs'])} jobs)")
    elif arguments.action == "launch":
        launch(arguments.plan, arguments.machines)
    elif arguments.action == "status":
        status(arguments.plan, arguments.machines)
    elif arguments.action == "logs":
        logs(arguments.plan, arguments.machines, arguments.job, arguments.lines)
    else:
        collect(arguments.plan, arguments.machines)


if __name__ == "__main__":
    main()
