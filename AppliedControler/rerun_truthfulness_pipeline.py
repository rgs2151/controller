"""Resumable truthfulness rerun pipeline with per-method status files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "AppliedControler" / "results_reports"
RUNS_DIR = REPORTS_DIR / "truthfulness_rerun_runs"


@dataclass(frozen=True)
class Job:
    name: str
    model: str
    method: str
    output_csv: str
    args: list[str]
    timeout_seconds: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", action="store_true", help="Delete prior rerun-run artifacts first.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip jobs whose output CSV already exists.")
    parser.add_argument(
        "--final-csv",
        type=Path,
        default=REPORTS_DIR / "paper_style_table_truthfulness_ours_plus_qwen14b_methods.csv",
    )
    parser.add_argument(
        "--final-tex",
        type=Path,
        default=REPORTS_DIR / "table2_style_truthfulness_ours_plus_qwen14b_methods.tex",
    )
    parser.add_argument(
        "--final-pdf",
        type=Path,
        default=REPORTS_DIR / "table2_style_truthfulness_ours_plus_qwen14b_methods.pdf",
    )
    return parser.parse_args()


def _job_status_path(job: Job) -> Path:
    return RUNS_DIR / f"{job.name}.status.json"


def _job_log_path(job: Job) -> Path:
    return RUNS_DIR / f"{job.name}.log"


def _job_output_path(job: Job) -> Path:
    return REPORTS_DIR / job.output_csv


def _jobs() -> list[Job]:
    py = sys.executable
    base = [py, "-u", "-m", "AppliedControler.eval_truthfulness_methods"]
    return [
        Job(
            name="distilgpt2_original",
            model="DistilGPT-2-ours",
            method="Original",
            output_csv="paper_style_table_truthfulness_distilgpt2_original_rerun.csv",
            args=base + [
                "--model-label", "DistilGPT-2-ours",
                "--model-name", "distilgpt2",
                "--device", "cuda:0",
                "--only-original",
                "--truthful-samples", "16",
                "--mmlu-samples", "24",
                "--max-length", "192",
                "--max-new-tokens", "24",
                "--output-csv", "AppliedControler/results_reports/paper_style_table_truthfulness_distilgpt2_original_rerun.csv",
            ],
            timeout_seconds=1200,
        ),
        Job(
            name="distilgpt2_alqr",
            model="DistilGPT-2-ours",
            method="A-LQR",
            output_csv="paper_style_table_truthfulness_distilgpt2_alqr_rerun.csv",
            args=base + [
                "--model-label", "DistilGPT-2-ours",
                "--model-name", "distilgpt2",
                "--device", "cuda:0",
                "--methods", "alqr",
                "--greedy-decoding",
                "--truthful-samples", "16",
                "--mmlu-samples", "24",
                "--calibration-prompts", "4",
                "--jacobian-prompts", "1",
                "--jacobian-vjp-chunk", "64",
                "--max-length", "192",
                "--max-new-tokens", "24",
                "--output-csv", "AppliedControler/results_reports/paper_style_table_truthfulness_distilgpt2_alqr_rerun.csv",
            ],
            timeout_seconds=1200,
        ),
        Job(
            name="distilgpt2_spid",
            model="DistilGPT-2-ours",
            method="S-PID",
            output_csv="paper_style_table_truthfulness_distilgpt2_spid_rerun.csv",
            args=base + [
                "--model-label", "DistilGPT-2-ours",
                "--model-name", "distilgpt2",
                "--device", "cuda:0",
                "--methods", "spid",
                "--kp", "0.05",
                "--ki", "0.0",
                "--kd", "0.0",
                "--greedy-decoding",
                "--truthful-samples", "16",
                "--mmlu-samples", "24",
                "--calibration-prompts", "4",
                "--jacobian-prompts", "1",
                "--max-length", "192",
                "--max-new-tokens", "24",
                "--output-csv", "AppliedControler/results_reports/paper_style_table_truthfulness_distilgpt2_spid_rerun.csv",
            ],
            timeout_seconds=1200,
        ),
        Job(
            name="distilgpt2_hinf",
            model="DistilGPT-2-ours",
            method="H-infinity",
            output_csv="paper_style_table_truthfulness_distilgpt2_hinf_rerun.csv",
            args=base + [
                "--model-label", "DistilGPT-2-ours",
                "--model-name", "distilgpt2",
                "--device", "cuda:0",
                "--methods", "hinf",
                "--greedy-decoding",
                "--truthful-samples", "16",
                "--mmlu-samples", "24",
                "--calibration-prompts", "4",
                "--jacobian-prompts", "1",
                "--jacobian-vjp-chunk", "64",
                "--max-length", "192",
                "--max-new-tokens", "24",
                "--output-csv", "AppliedControler/results_reports/paper_style_table_truthfulness_distilgpt2_hinf_rerun.csv",
            ],
            timeout_seconds=1500,
        ),
        Job(
            name="qwen14b_original",
            model="Qwen-2.5-14B-ours",
            method="Original",
            output_csv="paper_style_table_truthfulness_qwen14b_original_rerun.csv",
            args=base + [
                "--model-label", "Qwen-2.5-14B-ours",
                "--model-name", "Qwen/Qwen2.5-14B-Instruct",
                "--device", "cuda:0",
                "--quantized",
                "--quantized-device-map-auto",
                "--only-original",
                "--truthful-samples", "8",
                "--mmlu-samples", "12",
                "--max-length", "160",
                "--max-new-tokens", "16",
                "--output-csv", "AppliedControler/results_reports/paper_style_table_truthfulness_qwen14b_original_rerun.csv",
            ],
            timeout_seconds=1200,
        ),
        Job(
            name="qwen14b_alqr",
            model="Qwen-2.5-14B-ours",
            method="A-LQR",
            output_csv="paper_style_table_truthfulness_qwen14b_alqr_rerun.csv",
            args=base + [
                "--model-label", "Qwen-2.5-14B-ours",
                "--model-name", "Qwen/Qwen2.5-14B-Instruct",
                "--device", "cuda:0",
                "--controller-device", "cpu",
                "--quantized",
                "--quantized-device-map-auto",
                "--methods", "alqr",
                "--greedy-decoding",
                "--truthful-samples", "2",
                "--mmlu-samples", "2",
                "--calibration-prompts", "1",
                "--jacobian-prompts", "1",
                "--jacobian-vjp-chunk", "4",
                "--max-length", "96",
                "--max-new-tokens", "8",
                "--output-csv", "AppliedControler/results_reports/paper_style_table_truthfulness_qwen14b_alqr_rerun.csv",
            ],
            timeout_seconds=900,
        ),
        Job(
            name="qwen14b_spid",
            model="Qwen-2.5-14B-ours",
            method="S-PID",
            output_csv="paper_style_table_truthfulness_qwen14b_spid_rerun.csv",
            args=base + [
                "--model-label", "Qwen-2.5-14B-ours",
                "--model-name", "Qwen/Qwen2.5-14B-Instruct",
                "--device", "cuda:0",
                "--quantized",
                "--quantized-device-map-auto",
                "--methods", "spid",
                "--kp", "0.01",
                "--ki", "0.0",
                "--kd", "0.0",
                "--greedy-decoding",
                "--truthful-samples", "6",
                "--mmlu-samples", "8",
                "--calibration-prompts", "2",
                "--jacobian-prompts", "1",
                "--jacobian-vjp-chunk", "8",
                "--max-length", "160",
                "--max-new-tokens", "12",
                "--output-csv", "AppliedControler/results_reports/paper_style_table_truthfulness_qwen14b_spid_rerun.csv",
            ],
            timeout_seconds=1200,
        ),
        Job(
            name="qwen14b_hinf",
            model="Qwen-2.5-14B-ours",
            method="H-infinity",
            output_csv="paper_style_table_truthfulness_qwen14b_hinf_rerun.csv",
            args=base + [
                "--model-label", "Qwen-2.5-14B-ours",
                "--model-name", "Qwen/Qwen2.5-14B-Instruct",
                "--device", "cuda:0",
                "--controller-device", "cpu",
                "--quantized",
                "--quantized-device-map-auto",
                "--methods", "hinf",
                "--greedy-decoding",
                "--truthful-samples", "2",
                "--mmlu-samples", "2",
                "--calibration-prompts", "1",
                "--jacobian-prompts", "1",
                "--jacobian-vjp-chunk", "4",
                "--max-length", "96",
                "--max-new-tokens", "8",
                "--output-csv", "AppliedControler/results_reports/paper_style_table_truthfulness_qwen14b_hinf_rerun.csv",
            ],
            timeout_seconds=900,
        ),
    ]


def _write_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run_job(job: Job, skip_existing: bool) -> dict[str, object]:
    output_path = _job_output_path(job)
    status_path = _job_status_path(job)
    log_path = _job_log_path(job)
    if skip_existing and output_path.exists():
        payload = {
            "job": asdict(job),
            "status": "skipped_existing",
            "output_csv": str(output_path.relative_to(REPO_ROOT)),
            "timestamp": time.time(),
        }
        _write_status(status_path, payload)
        return payload

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    started = time.time()
    try:
        completed = subprocess.run(
            job.args,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=job.timeout_seconds,
            check=False,
        )
        log_path.write_text((completed.stdout or "") + (completed.stderr or ""), encoding="utf-8")
        payload = {
            "job": asdict(job),
            "status": "completed" if completed.returncode == 0 and output_path.exists() else "failed",
            "returncode": completed.returncode,
            "duration_seconds": time.time() - started,
            "output_csv": str(output_path.relative_to(REPO_ROOT)),
            "log_path": str(log_path.relative_to(REPO_ROOT)),
            "timestamp": time.time(),
        }
        _write_status(status_path, payload)
        return payload
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        log_path.write_text(output, encoding="utf-8")
        payload = {
            "job": asdict(job),
            "status": "timeout",
            "returncode": None,
            "duration_seconds": time.time() - started,
            "output_csv": str(output_path.relative_to(REPO_ROOT)),
            "log_path": str(log_path.relative_to(REPO_ROOT)),
            "timestamp": time.time(),
        }
        _write_status(status_path, payload)
        return payload


def _fresh_cleanup() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for path in RUNS_DIR.glob("*"):
        if path.is_file():
            path.unlink()
    for pattern in [
        "paper_style_table_truthfulness_*_rerun.csv",
        "paper_style_table_truthfulness_ours_plus_qwen14b_methods.csv",
        "table2_style_truthfulness_ours_plus_qwen14b_methods.tex",
        "table2_style_truthfulness_ours_plus_qwen14b_methods.pdf",
    ]:
        for path in REPORTS_DIR.glob(pattern):
            if path.is_file():
                path.unlink()


def _placeholder_row(job: Job, run_status: str) -> dict[str, object]:
    return {
        "model": job.model,
        "method": job.method,
        "ti_mean": float("nan"),
        "ti_std": float("nan"),
        "true_mean": float("nan"),
        "true_std": float("nan"),
        "info_mean": float("nan"),
        "info_std": float("nan"),
        "mmlu_mean": float("nan"),
        "mmlu_std": float("nan"),
        "run_status": run_status,
        "source_job": job.name,
    }


def _load_rows_from_csv(path: Path, run_status: str, source_job: str) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["run_status"] = run_status
        row["source_job"] = source_job
    return rows


def _combine_rows(statuses: list[dict[str, object]], final_csv: Path) -> None:
    rows: list[dict[str, object]] = []
    for status in statuses:
        job = Job(**status["job"])
        output_path = _job_output_path(job)
        if status["status"] in {"completed", "skipped_existing"} and output_path.exists():
            rows.extend(_load_rows_from_csv(output_path, str(status["status"]), job.name))
        else:
            rows.append(_placeholder_row(job, str(status["status"])))

    final_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "method",
        "ti_mean",
        "ti_std",
        "true_mean",
        "true_std",
        "info_mean",
        "info_std",
        "mmlu_mean",
        "mmlu_std",
        "run_status",
        "source_job",
    ]
    with final_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_final(final_csv: Path, final_tex: Path, final_pdf: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "AppliedControler.export_table2_truthfulness_style",
            "--input-csv",
            str(final_csv.relative_to(REPO_ROOT)),
            "--output-tex",
            str(final_tex.relative_to(REPO_ROOT)),
            "--output-pdf",
            str(final_pdf.relative_to(REPO_ROOT)),
            "--title",
            "Table 2. Summary of truthfulness evaluations (ours + Qwen 14B staged methods).",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def main() -> None:
    args = parse_args()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        _fresh_cleanup()

    statuses = [_run_job(job, args.skip_existing) for job in _jobs()]
    manifest_path = RUNS_DIR / "manifest.json"
    _write_status(manifest_path, {"jobs": statuses, "timestamp": time.time()})
    _combine_rows(statuses, args.final_csv)
    _render_final(args.final_csv, args.final_tex, args.final_pdf)
    print(f"Wrote final CSV: {args.final_csv}")
    print(f"Wrote final TeX: {args.final_tex}")
    print(f"Wrote final PDF: {args.final_pdf}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()