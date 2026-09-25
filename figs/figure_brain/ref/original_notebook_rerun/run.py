#!/usr/bin/env python3
"""Reproduce Erfan's matched P12 neural and GPT-2 Small empirical panels.

The scientific code remains in exact source notebooks under ``source/``. This
runner only supplies durable local inputs/caches and executes path-adjusted
copies inside this analysis unit.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import io
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient


UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[2]
SOURCE = UNIT / "source"
CACHE = UNIT / "cache"
RESULTS = UNIT / "results"
PLOTS = UNIT / "plots"
EXECUTED = UNIT / "executed"
DATA = REPO / "data" / "brain"

SOURCE_COMMIT = {
    "brain": "a8c866d (final_figure.ipynb)",
    "gpt2": "44f6f93 + 9d1e1bb (final_figure_llm_fitted.ipynb)",
    "prompt_banks": "8cd7a02f2057fe2af5c19aae958f9771d9c1a8ce",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_matlab_string(f: h5py.File, ref_or_ds) -> str:
    ds = f[ref_or_ds] if isinstance(ref_or_ds, h5py.Reference) else ref_or_ds
    return "".join(chr(int(c)) for c in np.asarray(ds).ravel())


def _read_string_list(f: h5py.File, ds) -> list[str]:
    return [_read_matlab_string(f, r) for r in np.asarray(ds).ravel()]


def _read_char_matrix(arr: np.ndarray) -> list[str]:
    return ["".join(chr(c) for c in arr[:, i] if c) for i in range(arr.shape[1])]


def load_session(path: Path) -> dict:
    """Exact loader from NeuroAnalysis/brain_local_linearization.ipynb."""
    with h5py.File(path, "r") as f:
        ft = f["ft_data3"]
        fs = float(np.asarray(ft["fsample"]).squeeze())
        trial_refs = np.asarray(ft["trial"]).ravel()
        trials_raw = [np.asarray(f[r], dtype=np.float32) for r in trial_refs]
        n_ch = len(np.asarray(ft["label"]).ravel())
        trials_raw = [t.T if t.shape[1] == n_ch else t for t in trials_raw]
        min_len = min(t.shape[1] for t in trials_raw)
        data = np.stack([t[:, :min_len] for t in trials_raw])
        time = np.asarray(f[np.asarray(ft["time"]).ravel()[0]]).ravel()[:min_len]
        tdh = _read_string_list(f, f["TrialDetHeaders"])
        trials = pd.DataFrame(np.asarray(f["TrialDet"]).T, columns=tdh)
        bank1 = np.asarray(f["ChannelPairNamesBank1"])
        bank2 = np.asarray(f["ChannelPairNamesBank2"])
        names = _read_char_matrix(bank1) + _read_char_matrix(bank2)
        channels = pd.DataFrame({"pair_name": names, "bank": [1] * bank1.shape[1] + [2] * bank2.shape[1]})
        parc = np.asarray(f["ParcellationValues"])
        for i in range(parc.shape[0]):
            channels[f"parc{i}"] = parc[i, : len(channels)]
        ictal = _read_char_matrix(np.asarray(f["ch_ictal"]))
        channels["ictal"] = channels["pair_name"].isin(ictal)
        stim_flag = float(np.asarray(f["stim"]).squeeze())
    assert data.shape[1] == len(channels)
    return {"data": data, "time": time, "fs": fs, "trials": trials,
            "channels": channels, "stim_flag": stim_flag}


def prepare_brain_sessions(force: bool) -> None:
    out_dir = CACHE / "brain" / "sessions"
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "P12_NoStim2": DATA / "BIPOLFieldTripFormat_AlignedToImagePresent_P12_NoStim2.mat",
        "P12_Stim1": DATA / "BIPOLFieldTripFormat_AlignedToImagePresent_P12_Stim1.mat",
    }
    for name, src in mapping.items():
        out = out_dir / f"{name}.npz"
        if out.exists() and not force:
            print(f"reuse session cache: {out}", flush=True)
            continue
        print(f"decode {src.name}", flush=True)
        sess = load_session(src)
        np.savez_compressed(
            out, data=sess["data"], time=sess["time"], fs=sess["fs"],
            trials_csv=sess["trials"].to_csv(index=False),
            channels_csv=sess["channels"].to_csv(index=False),
            stim_flag=sess["stim_flag"], source_sha256=sha256(src),
        )
        print(f"cached {out} ({out.stat().st_size / 1e9:.2f} GB)", flush=True)


def historical_prompt_banks() -> dict[str, list[str]]:
    """Read constants from the exact historical generator and expand identically."""
    source = (SOURCE / "network_size_residual_explore.py").read_text()
    tree = ast.parse(source)
    vals = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "WITHIN_PROMPTS", "PROMPT_PREFIXES", "PROMPT_SUFFIXES", "OOD_PROMPTS"
                }:
                    vals[target.id] = ast.literal_eval(node.value)

    def expand(base: list[str], n: int = 50) -> list[str]:
        expanded, seen = [], set()
        for prompt in base:
            for prefix in vals["PROMPT_PREFIXES"]:
                for suffix in vals["PROMPT_SUFFIXES"]:
                    candidate = f"{prefix}{prompt}{suffix}".strip()
                    if candidate not in seen:
                        expanded.append(candidate)
                        seen.add(candidate)
                    if len(expanded) >= n:
                        return expanded
        raise RuntimeError("historical prompt expansion produced fewer than 50 prompts")

    return {
        "within": expand(vals["WITHIN_PROMPTS"]),
        "spanish": expand(vals["OOD_PROMPTS"]["Spanish"]),
        "long_context": expand(vals["OOD_PROMPTS"]["Long context"]),
    }


def prepare_prompt_banks(force: bool) -> None:
    out = CACHE / "gpt2" / "prompt_banks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    banks = historical_prompt_banks()
    payload = {
        "provenance_commit": SOURCE_COMMIT["prompt_banks"],
        "source_sha256": sha256(SOURCE / "network_size_residual_explore.py"),
        "counts": {k: len(v) for k, v in banks.items()},
        "banks": banks,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if force or not out.exists() or out.read_text() != text:
        out.write_text(text)
    print(f"prompt banks: {payload['counts']} -> {out}", flush=True)


def make_executable_notebook(kind: str) -> Path:
    src = SOURCE / ("final_figure_brain.ipynb" if kind == "brain" else "final_figure_gpt2.ipynb")
    nb = nbformat.read(src, as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        cell.source = cell.source.replace(
            'HERE = Path.cwd() if (Path.cwd() / "ANALYSIS_SPEC.md").exists() else Path("NeuroAnalysis")',
            "HERE = Path.cwd()",
        )
        cell.source = cell.source.replace('FIG_DIR = HERE / "figures"', 'FIG_DIR = HERE / "plots"')
        if kind == "gpt2" and "# identical prompt banks" in cell.source:
            start = cell.source.index("# identical prompt banks")
            end = cell.source.index("from transformers", start)
            replacement = '''# exact historical prompt banks, materialized by run.py
import json
_prompt_payload = json.loads((HERE / "cache" / "gpt2" / "prompt_banks.json").read_text())
prompt_banks = _prompt_payload["banks"]
print({k: len(v) for k, v in prompt_banks.items()})

'''
            cell.source = cell.source[:start] + replacement + cell.source[end:]
        if kind == "gpt2" and "# trajectories at the primary layer" in cell.source:
            start = cell.source.index("# trajectories at the primary layer")
            replacement = '''# trajectories at the primary layer, cached without changing numerics
TRAJ_CACHE = HERE / "cache" / "gpt2" / "trajectories.pt"
rng = np.random.default_rng(RANDOM_SEED)
dims = np.sort(rng.choice(768, size=STATE_DIMS, replace=False))
if TRAJ_CACHE.exists():
    _tc = torch.load(TRAJ_CACHE, map_location="cpu", weights_only=False)
    traj, traj_layers = _tc["traj"], _tc["traj_layers"]
else:
    traj = {name: hidden_trajectories(prompts, LAYER) for name, prompts in prompt_banks.items()}
    traj_layers = {L: hidden_trajectories(prompt_banks["within"], L) for L in range(1, 13)}
    TRAJ_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"traj": traj, "traj_layers": traj_layers, "dims": dims,
                "model": MODEL_NAME, "layer": LAYER, "state_dims": STATE_DIMS,
                "seed": RANDOM_SEED}, TRAJ_CACHE)
print("trajectories ready:", {k: len(v) for k, v in traj.items()},
      "| token lengths (within):", sorted(len(t) for t in traj["within"])[:5], "...")
'''
            cell.source = cell.source[:start] + replacement
        if kind == "brain" and "del pw\n" in cell.source:
            injection = '''del pw
_pre = HERE / "cache" / "brain" / "preprocessed"
_pre.mkdir(parents=True, exist_ok=True)
np.savez_compressed(_pre / "P12_NoStim2_theta_tokens.npz", tokens=tokens, baseline=baseline,
                    token_t=token_t, clean_idx=clean_idx)
'''
            cell.source = cell.source.replace("del pw\n", injection, 1)
    out = EXECUTED / f"{kind}_executable.ipynb"
    EXECUTED.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out)
    return out


def stage_session_links() -> None:
    for name in ("P12_NoStim2", "P12_Stim1"):
        src = CACHE / "brain" / "sessions" / f"{name}.npz"
        dst = CACHE / f"{name}.npz"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.relative_to(CACHE))


def execute_notebook(kind: str, force: bool) -> None:
    completed = EXECUTED / f"{kind}_completed.ipynb"
    if completed.exists() and not force:
        print(f"reuse completed notebook: {completed}", flush=True)
        return
    path = make_executable_notebook(kind)
    nb = nbformat.read(path, as_version=4)
    timeout = 7200 if kind == "brain" else 3600
    print(f"execute {kind}: {path.name}", flush=True)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(UNIT)}})
    client.execute()
    nbformat.write(nb, completed)
    print(f"completed {kind}: {completed}", flush=True)


def write_manifest(status: str) -> None:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "h5py", "torch", "transformers", "scikit-learn", "nbclient"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {
        "status": status,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "canonical_sources": SOURCE_COMMIT,
        "source_sha256": {p.name: sha256(p) for p in sorted(SOURCE.iterdir()) if p.is_file()},
        "raw_data_sha256": {p.name: sha256(p) for p in sorted(DATA.glob("*.mat"))},
        "packages": packages,
    }
    (UNIT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brain", action="store_true")
    parser.add_argument("--gpt2", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_brain = args.all or args.brain or not (args.brain or args.gpt2)
    run_gpt2 = args.all or args.gpt2 or not (args.brain or args.gpt2)
    for d in (CACHE, RESULTS, PLOTS, EXECUTED):
        d.mkdir(parents=True, exist_ok=True)
    write_manifest("running")
    if run_brain:
        prepare_brain_sessions(args.force)
        stage_session_links()
        execute_notebook("brain", args.force)
    if run_gpt2:
        prepare_prompt_banks(args.force)
        execute_notebook("gpt2", args.force)
    write_manifest("complete")


if __name__ == "__main__":
    main()
