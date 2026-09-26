#!/usr/bin/env python3
"""Screen conflict-shift linear residuals across all no-stimulation subjects."""

from __future__ import annotations

import argparse
import io
import re
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn import plotting as niplot
from scipy.io import loadmat
from scipy.signal import butter, hilbert, sosfiltfilt


HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "brain"
CACHE = HERE / "cache" / "conflict_screen"
RESULTS = HERE / "results"
PLOTS = HERE / "plots"

HORIZONS_MS = (20, 40, 100, 200, 300, 400, 500, 700)
TOKEN_STRIDE_MS = 20
CAUSAL_WINDOW_MS = 40
EPOCH_MS = (-500, 1500)
RANDOM_SEED = 0
ID_COLOR = "#c9c9c9"
OOD_COLOR = "#8B1E1E"

AREA_ORDER = (
    "dorsolateral prefrontal",
    "ventrolateral prefrontal",
    "premotor / dorsomedial frontal",
    "sensorimotor",
    "temporal / peri-insular",
    "posterior temporal",
    "parieto-occipital",
)
AREA_COLORS = dict(zip(
    AREA_ORDER,
    ("#1f4ea1", "#5b8bd0", "#2e9c5c", "#8bc34a", "#e0a13c", "#c1272d", "#7b4fa6"),
))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
    "pdf.fonttype": 42,
})


def _h5_string(handle: h5py.File, ref_or_dataset) -> str:
    dataset = handle[ref_or_dataset] if isinstance(ref_or_dataset, h5py.Reference) else ref_or_dataset
    return "".join(chr(int(value)) for value in np.asarray(dataset).ravel())


def _h5_string_list(handle: h5py.File, dataset) -> list[str]:
    return [_h5_string(handle, ref) for ref in np.asarray(dataset).ravel()]


def _char_matrix(array) -> list[str]:
    array = np.asarray(array)
    if array.ndim < 2:
        return []
    return ["".join(chr(int(value)) for value in array[:, index] if value)
            for index in range(array.shape[1])]


def _trial_table(array: np.ndarray, headers: list[str]) -> pd.DataFrame:
    array = np.atleast_2d(array)
    width = min(array.shape[1], len(headers))
    return pd.DataFrame(array[:, :width], columns=headers[:width])


def metadata(path: Path) -> dict:
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            fieldtrip = handle["ft_data3"]
            headers = _h5_string_list(handle, handle["TrialDetHeaders"])
            trials = _trial_table(np.asarray(handle["TrialDet"]).T, headers)
            names = (_char_matrix(handle["ChannelPairNamesBank1"])
                     + _char_matrix(handle["ChannelPairNamesBank2"]))
            ictal = set(_char_matrix(handle["ch_ictal"]))
            n_trials = len(np.asarray(fieldtrip["trial"]).ravel())
    else:
        loaded = loadmat(
            path,
            variable_names=["TrialDet", "TrialDetHeaders", "ChannelPairNamesBank1",
                            "ChannelPairNamesBank2", "ch_ictal"],
            squeeze_me=True,
            struct_as_record=False,
        )
        headers = [str(value).strip() for value in np.atleast_1d(loaded["TrialDetHeaders"])]
        trials = _trial_table(np.atleast_2d(loaded["TrialDet"]), headers)
        names = [str(value).strip() for value in np.atleast_1d(loaded["ChannelPairNamesBank1"])]
        names += [str(value).strip() for value in np.atleast_1d(loaded["ChannelPairNamesBank2"])
                  if str(value).strip()]
        ictal = {str(value).strip() for value in np.atleast_1d(loaded["ch_ictal"])
                 if str(value).strip()}
        n_trials = len(trials)
    return {"trials": trials.iloc[:n_trials].reset_index(drop=True),
            "names": names, "ictal": ictal, "n_trials": n_trials}


def coarse_area(x: float, y: float, z: float) -> str:
    """Irfan's coordinate-derived coarse anatomical grouping."""
    if y <= -60:
        return "parieto-occipital"
    if y <= -35 and z < 25:
        return "posterior temporal"
    if z < 20:
        return "temporal / peri-insular"
    if z >= 40 and y >= 10:
        return "dorsolateral prefrontal"
    if y >= 10:
        return "ventrolateral prefrontal"
    if y <= -12:
        return "sensorimotor"
    return "premotor / dorsomedial frontal"


def electrode_table(path: Path) -> pd.DataFrame:
    """Load bipolar-pair MNI midpoints without reading neural trial arrays."""
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            names = (_char_matrix(handle["ChannelPairNamesBank1"])
                     + _char_matrix(handle["ChannelPairNamesBank2"]))
            ictal = set(_char_matrix(handle["ch_ictal"]))
            parcellation = np.asarray(handle["ParcellationValues"], dtype=float)
    else:
        loaded = loadmat(
            path,
            variable_names=["ChannelPairNamesBank1", "ChannelPairNamesBank2",
                            "ch_ictal", "ParcellationValues"],
            squeeze_me=True,
            struct_as_record=False,
        )
        names = [str(value).strip() for value in np.atleast_1d(loaded["ChannelPairNamesBank1"])]
        names += [str(value).strip() for value in np.atleast_1d(loaded["ChannelPairNamesBank2"])
                  if str(value).strip()]
        ictal = {str(value).strip() for value in np.atleast_1d(loaded["ch_ictal"])
                 if str(value).strip()}
        parcellation = np.asarray(loaded["ParcellationValues"], dtype=float)

    if parcellation.shape[0] == len(names):
        coordinates = parcellation[:, 4:7]
    elif parcellation.shape[1] == len(names):
        coordinates = parcellation[4:7].T
    else:
        raise RuntimeError(
            f"Parcellation/channel mismatch in {path.name}: "
            f"{parcellation.shape} versus {len(names)} names"
        )
    table = pd.DataFrame(coordinates, columns=["x", "y", "z"])
    table.insert(0, "pair_name", names)
    table["ictal"] = table["pair_name"].isin(ictal)
    table["lead"] = table["pair_name"].str.extract(r"^([A-Za-z]+)")
    valid = np.isfinite(table[["x", "y", "z"]]).all(axis=1)
    table.loc[valid, "area"] = [
        coarse_area(row.x, row.y, row.z)
        for row in table.loc[valid, ["x", "y", "z"]].itertuples(index=False)
    ]
    return table


def plot_electrodes(axis, selection: dict, selected_lead: str, view: str):
    """Project one participant's clean electrodes onto a lateral or frontal view."""
    table = electrode_table(selection["paths"][0])
    table = table[
        table["pair_name"].isin(selection["pair_names"])
        & ~table["ictal"]
        & table["area"].notna()
    ].copy()
    if view == "l":
        # A sagittal projection does not encode left-right depth. Reflect both
        # hemispheres onto the same lateral silhouette so right-sided implants
        # are not discarded by Nilearn's left-lateral display mode.
        table["x"] = -np.abs(table["x"])
    display = niplot.plot_glass_brain(
        None,
        display_mode=view,
        figure=axis.figure,
        axes=axis,
        annotate=False,
        black_bg=False,
        alpha=0.38,
    )
    for area in AREA_ORDER:
        subset = table[table["area"] == area]
        if len(subset):
            display.add_markers(
                subset[["x", "y", "z"]].to_numpy(),
                marker_color=AREA_COLORS[area],
                marker_size=11,
            )
    highlighted = table[table["lead"] == selected_lead]
    for area in AREA_ORDER:
        subset = highlighted[highlighted["area"] == area]
        if len(subset):
            display.add_markers(
                subset[["x", "y", "z"]].to_numpy(),
                marker_color=AREA_COLORS[area],
                marker_size=24,
            )


def discover_sessions() -> list[dict]:
    grouped: dict[tuple[int, int], list[Path]] = defaultdict(list)
    pattern = re.compile(r"_P(\d+)_NoStim(\d+)")
    for path in DATA.glob("BIPOLFieldTripFormat_AlignedToImagePresent_P*_NoStim*.mat"):
        match = pattern.search(path.name)
        if match:
            grouped[(int(match.group(1)), int(match.group(2)))].append(path)

    candidates = []
    for (subject, session), paths in grouped.items():
        infos = [metadata(path) for path in sorted(paths)]
        common_clean = set(infos[0]["names"]) - infos[0]["ictal"]
        for info in infos[1:]:
            common_clean &= set(info["names"]) - info["ictal"]
        lead_names: dict[str, list[str]] = defaultdict(list)
        ordered_common_clean = []
        for name in infos[0]["names"]:
            if name not in common_clean:
                continue
            ordered_common_clean.append(name)
            match = re.match(r"^([A-Za-z]+)", name)
            if match:
                lead_names[match.group(1)].append(name)
        lead_names = {lead: names for lead, names in lead_names.items() if len(names) >= 5}
        if not lead_names:
            continue
        n_trials = sum(info["n_trials"] for info in infos)
        candidates.append({"subject": subject, "session": session,
                           "paths": sorted(paths), "infos": infos,
                           "n_trials": n_trials, "lead_pairs": lead_names,
                           "pair_names": ordered_common_clean})

    selected = []
    for subject in sorted({item["subject"] for item in candidates}):
        options = [item for item in candidates if item["subject"] == subject]
        selected.append(sorted(options, key=lambda item: (item["n_trials"], item["session"]),
                               reverse=True)[0])
    return selected


def _load_hdf5_selected(path: Path, pair_names: list[str]) -> tuple[np.ndarray, np.ndarray, float, pd.DataFrame]:
    with h5py.File(path, "r") as handle:
        fieldtrip = handle["ft_data3"]
        names = (_char_matrix(handle["ChannelPairNamesBank1"])
                 + _char_matrix(handle["ChannelPairNamesBank2"]))
        indices = np.asarray([names.index(name) for name in pair_names])
        n_channels = len(names)
        trial_refs = np.asarray(fieldtrip["trial"]).ravel()
        arrays = []
        for ref in trial_refs:
            dataset = handle[ref]
            if dataset.shape[1] == n_channels:
                arrays.append(np.asarray(dataset[:, indices], dtype=np.float32).T)
            elif dataset.shape[0] == n_channels:
                arrays.append(np.asarray(dataset[indices, :], dtype=np.float32))
            else:
                raise RuntimeError(f"Unrecognized trial orientation in {path.name}: {dataset.shape}")
        min_length = min(array.shape[1] for array in arrays)
        data = np.stack([array[:, :min_length] for array in arrays])
        time = np.asarray(handle[np.asarray(fieldtrip["time"]).ravel()[0]]).ravel()[:min_length]
        fs = float(np.asarray(fieldtrip["fsample"]).squeeze())
        headers = _h5_string_list(handle, handle["TrialDetHeaders"])
        trials = _trial_table(np.asarray(handle["TrialDet"]).T, headers).iloc[:len(data)]
    return data, time, fs, trials.reset_index(drop=True)


def _load_v5_selected(path: Path, pair_names: list[str]) -> tuple[np.ndarray, np.ndarray, float, pd.DataFrame]:
    loaded = loadmat(
        path,
        variable_names=["ft_data3", "TrialDet", "TrialDetHeaders",
                        "ChannelPairNamesBank1", "ChannelPairNamesBank2"],
        squeeze_me=True,
        struct_as_record=False,
    )
    names = [str(value).strip() for value in np.atleast_1d(loaded["ChannelPairNamesBank1"])]
    names += [str(value).strip() for value in np.atleast_1d(loaded["ChannelPairNamesBank2"])
              if str(value).strip()]
    indices = np.asarray([names.index(name) for name in pair_names])
    fieldtrip = loaded["ft_data3"]
    raw_trials = list(np.atleast_1d(fieldtrip.trial).ravel())
    arrays = []
    for trial in raw_trials:
        trial = np.asarray(trial)
        if trial.shape[0] == len(names):
            arrays.append(np.asarray(trial[indices, :], dtype=np.float32))
        elif trial.shape[1] == len(names):
            arrays.append(np.asarray(trial[:, indices], dtype=np.float32).T)
        else:
            raise RuntimeError(f"Unrecognized trial orientation in {path.name}: {trial.shape}")
    min_length = min(array.shape[1] for array in arrays)
    data = np.stack([array[:, :min_length] for array in arrays])
    raw_time = np.atleast_1d(fieldtrip.time).ravel()[0]
    time = np.asarray(raw_time, dtype=float).ravel()[:min_length]
    # P14's first FieldTrip time cell is stored in integer milliseconds while
    # the remaining legacy files/cells use seconds.
    if np.nanmax(np.abs(time)) > 100:
        time = time / 1000.0
    fs = float(np.asarray(fieldtrip.fsample).squeeze())
    headers = [str(value).strip() for value in np.atleast_1d(loaded["TrialDetHeaders"])]
    trials = _trial_table(np.atleast_2d(loaded["TrialDet"]), headers).iloc[:len(data)]
    return data, time, fs, trials.reset_index(drop=True)


def load_selected_session(selection: dict):
    chunks = []
    for path in selection["paths"]:
        loader = _load_hdf5_selected if h5py.is_hdf5(path) else _load_v5_selected
        chunks.append(loader(path, selection["pair_names"]))
    fs_values = {round(chunk[2], 8) for chunk in chunks}
    if len(fs_values) != 1:
        raise RuntimeError(f"Sampling rates differ within P{selection['subject']} session")
    min_length = min(chunk[0].shape[-1] for chunk in chunks)
    data = np.concatenate([chunk[0][..., :min_length] for chunk in chunks], axis=0)
    trials = pd.concat([chunk[3] for chunk in chunks], ignore_index=True)
    return data, chunks[0][1][:min_length], chunks[0][2], trials


def theta_tokens(data: np.ndarray, time: np.ndarray, fs: float):
    token_times = np.arange(CAUSAL_WINDOW_MS, EPOCH_MS[1] + 1,
                            TOKEN_STRIDE_MS, dtype=float)
    time_ms = time * 1000.0
    token_masks = [(time_ms > value - CAUSAL_WINDOW_MS) & (time_ms <= value)
                   for value in token_times]
    baseline_mask = (time_ms >= EPOCH_MS[0]) & (time_ms < 0)
    sos = butter(4, [4, 8], btype="bandpass", fs=fs, output="sos")
    tokens = np.empty((len(data), len(token_times), data.shape[1]), dtype=np.float32)
    baseline = np.empty((len(data), data.shape[1]), dtype=np.float32)
    for start in range(0, len(data), 32):
        stop = min(start + 32, len(data))
        filtered = sosfiltfilt(sos, data[start:stop], axis=-1)
        power = np.abs(hilbert(filtered, axis=-1)).astype(np.float32) ** 2
        baseline[start:stop] = power[:, :, baseline_mask].mean(axis=-1)
        tokens[start:stop] = np.stack(
            [power[:, :, mask].mean(axis=-1) for mask in token_masks], axis=1)
    return tokens, baseline


def cached_tokens(selection: dict, force: bool):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"P{selection['subject']}_NoStim{selection['session']}_all_clean_theta.npz"
    if path.exists() and not force:
        cached = np.load(path, allow_pickle=False)
        if np.isfinite(cached["tokens"]).all() and np.isfinite(cached["baseline"]).all():
            return (cached["tokens"], cached["baseline"],
                    pd.read_csv(io.StringIO(str(cached["trials_csv"]))),
                    cached["pair_names"].astype(str))
    data, time, fs, trials = load_selected_session(selection)
    valid = ~np.isnan(data).any(axis=(1, 2))
    data, trials = data[valid], trials.loc[valid].reset_index(drop=True)
    tokens, baseline = theta_tokens(data, time, fs)
    np.savez_compressed(path, tokens=tokens, baseline=baseline,
                        trials_csv=trials.to_csv(index=False),
                        pair_names=np.asarray(selection["pair_names"]),
                        source_names=np.asarray([item.name for item in selection["paths"]]))
    return tokens, baseline, trials, np.asarray(selection["pair_names"])


def split_indices(n: int):
    order = np.random.default_rng(RANDOM_SEED).permutation(n)
    n_train = int(round(0.6 * n))
    n_validation = int(round(0.2 * n))
    return order[:n_train], order[n_train:n_train + n_validation], order[n_train + n_validation:]


def pairs(states: np.ndarray, horizon_steps: int):
    x = states[:, :-horizon_steps].reshape(-1, states.shape[-1])
    y = states[:, horizon_steps:].reshape(-1, states.shape[-1])
    return x, y


def population_r2(y, prediction):
    return 1.0 - ((y - prediction) ** 2).sum() / ((y - y.mean(axis=0)) ** 2).sum()


def fit_ridge(x_train, y_train, x_validation, y_validation):
    augmented = np.hstack([x_train, np.ones((len(x_train), 1))])
    xtx, xty = augmented.T @ augmented, augmented.T @ y_train
    eye = np.eye(augmented.shape[1]); eye[-1, -1] = 0
    best = None
    for penalty in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        weights = np.linalg.solve(xtx + penalty * eye, xty)
        prediction = np.hstack([x_validation, np.ones((len(x_validation), 1))]) @ weights
        score = population_r2(y_validation, prediction)
        if best is None or score > best[0]:
            best = score, weights, penalty
    _, weights, penalty = best
    return weights[:-1].T, weights[-1], penalty


def analyze(selection: dict, force: bool, horizons_ms=HORIZONS_MS) -> list[dict]:
    tokens, baseline, trials, pair_names = cached_tokens(selection, force)
    conflict = pd.to_numeric(trials["Conflict"], errors="coerce").to_numpy()
    low = np.flatnonzero(conflict <= 1)
    high = np.flatnonzero(conflict >= 2)
    train_rel, validation_rel, test_rel = split_indices(len(low))
    train, validation, test = low[train_rel], low[validation_rel], low[test_rel]
    rows = []
    for lead in sorted(selection["lead_pairs"]):
        lead_names = selection["lead_pairs"][lead]
        columns = np.asarray([np.flatnonzero(pair_names == name)[0] for name in lead_names])
        lead_tokens, lead_baseline = tokens[:, :, columns], baseline[:, columns]
        mean = lead_baseline[train].mean(axis=0)
        std = lead_baseline[train].std(axis=0) + 1e-12
        states = (lead_tokens - mean[None, None, :]) / std[None, None, :]
        for horizon_ms in horizons_ms:
            steps = horizon_ms // TOKEN_STRIDE_MS
            x_train, y_train = pairs(states[train], steps)
            x_validation, y_validation = pairs(states[validation], steps)
            a, b, penalty = fit_ridge(x_train, y_train, x_validation, y_validation)
            values = {}
            for split, indices in (("ID", test), ("OOD", high)):
                x, y = pairs(states[indices], steps)
                prediction = x @ a.T + b
                values[split] = float(np.median(np.sqrt(((y - prediction) ** 2).mean(axis=1))))
                rows.append({
                    "subject": f"P{selection['subject']}", "subject_number": selection["subject"],
                    "session": f"NoStim{selection['session']}", "lead": lead,
                    "n_channels": states.shape[-1], "n_low": len(low), "n_high": len(high),
                    "horizon_ms": horizon_ms, "split": split,
                    "linear_residual_rms": values[split], "ridge_lambda": penalty,
                })
            for row in rows[-2:]:
                row["ood_minus_id"] = values["OOD"] - values["ID"]
                row["ood_over_id"] = values["OOD"] / values["ID"]
    return rows


def plot(results: pd.DataFrame, selections: list[dict]):
    subjects = sorted(results["subject_number"].unique())
    selection_by_subject = {item["subject"]: item for item in selections}
    n_columns = 4
    n_rows = int(np.ceil(len(subjects) / n_columns))
    fig = plt.figure(figsize=(13, 13))
    grid = fig.add_gridspec(
        n_rows, n_columns, left=0.065, right=0.99, bottom=0.055, top=0.95,
        wspace=0.34, hspace=0.55,
    )
    bar_axes = []
    for index, subject_number in enumerate(subjects):
        row, column = divmod(index, n_columns)
        cell = grid[row, column].subgridspec(
            2, 1, height_ratios=[1.28, 0.50], hspace=0.42,
        )
        anatomy = cell[0, 0].subgridspec(1, 2, wspace=0.02)
        lateral_axis = fig.add_subplot(anatomy[0, 0])
        frontal_axis = fig.add_subplot(anatomy[0, 1])
        axis = fig.add_subplot(cell[1, 0])
        bar_axes.append(axis)
        subset = results[results["subject_number"] == subject_number]
        pivot = subset.pivot(index="horizon_ms", columns="split",
                             values="linear_residual_rms").sort_index()
        x = np.arange(len(pivot)); width = 0.36
        axis.bar(x - width / 2, pivot["ID"], width, color=ID_COLOR)
        axis.bar(x + width / 2, pivot["OOD"], width, color=OOD_COLOR)
        meta = subset.iloc[0]
        axis.set_title(f"P{subject_number} · {meta['lead']}", fontsize=13,
                       weight="bold", pad=5)
        axis.set_xticks(x, [str(value) for value in pivot.index], fontsize=9)
        axis.set_ylim(0, subset["linear_residual_rms"].max() * 1.12)
        axis.grid(axis="y", color="#D9DDE1", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("Horizon (ms)", fontsize=11)
        axis.set_ylabel("Linear residual RMS (z)", fontsize=11)
        axis.tick_params(axis="y", labelsize=10)
        plot_electrodes(
            lateral_axis, selection_by_subject[subject_number], str(meta["lead"]), "l"
        )
        plot_electrodes(
            frontal_axis, selection_by_subject[subject_number], str(meta["lead"]), "y"
        )
    handles = [plt.Rectangle((0, 0), 1, 1, color=ID_COLOR, label="ID (low conflict)"),
               plt.Rectangle((0, 0), 1, 1, color=OOD_COLOR, label="OOD (high conflict)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=2, frameon=False, fontsize=12)
    for extension in ("png", "pdf"):
        fig.savefig(PLOTS / f"all_subject_conflict_linear_residual.{extension}",
                    dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(exist_ok=True); PLOTS.mkdir(exist_ok=True)
    selections = discover_sessions()
    all_lead_path = RESULTS / "all_subject_all_leads_conflict_linear_residual.csv"
    if all_lead_path.exists() and not args.force:
        all_leads = pd.read_csv(all_lead_path)
        present = set(all_leads["horizon_ms"].astype(int).unique())
        missing = tuple(value for value in HORIZONS_MS if value not in present)
        if missing:
            print(f"extend cached all-lead fit table with horizons: {missing}", flush=True)
            new_rows = []
            for index, selection in enumerate(selections, start=1):
                print(f"[{index}/{len(selections)}] P{selection['subject']} "
                      f"NoStim{selection['session']} cached states; new horizons {missing}", flush=True)
                new_rows.extend(analyze(selection, False, missing))
            all_leads = pd.concat([all_leads, pd.DataFrame(new_rows)], ignore_index=True)
            all_leads = (all_leads.drop_duplicates(
                ["subject_number", "session", "lead", "horizon_ms", "split"], keep="last")
                .sort_values(["subject_number", "lead", "horizon_ms", "split"]))
            all_leads.to_csv(all_lead_path, index=False)
        else:
            print(f"reuse complete all-lead fit table: {all_lead_path}", flush=True)
    else:
        all_rows = []
        for index, selection in enumerate(selections, start=1):
            print(f"[{index}/{len(selections)}] P{selection['subject']} "
                  f"NoStim{selection['session']} all clean leads "
                  f"({len(selection['pair_names'])} channels, {selection['n_trials']} trials)", flush=True)
            all_rows.extend(analyze(selection, args.force))
        all_leads = pd.DataFrame(all_rows).sort_values(
            ["subject_number", "lead", "horizon_ms", "split"])
        all_leads.to_csv(all_lead_path, index=False)
    lead_scores = (all_leads.drop_duplicates(["subject_number", "lead", "horizon_ms"])
                   .groupby(["subject", "subject_number", "session", "lead", "n_channels"], as_index=False)
                   .agg(mean_ood_minus_id=("ood_minus_id", "mean"),
                        gap_at_max_horizon=("ood_minus_id", lambda values: values.iloc[-1]),
                        mean_ood_over_id=("ood_over_id", "mean")))
    chosen = (lead_scores.sort_values(
        ["subject_number", "mean_ood_over_id", "mean_ood_minus_id", "lead"],
        ascending=[True, False, False, True]).drop_duplicates("subject_number"))
    results = all_leads.merge(chosen[["subject_number", "lead"]],
                              on=["subject_number", "lead"], how="inner")
    results.to_csv(RESULTS / "all_subject_conflict_linear_residual.csv", index=False)
    lead_scores.sort_values(["subject_number", "mean_ood_minus_id"], ascending=[True, False]).to_csv(
        RESULTS / "all_subject_lead_ranking.csv", index=False)
    ranking = (results.drop_duplicates(["subject_number", "horizon_ms"])
               .groupby(["subject", "subject_number", "session", "lead", "n_channels"], as_index=False)
               .agg(mean_ood_minus_id=("ood_minus_id", "mean"),
                    gap_at_max_horizon=("ood_minus_id", lambda values: values.iloc[-1]),
                    mean_ood_over_id=("ood_over_id", "mean"))
               .sort_values(["mean_ood_over_id", "mean_ood_minus_id"], ascending=False))
    ranking.to_csv(RESULTS / "all_subject_conflict_ranking.csv", index=False)
    plot(results, selections)
    print(ranking.to_string(index=False), flush=True)
    print(PLOTS / "all_subject_conflict_linear_residual.png", flush=True)


if __name__ == "__main__":
    main()
