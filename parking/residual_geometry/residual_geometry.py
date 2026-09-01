from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[2]
UNIT_DIR = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "data" / "representation_dynamics" / "residual_trajectories.npz"
CACHE_DIR = UNIT_DIR / "cache"
PLOTS_DIR = UNIT_DIR / "plots"
CACHE_PATH = CACHE_DIR / "residual_geometry.pkl"
PLOT_PATH = PLOTS_DIR / "residual_geometry.pdf"
EXPLAINED_VARIANCE_THRESHOLD = 0.95
EPSILON = 1e-12
CONDITION_ORDER = ["id", "paraphrase", "ood", "adversarial"]
CONDITION_LABELS = {
    "id": "ID",
    "paraphrase": "Paraphrase",
    "ood": "OOD",
    "adversarial": "Adversarial",
}
CONDITION_COLORS = {
    "id": "midnightblue",
    "paraphrase": "darkgreen",
    "ood": "darkred",
    "adversarial": "purple",
}


def setup_style() -> None:
    sns.set_theme(context="talk", style="ticks", palette="dark")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["lines.linewidth"] = 1
    plt.rcParams["patch.linewidth"] = 0
    plt.rcParams["image.interpolation"] = "none"
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.format"] = "pdf"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.transparent"] = False


def load_input() -> dict[str, np.ndarray]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing analysis-ready input: {DATA_PATH}")
    with np.load(DATA_PATH, allow_pickle=False) as source:
        required = {
            "residuals",
            "state_deviations",
            "prompt_split",
            "shift_condition",
            "normalized_depth",
            "model_id",
            "behavior",
        }
        missing = required.difference(source.files)
        if missing:
            raise ValueError(f"Missing arrays in {DATA_PATH}: {sorted(missing)}")
        return {name: source[name] for name in required}


def validate_input(data: dict[str, np.ndarray]) -> None:
    residuals = data["residuals"]
    states = data["state_deviations"]
    prompt_split = data["prompt_split"].astype(str)
    conditions = data["shift_condition"].astype(str)
    depth = data["normalized_depth"]

    if residuals.ndim != 3 or states.shape != residuals.shape:
        raise ValueError("residuals and state_deviations must share shape (records, layers, state_dimensions)")
    if prompt_split.shape != (residuals.shape[0],) or conditions.shape != (residuals.shape[0],):
        raise ValueError("prompt_split and shift_condition must contain one label per record")
    if depth.shape != (residuals.shape[1],):
        raise ValueError("normalized_depth must contain one value per analyzed layer")
    if not np.any(prompt_split == "calibration") or not np.any(prompt_split == "test"):
        raise ValueError("prompt_split must include calibration and test records")
    if not np.all(np.isfinite(residuals)) or not np.all(np.isfinite(states)):
        raise ValueError("residuals and state_deviations must be finite")
    if not np.all(np.diff(depth) >= 0) or depth.min() < 0 or depth.max() > 1:
        raise ValueError("normalized_depth must be ordered and lie within [0, 1]")
    unknown_conditions = set(np.unique(conditions)).difference(CONDITION_ORDER)
    if unknown_conditions:
        raise ValueError(f"Unknown shift conditions: {sorted(unknown_conditions)}")


def fit_disturbance_bases(
    calibration_residuals: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    layer_count = calibration_residuals.shape[1]
    means = np.empty((layer_count, calibration_residuals.shape[2]))
    retained_rank = np.empty(layer_count, dtype=int)
    bases: list[np.ndarray] = []

    for layer in range(layer_count):
        layer_residuals = calibration_residuals[:, layer, :]
        means[layer] = layer_residuals.mean(axis=0)
        centered = layer_residuals - means[layer]
        _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
        variance = singular_values**2
        if variance.sum() == 0:
            raise ValueError(f"Calibration residual variance is zero at layer {layer}")
        cumulative_variance = np.cumsum(variance) / variance.sum()
        rank = int(np.searchsorted(cumulative_variance, EXPLAINED_VARIANCE_THRESHOLD) + 1)
        retained_rank[layer] = rank
        bases.append(right_vectors[:rank].T)

    return bases, means, retained_rank


def compute_result(data: dict[str, np.ndarray]) -> dict[str, object]:
    validate_input(data)
    residuals = data["residuals"].astype(float)
    states = data["state_deviations"].astype(float)
    prompt_split = data["prompt_split"].astype(str)
    conditions = data["shift_condition"].astype(str)
    depth = data["normalized_depth"].astype(float)

    normalized_magnitude = np.linalg.norm(residuals, axis=2) / (
        np.linalg.norm(states, axis=2) + EPSILON
    )
    calibration_mask = prompt_split == "calibration"
    test_mask = prompt_split == "test"
    bases, means, retained_rank = fit_disturbance_bases(residuals[calibration_mask])

    explained_energy = np.empty((test_mask.sum(), residuals.shape[1]))
    test_residuals = residuals[test_mask]
    for layer, basis in enumerate(bases):
        centered = test_residuals[:, layer, :] - means[layer]
        projected = centered @ basis
        explained_energy[:, layer] = np.sum(projected**2, axis=1) / np.maximum(
            np.sum(centered**2, axis=1), EPSILON
        )

    condition_names = [name for name in CONDITION_ORDER if np.any(conditions[test_mask] == name)]
    mean_normalized_magnitude = {
        name: normalized_magnitude[test_mask][conditions[test_mask] == name].mean(axis=0)
        for name in condition_names
    }
    mean_explained_energy = {
        name: explained_energy[conditions[test_mask] == name].mean(axis=0)
        for name in condition_names
    }

    return {
        "normalized_depth": depth,
        "condition_names": condition_names,
        "mean_normalized_magnitude": mean_normalized_magnitude,
        "mean_explained_energy": mean_explained_energy,
        "retained_rank_fraction": retained_rank / residuals.shape[2],
        "model_id": str(data["model_id"]),
        "behavior": str(data["behavior"]),
    }


def load_or_compute(recompute: bool) -> dict[str, object]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if recompute and CACHE_PATH.exists():
        CACHE_PATH.unlink()
    if CACHE_PATH.exists():
        with CACHE_PATH.open("rb") as handle:
            return pickle.load(handle)
    result = compute_result(load_input())
    with CACHE_PATH.open("wb") as handle:
        pickle.dump(result, handle)
    return result


def plot_result(result: dict[str, object]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    depth = np.asarray(result["normalized_depth"])
    condition_names = list(result["condition_names"])
    mean_magnitude = dict(result["mean_normalized_magnitude"])
    mean_explained = dict(result["mean_explained_energy"])
    retained_rank_fraction = np.asarray(result["retained_rank_fraction"])

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for name in condition_names:
        axes[0].plot(depth, mean_magnitude[name], color=CONDITION_COLORS[name], label=CONDITION_LABELS[name])
        axes[1].plot(depth, mean_explained[name], color=CONDITION_COLORS[name])

    axes[0].set_title("A  Residual magnitude")
    axes[0].set_xlabel("Normalized depth")
    axes[0].set_ylabel(r"Mean $\|\xi_k\|_2/(\|x_k\|_2+\epsilon)$")
    magnitude_max = max(float(np.max(mean_magnitude[name])) for name in condition_names)
    magnitude_endpoint = max(magnitude_max, EPSILON)
    axes[0].set_ylim(0, 1.05 * magnitude_endpoint)
    axes[0].set_yticks([0, magnitude_endpoint])
    axes[0].legend(loc="best", fontsize=8, frameon=False)

    axes[1].set_title("B  Held-out coverage")
    axes[1].set_xlabel("Normalized depth")
    axes[1].set_ylabel("Mean explained residual energy")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_yticks([0, 1])

    axes[2].plot(depth, retained_rank_fraction, color="black")
    axes[2].set_title("C  Calibration rank")
    axes[2].set_xlabel("Normalized depth")
    axes[2].set_ylabel("Retained state-dimension fraction")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_yticks([0, 1])

    for ax in axes:
        ax.set_xlim(float(depth.min()), float(depth.max()))
        ax.set_xticks([float(depth.min()), float(depth.max())])
        ax.set_box_aspect(1)
        sns.despine(ax=ax, trim=True, offset=10)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.84, wspace=0.55)
    fig.savefig(PLOT_PATH, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze calibrated representation-dynamics residual geometry.")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Delete the existing cache and recompute the unit before plotting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_result(load_or_compute(recompute=args.recompute))


if __name__ == "__main__":
    main()
