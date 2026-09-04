"""Low-rank disturbance geometry from calibration residuals."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DisturbanceGeometry:
    """Layer-wise covariance factors with zero padding to a common rank."""

    channels: torch.Tensor
    means: torch.Tensor
    retained_ranks: torch.Tensor
    explained_variance: torch.Tensor


def fit_disturbance_geometry(
    residuals: torch.Tensor,
    variance_threshold: float = 0.95,
) -> DisturbanceGeometry:
    """Fit PCA covariance factors from ``(records, layers, state)`` residuals."""

    if residuals.ndim != 3:
        raise ValueError("residuals must have shape (records, layers, state)")
    if not 0.0 < variance_threshold <= 1.0:
        raise ValueError("variance_threshold must be in (0, 1]")
    record_count, layer_count, state_dimension = residuals.shape
    if record_count < 2:
        raise ValueError("at least two calibration residuals are required")

    means = residuals.mean(dim=0)
    decompositions = []
    retained_ranks = []
    explained = []
    for layer_index in range(layer_count):
        centered = residuals[:, layer_index, :] - means[layer_index]
        _, singular_values, right_vectors = torch.linalg.svd(
            centered,
            full_matrices=False,
        )
        eigenvalues = singular_values.square() / (record_count - 1)
        fractions = eigenvalues / eigenvalues.sum()
        cumulative = torch.cumsum(fractions, dim=0)
        rank = int(
            torch.searchsorted(
                cumulative,
                torch.tensor(
                    variance_threshold,
                    device=cumulative.device,
                    dtype=cumulative.dtype,
                ),
            ).item()
            + 1
        )
        rank = min(rank, state_dimension, right_vectors.shape[0])
        decompositions.append((right_vectors, eigenvalues))
        retained_ranks.append(rank)
        explained.append(float(cumulative[rank - 1].item()))

    common_rank = max(retained_ranks)
    channels = torch.zeros(
        layer_count,
        state_dimension,
        common_rank,
        device=residuals.device,
        dtype=residuals.dtype,
    )
    for layer_index, ((right_vectors, eigenvalues), rank) in enumerate(
        zip(decompositions, retained_ranks, strict=True)
    ):
        channels[layer_index, :, :rank] = (
            right_vectors[:rank].T * eigenvalues[:rank].sqrt()
        )
    return DisturbanceGeometry(
        channels=channels,
        means=means,
        retained_ranks=torch.tensor(retained_ranks, device=residuals.device),
        explained_variance=torch.tensor(
            explained,
            device=residuals.device,
            dtype=residuals.dtype,
        ),
    )
