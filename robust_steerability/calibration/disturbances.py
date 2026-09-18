"""Kaz-aligned disturbance geometry from calibration residuals."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DisturbanceGeometry:
    """Layer-wise empirical covariance factors."""

    channels: torch.Tensor
    means: torch.Tensor
    retained_ranks: torch.Tensor
    explained_variance: torch.Tensor
    empirical_covariance: torch.Tensor
    covariance_relative_error: torch.Tensor


def fit_disturbance_geometry(
    residuals: torch.Tensor,
) -> DisturbanceGeometry:
    """Return ``D`` satisfying ``D D.T = Cov(residuals)`` at each layer."""

    if residuals.ndim != 3:
        raise ValueError("residuals must have shape (records, layers, state)")
    record_count, layer_count, state_dimension = residuals.shape
    if record_count < 2:
        raise ValueError("at least two calibration residuals are required")

    means = residuals.mean(dim=0)
    covariance_rank = min(record_count - 1, state_dimension)
    channels = torch.empty(
        layer_count,
        state_dimension,
        covariance_rank,
        device=residuals.device,
        dtype=residuals.dtype,
    )
    for layer_index in range(layer_count):
        centered = residuals[:, layer_index, :] - means[layer_index]
        _, singular_values, right_vectors = torch.linalg.svd(
            centered,
            full_matrices=False,
        )
        covariance_eigenvalue_roots = singular_values[:covariance_rank] / (
            record_count - 1
        ) ** 0.5
        channels[layer_index] = (
            right_vectors[:covariance_rank].T * covariance_eigenvalue_roots
        )

    centered = residuals - means.unsqueeze(0)
    empirical_covariance = torch.einsum(
        "nli,nlj->lij", centered, centered
    ) / (record_count - 1)
    reconstructed_covariance = channels @ channels.transpose(-1, -2)
    covariance_error = torch.linalg.matrix_norm(
        empirical_covariance - reconstructed_covariance,
        dim=(-2, -1),
    )
    covariance_norm = torch.linalg.matrix_norm(
        empirical_covariance,
        dim=(-2, -1),
    )
    relative_error = torch.where(
        covariance_norm > 0,
        covariance_error / covariance_norm,
        covariance_error,
    )
    if not torch.isfinite(relative_error).all() or relative_error.max() > 1e-4:
        raise ValueError(
            "D @ D.T does not reproduce the empirical residual covariance"
        )

    return DisturbanceGeometry(
        channels=channels,
        means=means,
        retained_ranks=torch.full(
            (layer_count,),
            covariance_rank,
            device=residuals.device,
            dtype=torch.int64,
        ),
        explained_variance=torch.ones(
            layer_count,
            device=residuals.device,
            dtype=residuals.dtype,
        ),
        empirical_covariance=empirical_covariance,
        covariance_relative_error=relative_error,
    )
