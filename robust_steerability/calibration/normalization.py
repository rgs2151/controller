"""Calibration-only coordinate normalization."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WhiteningTransform:
    mean: torch.Tensor
    matrix: torch.Tensor

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) @ self.matrix.T


def fit_whitening(
    calibration_values: torch.Tensor,
    eigenvalue_floor: float = 1e-8,
) -> WhiteningTransform:
    """Fit a whitening transform using calibration observations only."""

    if calibration_values.ndim != 2:
        raise ValueError("calibration_values must have shape (records, dimensions)")
    mean = calibration_values.mean(dim=0)
    centered = calibration_values - mean
    covariance = centered.T @ centered / (len(calibration_values) - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    inverse_scale = eigenvalues.clamp_min(eigenvalue_floor).rsqrt()
    matrix = eigenvectors @ torch.diag(inverse_scale) @ eigenvectors.T
    return WhiteningTransform(mean=mean, matrix=matrix)
