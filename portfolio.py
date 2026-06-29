"""Portfolio analytics utilities from the E1 notebook.

This module preserves the mean-variance optimization matrix algebra from E1
and exposes structured outputs for downstream reporting and Jupyter notebooks.

Examples
--------
>>> result = compute_mvo_scalars()
>>> {"A", "B", "C", "D"}.issubset(result)
True

>>> weights = compute_mvo_weights()
>>> round(weights["expected_return"], 6) == round(MARKET.MVO_TARGET_RETURN, 6)
True
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quant_platform.config import MARKET


def _as_1d_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Convert input values to a finite one-dimensional float array."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_square_matrix(values: ArrayLike, size: int, name: str) -> NDArray[np.float64]:
    """Convert input values to a finite square matrix of the requested size."""

    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size})")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _validate_portfolio_inputs(
    mu: ArrayLike,
    sigma: ArrayLike,
    rho: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Validate and normalize E1 MVO inputs."""

    mu_array = _as_1d_array(mu, "mu")
    sigma_array = _as_1d_array(sigma, "sigma")
    if mu_array.size != sigma_array.size:
        raise ValueError("mu and sigma must have the same length")
    if (sigma_array <= 0.0).any():
        raise ValueError("sigma must contain only positive values")

    rho_matrix = _as_square_matrix(rho, mu_array.size, "rho")
    if not np.allclose(rho_matrix, rho_matrix.T):
        raise ValueError("rho must be symmetric")
    if not np.allclose(np.diag(rho_matrix), 1.0):
        raise ValueError("rho must have ones on the diagonal")

    return mu_array, sigma_array, rho_matrix


def compute_mvo_scalars(
    mu: ArrayLike = MARKET.MVO_MU,
    sigma: ArrayLike = MARKET.MVO_SIGMA,
    rho: ArrayLike = MARKET.MVO_RHO,
) -> dict[str, Any]:
    """Compute E1 mean-variance optimization scalars.

    Preserves the notebook matrix algebra exactly:

    ``Sigma = diag(sigma) @ rho @ diag(sigma)``
    ``A = 1' Sigma^-1 1``
    ``B = 1' Sigma^-1 mu``
    ``C = mu' Sigma^-1 mu``
    ``D = A*C - B**2``
    """

    mu_array, sigma_array, rho_matrix = _validate_portfolio_inputs(mu, sigma, rho)
    covariance = np.diag(sigma_array) @ rho_matrix @ np.diag(sigma_array)

    try:
        inverse_covariance = np.linalg.inv(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance matrix must be invertible") from exc

    ones = np.ones(mu_array.size)
    a_scalar = float(ones @ inverse_covariance @ ones)
    b_scalar = float(ones @ inverse_covariance @ mu_array)
    c_scalar = float(mu_array @ inverse_covariance @ mu_array)
    d_scalar = float(a_scalar * c_scalar - b_scalar**2)

    if np.isclose(d_scalar, 0.0):
        raise ValueError("D scalar is too close to zero for stable MVO weights")

    return {
        "A": a_scalar,
        "B": b_scalar,
        "C": c_scalar,
        "D": d_scalar,
        "covariance": covariance,
        "inverse_covariance": inverse_covariance,
        "ones": ones,
        "mu": mu_array,
        "sigma": sigma_array,
        "rho": rho_matrix,
    }


def compute_mvo_weights(
    mu: ArrayLike = MARKET.MVO_MU,
    sigma: ArrayLike = MARKET.MVO_SIGMA,
    rho: ArrayLike = MARKET.MVO_RHO,
    target_return: float = MARKET.MVO_TARGET_RETURN,
) -> dict[str, Any]:
    """Compute analytical E1 mean-variance optimal weights.

    Preserves the notebook equations:

    ``lambda = (C - target_return * B) / D``
    ``gamma = (target_return * A - B) / D``
    ``weights = lambda * (Sigma^-1 @ 1) + gamma * (Sigma^-1 @ mu)``
    """

    if not np.isfinite(target_return):
        raise ValueError("target_return must be finite")

    scalars = compute_mvo_scalars(mu=mu, sigma=sigma, rho=rho)
    mu_array = scalars["mu"]
    covariance = scalars["covariance"]
    inverse_covariance = scalars["inverse_covariance"]
    ones = scalars["ones"]

    lambda_scalar = (scalars["C"] - target_return * scalars["B"]) / scalars["D"]
    gamma_scalar = (target_return * scalars["A"] - scalars["B"]) / scalars["D"]
    weights = lambda_scalar * (inverse_covariance @ ones) + gamma_scalar * (inverse_covariance @ mu_array)

    expected_return = float(weights @ mu_array)
    variance = float(weights @ covariance @ weights)
    if variance < 0.0 and np.isclose(variance, 0.0):
        variance = 0.0
    if variance < 0.0:
        raise ValueError("computed portfolio variance is negative")

    return {
        "weights": weights,
        "expected_return": expected_return,
        "variance": variance,
        "volatility": float(np.sqrt(variance)),
        "lambda": float(lambda_scalar),
        "gamma": float(gamma_scalar),
        "scalars": scalars,
        "asset_names": MARKET.MVO_ASSET_NAMES,
    }

