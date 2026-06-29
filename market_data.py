from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.alpha import (
    AssetUniverseConfig,
    build_asset_price_panel,
    build_asset_return_panel,
)

logger = logging.getLogger("quant_platform.market_data")


@dataclass(frozen=True)
class MarketDataset:
    prices: pd.DataFrame
    returns: pd.DataFrame
    covariance: pd.DataFrame
    correlation: pd.DataFrame


def compute_sample_covariance(
    returns_panel: pd.DataFrame,
    min_observations: int = 60,
) -> pd.DataFrame:

    if not isinstance(returns_panel, pd.DataFrame):
        raise TypeError(
            "returns_panel must be a pandas DataFrame."
        )

    if returns_panel.empty:
        raise ValueError(
            "returns_panel is empty."
        )

    if len(returns_panel) < min_observations:
        raise ValueError(
            f"Insufficient observations for covariance estimation. "
            f"Required >= {min_observations}, "
            f"found {len(returns_panel)}."
        )

    if returns_panel.isna().any().any():
        raise ValueError(
            "returns_panel contains NaN values."
        )

    values = returns_panel.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "returns_panel contains non-finite values."
        )

    covariance = returns_panel.cov()

    if covariance.isna().any().any():
        raise ValueError(
            "Covariance matrix contains NaN values."
        )

    eigvals = np.linalg.eigvalsh(
        covariance.to_numpy(dtype=float)
    )

    min_eigenvalue = float(eigvals.min())

    if min_eigenvalue < -1e-8:
        raise ValueError(
            f"Covariance matrix is not positive semi-definite. "
            f"Minimum eigenvalue = {min_eigenvalue:.8f}"
        )

    condition_number = float(
        np.linalg.cond(
            covariance.to_numpy(dtype=float)
        )
    )

    logger.info(
        "Covariance diagnostics | assets=%d | observations=%d | "
        "min_eigenvalue=%.8f | condition_number=%.2e",
        covariance.shape[0],
        len(returns_panel),
        min_eigenvalue,
        condition_number,
    )

    if condition_number > 1e8:
        logger.warning(
            "Covariance matrix is ill-conditioned "
            "(condition number %.2e).",
            condition_number,
        )

    return covariance


def compute_correlation_matrix(
    covariance: pd.DataFrame,
) -> pd.DataFrame:

    if not isinstance(covariance, pd.DataFrame):
        raise TypeError(
            "covariance must be a pandas DataFrame."
        )

    if covariance.empty:
        raise ValueError(
            "covariance matrix is empty."
        )

    sigma = np.sqrt(
        np.diag(
            covariance.to_numpy(dtype=float)
        )
    )

    if np.any(sigma <= 0):
        raise ValueError(
            "Covariance matrix contains "
            "non-positive variances."
        )

    correlation = covariance.div(
        sigma,
        axis=0,
    ).div(
        sigma,
        axis=1,
    )

    diag = np.diag(
        correlation.to_numpy(dtype=float)
    )

    if not np.allclose(
        diag,
        np.ones_like(diag),
        atol=1e-8,
    ):
        raise ValueError(
            "Correlation matrix diagonal is invalid."
        )

    return correlation


def build_market_dataset(
    universe: AssetUniverseConfig,
    min_observations: int = 60,
) -> MarketDataset:

    logger.info(
        "Building market dataset for %d assets.",
        len(universe.asset_names),
    )

    prices_panel = build_asset_price_panel(
        universe
    )

    returns_panel = build_asset_return_panel(
        prices_panel
    )

    covariance = compute_sample_covariance(
        returns_panel,
        min_observations=min_observations,
    )

    correlation = compute_correlation_matrix(
        covariance
    )

    logger.info(
        "Market dataset assembled | assets=%d | "
        "price_obs=%d | return_obs=%d",
        prices_panel.shape[1],
        len(prices_panel),
        len(returns_panel),
    )

    return MarketDataset(
        prices=prices_panel,
        returns=returns_panel,
        covariance=covariance,
        correlation=correlation,
    )