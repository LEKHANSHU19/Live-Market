"""Risk analytics utilities from the E1 notebook.

This module preserves the E1 risk mathematics while making the workflow
reusable from notebooks, scripts, and tests.  It covers log-return integration,
rolling and EWMA volatility, 10-day VaR, VaR/ES sensitivities, breach
classification, and Basel traffic-light backtesting.

Examples
--------
>>> import pandas as pd
>>> prices = pd.Series([100.0, 101.0, 99.0], index=pd.date_range("2024-01-02", periods=3))
>>> returns = compute_log_returns(prices)
>>> len(returns)
2

>>> comparison = get_breaches(pd.Series([-0.02, -0.03]), pd.Series([-0.01, -0.04]))
>>> comparison["Breach"].tolist()
[False, True]
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import binom, norm

from quant_platform.config import BACKTEST, VAR
from quant_platform.data import compute_log_returns as _data_compute_log_returns


@dataclass(frozen=True)
class TrafficLightResult:
    """Basel traffic-light backtest summary."""

    observations: int
    breaches: int
    breach_rate: float
    green_threshold: int
    yellow_threshold: int
    zone: str


def _as_series(values: pd.Series | ArrayLike, name: str) -> pd.Series:
    """Convert array-like values to a float Series without mutating input."""

    if isinstance(values, pd.Series):
        series = values.copy()
    else:
        series = pd.Series(values)

    if series.empty:
        raise ValueError(f"{name} must contain at least one observation")

    try:
        series = series.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc

    if not np.isfinite(series.dropna().to_numpy(dtype=float)).all():
        raise ValueError(f"{name} must contain only finite numeric values")

    return series


def _validate_probability(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0 or value >= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def compute_log_returns(prices: pd.Series | ArrayLike) -> pd.Series:
    """Compute log returns ``ln(P_t / P_{t-1})`` from prices.

    Delegates to ``quant_platform.data.compute_log_returns`` so the data
    ingestion layer remains the single source of truth.
    """

    return _data_compute_log_returns(prices)


def compute_rolling_vol(
    returns: pd.Series | ArrayLike,
    window: int = VAR.ROLLING_WINDOW,
) -> pd.Series:
    """Compute rolling daily volatility using the E1 21-day sample standard deviation."""

    _validate_positive_int("window", window)
    return_series = _as_series(returns, "returns")
    if len(return_series) < window:
        raise ValueError("returns length must be at least window")

    min_periods = VAR.ROLLING_MIN_PERIODS if window == VAR.ROLLING_WINDOW else window
    rolling_vol = return_series.rolling(window=window, min_periods=min_periods).std()
    rolling_vol.name = "rolling_vol"
    return rolling_vol


def compute_ewma_variance(
    returns: pd.Series | ArrayLike,
    lambda_: float = VAR.LAMBDA_EWMA,
) -> pd.Series:
    """Compute EWMA variance with E1 initialization and recursion.

    The notebook initializes ``ev[0]`` with the full-sample variance and then
    applies ``ev[t] = lambda * ev[t-1] + (1 - lambda) * returns[t-1]**2``.
    """

    _validate_probability("lambda_", lambda_)
    return_series = _as_series(returns, "returns")
    values = return_series.to_numpy(dtype=float)
    if values.size < 2:
        raise ValueError("returns must have at least 2 observations")

    ewma = np.zeros(values.size, dtype=float)
    ewma[0] = np.var(values, ddof=0)
    for idx in range(1, values.size):
        ewma[idx] = lambda_ * ewma[idx - 1] + (1.0 - lambda_) * values[idx - 1] ** 2

    return pd.Series(ewma, index=return_series.index, name="ewma_variance")


def compute_10d_var(
    daily_vol: pd.Series | ArrayLike,
    confidence: float = VAR.CONFIDENCE,
    horizon: int = VAR.HORIZON_DAYS,
) -> pd.Series:
    """Compute horizon-scaled parametric VaR from daily volatility.

    Preserves the E1 formula ``norm.ppf(1 - confidence) * daily_vol * sqrt(horizon)``.
    The returned VaR is negative, matching the notebook's loss-threshold
    comparison ``forward_return < VaR``.
    """

    _validate_probability("confidence", confidence)
    _validate_positive_int("horizon", horizon)
    vol_series = _as_series(daily_vol, "daily_vol")
    if (vol_series.dropna() < 0.0).any():
        raise ValueError("daily_vol must not contain negative values")

    var = norm.ppf(1.0 - confidence) * vol_series * sqrt(horizon)
    var.name = f"{horizon}d_var"
    return var


def compute_var_es_sensitivities(
    weights: ArrayLike,
    sigmas: ArrayLike,
    correlation: ArrayLike,
    confidence: float = VAR.CONFIDENCE,
    means: ArrayLike | None = None,
) -> dict[str, NDArray[np.float64] | float]:
    """Compute parametric VaR and ES sensitivities from the E1 formulas."""

    _validate_probability("confidence", confidence)
    w = np.asarray(weights, dtype=float)
    sigma = np.asarray(sigmas, dtype=float)
    rho = np.asarray(correlation, dtype=float)

    if w.ndim != 1 or sigma.ndim != 1:
        raise ValueError("weights and sigmas must be one-dimensional arrays")
    if w.size == 0 or sigma.size == 0:
        raise ValueError("weights and sigmas must not be empty")
    if w.size != sigma.size:
        raise ValueError("weights and sigmas must have the same length")
    if rho.shape != (w.size, w.size):
        raise ValueError("correlation must be a square matrix matching weights")
    if not np.isfinite(w).all() or not np.isfinite(sigma).all() or not np.isfinite(rho).all():
        raise ValueError("weights, sigmas, and correlation must contain only finite values")
    if (sigma <= 0.0).any():
        raise ValueError("sigmas must contain only positive values")

    if means is None:
        mu = np.zeros_like(w)
    else:
        mu = np.asarray(means, dtype=float)
        if mu.shape != w.shape:
            raise ValueError("means must match weights shape")
        if not np.isfinite(mu).all():
            raise ValueError("means must contain only finite values")

    covariance = np.diag(sigma) @ rho @ np.diag(sigma)
    portfolio_variance = float(w @ covariance @ w)
    if portfolio_variance <= 0.0:
        raise ValueError("portfolio variance must be positive")

    portfolio_std = sqrt(portfolio_variance)
    z_alpha = float(norm.ppf(1.0 - confidence))
    phi_z = float(norm.pdf(z_alpha))
    risk_contribution = (covariance @ w) / portfolio_std
    d_var = mu + z_alpha * risk_contribution
    d_es = mu - (phi_z / (1.0 - confidence)) * risk_contribution

    return {
        "covariance": covariance,
        "portfolio_std": portfolio_std,
        "z_alpha": z_alpha,
        "phi_z": phi_z,
        "dVaR": d_var,
        "dES": d_es,
    }


def get_breaches(
    var: pd.Series | ArrayLike,
    forward_returns: pd.Series | ArrayLike,
    start: Any | None = None,
    end: Any | None = None,
) -> pd.DataFrame:
    """Compare forward returns against VaR and flag breaches.

    This preserves the E1 comparison ``Fwd < VaR`` and returns columns
    ``VaR``, ``Fwd``, and ``Breach``.
    """

    var_series = _as_series(var, "var")
    fwd_series = _as_series(forward_returns, "forward_returns")

    if isinstance(var_series.index, pd.DatetimeIndex) or isinstance(fwd_series.index, pd.DatetimeIndex):
        comparison = pd.DataFrame({"VaR": var_series[start:end], "Fwd": fwd_series[start:end]})
    else:
        comparison = pd.DataFrame({"VaR": var_series, "Fwd": fwd_series})

    comparison = comparison.dropna()
    if comparison.empty:
        raise ValueError("no overlapping non-NaN observations are available for breach testing")

    comparison["Breach"] = comparison["Fwd"] < comparison["VaR"]
    return comparison


def traffic_light(
    comparison: pd.DataFrame,
    alpha: float = BACKTEST.H0_BREACH_PROB,
) -> TrafficLightResult:
    """Return Basel traffic-light classification for a breach comparison table."""

    _validate_probability("alpha", alpha)
    if not isinstance(comparison, pd.DataFrame):
        raise TypeError("comparison must be a pd.DataFrame")
    if "Breach" not in comparison.columns:
        raise ValueError("comparison must contain a 'Breach' column")
    if comparison.empty:
        raise ValueError("comparison must not be empty")

    observations = len(comparison)
    breaches = int(comparison["Breach"].astype(bool).sum())
    breach_rate = breaches / observations * 100.0
    green_threshold = int(binom.ppf(BACKTEST.TRAFFIC_GREEN_PROB, observations, alpha))
    yellow_threshold = int(binom.ppf(BACKTEST.TRAFFIC_YELLOW_PROB, observations, alpha))
    zone = (
        BACKTEST.ZONE_GREEN
        if breaches <= green_threshold
        else (BACKTEST.ZONE_YELLOW if breaches <= yellow_threshold else BACKTEST.ZONE_RED)
    )

    return TrafficLightResult(
        observations=observations,
        breaches=breaches,
        breach_rate=breach_rate,
        green_threshold=green_threshold,
        yellow_threshold=yellow_threshold,
        zone=zone,
    )
