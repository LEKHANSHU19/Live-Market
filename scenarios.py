"""Scenario analysis and stress testing utilities.

The functions in this module reuse existing platform engines only.  They do
not introduce new quantitative models; they call pricing, risk, and portfolio
functions with shocked inputs and return structured dictionaries for later
reporting or Jupyter consumption.

Examples
--------
>>> result = volatility_shock_analysis(shock_values=[0.10, 0.20, 0.30])
>>> {"base", "stressed", "difference", "pct_change"}.issubset(result)
True

>>> stress = portfolio_stress_test(scenario_returns=[-0.20, -0.25, -0.35, -0.15])
>>> "portfolio_pnl" in stress["stressed"]
True
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from quant_platform.config import MARKET, VAR
from quant_platform.portfolio import compute_mvo_weights
from quant_platform.pricing import black_scholes_values
from quant_platform.risk import compute_10d_var, compute_ewma_variance, compute_rolling_vol


ScenarioMethod = Literal["rolling", "ewma"]


def _as_1d_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Convert values to a finite one-dimensional float array."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _pct_change(base: float, stressed: float) -> float:
    """Return percentage change, using NaN when the base value is zero."""

    if np.isclose(base, 0.0):
        return float("nan")
    return float((stressed - base) / base * 100.0)


def _dict_difference(
    base: dict[str, float],
    stressed: dict[str, dict[str, float]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Compute difference and percentage-change dictionaries for price scenarios."""

    difference: dict[str, dict[str, float]] = {}
    pct_change: dict[str, dict[str, float]] = {}
    for scenario_name, prices in stressed.items():
        difference[scenario_name] = {option: prices[option] - base[option] for option in base}
        pct_change[scenario_name] = {
            option: _pct_change(base[option], prices[option]) for option in base
        }
    return difference, pct_change


def volatility_shock_analysis(
    shock_values: ArrayLike,
    s0: float = MARKET.S0,
    strike: float = MARKET.K,
    maturity: float = MARKET.T,
    rate: float = MARKET.R,
    base_volatility: float = MARKET.SIGMA,
) -> dict[str, Any]:
    """Price all supported options under volatility shocks.

    Each shocked value is passed directly to ``black_scholes_values``.  This is
    a sensitivity scenario, not a new pricing model.
    """

    shocks = _as_1d_array(shock_values, "shock_values")
    if (shocks <= 0.0).any():
        raise ValueError("shock_values must contain only positive volatilities")

    base = black_scholes_values(s0, strike, maturity, rate, base_volatility)
    stressed = {
        f"volatility={volatility:.6g}": black_scholes_values(
            s0, strike, maturity, rate, float(volatility)
        )
        for volatility in shocks
    }
    difference, pct_change = _dict_difference(base, stressed)

    return {
        "base": base,
        "stressed": stressed,
        "difference": difference,
        "pct_change": pct_change,
        "inputs": {
            "s0": s0,
            "strike": strike,
            "maturity": maturity,
            "rate": rate,
            "base_volatility": base_volatility,
            "shock_values": shocks,
        },
    }


def interest_rate_shock_analysis(
    shock_values: ArrayLike,
    s0: float = MARKET.S0,
    strike: float = MARKET.K,
    maturity: float = MARKET.T,
    base_rate: float = MARKET.R,
    volatility: float = MARKET.SIGMA,
) -> dict[str, Any]:
    """Price all supported options under interest-rate shocks."""

    shocks = _as_1d_array(shock_values, "shock_values")

    base = black_scholes_values(s0, strike, maturity, base_rate, volatility)
    stressed = {
        f"rate={rate:.6g}": black_scholes_values(s0, strike, maturity, float(rate), volatility)
        for rate in shocks
    }
    difference, pct_change = _dict_difference(base, stressed)

    return {
        "base": base,
        "stressed": stressed,
        "difference": difference,
        "pct_change": pct_change,
        "inputs": {
            "s0": s0,
            "strike": strike,
            "maturity": maturity,
            "base_rate": base_rate,
            "volatility": volatility,
            "shock_values": shocks,
        },
    }


def portfolio_stress_test(
    scenario_returns: ArrayLike,
    mu: ArrayLike = MARKET.MVO_MU,
    sigma: ArrayLike = MARKET.MVO_SIGMA,
    rho: ArrayLike = MARKET.MVO_RHO,
    target_return: float = MARKET.MVO_TARGET_RETURN,
) -> dict[str, Any]:
    """Compute baseline MVO weights and apply scenario returns to portfolio PnL."""

    baseline = compute_mvo_weights(mu=mu, sigma=sigma, rho=rho, target_return=target_return)
    weights = np.asarray(baseline["weights"], dtype=float)
    returns = _as_1d_array(scenario_returns, "scenario_returns")
    if returns.size != weights.size:
        raise ValueError("scenario_returns length must match the number of portfolio weights")

    portfolio_pnl = float(weights @ returns)
    base_expected_return = float(baseline["expected_return"])

    stressed = {
        "scenario_returns": returns,
        "portfolio_pnl": portfolio_pnl,
    }
    difference = {
        "portfolio_pnl_minus_expected_return": portfolio_pnl - base_expected_return,
    }
    pct_change = {
        "portfolio_pnl_vs_expected_return": _pct_change(base_expected_return, portfolio_pnl),
    }

    return {
        "base": baseline,
        "stressed": stressed,
        "difference": difference,
        "pct_change": pct_change,
        "inputs": {
            "scenario_returns": returns,
            "target_return": target_return,
        },
    }


def var_stress_test(
    returns: pd.Series | ArrayLike,
    shock_multipliers: ArrayLike,
    method: ScenarioMethod = "rolling",
    window: int = VAR.ROLLING_WINDOW,
    lambda_: float = VAR.LAMBDA_EWMA,
    confidence: float = VAR.CONFIDENCE,
    horizon: int = VAR.HORIZON_DAYS,
) -> dict[str, Any]:
    """Compute VaR under volatility multiplier stress scenarios.

    ``method="rolling"`` uses ``compute_rolling_vol``.  ``method="ewma"`` uses
    ``np.sqrt(compute_ewma_variance(...))``.  Stressed VaR is then computed by
    passing multiplied volatility series to ``compute_10d_var``.
    """

    multipliers = _as_1d_array(shock_multipliers, "shock_multipliers")
    if (multipliers <= 0.0).any():
        raise ValueError("shock_multipliers must contain only positive values")
    if method not in {"rolling", "ewma"}:
        raise ValueError("method must be 'rolling' or 'ewma'")

    if method == "rolling":
        base_vol = compute_rolling_vol(returns, window=window)
    else:
        base_vol = np.sqrt(compute_ewma_variance(returns, lambda_=lambda_))
        base_vol.name = "ewma_vol"

    base_var = compute_10d_var(base_vol, confidence=confidence, horizon=horizon)
    stressed = {
        f"multiplier={multiplier:.6g}": compute_10d_var(
            base_vol * float(multiplier), confidence=confidence, horizon=horizon
        )
        for multiplier in multipliers
    }
    difference = {name: stressed_var - base_var for name, stressed_var in stressed.items()}
    pct_change = {
        name: (stressed_var - base_var) / base_var.replace(0.0, np.nan) * 100.0
        for name, stressed_var in stressed.items()
    }

    return {
        "base": {
            "volatility": base_vol,
            "var": base_var,
        },
        "stressed": stressed,
        "difference": difference,
        "pct_change": pct_change,
        "inputs": {
            "method": method,
            "shock_multipliers": multipliers,
            "window": window,
            "lambda_": lambda_,
            "confidence": confidence,
            "horizon": horizon,
        },
    }

