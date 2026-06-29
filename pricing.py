"""Option pricing utilities from the E2 derivatives notebook.

The module keeps the notebook mathematics intact while exposing reusable,
validated functions for Black-Scholes prices, Monte Carlo terminal-price
simulation, payoff valuation, and antithetic variates.
"""

from __future__ import annotations

from collections.abc import Callable
from math import erf, exp, isfinite, log, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray


OptionName = str
SchemeName = str
PayoffFunction = Callable[[NDArray[np.float64], float], NDArray[np.float64]]

VALID_SCHEMES: tuple[SchemeName, ...] = ("Euler", "Milstein", "Exact")


def _european_call_payoff(st: NDArray[np.float64], strike: float) -> NDArray[np.float64]:
    return np.maximum(st - strike, 0.0)


def _european_put_payoff(st: NDArray[np.float64], strike: float) -> NDArray[np.float64]:
    return np.maximum(strike - st, 0.0)


def _binary_call_payoff(st: NDArray[np.float64], strike: float) -> NDArray[np.float64]:
    return (st > strike).astype(float)


def _binary_put_payoff(st: NDArray[np.float64], strike: float) -> NDArray[np.float64]:
    return (st <= strike).astype(float)


PAYOFF_REGISTRY: dict[OptionName, PayoffFunction] = {
    "European Call": _european_call_payoff,
    "European Put": _european_put_payoff,
    "Binary Call": _binary_call_payoff,
    "Binary Put": _binary_put_payoff,
}


def _validate_positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")


def _validate_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_market_inputs(s0: float, strike: float, maturity: float, rate: float, volatility: float) -> None:
    _validate_positive("s0", s0)
    _validate_positive("strike", strike)
    _validate_positive("maturity", maturity)
    _validate_finite("rate", rate)
    _validate_positive("volatility", volatility)


def _validate_rng(rng: np.random.Generator | None) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    return rng


def _validate_count(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def norm_cdf(x: float | ArrayLike) -> float | NDArray[np.float64]:
    """Return the standard normal cumulative distribution function.

    The implementation follows the notebook formula using the error function.
    Scalars return a float; array-like inputs return a NumPy array.
    """

    values = np.asarray(x, dtype=float)
    cdf = 0.5 * (1.0 + np.vectorize(erf, otypes=[float])(values / sqrt(2.0)))
    if np.isscalar(x):
        return float(cdf)
    return cdf


def black_scholes_values(
    s0: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> dict[OptionName, float]:
    """Return Black-Scholes prices for all supported option payoffs."""

    _validate_market_inputs(s0, strike, maturity, rate, volatility)

    sqrt_t = sqrt(maturity)
    d1 = (log(s0 / strike) + (rate + 0.5 * volatility**2) * maturity) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    discount = exp(-rate * maturity)

    return {
        "European Call": s0 * norm_cdf(d1) - strike * discount * norm_cdf(d2),
        "European Put": strike * discount * norm_cdf(-d2) - s0 * norm_cdf(-d1),
        "Binary Call": discount * norm_cdf(d2),
        "Binary Put": discount * norm_cdf(-d2),
    }


def simulate_terminal_prices(
    s0: float,
    maturity: float,
    rate: float,
    volatility: float,
    n_paths: int,
    n_steps: int,
    scheme: SchemeName = "Exact",
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Simulate terminal GBM prices under Euler, Milstein, or Exact schemes."""

    _validate_positive("s0", s0)
    _validate_positive("maturity", maturity)
    _validate_finite("rate", rate)
    _validate_positive("volatility", volatility)
    _validate_count("n_paths", n_paths)
    _validate_count("n_steps", n_steps)
    if scheme not in VALID_SCHEMES:
        raise ValueError(f"scheme must be one of {VALID_SCHEMES}")

    generator = _validate_rng(rng)
    dt = maturity / n_steps
    sqrt_dt = sqrt(dt)
    shocks = generator.standard_normal(size=(n_paths, n_steps))
    prices = np.full(n_paths, float(s0), dtype=float)

    if scheme == "Euler":
        for step in range(n_steps):
            d_w = sqrt_dt * shocks[:, step]
            prices = prices + rate * prices * dt + volatility * prices * d_w
    elif scheme == "Milstein":
        for step in range(n_steps):
            d_w = sqrt_dt * shocks[:, step]
            prices = (
                prices
                + rate * prices * dt
                + volatility * prices * d_w
                + 0.5 * volatility**2 * prices * (d_w**2 - dt)
            )
    else:
        for step in range(n_steps):
            prices = prices * np.exp(
                (rate - 0.5 * volatility**2) * dt + volatility * sqrt_dt * shocks[:, step]
            )

    return prices


def discounted_payoffs(
    terminal_prices: ArrayLike,
    strike: float,
    rate: float,
    maturity: float,
    option_name: OptionName | None = None,
) -> dict[OptionName, NDArray[np.float64]] | NDArray[np.float64]:
    """Return discounted payoff arrays for one option or all registered options."""

    _validate_positive("strike", strike)
    _validate_finite("rate", rate)
    _validate_positive("maturity", maturity)

    prices = np.asarray(terminal_prices, dtype=float)
    if prices.ndim == 0 or prices.size == 0:
        raise ValueError("terminal_prices must contain at least one value")
    if not np.all(np.isfinite(prices)):
        raise ValueError("terminal_prices must contain only finite values")

    discount = exp(-rate * maturity)
    if option_name is not None:
        if option_name not in PAYOFF_REGISTRY:
            raise ValueError(f"option_name must be one of {tuple(PAYOFF_REGISTRY)}")
        return discount * PAYOFF_REGISTRY[option_name](prices, strike)

    return {name: discount * payoff(prices, strike) for name, payoff in PAYOFF_REGISTRY.items()}


def price_from_payoffs(payoffs: ArrayLike) -> tuple[float, float, float, float]:
    """Return price, standard error, and 95% confidence interval from payoffs."""

    values = np.asarray(payoffs, dtype=float)
    if values.ndim == 0 or values.size < 2:
        raise ValueError("payoffs must contain at least two values")
    if not np.all(np.isfinite(values)):
        raise ValueError("payoffs must contain only finite values")

    price = float(values.mean())
    standard_error = float(values.std(ddof=1) / sqrt(values.size))
    return price, standard_error, price - 1.96 * standard_error, price + 1.96 * standard_error


def antithetic_price_all(
    s0: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    n_pairs: int,
    rng: np.random.Generator | None = None,
) -> dict[OptionName, tuple[float, float, float]]:
    """Price all supported options with antithetic variates.

    Returns a mapping of option name to ``(price, standard_error, payoff_corr)``.
    """

    _validate_market_inputs(s0, strike, maturity, rate, volatility)
    _validate_count("n_pairs", n_pairs)
    if n_pairs < 2:
        raise ValueError("n_pairs must be at least 2 to estimate standard error and correlation")

    generator = _validate_rng(rng)
    shocks = generator.standard_normal(n_pairs)
    drift = (rate - 0.5 * volatility**2) * maturity
    diffusion = volatility * sqrt(maturity) * shocks
    st_plus = s0 * np.exp(drift + diffusion)
    st_minus = s0 * np.exp(drift - diffusion)
    discount = exp(-rate * maturity)

    results: dict[OptionName, tuple[float, float, float]] = {}
    for option_name, payoff in PAYOFF_REGISTRY.items():
        x_plus = discount * payoff(st_plus, strike)
        x_minus = discount * payoff(st_minus, strike)
        paired = 0.5 * (x_plus + x_minus)
        price = float(paired.mean())
        standard_error = float(paired.std(ddof=1) / sqrt(n_pairs))
        corr = float(np.corrcoef(x_plus, x_minus)[0, 1])
        results[option_name] = (price, standard_error, corr)

    return results

