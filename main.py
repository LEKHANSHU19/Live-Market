"""main.py — Quantitative Risk & Derivatives Analytics Platform.

Single end-to-end orchestration script that runs the complete platform
against the seven frozen modules in ``quant_platform``:

    config.py -> data.py -> {pricing, risk, portfolio} -> scenarios.py
        -> reporting.py -> main.py

Sections
--------
    0. Bootstrap   — logging, output directory, RNG, package path setup
    1. Data        — price loading, log returns, console summary
    2. Portfolio   — MVO scalars, optimal weights, formatted output
    3. Risk        — rolling/EWMA volatility and VaR, breach analysis,
                      Basel traffic-light backtests, VaR/ES sensitivities
    4. Pricing     — Black-Scholes benchmark, parity checks, Monte Carlo
                      convergence study, antithetic-variates study
    5. Scenarios   — volatility shocks, rate shocks, portfolio stress,
                      VaR stress
    6. Reporting   — all figures, summary report, saved outputs

Integration fixes applied (see project integration audit)
-----------------------------------------------------------
    I1  Package bootstrap: ``sys.path`` is extended so that
        ``quant_platform`` resolves as a package without installation.
    C1  ``var_stress_test`` returns ``difference``/``pct_change`` as
        ``dict[str, pd.Series]``.  Before plotting, each series is reduced
        to its peak absolute value so ``plot_scenario_analysis`` receives
        a scalar-only mapping.
    C2  ``traffic_light`` returns a frozen ``TrafficLightResult``
        dataclass.  ``dataclasses.asdict`` converts it to a plain dict
        before it is handed to ``format_var_results``.
    C4  ``pricing.py`` has no convergence-study helper.  The convergence
        loop is implemented here and produces a DataFrame with the exact
        column names (``"Scheme"``, ``"Steps"``, ``"Abs error"``) that
        ``reporting.plot_convergence`` expects by default.
    J4  ``pricing.py`` uses ``strike``/``maturity``/``volatility`` while
        ``config.MARKET`` uses ``K``/``T``/``SIGMA``.  A single adapter
        dictionary, ``_BSP``, defines this mapping once and is reused at
        every pricing call site via ``**_BSP``.

No existing module is modified.  Every fix above is implemented entirely
within this file.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── I1: Package bootstrap ──────────────────────────────────────────────────
# Ensure the project root (the directory containing the ``quant_platform``
# package) is importable without requiring an editable install.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from quant_platform.config import BACKTEST, MARKET, MC, PLOT, VAR
from quant_platform.data import (
    compute_log_returns,
    load_price_data,
    summarise_price_data,
)
from quant_platform.portfolio import compute_mvo_scalars, compute_mvo_weights
from quant_platform.pricing import (
    antithetic_price_all,
    black_scholes_values,
    discounted_payoffs,
    price_from_payoffs,
    simulate_terminal_prices,
)
from quant_platform.reporting import (
    format_pricing_results,
    format_portfolio_results,
    format_scenario_results,
    format_var_results,
    generate_summary_report,
    plot_convergence,
    plot_scenario_analysis,
    plot_sensitivity,
    plot_var_backtest,
)
from quant_platform.risk import (
    TrafficLightResult,
    compute_10d_var,
    compute_ewma_variance,
    compute_rolling_vol,
    compute_var_es_sensitivities,
    get_breaches,
    traffic_light,
)
from quant_platform.scenarios import (
    interest_rate_shock_analysis,
    portfolio_stress_test,
    var_stress_test,
    volatility_shock_analysis,
)


# ── J4: Pricing parameter adapter ────────────────────────────────────────────
# ``pricing.py`` functions use ``strike``/``maturity``/``volatility`` while
# ``config.MARKET`` uses ``K``/``T``/``SIGMA``.  This dictionary is the single
# source of truth for that mapping; every pricing call site below unpacks it
# with ``**_BSP`` so a parameter can never be silently mismapped.
_BSP: dict[str, float] = {
    "s0": MARKET.S0,
    "strike": MARKET.K,
    "maturity": MARKET.T,
    "rate": MARKET.R,
    "volatility": MARKET.SIGMA,
}


def _setup_logging(output_dir: Path) -> logging.Logger:
    """Configure and return the platform logger.

    Two handlers are attached: a console ``StreamHandler`` at ``WARNING``
    level (so routine progress does not clutter the terminal) and a
    ``FileHandler`` at ``DEBUG`` level writing to ``run.log`` inside
    *output_dir* (so the full run is reproducible from the log alone).

    Parameters
    ----------
    output_dir:
        Directory in which ``run.log`` will be created.  Must already exist.

    Returns
    -------
    logging.Logger
        A logger named ``"quant_platform.main"`` with both handlers attached.
    """

    logger = logging.getLogger("quant_platform.main")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(output_dir / "run.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def _create_output_directory() -> Path:
    """Create and return a timestamped output directory under ``outputs/``.

    Returns
    -------
    Path
        Path to the newly created directory, e.g.
        ``outputs/20260613_142233``.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _PROJECT_ROOT / "outputs" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_figure(fig: Any, output_dir: Path, filename: str, logger: logging.Logger) -> None:
    """Save a matplotlib figure to *output_dir* and log its size.

    Parameters
    ----------
    fig:
        A ``matplotlib.figure.Figure`` instance.
    output_dir:
        Destination directory.
    filename:
        Output filename (e.g. from ``config.PLOT.OUT_TASK3``).
    logger:
        Logger used to record the saved path and file size.
    """

    path = output_dir / filename
    fig.savefig(path, dpi=PLOT.DPI, bbox_inches="tight")
    size_kb = path.stat().st_size / 1024.0
    logger.info("Saved figure: %s (%.1f KB)", path, size_kb)


def run_section_1_data(logger: logging.Logger) -> dict[str, Any]:
    """Run Section 1 — Data ingestion and preparation.

    Loads the configured price series, computes daily log returns, prints
    the standard loading summary, and derives the two series required by
    the risk section: the out-of-sample backtest window slice and the
    horizon-shifted forward returns used for breach testing.

    Parameters
    ----------
    logger:
        Platform logger.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys ``"prices"``, ``"returns"``, ``"prices_bt"``,
        ``"returns_bt"``, and ``"fwd_returns"``.

    Raises
    ------
    FileNotFoundError
        If the configured data file does not exist.
    ValueError
        If the workbook is malformed, the backtest window produces no
        observations, or the forward-return series is empty.
    """

    logger.info("SECTION 1 — Data: loading %r (ticker=%r)", MARKET.DATA_FILEPATH, MARKET.SP500_TICKER)

    prices = load_price_data(MARKET.DATA_FILEPATH, MARKET.SP500_TICKER)
    returns = compute_log_returns(prices)
    summarise_price_data(prices, returns)

    returns_bt = returns[BACKTEST.START : BACKTEST.END]
    prices_bt = prices[BACKTEST.START : BACKTEST.END]
    if returns_bt.empty:
        raise ValueError(
            f"Backtest window {BACKTEST.START!r}-{BACKTEST.END!r} produced no "
            f"return observations. Data spans {returns.index[0].date()} to "
            f"{returns.index[-1].date()}."
        )

    fwd_returns = returns.shift(-VAR.HORIZON_DAYS).dropna()
    if fwd_returns.empty:
        raise ValueError(
            f"Forward-return series is empty after shifting by "
            f"VAR.HORIZON_DAYS={VAR.HORIZON_DAYS}."
        )

    logger.info(
        "Backtest window: %s observations from %s to %s",
        len(returns_bt),
        returns_bt.index[0].date(),
        returns_bt.index[-1].date(),
    )

    return {
        "prices": prices,
        "returns": returns,
        "prices_bt": prices_bt,
        "returns_bt": returns_bt,
        "fwd_returns": fwd_returns,
    }


def run_section_2_portfolio(logger: logging.Logger) -> dict[str, Any]:
    """Run Section 2 — Mean-variance portfolio optimisation.

    Computes the E1 mean-variance scalars (A, B, C, D) and the analytical
    optimal weights at ``config.MARKET.MVO_TARGET_RETURN``, then formats
    both into display-ready tables.

    Parameters
    ----------
    logger:
        Platform logger.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys ``"scalars"``, ``"weights"``,
        ``"scalars_table"``, and ``"weights_table"``.

    Raises
    ------
    ValueError
        If the covariance matrix is not invertible, the ``D`` scalar is
        too close to zero, or the resulting portfolio variance is negative.
    """

    logger.info("SECTION 2 — Portfolio: computing MVO scalars and weights")

    universe = UniverseBuilder().build_universe("NIFTY50")

dataset = build_market_dataset(universe)

momentum = compute_momentum_signal(
    dataset.returns
)

mean_rev = compute_mean_reversion_signal(
    dataset.prices,
    dataset.returns,
)

vol_scalar = compute_volatility_scalar(
    dataset.returns
)

composite_alpha = blend_signals(
    momentum,
    mean_rev,
    vol_scalar,
)

mu = generate_expected_returns(
    composite_alpha
)

sigma = dataset.covariance.values

weights = compute_mvo_weights(
    mu=mu,
    covariance=sigma,
    target_return=MARKET.MVO_TARGET_RETURN,
)

    weights_array = np.asarray(weights["weights"], dtype=float)
    asset_names = MARKET.MVO_ASSET_NAMES
    weights_table = pd.DataFrame(
        {
            "Asset": list(asset_names),
            "Weight": weights_array,
        }
    )
    weights_table = format_portfolio_results(weights_table)

    summary_table = format_portfolio_results(
        {
            "expected_return": weights["expected_return"],
            "variance": weights["variance"],
            "volatility": weights["volatility"],
            "lambda": weights["lambda"],
            "gamma": weights["gamma"],
        }
    )

    logger.info(
        "MVO weights: %s (expected return=%.6f, volatility=%.6f)",
        np.round(weights_array, 6).tolist(),
        weights["expected_return"],
        weights["volatility"],
    )

    print("\n" + "=" * 60)
    print("SECTION 2 — PORTFOLIO OPTIMISATION (MVO)")
    print("=" * 60)
    print("\nFrontier scalars (A, B, C, D):")
    print(scalars_table.to_string(index=False))
    print(f"\nOptimal weights (target return = {MARKET.MVO_TARGET_RETURN:.4%}):")
    print(weights_table.to_string(index=False))
    print("\nPortfolio summary:")
    print(summary_table.to_string(index=False))

    return {
        "scalars": scalars,
        "weights": weights,
        "scalars_table": scalars_table,
        "weights_table": weights_table,
        "summary_table": summary_table,
    }


def run_section_3_risk(
    returns: pd.Series,
    fwd_returns: pd.Series,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Run Section 3 — Risk analytics.

    Computes rolling and EWMA volatility, the corresponding 10-day
    parametric VaR series, breach comparisons against forward returns,
    Basel traffic-light backtests for both volatility models, and the
    analytical VaR/ES sensitivity vectors for the E1 three-asset problem.

    Parameters
    ----------
    returns:
        Full daily log-return series (output of
        :func:`quant_platform.data.compute_log_returns`).
    fwd_returns:
        Horizon-shifted forward returns (output of
        :func:`run_section_1_data`).
    logger:
        Platform logger.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the rolling and EWMA volatility and VaR
        series, breach comparison tables, traffic-light results, formatted
        display tables, and the VaR/ES sensitivity dictionary.

    Raises
    ------
    ValueError
        If either volatility series is shorter than its required window,
        or if no overlapping non-NaN observations exist for breach testing.
    """

    logger.info("SECTION 3 — Risk: computing rolling and EWMA VaR backtests")

    # ── Rolling 21-day VaR ───────────────────────────────────────────────
    roll_vol = compute_rolling_vol(returns)
    roll_var = compute_10d_var(roll_vol)
    comp_roll = get_breaches(roll_var, fwd_returns, start=BACKTEST.START, end=BACKTEST.END)
    tl_roll = traffic_light(comp_roll)

    logger.info(
        "Rolling 21D VaR backtest: %d/%d breaches (%.2f%%) -> zone=%s",
        tl_roll.breaches,
        tl_roll.observations,
        tl_roll.breach_rate,
        tl_roll.zone,
    )

    # ── EWMA VaR ─────────────────────────────────────────────────────────
    ewma_variance = compute_ewma_variance(returns)
    ewma_vol = np.sqrt(ewma_variance)
    ewma_vol.name = "ewma_vol"
    ewma_var = compute_10d_var(ewma_vol)
    comp_ewma = get_breaches(ewma_var, fwd_returns, start=BACKTEST.START, end=BACKTEST.END)
    tl_ewma = traffic_light(comp_ewma)

    logger.info(
        "EWMA VaR backtest: %d/%d breaches (%.2f%%) -> zone=%s",
        tl_ewma.breaches,
        tl_ewma.observations,
        tl_ewma.breach_rate,
        tl_ewma.zone,
    )

    # ── C2: convert TrafficLightResult dataclasses to plain dicts ──────────
    tl_roll_dict = dataclasses.asdict(tl_roll)
    tl_ewma_dict = dataclasses.asdict(tl_ewma)

    tl_roll_table = format_var_results(tl_roll_dict)
    tl_ewma_table = format_var_results(tl_ewma_dict)

    # ── Side-by-side comparison table ───────────────────────────────────
    comparison_table = pd.DataFrame(
        {
            "Metric": [
                "Observations",
                "Breaches",
                "Breach rate (%)",
                "Green threshold",
                "Yellow threshold",
                "Zone",
                "Mean |VaR|",
                "Mean daily vol",
            ],
            "Rolling 21D": [
                tl_roll.observations,
                tl_roll.breaches,
                round(tl_roll.breach_rate, 4),
                tl_roll.green_threshold,
                tl_roll.yellow_threshold,
                tl_roll.zone,
                round(float(comp_roll["VaR"].abs().mean()), 6),
                round(float(roll_vol.dropna().mean()), 6),
            ],
            "EWMA (lambda=0.72)": [
                tl_ewma.observations,
                tl_ewma.breaches,
                round(tl_ewma.breach_rate, 4),
                tl_ewma.green_threshold,
                tl_ewma.yellow_threshold,
                tl_ewma.zone,
                round(float(comp_ewma["VaR"].abs().mean()), 6),
                round(float(ewma_vol.mean()), 6),
            ],
        }
    )

    # ── VaR / ES sensitivities (E1 three-asset problem) ─────────────────
    ves = compute_var_es_sensitivities(
        weights=MARKET.VES_WEIGHTS,
        sigmas=MARKET.VES_SIGMA,
        correlation=MARKET.VES_RHO,
        confidence=VAR.CONFIDENCE,
        means=MARKET.VES_MU,
    )

    ves_table = pd.DataFrame(
        {
            "Asset": list(MARKET.VES_ASSET_NAMES),
            "Weight": list(MARKET.VES_WEIGHTS),
            "sigma": list(MARKET.VES_SIGMA),
            "dVaR/dw": np.round(ves["dVaR"], 6),
            "dES/dw": np.round(ves["dES"], 6),
        }
    )

    logger.info(
        "VaR/ES sensitivities computed: portfolio_std=%.6f, z_alpha=%.6f",
        ves["portfolio_std"],
        ves["z_alpha"],
    )
    logger.debug("dVaR = %s", np.round(ves["dVaR"], 6).tolist())
    logger.debug("dES  = %s", np.round(ves["dES"], 6).tolist())

    print("\n" + "=" * 60)
    print("SECTION 3 — RISK ANALYTICS")
    print("=" * 60)
    print("\nRolling 21D VaR backtest:")
    print(tl_roll_table.to_string(index=False))
    print("\nEWMA VaR backtest:")
    print(tl_ewma_table.to_string(index=False))
    print("\nRolling vs EWMA comparison:")
    print(comparison_table.to_string(index=False))
    print("\nVaR/ES sensitivities (three-asset problem):")
    print(ves_table.to_string(index=False))

    return {
        "roll_vol": roll_vol,
        "roll_var": roll_var,
        "comp_roll": comp_roll,
        "tl_roll": tl_roll,
        "tl_roll_table": tl_roll_table,
        "ewma_vol": ewma_vol,
        "ewma_var": ewma_var,
        "comp_ewma": comp_ewma,
        "tl_ewma": tl_ewma,
        "tl_ewma_table": tl_ewma_table,
        "comparison_table": comparison_table,
        "ves": ves,
        "ves_table": ves_table,
    }


def run_section_4_pricing(rng: np.random.Generator, logger: logging.Logger) -> dict[str, Any]:
    """Run Section 4 — Option pricing and Monte Carlo analysis.

    Computes Black-Scholes benchmark prices for all four supported option
    payoffs, verifies put-call and binary parity, runs a Monte Carlo
    convergence study across every (scheme, step-count) combination in
    :class:`quant_platform.config.MC`, and runs an antithetic-variates
    study comparing crude Monte Carlo to antithetic sampling.

    Parameters
    ----------
    rng:
        Seeded random number generator shared across all simulations for
        reproducibility.
    logger:
        Platform logger.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys ``"bs_prices"``, ``"bs_table"``,
        ``"convergence_results"``, ``"av_result"``, and ``"av_table"``.

    Notes
    -----
    The convergence-study DataFrame uses the exact column names
    ``"Scheme"``, ``"Steps"``, ``"Abs error"`` required by
    :func:`quant_platform.reporting.plot_convergence` (integration fix C4).
    All pricing-function calls use the ``_BSP`` adapter dictionary to map
    ``config.MARKET`` field names onto ``pricing.py`` parameter names
    (integration fix J4).
    """

    logger.info("SECTION 4 — Pricing: Black-Scholes benchmark and Monte Carlo studies")

    # ── Black-Scholes benchmark ──────────────────────────────────────────
    bs_prices = black_scholes_values(**_BSP)
    bs_table = format_pricing_results(bs_prices)

    discount = float(np.exp(-MARKET.R * MARKET.T))
    put_call_parity_residual = (
        bs_prices["European Call"]
        - bs_prices["European Put"]
        - (MARKET.S0 - MARKET.K * discount)
    )
    binary_parity_residual = bs_prices["Binary Call"] + bs_prices["Binary Put"] - discount

    assert abs(put_call_parity_residual) < 1e-10, (
        f"Put-call parity violated: residual={put_call_parity_residual:.2e}"
    )
    assert abs(binary_parity_residual) < 1e-10, (
        f"Binary parity violated: residual={binary_parity_residual:.2e}"
    )

    logger.info(
        "Black-Scholes prices: %s (put-call parity residual=%.2e, binary parity residual=%.2e)",
        {name: round(price, 6) for name, price in bs_prices.items()},
        put_call_parity_residual,
        binary_parity_residual,
    )

    # ── C4: Monte Carlo convergence study ────────────────────────────────
    # Builds a DataFrame with columns ["Scheme", "Steps", "Option", "Price",
    # "SE", "CI_lo", "CI_hi", "BS_price", "Abs error"].  The "Scheme",
    # "Steps", and "Abs error" columns match the defaults expected by
    # reporting.plot_convergence(x="Steps", y="Abs error", group="Scheme").
    convergence_rows: list[dict[str, Any]] = []
    total_paths_simulated = 0

    for scheme in MC.SCHEMES:
        for n_steps in MC.STEPS_LIST:
            terminal_prices = simulate_terminal_prices(
                s0=_BSP["s0"],
                maturity=_BSP["maturity"],
                rate=_BSP["rate"],
                volatility=_BSP["volatility"],
                n_paths=MC.N_PATHS,
                n_steps=n_steps,
                scheme=scheme,
                rng=rng,
            )
            total_paths_simulated += MC.N_PATHS

            n_negative = int((terminal_prices < 0.0).sum())
            if n_negative > 0:
                logger.warning(
                    "%s scheme at %d steps produced %d/%d negative terminal "
                    "prices (%.4f%%); affected paths price to zero payoff "
                    "for vanilla options.",
                    scheme,
                    n_steps,
                    n_negative,
                    MC.N_PATHS,
                    100.0 * n_negative / MC.N_PATHS,
                )

            payoffs = discounted_payoffs(
                terminal_prices,
                strike=_BSP["strike"],
                rate=_BSP["rate"],
                maturity=_BSP["maturity"],
            )

            for option_name in MC.OPTION_NAMES:
                price, standard_error, ci_lo, ci_hi = price_from_payoffs(payoffs[option_name])
                bs_price = bs_prices[option_name]
                convergence_rows.append(
                    {
                        "Scheme": scheme,
                        "Steps": n_steps,
                        "Option": option_name,
                        "Price": price,
                        "SE": standard_error,
                        "CI_lo": ci_lo,
                        "CI_hi": ci_hi,
                        "BS_price": bs_price,
                        "Abs error": abs(price - bs_price),
                    }
                )
                logger.debug(
                    "Convergence: scheme=%s steps=%d option=%s price=%.6f SE=%.6f abs_error=%.6f",
                    scheme,
                    n_steps,
                    option_name,
                    price,
                    standard_error,
                    abs(price - bs_price),
                )

    convergence_results = pd.DataFrame(convergence_rows)
    logger.info(
        "Convergence study complete: %d (scheme, steps) combinations, %d total paths simulated",
        len(MC.SCHEMES) * len(MC.STEPS_LIST),
        total_paths_simulated,
    )

    # ── Antithetic variates study ────────────────────────────────────────
    av_rng = np.random.default_rng(MC.SEED)
    av_result = antithetic_price_all(
        s0=_BSP["s0"],
        strike=_BSP["strike"],
        maturity=_BSP["maturity"],
        rate=_BSP["rate"],
        volatility=_BSP["volatility"],
        n_pairs=MC.N_AV_PAIRS,
        rng=av_rng,
    )

    # Crude Monte Carlo comparison at the same total path budget
    # (N_AV_PAIRS pairs * 2 = N_PATHS crude draws), single-step Exact GBM
    # to match the antithetic engine's discretisation exactly.
    crude_rng = np.random.default_rng(MC.SEED)
    crude_terminal = simulate_terminal_prices(
        s0=_BSP["s0"],
        maturity=_BSP["maturity"],
        rate=_BSP["rate"],
        volatility=_BSP["volatility"],
        n_paths=MC.N_AV_PAIRS * 2,
        n_steps=1,
        scheme="Exact",
        rng=crude_rng,
    )
    crude_payoffs = discounted_payoffs(
        crude_terminal,
        strike=_BSP["strike"],
        rate=_BSP["rate"],
        maturity=_BSP["maturity"],
    )

    av_rows: list[dict[str, Any]] = []
    for option_name, (av_price, av_se, av_corr) in av_result.items():
        crude_price, crude_se, _, _ = price_from_payoffs(crude_payoffs[option_name])
        se_reduction_pct = (
            100.0 * (1.0 - av_se / crude_se) if crude_se > 0.0 else float("nan")
        )
        variance_reduction_ratio = (
            (crude_se / av_se) ** 2 if av_se > 0.0 else float("inf")
        )
        av_rows.append(
            {
                "Option": option_name,
                "Crude Price": crude_price,
                "Crude SE": crude_se,
                "AV Price": av_price,
                "AV SE": av_se,
                "SE Reduction (%)": se_reduction_pct,
                "Payoff Corr": av_corr,
                "Variance Reduction Ratio": variance_reduction_ratio,
            }
        )

    av_table_raw = pd.DataFrame(av_rows).sort_values(
        "SE Reduction (%)", ascending=False
    ).reset_index(drop=True)
    av_table = format_pricing_results(av_table_raw)

    logger.info(
        "Antithetic variates study complete: %d pairs per option, mean SE reduction=%.2f%%",
        MC.N_AV_PAIRS,
        float(av_table_raw["SE Reduction (%)"].mean()),
    )

    print("\n" + "=" * 60)
    print("SECTION 4 — OPTION PRICING")
    print("=" * 60)
    print("\nBlack-Scholes benchmark prices:")
    print(bs_table.to_string(index=False))
    print(f"\nPut-call parity residual : {put_call_parity_residual:.2e}  (OK)")
    print(f"Binary parity residual   : {binary_parity_residual:.2e}  (OK)")
    print("\nMonte Carlo convergence results (head):")
    print(convergence_results.head(12).to_string(index=False))
    print("\nAntithetic variates vs crude Monte Carlo:")
    print(av_table.to_string(index=False))

    return {
        "bs_prices": bs_prices,
        "bs_table": bs_table,
        "put_call_parity_residual": put_call_parity_residual,
        "binary_parity_residual": binary_parity_residual,
        "convergence_results": convergence_results,
        "av_result": av_result,
        "av_table": av_table,
    }


def run_section_5_scenarios(returns: pd.Series, logger: logging.Logger) -> dict[str, Any]:
    """Run Section 5 — Scenario and stress analysis.

    Runs four scenario analyses, all of which call existing pricing and
    risk engines with shocked inputs: a volatility shock sweep, an
    interest-rate shock sweep, a joint portfolio stress test, and a VaR
    stress test under volatility multipliers.

    Parameters
    ----------
    returns:
        Full daily log-return series, used as the basis for
        :func:`quant_platform.scenarios.var_stress_test`.
    logger:
        Platform logger.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys ``"vol_shock"``, ``"rate_shock"``,
        ``"port_stress"``, ``"var_stress"``, ``"var_stress_peak_table"``,
        and the formatted display tables for each scenario.

    Notes
    -----
    ``var_stress_test`` returns ``difference`` and ``pct_change`` as
    ``dict[str, pd.Series]``.  Before these are passed to
    :func:`quant_platform.reporting.plot_scenario_analysis`, each series is
    reduced to its peak absolute value, producing a scalar-only mapping
    (integration fix C1).
    """

    logger.info("SECTION 5 — Scenarios: volatility, rate, portfolio, and VaR stress tests")

    # ── Volatility shocks ────────────────────────────────────────────────
    vol_shock = volatility_shock_analysis(shock_values=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
    vol_shock_tables = format_scenario_results(vol_shock)

    # ── Interest rate shocks ─────────────────────────────────────────────
    rate_shock = interest_rate_shock_analysis(shock_values=[0.00, 0.01, 0.03, 0.05, 0.07, 0.10])
    rate_shock_tables = format_scenario_results(rate_shock)

    # ── Portfolio stress test ────────────────────────────────────────────
    scenario_returns = [-0.20, -0.25, -0.35, -0.15]
    if len(scenario_returns) != len(MARKET.MVO_MU):
        raise ValueError(
            f"scenario_returns length ({len(scenario_returns)}) must match "
            f"the number of portfolio assets ({len(MARKET.MVO_MU)})"
        )
    port_stress = portfolio_stress_test(scenario_returns=scenario_returns)

    port_stress_table = pd.DataFrame(
        {
            "Asset": list(MARKET.MVO_ASSET_NAMES),
            "Scenario Return": scenario_returns,
        }
    )
    port_stress_table = format_pricing_results(port_stress_table)
    portfolio_pnl = float(port_stress["stressed"]["portfolio_pnl"])
    base_expected_return = float(port_stress["base"]["expected_return"])

    # ── VaR stress test ───────────────────────────────────────────────────
    var_stress_result = var_stress_test(
        returns=returns,
        shock_multipliers=[1.0, 1.5, 2.0, 3.0],
        method="rolling",
    )

    # C1: reduce each Series in "difference" to its peak absolute value so
    # plot_scenario_analysis receives a scalar-only mapping.
    var_stress_peak_diff = {
        name: float(series.abs().max())
        for name, series in var_stress_result["difference"].items()
    }
    base_peak_var = float(var_stress_result["base"]["var"].abs().max())
    var_stress_peak_table = pd.DataFrame(
        {
            "Scenario": list(var_stress_peak_diff.keys()),
            "Peak |Delta VaR|": list(var_stress_peak_diff.values()),
            "Base Peak |VaR|": base_peak_var,
        }
    )
    var_stress_peak_table["Pct Change (%)"] = (
        100.0 * var_stress_peak_table["Peak |Delta VaR|"] / base_peak_var
    )
    var_stress_peak_table = format_pricing_results(var_stress_peak_table)

    logger.info(
        "Scenario analysis complete: %d volatility shocks, %d rate shocks, "
        "portfolio stress P&L=%.6f (vs expected return=%.6f)",
        len(vol_shock["stressed"]),
        len(rate_shock["stressed"]),
        portfolio_pnl,
        base_expected_return,
    )
    logger.warning(
        "var_stress_test 'difference' contains pd.Series; reduced to peak "
        "absolute values before plotting (integration fix C1)."
    )

    print("\n" + "=" * 60)
    print("SECTION 5 — SCENARIO & STRESS ANALYSIS")
    print("=" * 60)
    print("\nVolatility shock — stressed option prices:")
    print(vol_shock_tables["stressed"].to_string(index=False))
    print("\nInterest rate shock — stressed option prices:")
    print(rate_shock_tables["stressed"].to_string(index=False))
    print("\nPortfolio stress test:")
    print(port_stress_table.to_string(index=False))
    print(f"  Portfolio P&L           : {portfolio_pnl:.4%}")
    print(f"  Base expected return    : {base_expected_return:.4%}")
    print("\nVaR stress test — peak |VaR| change by multiplier:")
    print(var_stress_peak_table.to_string(index=False))

    return {
        "vol_shock": vol_shock,
        "vol_shock_tables": vol_shock_tables,
        "rate_shock": rate_shock,
        "rate_shock_tables": rate_shock_tables,
        "port_stress": port_stress,
        "port_stress_table": port_stress_table,
        "portfolio_pnl": portfolio_pnl,
        "base_expected_return": base_expected_return,
        "var_stress": var_stress_result,
        "var_stress_peak_diff": var_stress_peak_diff,
        "var_stress_peak_table": var_stress_peak_table,
    }


def run_section_6_reporting(
    output_dir: Path,
    data_results: dict[str, Any],
    portfolio_results: dict[str, Any],
    risk_results: dict[str, Any],
    pricing_results: dict[str, Any],
    scenario_results: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    """Run Section 6 — Reporting: figures, summary, and saved outputs.

    Generates every figure defined in the platform's reporting layer,
    saves each to *output_dir* as a PNG at ``config.PLOT.DPI``, builds a
    parameter-sensitivity sweep over volatility for the sensitivity chart,
    and assembles a structured summary report from all prior sections.

    Parameters
    ----------
    output_dir:
        Directory in which all figures and the summary are written.
    data_results, portfolio_results, risk_results, pricing_results, scenario_results:
        Output dictionaries from :func:`run_section_1_data` through
        :func:`run_section_5_scenarios`.
    logger:
        Platform logger.

    Returns
    -------
    dict[str, Any]
        Dictionary with key ``"summary"`` containing the structured
        summary report produced by
        :func:`quant_platform.reporting.generate_summary_report`.

    Notes
    -----
    The EWMA backtest figure's VaR line colour is corrected to
    ``config.PLOT.COLOR_EWMA_VAR`` after creation, since
    :func:`quant_platform.reporting.plot_var_backtest` always plots the VaR
    series in ``config.PLOT.COLOR_ROLLING_VAR`` (integration fix M1).
    """

    logger.info("SECTION 6 — Reporting: generating and saving figures")

    # ── Rolling VaR backtest figure ──────────────────────────────────────
    fig_roll = plot_var_backtest(
        risk_results["comp_roll"],
        title="Rolling 21D VaR Backtest",
    )
    _save_figure(fig_roll, output_dir, PLOT.OUT_TASK3, logger)

    # ── EWMA VaR backtest figure ─────────────────────────────────────────
    fig_ewma = plot_var_backtest(
        risk_results["comp_ewma"],
        title="EWMA VaR Backtest (lambda=0.72)",
    )
    # M1: plot_var_backtest always renders the VaR line in
    # PLOT.COLOR_ROLLING_VAR (navy). Recolour it to PLOT.COLOR_EWMA_VAR
    # (purple) so the two backtest charts are visually distinguishable.
    for line in fig_ewma.axes[0].lines:
        if line.get_label() == "VaR":
            line.set_color(PLOT.COLOR_EWMA_VAR)
    fig_ewma.axes[0].legend()
    _save_figure(fig_ewma, output_dir, PLOT.OUT_TASK4, logger)

    # ── Monte Carlo convergence figure ───────────────────────────────────
    fig_convergence = plot_convergence(
        pricing_results["convergence_results"],
        x="Steps",
        y="Abs error",
        group="Scheme",
        title="Monte Carlo Convergence: |Price - BS Price| vs Steps",
    )
    fig_convergence.axes[0].set_xscale("log")
    fig_convergence.axes[0].set_yscale("log")
    _save_figure(fig_convergence, output_dir, PLOT.OUT_CONVERGENCE, logger)

    # ── Sensitivity sweep (volatility) ───────────────────────────────────
    sigma_grid = np.linspace(*MC.SENS_SIGMA_RANGE, MC.SENSITIVITY_N_POINTS)
    sensitivity_rows: list[dict[str, Any]] = []
    for sigma_value in sigma_grid:
        row = black_scholes_values(
            s0=_BSP["s0"],
            strike=_BSP["strike"],
            maturity=_BSP["maturity"],
            rate=_BSP["rate"],
            volatility=float(sigma_value),
        )
        row["sigma"] = float(sigma_value)
        sensitivity_rows.append(row)
    sensitivity_df = pd.DataFrame(sensitivity_rows)

    fig_sensitivity = plot_sensitivity(
        sensitivity_df,
        x="sigma",
        value_columns=list(MC.OPTION_NAMES),
        title="Option Price Sensitivity to Volatility",
    )
    _save_figure(fig_sensitivity, output_dir, PLOT.OUT_SENSITIVITY, logger)

    # ── Scenario figures: volatility shock ───────────────────────────────
    fig_vol_shock = plot_scenario_analysis(
        scenario_results["vol_shock"],
        section="difference",
        title="Option Price Change Under Volatility Shock",
    )
    _save_figure(fig_vol_shock, output_dir, "scenario_volatility_shock.png", logger)

    # ── Scenario figures: interest rate shock ────────────────────────────
    fig_rate_shock = plot_scenario_analysis(
        scenario_results["rate_shock"],
        section="difference",
        title="Option Price Change Under Interest Rate Shock",
    )
    _save_figure(fig_rate_shock, output_dir, "scenario_rate_shock.png", logger)

    # ── Scenario figures: VaR stress (C1 peak-aggregated) ────────────────
    fig_var_stress = plot_scenario_analysis(
        {"difference": scenario_results["var_stress_peak_diff"]},
        section="difference",
        title="Peak |Delta VaR| Under Volatility Multiplier Stress",
    )
    _save_figure(fig_var_stress, output_dir, "scenario_var_stress.png", logger)

    # ── Antithetic variates: SE comparison bar chart ─────────────────────
    av_table_raw = pricing_results["av_table"]
    fig_av = plot_scenario_analysis(
        {
            "difference": {
                row["Option"]: row["SE Reduction (%)"]
                for _, row in av_table_raw.iterrows()
            }
        },
        section="difference",
        title="Antithetic Variates: Standard Error Reduction (%) by Option",
    )
    _save_figure(fig_av, output_dir, PLOT.OUT_AV, logger)

    # ── Summary report ────────────────────────────────────────────────────
    summary = generate_summary_report(
        portfolio={
            "weights": portfolio_results["weights"]["weights"].tolist(),
            "expected_return": portfolio_results["weights"]["expected_return"],
            "volatility": portfolio_results["weights"]["volatility"],
            "scalars": {
                "A": portfolio_results["scalars"]["A"],
                "B": portfolio_results["scalars"]["B"],
                "C": portfolio_results["scalars"]["C"],
                "D": portfolio_results["scalars"]["D"],
            },
        },
        risk={
            "rolling": dataclasses.asdict(risk_results["tl_roll"]),
            "ewma": dataclasses.asdict(risk_results["tl_ewma"]),
            "var_es_sensitivities": {
                "dVaR": risk_results["ves"]["dVaR"].tolist(),
                "dES": risk_results["ves"]["dES"].tolist(),
                "portfolio_std": risk_results["ves"]["portfolio_std"],
            },
        },
        pricing={
            "black_scholes": pricing_results["bs_prices"],
            "put_call_parity_residual": pricing_results["put_call_parity_residual"],
            "binary_parity_residual": pricing_results["binary_parity_residual"],
            "convergence_rows": len(pricing_results["convergence_results"]),
            "antithetic_mean_se_reduction_pct": float(
                pricing_results["av_table"]["SE Reduction (%)"].astype(float).mean()
            ),
        },
        scenarios={
            "portfolio_pnl": scenario_results["portfolio_pnl"],
            "base_expected_return": scenario_results["base_expected_return"],
            "var_stress_peak_diff": scenario_results["var_stress_peak_diff"],
        },
    )

    logger.info("Summary report assembled with %d top-level sections", len(summary))

    return {"summary": summary}


def main() -> int:
    """Run the full Quantitative Risk & Derivatives Analytics Platform.

    Executes Sections 0 through 6 in sequence.  Section 0 (Bootstrap) and
    Section 1 (Data) are required for the run to proceed; any failure
    there is logged at ``CRITICAL`` and the process exits with status 1.
    Failures in Sections 2 through 6 are logged at ``ERROR`` with full
    tracebacks written to ``run.log``; the corresponding section's results
    are then unavailable to later sections that depend on them, and the
    process exits with status 2 if any section failed.

    Returns
    -------
    int
        Process exit code: ``0`` on full success, ``1`` if Bootstrap or
        Data failed (abort), ``2`` if one or more later sections failed
        but the run otherwise completed.
    """

    start_time = time.perf_counter()

    # ── Section 0: Bootstrap ──────────────────────────────────────────────
    output_dir = _create_output_directory()
    logger = _setup_logging(output_dir)
    rng = np.random.default_rng(MC.SEED)

    logger.info("=" * 60)
    logger.info("Quantitative Risk & Derivatives Analytics Platform")
    logger.info("=" * 60)
    logger.info("Output directory : %s", output_dir)
    logger.info("Random seed      : %d", MC.SEED)
    logger.info(
        "Config summary   : S0=%.2f K=%.2f T=%.2f sigma=%.2f r=%.2f",
        MARKET.S0,
        MARKET.K,
        MARKET.T,
        MARKET.SIGMA,
        MARKET.R,
    )

    section_failed: dict[str, bool] = {
        "portfolio": False,
        "risk": False,
        "pricing": False,
        "scenarios": False,
        "reporting": False,
    }

    # ── Section 1: Data (abort on failure) ──────────────────────────────
    try:
        data_results = run_section_1_data(logger)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        logger.critical("Section 1 (Data) failed: %s", exc, exc_info=True)
        logger.critical(
            "The platform cannot proceed without data. Check that "
            "MARKET.DATA_FILEPATH (%r) points to a valid Excel file "
            "containing 'Date' and %r columns.",
            MARKET.DATA_FILEPATH,
            MARKET.SP500_TICKER,
        )
        return 1

    # ── Section 2: Portfolio ──────────────────────────────────────────────
    portfolio_results: dict[str, Any] = {}
    try:
        portfolio_results = run_section_2_portfolio(logger)
    except Exception:
        logger.error("Section 2 (Portfolio) failed", exc_info=True)
        section_failed["portfolio"] = True

    # ── Section 3: Risk ───────────────────────────────────────────────────
    risk_results: dict[str, Any] = {}
    try:
        risk_results = run_section_3_risk(
            data_results["returns"], data_results["fwd_returns"], logger
        )
    except Exception:
        logger.error("Section 3 (Risk) failed", exc_info=True)
        section_failed["risk"] = True

    # ── Section 4: Pricing ────────────────────────────────────────────────
    pricing_results: dict[str, Any] = {}
    try:
        pricing_results = run_section_4_pricing(rng, logger)
    except Exception:
        logger.error("Section 4 (Pricing) failed", exc_info=True)
        section_failed["pricing"] = True

    # ── Section 5: Scenarios ──────────────────────────────────────────────
    scenario_results: dict[str, Any] = {}
    try:
        scenario_results = run_section_5_scenarios(data_results["returns"], logger)
    except Exception:
        logger.error("Section 5 (Scenarios) failed", exc_info=True)
        section_failed["scenarios"] = True

    # ── Section 6: Reporting ──────────────────────────────────────────────
    if not any(
        section_failed[key] for key in ("portfolio", "risk", "pricing", "scenarios")
    ):
        try:
            run_section_6_reporting(
                output_dir,
                data_results,
                portfolio_results,
                risk_results,
                pricing_results,
                scenario_results,
                logger,
            )
        except Exception:
            logger.error("Section 6 (Reporting) failed", exc_info=True)
            section_failed["reporting"] = True
    else:
        logger.warning(
            "Skipping Section 6 (Reporting): upstream section(s) failed: %s",
            [name for name, failed in section_failed.items() if failed],
        )
        section_failed["reporting"] = True

    elapsed = time.perf_counter() - start_time
    failed_sections = [name for name, failed in section_failed.items() if failed]

    logger.info("=" * 60)
    logger.info("Run complete in %.2f seconds", elapsed)
    if failed_sections:
        logger.warning("Sections with failures: %s", failed_sections)
        logger.info("Outputs written to: %s", output_dir)
        return 2

    logger.info("All sections completed successfully")
    logger.info("Outputs written to: %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())