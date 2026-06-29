"""alpha.py — Alpha signal generation for the Quantitative Risk & Derivatives
Analytics Platform (Version 2.0).

This module sits between ``data.py`` and ``portfolio.py`` in the V2.0
architecture::

    data.py -> alpha.py -> portfolio.py -> benchmark.py -> attribution.py
        -> decision_engine.py -> reporting.py

Its sole responsibility is to transform a multi-asset price history into a
single cross-sectional expected-return vector (``mu``) that is passed
directly to :func:`quant_platform.portfolio.compute_mvo_weights` and
:func:`quant_platform.portfolio.compute_mvo_scalars`, replacing the static
``MARKET.MVO_MU`` default with a signal-driven forecast.

Pipeline overview
------------------
::

    AssetUniverseConfig
        |
        v
    build_asset_price_panel   -> prices_panel  (T x N)
        |
        v
    build_asset_return_panel  -> returns_panel ((T-1) x N)
        |
        +--> compute_momentum_signal       -> momentum       (N,)
        +--> compute_mean_reversion_signal -> mean_reversion (N,)
        +--> compute_volatility_scalar     -> vol_scalar     (N,)
        |
        v
    blend_signals (via zscore_cross_sectional) -> composite_alpha (N,)
        |
        +--> generate_expected_returns -> mu  (N,)  -> portfolio.compute_mvo_weights(mu=mu)
        +--> rank_assets               -> ranking DataFrame

Design principles
-------------------
- Every cross-sectional quantity (``momentum``, ``mean_reversion``,
  ``vol_scalar``, ``composite_alpha``, ``mu``) is a 1-D structure indexed by
  asset name, matching the shape ``portfolio.compute_mvo_weights`` expects
  for ``mu``.
- Every time-series quantity (``prices_panel``, ``returns_panel``) is a
  :class:`pandas.DataFrame` with a ``DatetimeIndex`` and one column per
  asset.
- All functions are pure: no global state, no file caching, no hidden
  persistence. Historical state required by diagnostics (turnover,
  stability) is the responsibility of the calling orchestration layer
  (``main.py``), not of this module.
- Volatility estimation reuses :func:`quant_platform.risk.compute_ewma_variance`
  exclusively -- there is no second definition of EWMA decay anywhere in
  this module.
- Validation failures raise descriptive exceptions immediately. Degenerate
  numerical results (NaN, zero variance, zero cross-sectional dispersion)
  are never silently suppressed or converted to a default value, except in
  the one documented case (:func:`compute_volatility_scalar`) where a
  zero-volatility asset has a well-defined, safe representation (zero
  weight) in the output's value range.

No existing module (``config.py``, ``data.py``, ``portfolio.py``,
``risk.py``, ``pricing.py``, ``scenarios.py``, ``reporting.py``) is modified
by this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from quant_platform.config import MARKET, VAR
from quant_platform.data import compute_log_returns, load_price_data
from quant_platform.risk import compute_ewma_variance

__all__ = [
    "AssetUniverseConfig",
    "build_asset_price_panel",
    "build_asset_return_panel",
    "compute_momentum_signal",
    "compute_mean_reversion_signal",
    "compute_volatility_scalar",
    "zscore_cross_sectional",
    "blend_signals",
    "generate_expected_returns",
    "rank_assets",
]

logger = logging.getLogger("quant_platform.alpha")


# ── AssetUniverseConfig ────────────────────────────────────────────────────


@dataclass(frozen=True)
class AssetUniverseConfig:
    """Configuration describing the multi-asset universe for alpha generation.

    ``config.py`` is frozen and contains no per-asset data file paths --
    ``MARKET.DATA_FILEPATH`` / ``MARKET.SP500_TICKER`` describe a single
    instrument (the S&P 500 series used for V1.0's VaR backtest), while
    ``MARKET.MVO_ASSET_NAMES`` describes a four-asset portfolio-optimisation
    universe with no corresponding data sources defined anywhere in V1.0.

    This dataclass is the single new "configuration-like" structure
    introduced by ``alpha.py``. It is deliberately scoped as a local
    dataclass rather than an addition to ``config.py``, preserving the
    "zero changes to frozen modules" constraint while giving V2.0 a place
    to specify where each asset's price history is loaded from.

    Parameters
    ----------
    asset_names:
        Ordered tuple of asset identifiers. Defaults to
        ``MARKET.MVO_ASSET_NAMES``. The order of this tuple determines the
        column order of every panel and the element order of every output
        vector (``momentum``, ``mean_reversion``, ``vol_scalar``,
        ``composite_alpha``, ``mu``), so that ``mu`` aligns with
        ``MARKET.MVO_MU``'s implicit asset ordering when passed to
        :func:`quant_platform.portfolio.compute_mvo_weights`.
    filepaths:
        Mapping from asset name to the Excel workbook path containing that
        asset's price history, suitable for
        :func:`quant_platform.data.load_price_data`.
    tickers:
        Mapping from asset name to the column name within that asset's
        workbook (the ``ticker`` argument to
        :func:`quant_platform.data.load_price_data`).

    Notes
    -----
    No default ``filepaths`` / ``tickers`` are provided for assets beyond
    what the caller supplies. :func:`build_asset_price_panel` raises
    ``ValueError`` if any name in ``asset_names`` is missing from either
    mapping -- this module does not fabricate placeholder data sources for
    an incomplete universe.

    Examples
    --------
    >>> universe = AssetUniverseConfig(
    ...     asset_names=("A", "B", "C", "D"),
    ...     filepaths={
    ...         "A": "data/raw/asset_a.xlsx",
    ...         "B": "data/raw/asset_b.xlsx",
    ...         "C": "data/raw/asset_c.xlsx",
    ...         "D": "data/raw/asset_d.xlsx",
    ...     },
    ...     tickers={"A": "ASSET_A", "B": "ASSET_B", "C": "ASSET_C", "D": "ASSET_D"},
    ... )
    """

    asset_names: tuple[str, ...] = MARKET.MVO_ASSET_NAMES
    filepaths: dict[str, str] = field(default_factory=dict)
    tickers: dict[str, str] = field(default_factory=dict)


# ── Layer 1: Data Assembly ──────────────────────────────────────────────────


def build_asset_price_panel(universe: AssetUniverseConfig) -> pd.DataFrame:
    """Load and assemble a multi-asset price panel from an asset universe.

    Calls :func:`quant_platform.data.load_price_data` once per asset in
    ``universe.asset_names``, then aligns the resulting per-asset price
    series onto a common ``DatetimeIndex`` -- the intersection of all
    assets' trading calendars -- so that every column of the returned panel
    has a value on every row.

    Parameters
    ----------
    universe:
        Asset universe configuration. ``universe.asset_names`` determines
        the column order of the returned panel.

    Returns
    -------
    pd.DataFrame
        Price panel with one column per asset (named after
        ``universe.asset_names``, in that order) and a sorted, ascending
        ``DatetimeIndex`` equal to the intersection of all assets' date
        ranges. Values are ``float64`` close prices.

    Raises
    ------
    ValueError
        If ``universe.asset_names`` is empty; if it contains a duplicate
        name; if any asset name is missing from ``universe.filepaths`` or
        ``universe.tickers``; or if the intersection of all assets'
        ``DatetimeIndex`` values is empty (no overlapping trading days).
    FileNotFoundError
        Propagated unchanged from :func:`quant_platform.data.load_price_data`
        if any asset's configured ``filepath`` does not exist.
    TypeError
        Propagated unchanged from :func:`quant_platform.data.load_price_data`
        if any asset's configured ``filepath`` is not a string or path-like
        object.

    Notes
    -----
    If the intersected date range is strictly smaller than any individual
    asset's native date range, a warning is logged (at ``WARNING`` level)
    naming the asset(s) whose history was truncated and by how many
    observations -- this is a data-quality signal worth surfacing, but not
    severe enough to raise, since a non-empty intersection is itself
    sufficient for the downstream pipeline to proceed.

    Examples
    --------
    >>> universe = AssetUniverseConfig(
    ...     asset_names=("A", "B"),
    ...     filepaths={"A": "data/raw/a.xlsx", "B": "data/raw/b.xlsx"},
    ...     tickers={"A": "ASSET_A", "B": "ASSET_B"},
    ... )
    >>> prices_panel = build_asset_price_panel(universe)
    >>> prices_panel.columns.tolist()
    ['A', 'B']
    """

    asset_names = universe.asset_names

    if len(asset_names) == 0:
        raise ValueError(
            "AssetUniverseConfig.asset_names is empty; at least one asset "
            "is required to build a price panel."
        )

    if len(set(asset_names)) != len(asset_names):
        seen: set[str] = set()
        duplicates: list[str] = []
        for name in asset_names:
            if name in seen:
                duplicates.append(name)
            seen.add(name)
        raise ValueError(
            f"AssetUniverseConfig.asset_names contains duplicate name(s): "
            f"{duplicates}. Each asset must appear exactly once."
        )

    missing_filepaths = [name for name in asset_names if name not in universe.filepaths]
    missing_tickers = [name for name in asset_names if name not in universe.tickers]
    if missing_filepaths or missing_tickers:
        raise ValueError(
            "AssetUniverseConfig is incomplete for the requested asset "
            f"universe {asset_names!r}. "
            f"Missing filepaths for: {missing_filepaths!r}. "
            f"Missing tickers for: {missing_tickers!r}. "
            "Every asset name in `asset_names` must have a corresponding "
            "entry in both `filepaths` and `tickers`."
        )

    logger.info(
        "Building asset price panel for %d asset(s): %s",
        len(asset_names),
        list(asset_names),
    )

    per_asset_series: dict[str, pd.Series] = {}
    for name in asset_names:
        filepath = universe.filepaths[name]
        ticker = universe.tickers[name]
        logger.debug("Loading asset %r from %r (ticker=%r)", name, filepath, ticker)
        series = load_price_data(filepath, ticker)
        per_asset_series[name] = series.rename(name)

    prices_panel = pd.concat(per_asset_series.values(), axis=1, join="outer")
    prices_panel = prices_panel[list(asset_names)]

    full_index_lengths = {name: len(series) for name, series in per_asset_series.items()}

    aligned_panel = prices_panel.dropna(how="any")

    if aligned_panel.empty:
        date_ranges = {
            name: (series.index[0].date(), series.index[-1].date())
            for name, series in per_asset_series.items()
        }
        raise ValueError(
            "No overlapping trading days found across the asset universe "
            f"{asset_names!r}. Individual date ranges: {date_ranges!r}. "
            "The intersection of all assets' DatetimeIndex values is empty."
        )

    intersection_length = len(aligned_panel)
    for name, native_length in full_index_lengths.items():
        if native_length > intersection_length:
            dropped = native_length - intersection_length
            logger.warning(
                "Asset %r has %d native observations but only %d overlap "
                "with the full asset universe; %d observation(s) were "
                "truncated when aligning to the common trading calendar.",
                name,
                native_length,
                intersection_length,
                dropped,
            )

    aligned_panel = aligned_panel.sort_index()
    aligned_panel = aligned_panel.astype(np.float64)

    logger.info(
        "Asset price panel assembled: %d assets x %d observations "
        "(from %s to %s).",
        len(asset_names),
        len(aligned_panel),
        aligned_panel.index[0].date(),
        aligned_panel.index[-1].date(),
    )

    return aligned_panel


def build_asset_return_panel(prices_panel: pd.DataFrame) -> pd.DataFrame:
    """Compute a multi-asset daily log-return panel from a price panel.

    Calls :func:`quant_platform.data.compute_log_returns` once per column of
    ``prices_panel`` and reassembles the results into a single panel. This
    function deliberately performs a per-column loop rather than a
    vectorised ``np.log(df / df.shift(1))`` operation, so that every
    validation check inside :func:`quant_platform.data.compute_log_returns`
    (non-positive price detection, minimum-length check, NaN handling) is
    applied identically to each asset, with no duplicated logic.

    Parameters
    ----------
    prices_panel:
        Price panel as returned by :func:`build_asset_price_panel`. Must
        have at least 2 rows and contain only positive values.

    Returns
    -------
    pd.DataFrame
        Return panel with the same columns as ``prices_panel``, and index
        equal to ``prices_panel.index[1:]`` (one row shorter, matching
        :func:`quant_platform.data.compute_log_returns`'s contract of
        dropping the leading ``NaN`` produced by the first difference).
        Values are daily log returns, ``float64``.

    Raises
    ------
    ValueError
        If ``prices_panel`` has fewer than 2 rows; or if any column raises
        ``ValueError`` from :func:`quant_platform.data.compute_log_returns`
        (e.g. a non-positive price in that column). In the latter case, the
        raised ``ValueError`` is re-raised with the offending asset's
        column name prepended to the message for diagnosability.
    TypeError
        Propagated unchanged from
        :func:`quant_platform.data.compute_log_returns` if ``prices_panel``
        is not a :class:`pandas.DataFrame` (raised when iterating columns
        produces a non-:class:`pandas.Series` object, which cannot occur for
        a valid ``DataFrame`` but is preserved here as a defensive contract).

    Examples
    --------
    >>> returns_panel = build_asset_return_panel(prices_panel)
    >>> returns_panel.shape[0] == prices_panel.shape[0] - 1
    True
    """

    if len(prices_panel) < 2:
        raise ValueError(
            f"prices_panel must have at least 2 observations to compute "
            f"returns; got {len(prices_panel)}."
        )

    logger.debug(
        "Computing log returns for %d asset(s) over %d observations.",
        prices_panel.shape[1],
        len(prices_panel),
    )

    per_asset_returns: dict[str, pd.Series] = {}
    for column in prices_panel.columns:
        try:
            returns_series = compute_log_returns(prices_panel[column])
        except ValueError as exc:
            raise ValueError(f"Asset {column!r}: {exc}") from exc
        per_asset_returns[column] = returns_series.rename(column)

    returns_panel = pd.concat(per_asset_returns.values(), axis=1)
    returns_panel = returns_panel[list(prices_panel.columns)]

    logger.info(
        "Asset return panel assembled: %d assets x %d observations.",
        returns_panel.shape[1],
        len(returns_panel),
    )

    return returns_panel


# ── Layer 2: Raw Signals ─────────────────────────────────────────────────────


def compute_momentum_signal(
    returns_panel: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.Series:
    """Compute a 12-1 month style cross-sectional momentum signal.

    For each asset, computes the cumulative log return over the trailing
    ``lookback`` trading days, excluding the most recent ``skip`` days. The
    ``skip`` window excludes short-term reversal effects: an asset that has
    just had a large recent move often partially reverses over the following
    few weeks, which would otherwise contaminate a pure momentum read.

    Parameters
    ----------
    returns_panel:
        Multi-asset log-return panel, as returned by
        :func:`build_asset_return_panel`. Must have at least ``lookback``
        rows.
    lookback:
        Total trailing window length in trading days. Default ``252``
        (approximately one trading year).
    skip:
        Number of most-recent trading days to exclude from the window.
        Default ``21`` (approximately one trading month).

    Returns
    -------
    pd.Series
        Index equal to ``returns_panel.columns`` (asset names), values equal
        to the cumulative log return over
        ``returns_panel.iloc[-lookback : -skip]`` (or
        ``returns_panel.iloc[-lookback:]`` when ``skip == 0``).

    Raises
    ------
    ValueError
        If ``skip >= lookback``; or if ``len(returns_panel) < lookback``.

    Notes
    -----
    A momentum value of exactly ``0.0`` for a given asset is a valid result
    (e.g. an asset with zero net log return, or zero returns throughout the
    window) and is not treated as an error or missing value.

    Examples
    --------
    >>> momentum = compute_momentum_signal(returns_panel)
    >>> momentum.index.tolist() == returns_panel.columns.tolist()
    True
    """

    if skip >= lookback:
        raise ValueError(
            f"skip ({skip}) must be strictly less than lookback ({lookback})."
        )

    if len(returns_panel) < lookback:
        raise ValueError(
            f"returns_panel has {len(returns_panel)} observation(s), but "
            f"lookback={lookback} requires at least {lookback}."
        )

    if skip == 0:
        window = returns_panel.iloc[-lookback:]
    else:
        window = returns_panel.iloc[-lookback:-skip]

    momentum = window.sum(axis=0)
    momentum.name = "momentum"

    logger.debug(
        "Momentum signal computed (lookback=%d, skip=%d): %s",
        lookback,
        skip,
        momentum.to_dict(),
    )

    return momentum


def compute_mean_reversion_signal(
    prices_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    short_window: int = 21,
    long_window: int = 126,
    lambda_: float = VAR.LAMBDA_EWMA,
) -> pd.Series:
    """Compute a volatility-normalised mean-reversion signal.

    For each asset, computes the deviation of the most recent price from its
    trailing ``long_window``-day moving average, normalised by the asset's
    most recent EWMA volatility (via
    :func:`quant_platform.risk.compute_ewma_variance`), with the sign
    flipped so that a price *below* its long-run average -- relative to its
    own volatility -- produces a *positive* (attractive) score.

    Parameters
    ----------
    prices_panel:
        Multi-asset price panel, as returned by
        :func:`build_asset_price_panel`. Must have at least ``long_window``
        rows.
    returns_panel:
        Multi-asset log-return panel, as returned by
        :func:`build_asset_return_panel`. Must have the same columns as
        ``prices_panel`` (in the same order).
    short_window:
        Short moving-average window in trading days. Default ``21``. Used
        only as a diagnostic comparison (see Notes); it does not enter the
        returned signal's formula.
    long_window:
        Long moving-average window in trading days. Default ``126``
        (approximately six months).
    lambda_:
        EWMA decay factor passed to
        :func:`quant_platform.risk.compute_ewma_variance`. Default
        ``VAR.LAMBDA_EWMA`` (``0.72``), ensuring this module's notion of
        "volatility" is identical to ``risk.py``'s.

    Returns
    -------
    pd.Series
        Index equal to ``prices_panel.columns`` (asset names), values equal
        to ``-((price - long_ma) / long_ma) / ewma_vol`` evaluated at the
        most recent observation for each asset.

    Raises
    ------
    ValueError
        If ``short_window >= long_window``; if
        ``len(prices_panel) < long_window``; if
        ``prices_panel.columns`` does not equal ``returns_panel.columns``
        (same set, same order); if any asset's most recent EWMA volatility
        is exactly zero (division by zero, raised per-asset with the
        asset's name); or if any asset's ``long_window``-day moving average
        is exactly zero.

    Notes
    -----
    The short-window moving average (``short_window``) is computed
    internally for diagnostic comparison -- confirming whether an asset
    flagged as attractive by the long-window mean-reversion read is *also*
    exhibiting short-term momentum that would contradict the mean-reversion
    interpretation -- but it does not appear in the returned signal's
    formula. This is intentional: the core signal is purely a function of
    the long-window deviation and EWMA volatility. A future maintainer
    should not "simplify" this function by removing the short-window
    computation without first verifying it is not consumed by a diagnostic
    caller, since its omission from the return value is by design, not an
    oversight.

    Unlike :func:`compute_volatility_scalar`, a zero EWMA volatility here is
    treated as a hard error rather than a soft default, because this
    function's output is a *ratio* with no safe representation for
    division by zero -- whereas :func:`compute_volatility_scalar`'s output
    is a *normalised weight* for which zero is a meaningful, safe value.

    Examples
    --------
    >>> mean_reversion = compute_mean_reversion_signal(prices_panel, returns_panel)
    >>> mean_reversion.index.tolist() == prices_panel.columns.tolist()
    True
    """

    if short_window >= long_window:
        raise ValueError(
            f"short_window ({short_window}) must be strictly less than "
            f"long_window ({long_window})."
        )

    if len(prices_panel) < long_window:
        raise ValueError(
            f"prices_panel has {len(prices_panel)} observation(s), but "
            f"long_window={long_window} requires at least {long_window}."
        )

    prices_columns = list(prices_panel.columns)
    returns_columns = list(returns_panel.columns)
    if prices_columns != returns_columns:
        mismatched = sorted(set(prices_columns) ^ set(returns_columns))
        raise ValueError(
            "prices_panel and returns_panel must have identical columns in "
            f"the same order. prices_panel columns: {prices_columns!r}, "
            f"returns_panel columns: {returns_columns!r}. "
            f"Mismatched asset(s): {mismatched!r}."
        )

    mean_reversion_values: dict[str, float] = {}

    for asset in prices_columns:
        asset_prices = prices_panel[asset]
        asset_returns = returns_panel[asset]

        short_ma = asset_prices.rolling(window=short_window).mean().iloc[-1]
        long_ma = asset_prices.rolling(window=long_window).mean().iloc[-1]

        logger.debug(
            "Asset %r: short_ma(%d)=%.6f, long_ma(%d)=%.6f (short_ma is "
            "diagnostic-only and does not enter the returned signal).",
            asset,
            short_window,
            short_ma,
            long_window,
            long_ma,
        )

        if long_ma == 0.0:
            raise ValueError(
                f"Asset {asset!r}: long-window ({long_window}-day) moving "
                f"average is zero; cannot compute relative deviation "
                f"(division by zero)."
            )

        current_price = asset_prices.iloc[-1]
        deviation = (current_price - long_ma) / long_ma

        ewma_variance = compute_ewma_variance(asset_returns, lambda_=lambda_)
        ewma_vol = float(np.sqrt(ewma_variance.iloc[-1]))

        if ewma_vol == 0.0:
            raise ValueError(
                f"Asset {asset!r}: most recent EWMA volatility is zero; "
                f"cannot compute mean-reversion signal (division by zero)."
            )

        mean_reversion_values[asset] = float(-deviation / ewma_vol)

    mean_reversion = pd.Series(mean_reversion_values, index=prices_columns, name="mean_reversion")

    logger.debug("Mean-reversion signal computed: %s", mean_reversion.to_dict())

    return mean_reversion


def compute_volatility_scalar(
    returns_panel: pd.DataFrame,
    lambda_: float = VAR.LAMBDA_EWMA,
) -> pd.Series:
    """Compute inverse-volatility weights normalised to sum to one.

    For each asset, computes the most recent EWMA volatility via
    :func:`quant_platform.risk.compute_ewma_variance`, then returns
    ``1 / volatility`` normalised across the asset universe so the result
    sums to ``1.0`` -- the same "risk-parity-style" sizing logic familiar
    from inverse-volatility portfolio construction, repurposed here as a
    multiplicative modulator on the momentum/mean-reversion composite score
    in :func:`blend_signals`.

    Parameters
    ----------
    returns_panel:
        Multi-asset log-return panel, as returned by
        :func:`build_asset_return_panel`.
    lambda_:
        EWMA decay factor passed to
        :func:`quant_platform.risk.compute_ewma_variance`. Default
        ``VAR.LAMBDA_EWMA`` (``0.72``). Must satisfy ``0 < lambda_ < 1``.

    Returns
    -------
    pd.Series
        Index equal to ``returns_panel.columns`` (asset names), values
        non-negative and summing to ``1.0``.

    Raises
    ------
    ValueError
        If ``lambda_`` is not strictly between ``0`` and ``1``; or if every
        asset's most recent EWMA volatility is exactly zero (in which case
        no asset can receive a non-zero inverse-volatility weight and the
        normalisation ``inv_vol / inv_vol.sum()`` would divide by zero).

    Notes
    -----
    An individual asset with zero EWMA volatility (while other assets have
    positive volatility) receives an inverse-volatility weight of exactly
    ``0.0`` -- not ``inf`` and not an error. This reflects the
    interpretation "a zero-volatility asset cannot be sized by inverse
    volatility, so it receives no weight from this signal" -- the opposite
    failure mode from :func:`compute_mean_reversion_signal`, where a zero
    EWMA volatility is a hard error because that function's output is a
    ratio with no safe zero-representation. Here, the output is a
    normalised weight vector, for which zero is the natural and safe
    representation.

    Examples
    --------
    >>> vol_scalar = compute_volatility_scalar(returns_panel)
    >>> abs(vol_scalar.sum() - 1.0) < 1e-12
    True
    """

    if not (0.0 < lambda_ < 1.0):
        raise ValueError(
            f"lambda_ must satisfy 0 < lambda_ < 1; got {lambda_}."
        )

    ewma_vols: dict[str, float] = {}
    for asset in returns_panel.columns:
        ewma_variance = compute_ewma_variance(returns_panel[asset], lambda_=lambda_)
        ewma_vols[asset] = float(np.sqrt(ewma_variance.iloc[-1]))

    logger.debug("Per-asset EWMA volatility (lambda=%.4f): %s", lambda_, ewma_vols)

    inv_vol = pd.Series(
        {asset: (1.0 / vol if vol > 0.0 else 0.0) for asset, vol in ewma_vols.items()},
        index=returns_panel.columns,
    )

    inv_vol_sum = float(inv_vol.sum())
    if inv_vol_sum == 0.0:
        raise ValueError(
            "All assets have zero EWMA volatility; cannot compute "
            "inverse-volatility weights (normalisation would divide by "
            f"zero). Per-asset EWMA volatilities: {ewma_vols!r}."
        )

    vol_scalar = inv_vol / inv_vol_sum
    vol_scalar.name = "vol_scalar"

    logger.debug("Volatility scalar computed: %s", vol_scalar.to_dict())

    return vol_scalar


# ── Layer 3: Composite Score ─────────────────────────────────────────────────


def zscore_cross_sectional(signal: pd.Series) -> pd.Series:
    """Compute the cross-sectional z-score of a signal.

    Standardises ``signal`` to zero mean and unit sample standard deviation
    across its index (interpreted as the asset universe at a single point
    in time, not as a time series).

    Parameters
    ----------
    signal:
        Cross-sectional signal, one value per asset.

    Returns
    -------
    pd.Series
        Same index as ``signal``, values equal to
        ``(signal - signal.mean()) / signal.std(ddof=1)``.

    Raises
    ------
    ValueError
        If ``len(signal) < 2`` (sample standard deviation requires at least
        2 observations); or if ``signal.std(ddof=1) == 0`` (every asset has
        an identical raw signal value, making the z-score ``0/0``,
        undefined).

    Notes
    -----
    A cross-section in which every asset has an identical raw signal value
    is treated as a data-quality error, not as "no signal" (which would be
    represented by a z-score of ``0.0``). These are different situations:
    "no signal" means the cross-section has normal dispersion but this
    particular asset sits at the mean; "identical values" means the
    cross-section has *no dispersion at all*, which is degenerate for any
    cross-sectional comparison and should surface rather than be silently
    mapped to all-zeros.

    Examples
    --------
    >>> z = zscore_cross_sectional(pd.Series([1.0, 2.0, 3.0, 4.0]))
    >>> abs(z.mean()) < 1e-12
    True
    """

    if len(signal) < 2:
        raise ValueError(
            f"signal must have at least 2 observations to compute a "
            f"cross-sectional z-score; got {len(signal)}."
        )

    std = signal.std(ddof=1)

    if std == 0.0:
        unique_values = signal.unique()
        raise ValueError(
            "Cross-sectional standard deviation is zero; all assets have "
            f"identical signal value {unique_values[0]!r}; cannot compute "
            "z-scores."
        )

    zscored = (signal - signal.mean()) / std
    zscored.name = signal.name

    return zscored


def blend_signals(
    momentum: pd.Series,
    mean_reversion: pd.Series,
    vol_scalar: pd.Series,
    momentum_weight: float = 0.5,
    mean_reversion_weight: float = 0.3,
) -> pd.Series:
    """Combine momentum, mean-reversion, and volatility signals into a
    single composite alpha score.

    The momentum and mean-reversion signals are each cross-sectionally
    z-scored (via :func:`zscore_cross_sectional`) and combined as a weighted
    sum. The result is then *modulated* (multiplied), not blended
    additively, by a re-centred version of ``vol_scalar`` -- assets with
    below-average volatility have their combined signal amplified, and
    assets with above-average volatility have it dampened.

    Parameters
    ----------
    momentum:
        Output of :func:`compute_momentum_signal`.
    mean_reversion:
        Output of :func:`compute_mean_reversion_signal`. Must share the
        same index as ``momentum``.
    vol_scalar:
        Output of :func:`compute_volatility_scalar`. Must share the same
        index as ``momentum``, must be non-negative, and must not be
        all-zero.
    momentum_weight:
        Weight applied to the momentum z-score. Default ``0.5``.
    mean_reversion_weight:
        Weight applied to the mean-reversion z-score. Default ``0.3``.
        Must satisfy ``momentum_weight + mean_reversion_weight <= 1.0``
        together with ``momentum_weight``.

    Returns
    -------
    pd.Series
        Index equal to ``momentum.index``, values equal to
        ``(momentum_weight * momentum_z + mean_reversion_weight *
        mean_reversion_z) * (vol_scalar * len(vol_scalar))``.

    Raises
    ------
    ValueError
        If ``momentum``, ``mean_reversion``, and ``vol_scalar`` do not share
        an identical index (same assets, same order); if
        ``momentum_weight < 0`` or ``mean_reversion_weight < 0``; if
        ``momentum_weight + mean_reversion_weight > 1.0``; if any value in
        ``vol_scalar`` is negative; or if ``vol_scalar`` is all-zero.

    Notes
    -----
    ``momentum_weight`` and ``mean_reversion_weight`` are *not* two terms of
    a three-way convex combination with an implicit "volatility weight" of
    ``1 - momentum_weight - mean_reversion_weight``. Volatility always acts
    as a *multiplicative modulator* on ``combined_z =
    momentum_weight * momentum_z + mean_reversion_weight * mean_reversion_z``,
    regardless of the explicit weights chosen. When
    ``momentum_weight + mean_reversion_weight == 1.0``, the modulation by
    ``vol_scalar`` still applies -- there is no configuration in which
    volatility additively contributes to the composite score.

    ``vol_scalar`` is re-centred by multiplying by ``len(vol_scalar)``
    (the number of assets): since ``vol_scalar`` sums to ``1.0`` across the
    universe by construction (the contract of
    :func:`compute_volatility_scalar`), ``vol_scalar * N`` has a
    cross-sectional mean of ``1.0``. An asset with average volatility
    therefore has its ``combined_z`` left unchanged; an asset with
    below-average volatility has its ``combined_z`` amplified
    (``vol_scalar_centred > 1``); an asset with above-average volatility has
    it dampened (``vol_scalar_centred < 1``).

    Examples
    --------
    >>> composite_alpha = blend_signals(momentum, mean_reversion, vol_scalar)
    >>> composite_alpha.index.tolist() == momentum.index.tolist()
    True
    """

    momentum_index = list(momentum.index)
    mean_reversion_index = list(mean_reversion.index)
    vol_scalar_index = list(vol_scalar.index)

    if momentum_index != mean_reversion_index or momentum_index != vol_scalar_index:
        all_assets = set(momentum_index) | set(mean_reversion_index) | set(vol_scalar_index)
        common_assets = set(momentum_index) & set(mean_reversion_index) & set(vol_scalar_index)
        mismatched = sorted(all_assets - common_assets)
        raise ValueError(
            "momentum, mean_reversion, and vol_scalar must share an "
            "identical index (same assets, same order). "
            f"momentum index: {momentum_index!r}, "
            f"mean_reversion index: {mean_reversion_index!r}, "
            f"vol_scalar index: {vol_scalar_index!r}. "
            f"Mismatched asset(s): {mismatched!r}."
        )

    if momentum_weight < 0.0:
        raise ValueError(f"momentum_weight must be non-negative; got {momentum_weight}.")
    if mean_reversion_weight < 0.0:
        raise ValueError(
            f"mean_reversion_weight must be non-negative; got {mean_reversion_weight}."
        )
    if momentum_weight + mean_reversion_weight > 1.0:
        raise ValueError(
            f"momentum_weight + mean_reversion_weight must be <= 1.0; got "
            f"{momentum_weight} + {mean_reversion_weight} = "
            f"{momentum_weight + mean_reversion_weight}."
        )

    if (vol_scalar < 0.0).any():
        negative_assets = vol_scalar[vol_scalar < 0.0].index.tolist()
        raise ValueError(
            f"vol_scalar must be non-negative; found negative value(s) for "
            f"asset(s): {negative_assets!r}."
        )

    if float(vol_scalar.sum()) == 0.0:
        raise ValueError(
            "vol_scalar is all-zero; composite alpha would be degenerate "
            "(every asset's combined signal would be multiplied by zero)."
        )

    momentum_z = zscore_cross_sectional(momentum)
    mean_reversion_z = zscore_cross_sectional(mean_reversion)

    combined_z = momentum_weight * momentum_z + mean_reversion_weight * mean_reversion_z

    n_assets = len(vol_scalar)
    vol_scalar_centred = vol_scalar * n_assets

    composite_alpha = combined_z * vol_scalar_centred
    composite_alpha = composite_alpha.reindex(momentum_index)
    composite_alpha.name = "composite_alpha"

    logger.info(
        "Composite alpha computed (momentum_weight=%.2f, "
        "mean_reversion_weight=%.2f): %s",
        momentum_weight,
        mean_reversion_weight,
        composite_alpha.to_dict(),
    )

    return composite_alpha


# ── Layer 4: Portfolio Interface ─────────────────────────────────────────────


def generate_expected_returns(
    composite_alpha: pd.Series,
    base_return: float = MARKET.MVO_TARGET_RETURN,
    scale: float = 0.05,
) -> NDArray[np.float64]:
    """Convert a composite alpha score into an expected-return vector.

    Applies the affine transform ``mu_i = base_return + scale *
    composite_alpha_i``. The resulting array is the ``mu`` vector passed
    directly to :func:`quant_platform.portfolio.compute_mvo_weights` and
    :func:`quant_platform.portfolio.compute_mvo_scalars`, replacing the
    static ``MARKET.MVO_MU`` default.

    Parameters
    ----------
    composite_alpha:
        Output of :func:`blend_signals`. Must not contain ``NaN`` or
        infinite values.
    base_return:
        The expected return assigned to an asset with
        ``composite_alpha_i == 0`` (no signal). Default
        ``MARKET.MVO_TARGET_RETURN`` (``0.045``). With this default, an
        all-zero ``composite_alpha`` produces ``mu = [base_return, ...,
        base_return]``, under which
        ``portfolio.compute_mvo_weights(mu=mu, target_return=base_return)``
        is satisfied by *any* portfolio weighting (since ``w^T mu =
        base_return`` for all ``w`` summing to 1 when every ``mu_i`` is
        equal), causing the optimizer to fall back to the global
        minimum-variance portfolio -- the textbook-correct "no view" result.
    scale:
        Multiplier controlling how strongly ``composite_alpha`` perturbs
        ``mu`` away from ``base_return``. Must be strictly positive. Default
        ``0.05``, calibrated so that ``composite_alpha_i = +-1.0`` (one
        cross-sectional standard deviation, the typical magnitude produced
        by :func:`zscore_cross_sectional`) produces a ``+-5`` percentage
        point swing in ``mu_i``.

    Returns
    -------
    NDArray[np.float64]
        1-D array of length ``len(composite_alpha)``, in the same order as
        ``composite_alpha.index``.

    Raises
    ------
    ValueError
        If ``scale <= 0``; or if ``composite_alpha`` contains any ``NaN`` or
        infinite value (the affected asset name(s) are listed in the
        exception message).

    Notes
    -----
    This function does **not** clip or floor ``mu`` at zero. A negative
    ``mu_i`` (e.g. ``composite_alpha_i = -1.0`` with the defaults produces
    ``mu_i = -0.005``) is a valid and expected output -- it signals to the
    optimizer that this asset's expected return is below the risk-free-rate
    range, which is the entire purpose of a cross-sectional alpha model
    that can recommend underweighting or shorting an asset.

    Examples
    --------
    >>> composite_alpha = pd.Series([1.0, -1.0, 0.0], index=["A", "B", "C"])
    >>> mu = generate_expected_returns(composite_alpha, base_return=0.045, scale=0.05)
    >>> np.allclose(mu, [0.095, -0.005, 0.045])
    True
    """

    if scale <= 0.0:
        raise ValueError(f"scale must be strictly positive; got {scale}.")

    values = composite_alpha.to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        bad_assets = composite_alpha.index[~np.isfinite(values)].tolist()
        raise ValueError(
            f"composite_alpha contains NaN or infinite value(s) for "
            f"asset(s): {bad_assets!r}. Cannot generate expected returns "
            f"from a non-finite composite alpha."
        )

    mu = (base_return + scale * values).astype(np.float64)

    logger.info(
        "Expected returns generated (base_return=%.4f, scale=%.4f): %s",
        base_return,
        scale,
        dict(zip(composite_alpha.index, mu.tolist())),
    )

    return mu


def rank_assets(composite_alpha: pd.Series, ascending: bool = False) -> pd.DataFrame:
    """Rank assets by composite alpha score.

    Parameters
    ----------
    composite_alpha:
        Output of :func:`blend_signals`.
    ascending:
        If ``False`` (default), the asset with the highest
        ``composite_alpha`` receives ``Rank == 1`` (most attractive first).
        If ``True``, the asset with the lowest ``composite_alpha`` receives
        ``Rank == 1``.

    Returns
    -------
    pd.DataFrame
        Columns ``["Asset", "Composite Alpha", "Rank"]``, sorted by ``Rank``
        ascending. ``Rank`` is a strict ``1..N`` permutation with no
        duplicate or missing ranks, with ties broken by the original index
        order of ``composite_alpha`` (``method="first"``).

    Raises
    ------
    ValueError
        If ``composite_alpha`` is empty.

    Notes
    -----
    Tied ``composite_alpha`` values are resolved deterministically by
    original index order (``method="first"``), not by an arbitrary or
    run-dependent tiebreak. This determinism is required by downstream
    turnover diagnostics, which compare rank *sets* across time periods --
    a non-deterministic tiebreak would produce spurious turnover between
    two periods with identical underlying scores.

    Examples
    --------
    >>> composite_alpha = pd.Series([0.5, -0.3, 1.2, 0.0], index=["A", "B", "C", "D"])
    >>> ranking = rank_assets(composite_alpha)
    >>> ranking.iloc[0]["Asset"]
    'C'
    >>> ranking.iloc[0]["Rank"]
    1
    """

    if len(composite_alpha) == 0:
        raise ValueError("composite_alpha is empty; cannot rank an empty asset universe.")

    ranking = pd.DataFrame(
        {
            "Asset": composite_alpha.index,
            "Composite Alpha": composite_alpha.to_numpy(dtype=np.float64),
        }
    )
    ranking["Rank"] = (
        ranking["Composite Alpha"].rank(ascending=ascending, method="first").astype(int)
    )
    ranking = ranking.sort_values("Rank").reset_index(drop=True)

    logger.debug("Asset ranking computed (ascending=%s): %s", ascending, ranking.to_dict("records"))

    return ranking