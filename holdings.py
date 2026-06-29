"""holdings.py — ETF holdings ingestion for the Quantitative ETF Risk &
Derivatives Analytics Platform (Version 3.0).

This module sits before ``alpha.py`` in the V3 architecture::

    holdings.py -> market_data.py -> alpha.py -> portfolio.py -> ...

Its responsibility is to read a real ETF's published constituent-holdings
file (a CSV containing, at minimum, ticker, weight, and sector columns),
represent it as a structured, validated, point-in-time snapshot, optionally
restrict that snapshot to its largest constituents, and convert the result
into the same configuration shape ``alpha.py`` already consumes --
:class:`quant_platform.alpha.AssetUniverseConfig` -- with one additive field
(``holdings_weights``) carrying each constituent's *current* ETF weight.

Pipeline overview
------------------
::

    holdings CSV (Ticker, Name, Weight, Sector, MarketValue, ...)
        |
        v
    load_holdings_snapshot   -> HoldingsSnapshot   (validated, point-in-time)
        |
        v
    filter_top_n_holdings    -> HoldingsSnapshot   (top N by weight, residual
        |                                            folded into cash_weight)
        v
    build_universe_config    -> HoldingsUniverseConfig
        |
        v
    alpha.build_asset_price_panel(universe)   <-- ZERO alpha.py changes
    alpha.build_asset_return_panel(...)       <-- ZERO alpha.py changes

Design principles
-------------------
- :class:`HoldingsUniverseConfig` is a structural superset of
  :class:`quant_platform.alpha.AssetUniverseConfig`: it exposes the same
  three attributes (``asset_names``, ``filepaths``, ``tickers``) with the
  same types, so any function written against
  ``alpha.AssetUniverseConfig`` (duck-typed via attribute access, as
  ``alpha.build_asset_price_panel`` is) accepts a
  :class:`HoldingsUniverseConfig` instance unchanged.
- No ETF is hardcoded anywhere in this module. Every function operates on
  whatever ``etf_ticker`` and holdings file the caller supplies.
- Validation failures raise descriptive :class:`ValueError` exceptions
  immediately, following the same philosophy as
  :func:`quant_platform.data.load_price_data`: no silent correction, no
  fabricated data, exception messages name the specific offending row(s) or
  ticker(s).
- Constituents whose price files are not yet available on disk are
  *excluded* from the resulting universe with a logged ``WARNING`` (not a
  raised error) -- a real ETF with many constituents will routinely have a
  handful of names without yet-downloaded price history, and excluding them
  (with their weight folded into ``cash_weight``) is the correct, non-fatal
  response. This is the one place this module diverges from
  ``alpha.AssetUniverseConfig``'s V1.0/V2.0 contract, where every configured
  asset was a hard precondition by design.

No existing module (``config.py``, ``data.py``, ``alpha.py``,
``portfolio.py``, ``risk.py``, ``pricing.py``, ``scenarios.py``,
``reporting.py``, ``main.py``) is modified by this module.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "Holding",
    "HoldingsSnapshot",
    "HoldingsUniverseConfig",
    "load_holdings_snapshot",
    "filter_top_n_holdings",
    "build_universe_config",
]

logger = logging.getLogger("quant_platform.holdings")


# ── Tolerances ────────────────────────────────────────────────────────────

# Published ETF holdings weights are themselves rounded (typically to 2-4
# decimal places), so the sum of all holding weights plus cash will not
# equal exactly 1.0. This tolerance defines "approximately equal to 1" for
# the purposes of load_holdings_snapshot's validation.
_WEIGHT_SUM_TOLERANCE = 1e-2


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Holding:
    """A single constituent holding within an ETF, as of one date.

    Parameters
    ----------
    ticker:
        Exchange ticker symbol of the constituent, e.g. ``"AAPL"``. Must be
        non-empty.
    name:
        Full security name, e.g. ``"Apple Inc"``.
    weight:
        Portfolio weight as a fraction of the ETF's net asset value, e.g.
        ``0.0712`` for 7.12%. Must be non-negative.
    sector:
        Sector classification as published by the ETF issuer, e.g.
        ``"Information Technology"``. May be an empty string if the issuer
        does not classify this holding (e.g. cash equivalents), but is
        never ``None``.
    market_value:
        Market value of this holding in the ETF's reporting currency. Must
        be non-negative.

    Notes
    -----
    This dataclass is intentionally minimal -- it captures exactly the
    fields :func:`build_universe_config` and downstream ``alpha.py``
    functions require (``ticker``, ``weight``) plus two fields
    (``name``, ``sector``) retained for display and future sector-level
    diagnostics. Additional issuer-specific fields present in a holdings
    CSV (e.g. CUSIP, exchange, currency) are read by
    :func:`load_holdings_snapshot` for validation purposes where relevant,
    but are not retained on :class:`Holding` itself -- if a future module
    requires them, they should be added here explicitly rather than passed
    through an untyped dict.

    Examples
    --------
    >>> holding = Holding(
    ...     ticker="AAPL",
    ...     name="Apple Inc",
    ...     weight=0.0712,
    ...     sector="Information Technology",
    ...     market_value=1_234_567.89,
    ... )
    """

    ticker: str
    name: str
    weight: float
    sector: str
    market_value: float


@dataclass(frozen=True)
class HoldingsSnapshot:
    """A point-in-time snapshot of an ETF's constituent holdings.

    Parameters
    ----------
    etf_ticker:
        Ticker symbol of the ETF itself, e.g. ``"XLK"``. Must be non-empty.
    as_of_date:
        Date on which this holdings composition was published.
    holdings:
        Tuple of :class:`Holding` instances. Must be non-empty, must
        contain no duplicate tickers, and every holding's ``weight`` must
        be non-negative.
    total_market_value:
        Total market value of the ETF's portfolio, in the same currency as
        each holding's ``market_value``. Must be non-negative.
    cash_weight:
        Residual weight not represented by ``holdings`` -- typically cash,
        cash equivalents, or rounding residual. Must be non-negative.
        ``sum(h.weight for h in holdings) + cash_weight`` must be
        approximately ``1.0`` (within :data:`_WEIGHT_SUM_TOLERANCE`).

    Notes
    -----
    This dataclass is frozen and therefore immutable. Functions that
    transform a :class:`HoldingsSnapshot` (such as
    :func:`filter_top_n_holdings`) return a **new** instance rather than
    mutating the input -- consistent with the platform's pure-function
    design philosophy established in ``alpha.py``.

    Examples
    --------
    >>> snapshot = HoldingsSnapshot(
    ...     etf_ticker="XLK",
    ...     as_of_date=pd.Timestamp("2026-06-13"),
    ...     holdings=(holding_a, holding_b),
    ...     total_market_value=50_000_000_000.0,
    ...     cash_weight=0.0021,
    ... )
    """

    etf_ticker: str
    as_of_date: pd.Timestamp
    holdings: tuple[Holding, ...]
    total_market_value: float
    cash_weight: float


@dataclass(frozen=True)
class HoldingsUniverseConfig:
    """Asset universe configuration derived from an ETF's holdings snapshot.

    This dataclass is a structural superset of
    :class:`quant_platform.alpha.AssetUniverseConfig`: it exposes the same
    three attributes (``asset_names``, ``filepaths``, ``tickers``) with the
    same types and the same meanings, plus one additive field
    (``holdings_weights``). Because
    :func:`quant_platform.alpha.build_asset_price_panel` accesses its
    ``universe`` argument only via these three attributes (duck typing, no
    ``isinstance`` check), an instance of this class can be passed directly
    to ``alpha.build_asset_price_panel`` and
    ``alpha.build_asset_return_panel`` with **no changes to alpha.py**.

    Parameters
    ----------
    asset_names:
        Ordered tuple of constituent tickers, ordered by descending ETF
        weight (the order ``alpha.py``'s output vectors -- ``momentum``,
        ``mean_reversion``, ``vol_scalar``, ``composite_alpha``, ``mu`` --
        will follow).
    filepaths:
        Mapping from ticker to the price-history workbook path for that
        ticker, suitable for :func:`quant_platform.data.load_price_data`.
    tickers:
        Mapping from ticker to the column name within that ticker's price
        workbook (the ``ticker`` argument to
        :func:`quant_platform.data.load_price_data`). In the common case
        where the price file's price column is named identically to the
        ticker itself, ``tickers[t] == t`` for every ``t`` in
        ``asset_names``.
    holdings_weights:
        Mapping from ticker to that constituent's current ETF weight (as a
        fraction of NAV), sourced from the
        :class:`HoldingsSnapshot` this configuration was built from. This
        is the ``current_weight`` input that the V2.0 decision-engine design
        compares against ``portfolio.compute_mvo_weights``'s
        ``target_weight`` output for portfolio drift analysis.

    Notes
    -----
    ``asset_names``, ``filepaths``, and ``tickers`` will in general contain
    *fewer* entries than the originating :class:`HoldingsSnapshot`'s
    ``holdings`` tuple: any constituent whose price file does not exist on
    disk at the time :func:`build_universe_config` is called is excluded
    (logged at ``WARNING``), since ``alpha.build_asset_price_panel`` would
    otherwise raise ``FileNotFoundError`` for the entire universe over a
    single missing constituent.

    Examples
    --------
    >>> universe = build_universe_config(snapshot, price_data_dir="data/raw")
    >>> prices_panel = alpha.build_asset_price_panel(universe)  # zero alpha.py changes
    """

    asset_names: tuple[str, ...]
    filepaths: dict[str, str] = field(default_factory=dict)
    tickers: dict[str, str] = field(default_factory=dict)
    holdings_weights: dict[str, float] = field(default_factory=dict)


# ── Loading ───────────────────────────────────────────────────────────────────


def load_holdings_snapshot(
    filepath: str | os.PathLike[str],
    etf_ticker: str,
    as_of_date: pd.Timestamp | None = None,
    *,
    ticker_column: str = "Ticker",
    name_column: str = "Name",
    weight_column: str = "Weight",
    sector_column: str = "Sector",
    market_value_column: str = "MarketValue",
) -> HoldingsSnapshot:
    """Load and validate an ETF holdings snapshot from a CSV file.

    Reads a holdings file containing one row per constituent, with at
    minimum a ticker column and a weight column. Column names are
    configurable via keyword arguments to accommodate different ETF
    issuers' export formats; the defaults match the field names specified
    in this module's design (``Ticker``, ``Name``, ``Weight``, ``Sector``,
    ``MarketValue``).

    Parameters
    ----------
    filepath:
        Path to the holdings CSV file.
    etf_ticker:
        Ticker symbol of the ETF whose holdings are being loaded, e.g.
        ``"XLK"``. Must be a non-empty string. This value is stored on the
        returned :class:`HoldingsSnapshot` and is not read from the file --
        no ETF is hardcoded by this module, but the caller must identify
        which ETF the file describes.
    as_of_date:
        Date on which this holdings composition was published. If ``None``
        (default), the file's modification time
        (:meth:`pathlib.Path.stat`'s ``st_mtime``) is used as a fallback,
        with a ``WARNING`` logged noting that an explicit ``as_of_date`` was
        not provided.
    ticker_column, name_column, weight_column, sector_column, market_value_column:
        Column names within the CSV corresponding to each
        :class:`Holding` field. Defaults match
        ``Ticker``, ``Name``, ``Weight``, ``Sector``, ``MarketValue``.
        ``name_column``, ``sector_column``, and ``market_value_column`` are
        optional: if a configured column name is not present in the file,
        the corresponding :class:`Holding` field is populated with
        ``""`` (for ``name``/``sector``) or ``0.0`` (for ``market_value``),
        and a ``WARNING`` is logged. ``ticker_column`` and ``weight_column``
        are required and raise ``ValueError`` if absent.

    Returns
    -------
    HoldingsSnapshot
        Validated snapshot. ``holdings`` is ordered by descending
        ``weight``. ``cash_weight`` is computed as
        ``max(0.0, 1.0 - sum(h.weight for h in holdings))``.
        ``total_market_value`` is the sum of all holdings' ``market_value``
        (which is ``0.0`` if ``market_value_column`` was not present in the
        file).

    Raises
    ------
    ValueError
        If ``etf_ticker`` is empty; if ``ticker_column`` or
        ``weight_column`` is missing from the file; if the file contains a
        duplicate ticker; if any ``weight`` value is negative or
        non-numeric; if any ``ticker`` value is empty or whitespace-only;
        or if the sum of all holding weights plus the computed
        ``cash_weight`` deviates from ``1.0`` by more than
        :data:`_WEIGHT_SUM_TOLERANCE` (which can only occur if the sum of
        weights *exceeds* ``1.0``, since ``cash_weight`` is the
        non-negative residual against ``1.0``).
    FileNotFoundError
        If ``filepath`` does not exist.

    Examples
    --------
    >>> snapshot = load_holdings_snapshot(
    ...     "data/holdings/XLK_holdings_2026-06-13.csv",
    ...     etf_ticker="XLK",
    ...     as_of_date=pd.Timestamp("2026-06-13"),
    ... )
    >>> snapshot.holdings[0].ticker  # largest holding by weight
    'NVDA'
    """

    if not isinstance(etf_ticker, str) or not etf_ticker.strip():
        raise ValueError(f"etf_ticker must be a non-empty string; got {etf_ticker!r}.")

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Holdings file not found: '{path.resolve()}'\n"
            f"Place the ETF holdings CSV at this path and retry."
        )

    logger.info("Loading holdings snapshot for %r from %r", etf_ticker, str(path))

    raw = pd.read_csv(path)

    if ticker_column not in raw.columns:
        raise ValueError(
            f"Required column {ticker_column!r} not found in {path.name!r}. "
            f"Available columns: {list(raw.columns)!r}."
        )
    if weight_column not in raw.columns:
        raise ValueError(
            f"Required column {weight_column!r} not found in {path.name!r}. "
            f"Available columns: {list(raw.columns)!r}."
        )

    has_name = name_column in raw.columns
    has_sector = sector_column in raw.columns
    has_market_value = market_value_column in raw.columns

    if not has_name:
        logger.warning(
            "Column %r not found in %r; Holding.name will default to ''.",
            name_column,
            path.name,
        )
    if not has_sector:
        logger.warning(
            "Column %r not found in %r; Holding.sector will default to ''.",
            sector_column,
            path.name,
        )
    if not has_market_value:
        logger.warning(
            "Column %r not found in %r; Holding.market_value and "
            "HoldingsSnapshot.total_market_value will default to 0.0.",
            market_value_column,
            path.name,
        )

    # ── Per-row validation: ticker ──────────────────────────────────────────
    raw_tickers = raw[ticker_column]
    blank_ticker_rows = raw.index[
        raw_tickers.isna() | (raw_tickers.astype(str).str.strip() == "")
    ].tolist()
    if blank_ticker_rows:
        raise ValueError(
            f"{path.name!r} contains empty or missing {ticker_column!r} "
            f"value(s) at row(s) {blank_ticker_rows!r} (0-indexed, excluding "
            f"header)."
        )

    tickers = raw_tickers.astype(str).str.strip()

    # ── Duplicate ticker check ──────────────────────────────────────────────
    duplicate_mask = tickers.duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_tickers = sorted(tickers[duplicate_mask].unique().tolist())
        raise ValueError(
            f"{path.name!r} contains duplicate ticker(s): "
            f"{duplicate_tickers!r}. Each constituent must appear exactly "
            f"once."
        )

    # ── Weight validation ────────────────────────────────────────────────────
    raw_weights = pd.to_numeric(raw[weight_column], errors="coerce")
    non_numeric_rows = raw.index[raw_weights.isna() & raw[weight_column].notna()].tolist()
    missing_weight_rows = raw.index[raw[weight_column].isna()].tolist()
    if non_numeric_rows:
        bad_tickers = tickers.loc[non_numeric_rows].tolist()
        raise ValueError(
            f"{path.name!r} contains non-numeric {weight_column!r} value(s) "
            f"for ticker(s): {bad_tickers!r} (row(s) {non_numeric_rows!r})."
        )
    if missing_weight_rows:
        bad_tickers = tickers.loc[missing_weight_rows].tolist()
        raise ValueError(
            f"{path.name!r} contains missing {weight_column!r} value(s) for "
            f"ticker(s): {bad_tickers!r} (row(s) {missing_weight_rows!r})."
        )

    negative_weight_rows = raw.index[raw_weights < 0.0].tolist()
    if negative_weight_rows:
        bad = {
            tickers.loc[i]: float(raw_weights.loc[i]) for i in negative_weight_rows
        }
        raise ValueError(
            f"{path.name!r} contains negative {weight_column!r} value(s): "
            f"{bad!r}. Holding weights must be non-negative."
        )

    # ── Build Holding instances ──────────────────────────────────────────────
    names = raw[name_column].astype(str) if has_name else pd.Series([""] * len(raw))
    sectors = raw[sector_column].astype(str) if has_sector else pd.Series([""] * len(raw))
    if has_market_value:
        market_values = pd.to_numeric(raw[market_value_column], errors="coerce").fillna(0.0)
        negative_mv_rows = raw.index[market_values < 0.0].tolist()
        if negative_mv_rows:
            bad = {
                tickers.loc[i]: float(market_values.loc[i]) for i in negative_mv_rows
            }
            raise ValueError(
                f"{path.name!r} contains negative {market_value_column!r} "
                f"value(s): {bad!r}. Market values must be non-negative."
            )
    else:
        market_values = pd.Series([0.0] * len(raw))

    holdings_list = [
        Holding(
            ticker=tickers.iloc[i],
            name=names.iloc[i],
            weight=float(raw_weights.iloc[i]),
            sector=sectors.iloc[i],
            market_value=float(market_values.iloc[i]),
        )
        for i in range(len(raw))
    ]

    # ── Order by descending weight ────────────────────────────────────────────
    holdings_list.sort(key=lambda h: h.weight, reverse=True)
    holdings_tuple = tuple(holdings_list)

    # ── Cash weight and total weight validation ────────────────────────────
    sum_of_weights = sum(h.weight for h in holdings_tuple)
    cash_weight = max(0.0, 1.0 - sum_of_weights)
    total_weight = sum_of_weights + cash_weight

    if abs(total_weight - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(
            f"{path.name!r}: sum of holding weights ({sum_of_weights:.6f}) "
            f"plus computed cash_weight ({cash_weight:.6f}) = "
            f"{total_weight:.6f}, which deviates from 1.0 by more than "
            f"the tolerance ({_WEIGHT_SUM_TOLERANCE}). This indicates the "
            f"sum of holding weights exceeds 1.0, which is inconsistent "
            f"with weights expressed as a fraction of NAV. Check that "
            f"{weight_column!r} values are fractions (e.g. 0.0712), not "
            f"percentages (e.g. 7.12)."
        )

    total_market_value = float(market_values.sum())

    # ── as_of_date ────────────────────────────────────────────────────────────
    if as_of_date is None:
        mtime = path.stat().st_mtime
        as_of_date = pd.Timestamp.fromtimestamp(mtime).normalize()
        logger.warning(
            "as_of_date not provided for %r; using file modification date "
            "%s as a fallback.",
            etf_ticker,
            as_of_date.date(),
        )
    else:
        as_of_date = pd.Timestamp(as_of_date)

    logger.info(
        "Holdings snapshot loaded for %r: %d constituent(s), "
        "cash_weight=%.6f, as_of_date=%s, total_market_value=%.2f",
        etf_ticker,
        len(holdings_tuple),
        cash_weight,
        as_of_date.date(),
        total_market_value,
    )

    return HoldingsSnapshot(
        etf_ticker=etf_ticker,
        as_of_date=as_of_date,
        holdings=holdings_tuple,
        total_market_value=total_market_value,
        cash_weight=cash_weight,
    )


# ── Filtering ────────────────────────────────────────────────────────────────


def filter_top_n_holdings(
    snapshot: HoldingsSnapshot,
    n: int,
    min_weight: float = 0.0,
) -> HoldingsSnapshot:
    """Restrict a holdings snapshot to its top-N constituents by weight.

    For ETFs with many constituents, running ``alpha.py``'s full signal
    pipeline on every holding is both computationally expensive and
    statistically noisy: many small-weight constituents have momentum and
    mean-reversion signals dominated by idiosyncratic noise rather than
    systematic factors. This function returns a new
    :class:`HoldingsSnapshot` retaining only the largest ``n`` constituents
    (optionally also subject to ``min_weight``), with every excluded
    constituent's weight folded into ``cash_weight``.

    Parameters
    ----------
    snapshot:
        Source snapshot, as returned by :func:`load_holdings_snapshot`.
        ``snapshot.holdings`` is assumed to already be ordered by descending
        weight (the contract of :func:`load_holdings_snapshot`); this
        function does not re-sort.
    n:
        Number of largest constituents to retain. Must satisfy
        ``1 <= n <= len(snapshot.holdings)``.
    min_weight:
        Minimum weight a constituent must have to be retained, applied in
        addition to the top-``n`` restriction (a constituent is retained
        only if it is both within the top ``n`` by rank *and* has
        ``weight >= min_weight``). Must be non-negative. Default ``0.0``
        (no additional filter).

    Returns
    -------
    HoldingsSnapshot
        New snapshot with the same ``etf_ticker``, ``as_of_date``, and
        ``total_market_value`` as ``snapshot``, ``holdings`` restricted to
        the retained constituents (order preserved), and ``cash_weight``
        increased by the sum of every excluded constituent's weight (plus
        ``snapshot.cash_weight``).

    Raises
    ------
    ValueError
        If ``n < 1`` or ``n > len(snapshot.holdings)``; or if
        ``min_weight < 0``.

    Notes
    -----
    If ``min_weight`` excludes some constituents that would otherwise fall
    within the top ``n``, fewer than ``n`` constituents may be retained --
    this is not an error; the returned snapshot's ``holdings`` simply has
    length ``<= n``. The resulting ``cash_weight`` correctly reflects
    whatever was excluded by either criterion.

    Examples
    --------
    >>> top_10 = filter_top_n_holdings(snapshot, n=10)
    >>> len(top_10.holdings)
    10
    >>> top_10.cash_weight >= snapshot.cash_weight
    True
    """

    if n < 1:
        raise ValueError(f"n must be at least 1; got {n}.")
    if n > len(snapshot.holdings):
        raise ValueError(
            f"n ({n}) exceeds the number of holdings in the snapshot "
            f"({len(snapshot.holdings)})."
        )
    if min_weight < 0.0:
        raise ValueError(f"min_weight must be non-negative; got {min_weight}.")

    top_n = snapshot.holdings[:n]
    retained = tuple(h for h in top_n if h.weight >= min_weight)
    excluded = tuple(h for h in snapshot.holdings if h not in retained)

    excluded_weight = sum(h.weight for h in excluded)
    new_cash_weight = snapshot.cash_weight + excluded_weight

    logger.info(
        "Filtered %r holdings to top %d (min_weight=%.6f): retained %d, "
        "excluded %d, cash_weight %.6f -> %.6f.",
        snapshot.etf_ticker,
        n,
        min_weight,
        len(retained),
        len(excluded),
        snapshot.cash_weight,
        new_cash_weight,
    )

    return HoldingsSnapshot(
        etf_ticker=snapshot.etf_ticker,
        as_of_date=snapshot.as_of_date,
        holdings=retained,
        total_market_value=snapshot.total_market_value,
        cash_weight=new_cash_weight,
    )


# ── Universe construction ──────────────────────────────────────────────────────


def build_universe_config(
    snapshot: HoldingsSnapshot,
    price_data_dir: str | os.PathLike[str],
    *,
    file_extension: str = "xlsx",
) -> HoldingsUniverseConfig:
    """Convert a holdings snapshot into an alpha.py-compatible universe.

    For each constituent in ``snapshot.holdings``, checks whether a price
    file exists at ``{price_data_dir}/{ticker}.{file_extension}``. Tickers
    with an existing price file are included in the returned
    :class:`HoldingsUniverseConfig`'s ``asset_names``, ``filepaths``, and
    ``tickers``; tickers without one are excluded, with their weight folded
    into the implicit residual (see Notes).

    Parameters
    ----------
    snapshot:
        Source snapshot, as returned by :func:`load_holdings_snapshot` or
        :func:`filter_top_n_holdings`. Must contain at least one holding.
    price_data_dir:
        Directory in which per-ticker price workbooks are expected to be
        found, named ``{ticker}.{file_extension}``.
    file_extension:
        File extension (without leading dot) of the expected price
        workbooks. Default ``"xlsx"``, matching
        :func:`quant_platform.data.load_price_data`'s supported formats.

    Returns
    -------
    HoldingsUniverseConfig
        ``asset_names`` contains the tickers (in the same descending-weight
        order as ``snapshot.holdings``) for which a price file was found.
        ``filepaths[ticker]`` is the path to that file. ``tickers[ticker]``
        equals ``ticker`` (the convention that the price column within each
        ticker's workbook is named after the ticker itself --
        :func:`quant_platform.data.load_price_data` requires the column name
        passed as its ``ticker`` argument to match a column in the
        workbook; this convention is the contract
        :mod:`quant_platform.market_data` is expected to honour when writing
        these files). ``holdings_weights[ticker]`` is
        ``snapshot``'s weight for that ticker.

    Raises
    ------
    ValueError
        If ``snapshot.holdings`` is empty; or if **no** constituent has a
        price file present in ``price_data_dir`` (an empty
        ``asset_names`` would make ``alpha.build_asset_price_panel`` raise
        on an empty ``asset_names`` tuple -- this function raises earlier,
        with a message that names the directory and extension searched, so
        the diagnosis is immediate).

    Notes
    -----
    Excluded constituents' weights are **not** added to a
    ``HoldingsUniverseConfig.cash_weight`` field, because
    :class:`HoldingsUniverseConfig` has no such field (unlike
    :class:`HoldingsSnapshot`) -- ``HoldingsUniverseConfig`` is a thin
    data-loading configuration, not a portfolio-composition record. If a
    caller needs the post-exclusion residual weight (e.g. for display),
    it can be computed as
    ``1.0 - sum(universe.holdings_weights.values()) - snapshot.cash_weight``.
    This function logs, at ``WARNING``, every excluded ticker and its
    individual weight, so the information is not lost -- only not
    re-aggregated into a new field.

    Examples
    --------
    >>> universe = build_universe_config(top_10, price_data_dir="data/raw")
    >>> prices_panel = alpha.build_asset_price_panel(universe)
    """

    if len(snapshot.holdings) == 0:
        raise ValueError(
            f"snapshot for {snapshot.etf_ticker!r} has no holdings; cannot "
            f"build an asset universe from an empty snapshot."
        )

    price_dir = Path(price_data_dir)

    asset_names: list[str] = []
    filepaths: dict[str, str] = {}
    tickers: dict[str, str] = {}
    holdings_weights: dict[str, float] = {}

    excluded: list[tuple[str, float]] = []

    for holding in snapshot.holdings:
        candidate_path = price_dir / f"{holding.ticker}.{file_extension}"
        if candidate_path.exists():
            asset_names.append(holding.ticker)
            filepaths[holding.ticker] = str(candidate_path)
            tickers[holding.ticker] = holding.ticker
            holdings_weights[holding.ticker] = holding.weight
        else:
            excluded.append((holding.ticker, holding.weight))

    if excluded:
        for ticker, weight in excluded:
            logger.warning(
                "Excluding %r from asset universe for %r: no price file "
                "found at '%s' (weight=%.6f).",
                ticker,
                snapshot.etf_ticker,
                price_dir / f"{ticker}.{file_extension}",
                weight,
            )

    if not asset_names:
        raise ValueError(
            f"No price file found for any of the {len(snapshot.holdings)} "
            f"constituent(s) of {snapshot.etf_ticker!r} in directory "
            f"'{price_dir.resolve()}' with extension {file_extension!r}. "
            f"Expected files named '<TICKER>.{file_extension}' for tickers: "
            f"{[h.ticker for h in snapshot.holdings]!r}."
        )

    logger.info(
        "Universe config built for %r: %d/%d constituent(s) have price "
        "data available (%.2f%% of snapshot by count); %d excluded.",
        snapshot.etf_ticker,
        len(asset_names),
        len(snapshot.holdings),
        100.0 * len(asset_names) / len(snapshot.holdings),
        len(excluded),
    )

    return HoldingsUniverseConfig(
        asset_names=tuple(asset_names),
        filepaths=filepaths,
        tickers=tickers,
        holdings_weights=holdings_weights,
    )


# ── Usage example ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # End-to-end usage example. Replace the filepath, etf_ticker, and
    # price_data_dir below with real values for the ETF being analysed.
    #
    #   python -m quant_platform.holdings
    #
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

    example_holdings_path = "data/holdings/example_etf_holdings.csv"
    example_etf_ticker = "EXAMPLE"
    example_price_data_dir = "data/raw"

    print(f"Loading holdings snapshot from {example_holdings_path!r} ...")
    try:
        snapshot = load_holdings_snapshot(
            example_holdings_path,
            etf_ticker=example_etf_ticker,
        )
    except FileNotFoundError as exc:
        print(f"[FileNotFoundError] {exc}")
        print(
            "\nThis is an example invocation -- place a real ETF holdings "
            "CSV at the path above (or edit example_holdings_path) to run "
            "this example end to end."
        )
        raise SystemExit(0)

    print(f"\nSnapshot for {snapshot.etf_ticker} as of {snapshot.as_of_date.date()}:")
    print(f"  Constituents : {len(snapshot.holdings)}")
    print(f"  Cash weight  : {snapshot.cash_weight:.4%}")
    print(f"  Total MV     : {snapshot.total_market_value:,.2f}")

    print("\nTop 5 holdings:")
    for holding in snapshot.holdings[:5]:
        print(f"  {holding.ticker:<8} {holding.weight:.4%}  {holding.name}  [{holding.sector}]")

    top_10 = filter_top_n_holdings(snapshot, n=min(10, len(snapshot.holdings)))
    print(f"\nFiltered to top {len(top_10.holdings)} holdings; cash weight now {top_10.cash_weight:.4%}.")

    universe = build_universe_config(top_10, price_data_dir=example_price_data_dir)
    print(f"\nUniverse config: {len(universe.asset_names)} asset(s) with available price data:")
    for ticker in universe.asset_names:
        print(
            f"  {ticker:<8} weight={universe.holdings_weights[ticker]:.4%}  "
            f"filepath={universe.filepaths[ticker]}"
        )