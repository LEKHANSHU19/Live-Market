from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from quant_platform.alpha import AssetUniverseConfig

try:
    from quant_platform.config import UNIVERSE
except ImportError:

    class UNIVERSE:
        MIN_HISTORY_DAYS = 252
        MAX_MISSING_PCT = 0.05
        TOP_N = 20


logger = logging.getLogger("quant_platform.universe")


class UniverseBuilder:
    """
    Investable universe construction engine.

    Responsibilities
    ----------------
    - Static universe definitions
    - Data quality filtering
    - Historical coverage validation
    - Shared-history survivorship validation
    - Liquidity ranking
    - Investable universe generation
    """

    NIFTY50: tuple[str, ...] = (
        "RELIANCE",
        "HDFCBANK",
        "ICICIBANK",
        "INFY",
        "TCS",
        "LT",
        "SBIN",
        "BHARTIARTL",
        "ITC",
        "AXISBANK",
        "KOTAKBANK",
        "HINDUNILVR",
        "BAJFINANCE",
        "MARUTI",
        "SUNPHARMA",
        "NTPC",
        "POWERGRID",
        "ULTRACEMCO",
        "TITAN",
        "ADANIENT",
    )

    SP500: tuple[str, ...] = (
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "META",
        "GOOGL",
        "GOOG",
        "BRK.B",
        "JPM",
        "V",
        "MA",
        "XOM",
        "UNH",
        "LLY",
        "AVGO",
        "COST",
        "HD",
        "WMT",
        "PG",
        "NFLX",
    )

    @classmethod
    def get_nifty50(cls) -> list[str]:
        logger.info(
            "Loaded NIFTY50 blueprint universe (%d assets).",
            len(cls.NIFTY50),
        )
        return list(cls.NIFTY50)

    @classmethod
    def get_sp500(cls) -> list[str]:
        logger.info(
            "Loaded SP500 blueprint universe (%d assets).",
            len(cls.SP500),
        )
        return list(cls.SP500)
@classmethod
def build_nifty50_config(
    cls,
    data_directory: str = "data/raw",
) -> AssetUniverseConfig:

    asset_names = tuple(cls.NIFTY50)

    filepaths = {
        asset: f"{data_directory}/{asset}.xlsx"
        for asset in asset_names
    }

    tickers = {
        asset: asset
        for asset in asset_names
    }

    return AssetUniverseConfig(
        asset_names=asset_names,
        filepaths=filepaths,
        tickers=tickers,
    )


@classmethod
def build_sp500_config(
    cls,
    data_directory: str = "data/raw",
) -> AssetUniverseConfig:

    asset_names = tuple(cls.SP500)

    filepaths = {
        asset: f"{data_directory}/{asset}.xlsx"
        for asset in asset_names
    }

    tickers = {
        asset: asset
        for asset in asset_names
    }

    return AssetUniverseConfig(
        asset_names=asset_names,
        filepaths=filepaths,
        tickers=tickers,
    )
    @staticmethod
    def _validate_price_panel(prices_df: pd.DataFrame) -> None:
        if not isinstance(prices_df, pd.DataFrame):
            raise TypeError(
                "prices_df must be a pandas DataFrame."
            )

        if prices_df.empty:
            raise ValueError(
                "prices_df is empty."
            )

        if not isinstance(prices_df.index, pd.DatetimeIndex):
            raise TypeError(
                "prices_df must use a DatetimeIndex."
            )

        if prices_df.columns.duplicated().any():
            duplicated = (
                prices_df.columns[
                    prices_df.columns.duplicated()
                ]
                .tolist()
            )
            raise ValueError(
                f"Duplicate tickers detected: {duplicated}"
            )

        values = prices_df.to_numpy(dtype=float)

        if np.isinf(values).any():
            raise ValueError(
                "prices_df contains infinite values."
            )

    @staticmethod
    def missing_data_filter(
        prices_df: pd.DataFrame,
        max_missing_pct: float | None = None,
    ) -> list[str]:

        UniverseBuilder._validate_price_panel(prices_df)

        if max_missing_pct is None:
            max_missing_pct = UNIVERSE.MAX_MISSING_PCT

        if not 0.0 <= max_missing_pct <= 1.0:
            raise ValueError(
                "max_missing_pct must be between 0 and 1."
            )

        starting_assets = prices_df.shape[1]

        missing_pct = prices_df.isna().mean(axis=0)

        survivors = (
            missing_pct[
                missing_pct <= max_missing_pct
            ]
            .index
            .tolist()
        )

        dropped = sorted(
            set(prices_df.columns) - set(survivors)
        )

        logger.info(
            "[Missing Data Filter] Started=%d | Dropped=%d | Survived=%d",
            starting_assets,
            len(dropped),
            len(survivors),
        )

        if dropped:
            logger.warning(
                "[Missing Data Filter] Dropped assets: %s",
                dropped,
            )

        if len(survivors) == 0:
            raise ValueError(
                "Universe construction failed. "
                "All assets removed by Missing Data Filter."
            )

        return survivors

    @staticmethod
    def history_filter(
        prices_df: pd.DataFrame,
        min_days: int | None = None,
    ) -> list[str]:

        UniverseBuilder._validate_price_panel(prices_df)

        if min_days is None:
            min_days = UNIVERSE.MIN_HISTORY_DAYS

        if min_days <= 0:
            raise ValueError(
                "min_days must be positive."
            )

        starting_assets = prices_df.shape[1]

        history_counts = prices_df.notna().sum(axis=0)

        survivors = (
            history_counts[
                history_counts >= min_days
            ]
            .index
            .tolist()
        )

        dropped = sorted(
            set(prices_df.columns) - set(survivors)
        )

        logger.info(
            "[History Filter] Started=%d | Dropped=%d | Survived=%d",
            starting_assets,
            len(dropped),
            len(survivors),
        )

        if dropped:
            logger.warning(
                "[History Filter] Dropped assets: %s",
                dropped,
            )

        if len(survivors) == 0:
            raise ValueError(
                "Universe construction failed. "
                "All assets removed by History Filter."
            )

        shared_panel = prices_df[survivors].dropna(
            how="any"
        )

        effective_shared_history = shared_panel.shape[0]

        minimum_individual_history = int(
            history_counts.loc[survivors].min()
        )

        logger.info(
            "[Shared History Check] "
            "Minimum Individual History=%d | "
            "Effective Shared History=%d",
            minimum_individual_history,
            effective_shared_history,
        )

        if minimum_individual_history > 0:

            retention_ratio = (
                effective_shared_history
                / minimum_individual_history
            )

            if retention_ratio < 0.70:
                logger.warning(
                    "[Shared History Fragmentation] "
                    "Shared history retention %.2f%%. "
                    "Potential severe temporal fragmentation.",
                    retention_ratio * 100.0,
                )

        if effective_shared_history < min_days:
            raise ValueError(
                "Asset intersection history is insufficient "
                "for stable covariance matrix estimation. "
                f"Required >= {min_days} observations, "
                f"found {effective_shared_history}."
            )

        return survivors

    @staticmethod
    def liquidity_filter(
        volumes_df: pd.DataFrame,
        top_n: int | None = None,
    ) -> list[str]:

        if not isinstance(volumes_df, pd.DataFrame):
            raise TypeError(
                "volumes_df must be a pandas DataFrame."
            )

        if volumes_df.empty:
            raise ValueError(
                "volumes_df is empty."
            )

        if top_n is None:
            top_n = UNIVERSE.TOP_N

        if top_n <= 0:
            raise ValueError(
                "top_n must be positive."
            )

        starting_assets = volumes_df.shape[1]

        coverage = volumes_df.notna().mean(axis=0)

        valid_columns = coverage[
            coverage >= 0.90
        ].index.tolist()

        filtered_volume = volumes_df[valid_columns]

        adv = filtered_volume.mean(
            axis=0,
            skipna=True,
        )

        adv = adv.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        ranked = adv.sort_values(
            ascending=False
        )

        survivors = ranked.head(
            min(top_n, len(ranked))
        ).index.tolist()

        dropped = sorted(
            set(volumes_df.columns) - set(survivors)
        )

        logger.info(
            "[Liquidity Filter] Started=%d | Dropped=%d | Survived=%d",
            starting_assets,
            len(dropped),
            len(survivors),
        )

        if dropped:
            logger.warning(
                "[Liquidity Filter] Dropped assets: %s",
                dropped,
            )

        if len(survivors) == 0:
            raise ValueError(
                "Universe construction failed. "
                "All assets removed by Liquidity Filter."
            )

        logger.info(
            "[Liquidity Filter] Top ADV Assets: %s",
            survivors,
        )

        return survivors

    @classmethod
    def build_investable_universe(
        cls,
        prices_df: pd.DataFrame,
        volumes_df: pd.DataFrame | None = None,
        min_days: int | None = None,
        max_missing_pct: float | None = None,
        top_n: int | None = None,
    ) -> list[str]:

        cls._validate_price_panel(prices_df)

        logger.info(
            "=================================================="
        )
        logger.info(
            "STARTING UNIVERSE CONSTRUCTION"
        )
        logger.info(
            "Initial Asset Count: %d",
            prices_df.shape[1],
        )

        survivors = cls.missing_data_filter(
            prices_df=prices_df,
            max_missing_pct=max_missing_pct,
        )

        prices_df = prices_df[survivors]

        if len(prices_df.columns) == 0:
            raise ValueError(
                "No assets remain after Missing Data Filter."
            )

        survivors = cls.history_filter(
            prices_df=prices_df,
            min_days=min_days,
        )

        prices_df = prices_df[survivors]

        if len(prices_df.columns) == 0:
            raise ValueError(
                "No assets remain after History Filter."
            )

        if volumes_df is not None:

            if not isinstance(
                volumes_df,
                pd.DataFrame,
            ):
                raise TypeError(
                    "volumes_df must be a pandas DataFrame."
                )

            common_assets = [
                col
                for col in prices_df.columns
                if col in volumes_df.columns
            ]

            if len(common_assets) == 0:
                raise ValueError(
                    "No overlap between price and volume panels."
                )

            common_dates = (
                prices_df.index.intersection(
                    volumes_df.index
                )
            )

            if len(common_dates) == 0:
                raise ValueError(
                    "No overlapping dates between "
                    "price and volume panels."
                )

            aligned_prices = prices_df.loc[
                common_dates,
                common_assets,
            ]

            aligned_volumes = volumes_df.loc[
                common_dates,
                common_assets,
            ]

            survivors = cls.liquidity_filter(
                volumes_df=aligned_volumes,
                top_n=top_n,
            )

            aligned_prices = aligned_prices[
                survivors
            ]

            shared_panel = aligned_prices.dropna(
                how="any"
            )

            effective_shared_history = (
                shared_panel.shape[0]
            )

            required_days = (
                min_days
                if min_days is not None
                else UNIVERSE.MIN_HISTORY_DAYS
            )

            if effective_shared_history < required_days:
                raise ValueError(
                    "Final asset set fails shared-history "
                    "requirement after liquidity filtering. "
                    f"Required >= {required_days}, "
                    f"found {effective_shared_history}."
                )

        else:

            logger.warning(
                "Volume panel not supplied. "
                "Liquidity filter skipped."
            )

            survivors = list(
                prices_df.columns
            )

        if len(survivors) < 10:
            logger.warning(
                "Universe concentration warning. "
                "Only %d assets remain.",
                len(survivors),
            )

        logger.info(
            "FINAL INVESTABLE UNIVERSE (%d assets)",
            len(survivors),
        )

        logger.info(
            "%s",
            survivors,
        )

        logger.info(
            "=================================================="
        )

        return survivors


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    print(
        "NIFTY50:",
        UniverseBuilder.get_nifty50(),
    )

    print(
        "SP500:",
        UniverseBuilder.get_sp500(),
    )