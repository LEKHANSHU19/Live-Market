"""Pure presentation helpers for the quant platform.

This module only formats precomputed engine outputs into tables, figures, and
summary structures.  It does not price options, compute risk measures, optimize
portfolios, run simulations, or generate scenarios.

Examples
--------
>>> table = format_pricing_results({"European Call": 10.5, "European Put": 5.6})
>>> list(table.columns)
['Metric', 'Value']

>>> summary = generate_summary_report(portfolio={"volatility": 0.058})
>>> "Portfolio Summary" in summary
True
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from quant_platform.config import PLOT


def _to_dataframe(results: Any, section_name: str = "Result") -> pd.DataFrame:
    """Convert common precomputed output structures into a clean DataFrame."""

    if isinstance(results, pd.DataFrame):
        return results.copy()

    if isinstance(results, pd.Series):
        return results.rename(section_name).reset_index()

    if isinstance(results, dict):
        rows: list[dict[str, Any]] = []
        for key, value in results.items():
            if isinstance(value, dict):
                row = {"Metric": key, **value}
            elif isinstance(value, pd.Series):
                row = {"Metric": key, "Value": value}
            elif isinstance(value, np.ndarray):
                row = {"Metric": key, "Value": value}
            else:
                row = {"Metric": key, "Value": value}
            rows.append(row)
        return pd.DataFrame(rows)

    if isinstance(results, (list, tuple, np.ndarray)):
        return pd.DataFrame(results)

    return pd.DataFrame([{"Metric": section_name, "Value": results}])


def _format_scalar(value: Any, decimals: int) -> Any:
    """Round scalar numeric values while preserving arrays, Series, and text."""

    if isinstance(value, (pd.Series, pd.DataFrame, np.ndarray, list, tuple, dict)):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        return round(float(value), decimals)
    return value


def _format_numeric_columns(table: pd.DataFrame, decimals: int) -> pd.DataFrame:
    """Return a copy with numeric columns rounded for display."""

    formatted = table.copy()
    for column in formatted.columns:
        if pd.api.types.is_numeric_dtype(formatted[column]):
            formatted[column] = formatted[column].round(decimals)
        else:
            formatted[column] = formatted[column].map(lambda value: _format_scalar(value, decimals))
    return formatted


def format_pricing_results(results: Any, decimals: int = 6) -> pd.DataFrame:
    """Format precomputed pricing results into a table.

    Parameters
    ----------
    results:
        Precomputed pricing output, such as a dict from pricing functions or a
        DataFrame assembled by the caller.
    decimals:
        Number of decimal places used for scalar numeric display values.
    """

    return _format_numeric_columns(_to_dataframe(results, "Pricing"), decimals)


def format_var_results(results: Any, decimals: int = 6) -> pd.DataFrame:
    """Format precomputed VaR or backtest results into a table."""

    return _format_numeric_columns(_to_dataframe(results, "VaR"), decimals)


def format_portfolio_results(results: Any, decimals: int = 6) -> pd.DataFrame:
    """Format precomputed portfolio analytics output into a table."""

    return _format_numeric_columns(_to_dataframe(results, "Portfolio"), decimals)


def format_scenario_results(results: Any, decimals: int = 6) -> dict[str, pd.DataFrame]:
    """Format precomputed scenario-analysis output into sectioned tables."""

    if not isinstance(results, dict):
        return {"Scenario Results": _format_numeric_columns(_to_dataframe(results, "Scenario"), decimals)}

    tables: dict[str, pd.DataFrame] = {}
    for section, value in results.items():
        tables[str(section)] = _format_numeric_columns(_to_dataframe(value, str(section)), decimals)
    return tables


def plot_var_backtest(
    comparison: pd.DataFrame,
    title: str = "VaR Backtest",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot precomputed VaR and forward returns with breach markers."""

    required = {"VaR", "Fwd", "Breach"}
    if not isinstance(comparison, pd.DataFrame):
        raise TypeError("comparison must be a pd.DataFrame")
    if not required.issubset(comparison.columns):
        raise ValueError("comparison must contain 'VaR', 'Fwd', and 'Breach' columns")

    if ax is None:
        fig, ax = plt.subplots(figsize=PLOT.BACKTEST_FIGSIZE)
    else:
        fig = ax.figure

    ax.plot(comparison.index, comparison["VaR"], color=PLOT.COLOR_ROLLING_VAR, lw=1.5, label="VaR")
    ax.plot(comparison.index, comparison["Fwd"], color=PLOT.COLOR_ACTUAL_RET, lw=0.8, alpha=0.75, label="Forward return")

    breaches = comparison[comparison["Breach"].astype(bool)]
    if not breaches.empty:
        ax.scatter(
            breaches.index,
            breaches["Fwd"],
            marker=PLOT.BREACH_MARKER,
            color=PLOT.COLOR_BREACH,
            s=PLOT.BREACH_MARKER_SIZE,
            lw=PLOT.BREACH_MARKER_LW,
            label="Breach",
        )

    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_title(title)
    ax.set_ylabel("Return / VaR")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_convergence(
    results: pd.DataFrame,
    x: str = "Steps",
    y: str = "Abs error",
    group: str = "Scheme",
    title: str = "Convergence",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot precomputed convergence results from a table."""

    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pd.DataFrame")
    for column in (x, y, group):
        if column not in results.columns:
            raise ValueError(f"results must contain column {column!r}")

    if ax is None:
        fig, ax = plt.subplots(figsize=PLOT.CONVERGENCE_FIGSIZE)
    else:
        fig = ax.figure

    for label, subset in results.groupby(group):
        ax.plot(subset[x], subset[y], marker="o", lw=1.5, label=str(label))

    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_sensitivity(
    results: pd.DataFrame,
    x: str = "Value",
    value_columns: list[str] | None = None,
    title: str = "Sensitivity Analysis",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot precomputed sensitivity columns against a selected x-axis."""

    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pd.DataFrame")
    if x not in results.columns:
        raise ValueError(f"results must contain column {x!r}")

    columns = value_columns or [
        column for column in results.columns if column != x and pd.api.types.is_numeric_dtype(results[column])
    ]
    if not columns:
        raise ValueError("no numeric value columns are available to plot")

    if ax is None:
        fig, ax = plt.subplots(figsize=PLOT.SENSITIVITY_FIGSIZE)
    else:
        fig = ax.figure

    for column in columns:
        ax.plot(results[x], results[column], lw=1.5, label=column)

    ax.set_title(title)
    ax.set_xlabel(x)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_scenario_analysis(
    scenario_results: dict[str, Any] | pd.DataFrame,
    section: str = "difference",
    title: str = "Scenario Analysis",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot a precomputed scenario section as a bar chart when possible."""

    if isinstance(scenario_results, dict) and section in scenario_results:
        table = _to_dataframe(scenario_results[section], section)
    else:
        table = _to_dataframe(scenario_results, "Scenario")

    if "Metric" not in table.columns:
        table = table.reset_index().rename(columns={"index": "Metric"})

    numeric_columns = [column for column in table.columns if pd.api.types.is_numeric_dtype(table[column])]
    if not numeric_columns:
        raise ValueError("scenario results do not contain scalar numeric columns to plot")

    if ax is None:
        fig, ax = plt.subplots(figsize=PLOT.AV_FIGSIZE)
    else:
        fig = ax.figure

    table.plot(kind="bar", x="Metric", y=numeric_columns, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def generate_summary_report(
    portfolio: Any | None = None,
    risk: Any | None = None,
    pricing: Any | None = None,
    scenarios: Any | None = None,
) -> dict[str, Any]:
    """Assemble supplied precomputed outputs into a structured summary dictionary."""

    summary: dict[str, Any] = {}
    if portfolio is not None:
        summary["Portfolio Summary"] = portfolio
    if risk is not None:
        summary["Risk Summary"] = risk
    if pricing is not None:
        summary["Pricing Summary"] = pricing
    if scenarios is not None:
        summary["Scenario Summary"] = scenarios
    return summary

