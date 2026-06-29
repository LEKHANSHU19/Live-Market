"""
config.py — Quantitative Risk & Derivatives Analytics Platform
==============================================================
Single source of truth for every hardcoded parameter used across the
platform.  All other modules import from here; nothing is hardcoded
downstream.

Usage
-----
    from config import MC, VAR, BACKTEST, MARKET, PLOT

Parameter groups
----------------
    MARKET    — underlying asset and option contract parameters
    MC        — Monte Carlo simulation controls
    VAR       — Value-at-Risk and Expected Shortfall settings
    BACKTEST  — rolling / EWMA backtesting window and traffic-light thresholds
    PLOT      — figure size, DPI, output filenames
"""

from __future__ import annotations


# ── 1. MARKET PARAMETERS ──────────────────────────────────────────────────────
# Source: E1 Cells 2–3 (portfolio / VaR-ES problem sets)
#         E2 Cell 2    (Monte Carlo base case)

class MARKET:
    # ── Option / GBM base case (E2 Cell 2) ──────────────────────────────────
    S0: float    = 100.0   # Initial stock price ($)
    K: float     = 100.0   # Strike price — at-the-money base case ($)
    T: float     = 1.0     # Time to expiry (years)
    SIGMA: float = 0.20    # Annualised volatility (20 %)
    R: float     = 0.05    # Continuously compounded risk-free rate (5 % p.a.)

    # ── Four-asset MVO problem (E1 Cell 2) ───────────────────────────────────
    # Expected returns vector  μ  (annualised)
    MVO_MU: tuple[float, ...] = (0.02, 0.07, 0.15, 0.20)

    # Asset volatilities  σ  (annualised)
    MVO_SIGMA: tuple[float, ...] = (0.05, 0.12, 0.17, 0.25)

    # Correlation matrix  ρ  (4 × 4, stored row-major)
    MVO_RHO: tuple[tuple[float, ...], ...] = (
        (1.00, 0.30, 0.30, 0.30),
        (0.30, 1.00, 0.60, 0.60),
        (0.30, 0.60, 1.00, 0.60),
        (0.30, 0.60, 0.60, 1.00),
    )

    # Target portfolio return for the MVO efficient-frontier query (E1 Cell 2)
    MVO_TARGET_RETURN: float = 0.045   # 4.5 % p.a.

    # ── Three-asset VaR / ES sensitivity problem (E1 Cell 3) ─────────────────
    # Portfolio weights  w
    VES_WEIGHTS: tuple[float, ...] = (0.50, 0.20, 0.30)

    # Individual asset volatilities  σ  (annualised)
    VES_SIGMA: tuple[float, ...] = (0.30, 0.20, 0.15)

    # Zero-mean assumption for the sensitivity derivation
    VES_MU: tuple[float, ...] = (0.0, 0.0, 0.0)

    # Correlation matrix  ρ  (3 × 3, stored row-major)
    VES_RHO: tuple[tuple[float, ...], ...] = (
        (1.0, 0.8, 0.5),
        (0.8, 1.0, 0.3),
        (0.5, 0.3, 1.0),
    )

    # Asset labels used in MVO output tables
    MVO_ASSET_NAMES: tuple[str, ...] = ("A", "B", "C", "D")

    # Asset labels used in VaR / ES output tables
    VES_ASSET_NAMES: tuple[str, ...] = ("Asset 1", "Asset 2", "Asset 3")

    # S&P 500 ticker column name as it appears in Indices_Download_2026.xlsx
    SP500_TICKER: str = "^GSPC"

    # Path to the S&P 500 Excel data file (override at runtime via CLI or env)
    DATA_FILEPATH: str = "data/raw/Indices_Download_2026.xlsx"


# ── 2. MONTE CARLO PARAMETERS ─────────────────────────────────────────────────
# Source: E2 Cells 2 and 4

class MC:
    # Number of simulation paths (E2 Cell 2)
    N_PATHS: int = 100_000

    # Time-step grid used in the weak-convergence study (E2 Cell 4)
    # Each entry is the number of Euler / Milstein steps per year
    STEPS_LIST: tuple[int, ...] = (1, 4, 12, 52, 252)

    # Simulation schemes to evaluate (E2 Cell 2)
    SCHEMES: tuple[str, ...] = ("Euler", "Milstein", "Exact")

    # Option types to price (E2 Cell 2)
    OPTION_NAMES: tuple[str, ...] = (
        "European Call",
        "European Put",
        "Binary Call",
        "Binary Put",
    )

    # Global random seed for reproducibility (E2 Cell 2)
    # Passed explicitly to np.random.default_rng(); never set as global state
    SEED: int = 42

    # Number of antithetic-variate pairs = N_PATHS // 2 (E2 Cell 9)
    # Kept as a derived constant so the relationship is explicit
    N_AV_PAIRS: int = N_PATHS // 2

    # Sample-size grid for the AV convergence study (E2 Cell 10)
    AV_SIZES: tuple[int, ...] = (1_000, 5_000, 10_000, 25_000, 50_000, 100_000)

    # Number of paths shown in the GBM path-visualisation panel (E2 Cell 7)
    N_PATHS_SHOW: int = 25

    # Number of time steps used for the path-visualisation panel (E2 Cell 7)
    STEPS_VIS: int = 252

    # Parameter grid sizes for the sensitivity sweep (E2 Cell 8)
    SENSITIVITY_N_POINTS: int = 12   # points per continuous parameter
    SENSITIVITY_K_POINTS: int = 13   # points for the strike / S0 grids (odd → ATM included)

    # Sensitivity sweep ranges (E2 Cell 8)
    SENS_SIGMA_RANGE: tuple[float, float] = (0.05, 0.60)   # volatility
    SENS_R_RANGE:     tuple[float, float] = (0.00, 0.15)   # risk-free rate
    SENS_K_RANGE:     tuple[float, float] = (70.0, 130.0)  # strike
    SENS_S0_RANGE:    tuple[float, float] = (70.0, 130.0)  # initial price
    SENS_T_RANGE:     tuple[float, float] = (0.25,  3.00)  # time to expiry


# ── 3. VAR PARAMETERS ─────────────────────────────────────────────────────────
# Source: E1 Cells 3, 4, 5, 7

class VAR:
    # Confidence level for VaR and ES (99 %) — used in both E1 Cell 3 and Cell 5
    CONFIDENCE: float = 0.99

    # Corresponding tail probability  α = 1 − confidence
    ALPHA: float = 1.0 - CONFIDENCE   # 0.01

    # Horizon for the 10-day regulatory VaR (trading days) — E1 Cells 5 and 7
    HORIZON_DAYS: int = 10

    # Rolling window for historical volatility estimation (trading days) — E1 Cell 5
    ROLLING_WINDOW: int = 21

    # EWMA decay factor  λ — RiskMetrics daily calibration (E1 Cell 7)
    LAMBDA_EWMA: float = 0.72

    # Minimum observations required before the rolling window produces a valid σ
    # (set equal to ROLLING_WINDOW so the first estimate uses a full window)
    ROLLING_MIN_PERIODS: int = ROLLING_WINDOW


# ── 4. BACKTEST PARAMETERS ────────────────────────────────────────────────────
# Source: E1 Cells 4, 5, 7

class BACKTEST:
    # In-sample period: full history loaded from Excel
    # (the actual start date is determined by the data file)

    # Out-of-sample backtesting window — E1 Cell 4
    START: str = "2025-01-02"
    END:   str = "2026-01-15"

    # Basel traffic-light binomial quantile thresholds (E1 Cell 5)
    # GREEN  zone: breach count ≤ binom.ppf(0.95,   T, 0.01)
    # YELLOW zone: breach count ≤ binom.ppf(0.9999, T, 0.01)
    # RED    zone: breach count >  YELLOW threshold
    TRAFFIC_GREEN_PROB:  float = 0.95
    TRAFFIC_YELLOW_PROB: float = 0.9999

    # Null-hypothesis breach probability (H₀: model is correctly specified)
    H0_BREACH_PROB: float = 0.01   # = 1 % for a 99 % VaR model

    # Zone labels — kept here so reporting.py and tests reference the same strings
    ZONE_GREEN:  str = "GREEN"
    ZONE_YELLOW: str = "YELLOW"
    ZONE_RED:    str = "RED"


# ── 5. PLOT PARAMETERS ────────────────────────────────────────────────────────
# Source: E1 Cells 6 and 8; E2 Cells 4, 7, 8, 9, 10

class PLOT:
    # Resolution for saved PNG figures (E1 Cells 6, 8)
    DPI: int = 150

    # Figure size for the dual-panel VaR backtest charts (E1 Cells 6, 8)
    BACKTEST_FIGSIZE: tuple[float, float] = (14.0, 12.0)

    # Height ratios for the three-panel backtest layout [VaR panel, return panel, price panel]
    BACKTEST_HEIGHT_RATIOS: tuple[float, ...] = (2.5, 2.5, 1.0)

    # Figure size for the MC convergence plot (E2 Cell 5)
    CONVERGENCE_FIGSIZE: tuple[float, float] = (13.0, 5.0)

    # Figure size for the GBM path visualisation (E2 Cell 7)
    PATHS_FIGSIZE: tuple[float, float] = (15.0, 4.0)

    # Figure size for the parameter sensitivity grid (E2 Cell 8)
    SENSITIVITY_FIGSIZE: tuple[float, float] = (16.0, 9.0)

    # Figure size for the antithetic-variate convergence plot (E2 Cell 10)
    AV_FIGSIZE: tuple[float, float] = (13.0, 5.0)

    # Global matplotlib DPI for interactive display (E2 Cell 2)
    DISPLAY_DPI: int = 120

    # Colour scheme for the backtest charts
    COLOR_ROLLING_VAR:  str = "navy"       # Rolling 21D VaR line (E1 Cell 6)
    COLOR_EWMA_VAR:     str = "purple"     # EWMA VaR line         (E1 Cell 8)
    COLOR_ACTUAL_RET:   str = "steelblue"  # Actual return series  (E1 Cells 6, 8)
    COLOR_BREACH:       str = "red"        # Breach marker ×       (E1 Cells 6, 8)
    COLOR_ROLLING_FILL: str = "lightcoral" # Rolling negative-return fill (E1 Cell 6)
    COLOR_EWMA_FILL:    str = "plum"       # EWMA negative-return fill    (E1 Cell 8)
    COLOR_SP500:        str = "darkgreen"  # S&P 500 price panel          (E1 Cell 6)

    # Colour scheme for the MC convergence / AV plots (E2 Cells 5, 10)
    COLOR_EULER:       str = "#1f77b4"  # Euler–Maruyama
    COLOR_MILSTEIN:    str = "#2ca02c"  # Milstein
    COLOR_EXACT:       str = "#d62728"  # Exact GBM
    COLOR_CRUDE_MC:    str = "#1f77b4"  # Crude MC (AV comparison)
    COLOR_ANTITHETIC:  str = "#2ca02c"  # Antithetic variates

    # Breach marker style (E1 Cells 6, 8)
    BREACH_MARKER:      str   = "x"
    BREACH_MARKER_SIZE: float = 130.0
    BREACH_MARKER_LW:   float = 2.5

    # Output filenames for saved figures
    OUT_TASK3:       str = "task3_rolling_var_backtest.png"
    OUT_TASK4:       str = "task4_ewma_var_backtest.png"
    OUT_CONVERGENCE: str = "mc_convergence.png"
    OUT_PATHS:       str = "gbm_paths.png"
    OUT_SENSITIVITY: str = "option_sensitivity.png"
    OUT_AV:          str = "antithetic_variance_reduction.png"
