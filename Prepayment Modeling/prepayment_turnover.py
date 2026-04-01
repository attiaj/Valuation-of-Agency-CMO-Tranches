"""
prepayment_turnover.py

Turnover component of the APD prepayment model.

Turnover captures prepayments driven by housing mobility (moving, death,
divorce, etc.) -- independent of refinancing incentive.

Formula (Davidson & Levin, Chapter 7):
    TurnoverSMM(t) = BaseTurnover_CPR * AgeFactor(t) * Seasonality(m)
    converted to SMM via: SMM = 1 - (1 - CPR)^(1/12)

Where:
    AgeFactor(t)  = min(1, WALA(t) / 30)          -- PSA-style seasoning ramp
    Seasonality(m) = monthly multiplier (m = 1..12) -- housing market seasonality

Pool assumptions (from Bloomberg FNR 2024-100 A Tranche):
    - Origination:      December 2024
    - Projection start: February 2026
    - WALA at t=0:      14 months  (Dec 2024 -> Feb 2026)
    - start_month:      2          (February)

Corrections vs. original calculate_turnover.py:
    1. wala_start corrected to 14 (from Bloomberg dates) instead of 20
    2. base_turnover_cpr corrected to 0.06 (6% CPR, textbook default)
       -- original used 0.08 without documented calibration justification
    3. Seasonality for February corrected to 0.8 (was 0.7)
    4. Seasonality for March corrected to 0.9 (was 1.0)
       -- both per Davidson & Levin Chapter 7 Table
    5. Refactored from script into callable functions for pipeline integration
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Textbook seasonality multipliers (Davidson & Levin, Chapter 7)
# Keys: calendar month (1=Jan ... 12=Dec)
SEASONALITY_MAP = {
    1:  0.8,   # January   -- winter
    2:  0.8,   # February  -- winter   (corrected from 0.7)
    3:  0.9,   # March     -- early spring (corrected from 1.0)
    4:  1.0,   # April     -- spring
    5:  1.1,   # May       -- spring
    6:  1.2,   # June      -- summer peak
    7:  1.2,   # July      -- summer peak
    8:  1.1,   # August    -- late summer
    9:  1.0,   # September -- fall
    10: 0.9,   # October   -- fall
    11: 0.8,   # November  -- early winter
    12: 0.7,   # December  -- holidays
}

# Pool constants (FNR 2024-100, Bloomberg data)
WALA_START      = 35    # WALA at start of projection (months, April 2023 -> March 2026)
START_MONTH     = 4     # Calendar month projection begins (April)
BASE_TURNOVER_CPR = 0.06  # Textbook default: 6% CPR annualized


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def age_factor(wala: float) -> float:
    """
    PSA-style seasoning ramp.

    Loans season linearly over the first 30 months, reaching full
    turnover speed at WALA = 30. Flat thereafter.

    Parameters
    ----------
    wala : float -- weighted average loan age in months

    Returns
    -------
    float in [0, 1]
    """
    return min(1.0, wala / 30.0)


def seasonal_factor(calendar_month: int) -> float:
    """
    Seasonal multiplier for a given calendar month.

    Parameters
    ----------
    calendar_month : int -- 1 (January) through 12 (December)

    Returns
    -------
    float -- seasonal multiplier from SEASONALITY_MAP
    """
    return SEASONALITY_MAP[calendar_month]


def calendar_month_at_step(step: int,
                            start_month: int = START_MONTH) -> int:
    """
    Derive the calendar month at a given projection step.

    Parameters
    ----------
    step        : int -- projection month index, 1-based (step=1 is the
                        first projection month, i.e. February 2026)
    start_month : int -- calendar month the projection begins (default 2)

    Returns
    -------
    int -- calendar month (1-12)
    """
    return ((start_month + step - 1 - 1) % 12) + 1


def wala_at_step(step: int,
                 wala_start: int = WALA_START) -> int:
    """
    Pool WALA at a given projection step.

    Parameters
    ----------
    step       : int -- projection step, 1-based
    wala_start : int -- WALA at the start of projection (default 14)

    Returns
    -------
    int -- WALA in months
    """
    return wala_start + step


def turnover_smm(step: int,
                 wala_start: int = WALA_START,
                 start_month: int = START_MONTH,
                 base_turnover_cpr: float = BASE_TURNOVER_CPR) -> float:
    """
    Compute TurnoverSMM at a single projection step.

    TurnoverCPR = BaseTurnover_CPR * AgeFactor(WALA) * Seasonality(month)
    TurnoverSMM = 1 - (1 - TurnoverCPR)^(1/12)

    Parameters
    ----------
    step              : int   -- projection step, 1-based
    wala_start        : int   -- WALA at projection start (months)
    start_month       : int   -- calendar month projection begins (1-12)
    base_turnover_cpr : float -- annualized base turnover CPR (decimal)

    Returns
    -------
    float -- TurnoverSMM as monthly decimal (e.g. 0.0049)
    """
    wala         = wala_at_step(step, wala_start)
    m_age        = age_factor(wala)
    cal_month    = calendar_month_at_step(step, start_month)
    m_season     = seasonal_factor(cal_month)

    turnover_cpr = base_turnover_cpr * m_age * m_season
    turnover_smm = 1.0 - (1.0 - turnover_cpr) ** (1.0 / 12.0)

    return turnover_smm


def generate_turnover_vector(projection_months: int = 360,
                             wala_start: int = WALA_START,
                             start_month: int = START_MONTH,
                             base_turnover_cpr: float = BASE_TURNOVER_CPR) -> np.ndarray:
    """
    Generate a full TurnoverSMM vector for all projection steps.

    Convenience wrapper for the Monte Carlo loop -- precomputes all
    turnover values so they don't need to be recalculated per path.

    Parameters
    ----------
    projection_months : int   -- number of monthly steps (default 360)
    wala_start        : int   -- WALA at projection start (months)
    start_month       : int   -- calendar month projection begins (1-12)
    base_turnover_cpr : float -- annualized base turnover CPR (decimal)

    Returns
    -------
    np.ndarray of shape (projection_months,) -- TurnoverSMM at each step,
    indexed from 0 (step=1) to projection_months-1 (step=projection_months)
    """
    return np.array([
        turnover_smm(step, wala_start, start_month, base_turnover_cpr)
        for step in range(1, projection_months + 1)
    ])


def generate_turnover_table(projection_months: int = 360,
                            wala_start: int = WALA_START,
                            start_month: int = START_MONTH,
                            base_turnover_cpr: float = BASE_TURNOVER_CPR) -> pd.DataFrame:
    """
    Generate a full diagnostic DataFrame (mirrors Noah's original output).

    Useful for inspection and verification. Not needed in the main pipeline.

    Returns
    -------
    pd.DataFrame with columns:
        Month, WALA, M_age, Calendar_Month, M_season,
        Turnover_CPR, Turnover_SMM
    """
    steps = np.arange(1, projection_months + 1)
    wala_vals    = wala_start + steps
    m_age_vals   = np.minimum(1.0, wala_vals / 30.0)
    cal_months   = np.array([calendar_month_at_step(s, start_month) for s in steps])
    m_season_vals = np.array([SEASONALITY_MAP[m] for m in cal_months])
    cpr_vals     = base_turnover_cpr * m_age_vals * m_season_vals
    smm_vals     = 1.0 - (1.0 - cpr_vals) ** (1.0 / 12.0)

    return pd.DataFrame({
        'Month':          steps,
        'WALA':           wala_vals,
        'M_age':          m_age_vals,
        'Calendar_Month': cal_months,
        'M_season':       m_season_vals,
        'Turnover_CPR':   cpr_vals,
        'Turnover_SMM':   smm_vals,
    })


# ---------------------------------------------------------------------------
# Quick verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("TURNOVER MODULE VERIFICATION")
    print("=" * 60)

    print(f"\nPool constants:")
    print(f"  WALA at projection start : {WALA_START} months")
    print(f"  Projection start month   : {START_MONTH} (February)")
    print(f"  Base turnover CPR        : {BASE_TURNOVER_CPR:.1%}")

    print(f"\nFirst 24 months:")
    df = generate_turnover_table(projection_months=24)
    print(df.to_string(index=False))

    print(f"\nSingle-step call examples:")
    print(f"  turnover_smm(step=1)  = {turnover_smm(1):.8f}")
    print(f"  turnover_smm(step=12) = {turnover_smm(12):.8f}")

    vec = generate_turnover_vector(projection_months=360)
    print(f"\nFull vector shape: {vec.shape}")
    print(f"  Min SMM : {vec.min():.6f}")
    print(f"  Max SMM : {vec.max():.6f}")
    print(f"  SMM at t=1  : {vec[0]:.6f}")
    print(f"  SMM at t=12 : {vec[11]:.6f}")
    print(f"  SMM at t=30 : {vec[29]:.6f}")