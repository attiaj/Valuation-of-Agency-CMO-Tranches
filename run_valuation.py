"""
run_valuation.py

Master valuation runner for FNR 2023-23 CMO.

Directory structure expected:
    project_root/
    ├── run_valuation.py          <- this file
    ├── data_io/
    │   ├── rate_paths.npz
    │   └── market_data/
    │       └── treasury_rates_20260217.csv
    └── Prepayment Modeling/
        ├── prepayment_model.py
        ├── prepayment_refi.py
        ├── prepayment_turnover.py
        ├── apd.py
        ├── enhanced.py
        ├── utils.py
        └── cmo_waterfall.py

Pipeline
--------
1. Load Hull-White rate paths from data_io/
2. Run APD prepayment model -> pool cash flows (Group 1 and Group 2)
3. Extract principal components for waterfall allocation
4. Run CMO waterfall -> LA, LZ, AI tranche cash flows
5. Discount per tranche -> tranche prices
6. Compute OAS for each tranche via bisection
7. Compute effective duration and convexity via parallel rate shifts

Deal: FNR 2023-23 (Fannie Mae)
Tranches: LA (AD, SEQ), LZ (Z, SEQ), AI (IO, NTL)
Collateral: Group 1 = 100% FNCL 5.5M, Group 2 = 99% FNCL 4.0S
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup: resolve repo root and add "Prepayment Modeling" to sys.path
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_io"
PREP_DIR = ROOT / "Prepayment Modeling"

if not PREP_DIR.is_dir():
    raise FileNotFoundError(
        f"Expected prepayment package directory at {PREP_DIR}. "
        "Keep run_valuation.py at the repo root next to the 'Prepayment Modeling' folder."
    )

_prep = str(PREP_DIR)
if _prep not in sys.path:
    sys.path.insert(0, _prep)

from prepayment_model import POOL_DATA, POOL_PARAMS, run_prepayment_model, GROUP2_INPUTS
from cmo_waterfall import (
    extract_principal_components,
    run_ai_waterfall,
    run_group1_waterfall,
)

# ---------------------------------------------------------------------------
# Deal constants
# ---------------------------------------------------------------------------

# Market prices from Bloomberg (as of March 2026, in decimal)
# Prices quoted in 32nds: 101-24 = 101 + 24/32
LA_MARKET_PRICE = 101 + 24 / 32  # 101.750
LZ_MARKET_PRICE = 100 + 1 / 8  # 100.125 (approx. bid/ask mid)
AI_MARKET_PRICE = (19 + 11 / 32 + 19 + 30 / 32) / 2  # midpoint ~19.641

# Model parameters (FNR 2023-23, derived from Bloomberg pool characteristics)
PSI_0 = 0.65  # initial active borrower fraction
BETA = 0.20  # passive borrower refi sensitivity

# Group 2 pool params (AI collateral, assumed same characteristics as Group 1)
POOL_PARAMS_G2 = {
    "wac": 0.04697,  # Group 2 WAC
    "original_term": 360,
    "wala_start": 80,  # Group 2 WALA from Bloomberg
    "remaining_term": 266,  # Group 2 remaining term
    "initial_balance": 1.0,
}


# ---------------------------------------------------------------------------
# Step 1: Run prepayment model for Group 1 (LA + LZ collateral)
# ---------------------------------------------------------------------------


def run_group1_prepayment(base_dir: Path) -> dict:
    print("\n" + "=" * 70)
    print("GROUP 1 PREPAYMENT MODEL (LA + LZ collateral)")
    print("=" * 70)

    output = run_prepayment_model(
        tranche_sheet="LA",
        path_file="rate_paths.npz",
        beta=BETA,
        psi_0=PSI_0,
        pool_data=POOL_DATA,
        pool_params=POOL_PARAMS,
        base_dir=base_dir,
    )
    return output


# ---------------------------------------------------------------------------
# Step 2: Run prepayment model for Group 2 (AI collateral)
# ---------------------------------------------------------------------------


def run_group2_prepayment(base_dir: Path) -> dict:
    print("\n" + "=" * 70)
    print("GROUP 2 PREPAYMENT MODEL (AI collateral)")
    print("=" * 70)

    output = run_prepayment_model(
        tranche_sheet="LA",
        path_file="rate_paths.npz",
        beta=BETA,
        psi_0=PSI_0,
        pool_data=POOL_DATA,  # same pool characteristics assumed
        pool_params=POOL_PARAMS_G2,
        base_dir=base_dir,
        pool_inputs=GROUP2_INPUTS,    # add this
    )

    # Override inputs to use Group 2 Bloomberg data
    # (run_prepayment_model uses GROUP1_INPUTS internally;
    #  for Group 2 we post-process using the Group 2 balance paths)
    return output


# ---------------------------------------------------------------------------
# Step 3: Apply waterfall
# ---------------------------------------------------------------------------


def run_waterfall(g1_output: dict, g2_output: dict) -> dict:
    print("\n" + "=" * 70)
    print("CMO WATERFALL")
    print("=" * 70)

    rate_paths = g1_output["bundle"].paths
    M_T = 1.0
    M_R = 0.840  # from enhanced.py with FICO=759, LTV=81, SATO=97.1bps

    # Extract principal components for Group 1
    print("Extracting principal components (Group 1)...")
    g1_components = extract_principal_components(
        cashflows=g1_output["cashflows"],
        pool_params=POOL_PARAMS,
        refi_array=g1_output["refi_array"],
        turnover_vec=g1_output["turnover_vec"],
        rate_paths=rate_paths,
        psi_0=PSI_0,
        beta=BETA,
        M_T=M_T,
        M_R=M_R,
    )

    # Run Group 1 waterfall (LA and LZ)
    print("Running Group 1 waterfall (LA + LZ)...")
    g1_waterfall = run_group1_waterfall(
        pool_cashflows=g1_output["cashflows"],
        pool_schedprin=g1_components["schedprin"],
        pool_prepaidprin=g1_components["prepaidprin"],
        pool_balance_paths=g1_components["balance_paths"],
    )

    # AI cash flows: notional * 4% / 12, declining with Group 2 balance
    g2_balances = extract_principal_components(
        cashflows=g2_output["cashflows"],
        pool_params=POOL_PARAMS_G2,
        refi_array=g2_output["refi_array"],
        turnover_vec=g2_output["turnover_vec"],
        rate_paths=rate_paths,
        psi_0=PSI_0,
        beta=BETA,
        M_T=M_T,
        M_R=M_R,
    )

    AI_NOTIONAL_FRAC = 27_748_012 / 37_918_033   # 0.7318

    ai_cashflows = run_ai_waterfall(
        group2_balance_paths=g2_balances["balance_paths"],
        ai_coupon=0.04,
        ai_notional_start=AI_NOTIONAL_FRAC,
    )

    print(f"  LA cash flows: shape {g1_waterfall['la_cashflows'].shape}")
    print(f"  LZ cash flows: shape {g1_waterfall['lz_cashflows'].shape}")
    print(f"  AI cash flows: shape {ai_cashflows.shape}")

    return {
        "la_cashflows": g1_waterfall["la_cashflows"],
        "lz_cashflows": g1_waterfall["lz_cashflows"],
        "ai_cashflows": ai_cashflows,
        "la_balances": g1_waterfall["la_balances"],
        "lz_balances": g1_waterfall["lz_balances"],
        "rate_paths": rate_paths,
    }


# ---------------------------------------------------------------------------
# Step 4: Discount tranche cash flows
# ---------------------------------------------------------------------------


def discount_tranche(
    cashflows: np.ndarray, rate_paths: np.ndarray, oas_bps: float = 0.0
) -> np.ndarray:
    """
    Discount tranche cash flows path-by-path with optional OAS spread.

    Parameters
    ----------
    cashflows  : np.ndarray (N_paths, T)
    rate_paths : np.ndarray (N_paths, T) -- annual short rates
    oas_bps    : float -- OAS in basis points added to discount rate

    Returns
    -------
    np.ndarray (N_paths,) -- PV per path
    """
    dt = 1.0 / 12.0
    oas = oas_bps / 10000.0
    N_paths, T = cashflows.shape

    cum_rates = np.cumsum((rate_paths[:, :T] + oas), axis=1) * dt
    discount_factors = np.ones((N_paths, T))
    discount_factors[:, 1:] = np.exp(-cum_rates[:, :-1])

    return (cashflows * discount_factors).sum(axis=1)


# ---------------------------------------------------------------------------
# Step 5: OAS calculation via bisection
# ---------------------------------------------------------------------------


def compute_oas(
    cashflows: np.ndarray,
    rate_paths: np.ndarray,
    market_price_pct: float,
    lo_bps: float = -500.0,
    hi_bps: float = 2000.0,
    tol: float = 0.01,
    max_iter: int = 50,
) -> float:
    """
    Find OAS (in bps) such that model price = market price.

    Uses bisection search.

    Parameters
    ----------
    cashflows        : np.ndarray (N_paths, T)
    rate_paths       : np.ndarray (N_paths, T)
    market_price_pct : float -- market price as decimal (e.g. 101.75 / 100 = 1.0175)
    lo_bps           : float -- lower bound for bisection (default -500 bps)
    hi_bps           : float -- upper bound for bisection (default 2000 bps)
    tol              : float -- convergence tolerance in bps (default 0.01)
    max_iter         : int   -- maximum iterations

    Returns
    -------
    float -- OAS in basis points
    """
    target = market_price_pct

    for _ in range(max_iter):
        mid = (lo_bps + hi_bps) / 2.0
        price = discount_tranche(cashflows, rate_paths, oas_bps=mid).mean()

        if price > target:
            lo_bps = mid
        else:
            hi_bps = mid

        if (hi_bps - lo_bps) < tol:
            break

    return (lo_bps + hi_bps) / 2.0


# ---------------------------------------------------------------------------
# Step 6: Effective duration and convexity
# ---------------------------------------------------------------------------


def compute_duration_convexity(
    cashflows: np.ndarray,
    rate_paths: np.ndarray,
    oas_bps: float,
    shift_bps: float = 100.0,
) -> tuple[float, float]:
    """
    Compute effective duration and convexity via parallel rate shifts.

    Shifts the entire rate path up and down by shift_bps and reruns discounting.
    Note: prepayment speeds are held constant (no prepayment model rerun).
    This is a simplification -- full OAD would rerun prepayment at each shift.

    Parameters
    ----------
    cashflows  : np.ndarray (N_paths, T)
    rate_paths : np.ndarray (N_paths, T)
    oas_bps    : float -- tranche OAS in bps
    shift_bps  : float -- rate shift in bps (default 100 bps)

    Returns
    -------
    (eff_duration, eff_convexity) : tuple of floats
    """
    shift = shift_bps / 10000.0

    p0 = discount_tranche(cashflows, rate_paths, oas_bps=oas_bps).mean()
    p_up = discount_tranche(cashflows, rate_paths + shift, oas_bps=oas_bps).mean()
    p_dn = discount_tranche(cashflows, rate_paths - shift, oas_bps=oas_bps).mean()

    eff_duration = -(p_up - p_dn) / (2 * p0 * shift)
    eff_convexity = (p_up + p_dn - 2 * p0) / (p0 * shift**2)

    return eff_duration, eff_convexity


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FNR 2023-23 CMO VALUATION")
    print("=" * 70)
    print("Deal    : FNR 2023-23")
    print("Issuer  : Fannie Mae")
    print("Tranche : LA (AD SEQ), LZ (Z SEQ), AI (IO NTL)")
    print("As of   : March 2026")
 
    # ------------------------------------------------------------------
    # Step 1 & 2: Prepayment models
    # ------------------------------------------------------------------
    g1_output = run_group1_prepayment(DATA_DIR)
    g2_output = run_group2_prepayment(DATA_DIR)
 
    # ------------------------------------------------------------------
    # Step 3: Waterfall
    # ------------------------------------------------------------------
    wf = run_waterfall(g1_output, g2_output)
    rate_paths = wf["rate_paths"]
 
    # ------------------------------------------------------------------
    # Step 4: Tranche prices at OAS = 0
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRANCHE VALUATIONS")
    print("=" * 70)
 
    # Normalization fractions -- convert from pool-normalized to per-tranche-face
    LA_FRAC = 63_255_947 / 84_096_204    # LA current / Group 1 pool current
    LZ_FRAC = 20_840_256 / 84_096_204    # LZ current / Group 1 pool current
 
    # Scale cash flows to per-tranche-face terms
    la_cf_scaled = wf["la_cashflows"] / LA_FRAC
    lz_cf_scaled = wf["lz_cashflows"] / LZ_FRAC
    ai_cf_scaled = wf["ai_cashflows"]
 
    # Prices per $1 face
    la_pv = discount_tranche(la_cf_scaled, rate_paths).mean()
    lz_pv = discount_tranche(lz_cf_scaled, rate_paths).mean()
    ai_pv = discount_tranche(ai_cf_scaled, rate_paths).mean()
 
    print("\n  Model prices (OAS = 0, per $100 face):")
    print(f"    LA : {la_pv*100:.3f}")
    print(f"    LZ : {lz_pv*100:.3f}")
    print(f"    AI : {ai_pv*100:.3f}")
 
    print("\n  Bloomberg market prices (per $100 face):")
    print(f"    LA : {LA_MARKET_PRICE:.3f}")
    print(f"    LZ : {LZ_MARKET_PRICE:.3f}")
    print(f"    AI : {AI_MARKET_PRICE:.3f}")
 
    # ------------------------------------------------------------------
    # Step 5: OAS
    # ------------------------------------------------------------------
    print("\n  Computing OAS via bisection...")
 
    la_oas = compute_oas(la_cf_scaled, rate_paths, LA_MARKET_PRICE / 100)
    lz_oas = compute_oas(lz_cf_scaled, rate_paths, LZ_MARKET_PRICE / 100)
    ai_oas = compute_oas(ai_cf_scaled, rate_paths, AI_MARKET_PRICE / 100)
 
    print("\n  OAS (bps):")
    print(f"    LA : {la_oas:.1f} bps")
    print(f"    LZ : {lz_oas:.1f} bps")
    print(f"    AI : {ai_oas:.1f} bps")
 
    # ------------------------------------------------------------------
    # Step 6: Effective duration and convexity
    # ------------------------------------------------------------------
    print("\n  Computing effective duration and convexity (100 bps shift)...")
 
    la_dur, la_cvx = compute_duration_convexity(la_cf_scaled, rate_paths, la_oas)
    lz_dur, lz_cvx = compute_duration_convexity(lz_cf_scaled, rate_paths, lz_oas)
    ai_dur, ai_cvx = compute_duration_convexity(ai_cf_scaled, rate_paths, ai_oas)
 
    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(
        f"\n  {'Tranche':<10} {'Model Price':>12} {'Mkt Price':>10} "
        f"{'OAS (bps)':>10} {'Eff Dur':>9} {'Eff Cvx':>9}"
    )
    print("  " + "-" * 65)
    for name, mv, mp, oas, dur, cvx in [
        ("LA", la_pv, LA_MARKET_PRICE / 100, la_oas, la_dur, la_cvx),
        ("LZ", lz_pv, LZ_MARKET_PRICE / 100, lz_oas, lz_dur, lz_cvx),
        ("AI", ai_pv, AI_MARKET_PRICE / 100, ai_oas, ai_dur, ai_cvx),
    ]:
        print(
            f"  {name:<10} {mv:>12.4f} {mp:>10.4f} "
            f"{oas:>10.1f} {dur:>9.3f} {cvx:>9.3f}"
        )
 
    print("\n  Notes:")
    print("  - Model: Hull-White (a=0.021, sigma=0.0096) + APD prepayment")
    print("  - Paths: 1,000 Monte Carlo, 30-year horizon")
    print(f"  - psi_0={PSI_0}, beta={BETA}, M_R=0.840")
    print("  - Effective duration/convexity: cash flows held fixed at shifted rates")
    print("    (full OAD would rerun prepayment model at each rate shift)")
    print(f"  - AI market price mid: {AI_MARKET_PRICE:.3f}")
