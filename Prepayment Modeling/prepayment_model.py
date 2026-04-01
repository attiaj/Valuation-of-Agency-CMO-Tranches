"""
prepayment_model.py

Top-level integration of the APD prepayment model.

Consumes:
    - Hull-White short rate paths        (rate_paths.npz)
    - Turnover vector                    (prepayment_turnover.py)
    - Refinancing SMM paths              (prepayment_refi.py)
    - Enhanced variable multipliers      (enhanced.py)
    - APD recursion                      (apd.py)

Returns per-path pool-level cash flows ready for discounting.

APD model (Davidson & Levin, Chapter 7):
    lambda(t) = psi(t) * lambda_a(t) + (1 - psi(t)) * lambda_p(t)
    lambda_a(t) = M_T * TurnoverSMM(t) + M_R * RefiSMM(t)
    lambda_p(t) = M_T * TurnoverSMM(t) + beta * M_R * RefiSMM(t)
    psi(t+1)   = psi(t) * (1 - lambda_a(t)) / (1 - lambda(t))
"""

import numpy as np
from pathlib import Path

from prepayment_turnover import generate_turnover_vector
from prepayment_refi import (
    RefiParams,
    load_project_data_bundle,
    infer_refi_inputs_from_bundle,
    refinancing_smm_paths,
    calibrate_max_refi_from_observed_cpr,
)
from enhanced import get_multipliers
from apd import init_psi, lambda_active, lambda_passive, aggregate_lambda, update_psi
from utils import scheduled_principal


# ---------------------------------------------------------------------------
# Pool constants (FNR 2024-100, Bloomberg data)
# ---------------------------------------------------------------------------

POOL_DATA = {
    'fico':      759,   # not available at pool level from Bloomberg tranche sheet
    'ltv':       81,
    'sato':      97.1,
    'loan_size': 498370,
}

POOL_PARAMS = {
    'wac':               0.06471,   # 6.471% collateral WAC from Bloomberg
    'origination_month': 4,      # December 2024
    'original_term':     360,     # 30-year mortgages
    'wala_start':        35,      # months seasoned at projection start (Feb 2026)
    'remaining_term':    316,     # months from next pay to maturity (Bloomberg)
    'initial_balance':   1.0,     # normalized to 1.0
}

# FNR 2023-23 Group 1 (LA and LZ collateral)
GROUP1_INPUTS = {
    "current_note_rate":    0.06471,   # WAC from Bloomberg
    "remaining_term_months": 316,
    "observed_cpr_1m":      19.92,     # Bloomberg 1m CPR
}

# FNR 2023-23 Group 2 (AI collateral)
GROUP2_INPUTS = {
    "current_note_rate":    0.04697,   # WAC from Bloomberg
    "remaining_term_months": 266,
    "observed_cpr_1m":      3.66,      # Bloomberg 1m CPR
}


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def run_prepayment_model(
    tranche_sheet: str = "A Tranche",
    path_file: str = "rate_paths.npz",
    beta: float = 0.25,
    psi_0: float = 0.70,
    pool_data: dict = None,
    pool_params: dict = None,
    base_dir=None,
    pool_inputs: dict = None,    # add this
) -> dict:
    """
    Run the full APD prepayment model and return per-path cash flows.

    Parameters
    ----------
    tranche_sheet : str        -- Bloomberg Excel sheet to parse (default 'A Tranche')
    path_file     : str        -- filename of Hull-White .npz paths
    beta          : float      -- passive borrower refi sensitivity (default 0.25)
    psi_0         : float      -- initial active borrower fraction (default 0.70)
    pool_data     : dict       -- Bloomberg pool characteristics for enhanced multipliers
                                 Keys: 'fico', 'ltv', 'sato', 'loan_size'
                                 Pass None to use module-level POOL_DATA defaults
    pool_params   : dict       -- pool specifics
                                 Pass None to use module-level POOL_PARAMS defaults
    base_dir      : str | Path -- directory containing rate_paths.npz,
                                 CMO_BLOOMBERG_DATA.xlsx, and treasury CSV.
                                 Defaults to data_io/ one level above this file.

    Returns
    -------
    dict with keys:
        'cashflows'     : np.ndarray (N_paths, T) -- total CF per path per step
        'refi_array'    : np.ndarray (N_paths, T) -- RefiSMM per path per step
        'turnover_vec'  : np.ndarray (T,)         -- TurnoverSMM vector
        'bundle'        : ProjectDataBundle       -- loaded data files
        'params'        : RefiParams              -- calibrated refi parameters
    """
    if pool_data is None:
        pool_data = POOL_DATA
    if pool_params is None:
        pool_params = POOL_PARAMS

    # Resolve base_dir: default to data_io/ one level above this file
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent / "data_io"
    base_dir = Path(base_dir)

    wac_annual      = pool_params['wac']
    wac_monthly     = wac_annual / 12.0
    original_term   = pool_params['original_term']
    remaining_term  = pool_params['remaining_term']
    wala_start      = pool_params['wala_start']
    initial_balance = pool_params.get('initial_balance', 1.0)

    # ------------------------------------------------------------------
    # Step 1: Load all data files via refi bundle
    # ------------------------------------------------------------------
    print("Loading project data bundle...")
    print(f"  base_dir: {base_dir}")
    bundle = load_project_data_bundle(
        base_dir=base_dir,
        tranche_sheet=tranche_sheet,
        path_file=path_file,
    )
    inputs = inputs = pool_inputs if pool_inputs is not None else GROUP1_INPUTS
    rate_paths = bundle.paths          # shape (N_paths, T)
    N_paths, T = rate_paths.shape
    print(f"  Loaded {N_paths:,} paths x {T} steps")
    print(f"  r0 = {bundle.r0:.4%}")
    if bundle.bloomberg is not None:
        b = bundle.bloomberg
        print(f"  Bloomberg: {b.deal} / {b.tranche_class}  "
              f"WAC={b.collateral_coupon:.2%}  1m CPR={b.current_cpr_1m}")

    # ------------------------------------------------------------------
    # Step 2: Precompute turnover vector (path-independent)
    # ------------------------------------------------------------------
    print("Computing turnover vector...")
    turnover_vec = generate_turnover_vector(
        projection_months=T,
        wala_start=wala_start,
    )
    if len(turnover_vec) < T:
        turnover_vec = np.append(
            turnover_vec,
            np.full(T - len(turnover_vec), turnover_vec[-1])
        )
    turnover_vec = turnover_vec[:T]
    print(f"  TurnoverSMM range: [{turnover_vec.min():.4%}, {turnover_vec.max():.4%}]")

    # ------------------------------------------------------------------
    # Step 3: Compute refi SMM for all paths (vectorized)
    # ------------------------------------------------------------------
    print("Computing refinancing SMM paths...")
    params = RefiParams(dispersion=0.10)

    # First pass: get t=0 payment ratio for CPR calibration
    prelim = refinancing_smm_paths(
        short_rate_paths=rate_paths,
        current_note_rate=inputs["current_note_rate"],
        remaining_term_months=remaining_term,
        params=params,
        treasury_curve=bundle.treasury_curve,
        r0=bundle.r0,
    )
    pay_ratio_0 = float(prelim["payment_ratio"].mean(axis=0)[0])

    # Calibrate max_refi_smm to observed Bloomberg 1m CPR
    params.max_refi_smm = calibrate_max_refi_from_observed_cpr(
        observed_cpr_pct=inputs["observed_cpr_1m"],
        pay_ratio_0=pay_ratio_0,
        turnover_smm_0=float(turnover_vec[0]),
        psi_0=psi_0,
        beta_passive=beta,
        threshold=params.threshold,
        dispersion=params.dispersion,
        default_max_refi_smm=params.max_refi_smm,
    )
    print(f"  Calibrated max_refi_smm: {params.max_refi_smm:.4%}")

    # Final refi computation with calibrated params
    refi_result = refinancing_smm_paths(
        short_rate_paths=rate_paths,
        current_note_rate=inputs["current_note_rate"],
        remaining_term_months=remaining_term,
        params=params,
        treasury_curve=bundle.treasury_curve,
        r0=bundle.r0,
    )
    refi_array = refi_result["refi_smm"]   # shape (N_paths, T)
    print(f"  RefiSMM at t=0 (mean): {refi_array[:, 0].mean():.4%}")

    # ------------------------------------------------------------------
    # Step 4: Enhanced variable multipliers (Bloomberg pool data)
    # ------------------------------------------------------------------
    M_T, M_R = get_multipliers(pool_data)
    print(f"Enhanced multipliers: M_T={M_T:.3f}, M_R={M_R:.3f}")

    # ------------------------------------------------------------------
    # Step 5: Monte Carlo cash flow loop -- per path, per step
    # ------------------------------------------------------------------
    print(f"Running Monte Carlo cash flow loop ({N_paths:,} paths x {T} steps)...")
    cashflows = np.zeros((N_paths, T))

    for path in range(N_paths):

        balance = initial_balance
        psi     = psi_0

        for t in range(T):

            if balance <= 1e-6:
                break

            rem = max(remaining_term - t, 1)

            # Component SMMs with enhanced multipliers
            t_smm = M_T * turnover_vec[t]
            r_smm = M_R * refi_array[path, t]

            # APD sub-population speeds
            la  = lambda_active(t_smm, r_smm)
            lp  = lambda_passive(t_smm, r_smm, beta=beta)
            lam = aggregate_lambda(psi, la, lp)

            # Update psi for next period
            psi = update_psi(psi, la, lam)

            # Cash flow components
            interest     = balance * wac_monthly
            sched_prin   = scheduled_principal(balance, wac_monthly, rem)
            prepaid_prin = balance * lam

            cashflows[path, t] = interest + sched_prin + prepaid_prin

            # Update balance
            balance = max(balance - sched_prin - prepaid_prin, 0.0)

    print("  Cash flow loop complete.")

    return {
        "cashflows":    cashflows,
        "refi_array":   refi_array,
        "turnover_vec": turnover_vec,
        "bundle":       bundle,
        "params":       params,
    }


# ---------------------------------------------------------------------------
# Discounting utility
# ---------------------------------------------------------------------------

def discount_cashflows(cashflows: np.ndarray, rate_paths: np.ndarray) -> np.ndarray:
    """
    Discount each path's cash flows using that path's own short rates.

    DF(t) = exp(-sum(r(0)...r(t-1)) * dt)  where dt = 1/12 (monthly)

    Parameters
    ----------
    cashflows  : np.ndarray (N_paths, T)
    rate_paths : np.ndarray (N_paths, T) -- annual short rates

    Returns
    -------
    pv_per_path : np.ndarray (N_paths,) -- present value of CFs on each path
    """
    dt = 1.0 / 12.0
    N_paths, T = cashflows.shape

    cum_rates = np.cumsum(rate_paths[:, :T], axis=1) * dt

    discount_factors = np.ones((N_paths, T))
    discount_factors[:, 1:] = np.exp(-cum_rates[:, :-1])

    pv_per_path = (cashflows * discount_factors).sum(axis=1)
    return pv_per_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    
    import warnings
    warnings.filterwarnings("ignore")

    print("\n" + "=" * 70)
    print("APD PREPAYMENT MODEL + CASH FLOW VALUATION")
    print("=" * 70)

    base_dir = Path(__file__).resolve().parent.parent / "data_io"

    output = run_prepayment_model(
        tranche_sheet="LA",
        path_file="rate_paths.npz",
        beta=0.20,       # updated: high SATO pool, more passive borrowers
        psi_0=0.65,      # updated: purchase-dominated pool, less refi-prone
        base_dir=base_dir,
    )

    cashflows  = output["cashflows"]
    rate_paths = output["bundle"].paths
    T          = cashflows.shape[1]

    print("\nDiscounting cash flows...")
    pv_per_path = discount_cashflows(cashflows, rate_paths)

    pool_price = pv_per_path.mean()
    print("\n" + "=" * 70)
    print("FIRST DRAFT VALUATION RESULTS")
    print("=" * 70)
    print(f"  Paths simulated        : {cashflows.shape[0]:,}")
    print(f"  Time steps             : {T}")
    print(f"  Mean PV (pool price)   : {pool_price:.6f}")
    print(f"  Std dev across paths   : {pv_per_path.std():.6f}")
    print(f"  Min / Max PV           : {pv_per_path.min():.6f} / {pv_per_path.max():.6f}")

    print(f"\nFirst 12 months (mean cash flow across paths):")
    print(f"  {'Month':>5} | {'Mean CF':>10} | {'Turnover SMM':>13} | {'Mean Refi SMM':>14}")
    turnover_vec = output["turnover_vec"]
    refi_array   = output["refi_array"]
    for t in range(min(12, T)):
        print(f"  {t+1:5d} | {cashflows[:, t].mean():10.6f} | "
              f"{turnover_vec[t]:12.4%} | {refi_array[:, t].mean():13.4%}")