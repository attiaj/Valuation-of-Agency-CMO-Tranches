"""
cmo_waterfall.py

CMO cash flow waterfall for FNR 2023-23.

Tranches
--------
LA  : AD, SEQ  -- senior sequential pay, receives all principal first
                  coupon 5.50%, original balance $100M, current $63.26M
LZ  : Z, SEQ   -- Z bond sequential, accrues interest while LA outstanding,
                  receives principal after LA retired
                  coupon 5.50%, original balance $17.84M, current $20.84M (accrued)
AI  : IO, NTL  -- interest only strip on Group 2 collateral (separate pool)
                  coupon 4.00%, notional balance $27.75M
                  valued independently using Group 2 prepayment model

Waterfall rules (Group 1 collateral -- LA and LZ)
--------------------------------------------------
Each period:
  1. Compute total pool interest = pool_balance * WAC/12
  2. Compute total principal available = scheduled + prepaid principal
  3. While LA balance > 0:
       - LA receives its coupon: LA_balance * 5.5%/12
       - LA receives ALL available principal (up to its remaining balance)
       - LZ accrues interest: LZ_balance += LZ_balance * 5.5%/12
         (interest is NOT paid out, added to LZ balance instead)
  4. After LA is retired:
       - LZ receives its coupon (now paid out, not accrued)
       - LZ receives all remaining principal

AI tranche (Group 2 collateral)
--------------------------------
AI receives interest only -- no principal ever.
Cash flow each period = AI_notional_balance * 4.0% / 12
Notional balance declines with Group 2 prepayments (passed in separately).

Reference: Davidson & Levin (2014), Chapter 2 (CMO Structure)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Tranche constants (FNR 2023-23, Bloomberg data, current as of Mar 2026)
# ---------------------------------------------------------------------------

# Group 1 collateral (LA and LZ)
LA_BALANCE_CURRENT  = 63_255_947.0   # current outstanding balance
LZ_BALANCE_CURRENT  = 20_840_256.0   # current outstanding balance (includes accrued)
LA_COUPON_ANNUAL    = 0.055           # 5.50%
LZ_COUPON_ANNUAL    = 0.055           # 5.50%

# Group 2 collateral (AI)
AI_NOTIONAL_CURRENT = 27_748_012.0   # current notional balance
AI_COUPON_ANNUAL    = 0.04            # 4.00%

GROUP1_POOL_CURRENT = 84_096_204.0   # current pool balance from Bloomberg
LA_BALANCE_NORM = LA_BALANCE_CURRENT / GROUP1_POOL_CURRENT
# = 63,255,947 / 84,096,204 = 0.7522
LZ_BALANCE_NORM = LZ_BALANCE_CURRENT / GROUP1_POOL_CURRENT
# = 20,840,256 / 84,096,204 = 0.2478

GROUP2_POOL_ORIGINAL = 37_918_033.0
AI_NOTIONAL_NORM = AI_NOTIONAL_CURRENT / GROUP2_POOL_ORIGINAL  # 0.7318


# ---------------------------------------------------------------------------
# Group 1 Waterfall: LA and LZ
# ---------------------------------------------------------------------------

def run_group1_waterfall(
    pool_cashflows: np.ndarray,
    pool_schedprin: np.ndarray,
    pool_prepaidprin: np.ndarray,
    pool_balance_paths: np.ndarray,
    wac_annual: float = 0.06471,
    la_balance_start: float = LA_BALANCE_NORM,
    lz_balance_start: float = LZ_BALANCE_NORM,
    la_coupon: float = LA_COUPON_ANNUAL,
    lz_coupon: float = LZ_COUPON_ANNUAL,
) -> dict:
    """
    Apply sequential pay and Z bond waterfall rules to Group 1 pool cash flows.

    Parameters
    ----------
    pool_cashflows    : np.ndarray (N_paths, T) -- total pool CF per path per step
    pool_schedprin    : np.ndarray (N_paths, T) -- scheduled principal per path per step
    pool_prepaidprin  : np.ndarray (N_paths, T) -- prepaid principal per path per step
    pool_balance_paths: np.ndarray (N_paths, T) -- pool balance at start of each period
    wac_annual        : float -- pool WAC (default 6.471%)
    la_balance_start  : float -- LA starting balance (normalized, default from Bloomberg)
    lz_balance_start  : float -- LZ starting balance (normalized, default from Bloomberg)
    la_coupon         : float -- LA annual coupon rate (default 5.50%)
    lz_coupon         : float -- LZ annual coupon rate (default 5.50%)

    Returns
    -------
    dict with keys:
        'la_cashflows' : np.ndarray (N_paths, T) -- LA cash flows per path per step
        'lz_cashflows' : np.ndarray (N_paths, T) -- LZ cash flows per path per step
        'la_balances'  : np.ndarray (N_paths, T) -- LA balance at start of each period
        'lz_balances'  : np.ndarray (N_paths, T) -- LZ balance at start of each period
    """
    N_paths, T = pool_cashflows.shape
    la_monthly = la_coupon / 12.0
    lz_monthly = lz_coupon / 12.0

    la_cashflows = np.zeros((N_paths, T))
    lz_cashflows = np.zeros((N_paths, T))
    la_balances  = np.zeros((N_paths, T))
    lz_balances  = np.zeros((N_paths, T))

    for path in range(N_paths):

        la_bal = la_balance_start
        lz_bal = lz_balance_start

        for t in range(T):

            la_balances[path, t] = la_bal
            lz_balances[path, t] = lz_bal

            if la_bal <= 1e-8 and lz_bal <= 1e-8:
                break

            # Total principal available this period
            principal_available = pool_schedprin[path, t] + pool_prepaidprin[path, t]

            if la_bal > 1e-8:
                # ---- LA is still outstanding ----

                # LA receives its coupon
                la_interest = la_bal * la_monthly
                la_cf = la_interest

                # LA receives all available principal (up to remaining balance)
                la_prin = min(principal_available, la_bal)
                la_cf  += la_prin
                la_bal  = max(la_bal - la_prin, 0.0)

                # LZ accrues interest (added to balance, NOT paid out)
                lz_accrual = lz_bal * lz_monthly
                lz_bal    += lz_accrual
                lz_cf      = 0.0

                # Any excess principal after LA is paid off goes to LZ
                excess_prin = max(principal_available - la_prin, 0.0)
                if excess_prin > 0 and la_bal <= 1e-8:
                    lz_prin = min(excess_prin, lz_bal)
                    lz_cf  += lz_prin
                    lz_bal  = max(lz_bal - lz_prin, 0.0)

            else:
                # ---- LA retired, LZ receives principal ----

                la_cf = 0.0

                # LZ receives its coupon (now paid out, not accrued)
                lz_interest = lz_bal * lz_monthly
                lz_prin     = min(principal_available, lz_bal)
                lz_cf       = lz_interest + lz_prin
                lz_bal      = max(lz_bal - lz_prin, 0.0)

            la_cashflows[path, t] = la_cf
            lz_cashflows[path, t] = lz_cf

    return {
        "la_cashflows": la_cashflows,
        "lz_cashflows": lz_cashflows,
        "la_balances":  la_balances,
        "lz_balances":  lz_balances,
    }


# ---------------------------------------------------------------------------
# Group 2 Waterfall: AI (IO strip)
# ---------------------------------------------------------------------------

def run_ai_waterfall(
    group2_balance_paths: np.ndarray,
    ai_coupon: float = AI_COUPON_ANNUAL,
    ai_notional_start: float = 1.0,
) -> np.ndarray:
    """
    Compute AI tranche cash flows (interest only on Group 2 notional balance).

    AI receives no principal -- only interest on its declining notional balance.
    The notional balance declines in line with Group 2 pool prepayments.

    Parameters
    ----------
    group2_balance_paths : np.ndarray (N_paths, T)
                           Group 2 pool balance at start of each period
                           (normalized to 1.0 at t=0)
    ai_coupon            : float -- AI annual coupon rate (default 4.00%)
    ai_notional_start    : float -- AI starting notional (normalized, default 1.0)

    Returns
    -------
    np.ndarray (N_paths, T) -- AI cash flows per path per step
    """
    ai_monthly = ai_coupon / 12.0

    # Scale notional balance by the Group 2 pool balance path
    # AI notional declines proportionally with Group 2 prepayments
    ai_notional_paths = group2_balance_paths * ai_notional_start

    # IO cash flow = notional * monthly coupon rate
    ai_cashflows = ai_notional_paths * ai_monthly

    return ai_cashflows


# ---------------------------------------------------------------------------
# Helper: extract balance paths and principal components from model output
# ---------------------------------------------------------------------------

def extract_principal_components(
    cashflows: np.ndarray,
    pool_params: dict,
    refi_array: np.ndarray,
    turnover_vec: np.ndarray,
    rate_paths: np.ndarray,
    psi_0: float = 0.65,
    beta: float = 0.20,
    M_T: float = 1.0,
    M_R: float = 0.840,
) -> dict:
    """
    Re-derive scheduled principal, prepaid principal, and balance paths
    from the pool model outputs.

    These are needed to apply the waterfall correctly -- the waterfall
    needs to know how much principal is available each period, which
    requires separating interest from principal in total cash flows.

    Parameters
    ----------
    cashflows    : np.ndarray (N_paths, T) -- total pool cash flows
    pool_params  : dict -- pool parameters (wac, wala_start, original_term, remaining_term)
    refi_array   : np.ndarray (N_paths, T) -- RefiSMM per path per step
    turnover_vec : np.ndarray (T,)         -- TurnoverSMM per step
    rate_paths   : np.ndarray (N_paths, T) -- H-W short rate paths (unused here, for signature)
    psi_0        : float -- initial active fraction
    beta         : float -- passive sensitivity
    M_T          : float -- turnover multiplier
    M_R          : float -- refi multiplier

    Returns
    -------
    dict with keys:
        'schedprin'     : np.ndarray (N_paths, T)
        'prepaidprin'   : np.ndarray (N_paths, T)
        'balance_paths' : np.ndarray (N_paths, T)
    """
    from apd import lambda_active, lambda_passive, aggregate_lambda, update_psi
    from utils import scheduled_principal

    N_paths, T = cashflows.shape
    wac_monthly    = pool_params['wac'] / 12.0
    original_term  = pool_params['original_term']
    remaining_term = pool_params['remaining_term']
    wala_start     = pool_params['wala_start']
    initial_balance = pool_params.get('initial_balance', 1.0)

    schedprin_arr    = np.zeros((N_paths, T))
    prepaidprin_arr  = np.zeros((N_paths, T))
    balance_paths    = np.zeros((N_paths, T))

    for path in range(N_paths):
        balance = initial_balance
        psi     = psi_0

        for t in range(T):
            balance_paths[path, t] = balance

            if balance <= 1e-6:
                break

            rem = max(remaining_term - t, 1)

            t_smm = M_T * turnover_vec[t]
            r_smm = M_R * refi_array[path, t]

            la  = lambda_active(t_smm, r_smm)
            lp  = lambda_passive(t_smm, r_smm, beta=beta)
            lam = aggregate_lambda(psi, la, lp)
            psi = update_psi(psi, la, lam)

            sched_prin   = scheduled_principal(balance, wac_monthly, rem)
            prepaid_prin = balance * lam

            schedprin_arr[path, t]   = sched_prin
            prepaidprin_arr[path, t] = prepaid_prin

            balance = max(balance - sched_prin - prepaid_prin, 0.0)

    return {
        "schedprin":     schedprin_arr,
        "prepaidprin":   prepaidprin_arr,
        "balance_paths": balance_paths,
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("CMO WATERFALL VERIFICATION (single path, 60 steps)")
    print("=" * 60)

    # Synthetic single path for verification
    N_paths, T = 1, 60
    wac_monthly = 0.06471 / 12.0

    # Flat prepayment speed for illustration
    lam = 0.018  # ~20% CPR
    balance = 1.0
    sched_arr   = np.zeros((N_paths, T))
    prepaid_arr = np.zeros((N_paths, T))
    cf_arr      = np.zeros((N_paths, T))
    bal_arr     = np.zeros((N_paths, T))

    from utils import scheduled_principal
    for t in range(T):
        bal_arr[0, t] = balance
        rem = max(316 - t, 1)
        sched = scheduled_principal(balance, wac_monthly, rem)
        prepaid = balance * lam
        cf_arr[0, t] = balance * wac_monthly + sched + prepaid
        sched_arr[0, t] = sched
        prepaid_arr[0, t] = prepaid
        balance = max(balance - sched - prepaid, 0.0)

    result = run_group1_waterfall(
        pool_cashflows=cf_arr,
        pool_schedprin=sched_arr,
        pool_prepaidprin=prepaid_arr,
        pool_balance_paths=bal_arr,
    )

    la_cf = result["la_cashflows"]
    lz_cf = result["lz_cashflows"]
    la_bal = result["la_balances"]
    lz_bal = result["lz_balances"]

    print(f"\n{'Month':>5} | {'Pool CF':>9} | {'LA CF':>9} | {'LZ CF':>9} | {'LA Bal':>9} | {'LZ Bal':>9}")
    print("-" * 60)
    for t in range(min(24, T)):
        print(f"{t+1:5d} | {cf_arr[0,t]:9.6f} | {la_cf[0,t]:9.6f} | "
              f"{lz_cf[0,t]:9.6f} | {la_bal[0,t]:9.6f} | {lz_bal[0,t]:9.6f}")

    # Check: LA + LZ CFs should approximately equal pool CFs
    # (small difference due to Z accrual mechanics)
    total_la = la_cf.sum()
    total_lz = lz_cf.sum()
    total_pool = cf_arr.sum()
    print(f"\nTotal LA CF : {total_la:.6f}")
    print(f"Total LZ CF : {total_lz:.6f}")
    print(f"Total pool  : {total_pool:.6f}")
    print(f"\nLA retired at month: {(la_bal[0] <= 1e-8).argmax()}")