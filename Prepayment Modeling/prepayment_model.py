"""
prepayment_model.py

Top-level integration: runs the APD prepayment model over all
Hull-White rate paths and returns pool-level cash flows per path.

Inputs:
    - rate_paths  : (N_paths x T) array of monthly short rates from Hull-White
                    in annual decimal form (will be converted to monthly internally)
    - pool_data   : dict of Bloomberg pool characteristics
    - pool_params : dict of pool specifics (WAC, origination_month, original_term)

Output:
    - cashflows   : (N_paths x T) array of total pool cash flows per period
"""

import numpy as np

from components.turnover import turnover_smm
from components.refinancing import refi_smm
from components.enhanced import get_multipliers
from apd import init_psi, lambda_active, lambda_passive, aggregate_lambda, update_psi
from utils import annual_to_monthly_rate, scheduled_principal


def run_prepayment_model(
    rate_paths: np.ndarray,
    pool_data: dict,
    pool_params: dict,
    beta: float = 0.25,
) -> np.ndarray:
    """
    Run the APD prepayment model across all Hull-White rate paths.

    Parameters
    ----------
    rate_paths  : np.ndarray, shape (N_paths, T)
                  Annual short rates from Hull-White (decimal, e.g. 0.05 for 5%)
                  NOTE: these are short rates -- a mortgage spread is added internally
                  to approximate current mortgage rates. Adjust MORTGAGE_SPREAD as needed.

    pool_data   : dict with Bloomberg pool characteristics
                  Keys: 'fico', 'ltv', 'sato', 'loan_size'

    pool_params : dict with keys:
                  'wac'               : float -- annual WAC (decimal)
                  'origination_month' : int   -- calendar month pool was originated (1-12)
                  'original_term'     : int   -- original loan term in months (e.g. 360)
                  'initial_balance'   : float -- starting pool balance (e.g. 1.0 for normalized)

    beta        : float -- passive borrower refi sensitivity (default 0.25)

    Returns
    -------
    cashflows : np.ndarray, shape (N_paths, T)
                Total cash flow (interest + scheduled principal + prepaid principal)
                for each path and time step.
    """

    # Approximate spread from short rate to current mortgage rate
    # TODO: replace with term structure spread once available
    MORTGAGE_SPREAD = 0.015  # 150 bps over short rate as rough proxy

    N_paths, T = rate_paths.shape

    wac_annual         = pool_params['wac']
    origination_month  = pool_params['origination_month']
    original_term      = pool_params['original_term']
    initial_balance    = pool_params.get('initial_balance', 1.0)

    wac_monthly = annual_to_monthly_rate(wac_annual)

    # Pool-level multipliers from Bloomberg data (constant across paths)
    M_T, M_R = get_multipliers(pool_data)

    # Initial active fraction
    psi_0 = init_psi(pool_data)

    cashflows = np.zeros((N_paths, T))

    for path in range(N_paths):

        balance    = initial_balance
        psi        = psi_0

        for t in range(T):

            if balance <= 0:
                break

            pool_age       = t + 1
            remaining_term = max(original_term - t, 1)
            month          = ((origination_month - 1 + t) % 12) + 1

            # Current mortgage rate approximation
            r_annual  = rate_paths[path, t] + MORTGAGE_SPREAD
            r_monthly = annual_to_monthly_rate(r_annual)

            # Component SMMs (scaled by enhanced multipliers)
            t_smm = M_T * turnover_smm(pool_age, month)
            r_smm = M_R * refi_smm(r_monthly, wac_monthly, remaining_term)

            # APD sub-population speeds
            la  = lambda_active(t_smm, r_smm)
            lp  = lambda_passive(t_smm, r_smm, beta=beta)
            lam = aggregate_lambda(psi, la, lp)

            # Update psi for next period
            psi = update_psi(psi, la, lam)

            # Cash flow components
            interest           = balance * wac_monthly
            sched_principal    = scheduled_principal(balance, wac_monthly, remaining_term)
            prepaid_principal  = balance * lam

            total_cf = interest + sched_principal + prepaid_principal
            cashflows[path, t] = total_cf

            # Update balance
            balance -= (sched_principal + prepaid_principal)

    return cashflows