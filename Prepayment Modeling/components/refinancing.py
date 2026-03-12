"""
components/refinancing.py

Refinancing component of the prepayment model.
Computes RefiSMM using a payment ratio incentive measure
fed through a normal CDF S-curve.

Output: RefiSMM -- monthly prepayment rate as a decimal
"""

import numpy as np
from scipy.stats import norm


def payment_ratio(current_rate: float, wac: float, remaining_term: int) -> float:
    """
    Compute the refinancing incentive as new payment / old payment.

    P = r * B / (1 - (1+r)^-N)  for a unit balance B=1

    Ratio < 1 : refinancing is attractive (new payment is lower)
    Ratio = 1 : no incentive
    Ratio > 1 : no refinancing

    Parameters
    ----------
    current_rate   : float -- current market mortgage rate, monthly decimal
    wac            : float -- pool weighted average coupon, monthly decimal
    remaining_term : int   -- remaining months on loan

    Returns
    -------
    float -- payment ratio P_new / P_old
    """
    def monthly_payment(r, n):
        if r == 0:
            return 1.0 / n
        return r / (1 - (1 + r) ** (-n))

    p_old = monthly_payment(wac, remaining_term)
    p_new = monthly_payment(current_rate, remaining_term)

    return p_new / p_old


def refi_smm(
    current_rate: float,
    wac: float,
    remaining_term: int,
    max_refi: float = 0.12,
    mu: float = 0.90,
    sigma: float = 0.10,
) -> float:
    """
    Compute the refinancing component of SMM via normal CDF S-curve.

    RefiSMM = MaxRefi * Phi((mu - PaymentRatio) / sigma)

    As PaymentRatio falls below mu, Phi increases toward 1,
    pushing RefiSMM toward MaxRefi.

    Parameters
    ----------
    current_rate  : float -- current market mortgage rate, monthly decimal
    wac           : float -- pool WAC, monthly decimal
    remaining_term: int   -- remaining months
    max_refi      : float -- maximum monthly refi speed (default 0.12 ~ 80% CPR)
    mu            : float -- payment ratio at which 50% of borrowers refinance (default 0.90)
    sigma         : float -- dispersion of borrower heterogeneity (default 0.10)

    Returns
    -------
    float -- RefiSMM as monthly decimal
    """
    pr = payment_ratio(current_rate, wac, remaining_term)
    return max_refi * norm.cdf((mu - pr) / sigma)