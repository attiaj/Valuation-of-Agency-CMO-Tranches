"""
apd.py

Active/Passive Decomposition (APD) model.
Manages psi(t) -- the proportion of active borrowers remaining in the pool --
and computes the APD aggregate prepayment rate lambda(t).

Reference: Levin (2001), Davidson & Levin (2014) Chapter 7
"""

#init_psi is bypassed in our current CMO, we use 0.65 as derived from the text
def init_psi(pool_data: dict = None, default: float = 0.70) -> float:
    """
    Initialize psi(0): the starting active borrower fraction.

    Higher credit quality pools (high FICO, low LTV) tend to have
    a larger active fraction since more borrowers are able to refinance.

    Parameters
    ----------
    pool_data : dict (optional) with keys 'fico', 'ltv'
    default   : float -- fallback if no pool data provided (default 0.70)

    Returns
    -------
    float -- psi(0) in (0, 1)
    """
    if pool_data is None:
        return default

    psi = default

    fico = pool_data.get('fico')
    ltv  = pool_data.get('ltv')

    if fico is not None:
        if fico >= 750:
            psi = min(psi + 0.05, 0.90)
        elif fico < 660:
            psi = max(psi - 0.10, 0.40)

    if ltv is not None:
        if ltv >= 90:
            psi = max(psi - 0.10, 0.40)
        elif ltv <= 75:
            psi = min(psi + 0.05, 0.90)

    return psi


def lambda_active(turnover_smm: float, refi_smm: float) -> float:
    """
    Prepayment rate for the active sub-population.

    lambda_a(t) = TurnoverSMM(t) + RefiSMM(t)

    Parameters
    ----------
    turnover_smm : float -- turnover component (monthly decimal)
    refi_smm     : float -- refinancing component (monthly decimal)

    Returns
    -------
    float -- lambda_a(t)
    """
    return turnover_smm + refi_smm


def lambda_passive(turnover_smm: float, refi_smm: float, beta: float = 0.25) -> float:
    """
    Prepayment rate for the passive sub-population.

    lambda_p(t) = TurnoverSMM(t) + beta * RefiSMM(t)

    Parameters
    ----------
    turnover_smm : float -- turnover component (monthly decimal)
    refi_smm     : float -- refinancing component (monthly decimal)
    beta         : float -- passive responsiveness to refi incentive, beta < 1 (default 0.25)

    Returns
    -------
    float -- lambda_p(t)
    """
    return turnover_smm + beta * refi_smm


def aggregate_lambda(psi: float, la: float, lp: float) -> float:
    """
    APD aggregate prepayment rate.

    lambda(t) = psi(t) * lambda_a(t) + (1 - psi(t)) * lambda_p(t)

    Parameters
    ----------
    psi : float -- current active fraction psi(t)
    la  : float -- lambda_a(t)
    lp  : float -- lambda_p(t)

    Returns
    -------
    float -- lambda(t), the pool-level SMM
    """
    return psi * la + (1 - psi) * lp


def update_psi(psi: float, la: float, lam: float) -> float:
    """
    Update active fraction for next period.

    psi(t+1) = psi(t) * (1 - lambda_a(t)) / (1 - lambda(t))

    Active borrowers prepay faster than pool average, so their
    share of the surviving balance shrinks over time. This is
    what produces burnout endogenously.

    Parameters
    ----------
    psi : float -- current psi(t)
    la  : float -- lambda_a(t)
    lam : float -- lambda(t), pool-level SMM

    Returns
    -------
    float -- psi(t+1)
    """
    denominator = 1.0 - lam
    if denominator <= 0:
        return 0.0
    return psi * (1.0 - la) / denominator