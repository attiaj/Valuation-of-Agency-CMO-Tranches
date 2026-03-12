"""
utils.py

Shared utility functions for the prepayment model.
"""


def smm_to_cpr(smm: float) -> float:
    """
    Convert Single Monthly Mortality to Conditional Prepayment Rate (annualized).

    CPR = 1 - (1 - SMM)^12

    Parameters
    ----------
    smm : float -- monthly prepayment rate as decimal

    Returns
    -------
    float -- annualized CPR as decimal
    """
    return 1.0 - (1.0 - smm) ** 12


def cpr_to_smm(cpr: float) -> float:
    """
    Convert Conditional Prepayment Rate (annualized) to SMM.

    SMM = 1 - (1 - CPR)^(1/12)

    Parameters
    ----------
    cpr : float -- annualized CPR as decimal

    Returns
    -------
    float -- monthly SMM as decimal
    """
    return 1.0 - (1.0 - cpr) ** (1.0 / 12.0)


def annual_to_monthly_rate(annual_rate: float) -> float:
    """
    Convert an annual rate (decimal) to a monthly rate.

    Parameters
    ----------
    annual_rate : float -- e.g. 0.06 for 6%

    Returns
    -------
    float -- monthly rate
    """
    return annual_rate / 12.0


def scheduled_principal(balance: float, monthly_rate: float, remaining_term: int) -> float:
    """
    Compute the scheduled principal payment for the current period.

    Parameters
    ----------
    balance        : float -- current outstanding balance
    monthly_rate   : float -- monthly interest rate (decimal)
    remaining_term : int   -- months remaining

    Returns
    -------
    float -- scheduled principal payment
    """
    if monthly_rate == 0:
        return balance / remaining_term
    pmt = balance * monthly_rate / (1 - (1 + monthly_rate) ** (-remaining_term))
    interest = balance * monthly_rate
    return pmt - interest