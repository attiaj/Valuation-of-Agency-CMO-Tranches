"""
components/turnover.py

Turnover component of the prepayment model.
Captures prepayments due to housing turnover (moving, death, etc.)
independent of refinancing incentive.

Output: TurnoverSMM — monthly prepayment rate as a decimal (e.g. 0.005)
"""

# Seasonal multipliers by calendar month (1=Jan ... 12=Dec)
SEASONALITY = {
    1:  0.8,
    2:  0.8,
    3:  0.9,
    4:  1.0,
    5:  1.1,
    6:  1.2,
    7:  1.2,
    8:  1.1,
    9:  1.0,
    10: 0.9,
    11: 0.8,
    12: 0.7,
}


def age_factor(pool_age: int) -> float:
    """
    Seasoning ramp: new loans prepay slowly, ramping up over first 30 months.

    Parameters
    ----------
    pool_age : int
        Age of pool in months since origination.

    Returns
    -------
    float in [0, 1]
    """
    return min(1.0, pool_age / 30.0)


def seasonal_factor(month: int) -> float:
    """
    Seasonal multiplier for calendar month.

    Parameters
    ----------
    month : int
        Calendar month (1-12).

    Returns
    -------
    float
    """
    return SEASONALITY[month]


def turnover_smm(pool_age: int, month: int, base_turnover: float = 0.005) -> float:
    """
    Compute the turnover component of SMM.

    TurnoverSMM(t) = BaseTurnover * AgeFactor(t) * Seasonality(m)

    Parameters
    ----------
    pool_age      : int   -- months since origination
    month         : int   -- calendar month (1-12)
    base_turnover : float -- baseline monthly turnover rate (default 0.005 = 6% CPR annualized)

    Returns
    -------
    float -- TurnoverSMM as monthly decimal
    """
    return base_turnover * age_factor(pool_age) * seasonal_factor(month)