"""
components/enhanced.py

Enhanced variable multipliers derived from Bloomberg pool-level data.
Produces M_T (turnover multiplier) and M_R (refinancing multiplier)
that scale the base TurnoverSMM and RefiSMM for pool characteristics.

Populate get_multipliers() with logic derived from your Bloomberg data
(FICO, LTV, SATO, loan size, etc.).
"""


def _refi_multiplier(fico: float = None, ltv: float = None, sato: float = None) -> float:
    """
    Compute M_R: scaling factor on RefiSMM based on credit and collateral.

    Higher FICO  -> higher M_R (easier to qualify for refi)
    Higher LTV   -> lower M_R  (harder to refi, closer to underwater)
    Higher SATO  -> lower M_R  (borrower paid penalty rate, impaired credit)

    Parameters
    ----------
    fico : float -- weighted average FICO score of pool (e.g. 720)
    ltv  : float -- weighted average LTV of pool (e.g. 80.0 for 80%)
    sato : float -- spread at origination in bps (e.g. 50)

    Returns
    -------
    float -- M_R multiplier
    """
    m_r = 1.0

    # FICO adjustment
    if fico is not None:
        if fico >= 750:
            m_r *= 1.20
        elif fico >= 720:
            m_r *= 1.05
        elif fico >= 680:
            m_r *= 0.90
        else:
            m_r *= 0.50

    # LTV adjustment
    if ltv is not None:
        if ltv <= 75:
            m_r *= 1.10
        elif ltv <= 85:
            m_r *= 1.00
        elif ltv <= 95:
            m_r *= 0.70
        else:
            m_r *= 0.40

    # SATO adjustment (bps)
    if sato is not None:
        if sato <= 25:
            m_r *= 1.05
        elif sato <= 75:
            m_r *= 0.90
        else:
            m_r *= 0.70

    return m_r


def _turnover_multiplier(ltv: float = None, loan_size: float = None) -> float:
    """
    Compute M_T: scaling factor on TurnoverSMM.

    Higher LTV       -> lower M_T (less mobility, harder to sell)
    Larger loan size -> slight positive effect on turnover

    Parameters
    ----------
    ltv       : float -- weighted average LTV (e.g. 80.0)
    loan_size : float -- weighted average loan size in dollars (e.g. 300000)

    Returns
    -------
    float -- M_T multiplier
    """
    m_t = 1.0

    if ltv is not None:
        if ltv <= 75:
            m_t *= 1.10
        elif ltv <= 85:
            m_t *= 1.00
        elif ltv <= 95:
            m_t *= 0.85
        else:
            m_t *= 0.60

    return m_t


def get_multipliers(pool_data: dict) -> tuple:
    """
    Return (M_T, M_R) multipliers from Bloomberg pool-level data.

    Parameters
    ----------
    pool_data : dict with any of the following keys:
        'fico'      : float -- weighted avg FICO
        'ltv'       : float -- weighted avg LTV (0-100 scale)
        'sato'      : float -- spread at origination in bps
        'loan_size' : float -- weighted avg loan balance in dollars

    Returns
    -------
    (M_T, M_R) : tuple of floats
    """
    fico      = pool_data.get('fico')
    ltv       = pool_data.get('ltv')
    sato      = pool_data.get('sato')
    loan_size = pool_data.get('loan_size')

    m_r = _refi_multiplier(fico=fico, ltv=ltv, sato=sato)
    m_t = _turnover_multiplier(ltv=ltv, loan_size=loan_size)

    return m_t, m_r