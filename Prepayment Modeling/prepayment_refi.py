"""Prepayment refinancing module for the Agency CMO project.

Purpose
-------
Convert simulated short-rate paths into a monthly refinancing SMM series,
while wiring in as much of the team's uploaded data as is actually usable.

This version uses:
- rate_paths.npz / rate_paths_full_pipeline.npz  -> simulated short-rate paths
  (expected under repo ``data_io/``; CSVs may live in ``data_io/market_data/``)
- treasury_rates_20260217.csv                   -> current Treasury anchor
- ust.csv                                       -> fallback historical Treasury curve
- CMO_BLOOMBERG_DATA.xlsx                       -> tranche/deal metadata, coupon,
                                                   balance, WAL, proxy term, CPR
- swaption_vols_20260217.csv                    -> loaded as calibration metadata
                                                   (not directly used in refi formula)

Loaders search cwd, this notebook's folder, repo root, ``data_io/``, then
``data_io/market_data/`` by filename.

Project role
------------
Jialin's piece is the refinancing component in Chapter 7:

    RefiSMM = f(payment_ratio)

Then the team can combine it with
- Noah's turnover(t)
- Clay's psi(t)
- optional passive-borrower scaling beta

to form the APD aggregate prepayment speed:

    lambda(t) = psi(t) * lambda_a(t) + (1 - psi(t)) * lambda_p(t)
    lambda_a(t) = TurnoverSMM(t) + RefiSMM(t)
    lambda_p(t) = TurnoverSMM(t) + beta * RefiSMM(t)

This file is intentionally modular so it can run now with mock inputs,
and later be plugged into the full pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

# Make __file__ behave in a notebook so helper functions work unchanged
if "__file__" not in globals():
    _nb_name = "prepayment_refi.ipynb"
    _cwd = Path.cwd()
    _found: Path | None = None
    for _d in [_cwd, *_cwd.parents]:
        for _p in (
            _d / "Prepayment Modeling" / "components" / _nb_name,
            _d / "components" / _nb_name,
            _d / _nb_name,
        ):
            if _p.is_file():
                _found = _p.resolve()
                break
        if _found is not None:
            break
    __file__ = str(_found if _found is not None else (_cwd / _nb_name))

# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class RefiParams:
    """Parameters for the refinancing S-curve.

    Attributes
    ----------
    mortgage_treasury_spread : float
        Additive spread over the 10Y Treasury used to anchor the current market
        mortgage rate level. Example: 150 bps.
    pass_through_beta : float
        Sensitivity used to translate short-rate path changes into market
        mortgage-rate changes. 1.0 means one-for-one with the simulated short rate.
    max_refi_smm : float
        Maximum monthly refinancing SMM reached when incentive is very strong.
    threshold : float
        Payment-ratio threshold for "meaningful" refinancing incentive.
        Example: 0.95 means borrowers become interested once the new payment is
        ~5% lower than the old payment.
    dispersion : float
        Controls how steep/smooth the S-curve is around the threshold.
    refi_term_months : int
        Term of the new refinanced mortgage. 360 months = 30 years.
    rate_floor : float
        Floor applied to current mortgage rates to avoid impossible negatives.
    beta_passive : float
        Passive-borrower multiplier used in APD aggregation.
    calibrate_max_refi_from_observed_cpr : bool
        If True, use Bloomberg 1m CPR as a rough anchor for max_refi_smm at t=0.
        This is only an approximation because the uploaded Excel is tranche-level,
        not a full pool-level collateral file.
    """

    mortgage_treasury_spread: float = 0.015
    pass_through_beta: float = 1.0
    max_refi_smm: float = 0.06
    threshold: float = 0.95
    dispersion: float = 0.10
    refi_term_months: int = 360
    rate_floor: float = 0.0001
    beta_passive: float = 0.25
    calibrate_max_refi_from_observed_cpr: bool = True


@dataclass
class BloombergTrancheData:
    """Parsed Bloomberg tranche/deal metadata from the uploaded Excel."""

    deal: str
    tranche_class: str
    tranche_sheet: str
    tranche_coupon: Optional[float]
    collateral_coupon: Optional[float]
    current_balance: Optional[float]
    original_balance: Optional[float]
    factor: Optional[float]
    wal_years: Optional[float]
    maturity: Optional[pd.Timestamp]
    next_pay: Optional[pd.Timestamp]
    dated_date: Optional[pd.Timestamp]
    current_cpr_1m: Optional[float]
    cpr_3m: Optional[float]
    cpr_6m: Optional[float]
    cpr_12m: Optional[float]
    cpr_life: Optional[float]
    current_psa_1m: Optional[float]
    proxy_remaining_term_months: Optional[int]


@dataclass
class ProjectDataBundle:
    """Convenience bundle of project inputs relevant for the refi module."""

    paths: np.ndarray
    t_grid: np.ndarray
    r0: float
    treasury_curve: Optional[pd.DataFrame]
    swaption_vols: Optional[pd.DataFrame]
    bloomberg: Optional[BloombergTrancheData]
    path_file: Path


# -----------------------------------------------------------------------------
# Helpers: parsing / loading
# -----------------------------------------------------------------------------


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        if isinstance(x, str):
            x = x.replace(",", "").replace("%", "").strip()
            if x == "":
                return None
        return float(x)
    except Exception:
        return None



def _extract_first(pattern: str, text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = re.search(pattern, text)
    return m.group(1) if m else None



def _parse_date_str(date_str: Optional[str]) -> Optional[pd.Timestamp]:
    if not date_str:
        return None
    try:
        return pd.to_datetime(date_str)
    except Exception:
        return None



def _months_between(start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> Optional[int]:
    if start is None or end is None:
        return None
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day >= start.day:
        months += 0
    return max(months, 1)


def _project_root_from_notebook() -> Path:
    """Repository root (parent of ``Prepayment Modeling``)."""
    p = Path(__file__).resolve()
    return p.parents[2]


def _resolve_data_file(path: str | Path) -> Path:
    """Resolve data files from cwd, notebook folder, repo root, ``data_io/``, then ``data_io/market_data/``.

    Rate ``.npz`` files are typically in ``data_io/``; Treasury/swaption CSVs may be under ``market_data/``.
    """
    path = Path(path)
    if path.is_file():
        return path
    name = path.name
    nb_dir = Path(__file__).resolve().parent
    try:
        root = _project_root_from_notebook()
    except IndexError:
        root = nb_dir
    roots = [
        Path.cwd(),
        nb_dir,
        nb_dir.parent,
        root,
        root / "data_io",
        root / "data_io" / "market_data",
    ]
    seen: set[str] = set()
    for r in roots:
        key = str(r.resolve())
        if key in seen:
            continue
        seen.add(key)
        cand = (r / name).resolve()
        if cand.is_file():
            return cand
    return path


def _synthetic_rate_paths(
    n_paths: int = 500,
    n_steps: int = 361,
    r0: float = 0.04,
    vol: float = 0.008,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Deterministic short-rate paths when no ``.npz`` is available (local demo only)."""
    rng = np.random.default_rng(seed)
    shock = rng.standard_normal((n_paths, n_steps - 1)) * (vol / np.sqrt(12.0))
    paths = np.empty((n_paths, n_steps), dtype=np.float64)
    paths[:, 0] = r0
    paths[:, 1:] = r0 + np.cumsum(shock, axis=1)
    paths = np.clip(paths, 1e-4, 0.5)
    t_grid = np.arange(n_steps, dtype=np.float64)
    return {"paths": paths, "t_grid": t_grid}


def load_rate_paths(npz_path: str | Path = "rate_paths.npz") -> Dict[str, np.ndarray]:
    """Load rate paths from a project .npz file, or synthesize paths if the file is absent."""
    npz_path = _resolve_data_file(npz_path)
    if not npz_path.is_file():
        warnings.warn(
            f"Rate path file not found ({npz_path.name}); using synthetic short-rate paths for demo.",
            stacklevel=2,
        )
        return _synthetic_rate_paths()
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}



def load_treasury_curve(
    treasury_csv: str | Path = "treasury_rates_20260217.csv",
    ust_csv: str | Path = "ust.csv",
) -> Optional[pd.DataFrame]:
    """Load current Treasury curve, falling back to the latest row of ust.csv."""
    treasury_csv = _resolve_data_file(treasury_csv)

    if treasury_csv.is_file():
        df = pd.read_csv(treasury_csv)
        return df.sort_values("maturity_years").reset_index(drop=True)

    ust_csv = _resolve_data_file(ust_csv)
    if not ust_csv.is_file():
        return None

    ust = pd.read_csv(ust_csv)
    last = ust.dropna(how="all").iloc[-1]
    mapping = [
        ("1M", 1 / 12, "BC_1MONTH"),
        ("3M", 0.25, "BC_3MONTH"),
        ("6M", 0.50, "BC_6MONTH"),
        ("1Y", 1.0, "BC_1YEAR"),
        ("2Y", 2.0, "BC_2YEAR"),
        ("3Y", 3.0, "BC_3YEAR"),
        ("5Y", 5.0, "BC_5YEAR"),
        ("7Y", 7.0, "BC_7YEAR"),
        ("10Y", 10.0, "BC_10YEAR"),
        ("20Y", 20.0, "BC_20YEAR"),
        ("30Y", 30.0, "BC_30YEAR"),
    ]
    rows = []
    for tenor, maturity, col in mapping:
        val = _safe_float(last.get(col))
        if val is not None and val > 0:
            rows.append({"tenor": tenor, "maturity_years": maturity, "rate": val / 100.0})
    return pd.DataFrame(rows)



def load_swaption_vols(csv_path: str | Path = "swaption_vols_20260217.csv") -> Optional[pd.DataFrame]:
    """Load swaption vol file for metadata / provenance."""
    csv_path = _resolve_data_file(csv_path)
    if not csv_path.is_file():
        return None
    return pd.read_csv(csv_path)



def parse_random_remic_sheet(xlsx_path: str | Path = "CMO_BLOOMBERG_DATA.xlsx") -> pd.DataFrame:
    """Parse the structured Security Finder table from the Random REMIC sheet."""
    xlsx_path = _resolve_data_file(xlsx_path)

    raw = pd.read_excel(xlsx_path, sheet_name="Random REMIC", header=None)
    header = raw.iloc[8, 1:16].tolist()
    data = raw.iloc[10:, 1:16].copy()
    data.columns = header
    data = data.dropna(how="all")
    data = data.rename(columns={"Cpn": "coupon", "Orig Amt": "original_amount", "Class": "class", "Series": "series", "Type": "type"})
    if "coupon" in data:
        data["coupon"] = pd.to_numeric(data["coupon"], errors="coerce") / 100.0
    if "original_amount" in data:
        data["original_amount"] = pd.to_numeric(data["original_amount"], errors="coerce")
    return data.reset_index(drop=True)



def parse_bloomberg_tranche_sheet(
    xlsx_path: str | Path = "CMO_BLOOMBERG_DATA.xlsx",
    tranche_sheet: str = "A Tranche",
) -> BloombergTrancheData:
    """Parse one Bloomberg tranche description sheet.

    This Excel is tranche/deal-level, not a full collateral pool file. We extract
    what is actually usable for the refinancing module:
    - tranche coupon and balances
    - collateral coupon proxy from the security header (e.g. '100% FNCL 6.5 N')
    - current CPR / PSA snapshot
    - WAL
    - proxy remaining term from next pay to maturity
    """
    xlsx_path = _resolve_data_file(xlsx_path)

    raw = pd.read_excel(xlsx_path, sheet_name=tranche_sheet, header=None)

    header_1 = raw.iloc[4, 1] if raw.shape[0] > 4 and raw.shape[1] > 1 else ""
    header_2 = raw.iloc[5, 1] if raw.shape[0] > 5 and raw.shape[1] > 1 else ""
    row_8 = raw.iloc[8, 1] if raw.shape[0] > 8 and raw.shape[1] > 1 else ""
    row_12 = raw.iloc[12, 1] if raw.shape[0] > 12 and raw.shape[1] > 1 else ""
    row_13 = raw.iloc[13, 1] if raw.shape[0] > 13 and raw.shape[1] > 1 else ""
    row_14 = raw.iloc[14, 1] if raw.shape[0] > 14 and raw.shape[1] > 1 else ""
    row_18 = raw.iloc[18, 1] if raw.shape[0] > 18 and raw.shape[1] > 1 else ""

    deal = _extract_first(r"([A-Z]{3}\s+\d{4}-\d+)", str(header_1)) or "UNKNOWN DEAL"
    tranche_class = _extract_first(r"Class\s+([A-Z0-9]+)", str(row_8)) or tranche_sheet.replace(" Tranche", "")

    collateral_coupon = _safe_float(_extract_first(r"FNCL\s+([0-9.]+)", str(header_2)))
    if collateral_coupon is not None:
        collateral_coupon /= 100.0

    maturity = _parse_date_str(_extract_first(r"Mty\s+([0-9]{2}/[0-9]{2}/[0-9]{4})", str(row_8)))
    next_pay = _parse_date_str(_extract_first(r"Next Pay\s+([0-9/]+)", str(row_12)))
    dated_date = _parse_date_str(_extract_first(r"Dated Date\s+([0-9/]+)", str(row_18)))

    bal_matches = re.findall(r"Bal USD\s*([\d,]+)", str(row_12))
    current_balance = _safe_float(bal_matches[0]) if len(bal_matches) >= 1 else None
    original_balance = _safe_float(bal_matches[1]) if len(bal_matches) >= 2 else None

    factor = _safe_float(_extract_first(r"Fct \([^)]*\)\s*([0-9.]+)", str(row_13)))
    wal_years = _safe_float(_extract_first(r"WAL\s*([0-9.]+)Yrs", str(row_13)))
    tranche_coupon = _safe_float(_extract_first(r"Cpn \([^)]*\)\s*([0-9.]+)%", str(row_14)))
    if tranche_coupon is not None:
        tranche_coupon /= 100.0

    current_cpr_1m = _safe_float(raw.iloc[12, 3] if raw.shape[0] > 12 and raw.shape[1] > 3 else None)
    cpr_3m = _safe_float(raw.iloc[13, 3] if raw.shape[0] > 13 and raw.shape[1] > 3 else None)
    cpr_6m = _safe_float(raw.iloc[14, 3] if raw.shape[0] > 14 and raw.shape[1] > 3 else None)
    cpr_12m = _safe_float(raw.iloc[15, 3] if raw.shape[0] > 15 and raw.shape[1] > 3 else None)
    cpr_life = _safe_float(raw.iloc[16, 3] if raw.shape[0] > 16 and raw.shape[1] > 3 else None)

    current_psa_1m = _safe_float(raw.iloc[12, 4] if raw.shape[0] > 12 and raw.shape[1] > 4 else None)

    proxy_remaining_term_months = _months_between(next_pay, maturity)

    return BloombergTrancheData(
        deal=deal,
        tranche_class=tranche_class,
        tranche_sheet=tranche_sheet,
        tranche_coupon=tranche_coupon,
        collateral_coupon=collateral_coupon,
        current_balance=current_balance,
        original_balance=original_balance,
        factor=factor,
        wal_years=wal_years,
        maturity=maturity,
        next_pay=next_pay,
        dated_date=dated_date,
        current_cpr_1m=current_cpr_1m,
        cpr_3m=cpr_3m,
        cpr_6m=cpr_6m,
        cpr_12m=cpr_12m,
        cpr_life=cpr_life,
        current_psa_1m=current_psa_1m,
        proxy_remaining_term_months=proxy_remaining_term_months,
    )



def load_project_data_bundle(
    base_dir: str | Path | None = None,
    tranche_sheet: str = "A Tranche",
    path_file: str = "rate_paths.npz",
) -> ProjectDataBundle:
    """Load the project files relevant to the refi module."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    base_dir = Path(base_dir)

    npz_logical = base_dir / path_file
    npz_resolved = _resolve_data_file(npz_logical)
    path_data = load_rate_paths(npz_logical)
    path_file_out = npz_resolved if npz_resolved.is_file() else npz_logical

    treasury = load_treasury_curve(base_dir / "treasury_rates_20260217.csv", base_dir / "ust.csv")
    swaption = load_swaption_vols(base_dir / "swaption_vols_20260217.csv")

    bloomberg = None
    xlsx = _resolve_data_file(base_dir / "CMO_BLOOMBERG_DATA.xlsx")
    if xlsx.is_file():
        try:
            bloomberg = parse_bloomberg_tranche_sheet(xlsx, tranche_sheet=tranche_sheet)
        except Exception:
            bloomberg = None

    _r0 = path_data.get("r0", path_data["paths"][0, 0])
    r0_scalar = float(np.asarray(_r0).reshape(-1)[0])

    return ProjectDataBundle(
        paths=path_data["paths"],
        t_grid=path_data["t_grid"],
        r0=r0_scalar,
        treasury_curve=treasury,
        swaption_vols=swaption,
        bloomberg=bloomberg,
        path_file=path_file_out,
    )


# -----------------------------------------------------------------------------
# Core refinancing math
# -----------------------------------------------------------------------------


def level_payment(annual_rate: float | np.ndarray, term_months: int | np.ndarray, balance: float = 1.0) -> np.ndarray:
    """Monthly payment for a standard fully amortizing mortgage."""
    annual_rate = np.asarray(annual_rate, dtype=float)
    term_months = np.asarray(term_months, dtype=float)

    term_months = np.maximum(term_months, 1.0)
    monthly_rate = annual_rate / 12.0
    payment = np.empty_like(monthly_rate, dtype=float)

    zero_mask = np.isclose(monthly_rate, 0.0)
    nonzero_mask = ~zero_mask

    payment[zero_mask] = balance / term_months[zero_mask]

    r = monthly_rate[nonzero_mask]
    n = term_months[nonzero_mask]
    payment[nonzero_mask] = balance * r / (1.0 - (1.0 + r) ** (-n))
    return payment



def market_mortgage_rate_from_paths(
    short_rate_paths: np.ndarray,
    r0: float,
    treasury_curve: Optional[pd.DataFrame],
    params: RefiParams,
) -> np.ndarray:
    """Map short-rate paths into current market mortgage-rate paths.

    We anchor the level to:
        current 10Y Treasury + mortgage_treasury_spread

    and then let it move with the simulated short-rate changes:
        mort_rate(t) = mort_rate(0) + beta * (r(t) - r0)

    This uses uploaded Treasury data while still respecting the team's simulated
    Hull-White paths.
    """
    short_rate_paths = np.asarray(short_rate_paths, dtype=float)

    if treasury_curve is not None and "maturity_years" in treasury_curve and "rate" in treasury_curve:
        ten_year = treasury_curve.loc[np.isclose(treasury_curve["maturity_years"], 10.0), "rate"]
        benchmark_10y = float(ten_year.iloc[0]) if not ten_year.empty else float(treasury_curve.iloc[-1]["rate"])
    else:
        benchmark_10y = float(r0)

    market_rate_0 = benchmark_10y + params.mortgage_treasury_spread
    delta_r = short_rate_paths - float(r0)
    mortgage_rates = market_rate_0 + params.pass_through_beta * delta_r
    return np.maximum(mortgage_rates, params.rate_floor)



def payment_ratio(
    current_note_rate: float | np.ndarray,
    market_mortgage_rate: float | np.ndarray,
    remaining_term_months: int | np.ndarray,
    refi_term_months: int = 360,
) -> np.ndarray:
    """Compute payment ratio = new payment / current payment."""
    current_note_rate = np.asarray(current_note_rate, dtype=float)
    market_mortgage_rate = np.asarray(market_mortgage_rate, dtype=float)
    remaining_term_months = np.asarray(remaining_term_months, dtype=float)

    old_payment = level_payment(current_note_rate, remaining_term_months, balance=1.0)
    new_payment = level_payment(
        market_mortgage_rate,
        np.full_like(remaining_term_months, refi_term_months, dtype=float),
        balance=1.0,
    )
    return new_payment / old_payment



def refinancing_smm_from_payment_ratio(
    pay_ratio: float | np.ndarray,
    max_refi_smm: float = 0.06,
    threshold: float = 0.95,
    dispersion: float = 0.03,
    burnout_multiplier: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Convert payment ratio into monthly refinancing SMM using an S-curve."""
    pay_ratio = np.asarray(pay_ratio, dtype=float)
    burnout_multiplier = np.asarray(burnout_multiplier, dtype=float)

    z = (threshold - pay_ratio) / dispersion
    refi_smm = max_refi_smm * norm.cdf(z) * burnout_multiplier
    return np.clip(refi_smm, 0.0, max_refi_smm)



def calibrate_max_refi_from_observed_cpr(
    observed_cpr_pct: Optional[float],
    pay_ratio_0: float,
    turnover_smm_0: float,
    psi_0: float,
    beta_passive: float,
    threshold: float,
    dispersion: float,
    default_max_refi_smm: float,
) -> float:
    """Back out max_refi_smm from observed 1m CPR as a rough t=0 anchor.

    Since the uploaded Bloomberg file is tranche-level rather than full pool-level,
    this should be treated as a rough calibration aid / sanity check, not a final
    production calibration.
    """
    if observed_cpr_pct is None or np.isnan(observed_cpr_pct):
        return default_max_refi_smm

    observed_cpr = observed_cpr_pct / 100.0
    observed_smm = 1.0 - (1.0 - observed_cpr) ** (1.0 / 12.0)

    responsive_weight = beta_passive + (1.0 - beta_passive) * psi_0
    cdf0 = float(norm.cdf((threshold - pay_ratio_0) / dispersion))
    if cdf0 <= 1e-8 or responsive_weight <= 1e-8:
        return default_max_refi_smm

    implied_refi_smm_0 = max((observed_smm - turnover_smm_0) / responsive_weight, 0.0)
    implied_max = implied_refi_smm_0 / cdf0

    # Clamp to a reasonable monthly range.
    implied_max = float(np.clip(implied_max, 0.001, 0.20))
    return implied_max



def refinancing_smm_paths(
    short_rate_paths: np.ndarray,
    current_note_rate: float,
    remaining_term_months: int,
    params: RefiParams | None = None,
    burnout_multiplier: Optional[np.ndarray] = None,
    treasury_curve: Optional[pd.DataFrame] = None,
    r0: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Vectorized refinancing SMM for all paths."""
    if params is None:
        params = RefiParams()

    short_rate_paths = np.asarray(short_rate_paths, dtype=float)
    n_paths, n_steps = short_rate_paths.shape
    if r0 is None:
        r0 = float(short_rate_paths[0, 0])

    rem_term = np.maximum(remaining_term_months - np.arange(n_steps), 1)
    rem_term_2d = np.broadcast_to(rem_term, (n_paths, n_steps))

    market_mortgage_rate = market_mortgage_rate_from_paths(
        short_rate_paths=short_rate_paths,
        r0=float(r0),
        treasury_curve=treasury_curve,
        params=params,
    )

    pay_ratio = payment_ratio(
        current_note_rate=np.full((n_paths, n_steps), current_note_rate),
        market_mortgage_rate=market_mortgage_rate,
        remaining_term_months=rem_term_2d,
        refi_term_months=params.refi_term_months,
    )

    if burnout_multiplier is None:
        burnout_multiplier = np.ones((n_paths, n_steps))
    elif np.ndim(burnout_multiplier) == 1:
        burnout_multiplier = np.broadcast_to(burnout_multiplier, (n_paths, n_steps))

    refi_smm = refinancing_smm_from_payment_ratio(
        pay_ratio,
        max_refi_smm=params.max_refi_smm,
        threshold=params.threshold,
        dispersion=params.dispersion,
        burnout_multiplier=burnout_multiplier,
    )

    return {
        "market_mortgage_rate": market_mortgage_rate,
        "payment_ratio": pay_ratio,
        "remaining_term_months": rem_term_2d.astype(int),
        "refi_smm": refi_smm,
    }


# -----------------------------------------------------------------------------
# APD aggregation and team mocks
# -----------------------------------------------------------------------------


def apd_aggregate_smm(
    refi_smm: np.ndarray,
    turnover_smm: np.ndarray,
    psi_t: np.ndarray,
    beta: float = 0.25,
) -> np.ndarray:
    """Combine team components into APD aggregate SMM."""
    refi_smm = np.asarray(refi_smm, dtype=float)
    turnover_smm = np.asarray(turnover_smm, dtype=float)
    psi_t = np.asarray(psi_t, dtype=float)

    responsive_weight = beta + (1.0 - beta) * psi_t
    smm = turnover_smm + responsive_weight * refi_smm
    return np.clip(smm, 0.0, 1.0)



# -----------------------------------------------------------------------------
# High-level convenience: use the uploaded project files directly
# -----------------------------------------------------------------------------


def infer_refi_inputs_from_bundle(bundle: ProjectDataBundle) -> Dict[str, Any]:
    """Infer the best available refi inputs from uploaded files.

    Priority order:
    1) Bloomberg collateral coupon proxy from the security header (best available)
    2) Bloomberg tranche coupon
    3) Hardcoded fallback

    Remaining term:
    1) months from Bloomberg next-pay to maturity (proxy term)
    2) 360-month fallback
    """
    bloom = bundle.bloomberg
    current_note_rate = 0.055
    remaining_term_months = 360
    observed_cpr_1m = None

    if bloom is not None:
        if bloom.collateral_coupon is not None:
            current_note_rate = bloom.collateral_coupon
        elif bloom.tranche_coupon is not None:
            current_note_rate = bloom.tranche_coupon

        if bloom.proxy_remaining_term_months is not None:
            remaining_term_months = bloom.proxy_remaining_term_months

        observed_cpr_1m = bloom.current_cpr_1m

    return {
        "current_note_rate": float(current_note_rate),
        "remaining_term_months": int(remaining_term_months),
        "observed_cpr_1m": observed_cpr_1m,
    }


# -----------------------------------------------------------------------------
# Demo / script entrypoint
# -----------------------------------------------------------------------------


def demo_run(tranche_sheet: str = "A Tranche", path_file: str = "rate_paths.npz") -> None:
    """Quick demo using the team's uploaded files.

    Loads all data files, infers pool inputs from Bloomberg, and runs
    refinancing_smm_paths directly. Prints a 12-month summary table.
    """
    bundle = load_project_data_bundle(tranche_sheet=tranche_sheet, path_file=path_file)
    inputs = infer_refi_inputs_from_bundle(bundle)
    params = RefiParams()

    # First pass to get t=0 payment ratio
    prelim = refinancing_smm_paths(
        short_rate_paths=bundle.paths,
        current_note_rate=inputs["current_note_rate"],
        remaining_term_months=inputs["remaining_term_months"],
        params=params,
        treasury_curve=bundle.treasury_curve,
        r0=bundle.r0,
    )
    pay_ratio_0 = float(prelim["payment_ratio"].mean(axis=0)[0])
    params.max_refi_smm = calibrate_max_refi_from_observed_cpr(
        observed_cpr_pct=inputs["observed_cpr_1m"],
        pay_ratio_0=pay_ratio_0,
        turnover_smm_0=0.002022,   # turnover_smm(step=1) from prepayment_turnover
        psi_0=0.70,
        beta_passive=params.beta_passive,
        threshold=params.threshold,
        dispersion=params.dispersion,
        default_max_refi_smm=params.max_refi_smm,
    )

    # Main pass - uses calibrated max_refi_smm
    result = refinancing_smm_paths(
        short_rate_paths=bundle.paths,
        current_note_rate=inputs["current_note_rate"],
        remaining_term_months=inputs["remaining_term_months"],
        params=params,
        treasury_curve=bundle.treasury_curve,
        r0=bundle.r0,
    )

    avg_pr   = result["payment_ratio"].mean(axis=0)
    avg_refi = result["refi_smm"].mean(axis=0)
    bloom    = bundle.bloomberg

    print("=" * 78)
    print("PREPAYMENT REFINANCING DEMO")
    print("=" * 78)
    print(f"Rate path file       : {bundle.path_file.name}")
    print(f"Loaded paths         : {bundle.paths.shape[0]:,} x {bundle.paths.shape[1]}")
    print(f"Initial short rate r0: {bundle.r0:.4%}")

    if bundle.treasury_curve is not None:
        ten_year = bundle.treasury_curve.loc[
            np.isclose(bundle.treasury_curve["maturity_years"], 10.0), "rate"
        ]
        if not ten_year.empty:
            print(f"Treasury 10Y anchor  : {float(ten_year.iloc[0]):.4%}")

    if bloom is not None:
        print(f"\nDeal / tranche       : {bloom.deal} / {bloom.tranche_class} ({bloom.tranche_sheet})")
        print(f"Collateral WAC       : {bloom.collateral_coupon if bloom.collateral_coupon is not None else float('nan'):.4%}")
        print(f"Tranche coupon       : {bloom.tranche_coupon if bloom.tranche_coupon is not None else float('nan'):.4%}")
        print(f"1m CPR / PSA         : {bloom.current_cpr_1m} / {bloom.current_psa_1m}")
        print(f"Proxy remaining term : {bloom.proxy_remaining_term_months} months")

    print(f"\nInputs used")
    print(f"  Current note rate : {inputs['current_note_rate']:.4%}")
    print(f"  Remaining term    : {inputs['remaining_term_months']} months")
    print(f"  Max refi SMM      : {params.max_refi_smm:.4%}")
    print(f"  Threshold         : {params.threshold:.4f}")
    print(f"  Dispersion        : {params.dispersion:.4f}")

    print(f"\nFirst 12 months (path averages)")
    print("month | pay_ratio | refi_smm")
    for m in range(min(12, len(bundle.t_grid))):
        print(f"{m+1:5d} | {avg_pr[m]:9.4f} | {avg_refi[m]:8.4%}")


if __name__ == "__main__":
    demo_run(tranche_sheet="A Tranche", path_file="rate_paths.npz")