# hull_white/calibration.py

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt  # type: ignore

def hw_swaption_vol(a, sigma, T_opt, T_swap):
    """
    Compute implied Black volatility for a swaption in Hull-White model
    
    This is the volatility that would be used in Black's formula to price
    the swaption, given Hull-White parameters (a, sigma)
    
    Args:
        a: Mean reversion speed
        sigma: Hull-White volatility parameter
        T_opt: Option expiry in years (time to swaption expiration)
        T_swap: Swap tenor in years (length of underlying swap)
        
    Returns:
        Implied Black volatility (annualized)
    """
    # Avoid division by zero
    if a < 1e-6:
        # Ho-Lee limit (a -> 0)
        B = 1.0
        var_r = sigma**2 * T_opt
    else:
        # B(T) factor: sensitivity of bond price to short rate
        # This measures how much the swap rate moves with the short rate
        B = (1 - np.exp(-a * T_swap)) / (a * T_swap)
        
        # Variance of short rate over option life
        var_r = (sigma**2 / (2*a)) * (1 - np.exp(-2*a*T_opt))
    
    # Implied Black vol for the swap rate
    # This comes from the Hull-White analytical swaption formula
    sigma_black = np.sqrt(var_r / T_opt) * B
    
    return sigma_black


def price_swaption_black(swap_rate, strike, vol, T_opt, notional=1.0):
    """
    Price a payer swaption using Black's formula
    
    Args:
        swap_rate: Forward swap rate (e.g., 0.045 for 4.5%)
        strike: Strike rate (e.g., 0.045 for ATM)
        vol: Black volatility (e.g., 0.0085 for 85 bps)
        T_opt: Option expiry in years
        notional: Notional amount (default 1.0)
        
    Returns:
        Swaption price as % of notional
    """
    if vol <= 0 or T_opt <= 0:
        return 0.0
    
    # Black's formula (simplified for ATM and annuity = 1)
    d1 = (np.log(swap_rate / strike) + 0.5 * vol**2 * T_opt) / (vol * np.sqrt(T_opt))
    d2 = d1 - vol * np.sqrt(T_opt)
    
    price = notional * (swap_rate * norm.cdf(d1) - strike * norm.cdf(d2))
    
    return price


def calibration_objective(params, yield_curve, swaption_data):
    """
    Objective function for Hull-White calibration
    
    Minimizes sum of squared errors between model and market swaption vols
    
    Args:
        params: [a, sigma] parameters to calibrate
        yield_curve: YieldCurve object with forward rates
        swaption_data: DataFrame with columns:
            - option_expiry: years
            - swap_tenor: years  
            - implied_vol: market vol (decimal, e.g., 0.0085 for 85 bps)
            
    Returns:
        Sum of squared volatility errors
    """
    a, sigma = params
    
    # Sanity checks
    if a <= 0 or a > 0.5:  # Mean reversion should be positive and reasonable
        return 1e10
    if sigma <= 0 or sigma > 0.05:  # Vol should be positive and < 500 bps
        return 1e10
    
    total_error = 0.0
    
    for _, row in swaption_data.iterrows():
        T_opt = row['option_expiry']
        T_swap = row['swap_tenor']
        market_vol = row['implied_vol']
        
        # Compute model implied vol
        model_vol = hw_swaption_vol(a, sigma, T_opt, T_swap)
        
        # Squared error in volatility
        error = (model_vol - market_vol)**2
        total_error += error
    
    return total_error


def calibrate_hull_white(yield_curve, swaption_data, initial_guess=None, verbose=True):
    """
    Calibrate Hull-White model to market swaption volatilities
    
    Args:
        yield_curve: YieldCurve object
        swaption_data: DataFrame with swaption market data
        initial_guess: Optional [a, sigma] starting point
        verbose: Print calibration progress
        
    Returns:
        dict with:
            - 'a': calibrated mean reversion
            - 'sigma': calibrated volatility
            - 'success': whether optimization succeeded
            - 'message': optimization message
            - 'fit_quality': DataFrame showing model vs market
    """
    if verbose:
        print("\n" + "="*60)
        print("HULL-WHITE CALIBRATION")
        print("="*60)
        print(f"\nCalibrating to {len(swaption_data)} swaptions...")
    
    # Initial guess
    if initial_guess is None:
        initial_guess = [0.05, 0.015]  # Typical values
    
    # Bounds for parameters
    bounds = [
        (0.01, 0.20),   # a: 1% to 20% mean reversion
        (0.005, 0.030)  # sigma: 50 bps to 300 bps
    ]
    
    # Run optimization
    if verbose:
        print(f"\nInitial guess: a={initial_guess[0]:.4f}, sigma={initial_guess[1]:.4f}")
        print("Optimizing...")
    
    result = minimize(
        calibration_objective,
        initial_guess,
        args=(yield_curve, swaption_data),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'ftol': 1e-8}
    )
    
    # Extract results
    a_calibrated = result.x[0]
    sigma_calibrated = result.x[1]
    
    if verbose:
        print(f"\n{'='*60}")
        print("CALIBRATION RESULTS")
        print("="*60)
        print(f"[OK] Mean reversion (a) = {a_calibrated:.6f} ({a_calibrated*100:.4f}% per year)")
        print(f"[OK] Volatility (sigma) = {sigma_calibrated:.6f} ({sigma_calibrated*10000:.2f} bps)")
        print(f"[OK] Optimization success: {result.success}")
        print(f"[OK] Final error: {result.fun:.8f}")
    
    # Compute fit quality
    fit_data = []
    for _, row in swaption_data.iterrows():
        T_opt = row['option_expiry']
        T_swap = row['swap_tenor']
        market_vol = row['implied_vol']
        
        model_vol = hw_swaption_vol(a_calibrated, sigma_calibrated, T_opt, T_swap)
        
        fit_data.append({
            'swaption': f"{T_opt}Y x {T_swap}Y",
            'market_vol_bps': market_vol * 10000,
            'model_vol_bps': model_vol * 10000,
            'error_bps': (model_vol - market_vol) * 10000
        })
    
    fit_quality = pd.DataFrame(fit_data)
    
    if verbose:
        print("\n" + "="*60)
        print("FIT QUALITY")
        print("="*60)
        print(fit_quality.to_string(index=False))
        print(f"\nRMSE: {np.sqrt(np.mean(fit_quality['error_bps']**2)):.2f} bps")
    
    return {
        'a': a_calibrated,
        'sigma': sigma_calibrated,
        'success': result.success,
        'message': result.message,
        'fit_quality': fit_quality,
        'optimization_result': result
    }


def compute_theta(yield_curve, a, sigma, t_min=1.0):
    from scipy.ndimage import uniform_filter1d
    
    print("\n" + "="*60)
    print("COMPUTING theta(t)")
    print("="*60)
    
    t_grid = yield_curve.t_grid
    f = yield_curve.forward_rates
    df_dt = yield_curve.forward_slope
    
    # Heavy smoothing
    df_dt_smooth = uniform_filter1d(df_dt, size=36, mode='nearest')
    
    mask = t_grid >= t_min
    
    # Three terms
    term1 = f.copy()
    term3 = (sigma**2 / (2*a)) * (1 - np.exp(-2*a*t_grid)) if a > 1e-6 else (sigma**2 / 2) * t_grid
    
    # Slope term - compute and cap
    term2_uncapped = (1 / a) * df_dt_smooth if a > 1e-6 else df_dt_smooth / 1e-6
    term2_capped = np.clip(term2_uncapped, -0.02, 0.02)
    
    # Apply cap and zero out unstable region
    term2 = term2_capped.copy()
    term2[~mask] = 0
    
    # Debug: check what's happening at t=30Y
    idx_30 = int(30 * 12)
    if idx_30 < len(term2):
        print(f"\nDEBUG at t=30Y:")
        print(f"  term2_uncapped = {term2_uncapped[idx_30]:.4%}")
        print(f"  term2_capped   = {term2_capped[idx_30]:.4%}")
        print(f"  term2_final    = {term2[idx_30]:.4%}")
    
    theta = term1 + term2 + term3
    
    print(f"\ntheta(t) computed (stable for t >= {t_min}Y)")
    print(f"  Slope capped at ±2%")
    
    check_times = [1, 5, 10, 20, 30]
    for t_years in check_times:
        idx = int(t_years * 12)
        if idx < len(theta):
            print(f"  theta({t_years:2d}Y) = {theta[idx]:7.4%}  "
                  f"[f={f[idx]:6.4%}, slope={term2[idx]:7.4%}, convex={term3[idx]:6.4%}]")
    
    print(f"\n[OK] theta(t) range (t>={t_min}): [{theta[mask].min():.4%}, {theta[mask].max():.4%}]")
    
    return theta


def plot_calibration_results(calibrated_params, yield_curve, t_min=1.0):
    """
    Visualize calibration results with auto-scaled axes
    
    Args:
        calibrated_params: Dict from calibrate_hull_white()
        yield_curve: YieldCurve object
        t_min: Minimum time to plot (skip unstable short end)
    """
    import matplotlib.pyplot as plt
    
    a = calibrated_params['a']
    sigma = calibrated_params['sigma']
    
    # Compute theta
    theta = compute_theta(yield_curve, a, sigma, t_min=t_min)
    
    # Filter to stable region for plotting
    mask = yield_curve.t_grid >= t_min
    t_plot = yield_curve.t_grid[mask]
    f_plot = yield_curve.forward_rates[mask]
    theta_plot = theta[mask]
    slope_plot = yield_curve.forward_slope[mask]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Forward rate and theta
    axes[0, 0].plot(t_plot, f_plot * 100,
                    'b-', linewidth=2, label='Forward f(0,t)')
    axes[0, 0].plot(t_plot, theta_plot * 100,
                    'r--', linewidth=2, label='theta(t)')
    axes[0, 0].set_xlabel('Time (years)')
    axes[0, 0].set_ylabel('Rate (%)')
    axes[0, 0].set_title(f'Forward Rate vs Mean Reversion Level (t >= {t_min}Y)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([t_min, 30])
    axes[0, 0].margins(y=0.1)  # Add 10% margin on y-axis
    
    # Panel 2: Theta components
    if a > 1e-6:
        term2 = (1 / a) * slope_plot
        term3 = (sigma**2 / (2*a)) * (1 - np.exp(-2*a*t_plot))
    else:
        term2 = slope_plot / 1e-6
        term3 = (sigma**2 / 2) * t_plot
    
    axes[0, 1].plot(t_plot, f_plot * 100, label='f(0,t)', linewidth=2)
    axes[0, 1].plot(t_plot, term2 * 100, label='(1/a)·df/dt', linewidth=2)
    axes[0, 1].plot(t_plot, term3 * 100, label='Convexity', linewidth=2)
    axes[0, 1].set_xlabel('Time (years)')
    axes[0, 1].set_ylabel('Rate (%)')
    axes[0, 1].set_title('theta(t) Decomposition')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([t_min, 30])
    axes[0, 1].margins(y=0.1)  # Add 10% margin on y-axis
    
    # Panel 3: Swaption fit
    fit_df = calibrated_params['fit_quality']
    x = np.arange(len(fit_df))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, fit_df['market_vol_bps'], width,
                   label='Market', alpha=0.8)
    axes[1, 0].bar(x + width/2, fit_df['model_vol_bps'], width,
                   label='Model', alpha=0.8)
    axes[1, 0].set_xlabel('Swaption')
    axes[1, 0].set_ylabel('Implied Vol (bps)')
    axes[1, 0].set_title('Calibration Fit: Market vs Model')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(fit_df['swaption'], rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].margins(y=0.15)  # Add 15% margin for bars
    
    # Panel 4: Calibration errors
    axes[1, 1].bar(x, fit_df['error_bps'], color='red', alpha=0.7)
    axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1, 1].set_xlabel('Swaption')
    axes[1, 1].set_ylabel('Error (bps)')
    axes[1, 1].set_title('Calibration Errors (Model - Market)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(fit_df['swaption'], rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].margins(y=0.15)  # Add 15% margin
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import pandas as pd
    from yield_curve import YieldCurve
    from data_loading import load_sample_swaption_vols
    
    print("="*60)
    print("Testing Hull-White Calibration with REAL DATA")
    print("="*60)
    
    # Load REAL Treasury data from saved file
    treasury_df = pd.read_csv('../data_io/market_data/treasury_rates_20260217.csv')
    
    print("\nUsing real Treasury data:")
    print(treasury_df)
    
    # Build curve from REAL data
    curve = YieldCurve.from_par_yields(
        treasury_df['rate'].values,
        treasury_df['maturity_years'].values,
        t_max=30,
        dt=1/12
    )
    
    # Load swaption data
    swaption_df = load_sample_swaption_vols()
    
    # Calibrate
    calibrated = calibrate_hull_white(curve, swaption_df, verbose=True)
    
    # Compute theta
    theta = compute_theta(curve, calibrated['a'], calibrated['sigma'], t_min=1.0)
    
    # Plot
    print("\nGenerating plots...")
    plot_calibration_results(calibrated, curve, t_min=1.0)
    
    print("\n[OK] Calibration test complete!")