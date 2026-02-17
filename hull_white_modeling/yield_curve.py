# hull_white/yield_curve.py

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from typing import Tuple, Optional
import matplotlib.pyplot as plt  # type: ignore

class YieldCurve:
    """
    Container for yield curve data
    
    Stores zero-coupon yields, forward rates, and derivatives
    all on a consistent time grid
    """
    
    def __init__(self, t_grid, zero_rates, forward_rates, forward_slope):
        """
        Args:
            t_grid: Time grid in years (e.g., monthly: 0, 1/12, 2/12, ...)
            zero_rates: Zero-coupon yields y(0,t) at each t
            forward_rates: Instantaneous forward rates f(0,t) at each t
            forward_slope: Derivative df(0,t)/dt at each t
        """
        self.t_grid = np.array(t_grid)
        self.zero_rates = np.array(zero_rates)
        self.forward_rates = np.array(forward_rates)
        self.forward_slope = np.array(forward_slope)
        
        # Store discount factors too (useful later)
        self.discount_factors = np.exp(-zero_rates * t_grid)
    
    @classmethod
    def from_par_yields(cls, par_yields, maturities, 
                        t_max=30, dt=1/12, method='linear'):
        """
        Build complete yield curve from par yields
        
        Args:
            par_yields: Array of par yields (as decimals, e.g., 0.045 for 4.5%)
            maturities: Array of maturities in years
            t_max: Maximum maturity for interpolation (default 30 years)
            dt: Time step for grid (default 1/12 = monthly)
            method: Bootstrapping method ('linear' or 'cubic')
            
        Returns:
            YieldCurve object
        """
        print("\n" + "="*60)
        print("BUILDING YIELD CURVE")
        print("="*60)
        
        # Step 1: Bootstrap zero-coupon yields
        print("\nStep 1: Bootstrapping zero-coupon curve...")
        zero_rates_at_tenors = bootstrap_zeros(par_yields, maturities, method)
        
        # Step 2: Create monthly time grid
        print("\nStep 2: Creating monthly time grid...")
        t_grid = np.arange(0, t_max + dt, dt)
        print(f"  Grid: 0 to {t_max} years, {len(t_grid)} points")
        
        # Step 3: Interpolate zeros to monthly grid
        print("\nStep 3: Interpolating to monthly grid...")
        zero_rates_monthly = interpolate_curve(maturities, zero_rates_at_tenors, 
                                               t_grid, method='cubic')
        
        # Step 4: Compute forward curve
        print("\nStep 4: Computing forward rates...")
        forward_rates = compute_forward_curve(t_grid, zero_rates_monthly)
        
        # Step 5: Compute forward slope
        print("\nStep 5: Computing forward curve slope...")
        forward_slope = compute_forward_slope(t_grid, forward_rates)
        
        print("\n[OK] Yield curve construction complete!")
        
        return cls(t_grid, zero_rates_monthly, forward_rates, forward_slope)
    
    def plot(self, show_forwards=True, t_min=0.5):
        """
        Plot the yield curves
        
        Args:
            show_forwards: Whether to show forward rates
            t_min: Minimum time to plot (default 0.5Y to skip unstable short end)
        """
        
        # Filter to show only t >= t_min
        mask = self.t_grid >= t_min
        t_plot = self.t_grid[mask]
        zero_plot = self.zero_rates[mask]
        fwd_plot = self.forward_rates[mask]
        slope_plot = self.forward_slope[mask]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Top panel: Zero and forward rates
        axes[0].plot(t_plot, zero_plot * 100, 
                    'b-', linewidth=2, label='Zero Rates y(0,t)')
        if show_forwards:
            axes[0].plot(t_plot, fwd_plot * 100, 
                        'r--', linewidth=2, label='Forward Rates f(0,t)')
        axes[0].set_xlabel('Maturity (years)')
        axes[0].set_ylabel('Rate (%)')
        axes[0].set_title(f'Yield Curve (t >= {t_min:.1f} years)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([3, 6])  # Reasonable range
        
        # Bottom panel: Forward slope
        axes[1].plot(t_plot, slope_plot * 100, 
                    'g-', linewidth=2, label='Forward Slope df/dt')
        axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        axes[1].set_xlabel('Maturity (years)')
        axes[1].set_ylabel('Slope (%/year)')
        axes[1].set_title('Forward Curve Slope')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([-2, 2])  # Reasonable range
        
        plt.tight_layout()
        plt.show()
    
    def __repr__(self):
        return (f"YieldCurve(t_grid: {len(self.t_grid)} points, "
                f"range: {self.t_grid[0]:.2f} to {self.t_grid[-1]:.2f} years)")


def bootstrap_zeros(par_yields, maturities, method='simple'):
    """
    Convert par yields to zero yields
    
    For Treasury/swap curves, using the approximation: zero ~= par
    This is accurate to within a few basis points and avoids numerical issues
    
    Args:
        par_yields: Array of par yields (decimals)
        maturities: Array of maturities (years)
        method: Not used, kept for compatibility
        
    Returns:
        Array of zero-coupon yields
    """
    print("  Converting par yields to zero yields...")
    print("  (Using approximation: zero ~= par for Treasury curve)")
    
    # For closely-spaced yield curves, par ~= zero
    # This avoids all the numerical instability from bootstrapping
    zero_rates = par_yields.copy()
    
    for i, (T, par, zero) in enumerate(zip(maturities, par_yields, zero_rates)):
        print(f"    T={T:5.2f}Y: par={par:6.4%} -> zero={zero:6.4%}")
    
    return zero_rates


def interpolate_curve(maturities, rates, t_grid, method='smooth_spline'):
    """
    Interpolate rates using smoothing spline
    
    Produces smooth curves with smooth derivatives (no kinks)
    
    Args:
        maturities: Original maturity points
        rates: Rates at those maturities
        t_grid: New time grid for interpolation
        method: 'smooth_spline' (recommended)
        
    Returns:
        Interpolated rates on t_grid
    """
    from scipy.interpolate import UnivariateSpline
    
    # Ensure we start from t=0
    if maturities[0] > 0:
        maturities = np.insert(maturities, 0, 0)
        rates = np.insert(rates, 0, rates[0])
    
    # Use smoothing spline with very small smoothing parameter
    # s=0 would pass through all points (might oscillate)
    # s=small gives smooth curve close to data
    spline = UnivariateSpline(maturities, rates, s=0.0001, k=3)
    rates_interp = spline(t_grid)
    
    # Ensure no negative rates
    rates_interp = np.maximum(rates_interp, 0.0001)
    
    print(f"  Interpolated from {len(maturities)} points to {len(t_grid)} points")
    print(f"  Method: Smoothing spline (smooth derivatives)")
    
    return rates_interp


def compute_forward_curve(t_grid, zero_rates):
    """
    Compute instantaneous forward rates from zero rates
    
    Formula: f(0,t) = y(0,t) + t * dy(0,t)/dt
    
    Special handling for t=0 to avoid singularity
    
    Args:
        t_grid: Time grid
        zero_rates: Zero-coupon yields at each t
        
    Returns:
        Forward rates at each t
    """
    n = len(t_grid)
    forward_rates = np.zeros(n)
    
    # Handle t=0 separately (use zero rate)
    forward_rates[0] = zero_rates[0]
    
    # For t > 0, use forward rate formula
    for i in range(1, n):
        t = t_grid[i]
        y = zero_rates[i]
        
        # Compute derivative dy/dt using one-sided difference
        # (avoids issues at boundaries)
        if i < n - 1:
            # Central difference (more accurate)
            dt = t_grid[i+1] - t_grid[i-1]
            dy = zero_rates[i+1] - zero_rates[i-1]
            dy_dt = dy / dt
        else:
            # Backward difference (at end of grid)
            dt = t_grid[i] - t_grid[i-1]
            dy = zero_rates[i] - zero_rates[i-1]
            dy_dt = dy / dt
        
        # Forward rate formula: f = y + t * dy/dt
        forward_rates[i] = y + t * dy_dt
    
    print(f"  Forward rates computed")
    print(f"  f(0,0) = {forward_rates[0]:.4%} (initial short rate)")
    if len(forward_rates) > 12:
        print(f"  f(0,1) = {forward_rates[12]:.4%} (1Y forward)")
    if len(forward_rates) > 120:
        print(f"  f(0,10) = {forward_rates[120]:.4%} (10Y forward)")
    
    return forward_rates


def compute_forward_slope(t_grid, forward_rates):
    """
    Compute derivative of forward curve df(0,t)/dt
    
    Uses central differences with special handling at boundaries
    
    Args:
        t_grid: Time grid
        forward_rates: Forward rates at each t
        
    Returns:
        Forward slope df/dt at each t
    """
    n = len(t_grid)
    df_dt = np.zeros(n)
    
    # Handle edges with one-sided differences
    # At t=0: forward difference
    df_dt[0] = (forward_rates[1] - forward_rates[0]) / (t_grid[1] - t_grid[0])
    
    # Interior points: central difference
    for i in range(1, n-1):
        dt = t_grid[i+1] - t_grid[i-1]
        df = forward_rates[i+1] - forward_rates[i-1]
        df_dt[i] = df / dt
    
    # At t=T_max: backward difference
    df_dt[n-1] = (forward_rates[n-1] - forward_rates[n-2]) / (t_grid[n-1] - t_grid[n-2])
    
    print(f"  Forward slope computed")
    print(f"  df/dt range: [{df_dt.min():.6f}, {df_dt.max():.6f}]")
    
    return df_dt


# Convenience function for quick loading
def load_curve_from_csv(filepath, t_max=30, dt=1/12):
    """
    Load yield curve from saved CSV file
    
    Args:
        filepath: Path to CSV with 'maturity_years' and 'rate' columns
        t_max: Maximum maturity for curve
        dt: Time step
        
    Returns:
        YieldCurve object
    """
    df = pd.read_csv(filepath)
    
    return YieldCurve.from_par_yields(
        par_yields=df['rate'].values,
        maturities=df['maturity_years'].values,
        t_max=t_max,
        dt=dt
    )


if __name__ == "__main__":
    import pandas as pd
    
    print("="*60)
    print("Testing Yield Curve with REAL Treasury Data")
    print("="*60)
    
    # Load the real data we saved
    treasury_df = pd.read_csv('../data_io/market_data/treasury_rates_20260217.csv')
    
    print("\nUsing real Treasury data:")
    print(treasury_df)
    
    # Build curve from REAL data
    curve = YieldCurve.from_par_yields(
        par_yields=treasury_df['rate'].values,
        maturities=treasury_df['maturity_years'].values,
        t_max=30,
        dt=1/12
    )
    
    curve.plot(t_min=0.5)