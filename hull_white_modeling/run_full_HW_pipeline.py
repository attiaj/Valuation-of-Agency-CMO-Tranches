# run_hull_white_full_pipeline.py

"""
Complete Hull-White Model Pipeline
Demonstrates full workflow from data loading to path generation and validation
"""

import sys
import os
import importlib
from pathlib import Path

# Add parent directory to path to allow imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import pandas as pd
import numpy as np
from datetime import datetime

# Import from submodules
from hull_white_modeling.data_loading import (
    load_treasury_rates_from_fred,
    load_sample_swaption_vols,
    save_market_data
)
from hull_white_modeling.yield_curve import YieldCurve
from hull_white_modeling.calibration import (
    calibrate_hull_white,
    compute_theta,
    plot_calibration_results
)
from hull_white_modeling.path_generation import (
    generate_paths,
    save_paths
)
from hull_white_modeling.validation import (
    plot_rate_paths,
    plot_rate_histogram,
    validate_paths
)

def main():
    """
    Run complete Hull-White pipeline
    """
    print("\n" + "="*80)
    print(" "*20 + "HULL-WHITE MODEL - COMPLETE PIPELINE")
    print("="*80)
    
    # -------------------------------------------------------------------------
    # STEP 1: DATA LOADING
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 1: LOADING MARKET DATA")
    print("="*80)
    
    use_saved_data = True  # Set to False to fetch fresh data from FRED
    
    if use_saved_data:
        print("\nLoading saved Treasury data...")
        try:
            # Use path relative to project root (parent_dir)
            treasury_file = parent_dir / 'data_io' / 'market_data' / 'treasury_rates_20260217.csv'
            treasury_df = pd.read_csv(treasury_file)
            swaption_df = load_sample_swaption_vols()
            print("[OK] Loaded saved data successfully")
        except FileNotFoundError:
            print("[WARNING] Saved data not found, fetching from FRED...")
            use_saved_data = False
    
    if not use_saved_data:
        print("\nFetching fresh data from FRED...")
        treasury_df = load_treasury_rates_from_fred(
            start_date='2026-02-01',
            end_date='2026-02-17'
        )
        swaption_df = load_sample_swaption_vols()
        
        # Save for future use
        save_market_data(treasury_df, swaption_df)
        print("[OK] Data fetched and saved")
    
    print("\nTreasury rates:")
    print(treasury_df)
    print("\nSwaption volatilities:")
    print(swaption_df)
    
    # -------------------------------------------------------------------------
    # STEP 2: YIELD CURVE CONSTRUCTION
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 2: CONSTRUCTING YIELD CURVE")
    print("="*80)
    
    curve = YieldCurve.from_par_yields(
        par_yields=treasury_df['rate'].values,
        maturities=treasury_df['maturity_years'].values,
        t_max=30,
        dt=1/12
    )
    
    print(f"\n[OK] Yield curve constructed: {curve}")
    print(f"  Initial forward rate f(0,0) = {curve.forward_rates[0]:.4%}")
    
    # Plot yield curve
    print("\nDisplaying yield curve...")
    curve.plot(t_min=0.5)
    
    # -------------------------------------------------------------------------
    # STEP 3: HULL-WHITE CALIBRATION
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 3: CALIBRATING HULL-WHITE MODEL")
    print("="*80)
    
    calibrated = calibrate_hull_white(curve, swaption_df, verbose=True)
    
    a = calibrated['a']
    sigma = calibrated['sigma']
    
    print(f"\n[OK] Calibration complete!")
    print(f"  Mean reversion (a) = {a:.6f} ({a*100:.4f}% per year)")
    print(f"  Volatility (sigma) = {sigma:.6f} ({sigma*10000:.2f} bps)")
    
    # Compute theta
    theta = compute_theta(curve, a, sigma, t_min=1.0)
    
    # Plot calibration results
    print("\nDisplaying calibration results...")
    plot_calibration_results(calibrated, curve, t_min=1.0)
    
    # -------------------------------------------------------------------------
    # STEP 4: RATE PATH GENERATION
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 4: GENERATING INTEREST RATE PATHS")
    print("="*80)
    
    # Prepare parameters
    params = {
        'a': a,
        'sigma': sigma,
        'theta': theta,
        'r0': curve.forward_rates[0],  # Initial short rate
        't_grid': curve.t_grid
    }
    
    print(f"\nGenerating paths with:")
    print(f"  Initial rate r(0) = {params['r0']:.4%}")
    print(f"  Mean reversion a  = {a:.6f}")
    print(f"  Volatility sigma  = {sigma:.6f}")
    
    # Generate paths
    N_PATHS = 1000
    paths = generate_paths(params, N_paths=N_PATHS, random_seed=42)
    
    # Save paths
    output_file = parent_dir / 'data_io' / 'rate_paths.npz'
    save_paths(paths, params, filepath=str(output_file))
    
    # -------------------------------------------------------------------------
    # STEP 5: VALIDATION AND VISUALIZATION
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 5: VALIDATING AND VISUALIZING PATHS")
    print("="*80)
    
    # Validate paths
    validate_paths(paths, params, curve)
    
    # Visualize paths
    print("\nGenerating path visualizations...")
    plot_rate_paths(paths, params, n_paths_display=100, t_max=30)
    
    # Histogram at key times
    print("\nGenerating distribution histograms...")
    plot_rate_histogram(paths, params, times=[1, 5, 10, 20, 30])
    
    # -------------------------------------------------------------------------
    # STEP 6: SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    
    print("\n[DATA]")
    print(f"  Treasury curve: {len(treasury_df)} maturities")
    print(f"  Swaption vols:  {len(swaption_df)} instruments")
    print(f"  Date:           February 17, 2026")
    
    print("\n[YIELD CURVE]")
    print(f"  Time grid:      0 to 30 years, monthly steps")
    print(f"  Grid points:    {len(curve.t_grid)}")
    print(f"  Initial rate:   {curve.forward_rates[0]:.4%}")
    
    print("\n[CALIBRATION]")
    print(f"  Mean reversion: {a:.6f}")
    print(f"  Volatility:     {sigma:.6f}")
    print(f"  Fit RMSE:       {calibrated.get('fit_quality', pd.DataFrame()).get('error_bps', pd.Series()).std():.2f} bps")
    
    print("\n[SIMULATION]")
    print(f"  Paths:          {N_PATHS:,}")
    print(f"  Time steps:     {paths.shape[1]}")
    print(f"  Horizon:        30 years")
    print(f"  Output file:    {str(output_file)}")
    
    # Path statistics
    neg_count = np.sum(paths < 0)
    neg_pct = 100 * neg_count / paths.size
    
    print("\n[PATH STATISTICS]")
    print(f"  Rate at 1Y:     {paths[:, 12].mean():.4%} ± {paths[:, 12].std():.4%}")
    print(f"  Rate at 5Y:     {paths[:, 60].mean():.4%} ± {paths[:, 60].std():.4%}")
    print(f"  Rate at 10Y:    {paths[:, 120].mean():.4%} ± {paths[:, 120].std():.4%}")
    print(f"  Negative rates: {neg_count:,} ({neg_pct:.2f}%)")
    
    print("\n" + "="*80)
    print("[OK] PIPELINE COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  • Use generated paths for MBS valuation")
    print("  • Analyze prepayment sensitivity to rate paths")
    print("  • Price CMO tranches under different scenarios")
    print("\n")

    plt = importlib.import_module("matplotlib.pyplot")

    fig, ax = plt.subplots(figsize=(14, 7))
    t_grid = params['t_grid']

    n_display = min(100, paths.shape[0])
    theta_plot = params['theta'][:paths.shape[1]]
    path_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for i in range(n_display):
        ax.plot(
            t_grid,
            paths[i, :] * 100,
            alpha=0.45,
            linewidth=0.6,
            color=path_colors[i % len(path_colors)],
        )

    ax.plot(t_grid, paths.mean(axis=0) * 100, 'r-', linewidth=2, label='Mean path')
    ax.plot(t_grid, theta_plot * 100, 'k--', linewidth=2, label='theta(t)')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Short Rate (%)')
    ax.set_title('Hull-White Interest Rate Paths (100 of 1,000 shown)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_file = parent_dir / "data_io" / "hw_paths.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved final path plot to {plot_file}")
    plt.show()


if __name__ == "__main__":
    main()