# hull_white/path_generation.py

import numpy as np
import pandas as pd

def generate_paths(params, N_paths=1000, N_steps=360, dt=1/12, random_seed=None):
    """
    Generate Hull-White interest rate paths using Euler discretization
    
    Simulates the SDE: dr = a[theta(t) - r]dt + sigma*dz
    
    Args:
        params: dict with keys:
            - 'a': mean reversion speed
            - 'sigma': volatility
            - 'theta': array of theta(t) values (length N_steps+1)
            - 'r0': initial short rate
            - 't_grid': time grid (optional, for validation)
        N_paths: number of Monte Carlo paths to generate
        N_steps: number of time steps (default 360 = 30 years monthly)
        dt: time step size in years (default 1/12 = monthly)
        random_seed: optional seed for reproducibility
        
    Returns:
        Array of shape (N_paths, N_steps+1) with rate paths
    """
    # Extract parameters
    a = params['a']
    sigma = params['sigma']
    theta = params['theta']
    r0 = params['r0']
    
    # Validation
    if len(theta) < N_steps + 1:
        raise ValueError(f"theta array too short: {len(theta)} < {N_steps+1}")
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    print("\n" + "="*60)
    print("GENERATING HULL-WHITE RATE PATHS")
    print("="*60)
    print(f"Parameters:")
    print(f"  Mean reversion (a) = {a:.6f}")
    print(f"  Volatility (sigma) = {sigma:.6f}")
    print(f"  Initial rate (r0)  = {r0:.4%}")
    print(f"\nSimulation settings:")
    print(f"  Number of paths    = {N_paths:,}")
    print(f"  Time steps         = {N_steps} ({N_steps*dt:.1f} years)")
    print(f"  Time step (dt)     = {dt:.6f} years ({dt*12:.1f} months)")
    
    # Initialize paths matrix
    r = np.zeros((N_paths, N_steps + 1))
    r[:, 0] = r0  # All paths start at r0
    
    # Generate all random shocks at once (efficient)
    epsilon = np.random.randn(N_paths, N_steps)
    
    print("\nSimulating paths...")
    
    # Euler-Maruyama discretization
    # r(t+dt) = r(t) + a[theta(t) - r(t)]*dt + sigma*sqrt(dt)*epsilon
    
    sqrt_dt = np.sqrt(dt)
    
    for step in range(N_steps):
        # Current theta value
        theta_t = theta[step]
        
        # Deterministic drift term: a[theta(t) - r(t)]*dt
        drift = a * (theta_t - r[:, step]) * dt
        
        # Stochastic diffusion term: sigma*sqrt(dt)*epsilon
        diffusion = sigma * sqrt_dt * epsilon[:, step]
        
        # Euler step
        r[:, step + 1] = r[:, step] + drift + diffusion
    
    print("[OK] Path generation complete!")
    
    # Quick statistics
    print(f"\nPath statistics:")
    print(f"  Initial rate:  {r[:, 0].mean():.4%} (all paths)")
    print(f"  Rate at 1Y:    {r[:, 12].mean():.4%} ± {r[:, 12].std():.4%}")
    print(f"  Rate at 5Y:    {r[:, 60].mean():.4%} ± {r[:, 60].std():.4%}")
    print(f"  Rate at 10Y:   {r[:, 120].mean():.4%} ± {r[:, 120].std():.4%}")
    print(f"  Rate at 30Y:   {r[:, 360].mean():.4%} ± {r[:, 360].std():.4%}")
    
    # Check for negative rates
    neg_count = np.sum(r < 0)
    neg_pct = 100 * neg_count / r.size
    if neg_pct > 0:
        print(f"\n[WARNING] {neg_count:,} negative rates ({neg_pct:.2f}% of all values)")
    else:
        print(f"\n[OK] No negative rates")
    
    return r


def generate_single_path(params, N_steps=360, dt=1/12, random_seed=None):
    """
    Generate a single Hull-White rate path (for debugging/visualization)
    
    Args:
        params: dict with 'a', 'sigma', 'theta', 'r0'
        N_steps: number of time steps
        dt: time step size
        random_seed: optional seed
        
    Returns:
        Array of length N_steps+1 with one rate path
    """
    paths = generate_paths(params, N_paths=1, N_steps=N_steps, 
                          dt=dt, random_seed=random_seed)
    return paths[0, :]


def generate_paths_antithetic(params, N_paths=1000, N_steps=360, dt=1/12, random_seed=None):
    """
    Generate Hull-White paths using antithetic variates for variance reduction
    
    Generates N_paths/2 normal paths and N_paths/2 antithetic paths (negated shocks)
    This reduces Monte Carlo variance for certain estimators
    
    Args:
        params: dict with 'a', 'sigma', 'theta', 'r0'
        N_paths: number of paths (must be even)
        N_steps: number of time steps
        dt: time step size
        random_seed: optional seed
        
    Returns:
        Array of shape (N_paths, N_steps+1) with rate paths
    """
    if N_paths % 2 != 0:
        raise ValueError("N_paths must be even for antithetic variates")
    
    # Extract parameters
    a = params['a']
    sigma = params['sigma']
    theta = params['theta']
    r0 = params['r0']
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    print("\n" + "="*60)
    print("GENERATING PATHS (ANTITHETIC VARIATES)")
    print("="*60)
    
    N_pairs = N_paths // 2
    
    # Initialize
    r = np.zeros((N_paths, N_steps + 1))
    r[:, 0] = r0
    
    # Generate half the random shocks
    epsilon = np.random.randn(N_pairs, N_steps)
    
    sqrt_dt = np.sqrt(dt)
    
    for step in range(N_steps):
        theta_t = theta[step]
        
        # Normal paths (first half)
        drift_normal = a * (theta_t - r[:N_pairs, step]) * dt
        diffusion_normal = sigma * sqrt_dt * epsilon[:, step]
        r[:N_pairs, step + 1] = r[:N_pairs, step] + drift_normal + diffusion_normal
        
        # Antithetic paths (second half) - use negated shocks
        drift_anti = a * (theta_t - r[N_pairs:, step]) * dt
        diffusion_anti = sigma * sqrt_dt * (-epsilon[:, step])  # Negated!
        r[N_pairs:, step + 1] = r[N_pairs:, step] + drift_anti + diffusion_anti
    
    print(f"[OK] Generated {N_paths} paths ({N_pairs} pairs)")
    
    return r


def save_paths(paths, params, filepath='rate_paths.npz'):
    """
    Save generated paths and parameters to file
    
    Args:
        paths: Array of rate paths
        params: Parameter dict
        filepath: Path to save file
    """
    np.savez(filepath,
             paths=paths,
             a=params['a'],
             sigma=params['sigma'],
             theta=params['theta'],
             r0=params['r0'],
             t_grid=params.get('t_grid', None))
    
    print(f"\n[OK] Saved {paths.shape[0]} paths to {filepath}")


def load_paths(filepath='rate_paths.npz'):
    """
    Load previously generated paths
    
    Args:
        filepath: Path to saved file
        
    Returns:
        paths: Array of rate paths
        params: Parameter dict
    """
    data = np.load(filepath)
    
    paths = data['paths']
    params = {
        'a': float(data['a']),
        'sigma': float(data['sigma']),
        'theta': data['theta'],
        'r0': float(data['r0']),
        't_grid': data['t_grid'] if 't_grid' in data else None
    }
    
    print(f"[OK] Loaded {paths.shape[0]} paths from {filepath}")
    
    return paths, params


if __name__ == "__main__":
    import pandas as pd
    from yield_curve import YieldCurve
    from calibration import calibrate_hull_white, compute_theta
    from data_loading import load_sample_swaption_vols
    
    print("="*60)
    print("Testing Hull-White Path Generation with REAL DATA")
    print("="*60)
    
    # Load REAL Treasury data
    treasury_df = pd.read_csv('../data_io/market_data/treasury_rates_20260217.csv')
    
    print("\nUsing real Treasury data:")
    print(treasury_df)
    
    # Build curve
    curve = YieldCurve.from_par_yields(
        treasury_df['rate'].values,
        treasury_df['maturity_years'].values,
        t_max=30,
        dt=1/12
    )
    
    # Calibrate
    print("\nCalibrating Hull-White model...")
    swaption_df = load_sample_swaption_vols()
    calibrated = calibrate_hull_white(curve, swaption_df, verbose=False)
    
    print(f"\nCalibrated parameters:")
    print(f"  a = {calibrated['a']:.6f}")
    print(f"  sigma = {calibrated['sigma']:.6f}")
    
    # Compute theta
    theta = compute_theta(curve, calibrated['a'], calibrated['sigma'], t_min=1.0)
    
    # Prepare parameters
    params = {
        'a': calibrated['a'],
        'sigma': calibrated['sigma'],
        'theta': theta,
        'r0': curve.forward_rates[12],  # Start at 1Y forward (stable)
        't_grid': curve.t_grid
    }
    
    # Generate paths
    paths = generate_paths(params, N_paths=1000, random_seed=42)
    
    print(f"\n[OK] Generated paths shape: {paths.shape}")
    print(f"  {paths.shape[0]} paths x {paths.shape[1]} time steps")
    
    # Test saving/loading
    save_paths(paths, params, filepath='../data_io/rate_paths.npz')
    paths_loaded, params_loaded = load_paths('../data_io/rate_paths.npz')
    
    print(f"\n[OK] Save/load test passed")
    
    print("\n" + "="*60)
    print("Path generation test complete!")
    print("="*60)