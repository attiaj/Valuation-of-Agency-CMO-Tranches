# hull_white/validation.py

import numpy as np
import pandas as pd
import importlib

plt = importlib.import_module("matplotlib.pyplot")

def plot_rate_paths(paths, params, n_paths_display=50, t_max=30):
    """
    Visualize Hull-White rate paths
    
    Args:
        paths: Array of rate paths (N_paths, N_steps+1)
        params: Dict with 'a', 'sigma', 'theta', 'r0', 't_grid'
        n_paths_display: Number of paths to display (default 50)
        t_max: Maximum time to display (years)
    """
    N_paths, N_steps = paths.shape
    t_grid = params['t_grid'][:N_steps]
    theta = params['theta'][:N_steps]
    
    # Limit display time
    mask = t_grid <= t_max
    t_plot = t_grid[mask]
    paths_plot = paths[:, mask]
    theta_plot = theta[mask]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Panel 1: Individual paths
    ax = axes[0, 0]
    for i in range(min(n_paths_display, N_paths)):
        ax.plot(t_plot, paths_plot[i, :] * 100, alpha=0.3, linewidth=0.5)
    
    # Add mean path
    mean_path = paths_plot.mean(axis=0)
    ax.plot(t_plot, mean_path * 100, 'r-', linewidth=2, label='Mean path')
    ax.plot(t_plot, theta_plot * 100, 'k--', linewidth=2, label='theta(t)')
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Rate (%)')
    ax.set_title(f'Simulated Rate Paths ({n_paths_display} of {N_paths} shown)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.margins(y=0.1)
    
    # Panel 2: Mean and percentiles over time
    ax = axes[0, 1]
    percentiles = [5, 25, 50, 75, 95]
    percentile_paths = np.percentile(paths_plot * 100, percentiles, axis=0)
    
    ax.fill_between(t_plot, percentile_paths[0], percentile_paths[4], 
                     alpha=0.2, label='5th-95th percentile')
    ax.fill_between(t_plot, percentile_paths[1], percentile_paths[3], 
                     alpha=0.3, label='25th-75th percentile')
    ax.plot(t_plot, percentile_paths[2], 'b-', linewidth=2, label='Median')
    ax.plot(t_plot, theta_plot * 100, 'k--', linewidth=2, label='theta(t)')
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Rate (%)')
    ax.set_title('Path Distribution Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.margins(y=0.1)
    
    # Panel 3: Distribution at specific times
    ax = axes[1, 0]
    times_to_check = [1, 5, 10, 20, 30]
    times_available = [t for t in times_to_check if t <= t_max]
    
    positions = []
    labels = []
    for i, t_years in enumerate(times_available):
        idx = int(t_years * 12)
        if idx < paths_plot.shape[1]:
            data = paths_plot[:, idx] * 100
            pos = ax.violinplot([data], positions=[i], widths=0.7, 
                               showmeans=True, showmedians=True)
            positions.append(i)
            labels.append(f'{t_years}Y')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Rate (%)')
    ax.set_title('Rate Distribution at Different Times')
    ax.grid(True, alpha=0.3, axis='y')
    ax.margins(y=0.1)
    
    # Panel 4: Statistics summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Compute statistics
    stats_text = "SIMULATION STATISTICS\n" + "="*50 + "\n\n"
    stats_text += f"Number of paths: {N_paths:,}\n"
    stats_text += f"Time steps: {N_steps}\n"
    stats_text += f"Time horizon: {t_plot[-1]:.1f} years\n"
    stats_text += f"Initial rate r(0): {params['r0']:.4%}\n\n"
    
    stats_text += "CALIBRATED PARAMETERS:\n"
    stats_text += f"  Mean reversion (a): {params['a']:.6f}\n"
    stats_text += f"  Volatility (sigma): {params['sigma']:.6f}\n\n"
    
    stats_text += "RATE STATISTICS AT KEY TIMES:\n"
    for t_years in [1, 5, 10, 20, 30]:
        idx = int(t_years * 12)
        if idx < paths_plot.shape[1]:
            rates = paths_plot[:, idx]
            stats_text += f"\n  t = {t_years}Y:\n"
            stats_text += f"    Mean:   {rates.mean():.4%}\n"
            stats_text += f"    Std:    {rates.std():.4%}\n"
            stats_text += f"    Min:    {rates.min():.4%}\n"
            stats_text += f"    Max:    {rates.max():.4%}\n"
    
    # Count negative rates
    neg_count = np.sum(paths < 0)
    neg_pct = 100 * neg_count / paths.size
    stats_text += f"\n\nNEGATIVE RATES:\n"
    stats_text += f"  Count: {neg_count:,} ({neg_pct:.2f}%)\n"
    
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()


def plot_rate_histogram(paths, params, times=[1, 5, 10, 20, 30]):
    """
    Plot histograms of rate distributions at specific times
    
    Args:
        paths: Array of rate paths
        params: Parameter dict
        times: List of times (years) to display
    """
    t_grid = params['t_grid'][:paths.shape[1]]
    
    # Filter to available times
    times_available = [t for t in times if t <= t_grid[-1]]
    n_plots = len(times_available)
    
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    
    for ax, t_years in zip(axes, times_available):
        idx = int(t_years * 12)
        if idx < paths.shape[1]:
            rates = paths[:, idx] * 100
            
            ax.hist(rates, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(rates.mean(), color='r', linestyle='--', 
                      linewidth=2, label=f'Mean: {rates.mean():.2f}%')
            median_rate = float(np.median(rates))
            ax.axvline(median_rate, color='g', linestyle='--', 
                      linewidth=2, label=f'Median: {median_rate:.2f}%')
            
            ax.set_xlabel('Rate (%)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Rate Distribution at t = {t_years}Y')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def validate_paths(paths, params, yield_curve):
    """
    Validate that simulated paths match theoretical expectations
    
    Args:
        paths: Array of rate paths
        params: Parameter dict
        yield_curve: YieldCurve object
    """
    print("\n" + "="*60)
    print("PATH VALIDATION")
    print("="*60)
    
    a = params['a']
    sigma = params['sigma']
    t_grid = params['t_grid'][:paths.shape[1]]
    theta = params['theta'][:paths.shape[1]]
    
    # Theoretical mean: E[r(t)]
    r0 = params['r0']
    
    print("\nComparing simulated vs theoretical statistics:\n")
    print(f"{'Time':>6} {'Sim Mean':>10} {'Sim Std':>10} {'Theo Std':>10} {'Difference':>12}")
    print("-" * 60)
    
    for t_years in [1, 5, 10, 20, 30]:
        idx = int(t_years * 12)
        if idx < len(t_grid):
            t = t_grid[idx]
            
            # Simulated statistics
            sim_mean = paths[:, idx].mean()
            sim_std = paths[:, idx].std()
            
            # Theoretical std: sqrt((sigma^2/2a)(1 - e^(-2at)))
            if a > 1e-6:
                theo_var = (sigma**2 / (2*a)) * (1 - np.exp(-2*a*t))
            else:
                theo_var = sigma**2 * t
            theo_std = np.sqrt(theo_var)
            
            diff = abs(sim_std - theo_std) / theo_std * 100
            
            print(f"{t_years:>6}Y {sim_mean:>9.4%} {sim_std:>9.4%} {theo_std:>9.4%} {diff:>10.2f}%")
    
    # Check for negative rates
    neg_count = np.sum(paths < 0)
    neg_pct = 100 * neg_count / paths.size
    
    print(f"\n{'='*60}")
    print(f"Negative rates: {neg_count:,} ({neg_pct:.2f}% of all values)")
    
    if neg_pct < 1:
        print("[OK] Low negative rate probability - acceptable")
    elif neg_pct < 5:
        print("[WARNING] Moderate negative rate probability")
    else:
        print("[ERROR] High negative rate probability - consider adjusting parameters")
    
    print("="*60)

def plot_paths_simple(paths, params, n_paths_display=75, save_file=None):
    """
    Simple single-panel path visualization with colorful, visible paths
    
    Args:
        paths: Array of rate paths
        params: Parameter dict
        n_paths_display: Number of paths to show (default 75)
        save_file: Optional filename to save (e.g., 'paths.png')
    """
    N_paths = paths.shape[0]
    t_grid = params['t_grid'][:paths.shape[1]]
    theta = params['theta'][:paths.shape[1]]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    
    # Individual paths with different colors
    colors = plt.cm.tab20c(np.linspace(0, 1, n_paths_display))
    
    for i in range(min(n_paths_display, N_paths)):
        ax.plot(t_grid, paths[i, :] * 100, 
                color=colors[i], alpha=0.5, linewidth=1.0)
    
    # Mean path (red, slightly thicker)
    mean_path = paths.mean(axis=0)
    ax.plot(t_grid, mean_path * 100, 
            'r-', linewidth=2.5, label='Mean Path', zorder=10)
    
    # Theta (black dashed, slightly thicker)
    ax.plot(t_grid, theta * 100, 
            'k--', linewidth=2.5, label='θ(t) - Target', zorder=10)
    
    # Formatting
    ax.set_xlabel('Time (years)', fontsize=13)
    ax.set_ylabel('Interest Rate (%)', fontsize=13)
    ax.set_title(f'Hull-White Interest Rate Paths ({n_paths_display} of {N_paths} shown)', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, t_grid[-1]])
    ax.margins(y=0.1)
    
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file, dpi=300, facecolor='white', bbox_inches='tight')
        print(f"✓ Saved to {save_file}")
    
    plt.show()


if __name__ == "__main__":
    # Test visualization with saved paths
    import sys
    sys.path.append('..')
    from path_generation import load_paths
    
    print("Loading saved paths...")
    paths, params = load_paths('../data_io/rate_paths.npz')
    
    print(f"Loaded {paths.shape[0]} paths with {paths.shape[1]} time steps")
    
    # Validate
    from yield_curve import YieldCurve
    import pandas as pd
    
    treasury_df = pd.read_csv('../data_io/market_data/treasury_rates_20260217.csv')
    curve = YieldCurve.from_par_yields(
        treasury_df['rate'].values,
        treasury_df['maturity_years'].values
    )
    
    validate_paths(paths, params, curve)
    
    # Visualize
    print("\nGenerating visualizations...")
    plot_paths_simple(paths, params, n_paths_display=100)
    plot_rate_histogram(paths, params)