# hull_white/__init__.py

"""
Hull-White Interest Rate Model Implementation
"""
from .validation import (
    plot_rate_paths,
    plot_rate_histogram,
    validate_paths
)

from .data_loading import (
    load_treasury_rates_from_fred,
    load_sample_swaption_vols,
    validate_curve_data,
    save_market_data
)

from .yield_curve import (
    YieldCurve,
    bootstrap_zeros,
    interpolate_curve,
    compute_forward_curve,
    compute_forward_slope,
    load_curve_from_csv
)

from .calibration import (
    calibrate_hull_white,
    compute_theta,
    hw_swaption_vol,
    plot_calibration_results
)

from .path_generation import (
    generate_paths,
    generate_single_path,
    generate_paths_antithetic,
    save_paths,
    load_paths
)

__all__ = [
    # Data loading
    'load_treasury_rates_from_fred',
    'load_sample_swaption_vols',
    'validate_curve_data',
    'save_market_data',
    
    # Yield curve
    'YieldCurve',
    'bootstrap_zeros',
    'interpolate_curve',
    'compute_forward_curve',
    'compute_forward_slope',
    'load_curve_from_csv',
    
    # Calibration
    'calibrate_hull_white',
    'compute_theta',
    'hw_swaption_vol',
    'plot_calibration_results',
    
    # Path generation
    'generate_paths',
    'generate_single_path',
    'generate_paths_antithetic',
    'save_paths',
    'load_paths',

    # Validation
    'plot_rate_paths',
    'plot_rate_histogram',
    'validate_paths'
]