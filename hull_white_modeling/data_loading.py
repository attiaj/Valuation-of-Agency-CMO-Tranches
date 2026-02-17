# hull_white/data_loading.py

import pandas as pd
import numpy as np
from datetime import datetime

def load_treasury_rates_from_fred(start_date='2026-02-01', end_date='2026-02-17'):
    """
    Load Treasury rates from FRED (Federal Reserve Economic Data)
    
    Args:
        start_date: Start date for data (YYYY-MM-DD)
        end_date: End date for data (YYYY-MM-DD)
        
    Returns:
        DataFrame with maturities and rates
    """
    # FRED series IDs for Treasury rates
    series_ids = {
        '1M': 'DGS1MO',
        '3M': 'DGS3MO',
        '6M': 'DGS6MO',
        '1Y': 'DGS1',
        '2Y': 'DGS2',
        '3Y': 'DGS3',
        '5Y': 'DGS5',
        '7Y': 'DGS7',
        '10Y': 'DGS10',
        '20Y': 'DGS20',
        '30Y': 'DGS30'
    }
    
    # Maturity in years
    maturities_map = {
        '1M': 1/12,
        '3M': 0.25,
        '6M': 0.5,
        '1Y': 1.0,
        '2Y': 2.0,
        '3Y': 3.0,
        '5Y': 5.0,
        '7Y': 7.0,
        '10Y': 10.0,
        '20Y': 20.0,
        '30Y': 30.0
    }
    
    print(f"Fetching Treasury rates from FRED...")
    print(f"Date range: {start_date} to {end_date}")
    
    rates_dict = {}
    
    for tenor, series_id in series_ids.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        
        try:
            # Directly read CSV from URL using pandas (no external requests dependency)
            df = pd.read_csv(url)
            df.columns = ['date', 'rate']
            df['date'] = pd.to_datetime(df['date'])
            df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
            
            # Filter by date range
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            if not df.empty:
                # Take most recent non-null value
                latest_rate = df.dropna()['rate'].iloc[-1] if not df.dropna().empty else None
                if latest_rate is not None:
                    rates_dict[tenor] = {
                        'maturity_years': maturities_map[tenor],
                        'rate': latest_rate / 100  # Convert from % to decimal
                    }
                    print(f"  {tenor}: {latest_rate:.2f}%")
            
        except Exception as e:
            print(f"  Warning: Could not fetch {tenor} ({series_id}): {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(rates_dict, orient='index')
    df = df.sort_values('maturity_years').reset_index()
    df = df.rename(columns={'index': 'tenor'})
    
    print(f"\nSuccessfully loaded {len(df)} Treasury rates")
    
    return df


def load_sample_swaption_vols():
    """
    Load sample ATM swaption volatilities
    
    For now, using typical market values
    Later, you can replace with real data
    
    Returns:
        DataFrame with swaption data
    """
    # Typical market values (in basis points)
    data = {
        'option_expiry': [1, 2, 3, 5, 1, 2, 5],
        'swap_tenor': [5, 5, 5, 5, 10, 10, 10],
        'implied_vol_bps': [95, 88, 85, 80, 92, 87, 82]
    }
    
    df = pd.DataFrame(data)
    df['implied_vol'] = df['implied_vol_bps'] / 10000  # Convert to decimal
    
    print("\nUsing sample swaption volatilities:")
    for _, row in df.iterrows():
        print(f"  {row['option_expiry']}Y x {row['swap_tenor']}Y: {row['implied_vol_bps']:.0f} bps")
    
    return df


def validate_curve_data(df):
    """
    Validate that curve data is reasonable
    
    Args:
        df: DataFrame with 'maturity_years' and 'rate' columns
        
    Returns:
        bool: True if valid
    """
    print("\nValidating data...")
    
    # Check for NaNs
    if df['rate'].isna().any():
        print("  [WARNING] Found NaN values in rates")
        return False
    
    # Check for negative rates
    if (df['rate'] < 0).any():
        print("  [WARNING] Found negative rates")
        return False
    
    # Check for unreasonable rates (outside 0-20%)
    if (df['rate'] > 0.20).any():
        print("  [WARNING] Found rates above 20%")
        return False
    
    # Check maturities are sorted
    if not df['maturity_years'].is_monotonic_increasing:
        print("  [WARNING] Maturities not sorted")
        return False
    
    print("  [OK] Data validation passed")
    return True


def save_market_data(swap_df, swaption_df, date_str=None):
    """
    Save market data to CSV files
    
    Args:
        swap_df: DataFrame with swap/treasury rates
        swaption_df: DataFrame with swaption vols
        date_str: Optional date string for filename
    """
    if date_str is None:
        date_str = datetime.now().strftime('%Y%m%d')
    
    # Create directory if needed
    import os
    os.makedirs('../data_io/market_data', exist_ok=True)
    
    # Save files
    swap_file = f'../data_io/market_data/treasury_rates_{date_str}.csv'
    swaption_file = f'../data_io/market_data/swaption_vols_{date_str}.csv'
    
    swap_df.to_csv(swap_file, index=False)
    swaption_df.to_csv(swaption_file, index=False)
    
    print(f"\n[OK] Saved data to:")
    print(f"  {swap_file}")
    print(f"  {swaption_file}")


if __name__ == "__main__":
    # Test the data loading
    print("="*60)
    print("Testing Hull-White Data Loading")
    print("="*60)
    
    # Load Treasury rates
    treasury_df = load_treasury_rates_from_fred(
        start_date='2026-02-01',
        end_date='2026-02-17'
    )
    
    # Load swaption vols
    swaption_df = load_sample_swaption_vols()
    
    # Validate
    if validate_curve_data(treasury_df):
        print("\n[OK] Data ready for calibration")
        
        # Save for later use
        save_market_data(treasury_df, swaption_df)
    else:
        print("\n[ERROR] Data validation failed")