#!/usr/bin/env python
"""Example script demonstrating multi-asset data loading and usage."""

import sys
import os
import logging
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from data.utils.multi_asset_data_loader import MultiAssetDataLoader
from data.utils.data_synchronization import align_timestamps, create_unified_dataframe, detect_and_fix_outliers

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger('multi_asset_example')

def main():
    """Main function demonstrating multi-asset data workflow."""
    logger.info("Starting multi-asset data example")
    
    # 1. Define assets to fetch
    assets = [
        {'symbol': 'BTC/USDT', 'exchange': 'binance', 'type': 'crypto', 'alias': 'BTC'},
        {'symbol': 'ETH/USDT', 'exchange': 'binance', 'type': 'crypto', 'alias': 'ETH'},
        {'symbol': 'SOL/USDT', 'exchange': 'binance', 'type': 'crypto', 'alias': 'SOL'},
    ]
    
    # 2. Initialize the data loader
    loader = MultiAssetDataLoader(
        assets=assets,
        timeframe='1d',  # Daily data
        use_ccxt=True,   # Use CCXT for data fetching
        fill_method='ffill'  # Forward fill missing values
    )
    
    # 3. Fetch data for the last 30 days
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    logger.info(f"Fetching data from {start_date} to {end_date}")
    
    # Option 1: Get unified dataframe directly
    unified_df = loader.fetch_multi_asset_data(
        start_date=start_date,
        end_date=end_date,
        format_type='unified'  # Return a unified DataFrame
    )
    
    if unified_df.empty:
        logger.error("Failed to fetch unified data")
        return
        
    logger.info(f"Fetched unified data: {unified_df.shape} shape")
    
    # Option 2: Get separate dataframes
    separate_dfs = loader.fetch_multi_asset_data(
        start_date=start_date,
        end_date=end_date,
        format_type='separate'  # Return dict of separate DataFrames
    )
    
    if not separate_dfs:
        logger.error("Failed to fetch separate data")
        return
        
    logger.info(f"Fetched separate data for {len(separate_dfs)} assets")
    
    # 4. Demonstrate data synchronization
    # First create a scenario with different time indices
    btc_df = separate_dfs.get('BTC')
    eth_df = separate_dfs.get('ETH')
    
    if btc_df is None or eth_df is None:
        logger.error("Missing required asset data")
        return
    
    # Create a sample with missing days
    if len(btc_df) > 5:
        modified_btc = btc_df.iloc[::2]  # Every other row
        modified_eth = eth_df.iloc[1::2]  # Every other row, offset by 1
        
        logger.info(f"Modified BTC data: {len(modified_btc)} rows")
        logger.info(f"Modified ETH data: {len(modified_eth)} rows")
        
        # Align timestamps
        aligned_dfs, common_idx = align_timestamps(
            {'BTC': modified_btc, 'ETH': modified_eth},
            method='union',  # Use union of all timestamps
            fill_method='ffill'  # Forward fill missing values
        )
        
        logger.info(f"Aligned data using union method: {len(common_idx)} timestamps")
        
        # Create unified dataframe from aligned data
        aligned_unified_df = create_unified_dataframe(aligned_dfs)
        logger.info(f"Created unified dataframe from aligned data: {aligned_unified_df.shape} shape")
    
    # 5. Clean data (fix outliers)
    # Add some artificial outliers to demonstrate outlier detection
    if not unified_df.empty and 'BTC_$close' in unified_df.columns and len(unified_df) > 5:
        # Create a copy with outliers
        outlier_df = unified_df.copy()
        
        # Add outliers at random positions
        np.random.seed(42)
        for col in [c for c in outlier_df.columns if '$close' in c]:
            # Create 1-2 random outliers
            n_outliers = np.random.randint(1, 3)
            for _ in range(n_outliers):
                idx = np.random.randint(0, len(outlier_df))
                # Make the outlier 30-50% higher than normal
                outlier_df.iloc[idx, outlier_df.columns.get_loc(col)] *= np.random.uniform(1.3, 1.5)
                
        logger.info("Added artificial outliers to demonstrate cleaning")
        
        # Fix outliers
        cleaned_df = detect_and_fix_outliers(
            outlier_df,
            method='zscore',
            threshold=2.0  # Lowered threshold to catch our artificial outliers
        )
        
        # Compare before and after cleaning
        fig, axes = plt.subplots(len(assets), 1, figsize=(12, 4 * len(assets)), sharex=True)
        
        for i, asset in enumerate(assets):
            ax = axes[i] if len(assets) > 1 else axes
            
            close_col = f"{asset['alias']}_$close"
            if close_col in outlier_df.columns:
                ax.plot(outlier_df.index, outlier_df[close_col], 'r.-', label=f"{asset['alias']} with outliers")
                ax.plot(cleaned_df.index, cleaned_df[close_col], 'b-', label=f"{asset['alias']} cleaned")
                ax.set_title(f"{asset['alias']} Price - Before and After Outlier Removal")
                ax.legend()
                ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('outlier_cleaning_example.png')
        logger.info("Saved outlier cleaning visualization to outlier_cleaning_example.png")
    
    # 6. Generate basic correlation analysis
    if not unified_df.empty:
        # Extract close prices
        close_cols = [col for col in unified_df.columns if '$close' in col]
        price_df = unified_df[close_cols]
        
        # Calculate returns
        returns_df = price_df.pct_change().dropna()
        
        # Calculate correlation
        corr_matrix = returns_df.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            vmin=-1, 
            vmax=1,
            linewidths=0.5
        )
        plt.title('Correlation Matrix of Asset Returns')
        plt.tight_layout()
        plt.savefig('return_correlation.png')
        logger.info("Saved return correlation heatmap to return_correlation.png")
        
        # Save the unified data to CSV
        output_path = 'multi_asset_data_example.csv'
        unified_df.to_csv(output_path)
        logger.info(f"Saved unified multi-asset data to {output_path}")
    
    logger.info("Multi-asset data example completed successfully")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error in multi-asset example: {e}") 