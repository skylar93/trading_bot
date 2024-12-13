import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from typing import Optional, Dict, List
from utils.data_loader import DataLoader
from utils.feature_generator import FeatureGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetCreator:
    """Creates and manages datasets for trading"""
    
    def __init__(self, config_path: str = "../config/default_config.yaml"):
        """
        Initialize DatasetCreator
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.data_loader = DataLoader(
            exchange_id=self.config['data']['exchange'],
            symbol=self.config['data']['symbols'][0],
            timeframe=self.config['data']['timeframe'],
            cache_dir="data/raw"
        )
        
        self.feature_generator = FeatureGenerator()
        
        # Create necessary directories
        for dir_path in ['data/raw', 'data/processed', 'data/features']:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def create_dataset(self, start_date: Optional[str] = None, 
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Create complete dataset with all features
        
        Args:
            start_date: Start date (default: from config)
            end_date: End date (default: from config)
            
        Returns:
            DataFrame with all features
        """
        try:
            # Use dates from config if not provided
            start_date = start_date or self.config['data']['start_date']
            end_date = end_date or self.config['data']['end_date']
            
            logger.info(f"Creating dataset from {start_date} to {end_date}")
            
            # 1. Load raw data
            logger.info("Fetching raw data...")
            df = self.data_loader.fetch_data(start_date, end_date)
            logger.info(f"Fetched {len(df)} data points")
            
            # 2. Generate features
            logger.info("Generating features...")
            df_features = self.feature_generator.generate_features(df)
            
            # 3. Save processed data
            output_path = Path("data/processed/features.csv")
            df_features.to_csv(output_path)
            logger.info(f"Saved processed data to {output_path}")
            
            # 4. Save feature metadata
            self._save_feature_metadata(df_features)
            
            return df_features
            
        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}")
            raise
    
    def _save_feature_metadata(self, df: pd.DataFrame) -> None:
        """Save feature metadata for reference"""
        try:
            metadata = {
                'feature_count': len(df.columns),
                'features': list(df.columns),
                'stats': {
                    col: {
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'min': df[col].min(),
                        'max': df[col].max()
                    }
                    for col in df.columns
                }
            }
            
            output_path = Path("data/processed/feature_metadata.yaml")
            with open(output_path, 'w') as f:
                yaml.dump(metadata, f)
            
            logger.info(f"Saved feature metadata to {output_path}")
            
        except Exception as e:
            logger.warning(f"Error saving feature metadata: {str(e)}")
    
    def update_dataset(self, lookback_days: int = 1) -> pd.DataFrame:
        """
        Update dataset with latest data
        
        Args:
            lookback_days: Number of days to look back for updating
            
        Returns:
            DataFrame with updated data and features
        """
        try:
            logger.info(f"Updating dataset with last {lookback_days} days of data")
            
            # 1. Get latest data
            df_new = self.data_loader.get_latest_data(lookback_days)
            
            # 2. Generate features
            df_features = self.feature_generator.generate_features(df_new)
            
            # 3. Load existing data
            existing_data_path = Path("data/processed/features.csv")
            if existing_data_path.exists():
                df_existing = pd.read_csv(existing_data_path)
                df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'])
                df_existing.set_index('timestamp', inplace=True)
                
                # 4. Merge and remove duplicates
                df_combined = pd.concat([df_existing, df_features])
                df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                df_combined = df_combined.sort_index()
                
                # 5. Save updated dataset
                df_combined.to_csv(existing_data_path)
                logger.info(f"Updated dataset saved to {existing_data_path}")
                
                return df_combined
            else:
                logger.warning("No existing dataset found, creating new one")
                return self.create_dataset()
            
        except Exception as e:
            logger.error(f"Error updating dataset: {str(e)}")
            raise

def main():
    """Main function to create or update dataset"""
    creator = DatasetCreator()
    
    # Create full dataset
    df = creator.create_dataset()
    print("\nDataset shape:", df.shape)
    print("\nFeatures:", list(df.columns))
    print("\nSample data:")
    print(df.head())

if __name__ == "__main__":
    main()