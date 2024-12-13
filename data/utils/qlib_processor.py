import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging
from qlib.data import D
from qlib.config import C

logger = logging.getLogger(__name__)

class QlibProcessor:
    """Process data into Qlib format"""
    
    def __init__(self, data_dir: str = "data/qlib_data"):
        """
        Initialize QlibProcessor
        
        Args:
            data_dir: Directory to store Qlib formatted data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def process_data(self, df: pd.DataFrame, symbol: str) -> None:
        """
        Process DataFrame into Qlib format
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol name (e.g., "BTC/USDT")
        """
        try:
            # Prepare data in Qlib format
            formatted_data = {
                "instrument": symbol.replace("/", ""),
                "start_time": df.index.min(),
                "end_time": df.index.max(),
                "fields": ["open", "high", "low", "close", "volume"],
                "data": df
            }
            
            # Save to csv in Qlib format
            output_file = self.data_dir / f"{symbol.replace('/', '')}.csv"
            
            # Format data for Qlib
            qlib_df = df.copy()
            qlib_df.index.name = "datetime"
            qlib_df.reset_index(inplace=True)
            
            # Save to csv
            qlib_df.to_csv(output_file, index=False)
            logger.info(f"Saved Qlib formatted data to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise