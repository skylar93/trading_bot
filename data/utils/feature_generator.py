import pandas as pd
import numpy as np
import ta
import logging
from typing import List, Callable

class FeatureGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_step = 0
        self.total_steps = 7  # Total number of feature generation steps

    def generate_features(self, df: pd.DataFrame, progress_callback: Callable = None) -> pd.DataFrame:
        """Generate technical features from OHLCV data"""
        result = df.copy()
        
        def update_progress(message: str):
            self.current_step += 1
            if progress_callback:
                progress_callback(self.current_step / self.total_steps, message)
            self.logger.info(message)
        
        # Start generating features
        update_progress("Starting feature generation...")

        # Moving Averages
        update_progress("Generating moving averages...")
        windows = [5, 10, 20, 50]
        for window in windows:
            result[f'sma_{window}'] = ta.trend.sma_indicator(df['close'], window=window)
            result[f'ema_{window}'] = ta.trend.ema_indicator(df['close'], window=window)

        # Bollinger Bands
        update_progress("Generating Bollinger Bands...")
        for window in [20]:
            indicator_bb = ta.volatility.BollingerBands(close=df["close"], window=window, window_dev=2)
            result[f'bb_high_{window}'] = indicator_bb.bollinger_hband()
            result[f'bb_mid_{window}'] = indicator_bb.bollinger_mavg()
            result[f'bb_low_{window}'] = indicator_bb.bollinger_lband()

        # RSI
        update_progress("Generating RSI...")
        for window in [14]:
            result[f'rsi_{window}'] = ta.momentum.rsi(df['close'], window=window)

        # MACD
        update_progress("Generating MACD...")
        macd = ta.trend.MACD(close=df['close'])
        result['macd'] = macd.macd()
        result['macd_signal'] = macd.macd_signal()
        result['macd_diff'] = macd.macd_diff()

        # Momentum
        update_progress("Generating momentum indicators...")
        result['momentum'] = ta.momentum.awesome_oscillator(df['high'], df['low'])
        result['roc'] = ta.momentum.roc(df['close'])

        # Clean data
        update_progress("Cleaning and finalizing data...")
        result = result.fillna(method='bfill').fillna(method='ffill')
        
        num_features = len(result.columns) - len(df.columns)
        self.logger.info(f"Generated {num_features} new features")
        
        return result