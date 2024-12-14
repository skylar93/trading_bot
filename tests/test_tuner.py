"""Test hyperparameter tuning system"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from training.utils.tuner import HyperParameterTuner

def generate_test_data(length: int = 1000) -> pd.DataFrame:
    """Generate test market data"""
    dates = pd.date_range(start='2024-01-01', periods=length, freq='1h')
    
    # Generate price process
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, length)
    price = 100 * np.exp(np.cumsum(returns))
    
    # Generate features
    df = pd.DataFrame({
        'timestamp': dates,
        'open': price * (1 + np.random.uniform(-0.001, 0.001, length)),
        'high': price * (1 + np.random.uniform(0, 0.002, length)),
        'low': price * (1 - np.random.uniform(0, 0.002, length)),
        'close': price,
        'volume': np.random.uniform(1000, 5000, length)
    })
    
    # Add some technical indicators
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['rsi'] = calculate_rsi(df['close'])
    
    return df.dropna()

def calculate_rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
    """Calculate RSI technical indicator"""
    delta = prices.diff()
    gain = (delta.clip(lower=0)).rolling(window=periods).mean()
    loss = (-delta.clip(upper=0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

class TestHyperParameterTuner(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Generate test data
        data = generate_test_data(1000)
        split_idx = int(len(data) * 0.7)
        
        self.train_data = data[:split_idx]
        self.val_data = data[split_idx:]
        
        # Initialize tuner
        self.tuner = HyperParameterTuner(
            data_train=self.train_data,
            data_val=self.val_data
        )
    
    def test_optimization_run(self):
        """Test hyperparameter optimization with small search space"""
        # Run optimization with minimal settings for testing
        best_config = self.tuner.run_optimization(
            num_samples=2,  # Small number for testing
            num_epochs=2,
            gpus_per_trial=0  # CPU only for testing
        )
        
        # Check if best config contains all expected parameters
        expected_params = {
            'window_size',
            'hidden_size',
            'num_layers',
            'learning_rate',
            'gamma',
            'lambda',
            'clip_param',
            'batch_size',
            'num_episodes'
        }
        
        self.assertEqual(set(best_config.keys()), expected_params)
        
        # Check parameter ranges
        self.assertTrue(0 < best_config['learning_rate'] < 1)
        self.assertTrue(0.9 <= best_config['gamma'] <= 0.999)
        self.assertTrue(0.9 <= best_config['lambda'] <= 1.0)
    
    def test_best_config_application(self):
        """Test applying best configuration"""
        # Create a sample best config
        sample_config = {
            'window_size': 20,
            'hidden_size': 128,
            'num_layers': 2,
            'learning_rate': 0.001,
            'gamma': 0.99,
            'lambda': 0.95,
            'clip_param': 0.2,
            'batch_size': 64,
            'num_episodes': 100
        }
        
        # Apply config
        pipeline = self.tuner.apply_best_config(sample_config)
        
        # Check if config was properly applied
        self.assertEqual(pipeline.config['env']['window_size'], 20)
        self.assertEqual(pipeline.config['model']['hidden_size'], 128)
        self.assertEqual(pipeline.config['training']['batch_size'], 64)

if __name__ == '__main__':
    unittest.main()