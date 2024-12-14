"""Test hyperparameter optimization"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from training.hyperopt import HyperparameterOptimizer

def generate_test_data(length: int = 1000) -> pd.DataFrame:
    """Generate test market data"""
    dates = pd.date_range(start='2024-01-01', periods=length, freq='1h')
    
    # Generate price process
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, length)
    price = 100 * np.exp(np.cumsum(returns))
    
    return pd.DataFrame({
        'open': price,
        'high': price * (1 + np.random.uniform(0, 0.001, length)),
        'low': price * (1 - np.random.uniform(0, 0.001, length)),
        'close': price,
        'volume': np.random.uniform(1000, 5000, length)
    }, index=dates)

class TestHyperparameterOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Generate test data
        data = generate_test_data(1000)
        train_size = int(len(data) * 0.7)
        cls.train_data = data[:train_size]
        cls.val_data = data[train_size:]
        
        # Create test config
        config_path = Path("tests/test_config.yaml")
        if not config_path.exists():
            config = {
                'env': {
                    'initial_balance': 10000,
                    'trading_fee': 0.001,
                    'window_size': 20
                },
                'model': {
                    'hidden_size': 256,
                    'num_layers': 2
                },
                'training': {
                    'batch_size': 128,
                    'learning_rate': 0.0003,
                    'num_episodes': 10
                }
            }
            config_path.parent.mkdir(exist_ok=True)
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
                
        cls.config_path = str(config_path)
    
    def setUp(self):
        """Set up each test"""
        self.optimizer = HyperparameterOptimizer(self.config_path)
    
    def test_optimization_run(self):
        """Test if optimization runs without errors"""
        # Run optimization with minimal settings for testing
        best_params = self.optimizer.optimize(
            train_data=self.train_data,
            val_data=self.val_data,
            num_trials=2,  # Small number for testing
            cpus_per_trial=1,
            gpus_per_trial=0
        )
        
        # Check if best parameters were found
        self.assertIsNotNone(best_params)
        self.assertIsInstance(best_params, dict)
        
        # Check if all expected parameters are present
        expected_params = [
            'hidden_size', 'num_layers', 'learning_rate',
            'batch_size', 'gamma'
        ]
        for param in expected_params:
            self.assertIn(param, best_params)
    
    def test_results_logging(self):
        """Test if results are properly logged"""
        # Run optimization
        self.optimizer.optimize(
            train_data=self.train_data,
            val_data=self.val_data,
            num_trials=2
        )
        
        # Check if results file was created
        results_path = Path("results/hyperopt/optimization_results.csv")
        self.assertTrue(results_path.exists())
        
        # Check if results file contains expected data
        results_df = pd.read_csv(results_path)
        expected_columns = [
            'sharpe_ratio', 'sortino_ratio',
            'max_drawdown', 'total_return'
        ]
        for col in expected_columns:
            self.assertIn(col, results_df.columns)

if __name__ == '__main__':
    unittest.main()