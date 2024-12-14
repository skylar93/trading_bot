"""Test hyperparameter optimization"""

import ray
from ray import tune
import numpy as np
import unittest
import pandas as pd
from pathlib import Path

class DummyEnv:
    def __init__(self):
        self.observation_space = np.zeros((10, 10))
        self.action_space = np.zeros(1)
        
    def step(self, action):
        return (
            np.random.random((10, 10)),  # observation
            float(np.random.random()),   # reward
            False,                       # done
            False,                       # truncated
            {}                          # info
        )
        
    def reset(self):
        return np.random.random((10, 10)), {}

def train_dummy_func(config):
    """Simple trainable function for testing"""
    for i in range(10):
        result = {
            "sharpe_ratio": float(np.random.random()),
            "sortino_ratio": float(np.random.random()),
            "max_drawdown": float(np.random.random()),
            "total_return": float(np.random.random())
        }
        tune.report(**result)

class TestHyperparameterOptimization(unittest.TestCase):
    def test_simple_tune(self):
        """Test basic Ray Tune functionality"""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            
        analysis = tune.run(
            train_dummy_func,
            config={
                "lr": tune.loguniform(1e-4, 1e-1),
                "batch_size": tune.choice([16, 32, 64])
            },
            num_samples=2,
            resources_per_trial={"cpu": 1}
        )
        
        # Check basic results
        self.assertIsNotNone(analysis.best_result)
        self.assertTrue("sharpe_ratio" in analysis.best_result)

    def setUp(self):
        """Clear Ray state before each test"""
        ray.shutdown()
        ray.init(ignore_reinit_error=True)
        
    def tearDown(self):
        """Clean up after each test"""
        ray.shutdown()

if __name__ == "__main__":
    unittest.main()