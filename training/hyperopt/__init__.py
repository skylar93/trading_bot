"""
Hyperparameter optimization module
"""

from .hyperopt_env import SimplifiedTradingEnv
from .hyperopt_agent import MinimalPPOAgent
from .hyperopt_tuner import MinimalTuner, train_agent

# Expose MinimalTuner as HyperparameterOptimizer for backward compatibility
HyperparameterOptimizer = MinimalTuner

__all__ = ['SimplifiedTradingEnv', 'MinimalPPOAgent', 'MinimalTuner', 'train_agent']