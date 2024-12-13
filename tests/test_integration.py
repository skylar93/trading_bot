import pytest
import pandas as pd
import numpy as np
import os
import yaml
from pathlib import Path
from datetime import datetime

from data.utils.data_loader import DataLoader
from envs.base_env import TradingEnvironment
from envs.wrap_env import make_env
from training.train import load_config, create_env

class TestIntegration:
    @pytest.fixture
    def config(self):
        """Load configuration"""
        config_path = Path("config/default_config.yaml")
        assert config_path.exists(), "Configuration file not found"
        return load_config(str(config_path))
    
    def test_data_to_env_pipeline(self, config):
        """Test data pipeline integration with environment"""
        # 1. Load data
        loader = DataLoader(config['data']['exchange'])
        df = loader.fetch_and_process(
            symbol=config['data']['symbols'][0],
            timeframe=config['data']['timeframe'],
            start_date=config['data']['start_date'],
            limit=100  # Use small dataset for testing
        )
        
        assert not df.empty, "Failed to load data"
        assert all(col in df.columns for col in ['$open', '$high', '$low', '$close', '$volume'])
        
        # 2. Create environment
        env = TradingEnvironment(
            df=df,
            initial_balance=config['env']['initial_balance'],
            trading_fee=config['env']['trading_fee'],
            window_size=config['env']['window_size'],
            max_position_size=config['env']['max_position_size']
        )
        
        # 3. Apply wrappers
        wrapped_env = make_env(
            env,
            normalize=config['env']['normalize'],
            stack_size=config['env']['stack_size']
        )
        
        # 4. Test environment functionality
        obs = wrapped_env.reset()
        assert isinstance(obs, np.ndarray), "Observation should be numpy array"
        assert obs.shape[0] == (len(env.feature_columns) * env.window_size + 4) * config['env']['stack_size'], \
            "Observation shape mismatch"
        
        # 5. Test environment step
        action = np.array([0.5])  # Buy position with 50% size
        obs, reward, done, info = wrapped_env.step(action)
        
        assert isinstance(obs, np.ndarray), "Step observation should be numpy array"
        assert isinstance(reward, float), "Reward should be float"
        assert isinstance(done, bool), "Done should be boolean"
        assert isinstance(info, dict), "Info should be dictionary"
        assert info['balance'] > 0, "Balance should be positive"
        
    def test_create_env_function(self, config):
        """Test environment creation function used by RLlib"""
        # Create environment config
        env_config = {
            **config['env'],
            **config['data'],
            "symbol": config['data']['symbols'][0]
        }
        
        # Create environment
        env = create_env(env_config)
        
        # Basic environment tests
        obs = env.reset()
        assert isinstance(obs, np.ndarray), "Observation should be numpy array"
        
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        assert isinstance(obs, np.ndarray), "Step observation should be numpy array"
        assert isinstance(reward, float), "Reward should be float"
        assert isinstance(done, bool), "Done should be boolean"
        assert isinstance(info, dict), "Info should be dictionary"
    
    def test_full_episode(self, config):
        """Test running a full episode"""
        # Create environment config
        env_config = {
            **config['env'],
            **config['data'],
            "symbol": config['data']['symbols'][0]
        }
        
        # Create environment
        env = create_env(env_config)
        obs = env.reset()
        
        # Run episode
        done = False
        total_reward = 0
        step_count = 0
        max_steps = 100  # Limit steps for testing
        
        while not done and step_count < max_steps:
            # Take random action
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            # Validate step outputs
            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert isinstance(done, bool)
            assert isinstance(info, dict)
            
            total_reward += reward
            step_count += 1
            
            # Validate observation bounds if normalized
            if config['env']['normalize']:
                assert np.all(obs >= -1) and np.all(obs <= 1), \
                    "Normalized observation out of bounds"
        
        assert step_count > 0, "Episode should have at least one step"
    
    def test_state_transitions(self, config):
        """Test environment state transitions"""
        # Create environment
        env_config = {
            **config['env'],
            **config['data'],
            "symbol": config['data']['symbols'][0]
        }
        env = create_env(env_config)
        
        # Test buy -> sell transition
        obs = env.reset()
        
        # Open buy position
        obs1, reward1, done1, info1 = env.step(np.array([1.0]))
        assert info1.get('position') is not None, "Position should be opened"
        assert info1['position'].type == 'long', "Position should be long"
        
        # Close position with sell
        obs2, reward2, done2, info2 = env.step(np.array([-1.0]))
        assert info2.get('position') is not None, "Position should be opened"
        assert info2['position'].type == 'short', "Position should be short"
        
        # Test position size limits
        obs = env.reset()
        action = np.array([1.5])  # Try to open position larger than max
        obs, reward, done, info = env.step(action)
        assert info['position'].size <= env_config['max_position_size'] * info['balance'], \
            "Position size should be limited"
    
    def test_reward_calculation(self, config):
        """Test reward calculation"""
        # Create environment
        env_config = {
            **config['env'],
            **config['data'],
            "symbol": config['data']['symbols'][0]
        }
        env = create_env(env_config)
        
        # Reset environment
        obs = env.reset()
        initial_balance = env_config['initial_balance']
        
        # Test reward for successful trade
        # 1. Open long position
        obs, reward1, _, info1 = env.step(np.array([1.0]))
        entry_price = info1['current_price']
        
        # 2. Close position after price increase
        # (This is a simplified test as we can't directly control price movement
        # in real market data)
        obs, reward2, _, info2 = env.step(np.array([-1.0]))
        exit_price = info2['current_price']
        
        # At least verify reward signs make sense
        if exit_price > entry_price:
            assert reward2 >= 0, "Profit should give positive reward"
        elif exit_price < entry_price:
            assert reward2 <= 0, "Loss should give negative reward"