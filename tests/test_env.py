import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from envs.base_env import TradingEnvironment
from envs.wrap_env import make_env, NormalizeObservation, StackObservation

@pytest.fixture
def sample_env_data():
    """Create sample data for environment testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    np.random.seed(42)
    
    # Ensure price consistency (high > open/close > low)
    base_price = 100
    data = {
        'datetime': dates,
        '$open': base_price + np.random.normal(0, 1, len(dates)),
        '$close': base_price + np.random.normal(0, 1, len(dates)),
    }
    data['$high'] = np.maximum(data['$open'], data['$close']) + abs(np.random.normal(0, 0.5, len(dates)))
    data['$low'] = np.minimum(data['$open'], data['$close']) - abs(np.random.normal(0, 0.5, len(dates)))
    
    # Add other features
    data.update({
        '$volume': np.abs(np.random.normal(1000, 100, len(dates))),
        '$amount': np.abs(np.random.normal(100000, 10000, len(dates))),
        'RSI': np.random.uniform(0, 100, len(dates)),
        'MACD': np.random.normal(0, 1, len(dates)),
        'Signal': np.random.normal(0, 1, len(dates)),
        'BB_upper': base_price + 2,
        'BB_middle': base_price,
        'BB_lower': base_price - 2
    })
    
    return pd.DataFrame(data)

@pytest.fixture
def trading_env(sample_env_data):
    """Create trading environment instance"""
    return TradingEnvironment(
        df=sample_env_data,
        initial_balance=10000.0,
        trading_fee=0.001,
        window_size=10,
        max_position_size=1.0
    )

class TestTradingEnvironment:
    def test_initialization(self, trading_env):
        """Test environment initialization"""
        assert trading_env.initial_balance == 10000.0
        assert trading_env.trading_fee == 0.001
        assert trading_env.window_size == 10
        assert trading_env.max_position_size == 1.0
        assert trading_env.position is None
        
        # Check spaces
        assert trading_env.action_space.shape == (1,)
        assert trading_env.action_space.low == -1
        assert trading_env.action_space.high == 1
    
    def test_reset(self, trading_env):
        """Test environment reset"""
        obs, info = trading_env.reset(seed=42)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert trading_env.balance == trading_env.initial_balance
        assert trading_env.position is None
        assert len(trading_env.position_history) == 0
        
        # Check observation shape
        expected_obs_size = (trading_env.window_size, len(trading_env.feature_columns))
        assert obs.shape == expected_obs_size
    
    def test_step_buy_action(self, trading_env):
        """Test environment step with buy action"""
        trading_env.reset(seed=42)
        obs, reward, terminated, truncated, info = trading_env.step(np.array([1.0]))  # Full buy
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert trading_env.position is not None
        assert trading_env.position.type == 'long'
        
        # Check position size constraints
        assert trading_env.position.size > 0
        max_allowed_size = trading_env.max_position_size * trading_env.balance
        assert trading_env.position.size <= max_allowed_size
    
    def test_step_sell_action(self, trading_env):
        """Test environment step with sell action"""
        trading_env.reset(seed=42)
        obs, reward, terminated, truncated, info = trading_env.step(np.array([-1.0]))  # Full sell
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert trading_env.position is not None
        assert trading_env.position.type == 'short'
    
    def test_step_hold_action(self, trading_env):
        """Test environment step with hold action"""
        trading_env.reset(seed=42)
        obs, reward, terminated, truncated, info = trading_env.step(np.array([0.0]))  # Hold
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert trading_env.position is None  # No position should be opened
    
    def test_position_pnl(self, trading_env):
        """Test position profit/loss calculation"""
        trading_env.reset(seed=42)
        
        # Open long position
        _, _, _, _, info = trading_env.step(np.array([1.0]))
        entry_price = info['current_price']
        
        # Calculate expected PnL
        position_size = trading_env.position.size
        test_price_increase = entry_price * 1.1
        expected_profit = (test_price_increase - entry_price) * position_size
        actual_profit = trading_env.position.calculate_pnl(test_price_increase)
        assert np.isclose(actual_profit, expected_profit)
        
        test_price_decrease = entry_price * 0.9
        expected_loss = (test_price_decrease - entry_price) * position_size
        actual_loss = trading_env.position.calculate_pnl(test_price_decrease)
        assert np.isclose(actual_loss, expected_loss)

class TestEnvironmentWrappers:
    def test_normalize_observation(self, trading_env):
        """Test observation normalization wrapper"""
        env = NormalizeObservation(trading_env)
        obs, _ = env.reset(seed=42)
        
        assert isinstance(obs, np.ndarray)
        assert np.all(obs >= -1) and np.all(obs <= 1)  # Check bounds
    
    def test_stack_observation(self, trading_env):
        """Test observation stacking"""
        stack_size = 4
        wrapped_env = StackObservation(trading_env, stack_size=stack_size)
        obs, _ = wrapped_env.reset()
        
        # Check stacked observation shape
        n_features = len(trading_env.feature_columns)
        expected_shape = (trading_env.window_size, n_features)
        assert obs.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {obs.shape}"
    
    def test_full_wrapped_env(self, trading_env):
        """Test fully wrapped environment"""
        wrapped_env = make_env(
            trading_env,
            normalize=True,
            stack_size=4
        )
        
        obs, _ = wrapped_env.reset()
        n_features = len(trading_env.feature_columns)
        expected_shape = (trading_env.window_size, n_features)
        assert obs.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {obs.shape}"
        
        # Test step
        action = wrapped_env.action_space.sample()
        obs, reward, done, truncated, info = wrapped_env.step(action)
        assert obs.shape == expected_shape
        
        # Run full episode
        total_reward = 0
        step_count = 0
        done = False
        
        obs, info = wrapped_env.reset()
        while not done and step_count < 100:  # Limit steps for testing
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated
        
        assert step_count > 0, "Episode should have at least one step"
        assert 'episode' in info