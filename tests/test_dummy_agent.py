import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from deployment.web_interface.utils.backtest import DummyAgent
from deployment.web_interface.utils.data_stream import DataStream

def create_test_data(periods: int = 100) -> pd.DataFrame:
    """Create test market data"""
    dates = pd.date_range(start=datetime.now(), periods=periods, freq='1min')
    data = pd.DataFrame(index=dates)
    
    # Generate sample price data
    base_price = 100.0
    noise = np.random.normal(0, 0.001, periods)
    trend = np.linspace(0, 0.1, periods)
    prices = base_price * (1 + noise + trend)
    
    data["$open"] = prices * (1 + np.random.normal(0, 0.0002, periods))
    data["$high"] = data["$open"] * (1 + abs(np.random.normal(0, 0.001, periods)))
    data["$low"] = data["$open"] * (1 - abs(np.random.normal(0, 0.001, periods)))
    data["$close"] = prices
    data["$volume"] = np.random.lognormal(10, 1, periods)
    
    return data

def test_dummy_agent_action_generation():
    """Test that DummyAgent generates valid actions"""
    agent = DummyAgent()
    data = create_test_data(100)
    
    # Test multiple actions
    actions = []
    for i in range(20):
        action = agent.get_action(data)
        assert isinstance(action, (int, float, np.ndarray)), f"Action should be numeric, got {type(action)}"
        assert -1.0 <= action <= 1.0, f"Action should be between -1 and 1, got {action}"
        actions.append(action)
    
    # Verify some actions are non-zero (agent should trade sometimes)
    non_zero_actions = [a for a in actions if abs(a) > 0]
    assert len(non_zero_actions) > 0, "Agent should generate some non-zero actions"
    
    # Verify trading frequency (every 5 steps)
    zero_actions = [a for a in actions if abs(a) == 0]
    assert len(zero_actions) >= 15, "Agent should hold (action=0) most of the time"

def test_dummy_agent_with_datastream():
    """Test DummyAgent with actual DataStream data"""
    data_stream = DataStream()
    agent = DummyAgent()
    
    # Get data from stream
    data = data_stream.get_current_data(lookback=100)
    assert not data.empty, "DataStream should provide non-empty data"
    
    # Test action generation with stream data
    action = agent.get_action(data)
    assert isinstance(action, (int, float, np.ndarray)), f"Action should be numeric, got {type(action)}"
    assert -1.0 <= action <= 1.0, f"Action should be between -1 and 1, got {action}"

def test_dummy_agent_consistency():
    """Test that DummyAgent's actions are consistent with its trading frequency"""
    agent = DummyAgent()
    data = create_test_data(100)
    
    # Test actions for 100 steps
    actions = []
    for i in range(100):
        action = agent.get_action(data)
        actions.append(action)
        
        # Every 5th step should potentially have a trade
        if i % 5 == 0:
            assert abs(action) > 0, f"Expected non-zero action at step {i}"
        else:
            assert abs(action) == 0, f"Expected zero action at step {i}"

def test_dummy_agent_action_values():
    """Test that DummyAgent's action values are appropriate"""
    agent = DummyAgent()
    data = create_test_data(100)
    
    # Collect trading actions (non-zero)
    trading_actions = []
    for i in range(100):
        action = agent.get_action(data)
        if abs(action) > 0:
            trading_actions.append(action)
    
    # Verify action values
    for action in trading_actions:
        assert abs(action) == 0.5, f"Trading action should be 0.5 or -0.5, got {action}" 