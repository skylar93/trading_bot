'''
Backtest Execution Tests: ensures that BaseBacktester.run(...) 
with a dummy strategy can produce trades, compute metrics, and handle basic scenarios. 
Not testing advanced risk or scenario logic here.
'''

import pandas as pd
import numpy as np
from training.backtesting.base_backtester import BaseBacktester
from training.backtesting.risk_aware_backtester import RiskAwareBacktester
from training.backtesting.risk_manager import RiskConfig

def create_test_settings():
    """Create test settings for backtesting"""
    return {
        "symbol": "BTC/USDT",
        "timeframe": "1m",
        "initial_capital": 10000.0,
        "max_position": 0.5,  # 50%
        "stop_loss": 0.02,  # 2%
        "max_drawdown": 0.15,  # 15%
        "trading_fee": 0.001
    }

def create_test_data(n_samples: int = 100) -> pd.DataFrame:
    """Create test market data with simple $-prefixed columns for testing"""
    np.random.seed(42)
    
    # Generate random price data
    base_price = 100
    returns = np.random.normal(0, 0.02, n_samples)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create simple $-prefixed columns for testing
    df = pd.DataFrame({
        "$open": prices * (1 + np.random.normal(0, 0.001, n_samples)),
        "$high": prices * (1 + np.random.normal(0.005, 0.001, n_samples)),
        "$low": prices * (1 + np.random.normal(-0.005, 0.001, n_samples)),
        "$close": prices,
        "$volume": np.random.lognormal(10, 1, n_samples)
    })
    
    # Add datetime index
    df.index = pd.date_range(
        start=pd.Timestamp("2023-01-01"),
        periods=n_samples,
        freq="1min"
    )
    
    return df

class DummyStrategy:
    def __init__(self):
        self.last_action = -1  # Start with sell so first action will be buy
        
    def get_action(self, window_data):
        """Simple strategy that alternates between buy and sell"""
        self.last_action *= -1  # Alternate between 1 and -1
        return self.last_action  # Always trade with full size

def test_backtest_initialization():
    """Test backtest manager initialization"""
    settings = create_test_settings()
    data = create_test_data()
    backtester = BaseBacktester(
        data=data,
        initial_capital=settings["initial_capital"],
        trading_fee=settings["trading_fee"],
        max_position=settings["max_position"]
    )
    assert backtester is not None
    assert backtester.data is not None
    assert len(backtester.data) == len(data)

def test_backtest_execution():
    """Test basic backtest execution"""
    settings = create_test_settings()
    data = create_test_data()
    strategy = DummyStrategy()
    
    backtester = BaseBacktester(
        data=data,
        initial_capital=settings["initial_capital"],
        trading_fee=settings["trading_fee"],
        max_position=settings["max_position"]
    )
    
    results = backtester.run(strategy)
    assert len(results["trades"]) > 0, "Should execute some trades"

def test_metrics_calculation():
    """Test that metrics are correctly calculated"""
    settings = create_test_settings()
    data = create_test_data()
    strategy = DummyStrategy()
    
    backtester = BaseBacktester(
        data=data,
        initial_capital=settings["initial_capital"],
        trading_fee=settings["trading_fee"],
        max_position=settings["max_position"]
    )
    
    results = backtester.run(strategy)
    metrics = results["metrics"]
    
    assert "total_return" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
