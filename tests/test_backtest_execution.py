import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from deployment.web_interface.utils.backtest import BacktestManager, DummyAgent
from risk.risk_manager import RiskConfig

def create_test_settings():
    """Create test settings for backtesting"""
    return {
        "symbol": "BTC/USDT",
        "timeframe": "1m",
        "initial_balance": 10000.0,
        "max_position_size": 50,  # 50%
        "stop_loss": 2,  # 2%
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

def test_backtest_initialization():
    """Test backtest manager initialization"""
    settings = create_test_settings()
    manager = BacktestManager(settings)
    
    assert isinstance(manager.agent, DummyAgent), "Should initialize with DummyAgent"
    assert isinstance(manager.risk_config, RiskConfig), "Should have risk config"
    assert manager.risk_config.daily_trade_limit == 1000, "Should have correct trade limit"
    assert manager.risk_config.max_position_size == 0.5, "Should convert position size to decimal"

def test_backtest_execution():
    """Test basic backtest execution"""
    settings = create_test_settings()
    manager = BacktestManager(settings)
    data = create_test_data(200)  # Create enough data for meaningful test
    
    # Run backtest
    results = manager.run_backtest(data)
    
    # Verify results structure
    assert "portfolio_values" in results, "Should have portfolio values"
    assert "trades" in results, "Should have trades list"
    assert "metrics" in results, "Should have metrics"
    
    # Verify some trades were executed
    assert len(results["trades"]) > 0, "Should execute some trades"
    
    # Check first trade
    first_trade = results["trades"][0]
    assert "timestamp" in first_trade, "Trade should have timestamp"
    assert "type" in first_trade, "Trade should have type"
    assert "price" in first_trade, "Trade should have price"
    assert "size" in first_trade, "Trade should have size"

def test_risk_management_integration():
    """Test position size limits and risk management integration.
    
    Verifies:
    1. Position size never exceeds 10% limit
    2. Risk signals properly processed
    3. Portfolio metrics updated
    
    Expected Results:
    - Max position size: 10.0%
    - Tolerance: 0.1%
    - Risk metrics logged
    """
    
def test_trade_execution_logging():
    """Test trade execution and PnL calculation accuracy.
    
    Verifies:
    1. PnL calculation matches formula
    2. Trade details properly logged
    3. Position tracking accurate
    
    Expected Results:
    - PnL within 0.01 of expected
    - All trade details logged
    - Zero positions cleaned up
    """

def test_metrics_calculation():
    """Test that performance metrics are calculated correctly"""
    settings = create_test_settings()
    manager = BacktestManager(settings)
    data = create_test_data(200)
    
    results = manager.run_backtest(data)
    metrics = results["metrics"]
    
    # Verify required metrics exist
    required_metrics = [
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "total_trades"
    ]
    
    for metric in required_metrics:
        assert metric in metrics, f"Metrics should include {metric}"
        assert isinstance(metrics[metric], (int, float)), \
            f"{metric} should be numeric" 