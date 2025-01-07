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
    """Test that risk management is properly integrated"""
    settings = create_test_settings()
    settings["max_position_size"] = 10  # Set to 10% to test position size limits
    manager = BacktestManager(settings)
    data = create_test_data(200)
    
    results = manager.run_backtest(data)
    
    # Check position sizes in trades
    for trade in results["trades"]:
        if trade["type"] == "none":
            continue
            
        position_value = trade["size"] * trade["price"]
        portfolio_value = trade.get("portfolio_value", settings["initial_balance"])
        position_size_pct = (position_value / portfolio_value) * 100
        
        # Position size should not exceed limit (with small tolerance for floating point errors)
        assert position_size_pct <= settings["max_position_size"] * 1.001, \
            f"Position size {position_size_pct}% exceeds limit {settings['max_position_size']}%"

def test_trade_execution_logging():
    """Test detailed logging of trade execution"""
    settings = create_test_settings()
    manager = BacktestManager(settings)
    data = create_test_data(200)
    
    # Run backtest
    results = manager.run_backtest(data)
    
    # Track previous trades to verify PnL calculations
    position_size = 0
    entry_price = 0
    
    # Verify trade details
    for trade in results["trades"]:
        assert all(key in trade for key in ["timestamp", "type", "price", "size"]), \
            "Trade should have all required fields"
        
        if trade["type"] == "buy":
            assert "cost" in trade, "Buy trade should have cost"
            # Update position tracking
            position_size = trade["size"]
            entry_price = trade["price"]
            # Verify cost calculation
            expected_cost = trade["size"] * trade["price"] * (1 + settings["trading_fee"])
            assert abs(trade["cost"] - expected_cost) < 0.01, \
                f"Cost calculation error: {trade['cost']} != {expected_cost}"
            
        elif trade["type"] == "sell":
            assert "revenue" in trade, "Sell trade should have revenue"
            # Verify revenue calculation
            expected_revenue = trade["size"] * trade["price"] * (1 - settings["trading_fee"])
            assert abs(trade["revenue"] - expected_revenue) < 0.01, \
                f"Revenue calculation error: {trade['revenue']} != {expected_revenue}"
            # Verify PnL calculation if we had a previous position
            if position_size > 0:
                expected_pnl = trade["size"] * (trade["price"] - entry_price) - \
                             trade["size"] * trade["price"] * settings["trading_fee"]
                assert abs(trade["pnl"] - expected_pnl) < 0.01, \
                    f"PnL calculation error: {trade['pnl']} != {expected_pnl}"
            
        # Verify portfolio value is tracked
        assert "portfolio_value" in trade, "Trade should track portfolio value"
        assert trade["portfolio_value"] > 0, "Portfolio value should be positive"
        
        # Verify risk metrics
        assert "risk_metrics" in trade, "Trade should have risk metrics"
        assert all(key in trade["risk_metrics"] for key in ["volatility", "leverage", "position_pnl"]), \
            "Risk metrics should have required fields"

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