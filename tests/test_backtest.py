import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from training.backtest import Backtester
from training.evaluation import TradingMetrics

class DummyAgent:
    """Dummy agent for testing"""
    def get_action(self, state):
        """Random action between -1 and 1"""
        return np.random.uniform(-1, 1)

@pytest.fixture
def sample_data():
    """Create sample price data"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1h')
    df = pd.DataFrame({
        'open': np.random.randn(len(dates)) * 10 + 100,
        'high': np.random.randn(len(dates)) * 10 + 100,
        'low': np.random.randn(len(dates)) * 10 + 100,
        'close': np.random.randn(len(dates)) * 10 + 100,
        'volume': np.abs(np.random.randn(len(dates)) * 1000)
    }, index=dates)
    
    # Ensure high is highest and low is lowest
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    return df

def test_backtester_initialization(sample_data):
    """Test Backtester initialization"""
    backtester = Backtester(sample_data)
    assert backtester.initial_balance == 10000.0
    assert backtester.position == 0
    assert len(backtester.trades) == 0
    assert backtester.portfolio_values == [10000.0]

def test_trade_execution(sample_data):
    """Test trade execution"""
    backtester = Backtester(sample_data)
    
    # Test buy action
    result = backtester.execute_trade(
        timestamp=sample_data.index[0],
        action=0.5,  # Buy 50% of possible position
        price_data={
            'open': 100,
            'high': 102,
            'low': 98,
            'close': 101
        }
    )
    
    assert result['action'] == 0.5
    assert result['balance'] < 10000.0  # Balance should decrease after buy
    assert result['position'] > 0  # Should have positive position
    
    # Test sell action
    result = backtester.execute_trade(
        timestamp=sample_data.index[1],
        action=-0.5,  # Sell 50% of position
        price_data={
            'open': 101,
            'high': 103,
            'low': 99,
            'close': 102
        }
    )
    
    assert result['action'] == -0.5
    assert backtester.position >= 0  # Position should never go negative

def test_backtest_run(sample_data):
    """Test complete backtest run"""
    backtester = Backtester(sample_data)
    agent = DummyAgent()
    
    results = backtester.run(agent, window_size=5, verbose=False)
    
    # Check results structure
    assert 'metrics' in results
    assert 'trades' in results
    assert 'portfolio_values' in results
    assert 'timestamps' in results
    
    # Check metrics
    metrics = results['metrics']
    assert 'total_return' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'sortino_ratio' in metrics
    assert 'max_drawdown' in metrics
    
    # Verify portfolio values
    assert len(results['portfolio_values']) > 1
    assert isinstance(results['portfolio_values'][0], (int, float))
    
    # Verify trades if any
    if results['trades']:
        trade = results['trades'][0]
        assert 'entry_time' in trade
        assert 'exit_time' in trade
        assert 'pnl' in trade

def test_metrics_calculation():
    """Test trading metrics calculation"""
    # Test Sharpe Ratio
    returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    sharpe = TradingMetrics.calculate_sharpe_ratio(returns)
    assert isinstance(sharpe, float)
    
    # Test Maximum Drawdown
    portfolio_values = np.array([100, 95, 105, 90, 100, 110])
    mdd, start_idx, end_idx = TradingMetrics.calculate_maximum_drawdown(portfolio_values)
    assert isinstance(mdd, float)
    assert mdd <= 0  # Drawdown should be negative
    assert start_idx < end_idx
    
    # Test Win Rate
    trades = [
        {'pnl': 100},
        {'pnl': -50},
        {'pnl': 75},
        {'pnl': 25}
    ]
    win_rate = TradingMetrics.calculate_win_rate(trades)
    assert win_rate == 75.0  # 3 out of 4 trades are profitable
    
    # Test Profit/Loss Ratio
    pnl_ratio = TradingMetrics.calculate_profit_loss_ratio(trades)
    assert pnl_ratio > 0
    assert isinstance(pnl_ratio, float)

def test_backtester_save_results(sample_data, tmp_path):
    """Test saving backtest results"""
    backtester = Backtester(sample_data)
    agent = DummyAgent()
    
    # Run backtest
    results = backtester.run(agent, window_size=5, verbose=False)
    
    # Save results
    save_dir = tmp_path / "backtest_results"
    backtester.save_results(results, save_dir)
    
    # Check if files were created
    assert (save_dir / "metrics.csv").exists()
    assert (save_dir / "trades.csv").exists()
    assert (save_dir / "portfolio_values.csv").exists()
    
    # Verify file contents
    metrics_df = pd.read_csv(save_dir / "metrics.csv")
    assert not metrics_df.empty
    assert "total_return" in metrics_df.columns
    
    if results['trades']:
        trades_df = pd.read_csv(save_dir / "trades.csv")
        assert not trades_df.empty
        assert "pnl" in trades_df.columns

def test_edge_cases(sample_data):
    """Test edge cases and error handling"""
    backtester = Backtester(sample_data)
    
    # Test zero action
    result = backtester.execute_trade(
        timestamp=sample_data.index[0],
        action=0,
        price_data={
            'open': 100,
            'high': 102,
            'low': 98,
            'close': 101
        }
    )
    assert result['action'] == 0
    assert result['position'] == 0
    
    # Test very small action
    result = backtester.execute_trade(
        timestamp=sample_data.index[0],
        action=1e-6,
        price_data={
            'open': 100,
            'high': 102,
            'low': 98,
            'close': 101
        }
    )
    assert result['action'] == 0
    
    # Test insufficient balance
    backtester.balance = 1  # Set very low balance
    result = backtester.execute_trade(
        timestamp=sample_data.index[0],
        action=1,
        price_data={
            'open': 100,
            'high': 102,
            'low': 98,
            'close': 101
        }
    )
    assert result['position'] == 0  # Should not execute trade

if __name__ == "__main__":
    pytest.main([__file__])