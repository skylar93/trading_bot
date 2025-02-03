'''
BaseBacktester Unit Tests: verifies single‐asset vs multi‐asset, 
fees, partial sells, basic metrics, etc. We call update(...) directly.
'''

import pytest
import pandas as pd
from training.backtesting.base_backtester import BaseBacktester

def test_single_asset_mode():
    """Test backtester in single-asset mode"""
    backtester = BaseBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        max_position=1.0
    )
    
    # Test initial state
    assert backtester.cash == 10000.0
    assert len(backtester.positions) == 0  # Empty positions dict
    assert len(backtester.trades) == 0
    assert len(backtester.portfolio_history) == 1
    assert backtester.portfolio_history[0] == 10000.0
    
    # Test simple buy
    timestamp = pd.Timestamp('2023-01-01')
    result = backtester.update(
        timestamp=timestamp,
        prices={'default': 100.0},
        actions={'default': 0.5}  # Buy 50% position
    )
    
    assert result['trades']['default']['success']
    assert backtester.positions['default'] > 0
    assert len(backtester.trades) == 1
    assert backtester.portfolio_history[-1] < 10000.0  # Should be less due to fees

def test_multi_asset_mode():
    """Test backtester in multi-asset mode"""
    backtester = BaseBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        max_position=1.0
    )
    
    # Test buying multiple assets
    timestamp = pd.Timestamp('2023-01-01')
    result = backtester.update(
        timestamp=timestamp,
        prices={'BTC': 100.0, 'ETH': 50.0},
        actions={'BTC': 0.3, 'ETH': 0.3}  # 30% position in each
    )
    
    assert result['trades']['BTC']['success']
    assert result['trades']['ETH']['success']
    assert 'BTC' in backtester.positions
    assert 'ETH' in backtester.positions
    assert len(backtester.trades) == 2
    assert backtester.portfolio_history[-1] < 10000.0  # Should be less due to fees

def test_trading_fees():
    """Test proper handling of trading fees"""
    initial_capital = 10000.0
    fee_rate = 0.01  # 1% fee
    backtester = BaseBacktester(
        initial_capital=initial_capital,
        trading_fee=fee_rate,  # 1% fee for clear impact
        max_position=1.0
    )
    
    # Buy with significant fee
    price = 100.0
    timestamp = pd.Timestamp('2023-01-01')
    result = backtester.update(
        timestamp=timestamp,
        prices={'default': price},
        actions={'default': 1.0}  # Full position
    )
    
    # Calculate expected values
    # For a full position with 1% fee:
    # If x is the trade value, then x + 0.01x = 10000
    # 1.01x = 10000
    # x = 10000/1.01
    trade_value = initial_capital / (1 + fee_rate)
    expected_fee = trade_value * fee_rate
    expected_position_units = trade_value / price
    expected_cash = initial_capital - trade_value - expected_fee  # Should be close to 0
    expected_portfolio_value = (expected_position_units * price)  # Current position value
    
    # Verify fee calculation
    assert result['trades']['default']['fee'] == pytest.approx(expected_fee, rel=1e-2)
    
    # Verify position
    assert backtester.positions['default'] == pytest.approx(expected_position_units, rel=1e-2)
    
    # Verify cash balance
    assert backtester.cash == pytest.approx(expected_cash, rel=1e-2)
    
    # Verify portfolio value
    assert backtester.portfolio_history[-1] == pytest.approx(expected_portfolio_value, rel=1e-2)
    assert backtester.portfolio_history[-1] < initial_capital  # Should be less due to fees

def test_position_limits():
    """Test max position size limits"""
    backtester = BaseBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        max_position=0.5  # 50% max position
    )
    
    # Try to take full position
    timestamp = pd.Timestamp('2023-01-01')
    result = backtester.update(
        timestamp=timestamp,
        prices={'default': 100.0},
        actions={'default': 1.0}  # Try full position
    )
    
    # Should be limited to 50%
    position_value = backtester.positions['default'] * 100.0
    assert position_value == pytest.approx(5000.0, rel=1e-2)

def test_insufficient_funds():
    """Test handling of insufficient funds"""
    backtester = BaseBacktester(
        initial_capital=100.0,  # Small capital
        trading_fee=0.001,
        max_position=1.0
    )
    
    # Try to buy more than we can afford
    timestamp = pd.Timestamp('2023-01-01')
    result = backtester.update(
        timestamp=timestamp,
        prices={'default': 200.0},  # Price higher than capital
        actions={'default': 1.0}
    )
    
    # Should fail gracefully
    assert not result['trades']['default']['success']
    assert result['trades']['default'].get('reason') == 'insufficient_funds'

def test_buy_sell_sequence():
    """Test a sequence of buy and sell trades across multiple assets"""
    backtester = BaseBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        max_position=1.0
    )
    
    # Buy sequence
    timestamps = [pd.Timestamp(f"2024-01-0{i}") for i in range(1, 4)]
    prices = [
        {"BTC": 100.0, "ETH": 50.0},  # Day 1
        {"BTC": 110.0, "ETH": 45.0},  # Day 2
        {"BTC": 90.0, "ETH": 55.0},   # Day 3
    ]
    actions = [
        {"BTC": 0.5, "ETH": 0.3},     # Buy both
        {"BTC": -0.2, "ETH": 0.2},    # Reduce BTC, increase ETH
        {"BTC": 0.0, "ETH": 0.0},     # Sell all
    ]
    
    portfolio_values = []
    for t, p, a in zip(timestamps, prices, actions):
        result = backtester.update(t, p, a)
        portfolio_values.append(result['portfolio_value'])
        
    # Verify trading sequence
    assert len(backtester.trades) > 0
    
    # First day trades
    day1_trades = {t['symbol']: t for t in backtester.trades[:2]}
    assert day1_trades['BTC']['type'] == 'buy'
    assert day1_trades['ETH']['type'] == 'buy'
    
    # Second day trades
    day2_trades = {t['symbol']: t for t in backtester.trades[2:4]}
    assert day2_trades['BTC']['type'] == 'sell'  # Reducing position
    assert day2_trades['ETH']['type'] == 'buy'   # Increasing position
    
    # Final day - all positions should be closed
    assert len(backtester.positions) == 0
    
    # Get trade history
    trade_df = backtester.get_trade_history()
    assert len(trade_df) == len(backtester.trades)
    assert 'symbol' in trade_df.columns
    assert 'type' in trade_df.columns
    
    # Get position history
    pos_df = backtester.get_position_history()
    assert len(pos_df) == len(backtester.portfolio_history)
    assert 'total' in pos_df.columns
    
    # Get returns
    returns = backtester.get_returns()
    assert len(returns) == len(backtester.portfolio_history)
    assert isinstance(returns, pd.Series)

def test_metrics_calculation():
    """
    Test that the backtester calculates cost/revenue/profit for each trade,
    and that overall metrics (win rate, final balance, etc.) are non-zero
    when there are winning and losing trades.
    """
    backtester = BaseBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,  # 0.1% fee
        max_position=1.0
    )

    # Day 1: Buy half position at price=100
    t1 = pd.Timestamp("2025-01-01")
    backtester.update(
        timestamp=t1,
        prices={"default": 100.0},
        actions={"default": 0.5}  # 50% position
    )

    # Day 2: Price rises to 120, buy more (making cost higher),
    # so that partial sells later can realize different PnLs
    t2 = pd.Timestamp("2025-01-02")
    backtester.update(
        timestamp=t2,
        prices={"default": 120.0},
        actions={"default": 0.3}  # further +30% => bigger position
    )

    # Day 3: Price at 110, let's do a partial sell (some profit realized, but price is down from 120)
    t3 = pd.Timestamp("2025-01-03")
    backtester.update(
        timestamp=t3,
        prices={"default": 110.0},
        actions={"default": -0.2}  # reduce position by 20% => partial close
    )

    # Day 4: Price rises to 130, sell everything => big profit
    t4 = pd.Timestamp("2025-01-04")
    backtester.update(
        timestamp=t4,
        prices={"default": 130.0},
        actions={"default": 0.0}  # fully close position
    )

    # Day 5: Price is 90, intentionally buy 80% => big position
    # Next day price falls => losing trade
    t5 = pd.Timestamp("2025-01-05")
    backtester.update(
        timestamp=t5,
        prices={"default": 90.0},
        actions={"default": 0.8}  # 80% position
    )

    # Day 6: Price drops to 70 => sell all => losing trade
    t6 = pd.Timestamp("2025-01-06")
    backtester.update(
        timestamp=t6,
        prices={"default": 70.0},
        actions={"default": 0.0}  # close (loss)
    )

    # 1) Check each recorded trade for cost/revenue/profit
    for i, trade in enumerate(backtester.trades, start=1):
        assert "cost" in trade, f"Trade #{i} missing 'cost'"
        assert "revenue" in trade, f"Trade #{i} missing 'revenue'"
        assert "profit" in trade, f"Trade #{i} missing 'profit'"

    # 2) Check the final metrics
    metrics = backtester._calculate_metrics()

    # We expect:
    # - total_trades > 0
    # - some trades were winning, some losing => non-zero 'win_rate' if at least one winner
    # - final_balance different from initial
    assert metrics["total_trades"] > 0, "No trades recorded"
    assert metrics["win_rate"] > 0, "Expected some winning trades => win_rate should be > 0"
    # We also expect the final balance could be > or < initial depending on net PnL
    assert metrics["final_balance"] != 10000.0, "Final balance unchanged => no realized PnL?"

    # Optionally check that final_portfolio_value is also not the same as initial
    assert metrics["final_portfolio_value"] != 10000.0, "No net change in portfolio value?"

    # 3) Check Sharpe ratio is non-zero (we have both winning and losing trades)
    assert metrics["sharpe_ratio"] != 0, "Sharpe ratio unexpectedly zero" 