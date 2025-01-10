# Single-Asset Backtesting System

## Overview

The `Backtester` class provides a simple and efficient implementation for backtesting single-asset trading strategies. It focuses on clarity and ease of use, making it ideal for strategy development and testing.

## Core Features

### 1. Trade Execution
- Position sizing based on strategy actions (-1 to 1)
- Transaction cost consideration
- Balance and position tracking
- Dust position cleanup (< 1e-4)

### 2. Performance Tracking
- Portfolio value history
- Trade history logging
- Performance metrics calculation
- Peak value tracking for drawdown

### 3. Strategy Integration
- Window-based strategy execution
- Action validation and processing
- Trade result accumulation

## Implementation Details

### Data Requirements
- OHLCV data format with '$' prefix
- Required columns: $open, $high, $low, $close, $volume
- DataFrame index serves as timestamp

### Position Management
- Single position tracking
- No multi-asset support
- Balance updates include transaction fees
- Automatic dust position cleanup

### Performance Metrics
- Sharpe Ratio (annualized, risk-free rate = 0)
- Sortino Ratio (downside deviation)
- Maximum Drawdown
- Win Rate calculation

## Usage Example

```python
# Prepare OHLCV data
data = pd.DataFrame({
    '$open': [...],
    '$high': [...],
    '$low': [...],
    '$close': [...],
    '$volume': [...]
})

# Initialize backtester
backtester = Backtester(
    data=data,
    initial_balance=10000.0,
    trading_fee=0.001  # 0.1% fee
)

# Create strategy
class SimpleStrategy:
    def get_action(self, window_data):
        return 0.5  # Example: always buy with 50% size

# Run backtest
results = backtester.run(
    strategy=SimpleStrategy(),
    window_size=20,
    verbose=True
)
```

## Logging Structure

The system uses Python's logging module with the following levels:
- ERROR: Critical failures (e.g., strategy errors)
- WARNING: Potential issues (e.g., invalid actions)
- INFO: Trade execution, progress updates
- DEBUG: Detailed calculations, state changes

## Recent Changes

- Enhanced logging for better debugging
- Improved PnL calculation accuracy
- Added detailed trade history
- Fixed position size validation

## Related Components

- BacktestEngine: Multi-asset version
- RiskAwareBacktester: Version with risk management
- ExperimentalBacktester: Advanced features testing 