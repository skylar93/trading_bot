# Risk-Aware Backtesting System

## Overview
The Risk-Aware Backtesting System extends the base backtester with comprehensive risk management capabilities, providing accurate position sizing, PnL tracking, and multi-asset risk monitoring.

## File Structure

| File | Class/Function | Description | Dependencies |
|------|---------------|-------------|--------------|
| `risk_backtest.py` | `RiskAwareBacktester` | Main backtesting class with risk management | `Backtester`, `RiskManager`, `RiskConfig` |
| | `__init__()` | Initializes backtester with risk config | `RiskManager` |
| | `reset()` | Resets backtester state | Parent `reset()` |
| | `get_position_value()` | Calculates position value | `full_data` |
| | `calculate_volatility()` | Computes rolling volatility | `numpy` |
| | `get_current_leverage()` | Calculates leverage ratio | `get_position_value()` |
| | `execute_trade()` | Core trade execution with risk checks | `RiskManager` |
| | `run()` | Main backtesting loop | Trading agent |

## Key Features

### 1. Position Size Management
- Strict enforcement of maximum position size (e.g., 10%)
- Uses 0.999 buffer for float precision
- Dynamic adjustment based on portfolio value
```python
max_position_value = portfolio_value * max_position_size * 0.999
```

### 2. PnL Calculation
- Accurate formula with fee consideration:
```python
pnl = size * (price - entry_price) - size * price * fee
```
- Separate tracking for each asset
- Entry price updates on position changes

### 3. Multi-Asset Support
- Correlation matrix tracking
- Portfolio VaR monitoring
- Asset-specific position limits
- Unified data handling for single/multi-asset scenarios

## Implementation Details

### Data Handling
```python
# Multi-asset data structure
{
    "BTC_$open": [...],
    "BTC_$close": [...],
    "ETH_$open": [...],
    "ETH_$close": [...]
}
```

### Risk Integration
- Risk signal processing through `RiskManager`
- Position size adjustments based on:
  - Portfolio VaR limits
  - Asset correlations
  - Maximum drawdown
  - Leverage constraints

### Configuration Parameters
- `max_position_size`: Maximum position size (default: 0.1)
- `stop_loss_pct`: Stop-loss percentage
- `max_drawdown_pct`: Maximum drawdown limit
- `portfolio_var_limit`: Portfolio VaR limit
- `max_correlation`: Maximum allowed asset correlation
- `trading_fee`: Trading fee as decimal

## Usage Example

```python
# Initialize with risk configuration
risk_config = RiskConfig(max_position_size=0.1)
backtester = RiskAwareBacktester(df, risk_config)

# Run backtest
results = backtester.run(strategy)
print(f"Final PnL: {results['total_pnl']}")
```

## Testing

The system includes comprehensive test coverage:
- `test_risk_backtest.py`: Core functionality
- `test_risk_management.py`: Risk manager integration
- `test_portfolio_var_limits.py`: VaR constraints
- `test_correlation_based_position_sizing.py`: Multi-asset sizing

## Recent Updates
- Fixed position size limit enforcement
- Improved PnL calculation accuracy
- Added position verification steps
- Enhanced logging for debugging
- Added correlation-based position sizing
- Implemented portfolio VaR monitoring 