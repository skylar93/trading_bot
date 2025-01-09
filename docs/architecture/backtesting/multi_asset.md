# Multi-Asset Backtesting System

## Overview

The `BacktestEngine` class implements a sophisticated backtesting system supporting multiple assets simultaneously. It provides comprehensive portfolio management and performance tracking capabilities.

## Core Features

### 1. Portfolio Management
- Multi-asset position tracking
- Position size limits per asset
- Portfolio value calculation
- Cash balance management
- Entry price tracking per asset

### 2. Trade Execution
- Action processing (-1 to 1 per asset)
- Transaction cost handling
- Position updates and tracking
- Trade history logging
- Dust position cleanup

### 3. Performance Analysis
- Portfolio value history
- Return calculation
- Position history tracking
- Trade history management
- Cash flow tracking

## Implementation Details

### Position Management
- Uses dictionary for multi-asset positions
- Implements position limits per asset
- Handles cash balance separately
- Tracks entry prices for PnL
- Automatic position cleanup

### Trade Processing
- Supports simultaneous trades across assets
- Validates position limits before execution
- Updates portfolio state after each trade
- Maintains detailed trade history
- Handles transaction costs accurately

### Data Requirements
- Timestamp-indexed price data
- Asset identifiers as dictionary keys
- Price data for all tracked assets
- Consistent price format across assets

## Usage Example

```python
# Initialize engine
engine = BacktestEngine(
    initial_capital=100000.0,
    transaction_cost=0.001,  # 0.1% fee
    max_position=0.2        # 20% max per asset
)

# Prepare multi-asset data
prices = {
    'BTC': 50000.0,
    'ETH': 3000.0
}

# Define actions (-1 to 1 for each asset)
actions = {
    'BTC': 0.5,   # Buy 50% of allowed size
    'ETH': -0.3   # Sell 30% of allowed size
}

# Update portfolio
engine.update(
    timestamp=pd.Timestamp.now(),
    prices=prices,
    actions=actions
)

# Get results
print(f"Portfolio Value: {engine.get_portfolio_value(prices)}")
print(f"Returns: {engine.get_returns()}")
print(f"Positions:\\n{engine.get_position_history()}")
```

## Portfolio Tracking

### 1. Value Components
- Cash balance tracking
- Asset position values
- Total portfolio value
- Unrealized PnL

### 2. History Tracking
- Portfolio value over time
- Position sizes per asset
- Trade execution details
- Cash balance changes
- Entry and exit prices

### 3. Performance Metrics
- Returns calculation
- Position exposure
- Trading activity
- Asset allocation

## Recent Changes

- Added position history tracking
- Improved transaction cost handling
- Enhanced portfolio value calculation
- Added detailed trade logging

## Related Components

- Backtester: Single-asset version
- RiskAwareBacktester: Risk-managed version
- ExperimentalBacktester: Advanced features

## Notes

- The engine supports both spot and margin trading
- Position limits are enforced at the asset level
- Transaction costs are considered in all calculations
- Dust positions are automatically cleaned up 