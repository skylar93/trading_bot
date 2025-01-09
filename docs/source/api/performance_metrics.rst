Performance Metrics API
=====================

Overview
--------

The performance metrics module provides comprehensive tools for calculating and analyzing trading performance across returns, trades, and positions.

Key Functions
------------

Returns Metrics
^^^^^^^^^^^^^^

.. code-block:: python

    def calculate_returns_metrics(returns):
        """Calculate metrics based on returns series.
        
        Args:
            returns: Series of period returns
            
        Returns:
            dict: Metrics including:
                - total_return
                - annual_return
                - volatility
                - sharpe_ratio
                - sortino_ratio
                - max_drawdown
        """
        metrics = {}
        
        # Total and annual returns
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annual_return'] = (
            (1 + metrics['total_return']) ** 
            (252 / len(returns)) - 1
        )
        
        # Risk metrics
        metrics['volatility'] = returns.std() * np.sqrt(252)
        metrics['sharpe_ratio'] = (
            metrics['annual_return'] / metrics['volatility']
        )
        
        # Drawdown
        metrics['max_drawdown'] = calculate_max_drawdown(returns)
        
        return metrics

Trade Metrics
^^^^^^^^^^^^

.. code-block:: python

    def calculate_trade_metrics(trades_df):
        """Calculate metrics based on individual trades.
        
        Args:
            trades_df: DataFrame of trades with columns:
                     - entry_price
                     - exit_price
                     - size
                     - pnl
                     
        Returns:
            dict: Metrics including:
                - win_rate
                - profit_factor
                - avg_win
                - avg_loss
        """
        metrics = {}
        
        # Win rate
        wins = trades_df['pnl'] > 0
        metrics['win_rate'] = wins.mean()
        
        # Profit factor
        gross_profit = trades_df.loc[wins, 'pnl'].sum()
        gross_loss = abs(trades_df.loc[~wins, 'pnl'].sum())
        metrics['profit_factor'] = gross_profit / gross_loss
        
        # Average trade metrics
        metrics['avg_win'] = trades_df.loc[wins, 'pnl'].mean()
        metrics['avg_loss'] = trades_df.loc[~wins, 'pnl'].mean()
        
        return metrics

Position Metrics
^^^^^^^^^^^^^^^

.. code-block:: python

    def calculate_position_metrics(positions_df):
        """Calculate metrics based on position history.
        
        Args:
            positions_df: DataFrame of positions with columns:
                        - symbol
                        - size
                        - value
                        
        Returns:
            dict: Metrics including:
                - avg_position_size
                - max_position_size
                - turnover
                - concentration
        """
        metrics = {}
        
        # Position sizing
        metrics['avg_position_size'] = positions_df['size'].mean()
        metrics['max_position_size'] = positions_df['size'].max()
        
        # Turnover
        metrics['turnover'] = (
            positions_df['size'].diff().abs().sum() / 
            len(positions_df)
        )
        
        # Concentration
        total_value = positions_df['value'].sum()
        metrics['concentration'] = (
            (positions_df['value'] / total_value) ** 2
        ).sum()
        
        return metrics

Implementation Details
--------------------

Returns Analysis
^^^^^^^^^^^^^^^

1. Return Calculations:
   - Period returns
   - Cumulative returns
   - Annualized metrics

2. Risk Metrics:
   - Volatility (standard deviation)
   - Downside deviation
   - Maximum drawdown

3. Risk-Adjusted Returns:
   - Sharpe ratio
   - Sortino ratio
   - Calmar ratio

Trade Analysis
^^^^^^^^^^^^^

1. Trade Statistics:
   - Win/loss ratio
   - Average trade P&L
   - Trade duration

2. Risk Management:
   - Maximum drawdown
   - Value at Risk (VaR)
   - Expected Shortfall

3. Transaction Analysis:
   - Trading costs
   - Slippage
   - Market impact

Dependencies
-----------

- NumPy (statistical calculations)
- Pandas (DataFrame operations)
- SciPy (optional, for advanced statistics)

Usage Example
------------

Basic Usage
^^^^^^^^^^

.. code-block:: python

    # Calculate returns metrics
    returns = calculate_returns_metrics(daily_returns)
    print(f"Sharpe Ratio: {returns['sharpe_ratio']:.2f}")
    
    # Calculate trade metrics
    trades = calculate_trade_metrics(trades_df)
    print(f"Win Rate: {trades['win_rate']:.2%}")
    
    # Calculate position metrics
    positions = calculate_position_metrics(positions_df)
    print(f"Turnover: {positions['turnover']:.2f}")

Best Practices
------------

1. Data Quality
^^^^^^^^^^^^^

- Clean and validate input data
- Handle missing values
- Check for outliers

2. Time Periods
^^^^^^^^^^^^^

- Use consistent periods
- Consider market hours
- Account for holidays

3. Risk Assessment
^^^^^^^^^^^^^^^

- Use multiple metrics
- Consider drawdowns
- Track correlations

4. Performance Attribution
^^^^^^^^^^^^^^^^^^^^^^^

- Analyze by strategy
- Track factor exposure
- Monitor style drift 