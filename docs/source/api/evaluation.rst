Trading Evaluation
=================

Overview
--------

The Trading Evaluation module provides comprehensive performance and risk metrics calculation for trading strategies, including Sharpe ratio, Sortino ratio, maximum drawdown, and trade statistics.

Key Components
-------------

TradingMetrics
^^^^^^^^^^^^

Static methods for calculating trading performance metrics:

Core Methods:
    * ``calculate_returns(portfolio_values)``: Calculate period returns from portfolio values
    * ``calculate_sharpe_ratio(returns, risk_free_rate=0.0)``: Calculate annualized Sharpe ratio
    * ``calculate_sortino_ratio(returns, risk_free_rate=0.0)``: Calculate Sortino ratio using downside risk
    * ``calculate_maximum_drawdown(portfolio_values)``: Calculate maximum drawdown and duration
    * ``calculate_win_rate(trades)``: Calculate percentage of profitable trades
    * ``calculate_profit_loss_ratio(trades)``: Calculate profit/loss ratio
    * ``calculate_trade_statistics(trades)``: Calculate comprehensive trade statistics
    * ``evaluate_strategy(portfolio_values, trades, risk_free_rate=0.0)``: Calculate all key metrics

Standalone Functions
^^^^^^^^^^^^^^^^^

Additional metric calculation functions:

* ``calculate_metrics(portfolio_values, returns)``: Basic portfolio metrics
* ``calculate_trade_metrics(trades)``: Trade-specific metrics
* ``calculate_risk_metrics(returns)``: Risk-focused metrics
* ``calculate_all_metrics(portfolio_values, returns, trades)``: Comprehensive metrics

Implementation Details
--------------------

Return Calculations
^^^^^^^^^^^^^^^

* Daily returns from portfolio values
* Annualization factor (√252)
* Risk-free rate adjustment
* Handling of edge cases (empty arrays, zero values)

Risk Metrics
^^^^^^^^^^

* Sharpe Ratio: (returns - risk_free_rate).mean() / std() * √252
* Sortino Ratio: Using downside deviation
* Maximum Drawdown: Peak to trough calculation
* VaR/CVaR (95% confidence)
* Annualized volatility

Trade Statistics
^^^^^^^^^^^^^

* Win rate calculation
* Profit/Loss ratio
* Average trade metrics
* Trade duration analysis
* Profit factor computation

Dependencies
-----------

* ``numpy``: Numerical computations
* ``pandas``: Time series operations
* ``logging``: Error tracking
* ``typing``: Type hints

Usage Example
------------

.. code-block:: python

    # Calculate basic metrics
    portfolio_values = np.array([10000, 10100, 10200, 10150, 10300])
    returns = TradingMetrics.calculate_returns(portfolio_values)
    
    # Calculate Sharpe ratio
    sharpe = TradingMetrics.calculate_sharpe_ratio(returns, risk_free_rate=0.02)
    
    # Calculate trade statistics
    trades = [
        {"pnl": 100, "duration": 2},
        {"pnl": -50, "duration": 1},
        {"pnl": 150, "duration": 3}
    ]
    stats = TradingMetrics.calculate_trade_statistics(trades)
    
    # Evaluate complete strategy
    metrics = TradingMetrics.evaluate_strategy(
        portfolio_values=portfolio_values,
        trades=trades,
        risk_free_rate=0.02
    )
    
    # Calculate risk metrics
    risk_metrics = calculate_risk_metrics(returns)
    print(f"VaR (95%): {risk_metrics['var_95']}")
    print(f"CVaR (95%): {risk_metrics['cvar_95']}")

Best Practices
-------------

1. Return Calculations
   * Handle missing data appropriately
   * Apply proper annualization factors
   * Consider risk-free rate adjustments

2. Risk Assessment
   * Use appropriate confidence levels
   * Consider multiple risk metrics
   * Handle extreme values

3. Trade Analysis
   * Account for transaction costs
   * Consider trade duration
   * Handle zero P/L trades

4. Performance Monitoring
   * Track metrics over time
   * Compare against benchmarks
   * Monitor risk limits

Recent Changes
-------------

* Added VaR/CVaR calculations
* Enhanced trade statistics
* Improved edge case handling
* Added comprehensive metrics function 