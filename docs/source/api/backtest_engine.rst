Backtest Engine API
=================

Overview
--------

The ``BacktestEngine`` class provides a robust engine for backtesting trading strategies across multiple assets.

Class Structure
--------------

.. code-block:: python

    class BacktestEngine:
        def __init__(self, initial_capital: float = 100000.0, transaction_cost: float = 0.001, max_position: float = 1.0):
            """Initialize the backtest engine.

            Args:
                initial_capital (float): Starting capital
                transaction_cost (float): Transaction cost as decimal
                max_position (float): Maximum position size as fraction of capital
            """

Key Methods
----------

- ``reset()``: Reset the engine state
- ``update(timestamp, prices, actions)``: Process updates for multiple assets
- ``get_portfolio_value(prices)``: Calculate total portfolio value
- ``get_returns()``: Get historical returns
- ``get_trade_history()``: Get detailed trade history
- ``get_position_history()``: Get position size history

Implementation Details
--------------------

Portfolio Management
^^^^^^^^^^^^^^^^^

The engine manages a portfolio by:

- Tracking positions across multiple assets
- Calculating portfolio-level metrics
- Managing risk exposure

Trade Processing
^^^^^^^^^^^^^

Trade execution includes:

1. Position sizing calculations
2. Transaction cost application
3. Portfolio rebalancing
4. Performance tracking

Dependencies
-----------

- pandas
- numpy
- logging

Usage Example
-----------

.. code-block:: python

    engine = BacktestEngine(initial_capital=100000.0, transaction_cost=0.001)
    engine.update(timestamp, prices, actions)
    portfolio_value = engine.get_portfolio_value(current_prices)

Recent Changes
------------

- Added portfolio rebalancing functionality
- Improved risk management features
- Enhanced performance tracking 