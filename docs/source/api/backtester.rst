Backtester API
=============

Overview
--------

The ``Backtester`` class provides functionality for backtesting trading strategies on historical data.

Class Structure
--------------

.. code-block:: python

    class Backtester:
        def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0, trading_fee: float = 0.001):
            """Initialize the backtester.

            Args:
                data (pd.DataFrame): Historical price data with OHLCV columns
                initial_balance (float): Starting account balance
                trading_fee (float): Transaction fee as a decimal (e.g., 0.001 for 0.1%)
            """

Key Methods
----------

- ``reset()``: Reset the backtester state
- ``run(strategy, window_size, verbose=False)``: Execute backtest with given strategy
- ``execute_trade(timestamp, action, price_data)``: Process a single trade
- ``_calculate_portfolio_value(current_price)``: Calculate current portfolio value
- ``_calculate_metrics()``: Calculate performance metrics

Implementation Details
--------------------

Position Management
^^^^^^^^^^^^^^^^^

The backtester tracks positions using:

- Current position size
- Entry price
- Unrealized PnL

Trade Processing
^^^^^^^^^^^^^

Each trade is processed by:

1. Validating the action
2. Calculating transaction costs
3. Updating position and balance
4. Recording trade details

Dependencies
-----------

- pandas
- numpy
- logging

Usage Example
-----------

.. code-block:: python

    backtester = Backtester(data, initial_balance=10000.0, trading_fee=0.001)
    results = backtester.run(strategy, window_size=20)

Recent Changes
------------

- Added support for multiple position entry points
- Improved transaction cost modeling
- Enhanced logging capabilities 