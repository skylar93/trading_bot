ExperimentalBacktester
=====================

Overview
--------

The ``ExperimentalBacktester`` class extends the ``BacktestEngine`` class to provide advanced position management features such as partial liquidation and average entry price tracking.

Class Structure
-------------

.. code-block:: python

    class ExperimentalBacktester(BacktestEngine):
        def __init__(self, data, risk_config, initial_balance):
            super().__init__(
                initial_capital=initial_balance,
                transaction_cost=0.001,
                max_position=risk_config.max_position_size
            )
            self.data = data
            self.entry_prices = {}
            self.trade_sizes = {}

Key Components
------------

Constructor Parameters
^^^^^^^^^^^^^^^^^^
* ``data``: Time series data containing OHLCV information
* ``risk_config``: Risk configuration object containing max_position_size and min_trade_size
* ``initial_balance``: Starting capital for backtesting

Instance Attributes
^^^^^^^^^^^^^^^
* ``self.data``: Time series data for backtesting (OHLCV)
* ``self.positions``: Dictionary of {symbol: float} tracking position sizes
* ``self.entry_prices``: Tracks average entry prices per asset
* ``self.trade_sizes``: Records trade sizes at each price for weighted average calculations
* ``self.trades``: List of executed trade records

Key Methods
^^^^^^^^^

execute_trade
~~~~~~~~~~~
.. code-block:: python

    def execute_trade(self, timestamp, action, price_data):
        """
        Execute trading logic including partial liquidation and position building.
        
        Args:
            timestamp: Current timestamp
            action: Trading action (-1 to 1)
            price_data: Current price information
            
        Returns:
            Trade execution result including PnL and fees
        """

Features:
* Handles partial position liquidation when action < 0
* Updates average entry price for new positions
* Tracks trade sizes for weighted average calculations
* Updates cash balance and trade history

_calculate_trade_revenue
~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    def _calculate_trade_revenue(self, asset, trade_size, current_price):
        """
        Calculate trade revenue, fees, and realized PnL.
        
        Args:
            asset: Asset symbol
            trade_size: Size of the trade
            current_price: Current market price
            
        Returns:
            Tuple of (revenue, fees, PnL)
        """

Features:
* Calculates actual revenue for position liquidation
* Computes transaction fees
* Determines realized PnL based on entry price

Implementation Details
-------------------

Partial Liquidation Logic
^^^^^^^^^^^^^^^^^^^^^^
* When action < 0 and current_position > 0:
    * close_size = current_position * (2 * abs(action))
    * Full liquidation if abs(action) >= 0.5

Average Entry Price Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Records trade sizes in trade_sizes dictionary (price -> size)
* Updates entry price using weighted average: weighted_sum / total_size

Risk Management
^^^^^^^^^^^^
* Enforces minimum trade size (risk_config.min_trade_size)
* Maintains maximum position size (risk_config.max_position_size)
* Handles transaction costs in PnL calculations

Dependencies
----------

* ``backtest_engine.py``: Parent class BacktestEngine
* ``pandas``: Data manipulation
* ``numpy``: Numerical operations
* ``risk.risk_manager.RiskConfig``: Risk parameters

Recent Changes
------------

* Added partial liquidation support
* Implemented weighted average entry price tracking
* Enhanced position size management
* Improved trade history logging 