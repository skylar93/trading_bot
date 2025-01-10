Risk Manager API
==============

Overview
--------

The ``RiskManager`` class provides comprehensive risk management functionality for trading systems.

Class Structure
--------------

.. code-block:: python

    class RiskManager:
        def __init__(self, max_position_size: float = 1.0, stop_loss_pct: float = 0.02, 
                     max_drawdown: float = 0.2, correlation_threshold: float = 0.7):
            """Initialize the risk manager.

            Args:
                max_position_size (float): Maximum position size as fraction of capital
                stop_loss_pct (float): Stop loss percentage
                max_drawdown (float): Maximum allowed drawdown
                correlation_threshold (float): Correlation threshold for risk signals
            """

Key Methods
----------

- ``process_trade_signal(signal, current_price, position)``: Process and validate trade signals
- ``update_correlation_matrix()``: Update asset correlation matrix
- ``check_risk_limits(portfolio_value, drawdown)``: Check if risk limits are breached
- ``calculate_position_size(signal_strength, volatility)``: Calculate risk-adjusted position size

Implementation Details
--------------------

Risk Assessment
^^^^^^^^^^^^^

The risk manager evaluates:

- Position size limits
- Stop loss levels
- Portfolio drawdown
- Asset correlations

Signal Processing
^^^^^^^^^^^^^^

Trade signals are processed by:

1. Validating against risk limits
2. Adjusting for position sizing
3. Checking correlation constraints
4. Applying stop loss rules

Dependencies
-----------

- pandas
- numpy
- scipy

Usage Example
-----------

.. code-block:: python

    risk_manager = RiskManager(max_position_size=1.0, stop_loss_pct=0.02)
    adjusted_signal = risk_manager.process_trade_signal(signal, price, position)

Recent Changes
------------

- Added correlation-based risk signals
- Improved drawdown monitoring
- Enhanced position sizing logic 