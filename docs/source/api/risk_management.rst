Risk Management
===============

Overview
--------

The Risk Management module provides comprehensive risk control functionality for trading operations, including position sizing, stop-loss management, and trade limits.

Key Components
-------------

RiskConfig
^^^^^^^^^

Configuration dataclass for risk management parameters:

* ``max_position_size``: Maximum position size as fraction of portfolio (default: 0.2)
* ``stop_loss_pct``: Stop loss percentage (default: 0.02)
* ``max_drawdown_pct``: Maximum allowable drawdown (default: 0.15)
* ``daily_trade_limit``: Maximum trades per day (default: 10)
* ``min_trade_size``: Minimum trade size as fraction of portfolio (default: 0.01)
* ``max_leverage``: Maximum leverage (default: 1.0 = no leverage)
* ``position_scaling``: Whether to scale position size based on volatility (default: True)

RiskManager
^^^^^^^^^^

Main risk management class with comprehensive risk control features:

Core Methods:
    * ``__init__(config: RiskConfig)``: Initialize risk manager
    * ``reset()``: Reset risk manager state
    * ``calculate_position_size(portfolio_value, price, volatility)``: Calculate allowed position size
    * ``check_trade_limits(timestamp)``: Check daily trade limits
    * ``set_stop_loss(trade_id, entry_price, position_type)``: Set stop loss price
    * ``check_stop_loss(trade_id, current_price)``: Check if stop loss hit
    * ``update_drawdown(portfolio_value)``: Update and check drawdown
    * ``adjust_for_leverage(position_size, current_leverage)``: Adjust for leverage limits
    * ``process_trade_signal(...)``: Process trade through all risk checks
    * ``update_after_trade(...)``: Update state after trade execution

Implementation Details
--------------------

Position Sizing
^^^^^^^^^^^^^

* Base size calculated as fraction of portfolio
* Optional volatility-based scaling
* Minimum trade size enforcement
* Leverage adjustment

Risk Controls
^^^^^^^^^^^

* Daily trade limit tracking
* Stop loss management
* Drawdown monitoring
* Leverage limits

Dependencies
-----------

* ``numpy``: Numerical operations
* ``pandas``: Timestamp handling
* ``dataclasses``: Configuration class
* ``logging``: Error tracking

Usage Example
------------

.. code-block:: python

    # Initialize risk manager
    config = RiskConfig(
        max_position_size=0.2,
        stop_loss_pct=0.02,
        daily_trade_limit=10
    )
    risk_manager = RiskManager(config)

    # Process trade signal
    result = risk_manager.process_trade_signal(
        timestamp=pd.Timestamp.now(),
        portfolio_value=100000.0,
        price=50.0,
        volatility=0.2,
        current_leverage=0.5
    )

    if result["allowed"]:
        # Execute trade
        trade_id = "trade_1"
        risk_manager.update_after_trade(
            trade_id,
            pd.Timestamp.now(),
            entry_price=50.0,
            position_type="long"
        )

Best Practices
-------------

1. Position Management
   * Configure appropriate position size limits
   * Consider volatility in sizing
   * Monitor leverage carefully

2. Risk Limits
   * Set conservative stop loss levels
   * Monitor drawdown closely
   * Respect daily trade limits

3. Trade Execution
   * Verify all risk checks before trading
   * Update risk state after trades
   * Track stop losses carefully

4. State Management
   * Reset state appropriately
   * Track daily limits properly
   * Monitor drawdown peaks

Recent Changes
-------------

* Added volatility-based position scaling
* Enhanced drawdown monitoring
* Improved leverage management
* Added comprehensive trade signal processing 