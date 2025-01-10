Paper Trading Environment
======================

Overview
--------

The ``PaperTradingEnvironment`` class provides a simulated trading environment with support for various order types and real-time market data integration.

Class Structure
-------------

Order Types and Status
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class OrderType(Enum):
        """Order type definitions."""
        MARKET = 'market'
        LIMIT = 'limit'
        STOP_LIMIT = 'stop_limit'
        TRAILING_STOP = 'trailing_stop'
        ICEBERG = 'iceberg'
        OCO = 'oco'

    class OrderStatus(Enum):
        """Order status definitions."""
        PENDING = 'pending'
        FILLED = 'filled'
        CANCELLED = 'cancelled'
        EXPIRED = 'expired'

Order Class
^^^^^^^^^

.. code-block:: python

    class Order:
        """Order representation with advanced features."""
        def __init__(self, order_id, type, side, size, price,
                    trailing_price=None, visible_qty=None):
            self.order_id = order_id
            self.type = type
            self.side = side
            self.size = size
            self.price = price
            self.status = OrderStatus.PENDING
            self.filled_size = 0.0
            self.filled_price = 0.0
            self.trailing_price = trailing_price
            self.visible_qty = visible_qty
            self.executed_qty = 0.0
            self.timestamp = datetime.now()

Key Components
------------

Environment Initialization
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class PaperTradingEnvironment:
        def __init__(self, symbol, initial_balance, trading_fee,
                    window_size=20, test_mode=False):
            self.symbol = symbol
            self.initial_balance = initial_balance
            self.trading_fee = trading_fee
            self.window_size = window_size
            self.test_mode = test_mode
            self.websocket = WebSocketLoader() if not test_mode else None
            self._initialize_state()

Features:
* Real-time or simulated market data
* Multiple order type support
* Position and balance tracking
* Price history management

Key Methods
^^^^^^^^^

Market Data Management
""""""""""""""""""

.. code-block:: python

    async def update_market_data(self, data: Dict):
        """
        Update market data and process orders.
        
        Args:
            data: Market data dictionary with price information
        """

Features:
* Updates latest price
* Maintains price history
* Checks stop-loss conditions
* Processes pending orders

Order Processing
"""""""""""""

.. code-block:: python

    async def place_order(self, order: Order):
        """
        Place a new order.
        
        Args:
            order: Order object with type, size, price, etc.
            
        Returns:
            Placed order with updated status
        """

Features:
* Handles different order types
* Immediate execution for market orders
* Queues other orders for later processing
* Updates account state

Order Execution
""""""""""""

.. code-block:: python

    async def _execute_order_fill(self, order):
        """
        Execute order fill with slippage simulation.
        
        Args:
            order: Order to execute
        """

Features:
* Simulates realistic fills
* Applies transaction fees
* Updates position and balance
* Records trade history

Implementation Details
-------------------

Order Type Processing
^^^^^^^^^^^^^^^^^

Market Orders
"""""""""""
* Immediate execution at current price
* Slippage simulation in non-test mode
* Fee calculation and balance update

Limit Orders
""""""""""
* Price condition checking
* Partial fills for iceberg orders
* Order expiration handling

Stop and Trailing Orders
"""""""""""""""""""""
* Stop price triggering
* Trailing price updates
* Conversion to market orders

Dependencies
----------

* ``asyncio``: Asynchronous operations
* ``WebSocketLoader``: Real-time data feed
* ``numpy``: Slippage simulation
* ``pandas``: Price history and metrics

Usage Example
-----------

Basic Usage
^^^^^^^^^

.. code-block:: python

    # Initialize environment
    env = PaperTradingEnvironment(
        symbol="BTC/USD",
        initial_balance=10000.0,
        trading_fee=0.001,
        test_mode=True
    )
    
    # Place market order
    order = Order(
        order_id="1",
        type=OrderType.MARKET,
        side="buy",
        size=1.0,
        price=None
    )
    await env.place_order(order)

Advanced Orders
^^^^^^^^^^^^

.. code-block:: python

    # Place trailing stop order
    order = Order(
        order_id="2",
        type=OrderType.TRAILING_STOP,
        side="sell",
        size=1.0,
        price=50000.0,
        trailing_price=100.0
    )
    await env.place_order(order)
    
    # Place iceberg order
    order = Order(
        order_id="3",
        type=OrderType.ICEBERG,
        side="buy",
        size=10.0,
        price=49000.0,
        visible_qty=1.0
    )
    await env.place_order(order)

Best Practices
-----------

1. Order Management
^^^^^^^^^^^^^^^
* Validate orders before placement
* Monitor fill rates
* Handle partial executions
* Track order history

2. Risk Management
^^^^^^^^^^^^^
* Set position limits
* Implement stop-loss
* Monitor drawdown
* Track exposure

3. Performance Monitoring
^^^^^^^^^^^^^^^^^^^
* Calculate metrics regularly
* Track execution quality
* Monitor slippage
* Record trade statistics

4. Error Handling
^^^^^^^^^^^^
* Handle network issues
* Manage timeouts
* Log exceptions
* Maintain state consistency

Recent Changes
------------

* Added advanced order types
* Improved slippage simulation
* Enhanced metric calculations
* Added real-time data support 