Live Trading Environment
=====================

Overview
--------

The ``LiveTradingEnvironment`` class provides a real-time trading environment that connects to actual exchanges (or simulated ones) using the OpenAI Gym interface with async methods.

Class Structure
-------------

.. code-block:: python

    class LiveTradingEnvironment(gym.Env):
        def __init__(self, symbol, initial_balance, trading_fee,
                    websocket=None, exchange=None, test_mode=False):
            self.symbol = symbol
            self.initial_balance = initial_balance
            self.trading_fee = trading_fee
            self.websocket = websocket or WebSocketLoader()
            self.exchange = exchange or ccxt.async_support.exchange()
            self.test_mode = test_mode

Key Components
------------

Order Status and Management
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class OrderStatus:
        """Order status constants."""
        PENDING = 'pending'
        OPEN = 'open'
        CLOSED = 'closed'
        CANCELED = 'canceled'
        REJECTED = 'rejected'

    class Order:
        """Order representation."""
        def __init__(self, order_id, symbol, order_type, 
                    size, price, timestamp):
            self.order_id = order_id
            self.symbol = symbol
            self.type = order_type
            self.size = size
            self.price = price
            self.timestamp = timestamp
            self.status = OrderStatus.PENDING
            self.filled_size = 0.0

Key Methods
^^^^^^^^^

create_order
""""""""""
.. code-block:: python

    async def create_order(self, side, amount, price):
        """
        Create an order in the exchange or test environment.
        
        Args:
            side: 'buy' or 'sell'
            amount: Order size
            price: Order price
            
        Returns:
            Order object with status updates
        """

Features:
* Creates real or simulated orders
* Handles test mode differently
* Manages order tracking
* Updates account state

cancel_order
"""""""""""
.. code-block:: python

    async def cancel_order(self, order_id):
        """
        Cancel an existing order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if canceled, False otherwise
        """

Features:
* Cancels orders in exchange or test mode
* Updates order status
* Manages order lists
* Handles exceptions

monitor_order
""""""""""""
.. code-block:: python

    async def monitor_order(self, order_id, timeout=30.0):
        """
        Monitor order status until completion or timeout.
        
        Args:
            order_id: Order to monitor
            timeout: Maximum wait time
            
        Yields:
            Order status updates
        """

Features:
* Async generator pattern
* Periodic status checks
* Timeout handling
* Order completion detection

Implementation Details
-------------------

Test Mode
^^^^^^^^
* Uses mock data for price simulation
* Local order execution logic
* No network calls required
* Useful for strategy testing

Order Management
^^^^^^^^^^^^^
* Tracks orders in different states:
    * active_orders: Currently open
    * filled_orders: Successfully executed
    * canceled_orders: Manually canceled

Async Structure
^^^^^^^^^^^^
* Uses async/await for I/O operations
* Efficient handling of network calls
* Integration with WebSocket feeds
* Non-blocking order monitoring

Dependencies
----------

* ``gymnasium``: OpenAI Gym interface
* ``ccxt.async_support``: Exchange API
* ``WebSocketLoader``: Real-time data
* ``numpy``: Numerical operations
* ``pandas``: Data handling
* ``asyncio``: Async support

Usage Example
-----------

Basic Usage
^^^^^^^^^

.. code-block:: python

    # Initialize environment
    env = LiveTradingEnvironment(
        symbol="BTC/USD",
        initial_balance=10000.0,
        trading_fee=0.001,
        test_mode=True
    )
    
    # Reset environment
    obs, info = await env.reset()
    
    # Execute trades
    action = 0.5  # Buy with 50% of max size
    obs, reward, done, truncated, info = await env.step(action)

Advanced Usage
^^^^^^^^^^^

.. code-block:: python

    # Create specific orders
    order = await env.create_order(
        side='buy',
        amount=1.0,
        price=50000.0
    )
    
    # Monitor order execution
    async for status in env.monitor_order(order.order_id):
        print(f"Order status: {status}")
        if status == OrderStatus.CLOSED:
            break

Best Practices
-----------

1. Error Handling
^^^^^^^^^^^^^
* Handle network errors
* Manage rate limits
* Implement retry logic
* Log all exceptions

2. Order Management
^^^^^^^^^^^^^^^
* Monitor order timeouts
* Validate order parameters
* Track fill rates
* Handle partial fills

3. State Management
^^^^^^^^^^^^^^^
* Maintain accurate balances
* Track position sizes
* Monitor portfolio value
* Handle edge cases

4. Testing
^^^^^^^
* Use test_mode for development
* Validate order logic
* Check error handling
* Monitor performance

Recent Changes
------------

* Added async/await support
* Improved order monitoring
* Enhanced test mode features
* Added WebSocket integration 