Trading Environment Base
===================

Overview
--------

The trading environment module provides OpenAI Gym-compatible classes for reinforcement learning experiments in trading.

Position Class
------------

.. code-block:: python

    @dataclass
    class Position:
        side: str  # 'long' or 'short'
        size: float
        entry_price: float
        entry_time: pd.Timestamp

Features:
* Stores individual position information
* Provides value property (positive for long, negative for short)
* Calculates unrealized PnL

Properties
^^^^^^^^^
* ``value``: Returns conceptual position size (positive for long, negative for short)
* ``calculate_pnl(current_price)``: Calculates unrealized PnL against current price

TradingEnvironment Class
---------------------

Overview
^^^^^^^
The ``TradingEnvironment`` class implements the OpenAI Gym interface for trading simulations.

.. code-block:: python

    class TradingEnvironment(gym.Env):
        def __init__(self, df, initial_balance, trading_fee, window_size=20,
                    max_position_size=1.0):
            super().__init__()
            self.df = df
            self.initial_balance = initial_balance
            self.trading_fee = trading_fee
            self.window_size = window_size
            self.max_position_size = max_position_size
            ...

Key Components
^^^^^^^^^^^

Constructor Parameters
""""""""""""""""""
* ``df``: DataFrame with OHLCV and feature data
* ``initial_balance``: Starting capital
* ``trading_fee``: Transaction fee rate
* ``window_size``: Number of time steps in observation window
* ``max_position_size``: Maximum allowed position size

Instance Attributes
"""""""""""""""
* ``observation_space``: Box space of shape (window_size, n_features)
* ``action_space``: Box space for continuous actions [-1, 1]
* ``current_step``: Current time step in episode
* ``balance``: Current account balance
* ``position``: Current Position object

Key Methods
^^^^^^^^^

reset
"""""
.. code-block:: python

    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state.
        
        Returns:
            tuple: (observation, info)
        """

Features:
* Initializes episode starting point
* Resets balance and position
* Returns initial observation and info

step
""""
.. code-block:: python

    def step(self, action):
        """
        Execute one time step in the environment.
        
        Args:
            action: numpy array with trading action [-1, 1]
            
        Returns:
            tuple: (observation, reward, done, truncated, info)
        """

Features:
* Processes trading action
* Updates position and balance
* Calculates reward
* Determines episode completion
* Returns next state information

_get_observation
"""""""""""""
.. code-block:: python

    def _get_observation(self):
        """
        Get current window of normalized feature data.
        
        Returns:
            numpy.ndarray: Normalized feature window
        """

Features:
* Slices window_size period of data
* Applies min-max scaling to [-1, 1]
* Returns numpy array of features

Implementation Details
-------------------

State Space
^^^^^^^^^
* Features extracted from DataFrame (excluding "datetime" and "instrument")
* Normalized to [-1, 1] range using min-max scaling
* Window of size window_size forms the observation

Action Space
^^^^^^^^^^
* Continuous action space [-1, 1]
* Negative values for short positions
* Positive values for long positions
* Absolute value represents position intensity

Position Management
^^^^^^^^^^^^^^^
* Single position tracking via Position class
* Long/short distinction
* Entry price and time tracking
* PnL calculation support

Reward Structure
^^^^^^^^^^^^^
* Based on portfolio value changes
* Considers transaction fees
* Can be customized for specific strategies

Dependencies
----------

* ``gymnasium``: OpenAI Gym interface
* ``numpy``: Numerical operations
* ``pandas``: Data handling
* ``dataclasses``: Position class structure

Usage Guide
---------

For Reinforcement Learning
^^^^^^^^^^^^^^^^^^^^^^^
* Environment follows Gym interface conventions
* Suitable for training RL agents with continuous action spaces
* Provides standard step-by-step interaction
* Customizable reward function

For Strategy Development
^^^^^^^^^^^^^^^^^^^^
* Can be used for strategy prototyping
* Supports feature engineering via DataFrame
* Provides realistic trading simulation
* Tracks performance metrics

Recent Changes
------------

* Implemented Gym v0.26+ interface
* Added position tracking improvements
* Enhanced reward calculation
* Updated observation normalization 