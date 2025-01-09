Trading Environment
=================

Overview
--------

The ``TradingEnvironment`` class implements a single-asset trading environment following the OpenAI Gym interface for reinforcement learning experiments.

Class Structure
-------------

.. code-block:: python

    class TradingEnvironment(gym.Env):
        def __init__(self, df, initial_balance=10000.0,
                    trading_fee=0.001, window_size=20,
                    max_position_size=1.0):
            self.df = self._validate_columns(df)
            self.initial_balance = initial_balance
            self.trading_fee = trading_fee
            self.window_size = window_size
            self.max_position_size = max_position_size
            self._setup_spaces()

Key Components
------------

Observation Space
^^^^^^^^^^^^^
* Shape: (window_size, 5) for OHLCV data
* Normalized features
* Price ratios relative to first value
* Volume normalized by mean

Action Space
^^^^^^^^^^
* Continuous: Box(-1, 1)
* Negative values for short positions
* Positive values for long positions
* Absolute value determines position size

Key Methods
^^^^^^^^^

reset
"""""
.. code-block:: python

    def reset(self):
        """
        Reset environment to initial state.
        
        Returns:
            tuple: (observation, info)
        """

Features:
* Resets current step
* Initializes balance
* Clears position
* Returns initial observation

step
""""
.. code-block:: python

    def step(self, action):
        """
        Execute one environment step.
        
        Args:
            action: numpy array with values in [-1, 1]
            
        Returns:
            tuple: (observation, reward, done, truncated, info)
        """

Features:
* Processes trading action
* Updates position and balance
* Calculates reward
* Returns next state

Implementation Details
-------------------

Position Sizing
^^^^^^^^^^^^

.. code-block:: python

    def _calculate_target_position(self, action):
        """Calculate target position from action."""
        return float(action[0]) * self.max_position_size

Features:
* Scales action to position size
* Respects maximum position limit
* Handles both long and short positions

Portfolio Value
^^^^^^^^^^^^

.. code-block:: python

    def _calculate_portfolio_value(self, step):
        """Calculate total portfolio value."""
        price = self.df.iloc[step]["$close"]
        return self.balance + self.position * price

Features:
* Combines cash and position value
* Uses current market price
* Basis for reward calculation

Observation Processing
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def _get_observation(self):
        """Get normalized price window."""
        start = self.current_step - self.window_size
        end = self.current_step
        
        window = self.df.iloc[start:end]
        prices = window[["$open", "$high", "$low", "$close"]]
        volume = window["$volume"]
        
        # Normalize prices relative to first value
        price_matrix = prices.values / prices.iloc[0].values
        
        # Normalize volume by mean
        volume_matrix = volume.values / volume.mean()
        
        return np.column_stack([price_matrix, volume_matrix])

Features:
* Extracts price window
* Normalizes features
* Handles missing data
* Returns numpy array

Dependencies
----------

* ``gymnasium``: OpenAI Gym interface
* ``numpy``: Numerical operations
* ``pandas``: Data handling
* ``logging``: Error tracking

Usage Example
-----------

Basic Setup
^^^^^^^^^

.. code-block:: python

    # Prepare data
    df = pd.DataFrame({
        "$open": [...],
        "$high": [...],
        "$low": [...],
        "$close": [...],
        "$volume": [...]
    })
    
    # Create environment
    env = TradingEnvironment(
        df=df,
        initial_balance=10000.0,
        trading_fee=0.001,
        window_size=20
    )

Training Loop
^^^^^^^^^^

.. code-block:: python

    # Reset environment
    obs, info = env.reset()
    
    for _ in range(1000):
        # Get action from agent
        action = agent.get_action(obs)
        
        # Execute step
        obs, reward, done, truncated, info = env.step(action)
        
        if done:
            break

Best Practices
-----------

1. Data Preparation
^^^^^^^^^^^^^^^
* Ensure OHLCV columns have '$' prefix
* Handle missing values
* Check data quality
* Validate timestamps

2. Feature Engineering
^^^^^^^^^^^^^^^^^
* Normalize observations
* Handle outliers
* Consider additional features
* Validate calculations

3. Position Management
^^^^^^^^^^^^^^^^^
* Set appropriate position limits
* Consider transaction costs
* Monitor exposure
* Track performance

4. Reward Design
^^^^^^^^^^^^^^
* Use portfolio value changes
* Consider risk-adjusted returns
* Handle edge cases
* Validate calculations

Recent Changes
-------------

* Added observation normalization
* Improved reward calculation
* Enhanced position management
* Added performance tracking 