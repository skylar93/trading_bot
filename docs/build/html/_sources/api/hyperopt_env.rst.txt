Hyperopt Environment API
=====================

Overview
--------

The ``SimplifiedTradingEnv`` provides a streamlined trading environment specifically designed for hyperparameter optimization experiments, using OHLCV data and simplified trading mechanics.

Class Structure
-------------

.. code-block:: python

    class SimplifiedTradingEnv(gym.Env):
        def __init__(self, df, window_size=10):
            """Initialize simplified trading environment.
            
            Args:
                df: DataFrame with OHLCV data
                window_size: Observation window size
            """
            self.df = df
            self.window_size = window_size
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(window_size, 5)
            )

Key Methods
---------

Step Function
^^^^^^^^^^

.. code-block:: python

    def step(self, action):
        """Execute one step in the environment.
        
        Args:
            action: Float in [-1, 1] range
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Trading logic
        if action > 0:  # Buy
            self._buy(abs(action))
        elif action < 0:  # Sell
            self._sell(abs(action))
            
        # Update portfolio and calculate reward
        self._update_portfolio()
        reward = self._calculate_reward()
        
        return self._get_observation(), reward, done, info

Observation Processing
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def _get_observation(self):
        """Get normalized OHLCV window.
        
        Returns:
            ndarray: Normalized price and volume data
        """
        # Get window slice
        window = self.df[self.current_step-self.window_size:
                        self.current_step]
        
        # Normalize separately for price and volume
        price_cols = ['open', 'high', 'low', 'close']
        volume_col = ['volume']
        
        prices = window[price_cols].values
        volumes = window[volume_col].values
        
        # Standardize
        prices = (prices - prices.mean()) / prices.std()
        volumes = (volumes - volumes.mean()) / volumes.std()
        
        return np.column_stack([prices, volumes])

Reward Calculation
^^^^^^^^^^^^^^^

.. code-block:: python

    def _calculate_sharpe_ratio(self, returns):
        """Calculate Sharpe ratio for returns series.
        
        Args:
            returns: Array of returns
            
        Returns:
            float: Annualized Sharpe ratio
        """
        if len(returns) < 2:
            return 0
            
        std = np.std(returns, ddof=1)
        if std == 0:
            return 0
            
        sharpe = np.sqrt(252) * (np.mean(returns) / std)
        return sharpe

Implementation Details
-------------------

Trading Mechanics
^^^^^^^^^^^^^

1. Action Processing:
   - Positive action: Buy proportional amount
   - Negative action: Sell proportional amount
   - Zero action: Hold position

2. Portfolio Tracking:
   - Track cash balance
   - Track position size
   - Calculate total value

3. Transaction Costs:
   - Apply trading fees
   - Update available cash

Observation Space
^^^^^^^^^^^^^

1. Window Structure:
   - Shape: (window_size, 5)
   - Features: OHLCV data
   - Normalized values

2. Normalization:
   - Price features: Shared statistics
   - Volume: Separate statistics
   - Rolling window basis

Dependencies
----------

- Gymnasium (gym.Env)
- Pandas (DataFrame operations)
- Numpy (array operations)

Usage Example
-----------

Basic Usage
^^^^^^^^^

.. code-block:: python

    # Create environment
    df = pd.read_csv('price_data.csv')
    env = SimplifiedTradingEnv(df, window_size=10)
    
    # Reset environment
    obs = env.reset()
    
    # Take action
    action = 0.5  # Buy 50% of available cash
    obs, reward, done, info = env.step(action)

Best Practices
-----------

1. Data Preparation
^^^^^^^^^^^^^^^

- Ensure clean OHLCV data
- Handle missing values
- Consider data frequency

2. Window Size
^^^^^^^^^^

- Match to strategy horizon
- Consider memory constraints
- Balance information content

3. Action Space
^^^^^^^^^^^

- Understand position sizing
- Consider transaction costs
- Monitor portfolio exposure

4. Reward Design
^^^^^^^^^^^

- Use appropriate time scale
- Balance risk/return
- Consider trading frequency 