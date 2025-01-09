Multi-Agent Trading Environment
=========================

Overview
--------

The ``MultiAgentTradingEnv`` class implements a trading environment supporting multiple agents with different strategies and risk profiles.

Class Structure
-------------

.. code-block:: python

    class MultiAgentTradingEnv(gym.Env):
        def __init__(self, data, agent_configs, window_size=20,
                    trading_fee=0.001):
            self.data = data
            self.agent_configs = agent_configs
            self.window_size = window_size
            self.trading_fee = trading_fee
            self.agents = [config["id"] for config in agent_configs]
            self._setup_spaces()

Key Components
------------

Agent Configuration
^^^^^^^^^^^^^^^
.. code-block:: python

    agent_config = {
        "id": "agent_1",
        "strategy": "momentum",  # or "mean_reversion", "market_making"
        "initial_balance": 10000.0,
        "risk_config": {
            "max_position": 1.0,
            "min_trade_size": 0.1
        }
    }

Features:
* Individual strategy assignment
* Custom risk parameters
* Separate balance tracking
* Strategy-specific features

Observation/Action Spaces
^^^^^^^^^^^^^^^^^^^^^
* Observation space: Box(window_size, n_features)
* Action space: Box(-1, 1)
* Spaces defined per agent
* Features customized by strategy

Key Methods
^^^^^^^^^

reset
"""""
.. code-block:: python

    def reset(self):
        """
        Reset environment for all agents.
        
        Returns:
            tuple: (observations dict, info dict)
        """

Features:
* Resets simulation step
* Initializes agent states
* Returns initial observations
* Provides agent info

step
""""
.. code-block:: python

    def step(self, actions):
        """
        Execute one step for all agents.
        
        Args:
            actions: Dict[agent_id, np.ndarray]
            
        Returns:
            tuple: (observations, rewards, dones, truncated, infos)
        """

Features:
* Processes multiple agent actions
* Updates positions and balances
* Calculates individual rewards
* Returns agent-specific results

Implementation Details
-------------------

Strategy Features
^^^^^^^^^^^^^

Momentum Strategy
"""""""""""""""
.. code-block:: python

    def _calculate_momentum_features(self, agent_id):
        """Calculate momentum indicators."""
        lookback = self.agent_configs[agent_id]["lookback"]
        returns = self.data["$close"].pct_change(lookback)
        volatility = returns.rolling(lookback).std()
        return np.column_stack([returns, volatility])

Mean Reversion Strategy
""""""""""""""""""""
.. code-block:: python

    def _calculate_mean_reversion_features(self, agent_id):
        """Calculate mean reversion indicators."""
        window = self.agent_configs[agent_id]["window"]
        rolling_mean = self.data["$close"].rolling(window).mean()
        deviation = (self.data["$close"] - rolling_mean) / rolling_mean
        return np.column_stack([rolling_mean, deviation])

Market Making Strategy
""""""""""""""""""
.. code-block:: python

    def _calculate_market_making_features(self, agent_id):
        """Calculate market making indicators."""
        spread = self.data["$high"] - self.data["$low"]
        volume = self.data["$volume"]
        return np.column_stack([spread, volume])

Reward Calculation
^^^^^^^^^^^^^^^

.. code-block:: python

    def _calculate_reward(self, agent_id, portfolio_value):
        """
        Calculate strategy-specific rewards.
        
        Args:
            agent_id: Agent identifier
            portfolio_value: Current portfolio value
            
        Returns:
            float: Reward value
        """

Features:
* Strategy-based reward computation
* Risk-adjusted returns
* Transaction cost consideration
* Performance penalties

Dependencies
----------

* ``gymnasium``: OpenAI Gym interface
* ``numpy``: Numerical operations
* ``pandas``: Data handling
* ``torch``: Optional GPU support

Usage Example
-----------

Basic Setup
^^^^^^^^^

.. code-block:: python

    # Prepare agent configurations
    agent_configs = [
        {
            "id": "momentum_trader",
            "strategy": "momentum",
            "initial_balance": 10000.0,
            "lookback": 20
        },
        {
            "id": "mean_rev_trader",
            "strategy": "mean_reversion",
            "initial_balance": 10000.0,
            "window": 50
        }
    ]
    
    # Create environment
    env = MultiAgentTradingEnv(
        data=price_data,
        agent_configs=agent_configs,
        window_size=50,
        trading_fee=0.001
    )

Running Simulation
^^^^^^^^^^^^^^^

.. code-block:: python

    # Reset environment
    observations = env.reset()
    
    # Simulation loop
    for _ in range(1000):
        # Get actions from agents
        actions = {
            agent_id: agent.get_action(observations[agent_id])
            for agent_id in env.agents
        }
        
        # Execute step
        observations, rewards, dones, truncated, infos = env.step(actions)
        
        # Check if episode is done
        if all(dones.values()):
            break

Best Practices
-----------

1. Agent Configuration
^^^^^^^^^^^^^^^^^
* Define clear strategy parameters
* Set appropriate risk limits
* Consider agent interactions
* Document feature requirements

2. Feature Engineering
^^^^^^^^^^^^^^^^^
* Customize features per strategy
* Ensure proper normalization
* Handle missing data
* Validate calculations

3. Reward Design
^^^^^^^^^^^^
* Align with strategy objectives
* Consider risk-adjusted returns
* Account for transaction costs
* Implement penalties

4. Performance Monitoring
^^^^^^^^^^^^^^^^^^^
* Track individual agent performance
* Monitor portfolio metrics
* Analyze agent interactions
* Evaluate strategy effectiveness

Recent Changes
------------

* Added strategy-specific features
* Enhanced reward calculations
* Improved agent interaction handling
* Added performance tracking 