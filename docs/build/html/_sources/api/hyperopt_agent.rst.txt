Hyperopt Agent API
================

Overview
--------

The hyperopt agent module provides minimal implementations of neural networks and PPO agents specifically designed for hyperparameter optimization experiments.

Class Structure
-------------

MinimalNetwork
^^^^^^^^^^^^

.. code-block:: python

    class MinimalNetwork(nn.Module):
        def __init__(self, input_dim, hidden_size=64):
            """Initialize minimal network.
            
            Args:
                input_dim: Input dimension
                hidden_size: Hidden layer size (default: 64)
            """
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )
            self.policy_head = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Tanh()
            )
            self.value_head = nn.Linear(hidden_size, 1)

MinimalPPOAgent
^^^^^^^^^^^^

.. code-block:: python

    class MinimalPPOAgent:
        def __init__(self, state_dim, learning_rate=3e-4):
            """Initialize minimal PPO agent.
            
            Args:
                state_dim: State space dimension
                learning_rate: Learning rate for optimizer
            """
            self.network = MinimalNetwork(state_dim)
            self.optimizer = optim.Adam(self.network.parameters(), 
                                      lr=learning_rate)

Key Methods
---------

Network Forward Pass
^^^^^^^^^^^^^^^^^

.. code-block:: python

    def forward(self, x):
        """Forward pass through network.
        
        Args:
            x: Input tensor
            
        Returns:
            tuple: (policy_output, value_output)
        """
        shared_features = self.shared(x)
        return (self.policy_head(shared_features), 
                self.value_head(shared_features))

Agent Actions
^^^^^^^^^^

.. code-block:: python

    def get_action(self, state):
        """Get action from current policy.
        
        Args:
            state: Current observation
            
        Returns:
            float: Action in [-1, 1] range
        """

Agent Training
^^^^^^^^^^^

.. code-block:: python

    def train(self, state, action, reward, next_state, done):
        """Train agent on single transition.
        
        Args:
            state: Current state
            action: Taken action
            reward: Received reward
            next_state: Next state
            done: Episode done flag
            
        Returns:
            dict: Training metrics
        """

Implementation Details
-------------------

Network Architecture
^^^^^^^^^^^^^^^^^

1. Shared Layers:
   - Two fully connected layers with ReLU
   - Hidden size configurable (default 64)

2. Policy Head:
   - Single linear layer
   - Tanh activation for [-1, 1] output

3. Value Head:
   - Single linear layer
   - No activation (raw value)

PPO Implementation
^^^^^^^^^^^^^^^

1. Action Selection:
   - Deterministic policy output
   - No exploration noise

2. Training Process:
   - Single transition updates
   - Advantage calculation
   - Clipped objective
   - Value loss (MSE)

Dependencies
----------

- PyTorch (torch, nn.Module)
- Numpy (array operations)
- Gym (spaces, optional)

Usage Example
-----------

Basic Usage
^^^^^^^^^

.. code-block:: python

    # Initialize agent
    agent = MinimalPPOAgent(state_dim=5)
    
    # Get action
    action = agent.get_action(state)
    
    # Train on transition
    metrics = agent.train(state, action, reward, next_state, done)
    
    # Save/load
    agent.save("minimal_ppo.pt")
    agent.load("minimal_ppo.pt")

Best Practices
-----------

1. Network Configuration
^^^^^^^^^^^^^^^^^^^

- Match hidden_size to problem complexity
- Consider input normalization
- Monitor gradient norms

2. Training Stability
^^^^^^^^^^^^^^^^^

- Start with small learning rates
- Monitor value loss convergence
- Check policy entropy

3. Hyperparameter Tuning
^^^^^^^^^^^^^^^^^^^

- Adjust clip epsilon if needed
- Tune learning rate schedule
- Experiment with hidden sizes 