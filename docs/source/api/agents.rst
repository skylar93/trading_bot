Trading Agents
==============

Overview
--------

The Trading Agents module implements reinforcement learning agents for trading, including a base neural network architecture and a PPO (Proximal Policy Optimization) agent implementation.

Key Components
-------------

BaseNetwork
^^^^^^^^^^

Base neural network architecture inheriting from PyTorch's nn.Module:

* ``input_dim``: Input dimension (observation space size)
* ``hidden_dim``: Hidden layer dimension (default: 128)
* Architecture: Input -> 2 hidden layers -> 3 actions (sell, hold, buy)

Methods:
    * ``__init__(input_dim, hidden_dim=128)``: Initialize network layers
    * ``forward(x: torch.Tensor)``: Forward pass returning action logits

PPOAgent
^^^^^^^

PPO algorithm implementation for trading:

Core Methods:
    * ``__init__(env, learning_rate, gamma, ...)``: Initialize agent
    * ``reset_memory()``: Clear episode memory
    * ``get_action(state)``: Get action from policy network
    * ``train(state, action, reward, next_state, done)``: Training step
    * ``_update()``: Core PPO update logic
    * ``evaluate(env)``: Evaluate agent performance
    * ``save(path)``: Save agent state
    * ``load(path)``: Load agent state

Implementation Details
--------------------

Neural Network Architecture
^^^^^^^^^^^^^^^^^^^^^^^

* Fully connected layers with ReLU activation
* Input layer matches environment observation space
* Two hidden layers with configurable size
* Output layer with 3 neurons (sell, hold, buy actions)

PPO Implementation
^^^^^^^^^^^^^^^

* Separate policy and value networks
* Episode memory management
* Advantage calculation
* PPO clipping and loss computation
* Multiple epochs of updates

Memory Management
^^^^^^^^^^^^^^

* States: Environment observations
* Actions: Selected trading actions
* Rewards: Trading returns
* Values: Value network predictions
* Log probabilities: Action probabilities

Dependencies
-----------

* ``torch``: Deep learning framework
* ``torch.nn``: Neural network modules
* ``torch.optim``: Optimization algorithms
* ``torch.distributions``: Probability distributions
* ``numpy``: Numerical operations
* ``typing``: Type hints

Usage Example
------------

.. code-block:: python

    # Create base network
    network = BaseNetwork(
        input_dim=20,  # observation space size
        hidden_dim=128
    )

    # Initialize PPO agent
    agent = PPOAgent(
        env=trading_env,
        learning_rate=3e-4,
        gamma=0.99,
        epsilon=0.2
    )

    # Training loop
    state = env.reset()
    for step in range(max_steps):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        
        agent.train(state, action, reward, next_state, done)
        
        if done:
            state = env.reset()
        else:
            state = next_state

    # Evaluate agent
    metrics = agent.evaluate(eval_env)
    print(f"Sharpe ratio: {metrics['sharpe_ratio']}")

    # Save agent
    agent.save("models/ppo_agent.pt")

Best Practices
-------------

1. Network Architecture
   * Match input dimension to observation space
   * Choose appropriate hidden layer sizes
   * Initialize weights properly

2. PPO Training
   * Set reasonable hyperparameters
   * Monitor value/policy losses
   * Track advantage estimates

3. Memory Management
   * Clear memory after updates
   * Handle episode termination
   * Manage batch sizes

4. Model Management
   * Save checkpoints regularly
   * Track training metrics
   * Version control models

Recent Changes
-------------

* Added PPO implementation
* Enhanced network architecture
* Improved memory management
* Added evaluation metrics 