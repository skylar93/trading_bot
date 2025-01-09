PPO Agent API
============

Overview
--------

The ``PPOAgent`` class implements the Proximal Policy Optimization (PPO) algorithm with an actor-critic architecture, supporting both single-agent and shared buffer learning.

Class Structure
-------------

.. code-block:: python

    class PPOAgent(BaseAgent):
        def __init__(self, state_dim, action_dim, hidden_dim=256,
                     lr=3e-4, gamma=0.99, clip_epsilon=0.2,
                     c1=1.0, c2=0.01, c3=0.01, target_kl=0.015,
                     batch_size=64, n_epochs=10, buffer_size=2048):
            """Initialize PPO agent.

            Args:
                state_dim (int): Dimension of state space
                action_dim (int): Dimension of action space
                hidden_dim (int): Hidden layer dimension
                lr (float): Learning rate
                gamma (float): Discount factor
                clip_epsilon (float): PPO clip ratio
                c1 (float): Value loss coefficient
                c2 (float): Entropy bonus coefficient
                c3 (float): KL penalty coefficient
                target_kl (float): Target KL divergence
                batch_size (int): Mini-batch size
                n_epochs (int): Number of epochs per update
                buffer_size (int): Experience buffer size
            """

Key Components
------------

Actor-Critic Architecture
^^^^^^^^^^^^^^^^^^^^^

- PolicyNetwork (Actor): Outputs action mean and standard deviation
- ValueNetwork (Critic): Estimates state values
- Shared feature extraction layers
- CosineAnnealing learning rate scheduler

State Processing
^^^^^^^^^^^^^

- Running statistics for state normalization
- Support for various input shapes (batch, window, features)
- Momentum-based updates for mean/std

Key Methods
---------

get_action
^^^^^^^^^

.. code-block:: python

    def get_action(self, state, deterministic=False):
        """Get action from current policy.
        
        Args:
            state: Current observation
            deterministic: If True, return mean action
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: Estimated state value
        """

train
^^^^^

.. code-block:: python

    def train(self, env_or_experiences, total_timesteps=None,
              eval_env=None, eval_freq=10000):
        """Train the agent.
        
        Args:
            env_or_experiences: Gym env or list of experiences
            total_timesteps: Total training timesteps
            eval_env: Optional environment for evaluation
            eval_freq: Evaluation frequency
            
        Returns:
            dict: Training statistics
        """

update
^^^^^^

.. code-block:: python

    def update(self, states, actions, rewards, values,
               log_probs, dones, next_values):
        """Update networks using PPO.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            values: Batch of values
            log_probs: Batch of log probabilities
            dones: Batch of done flags
            next_values: Batch of next state values
            
        Returns:
            dict: Update statistics
        """

Implementation Details
-------------------

PPO Algorithm
^^^^^^^^^^^

1. Collect experiences using current policy
2. Compute advantages using GAE
3. Update policy using clipped objective
4. Monitor KL divergence for early stopping
5. Update value network to minimize MSE
6. Apply entropy bonus for exploration

Advantage Estimation
^^^^^^^^^^^^^^^^^

.. code-block:: python

    def _compute_gae(self, rewards, values, dones, next_values,
                     gamma=0.99, gae_lambda=0.95):
        """Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Episode rewards
            values: Value estimates
            dones: Episode done flags
            next_values: Next state values
            gamma: Discount factor
            gae_lambda: GAE parameter
            
        Returns:
            advantages: GAE advantages
            returns: Value function targets
        """

Learning Rate Schedule
^^^^^^^^^^^^^^^^^^

- CosineAnnealing scheduler
- Decay over training epochs
- Minimum learning rate threshold

Dependencies
----------

- torch: Neural network implementation
- numpy: Numerical operations
- gym: Environment interface
- logging: Debug information

Usage Example
-----------

Basic Usage
^^^^^^^^^

.. code-block:: python

    # Initialize agent
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )
    
    # Train agent
    stats = agent.train(env, total_timesteps=1000000)
    
    # Save agent
    agent.save("ppo_agent.pt")

Shared Experience
^^^^^^^^^^^^^

.. code-block:: python

    # Train from shared buffer
    agent.learn_from_shared_experience(
        shared_buffer,
        batch_size=64,
        n_epochs=10
    )

Best Practices
-----------

1. Hyperparameter Tuning
^^^^^^^^^^^^^^^^^^^

- Adjust clip_epsilon based on task complexity
- Monitor KL divergence and adjust target_kl
- Balance value loss and policy loss coefficients
- Tune entropy bonus for exploration/exploitation

2. Buffer Management
^^^^^^^^^^^^^^^

- Use appropriate buffer_size for task
- Consider episode boundaries
- Handle incomplete episodes
- Monitor buffer statistics

3. Training Stability
^^^^^^^^^^^^^^^^^

- Use state normalization
- Monitor value loss convergence
- Check policy entropy
- Validate advantage estimates

4. Model Deployment
^^^^^^^^^^^^^^^

- Save checkpoints regularly
- Validate loaded models
- Monitor inference performance
- Track environment changes

Recent Changes
------------

- Added shared experience learning
- Improved state normalization
- Enhanced logging capabilities
- Added early stopping based on KL 