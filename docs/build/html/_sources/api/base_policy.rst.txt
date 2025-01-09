Base Policy API
=============

Overview
--------

The ``BasePolicy`` class serves as an abstract base class for all policy networks in the system, providing a common interface and structure that all concrete policy implementations must follow.

Class Structure
-------------

.. code-block:: python

    class BasePolicy(nn.Module):
        def __init__(self, observation_space, action_space):
            """Initialize base policy.
            
            Args:
                observation_space: Gym space defining observations
                action_space: Gym space defining actions
            """
            super().__init__()
            self.observation_space = observation_space
            self.action_space = action_space

Key Methods
---------

forward
^^^^^^^

.. code-block:: python

    def forward(self, x):
        """Forward pass of the policy network.
        
        Args:
            x: Input observation tensor
            
        Returns:
            action_mean: Mean of action distribution
            action_std: Standard deviation of action distribution
            
        Raises:
            NotImplementedError: Must be implemented by child classes
        """
        raise NotImplementedError

get_architecture_type
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def get_architecture_type(self):
        """Get identifier for network architecture type.
        
        Returns:
            str: Architecture type identifier
            
        Raises:
            NotImplementedError: Must be implemented by child classes
        """
        raise NotImplementedError

Implementation Details
-------------------

Abstract Methods
^^^^^^^^^^^^^

The class enforces implementation of two key methods in child classes:

1. ``forward(x)``: Define the network's forward pass
   - Must output action mean and standard deviation
   - Input processing specific to architecture

2. ``get_architecture_type()``: Provide architecture identifier
   - Used for model saving/loading
   - Helps with architecture-specific processing

Dependencies
----------

- PyTorch (torch.nn.Module)
- Gym spaces (observation_space, action_space)

Usage Example
-----------

Basic Implementation
^^^^^^^^^^^^^^^^^

.. code-block:: python

    class MLPPolicy(BasePolicy):
        def __init__(self, observation_space, action_space):
            super().__init__(observation_space, action_space)
            self.net = nn.Sequential(
                nn.Linear(observation_space.shape[0], 64),
                nn.ReLU(),
                nn.Linear(64, action_space.shape[0] * 2)
            )
            
        def forward(self, x):
            out = self.net(x)
            mean, log_std = torch.chunk(out, 2, dim=-1)
            return mean, log_std.exp()
            
        def get_architecture_type(self):
            return "mlp"

Best Practices
-----------

1. Network Architecture
^^^^^^^^^^^^^^^^^^

- Keep architecture-specific logic in child classes
- Use appropriate initialization for weights
- Consider input/output dimensions carefully

2. Action Distribution
^^^^^^^^^^^^^^^^^

- Ensure valid action ranges
- Handle continuous vs discrete actions
- Validate output shapes

3. State Processing
^^^^^^^^^^^^^^

- Normalize inputs appropriately
- Handle different observation types
- Validate input shapes

4. Implementation
^^^^^^^^^^^^

- Document architecture details
- Add type hints where helpful
- Include validation checks 