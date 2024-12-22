import torch
import torch.nn as nn
import numpy as np
from agents.models.architectures.base import BaseNetwork


class ValueNetwork(BaseNetwork):
    """Value network for PPO agent using MLP architecture"""

    def __init__(self, observation_space, hidden_size: int = 256):
        """Initialize value network

        Args:
            observation_space: Gym observation space
            hidden_size: Size of hidden layers
        """
        super().__init__()

        # Get input dimension
        if hasattr(observation_space, "shape"):
            input_dim = int(np.prod(observation_space.shape))
        else:
            input_dim = observation_space.n

        # Create network
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network

        Args:
            x: Input tensor of shape (batch_size, window_size, features) or (batch_size, flattened_dim)

        Returns:
            Value prediction
        """
        # Ensure input is float tensor
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)

        # Flatten input if it's 3D
        if len(x.shape) == 3:  # (batch_size, window_size, features)
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
        elif len(x.shape) == 2:  # (window_size, features)
            if x.shape[1] != self.network[0].in_features:
                x = x.reshape(1, -1)

        return self.network(x)

    def get_architecture_type(self) -> str:
        """Get architecture type

        Returns:
            String identifier for architecture type
        """
        return "mlp"
