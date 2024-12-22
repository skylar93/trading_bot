import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from agents.models.architectures.base import BaseNetwork


class PolicyNetwork(BaseNetwork):
    """Policy network for PPO agent using MLP architecture"""

    def __init__(
        self, observation_space, action_space, hidden_size: int = 256
    ):
        """Initialize policy network

        Args:
            observation_space: Gym observation space
            action_space: Gym action space
            hidden_size: Size of hidden layers
        """
        super().__init__()

        # Get input and output dimensions
        if hasattr(observation_space, "shape"):
            # For (window_size, features) shape
            input_dim = int(np.prod(observation_space.shape))
        else:
            input_dim = observation_space.n

        if hasattr(action_space, "shape"):
            output_dim = int(np.prod(action_space.shape))
        else:
            output_dim = action_space.n

        # Create shared network layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        # Output layers for mean and standard deviation
        self.mean = nn.Linear(hidden_size, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)

        # Initialize output layers
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.constant_(self.mean.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through network

        Args:
            x: Input tensor of shape (batch_size, window_size, features) or (batch_size, flattened_dim)

        Returns:
            Tuple of (action mean, action standard deviation)
        """
        # Ensure input is float tensor
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)

        # Flatten input if it's 3D or 2D with window_size
        if len(x.shape) == 3:  # (batch_size, window_size, features)
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
        elif (
            len(x.shape) == 2 and len(self.shared) > 0
        ):  # (window_size, features) or (batch_size, flattened_dim)
            if x.shape[1] != self.shared[0].in_features:
                x = x.reshape(1, -1)

        # Forward pass through shared layers
        features = self.shared(x)

        # Get mean and standard deviation
        action_mean = self.mean(features)
        action_std = torch.exp(self.log_std)

        # Expand std to match batch size if needed
        if len(action_mean.shape) > 1:
            action_std = action_std.expand_as(action_mean)

        return action_mean, action_std

    def get_architecture_type(self) -> str:
        """Get architecture type

        Returns:
            String identifier for architecture type
        """
        return "mlp"
