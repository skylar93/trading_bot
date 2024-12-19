import torch
import torch.nn as nn
import numpy as np

class ValueNetwork(nn.Module):
    """Value network for PPO agent"""
    
    def __init__(self,
                 observation_space,
                 hidden_size: int = 256):
        """Initialize value network
        
        Args:
            observation_space: Gym observation space
            hidden_size: Size of hidden layers
        """
        super().__init__()
        
        # Get input dimension
        if hasattr(observation_space, 'shape'):
            # For (window_size, features) shape
            input_dim = int(np.prod(observation_space.shape))
        else:
            input_dim = observation_space.n
        
        # Create network layers
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)  # Output single value
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
            Value estimate
        """
        # Ensure input is float tensor
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        
        # Flatten input if it's 3D or 2D with window_size
        if len(x.shape) == 3:  # (batch_size, window_size, features)
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
        elif len(x.shape) == 2 and len(self.network) > 0:  # (window_size, features) or (batch_size, flattened_dim)
            if x.shape[1] != self.network[0].in_features:
                x = x.reshape(1, -1)
        
        # Forward pass
        value = self.network(x)
        
        return value