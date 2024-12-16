import torch
import torch.nn as nn

class ValueNetwork(nn.Module):
    """Value network for PPO agent"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Value output
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Value tensor of shape (batch_size, 1)
        """
        return self.network(x) 