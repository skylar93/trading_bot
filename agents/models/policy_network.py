import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    """Policy network for PPO agent"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # Mean and log_std
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of mean and log_std tensors
        """
        output = self.network(x)
        mean, log_std = output.chunk(2, dim=-1)
        return mean, log_std 