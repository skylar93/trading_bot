import torch
import torch.nn as nn
from typing import Tuple

class BasePolicy(nn.Module):
    """Base class for all policy networks"""
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network
        
        Args:
            x: Input tensor
        
        Returns:
            Tuple of (action mean, action standard deviation)
        """
        raise NotImplementedError
    
    def get_architecture_type(self) -> str:
        """Get architecture type
        
        Returns:
            String identifier for architecture type
        """
        raise NotImplementedError 