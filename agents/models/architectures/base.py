from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Tuple, Any
import torch
import logging

logger = logging.getLogger(__name__)

class BaseNetwork(nn.Module, ABC):
    """Base class for all networks.
    
    Features:
    - Common save/load functionality
    - Logging setup
    - Abstract methods for network-specific functionality
    """
    
    def __init__(self):
        """Initialize base network."""
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def save(self, path: str):
        """Save model state."""
        torch.save(self.state_dict(), path)
        self.logger.info(f"Saved model state to {path}")
        
    def load(self, path: str):
        """Load model state."""
        self.load_state_dict(torch.load(path))
        self.logger.info(f"Loaded model state from {path}")
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network."""
        pass

    @abstractmethod
    def get_architecture_type(self) -> str:
        """Get architecture type

        Returns:
            String identifier for architecture type
        """
        pass

    def save(self, path: str):
        """Save network state

        Args:
            path: Path to save state to
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Load network state

        Args:
            path: Path to load state from
        """
        self.load_state_dict(torch.load(path)) 