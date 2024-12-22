from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Tuple, Any
import torch


class BaseNetwork(nn.Module, ABC):
    """Base class for all network architectures"""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass through network

        Args:
            x: Input tensor

        Returns:
            Network output
        """
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