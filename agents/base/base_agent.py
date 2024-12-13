from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Any

class BaseAgent(ABC):
    """Abstract base class for all trading agents"""
    
    def __init__(self, observation_space: Any, action_space: Any):
        self.observation_space = observation_space
        self.action_space = action_space
        
    @abstractmethod
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> float:
        """Get action from the agent"""
        pass
    
    @abstractmethod
    def train(self, replay_buffer: List[Tuple]) -> Dict[str, float]:
        """Train the agent using experiences from the replay buffer"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save the agent to disk"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load the agent from disk"""
        pass