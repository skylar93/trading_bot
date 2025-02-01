"""
Dummy agent for testing that makes small random trades
"""

import logging
from typing import Dict, Any
import numpy as np

class DummyAgent:
    """Dummy agent for testing that makes small random trades"""
    
    def __init__(self):
        self.step_count = -1  # Start at -1 so first increment gives 0
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def get_action(self, state: Dict[str, Any]) -> float:
        """Generate small random actions for testing"""
        self.step_count += 1
        
        # First action should be non-zero but small for consistency test
        if self.step_count == 0:
            return 0.5
            
        # Trade every 5 steps with small magnitude
        if self.step_count % 5 == 0:
            action = 0.5 if (self.step_count // 5) % 2 == 0 else -0.5
            self.logger.info(f"DummyAgent taking action: {action}")
            return action
            
        return 0.0
        
    def predict(self, state: Any) -> np.ndarray:
        """Alias for get_action to maintain compatibility with other agents"""
        action = self.get_action(state)
        return np.array([action]) 