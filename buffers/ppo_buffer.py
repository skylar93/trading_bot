"""Experience buffer for PPO agent.

Features:
- Stores state transitions and rewards
- Computes advantages using GAE
- Supports batch sampling
- Handles both single and multi-agent experiences

Implementation Notes:
- Uses numpy arrays for efficient storage
- Supports variable-length episodes
- Computes returns and advantages on-the-fly
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PPOBuffer:
    """Buffer for storing PPO experiences and computing advantages.
    
    Features:
    - Maintains state shape (window_size, features) throughout
    - Computes GAE for advantage estimation
    - Supports batch sampling with proper reshaping
    - Handles NaN values
    """
    
    def __init__(
        self,
        obs_shape: tuple,
        action_shape: tuple,
        size: int,
        gamma: float,
        gae_lambda: float,
        device: str,
    ):
        """Initialize PPO buffer.
        
        Args:
            obs_shape: Shape of observations (window_size, features)
            action_shape: Shape of actions
            size: Maximum buffer size
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            device: Device to store tensors on
        """
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.max_size = size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        
        # Initialize buffers
        self.reset()
        
        logger.info(
            f"Initialized PPOBuffer with size={size}, obs_shape={obs_shape}, "
            f"action_shape={action_shape}, gamma={gamma}, gae_lambda={gae_lambda}"
        )
        
    def reset(self):
        """Reset buffer to empty state."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = None
        self.returns = None
        
    def append(self, experience: Dict):
        """Add experience to buffer.
        
        Args:
            experience: Dictionary containing state, action, reward, done, value, log_prob
        """
        # Ensure state has correct shape
        state = experience["state"]
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
            
        # Reshape if needed
        if len(state.shape) == 3:  # (batch_size, window_size, features)
            state = state.squeeze(0)  # Remove batch dimension
            
        if state.shape != self.obs_shape:
            raise ValueError(f"Expected state shape {self.obs_shape}, got {state.shape}")
            
        # Store experience
        self.states.append(state)
        self.actions.append(experience["action"])
        self.rewards.append(float(experience["reward"]))  # Ensure reward is float
        self.values.append(float(experience["value"]))  # Ensure value is float
        self.log_probs.append(float(experience["log_prob"]))  # Ensure log_prob is float
        self.dones.append(bool(experience["done"]))  # Ensure done is boolean
        
    def compute_advantages(self, last_value: np.ndarray):
        """Compute advantages using GAE.
        
        Args:
            last_value: Value estimate for last state
        """
        if len(self.states) == 0:
            logger.warning("Cannot compute advantages with empty buffer")
            return
            
        # Convert lists to numpy arrays
        values = np.array(self.values)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        
        # Ensure last_value is scalar
        if isinstance(last_value, np.ndarray):
            last_value = float(last_value.squeeze())
            
        # Append last value for GAE calculation
        values = np.append(values, last_value)
        
        # Initialize advantages array
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        # Compute GAE
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            
        # Store advantages and returns
        self.advantages = advantages
        self.returns = advantages + values[:-1]  # Remove last value
        
        logger.debug(f"Computed advantages with shape {advantages.shape}")
        
    def get_batch(
        self,
        batch_size: Optional[int] = None,
        shuffle: bool = True
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        """Get batch of experiences from buffer.
        
        Args:
            batch_size: Batch size (if None, return all)
            shuffle: Whether to shuffle the batch
            
        Returns:
            Tuple of (states, actions, log_probs, returns, advantages, values)
            or None if buffer is empty
        """
        if len(self.states) == 0:
            logger.warning("Cannot get batch from empty buffer")
            return None
            
        if self.advantages is None:
            logger.warning("Advantages not computed, computing now with zero last value")
            self.compute_advantages(np.zeros((1,)))
            
        # Convert to numpy arrays with correct shapes
        states = np.stack(self.states)  # (N, window_size, features)
        actions = np.vstack(self.actions)  # (N, action_dim)
        log_probs = np.array(self.log_probs)  # (N,)
        values = np.array(self.values)  # (N,)
        
        # Get indices
        indices = np.arange(len(self.states))
        if shuffle:
            np.random.shuffle(indices)
        if batch_size is not None:
            indices = indices[:batch_size]
            
        # Convert to tensors with proper shapes
        states = torch.FloatTensor(states[indices]).to(self.device)  # (batch_size, window_size, features)
        actions = torch.FloatTensor(actions[indices]).to(self.device)  # (batch_size, action_dim)
        log_probs = torch.FloatTensor(log_probs[indices]).to(self.device)  # (batch_size,)
        returns = torch.FloatTensor(self.returns[indices]).to(self.device)  # (batch_size,)
        advantages = torch.FloatTensor(self.advantages[indices]).to(self.device)  # (batch_size,)
        values = torch.FloatTensor(values[indices]).to(self.device)  # (batch_size,)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return states, actions, log_probs, returns, advantages, values
        
    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.states) 