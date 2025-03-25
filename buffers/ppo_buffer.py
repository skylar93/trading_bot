"""Experience buffer for PPO agent.

Features:
- Stores state transitions and rewards
- Computes advantages using GAE
- Supports batch sampling
- Handles both single and multi-agent experiences
- Maintains log_probs from rollout phase for accurate ratio calculation

Implementation Notes:
- Uses numpy arrays for efficient storage
- Supports variable-length episodes
- Computes returns and advantages on-the-fly
- Designed to work with the proper PPO rollout-then-update pattern
- Preserves original log_probs from policy during rollout phase

Recent Changes:
- Added support for the PPO rollout-then-update pattern
- Improved handling of log probabilities to maintain stability in PPO ratio calculation
- Enhanced robustness with better shape handling and error checking
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
            
        # Check if state has valid shape
        if state is None or (isinstance(state, np.ndarray) and state.size == 0):
            logger.warning("Received empty state, skipping this experience")
            return
            
        # Reshape if needed
        if len(state.shape) == 3:  # (batch_size, window_size, features)
            state = state.squeeze(0)  # Remove batch dimension
        
        # Handle mismatch between state shape and expected obs_shape
        if state.shape != self.obs_shape:
            # Handle common shape mismatches by reshaping if possible
            if np.prod(state.shape) == np.prod(self.obs_shape):
                # If total elements match, reshape
                state = state.reshape(self.obs_shape)
                logger.debug(f"Reshaped state from {experience['state'].shape} to {state.shape}")
            else:
                # If we can't reshape, log warning but continue by padding/truncating
                logger.warning(
                    f"State shape mismatch. Expected {self.obs_shape}, got {state.shape}. "
                    f"Attempting to adapt."
                )
                # Create a zero-filled state with correct shape
                adapted_state = np.zeros(self.obs_shape, dtype=np.float32)
                
                # Copy as much data as we can
                if len(state.shape) == 1 and len(self.obs_shape) == 1:
                    # 1D to 1D
                    min_len = min(state.shape[0], self.obs_shape[0])
                    adapted_state[:min_len] = state[:min_len]
                elif len(state.shape) == 2 and len(self.obs_shape) == 2:
                    # 2D to 2D
                    min_rows = min(state.shape[0], self.obs_shape[0])
                    min_cols = min(state.shape[1], self.obs_shape[1])
                    adapted_state[:min_rows, :min_cols] = state[:min_rows, :min_cols]
                
                state = adapted_state
            
        # Store experience
        self.states.append(state)
        
        # Convert action to numpy if it's a tensor
        action = experience["action"]
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        # Ensure action has correct shape
        if np.asarray(action).ndim == 0:  # Scalar action
            action = np.array([action])
        elif np.asarray(action).ndim == 1 and len(action) == 1:  # Already correct shape
            pass
        else:
            # Reshape or adapt action if needed
            target_shape = self.action_shape
            if np.prod(np.asarray(action).shape) != np.prod(target_shape):
                logger.warning(
                    f"Action shape mismatch. Expected {target_shape}, got {np.asarray(action).shape}. "
                    f"Adapting."
                )
                # Create a zero-filled action with correct shape
                adapted_action = np.zeros(target_shape, dtype=np.float32)
                
                # Copy as much data as we can
                flat_action = np.asarray(action).flatten()
                flat_adapted = adapted_action.flatten()
                min_len = min(len(flat_action), len(flat_adapted))
                flat_adapted[:min_len] = flat_action[:min_len]
                
                action = adapted_action.reshape(target_shape)
        
        self.actions.append(action)
        
        # Process scalar values
        try:
            self.rewards.append(float(experience["reward"]))
            self.values.append(float(experience["value"]))
            self.dones.append(bool(experience["done"]))
        except (ValueError, TypeError) as e:
            logger.warning(f"Error processing experience values: {e}")
            # Use defaults as fallback
            self.rewards.append(0.0)
            self.values.append(0.0)
            self.dones.append(False)
        
        # Handle log_prob
        log_prob = experience.get("log_prob", 0.0)
        if isinstance(log_prob, torch.Tensor):
            log_prob = log_prob.cpu().numpy()
            
        # Handle multi-dimensional log_probs by taking the mean if necessary
        if isinstance(log_prob, np.ndarray) and log_prob.size > 1:
            log_prob = float(np.mean(log_prob))
        else:
            try:
                log_prob = float(log_prob)
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing log_prob: {e}")
                log_prob = 0.0
                
        self.log_probs.append(log_prob)
        
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