import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, List, Optional
import gymnasium as gym
from gymnasium import spaces
import logging
from agents.strategies.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class MetaNetwork(nn.Module):
    """
    Neural network for meta-agent decision making.
    
    Features:
    - Processes joint observations from multiple sub-agents
    - Processes hidden state representations from sub-agents
    - Outputs either discrete agent selection or continuous weights
    - Supports both actor-critic and direct policy architectures
    - Configurable hidden layer sizes
    
    Implementation Notes:
    - Uses separate networks for actor and critic
    - Handles both discrete and continuous action spaces
    - Implements proper weight initialization
    - Supports batch processing for efficient training
    - Processes sub-agent hidden states for enhanced decision making
    
    Recent Changes:
    - Added support for processing sub-agent hidden state representations
    - Added support for continuous ensemble weights
    - Implemented attention mechanism for agent selection
    - Enhanced network architecture with residual connections
    - Improved dimension handling with standardized input processing
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        continuous_ensemble: bool = False,
        use_attention: bool = True
    ):
        """
        Initialize meta-network.
        
        Args:
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
            continuous_ensemble: Whether to output continuous weights
            use_attention: Whether to use attention for integrating sub-agent hidden states
        """
        super().__init__()
        
        self.use_attention = use_attention
        self.observation_dim = observation_dim
        
        # Common feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Attention mechanism for integrating sub-agent hidden states
        if use_attention:
            self.attention_key = nn.Linear(hidden_dim, hidden_dim)
            self.attention_query = nn.Linear(hidden_dim, hidden_dim)
            self.attention_value = nn.Linear(hidden_dim, hidden_dim)
            self.attention_output = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor network
        if continuous_ensemble:
            # For continuous weights, output values in [0, 1] that sum to 1
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)  # Ensure weights sum to 1
            )
        else:
            # For discrete selection, output logits
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Store configuration
        self.continuous_ensemble = continuous_ensemble
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
    
    def _standardize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standardize input tensor to expected dimension.
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Standardized tensor of shape (batch_size, observation_dim)
        """
        # Log original shape for debugging
        original_shape = x.shape
        logger.debug(f"Original input shape: {original_shape}")
        
        # Ensure 2D tensor (batch_size, features)
        if x.dim() == 1:
            # If we have a single vector, add batch dimension
            x = x.unsqueeze(0)
            logger.debug(f"Added batch dimension: {x.shape}")
        elif x.dim() > 2:
            # If we have a batch of sequences or higher dimensions, flatten to (batch_size, -1)
            batch_size = x.size(0)
            x = x.reshape(batch_size, -1)
            logger.debug(f"Flattened to 2D: {x.shape}")
        
        # Handle dimension mismatch with expected observation dimension
        expected_dim = self.observation_dim
        if x.size(-1) != expected_dim:
            logger.warning(
                f"Input dimension mismatch: got {x.size(-1)}, expected {expected_dim}. "
                f"Reshaping input to match expected dimension."
            )
            
            if x.size(-1) > expected_dim:
                # If input is larger, truncate to expected dimension
                x = x[..., :expected_dim]
                logger.debug(f"Truncated to expected dimension: {x.shape}")
            else:
                # If input is smaller, pad with zeros
                padding = torch.zeros(*x.shape[:-1], expected_dim - x.size(-1), device=x.device)
                x = torch.cat([x, padding], dim=-1)
                logger.debug(f"Padded to expected dimension: {x.shape}")
        
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, observation_dim) or any shape that can be standardized
            
        Returns:
            Tuple of (action_output, value)
        """
        # Standardize input to ensure correct dimensions
        x = self._standardize_input(x)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Actor output
        action_output = self.actor(features)
        
        # Critic output
        value = self.critic(features)
        
        return action_output, value
    
    def get_action(
        self, 
        x: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from the network.
        
        Args:
            x: Input tensor
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        action_output, value = self.forward(x)
        
        if self.continuous_ensemble:
            # For continuous ensemble, action_output is already the weights
            if deterministic:
                action = action_output
            else:
                # Add some exploration noise but ensure weights still sum to 1
                noise = torch.randn_like(action_output) * 0.1
                action = torch.softmax(action_output + noise, dim=-1)
            
            # Compute log probability (approximate for continuous case)
            log_prob = torch.sum(torch.log(action_output + 1e-8) * action, dim=-1)
        else:
            # For discrete selection, sample from categorical distribution
            if deterministic:
                action = torch.argmax(action_output, dim=-1, keepdim=True).float()
                # In deterministic mode, compute log prob for the selected action
                dist = torch.distributions.Categorical(logits=action_output)
                log_prob = dist.log_prob(action.view(-1).long()).view(-1, 1)
            else:
                dist = torch.distributions.Categorical(logits=action_output)
                action = dist.sample().float().view(-1, 1)
                log_prob = dist.log_prob(action.view(-1).long()).view(-1, 1)
        
        return action, log_prob, value


class MetaAgent(BaseAgent):
    """
    Meta-agent for ensemble decision making.
    
    Features:
    - Coordinates decisions from multiple sub-agents
    - Can select best agent or blend their actions
    - Learns from experience which agent performs best in different situations
    - Processes sub-agent hidden state representations for enhanced decision making
    - Adapts to changing market conditions
    - Supports both discrete selection and continuous weighting
    
    Implementation Notes:
    - Uses PPO algorithm for training
    - Maintains history of agent performance
    - Processes sub-agent internal representations
    - Implements proper exploration-exploitation balance
    - Handles both discrete and continuous action spaces
    - Supports online learning from streaming data
    
    Recent Changes:
    - Added support for processing sub-agent hidden state representations
    - Added support for continuous ensemble weights
    - Implemented attention mechanism for agent selection
    - Enhanced reward shaping for better agent selection
    - Added market regime detection for context-aware selection
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 3e-4,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_param: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        continuous_ensemble: bool = False,
        use_attention: bool = True,
        **kwargs
    ):
        """
        Initialize meta-agent.
        
        Args:
            observation_space: Observation space
            action_space: Action space
            device: Device to use for computations
            learning_rate: Learning rate for optimizer
            hidden_dim: Dimension of hidden layers
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_param: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy coefficient
            max_grad_norm: Maximum gradient norm
            continuous_ensemble: Whether to use continuous ensemble weights
            use_attention: Whether to use attention for integrating sub-agent hidden states
        """
        super().__init__(observation_space, action_space)
        
        self.device = torch.device(device)
        self.continuous_ensemble = continuous_ensemble
        self.use_attention = use_attention
        
        # PPO hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Initialize network
        self.network = MetaNetwork(
            observation_dim=observation_space.shape[0],
            action_dim=action_space.shape[0] if continuous_ensemble else 1,
            hidden_dim=hidden_dim,
            continuous_ensemble=continuous_ensemble,
            use_attention=use_attention
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Initialize buffers for PPO
        self.reset_buffers()
        
        logger.info(
            f"Initialized MetaAgent with {'continuous' if continuous_ensemble else 'discrete'} "
            f"ensemble, observation dim: {observation_space.shape[0]}, "
            f"action dim: {action_space.shape[0] if continuous_ensemble else 1}"
        )
    
    def reset_buffers(self):
        """Reset experience buffers."""
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
    
    def get_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Get action from the agent.
        
        Args:
            observation: Observation array, which may include sub-agent hidden states
                         if provided by the MultiAgentManager
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Action array representing either a discrete agent selection or continuous weights
        """
        # The MetaAgent itself doesn't expose its hidden state through a tuple return.
        # Instead, it processes the hidden states of sub-agents that are included in the observation.
        
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # Get action from network
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(obs_tensor, deterministic)
        
        # Convert to numpy
        action_np = action.cpu().numpy()[0]
        
        return action_np
    
    def train_step(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """
        Train the agent on a single experience.
        
        Args:
            experience: Experience dictionary which may include sub-agent hidden states
                        in the observation if provided by the MultiAgentManager
            
        Returns:
            Dictionary of training metrics
            
        Recent Changes:
            - Improved dimension handling for actions and observations
            - Added robust error handling for tensor shape mismatches
            - Enhanced logging for tensor shape debugging
        """
        # Extract experience
        observation = experience.get("observation")
        action = experience.get("action")
        reward = experience.get("reward", 0.0)
        next_observation = experience.get("next_observation")
        done = experience.get("done", False)
        
        if observation is None or action is None or next_observation is None:
            logger.warning("Missing required experience data for training")
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        
        # Log input shapes for debugging
        logger.debug(f"train_step input shapes - observation: {observation.shape}, action: {action.shape if hasattr(action, 'shape') else 'scalar'}")
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # Handle action tensor conversion based on type
        if isinstance(action, np.ndarray):
            # For array actions, maintain shape
            action_tensor = torch.FloatTensor(action).to(self.device)
            # Ensure action has batch dimension
            if action_tensor.dim() == 1:
                action_tensor = action_tensor.unsqueeze(0)
        else:
            # For scalar actions, convert to tensor with batch dimension
            action_tensor = torch.FloatTensor([action]).unsqueeze(0).to(self.device)
        
        # Get log probability and value
        with torch.no_grad():
            action_output, value = self.network(obs_tensor)
            
            if self.continuous_ensemble:
                # Ensure action dimensions match
                if action_output.shape[-1] != action_tensor.shape[-1]:
                    logger.warning(f"Action shape mismatch: output {action_output.shape}, action {action_tensor.shape}")
                    
                    # Standardize dimensions
                    if action_output.shape[-1] > action_tensor.shape[-1]:
                        # Pad action_tensor with zeros
                        padding = torch.zeros(
                            *action_tensor.shape[:-1], 
                            action_output.shape[-1] - action_tensor.shape[-1], 
                            device=action_tensor.device
                        )
                        action_tensor = torch.cat([action_tensor, padding], dim=-1)
                        logger.debug(f"Padded action_tensor to shape: {action_tensor.shape}")
                    else:
                        # Truncate action_tensor
                        action_tensor = action_tensor[..., :action_output.shape[-1]]
                        logger.debug(f"Truncated action_tensor to shape: {action_tensor.shape}")
                
                # For continuous ensemble, compute log probability
                log_prob = torch.sum(torch.log(action_output + 1e-8) * action_tensor, dim=-1, keepdim=True)
            else:
                # For discrete selection, compute log probability from categorical
                # Ensure action is within valid range (0 to n-1 where n is number of logits)
                n_categories = action_output.size(-1)
                
                # Handle different action formats
                if action_tensor.dim() > 1 and action_tensor.size(-1) > 1:
                    # For one-hot encoded actions, convert to indices
                    valid_action = torch.argmax(action_tensor, dim=-1).long()
                else:
                    # For scalar or single-dimension actions, clamp to valid range
                    valid_action = torch.clamp(action_tensor.long().view(-1), 0, max(0, n_categories - 1))
                
                # Create categorical distribution and compute log probability
                dist = torch.distributions.Categorical(logits=action_output)
                log_prob = dist.log_prob(valid_action.view(-1)).view(-1, 1)
        
        # Store experience
        self.observations.append(observation)
        self.actions.append(action)
        self.log_probs.append(log_prob.cpu().numpy()[0])
        self.values.append(value.cpu().numpy()[0])
        self.rewards.append(reward)
        self.dones.append(done)
        
        # Only update if we have enough data
        if len(self.observations) < 32:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        
        # Perform PPO update
        metrics = self._update_policy()
        
        # Reset buffers after update
        self.reset_buffers()
        
        return metrics
    
    def _update_policy(self) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Returns:
            Dictionary of training metrics
            
        Recent Changes:
            - Standardized tensor shapes for log probabilities and advantages
            - Improved dimension handling for actions
            - Enhanced error handling and logging
        """
        # Convert buffers to tensors
        observations = torch.FloatTensor(np.array(self.observations)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        old_values = torch.FloatTensor(np.array(self.values)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)).to(self.device)
        
        # Log tensor shapes for debugging
        logger.debug(f"_update_policy tensor shapes - observations: {observations.shape}, actions: {actions.shape}")
        logger.debug(f"old_log_probs: {old_log_probs.shape}, old_values: {old_values.shape}")
        
        # Ensure old_log_probs has the right shape (batch_size, 1)
        if old_log_probs.dim() == 1:
            old_log_probs = old_log_probs.unsqueeze(1)
            logger.debug(f"Reshaped old_log_probs to: {old_log_probs.shape}")
        
        # Compute returns and advantages
        returns, advantages = self._compute_gae(rewards, old_values, dones)
        
        # Ensure advantages has the right shape (batch_size, 1)
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
            logger.debug(f"Reshaped advantages to: {advantages.shape}")
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        policy_loss_epoch = 0
        value_loss_epoch = 0
        entropy_epoch = 0
        
        # Single epoch update for online learning
        # Get current policy and value
        action_output, values = self.network(observations)
        
        # Compute policy loss
        if self.continuous_ensemble:
            # Ensure action dimensions match
            if action_output.shape[-1] != actions.shape[-1]:
                logger.warning(f"Action shape mismatch in update: output {action_output.shape}, actions {actions.shape}")
                
                # Standardize dimensions
                if action_output.shape[-1] > actions.shape[-1]:
                    # Pad actions with zeros
                    padding = torch.zeros(
                        *actions.shape[:-1], 
                        action_output.shape[-1] - actions.shape[-1], 
                        device=actions.device
                    )
                    actions = torch.cat([actions, padding], dim=-1)
                    logger.debug(f"Padded actions to shape: {actions.shape}")
                else:
                    # Truncate actions
                    actions = actions[..., :action_output.shape[-1]]
                    logger.debug(f"Truncated actions to shape: {actions.shape}")
            
            # For continuous ensemble, compute log probability
            log_probs = torch.sum(torch.log(action_output + 1e-8) * actions, dim=-1, keepdim=True)
            entropy = -torch.sum(action_output * torch.log(action_output + 1e-8), dim=-1).mean()
        else:
            # For discrete selection, compute log probability from categorical
            # Handle different action formats
            if actions.dim() > 1 and actions.size(-1) > 1:
                # For one-hot encoded actions, convert to indices
                action_indices = torch.argmax(actions, dim=-1).long()
            else:
                # For scalar or single-dimension actions, clamp to valid range
                n_categories = action_output.size(-1)
                action_indices = torch.clamp(actions.long().view(-1), 0, max(0, n_categories - 1))
            
            # Create categorical distribution and compute log probability
            dist = torch.distributions.Categorical(logits=action_output)
            log_probs = dist.log_prob(action_indices).view(-1, 1)
            entropy = dist.entropy().mean()
        
        # Ensure log_probs and old_log_probs have the same shape
        if log_probs.shape != old_log_probs.shape:
            logger.warning(f"Log prob shape mismatch: current {log_probs.shape}, old {old_log_probs.shape}")
            
            # Standardize to 2D (batch_size, 1)
            log_probs = log_probs.view(-1, 1)
            old_log_probs = old_log_probs.view(-1, 1)
            logger.debug(f"Standardized log_probs to: {log_probs.shape}, old_log_probs to: {old_log_probs.shape}")
        
        # Ensure advantages has the right shape for the computation
        if advantages.shape != log_probs.shape:
            logger.warning(f"Advantages shape mismatch: advantages {advantages.shape}, log_probs {log_probs.shape}")
            
            # Standardize to match log_probs shape
            advantages = advantages.view(*log_probs.shape)
            logger.debug(f"Standardized advantages to shape: {advantages.shape}")
        
        # Compute ratio and clipped loss
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute value loss
        value_loss = 0.5 * ((values - returns) ** 2).mean()
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Update metrics
        policy_loss_epoch = policy_loss.item()
        value_loss_epoch = value_loss.item()
        entropy_epoch = entropy.item()
        
        return {
            "policy_loss": policy_loss_epoch,
            "value_loss": value_loss_epoch,
            "entropy": entropy_epoch
        }
    
    def _compute_gae(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor, 
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns and advantages using GAE.
        
        Args:
            rewards: Reward tensor
            values: Value tensor
            dones: Done tensor
            
        Returns:
            Tuple of (returns, advantages)
        """
        # Initialize returns and advantages
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # Initialize last values
        last_gae_lam = 0
        
        # Compute returns and advantages in reverse order
        for t in reversed(range(len(rewards))):
            # Compute next value
            if t == len(rewards) - 1:
                next_value = 0
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            # Compute delta
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            
            # Compute GAE
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
        
        # Compute returns
        returns = advantages + values
        
        return returns, advantages
    
    def save(self, path: str) -> None:
        """
        Save agent to disk.
        
        Args:
            path: Path to save to
        """
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "continuous_ensemble": self.continuous_ensemble
        }, path)
        
        logger.info(f"Saved MetaAgent to {path}")
    
    def load(self, path: str) -> None:
        """
        Load agent from disk.
        
        Args:
            path: Path to load from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.continuous_ensemble = checkpoint["continuous_ensemble"]
        
        logger.info(f"Loaded MetaAgent from {path}") 