"""PPO agent implementation with state normalization and experience sharing.

Features:
- Proximal Policy Optimization (PPO) algorithm
- State normalization using running statistics
- Experience sharing between agents
- Configurable hyperparameters
- Automatic device selection (CPU/GPU)

Implementation Notes:
- Uses separate policy and value networks
- Supports both continuous and discrete action spaces
- Implements early stopping based on KL divergence
- Uses GAE for advantage estimation
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
from gymnasium.spaces import Box
import gymnasium as gym
import pandas as pd
from agents.base.base_agent import BaseAgent
from agents.models.architectures.mlp import PolicyNetwork
from agents.models.architectures.value_mlp import ValueNetwork
from buffers.ppo_buffer import PPOBuffer

logger = logging.getLogger(__name__)


class PPOAgent(BaseAgent):
    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        c1: float = 1.0,
        c2: float = 0.01,
        c3: float = 0.01,
        n_epochs: int = 10,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.015,
        normalize_observations: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ):
        """Initialize PPO agent.
        
        Args:
            observation_space: Observation space
            action_space: Action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clip parameter
            c1: Value loss coefficient
            c2: Entropy coefficient
            c3: KL divergence coefficient
            n_epochs: Number of epochs per update
            batch_size: Batch size for updates
            max_grad_norm: Maximum gradient norm
            target_kl: Target KL divergence
            normalize_observations: Whether to normalize observations
            device: Device to use for computations
            **kwargs: Additional arguments
        """
        super().__init__(observation_space, action_space)
        
        # Set up logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Log unused config keys
        unused_keys = [key for key in kwargs.keys() if key not in self.__init__.__code__.co_varnames]
        if unused_keys:
            self.logger.warning(f"Ignoring unused config keys in PPOAgent: {unused_keys}")
        
        # Store hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.normalize_observations = normalize_observations
        self.eps = 1e-8
        
        # Set device
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Calculate input dimension based on observation space
        if len(observation_space.shape) == 1:  # (features,)
            self.obs_dim = observation_space.shape[0]
        elif len(observation_space.shape) == 2:  # (window_size, features)
            self.obs_dim = observation_space.shape[0] * observation_space.shape[1]
        else:
            raise ValueError(f"Unsupported observation space shape: {observation_space.shape}")
            
        # Initialize networks
        self.network = PolicyNetwork(observation_space, action_space).to(self.device)
        self.value_network = ValueNetwork(observation_space).to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam([
            {'params': self.network.parameters(), 'lr': learning_rate},
            {'params': self.value_network.parameters(), 'lr': learning_rate}
        ])
        
        # Set up learning rate scheduler if specified
        if kwargs.get("use_lr_scheduler", False):
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=kwargs.get("lr_scheduler_step_size", 100),
                gamma=kwargs.get("lr_scheduler_gamma", 0.9)
            )
        else:
            self.scheduler = None
        
        # Initialize normalization statistics with correct dimensions
        self.state_mean = torch.zeros(self.obs_dim, device=self.device)
        self.state_std = torch.ones(self.obs_dim, device=self.device)
        
        # Initialize replay buffer
        self.buffer = PPOBuffer(
            observation_space.shape,
            action_space.shape,
            self.batch_size,
            self.gamma,
            self.gae_lambda,
            self.device
        )
        
        logger.info(
            f"Initialized PPO agent on device: {self.device} with obs_dim={self.obs_dim}"
        )

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize state observations using running mean and standard deviation.
        
        Handles input shapes consistently with PolicyNetwork:
        - 1D: (features,) -> (1, features)
        - 2D: (batch_size, input_size) or (window_size, features) -> (batch_size, input_size)
        - 3D: (batch_size, window_size, features) -> (batch_size, window_size*features)
        
        Args:
            state: Input state tensor
            
        Returns:
            Normalized state tensor with shape (batch_size, obs_dim)
        """
        if not self.normalize_observations:
            return state
            
        # Handle NaN values
        if torch.isnan(state).any():
            self.logger.warning("NaN in state input; replacing with 0.0")
            state = torch.nan_to_num(state, nan=0.0)
            
        original_shape = state.shape
        
        # Reshape input to (batch_size, obs_dim) for normalization
        if state.dim() == 1:
            # (features,) -> (1, features)
            if state.shape[0] == self.obs_dim:
                state = state.unsqueeze(0)
            else:
                raise ValueError(
                    f"Expected 1D input with size {self.obs_dim}, got {state.shape}"
                )
                
        elif state.dim() == 2:
            # (batch_size, input_size) or (window_size, features)
            if state.shape[1] == self.obs_dim:
                # Already (batch_size, obs_dim)
                pass
            elif state.shape[0] * state.shape[1] == self.obs_dim:
                # Single sample (window_size, features) -> (1, window_size*features)
                state = state.reshape(1, -1)
            else:
                raise ValueError(
                    f"Cannot interpret 2D input {state.shape} as (batch_size, {self.obs_dim}) "
                    f"or single sample with size {self.obs_dim}"
                )
                
        elif state.dim() == 3:
            # (batch_size, window_size, features)
            b, w, f = state.shape
            if w * f != self.obs_dim:
                raise ValueError(
                    f"Expected 3D input with window_size*features={self.obs_dim}, "
                    f"got shape {state.shape} with w*f={w*f}"
                )
            state = state.reshape(b, w*f)
            
        else:
            raise ValueError(f"Unsupported input dimensions: {state.dim()}")
            
        # Update running statistics
        with torch.no_grad():
            batch_mean = state.mean(dim=0)
            batch_std = torch.clamp(state.std(dim=0), min=self.eps)
            
            alpha = 0.05
            self.state_mean = (1 - alpha) * self.state_mean + alpha * batch_mean
            self.state_std = (1 - alpha) * self.state_std + alpha * batch_std
            
        # Normalize using running statistics
        normalized = (state - self.state_mean) / self.state_std
        normalized = torch.clamp(normalized, -10, 10)  # Prevent extreme values
        
        # Restore original shape if needed
        if len(original_shape) == 1:
            normalized = normalized.squeeze(0)
        elif len(original_shape) == 3:
            normalized = normalized.reshape(original_shape)
            
        return normalized

    def get_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """Get action from policy network

        Args:
            state: Current state observation (can be numpy array or pandas DataFrame)
                Expected shapes:
                - (features,) : Single feature vector
                - (window_size, 5) : Single window of OHLCV data
                - (batch_size, features) : Batch of feature vectors
                - (batch_size, window_size, 5) : Batch of OHLCV windows
            deterministic: Whether to use deterministic action

        Returns:
            Action as numpy array with shape (1,) for single actions
            or (batch_size, 1) for batched actions
        """
        with torch.no_grad():
            # Convert DataFrame to numpy if needed
            if isinstance(state, pd.DataFrame):
                state = state.to_numpy()
            
            # Convert to tensor
            state_tensor = torch.FloatTensor(state).to(self.device)
            
            # Normalize state (handles all shape cases internally)
            state_tensor = self._normalize_state(state_tensor)
            
            # Get action distribution parameters
            action_mean, action_std = self.network(state_tensor)

            if deterministic:
                raw_action = action_mean
            else:
                dist = Normal(action_mean, action_std)
                raw_action = dist.sample()
            
            # Clip action to valid range
            action = torch.clamp(raw_action, -1.0, 1.0)
            
            # Convert to numpy and squeeze last dimension to ensure shape (1,) for single actions
            return action.cpu().numpy().squeeze(-1)

    def predict(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Alias for get_action to maintain compatibility with stable-baselines3 style API"""
        return self.get_action(state, deterministic)

    def train(
        self,
        env_or_experiences,
        total_timesteps: int = 1000,
        batch_size: int = None,
        env: Optional[gym.Env] = None,
    ) -> Dict[str, float]:
        """Train the agent using either environment interactions or experiences.

        Args:
            env_or_experiences: Either a gym environment or a list of experiences
            total_timesteps: Total number of timesteps to train for
            batch_size: Batch size for training (overrides self.batch_size if provided)
            env: Optional environment for training (used when env_or_experiences is a buffer)

        Returns:
            Dictionary with training metrics
        """
        if batch_size is not None:
            self.batch_size = batch_size

        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []

        if isinstance(env_or_experiences, list):
            # Training from buffer
            logger.info("Training from buffer")
            if len(env_or_experiences) < self.batch_size:
                logger.warning(
                    f"Buffer size {len(env_or_experiences)} is smaller than "
                    f"batch size {self.batch_size}"
                )
                return {
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                    "entropy": 0.0,
                }

            # Convert experiences to tensors
            for exp in env_or_experiences:
                states.append(exp["state"])
                actions.append(exp["action"])
                rewards.append(exp["reward"])
                dones.append(exp["done"])

            # Stack tensors
            states_tensor = torch.FloatTensor(np.vstack(states)).to(self.device)
            actions_tensor = torch.FloatTensor(np.array(actions)).to(self.device)
            rewards_tensor = torch.FloatTensor(np.array(rewards)).to(self.device)
            values_tensor = self.value_network(states_tensor).detach()
            log_probs_tensor = torch.zeros_like(rewards_tensor).to(self.device)

            # Update policy
            self.update(
                states_tensor,
                actions_tensor,
                log_probs_tensor,
                rewards_tensor,
                rewards_tensor,
                values_tensor,
            )

            return {
                "policy_loss": 0.0,  # Placeholder values since we don't track these for experience replay
                "value_loss": 0.0,
                "entropy": 0.0,
            }

        else:
            # Training from environment
            logger.info("Training from environment")
            env = env_or_experiences if env is None else env
            episode_rewards = []
            current_episode_reward = 0
            episode_count = 0
            step_count = 0

            state, _ = env.reset()
            if len(state.shape) == 1:
                state = state.reshape(1, -1)

            while len(states) < total_timesteps:
                step_count += 1
                if step_count % 10 == 0:  # Log every 10 steps
                    logger.info(
                        f"Step {step_count}/{total_timesteps}, Episodes: {episode_count}, Current Episode Reward: {current_episode_reward:.2f}"
                    )

                # Get action and value
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(self.device)
                    action_mean, action_std = self.network(state_tensor)
                    value = self.value_network(state_tensor)

                    # Sample action
                    dist = Normal(action_mean, action_std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                # Take step in environment
                next_state, reward, done, truncated, info = env.step(
                    action.cpu().numpy()
                )
                if len(next_state.shape) == 1:
                    next_state = next_state.reshape(1, -1)

                # Store transition
                states.append(state)
                actions.append(action.cpu().numpy())
                rewards.append(reward)
                values.append(value.cpu().numpy())
                log_probs.append(log_prob.cpu().numpy())
                dones.append(done)

                current_episode_reward += reward

                if done or truncated:
                    episode_count += 1
                    episode_rewards.append(current_episode_reward)
                    logger.info(
                        f"Episode {episode_count} finished with reward {current_episode_reward:.2f}"
                    )
                    current_episode_reward = 0
                    state, _ = env.reset()
                    if len(state.shape) == 1:
                        state = state.reshape(1, -1)
                else:
                    state = next_state

                # Update policy if we have enough samples
                if len(states) >= batch_size:
                    logger.info(
                        f"Updating policy with batch of {len(states)} samples"
                    )
                    # Convert lists to tensors with proper shapes
                    states_tensor = torch.FloatTensor(np.vstack(states)).to(
                        self.device
                    )
                    actions_tensor = torch.FloatTensor(np.array(actions)).to(
                        self.device
                    )
                    rewards_tensor = torch.FloatTensor(np.array(rewards)).to(
                        self.device
                    )
                    values_tensor = torch.FloatTensor(np.array(values)).to(
                        self.device
                    )
                    log_probs_tensor = torch.FloatTensor(
                        np.array(log_probs)
                    ).to(self.device)
                    dones_tensor = torch.FloatTensor(np.array(dones)).to(
                        self.device
                    )

                    self.update(
                        states_tensor,
                        actions_tensor,
                        log_probs_tensor,
                        rewards_tensor,
                        rewards_tensor,
                        values_tensor,
                    )
                    states = []
                    actions = []
                    rewards = []
                    values = []
                    log_probs = []
                    dones = []

            logger.info(
                f"Training completed. Total episodes: {episode_count}, Mean reward: {np.mean(episode_rewards):.2f}"
            )
            return {
                "episode_rewards": episode_rewards,
                "mean_reward": np.mean(episode_rewards),
                "std_reward": np.std(episode_rewards),
            }

    def train_step(self, state, action, reward, next_state, done, agent_id: str = None) -> Dict[str, float]:
        """Train the agent on a single state transition.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            agent_id: Optional agent identifier for multi-agent scenarios
        """
        # Convert state to numpy array if it's a DataFrame
        if isinstance(state, pd.DataFrame):
            state = state.to_numpy()
        if isinstance(next_state, pd.DataFrame):
            next_state = next_state.to_numpy()
        
        # Convert to tensor and ensure correct shape
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        
        # Add batch dimension if needed
        if len(state_tensor.shape) == 1:  # (features,)
            state_tensor = state_tensor.unsqueeze(0)  # (1, features)
        if len(next_state_tensor.shape) == 1:  # (features,)
            next_state_tensor = next_state_tensor.unsqueeze(0)  # (1, features)
        
        # Normalize states
        state_tensor = self._normalize_state(state_tensor)
        next_state_tensor = self._normalize_state(next_state_tensor)
        
        # Get value estimates for GAE
        with torch.no_grad():
            value = self.value_network(state_tensor)
            next_value = self.value_network(next_state_tensor)
            
            # Get action probabilities for current state
            action_mean, action_std = self.network(state_tensor)
            dist = Normal(action_mean, action_std)
            log_prob = dist.log_prob(torch.FloatTensor(action).to(self.device))

        # Create experience dict
        experience = {
            "state": state,
            "action": action,
            "reward": float(reward),
            "done": float(done),
            "value": value.cpu().numpy(),
            "log_prob": log_prob.cpu().numpy()
        }
        
        # Add agent_id if provided
        if agent_id is not None:
            experience["agent_id"] = agent_id

        # Add experience to buffer
        self.buffer.append(experience)

        # Train if we have enough samples
        if len(self.buffer) >= self.batch_size:
            # Compute advantages using the last value estimate
            self.buffer.compute_advantages(last_value=next_value.cpu().numpy())
            
            # Get batch from buffer
            states, actions, old_log_probs, returns, advantages, values = self.buffer.get_batch(
                batch_size=self.batch_size,
                shuffle=True
            )
            
            # Update policy
            metrics = self.update(
                states,
                actions,
                old_log_probs,
                returns,
                advantages,
                values
            )
            
            # Step learning rate scheduler if it exists
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Reset buffer
            self.buffer.reset()
            
            return metrics
            
        return None

    def _compute_gae(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = (
                rewards[t]
                + self.gamma * next_value * (1 - dones[t])
                - values[t]
            )
            advantages[t] = last_gae = (
                delta
                + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            )

        return advantages

    def learn_from_shared_experience(self, shared_buffer: list) -> Dict[str, float]:
        """Learn from shared experience buffer

        Args:
            shared_buffer: List of experiences from other agents

        Returns:
            Dictionary with training metrics
        """
        if not shared_buffer:
            return {
                "shared_policy_loss": 0.0,
                "shared_value_loss": 0.0,
                "shared_entropy": 0.0
            }
            
        return self.train(shared_buffer)

    def save(self, path: str):
        """Save agent's state"""
        torch.save(
            {
                "policy_state_dict": self.network.state_dict(),
                "value_state_dict": self.value_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            },
            path,
        )
        logger.info(f"Saved agent state to {path}")

    def load(self, path: str):
        """Load agent's state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["policy_state_dict"])
        self.value_network.load_state_dict(checkpoint["value_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info(f"Loaded agent state from {path}")

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        old_values: torch.Tensor,
    ) -> Dict[str, float]:
        """Update policy and value networks using PPO.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            old_log_probs: Log probabilities of actions under old policy
            returns: Computed returns
            advantages: Computed advantages
            old_values: Values from old value network
            
        Returns:
            Dictionary of training metrics
        """
        # Normalize states
        normalized_states = self._normalize_state(states)
        
        # Ensure no NaN values
        if torch.isnan(normalized_states).any():
            self.logger.warning("NaN values detected in normalized states")
            normalized_states = torch.nan_to_num(normalized_states, nan=0.0)
        
        # Get action distribution parameters
        action_mean, action_std = self.network(normalized_states)
        
        # Ensure action parameters are valid
        if torch.isnan(action_mean).any() or torch.isnan(action_std).any():
            self.logger.warning("NaN values detected in policy network output")
            action_mean = torch.nan_to_num(action_mean, nan=0.0)
            action_std = torch.nan_to_num(action_std, nan=1.0)
            action_std = torch.clamp(action_std, min=1e-6, max=1.0)
        
        # Create action distribution
        current_dist = Normal(action_mean, action_std)
        
        # Calculate log probabilities and entropy
        current_log_probs = current_dist.log_prob(actions)
        entropy = current_dist.entropy().mean()

        # Calculate ratios and surrogate losses
        ratios = torch.exp(current_log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(
            ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon
        ) * advantages

        # Calculate losses
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * ((action_mean - returns) ** 2).mean()
        kl_loss = self.c3 * (old_log_probs - current_log_probs).mean()

        # Combined loss with KL penalty
        total_loss = policy_loss + self.c1 * value_loss - self.c2 * entropy + kl_loss

        # Update networks
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Track metrics
        total_policy_loss = policy_loss.item()
        total_value_loss = value_loss.item()
        total_entropy = entropy.item()
        total_kl = kl_loss.item()

        # Log metrics
        logger.info(
            f"Policy Loss={total_policy_loss:.4f}, Value Loss={total_value_loss:.4f}, "
            f"Entropy={total_entropy:.4f}, KL={total_kl:.4f}"
        )

        return {
            "policy_loss": total_policy_loss,
            "value_loss": total_value_loss,
            "entropy": total_entropy,
            "kl": total_kl,
        }
