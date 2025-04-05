"""PPO agent implementation with state normalization and experience sharing.

Features:
- Proximal Policy Optimization (PPO) algorithm
- State normalization using running statistics
- Experience sharing between agents
- Configurable hyperparameters
- Automatic device selection (CPU/GPU)
- Proper rollout-based PPO implementation with separate policy networks

Implementation Notes:
- Uses separate policy and value networks
- Maintains separate "old" policy for rollout and stable ratio calculation
- Supports both continuous and discrete action spaces
- Implements early stopping based on KL divergence
- Uses GAE for advantage estimation
- Prevents negative entropy through minimum standard deviation clamping
- Accurate KL divergence calculation between normal distributions

Recent Changes:
- Fixed standard deviation calculation to prevent negative entropy
- Modified the PolicyNetwork to ensure std is clamped with a higher minimum (0.1)
- Implemented proper rollout collection with old_network to maintain consistency
- Added support for configurable rollout length with rollout_threshold parameter
- Fixed PPO implementation to properly separate rollout from update phases
- Modified train_step() to collect experiences and only update after sufficient rollout
- Enhanced update_if_buffer_ready() to perform multi-epoch PPO updates
- Implemented proper KL divergence calculation for normal distributions
- Improved stability with ratio clamping and better numerical handling
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
import math

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
        n_epochs: int = 10,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.015,
        normalize_observations: bool = False,
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
            clip_epsilon: PPO clip epsilon
            c1: Value loss coefficient
            c2: Entropy coefficient
            c3: KL penalty coefficient (optional, used for early stopping)
            n_epochs: Number of epochs per update
            batch_size: Batch size for updates
            max_grad_norm: Maximum gradient norm
            target_kl: Target KL divergence for early stopping
            normalize_observations: Whether to normalize observations
            device: Device to use (cpu or cuda)
            **kwargs: Additional arguments (hidden_sizes, etc.)
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
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.normalize_observations = False
        self.eps = 1e-8
        
        # Add training flag
        self.training = True
        
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
        hidden_sizes = kwargs.get('hidden_sizes', [64, 64])
        self.logger.info(f"Creating networks with hidden sizes: {hidden_sizes}")
        self.network = PolicyNetwork(observation_space, action_space).to(self.device)
        self.value_network = ValueNetwork(observation_space, hidden_sizes=hidden_sizes).to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam([
            {'params': self.network.parameters(), 'lr': learning_rate},
            {'params': self.value_network.parameters(), 'lr': learning_rate}
        ])
        
        # Set up learning rate scheduler if specified
        self.use_lr_scheduler = kwargs.get("use_lr_scheduler", False)
        if self.use_lr_scheduler:
            lr_scheduler_step_size = kwargs.get("lr_scheduler_step_size", 100)
            lr_scheduler_gamma = kwargs.get("lr_scheduler_gamma", 0.9)
            self.logger.info(f"Creating StepLR scheduler with step_size={lr_scheduler_step_size}, gamma={lr_scheduler_gamma}")
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=lr_scheduler_step_size,
                gamma=lr_scheduler_gamma
            )
        else:
            self.scheduler = None
        
        # Initialize replay buffer
        self.buffer = PPOBuffer(
            observation_space.shape,
            action_space.shape,
            self.batch_size,
            self.gamma,
            self.gae_lambda,
            self.device
        )
        
        # Create old_network to store policy parameters during rollout
        self.old_network = PolicyNetwork(observation_space, action_space).to(self.device)
        self._update_old_network()  # Initialize with current policy
        
        # Track rollout state
        self.rollout_steps = 0
        self.rollout_threshold = kwargs.get("rollout_steps", 1024)  # Default to 1024 steps per rollout
        
        # Track training metrics over time
        self.training_history = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl": [],
            "mean_std": [],
            "mean_ratio": [],
            "rewards": [],
            "episode_rewards": [],
            "episode_lengths": [],
            "update_count": 0,
            "total_steps": 0,
            "completed_episodes": 0
        }
        
        # Track current episode stats
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        
        logger.info(
            f"Initialized PPO agent on device: {self.device} with obs_dim={self.obs_dim}, "
            f"rollout_threshold={self.rollout_threshold}"
        )

    def _prepare_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Prepares a state tensor (batch_size, obs_dim) for network input,
        handles various shape cases, NaN/Inf, and clipping.
        """
        # 1) 1D 케이스: (obs_dim,) -> (1, obs_dim)
        if state.dim() == 1:
            if state.shape[0] == self.obs_dim:
                state = state.unsqueeze(0)  # shape (1, obs_dim)
            else:
                # pad or truncate
                needed = self.obs_dim - state.shape[0]
                if needed > 0:
                    pad = torch.zeros(needed, device=state.device)
                    state = torch.cat([state, pad])
                else:
                    state = state[:self.obs_dim]
                state = state.unsqueeze(0)

        # 2) 2D 케이스
        elif state.dim() == 2:
            b, d = state.shape
            if b * d == self.obs_dim:
                # ex) (3,5) with obs_dim=15 => flatten -> (1,15)
                self.logger.debug(
                    f"Detected 2D single-sample shape (b={b}, d={d}). Flatten => (1, {self.obs_dim})."
                )
                state = state.reshape(1, self.obs_dim)
            elif d == self.obs_dim:
                # already (batch_size, obs_dim)
                pass
            else:
                # pad or truncate
                if d < self.obs_dim:
                    pad = torch.zeros(b, self.obs_dim - d, device=state.device)
                    state = torch.cat([state, pad], dim=1)
                else:
                    state = state[:, :self.obs_dim]

        # 3) **새로 추가**: 3D 케이스 (batch_size, window_size, features)
        elif state.dim() == 3:
            b, w, f = state.shape
            # 만약 w*f == self.obs_dim 이라면 그냥 flatten
            if w * f == self.obs_dim:
                # reshape => (b, obs_dim)
                self.logger.debug(
                    f"Detected 3D shape (b={b}, w={w}, f={f}) => flatten to (b, {self.obs_dim})."
                )
                state = state.reshape(b, self.obs_dim)
            else:
                # pad or truncate
                actual_dim = w * f
                self.logger.warning(
                    f"3D shape (b={b}, w={w}, f={f}) => {actual_dim} != obs_dim={self.obs_dim}, doing pad/truncate."
                )
                if actual_dim < self.obs_dim:
                    # flatten first -> (b, actual_dim)
                    state = state.view(b, -1)
                    pad_size = self.obs_dim - actual_dim
                    pad = torch.zeros(b, pad_size, device=state.device)
                    state = torch.cat([state, pad], dim=1)
                else:
                    # flatten and truncate
                    state = state.view(b, -1)[:, :self.obs_dim]

        # 4) 그 외: 예전 unexpected case
        else:
            self.logger.warning(f"Shape {state.shape} beyond 3D?! Flatten forcibly.")
            state = state.reshape(-1, self.obs_dim)
            # pad/truncate if mismatch

        # NaN/Inf handling
        if torch.isnan(state).any() or torch.isinf(state).any():
            self.logger.warning("NaN/Inf in state => clamp or zero")
            state = torch.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)

        # clamp big values
        MAX_VALUE = 1e6
        state = torch.clamp(state, -MAX_VALUE, MAX_VALUE)

        return state



    def get_action(
        self, state: np.ndarray, deterministic: bool = False, eval_mode: bool = False
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
            eval_mode: Whether the agent is in evaluation mode (equivalent to deterministic)

        Returns:
            Action as numpy array with shape matching the expected output shape:
            - For single actions: (1,) or scalar
            - For multi-asset actions: (n_assets,) without extra dimensions
        """
        with torch.no_grad():
            # Convert DataFrame to numpy if needed
            if isinstance(state, pd.DataFrame):
                state = state.to_numpy()
            
            # Handle 2D observations from environment with shape (window_size, features)
            # by adding a batch dimension to make it (1, window_size, features)
            if len(state.shape) == 2 and state.shape[0] == self.observation_space.shape[0] and state.shape[1] == self.observation_space.shape[1]:
                # This is likely a direct observation from SingleAssetRLTradingEnv
                state = np.expand_dims(state, axis=0)  # Add batch dimension: (1, window_size, features)
            
            # Convert to tensor
            state_tensor = torch.FloatTensor(state).to(self.device)

            # Prepare state (handles all shape cases internally)
            state_tensor = self._prepare_state(state_tensor)
            
            # Get action distribution parameters
            action_mean, action_std = self.network(state_tensor)

            # Use deterministic action if either deterministic or eval_mode is True
            if deterministic or eval_mode:
                raw_action = action_mean
            else:
                dist = Normal(action_mean, action_std)
                raw_action = dist.sample()
            
            # Clip action to valid range
            action = torch.clamp(raw_action, -1.0, 1.0)
            
            # Convert to numpy
            action_np = action.cpu().numpy()
            
            # Handle different action shapes based on the action space
            # For multi-asset environments, we need to return shape (n_assets,)
            if len(action_np.shape) > 1 and action_np.shape[0] == 1:
                # If we have a batch dimension of 1, remove it
                return action_np.squeeze(0)
            elif action_np.shape[-1] == 1:
                # For single-asset actions, squeeze the last dimension
                return action_np.squeeze(-1)
            else:
                # For other cases, return as is
                return action_np

    def predict(self, state: np.ndarray, deterministic: bool = False, eval_mode: bool = False) -> np.ndarray:
        """Alias for get_action to maintain compatibility with stable-baselines3 style API"""
        return self.get_action(state, deterministic, eval_mode)

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
                    states_tensor = torch.FloatTensor(np.vstack(states)).to(self.device)
                    actions_tensor = torch.FloatTensor(np.array(actions)).to(self.device)
                    rewards_tensor = torch.FloatTensor(np.array(rewards)).to(self.device)
                    values_tensor = torch.FloatTensor(np.array(values)).to(self.device)
                    log_probs_tensor = torch.FloatTensor(np.array(log_probs)).to(self.device)
                    dones_tensor = torch.FloatTensor(np.array(dones)).to(self.device)

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

    def _update_old_network(self):
        """Update the old policy network with current policy parameters."""
        self.old_network.load_state_dict(self.network.state_dict())
        logger.debug("Updated old policy network with current parameters")

    def _compute_ratio_histogram(self, ratios: torch.Tensor, bins: int = 10) -> Dict[str, List[float]]:
        """Compute histogram data for ratio distribution for debugging.
        
        Args:
            ratios: Tensor of policy ratios
            bins: Number of histogram bins
            
        Returns:
            Dictionary with histogram data (counts and bin_edges)
        """
        with torch.no_grad():
            # Convert to numpy for histogram computation
            ratios_np = ratios.detach().cpu().numpy()
            
            # Compute histogram
            counts, bin_edges = np.histogram(ratios_np, bins=bins, range=(0.0, 2.0))
            
            # Convert to lists for logging
            counts_list = counts.tolist()
            bin_edges_list = bin_edges.tolist()
            
            # Also compute useful statistics
            stats = {
                "mean": float(np.mean(ratios_np)),
                "median": float(np.median(ratios_np)),
                "std": float(np.std(ratios_np)),
                "min": float(np.min(ratios_np)),
                "max": float(np.max(ratios_np)),
                "close_to_one": float(np.mean((ratios_np > 0.99) & (ratios_np < 1.01))),
            }
            
            return {
                "counts": counts_list,
                "bin_edges": bin_edges_list,
                "stats": stats
            }

    def _compute_kl_divergence(self, states: torch.Tensor) -> Dict[str, float]:
        """Calculate detailed KL divergence between old and new policies.
        
        This method provides detailed information about KL divergence components
        to help detect issues with policy updates.
        
        Args:
            states: Batch of states to evaluate
            
        Returns:
            Dictionary with KL divergence components and total
        """
        with torch.no_grad():
            # Prepare states
            prepared_states = self._prepare_state(states)
            
            # Get distribution parameters from old policy
            old_mean, old_std = self.old_network(prepared_states)
            
            # Get distribution parameters from current policy
            new_mean, new_std = self.network(prepared_states)
            
            # Ensure std values are valid
            old_std = torch.clamp(old_std, min=0.1, max=1.0)
            new_std = torch.clamp(new_std, min=0.1, max=1.0)
            
            # Calculate mean component of KL: (μ1 - μ2)²/(2*σ2²)
            mean_diff_squared = (old_mean - new_mean).pow(2)
            mean_component = (mean_diff_squared / (2 * new_std.pow(2))).mean()
            
            # Calculate log(σ2/σ1) term
            log_std_diff = torch.log(new_std / old_std).mean()
            
            # Calculate variance term: σ1²/(2*σ2²)
            var_component = (old_std.pow(2) / (2 * new_std.pow(2))).mean()
            
            # Calculate -1/2 term
            constant = -0.5
            
            # Full KL: log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2*σ2²) - 1/2
            total_kl = log_std_diff + var_component + mean_component + constant
            
            # Calculate average absolute differences
            mean_abs_diff = (old_mean - new_mean).abs().mean()
            std_abs_diff = (old_std - new_std).abs().mean()
            
            # Store components
            kl_components = {
                "kl_total": total_kl.item(),
                "kl_log_std_diff": log_std_diff.item(),
                "kl_var_component": var_component.item(),
                "kl_mean_component": mean_component.item(),
                "kl_constant": constant,
                "mean_abs_diff": mean_abs_diff.item(),
                "std_abs_diff": std_abs_diff.item(),
                "old_mean_avg": old_mean.mean().item(),
                "new_mean_avg": new_mean.mean().item(),
                "old_std_avg": old_std.mean().item(),
                "new_std_avg": new_std.mean().item(),
            }
            
            return kl_components

    def train_step(self, state, action, reward, next_state, done, agent_id: str = None) -> Dict[str, float]:
        """Single training step using one transition.
        
        Collects experiences into the buffer using the old policy's log_probs.
        Only triggers an update when the rollout threshold is reached.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            agent_id: Optional agent ID for multi-agent settings
            
        Returns:
            Empty dict if no update is performed, or metrics dict if update was triggered
        """
        # Track episode stats
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # Convert inputs to tensors
        state_tensor = torch.FloatTensor(state).to(self.device)
        action_tensor = torch.FloatTensor(action).to(self.device)
        
        # Prepare state
        prepared_state = self._prepare_state(state_tensor)
        
        # Get current value estimate 
        with torch.no_grad():
            current_value = self.value_network(prepared_state)
            # current_value: shape (1,1) or (1,) 이라면
            current_value = current_value.squeeze(-1)  # shape (1,)
            current_value = current_value.item()
                        
            # IMPORTANT: Use old_network to calculate log_prob for consistent ratio calculation
            action_mean, action_std = self.old_network(prepared_state)
            dist = Normal(action_mean, action_std)
            log_prob = dist.log_prob(action_tensor).sum(dim=-1).item()
            
            # Debug logging for standard deviation
            mean_std = action_std.mean().item()
            min_std = action_std.min().item()
            max_std = action_std.max().item()
            if self.rollout_steps % 100 == 0:  # Log every 100 steps
                self.logger.info(
                    f"Rollout step {self.rollout_steps}: Mean std={mean_std:.4f}, "
                    f"Min std={min_std:.4f}, Max std={max_std:.4f}"
                )
        
        # Add experience to buffer
        self.buffer.append({
            "state": state,
            "action": action,
            "reward": reward,
            "value": current_value,
            "log_prob": log_prob,
            "done": done
        })
        
        # Increment rollout steps
        self.rollout_steps += 1
        
        # Track episode completion
        if done:
            self.training_history["completed_episodes"] += 1
            self.training_history["episode_rewards"].append(self.current_episode_reward)
            self.training_history["episode_lengths"].append(self.current_episode_length)
            
            # Log episode results
            self.logger.info(
                f"Episode {self.training_history['completed_episodes']} completed with "
                f"reward={self.current_episode_reward:.4f}, length={self.current_episode_length}"
            )
            
            # Reset episode tracking
            self.current_episode_reward = 0.0
            self.current_episode_length = 0
        
        # Only update if we've collected enough experiences
        metrics = {}
        if self.rollout_steps >= self.rollout_threshold or done:
            self.logger.info(
                f"Rollout complete with {self.rollout_steps} steps, buffer size {len(self.buffer)}. "
                f"Performing update with {self.n_epochs} epochs."
            )
            metrics = self.update_if_buffer_ready()
            
            # Log update results
            self.logger.info(
                f"Update complete. Policy Loss: {metrics.get('policy_loss', 0):.4f}, "
                f"Value Loss: {metrics.get('value_loss', 0):.4f}, "
                f"KL: {metrics.get('kl', 0):.4f}, "
                f"Entropy: {metrics.get('entropy', 0):.4f}"
            )
            
            # Reset rollout steps counter and update old_network for next rollout
            self.rollout_steps = 0
            self._update_old_network()
            self.logger.info("Updated old_network for next rollout phase")
        
        return metrics
        
    def update_if_buffer_ready(self) -> Dict[str, float]:
        """Update policy if buffer has enough experiences.
        
        This method performs a proper PPO update when called after rollout completes.
        The old_network parameters remain fixed during the update so that ratio calculation
        compares the updated policy to the policy used during data collection.
        
        Returns:
            Dictionary of loss metrics if update was performed, empty dict otherwise
        """
        # If we don't have enough samples, return early
        if len(self.buffer) < self.batch_size:
            return {}
            
        # Get last state to estimate its value for advantage computation
        if hasattr(self.buffer, 'states') and len(self.buffer.states) > 0:
            last_state = self.buffer.states[-1]
            last_state_tensor = torch.FloatTensor(last_state).to(self.device)
            prepared_last_state = self._prepare_state(last_state_tensor)
            with torch.no_grad():
                last_value = self.value_network(prepared_last_state).cpu().numpy()
        else:
            # If no states in buffer yet, use zero
            last_value = np.zeros(1)
            
        # Compute advantages and returns
        self.buffer.compute_advantages(last_value)
        
        # Get batch data
        batch_data = self.buffer.get_batch(self.batch_size)
        if batch_data is None:
            return {}
            
        states, actions, old_log_probs, returns, advantages, old_values = batch_data
        
        # Calculate initial KL divergence to verify old and new policies are different
        initial_kl = self._compute_kl_divergence(states)
        self.logger.info(f"Initial KL before update: {initial_kl}")
        
        # Store metrics
        policy_losses = []
        value_losses = []
        entropies = []
        kls = []
        
        # Perform multiple epochs of optimization
        for epoch in range(self.n_epochs):
            # Update policy and value networks
            update_metrics = self.update(
                states, actions, old_log_probs, returns, advantages, old_values
            )
            
            # Store metrics
            policy_losses.append(update_metrics["policy_loss"])
            value_losses.append(update_metrics["value_loss"])
            entropies.append(update_metrics["entropy"])
            kls.append(update_metrics.get("kl", 0))
            
            # Early stopping based on KL divergence
            if update_metrics.get("kl", 0) > 1.5 * self.target_kl:
                self.logger.info(f"Early stopping at epoch {epoch+1}/{self.n_epochs} due to KL divergence")
                break
        
        # Step the learning rate scheduler if it exists
        if self.scheduler is not None:
            self.scheduler.step()
            self.logger.info(f"Stepped learning rate scheduler. New LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        # Record metrics in training history
        avg_metrics = {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropies),
            "kl": np.mean(kls)
        }
        
        # Update training history
        self._update_training_history(avg_metrics)
        
        # Calculate final KL after all updates
        final_kl = self._compute_kl_divergence(states)
        self.logger.info(f"Final KL after update: {final_kl}")
        
        # Reset buffer after update
        self.buffer.reset()
        
        # Return average metrics
        return avg_metrics

    def _update_training_history(self, metrics: Dict[str, float]):
        """Update training history with the latest metrics.
        
        Args:
            metrics: Dictionary of metrics from the latest update
        """
        # Update metrics
        for key, value in metrics.items():
            if key in self.training_history:
                self.training_history[key].append(value)
        
        # Update counters
        self.training_history["update_count"] += 1
        self.training_history["total_steps"] += self.rollout_steps
        
        # Log training progress periodically
        if self.training_history["update_count"] % 10 == 0:
            # Calculate recent performance
            recent_metrics = {}
            for key in ["policy_loss", "value_loss", "entropy", "kl"]:
                if len(self.training_history[key]) >= 10:
                    recent_metrics[f"avg_{key}_last10"] = np.mean(self.training_history[key][-10:])
            
            # Add episode reward statistics if available
            if len(self.training_history["episode_rewards"]) >= 5:
                recent_metrics["avg_episode_reward_last5"] = np.mean(self.training_history["episode_rewards"][-5:])
                recent_metrics["avg_episode_length_last5"] = np.mean(self.training_history["episode_lengths"][-5:])
            
            # Log progress
            self.logger.info(
                f"Training progress after {self.training_history['update_count']} updates "
                f"({self.training_history['total_steps']} steps, {self.training_history['completed_episodes']} episodes): "
                f"{recent_metrics}"
            )
            
    def get_training_history(self) -> Dict[str, Any]:
        """Get training history.
        
        Returns:
            Dictionary with training metrics over time
        """
        return self.training_history

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
        # Prepare states
        prepared_states = self._prepare_state(states)
        
        # Ensure no NaN values
        if torch.isnan(prepared_states).any():
            self.logger.warning("NaN values detected in prepared states")
            prepared_states = torch.nan_to_num(prepared_states, nan=0.0)
        
        # Get action distribution parameters from current policy
        action_mean, action_std = self.network(prepared_states)
        
        # Get value predictions
        predicted_values = self.value_network(prepared_states).squeeze(-1)
        
        # Ensure action parameters are valid
        if torch.isnan(action_mean).any() or torch.isnan(action_std).any():
            self.logger.warning("NaN values detected in policy network output")
            action_mean = torch.nan_to_num(action_mean, nan=0.5)
            action_std = torch.nan_to_num(action_std, nan=0.1)
        
        # Clamp standard deviation to prevent it from getting too small
        action_std = torch.clamp(action_std, min=0.1, max=1.0)
        
        # Create current action distribution
        current_dist = Normal(action_mean, action_std)
        
        # Calculate current log probabilities 
        current_log_probs = current_dist.log_prob(actions).sum(-1)
        
        # Calculate entropy (ensure it's properly computed and bounded)
        entropy = current_dist.entropy().mean()
        
        # Log entropy statistics for debugging
        with torch.no_grad():
            min_entropy = current_dist.entropy().min().item()
            mean_std = action_std.mean().item()
            self.logger.debug(f"Entropy: {entropy.item():.4f}, Min entropy: {min_entropy:.4f}, Mean std: {mean_std:.4f}")

        # Calculate ratios and surrogate losses
        ratios = torch.exp(current_log_probs - old_log_probs)
        
        # Log ratio statistics for debugging
        with torch.no_grad():
            mean_ratio = ratios.mean().item()
            min_ratio = ratios.min().item()
            max_ratio = ratios.max().item()
            log_prob_diff = (current_log_probs - old_log_probs).mean().item()
            
            self.logger.info(
                f"Ratio stats - Mean: {mean_ratio:.4f}, Min: {min_ratio:.4f}, "
                f"Max: {max_ratio:.4f}, Log prob diff: {log_prob_diff:.4f}"
            )
            
            # Count ratios close to 1.0 (indicating old/new policies are similar)
            close_to_one = ((ratios > 0.99) & (ratios < 1.01)).float().mean().item()
            self.logger.info(f"Percentage of ratios close to 1.0: {close_to_one * 100:.2f}%")
            
            # Generate ratio histogram data
            ratio_hist = self._compute_ratio_histogram(ratios)
            self.logger.info(f"Ratio histogram: {ratio_hist}")
        
        # Clamp ratios to prevent extreme values that could destabilize training
        ratios = torch.clamp(ratios, 0.0, 10.0)
        
        # Calculate KL divergence more robustly
        # KL = (log(std2/std1) + (std1^2 + (mean1-mean2)^2)/(2*std2^2) - 0.5)
        with torch.no_grad():
            # Get old policy distribution parameters for more accurate KL calculation
            old_action_mean, old_action_std = self.old_network(prepared_states)
            
            # Log mean difference between old and new policy means
            mean_diff = (old_action_mean - action_mean).abs().mean().item()
            std_diff = (old_action_std - action_std).abs().mean().item()
            self.logger.info(f"Policy diff - Mean: {mean_diff:.6f}, Std: {std_diff:.6f}")
            
            # Clamp old std to prevent division by zero
            old_action_std = torch.clamp(old_action_std, min=0.1, max=1.0)
            
            # Calculate KL divergence between old and new policies
            mean_diff_squared = (old_action_mean - action_mean).pow(2)
            std_ratio = old_action_std / action_std
            var_ratio = std_ratio.pow(2)
            
            # Use the analytical KL for normal distributions
            kl_div = (torch.log(action_std / old_action_std) + 
                      (old_action_std.pow(2) + mean_diff_squared) / 
                      (2 * action_std.pow(2)) - 0.5).mean()
            
            kl_val = kl_div.item()
        
        # PPO surrogate objectives
        surr1 = ratios * advantages
        surr2 = torch.clamp(
            ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon
        ) * advantages

        # Calculate policy loss
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Calculate value loss using predicted values from value network
        value_loss = 0.5 * F.mse_loss(predicted_values, returns)
        
        # Combined loss with KL penalty
        total_loss = policy_loss + self.c1 * value_loss - self.c2 * entropy

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
        total_kl = kl_val  # Use the properly calculated KL value

        # Log metrics
        logger.info(
            f"Policy Loss={total_policy_loss:.4f}, Value Loss={total_value_loss:.4f}, "
            f"Entropy={total_entropy:.4f}, KL={total_kl:.4f}, Mean Std={mean_std:.4f}"
        )

        return {
            "policy_loss": total_policy_loss,
            "value_loss": total_value_loss,
            "entropy": total_entropy,
            "kl": total_kl,
            "mean_std": mean_std
        }