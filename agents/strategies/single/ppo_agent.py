import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, kl_divergence
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Tuple, Optional, Any
import logging
from agents.base.base_agent import BaseAgent
from agents.models.architectures.mlp import PolicyNetwork
from agents.models.architectures.value_mlp import ValueNetwork

logger = logging.getLogger(__name__)


class PPOAgent(BaseAgent):
    def __init__(
        self,
        observation_space=None,
        action_space=None,
        env=None,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        c1: float = 1.0,
        c2: float = 0.01,
        c3: float = 0.5,
        batch_size: int = 64,
        n_epochs: int = 10,
        target_kl: float = 0.015,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        PPO Agent with shared experience learning capability

        Args:
            observation_space: Gym observation space (optional if env is provided)
            action_space: Gym action space (optional if env is provided)
            env: Training environment (optional)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clip parameter
            c1: Value loss coefficient
            c2: Entropy coefficient
            c3: KL penalty coefficient
            batch_size: Mini-batch size for training
            n_epochs: Number of epochs to train on each batch
            target_kl: Target KL divergence
            device: Device to use for tensor operations
        """
        # Get spaces from env if provided
        if env is not None:
            observation_space = env.observation_space
            action_space = env.action_space

        if observation_space is None or action_space is None:
            raise ValueError(
                "Must provide either env or both observation_space and action_space"
            )

        super().__init__(observation_space, action_space)

        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

        # Initialize networks
        self.network = PolicyNetwork(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=256,
        ).to(device)

        self.value_network = ValueNetwork(
            observation_space=observation_space, hidden_size=256
        ).to(device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            [
                {"params": self.network.parameters(), "lr": learning_rate},
                {
                    "params": self.value_network.parameters(),
                    "lr": learning_rate,
                },
            ]
        )

        # Initialize scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=n_epochs,
            eta_min=learning_rate * 0.1
        )

        # Store hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.target_kl = target_kl

        # Initialize action standard deviation
        self.action_std = torch.ones(1).to(device)

        # Initialize experience buffer
        self.buffer = []

        # Initialize running statistics for normalization
        self.state_mean = None
        self.state_std = None

        logger.info(f"Initialized PPO agent on device: {device}")

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using running statistics"""
        # Ensure state is 2D for proper normalization
        original_shape = state.shape
        if len(state.shape) == 3:
            # For (batch_size, window_size, features)
            batch_size, window_size, n_features = state.shape
            state = state.reshape(-1, n_features)
        elif len(state.shape) == 1:
            state = state.reshape(1, -1)

        if self.state_mean is None:
            self.state_mean = np.mean(state, axis=0)
            self.state_std = np.std(state, axis=0) + 1e-8
        else:
            # Update running statistics with momentum
            momentum = 0.99
            batch_mean = np.mean(state, axis=0)
            batch_std = np.std(state, axis=0) + 1e-8

            # Ensure shapes match
            if self.state_mean.shape != batch_mean.shape:
                # Reinitialize statistics if shape mismatch
                self.state_mean = batch_mean
                self.state_std = batch_std
            else:
                self.state_mean = (
                    momentum * self.state_mean + (1 - momentum) * batch_mean
                )
                self.state_std = (
                    momentum * self.state_std + (1 - momentum) * batch_std
                )

        # Normalize
        normalized = (state - self.state_mean) / self.state_std

        # Restore original shape if needed
        if len(original_shape) == 3:
            normalized = normalized.reshape(original_shape)

        return normalized.astype(np.float32)

    def get_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """Get action from policy network

        Args:
            state: Current state observation
            deterministic: Whether to use deterministic action

        Returns:
            Action as numpy array with shape (1,)
        """
        with torch.no_grad():
            # Flatten and normalize state
            if len(state.shape) > 1:
                state = state.reshape(-1)
            state = self._normalize_state(state)

            state_tensor = (
                torch.FloatTensor(state).reshape(1, -1).to(self.device)
            )
            action_mean, action_std = self.network(state_tensor)

            if deterministic:
                action = action_mean
            else:
                dist = Normal(action_mean, action_std)
                action = dist.sample()

            # Ensure action is a numpy array with shape (1,)
            action = action.clamp(-1, 1).cpu().numpy()
            return action.reshape(1)

    def train(
        self,
        env_or_experiences,
        total_timesteps: int = 1000,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """Train the agent in batch mode

        Args:
            env_or_experiences: Either a gym environment or a list of experiences
            total_timesteps: Number of timesteps to train for (if env provided)
            batch_size: Size of batch for updates

        Returns:
            Dictionary with training metrics
        """
        logger.info(
            f"Starting training for {total_timesteps} timesteps with batch size {batch_size}"
        )
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []

        # Check if we're training from experiences or environment
        if isinstance(env_or_experiences, list):
            logger.info("Training from experiences")
            # Training from experiences
            if not env_or_experiences:
                return {
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                    "entropy": 0.0
                }
                
            for exp in env_or_experiences:
                state = self._normalize_state(exp["state"])
                states.append(state)
                actions.append(exp["action"])
                rewards.append(exp["reward"])

                # Get value and log prob for the state
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(self.device)
                    value = self.value_network(state_tensor)
                    action_mean, action_std = self.network(state_tensor)
                    
                    # Check for NaN values
                    if torch.isnan(action_mean).any() or torch.isnan(action_std).any():
                        logger.warning("NaN values detected in policy network output")
                        continue
                    
                    # Ensure positive standard deviation
                    action_std = torch.clamp(action_std, min=1e-6)
                    
                    dist = Normal(action_mean, action_std)
                    log_prob = dist.log_prob(
                        torch.FloatTensor([exp["action"]]).to(self.device)
                    )

                values.append(value.cpu().numpy())
                log_probs.append(log_prob.cpu().numpy())
                dones.append(exp.get("done", False))
            
            # Skip update if we don't have enough valid experiences
            if len(states) < 2:
                return {
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                    "entropy": 0.0
                }

            # Convert to tensors and update
            states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
            actions_tensor = torch.FloatTensor(np.array(actions)).to(
                self.device
            )
            rewards_tensor = torch.FloatTensor(np.array(rewards)).to(
                self.device
            )
            values_tensor = torch.FloatTensor(np.array(values)).to(self.device)
            log_probs_tensor = torch.FloatTensor(np.array(log_probs)).to(
                self.device
            )
            dones_tensor = torch.FloatTensor(np.array(dones)).to(self.device)

            self.update(
                states_tensor,
                actions_tensor,
                rewards_tensor,
                values_tensor,
                log_probs_tensor,
                dones_tensor,
            )

            return {
                "policy_loss": 0.0,  # Placeholder values since we don't track these for experience replay
                "value_loss": 0.0,
                "entropy": 0.0,
            }

        else:
            # Training from environment
            logger.info("Training from environment")
            env = env_or_experiences
            episode_rewards = []
            current_episode_reward = 0
            episode_count = 0
            step_count = 0

            state, _ = env.reset()
            state = self._normalize_state(state)

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
                    
                    # Check for NaN values
                    if torch.isnan(action_mean).any() or torch.isnan(action_std).any():
                        logger.warning("NaN values detected in policy network output")
                        state, _ = env.reset()
                        state = self._normalize_state(state)
                        continue
                    
                    # Ensure positive standard deviation
                    action_std = torch.clamp(action_std, min=1e-6)
                    
                    value = self.value_network(state_tensor)

                    # Sample action
                    dist = Normal(action_mean, action_std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                # Take step in environment
                next_state, reward, done, truncated, info = env.step(
                    action.cpu().numpy()
                )
                next_state = self._normalize_state(next_state)

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
                    state = self._normalize_state(state)
                else:
                    state = next_state

                # Update policy if we have enough samples
                if len(states) >= batch_size:
                    logger.info(
                        f"Updating policy with batch of {len(states)} samples"
                    )
                    # Convert lists to tensors with proper shapes
                    states_tensor = torch.FloatTensor(np.array(states)).to(
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
                        rewards_tensor,
                        values_tensor,
                        log_probs_tensor,
                        dones_tensor,
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

    def train_step(self, state, action, reward, next_state, done) -> None:
        """Train the agent on a single state transition.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Store experience
        self.buffer.append(
            {"state": state, "action": action, "reward": reward, "done": done}
        )

        # Train if we have enough samples
        if len(self.buffer) >= self.batch_size:
            self.train(self.buffer)
            self.buffer = []  # Clear buffer after training

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
        logger.info(f"Loaded agent state from {path}")

    def update(self, states, actions, rewards, values, log_probs, dones):
        """Update policy and value networks using PPO"""
        # Compute returns and advantages
        advantages = self._compute_gae(rewards, values, dones)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors if not already
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states).to(self.device)
        if not isinstance(actions, torch.Tensor):
            actions = torch.FloatTensor(actions).to(self.device)
        if not isinstance(returns, torch.Tensor):
            returns = torch.FloatTensor(returns).to(self.device)
        if not isinstance(advantages, torch.Tensor):
            advantages = torch.FloatTensor(advantages).to(self.device)
        if not isinstance(log_probs, torch.Tensor):
            log_probs = torch.FloatTensor(log_probs).to(self.device)

        # Get initial policy distribution for KL tracking
        with torch.no_grad():
            init_mean, init_std = self.network(states)
            init_dist = Normal(init_mean, init_std)

        # Mini-batch updates
        batch_size = len(states)
        if batch_size < 4:  # If batch is too small, use the entire batch
            mini_batch_size = batch_size
        else:
            mini_batch_size = batch_size // 4  # Use 4 mini-batches

        for epoch in range(self.n_epochs):
            # Generate random permutation
            indices = torch.randperm(batch_size)
            total_kl = 0
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0
            n_updates = 0

            # Mini-batch updates
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_indices = indices[start:end]

                # Get mini-batch data
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_old_log_probs = log_probs[mb_indices]

                # Get current policy outputs
                action_mean, action_std = self.network(mb_states)
                current_values = self.value_network(mb_states)

                # Calculate distributions
                current_dist = Normal(action_mean, action_std)
                
                # Calculate KL divergence with initial policy
                with torch.no_grad():
                    mb_init_mean = init_mean[mb_indices]
                    mb_init_std = init_std[mb_indices]
                    mb_init_dist = Normal(mb_init_mean, mb_init_std)
                    kl = torch.distributions.kl.kl_divergence(
                        mb_init_dist, current_dist
                    ).mean()
                    total_kl += kl.item()

                # Early stopping if KL is too high
                if kl > self.target_kl * 1.5:
                    logger.info(f"Early stopping at epoch {epoch} due to high KL: {kl:.4f}")
                    self.scheduler.step(epoch)  # Update scheduler with current epoch
                    return

                # Calculate log probabilities and entropy
                current_log_probs = current_dist.log_prob(mb_actions)
                entropy = current_dist.entropy().mean()

                # Calculate ratios and surrogate losses
                ratios = torch.exp(current_log_probs - mb_old_log_probs)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(
                    ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                ) * mb_advantages

                # Calculate losses
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((current_values - mb_returns) ** 2).mean()
                kl_loss = self.c3 * kl

                # Combined loss with KL penalty
                total_loss = policy_loss + self.c1 * value_loss - self.c2 * entropy + kl_loss

                # Update networks
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
                self.optimizer.step()

                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

            # Calculate averages
            avg_kl = total_kl / n_updates
            avg_policy_loss = total_policy_loss / n_updates
            avg_value_loss = total_value_loss / n_updates
            avg_entropy = total_entropy / n_updates

            # Log metrics
            logger.info(
                f"Epoch {epoch}: KL={avg_kl:.4f}, Policy Loss={avg_policy_loss:.4f}, "
                f"Value Loss={avg_value_loss:.4f}, Entropy={avg_entropy:.4f}"
            )

            # Early stopping if average KL is too high
            if avg_kl > self.target_kl:
                logger.info(f"Early stopping training due to high average KL: {avg_kl:.4f}")
                self.scheduler.step(epoch)  # Update scheduler with current epoch
                return

            # Step the scheduler
            self.scheduler.step()
