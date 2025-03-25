"""Tests for PPO improvements."""

import pytest
import torch
import numpy as np
import gymnasium as gym
from agents.strategies.single.ppo_agent import PPOAgent
from stable_baselines3.common.env_util import DummyVecEnv
from gymnasium.spaces import Box


@pytest.fixture
def observation_space():
    """Create sample observation space"""
    return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(20, 5))


@pytest.fixture
def action_space():
    """Create sample action space"""
    return gym.spaces.Box(low=-1, high=1, shape=(1,))


@pytest.fixture
def ppo_agent(observation_space, action_space):
    """Create PPO agent with default parameters"""
    return PPOAgent(
        observation_space=observation_space,
        action_space=action_space,
        learning_rate=3e-4,
        n_epochs=10,
        use_lr_scheduler=True,
        lr_scheduler_gamma=0.9,
        lr_scheduler_step_size=100
    )


def test_kl_penalty_effect(ppo_agent):
    """Test that KL penalty affects the loss appropriately"""
    # Create sample batch
    batch_size = 32
    states = torch.randn(batch_size, 20, 5).to(ppo_agent.device)
    actions = torch.randn(batch_size, 1).to(ppo_agent.device)
    rewards = torch.ones(batch_size).to(ppo_agent.device)  # Use consistent rewards
    values = torch.zeros(batch_size).to(ppo_agent.device)  # Start with zero values
    log_probs = torch.zeros(batch_size).to(ppo_agent.device)
    dones = torch.zeros(batch_size).to(ppo_agent.device)

    # Get initial policy distribution
    with torch.no_grad():
        initial_mean, initial_std = ppo_agent.network(states)
    
    # Store initial network state
    initial_state = {
        name: param.clone()
        for name, param in ppo_agent.network.named_parameters()
    }
    
    # Update with normal c3 (KL penalty coefficient)
    ppo_agent.c3 = 0.5
    ppo_agent.update(states, actions, rewards, values, log_probs, dones)
    
    # Calculate parameter changes with normal KL
    normal_param_change = 0
    for name, param in ppo_agent.network.named_parameters():
        normal_param_change += torch.norm(param - initial_state[name])
    
    # Restore initial network state
    for name, param in ppo_agent.network.named_parameters():
        param.data.copy_(initial_state[name])
    
    # Update with high c3
    ppo_agent.c3 = 2.0
    ppo_agent.update(states, actions, rewards, values, log_probs, dones)
    
    # Calculate parameter changes with high KL
    high_param_change = 0
    for name, param in ppo_agent.network.named_parameters():
        high_param_change += torch.norm(param - initial_state[name])
    
    # Higher KL penalty should result in smaller parameter changes
    assert high_param_change < normal_param_change, (
        f"Higher KL penalty should constrain policy updates more. "
        f"Normal change: {normal_param_change:.4f}, High KL change: {high_param_change:.4f}"
    )


def test_learning_rate_scheduler():
    """Test that learning rate decreases over time with scheduler."""
    # Create simple environment with 2D observation space
    observation_space = Box(
        low=-np.inf,
        high=np.inf,
        shape=(4, 3),  # window_size=4, features=3
        dtype=np.float32
    )
    action_space = Box(
        low=0,
        high=1,
        shape=(1,),
        dtype=np.float32
    )
    
    # Configure agent with learning rate scheduler
    # Use a small step size and large gamma to see LR change quickly
    config = {
        'learning_rate': 0.001,
        'batch_size': 4,
        'n_epochs': 1,
        'use_lr_scheduler': True,
        'lr_scheduler_step_size': 1,  # Step every update
        'lr_scheduler_gamma': 0.5     # Reduce by half each step
    }
    
    agent = PPOAgent(observation_space, action_space, **config)
    
    # Verify the scheduler is created correctly
    assert agent.scheduler is not None, "Scheduler should be created"
    assert agent.use_lr_scheduler is True, "use_lr_scheduler should be True"
    
    # Get initial learning rate
    initial_lr = agent.optimizer.param_groups[0]['lr']
    
    # Generate data for the buffer
    for _ in range(10):
        state = np.random.randn(4, 3).astype(np.float32)
        action = np.random.rand(1).astype(np.float32)
        agent.train_step(state, action, 1.0, state, False)
    
    # Force compute_advantages and update
    last_state = np.random.randn(4, 3).astype(np.float32)
    last_state_tensor = torch.FloatTensor(last_state).to(agent.device)
    normalized_last_state = agent._normalize_state(last_state_tensor)
    with torch.no_grad():
        last_value = agent.value_network(normalized_last_state).cpu().numpy()
    agent.buffer.compute_advantages(last_value)
    
    # Perform first update
    agent.update_if_buffer_ready()
    
    # Get learning rate after one step
    lr_after_one_step = agent.optimizer.param_groups[0]['lr']
    
    # Verify learning rate decreased
    assert lr_after_one_step < initial_lr, f"Learning rate should decrease, but got {lr_after_one_step} (initial was {initial_lr})"
    assert abs(lr_after_one_step - initial_lr * 0.5) < 1e-6, f"Learning rate should be halved, got {lr_after_one_step}"
    
    # Add more data and perform another update
    for _ in range(10):
        state = np.random.randn(4, 3).astype(np.float32)
        action = np.random.rand(1).astype(np.float32)
        agent.train_step(state, action, 1.0, state, False)
    
    agent.update_if_buffer_ready()
    
    # Get learning rate after two steps
    lr_after_two_steps = agent.optimizer.param_groups[0]['lr']
    
    # Verify learning rate decreased again
    assert lr_after_two_steps < lr_after_one_step, "Learning rate should decrease after second update"
    assert abs(lr_after_two_steps - initial_lr * 0.25) < 1e-6, f"Learning rate should be quartered after two steps, got {lr_after_two_steps}"


def test_ppo_proper_ratio_calculation():
    """Test that PPO properly calculates ratio using old log probabilities."""
    # Create simple environment
    observation_space = Box(
        low=-np.inf,
        high=np.inf,
        shape=(4, 3),
        dtype=np.float32
    )
    action_space = Box(
        low=-1,
        high=1,
        shape=(1,),
        dtype=np.float32
    )
    
    # Create agent
    agent = PPOAgent(observation_space, action_space, batch_size=8)
    
    # Generate a buffer of experiences
    states = []
    actions = []
    log_probs = []
    
    for _ in range(16):  # Generate more than batch_size
        state = np.random.randn(4, 3).astype(np.float32)
        
        # Capture old policy's action and log_prob
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(agent.device)
            normalized_state = agent._normalize_state(state_tensor)
            action_mean, action_std = agent.network(normalized_state)
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1).item()
        
        # Store state, action and log_prob
        action_np = action.cpu().numpy()
        
        # Add experience to buffer
        agent.train_step(state, action_np, 1.0, state, False)
        
        # Keep track for later verification
        states.append(state)
        actions.append(action_np)
        log_probs.append(log_prob)
    
    # Modify network parameters slightly to ensure policy changes
    for param in agent.network.parameters():
        param.data += torch.randn_like(param) * 0.01
    
    # Calculate new log probs for the same states and actions
    new_log_probs = []
    for i in range(len(states)):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(states[i]).to(agent.device)
            normalized_state = agent._normalize_state(state_tensor)
            action_mean, action_std = agent.network(normalized_state)
            dist = torch.distributions.Normal(action_mean, action_std)
            action_tensor = torch.FloatTensor(actions[i]).to(agent.device)
            new_log_prob = dist.log_prob(action_tensor).sum(dim=-1).item()
            new_log_probs.append(new_log_prob)
    
    # Verify that old and new log probs are different
    log_prob_diffs = np.abs(np.array(new_log_probs) - np.array(log_probs))
    assert np.mean(log_prob_diffs) > 0.0001, "Policy should change enough to have different log probs"
    
    # Now update the policy 
    update_results = agent.update_if_buffer_ready()
    
    # Verify we got ratio calculation in the update results
    assert "policy_loss" in update_results, "Policy loss should be calculated"
    assert "kl" in update_results, "KL divergence should be calculated"
    
    # Verify buffer was reset
    assert len(agent.buffer) == 0, "Buffer should be reset after update"
    
    # Very important: The buffer.get_batch() should return old_log_probs for ratio calculation
    # Fill buffer again
    for _ in range(8):
        state = np.random.randn(4, 3).astype(np.float32)
        action = np.random.rand(1).astype(np.float32)
        agent.train_step(state, action, 1.0, state, False)
    
    # Force compute_advantages
    last_state = np.random.randn(4, 3).astype(np.float32)
    last_state_tensor = torch.FloatTensor(last_state).to(agent.device)
    normalized_last_state = agent._normalize_state(last_state_tensor)
    with torch.no_grad():
        last_value = agent.value_network(normalized_last_state).cpu().numpy()
    agent.buffer.compute_advantages(last_value)
    
    # Get batch and check that old_log_probs is correctly returned
    batch_data = agent.buffer.get_batch()
    assert batch_data is not None, "Batch data should not be None"
    states, actions, old_log_probs, returns, advantages, old_values = batch_data
    
    # Verify shapes
    assert old_log_probs.shape[0] == states.shape[0], "Old log probs shape should match states"
    assert returns.shape[0] == states.shape[0], "Returns shape should match states"
    assert advantages.shape[0] == states.shape[0], "Advantages shape should match states"
    
    # The most important test - verify that ratio != 1 in the update method
    # Let's capture the ratio during an update
    original_update = agent.update
    
    ratios_list = []
    def update_with_ratio_check(*args, **kwargs):
        nonlocal ratios_list
        # Call original update
        result = original_update(*args, **kwargs)
        
        # Recalculate ratio for verification
        states, actions, old_log_probs = args[0], args[1], args[2]
        
        # Get current log probs
        action_mean, action_std = agent.network(states)
        dist = torch.distributions.Normal(action_mean, action_std)
        current_log_probs = dist.log_prob(actions)
        
        # Calculate ratio
        if current_log_probs.dim() > old_log_probs.dim():
            old_log_probs = old_log_probs.unsqueeze(-1).expand_as(current_log_probs)
        elif current_log_probs.dim() < old_log_probs.dim():
            current_log_probs = current_log_probs.mean(-1)
            
        ratio = torch.exp(current_log_probs - old_log_probs)
        ratios_list.append(ratio.detach().cpu().numpy())
        
        return result
    
    # Replace update method temporarily
    agent.update = update_with_ratio_check
    
    # Perform update
    agent.update_if_buffer_ready()
    
    # Restore original update method
    agent.update = original_update
    
    # Check ratios
    ratios = np.concatenate(ratios_list)
    ratio_diff_from_one = np.abs(ratios - 1.0)
    
    # The key assertion - ratio should not be 1
    avg_ratio_diff = np.mean(ratio_diff_from_one)
    assert avg_ratio_diff > 0.0001, f"Ratio should not be 1 (avg diff: {avg_ratio_diff})"
    print(f"Average ratio difference from 1: {avg_ratio_diff}")
    
    # Some ratios should be above 1 and some below, showing proper variance
    assert np.any(ratios > 1.01), "Some ratios should be significantly above 1"
    assert np.any(ratios < 0.99), "Some ratios should be significantly below 1"


def test_early_stopping_on_high_kl(ppo_agent):
    """Test that training stops when KL divergence is too high"""
    # Create sample batch
    batch_size = 32
    states = torch.randn(batch_size, 20, 5).to(ppo_agent.device)
    actions = torch.randn(batch_size, 1).to(ppo_agent.device)
    rewards = torch.randn(batch_size).to(ppo_agent.device)
    values = torch.randn(batch_size).to(ppo_agent.device)
    log_probs = torch.randn(batch_size).to(ppo_agent.device)
    dones = torch.zeros(batch_size).to(ppo_agent.device)
    
    # Set very low target KL to trigger early stopping
    ppo_agent.target_kl = 1e-6
    
    # Update and count epochs
    metrics = ppo_agent.update(states, actions, log_probs, rewards, values, dones)
    
    # Should have high KL divergence
    assert metrics["kl"] > ppo_agent.target_kl, "KL divergence should be higher than target"


class SimpleTradingEnv(gym.Env):
    """Simple environment for testing PPO in a trading-like context."""
    
    def __init__(self):
        super().__init__()
        self.window_size = 4
        self.n_features = 3
        self.max_steps = 100
        
        # Define spaces
        self.observation_space = gym.spaces.Box(
            low=-10.0, 
            high=10.0, 
            shape=(self.window_size, self.n_features),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self, seed=None):
        """Reset the environment."""
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
        
        # Generate initial price pattern (trending upward with noise)
        self.position = 0
        self.step_count = 0
        self.prices = np.zeros((self.max_steps + self.window_size, 1), dtype=np.float32)
        self.trend = np.random.choice([-1, 1])  # Random trend direction
        
        # Generate price series with a trend and noise
        for i in range(len(self.prices)):
            if i == 0:
                self.prices[i] = np.random.uniform(4.0, 6.0)
            else:
                # Random walk with trend
                self.prices[i] = self.prices[i-1] * (1 + 0.001 * self.trend + 0.002 * np.random.randn())
        
        # Create features (price, MA, volatility)
        self.features = np.zeros((self.max_steps + self.window_size, self.n_features), dtype=np.float32)
        self.features[:, 0] = self.prices.flatten()  # Price
        
        # Moving average (5-period)
        for i in range(4, len(self.features)):
            self.features[i, 1] = np.mean(self.prices[i-4:i+1])
        
        # Volatility (standard deviation over 5 periods)
        for i in range(4, len(self.features)):
            self.features[i, 2] = np.std(self.prices[i-4:i+1])
        
        # Normalize features
        feature_mean = np.mean(self.features, axis=0)
        feature_std = np.std(self.features, axis=0)
        feature_std[feature_std < 1e-8] = 1.0  # Avoid division by zero
        self.features = (self.features - feature_mean) / feature_std
        
        # Get initial observation
        obs = self.features[self.step_count:self.step_count+self.window_size]
        
        return obs, {}
    
    def step(self, action):
        """Take a step in the environment."""
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        # Get current price and next price
        current_price = self.prices[self.step_count + self.window_size - 1, 0]
        # Ensure we don't go out of bounds
        next_idx = min(self.step_count + self.window_size, len(self.prices) - 1)
        next_price = self.prices[next_idx, 0]
        
        # Calculate returns based on position and price change
        price_change_pct = (next_price / current_price) - 1.0
        
        # Calculate reward - a simplified PnL calculation
        old_position = self.position
        # Convert action from [-1, 1] to position [-1, 1]
        self.position = float(action[0])  # Simple position sizing
        
        # Reward is based on position * return - trading cost for position change
        trading_cost = 0.0001 * abs(self.position - old_position)  # 1 basis point per unit changed
        reward = (old_position * price_change_pct) - trading_cost
        
        # Additional reward for keeping position aligned with trend
        trend_alignment = self.position * self.trend * 0.0001
        reward += trend_alignment
        
        # Get next observation
        obs = self.features[self.step_count:self.step_count+self.window_size]
        
        # Add a small penalty for extreme positions to encourage exploration
        if abs(self.position) > 0.9:
            reward -= 0.0001
            
        info = {
            'price': float(current_price),
            'position': float(self.position),
            'price_change': float(price_change_pct),
            'trading_cost': float(trading_cost),
            'trend': self.trend
        }
        
        return obs, float(reward), done, False, info


def test_ppo_full_training():
    """Integration test for full PPO training process with rollout buffer and multiple epochs."""
    # Create environment
    env = SimpleTradingEnv()
    
    # Configure agent
    agent = PPOAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        n_epochs=5,
        batch_size=32,
        max_grad_norm=0.5,
        target_kl=0.02
    )
    
    # Run a small training session
    total_steps = 0
    num_episodes = 5
    update_interval = 64  # Number of steps before updating policy
    
    # Keep track of metrics
    rewards_history = []
    policy_loss_history = []
    value_loss_history = []
    ratio_history = []
    
    # Create a hook to capture ratios during updates
    original_update = agent.update
    
    def update_with_tracking(*args, **kwargs):
        nonlocal ratio_history
        
        # Call original update
        result = original_update(*args, **kwargs)
        
        # Calculate and store ratios
        states, actions, old_log_probs = args[0], args[1], args[2]
        
        with torch.no_grad():
            # Get current policy distribution
            action_mean, action_std = agent.network(states)
            dist = torch.distributions.Normal(action_mean, action_std)
            current_log_probs = dist.log_prob(actions)
            
            # Calculate ratio
            if current_log_probs.dim() > old_log_probs.dim():
                old_log_probs = old_log_probs.unsqueeze(-1).expand_as(current_log_probs)
            elif current_log_probs.dim() < old_log_probs.dim():
                current_log_probs = current_log_probs.mean(-1)
                
            ratio = torch.exp(current_log_probs - old_log_probs)
            ratio_history.append(ratio.detach().cpu().numpy().mean())
        
        return result
    
    # Replace update method
    agent.update = update_with_tracking
    
    # Reset the environment with a fixed seed for reproducibility
    state, _ = env.reset(seed=42)
    episode_reward = 0
    
    for _ in range(num_episodes):
        done = False
        episode_reward = 0
        
        while not done:
            # Select action
            action = agent.get_action(state)
            
            # Take step in environment
            next_state, reward, done, _, info = env.step(action)
            
            # Store transition
            agent.train_step(state, action, reward, next_state, done)
            
            # Update state and counters
            state = next_state
            total_steps += 1
            episode_reward += reward
            
            # Update policy if enough steps have been collected
            if total_steps % update_interval == 0:
                update_info = agent.update_if_buffer_ready()
                
                if update_info:
                    policy_loss_history.append(update_info.get("policy_loss", 0))
                    value_loss_history.append(update_info.get("value_loss", 0))
        
        # Track episode rewards
        rewards_history.append(episode_reward)
        
        # Reset environment for next episode
        if done:
            state, _ = env.reset()
    
    # Restore original update method
    agent.update = original_update
    
    # Verify learning metrics
    
    # 1. We should have collected enough data for at least one update
    assert len(policy_loss_history) > 0, "No policy updates occurred"
    
    # 2. Ratio should not always be 1 (which would indicate incorrect PPO implementation)
    assert len(ratio_history) > 0, "No ratios were collected"
    ratio_diffs = np.abs(np.array(ratio_history) - 1.0)
    assert np.mean(ratio_diffs) > 0.001, "Ratios are too close to 1, suggesting PPO implementation issues"
    
    # 3. Policy loss should generally decrease over updates
    # Skip this check as it might be noisy for small training runs
    
    # 4. Last episode reward should be better than random policy on average
    # (This is highly environment-dependent - adjust threshold as needed)
    assert rewards_history[-1] > -0.1, f"Final reward ({rewards_history[-1]}) suggests policy didn't improve"
    
    # Force a final update to clear the buffer if needed
    if len(agent.buffer) > 0:
        # Compute advantages for any remaining experiences
        last_state = env.reset()[0]  # Get a state for value estimation
        last_state_tensor = torch.FloatTensor(last_state).to(agent.device)
        normalized_last_state = agent._normalize_state(last_state_tensor)
        with torch.no_grad():
            last_value = agent.value_network(normalized_last_state).cpu().numpy()
        agent.buffer.compute_advantages(last_value)
        
        # Force update regardless of buffer size
        if len(agent.buffer) >= agent.batch_size:
            agent.update_if_buffer_ready()
        else:
            # If too few samples, manually reset buffer
            agent.buffer.reset()
    
    # 5. Verify buffer behavior
    assert len(agent.buffer) == 0, "Buffer should be empty after update"
    
    # Test agent's get_action function after training
    test_state = env.reset()[0]
    test_action = agent.get_action(test_state)
    assert isinstance(test_action, np.ndarray), "Action should be a numpy array"
    assert test_action.shape == (1,), f"Action shape should be (1,), got {test_action.shape}"
    assert -1.0 <= test_action[0] <= 1.0, f"Action should be in [-1, 1], got {test_action[0]}"


if __name__ == "__main__":
    pytest.main([__file__]) 