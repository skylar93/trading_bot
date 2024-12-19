import pytest
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from training.train_multi_agent import train_multi_agent_system
from envs.multi_agent_env import MultiAgentTradingEnv
from agents.strategies.ppo_agent import PPOAgent

@pytest.fixture
def sample_data():
    """Create sample price data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1h')
    df = pd.DataFrame({
        '$open': np.random.randn(len(dates)) * 10 + 100,
        '$high': np.random.randn(len(dates)) * 10 + 100,
        '$low': np.random.randn(len(dates)) * 10 + 100,
        '$close': np.random.randn(len(dates)) * 10 + 100,
        '$volume': np.abs(np.random.randn(len(dates)) * 1000)
    }, index=dates)
    
    # Ensure high is highest and low is lowest
    df['$high'] = df[['$open', '$high', '$low', '$close']].max(axis=1)
    df['$low'] = df[['$open', '$high', '$low', '$close']].min(axis=1)
    return df

@pytest.fixture
def agent_configs():
    """Create test agent configurations"""
    return [
        {
            'id': 'momentum_trader',
            'strategy': 'momentum',
            'initial_balance': 10000.0,
            'fee_multiplier': 1.0
        },
        {
            'id': 'mean_reversion_trader',
            'strategy': 'mean_reversion',
            'initial_balance': 10000.0,
            'fee_multiplier': 1.0
        }
    ]

def test_multi_agent_env_creation(sample_data, agent_configs):
    """Test multi-agent environment creation"""
    env = MultiAgentTradingEnv(sample_data, agent_configs)
    
    # Check observation and action spaces
    for config in agent_configs:
        assert config['id'] in env.observation_spaces
        assert config['id'] in env.action_spaces
        
        # Verify observation space dimensions
        obs_space = env.observation_spaces[config['id']]
        assert len(obs_space.shape) == 2  # (window_size, features)
        
        # Verify action space
        action_space = env.action_spaces[config['id']]
        assert action_space.shape == (1,)  # Continuous action space

def test_multi_agent_reset(sample_data, agent_configs):
    """Test environment reset"""
    env = MultiAgentTradingEnv(sample_data, agent_configs)
    observations, info = env.reset()
    
    # Check observations
    for agent_id in env.agents:
        assert agent_id in observations
        assert isinstance(observations[agent_id], np.ndarray)
        assert not np.isnan(observations[agent_id]).any()

def test_multi_agent_step(sample_data, agent_configs):
    """Test environment step"""
    env = MultiAgentTradingEnv(sample_data, agent_configs)
    observations, _ = env.reset()
    
    # Create random actions
    actions = {
        agent_id: np.array([np.random.uniform(-1, 1)])
        for agent_id in env.agents
    }
    
    # Take step
    next_obs, rewards, dones, truncated, infos = env.step(actions)
    
    # Check outputs
    for agent_id in env.agents:
        assert agent_id in next_obs
        assert agent_id in rewards
        assert agent_id in dones
        assert agent_id in infos
        
        assert isinstance(rewards[agent_id], float)
        assert isinstance(dones[agent_id], bool)
        assert 'portfolio_value' in infos[agent_id]

def test_experience_sharing():
    """Test experience sharing between agents"""
    # Create shared replay buffer
    shared_buffer = []
    
    # Add experience from agent 1
    exp1 = {
        'state': np.random.randn(10, 5),
        'action': np.array([0.5]),
        'reward': 1.0,
        'next_state': np.random.randn(10, 5),
        'done': False
    }
    shared_buffer.append(exp1)
    
    # Add experience from agent 2
    exp2 = {
        'state': np.random.randn(10, 5),
        'action': np.array([-0.3]),
        'reward': -0.5,
        'next_state': np.random.randn(10, 5),
        'done': False
    }
    shared_buffer.append(exp2)
    
    # Verify buffer
    assert len(shared_buffer) == 2
    assert all(isinstance(exp['reward'], float) for exp in shared_buffer)

def test_gpu_utilization(sample_data, agent_configs):
    """Test GPU resource utilization"""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")
    
    env = MultiAgentTradingEnv(sample_data, agent_configs)
    
    # Create agents with GPU support
    agents = {
        config['id']: PPOAgent(
            env.observation_spaces[config['id']], 
            env.action_spaces[config['id']],
            device='cuda'
        )
        for config in agent_configs
    }
    
    # Verify agents are on GPU
    for agent in agents.values():
        assert str(agent.device) == 'cuda'
        assert next(agent.network.parameters()).is_cuda

def test_different_strategies(sample_data):
    """Test different trading strategies"""
    # Define agents with different strategies
    agent_configs = [
        {
            'id': 'momentum',
            'strategy': 'momentum',
            'lookback': 20,
            'threshold': 0.02,
            'initial_balance': 10000.0,
            'fee_multiplier': 1.0
        },
        {
            'id': 'mean_reversion',
            'strategy': 'mean_reversion',
            'window': 50,
            'std_dev': 2.0,
            'initial_balance': 10000.0,
            'fee_multiplier': 1.0
        },
        {
            'id': 'market_maker',
            'strategy': 'market_making',
            'spread': 0.001,
            'inventory_limit': 100,
            'initial_balance': 10000.0,
            'fee_multiplier': 0.8
        }
    ]
    
    env = MultiAgentTradingEnv(sample_data, agent_configs)
    
    # Test that each agent gets appropriate observations
    observations, _ = env.reset()
    for agent_id, config in zip(env.agents, agent_configs):
        obs = observations[agent_id]
        if config['strategy'] == 'momentum':
            assert obs.shape[0] >= config['lookback']
        elif config['strategy'] == 'mean_reversion':
            assert obs.shape[0] >= config['window']

def test_training_stability(sample_data, agent_configs, tmp_path):
    """Test stability of multi-agent training"""
    # Create environment and agents
    env = MultiAgentTradingEnv(sample_data, agent_configs)
    agents = {
        config['id']: PPOAgent(
            env.observation_spaces[config['id']], 
            env.action_spaces[config['id']]
        )
        for config in agent_configs
    }
    
    # Create model save directory
    save_dir = tmp_path / "multi_agent_models"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Run short training
    metrics = train_multi_agent_system(
        env=env,
        agents=agents,
        num_episodes=5,
        save_freq=2,
        save_path=str(save_dir)
    )
    
    # Check metrics
    for agent_id in agents:
        assert 'episode_rewards' in metrics[agent_id]
        assert 'portfolio_values' in metrics[agent_id]
        assert len(metrics[agent_id]['episode_rewards']) == 5
        
        # Check for NaN values
        assert not np.isnan(metrics[agent_id]['episode_rewards']).any()
        assert not np.isnan(metrics[agent_id]['portfolio_values']).any()
        
        # Check model files
        model_files = list(save_dir.glob(f"{agent_id}_episode_*.pt"))
        assert len(model_files) > 0, f"No model files found for agent {agent_id}"

if __name__ == "__main__":
    pytest.main([__file__]) 