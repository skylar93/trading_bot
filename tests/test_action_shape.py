import pytest
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_action_shape")

@pytest.mark.shape_verification
def test_agent_action_shape():
    """
    Test to verify action shape compatibility between agents and environment.
    The environment expects action shape (n_assets,) but agents may return (1,).
    """
    # Define a simplified environment mimicking MultiAgentMultiAssetEnv
    class TestMultiAssetEnv:
        def __init__(self, n_assets=3):
            self.n_assets = n_assets
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(n_assets,), dtype=np.float32
            )
            
        def step(self, action):
            """Verify action shape matches what environment expects"""
            assert action.shape == (self.n_assets,), \
                f"Expected action shape ({self.n_assets},), got {action.shape}"
            return True
            
    # Define a problematic agent that returns (1,) shaped actions
    class ProblemAgent:
        def get_action(self, observation):
            # This shape (1,) doesn't match environment's expected (3,)
            return np.array([0.5], dtype=np.float32)
            
    # Define a fixed agent that returns properly shaped actions
    class FixedAgent:
        def __init__(self, n_assets=3):
            self.n_assets = n_assets
            
        def get_action(self, observation):
            # This returns the correct shape (n_assets,)
            return np.array([0.1] * self.n_assets, dtype=np.float32)
    
    # Create environment and agents
    env = TestMultiAssetEnv(n_assets=3)
    problem_agent = ProblemAgent()
    fixed_agent = FixedAgent(n_assets=3)
    
    # Test the problematic agent - should fail assertion
    with pytest.raises(AssertionError):
        observation = np.zeros(10)  # Dummy observation
        action = problem_agent.get_action(observation)
        logger.debug(f"Problem agent action shape: {action.shape}")
        env.step(action)  # This should fail the assertion
    
    # Test the fixed agent - should pass
    observation = np.zeros(10)  # Dummy observation
    action = fixed_agent.get_action(observation)
    logger.debug(f"Fixed agent action shape: {action.shape}")
    assert env.step(action) is True  # This should pass the assertion
    
    logger.info("Test completed successfully!")

@pytest.mark.shape_verification
def test_real_agent_shape_compatibility():
    """
    Test the actual agent implementations to verify they can output 
    actions that match what MultiAgentMultiAssetEnv expects.
    """
    # Import your actual agents and environment 
    from agents.strategies.single.ppo_agent import PPOAgent
    from agents.strategies.multi.momentum_ppo_agent import MomentumPPOAgent
    
    # Create a dummy observation space and action space
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(60, 5), dtype=np.float32)
    
    # Test with different action space dimensions
    for n_assets in [1, 3, 5]:
        logger.info(f"Testing with {n_assets} assets...")
        
        # Create action space for this test
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_assets,), dtype=np.float32)
        
        # Create agents
        agent1 = PPOAgent(obs_space, action_space)
        agent2 = MomentumPPOAgent(obs_space, action_space)
        
        # Test with a dummy observation
        obs = np.zeros((60, 5))
        
        # Get and verify actions
        action1 = agent1.get_action(obs)
        action2 = agent2.get_action(obs)
        
        logger.info(f"PPOAgent action shape: {action1.shape}")
        logger.info(f"MomentumPPOAgent action shape: {action2.shape}")
        
        assert action1.shape == (n_assets,), f"PPOAgent action shape: expected ({n_assets},), got {action1.shape}"
        assert action2.shape == (n_assets,), f"MomentumPPOAgent action shape: expected ({n_assets},), got {action2.shape}"
    
    logger.info("All agents produce correctly shaped actions!")

if __name__ == "__main__":
    test_agent_action_shape()
    test_real_agent_shape_compatibility() 