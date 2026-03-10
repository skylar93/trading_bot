# MultiAgentManager and Meta-Agent Training

This document provides a comprehensive overview of the multi-agent trading system that uses the `MultiAgentManager` for coordinated training and decision making.

## Overview

The `MultiAgentManager` provides a sophisticated way to coordinate multiple trading agents with different strategies. It supports:

1. **Ensemble Decision Making** - Combining decisions from multiple agents
2. **Experience Sharing** - Using a shared buffer to let agents learn from each other
3. **Meta-Agent Training** - Learning which agent performs best in different situations
4. **Adaptive Weighting** - Dynamically adjusting strategy weights based on performance

## Architecture

The multi-agent architecture consists of:

- **MultiAgentTradingEnv** - Environment supporting multiple agents
- **MultiAgentManager** - Coordinates agents and handles ensemble logic
- **Sub-Agents** - Individual trading strategies (Momentum, MeanReversion, etc.)
- **Meta-Agent** - Optional higher-level agent for optimal selection/weighting of sub-agents

```
┌─────────────────────────────────────────┐
│             MultiAgentManager           │
├─────────────┬──────────────┬────────────┤
│  Meta-Agent │ Shared Buffer│  Ensemble  │
├─────────────┴──────────────┴────────────┤
│                                         │
├─────────────┬──────────────┬────────────┤
│ Agent 1     │ Agent 2      │ Agent N    │
│ (Momentum)  │ (MeanRev)    │ (...)      │
└─────────────┴──────────────┴────────────┘
        │             │            │
        v             v            v
┌─────────────────────────────────────────┐
│          MultiAgentTradingEnv           │
└─────────────────────────────────────────┘
```

## Ensemble Methods

The MultiAgentManager supports three main ensemble methods:

1. **Weighted** - Combine actions from all agents using a weighted average
2. **Best** - Select the action from the best-performing agent
3. **Meta** - Use a meta-agent to learn optimal agent selection/weighting

## Meta-Agent

The meta-agent is a higher-level agent that learns to coordinate the sub-agents. It:

- Takes observations from all sub-agents
- Optionally processes sub-agents' hidden states
- Outputs either:
  - Discrete selection of which agent to follow
  - Continuous weights for blending agent actions

## Shared Experience Buffer

The shared experience buffer allows agents to learn from each other's experiences:

- Experiences from all agents are stored in a shared buffer
- Agents periodically train on experiences from other agents
- Experiences are adapted to match each agent's observation/action space
- Valuable experiences (high rewards) are prioritized

## Configuration

To use the MultiAgentManager with a meta-agent, configure your YAML file as follows:

```yaml
env:
  type: "multi_agent_rl"
  use_manager: true
  ensemble_method: "meta"  # Options: "weighted", "best", "meta"
  
  # Sub-agent configurations
  multi_agent_configs:
    - id: "momentum_agent"
      agent_type: "ppo"
      strategy: "momentum"
      # Agent-specific parameters...
    
    - id: "mean_reversion_agent"
      agent_type: "ppo"
      strategy: "mean_reversion"
      # Agent-specific parameters...
  
  # Meta-agent configuration (when ensemble_method is "meta")
  meta_config:
    id: "meta_agent"
    type: "meta"
    model: "ppo"
    continuous_ensemble: true  # Use weights instead of selection
    use_attention: true  # Use attention mechanism for hidden states
```

## Usage

### Command-Line Interface

Run multi-agent training with the manager:

```bash
python run_multi_agent_manager.py --config config/multi_agent_config.yaml
```

### Python API

```python
from training.train_pipeline import train_multi_agent_with_manager
from envs.multi_agent_env import MultiAgentTradingEnv

# Create environment
env = MultiAgentTradingEnv(data, agent_configs)

# Define agent configurations
agent_configs = [
    {"id": "momentum_agent", "agent_type": "ppo", "strategy": "momentum"},
    {"id": "mean_reversion_agent", "agent_type": "ppo", "strategy": "mean_reversion"}
]

# Train with manager
results = train_multi_agent_with_manager(
    env=env,
    agent_configs=agent_configs,
    ensemble_method="meta",
    config=config
)
```

## Customization

### Adding a New Sub-Agent

1. Implement your agent class extending `BaseAgent`
2. Add your agent to the `agent_factory.py`
3. Include it in the `multi_agent_configs` list in your configuration

### Custom Meta-Agent Architecture

You can customize the meta-agent by modifying:

- The observation space to include additional market features
- The network architecture in `MetaNetwork` class
- The attention mechanism if using hidden states
- The reward shaping for better agent selection

## Performance Considerations

For optimal performance:

- Balance the number of sub-agents (2-5 is typically optimal)
- Use the shared experience buffer for greater sample efficiency
- Consider the computational cost of meta-agent training
- Save the entire manager for proper ensemble behavior during evaluation

## Best Practices

1. **Start Simple**: Begin with the "weighted" ensemble method before moving to "meta"
2. **Complementary Strategies**: Choose sub-agents with complementary trading strategies
3. **Hyperparameter Tuning**: Tune the meta-agent separately after sub-agents are trained
4. **Evaluation**: Evaluate the system on out-of-sample data with `evaluate_with_manager`
5. **Checkpointing**: Save both individual agents and the manager for proper restoration 