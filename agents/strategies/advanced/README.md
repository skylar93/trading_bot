# Advanced Trading Agents

This directory contains specialized agent implementations that go beyond basic single or multi-agent architectures.

## Agent Types

### Asset-Specific Agents (`asset_specific_agents.py`)

Specialized agents tailored for different asset classes:

- `CryptoAgent`: Optimized for cryptocurrency markets
  - Handles 24/7 markets with high volatility
  - Includes flash crash protection
  - Adapts to varying liquidity conditions

- `EquityAgent`: Optimized for stock markets
  - Handles exchange trading hours
  - Considers market session dynamics
  - Processes fundamentals alongside technicals

### Hierarchical Agent (`hierarchical_agent.py`)

A manager-worker architecture for temporal abstraction:

- `ManagerNetwork`: Sets high-level goals
- `WorkerNetwork`: Executes actions to achieve goals
- Operates at different time scales
- Supports curriculum learning

### Meta-Agent (`meta_agent.py`)

Ensemble decision-making architecture:

- Coordinates decisions from multiple sub-agents
- Can select best agent or blend their actions
- Learns which agents perform best in different market conditions
- Handles both discrete selection and continuous weighting

## Integration

These agents are integrated into the trading system through `agents/strategies/agent_factory.py`,
which creates the appropriate agent based on configuration.

## Usage

Example:

```python
from agents.strategies.agent_factory import create_agent

# Create a crypto-specific agent
crypto_agent = create_agent(
    agent_type="assetspecific",
    config={
        "asset_id": "BTC",
        "asset_type": "crypto"
    }
)

# Create a hierarchical agent
hierarchical_agent = create_agent(
    agent_type="hierarchical",
    config={
        "goal_dim": 8,
        "goal_horizon": 10
    }
)

# Create a meta-agent
meta_agent = create_agent(
    agent_type="meta",
    config={
        "ensemble_type": "continuous"
    }
)
``` 