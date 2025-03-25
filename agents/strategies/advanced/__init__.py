"""
Advanced agent strategies for trading.

This package contains specialized agent implementations that go beyond basic
single or multi-agent architectures, including:

1. Asset-specific agents tailored for different asset classes
2. Hierarchical agents with manager-worker architectures
3. Meta-agents for ensemble decision making
"""

from agents.strategies.advanced.meta_agent import MetaAgent
from agents.strategies.advanced.hierarchical_agent import HierarchicalAgent
from agents.strategies.advanced.asset_specific_agents import (
    AssetSpecificAgent,
    CryptoAgent,
    EquityAgent,
    AssetSpecificAgentFactory
)

__all__ = [
    'MetaAgent',
    'HierarchicalAgent',
    'AssetSpecificAgent',
    'CryptoAgent', 
    'EquityAgent',
    'AssetSpecificAgentFactory'
]
