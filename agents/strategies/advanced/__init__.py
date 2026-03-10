"""
Advanced agent strategies for trading.

Asset-specific agents tailored for different asset classes.
"""

from agents.strategies.advanced.asset_specific_agents import (
    AssetSpecificAgent,
    CryptoAgent,
    EquityAgent,
    AssetSpecificAgentFactory,
)

__all__ = [
    "AssetSpecificAgent",
    "CryptoAgent",
    "EquityAgent",
    "AssetSpecificAgentFactory",
]
