"""
FLAG-Trader: Financial LLM Agent with Grounding.

Based on FLAG-Trader (ACL 2025, Harvard/Columbia/NVIDIA).
Small LLM (SmolLM2-135M) + LoRA + PPO fine-tuning for trading.
"""

from agents.llm_rl.flag_trader import (
    FLAGTrader,
    FLAGTraderConfig,
    FLAGTraderTrainer,
    MarketStateFormatter,
)

__all__ = [
    "FLAGTrader",
    "FLAGTraderConfig",
    "FLAGTraderTrainer",
    "MarketStateFormatter",
]
