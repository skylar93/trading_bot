"""
Offline RL / pre-training agents.

Modules
-------
trajectory_dataset
    TradingTrajectoryDataset — loads expert rollouts as (RTG, state, action) sequences
decision_transformer
    TradingDecisionTransformer — GPT-2-style causal transformer with optional LoRA
    DecisionTransformerTrainer  — supervised training loop (MSE on action targets)
"""

from agents.offline.trajectory_dataset import (
    Trajectory,
    TradingTrajectoryDataset,
)
from agents.offline.decision_transformer import (
    DecisionTransformerConfig,
    TradingDecisionTransformer,
    DecisionTransformerTrainer,
    _PEFT_AVAILABLE,
    _TRANSFORMERS_AVAILABLE,
)
from agents.offline.diffusion_augmentor import (
    DiffusionConfig,
    TradingDiffusionAugmentor,
)

__all__ = [
    "Trajectory",
    "TradingTrajectoryDataset",
    "DecisionTransformerConfig",
    "TradingDecisionTransformer",
    "DecisionTransformerTrainer",
    "_PEFT_AVAILABLE",
    "_TRANSFORMERS_AVAILABLE",
    "DiffusionConfig",
    "TradingDiffusionAugmentor",
]
