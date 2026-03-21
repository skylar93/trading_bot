"""
Offline RL / pre-training agents.

Modules
-------
trajectory_dataset
    TradingTrajectoryDataset — loads expert rollouts as (RTG, state, action) sequences
decision_transformer
    TradingDecisionTransformer — GPT-2-style causal transformer with optional LoRA
    DecisionTransformerTrainer  — supervised training loop (MSE on action targets)
cql_agent
    CQLAgent — Conservative Q-Learning offline RL baseline
    CQLConfig — configuration dataclass
dt_finetuner
    DecisionTransformerFineTuner — online PPO fine-tuning from a pre-trained DT
    DTFeatureExtractor — SB3 feature extractor using DT state embedding
    FineTunerConfig — configuration dataclass
diffusion_augmentor
    TradingDiffusionAugmentor — DDPM-based trajectory data augmentation
    DiffusionConfig — configuration dataclass
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
from agents.offline.cql_agent import (
    CQLConfig,
    CQLAgent,
)
from agents.offline.dt_finetuner import (
    FineTunerConfig,
    DecisionTransformerFineTuner,
    DTFeatureExtractor,
    _SB3_AVAILABLE,
)
from agents.offline.diffusion_augmentor import (
    DiffusionConfig,
    TradingDiffusionAugmentor,
)

__all__ = [
    # trajectory dataset
    "Trajectory",
    "TradingTrajectoryDataset",
    # decision transformer
    "DecisionTransformerConfig",
    "TradingDecisionTransformer",
    "DecisionTransformerTrainer",
    "_PEFT_AVAILABLE",
    "_TRANSFORMERS_AVAILABLE",
    # CQL
    "CQLConfig",
    "CQLAgent",
    # DT fine-tuner
    "FineTunerConfig",
    "DecisionTransformerFineTuner",
    "DTFeatureExtractor",
    "_SB3_AVAILABLE",
    # diffusion augmentation
    "DiffusionConfig",
    "TradingDiffusionAugmentor",
]
