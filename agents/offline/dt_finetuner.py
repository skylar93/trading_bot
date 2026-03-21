"""
Decision Transformer → online PPO fine-tuning pipeline.

Strategy
--------
1. Pre-train a :class:`~agents.offline.decision_transformer.TradingDecisionTransformer`
   on expert trajectories (supervised offline).
2. Extract the DT's *state embedding* layer as an SB3 ``BaseFeaturesExtractor``.
3. Build an SB3 PPO agent whose policy uses that feature extractor.
4. Fine-tune online against a live Gymnasium environment.

The DT's transformer backbone is **optionally frozen** (``freeze_backbone=True``)
so only the policy head adapts — useful when the dataset is large and the backbone
is well-trained.

Requirements
------------
stable_baselines3 >= 2.0

Usage::

    from agents.offline.decision_transformer import TradingDecisionTransformer
    from agents.offline.dt_finetuner import DecisionTransformerFineTuner

    # Load pre-trained DT
    dt = TradingDecisionTransformer.load("dt_pretrained.pt")

    # Wrap with online fine-tuner
    finetuner = DecisionTransformerFineTuner(
        dt_model=dt,
        env=gym.make("MyTradingEnv-v0"),
        freeze_backbone=False,
    )

    # Fine-tune online
    metrics = finetuner.fine_tune(total_timesteps=50_000)

    # Inference
    action, _ = finetuner.get_action(obs)

    # Persist (saves DT config + SB3 PPO checkpoint)
    finetuner.save("finetuned_dt_ppo.pt")
    finetuner2 = DecisionTransformerFineTuner.load("finetuned_dt_ppo.pt", env=env)
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from agents.offline.decision_transformer import (
    TradingDecisionTransformer,
    DecisionTransformerConfig,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional SB3 dependency guard
# ---------------------------------------------------------------------------

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    import gymnasium as gym

    _SB3_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SB3_AVAILABLE = False
    # Placeholder so class definitions below don't fail at import time
    BaseFeaturesExtractor = object  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class DTFeatureExtractor(BaseFeaturesExtractor):
    """
    SB3 ``BaseFeaturesExtractor`` that uses the DT's state embedding layer.

    The observation is assumed to be a **flat** vector of shape ``(state_dim,)``,
    matching :attr:`~agents.offline.decision_transformer.DecisionTransformerConfig.state_dim`.

    The output feature dimension equals
    :attr:`~agents.offline.decision_transformer.DecisionTransformerConfig.hidden_size`.

    Parameters
    ----------
    observation_space:
        SB3 / Gymnasium observation space (must be a flat ``Box``).
    dt_model:
        A :class:`TradingDecisionTransformer` instance whose ``state_embed``
        (and optionally ``transformer``) weights will be reused.
    freeze_backbone:
        If ``True``, ``state_embed`` and ``transformer`` weights are frozen;
        only the downstream SB3 policy head trains. Default ``False``.
    """

    def __init__(
        self,
        observation_space: Any,
        dt_model: TradingDecisionTransformer,
        freeze_backbone: bool = False,
    ) -> None:
        if not _SB3_AVAILABLE:
            raise ImportError(
                "stable_baselines3 is required for DTFeatureExtractor. "
                "Install with: pip install stable_baselines3"
            )

        features_dim = dt_model.config.hidden_size
        super().__init__(observation_space, features_dim=features_dim)

        # Reuse DT's state embedding (Linear → Tanh)
        self.state_embed = dt_model.state_embed

        if freeze_backbone:
            for p in self.state_embed.parameters():
                p.requires_grad_(False)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        observations: (B, state_dim)

        Returns
        -------
        features: (B, hidden_size)
        """
        return self.state_embed(observations)


# ---------------------------------------------------------------------------
# Fine-tuner
# ---------------------------------------------------------------------------

@dataclass
class FineTunerConfig:
    """Configuration for the DT→PPO fine-tuner."""

    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Feature extractor
    freeze_backbone: bool = False


class DecisionTransformerFineTuner:
    """
    Online PPO fine-tuner that initialises from a pre-trained Decision Transformer.

    Parameters
    ----------
    dt_model:
        Pre-trained :class:`TradingDecisionTransformer`.
    env:
        Gymnasium environment. Observation space must be a flat ``Box`` whose
        shape matches ``(dt_model.config.state_dim,)``.
    config:
        :class:`FineTunerConfig` — PPO and feature-extractor settings.
    device:
        ``"auto"`` selects CUDA if available, otherwise CPU.
    """

    def __init__(
        self,
        dt_model: TradingDecisionTransformer,
        env: Any,
        config: Optional[FineTunerConfig] = None,
        device: str = "auto",
    ) -> None:
        if not _SB3_AVAILABLE:
            raise ImportError(
                "stable_baselines3 is required for DecisionTransformerFineTuner. "
                "Install with: pip install stable_baselines3"
            )

        self.dt_model = dt_model
        self.config = config or FineTunerConfig()
        self.env = env

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        cfg = self.config
        policy_kwargs = {
            "features_extractor_class": DTFeatureExtractor,
            "features_extractor_kwargs": {
                "dt_model": dt_model,
                "freeze_backbone": cfg.freeze_backbone,
            },
        }

        self.ppo = PPO(
            "MlpPolicy",
            env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=0,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fine_tune(
        self,
        total_timesteps: int,
        progress_bar: bool = False,
    ) -> Dict[str, Any]:
        """
        Run SB3 PPO online learning for ``total_timesteps`` environment steps.

        Returns
        -------
        dict with key ``"total_timesteps"``
        """
        self.ppo.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)
        logger.debug("Fine-tuning complete: %d timesteps", total_timesteps)
        return {"total_timesteps": total_timesteps}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_action(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict action using the fine-tuned PPO policy.

        Parameters
        ----------
        obs: np.ndarray — observation from the environment
        deterministic: bool — if True use the policy mean (no sampling)

        Returns
        -------
        (action, state) — same as ``PPO.predict``
        """
        return self.ppo.predict(obs, deterministic=deterministic)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the fine-tuner to ``path``.

        Stores the DT config + state-dict alongside the SB3 PPO checkpoint
        in a single torch file.
        """
        # SB3 PPO saves to a zip file; we write it to a temp location and
        # embed the bytes alongside the DT state.
        import io
        buf = io.BytesIO()
        self.ppo.save(buf)
        buf.seek(0)
        ppo_bytes = buf.read()

        torch.save(
            {
                "dt_config": self.dt_model.config,
                "dt_state_dict": self.dt_model.state_dict(),
                "finetuner_config": self.config,
                "ppo_bytes": ppo_bytes,
                "device": self.device,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        env: Any,
        map_location: str = "cpu",
    ) -> "DecisionTransformerFineTuner":
        """
        Load a fine-tuner saved with :meth:`save`.

        Parameters
        ----------
        path: str — checkpoint path
        env: Gymnasium environment (required by SB3 for re-instantiation)
        map_location: str — torch device string
        """
        if not _SB3_AVAILABLE:
            raise ImportError(
                "stable_baselines3 is required to load DecisionTransformerFineTuner."
            )
        import io

        data = torch.load(path, map_location=map_location, weights_only=False)

        dt_model = TradingDecisionTransformer(data["dt_config"])
        dt_model.load_state_dict(data["dt_state_dict"])

        finetuner = cls(
            dt_model=dt_model,
            env=env,
            config=data["finetuner_config"],
            device=data.get("device", map_location),
        )

        # Restore the PPO weights
        ppo_bytes = data["ppo_bytes"]
        buf = io.BytesIO(ppo_bytes)
        buf.seek(0)
        finetuner.ppo = PPO.load(buf, env=env, device=map_location)

        return finetuner

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def count_parameters(self) -> Dict[str, int]:
        """Return total and trainable parameter counts of the PPO policy."""
        policy = self.ppo.policy
        total = sum(p.numel() for p in policy.parameters())
        trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ---------------------------------------------------------------------------
# Module flag
# ---------------------------------------------------------------------------

__all__ = [
    "_SB3_AVAILABLE",
    "DTFeatureExtractor",
    "FineTunerConfig",
    "DecisionTransformerFineTuner",
]
