"""SB3 callback that feeds step rewards into a DriftDetector.

When drift is detected the callback:
  1. Logs a warning via Python logging.
  2. Optionally saves a model checkpoint.
  3. Optionally switches the environment to a conservative action-scaling mode
     (requires the env to expose ``set_conservative_mode(bool)``).

Usage
-----
    from agents.sb3.drift_callback import DriftCallback
    from training.monitoring.drift_detector import DriftDetector

    detector = DriftDetector(method="adwin", confidence=0.002)
    callback = DriftCallback(
        drift_detector=detector,
        checkpoint_dir="checkpoints/drift",
        conservative_scale=0.5,
    )
    model.learn(total_timesteps=100_000, callback=callback)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback

from training.monitoring.drift_detector import DriftDetector

logger = logging.getLogger(__name__)


class DriftCallback(BaseCallback):
    """SB3 BaseCallback that monitors for concept drift.

    Parameters
    ----------
    drift_detector : DriftDetector
        Pre-configured detector instance (ADWIN or Page-Hinkley).
    checkpoint_dir : str or None
        Directory to save model checkpoints on drift detection.
        Pass ``None`` to disable checkpointing.
    conservative_scale : float or None
        If not None, multiply the environment's action magnitude by this
        factor after drift is detected (requires env to expose
        ``set_conservative_mode`` or ``action_scale`` attribute).
        Pass ``None`` to skip action scaling.
    cooldown_steps : int
        Minimum number of steps between two consecutive drift reactions
        to avoid repeatedly triggering.  Default: 1000.
    verbose : int
        SB3 verbosity level (0 = silent, 1 = info).
    """

    def __init__(
        self,
        drift_detector: Optional[DriftDetector] = None,
        checkpoint_dir: Optional[str] = None,
        conservative_scale: Optional[float] = None,
        cooldown_steps: int = 1000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.drift_detector = drift_detector if drift_detector is not None else DriftDetector()
        self.checkpoint_dir = checkpoint_dir
        self.conservative_scale = conservative_scale
        self.cooldown_steps = cooldown_steps
        self._last_drift_step: int = -cooldown_steps  # allow detection from step 0

    # ------------------------------------------------------------------
    # BaseCallback interface
    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        """Called after every environment step."""
        # Feed the most recent reward(s) into the detector.
        # ``self.locals["rewards"]`` is a numpy array of shape (n_envs,).
        rewards = self.locals.get("rewards")
        if rewards is None:
            return True

        for reward in rewards:
            self.drift_detector.update(float(reward))

        if (
            self.drift_detector.drift_detected
            and (self.num_timesteps - self._last_drift_step) >= self.cooldown_steps
        ):
            self._handle_drift()
            self._last_drift_step = self.num_timesteps

        return True  # continue training

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _handle_drift(self) -> None:
        """React to a confirmed drift event."""
        logger.warning(
            "Concept drift detected at step %d (total detections: %d).",
            self.num_timesteps,
            self.drift_detector.n_detections,
        )

        # 1. Checkpoint
        if self.checkpoint_dir is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(
                self.checkpoint_dir,
                f"drift_ckpt_step_{self.num_timesteps}",
            )
            try:
                self.model.save(ckpt_path)
                logger.info("Drift checkpoint saved to %s.", ckpt_path)
            except Exception as exc:
                logger.error("Failed to save drift checkpoint: %s", exc)

        # 2. Conservative mode
        if self.conservative_scale is not None:
            self._set_conservative_mode(self.conservative_scale)

    def _set_conservative_mode(self, scale: float) -> None:
        """Attempt to set conservative action scaling on the vectorised env."""
        try:
            env = self.training_env
            # VecEnv: iterate over wrapped envs
            if hasattr(env, "envs"):
                for sub_env in env.envs:
                    _apply_conservative(sub_env, scale)
            elif hasattr(env, "env"):
                _apply_conservative(env.env, scale)
            else:
                _apply_conservative(env, scale)
            logger.info("Switched to conservative action scale %.2f.", scale)
        except Exception as exc:
            logger.debug("Could not set conservative mode: %s", exc)


def _apply_conservative(env, scale: float) -> None:
    """Helper: set conservative mode on a single env if supported."""
    if hasattr(env, "set_conservative_mode"):
        env.set_conservative_mode(True, scale)
    elif hasattr(env, "action_scale"):
        env.action_scale = scale
