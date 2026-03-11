"""
CVaR (Conditional Value at Risk / Expected Shortfall) callback for SB3.

Enforces tail-risk constraints during RL training by:
1. Computing CVaR at the end of each rollout (on-policy: PPO, A2C) or
   periodically from the replay buffer (off-policy: SAC, TD3).
2. Scaling rollout advantages/returns downward when CVaR violates the threshold
   so gradient signal from tail-risk-inducing episodes is reduced.
3. Adjusting the entropy coefficient upward for more conservative exploration
   when the constraint is violated, and decaying it back when satisfied.
4. Optionally maintaining a Lagrangian dual variable λ for soft constraint
   enforcement via dual ascent (λ grows proportional to violation magnitude).
5. Logging all CVaR metrics to the SB3 logger and optionally to MLflow.

Algorithm compatibility:
  On-policy  (PPO, A2C): hooks _on_rollout_end, reads rollout_buffer.rewards,
                          scales advantages and returns in-place.
  Off-policy (SAC, TD3): hooks _on_step every ``off_policy_check_interval`` steps,
                          samples the replay buffer for CVaR estimation.
                          Reward buffer is NOT modified (preserves replay integrity).
"""

import logging
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone CVaR utility
# ---------------------------------------------------------------------------

def compute_cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Compute CVaR (Conditional Value at Risk / Expected Shortfall) at level alpha.

    CVaR_alpha = E[R | R <= quantile_alpha(R)]
    i.e. the mean of the worst ``alpha`` fraction of ``returns``.

    Args:
        returns: Array of returns / rewards (any shape, flattened internally).
        alpha:   Tail probability in (0, 1].  alpha=0.05 → worst 5%.

    Returns:
        CVaR as a float.  Returns 0.0 for an empty array.
    """
    flat = np.asarray(returns, dtype=np.float64).flatten()
    if flat.size == 0:
        return 0.0
    sorted_r = np.sort(flat)
    n_tail = max(1, int(np.ceil(alpha * len(sorted_r))))
    return float(np.mean(sorted_r[:n_tail]))


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class CVaRCallback(BaseCallback):
    """
    SB3 callback that enforces a CVaR (Expected Shortfall) constraint.

    Parameters
    ----------
    alpha : float
        Tail probability for CVaR (default 0.05 = worst 5% of rewards).
    cvar_threshold : float
        Constraint threshold.  CVaR values below this trigger a penalty.
        Units match the environment reward scale (default -0.02).
    penalty_scale : float
        Divisor applied to rollout advantages/returns when violated (>= 1.0).
        Larger values reduce gradient signal from tail-risk episodes.
    ent_coef_scale : float
        Multiplier applied to ``ent_coef`` when the constraint is violated,
        and used as a divisor when decaying back (>= 1.0).
    max_ent_coef : float
        Hard ceiling on the entropy coefficient (default 0.1).
    use_lagrangian : bool
        If True, maintain a Lagrangian multiplier λ via dual ascent.
        λ is added to the CVaR penalty (informational; stored in self.lambda_cvar).
    lagrangian_lr : float
        Step-size for the Lagrangian dual variable update.
    lambda_init : float
        Initial value of the Lagrangian multiplier.
    off_policy_check_interval : int
        For SAC/TD3: CVaR is computed from the replay buffer every this many steps.
    log_interval : int
        Log metrics every ``log_interval`` rollout-ends (on-policy) or
        CVaR checks (off-policy).
    mlflow_manager : optional
        Unified MLflow manager instance. If provided, metrics are forwarded via
        ``mlflow_manager.log_metric(key, value, step=...)``.
    verbose : int
        SB3 verbosity level (0=silent, 1=warnings, 2=debug).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        cvar_threshold: float = -0.02,
        penalty_scale: float = 2.0,
        ent_coef_scale: float = 2.0,
        max_ent_coef: float = 0.1,
        use_lagrangian: bool = False,
        lagrangian_lr: float = 0.01,
        lambda_init: float = 0.0,
        off_policy_check_interval: int = 1000,
        log_interval: int = 1,
        mlflow_manager=None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)

        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if penalty_scale < 1.0:
            raise ValueError(f"penalty_scale must be >= 1.0, got {penalty_scale}")
        if ent_coef_scale < 1.0:
            raise ValueError(f"ent_coef_scale must be >= 1.0, got {ent_coef_scale}")

        self.alpha = alpha
        self.cvar_threshold = cvar_threshold
        self.penalty_scale = penalty_scale
        self.ent_coef_scale = ent_coef_scale
        self.max_ent_coef = max_ent_coef
        self.use_lagrangian = use_lagrangian
        self.lagrangian_lr = lagrangian_lr
        self.lambda_cvar = float(lambda_init)
        self.off_policy_check_interval = off_policy_check_interval
        self.log_interval = log_interval
        self.mlflow_manager = mlflow_manager

        # State reset in _on_training_start
        self.rollout_count: int = 0
        self.violation_count: int = 0
        self.last_cvar: float = 0.0
        self._original_ent_coef: Optional[float] = None
        self._is_on_policy: Optional[bool] = None
        self._off_policy_check_count: int = 0

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        """Detect algorithm type and snapshot the original entropy coefficient."""
        self._is_on_policy = hasattr(self.model, "rollout_buffer")

        # Only snapshot float ent_coef (PPO/A2C); SAC uses a tensor.
        if hasattr(self.model, "ent_coef") and isinstance(self.model.ent_coef, float):
            self._original_ent_coef = float(self.model.ent_coef)
        else:
            self._original_ent_coef = None

        if self.verbose >= 1:
            algo = "on-policy" if self._is_on_policy else "off-policy"
            logger.info(
                "CVaRCallback attached (%s): alpha=%.2f, threshold=%.4f",
                algo,
                self.alpha,
                self.cvar_threshold,
            )

    def _on_rollout_end(self) -> None:
        """
        Called at the end of each on-policy rollout (PPO, A2C).

        Reads rewards from the rollout buffer, computes CVaR, and applies
        the constraint penalty (scales advantages and returns in-place).
        """
        if not hasattr(self.model, "rollout_buffer"):
            return  # Off-policy — handled in _on_step

        rewards = self.model.rollout_buffer.rewards.copy()
        self._apply_cvar_constraint(rewards, can_modify_buffer=True)

    def _on_step(self) -> bool:
        """
        Called every environment step.

        For on-policy algorithms the constraint is applied in ``_on_rollout_end``.
        For off-policy algorithms (SAC, TD3) this method samples the replay buffer
        every ``off_policy_check_interval`` steps and applies the constraint
        (entropy adjustment + Lagrangian only; buffer is not modified).
        """
        # On-policy: constraint is handled in _on_rollout_end
        if self._is_on_policy:
            return True

        self._off_policy_check_count += 1
        if self._off_policy_check_count % self.off_policy_check_interval != 0:
            return True

        if not hasattr(self.model, "replay_buffer"):
            return True
        rb = self.model.replay_buffer
        if rb.size() == 0:
            return True

        sample_size = min(1000, rb.size())
        try:
            samples = rb.sample(sample_size)
            rewards = samples.rewards.cpu().numpy().flatten()
        except Exception as exc:  # pragma: no cover
            logger.debug("Could not sample replay buffer for CVaR: %s", exc)
            return True

        # can_modify_buffer=False: preserve replay buffer integrity
        self._apply_cvar_constraint(rewards, can_modify_buffer=False)
        return True

    # ------------------------------------------------------------------
    # Core constraint logic
    # ------------------------------------------------------------------

    def _apply_cvar_constraint(
        self, rewards: np.ndarray, can_modify_buffer: bool
    ) -> None:
        """Compute CVaR and enforce the tail-risk constraint."""
        cvar = compute_cvar(rewards, self.alpha)
        self.last_cvar = cvar
        self.rollout_count += 1

        violated = cvar < self.cvar_threshold

        if violated:
            self.violation_count += 1

            # For on-policy algorithms: scale advantages and returns to reduce
            # gradient signal from tail-risk-inducing episodes.
            if can_modify_buffer and self._is_on_policy:
                buf = self.model.rollout_buffer
                buf.advantages /= self.penalty_scale
                buf.returns /= self.penalty_scale

            self._adjust_ent_coef(violated=True)
            self._update_lagrangian(cvar)

        else:
            # Decay ent_coef back toward the original when constraint is met
            self._adjust_ent_coef(violated=False)
            self._update_lagrangian(cvar)

        if self.rollout_count % self.log_interval == 0:
            self._log_metrics(cvar, violated)

    def _adjust_ent_coef(self, violated: bool) -> None:
        """
        Scale the entropy coefficient up on violation, decay back otherwise.

        Only operates when the model exposes a mutable float ``ent_coef``
        (i.e. PPO/A2C with a fixed entropy coefficient).
        """
        if self._original_ent_coef is None:
            return
        if not hasattr(self.model, "ent_coef"):
            return
        if not isinstance(self.model.ent_coef, float):
            return  # SAC auto-entropy uses a Tensor — skip

        if violated:
            new_ent = min(self.model.ent_coef * self.ent_coef_scale, self.max_ent_coef)
        else:
            new_ent = max(
                self.model.ent_coef / self.ent_coef_scale,
                self._original_ent_coef,
            )
        self.model.ent_coef = new_ent

    def _update_lagrangian(self, cvar: float) -> None:
        """
        Dual ascent update for the Lagrangian multiplier.

        λ ← max(0, λ + lr * (threshold - CVaR))

        When CVaR < threshold (violated), the constraint_violation > 0
        and λ increases.  When satisfied, λ decays toward zero.
        """
        if not self.use_lagrangian:
            return
        constraint_violation = self.cvar_threshold - cvar  # > 0 when violated
        self.lambda_cvar = max(
            0.0, self.lambda_cvar + self.lagrangian_lr * constraint_violation
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_metrics(self, cvar: float, violated: bool) -> None:
        """Emit CVaR metrics to the SB3 logger and optionally to MLflow."""
        violation_rate = self.violation_count / max(1, self.rollout_count)
        metrics = {
            "risk/cvar": cvar,
            "risk/cvar_threshold": self.cvar_threshold,
            "risk/cvar_violated": float(violated),
            "risk/violation_count": float(self.violation_count),
            "risk/violation_rate": violation_rate,
            "risk/lambda_cvar": self.lambda_cvar,
        }

        if self._original_ent_coef is not None and hasattr(self.model, "ent_coef"):
            if isinstance(self.model.ent_coef, float):
                metrics["risk/ent_coef"] = self.model.ent_coef

        for key, val in metrics.items():
            self.logger.record(key, val)

        if violated and self.verbose >= 1:
            logger.warning(
                "CVaR constraint violated: CVaR=%.4f < threshold=%.4f "
                "(violations=%d/%d, rate=%.1f%%)",
                cvar,
                self.cvar_threshold,
                self.violation_count,
                self.rollout_count,
                violation_rate * 100,
            )

        if self.mlflow_manager is not None:
            try:
                step = self.num_timesteps
                for key, val in metrics.items():
                    safe_key = key.replace("/", "_")
                    self.mlflow_manager.log_metric(safe_key, float(val), step=step)
            except Exception as exc:
                logger.debug("MLflow logging failed in CVaRCallback: %s", exc)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def violation_rate(self) -> float:
        """Fraction of rollouts in which the CVaR constraint was violated."""
        return self.violation_count / max(1, self.rollout_count)
