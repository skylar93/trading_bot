"""CVaRPPO: SB3 PPO with Lagrangian CVaR constraint in the loss function.

The Lagrangian method maintains a dual variable ν (nu) that penalises
violations of the CVaR tail-loss threshold:

    total_loss = ppo_loss + ν * relu(CVaR_α - threshold)

After each mini-batch update the dual variable is projected back onto [0, nu_max]:

    ν ← clip(ν + lr_ν * (CVaR_α - threshold), 0, nu_max)

This soft-constraint formulation keeps the standard PPO guarantee intact while
progressively tightening the tail-risk budget when the constraint is violated.

Reference: Chow & Ghavamzadeh (2014), "Algorithms for CVaR Optimization in MDPs".
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance


class CVaRPPO(PPO):
    """PPO with Lagrangian CVaR constraint injected into the mini-batch loss.

    Parameters
    ----------
    policy:
        SB3 policy class or string (e.g. "MlpPolicy").
    env:
        Gymnasium environment or vectorised environment.
    cvar_alpha : float
        Tail probability α ∈ (0, 1).  CVaR_α is the mean of the worst
        α-fraction of returns in each mini-batch.  Default: 0.05 (5 % tail).
    cvar_threshold : float
        Maximum acceptable CVaR value.  Negative values correspond to a
        loss budget (e.g. -0.02 means "tail mean ≥ -2 %").  Default: -0.02.
    lr_nu : float
        Step size for the dual-variable gradient ascent step.  Default: 0.01.
    nu_max : float
        Hard upper bound on the dual variable to prevent divergence.
        Default: 10.0.
    **kwargs:
        All remaining keyword arguments are forwarded to ``stable_baselines3.PPO``.
    """

    def __init__(
        self,
        policy: Union[str, Type],
        env: Union[GymEnv, str],
        cvar_alpha: float = 0.05,
        cvar_threshold: float = -0.02,
        lr_nu: float = 0.01,
        nu_max: float = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(policy, env, **kwargs)
        self.cvar_alpha = cvar_alpha
        self.cvar_threshold = cvar_threshold
        self.lr_nu = lr_nu
        self.nu_max = nu_max
        # Lagrangian dual variable — updated manually (not a torch Parameter)
        self._nu: float = 0.0

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def nu(self) -> float:
        """Current value of the Lagrangian dual variable ν."""
        return self._nu

    def get_cvar_info(self) -> Dict[str, float]:
        """Return CVaR constraint state for logging / inspection."""
        return {
            "nu": self._nu,
            "alpha": self.cvar_alpha,
            "threshold": self.cvar_threshold,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_cvar(self, returns: th.Tensor) -> th.Tensor:
        """CVaR_α = mean of the worst α-fraction of *returns* in the batch.

        A smaller (more negative) CVaR means heavier tail losses.
        """
        n = returns.numel()
        k = max(1, int(np.ceil(n * self.cvar_alpha)))
        worst_k, _ = th.topk(returns, k, largest=False, sorted=False)
        return worst_k.mean()

    def _nu_update(self, cvar_value: float) -> None:
        """Dual-variable gradient ascent step (projected onto [0, nu_max])."""
        violation = cvar_value - self.cvar_threshold
        self._nu = float(np.clip(self._nu + self.lr_nu * violation, 0.0, self.nu_max))

    # ------------------------------------------------------------------
    # Overridden training loop
    # ------------------------------------------------------------------

    def train(self) -> None:  # noqa: C901 (complexity accepted — mirrors SB3 source)
        """PPO training loop with Lagrangian CVaR penalty added to the loss.

        The implementation mirrors ``stable_baselines3.PPO.train()`` exactly,
        with two additions per mini-batch:
          1. ``cvar_loss = ν * relu(CVaR_α(batch_returns) - threshold)``
             is added to the total loss before ``loss.backward()``.
          2. After all epochs, ν is updated once using the CVaR computed on
             the full rollout buffer (stable estimate).
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses: list = []
        pg_losses: list = []
        value_losses: list = []
        cvar_losses: list = []
        clip_fractions: list = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs: list = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())

                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # ----- CVaR Lagrangian penalty -----
                batch_returns = rollout_data.returns.detach()
                cvar_val = self._compute_cvar(batch_returns)
                nu_tensor = th.tensor(self._nu, dtype=th.float32, device=self.device)
                cvar_loss = nu_tensor * F.relu(cvar_val - self.cvar_threshold)
                cvar_losses.append(cvar_loss.item())
                # -----------------------------------

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + cvar_loss
                )

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at epoch {epoch} due to reaching max kl: "
                            f"{approx_kl_div:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        # ---- Dual-variable update on full rollout buffer (stable estimate) ----
        all_returns = self.rollout_buffer.returns.flatten()
        all_returns_t = th.as_tensor(all_returns, dtype=th.float32, device=self.device)
        buffer_cvar = self._compute_cvar(all_returns_t).item()
        self._nu_update(buffer_cvar)

        # ---- Logging ----
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/cvar_loss", np.mean(cvar_losses))
        self.logger.record("train/cvar_alpha", buffer_cvar)
        self.logger.record("train/cvar_nu", self._nu)
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
