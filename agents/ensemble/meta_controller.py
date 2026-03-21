"""
Meta-Controller: learned agent weighting via a small MLP trained with PPO.

Architecture
------------
Input:
    - regime_probs   : (n_regimes,)   — softmax probabilities from regime detector
    - sharpe_history : (n_agents,)    — rolling Sharpe ratio of each sub-agent
    - market_features: (n_market_features,)  — optional extra market signals
      (e.g. volatility, momentum, prediction-market probability)

Output:
    - agent_weights  : (n_agents,)    — softmax weights in [min_weight, 1]

Training
--------
Online PPO update every ``rebalance_interval`` environment steps.
The reward signal is the weighted-portfolio return for that period.

Safety
------
- Minimum weight per agent: ``min_weight`` (default 0.05)
- Emergency cash mode: if all agents post negative Sharpe for
  ``emergency_window`` consecutive rebalance periods the controller
  returns a zero-action weight vector (all cash).

Usage
-----
    mc = MetaController(n_agents=3)
    weights = mc.get_weights(regime_probs, sharpe_history)

    # After collecting a rebalance window's worth of data:
    mc.update(observations, actions, rewards, dones)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MetaControllerConfig:
    """Hyper-parameters for the MetaController."""

    # Architecture
    n_regimes: int = 3          # number of market regime classes
    n_market_features: int = 4  # extra market/prediction-market features
    hidden_dim: int = 64        # MLP hidden layer width

    # PPO
    lr: float = 3e-4
    gamma: float = 0.99
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4         # gradient steps per update
    mini_batch_size: int = 32

    # Operational
    rebalance_interval: int = 20   # env steps between updates
    min_weight: float = 0.05       # floor per agent
    emergency_window: int = 5      # consecutive negative-Sharpe periods → cash
    buffer_size: int = 256         # max transitions stored before update


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class _MetaPolicy(nn.Module):
    """Shared-trunk actor-critic MLP."""

    def __init__(self, obs_dim: int, n_agents: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_dim, n_agents)   # logits → softmax
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.actor_head(h), self.critic_head(h).squeeze(-1)

    def get_dist(self, x: torch.Tensor):
        logits, value = self(x)
        return torch.distributions.Dirichlet(F.softplus(logits) + 1e-6), value


# ---------------------------------------------------------------------------
# Rollout buffer (minimal, no external deps)
# ---------------------------------------------------------------------------

class _RolloutBuffer:
    def __init__(self, max_size: int, obs_dim: int, n_agents: int) -> None:
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.reset()

    def reset(self) -> None:
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        if len(self.obs) >= self.max_size:
            return
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def __len__(self) -> int:
        return len(self.obs)

    def compute_returns(self, gamma: float, last_value: float = 0.0) -> np.ndarray:
        """Discounted returns (no GAE for simplicity)."""
        returns = np.zeros(len(self.rewards), dtype=np.float32)
        R = last_value
        for t in reversed(range(len(self.rewards))):
            R = self.rewards[t] + gamma * R * (1.0 - float(self.dones[t]))
            returns[t] = R
        return returns

    def as_tensors(self, gamma: float, device: str):
        returns = self.compute_returns(gamma)
        obs_t = torch.tensor(np.stack(self.obs), dtype=torch.float32, device=device)
        act_t = torch.tensor(np.stack(self.actions), dtype=torch.float32, device=device)
        lp_t = torch.tensor(self.log_probs, dtype=torch.float32, device=device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
        val_t = torch.tensor(self.values, dtype=torch.float32, device=device)
        adv_t = ret_t - val_t
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        return obs_t, act_t, lp_t, ret_t, adv_t


# ---------------------------------------------------------------------------
# MetaController
# ---------------------------------------------------------------------------

class MetaController:
    """
    Learned meta-controller that outputs per-agent ensemble weights.

    Parameters
    ----------
    n_agents : int
        Number of sub-agents in the ensemble.
    config : MetaControllerConfig, optional
        Hyper-parameters; defaults to ``MetaControllerConfig()``.
    device : str, optional
        Torch device string (default: auto).

    Example
    -------
    >>> mc = MetaController(n_agents=3)
    >>> regime_probs = np.array([0.6, 0.3, 0.1])
    >>> sharpe_history = np.array([0.8, -0.2, 0.5])
    >>> weights = mc.get_weights(regime_probs, sharpe_history)
    >>> assert weights.shape == (3,)
    >>> assert abs(weights.sum() - 1.0) < 1e-5
    """

    def __init__(
        self,
        n_agents: int,
        config: Optional[MetaControllerConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.n_agents = n_agents
        self.cfg = config or MetaControllerConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_dim = (
            self.cfg.n_regimes
            + n_agents              # sharpe history
            + self.cfg.n_market_features
        )

        self.policy = _MetaPolicy(
            obs_dim=self.obs_dim,
            n_agents=n_agents,
            hidden_dim=self.cfg.hidden_dim,
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg.lr)
        self.buffer = _RolloutBuffer(
            max_size=self.cfg.buffer_size,
            obs_dim=self.obs_dim,
            n_agents=n_agents,
        )

        # Emergency-mode tracking
        self._consecutive_neg_sharpe: int = 0
        self._emergency_mode: bool = False

        # Step counter for rebalance triggers
        self._steps_since_update: int = 0

        # Latest weights (cached)
        self._last_weights: np.ndarray = np.full(n_agents, 1.0 / n_agents)

        logger.info(
            "MetaController initialised — %d agents, obs_dim=%d, device=%s",
            n_agents,
            self.obs_dim,
            self.device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_weights(
        self,
        regime_probs: np.ndarray,
        sharpe_history: np.ndarray,
        market_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return normalised agent weights.

        Parameters
        ----------
        regime_probs : (n_regimes,) float array
            Softmax probabilities of current market regime.
        sharpe_history : (n_agents,) float array
            Rolling Sharpe ratio for each sub-agent (recent window).
        market_features : (n_market_features,) float array, optional
            Extra signals (e.g. prediction-market probabilities).
            Defaults to zeros if not provided.

        Returns
        -------
        weights : (n_agents,) float array summing to 1.0
            Each element >= ``cfg.min_weight``.
        """
        if self._emergency_mode:
            logger.warning("MetaController in emergency mode — all-cash weights")
            return np.zeros(self.n_agents, dtype=np.float32)

        obs = self._build_obs(regime_probs, sharpe_history, market_features)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            dist, _ = self.policy.get_dist(obs_t)
            raw_weights = dist.mean.squeeze(0).cpu().numpy()

        weights = self._apply_min_weight(raw_weights)
        self._last_weights = weights
        return weights

    def step(
        self,
        regime_probs: np.ndarray,
        sharpe_history: np.ndarray,
        portfolio_return: float,
        done: bool = False,
        market_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Get weights AND record transition for PPO update.

        Call this every rebalance step when you want the meta-controller
        to learn online.  After ``rebalance_interval`` calls the internal
        PPO update is triggered automatically.

        Returns
        -------
        weights : (n_agents,) float array
        """
        obs = self._build_obs(regime_probs, sharpe_history, market_features)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            dist, value = self.policy.get_dist(obs_t)
            action = dist.sample().squeeze(0)
            log_prob = dist.log_prob(action).item()
            value_scalar = value.item()

        raw_weights = action.cpu().numpy()
        weights = self._apply_min_weight(raw_weights)

        self.buffer.add(
            obs=obs,
            action=raw_weights,
            log_prob=log_prob,
            reward=portfolio_return,
            done=done,
            value=value_scalar,
        )

        # Update emergency mode
        all_neg = bool(np.all(sharpe_history < 0))
        if all_neg:
            self._consecutive_neg_sharpe += 1
        else:
            self._consecutive_neg_sharpe = 0
        self._emergency_mode = (
            self._consecutive_neg_sharpe >= self.cfg.emergency_window
        )

        self._steps_since_update += 1
        if self._steps_since_update >= self.cfg.rebalance_interval:
            if len(self.buffer) >= self.cfg.mini_batch_size:
                self._ppo_update()
            self._steps_since_update = 0

        self._last_weights = weights
        return weights

    def update(
        self,
        observations: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        dones: List[bool],
    ) -> dict:
        """
        Manual PPO update from externally-provided trajectories.

        Returns
        -------
        dict with 'policy_loss', 'value_loss', 'entropy'
        """
        self.buffer.reset()
        for obs, act, rew, done in zip(observations, actions, rewards, dones):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                dist, value = self.policy.get_dist(obs_t)
                act_t = torch.tensor(act, dtype=torch.float32, device=self.device)
                log_prob = dist.log_prob(act_t).item()
            self.buffer.add(obs, act, log_prob, rew, done, value.item())

        return self._ppo_update()

    @property
    def last_weights(self) -> np.ndarray:
        return self._last_weights.copy()

    @property
    def is_emergency(self) -> bool:
        return self._emergency_mode

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy_state": self.policy.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "n_agents": self.n_agents,
                "cfg": self.cfg,
            },
            path,
        )
        logger.info("MetaController saved to %s", path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "MetaController":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        mc = cls(n_agents=ckpt["n_agents"], config=ckpt["cfg"], device=device)
        mc.policy.load_state_dict(ckpt["policy_state"])
        mc.optimizer.load_state_dict(ckpt["optimizer_state"])
        logger.info("MetaController loaded from %s", path)
        return mc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(
        self,
        regime_probs: np.ndarray,
        sharpe_history: np.ndarray,
        market_features: Optional[np.ndarray],
    ) -> np.ndarray:
        rp = np.asarray(regime_probs, dtype=np.float32)
        sh = np.asarray(sharpe_history, dtype=np.float32)

        if market_features is None:
            mf = np.zeros(self.cfg.n_market_features, dtype=np.float32)
        else:
            mf = np.asarray(market_features, dtype=np.float32)

        # Validate / pad / clip
        if rp.shape[0] != self.cfg.n_regimes:
            rp = np.resize(rp, self.cfg.n_regimes)
        if sh.shape[0] != self.n_agents:
            sh = np.resize(sh, self.n_agents)
        if mf.shape[0] != self.cfg.n_market_features:
            mf = np.resize(mf, self.cfg.n_market_features)

        # Clip Sharpe to reasonable range
        sh = np.clip(sh, -5.0, 5.0)

        return np.concatenate([rp, sh, mf])

    def _apply_min_weight(self, raw: np.ndarray) -> np.ndarray:
        """Project softmax weights so each >= min_weight, then re-normalise."""
        w = np.clip(raw, 0.0, 1.0)
        w = w / (w.sum() + 1e-8)
        # Floor
        w = np.maximum(w, self.cfg.min_weight)
        w = w / w.sum()
        return w.astype(np.float32)

    def _ppo_update(self) -> dict:
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        obs_t, act_t, old_lp_t, ret_t, adv_t = self.buffer.as_tensors(
            gamma=self.cfg.gamma, device=self.device
        )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.cfg.ppo_epochs):
            # Mini-batch shuffle
            indices = torch.randperm(len(obs_t), device=self.device)
            for start in range(0, len(obs_t), self.cfg.mini_batch_size):
                idx = indices[start : start + self.cfg.mini_batch_size]
                if len(idx) < 2:
                    continue

                dist, values = self.policy.get_dist(obs_t[idx])
                new_lp = dist.log_prob(act_t[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - old_lp_t[idx])
                adv = adv_t[idx]
                policy_loss = -torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv,
                ).mean()

                value_loss = F.mse_loss(values, ret_t[idx])

                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        self.buffer.reset()

        stats = {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }
        logger.debug("MetaController PPO update: %s", stats)
        return stats
