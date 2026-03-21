"""
Regime-Aware Experience Store with Elastic Weight Consolidation (EWC).

Week 23 — Continual Learning Pipeline.

Key components
--------------
RegimeAwareExperienceStore
    Circular replay buffer that tags each transition with a regime_id.
    Balanced sampling: current-regime 70 %, past-regime 30 % (configurable).

EWCRegularizer
    Lightweight diagonal EWC:
      1. After finishing a regime, call consolidate(model) to record
         θ* and diagonal Fisher F (estimated from a mini-dataset).
      2. During fine-tuning, add ewc_loss(model) to the PPO objective.
      Loss: Σ_i  F_i * (θ_i − θ*_i)² * λ / 2

    Based on: Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting
    in neural networks." PNAS 114(13):3521-3526.

Usage
-----
    store = RegimeAwareExperienceStore(obs_dim=18, max_size_per_regime=50_000)
    store.add(obs, action, reward, next_obs, done, regime_id=1)
    batch = store.sample(batch_size=256, current_regime=1)

    ewc = EWCRegularizer(ewc_lambda=0.4)
    ewc.consolidate(model, dataset_tensors)
    loss = policy_loss + ewc.ewc_loss(model)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transition dataclass
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    regime_id: int = 0


# ---------------------------------------------------------------------------
# Per-regime circular buffer
# ---------------------------------------------------------------------------

class _RegimeBuffer:
    """Fixed-size circular buffer for a single regime."""

    def __init__(self, max_size: int, obs_dim: int, act_dim: int) -> None:
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self._actions = np.zeros((max_size, act_dim), dtype=np.float32)
        self._rewards = np.zeros(max_size, dtype=np.float32)
        self._dones = np.zeros(max_size, dtype=bool)
        self._ptr = 0
        self._size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float,
            next_obs: np.ndarray, done: bool) -> None:
        self._obs[self._ptr] = obs.reshape(self.obs_dim)
        self._actions[self._ptr] = np.atleast_1d(action).reshape(self.act_dim)
        self._rewards[self._ptr] = reward
        self._next_obs[self._ptr] = next_obs.reshape(self.obs_dim)
        self._dones[self._ptr] = done
        self._ptr = (self._ptr + 1) % self.max_size
        self._size = min(self._size + 1, self.max_size)

    def sample(self, n: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self._size, size=n)
        return {
            "obs": self._obs[idx],
            "actions": self._actions[idx],
            "rewards": self._rewards[idx],
            "next_obs": self._next_obs[idx],
            "dones": self._dones[idx],
        }

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# RegimeAwareExperienceStore
# ---------------------------------------------------------------------------

class RegimeAwareExperienceStore:
    """
    Replay buffer with per-regime partitioning and balanced sampling.

    Parameters
    ----------
    obs_dim:
        Observation dimensionality (flattened).
    act_dim:
        Action dimensionality.
    max_size_per_regime:
        Maximum transitions stored per regime.
    n_regimes:
        Number of regimes (default 3: low-vol, mid-vol, crisis).
    current_regime_ratio:
        Fraction of batch drawn from the current regime.
        Remaining (1 - ratio) is drawn uniformly from other regimes.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 1,
        max_size_per_regime: int = 50_000,
        n_regimes: int = 3,
        current_regime_ratio: float = 0.70,
    ) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_regimes = n_regimes
        self.current_regime_ratio = current_regime_ratio
        self._buffers: Dict[int, _RegimeBuffer] = {
            r: _RegimeBuffer(max_size_per_regime, obs_dim, act_dim)
            for r in range(n_regimes)
        }

    # ------------------------------------------------------------------
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        regime_id: int = 0,
    ) -> None:
        """Add a single transition to the appropriate regime buffer."""
        regime_id = int(regime_id) % self.n_regimes
        self._buffers[regime_id].add(obs, action, reward, next_obs, done)

    def add_transition(self, t: Transition) -> None:
        self.add(t.obs, t.action, t.reward, t.next_obs, t.done, t.regime_id)

    # ------------------------------------------------------------------
    def sample(
        self,
        batch_size: int,
        current_regime: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Sample a mixed batch.

        current_regime_ratio * batch_size from ``current_regime``,
        the rest sampled uniformly from all other non-empty regimes.
        Falls back to pure current-regime sampling if other regimes are empty.
        """
        current_buf = self._buffers[current_regime % self.n_regimes]
        other_bufs = [
            b for rid, b in self._buffers.items()
            if rid != current_regime and len(b) > 0
        ]

        n_current = int(batch_size * self.current_regime_ratio)
        n_other = batch_size - n_current

        if len(current_buf) == 0:
            raise ValueError(
                f"No transitions stored for regime {current_regime}. "
                "Add some data before sampling."
            )

        # Clamp to available size
        n_current = min(n_current, len(current_buf))
        parts = [current_buf.sample(n_current)]

        if n_other > 0 and other_bufs:
            # Round-robin across other regimes
            n_each = max(1, n_other // len(other_bufs))
            for b in other_bufs:
                parts.append(b.sample(min(n_each, len(b))))
        elif n_other > 0:
            # No past-regime data — fill from current
            extra = min(n_other, len(current_buf))
            parts.append(current_buf.sample(extra))

        batch: Dict[str, np.ndarray] = {}
        for key in parts[0]:
            batch[key] = np.concatenate([p[key] for p in parts], axis=0)

        # Shuffle so current-regime transitions are not always first
        perm = np.random.permutation(len(batch["obs"]))
        return {k: v[perm] for k, v in batch.items()}

    # ------------------------------------------------------------------
    def total_size(self) -> int:
        return sum(len(b) for b in self._buffers.values())

    def regime_sizes(self) -> Dict[int, int]:
        return {rid: len(b) for rid, b in self._buffers.items()}

    def __repr__(self) -> str:
        sizes = self.regime_sizes()
        return f"RegimeAwareExperienceStore(regime_sizes={sizes})"


# ---------------------------------------------------------------------------
# EWC Regularizer
# ---------------------------------------------------------------------------

class EWCRegularizer:
    """
    Elastic Weight Consolidation (diagonal Fisher approximation).

    After training on regime k, call ``consolidate(model, dataset)`` to
    record θ* and the diagonal Fisher F.  During subsequent fine-tuning
    add ``ewc_loss(model)`` to the PPO/SAC loss.

    ewc_loss = Σ_i  F_i · (θ_i − θ*_i)²  · λ / 2

    Multiple consolidations are supported — the penalty accumulates over
    all past regimes (sum of per-regime EWC terms).

    Parameters
    ----------
    ewc_lambda:
        Regularization strength.  0 disables EWC.  Recommended: 0.1 – 1.0.
    n_fisher_samples:
        Number of transitions to use when estimating the Fisher.
    """

    def __init__(self, ewc_lambda: float = 0.4, n_fisher_samples: int = 512) -> None:
        self.ewc_lambda = ewc_lambda
        self.n_fisher_samples = n_fisher_samples
        # List of (theta_star, fisher_diag) per consolidation
        self._consolidations: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = []

    # ------------------------------------------------------------------
    def consolidate(
        self,
        model: nn.Module,
        obs_tensors: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ) -> None:
        """
        Record θ* and estimate diagonal Fisher from ``obs_tensors``.

        If ``obs_tensors`` is None the Fisher is assumed to be uniform
        (all ones), which reduces EWC to a simple L2 penalty toward θ*.

        Parameters
        ----------
        model:
            The policy/value network whose parameters to protect.
        obs_tensors:
            Tensor of shape (N, obs_dim) used to estimate Fisher.
            Typically a sample from the experience store.
        device:
            Torch device.
        """
        model.eval()
        model.to(device)

        # Save θ*
        theta_star: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                theta_star[name] = param.detach().clone()

        # Estimate diagonal Fisher
        fisher: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)

        if obs_tensors is not None and len(obs_tensors) > 0:
            n = min(self.n_fisher_samples, len(obs_tensors))
            idx = torch.randperm(len(obs_tensors))[:n]
            sample = obs_tensors[idx].to(device).float()

            model.zero_grad()
            try:
                out = model(sample)
                # Use log-prob of the output as the "log-likelihood" proxy
                if isinstance(out, tuple):
                    out = out[0]
                log_prob = torch.log_softmax(out, dim=-1) if out.dim() > 1 else out
                loss = log_prob.mean()
                loss.backward()
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher[name] += (param.grad.detach() ** 2) / n
            except Exception as exc:
                logger.warning("Fisher estimation failed (%s). Using uniform Fisher.", exc)
                for name in fisher:
                    fisher[name] = torch.ones_like(fisher[name])
        else:
            logger.info("No obs_tensors provided; using uniform Fisher (L2 penalty).")
            for name in fisher:
                fisher[name] = torch.ones_like(fisher[name])

        self._consolidations.append((theta_star, fisher))
        logger.info(
            "EWC consolidation #%d complete. Protecting %d parameter tensors.",
            len(self._consolidations),
            len(theta_star),
        )

    # ------------------------------------------------------------------
    def ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Compute the total EWC penalty across all past consolidations.

        Returns a scalar tensor (0.0 if no consolidations yet or λ=0).
        """
        if not self._consolidations or self.ewc_lambda == 0.0:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0)
        for theta_star, fisher in self._consolidations:
            for name, param in model.named_parameters():
                if name in theta_star and param.requires_grad:
                    th = theta_star[name].to(param.device)
                    fi = fisher[name].to(param.device)
                    loss = loss + (fi * (param - th) ** 2).sum()

        return self.ewc_lambda / 2.0 * loss

    # ------------------------------------------------------------------
    @property
    def n_consolidations(self) -> int:
        return len(self._consolidations)

    def clear(self) -> None:
        """Remove all consolidation history."""
        self._consolidations.clear()
