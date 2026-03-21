"""
Conservative Q-Learning (CQL) for offline RL — trading baseline.

CQL adds a conservative penalty on top of standard TD learning:

    J_CQL(Q) = TD_loss + α · (E[log Σ_a exp(Q(s,a))] - E[Q(s, a_data)])

The logsumexp is approximated by sampling ``n_action_samples`` random actions
uniformly from [-1, 1]^act_dim at each state.

This gives a simple offline RL baseline that can be compared against the
Decision Transformer on the same trajectory datasets.

Architecture
------------
- Two independent Q-networks (+ target copies) for variance reduction
- Soft target-network updates (Polyak averaging, τ)
- Action selection at inference: sample N random actions, return the one
  with the highest min(Q1, Q2) estimate

Usage::

    config = CQLConfig(state_dim=100, act_dim=1)
    agent  = CQLAgent(config)

    # Offline training from a trajectory dataset
    metrics = agent.train(dataset, n_epochs=10)
    # → {"train_td_loss": [...], "train_cql_loss": [...], "train_total_loss": [...]}

    # Inference (greedy w.r.t. Q)
    action = agent.get_action(state_array)   # → np.ndarray of shape (act_dim,)

    # Persistence
    agent.save("cql_checkpoint.pt")
    agent2 = CQLAgent.load("cql_checkpoint.pt")
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from agents.offline.trajectory_dataset import TradingTrajectoryDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CQLConfig:
    """Hyper-parameters for the CQL offline RL agent."""

    # --- Environment dims ---------------------------------------------------
    state_dim: int = 100   # flattened observation dimension
    act_dim: int = 1       # action dimension

    # --- Q-network architecture --------------------------------------------
    hidden_size: int = 256  # width of each hidden layer
    n_layers: int = 2       # number of hidden layers

    # --- Training -----------------------------------------------------------
    learning_rate: float = 3e-4
    gamma: float = 0.99          # discount factor
    tau: float = 5e-3            # Polyak soft-update coefficient
    alpha: float = 1.0           # CQL conservative penalty weight
    n_action_samples: int = 10   # samples used for logsumexp CQL penalty
    batch_size: int = 256

    # --- Inference ----------------------------------------------------------
    n_inference_samples: int = 50  # random actions to evaluate at inference

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if not (0.0 < self.tau <= 1.0):
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")


# ---------------------------------------------------------------------------
# Q-network
# ---------------------------------------------------------------------------

class _QNetwork(nn.Module):
    """MLP that maps (state, action) → scalar Q-value."""

    def __init__(self, state_dim: int, act_dim: int, hidden_size: int, n_layers: int) -> None:
        super().__init__()
        in_dim = state_dim + act_dim
        layers: List[nn.Module] = []
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_size), nn.ReLU()]
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state:  (B, state_dim)
        action: (B, act_dim)

        Returns
        -------
        q: (B, 1)
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# CQL Agent
# ---------------------------------------------------------------------------

class CQLAgent:
    """
    Conservative Q-Learning offline RL agent for trading.

    Parameters
    ----------
    config: CQLConfig
    device: str
        "cpu" or "cuda" or "auto" (auto-selects CUDA if available)
    """

    def __init__(
        self,
        config: CQLConfig,
        device: str = "auto",
    ) -> None:
        self.config = config
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        cfg = config
        self.q1 = _QNetwork(cfg.state_dim, cfg.act_dim, cfg.hidden_size, cfg.n_layers).to(device)
        self.q2 = _QNetwork(cfg.state_dim, cfg.act_dim, cfg.hidden_size, cfg.n_layers).to(device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        # Targets are not trained directly
        for p in self.q1_target.parameters():
            p.requires_grad_(False)
        for p in self.q2_target.parameters():
            p.requires_grad_(False)

        params = list(self.q1.parameters()) + list(self.q2.parameters())
        self.optimizer = torch.optim.Adam(params, lr=cfg.learning_rate)

    # ------------------------------------------------------------------
    # Target network soft update
    # ------------------------------------------------------------------

    def _soft_update(self) -> None:
        tau = self.config.tau
        for target, src in zip(self.q1_target.parameters(), self.q1.parameters()):
            target.data.copy_(tau * src.data + (1.0 - tau) * target.data)
        for target, src in zip(self.q2_target.parameters(), self.q2.parameters()):
            target.data.copy_(tau * src.data + (1.0 - tau) * target.data)

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def train_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        """
        One gradient step on a batch of transitions.

        All tensors must already be on ``self.device``.

        Parameters
        ----------
        states:      (B, state_dim)
        actions:     (B, act_dim)
        rewards:     (B,)
        next_states: (B, state_dim)
        dones:       (B,)   — 1.0 at terminal steps

        Returns
        -------
        dict with keys: "td_loss", "cql_loss", "total_loss"
        """
        cfg = self.config
        B = states.shape[0]

        # ── 1. TD target ──────────────────────────────────────────────
        with torch.no_grad():
            # Sample random next actions for the target (simplified; no policy needed)
            next_actions = torch.empty(B, cfg.act_dim, device=self.device).uniform_(-1.0, 1.0)
            target_q = torch.min(
                self.q1_target(next_states, next_actions),
                self.q2_target(next_states, next_actions),
            ).squeeze(-1)  # (B,)
            td_target = rewards + cfg.gamma * (1.0 - dones) * target_q  # (B,)

        q1_pred = self.q1(states, actions).squeeze(-1)  # (B,)
        q2_pred = self.q2(states, actions).squeeze(-1)  # (B,)
        td_loss = F.mse_loss(q1_pred, td_target) + F.mse_loss(q2_pred, td_target)

        # ── 2. CQL conservative penalty ───────────────────────────────
        # Approximate logsumexp over action space by sampling random actions
        # Shape: (B, n_action_samples, act_dim)
        rand_acts = torch.empty(
            B, cfg.n_action_samples, cfg.act_dim, device=self.device
        ).uniform_(-1.0, 1.0)

        # Expand states: (B, n_action_samples, state_dim)
        states_exp = states.unsqueeze(1).expand(B, cfg.n_action_samples, cfg.state_dim)

        # Flatten for Q-network: (B*n, state_dim), (B*n, act_dim)
        s_flat = states_exp.reshape(B * cfg.n_action_samples, cfg.state_dim)
        a_flat = rand_acts.reshape(B * cfg.n_action_samples, cfg.act_dim)

        q1_rand = self.q1(s_flat, a_flat).reshape(B, cfg.n_action_samples)  # (B, n)
        q2_rand = self.q2(s_flat, a_flat).reshape(B, cfg.n_action_samples)  # (B, n)

        # logsumexp - Q(s, a_data): encourage lower Q outside data distribution
        cql_loss = (
            torch.logsumexp(q1_rand, dim=1).mean() - q1_pred.mean()
            + torch.logsumexp(q2_rand, dim=1).mean() - q2_pred.mean()
        )

        # ── 3. Combined loss ──────────────────────────────────────────
        total_loss = td_loss + cfg.alpha * cql_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self._soft_update()

        return {
            "td_loss": td_loss.item(),
            "cql_loss": cql_loss.item(),
            "total_loss": total_loss.item(),
        }

    # ------------------------------------------------------------------
    # Dataset-level training
    # ------------------------------------------------------------------

    def _dataset_to_transitions(
        self, dataset: TradingTrajectoryDataset
    ) -> TensorDataset:
        """
        Convert TradingTrajectoryDataset context windows into (s, a, r, s', done) tuples.

        The last real timestep in each window is used as the transition:
            s  = context[-1] state
            a  = context[-1] action
            r  = reward of context[-1] (estimated from RTG change)
            s' = "next" state (we reuse s here as a simple approximation for
                 offline datasets that lack explicit next-state info)
            done = 0.0

        For datasets converted from raw rollouts, we use the per-step rewards
        computed inside the dataset's RTG arrays.
        """
        states_list: List[torch.Tensor] = []
        actions_list: List[torch.Tensor] = []
        rewards_list: List[torch.Tensor] = []
        next_states_list: List[torch.Tensor] = []
        dones_list: List[torch.Tensor] = []

        for i in range(len(dataset)):
            sample = dataset[i]
            mask = sample["attention_mask"]       # (K,) 0/1 float
            states = sample["states"]             # (K, state_dim)
            actions = sample["actions"]           # (K, act_dim)
            rtg = sample["returns_to_go"]         # (K, 1)

            # Valid (non-padded) indices
            valid_idx = (mask > 0.5).nonzero(as_tuple=True)[0]
            if len(valid_idx) < 2:
                continue

            # Use transitions at all valid positions except the last
            for k in valid_idx[:-1]:
                k = k.item()
                s = states[k]                      # (state_dim,)
                a = actions[k]                     # (act_dim,)
                # reward ≈ RTG[k] - RTG[k+1]
                r = (rtg[k] - rtg[k + 1]).squeeze(-1)  # scalar
                s_next = states[k + 1]             # (state_dim,)
                d = torch.tensor(0.0)

                states_list.append(s)
                actions_list.append(a)
                rewards_list.append(r)
                next_states_list.append(s_next)
                dones_list.append(d)

        if not states_list:
            raise ValueError("Dataset produced zero valid transitions.")

        return TensorDataset(
            torch.stack(states_list),
            torch.stack(actions_list),
            torch.stack(rewards_list),
            torch.stack(next_states_list),
            torch.stack(dones_list),
        )

    def train(
        self,
        dataset: TradingTrajectoryDataset,
        n_epochs: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Train for ``n_epochs`` full passes over the dataset.

        Returns
        -------
        dict with keys "train_td_loss", "train_cql_loss", "train_total_loss"
        Each value is a list of per-epoch mean losses.
        """
        transition_ds = self._dataset_to_transitions(dataset)
        loader = DataLoader(
            transition_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        td_losses: List[float] = []
        cql_losses: List[float] = []
        total_losses: List[float] = []

        for epoch in range(n_epochs):
            ep_td, ep_cql, ep_total, n = 0.0, 0.0, 0.0, 0
            for batch in loader:
                s, a, r, s_next, d = [t.to(self.device) for t in batch]
                metrics = self.train_batch(s, a, r, s_next, d)
                ep_td += metrics["td_loss"]
                ep_cql += metrics["cql_loss"]
                ep_total += metrics["total_loss"]
                n += 1
            td_losses.append(ep_td / max(n, 1))
            cql_losses.append(ep_cql / max(n, 1))
            total_losses.append(ep_total / max(n, 1))
            logger.debug(
                "Epoch %d/%d  total_loss=%.6f  td=%.6f  cql=%.6f",
                epoch + 1, n_epochs,
                total_losses[-1], td_losses[-1], cql_losses[-1],
            )

        return {
            "train_td_loss": td_losses,
            "train_cql_loss": cql_losses,
            "train_total_loss": total_losses,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select the action with the highest conservative Q-value estimate.

        Samples ``config.n_inference_samples`` random actions from [-1, 1]^act_dim
        and returns the one maximising min(Q1, Q2).

        Parameters
        ----------
        state: np.ndarray of shape (state_dim,)

        Returns
        -------
        np.ndarray of shape (act_dim,)
        """
        self.q1.eval()
        self.q2.eval()

        cfg = self.config
        state_t = torch.from_numpy(
            np.asarray(state, dtype=np.float32)
        ).unsqueeze(0).to(self.device)  # (1, state_dim)

        n = cfg.n_inference_samples
        rand_acts = torch.empty(n, cfg.act_dim, device=self.device).uniform_(-1.0, 1.0)
        states_exp = state_t.expand(n, cfg.state_dim)  # (n, state_dim)

        q_min = torch.min(
            self.q1(states_exp, rand_acts),
            self.q2(states_exp, rand_acts),
        ).squeeze(-1)  # (n,)

        best_idx = q_min.argmax().item()
        return rand_acts[best_idx].cpu().numpy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save config + network weights to ``path`` (torch format)."""
        torch.save(
            {
                "config": self.config,
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_target": self.q1_target.state_dict(),
                "q2_target": self.q2_target.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "CQLAgent":
        """Load a checkpoint saved with :meth:`save`."""
        data = torch.load(path, map_location=map_location, weights_only=False)
        agent = cls(data["config"], device=map_location)
        agent.q1.load_state_dict(data["q1"])
        agent.q2.load_state_dict(data["q2"])
        agent.q1_target.load_state_dict(data["q1_target"])
        agent.q2_target.load_state_dict(data["q2_target"])
        return agent

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "CQLAgent":
        """Create from the project's unified YAML config dict."""
        cql = config_dict.get("cql", {})
        cfg = CQLConfig(
            state_dim=cql.get("state_dim", 100),
            act_dim=cql.get("act_dim", 1),
            hidden_size=cql.get("hidden_size", 256),
            n_layers=cql.get("n_layers", 2),
            learning_rate=cql.get("learning_rate", 3e-4),
            gamma=cql.get("gamma", 0.99),
            tau=cql.get("tau", 5e-3),
            alpha=cql.get("alpha", 1.0),
            n_action_samples=cql.get("n_action_samples", 10),
            batch_size=cql.get("batch_size", 256),
            n_inference_samples=cql.get("n_inference_samples", 50),
        )
        return cls(cfg, device=cql.get("device", "auto"))
