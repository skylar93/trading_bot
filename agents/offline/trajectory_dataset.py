"""
Expert trajectory storage and dataset for offline RL pre-training.

A *trajectory* is a single episode: arrays of observations, actions, rewards,
and done flags.  The dataset slices each trajectory into overlapping context
windows of length K and computes return-to-go (RTG) for each window.

Usage::

    # Build from rollout arrays
    dataset = TradingTrajectoryDataset.from_rollouts(
        observations, actions, rewards, dones,
        context_len=20,
    )

    # Access one sample (returns tensors of shape (K, …))
    sample = dataset[0]
    # Keys: "states"  (K, state_dim)
    #       "actions" (K, act_dim)
    #       "returns_to_go" (K, 1)
    #       "timesteps"     (K,)  long
    #       "attention_mask"(K,)  float  — 0 for padded, 1 for real

    # Persist
    dataset.save("trajectories.pkl")
    dataset2 = TradingTrajectoryDataset.load("trajectories.pkl")
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Trajectory dataclass
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """One episode's worth of experience."""

    observations: np.ndarray   # (T, obs_dim)  float32
    actions: np.ndarray        # (T, act_dim)  float32  (or (T,) for scalar actions)
    rewards: np.ndarray        # (T,)           float32
    dones: np.ndarray          # (T,)           float32  (0 or 1)

    def __post_init__(self) -> None:
        self.observations = np.asarray(self.observations, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        self.rewards = np.asarray(self.rewards, dtype=np.float32)
        self.dones = np.asarray(self.dones, dtype=np.float32)

        # Ensure actions are 2-D: (T, act_dim)
        if self.actions.ndim == 1:
            self.actions = self.actions[:, np.newaxis]

        T = len(self.rewards)
        if len(self.observations) != T:
            raise ValueError("observations and rewards length mismatch")
        if len(self.actions) != T:
            raise ValueError("actions and rewards length mismatch")
        if len(self.dones) != T:
            raise ValueError("dones and rewards length mismatch")

    def __len__(self) -> int:
        return len(self.rewards)

    @property
    def obs_dim(self) -> int:
        return int(np.prod(self.observations.shape[1:]))

    @property
    def act_dim(self) -> int:
        return self.actions.shape[1]

    def compute_rtg(self, gamma: float = 1.0) -> np.ndarray:
        """
        Compute discounted return-to-go for every timestep.

        RTG[t] = Σ_{k=t}^{T-1}  γ^{k-t} · r[k]   (episode boundary at done)
        """
        T = len(self.rewards)
        rtg = np.zeros(T, dtype=np.float64)
        cumulative = 0.0
        for t in reversed(range(T)):
            if t < T - 1 and self.dones[t]:
                cumulative = 0.0
            cumulative = float(self.rewards[t]) + gamma * cumulative
            rtg[t] = cumulative
        return rtg.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TradingTrajectoryDataset(Dataset):
    """
    Decision Transformer dataset.

    Each sample is a context window of K consecutive timesteps from one
    trajectory, formatted as::

        {
          "states":         (K, obs_dim)   float32
          "actions":        (K, act_dim)   float32
          "returns_to_go":  (K, 1)         float32
          "timesteps":      (K,)           int64
          "attention_mask": (K,)           float32   1=real, 0=padded
        }

    Short windows (at the start of a trajectory) are left-padded with zeros.

    Parameters
    ----------
    trajectories:
        List of :class:`Trajectory` objects.
    context_len:
        Context window length K.
    gamma:
        Discount factor for computing RTG.
    normalize_states:
        If True, z-score normalise observations using dataset-wide mean/std.
    normalize_returns:
        If True, divide RTG by ``max(|RTG|) + ε`` so values ∈ (-1, 1].
    """

    def __init__(
        self,
        trajectories: List[Trajectory],
        context_len: int = 20,
        gamma: float = 1.0,
        normalize_states: bool = True,
        normalize_returns: bool = True,
    ) -> None:
        if not trajectories:
            raise ValueError("trajectories list must not be empty")

        self.context_len = context_len
        self.gamma = gamma
        self.normalize_states = normalize_states
        self.normalize_returns = normalize_returns

        # Compute RTG for all trajectories first
        self.trajectories = trajectories
        self.rtgs: List[np.ndarray] = [t.compute_rtg(gamma) for t in trajectories]

        # Normalisation statistics
        all_obs = np.concatenate([t.observations.reshape(len(t), -1) for t in trajectories], axis=0)
        all_rtg = np.concatenate(self.rtgs)

        if normalize_states:
            self.obs_mean: np.ndarray = all_obs.mean(axis=0).astype(np.float32)
            self.obs_std: np.ndarray = (all_obs.std(axis=0) + 1e-6).astype(np.float32)
        else:
            self.obs_mean = np.zeros(all_obs.shape[1], dtype=np.float32)
            self.obs_std = np.ones(all_obs.shape[1], dtype=np.float32)

        if normalize_returns:
            self.rtg_scale = float(np.abs(all_rtg).max()) + 1e-6
        else:
            self.rtg_scale = 1.0

        # Build flat index: list of (traj_idx, end_t)
        self._windows: List[Tuple[int, int]] = []
        for i, traj in enumerate(trajectories):
            for t in range(len(traj)):
                self._windows.append((i, t))

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx, end_t = self._windows[idx]
        traj = self.trajectories[traj_idx]
        rtg = self.rtgs[traj_idx]
        K = self.context_len

        # Slice context window [start_t … end_t] (inclusive)
        start_t = max(0, end_t - K + 1)
        actual_len = end_t - start_t + 1  # ≤ K

        # Flatten observations for the window
        obs_win = traj.observations[start_t : end_t + 1].reshape(actual_len, -1)
        act_win = traj.actions[start_t : end_t + 1]                    # (actual_len, act_dim)
        rtg_win = rtg[start_t : end_t + 1]                             # (actual_len,)
        ts_win = np.arange(start_t, end_t + 1, dtype=np.int64)         # (actual_len,)

        # Normalise
        obs_win = (obs_win - self.obs_mean) / self.obs_std
        rtg_win = rtg_win / self.rtg_scale

        # Left-pad to exactly K timesteps
        pad_len = K - actual_len
        obs_dim = obs_win.shape[1]
        act_dim = act_win.shape[1]

        if pad_len > 0:
            obs_win = np.concatenate([np.zeros((pad_len, obs_dim), dtype=np.float32), obs_win])
            act_win = np.concatenate([np.zeros((pad_len, act_dim), dtype=np.float32), act_win])
            rtg_win = np.concatenate([np.zeros(pad_len, dtype=np.float32), rtg_win])
            ts_win = np.concatenate([np.zeros(pad_len, dtype=np.int64), ts_win])
            mask = np.array([0.0] * pad_len + [1.0] * actual_len, dtype=np.float32)
        else:
            mask = np.ones(K, dtype=np.float32)

        return {
            "states": torch.tensor(obs_win, dtype=torch.float32),                  # (K, obs_dim)
            "actions": torch.tensor(act_win, dtype=torch.float32),                 # (K, act_dim)
            "returns_to_go": torch.tensor(rtg_win, dtype=torch.float32).unsqueeze(-1),  # (K, 1)
            "timesteps": torch.tensor(ts_win, dtype=torch.long),                   # (K,)
            "attention_mask": torch.tensor(mask, dtype=torch.float32),             # (K,)
        }

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        return int(np.prod(self.trajectories[0].observations.shape[1:]))

    @property
    def act_dim(self) -> int:
        return self.trajectories[0].act_dim

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_rollouts(
        cls,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        context_len: int = 20,
        **kwargs,
    ) -> "TradingTrajectoryDataset":
        """
        Build a dataset from flat rollout arrays.

        Episodes are split at ``done=True`` boundaries.

        Parameters
        ----------
        observations:
            (N, obs_dim) or (N, window_size, n_features)
        actions:
            (N,) or (N, act_dim)
        rewards:
            (N,)
        dones:
            (N,) — 1 where an episode ends, 0 otherwise
        """
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        N = len(rewards)
        trajectories: List[Trajectory] = []
        start = 0

        for t in range(N):
            is_last = (t == N - 1)
            if dones[t] or is_last:
                end = t + 1
                if end > start:
                    trajectories.append(
                        Trajectory(
                            observations=observations[start:end],
                            actions=actions[start:end],
                            rewards=rewards[start:end],
                            dones=dones[start:end],
                        )
                    )
                start = end

        if not trajectories:
            raise ValueError("No complete trajectories found in rollout arrays")

        return cls(trajectories, context_len=context_len, **kwargs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Pickle the dataset to ``path``."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "trajectories": self.trajectories,
                    "context_len": self.context_len,
                    "gamma": self.gamma,
                    "normalize_states": self.normalize_states,
                    "normalize_returns": self.normalize_returns,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "TradingTrajectoryDataset":
        """Load a previously saved dataset from ``path``."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(
            trajectories=data["trajectories"],
            context_len=data["context_len"],
            gamma=data.get("gamma", 1.0),
            normalize_states=data.get("normalize_states", True),
            normalize_returns=data.get("normalize_returns", True),
        )
