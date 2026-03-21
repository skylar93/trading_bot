"""
Diffusion-based trajectory data augmentation for offline RL.

Overview
--------
Implements a simplified Gaussian denoising diffusion process for augmenting
expert trading trajectories.  A small MLP denoising network learns the
score function; at inference time it reverses the diffusion to generate
new trajectory variants that preserve the statistical structure of the
originals while adding diversity.

Design choices
--------------
- State-only diffusion: only observation arrays are diffused.
  Actions and rewards are kept from the source trajectory (option: jitter).
- Linear beta schedule (cosine optional via config).
- Denoising network: 2-layer MLP with sinusoidal time-step embedding.
- Pure-PyTorch, no diffusers library dependency.

Usage
-----
    aug = TradingDiffusionAugmentor(obs_dim=20)
    aug.fit(dataset, n_epochs=50)

    new_traj = aug.augment(source_traj)         # one augmented trajectory
    new_ds   = aug.augment_dataset(dataset, n_aug=3)  # 3× dataset

    aug.save("diffusion.pt")
    aug2 = TradingDiffusionAugmentor.load("diffusion.pt")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from agents.offline.trajectory_dataset import Trajectory, TradingTrajectoryDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DiffusionConfig:
    """Hyper-parameters for TradingDiffusionAugmentor."""

    # Diffusion schedule
    n_diffusion_steps: int = 50       # T — number of forward/reverse steps
    beta_start: float = 1e-4          # β₁
    beta_end: float = 0.02            # β_T
    schedule: str = "linear"          # "linear" | "cosine"

    # Denoising network
    hidden_dim: int = 128
    n_layers: int = 2                 # hidden layers (excluding in/out)
    time_emb_dim: int = 32            # sinusoidal time embedding size

    # Training
    lr: float = 1e-3
    batch_size: int = 64
    n_epochs: int = 100

    # Augmentation
    jitter_actions: bool = False      # add small noise to actions too
    action_noise_std: float = 0.01
    reward_noise_std: float = 0.0     # 0 = no reward noise


# ---------------------------------------------------------------------------
# Sinusoidal time-step embedding
# ---------------------------------------------------------------------------

def _sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal embedding for diffusion timestep t.

    Parameters
    ----------
    timesteps : (B,) long tensor of step indices in [0, T)
    dim       : embedding dimensionality (must be even)

    Returns
    -------
    emb : (B, dim) float tensor
    """
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, device=timesteps.device, dtype=torch.float32)
        * (np.log(10000.0) / (half - 1))
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


# ---------------------------------------------------------------------------
# Denoising MLP
# ---------------------------------------------------------------------------

class _DenoisingMLP(nn.Module):
    """
    Predicts the original clean signal x₀ from noisy xₜ and step t.

    Input  : [xₜ || time_emb]   shape (B, obs_dim + time_emb_dim)
    Output : x̂₀                 shape (B, obs_dim)
    """

    def __init__(self, obs_dim: int, cfg: DiffusionConfig) -> None:
        super().__init__()
        in_dim = obs_dim + cfg.time_emb_dim
        layers: list = [nn.Linear(in_dim, cfg.hidden_dim), nn.SiLU()]
        for _ in range(cfg.n_layers - 1):
            layers += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(cfg.hidden_dim, obs_dim))
        self.net = nn.Sequential(*layers)
        self.time_emb_dim = cfg.time_emb_dim

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = _sinusoidal_embedding(t, self.time_emb_dim)
        h = torch.cat([x_noisy, t_emb], dim=-1)
        return self.net(h)


# ---------------------------------------------------------------------------
# TradingDiffusionAugmentor
# ---------------------------------------------------------------------------

class TradingDiffusionAugmentor:
    """
    Denoising diffusion model for trading trajectory augmentation.

    Parameters
    ----------
    obs_dim : int
        Observation dimensionality.
    config : DiffusionConfig, optional
        Hyper-parameters; defaults to ``DiffusionConfig()``.
    device : str, optional
        Torch device (auto-selected by default).

    Attributes
    ----------
    is_fitted : bool
        True after ``fit()`` completes successfully.
    """

    def __init__(
        self,
        obs_dim: int,
        config: Optional[DiffusionConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.cfg = config or DiffusionConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.is_fitted = False

        # Build noise schedule
        self._build_schedule()

        # Denoising network
        self._net = _DenoisingMLP(obs_dim, self.cfg).to(self.device)
        self._optimizer = optim.Adam(self._net.parameters(), lr=self.cfg.lr)

    # ------------------------------------------------------------------
    # Noise schedule
    # ------------------------------------------------------------------

    def _build_schedule(self) -> None:
        T = self.cfg.n_diffusion_steps
        if self.cfg.schedule == "cosine":
            # Cosine schedule (Nichol & Dhariwal 2021)
            s = 0.008
            steps = np.linspace(0, T, T + 1)
            f = np.cos((steps / T + s) / (1 + s) * np.pi / 2) ** 2
            alphas_cumprod = f / f[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            betas = np.clip(betas, 0, 0.999)
        else:
            betas = np.linspace(self.cfg.beta_start, self.cfg.beta_end, T)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        self._betas = torch.tensor(betas, dtype=torch.float32)
        self._alphas = torch.tensor(alphas, dtype=torch.float32)
        self._alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32)
        self._sqrt_alphas_cumprod = torch.sqrt(self._alphas_cumprod)
        self._sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self._alphas_cumprod)

    def _to_device(self) -> None:
        self._betas = self._betas.to(self.device)
        self._alphas = self._alphas.to(self.device)
        self._alphas_cumprod = self._alphas_cumprod.to(self.device)
        self._sqrt_alphas_cumprod = self._sqrt_alphas_cumprod.to(self.device)
        self._sqrt_one_minus_alphas_cumprod = self._sqrt_one_minus_alphas_cumprod.to(self.device)

    # ------------------------------------------------------------------
    # Forward diffusion (analytical)
    # ------------------------------------------------------------------

    def _q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample xₜ ~ q(xₜ | x₀) using the reparametrisation trick.

        xₜ = √ᾱₜ · x₀ + √(1−ᾱₜ) · ε,   ε ~ N(0, I)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self._sqrt_alphas_cumprod[t].unsqueeze(-1)          # (B, 1)
        sqrt_omc = self._sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_ac * x0 + sqrt_omc * noise

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset: TradingTrajectoryDataset,
        n_epochs: Optional[int] = None,
        verbose: bool = False,
    ) -> "TradingDiffusionAugmentor":
        """
        Train the denoising network on all observations in *dataset*.

        Parameters
        ----------
        dataset : TradingTrajectoryDataset
        n_epochs : int, optional
            Overrides config.n_epochs.
        verbose : bool
            Print loss every 10 epochs.
        """
        n_epochs = n_epochs or self.cfg.n_epochs
        self._to_device()

        # Collect all observation vectors from dataset
        all_obs = self._collect_observations(dataset)
        if len(all_obs) == 0:
            raise ValueError("Dataset contains no observations.")

        X = torch.tensor(all_obs, dtype=torch.float32)  # (N, obs_dim)
        loader = DataLoader(
            TensorDataset(X),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

        self._net.train()
        T = self.cfg.n_diffusion_steps

        for epoch in range(1, n_epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            for (x0_batch,) in loader:
                x0_batch = x0_batch.to(self.device)
                B = x0_batch.shape[0]

                # Random timesteps
                t = torch.randint(0, T, (B,), device=self.device, dtype=torch.long)

                # Forward diffusion
                noise = torch.randn_like(x0_batch)
                x_noisy = self._q_sample(x0_batch, t, noise)

                # Predict x₀ (x0-prediction parameterisation)
                x0_pred = self._net(x_noisy, t)

                loss = nn.functional.mse_loss(x0_pred, x0_batch)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if verbose and epoch % 10 == 0:
                logger.info("Diffusion epoch %d/%d  loss=%.5f", epoch, n_epochs, epoch_loss / max(n_batches, 1))

        self._net.eval()
        self.is_fitted = True
        logger.info("TradingDiffusionAugmentor fitted on %d observations.", len(all_obs))
        return self

    # ------------------------------------------------------------------
    # Inference: reverse diffusion
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _reverse_sample(self, x_T: torch.Tensor) -> torch.Tensor:
        """
        Run DDPM reverse chain from xₜ=x_T down to x₀.

        Uses the simplified posterior mean:
            x̂₀ = net(xₜ, t)
            μ   = (√ᾱ_{t-1} β_t x̂₀  +  √αₜ (1−ᾱ_{t-1}) xₜ) / (1−ᾱₜ)
        """
        x = x_T.clone()
        T = self.cfg.n_diffusion_steps

        for t_idx in reversed(range(T)):
            t_tensor = torch.full((x.shape[0],), t_idx, device=self.device, dtype=torch.long)
            x0_pred = self._net(x, t_tensor)

            if t_idx == 0:
                x = x0_pred
                break

            alpha_t = self._alphas[t_idx]
            alpha_bar_t = self._alphas_cumprod[t_idx]
            alpha_bar_prev = self._alphas_cumprod[t_idx - 1]
            beta_t = self._betas[t_idx]

            # Posterior mean
            coef1 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
            coef2 = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            mu = coef1 * x0_pred + coef2 * x

            # Posterior variance (simplified)
            var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            noise = torch.randn_like(x) * torch.sqrt(var)
            x = mu + noise

        return x

    # ------------------------------------------------------------------
    # Augmentation API
    # ------------------------------------------------------------------

    def augment(
        self,
        trajectory: Trajectory,
        noise_level: float = 0.5,
    ) -> Trajectory:
        """
        Generate one augmented variant of *trajectory*.

        Parameters
        ----------
        trajectory : Trajectory
            Source trajectory to augment.
        noise_level : float in (0, 1]
            How far into the forward process to go (fraction of T).
            Lower values → smaller perturbations (closer to original).

        Returns
        -------
        Trajectory with augmented observations.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before augment().")

        self._to_device()
        T_steps = max(1, int(noise_level * self.cfg.n_diffusion_steps))
        T_steps = min(T_steps, self.cfg.n_diffusion_steps - 1)

        obs = trajectory.observations  # (L, obs_dim)
        x0 = torch.tensor(obs, dtype=torch.float32, device=self.device)

        # Partially noise
        t_tensor = torch.full((len(x0),), T_steps, device=self.device, dtype=torch.long)
        x_noisy = self._q_sample(x0, t_tensor)

        # Denoise back
        self._net.eval()
        x_aug = self._reverse_sample(x_noisy)  # (L, obs_dim)
        obs_aug = x_aug.cpu().numpy()

        # Optionally jitter actions
        actions = trajectory.actions.copy()
        if self.cfg.jitter_actions and self.cfg.action_noise_std > 0:
            actions = actions + np.random.randn(*actions.shape) * self.cfg.action_noise_std

        rewards = trajectory.rewards.copy()
        if self.cfg.reward_noise_std > 0:
            rewards = rewards + np.random.randn(*rewards.shape) * self.cfg.reward_noise_std

        return Trajectory(
            observations=obs_aug.astype(np.float32),
            actions=actions,
            rewards=rewards,
            dones=trajectory.dones.copy(),
        )

    def augment_dataset(
        self,
        dataset: TradingTrajectoryDataset,
        n_aug: int = 1,
        noise_level: float = 0.5,
    ) -> TradingTrajectoryDataset:
        """
        Augment every trajectory in *dataset* ``n_aug`` times and return a
        new ``TradingTrajectoryDataset`` containing originals + augmented.

        Parameters
        ----------
        dataset : TradingTrajectoryDataset
        n_aug : int
            Number of augmented copies per original trajectory.
        noise_level : float
            Passed to ``augment()``.

        Returns
        -------
        TradingTrajectoryDataset  (len = (1 + n_aug) × len(original))
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before augment_dataset().")

        new_trajectories = list(dataset.trajectories)

        for traj in dataset.trajectories:
            for _ in range(n_aug):
                aug = self.augment(traj, noise_level=noise_level)
                new_trajectories.append(aug)

        return TradingTrajectoryDataset(
            new_trajectories,
            context_len=dataset.context_len,
            gamma=dataset.gamma,
            normalize_states=dataset.normalize_states,
            normalize_returns=dataset.normalize_returns,
        )

    def count_parameters(self) -> int:
        """Total trainable parameters in the denoising network."""
        return sum(p.numel() for p in self._net.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "cfg": self.cfg,
                "net_state": self._net.state_dict(),
                "optimizer_state": self._optimizer.state_dict(),
                "is_fitted": self.is_fitted,
                # save schedule tensors on CPU
                "betas": self._betas.cpu(),
                "alphas": self._alphas.cpu(),
                "alphas_cumprod": self._alphas_cumprod.cpu(),
            },
            path,
        )
        logger.info("TradingDiffusionAugmentor saved to %s", path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "TradingDiffusionAugmentor":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        aug = cls(obs_dim=ckpt["obs_dim"], config=ckpt["cfg"], device=device)
        aug._net.load_state_dict(ckpt["net_state"])
        aug._optimizer.load_state_dict(ckpt["optimizer_state"])
        aug.is_fitted = ckpt["is_fitted"]
        aug._betas = ckpt["betas"]
        aug._alphas = ckpt["alphas"]
        aug._alphas_cumprod = ckpt["alphas_cumprod"]
        aug._sqrt_alphas_cumprod = torch.sqrt(aug._alphas_cumprod)
        aug._sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - aug._alphas_cumprod)
        logger.info("TradingDiffusionAugmentor loaded from %s", path)
        return aug

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_observations(self, dataset: TradingTrajectoryDataset) -> np.ndarray:
        """Flatten all observations across all trajectories."""
        all_obs = []
        for traj in dataset.trajectories:
            all_obs.append(traj.observations)  # (T, obs_dim)
        if not all_obs:
            return np.empty((0, self.obs_dim), dtype=np.float32)
        return np.concatenate(all_obs, axis=0)  # (N, obs_dim)
