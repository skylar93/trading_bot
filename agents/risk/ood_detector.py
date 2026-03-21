"""VAE-based Out-of-Distribution Detector (Abstain Gate).

Trains a Variational Autoencoder on the observation space seen during
training.  At inference time, high reconstruction error signals that the
current market state is outside the learned distribution → the ensemble
should abstain (reduce position towards cash).

Research basis: FineFT (KDD '26) showed ~40% risk reduction by refusing
to trade when the market state is OOD.

Usage
-----
    ood = VAEOODDetector(obs_dim=100, latent_dim=16)
    ood.fit(training_observations)               # (N, obs_dim)
    signal = ood.get_abstain_signal(new_obs)      # 0.0 = normal, 1.0 = OOD
    final_action = meta_action * (1 - signal)     # scale down when OOD
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class _VAE(nn.Module):
    """Simple VAE: Encoder → (mu, logvar) → z → Decoder → x_hat."""

    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def _vae_loss(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """ELBO loss = reconstruction + KL divergence."""
    recon = nn.functional.mse_loss(x_hat, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl


class VAEOODDetector:
    """VAE-based OOD detector that produces an abstain signal.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of (flattened) observations.
    latent_dim : int
        VAE latent space size.
    hidden_dim : int
        Hidden layer width.
    threshold_percentile : float
        Percentile of training reconstruction errors used as the OOD
        threshold (default 95 → top 5% of training errors are "OOD").
    device : str, optional
        Torch device.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        threshold_percentile: float = 95.0,
        device: Optional[str] = None,
    ):
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.threshold_percentile = threshold_percentile
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._vae = _VAE(obs_dim, latent_dim, hidden_dim).to(self.device)
        self._threshold: float = float("inf")
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._fitted = False

    def fit(
        self,
        observations: np.ndarray,
        epochs: int = 50,
        batch_size: int = 128,
        lr: float = 1e-3,
    ) -> "VAEOODDetector":
        """Train VAE on in-distribution observations.

        Parameters
        ----------
        observations : (N, obs_dim) array of training observations.
        """
        obs = np.asarray(observations, dtype=np.float32)
        if obs.ndim == 3:
            # Flatten (N, window, features) → (N, window*features)
            obs = obs.reshape(obs.shape[0], -1)

        assert obs.shape[1] == self.obs_dim, (
            f"Expected obs_dim={self.obs_dim}, got {obs.shape[1]}"
        )

        # Normalize
        self._mean = obs.mean(axis=0)
        self._std = obs.std(axis=0) + 1e-8
        obs_norm = (obs - self._mean) / self._std

        dataset = TensorDataset(torch.tensor(obs_norm, device=self.device))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self._vae.parameters(), lr=lr)

        self._vae.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for (batch,) in loader:
                x_hat, mu, logvar = self._vae(batch)
                loss = _vae_loss(batch, x_hat, mu, logvar)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.debug("VAE epoch %d/%d — loss=%.2f", epoch + 1, epochs, total_loss / len(obs))

        # Compute reconstruction error threshold from training set
        self._vae.eval()
        with torch.no_grad():
            all_obs = torch.tensor(obs_norm, device=self.device)
            x_hat, _, _ = self._vae(all_obs)
            errors = ((all_obs - x_hat) ** 2).mean(dim=1).cpu().numpy()

        self._threshold = float(np.percentile(errors, self.threshold_percentile))
        self._fitted = True

        logger.info(
            "VAEOODDetector fitted — %d observations, threshold=%.6f (p%.0f)",
            len(obs), self._threshold, self.threshold_percentile,
        )
        return self

    def reconstruction_error(self, observation: np.ndarray) -> float:
        """Mean squared reconstruction error for a single observation."""
        if not self._fitted:
            return 0.0

        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 2:
            obs = obs.flatten()
        obs_norm = (obs - self._mean) / self._std

        self._vae.eval()
        with torch.no_grad():
            x = torch.tensor(obs_norm, device=self.device).unsqueeze(0)
            x_hat, _, _ = self._vae(x)
            error = ((x - x_hat) ** 2).mean().item()
        return error

    def is_ood(self, observation: np.ndarray) -> Tuple[bool, float]:
        """Check if observation is out-of-distribution.

        Returns
        -------
        (is_ood, reconstruction_error) tuple.
        """
        error = self.reconstruction_error(observation)
        return error > self._threshold, error

    def get_abstain_signal(self, observation: np.ndarray) -> float:
        """Smooth abstain signal in [0, 1].

        0.0 = clearly in-distribution (no abstention needed)
        1.0 = clearly OOD (should go to cash)

        Uses sigmoid scaling around the threshold for smooth transitions.
        """
        if not self._fitted:
            return 0.0

        error = self.reconstruction_error(observation)

        # Sigmoid centered at threshold, with steepness based on threshold scale
        k = 5.0 / max(self._threshold, 1e-8)
        signal = 1.0 / (1.0 + np.exp(-k * (error - self._threshold)))
        return float(np.clip(signal, 0.0, 1.0))

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def save(self, path: str) -> None:
        torch.save({
            "vae_state": self._vae.state_dict(),
            "threshold": self._threshold,
            "mean": self._mean,
            "std": self._std,
            "config": {
                "obs_dim": self.obs_dim,
                "latent_dim": self.latent_dim,
                "hidden_dim": self.hidden_dim,
                "threshold_percentile": self.threshold_percentile,
            },
        }, path)
        logger.info("VAEOODDetector saved to %s", path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "VAEOODDetector":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        detector = cls(
            obs_dim=cfg["obs_dim"],
            latent_dim=cfg["latent_dim"],
            hidden_dim=cfg["hidden_dim"],
            threshold_percentile=cfg["threshold_percentile"],
            device=device,
        )
        detector._vae.load_state_dict(ckpt["vae_state"])
        detector._threshold = ckpt["threshold"]
        detector._mean = ckpt["mean"]
        detector._std = ckpt["std"]
        detector._fitted = True
        logger.info("VAEOODDetector loaded from %s", path)
        return detector
