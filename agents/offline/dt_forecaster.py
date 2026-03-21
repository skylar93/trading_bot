"""
DTForecaster: Decision Transformer repurposed as a return forecaster.

Week 22 implementation.

Instead of outputting actions directly (as the original Decision Transformer does),
this module learns to *predict* future returns from a state history window.
The predictions are used as auxiliary observation features for RL agents, giving
them an explicit "crystal ball" signal without requiring lookahead in the env.

Architecture
------------
- Input  : state_history  (seq_len, state_dim) — recent OHLCV/feature window
- Encoder: causal transformer (shared blocks with Decision Transformer)
- Heads  : two MLPs on top of the final hidden state
    - return_1step: scalar — expected return 1 step ahead
    - return_5step: scalar — expected return 5 steps ahead
- Confidence is derived from a Monte-Carlo dropout ensemble at inference time
  (no separate head needed; just run predict() N times with dropout active).

Training
--------
Supervised learning (walk-forward):
  - Target: realised log-returns at t+1 and t+5
  - Loss  : weighted MSE (w_5step default 0.5, since 5-step is noisier)
  - Walk-forward split: last ``val_frac`` rows of each training segment held out

Usage
-----
    forecaster = DTForecaster(state_dim=5, seq_len=20)
    forecaster.train_supervised(states, returns_1, returns_5, n_epochs=10)

    pred = forecaster.predict(state_history)   # (seq_len, state_dim) ndarray
    # pred == {'return_1step': 0.003, 'return_5step': 0.012, 'confidence': 0.85}

    # Factory from YAML config
    forecaster = DTForecaster.from_config('config/training_config.yaml')
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DTForecasterConfig:
    """Hyper-parameters for DTForecaster."""

    # Architecture
    state_dim: int = 5           # number of features per timestep
    seq_len: int = 20            # input sequence length (must match env window_size)
    hidden_size: int = 64        # transformer embedding dim
    n_layer: int = 2             # number of transformer blocks
    n_head: int = 4              # attention heads (hidden_size divisible by n_head)
    dropout: float = 0.1        # dropout for regularisation AND MC-dropout at test time
    n_inner: Optional[int] = None  # FFN width; None → 4 × hidden_size

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    n_epochs: int = 20
    max_grad_norm: float = 1.0
    w_5step: float = 0.5         # weight for 5-step head in combined MSE loss
    val_frac: float = 0.1        # fraction of data used for validation in walk-forward

    # Inference (Monte-Carlo dropout)
    mc_samples: int = 10         # forward passes for confidence estimation
    confidence_mode: str = "mc"  # "mc" | "entropy" (entropy not implemented yet)

    # Checkpoint
    checkpoint_dir: str = "checkpoints/dt_forecaster"

    def __post_init__(self) -> None:
        if self.hidden_size % self.n_head != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"n_head ({self.n_head})"
            )
        if self.n_inner is None:
            object.__setattr__(self, "n_inner", 4 * self.hidden_size)


# ---------------------------------------------------------------------------
# Causal transformer block (standalone — no dependency on decision_transformer.py)
# ---------------------------------------------------------------------------

class _CausalBlock(nn.Module):
    """Pre-LN causal self-attention + FFN block."""

    def __init__(self, hidden: int, n_head: int, n_inner: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, n_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_inner, hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=causal, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# DTForecaster model
# ---------------------------------------------------------------------------

class _ForecasterNet(nn.Module):
    """Transformer encoder → dual return prediction heads."""

    def __init__(self, cfg: DTForecasterConfig) -> None:
        super().__init__()
        n_inner = cfg.n_inner or (4 * cfg.hidden_size)

        # Input projection: (state_dim,) → (hidden_size,) per timestep
        self.input_proj = nn.Linear(cfg.state_dim, cfg.hidden_size)

        # Sinusoidal positional encoding (fixed, not learned)
        pe = torch.zeros(cfg.seq_len, cfg.hidden_size)
        pos = torch.arange(cfg.seq_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, cfg.hidden_size, 2, dtype=torch.float32)
            * (-math.log(10000.0) / cfg.hidden_size)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, seq_len, hidden_size)

        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            _CausalBlock(cfg.hidden_size, cfg.n_head, n_inner, cfg.dropout)
            for _ in range(cfg.n_layer)
        ])

        self.ln_f = nn.LayerNorm(cfg.hidden_size)

        # Prediction heads — operate on the *last* token's hidden state
        self.head_1step = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size // 2, 1),
        )
        self.head_5step = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, seq_len, state_dim)

        Returns:
            pred_1step: (B,)
            pred_5step: (B,)
        """
        h = self.drop(self.input_proj(x) + self.pe)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)

        # Use last token representation for prediction
        last = h[:, -1, :]  # (B, hidden_size)
        return self.head_1step(last).squeeze(-1), self.head_5step(last).squeeze(-1)


# ---------------------------------------------------------------------------
# DTForecaster (public API)
# ---------------------------------------------------------------------------

class DTForecaster:
    """
    Decision Transformer repurposed as a return forecaster.

    Parameters
    ----------
    state_dim : int
        Feature dimension per timestep (e.g. 5 for raw OHLCV, 18 for engineered).
    seq_len : int
        Input sequence length — must match the environment's window_size.
    config : DTForecasterConfig, optional
        Full config object; if provided, ``state_dim`` and ``seq_len`` are
        taken from it and the explicit arguments are ignored.
    device : str, optional
        Torch device.

    Example
    -------
    >>> forecaster = DTForecaster(state_dim=5, seq_len=20)
    >>> state_history = np.random.randn(20, 5).astype(np.float32)
    >>> pred = forecaster.predict(state_history)
    >>> assert 'return_1step' in pred
    >>> assert 'return_5step' in pred
    >>> assert 'confidence' in pred
    """

    def __init__(
        self,
        state_dim: int = 5,
        seq_len: int = 20,
        config: Optional[DTForecasterConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        if config is not None:
            self.cfg = config
        else:
            self.cfg = DTForecasterConfig(state_dim=state_dim, seq_len=seq_len)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _ForecasterNet(self.cfg).to(self.device)
        self._trained = False

        logger.info(
            "DTForecaster initialised — state_dim=%d, seq_len=%d, device=%s",
            self.cfg.state_dim,
            self.cfg.seq_len,
            self.device,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_supervised(
        self,
        states: np.ndarray,
        returns_1: np.ndarray,
        returns_5: np.ndarray,
        n_epochs: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Train with supervised MSE on historical returns.

        Parameters
        ----------
        states : (N, seq_len, state_dim) float array
            Sliding-window state sequences.
        returns_1 : (N,) float array
            Realised 1-step log-return for each window.
        returns_5 : (N,) float array
            Realised 5-step log-return for each window.
        n_epochs : int, optional
            Overrides ``cfg.n_epochs`` when provided.

        Returns
        -------
        dict with keys 'train_loss', 'val_loss'.
        """
        n_epochs = n_epochs or self.cfg.n_epochs

        states_t = torch.tensor(states, dtype=torch.float32)
        r1_t = torch.tensor(returns_1, dtype=torch.float32)
        r5_t = torch.tensor(returns_5, dtype=torch.float32)

        # Walk-forward split: last val_frac of samples for validation
        n_val = max(1, int(len(states) * self.cfg.val_frac))
        n_train = len(states) - n_val

        train_ds = TensorDataset(states_t[:n_train], r1_t[:n_train], r5_t[:n_train])
        val_ds = TensorDataset(states_t[n_train:], r1_t[n_train:], r5_t[n_train:])

        train_dl = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=self.cfg.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        best_val = float("inf")
        best_state = None

        for epoch in range(1, n_epochs + 1):
            # --- Train ---
            self.net.train()
            train_loss_acc = 0.0
            for s_b, r1_b, r5_b in train_dl:
                s_b = s_b.to(self.device)
                r1_b = r1_b.to(self.device)
                r5_b = r5_b.to(self.device)

                pred1, pred5 = self.net(s_b)
                loss = F.mse_loss(pred1, r1_b) + self.cfg.w_5step * F.mse_loss(pred5, r5_b)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                optimizer.step()
                train_loss_acc += loss.item() * len(s_b)

            train_loss = train_loss_acc / n_train

            # --- Validate ---
            self.net.eval()
            val_loss_acc = 0.0
            with torch.no_grad():
                for s_b, r1_b, r5_b in val_dl:
                    s_b = s_b.to(self.device)
                    r1_b = r1_b.to(self.device)
                    r5_b = r5_b.to(self.device)
                    pred1, pred5 = self.net(s_b)
                    loss = F.mse_loss(pred1, r1_b) + self.cfg.w_5step * F.mse_loss(pred5, r5_b)
                    val_loss_acc += loss.item() * len(s_b)
            val_loss = val_loss_acc / n_val

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}

            if verbose and (epoch % max(1, n_epochs // 5) == 0 or epoch == n_epochs):
                logger.info(
                    "DTForecaster epoch %d/%d | train_loss=%.6f | val_loss=%.6f",
                    epoch,
                    n_epochs,
                    train_loss,
                    val_loss,
                )

        # Restore best weights
        if best_state is not None:
            self.net.load_state_dict(best_state)

        self._trained = True
        return {"train_loss": train_loss, "val_loss": best_val}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        state_history: np.ndarray,
        mc_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Forecast returns for the next 1 and 5 steps.

        Parameters
        ----------
        state_history : (seq_len, state_dim) float array
            Most-recent state window (same shape as env observation).
        mc_samples : int, optional
            Number of MC-dropout passes for confidence estimation.
            Defaults to ``cfg.mc_samples``.

        Returns
        -------
        dict with keys:
            'return_1step'  : float — expected 1-step log-return
            'return_5step'  : float — expected 5-step log-return
            'confidence'    : float ∈ (0, 1] — higher = more certain
        """
        mc_samples = mc_samples or self.cfg.mc_samples

        state_arr = np.asarray(state_history, dtype=np.float32)
        if state_arr.ndim == 2:
            state_arr = state_arr[np.newaxis, ...]  # (1, seq_len, state_dim)

        x = torch.tensor(state_arr, dtype=torch.float32, device=self.device)

        # Monte-Carlo dropout: keep dropout *active* during inference
        self.net.train()
        preds_1, preds_5 = [], []
        with torch.no_grad():
            for _ in range(mc_samples):
                p1, p5 = self.net(x)
                preds_1.append(p1.item())
                preds_5.append(p5.item())
        self.net.eval()

        preds_1 = np.array(preds_1)
        preds_5 = np.array(preds_5)

        mean_1 = float(preds_1.mean())
        mean_5 = float(preds_5.mean())

        # Confidence: inverse of normalised std across MC samples
        # std_combined ∈ [0, ∞) → confidence ∈ (0, 1] via sigmoid-like transform
        std_combined = float(np.sqrt(preds_1.var() + preds_5.var()) / 2.0)
        confidence = float(1.0 / (1.0 + std_combined * 10.0))  # scale factor 10 is heuristic

        return {
            "return_1step": mean_1,
            "return_5step": mean_5,
            "confidence": confidence,
        }

    def predict_batch(self, state_histories: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Batch prediction (no MC dropout — uses single forward pass for speed).

        Parameters
        ----------
        state_histories : (B, seq_len, state_dim) float array

        Returns
        -------
        dict with keys 'return_1step', 'return_5step' — each (B,) float array.
        'confidence' is omitted for batch mode.
        """
        x = torch.tensor(
            np.asarray(state_histories, dtype=np.float32), dtype=torch.float32, device=self.device
        )
        self.net.eval()
        with torch.no_grad():
            p1, p5 = self.net(x)
        return {
            "return_1step": p1.cpu().numpy(),
            "return_5step": p5.cpu().numpy(),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        """Save model weights. Returns the path where saved."""
        if path is None:
            ckpt_dir = Path(self.cfg.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            path = str(ckpt_dir / "dt_forecaster.pt")

        torch.save(
            {
                "cfg": self.cfg,
                "net_state": self.net.state_dict(),
                "trained": self._trained,
            },
            path,
        )
        logger.info("DTForecaster saved → %s", path)
        return path

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "DTForecaster":
        """Load model from a checkpoint created by :meth:`save`."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        forecaster = cls(config=ckpt["cfg"], device=device)
        forecaster.net.load_state_dict(ckpt["net_state"])
        forecaster._trained = ckpt.get("trained", True)
        logger.info("DTForecaster loaded ← %s", path)
        return forecaster

    @classmethod
    def from_config(cls, config_path: str, device: Optional[str] = None) -> "DTForecaster":
        """
        Build a DTForecaster from a YAML training config file.

        Reads the ``dt_forecaster`` section; falls back to defaults if absent.
        """
        import yaml

        with open(config_path) as f:
            full_cfg = yaml.safe_load(f)

        sec = full_cfg.get("dt_forecaster", {})

        cfg = DTForecasterConfig(
            state_dim=sec.get("state_dim", 5),
            seq_len=sec.get("seq_len", full_cfg.get("env", {}).get("window_size", 20)),
            hidden_size=sec.get("hidden_size", 64),
            n_layer=sec.get("n_layer", 2),
            n_head=sec.get("n_head", 4),
            dropout=sec.get("dropout", 0.1),
            learning_rate=sec.get("learning_rate", 1e-4),
            weight_decay=sec.get("weight_decay", 1e-4),
            batch_size=sec.get("batch_size", 64),
            n_epochs=sec.get("n_epochs", 20),
            w_5step=sec.get("w_5step", 0.5),
            val_frac=sec.get("val_frac", 0.1),
            mc_samples=sec.get("mc_samples", 10),
            checkpoint_dir=sec.get("checkpoint_dir", "checkpoints/dt_forecaster"),
        )

        return cls(config=cfg, device=device)

    # ------------------------------------------------------------------
    # Utility: build sliding-window dataset from raw price series
    # ------------------------------------------------------------------

    @staticmethod
    def build_dataset(
        features: np.ndarray,
        seq_len: int = 20,
        horizon_short: int = 1,
        horizon_long: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create supervised training data from a feature matrix.

        Parameters
        ----------
        features : (T, state_dim) float array
            Full feature sequence (e.g. normalised OHLCV).
        seq_len : int
            Window length per sample.
        horizon_short, horizon_long : int
            Return forecast horizons (in timesteps).

        Returns
        -------
        states   : (N, seq_len, state_dim)
        returns_1: (N,)   — log-return at t + horizon_short
        returns_5: (N,)   — log-return at t + horizon_long
        """
        T, D = features.shape
        close_col = 3  # assumes $close is column index 3 (OHLCV order)

        # Work out how many complete windows fit
        n = T - seq_len - max(horizon_short, horizon_long)
        if n <= 0:
            raise ValueError(
                f"Not enough rows ({T}) for seq_len={seq_len} + "
                f"horizon={max(horizon_short, horizon_long)}"
            )

        states = np.zeros((n, seq_len, D), dtype=np.float32)
        r1 = np.zeros(n, dtype=np.float32)
        r5 = np.zeros(n, dtype=np.float32)

        for i in range(n):
            states[i] = features[i : i + seq_len]
            p_now = features[i + seq_len - 1, close_col]
            p_short = features[i + seq_len - 1 + horizon_short, close_col]
            p_long = features[i + seq_len - 1 + horizon_long, close_col]
            eps = 1e-8
            r1[i] = math.log((p_short + eps) / (p_now + eps))
            r5[i] = math.log((p_long + eps) / (p_now + eps))

        return states, r1, r5
