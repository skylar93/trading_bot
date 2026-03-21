"""
Decision Transformer for trading — offline RL pre-training.

Architecture (default, no GPT-2 download required):
  - Lightweight causal transformer backbone (configurable size)
  - Input: interleaved (return-to-go, state, action) token sequences
  - Context window K timesteps  →  3K tokens total
  - Action prediction head: Linear(hidden_size, act_dim)

Optional GPT-2 + LoRA backbone (``use_gpt2_backbone=True``):
  - Requires ``transformers`` and (for LoRA) ``peft``
  - LoRA rank=16 on attention layers  ~900K trainable params from 124M total

Usage::

    config = DecisionTransformerConfig(state_dim=100, act_dim=1)
    model = TradingDecisionTransformer(config)

    # Supervised training
    trainer = DecisionTransformerTrainer(model)
    metrics = trainer.train(dataset, n_epochs=10)

    # Inference — condition on high target RTG
    action = model.get_action(states, actions, returns_to_go, timesteps)

    # Factory from YAML config dict
    model = TradingDecisionTransformer.from_config(full_config_dict)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from agents.offline.trajectory_dataset import TradingTrajectoryDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import transformers  # noqa: F401
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    import peft  # noqa: F401
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DecisionTransformerConfig:
    """All hyper-parameters for the Decision Transformer."""

    # --- Model architecture ------------------------------------------------
    state_dim: int = 100        # flattened obs dim (window_size × n_features)
    act_dim: int = 1            # action dimension (1 for scalar position)
    hidden_size: int = 128      # transformer embedding dimension
    context_len: int = 20       # K — number of timesteps in one context window
    n_layer: int = 3            # transformer depth
    n_head: int = 1             # attention heads (hidden_size must be divisible)
    n_inner: Optional[int] = None  # FFN dim; defaults to 4 × hidden_size
    dropout: float = 0.1

    # --- Backbone choice ---------------------------------------------------
    use_gpt2_backbone: bool = False  # True → GPT-2 (downloads weights)

    # --- LoRA (only when use_gpt2_backbone=True) ----------------------------
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    lora_target_modules: Tuple[str, ...] = ("c_attn", "c_proj")

    # --- Training -----------------------------------------------------------
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_grad_norm: float = 0.25
    batch_size: int = 64

    # --- Inference ----------------------------------------------------------
    target_return: float = 1.0   # RTG target for conditioning (normalised scale)

    def __post_init__(self) -> None:
        if self.hidden_size % self.n_head != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by n_head ({self.n_head})"
            )


# ---------------------------------------------------------------------------
# Custom causal transformer backbone
# ---------------------------------------------------------------------------

class _CausalTransformerBlock(nn.Module):
    """Single causal self-attention + FFN block (pre-LN)."""

    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        n_inner: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size, n_head, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, n_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_inner, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, H)
            key_padding_mask: (B, T) bool — True positions are *ignored* (padding)
        """
        T = x.shape[1]
        # Causal (upper-triangular) mask — True means "block attention"
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        normed = self.ln1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class _CausalTransformerBackbone(nn.Module):
    """Stack of causal transformer blocks with final layer-norm."""

    def __init__(
        self,
        hidden_size: int,
        n_layer: int,
        n_head: int,
        n_inner: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_CausalTransformerBlock(hidden_size, n_head, n_inner, dropout)
             for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, H)
            attention_mask: (B, T) float — 1=real, 0=padding
        Returns:
            (B, T, H)
        """
        key_padding_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # True → ignore

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return self.ln_f(x)


# ---------------------------------------------------------------------------
# GPT-2 backbone wrapper
# ---------------------------------------------------------------------------

class _GPT2BackboneWrapper(nn.Module):
    """Thin wrapper so GPT-2 / PEFT model shares the same forward signature."""

    def __init__(self, gpt2_model: Any) -> None:
        super().__init__()
        self.model = gpt2_model

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """GPT-2 expects `inputs_embeds`; we pass pre-embedded tokens."""
        out = self.model(
            inputs_embeds=x,
            attention_mask=attention_mask,
        )
        # GPT-2Model returns a BaseModelOutputWithPastAndCrossAttentions
        return out.last_hidden_state  # (B, T, H)


# ---------------------------------------------------------------------------
# Decision Transformer
# ---------------------------------------------------------------------------

class TradingDecisionTransformer(nn.Module):
    """
    Decision Transformer for continuous-action trading environments.

    Input sequences of K timesteps, each represented by three tokens::

        [RTG_1, s_1, a_1,  RTG_2, s_2, a_2, …,  RTG_K, s_K, a_K]
         ─────────────────────────────────────── 3K tokens total

    At each *state* token position the model predicts the optimal action.
    """

    def __init__(self, config: DecisionTransformerConfig) -> None:
        super().__init__()
        self.config = config
        H = config.hidden_size
        K = config.context_len
        n_inner = config.n_inner if config.n_inner is not None else 4 * H

        # ── Embedding layers ─────────────────────────────────────────────
        self.state_embed = nn.Sequential(nn.Linear(config.state_dim, H), nn.Tanh())
        self.action_embed = nn.Sequential(nn.Linear(config.act_dim, H), nn.Tanh())
        self.rtg_embed = nn.Sequential(nn.Linear(1, H), nn.Tanh())
        self.pos_embed = nn.Embedding(K, H)
        self.embed_ln = nn.LayerNorm(H)

        # ── Transformer backbone ─────────────────────────────────────────
        if config.use_gpt2_backbone:
            self.transformer: nn.Module = self._build_gpt2_backbone(config, n_inner)
        else:
            self.transformer = _CausalTransformerBackbone(
                hidden_size=H,
                n_layer=config.n_layer,
                n_head=config.n_head,
                n_inner=n_inner,
                dropout=config.dropout,
            )

        # ── Action prediction head ───────────────────────────────────────
        self.action_head = nn.Linear(H, config.act_dim)

    # ------------------------------------------------------------------
    # Backbone builders
    # ------------------------------------------------------------------

    def _build_gpt2_backbone(
        self, config: DecisionTransformerConfig, n_inner: int
    ) -> nn.Module:
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "The 'transformers' package is required for the GPT-2 backbone. "
                "Install it with: pip install transformers"
            )
        from transformers import GPT2Config, GPT2Model  # type: ignore

        K = config.context_len
        max_seq_len = 3 * K

        gpt2_cfg = GPT2Config(
            vocab_size=1,           # unused — we pass inputs_embeds
            n_positions=max_seq_len,
            n_embd=config.hidden_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_inner=n_inner,
            resid_pdrop=config.dropout,
            attn_pdrop=config.dropout,
            embd_pdrop=0.0,
        )
        gpt2 = GPT2Model(gpt2_cfg)

        if config.use_lora:
            if not _PEFT_AVAILABLE:
                logger.warning(
                    "peft not installed — LoRA skipped; using full GPT-2 parameters."
                )
            else:
                from peft import LoraConfig, get_peft_model  # type: ignore

                lora_cfg = LoraConfig(
                    r=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    target_modules=list(config.lora_target_modules),
                    lora_dropout=config.lora_dropout,
                    bias="none",
                )
                gpt2 = get_peft_model(gpt2, lora_cfg)

        return _GPT2BackboneWrapper(gpt2)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        states:          (B, K, state_dim)
        actions:         (B, K, act_dim)
        returns_to_go:   (B, K, 1)
        timesteps:       (B, K)  long
        attention_mask:  (B, K)  float  — 1=real, 0=padded (optional)

        Returns
        -------
        action_preds:    (B, K, act_dim)
        """
        B, K, _ = states.shape

        # Clamp timesteps to valid embedding index range
        ts = timesteps.clamp(0, self.config.context_len - 1)
        pos_emb = self.pos_embed(ts)  # (B, K, H)

        state_emb = self.state_embed(states) + pos_emb       # (B, K, H)
        action_emb = self.action_embed(actions) + pos_emb    # (B, K, H)
        rtg_emb = self.rtg_embed(returns_to_go) + pos_emb   # (B, K, H)

        # Interleave into 3K-token sequence: [RTG_1, s_1, a_1, RTG_2, …]
        # stack → (B, K, 3, H) then reshape → (B, 3K, H)
        tokens = torch.stack([rtg_emb, state_emb, action_emb], dim=2)
        tokens = tokens.reshape(B, 3 * K, self.config.hidden_size)
        tokens = self.embed_ln(tokens)

        # Expand attention mask to 3K positions
        attn_mask_3k: Optional[torch.Tensor] = None
        if attention_mask is not None:
            attn_mask_3k = (
                attention_mask.unsqueeze(-1)          # (B, K, 1)
                .expand(B, K, 3)                     # (B, K, 3)
                .reshape(B, 3 * K)                   # (B, 3K)
            )

        # Transformer forward
        transformer_out = self.transformer(tokens, attention_mask=attn_mask_3k)
        # shape: (B, 3K, H)

        # State-token positions: 1, 4, 7, … (0-indexed: RTG=0, state=1, action=2 per step)
        state_outs = transformer_out[:, 1::3, :]     # (B, K, H)
        action_preds = self.action_head(state_outs)  # (B, K, act_dim)
        return action_preds

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """MSE loss on non-padded action positions."""
        action_preds = self.forward(
            states, actions, returns_to_go, timesteps, attention_mask
        )
        if attention_mask is not None:
            # Only compute loss on real (non-padded) positions
            mask = attention_mask.unsqueeze(-1)  # (B, K, 1)
            loss = ((action_preds - actions) ** 2 * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = ((action_preds - actions) ** 2).mean()
        return loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> np.ndarray:
        """
        Predict the action for the *last* timestep in the context window.

        All tensors are (K, …) — no batch dimension.
        Returns a numpy array of shape (act_dim,).
        """
        self.eval()
        action_preds = self.forward(
            states.unsqueeze(0),           # (1, K, state_dim)
            actions.unsqueeze(0),          # (1, K, act_dim)
            returns_to_go.unsqueeze(0),    # (1, K, 1)
            timesteps.unsqueeze(0),        # (1, K)
        )
        return action_preds[0, -1].cpu().numpy()  # (act_dim,)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def count_parameters(self) -> Dict[str, int]:
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save config + state-dict to ``path`` (torch format)."""
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "TradingDecisionTransformer":
        """Load from a file saved with :meth:`save`."""
        data = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(data["config"])
        model.load_state_dict(data["state_dict"])
        return model

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "TradingDecisionTransformer":
        """
        Create from the project's unified YAML config dict.

        Reads the ``decision_transformer`` sub-section; falls back to
        defaults for any missing key.
        """
        dt = config_dict.get("decision_transformer", {})
        cfg = DecisionTransformerConfig(
            state_dim=dt.get("state_dim", 100),
            act_dim=dt.get("act_dim", 1),
            hidden_size=dt.get("hidden_size", 128),
            context_len=dt.get("context_len", 20),
            n_layer=dt.get("n_layer", 3),
            n_head=dt.get("n_head", 1),
            n_inner=dt.get("n_inner", None),
            dropout=dt.get("dropout", 0.1),
            use_gpt2_backbone=dt.get("use_gpt2_backbone", False),
            use_lora=dt.get("use_lora", True),
            lora_rank=dt.get("lora_rank", 16),
            lora_alpha=dt.get("lora_alpha", 32.0),
            lora_dropout=dt.get("lora_dropout", 0.1),
            learning_rate=dt.get("learning_rate", 1e-4),
            weight_decay=dt.get("weight_decay", 1e-4),
            warmup_steps=dt.get("warmup_steps", 1000),
            max_grad_norm=dt.get("max_grad_norm", 0.25),
            batch_size=dt.get("batch_size", 64),
            target_return=dt.get("target_return", 1.0),
        )
        return cls(cfg)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DecisionTransformerTrainer:
    """
    Supervised training loop for :class:`TradingDecisionTransformer`.

    Optimiser: AdamW with optional linear-warmup scheduler.
    Loss:       Masked MSE on action predictions.

    Usage::

        trainer = DecisionTransformerTrainer(model, learning_rate=1e-4)
        metrics = trainer.train(train_dataset, n_epochs=10, eval_dataset=val_dataset)
        # metrics = {"train_loss": [...], "eval_loss": [...]}
    """

    def __init__(
        self,
        model: TradingDecisionTransformer,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_steps: int = 1000,
        max_grad_norm: float = 0.25,
        batch_size: int = 64,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self._step = 0
        self.warmup_steps = warmup_steps

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        def _lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step + 1) / float(max(warmup_steps, 1))
            return 1.0

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=_lr_lambda
        )

    # ------------------------------------------------------------------
    # Training / evaluation passes
    # ------------------------------------------------------------------

    def train_epoch(self, dataset: TradingTrajectoryDataset) -> float:
        """Run one full epoch. Returns mean loss over all batches."""
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            states = batch["states"].to(self.device)
            actions = batch["actions"].to(self.device)
            rtg = batch["returns_to_go"].to(self.device)
            timesteps = batch["timesteps"].to(self.device)
            mask = batch["attention_mask"].to(self.device)

            loss = self.model.compute_loss(states, actions, rtg, timesteps, mask)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self._step += 1

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate(self, dataset: TradingTrajectoryDataset) -> float:
        """Evaluate on ``dataset``. Returns mean loss (no grad)."""
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                states = batch["states"].to(self.device)
                actions = batch["actions"].to(self.device)
                rtg = batch["returns_to_go"].to(self.device)
                timesteps = batch["timesteps"].to(self.device)
                mask = batch["attention_mask"].to(self.device)

                loss = self.model.compute_loss(states, actions, rtg, timesteps, mask)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(
        self,
        train_dataset: TradingTrajectoryDataset,
        n_epochs: int = 10,
        eval_dataset: Optional[TradingTrajectoryDataset] = None,
    ) -> Dict[str, List[float]]:
        """
        Train for ``n_epochs`` epochs.

        Returns
        -------
        dict with keys:
            ``train_loss``:  list of per-epoch training losses
            ``eval_loss``:   list of per-epoch evaluation losses (empty if no eval_dataset)
        """
        train_losses: List[float] = []
        eval_losses: List[float] = []

        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_dataset)
            train_losses.append(train_loss)
            logger.debug("Epoch %d/%d  train_loss=%.6f", epoch + 1, n_epochs, train_loss)

            if eval_dataset is not None:
                eval_loss = self.evaluate(eval_dataset)
                eval_losses.append(eval_loss)
                logger.debug("              eval_loss=%.6f", eval_loss)

        return {"train_loss": train_losses, "eval_loss": eval_losses}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        model: TradingDecisionTransformer,
        config_dict: Dict[str, Any],
    ) -> "DecisionTransformerTrainer":
        """Create from the project's unified YAML config dict."""
        dt = config_dict.get("decision_transformer", {})
        return cls(
            model=model,
            learning_rate=dt.get("learning_rate", 1e-4),
            weight_decay=dt.get("weight_decay", 1e-4),
            warmup_steps=dt.get("warmup_steps", 1000),
            max_grad_norm=dt.get("max_grad_norm", 0.25),
            batch_size=dt.get("batch_size", 64),
            device=dt.get("device", "cpu"),
        )
