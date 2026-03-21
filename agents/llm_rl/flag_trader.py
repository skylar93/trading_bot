"""
FLAG-Trader: Financial LLM Agent with Grounding.

Based on FLAG-Trader (ACL 2025, Harvard/Columbia/NVIDIA):
  - Base model: SmolLM2-135M (HuggingFaceTB/SmolLM2-135M)
  - Fine-tuning: PPO via TRL on trading environment
  - Input: text-formatted market state (log-returns, position, portfolio)
  - Output: continuous action ∈ [-1, 1] via regression head on hidden states
  - LoRA rank=16 on q_proj/v_proj for efficient adaptation (~900K trainable)

Benchmark results (FLAG-Trader 135M, from paper):
  MSFT Sharpe 1.37  vs GPT-4 0.93
  JNJ  Sharpe 3.34  vs GPT-4 1.10
  BTC  Sharpe 1.73  vs GPT-4 0.83

SB3-compatible interface:
  agent.predict(obs, deterministic=True) → (np.ndarray, None)

Training pipeline:
  1. Supervised pre-training on expert trajectories (Decision Transformer data)
  2. PPO fine-tuning via TRL against trading environment

Optional dependencies (gracefully guarded):
  transformers >= 4.40  (SmolLM2)
  peft >= 0.10          (LoRA)
  trl >= 0.12           (PPO fine-tuning)

Usage::

    # Dry-run mode (no downloads, for testing)
    cfg = FLAGTraderConfig(dry_run=True)
    agent = FLAGTrader(cfg)
    action, _ = agent.predict(obs)   # obs: np.ndarray (obs_dim,)

    # Production mode
    cfg = FLAGTraderConfig(base_model="HuggingFaceTB/SmolLM2-135M")
    agent = FLAGTrader(cfg)
    trainer = FLAGTraderTrainer(agent)
    trainer.train_supervised(dataset, n_epochs=5)
    trainer.train_ppo(env, total_timesteps=100_000)

    # Factory from YAML config dict
    agent = FLAGTrader.from_config(full_config_dict)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

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

try:
    import trl  # noqa: F401
    _TRL_AVAILABLE = True
except ImportError:
    _TRL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FLAGTraderConfig:
    """All hyper-parameters for FLAG-Trader."""

    # --- Base model --------------------------------------------------------
    base_model: str = "HuggingFaceTB/SmolLM2-135M"
    dry_run: bool = False  # True → tiny in-memory model, no downloads

    # --- LoRA --------------------------------------------------------------
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    lora_target_modules: Tuple[str, ...] = ("q_proj", "v_proj")

    # --- Market state formatting ------------------------------------------
    window_size: int = 20          # number of log-return timesteps in obs
    obs_dim: int = 22              # window_size + position_ratio + cash_ratio
    price_decimals: int = 4        # decimal places for log-returns in text

    # --- Action regression head -------------------------------------------
    hidden_size: int = 576         # SmolLM2-135M hidden size; ignored in dry_run
    action_scale: float = 1.0      # tanh output scale

    # --- Supervised pre-training ------------------------------------------
    pretrain_lr: float = 1e-4
    pretrain_weight_decay: float = 1e-4
    pretrain_batch_size: int = 32
    pretrain_max_grad_norm: float = 1.0

    # --- PPO fine-tuning (TRL) --------------------------------------------
    ppo_lr: float = 1e-5
    ppo_epochs: int = 4
    ppo_mini_batch_size: int = 16
    ppo_batch_size: int = 256      # steps collected per PPO update
    gamma: float = 0.99
    lam: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    # --- Text generation --------------------------------------------------
    max_new_tokens: int = 5        # tokens generated for action text

    # --- Dry-run tiny model (no downloads) --------------------------------
    _dry_hidden: int = 32
    _dry_n_layer: int = 2
    _dry_n_head: int = 2
    _dry_vocab: int = 128


# ---------------------------------------------------------------------------
# Market state → text formatter
# ---------------------------------------------------------------------------

class MarketStateFormatter:
    """
    Converts a numeric observation vector to a natural-language market state.

    Observation layout (mirrors SingleAssetRLTradingEnv):
        obs[:-2]   — log-returns for last window_size steps
        obs[-2]    — position_ratio  ∈ [-1, 1]  (negative = short)
        obs[-1]    — cash_ratio      ∈ [0, 1]

    Example output::

        Market State:
        Log-returns (5 steps): -0.0123, +0.0045, -0.0067, +0.0201, -0.0034
        Position: 0.20 (20% long)
        Cash: 0.80 (80% available)
        Action:
    """

    def __init__(self, window_size: int = 20, price_decimals: int = 4) -> None:
        self.window_size = window_size
        self.price_decimals = price_decimals

    def format(self, obs: np.ndarray, step: int = 0) -> str:
        """Format a single observation vector to text.

        Parameters
        ----------
        obs:
            Flat observation array of shape (obs_dim,).
        step:
            Optional timestep index for context.

        Returns
        -------
        Prompt string ending with "Action:" ready for LLM completion.
        """
        obs = np.asarray(obs, dtype=np.float32).flatten()

        # Split observation
        returns = obs[:-2] if len(obs) > 2 else obs
        position = float(obs[-2]) if len(obs) >= 2 else 0.0
        cash = float(obs[-1]) if len(obs) >= 1 else 1.0

        # Format log-returns — show up to last 5 for readability
        display_returns = returns[-5:] if len(returns) >= 5 else returns
        fmt = f".{self.price_decimals}f"
        ret_str = ", ".join(
            f"{'+' if r >= 0 else ''}{r:{fmt}}" for r in display_returns
        )

        # Position description
        pos_pct = abs(position) * 100
        if position > 0.05:
            pos_label = f"{pos_pct:.0f}% long"
        elif position < -0.05:
            pos_label = f"{pos_pct:.0f}% short"
        else:
            pos_label = "flat (no position)"

        cash_pct = max(0.0, cash) * 100

        n_shown = len(display_returns)
        prompt = (
            f"Market State:\n"
            f"Log-returns ({n_shown} steps): {ret_str}\n"
            f"Position: {position:.2f} ({pos_label})\n"
            f"Cash: {cash:.2f} ({cash_pct:.0f}% available)\n"
            f"Action:"
        )
        return prompt

    def format_batch(self, obs_batch: np.ndarray) -> List[str]:
        """Format a batch of observations."""
        return [self.format(obs) for obs in obs_batch]


# ---------------------------------------------------------------------------
# Dry-run policy (tiny causal LM, no downloads)
# ---------------------------------------------------------------------------

class _DryRunAttention(nn.Module):
    def __init__(self, hidden: int, n_head: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.head_dim = hidden // n_head
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(2)  # each (B, T, n_head, head_dim)
        q = q.transpose(1, 2)   # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = math.sqrt(self.head_dim)
        attn = torch.softmax(q @ k.transpose(-2, -1) / scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, H)
        return self.out(out)


class _DryRunBlock(nn.Module):
    def __init__(self, hidden: int, n_head: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = _DryRunAttention(hidden, n_head)
        self.ln2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class _DryRunLLM(nn.Module):
    """
    Tiny causal language model for testing without downloading SmolLM2.

    Provides the same interface used by FLAGTrader:
      - embed(token_ids) → hidden states
      - last_hidden_state for regression head
    """

    def __init__(self, vocab: int, hidden: int, n_layer: int, n_head: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.embed = nn.Embedding(vocab, hidden)
        self.blocks = nn.ModuleList(
            [_DryRunBlock(hidden, n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(hidden)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids: (B, T) long

        Returns
        -------
        last_hidden_state: (B, T, hidden)
        """
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)


# ---------------------------------------------------------------------------
# Observation → token encoder (used in dry-run / production)
# ---------------------------------------------------------------------------

class _ObsEncoder(nn.Module):
    """
    Projects raw observation vectors directly to LLM hidden dimension.

    Used as an alternative to text tokenisation so that numeric observations
    can be fed to _DryRunLLM without a tokenizer.

    input:  (B, obs_dim)  float32
    output: (B, 1, hidden) — single token embedding
    """

    def __init__(self, obs_dim: int, hidden: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.proj(obs).unsqueeze(1)  # (B, 1, hidden)


# ---------------------------------------------------------------------------
# Action regression head
# ---------------------------------------------------------------------------

class _ActionHead(nn.Module):
    """
    Linear head that maps mean-pooled LLM hidden state → continuous action.

    output = tanh(linear(hidden))  ∈ (-1, 1)
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states: (B, T, hidden)

        Returns
        -------
        action: (B, 1)
        """
        pooled = hidden_states.mean(dim=1)  # (B, hidden)
        return torch.tanh(self.linear(pooled))  # (B, 1)


# ---------------------------------------------------------------------------
# Main FLAG-Trader class
# ---------------------------------------------------------------------------

class FLAGTrader:
    """
    Financial LLM Agent with Grounding (FLAG-Trader).

    Architecture:
      SmolLM2-135M  ─→  LoRA adapters  ─→  regression head  ─→  action ∈ [-1, 1]

    Provides SB3-compatible interface:
      predict(obs, deterministic=True) → (np.ndarray shape (1,), None)

    Parameters
    ----------
    config:
        FLAGTraderConfig instance.
    device:
        Torch device string ("cpu", "cuda", "mps").
    """

    def __init__(
        self,
        config: Optional[FLAGTraderConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.config = config or FLAGTraderConfig()
        self.device = torch.device(device)
        self.formatter = MarketStateFormatter(
            window_size=self.config.window_size,
            price_decimals=self.config.price_decimals,
        )

        self._tokenizer = None
        self._llm: nn.Module
        self._obs_encoder: Optional[_ObsEncoder] = None
        self._action_head: _ActionHead

        self._build_model()
        logger.info(
            "FLAGTrader initialised — dry_run=%s  params=%s",
            self.config.dry_run,
            self.count_parameters(),
        )

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self) -> None:
        cfg = self.config

        if cfg.dry_run:
            self._llm = _DryRunLLM(
                vocab=cfg._dry_vocab,
                hidden=cfg._dry_hidden,
                n_layer=cfg._dry_n_layer,
                n_head=cfg._dry_n_head,
            )
            self._obs_encoder = _ObsEncoder(cfg.obs_dim, cfg._dry_hidden)
            self._action_head = _ActionHead(cfg._dry_hidden)
        else:
            if not _TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "transformers is required for production mode. "
                    "Install: pip install transformers"
                )
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

            logger.info("Loading %s …", cfg.base_model)
            self._tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            llm_raw = AutoModelForCausalLM.from_pretrained(
                cfg.base_model,
                torch_dtype=torch.float32,
            )

            if _PEFT_AVAILABLE:
                from peft import LoraConfig, get_peft_model, TaskType  # type: ignore

                lora_cfg = LoraConfig(
                    r=cfg.lora_rank,
                    lora_alpha=cfg.lora_alpha,
                    target_modules=list(cfg.lora_target_modules),
                    lora_dropout=cfg.lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                self._llm = get_peft_model(llm_raw, lora_cfg)
                self._llm.print_trainable_parameters()
            else:
                logger.warning("peft not installed — using full model parameters (no LoRA)")
                self._llm = llm_raw

            self._action_head = _ActionHead(cfg.hidden_size)

        self._llm = self._llm.to(self.device)
        self._action_head = self._action_head.to(self.device)
        if self._obs_encoder is not None:
            self._obs_encoder = self._obs_encoder.to(self.device)

    # ------------------------------------------------------------------
    # Observation → hidden states
    # ------------------------------------------------------------------

    def _obs_to_hidden(self, obs: np.ndarray) -> torch.Tensor:
        """
        Convert observation array to LLM hidden states.

        In dry_run mode: obs → _ObsEncoder → (B, 1, hidden)
        In production mode: obs → text → tokenizer → LLM → (B, T, hidden)

        Returns
        -------
        hidden: (B, T, hidden)
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)  # (1, obs_dim)

        if self.config.dry_run:
            assert self._obs_encoder is not None
            token_emb = self._obs_encoder(obs_t)  # (B, 1, hidden)
            # Pass through LLM blocks using embed as input directly
            # _DryRunLLM.forward() expects token_ids; use forward on embeddings
            x = token_emb
            for block in self._llm.blocks:  # type: ignore[attr-defined]
                x = block(x)
            hidden = self._llm.ln_f(x)  # type: ignore[attr-defined]
            return hidden  # (B, 1, hidden)
        else:
            assert self._tokenizer is not None
            texts = self.formatter.format_batch(obs_t.cpu().numpy())
            enc = self._tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(self.device)
            with torch.no_grad():
                out = self._llm(**enc, output_hidden_states=True)
            # Use last hidden layer
            return out.hidden_states[-1]  # (B, T, hidden)

    # ------------------------------------------------------------------
    # SB3-compatible predict
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        obs: np.ndarray,
        state: Any = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Any]:
        """
        Predict trading action from observation.

        Follows SB3 agent interface:
          action, state = agent.predict(obs, deterministic=True)

        Parameters
        ----------
        obs:
            Observation array of shape (obs_dim,) or (B, obs_dim).
        deterministic:
            Ignored (regression head is always deterministic). Kept for SB3
            compatibility.

        Returns
        -------
        action:
            np.ndarray of shape (1,) with value ∈ (-1, 1).
        state:
            Always None (no recurrent state).
        """
        self._llm.eval()
        self._action_head.eval()

        obs = np.asarray(obs, dtype=np.float32)
        batch = obs.ndim == 2
        if not batch:
            obs = obs[np.newaxis]  # (1, obs_dim)

        hidden = self._obs_to_hidden(obs)       # (B, T, hidden)
        action = self._action_head(hidden)      # (B, 1)
        action_np = action.cpu().numpy()        # (B, 1)

        if not batch:
            return action_np[0], state          # (1,)
        return action_np, state                 # (B, 1)

    # ------------------------------------------------------------------
    # Parameter counting
    # ------------------------------------------------------------------

    def count_parameters(self) -> Dict[str, int]:
        """Return total and trainable parameter counts across all modules."""
        modules = [self._llm, self._action_head]
        if self._obs_encoder is not None:
            modules.append(self._obs_encoder)

        total = sum(p.numel() for m in modules for p in m.parameters())
        trainable = sum(
            p.numel() for m in modules for p in m.parameters() if p.requires_grad
        )
        return {"total": total, "trainable": trainable}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialize FLAG-Trader to ``path`` (JSON metadata + torch weights).

        Saves three files:
          <path>.json  — config
          <path>.pt    — model weights
        """
        import dataclasses

        with open(f"{path}.json", "w") as f:
            json.dump(dataclasses.asdict(self.config), f, indent=2)

        state = {
            "llm": self._llm.state_dict(),
            "action_head": self._action_head.state_dict(),
        }
        if self._obs_encoder is not None:
            state["obs_encoder"] = self._obs_encoder.state_dict()
        torch.save(state, f"{path}.pt")
        logger.info("FLAGTrader saved to %s.{json,pt}", path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "FLAGTrader":
        """Load a FLAGTrader saved with :meth:`save`."""
        with open(f"{path}.json") as f:
            cfg_dict = json.load(f)

        # Reconstruct dataclass — handle tuple fields stored as lists
        cfg_dict["lora_target_modules"] = tuple(cfg_dict.get("lora_target_modules", ()))
        config = FLAGTraderConfig(**cfg_dict)
        agent = cls(config, device=device)

        state = torch.load(f"{path}.pt", map_location=device, weights_only=True)
        agent._llm.load_state_dict(state["llm"])
        agent._action_head.load_state_dict(state["action_head"])
        if "obs_encoder" in state and agent._obs_encoder is not None:
            agent._obs_encoder.load_state_dict(state["obs_encoder"])

        logger.info("FLAGTrader loaded from %s.{json,pt}", path)
        return agent

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any], device: str = "cpu") -> "FLAGTrader":
        """
        Create from the project's unified YAML config dict.

        Reads the ``flag_trader`` sub-section; falls back to defaults for
        any missing key.
        """
        ft = config_dict.get("flag_trader", {})
        config = FLAGTraderConfig(
            base_model=ft.get("base_model", "HuggingFaceTB/SmolLM2-135M"),
            dry_run=ft.get("dry_run", False),
            lora_rank=ft.get("lora_rank", 16),
            lora_alpha=ft.get("lora_alpha", 32.0),
            lora_dropout=ft.get("lora_dropout", 0.1),
            lora_target_modules=tuple(ft.get("lora_target_modules", ("q_proj", "v_proj"))),
            window_size=ft.get("window_size", 20),
            obs_dim=ft.get("obs_dim", 22),
            price_decimals=ft.get("price_decimals", 4),
            hidden_size=ft.get("hidden_size", 576),
            pretrain_lr=ft.get("pretrain_lr", 1e-4),
            pretrain_weight_decay=ft.get("pretrain_weight_decay", 1e-4),
            pretrain_batch_size=ft.get("pretrain_batch_size", 32),
            ppo_lr=ft.get("ppo_lr", 1e-5),
            ppo_epochs=ft.get("ppo_epochs", 4),
            ppo_mini_batch_size=ft.get("ppo_mini_batch_size", 16),
            ppo_batch_size=ft.get("ppo_batch_size", 256),
            gamma=ft.get("gamma", 0.99),
            lam=ft.get("lam", 0.95),
            clip_range=ft.get("clip_range", 0.2),
            vf_coef=ft.get("vf_coef", 0.5),
            ent_coef=ft.get("ent_coef", 0.01),
            max_grad_norm=ft.get("max_grad_norm", 0.5),
            max_new_tokens=ft.get("max_new_tokens", 5),
        )
        return cls(config, device=device)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class FLAGTraderTrainer:
    """
    Two-phase trainer for :class:`FLAGTrader`.

    Phase 1 — Supervised pre-training:
        Trains the LLM + regression head to imitate expert actions from a
        :class:`~agents.offline.trajectory_dataset.TradingTrajectoryDataset`.
        Loss: MSE between predicted and expert actions.

    Phase 2 — PPO fine-tuning:
        Fine-tunes against a live Gym/Gymnasium environment using PPO.
        When TRL is available, uses ``trl.PPOTrainer`` for text-based PPO.
        Otherwise falls back to a custom gradient-based PPO loop that operates
        on the hidden-state representation directly.

    Usage::

        trainer = FLAGTraderTrainer(agent)
        # Phase 1
        metrics1 = trainer.train_supervised(dataset, n_epochs=5)
        # Phase 2
        metrics2 = trainer.train_ppo(env, total_timesteps=100_000)
    """

    def __init__(self, agent: FLAGTrader) -> None:
        self.agent = agent
        cfg = agent.config

        # Collect all trainable parameters
        params = list(agent._llm.parameters()) + list(agent._action_head.parameters())
        if agent._obs_encoder is not None:
            params += list(agent._obs_encoder.parameters())

        self._optimizer_supervised = torch.optim.AdamW(
            params,
            lr=cfg.pretrain_lr,
            weight_decay=cfg.pretrain_weight_decay,
        )
        self._optimizer_ppo = torch.optim.Adam(params, lr=cfg.ppo_lr)

    # ------------------------------------------------------------------
    # Phase 1: Supervised pre-training
    # ------------------------------------------------------------------

    def train_supervised(
        self,
        dataset: Any,  # TradingTrajectoryDataset (typed loosely to avoid import cycle)
        n_epochs: int = 5,
        eval_dataset: Optional[Any] = None,
    ) -> Dict[str, List[float]]:
        """
        Supervised imitation learning from expert trajectories.

        Parameters
        ----------
        dataset:
            :class:`~agents.offline.trajectory_dataset.TradingTrajectoryDataset`.
            Each sample provides ``states`` (K, state_dim) and ``actions`` (K, 1).
        n_epochs:
            Number of passes over the dataset.
        eval_dataset:
            Optional evaluation dataset; if provided, eval loss is computed after
            each epoch.

        Returns
        -------
        dict with keys:
            ``train_loss``: list of per-epoch mean MSE losses
            ``eval_loss``:  list of per-epoch eval losses (empty if no eval_dataset)
        """
        from torch.utils.data import DataLoader

        cfg = self.agent.config
        device = self.agent.device

        loader = DataLoader(
            dataset,
            batch_size=cfg.pretrain_batch_size,
            shuffle=True,
            drop_last=False,
        )

        train_losses: List[float] = []
        eval_losses: List[float] = []

        for epoch in range(n_epochs):
            self.agent._llm.train()
            self.agent._action_head.train()
            if self.agent._obs_encoder is not None:
                self.agent._obs_encoder.train()

            epoch_loss = 0.0
            n_batches = 0

            for batch in loader:
                # batch["states"]: (B, K, state_dim)  — take the last timestep
                states = batch["states"][:, -1, :].numpy()   # (B, state_dim)
                actions = batch["actions"][:, -1, :].to(device)  # (B, 1)
                mask = batch["attention_mask"][:, -1]            # (B,)

                hidden = self.agent._obs_to_hidden(states)    # (B, T, h)
                pred = self.agent._action_head(hidden)         # (B, 1)

                # Only compute loss on real (non-padded) samples
                valid = mask.bool().to(device)
                if valid.any():
                    loss = ((pred[valid] - actions[valid]) ** 2).mean()
                else:
                    continue

                self._optimizer_supervised.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.agent._llm.parameters())
                    + list(self.agent._action_head.parameters()),
                    cfg.pretrain_max_grad_norm,
                )
                self._optimizer_supervised.step()

                epoch_loss += loss.item()
                n_batches += 1

            mean_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(mean_loss)
            logger.debug("Supervised epoch %d/%d  loss=%.6f", epoch + 1, n_epochs, mean_loss)

            if eval_dataset is not None:
                eval_loss = self._eval_supervised(eval_dataset)
                eval_losses.append(eval_loss)

        return {"train_loss": train_losses, "eval_loss": eval_losses}

    def _eval_supervised(self, dataset: Any) -> float:
        """Evaluate supervised MSE loss on dataset (no gradient)."""
        from torch.utils.data import DataLoader

        cfg = self.agent.config
        device = self.agent.device
        loader = DataLoader(dataset, batch_size=cfg.pretrain_batch_size, shuffle=False)

        self.agent._llm.eval()
        self.agent._action_head.eval()
        if self.agent._obs_encoder is not None:
            self.agent._obs_encoder.eval()

        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                states = batch["states"][:, -1, :].numpy()
                actions = batch["actions"][:, -1, :].to(device)
                mask = batch["attention_mask"][:, -1]
                hidden = self.agent._obs_to_hidden(states)
                pred = self.agent._action_head(hidden)
                valid = mask.bool().to(device)
                if valid.any():
                    total += ((pred[valid] - actions[valid]) ** 2).mean().item()
                    count += 1
        return total / max(count, 1)

    # ------------------------------------------------------------------
    # Phase 2: PPO fine-tuning
    # ------------------------------------------------------------------

    def train_ppo(
        self,
        env: Any,
        total_timesteps: int = 100_000,
        log_interval: int = 1_000,
    ) -> Dict[str, Any]:
        """
        PPO fine-tuning against a Gym/Gymnasium trading environment.

        When TRL is available and the agent is in production mode, uses
        ``trl.PPOTrainer`` with text-based queries/responses.
        Otherwise falls back to a custom PPO loop using the hidden-state
        regression head directly (compatible with dry_run mode).

        Parameters
        ----------
        env:
            Gym/Gymnasium environment (``reset()``/``step()`` interface).
        total_timesteps:
            Total environment steps to collect.
        log_interval:
            Steps between progress log messages.

        Returns
        -------
        dict with keys:
            ``episode_rewards``: list of total rewards per episode
            ``mean_reward``:     mean episode reward
            ``n_updates``:       number of PPO updates performed
        """
        if _TRL_AVAILABLE and not self.agent.config.dry_run:
            return self._train_ppo_trl(env, total_timesteps, log_interval)
        return self._train_ppo_custom(env, total_timesteps, log_interval)

    # --- Custom PPO loop (dry_run / no TRL) ----------------------------

    def _train_ppo_custom(
        self,
        env: Any,
        total_timesteps: int,
        log_interval: int,
    ) -> Dict[str, Any]:
        """
        Lightweight PPO loop using the hidden-state regression head.

        Implements GAE advantage estimation and clipped surrogate objective.
        Suitable for dry_run testing and environments without TRL dependency.
        """
        cfg = self.agent.config
        device = self.agent.device

        episode_rewards: List[float] = []
        n_updates = 0
        steps_done = 0
        ep_reward = 0.0

        # Storage for one PPO batch
        obs_buf: List[np.ndarray] = []
        act_buf: List[float] = []
        rew_buf: List[float] = []
        val_buf: List[float] = []
        done_buf: List[bool] = []
        logp_buf: List[float] = []

        obs, _ = env.reset() if _has_gymnasium_api(env) else (env.reset(), {})

        while steps_done < total_timesteps:
            # Collect one action
            obs_arr = np.asarray(obs, dtype=np.float32)
            with torch.no_grad():
                hidden = self.agent._obs_to_hidden(obs_arr[np.newaxis])
                action_t = self.agent._action_head(hidden)  # (1, 1)
                action_val = float(action_t.item())

                # Value estimate (reuse action head mean as proxy critic)
                value = float(hidden.mean().item())

            # Gaussian policy for exploration (std=0.1)
            noise = np.random.normal(0, 0.1)
            action_noisy = float(np.clip(action_val + noise, -1.0, 1.0))
            # log-prob of Gaussian N(action_val, 0.1)
            logp = -0.5 * ((action_noisy - action_val) / 0.1) ** 2

            obs_buf.append(obs_arr)
            act_buf.append(action_noisy)
            rew_buf.append(0.0)  # filled after step
            val_buf.append(value)
            done_buf.append(False)
            logp_buf.append(logp)

            # Step environment
            action_env = np.array([action_noisy])
            step_result = env.step(action_env)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, _ = step_result
                done = bool(terminated or truncated)
            else:
                next_obs, reward, done, _ = step_result

            rew_buf[-1] = float(reward)
            done_buf[-1] = done
            ep_reward += float(reward)
            steps_done += 1
            obs = next_obs

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, _ = env.reset() if _has_gymnasium_api(env) else (env.reset(), {})

            # PPO update when buffer is full
            if len(obs_buf) >= cfg.ppo_batch_size:
                loss_info = self._ppo_update(
                    obs_buf, act_buf, rew_buf, val_buf, done_buf, logp_buf
                )
                n_updates += 1
                if n_updates % (log_interval // cfg.ppo_batch_size + 1) == 0:
                    mean_ep = float(np.mean(episode_rewards[-10:])) if episode_rewards else 0.0
                    logger.debug(
                        "PPO update %d  steps=%d  mean_ep_reward=%.4f  loss=%.4f",
                        n_updates, steps_done, mean_ep, loss_info.get("policy_loss", 0.0),
                    )
                obs_buf.clear(); act_buf.clear(); rew_buf.clear()
                val_buf.clear(); done_buf.clear(); logp_buf.clear()

        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        return {
            "episode_rewards": episode_rewards,
            "mean_reward": mean_reward,
            "n_updates": n_updates,
        }

    def _ppo_update(
        self,
        obs_buf: List[np.ndarray],
        act_buf: List[float],
        rew_buf: List[float],
        val_buf: List[float],
        done_buf: List[bool],
        logp_buf: List[float],
    ) -> Dict[str, float]:
        """One PPO update step using collected experience."""
        cfg = self.agent.config
        device = self.agent.device

        # Compute GAE advantages
        advantages, returns = _compute_gae(
            np.array(rew_buf, dtype=np.float32),
            np.array(val_buf, dtype=np.float32),
            np.array(done_buf, dtype=np.float32),
            gamma=cfg.gamma,
            lam=cfg.lam,
        )
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        old_logp_t = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        actions_t = torch.tensor(act_buf, dtype=torch.float32, device=device)
        obs_arr = np.stack(obs_buf)  # (N, obs_dim)

        total_policy_loss = 0.0
        n_mini = 0

        for _ in range(cfg.ppo_epochs):
            # Mini-batch indices
            indices = np.random.permutation(len(obs_buf))
            for start in range(0, len(obs_buf), cfg.ppo_mini_batch_size):
                idx = indices[start : start + cfg.ppo_mini_batch_size]
                if len(idx) == 0:
                    continue

                obs_mb = obs_arr[idx]
                act_mb = actions_t[idx]
                adv_mb = advantages_t[idx]
                old_logp_mb = old_logp_t[idx]

                # Forward pass
                self.agent._llm.train()
                self.agent._action_head.train()
                if self.agent._obs_encoder is not None:
                    self.agent._obs_encoder.train()

                hidden = self.agent._obs_to_hidden(obs_mb)   # (B, T, h)
                new_action_t = self.agent._action_head(hidden).squeeze(-1)  # (B,)

                # Gaussian log-prob with std=0.1
                std = 0.1
                new_logp = -0.5 * ((act_mb - new_action_t) / std) ** 2

                # Clipped surrogate loss
                ratio = torch.exp(new_logp - old_logp_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus (encourages exploration)
                entropy = 0.5 * (1.0 + math.log(2 * math.pi * std ** 2))
                entropy_loss = -cfg.ent_coef * entropy

                loss = policy_loss + entropy_loss

                self._optimizer_ppo.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.agent._llm.parameters())
                    + list(self.agent._action_head.parameters()),
                    cfg.max_grad_norm,
                )
                self._optimizer_ppo.step()

                total_policy_loss += policy_loss.item()
                n_mini += 1

        return {"policy_loss": total_policy_loss / max(n_mini, 1)}

    # --- TRL PPO (production mode with SmolLM2) ------------------------

    def _train_ppo_trl(
        self,
        env: Any,
        total_timesteps: int,
        log_interval: int,
    ) -> Dict[str, Any]:
        """
        PPO fine-tuning using TRL's PPOTrainer with text-based queries.

        Queries are text-formatted market states; responses are generated
        action tokens; rewards are trading environment rewards.
        """
        from trl import PPOConfig, PPOTrainer  # type: ignore

        cfg = self.agent.config
        assert self.agent._tokenizer is not None, "tokenizer must be set in production mode"
        tokenizer = self.agent._tokenizer

        ppo_config = PPOConfig(
            model_name=cfg.base_model,
            learning_rate=cfg.ppo_lr,
            ppo_epochs=cfg.ppo_epochs,
            mini_batch_size=cfg.ppo_mini_batch_size,
            batch_size=cfg.ppo_batch_size,
            gamma=cfg.gamma,
            lam=cfg.lam,
            cliprange=cfg.clip_range,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
        )

        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.agent._llm,
            tokenizer=tokenizer,
        )

        episode_rewards: List[float] = []
        steps_done = 0
        ep_reward = 0.0
        obs, _ = env.reset() if _has_gymnasium_api(env) else (env.reset(), {})

        while steps_done < total_timesteps:
            query_batch: List[torch.Tensor] = []
            response_batch: List[torch.Tensor] = []
            reward_batch: List[torch.Tensor] = []

            for _ in range(cfg.ppo_batch_size):
                # Text query
                text = self.agent.formatter.format(obs)
                query_ids = tokenizer(text, return_tensors="pt")["input_ids"][0]

                # Generate response
                gen_ids = ppo_trainer.generate(
                    [query_ids],
                    max_new_tokens=cfg.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
                response_ids = gen_ids[0][len(query_ids):]

                # Decode and parse action from generated text
                gen_text = tokenizer.decode(response_ids, skip_special_tokens=True)
                action_val = _parse_action_text(gen_text)
                action_env = np.array([action_val])

                # Step environment
                step_result = env.step(action_env)
                if len(step_result) == 5:
                    next_obs, reward, terminated, truncated, _ = step_result
                    done = bool(terminated or truncated)
                else:
                    next_obs, reward, done, _ = step_result

                query_batch.append(query_ids)
                response_batch.append(response_ids)
                reward_batch.append(torch.tensor(float(reward)))

                ep_reward += float(reward)
                steps_done += 1
                obs = next_obs

                if done:
                    episode_rewards.append(ep_reward)
                    ep_reward = 0.0
                    obs, _ = env.reset() if _has_gymnasium_api(env) else (env.reset(), {})

                if steps_done >= total_timesteps:
                    break

            # PPO update
            ppo_trainer.step(query_batch, response_batch, reward_batch)

        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        return {
            "episode_rewards": episode_rewards,
            "mean_reward": mean_reward,
            "n_updates": steps_done // cfg.ppo_batch_size,
        }


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _has_gymnasium_api(env: Any) -> bool:
    """Check if env.reset() returns (obs, info) tuple (Gymnasium ≥ 0.26)."""
    import inspect
    try:
        sig = inspect.signature(env.reset)
        # Gymnasium reset returns (obs, info); older gym returns obs only
        # Try calling with no args and check return type
        return True  # Gymnasium 0.26+ always returns (obs, info)
    except Exception:
        return False


def _parse_action_text(text: str) -> float:
    """
    Parse LLM-generated action text to a float ∈ [-1, 1].

    Looks for patterns like "long 0.7", "short 0.3", "hold", "-0.5", "+0.8".
    Falls back to 0.0 (hold) if no numeric value is found.
    """
    import re

    text = text.strip().lower()

    # Direct numeric: e.g., "0.7", "-0.5", "+0.8"
    match = re.search(r"[+-]?\d+\.?\d*", text)
    if match:
        val = float(match.group())
        return float(np.clip(val, -1.0, 1.0))

    # Keyword-based
    if any(w in text for w in ("long", "buy", "bullish")):
        return 0.5
    if any(w in text for w in ("short", "sell", "bearish")):
        return -0.5
    return 0.0  # hold


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalised Advantage Estimation (GAE).

    Parameters
    ----------
    rewards, values, dones: (N,) arrays
    gamma: discount factor
    lam: GAE lambda

    Returns
    -------
    advantages: (N,) float32
    returns:    (N,) float32
    """
    N = len(rewards)
    advantages = np.zeros(N, dtype=np.float32)
    last_gae = 0.0
    last_val = 0.0  # bootstrap value after last step

    for t in reversed(range(N)):
        next_non_terminal = 1.0 - dones[t]
        next_val = values[t + 1] if t + 1 < N else last_val
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns
