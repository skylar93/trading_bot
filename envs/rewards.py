"""
Multi-component reward function for RL trading environments.

All components are tanh-normalized to (-1, 1) before weighting.
Final reward is in (-1, 1) when weights sum to 1.0.
"""

import collections
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


@dataclass
class RewardConfig:
    """Configuration for MultiComponentReward.

    Weights should sum to 1.0.
    Scales control sensitivity before tanh:
      - pnl_scale=100  → 1% log-return  maps to tanh(1)  ≈ 0.76
      - sharpe_scale=1 → Sharpe of 1    maps to tanh(1)  ≈ 0.76
      - drawdown_scale=10 → 10% DD      maps to tanh(-1) ≈ -0.76
      - cost_scale=10000  → 0.01% cost  maps to tanh(-1) ≈ -0.76
    """

    # --- Component weights (must sum to 1.0) ---
    pnl_weight: float = 0.4
    sharpe_weight: float = 0.3
    drawdown_weight: float = 0.2
    cost_weight: float = 0.1

    # --- tanh scale factors ---
    pnl_scale: float = 100.0
    sharpe_scale: float = 1.0
    drawdown_scale: float = 10.0
    cost_scale: float = 10000.0

    # --- Sharpe rolling window ---
    sharpe_lookback: int = 30

    # --- Termination penalties (outside the weighted sum) ---
    bankruptcy_penalty: float = -1.0
    nan_inf_penalty: float = -0.5

    def validate(self) -> None:
        total = self.pnl_weight + self.sharpe_weight + self.drawdown_weight + self.cost_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Reward weights must sum to 1.0, got {total:.6f}")
        for name, val in [
            ("pnl_weight", self.pnl_weight),
            ("sharpe_weight", self.sharpe_weight),
            ("drawdown_weight", self.drawdown_weight),
            ("cost_weight", self.cost_weight),
        ]:
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")


class MultiComponentReward:
    """Computes a weighted, tanh-normalized reward from four trading components.

    Components:
        1. PnL      – log-return of portfolio value
        2. Sharpe   – rolling Sharpe ratio proxy
        3. Drawdown – penalty proportional to current drawdown
        4. Cost     – penalty proportional to transaction cost

    Each component is independently normalized via tanh to (-1, 1).
    The final reward is their weighted sum, also in (-1, 1) when weights
    sum to 1.0.

    Usage:
        reward_fn = MultiComponentReward(RewardConfig())
        reward_fn.reset()  # call on episode reset

        reward, components = reward_fn.compute(
            portfolio_value=10200,
            prev_portfolio_value=10000,
            peak_portfolio_value=10200,
            trade_cost=5.0,
        )
        # reward ∈ (-1, 1)
        # components: dict with 'pnl', 'sharpe', 'drawdown', 'cost', 'total'
    """

    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        self.config.validate()
        self._returns_buffer: collections.deque = collections.deque(
            maxlen=self.config.sharpe_lookback
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear internal state. Must be called on every episode reset."""
        self._returns_buffer.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        portfolio_value: float,
        prev_portfolio_value: float,
        peak_portfolio_value: float,
        trade_cost: float,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute the step reward.

        Args:
            portfolio_value:      Portfolio value after the step.
            prev_portfolio_value: Portfolio value before the step.
            peak_portfolio_value: Rolling maximum portfolio value.
            trade_cost:           Absolute transaction cost this step (e.g. in $).

        Returns:
            (reward, components)
              reward     – weighted sum in (-1, 1)
              components – dict with keys: pnl, sharpe, drawdown, cost,
                           log_return, total
        """
        eps = 1e-10
        pv = max(float(portfolio_value), eps)
        ppv = max(float(prev_portfolio_value), eps)
        pkv = max(float(peak_portfolio_value), eps)
        tc = max(float(trade_cost), 0.0)

        # 1. PnL component ─────────────────────────────────────────────
        log_ret = np.log(pv / ppv)
        log_ret = float(np.clip(log_ret, -5.0, 5.0))        # safety clip
        pnl_component = float(np.tanh(log_ret * self.config.pnl_scale))

        # Add log-return to Sharpe buffer
        self._returns_buffer.append(log_ret)

        # 2. Sharpe component ──────────────────────────────────────────
        sharpe_component = 0.0
        if len(self._returns_buffer) >= 5:
            arr = np.array(list(self._returns_buffer), dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if len(arr) >= 3:
                sharpe_raw = float(np.mean(arr) / (np.std(arr) + eps))
                sharpe_raw = float(np.clip(sharpe_raw, -5.0, 5.0))
                sharpe_component = float(np.tanh(sharpe_raw * self.config.sharpe_scale))

        # 3. Drawdown component ────────────────────────────────────────
        dd = float(np.clip((pkv - pv) / pkv, 0.0, 1.0))
        drawdown_component = float(np.tanh(-dd * self.config.drawdown_scale))

        # 4. Cost component ────────────────────────────────────────────
        cost_ratio = float(np.clip(tc / ppv, 0.0, 1.0))
        cost_component = float(np.tanh(-cost_ratio * self.config.cost_scale))

        # Weighted sum ─────────────────────────────────────────────────
        cfg = self.config
        reward = (
            cfg.pnl_weight * pnl_component
            + cfg.sharpe_weight * sharpe_component
            + cfg.drawdown_weight * drawdown_component
            + cfg.cost_weight * cost_component
        )
        reward = float(np.clip(reward, -1.0, 1.0))  # numerical safety

        components = {
            "pnl": pnl_component,
            "sharpe": sharpe_component,
            "drawdown": drawdown_component,
            "cost": cost_component,
            "log_return": log_ret,
            "total": reward,
        }
        return reward, components

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_sharpe_ratio(self) -> float:
        """Current rolling Sharpe ratio (0.0 if insufficient data)."""
        if len(self._returns_buffer) < 3:
            return 0.0
        arr = np.array(list(self._returns_buffer), dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 3:
            return 0.0
        return float(np.mean(arr) / (np.std(arr) + 1e-10))
