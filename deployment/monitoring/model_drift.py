"""
Model Prediction-Quality Drift Detector (Phase 8 A5).

Tracks 5 dimensions of agent behaviour degradation:
  1. Action distribution KL divergence (yesterday vs today)
  2. Meta-controller weight collapse (one agent dominant > N hours)
  3. Predicted vs realized return Spearman correlation
  4. Per-regime hit rate vs training baseline
  5. Action entropy collapse

All warnings are WARNING-level only — no halt (shadow policy, same as DeploymentDriftDetector).
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
from scipy.stats import spearmanr

if TYPE_CHECKING:
    from deployment.monitoring.alerter import TradingAlerter

logger = logging.getLogger(__name__)

_DEFAULTS: Dict[str, Any] = {
    "action_kl_threshold": 0.5,
    "meta_weight_collapse_pct": 0.80,
    "meta_weight_collapse_window_h": 6,
    "pred_realized_corr_min": 0.05,
    "pred_realized_window": 100,
    "hit_rate_drop_pct": 0.20,
    "shadow_mode_hours": 72,
    "steps_per_hour": 60,
    "action_bins": 10,
    "kl_window_steps": 500,
}


@dataclass
class ModelDriftEvent:
    """Single drift warning emitted by one detector dimension."""
    dimension: str
    metric_value: float
    threshold: float
    details: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class ModelDriftDetector:
    """5-dimension model prediction-quality drift detector.

    Parameters
    ----------
    config:
        Top-level alerts config dict. Reads ``model_drift`` sub-dict for
        thresholds, and ``drift.action_entropy_min`` for dim-5 threshold.
    alerter:
        Optional TradingAlerter for WARNING dispatches.
    baseline_hit_rates:
        {regime_label: hit_rate} from training.  Required for dim-4.
    _start_time:
        Override epoch start (for testing).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        alerter: Optional["TradingAlerter"] = None,
        baseline_hit_rates: Optional[Dict[int, float]] = None,
        _start_time: Optional[float] = None,
    ) -> None:
        md_cfg: Dict[str, Any] = {**_DEFAULTS, **config.get("model_drift", {})}
        drift_cfg: Dict[str, Any] = config.get("drift", {})

        self.action_kl_threshold: float = float(md_cfg["action_kl_threshold"])
        self.meta_collapse_pct: float = float(md_cfg["meta_weight_collapse_pct"])
        steps_per_hour: int = int(md_cfg["steps_per_hour"])
        self.meta_collapse_window: int = (
            int(md_cfg["meta_weight_collapse_window_h"]) * steps_per_hour
        )
        self.pred_corr_min: float = float(md_cfg["pred_realized_corr_min"])
        self.pred_corr_window: int = int(md_cfg["pred_realized_window"])
        self.hit_rate_drop: float = float(md_cfg["hit_rate_drop_pct"])
        self.action_entropy_min: float = float(
            drift_cfg.get("action_entropy_min", 0.5)
        )
        self.action_bins: int = int(md_cfg["action_bins"])
        kl_window: int = int(md_cfg["kl_window_steps"])
        shadow_hours: float = float(md_cfg["shadow_mode_hours"])

        start = _start_time if _start_time is not None else time.time()
        self._shadow_until: float = start + shadow_hours * 3600

        self.alerter = alerter
        self.baseline_hit_rates: Dict[int, float] = baseline_hit_rates or {}
        self._events: List[ModelDriftEvent] = []
        self._n_warnings: int = 0

        # Dim 1: rolling action buffer (split at midpoint for KL)
        self._actions: deque = deque(maxlen=kl_window * 2)
        self._kl_window: int = kl_window

        # Dim 2: per-agent weight history
        self._meta_weights: Dict[str, deque] = {}

        # Dim 3: predicted / realized return buffers (aligned by position)
        self._pred_returns: deque = deque(maxlen=self.pred_corr_window)
        self._realized_returns: deque = deque(maxlen=self.pred_corr_window)

        # Dim 4: (regime, won) trade outcomes
        self._regime_trades: deque = deque(maxlen=100)

        logger.info(
            "ModelDriftDetector init | shadow_h=%.1f kl_thr=%.2f meta_pct=%.0f%% "
            "corr_min=%.3f hit_drop=%.0f%% entropy_min=%.2f",
            shadow_hours,
            self.action_kl_threshold,
            self.meta_collapse_pct * 100,
            self.pred_corr_min,
            self.hit_rate_drop * 100,
            self.action_entropy_min,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def in_shadow_mode(self) -> bool:
        return time.time() < self._shadow_until

    @property
    def n_warnings(self) -> int:
        return self._n_warnings

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------

    def update(
        self,
        *,
        action: Optional[float] = None,
        predicted_return: Optional[float] = None,
        realized_return: Optional[float] = None,
        regime: Optional[int] = None,
        trade_won: Optional[bool] = None,
        meta_weights: Optional[Dict[str, float]] = None,
        timestamp: Optional[float] = None,
    ) -> List[ModelDriftEvent]:
        """Update all tracking buffers and return any new drift events.

        Parameters
        ----------
        action:
            Current ensemble action (scalar).
        predicted_return:
            Model's predicted next-step return (for dim-3).
        realized_return:
            Actual realized return at this step (for dim-3).
        regime:
            Current HMM regime label (for dim-4).
        trade_won:
            Whether the last completed trade was profitable (for dim-4).
        meta_weights:
            {agent_name: weight} from meta-controller (for dim-2).
        timestamp:
            Override current time (testing only).
        """
        ts = timestamp if timestamp is not None else time.time()
        new_events: List[ModelDriftEvent] = []

        # Dim 1 + 5: action buffer feeds both KL and entropy checks
        if action is not None:
            self._actions.append(float(action))
            ev = self._check_action_kl(ts)
            if ev:
                new_events.append(ev)
            ev = self._check_action_entropy(ts)
            if ev:
                new_events.append(ev)

        # Dim 2
        if meta_weights is not None:
            ev = self._check_meta_collapse(meta_weights, ts)
            if ev:
                new_events.append(ev)

        # Dim 3 — accept predicted and realized independently; correlate when
        # both buffers reach full window
        if predicted_return is not None:
            self._pred_returns.append(float(predicted_return))
        if realized_return is not None:
            self._realized_returns.append(float(realized_return))
        if (
            len(self._pred_returns) >= self.pred_corr_window
            and len(self._realized_returns) >= self.pred_corr_window
        ):
            ev = self._check_pred_corr(ts)
            if ev:
                new_events.append(ev)

        # Dim 4
        if regime is not None and trade_won is not None:
            self._regime_trades.append((int(regime), bool(trade_won)))
            ev = self._check_regime_hit_rate(int(regime), ts)
            if ev:
                new_events.append(ev)

        for ev in new_events:
            self._events.append(ev)
            self._n_warnings += 1
            self._dispatch(ev)

        return new_events

    # ------------------------------------------------------------------
    # Dimension checks
    # ------------------------------------------------------------------

    def _check_action_kl(self, ts: float) -> Optional[ModelDriftEvent]:
        """KL(first-half || second-half) of the rolling action buffer."""
        n = len(self._actions)
        if n < self._kl_window:
            return None
        actions = np.array(self._actions, dtype=float)
        half = n // 2
        lo = actions.min() - 1e-9
        hi = actions.max() + 1e-9
        if hi <= lo:
            return None
        bins = np.linspace(lo, hi, self.action_bins + 1)
        p, _ = np.histogram(actions[:half], bins=bins)
        q, _ = np.histogram(actions[half:], bins=bins)
        p = p.astype(float) + 1e-9
        q = q.astype(float) + 1e-9
        p /= p.sum()
        q /= q.sum()
        kl = float(np.sum(p * np.log(p / q)))
        if kl > self.action_kl_threshold:
            return ModelDriftEvent(
                dimension="action_kl",
                metric_value=kl,
                threshold=self.action_kl_threshold,
                details=f"action KL {kl:.3f} > {self.action_kl_threshold} (n={n})",
                timestamp=ts,
            )
        return None

    def _check_meta_collapse(
        self, weights: Dict[str, float], ts: float
    ) -> Optional[ModelDriftEvent]:
        """Warn when one agent holds > collapse_pct for the entire collapse window."""
        for agent, w in weights.items():
            if agent not in self._meta_weights:
                self._meta_weights[agent] = deque(maxlen=self.meta_collapse_window)
            self._meta_weights[agent].append(float(w))

        for agent, buf in self._meta_weights.items():
            if len(buf) < self.meta_collapse_window:
                continue
            if all(v > self.meta_collapse_pct for v in buf):
                mean_w = float(np.mean(buf))
                return ModelDriftEvent(
                    dimension="meta_weight_collapse",
                    metric_value=mean_w,
                    threshold=self.meta_collapse_pct,
                    details=(
                        f"agent '{agent}' weight > {self.meta_collapse_pct*100:.0f}% "
                        f"for {len(buf)} consecutive steps (mean={mean_w:.3f})"
                    ),
                    timestamp=ts,
                )
        return None

    def _check_pred_corr(self, ts: float) -> Optional[ModelDriftEvent]:
        """Spearman(predicted_return, realized_return) over last window steps."""
        pred = np.array(self._pred_returns, dtype=float)
        real = np.array(self._realized_returns, dtype=float)
        n = min(len(pred), len(real))
        corr_result = spearmanr(pred[-n:], real[-n:])
        corr = float(corr_result.statistic if hasattr(corr_result, "statistic") else corr_result[0])
        if np.isnan(corr):
            return None
        if corr < self.pred_corr_min:
            return ModelDriftEvent(
                dimension="pred_realized_corr",
                metric_value=corr,
                threshold=self.pred_corr_min,
                details=(
                    f"Spearman corr(pred, realized)={corr:.4f} "
                    f"< {self.pred_corr_min} (window={n})"
                ),
                timestamp=ts,
            )
        return None

    def _check_regime_hit_rate(self, regime: int, ts: float) -> Optional[ModelDriftEvent]:
        """Per-regime hit rate vs training baseline; requires ≥10 regime samples."""
        baseline = self.baseline_hit_rates.get(regime)
        if baseline is None:
            return None
        results = [won for (r, won) in self._regime_trades if r == regime]
        if len(results) < 10:
            return None
        hit_rate = float(sum(results)) / len(results)
        threshold = baseline - self.hit_rate_drop
        if hit_rate < threshold:
            return ModelDriftEvent(
                dimension="regime_hit_rate",
                metric_value=hit_rate,
                threshold=threshold,
                details=(
                    f"regime {regime} hit_rate={hit_rate:.3f} < "
                    f"baseline {baseline:.3f} - drop {self.hit_rate_drop:.2f} "
                    f"(n={len(results)})"
                ),
                timestamp=ts,
            )
        return None

    def _check_action_entropy(self, ts: float) -> Optional[ModelDriftEvent]:
        """Normalised Shannon entropy of recent action distribution."""
        n = len(self._actions)
        if n < 20:
            return None
        actions = np.array(list(self._actions)[-200:], dtype=float)
        lo = actions.min() - 1e-9
        hi = actions.max() + 1e-9
        if hi <= lo:
            # all identical actions → zero entropy
            norm_ent = 0.0
        else:
            bins = np.linspace(lo, hi, self.action_bins + 1)
            counts, _ = np.histogram(actions, bins=bins)
            probs = counts.astype(float) + 1e-9
            probs /= probs.sum()
            raw_ent = float(-np.sum(probs * np.log(probs)))
            max_ent = float(np.log(self.action_bins))
            norm_ent = raw_ent / max_ent if max_ent > 0 else 1.0

        if norm_ent < self.action_entropy_min:
            return ModelDriftEvent(
                dimension="action_entropy",
                metric_value=norm_ent,
                threshold=self.action_entropy_min,
                details=(
                    f"normalised action entropy {norm_ent:.4f} "
                    f"< {self.action_entropy_min}"
                ),
                timestamp=ts,
            )
        return None

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, ev: ModelDriftEvent) -> None:
        mode = "shadow" if self.in_shadow_mode else "active"
        msg = f"[ModelDrift/{mode}] [{ev.dimension}] {ev.details}"
        logger.warning(msg)
        if self.alerter is not None:
            self.alerter.send_alert(msg, level="WARNING")

    # ------------------------------------------------------------------
    # Introspection / dashboard
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return current metric values for dashboard / reporting."""
        actions = list(self._actions)
        n = len(actions)

        # KL (recompute without threshold check)
        kl: Optional[float] = None
        if n >= self._kl_window:
            arr = np.array(actions, dtype=float)
            half = n // 2
            lo, hi = arr.min() - 1e-9, arr.max() + 1e-9
            if hi > lo:
                bins = np.linspace(lo, hi, self.action_bins + 1)
                p, _ = np.histogram(arr[:half], bins=bins)
                q, _ = np.histogram(arr[half:], bins=bins)
                p = p.astype(float) + 1e-9
                q = q.astype(float) + 1e-9
                p /= p.sum()
                q /= q.sum()
                kl = float(np.sum(p * np.log(p / q)))

        # Entropy
        action_entropy: Optional[float] = None
        if n >= 20:
            arr = np.array(actions[-200:], dtype=float)
            lo, hi = arr.min() - 1e-9, arr.max() + 1e-9
            if hi <= lo:
                action_entropy = 0.0
            else:
                bins = np.linspace(lo, hi, self.action_bins + 1)
                counts, _ = np.histogram(arr, bins=bins)
                probs = counts.astype(float) + 1e-9
                probs /= probs.sum()
                raw = float(-np.sum(probs * np.log(probs)))
                mx = float(np.log(self.action_bins))
                action_entropy = raw / mx if mx > 0 else 1.0

        # Pred/realized correlation
        pred_corr: Optional[float] = None
        pred = list(self._pred_returns)
        real = list(self._realized_returns)
        if len(pred) >= self.pred_corr_window and len(real) >= self.pred_corr_window:
            r = spearmanr(pred, real)
            v = float(r.statistic if hasattr(r, "statistic") else r[0])
            pred_corr = None if np.isnan(v) else v

        # Meta weights (latest per agent)
        meta_latest: Dict[str, Optional[float]] = {
            agent: (float(list(buf)[-1]) if buf else None)
            for agent, buf in self._meta_weights.items()
        }

        # Per-regime hit rates
        by_regime: Dict[int, List[bool]] = defaultdict(list)
        for r, won in self._regime_trades:
            by_regime[r].append(won)
        regime_hit_rates = {
            r: float(sum(v)) / len(v) for r, v in by_regime.items() if v
        }

        return {
            "n_warnings": self._n_warnings,
            "in_shadow_mode": self.in_shadow_mode,
            "action_kl": kl,
            "action_entropy": action_entropy,
            "pred_realized_corr": pred_corr,
            "meta_weights": meta_latest,
            "regime_hit_rates": regime_hit_rates,
            "recent_events": [ev.to_dict() for ev in self._events[-10:]],
        }
