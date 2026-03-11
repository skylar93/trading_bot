"""
Kelly Criterion-Based Position Sizer (Week 10)

Computes optimal position sizes using the Kelly Criterion, with support for:
- Full Kelly, Half-Kelly, and fractional Kelly
- Regime-aware position scaling
- Multi-asset portfolio sizing with leverage cap
- Win-rate / avg-win / avg-loss interface (binary Kelly)
- Confidence-scaled sizing

Kelly formula (continuous, Gaussian returns):
    f* = (mu - r_f) / sigma^2

Binary Kelly:
    f* = (p * b - q) / b  where p=win_prob, q=1-p, b=win/loss ratio
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class PositionSizerConfig:
    """Configuration for Kelly-based position sizing."""

    # Core method: "kelly_full" | "kelly_half" | "kelly_fractional" | "fixed"
    method: str = "kelly_half"

    # Fraction to use when method="kelly_fractional" (ignored otherwise)
    kelly_fraction: float = 0.5

    # Portfolio-level limits
    max_position_fraction: float = 0.25   # hard cap per position (fraction of portfolio)
    min_position_fraction: float = 0.01   # trades below this threshold are set to 0
    max_leverage: float = 1.0             # total gross exposure cap

    # Signal confidence multiplier (set False to disable)
    confidence_scaling: bool = True

    # Regime-aware multipliers (applied after method scaling)
    regime_limits: Dict[str, float] = field(default_factory=lambda: {
        "low_vol": 1.0,
        "medium_vol": 0.75,
        "high_vol": 0.5,
    })

    # Whether to allow short positions (negative fraction)
    allow_short: bool = False


class PositionSizer:
    """
    Kelly Criterion-based position sizer.

    Usage::

        sizer = PositionSizer(PositionSizerConfig(method="kelly_half"))
        units, info = sizer.size_position(
            expected_return=0.15,   # 15% annualised
            volatility=0.20,        # 20% annualised
            portfolio_value=100_000,
            price=250.0,
            regime="low_vol",
        )
    """

    def __init__(self, config: Optional[PositionSizerConfig] = None):
        self.config = config or PositionSizerConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    # ──────────────────────────────────────────────────────────────────────────
    # Core Kelly computations
    # ──────────────────────────────────────────────────────────────────────────

    def kelly_fraction(
        self,
        expected_return: float,
        volatility: float,
        risk_free_rate: float = 0.0,
    ) -> float:
        """
        Compute full Kelly fraction using continuous Gaussian formula.

        f* = (mu - r_f) / sigma^2

        Returns:
            Fraction of portfolio to invest, clipped to [0, 1].
        """
        if volatility <= 0:
            return 0.0

        excess = expected_return - risk_free_rate
        if excess <= 0:
            return 0.0

        f_star = excess / (volatility ** 2)
        return float(np.clip(f_star, 0.0, 1.0))

    def kelly_fraction_binary(
        self,
        win_probability: float,
        win_loss_ratio: float,
    ) -> float:
        """
        Kelly fraction for binary win/loss outcomes.

        f* = (p * b - q) / b  where p=win_prob, q=1-p, b=win/loss ratio

        Returns:
            Fraction of portfolio to bet, clipped to [0, 1].
        """
        if win_loss_ratio <= 0 or not (0.0 < win_probability < 1.0):
            return 0.0

        p, q, b = win_probability, 1.0 - win_probability, win_loss_ratio
        f_star = (p * b - q) / b
        return float(np.clip(f_star, 0.0, 1.0))

    # ──────────────────────────────────────────────────────────────────────────
    # Single-asset sizing
    # ──────────────────────────────────────────────────────────────────────────

    def size_position(
        self,
        expected_return: float,
        volatility: float,
        portfolio_value: float,
        price: float,
        risk_free_rate: float = 0.0,
        confidence: float = 1.0,
        regime: Optional[str] = None,
    ) -> Tuple[float, Dict]:
        """
        Compute position size in asset units.

        Args:
            expected_return: Annualised expected return.
            volatility:      Annualised return standard deviation.
            portfolio_value: Current total portfolio value.
            price:           Current asset price.
            risk_free_rate:  Annualised risk-free rate.
            confidence:      Signal confidence in [0, 1].
            regime:          Market regime key (matches ``regime_limits``).

        Returns:
            (units, info_dict)
        """
        # 1. Raw Kelly fraction
        f_star = self.kelly_fraction(expected_return, volatility, risk_free_rate)

        # 2. Method scaling
        method = self.config.method
        if method == "kelly_full":
            f_scaled = f_star
        elif method == "kelly_half":
            f_scaled = f_star * 0.5
        elif method == "kelly_fractional":
            f_scaled = f_star * self.config.kelly_fraction
        elif method == "fixed":
            f_scaled = self.config.max_position_fraction if f_star > 0 else 0.0
        else:
            f_scaled = f_star * 0.5

        # 3. Confidence scaling
        if self.config.confidence_scaling and 0.0 < confidence < 1.0:
            f_scaled *= confidence

        # 4. Regime multiplier
        regime_mult = 1.0
        if regime is not None:
            regime_mult = self.config.regime_limits.get(regime, 1.0)
            f_scaled *= regime_mult

        # 5. Hard cap
        f_scaled = min(f_scaled, self.config.max_position_fraction)

        # 6. Minimum threshold → floor to 0
        if f_scaled < self.config.min_position_fraction:
            f_scaled = 0.0

        # 7. Units
        capital = portfolio_value * f_scaled
        units = capital / price if price > 0 else 0.0

        info = {
            "kelly_full": f_star,
            "kelly_scaled": f_scaled,
            "regime_multiplier": regime_mult,
            "confidence": confidence,
            "capital_to_invest": capital,
            "units": units,
            "method": method,
        }
        return float(units), info

    # ──────────────────────────────────────────────────────────────────────────
    # Multi-asset portfolio sizing
    # ──────────────────────────────────────────────────────────────────────────

    def size_portfolio(
        self,
        signals: Dict[str, Dict],
        portfolio_value: float,
        prices: Dict[str, float],
        regime: Optional[str] = None,
        risk_free_rate: float = 0.0,
    ) -> Dict[str, Tuple[float, Dict]]:
        """
        Size positions for multiple assets simultaneously.

        Applies a leverage-scaling pass so that total gross exposure never
        exceeds ``config.max_leverage``.

        Args:
            signals:         {asset: {"expected_return", "volatility", "confidence"}}
            portfolio_value: Current portfolio value.
            prices:          {asset: price}
            regime:          Market regime string.
            risk_free_rate:  Risk-free rate.

        Returns:
            {asset: (units, info_dict)}
        """
        raw: Dict[str, Tuple[float, Dict]] = {}
        total_fraction = 0.0

        for asset, sig in signals.items():
            price = prices.get(asset, 0.0)
            if price <= 0:
                continue

            units, info = self.size_position(
                expected_return=sig.get("expected_return", 0.0),
                volatility=sig.get("volatility", 0.1),
                portfolio_value=portfolio_value,
                price=price,
                risk_free_rate=risk_free_rate,
                confidence=sig.get("confidence", 1.0),
                regime=regime,
            )
            raw[asset] = (units, info)
            total_fraction += info["kelly_scaled"]

        # Leverage cap rescaling
        if total_fraction > self.config.max_leverage and total_fraction > 0:
            scale = self.config.max_leverage / total_fraction
            result: Dict[str, Tuple[float, Dict]] = {}
            for asset, (units, info) in raw.items():
                new_units = units * scale
                info["kelly_scaled"] *= scale
                info["capital_to_invest"] *= scale
                info["units"] = new_units
                info["leverage_scale"] = scale
                result[asset] = (new_units, info)
            return result

        for asset, (units, info) in raw.items():
            info["leverage_scale"] = 1.0

        return raw

    # ──────────────────────────────────────────────────────────────────────────
    # Win-rate interface
    # ──────────────────────────────────────────────────────────────────────────

    def from_win_rate(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        portfolio_value: float,
        price: float,
        regime: Optional[str] = None,
    ) -> Tuple[float, Dict]:
        """
        Size a position from historical win-rate statistics (binary Kelly).

        Args:
            win_rate:        Fraction of winning trades [0, 1].
            avg_win:         Average winning trade return (positive).
            avg_loss:        Average losing trade return (positive, absolute value).
            portfolio_value: Total portfolio value.
            price:           Current asset price.
            regime:          Market regime string.

        Returns:
            (units, info_dict)
        """
        if avg_loss <= 0 or not (0.0 < win_rate < 1.0):
            return 0.0, {"reason": "invalid inputs", "units": 0.0}

        win_loss_ratio = avg_win / avg_loss
        f_star = self.kelly_fraction_binary(win_rate, win_loss_ratio)

        method = self.config.method
        if method in ("kelly_full",):
            f_scaled = f_star
        elif method == "kelly_half":
            f_scaled = f_star * 0.5
        elif method == "kelly_fractional":
            f_scaled = f_star * self.config.kelly_fraction
        else:
            f_scaled = f_star * 0.5

        regime_mult = 1.0
        if regime is not None:
            regime_mult = self.config.regime_limits.get(regime, 1.0)
            f_scaled *= regime_mult

        f_scaled = min(f_scaled, self.config.max_position_fraction)
        if f_scaled < self.config.min_position_fraction:
            f_scaled = 0.0

        capital = portfolio_value * f_scaled
        units = capital / price if price > 0 else 0.0

        info = {
            "kelly_full": f_star,
            "kelly_scaled": f_scaled,
            "win_loss_ratio": win_loss_ratio,
            "regime_multiplier": regime_mult,
            "capital_to_invest": capital,
            "units": units,
            "method": method,
        }
        return float(units), info

    # ──────────────────────────────────────────────────────────────────────────
    # Factory helpers
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, risk_cfg: Dict) -> "PositionSizer":
        """
        Build a PositionSizer from the ``risk`` section of training_config.yaml.

        Expected keys (all optional, with defaults):
            kelly.method, kelly.kelly_fraction, kelly.max_position_fraction,
            kelly.min_position_fraction, kelly.max_leverage,
            kelly.confidence_scaling, regime_position_limits
        """
        kelly_cfg = risk_cfg.get("kelly", {})
        regime_limits_raw = risk_cfg.get("regime_position_limits", {})

        config = PositionSizerConfig(
            method=kelly_cfg.get("method", "kelly_half"),
            kelly_fraction=kelly_cfg.get("kelly_fraction", 0.5),
            max_position_fraction=kelly_cfg.get("max_position_fraction", 0.25),
            min_position_fraction=kelly_cfg.get("min_position_fraction", 0.01),
            max_leverage=kelly_cfg.get("max_leverage", 1.0),
            confidence_scaling=kelly_cfg.get("confidence_scaling", True),
            regime_limits=regime_limits_raw if regime_limits_raw else {
                "low_vol": 1.0,
                "medium_vol": 0.75,
                "high_vol": 0.5,
            },
        )
        return cls(config)
