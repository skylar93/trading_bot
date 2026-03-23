"""
Config Schema: Pydantic v1/v2 호환 FullConfig 검증.

Usage
-----
    from config.schema import FullConfig
    import yaml

    with open("config/local_3060ti.yaml") as f:
        raw = yaml.safe_load(f)
    config = FullConfig(**raw)          # ValidationError 발생 시 즉시 중단
    print(config.training.device)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic v1 / v2 compat shim
# ---------------------------------------------------------------------------
try:
    from pydantic import BaseModel, field_validator, model_validator
    from pydantic import __version__ as _pydantic_version
    _PYDANTIC_V2 = int(_pydantic_version.split(".")[0]) >= 2
except ImportError:
    raise ImportError("pydantic is required: pip install pydantic")

if _PYDANTIC_V2:
    from pydantic import field_validator as _fv, model_validator as _mv

    class _Base(BaseModel):
        model_config = {"extra": "allow"}   # unknown keys silently accepted

else:
    from pydantic import validator as _validator, root_validator as _rv   # type: ignore

    class _Base(BaseModel):                  # type: ignore
        class Config:
            extra = "allow"


# ---------------------------------------------------------------------------
# Sub-config sections
# ---------------------------------------------------------------------------

class EnvConfig(_Base):
    window_size: int = 20
    initial_balance: float = 10000.0
    trading_fee: float = 0.001
    max_position_size: float = 1.0
    sharpe_lookback: int = 60
    normalize: bool = True

    if _PYDANTIC_V2:
        @field_validator("window_size")
        @classmethod
        def _window_positive(cls, v: int) -> int:
            if v < 1:
                raise ValueError(f"window_size must be >= 1, got {v}")
            return v

        @field_validator("trading_fee")
        @classmethod
        def _fee_in_range(cls, v: float) -> float:
            if not 0.0 <= v <= 0.1:
                raise ValueError(f"trading_fee must be in [0, 0.1], got {v}")
            return v
    else:
        @_validator("window_size")  # type: ignore
        def _window_positive(cls, v: int) -> int:
            if v < 1:
                raise ValueError(f"window_size must be >= 1, got {v}")
            return v

        @_validator("trading_fee")  # type: ignore
        def _fee_in_range(cls, v: float) -> float:
            if not 0.0 <= v <= 0.1:
                raise ValueError(f"trading_fee must be in [0, 0.1], got {v}")
            return v


class AgentConfig(_Base):
    algo_type: str = "sb3_cvar_ppo"
    learning_rate: float = 3e-4
    feature_extractor: str = "conv1d"


class EnsembleAgentEntry(_Base):
    type: str
    params: Dict[str, Any] = {}


class EnsembleConfig(_Base):
    enabled: bool = True
    agents: List[EnsembleAgentEntry] = []

    if _PYDANTIC_V2:
        @field_validator("agents")
        @classmethod
        def _at_least_one(cls, v: list) -> list:
            if len(v) == 0:
                logger.warning("ensemble.agents is empty — single agent mode")
            return v
    else:
        @_validator("agents")  # type: ignore
        def _at_least_one(cls, v: list) -> list:
            if len(v) == 0:
                logger.warning("ensemble.agents is empty — single agent mode")
            return v


class TrainingConfig(_Base):
    total_timesteps: int = 500_000
    device: str = "cpu"
    eval_interval: int = 10_000
    use_gpu: bool = False
    ensemble_training: str = "sequential"

    if _PYDANTIC_V2:
        @field_validator("total_timesteps")
        @classmethod
        def _ts_positive(cls, v: int) -> int:
            if v < 1:
                raise ValueError(f"total_timesteps must be >= 1, got {v}")
            return v
    else:
        @_validator("total_timesteps")  # type: ignore
        def _ts_positive(cls, v: int) -> int:
            if v < 1:
                raise ValueError(f"total_timesteps must be >= 1, got {v}")
            return v


class RiskConfig(_Base):
    stop_loss_threshold: float = 0.05
    trailing_stop_buffer: float = 0.03
    max_drawdown_pct: float = 0.20
    portfolio_stop_loss_threshold: float = 0.15

    if _PYDANTIC_V2:
        @field_validator("max_drawdown_pct")
        @classmethod
        def _dd_positive(cls, v: float) -> float:
            if not 0.0 < v <= 1.0:
                raise ValueError(f"max_drawdown_pct must be in (0, 1], got {v}")
            return v
    else:
        @_validator("max_drawdown_pct")  # type: ignore
        def _dd_positive(cls, v: float) -> float:
            if not 0.0 < v <= 1.0:
                raise ValueError(f"max_drawdown_pct must be in (0, 1], got {v}")
            return v


class MonitoringConfig(_Base):
    enabled: bool = True
    alert_channels: List[str] = ["console"]
    drawdown_alert_threshold: float = 0.10
    daily_loss_alert: float = -500.0
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    webhook_url: Optional[str] = None


class RegimeConfig(_Base):
    method: str = "hmm"
    n_regimes: int = 3


class ValidationConfig(_Base):
    method: str = "walk_forward"
    n_folds: int = 5
    train_window: int = 252
    val_window: int = 63
    test_window: int = 21


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class FullConfig(_Base):
    env: EnvConfig = EnvConfig()
    training: TrainingConfig = TrainingConfig()
    risk_management: RiskConfig = RiskConfig()
    ensemble: EnsembleConfig = EnsembleConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    regime: RegimeConfig = RegimeConfig()
    validation: ValidationConfig = ValidationConfig()
