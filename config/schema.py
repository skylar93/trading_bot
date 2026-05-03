"""
Config Schema — Week 30 (강화: Week 41)

Pydantic v2 기반 config validation.
FullConfig(**raw_yaml_dict) 호출 시 잘못된 값이면 즉시 오류를 발생시킨다.
unknown 필드는 허용(extra='allow')하여 기존 YAML 키를 보존.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class EnvConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    window_size: int = 20
    initial_balance: float = 10000.0
    trading_fee: float = 0.001
    max_position_size: float = 1.0
    sharpe_lookback: int = 60

    @field_validator("window_size")
    @classmethod
    def window_size_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"window_size must be >= 1, got {v}")
        return v

    @field_validator("trading_fee")
    @classmethod
    def trading_fee_range(cls, v: float) -> float:
        if not (0.0 <= v <= 0.1):
            raise ValueError(f"trading_fee must be in [0, 0.1], got {v}")
        return v

    @field_validator("max_position_size")
    @classmethod
    def max_position_size_range(cls, v: float) -> float:
        if not (0.0 < v <= 10.0):
            raise ValueError(f"max_position_size must be in (0, 10], got {v}")
        return v

    cost_model: str = "spot_taker"

    @field_validator("cost_model")
    @classmethod
    def cost_model_valid(cls, v: str) -> str:
        allowed = {"spot_taker", "futures_maker"}
        if v not in allowed:
            raise ValueError(f"cost_model must be one of {allowed}, got '{v}'")
        return v


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    algo_type: str = "sb3_ppo"
    learning_rate: float = 3e-4
    feature_extractor: str = "conv1d"

    @field_validator("algo_type")
    @classmethod
    def algo_type_valid(cls, v: str) -> str:
        allowed = {"sb3_ppo", "sb3_sac", "sb3_td3", "sb3_cvar_ppo", "flag_trader"}
        if v not in allowed:
            raise ValueError(f"algo_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("learning_rate")
    @classmethod
    def learning_rate_range(cls, v: float) -> float:
        if not 0 < v <= 1.0:
            raise ValueError(f"learning_rate must be in (0, 1.0], got {v}")
        return v


class EnsembleConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    agents: List[Dict[str, Any]] = []


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    total_timesteps: int = 500_000
    device: str = "cuda"
    eval_interval: int = 10_000

    @field_validator("device")
    @classmethod
    def device_valid(cls, v: str) -> str:
        if v in ("cpu", "mps"):
            return v
        if v.startswith("cuda"):
            return v
        raise ValueError(f"device must be 'cpu', 'mps', or 'cuda[:<N>]', got '{v}'")

    @field_validator("total_timesteps")
    @classmethod
    def total_timesteps_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"total_timesteps must be >= 1, got {v}")
        return v


class FatFingerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    size_multiplier_limit: float = 5.0
    hard_cap: float = 0.0
    lookback: int = 20

    @field_validator("size_multiplier_limit")
    @classmethod
    def multiplier_nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"size_multiplier_limit must be >= 0, got {v}")
        return v

    @field_validator("hard_cap")
    @classmethod
    def hard_cap_nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"hard_cap must be >= 0, got {v}")
        return v

    @field_validator("lookback")
    @classmethod
    def lookback_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"lookback must be >= 1, got {v}")
        return v


class CircuitBreakerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    vol_threshold: float = 0.05
    window: int = 20
    cooldown: float = 300.0

    @field_validator("vol_threshold")
    @classmethod
    def vol_threshold_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"vol_threshold must be > 0, got {v}")
        return v

    @field_validator("window")
    @classmethod
    def window_ge2(cls, v: int) -> int:
        if v < 2:
            raise ValueError(f"window must be >= 2, got {v}")
        return v

    @field_validator("cooldown")
    @classmethod
    def cooldown_nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"cooldown must be >= 0, got {v}")
        return v


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    stop_loss_threshold: float = 0.05
    trailing_stop_buffer: float = 0.03
    max_drawdown_pct: float = 0.20
    portfolio_stop_loss_threshold: float = 0.15

    @field_validator("stop_loss_threshold", "trailing_stop_buffer",
                     "max_drawdown_pct", "portfolio_stop_loss_threshold")
    @classmethod
    def pct_range(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"threshold must be in (0, 1], got {v}")
        return v

    @model_validator(mode="after")
    def risk_thresholds_consistent(self):
        if self.stop_loss_threshold >= self.max_drawdown_pct:
            raise ValueError(
                f"stop_loss_threshold ({self.stop_loss_threshold}) must be < "
                f"max_drawdown_pct ({self.max_drawdown_pct})"
            )
        if self.trailing_stop_buffer >= self.max_drawdown_pct:
            raise ValueError(
                f"trailing_stop_buffer ({self.trailing_stop_buffer}) must be < "
                f"max_drawdown_pct ({self.max_drawdown_pct})"
            )
        return self


class MonitoringConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    alert_channels: List[str] = ["console"]
    drawdown_alert_threshold: float = 0.10
    daily_loss_alert: float = -500.0
    connection_timeout_seconds: float = 60.0
    verbose: bool = False
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    webhook_url: Optional[str] = None
    use_drift_detection: bool = False


class RegimeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    method: str = "hmm"
    n_regimes: int = 3


class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    method: str = "walk_forward"
    n_folds: int = 5
    train_window: int = 252
    val_window: int = 63
    test_window: int = 21


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    exchange: str = "binance"
    symbols: List[str] = ["BTC/USDT"]
    timeframe: str = "1h"
    cache_dir: str = "data/raw"

    @field_validator("timeframe")
    @classmethod
    def timeframe_valid(cls, v: str) -> str:
        allowed = {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"}
        if v not in allowed:
            raise ValueError(f"timeframe must be one of {allowed}, got '{v}'")
        return v


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    exchange_mode: str = "paper"   # "paper" | "sandbox" | "live"
    paper_mode: bool = True        # derived from exchange_mode when both present
    exchange_id: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1m"
    max_order_size: float = 0.1
    daily_loss_limit: float = -500.0
    initial_cash: float = 100_000.0
    rate_limit_calls: int = 10
    rate_limit_period: float = 1.0
    heartbeat_timeout: float = 60.0

    @field_validator("exchange_mode")
    @classmethod
    def exchange_mode_valid(cls, v: str) -> str:
        allowed = {"paper", "sandbox", "live"}
        if v not in allowed:
            raise ValueError(f"exchange_mode must be one of {allowed}, got '{v}'")
        return v

    @field_validator("max_order_size")
    @classmethod
    def max_order_size_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"max_order_size must be > 0, got {v}")
        return v

    @field_validator("daily_loss_limit")
    @classmethod
    def daily_loss_limit_negative(cls, v: float) -> float:
        if v > 0:
            raise ValueError(f"daily_loss_limit must be <= 0, got {v}")
        return v


class MLflowConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    experiment_name: str = "trading_bot"
    tracking_uri: str = "mlruns"


class PaperTradingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.02

    @field_validator("risk_per_trade")
    @classmethod
    def risk_per_trade_range(cls, v: float) -> float:
        if not (0.0 < v <= 0.5):
            raise ValueError(f"risk_per_trade must be in (0, 0.5], got {v}")
        return v


class DriftDetectionConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    method: str = "adwin"
    confidence: float = 0.01

    @field_validator("method")
    @classmethod
    def method_valid(cls, v: str) -> str:
        allowed = {"adwin", "ks", "psi", "page_hinkley"}
        if v not in allowed:
            raise ValueError(f"drift method must be one of {allowed}, got '{v}'")
        return v

    @field_validator("confidence")
    @classmethod
    def confidence_range(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError(f"confidence must be in (0, 1), got {v}")
        return v


class DataPipelineSafetyConfig(BaseModel):
    """Week 65 (S47-S49): data pipeline safety guards."""
    model_config = ConfigDict(extra="allow")

    # S47 — feed staleness
    max_staleness_sec: float = 60.0   # 0 = disabled
    staleness_enabled: bool = True

    # S48 — NaN/inf in computed features
    nan_halt_after_n: int = 5         # halt after N consecutive bad steps; 0 = never
    nan_check_enabled: bool = True

    # S49 — survivorship bias warning
    survivorship_warn: bool = True
    survivorship_min_lookback_bars: int = 0   # 0 = no minimum

    @field_validator("max_staleness_sec")
    @classmethod
    def staleness_nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"max_staleness_sec must be >= 0, got {v}")
        return v

    @field_validator("nan_halt_after_n")
    @classmethod
    def nan_halt_nonneg(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"nan_halt_after_n must be >= 0, got {v}")
        return v

    @field_validator("survivorship_min_lookback_bars")
    @classmethod
    def lookback_nonneg(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"survivorship_min_lookback_bars must be >= 0, got {v}")
        return v


class FullConfig(BaseModel):
    """
    최상위 config validation 모델.

    사용법:
        with open('config/local_3060ti.yaml') as f:
            raw = yaml.safe_load(f)
        config = FullConfig(**raw)   # validation error 발생 시 즉시 중단
    """
    model_config = ConfigDict(extra="allow")

    env: EnvConfig = EnvConfig()
    agent: AgentConfig = AgentConfig()
    ensemble: EnsembleConfig = EnsembleConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    execution: ExecutionConfig = ExecutionConfig()
    risk_management: RiskConfig = RiskConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    mlflow: MLflowConfig = MLflowConfig()
    paper_trading: PaperTradingConfig = PaperTradingConfig()
    drift_detection: DriftDetectionConfig = DriftDetectionConfig()
    regime: RegimeConfig = RegimeConfig()
    validation: ValidationConfig = ValidationConfig()
    data_pipeline_safety: DataPipelineSafetyConfig = DataPipelineSafetyConfig()
