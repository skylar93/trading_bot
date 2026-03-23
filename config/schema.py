"""
Config Schema — Week 30

Pydantic v2 기반 config validation.
FullConfig(**raw_yaml_dict) 호출 시 잘못된 값이면 즉시 오류를 발생시킨다.
unknown 필드는 허용(extra='allow')하여 기존 YAML 키를 보존.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator


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


class EnsembleConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    agents: List[dict] = []

    @field_validator("agents")
    @classmethod
    def at_least_one(cls, v: list) -> list:
        if len(v) < 1:
            raise ValueError("ensemble.agents must have at least 1 agent")
        return v


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    total_timesteps: int = 500_000
    device: str = "cuda"
    eval_interval: int = 10_000

    @field_validator("device")
    @classmethod
    def device_valid(cls, v: str) -> str:
        # cuda, cuda:0, cuda:1, mps, cpu 등 허용
        if v == "cpu" or v == "mps":
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
    training: TrainingConfig = TrainingConfig()
    risk_management: RiskConfig = RiskConfig()
