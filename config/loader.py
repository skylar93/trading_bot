"""
config/loader.py — Week 63: Config Consolidation (S38)

단일 진입점: load(env) -> dict

동작 순서:
1. 5개 base config (base, trading, risk, monitoring, deployment) 순서대로 deep_merge
2. config/env/{env}.yaml 오버라이드 적용
3. FullConfig(Pydantic v2) schema validation
4. 검증된 dict 반환

사용 예:
    from config.loader import load
    cfg = load("local_3060ti")
    cfg = load("test")
    cfg = load()  # 환경변수 TRADING_BOT_ENV 또는 기본값 "local_3060ti"
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# 5개 base config — 이 순서로 순차 deep_merge
_BASE_FILES = [
    "base.yaml",
    "trading.yaml",
    "risk.yaml",
    "monitoring.yaml",
    "deployment.yaml",
]

_CONFIG_DIR = Path(__file__).parent
_DEFAULT_ENV = "local_3060ti"


def load(env: str | None = None) -> dict[str, Any]:
    """
    Base configs + env override를 머지하고 schema validate한 뒤 dict 반환.

    Parameters
    ----------
    env:
        config/env/{env}.yaml 파일 이름 (확장자 제외).
        None이면 TRADING_BOT_ENV 환경변수, 그것도 없으면 "local_3060ti".

    Returns
    -------
    dict
        Pydantic FullConfig로 검증된 merged config dict.

    Raises
    ------
    FileNotFoundError
        base config 파일 또는 env override 파일이 없을 때.
    pydantic.ValidationError
        schema validation 실패 시.
    """
    env = env or os.environ.get("TRADING_BOT_ENV", _DEFAULT_ENV)

    # 1. Base configs 순서대로 deep_merge
    merged: dict[str, Any] = {}
    for fname in _BASE_FILES:
        fpath = _CONFIG_DIR / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Base config not found: {fpath}")
        with fpath.open() as f:
            part = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, part)
        logger.debug("Loaded base config: %s", fname)

    # 2. Env override
    env_path = _CONFIG_DIR / "env" / f"{env}.yaml"
    if not env_path.exists():
        raise FileNotFoundError(
            f"Env override not found: {env_path}. "
            f"Available envs: {_list_envs()}"
        )
    with env_path.open() as f:
        env_override = yaml.safe_load(f) or {}
    merged = _deep_merge(merged, env_override)
    logger.info("Config loaded: env=%s (%d top-level keys)", env, len(merged))

    # 3. Schema validation
    _validate(merged)

    return merged


def load_raw(path: str | Path) -> dict[str, Any]:
    """
    단일 YAML 파일을 로드하고 schema validate (레거시 호환용).

    기존 코드가 yaml.safe_load(open(path))로 config를 로드하던 것을
    점진적으로 이 함수로 교체하면 된다.
    """
    fpath = Path(path)
    if not fpath.exists():
        raise FileNotFoundError(f"Config not found: {fpath}")
    with fpath.open() as f:
        raw = yaml.safe_load(f) or {}
    _validate(raw)
    return raw


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    base에 override를 재귀적으로 덮어씌운다.
    dict는 재귀 merge, 그 외는 override 값으로 교체.
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _validate(cfg: dict[str, Any]) -> None:
    """
    FullConfig(Pydantic v2)로 schema validation 수행.
    extra='allow'이므로 unknown 키는 통과.
    """
    try:
        from config.schema import FullConfig
        FullConfig(**cfg)
    except ImportError:
        logger.warning("config.schema not importable — skipping Pydantic validation")
    except Exception as exc:
        raise exc


def _list_envs() -> list[str]:
    env_dir = _CONFIG_DIR / "env"
    if not env_dir.exists():
        return []
    return sorted(p.stem for p in env_dir.glob("*.yaml"))
