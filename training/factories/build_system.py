"""
build_system — single entry-point for assembling the full trading system.

Week 61 (S28): DI refactor.  Callers pass a config dict (or YAML path);
this function wires every component in the canonical order:

    data_source → env → risk_manager → agent → trader

Any component can be overridden by passing it explicitly as a keyword
argument.  If a component is omitted and cannot be built from config,
it is returned as None and a warning is logged.

Usage
-----
    from training.factories.build_system import build_system

    components = build_system(config)               # all from config
    components = build_system(config, agent=my_agent)  # inject trained agent
    env   = components.env
    trader = components.trader
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SystemComponents:
    """Assembled trading system components."""
    data_source: Any = None    # DataSource
    env: Any = None            # SingleAssetRLTradingEnv (or other Env)
    risk_manager: Any = None   # RiskManagerBase
    agent: Any = None          # SB3 / custom agent
    trader: Any = None         # PaperTrader


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_system(
    config: dict[str, Any],
    *,
    data: Optional[pd.DataFrame] = None,
    data_source=None,
    risk_manager=None,
    agent=None,
    trader=None,
) -> SystemComponents:
    """
    Assemble the full trading system from *config*.

    Parameters
    ----------
    config:
        Loaded config dict (e.g. from ``yaml.safe_load``).
    data:
        Optional DataFrame override.  When provided, wrapped in
        ``StaticDataSource`` and used as the data source.
    data_source:
        Explicit DataSource override (takes precedence over *data*).
    risk_manager:
        Explicit RiskManagerBase override.
    agent:
        Explicit agent override.  If None and a ``model_path`` is in config,
        the function attempts to load from that path.
    trader:
        Explicit PaperTrader override.

    Returns
    -------
    SystemComponents
        Populated dataclass; any component that could not be built is None.
    """
    out = SystemComponents()

    # ── 1. DataSource ─────────────────────────────────────────────────────
    out.data_source = _build_data_source(config, data=data, override=data_source)

    # ── 2. Environment ────────────────────────────────────────────────────
    out.env = _build_env(config, data_source=out.data_source)

    # ── 3. RiskManager ────────────────────────────────────────────────────
    out.risk_manager = _build_risk_manager(config, override=risk_manager)

    # ── 4. Agent ──────────────────────────────────────────────────────────
    out.agent = _build_agent(config, env=out.env, override=agent)

    # ── 5. Trader ─────────────────────────────────────────────────────────
    out.trader = _build_trader(
        config,
        agent=out.agent,
        risk_manager=out.risk_manager,
        override=trader,
    )

    return out


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_data_source(
    config: dict[str, Any],
    *,
    data: Optional[pd.DataFrame],
    override,
):
    if override is not None:
        return override

    from data.sources.base import StaticDataSource  # lazy import

    if data is not None:
        logger.info("build_system: wrapping provided DataFrame in StaticDataSource")
        return StaticDataSource(data)

    data_cfg = config.get("data", {})
    data_path = data_cfg.get("data_path")
    if data_path and Path(data_path).exists():
        try:
            df = pd.read_csv(data_path)
            logger.info("build_system: loaded data from %s (%d rows)", data_path, len(df))
            return StaticDataSource(df)
        except Exception as exc:
            logger.warning("build_system: failed to load %s — %s", data_path, exc)

    logger.warning("build_system: no data available — data_source will be None")
    return None


def _build_env(config: dict[str, Any], *, data_source):
    from training.env_factory import create_env  # lazy import

    if data_source is None:
        logger.warning("build_system: no data_source — env will be None")
        return None

    try:
        # Pass data_source so the env uses the DI path (S27)
        env_config = dict(config)
        # create_env still accepts data=df; we inject via data_source kwarg on
        # the env directly after construction for backwards-compatibility.
        from data.sources.base import StaticDataSource
        df = data_source.df if isinstance(data_source, StaticDataSource) else None
        if df is None:
            logger.warning(
                "build_system: data_source is not StaticDataSource; "
                "env construction via create_env may fail"
            )
            return None

        env = create_env(env_config, data=df)
        # Attach the data_source so downstream code can reach it via DI
        env.data_source = data_source
        logger.info("build_system: env created (%s)", type(env).__name__)
        return env
    except Exception as exc:
        logger.warning("build_system: env creation failed — %s", exc)
        return None


def _build_risk_manager(config: dict[str, Any], *, override):
    if override is not None:
        return override

    rm_cfg = config.get("risk_management", {})
    risk_type = rm_cfg.get("type", "rl")
    try:
        from risk_management.factory import create_risk_manager  # lazy import
        # Strip the "type" key before passing to factory
        rm_kwargs = {k: v for k, v in rm_cfg.items() if k != "type"}
        rm = create_risk_manager(risk_type, rm_kwargs)
        logger.info("build_system: risk_manager created (%s)", type(rm).__name__)
        return rm
    except Exception as exc:
        logger.warning("build_system: risk_manager creation failed — %s", exc)
        return None


def _build_agent(config: dict[str, Any], *, env, override):
    if override is not None:
        return override

    model_path = config.get("model_path") or config.get("agent", {}).get("model_path")
    if not model_path:
        logger.warning("build_system: no model_path in config — agent will be None")
        return None

    if not Path(model_path).exists():
        logger.warning("build_system: model_path %s not found — agent will be None", model_path)
        return None

    try:
        from stable_baselines3 import PPO  # lazy import
        agent = PPO.load(model_path, env=env)
        logger.info("build_system: agent loaded from %s", model_path)
        return agent
    except Exception as exc:
        logger.warning("build_system: agent load failed — %s", exc)
        return None


def _build_trader(
    config: dict[str, Any],
    *,
    agent,
    risk_manager,
    override,
):
    if override is not None:
        return override

    if agent is None:
        logger.warning("build_system: no agent — trader will be None")
        return None

    try:
        from deployment.paper_trader import PaperTrader  # lazy import
        trader = PaperTrader(
            agent=agent,
            config=config,
            risk_manager=risk_manager,
        )
        logger.info("build_system: PaperTrader created")
        return trader
    except Exception as exc:
        logger.warning("build_system: PaperTrader creation failed — %s", exc)
        return None
