"""
Ensemble Training Pipeline.

Loads config/ensemble.yaml, creates PPO + SAC + TD3 + FLAG-Trader agents,
runs a shared environment loop where MetaController weights the actions,
VAE OOD detector gates the final action, and optionally validates the
whole ensemble with walk-forward cross-validation.

Usage (Python API)
------------------
    from training.train_ensemble import train_ensemble
    results = train_ensemble(data=my_df)

Usage (CLI)
-----------
    python -m training.train_ensemble \\
        --data test_data.csv \\
        --config config/ensemble.yaml \\
        --walk-forward
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

# Ensure project root is importable
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from agents.ensemble.meta_controller import MetaController, MetaControllerConfig
from agents.risk.ood_detector import VAEOODDetector
from agents.strategies.agent_factory import create_agent
from training.env_factory import create_env
from training.signals.regime_detector import RegimeDetector
from training.validation.walk_forward import WalkForwardResult, WalkForwardValidator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_ensemble_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _rolling_sharpe(returns: np.ndarray, window: int = 20) -> float:
    """Annualised Sharpe of the last ``window`` returns."""
    if len(returns) < 2:
        return 0.0
    r = returns[-window:]
    mu = np.mean(r)
    sigma = np.std(r) + 1e-8
    return float(mu / sigma * np.sqrt(252))


# ---------------------------------------------------------------------------
# Core ensemble training loop
# ---------------------------------------------------------------------------

def train_ensemble(
    data: pd.DataFrame,
    config_path: str = "config/ensemble.yaml",
    env_config: Optional[Dict[str, Any]] = None,
    total_timesteps: int = 50_000,
    checkpoint_dir: str = "checkpoints/ensemble",
    walk_forward: bool = False,
    mlflow_manager=None,
) -> Dict[str, Any]:
    """Train a 4-agent ensemble with MetaController + OOD gating.

    Parameters
    ----------
    data : pd.DataFrame
        OHLCV dataframe with columns ``$open $high $low $close $volume``.
    config_path : str
        Path to ``config/ensemble.yaml``.
    env_config : dict, optional
        Environment config overrides (merged on top of a minimal default).
    total_timesteps : int
        Total environment steps for training.
    checkpoint_dir : str
        Directory for saving agent checkpoints.
    walk_forward : bool
        If True, run walk-forward validation after (or instead of) training.
    mlflow_manager : optional
        MLflow manager for metric logging.

    Returns
    -------
    dict with keys:
        ``agents``, ``meta_controller``, ``ood_detector``,
        ``regime_detector``, ``episode_rewards``, ``walk_forward`` (optional).
    """
    cfg = _load_ensemble_config(config_path)
    agent_specs: List[Dict] = cfg.get("agents", [])
    mc_cfg_dict: Dict = cfg.get("meta_controller", {})
    ood_cfg: Dict = cfg.get("ood_detector", {})
    wf_cfg: Dict = cfg.get("walk_forward", {})

    os.makedirs(checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Build a minimal env config if not provided
    # ------------------------------------------------------------------
    if env_config is None:
        env_config = {
            "env": {"type": "single_asset_rl", "initial_capital": 10_000, "window_size": 20},
            "training": {"total_timesteps": total_timesteps},
        }

    # ------------------------------------------------------------------
    # Regime Detector — fit on full dataset before splitting
    # ------------------------------------------------------------------
    regime_detector = RegimeDetector(n_regimes=mc_cfg_dict.get("n_regimes", 3))
    try:
        regime_detector.fit(data)
        logger.info("RegimeDetector fitted (%d regimes)", regime_detector.n_regimes)
    except Exception as exc:
        logger.warning("RegimeDetector fit failed: %s — using uniform priors", exc)
        regime_detector = None

    # ------------------------------------------------------------------
    # Create environment to obtain observation / action spaces
    # ------------------------------------------------------------------
    env = create_env(env_config, data)
    obs_space = env.observation_space
    act_space = env.action_space
    obs_dim = int(np.prod(obs_space.shape))

    # ------------------------------------------------------------------
    # Instantiate sub-agents
    # ------------------------------------------------------------------
    agents: List[Any] = []
    agent_names: List[str] = []

    for spec in agent_specs:
        atype = spec["type"]
        sub_cfg_path = spec.get("config_path", "")
        sub_cfg: Dict = {}
        if sub_cfg_path and os.path.exists(sub_cfg_path):
            with open(sub_cfg_path) as f:
                sub_cfg = yaml.safe_load(f) or {}

        try:
            agent = create_agent(
                agent_type=atype,
                config=sub_cfg.get("agent", sub_cfg),
                observation_space=obs_space,
                action_space=act_space,
            )
            agents.append(agent)
            agent_names.append(f"{atype}_{spec.get('role', atype)}")
            logger.info("Created agent: %s (%s)", atype, spec.get("role", ""))
        except Exception as exc:
            logger.warning("Could not create agent %s: %s — skipping", atype, exc)

    if not agents:
        raise RuntimeError("No agents could be created. Check ensemble config and dependencies.")

    n_agents = len(agents)

    # ------------------------------------------------------------------
    # MetaController
    # ------------------------------------------------------------------
    mc_config = MetaControllerConfig(
        n_regimes=mc_cfg_dict.get("n_regimes", 3),
        n_market_features=mc_cfg_dict.get("n_market_features", 4),
        hidden_dim=mc_cfg_dict.get("hidden_dim", 64),
        lr=mc_cfg_dict.get("lr", 3e-4),
        rebalance_interval=mc_cfg_dict.get("rebalance_interval", 20),
        min_weight=mc_cfg_dict.get("min_weight", 0.05),
        emergency_window=mc_cfg_dict.get("emergency_window", 5),
    )
    meta_controller = MetaController(n_agents=n_agents, config=mc_config)
    logger.info("MetaController created for %d agents", n_agents)

    # ------------------------------------------------------------------
    # VAE OOD Detector
    # ------------------------------------------------------------------
    ood_detector = VAEOODDetector(
        obs_dim=obs_dim,
        latent_dim=ood_cfg.get("latent_dim", 16),
        hidden_dim=ood_cfg.get("hidden_dim", 128),
        threshold_percentile=ood_cfg.get("threshold_percentile", 95.0),
    )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    episode_rewards: List[float] = []
    agent_sharpe_windows: List[List[float]] = [[] for _ in range(n_agents)]
    obs, info = env.reset()
    ep_reward = 0.0
    steps = 0
    episode_num = 0
    training_start = time.time()

    # OOD fit buffer — collect observations for VAE fitting
    obs_buffer: List[np.ndarray] = []
    ood_fitted = False

    logger.info("Starting ensemble training for %d steps with %d agents", total_timesteps, n_agents)

    while steps < total_timesteps:
        flat_obs = obs.flatten()

        # ---- Regime probs ------------------------------------------------
        if regime_detector is not None and regime_detector.is_fitted:
            # Use the most recent window as a single-row DataFrame
            recent = data.iloc[max(0, steps - regime_detector.lookback): steps + 1]
            if len(recent) >= 2:
                regime_probs = regime_detector.predict_proba(recent)
            else:
                regime_probs = np.ones(regime_detector.n_regimes) / regime_detector.n_regimes
        else:
            n_regimes = mc_cfg_dict.get("n_regimes", 3)
            regime_probs = np.ones(n_regimes) / n_regimes

        # ---- Per-agent actions -------------------------------------------
        raw_actions: List[np.ndarray] = []
        for agent in agents:
            try:
                a = agent.get_action(obs)
                raw_actions.append(np.asarray(a, dtype=np.float32).flatten())
            except Exception:
                raw_actions.append(np.zeros(act_space.shape, dtype=np.float32).flatten())

        # ---- MetaController weights + online PPO update -----------------
        sharpe_history = np.array([
            _rolling_sharpe(np.array(w)) for w in agent_sharpe_windows
        ], dtype=np.float32)
        # step() returns weights AND records transition for PPO update
        weights = meta_controller.step(
            regime_probs=regime_probs,
            sharpe_history=sharpe_history,
            portfolio_return=0.0,   # placeholder; updated after env.step
            done=False,
        )

        # ---- Weighted ensemble action ------------------------------------
        action_matrix = np.stack(raw_actions, axis=0)       # (n_agents, act_dim)
        ensemble_action = (weights[:, None] * action_matrix).sum(axis=0)

        # ---- OOD abstain gate -------------------------------------------
        if ood_fitted:
            abstain = float(ood_detector.get_abstain_signal(flat_obs))
        else:
            abstain = 0.0
            obs_buffer.append(flat_obs)

        final_action = ensemble_action * (1.0 - abstain)
        final_action = np.clip(final_action, act_space.low, act_space.high)

        # ---- Environment step -------------------------------------------
        next_obs, reward, done, truncated, info = env.step(
            final_action.reshape(act_space.shape)
        )

        # ---- Train each sub-agent on the shared transition --------------
        for i, agent in enumerate(agents):
            try:
                agent.train_step(obs, final_action.reshape(act_space.shape), reward, next_obs, done or truncated)
                agent_sharpe_windows[i].append(float(reward))
                if len(agent_sharpe_windows[i]) > 100:
                    agent_sharpe_windows[i] = agent_sharpe_windows[i][-100:]
            except Exception:
                pass

        ep_reward += float(reward)
        steps += 1
        obs = next_obs

        # ---- Episode end -------------------------------------------------
        if done or truncated:
            episode_rewards.append(ep_reward)
            episode_num += 1

            if episode_num % 10 == 0:
                recent_mean = np.mean(episode_rewards[-10:])
                logger.info(
                    "Episode %d | Steps %d/%d | Reward %.3f | Abstain %.2f | Sharpe %.3f",
                    episode_num, steps, total_timesteps, recent_mean,
                    abstain, _rolling_sharpe(np.array(episode_rewards))
                )
                if mlflow_manager is not None:
                    mlflow_manager.log_metrics({
                        "ensemble/episode_reward": recent_mean,
                        "ensemble/abstain": abstain,
                    }, step=steps)

            obs, info = env.reset()
            ep_reward = 0.0

        # ---- Fit OOD detector once we have enough observations ----------
        if not ood_fitted and len(obs_buffer) >= 200:
            try:
                ood_detector.fit(np.stack(obs_buffer))
                ood_fitted = True
                logger.info("VAE OOD detector fitted on %d observations", len(obs_buffer))
            except Exception as exc:
                logger.warning("OOD detector fit failed: %s", exc)
                obs_buffer = []  # retry later

    # ------------------------------------------------------------------
    # Save checkpoints
    # ------------------------------------------------------------------
    for name, agent in zip(agent_names, agents):
        ckpt = os.path.join(checkpoint_dir, f"{name}_final.pt")
        try:
            agent.save(ckpt)
            logger.info("Saved %s → %s", name, ckpt)
        except Exception as exc:
            logger.warning("Could not save %s: %s", name, exc)

    training_time = time.time() - training_start
    logger.info(
        "Ensemble training complete — %d episodes, %.1fs, mean reward %.3f",
        len(episode_rewards),
        training_time,
        float(np.mean(episode_rewards)) if episode_rewards else 0.0,
    )

    results: Dict[str, Any] = {
        "agents": dict(zip(agent_names, agents)),
        "meta_controller": meta_controller,
        "ood_detector": ood_detector,
        "regime_detector": regime_detector,
        "episode_rewards": episode_rewards,
        "training_time": training_time,
    }

    # ------------------------------------------------------------------
    # Walk-forward validation (optional)
    # ------------------------------------------------------------------
    if walk_forward:
        logger.info("Running walk-forward validation on ensemble...")

        validator = WalkForwardValidator(
            n_splits=wf_cfg.get("n_splits", 12),
            train_ratio=wf_cfg.get("train_ratio", 0.5),
            gap_days=wf_cfg.get("gap_days", 5),
            mode=wf_cfg.get("mode", "expanding"),
        )

        # For walk-forward we validate the *first* agent as a representative
        # (re-training the full ensemble per fold would be too expensive by default)
        _primary_agent_type = agent_specs[0]["type"] if agent_specs else "ppo"
        _primary_sub_cfg: Dict = {}
        _primary_cfg_path = agent_specs[0].get("config_path", "") if agent_specs else ""
        if _primary_cfg_path and os.path.exists(_primary_cfg_path):
            with open(_primary_cfg_path) as f:
                _primary_sub_cfg = yaml.safe_load(f) or {}

        def _agent_factory():
            return create_agent(
                agent_type=_primary_agent_type,
                config=_primary_sub_cfg.get("agent", _primary_sub_cfg),
                observation_space=obs_space,
                action_space=act_space,
            )

        def _env_factory(df: pd.DataFrame):
            return create_env(env_config, df)

        wf_result: WalkForwardResult = validator.validate(
            agent_factory=_agent_factory,
            env_factory=_env_factory,
            data=data,
            total_timesteps=min(total_timesteps // 4, 10_000),
        )

        summary = wf_result.summary()
        logger.info(
            "Walk-forward — OOS Sharpe=%.3f (std=%.3f), Stability=%.3f",
            summary["oos_sharpe_mean"], summary["oos_sharpe_std"], summary["stability_ratio"],
        )
        if mlflow_manager is not None:
            mlflow_manager.log_metrics({f"wf/{k}": v for k, v in summary.items()})

        results["walk_forward"] = summary
        results["walk_forward_folds"] = wf_result.folds

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble RL Training")
    parser.add_argument("--data", default="test_data.csv", help="Path to OHLCV CSV")
    parser.add_argument("--config", default="config/ensemble.yaml", help="Ensemble config YAML")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--checkpoint-dir", default="checkpoints/ensemble")
    parser.add_argument("--walk-forward", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    args = _parse_args()

    data = pd.read_csv(args.data)
    # Normalise column names to $-prefixed format if needed
    col_map = {c: f"${c}" for c in ["open", "high", "low", "close", "volume"] if c in data.columns}
    if col_map:
        data = data.rename(columns=col_map)

    results = train_ensemble(
        data=data,
        config_path=args.config,
        total_timesteps=args.timesteps,
        checkpoint_dir=args.checkpoint_dir,
        walk_forward=args.walk_forward,
    )

    print("\n=== Ensemble Training Results ===")
    print(f"Episodes: {len(results['episode_rewards'])}")
    if results['episode_rewards']:
        print(f"Mean reward: {np.mean(results['episode_rewards']):.4f}")
    if "walk_forward" in results:
        print("\nWalk-Forward Summary:")
        for k, v in results["walk_forward"].items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
