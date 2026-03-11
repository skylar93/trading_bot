"""
Hyperparameter Optimization with Optuna + ASHA.

Replaces the broken Ray Tune hyperopt with a working implementation using
Optuna (TPE sampler) and ASHA-style pruning (SuccessiveHalvingPruner).

Features
--------
- Single-objective: maximize OOS Sharpe with ASHA early stopping
- Multi-objective: maximize Sharpe + minimize max_drawdown (Pareto front)
- Full search space: learning_rate, n_steps, batch_size, n_epochs, gamma,
  gae_lambda, ent_coef, clip_range, vf_coef, max_grad_norm,
  feature_extractor, reward_weights.pnl, reward_weights.sharpe
- Trial failure handling:
    NaN / Inf  → optuna.TrialPruned
    OOM        → halve batch_size and retry once
    env/agent creation error → TrialPruned
- Intermediate Sharpe reported every 1/3 of total_timesteps for ASHA pruning
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    """Metrics recorded for a single Optuna trial."""
    trial_number: int
    params: Dict[str, Any]
    sharpe: float
    max_drawdown: float
    total_return: float
    n_timesteps: int
    duration_seconds: float
    pruned: bool = False
    failed: bool = False
    error_msg: str = ""


@dataclass
class HyperoptResult:
    """Aggregated results from a completed Optuna study."""
    n_trials: int
    n_completed: int
    n_pruned: int
    n_failed: int
    trials: List[TrialResult] = field(default_factory=list)

    # Best single-objective result
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_sharpe: float = -np.inf
    best_max_drawdown: float = 1.0

    # Multi-objective Pareto front: list of (sharpe, max_drawdown, params)
    pareto_front: List[Tuple[float, float, Dict]] = field(default_factory=list)

    study_name: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Search-space helpers
# ──────────────────────────────────────────────────────────────────────────────

def _suggest_params(trial, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Suggest hyperparameters for one trial.

    Default ranges follow the roadmap spec.
    Any range can be overridden via config['hyperopt']['parameters'].

    Returns
    -------
    dict mapping param name → suggested value.
    """
    hp_params = config.get("hyperopt", {}).get("parameters", {})

    def _rng(key: str, default: Dict) -> Dict:
        return hp_params.get(key, default)

    # ── PPO algorithm params ──────────────────────────────────────────────────
    lr = _rng("learning_rate", {})
    learning_rate = trial.suggest_float(
        "learning_rate", lr.get("min", 1e-4), lr.get("max", 1e-3), log=True
    )

    ns = _rng("n_steps", {})
    n_steps = trial.suggest_categorical(
        "n_steps", ns.get("values", [1024, 2048, 4096])
    )

    bs = _rng("batch_size", {})
    batch_size = trial.suggest_categorical(
        "batch_size", bs.get("values", [32, 64, 128, 256])
    )

    ne = _rng("n_epochs", {})
    n_epochs = trial.suggest_int(
        "n_epochs", ne.get("min", 3), ne.get("max", 15)
    )

    g = _rng("gamma", {})
    gamma = trial.suggest_float(
        "gamma", g.get("min", 0.95), g.get("max", 0.999)
    )

    gae = _rng("gae_lambda", {})
    gae_lambda = trial.suggest_float(
        "gae_lambda", gae.get("min", 0.9), gae.get("max", 0.99)
    )

    ent = _rng("ent_coef", {})
    ent_coef = trial.suggest_float(
        "ent_coef", ent.get("min", 1e-3), ent.get("max", 0.1), log=True
    )

    cr = _rng("clip_range", {})
    clip_range = trial.suggest_float(
        "clip_range", cr.get("min", 0.1), cr.get("max", 0.3)
    )

    vf = _rng("vf_coef", {})
    vf_coef = trial.suggest_float(
        "vf_coef", vf.get("min", 0.25), vf.get("max", 1.0)
    )

    mgn = _rng("max_grad_norm", {})
    max_grad_norm = trial.suggest_categorical(
        "max_grad_norm", mgn.get("values", [0.3, 0.5, 0.7, 1.0])
    )

    # ── Feature extractor ─────────────────────────────────────────────────────
    fe = _rng("feature_extractor", {})
    feature_extractor = trial.suggest_categorical(
        "feature_extractor", fe.get("values", ["conv1d", "lstm", "mlp"])
    )

    # ── Reward weights ────────────────────────────────────────────────────────
    pnl = _rng("reward_weights.pnl", {})
    reward_pnl = trial.suggest_float(
        "reward_weights.pnl", pnl.get("min", 0.2), pnl.get("max", 0.5)
    )

    sh = _rng("reward_weights.sharpe", {})
    reward_sharpe = trial.suggest_float(
        "reward_weights.sharpe", sh.get("min", 0.1), sh.get("max", 0.4)
    )

    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "feature_extractor": feature_extractor,
        "reward_weights.pnl": reward_pnl,
        "reward_weights.sharpe": reward_sharpe,
    }


def _apply_params_to_config(base_config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-copy *base_config* and overlay trial *params*.

    Special keys:
      "feature_extractor"   → config["agent"]["feature_extractor"]
      "reward_weights.pnl"  → config["env"]["reward"]["weights"]["pnl"]
      "reward_weights.sharpe" → config["env"]["reward"]["weights"]["sharpe"]
      Other dotted keys     → expanded into nested dict.
      Plain keys            → config["agent"]["sb3_params"]["ppo"][key]
    """
    cfg = copy.deepcopy(base_config)

    for key, value in params.items():
        if key == "feature_extractor":
            cfg.setdefault("agent", {})["feature_extractor"] = value

        elif key == "reward_weights.pnl":
            (cfg.setdefault("env", {})
               .setdefault("reward", {})
               .setdefault("weights", {})["pnl"]) = value

        elif key == "reward_weights.sharpe":
            (cfg.setdefault("env", {})
               .setdefault("reward", {})
               .setdefault("weights", {})["sharpe"]) = value

        elif "." in key:
            parts = key.split(".")
            d = cfg
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value

        else:
            # SB3 PPO params
            (cfg.setdefault("agent", {})
               .setdefault("sb3_params", {})
               .setdefault("ppo", {})[key]) = value

    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────────────────────────────────────

def _evaluate_agent(
    agent,
    env,
    n_episodes: int = 3,
) -> Tuple[float, float, float]:
    """
    Roll out *agent* in *env* for *n_episodes* and return
    (sharpe, max_drawdown, mean_total_return).

    Accepts SB3 models (.predict), BaseAgent-style (.get_action), or callables.
    Returns (0.0, 1.0, 0.0) on any failure.
    """
    all_log_returns: List[float] = []
    portfolio_curves: List[np.ndarray] = []

    for _ in range(n_episodes):
        try:
            obs, _ = env.reset()
        except Exception:
            return 0.0, 1.0, 0.0

        done = False
        ep_values = [1.0]

        while not done:
            try:
                if hasattr(agent, "predict"):
                    action, _ = agent.predict(obs, deterministic=True)
                elif hasattr(agent, "get_action"):
                    action = agent.get_action(obs)
                elif callable(agent):
                    action = agent(obs)
                else:
                    action = env.action_space.sample()
            except Exception:
                action = env.action_space.sample()

            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except Exception:
                break

            done = terminated or truncated

            # Track portfolio value
            port_val = info.get(
                "portfolio_value",
                ep_values[-1] * (1.0 + float(reward) * 0.01),
            )
            ep_values.append(float(port_val))

        ep_vals = np.array(ep_values, dtype=float)
        ep_vals = np.where(np.isfinite(ep_vals), ep_vals, 1.0)
        ep_vals = np.maximum(ep_vals, 1e-8)

        log_rets = np.diff(np.log(ep_vals))
        all_log_returns.extend(log_rets.tolist())
        portfolio_curves.append(ep_vals)

    if not all_log_returns:
        return 0.0, 1.0, 0.0

    returns_arr = np.array(all_log_returns, dtype=float)
    if not np.all(np.isfinite(returns_arr)):
        return 0.0, 1.0, 0.0

    # Annualised Sharpe (assuming hourly bars → 252 * 24 steps/year)
    mean_r = float(np.mean(returns_arr))
    std_r = float(np.std(returns_arr, ddof=1)) + 1e-8
    sharpe = float(mean_r / std_r * np.sqrt(252 * 24))

    # Max drawdown across all episodes
    max_dd = 0.0
    for curve in portfolio_curves:
        peak = np.maximum.accumulate(curve)
        dd = float(np.max((peak - curve) / (peak + 1e-8)))
        max_dd = max(max_dd, dd)

    # Mean total return across episodes
    total_return = float(np.mean([c[-1] / c[0] - 1.0 for c in portfolio_curves]))

    return sharpe, max_dd, total_return


# ──────────────────────────────────────────────────────────────────────────────
# OptunaHyperopt
# ──────────────────────────────────────────────────────────────────────────────

class OptunaHyperopt:
    """
    Optuna-based hyperparameter optimiser for SB3 trading agents.

    Parameters
    ----------
    config : dict
        Base training config; trial params are overlaid on a deep copy.
    env_factory : Callable[[pd.DataFrame, dict], gym.Env]
        Returns a (fresh) Gymnasium trading env.
    agent_factory : Callable[[gym.Env, dict], SB3AgentWrapper]
        Returns a (freshly constructed) SB3AgentWrapper.
    train_df : pd.DataFrame
        Data used for training inside each trial.
    val_df : pd.DataFrame
        Data used for evaluating each trial.
    n_trials : int
        Total Optuna trial budget.
    n_startup_trials : int
        Random (non-TPE) trials before the sampler warms up.
    eval_episodes : int
        Episodes per evaluation rollout.
    total_timesteps : int | None
        Steps per trial; overrides config['training']['total_timesteps'].
    multi_objective : bool
        True  → NSGAIISampler, directions=['maximize','minimize'] (Sharpe, DD).
        False → TPESampler + SuccessiveHalvingPruner (ASHA), direction='maximize'.
    timeout : float | None
        Wall-clock budget (seconds) for the whole study.
    study_name : str
        Optuna study name (for logging / storage).
    mlflow_manager : optional
        MLflowManager; if provided, best params & metrics are logged.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        env_factory: Callable,
        agent_factory: Callable,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        n_trials: int = 50,
        n_startup_trials: int = 10,
        eval_episodes: int = 3,
        total_timesteps: Optional[int] = None,
        multi_objective: bool = False,
        timeout: Optional[float] = None,
        study_name: str = "trading_hyperopt",
        mlflow_manager=None,
    ):
        self.config = config
        self.env_factory = env_factory
        self.agent_factory = agent_factory
        self.train_df = train_df
        self.val_df = val_df
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.eval_episodes = eval_episodes
        self.timeout = timeout
        self.multi_objective = multi_objective
        self.study_name = study_name
        self.mlflow_manager = mlflow_manager

        self._timesteps: int = (
            total_timesteps
            or config.get("training", {}).get("total_timesteps", 10_000)
        )
        self._results: List[TrialResult] = []

    # ── Study creation ────────────────────────────────────────────────────────

    def _create_study(self):
        import optuna

        seed = self.config.get("training", {}).get("seed", 42)

        if self.multi_objective:
            sampler = optuna.samplers.NSGAIISampler(
                population_size=20,
                seed=seed,
            )
            study = optuna.create_study(
                directions=["maximize", "minimize"],
                sampler=sampler,
                study_name=self.study_name,
            )
        else:
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=self.n_startup_trials,
                seed=seed,
            )
            pruner = optuna.pruners.SuccessiveHalvingPruner(
                min_resource=1,
                reduction_factor=3,
                min_early_stopping_rate=0,
            )
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
                study_name=self.study_name,
            )

        return study

    # ── Objective ────────────────────────────────────────────────────────────

    def _objective(self, trial):
        import optuna

        t0 = time.time()
        params = _suggest_params(trial, self.config)
        trial_cfg = _apply_params_to_config(self.config, params)
        trial_cfg.setdefault("training", {})["total_timesteps"] = self._timesteps

        # Build env + agent
        try:
            train_env = self.env_factory(self.train_df, trial_cfg)
            agent = self.agent_factory(train_env, trial_cfg)
        except Exception as exc:
            logger.warning(f"Trial {trial.number}: env/agent creation failed: {exc}")
            self._results.append(TrialResult(
                trial_number=trial.number,
                params=params,
                sharpe=0.0,
                max_drawdown=1.0,
                total_return=0.0,
                n_timesteps=0,
                duration_seconds=time.time() - t0,
                failed=True,
                error_msg=str(exc),
            ))
            raise optuna.TrialPruned()

        # Train (with OOM retry)
        try:
            self._train_agent(agent, train_env, trial_cfg, trial)
        except MemoryError:
            logger.warning(f"Trial {trial.number}: OOM — halving batch_size and retrying")
            ppo_params = (
                trial_cfg.get("agent", {})
                .get("sb3_params", {})
                .get("ppo", {})
            )
            ppo_params["batch_size"] = max(16, ppo_params.get("batch_size", 64) // 2)
            try:
                self._train_agent(agent, train_env, trial_cfg, trial)
            except Exception as exc2:
                logger.error(f"Trial {trial.number}: OOM retry failed: {exc2}")
                raise optuna.TrialPruned()
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            logger.warning(f"Trial {trial.number}: training failed: {exc}")
            self._results.append(TrialResult(
                trial_number=trial.number,
                params=params,
                sharpe=0.0,
                max_drawdown=1.0,
                total_return=0.0,
                n_timesteps=0,
                duration_seconds=time.time() - t0,
                failed=True,
                error_msg=str(exc),
            ))
            raise optuna.TrialPruned()

        # Final evaluation on val_df
        try:
            val_env = self.env_factory(self.val_df, trial_cfg)
            eval_model = getattr(agent, "model", agent)
            sharpe, max_dd, total_ret = _evaluate_agent(
                eval_model, val_env, self.eval_episodes
            )
        except Exception as exc:
            logger.warning(f"Trial {trial.number}: eval failed: {exc}")
            sharpe, max_dd, total_ret = 0.0, 1.0, 0.0

        # Prune NaN / Inf
        if not (np.isfinite(sharpe) and np.isfinite(max_dd)):
            logger.warning(
                f"Trial {trial.number}: non-finite metrics "
                f"(sharpe={sharpe}, dd={max_dd})"
            )
            self._results.append(TrialResult(
                trial_number=trial.number,
                params=params,
                sharpe=0.0,
                max_drawdown=1.0,
                total_return=total_ret,
                n_timesteps=self._timesteps,
                duration_seconds=time.time() - t0,
                pruned=True,
            ))
            raise optuna.TrialPruned()

        self._results.append(TrialResult(
            trial_number=trial.number,
            params=params,
            sharpe=sharpe,
            max_drawdown=max_dd,
            total_return=total_ret,
            n_timesteps=self._timesteps,
            duration_seconds=time.time() - t0,
        ))

        logger.info(
            f"Trial {trial.number}: sharpe={sharpe:.4f}, dd={max_dd:.4f}, "
            f"ret={total_ret:.4f} ({time.time()-t0:.1f}s)"
        )

        if self.multi_objective:
            return sharpe, max_dd  # (maximize, minimize)
        return sharpe

    def _train_agent(self, agent, env, cfg: Dict[str, Any], trial) -> None:
        """
        Train *agent* in *env*.

        Splits training into 3 equal chunks and reports intermediate Sharpe
        after each chunk so SuccessiveHalvingPruner can prune underperformers.
        """
        import optuna

        timesteps = cfg.get("training", {}).get("total_timesteps", self._timesteps)
        n_chunks = 3
        chunk_size = max(1, timesteps // n_chunks)

        for chunk_idx in range(n_chunks):
            this_chunk = (
                chunk_size
                if chunk_idx < n_chunks - 1
                else max(1, timesteps - chunk_idx * chunk_size)
            )
            reset_counter = chunk_idx == 0

            if hasattr(agent, "train"):
                agent.train(
                    env,
                    total_timesteps=this_chunk,
                    reset_num_timesteps=reset_counter,
                )
            elif hasattr(agent, "learn"):
                agent.learn(
                    this_chunk,
                    reset_num_timesteps=reset_counter,
                )
            else:
                raise ValueError("agent has no .train() or .learn() method")

            # Intermediate report for ASHA pruning (single-objective only)
            if not self.multi_objective:
                interm_sharpe = 0.0
                try:
                    val_env = self.env_factory(self.val_df, cfg)
                    eval_model = getattr(agent, "model", agent)
                    interm_sharpe, _, _ = _evaluate_agent(
                        eval_model, val_env, n_episodes=1
                    )
                    if not np.isfinite(interm_sharpe):
                        interm_sharpe = 0.0
                except Exception:
                    pass

                trial.report(interm_sharpe, step=chunk_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    # ── Run ──────────────────────────────────────────────────────────────────

    def optimize(self) -> HyperoptResult:
        """
        Run the Optuna study and return a :class:`HyperoptResult`.
        """
        import optuna

        study = self._create_study()
        try:
            study.optimize(
                self._objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                catch=(Exception,),
            )
        except KeyboardInterrupt:
            logger.info("Hyperopt interrupted by user.")

        return self._collect_results(study)

    def _collect_results(self, study) -> HyperoptResult:
        import optuna

        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        pruned = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        ]
        failed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.FAIL
        ]

        result = HyperoptResult(
            n_trials=len(study.trials),
            n_completed=len(completed),
            n_pruned=len(pruned),
            n_failed=len(failed),
            trials=list(self._results),
            study_name=self.study_name,
        )

        if self.multi_objective:
            pareto: List[Tuple[float, float, Dict]] = []
            for t in study.best_trials:
                sharpe = t.values[0] if t.values else 0.0
                max_dd = t.values[1] if (t.values and len(t.values) > 1) else 1.0
                pareto.append((sharpe, max_dd, dict(t.params)))
            result.pareto_front = pareto
            if pareto:
                best = max(pareto, key=lambda x: x[0])
                result.best_sharpe = best[0]
                result.best_max_drawdown = best[1]
                result.best_params = best[2]
        else:
            if completed:
                best_t = study.best_trial
                result.best_sharpe = float(best_t.value)
                result.best_params = dict(best_t.params)
                for tr in self._results:
                    if tr.trial_number == best_t.number:
                        result.best_max_drawdown = tr.max_drawdown
                        break

        # Log to MLflow
        if self.mlflow_manager is not None and result.best_params:
            try:
                self.mlflow_manager.log_params(result.best_params)
                self.mlflow_manager.log_metrics({
                    "hyperopt_best_sharpe": result.best_sharpe,
                    "hyperopt_best_max_drawdown": result.best_max_drawdown,
                    "hyperopt_n_trials": float(result.n_trials),
                    "hyperopt_n_completed": float(result.n_completed),
                })
            except Exception as exc:
                logger.warning(f"MLflow logging failed: {exc}")

        logger.info(
            f"Hyperopt complete: {result.n_completed}/{result.n_trials} completed, "
            f"best_sharpe={result.best_sharpe:.4f}, "
            f"best_max_drawdown={result.best_max_drawdown:.4f}"
        )
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Convenience entry-point
# ──────────────────────────────────────────────────────────────────────────────

def run_hyperopt(
    df: pd.DataFrame,
    config: Dict[str, Any],
    env_factory: Callable,
    agent_factory: Callable,
    n_trials: int = 50,
    train_ratio: float = 0.7,
    multi_objective: bool = False,
    timeout: Optional[float] = None,
    mlflow_manager=None,
    study_name: str = "trading_hyperopt",
) -> HyperoptResult:
    """
    Convenience wrapper: split *df* into train/val and run hyperopt.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset; split at *train_ratio* into train / val slices.
    config : dict
        Base training config.
    env_factory : Callable[[pd.DataFrame, dict], gym.Env]
        Creates a trading env from (df_slice, config).
    agent_factory : Callable[[gym.Env, dict], agent]
        Creates an agent from (env, config).
    n_trials : int
        Optuna trial budget.
    train_ratio : float
        Fraction of *df* used for within-trial training (rest = validation).
    multi_objective : bool
        True → Pareto optimisation (Sharpe, max_drawdown).
    timeout : float | None
        Wall-clock budget (seconds) for the whole study.
    mlflow_manager : optional
        MLflowManager for logging best results.
    study_name : str
        Optuna study name.

    Returns
    -------
    HyperoptResult
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)

    hp_cfg = config.get("hyperopt", {})
    n_startup = hp_cfg.get("n_startup_trials", 10)
    eval_eps = hp_cfg.get("eval_episodes", 3)
    trial_timesteps = hp_cfg.get("trial_timesteps", None)

    optimizer = OptunaHyperopt(
        config=config,
        env_factory=env_factory,
        agent_factory=agent_factory,
        train_df=train_df,
        val_df=val_df,
        n_trials=n_trials,
        n_startup_trials=n_startup,
        eval_episodes=eval_eps,
        total_timesteps=trial_timesteps,
        multi_objective=multi_objective,
        timeout=timeout,
        study_name=study_name,
        mlflow_manager=mlflow_manager,
    )
    return optimizer.optimize()
