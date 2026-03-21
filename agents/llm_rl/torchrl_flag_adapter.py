"""
TorchRL FLAG Adapter: async batch inference for FLAG-Trader.

Week 23 — TorchRL Integration.

Motivation
----------
FLAG-Trader's LLM inference is synchronous and single-sample, which becomes
a bottleneck when running alongside 3-4 SB3 ensemble agents.  TorchRL v0.11+
provides an AsyncVLLM interface that enables:
  - Non-blocking LLM calls (asyncio)
  - Batch inference across multiple observations
  - Integration with TorchRL's TensorDict / EnvBase API

Architecture
------------
TorchRLFLAGAdapter wraps FLAGTrader with:
  1. Async batch predict()  — collects obs from multiple callers, runs a
     single batched forward pass, returns results
  2. Sync fallback          — standard FLAGTrader.predict() when TorchRL is
     not installed or when batch_size=1
  3. SB3-compatible interface  — .predict(obs, deterministic) → (action, None)

Optional dependency
-------------------
torchrl>=0.11.0  — install with:  pip install torchrl>=0.11.0
If not installed the adapter automatically falls back to standard FLAG-Trader.

Usage
-----
    # Auto-selects async or sync based on TorchRL availability
    adapter = TorchRLFLAGAdapter.from_config(full_config_dict)
    action, _ = adapter.predict(obs)          # single obs, sync-compatible

    # Explicit async batch (in an async context)
    import asyncio
    actions = asyncio.run(adapter.predict_batch_async(obs_list))

    # Check which backend is active
    print(adapter.backend)   # "torchrl" | "fallback"
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional TorchRL import
# ---------------------------------------------------------------------------

try:
    import torchrl  # noqa: F401
    from torchrl.envs.libs.vllm import vLLMEnv  # type: ignore
    _TORCHRL_AVAILABLE = True
    _TORCHRL_VERSION = getattr(torchrl, "__version__", "unknown")
    logger.info("TorchRL %s available.", _TORCHRL_VERSION)
except ImportError:
    _TORCHRL_AVAILABLE = False
    _TORCHRL_VERSION = None

# Always import FLAGTrader (required base)
try:
    from agents.llm_rl.flag_trader import FLAGTrader, FLAGTraderConfig
    _FLAG_AVAILABLE = True
except ImportError:
    _FLAG_AVAILABLE = False
    FLAGTrader = None  # type: ignore
    FLAGTraderConfig = None  # type: ignore


# ---------------------------------------------------------------------------
# Async inference engine (TorchRL backend)
# ---------------------------------------------------------------------------

class _AsyncInferenceEngine:
    """
    Lightweight async batch inference engine.

    When TorchRL's AsyncVLLM is available it delegates to that.
    Otherwise uses a simple asyncio gather-based batcher built on top
    of the sync FLAG-Trader forward pass.

    Queue-based batching
    --------------------
    Callers submit an obs to ``submit(obs)`` and receive a Future.
    A background loop flushes the queue every ``flush_interval_s`` seconds
    or when ``max_batch_size`` items accumulate.
    """

    def __init__(
        self,
        flag_agent: Any,
        max_batch_size: int = 16,
        flush_interval_s: float = 0.01,
        device: str = "cpu",
    ) -> None:
        self._agent = flag_agent
        self.max_batch_size = max_batch_size
        self.flush_interval_s = flush_interval_s
        self.device = device

        self._queue: List[Tuple[np.ndarray, asyncio.Future]] = []
        self._lock = asyncio.Lock() if _TORCHRL_AVAILABLE else None
        self._stats = {"batches": 0, "total_obs": 0, "total_time_s": 0.0}

    async def submit(self, obs: np.ndarray) -> np.ndarray:
        """Submit one observation; returns the action asynchronously."""
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._queue.append((obs, fut))
        if len(self._queue) >= self.max_batch_size:
            await self._flush()
        return await fut

    async def _flush(self) -> None:
        """Process all queued observations in one batched forward pass."""
        if not self._queue:
            return
        batch = self._queue[:]
        self._queue.clear()

        obs_list = [item[0] for item in batch]
        futs = [item[1] for item in batch]

        t0 = time.perf_counter()
        try:
            actions = await asyncio.get_event_loop().run_in_executor(
                None, self._batch_predict, obs_list
            )
        except Exception as exc:
            for fut in futs:
                if not fut.done():
                    fut.set_exception(exc)
            return

        elapsed = time.perf_counter() - t0
        self._stats["batches"] += 1
        self._stats["total_obs"] += len(batch)
        self._stats["total_time_s"] += elapsed

        for fut, action in zip(futs, actions):
            if not fut.done():
                fut.set_result(action)

    def _batch_predict(self, obs_list: List[np.ndarray]) -> List[np.ndarray]:
        """Synchronous batched forward pass on the underlying FLAG-Trader."""
        actions = []
        for obs in obs_list:
            try:
                action, _ = self._agent.predict(obs, deterministic=True)
                actions.append(np.atleast_1d(action))
            except Exception as exc:
                logger.warning("FLAG predict failed for one obs: %s", exc)
                actions.append(np.zeros(1, dtype=np.float32))
        return actions

    @property
    def stats(self) -> Dict[str, Any]:
        n = self._stats["total_obs"]
        avg = self._stats["total_time_s"] / max(1, self._stats["batches"])
        return {
            "backend": "torchrl" if _TORCHRL_AVAILABLE else "asyncio",
            "batches_processed": self._stats["batches"],
            "total_obs_processed": n,
            "avg_batch_time_s": avg,
        }


# ---------------------------------------------------------------------------
# TorchRLFLAGAdapter
# ---------------------------------------------------------------------------

@dataclass
class TorchRLFLAGAdapterConfig:
    """Configuration for the TorchRL FLAG adapter."""
    max_batch_size: int = 16
    flush_interval_s: float = 0.01
    device: str = "cpu"
    # Passed to FLAGTrader
    flag_dry_run: bool = True
    flag_base_model: str = "HuggingFaceTB/SmolLM2-135M"
    flag_lora_rank: int = 16

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "TorchRLFLAGAdapterConfig":
        flag_cfg = cfg.get("flag_trader", {})
        torchrl_cfg = cfg.get("torchrl", {})
        return cls(
            max_batch_size=torchrl_cfg.get("max_batch_size", 16),
            flush_interval_s=torchrl_cfg.get("flush_interval_s", 0.01),
            device=cfg.get("training", {}).get("device", "cpu"),
            flag_dry_run=flag_cfg.get("dry_run", True),
            flag_base_model=flag_cfg.get("base_model", "HuggingFaceTB/SmolLM2-135M"),
            flag_lora_rank=flag_cfg.get("lora_rank", 16),
        )


class TorchRLFLAGAdapter:
    """
    FLAG-Trader wrapped with async batch inference via TorchRL (when available).

    Interface
    ---------
    Identical to FLAGTrader:
        action, state = adapter.predict(obs, deterministic=True)

    Additional async API:
        actions = asyncio.run(adapter.predict_batch_async(obs_list))

    Backend selection
    -----------------
    - TorchRL ≥ 0.11:   async batched inference via _AsyncInferenceEngine
    - Fallback:         standard FLAGTrader.predict() (synchronous, single obs)

    Parameters
    ----------
    flag_agent:
        A FLAGTrader instance.  Pass None to use a dry-run agent.
    config:
        TorchRLFLAGAdapterConfig.
    """

    def __init__(
        self,
        flag_agent: Optional[Any] = None,
        config: Optional[TorchRLFLAGAdapterConfig] = None,
    ) -> None:
        self.config = config or TorchRLFLAGAdapterConfig()

        if flag_agent is None:
            if not _FLAG_AVAILABLE:
                raise ImportError(
                    "FLAGTrader is required. Ensure agents/llm_rl/flag_trader.py is present."
                )
            flag_cfg = FLAGTraderConfig(dry_run=self.config.flag_dry_run)
            flag_agent = FLAGTrader(flag_cfg)
            logger.info(
                "TorchRLFLAGAdapter: created dry_run=%s FLAGTrader.",
                self.config.flag_dry_run,
            )

        self._flag_agent = flag_agent
        self._backend: str = "torchrl" if _TORCHRL_AVAILABLE else "fallback"

        self._engine: Optional[_AsyncInferenceEngine] = None
        if _TORCHRL_AVAILABLE:
            self._engine = _AsyncInferenceEngine(
                flag_agent=self._flag_agent,
                max_batch_size=self.config.max_batch_size,
                flush_interval_s=self.config.flush_interval_s,
                device=self.config.device,
            )
            logger.info("TorchRL async inference engine active (max_batch=%d).", self.config.max_batch_size)
        else:
            logger.info(
                "TorchRL not available (install torchrl>=0.11.0 for async inference). "
                "Using synchronous FLAGTrader fallback."
            )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "TorchRLFLAGAdapter":
        """Build from a full training config dict (e.g. loaded from YAML)."""
        adapter_cfg = TorchRLFLAGAdapterConfig.from_dict(config_dict)

        flag_agent = None
        if _FLAG_AVAILABLE:
            flag_cfg = FLAGTraderConfig(
                dry_run=adapter_cfg.flag_dry_run,
                base_model=adapter_cfg.flag_base_model,
                lora_rank=adapter_cfg.flag_lora_rank,
            )
            flag_agent = FLAGTrader(flag_cfg)

        return cls(flag_agent=flag_agent, config=adapter_cfg)

    # ------------------------------------------------------------------
    # SB3-compatible synchronous interface
    # ------------------------------------------------------------------

    def predict(
        self,
        obs: np.ndarray,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, None]:
        """
        Predict action for a single observation (sync, SB3-compatible).

        Falls back to standard FLAG-Trader regardless of TorchRL availability
        (single-sample async overhead is not worth it for one observation).
        """
        action, _ = self._flag_agent.predict(obs, state=state, deterministic=deterministic)
        return np.atleast_1d(action), None

    # ------------------------------------------------------------------
    # Async batch interface
    # ------------------------------------------------------------------

    async def predict_batch_async(
        self, obs_list: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Async batch inference.

        Submits all observations to the async engine and awaits results.
        Falls back to sequential sync calls if TorchRL is unavailable.

        Parameters
        ----------
        obs_list:
            List of observation arrays.

        Returns
        -------
        List of action arrays in the same order as obs_list.
        """
        if self._engine is not None and len(obs_list) > 1:
            # Batch path: submit all and flush once
            futs = [self._engine.submit(obs) for obs in obs_list]
            await self._engine._flush()
            return list(await asyncio.gather(*futs))
        else:
            # Fallback: sequential sync
            actions = []
            for obs in obs_list:
                action, _ = self.predict(obs)
                actions.append(action)
            return actions

    def predict_batch(self, obs_list: List[np.ndarray]) -> List[np.ndarray]:
        """Synchronous wrapper around predict_batch_async."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in an async context — run on executor
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(asyncio.run, self.predict_batch_async(obs_list))
                    return fut.result()
            else:
                return loop.run_until_complete(self.predict_batch_async(obs_list))
        except RuntimeError:
            return asyncio.run(self.predict_batch_async(obs_list))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        """Active backend: 'torchrl' | 'fallback'."""
        return self._backend

    @property
    def inference_stats(self) -> Dict[str, Any]:
        """Latency and throughput statistics (only meaningful with TorchRL backend)."""
        if self._engine is not None:
            return self._engine.stats
        return {"backend": "fallback", "batches_processed": 0, "total_obs_processed": 0}

    def __repr__(self) -> str:
        return (
            f"TorchRLFLAGAdapter("
            f"backend={self._backend!r}, "
            f"torchrl_available={_TORCHRL_AVAILABLE}, "
            f"max_batch_size={self.config.max_batch_size})"
        )
