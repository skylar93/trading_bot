"""
Agent Communication Protocol — Intention Sharing.

Week 22 implementation.

In a standard ensemble, each sub-agent independently outputs an action and the
meta-controller blends the results by weight.  This module adds an *intention
layer*: before the meta-controller aggregates, every sub-agent publishes a
lightweight "intention vector" that encodes its current market view.  The
aggregated intention is then fed back into every agent's next observation
(optional) and into the meta-controller's input feature space.

This follows the SeqML intention-aware communication paradigm, adapted for
the non-sequential SB3 ensemble setup used here.

Data model
----------
Each ``AgentIntention`` carries four scalars:

    direction        : float ∈ [-1, 1]   — strong sell ↔ strong buy
    confidence       : float ∈ [0, 1]    — certainty in the direction
    horizon          : int  ≥ 1          — expected holding period in env steps
    risk_assessment  : float ∈ [0, 1]    — perceived risk (0 = calm, 1 = alarming)

``CommunicationBus`` collects one intention per registered agent per step,
provides aggregated views, and resets at the end of each step.

Usage
-----
    bus = CommunicationBus(n_agents=4)

    # Each agent publishes its intention after computing its action:
    for i, agent in enumerate(agents):
        intention = AgentIntention(direction=agent.direction,
                                   confidence=agent.confidence,
                                   horizon=10,
                                   risk_assessment=agent.risk)
        bus.publish(i, intention)

    # Meta-controller reads aggregated features (n_agents * 4 floats):
    agg = bus.get_aggregated()    # shape (n_agents * 4,)

    # Or get per-agent summary:
    summary = bus.get_summary()   # shape (4,)  — mean across agents

    bus.reset()  # call before the next env step
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Number of fields in one AgentIntention vector
INTENTION_DIM: int = 4


# ---------------------------------------------------------------------------
# AgentIntention data structure
# ---------------------------------------------------------------------------

@dataclass
class AgentIntention:
    """
    Lightweight intention vector published by one sub-agent per step.

    Parameters
    ----------
    direction : float
        Market directional view in [-1, 1].
        -1 = strong sell, 0 = neutral, +1 = strong buy.
    confidence : float
        Certainty in the direction, in [0, 1].
    horizon : int
        Expected holding period in environment steps (≥ 1).
        Normalised to [0, 1] internally using a logarithmic scale
        (so short and long horizons are both representable).
    risk_assessment : float
        Perceived market risk in [0, 1].  High values may suppress
        the meta-controller's willingness to take concentrated positions.
    """

    direction: float = 0.0
    confidence: float = 0.5
    horizon: int = 1
    risk_assessment: float = 0.5

    def __post_init__(self) -> None:
        self.direction = float(np.clip(self.direction, -1.0, 1.0))
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        self.horizon = max(1, int(self.horizon))
        self.risk_assessment = float(np.clip(self.risk_assessment, 0.0, 1.0))

    def to_vector(self) -> np.ndarray:
        """Return a length-4 float32 feature vector.

        Encoding:
            [direction, confidence, log_horizon_norm, risk_assessment]

        ``horizon`` is normalised to [0, 1] as:
            log_horizon_norm = log(horizon) / log(max_horizon)
        where max_horizon = 1000 (arbitrary ceiling).
        """
        max_h = 1000.0
        h_norm = float(np.log1p(self.horizon) / np.log1p(max_h))
        return np.array(
            [self.direction, self.confidence, h_norm, self.risk_assessment],
            dtype=np.float32,
        )

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "AgentIntention":
        """Reconstruct from a length-4 vector (inverse of ``to_vector``)."""
        v = np.asarray(v, dtype=np.float32)
        max_h = 1000.0
        horizon = int(np.expm1(v[2] * np.log1p(max_h)))
        return cls(
            direction=float(v[0]),
            confidence=float(v[1]),
            horizon=max(1, horizon),
            risk_assessment=float(v[3]),
        )

    @classmethod
    def neutral(cls) -> "AgentIntention":
        """Return a zero-information (neutral) intention."""
        return cls(direction=0.0, confidence=0.0, horizon=1, risk_assessment=0.5)


# ---------------------------------------------------------------------------
# CommunicationBus
# ---------------------------------------------------------------------------

class CommunicationBus:
    """
    Collects agent intentions, aggregates them, and provides features to
    the meta-controller and optionally back to each agent.

    Parameters
    ----------
    n_agents : int
        Number of sub-agents registered on this bus.

    Attributes
    ----------
    intentions : dict[int, AgentIntention]
        Current-step intention published by each agent.
    history : list[dict]
        Per-step snapshots (kept for logging / analysis, not used internally).

    Example
    -------
    >>> bus = CommunicationBus(n_agents=4)
    >>> for i in range(4):
    ...     bus.publish(i, AgentIntention(direction=0.5, confidence=0.8,
    ...                                  horizon=10, risk_assessment=0.3))
    >>> agg = bus.get_aggregated()
    >>> assert agg.shape == (4 * 4,)  # n_agents * INTENTION_DIM
    >>> bus.reset()
    """

    def __init__(self, n_agents: int) -> None:
        if n_agents < 1:
            raise ValueError(f"n_agents must be ≥ 1, got {n_agents}")
        self.n_agents = n_agents
        self.intentions: Dict[int, AgentIntention] = {}
        self.history: List[Dict] = []

        logger.info("CommunicationBus initialised — n_agents=%d", n_agents)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, agent_id: int, intention: AgentIntention) -> None:
        """
        Register an agent's intention for the current step.

        Parameters
        ----------
        agent_id : int
            Index of the publishing agent (0 … n_agents-1).
        intention : AgentIntention
            The agent's current intention.
        """
        if not (0 <= agent_id < self.n_agents):
            raise ValueError(
                f"agent_id {agent_id} out of range [0, {self.n_agents})"
            )
        self.intentions[agent_id] = intention

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def get_aggregated(self) -> np.ndarray:
        """
        Return a flat feature vector of all agents' intentions.

        Missing (unpublished) agents are filled with neutral intentions.

        Returns
        -------
        (n_agents * INTENTION_DIM,) float32 array
        """
        parts = []
        for i in range(self.n_agents):
            intent = self.intentions.get(i, AgentIntention.neutral())
            parts.append(intent.to_vector())
        return np.concatenate(parts, axis=0)  # (n_agents * 4,)

    def get_summary(self) -> np.ndarray:
        """
        Return the *mean* intention across all agents — a 4-dim summary.

        Useful when the downstream consumer doesn't care about individual
        agent details, only the collective "mood".

        Returns
        -------
        (INTENTION_DIM,) float32 array
        """
        agg = self.get_aggregated().reshape(self.n_agents, INTENTION_DIM)
        return agg.mean(axis=0)

    def get_agent_intention(self, agent_id: int) -> AgentIntention:
        """Return the most-recently published intention for one agent."""
        return self.intentions.get(agent_id, AgentIntention.neutral())

    def consensus_direction(self) -> float:
        """
        Compute a confidence-weighted consensus direction across all agents.

        Returns
        -------
        float ∈ [-1, 1] — positive = net bullish, negative = net bearish
        """
        directions, weights = [], []
        for i in range(self.n_agents):
            intent = self.intentions.get(i, AgentIntention.neutral())
            directions.append(intent.direction)
            weights.append(intent.confidence + 1e-8)  # avoid zero-weight
        weights = np.array(weights, dtype=np.float32)
        directions = np.array(directions, dtype=np.float32)
        return float((directions * weights).sum() / weights.sum())

    def mean_risk(self) -> float:
        """Return the average risk assessment across all agents."""
        risks = [
            self.intentions.get(i, AgentIntention.neutral()).risk_assessment
            for i in range(self.n_agents)
        ]
        return float(np.mean(risks))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Snapshot current intentions to history, then clear for the next step.
        Call this at the end of each environment step.
        """
        if self.intentions:
            self.history.append(dict(self.intentions))
        self.intentions = {}

    def step_history_len(self) -> int:
        """Number of completed steps stored in history."""
        return len(self.history)

    # ------------------------------------------------------------------
    # Convenience: build from action (simple heuristic encoder)
    # ------------------------------------------------------------------

    @staticmethod
    def intention_from_action(
        action: float,
        policy_entropy: float = 0.5,
        horizon: int = 1,
        volatility: float = 0.5,
    ) -> AgentIntention:
        """
        Build a heuristic AgentIntention from a standard SB3 action scalar.

        This allows existing SB3 agents to participate in the communication
        protocol without adding an explicit intention head.

        Parameters
        ----------
        action : float
            Scalar action ∈ [-1, 1] (position size from SB3 agent).
        policy_entropy : float
            Entropy of the action distribution (high → low confidence).
            Should be normalised to [0, 1] by the caller.
        horizon : int
            Planned holding steps (caller's choice; default 1 step).
        volatility : float
            Perceived market volatility ∈ [0, 1] (used as risk proxy).

        Returns
        -------
        AgentIntention
        """
        direction = float(np.clip(action, -1.0, 1.0))
        confidence = max(0.0, 1.0 - float(np.clip(policy_entropy, 0.0, 1.0)))
        return AgentIntention(
            direction=direction,
            confidence=confidence,
            horizon=max(1, int(horizon)),
            risk_assessment=float(np.clip(volatility, 0.0, 1.0)),
        )
