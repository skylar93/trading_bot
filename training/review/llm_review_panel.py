"""
LLM Review Panel: Multi-perspective strategy review using Claude.

After each walk-forward fold or paper trading session the LLMReviewPanel:
  1. Collects an AgentBehaviorSummary (action distribution, P&L, trade list, …)
  2. Sends it to Claude with 5 structured review prompts
  3. Parses the response → ReviewResult with per-perspective critique and
     optional reward weight adjustments (max ±0.05 per review cycle)
  4. Logs the full review to MLflow as an artifact

Usage:
    summary = AgentBehaviorSummary(...)
    panel = LLMReviewPanel(api_key=os.getenv("ANTHROPIC_API_KEY"))
    result = panel.review(summary)
    new_weights = panel.adjust_reward_weights(result, current_weights)
    panel.log_to_mlflow(result, mlflow_manager)

Research basis:
  - TradingAgents (2025): multi-role review panel surfaces issues missed by
    single-perspective analysis.
  - FinRL-DeepSeek: LLM signals help only when combined with CVaR-PPO; bounded
    weight adjustments (±0.05) prevent over-correction.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default model; can be overridden in constructor
_DEFAULT_MODEL = "claude-opus-4-6"

# Max reward weight change per review cycle
_MAX_WEIGHT_CHANGE = 0.05

# Minimum weight floor
_MIN_WEIGHT = 0.01

# Canned dry-run response
_DRY_RUN_RESPONSE = {
    "strategy": (
        "The agent's behavior is broadly consistent with a momentum-following strategy. "
        "Long positions are opened after sustained upward moves and closed relatively quickly."
    ),
    "risk": (
        "Tail risk from correlated intraday drawdowns is present. "
        "Consider reducing position concentration during high-volatility regimes."
    ),
    "execution": (
        "Turnover is moderate. Entry timing looks well-calibrated; however, exits could be "
        "improved by holding profitable positions slightly longer."
    ),
    "data": (
        "Price momentum and volume features appear most influential. "
        "Sentiment features have low discriminative power in this period."
    ),
    "improvement": (
        "Increase the drawdown penalty weight slightly to discourage deep retracements. "
        "A modest reduction in the transaction_cost weight may improve responsiveness."
    ),
    "suggested_weight_changes": {
        "drawdown": 0.03,
        "transaction_cost": -0.02,
    },
    "confidence_score": 0.75,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentBehaviorSummary:
    """
    Snapshot of an agent's recent behavior for review.

    Parameters
    ----------
    actions : list[float]
        Raw action values taken (e.g. continuous [-1, 1]).
    portfolio_values : list[float]
        Portfolio value at each step.
    trades : list[dict]
        Each dict has keys: side, price, quantity, pnl.
    current_reward_weights : dict
        Current reward component weights (pnl, sharpe, drawdown, …).
    symbol : str
        Traded instrument.
    period_label : str
        Human-readable label, e.g. "fold-3" or "2024-Q1".
    feature_importances : dict, optional
        Feature name → importance score.
    extra_metrics : dict, optional
        Any additional metrics to include in the prompt.
    """

    actions: List[float]
    portfolio_values: List[float]
    trades: List[Dict[str, Any]]
    current_reward_weights: Dict[str, float]
    symbol: str = "BTC/USDT"
    period_label: str = ""
    feature_importances: Dict[str, float] = field(default_factory=dict)
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    # Derived statistics (computed in __post_init__)
    total_return: float = field(init=False)
    sharpe_ratio: float = field(init=False)
    max_drawdown: float = field(init=False)
    win_rate: float = field(init=False)
    avg_holding_steps: float = field(init=False)
    action_distribution: Dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        values = np.array(self.portfolio_values, dtype=float)
        if len(values) >= 2:
            returns = np.diff(values) / np.where(values[:-1] != 0, values[:-1], 1e-8)
            self.total_return = float((values[-1] - values[0]) / values[0])
            std = np.std(returns)
            self.sharpe_ratio = float(
                np.mean(returns) / std * np.sqrt(252) if std > 1e-10 else 0.0
            )
            peak = np.maximum.accumulate(values)
            dd = np.where(peak > 0, (peak - values) / peak, 0.0)
            self.max_drawdown = float(np.max(dd))
        else:
            self.total_return = 0.0
            self.sharpe_ratio = 0.0
            self.max_drawdown = 0.0

        closing_trades = [t for t in self.trades if t.get("side") == "sell"]
        wins = [t for t in closing_trades if t.get("pnl", 0.0) > 0]
        self.win_rate = len(wins) / len(closing_trades) if closing_trades else 0.0

        if self.trades:
            # crude holding period estimate: total steps / num round-trips
            num_rt = max(len(closing_trades), 1)
            self.avg_holding_steps = len(self.portfolio_values) / num_rt
        else:
            self.avg_holding_steps = 0.0

        actions = np.array(self.actions, dtype=float)
        self.action_distribution = {
            "buy_pct": float(np.mean(actions > 0.05)),
            "sell_pct": float(np.mean(actions < -0.05)),
            "hold_pct": float(np.mean(np.abs(actions) <= 0.05)),
            "mean_action": float(np.mean(actions)),
            "std_action": float(np.std(actions)),
        }


@dataclass
class ReviewResult:
    """Five-perspective review from the LLM panel."""

    strategy: str = ""
    risk: str = ""
    execution: str = ""
    data: str = ""
    improvement: str = ""
    suggested_weight_changes: Dict[str, float] = field(default_factory=dict)
    confidence_score: float = 0.0
    raw_response: str = ""
    reviewed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# LLMReviewPanel
# ---------------------------------------------------------------------------

class LLMReviewPanel:
    """
    Multi-perspective LLM review panel for RL trading agents.

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key. Falls back to ``ANTHROPIC_API_KEY`` env var.
    model : str
        Claude model ID.
    dry_run : bool
        When True, skip API call and return a canned ReviewResult.  Useful
        for testing without an API key.
    max_weight_change : float
        Maximum absolute reward weight adjustment per review cycle.
    """

    REVIEW_PERSPECTIVES = ["strategy", "risk", "execution", "data", "improvement"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        dry_run: bool = False,
        max_weight_change: float = _MAX_WEIGHT_CHANGE,
    ) -> None:
        self.model = model
        self.dry_run = dry_run
        self.max_weight_change = max_weight_change
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")

        if not dry_run and not self._api_key:
            logger.warning(
                "No Anthropic API key found; review calls will fail. "
                "Set ANTHROPIC_API_KEY or pass api_key=, or use dry_run=True."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def review(self, summary: AgentBehaviorSummary) -> ReviewResult:
        """
        Run the full 5-perspective review.

        Parameters
        ----------
        summary : AgentBehaviorSummary
            Populated behavior summary to review.

        Returns
        -------
        ReviewResult
            Structured review with per-perspective text + weight suggestions.
        """
        if self.dry_run:
            return self._dry_run_result()

        prompt = self._build_prompt(summary)
        raw = self._call_claude(prompt)
        return self._parse_response(raw)

    def adjust_reward_weights(
        self,
        review_result: ReviewResult,
        current_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply bounded weight adjustments suggested by the review.

        Each adjustment is clamped to ±max_weight_change, and weights are
        re-normalised so they sum to 1.0.  Minimum weight per component is
        _MIN_WEIGHT (0.01) to prevent collapse.

        Parameters
        ----------
        review_result : ReviewResult
        current_weights : dict
            Current reward weight mapping, e.g.
            {"pnl": 0.35, "sharpe": 0.25, ...}.

        Returns
        -------
        New weight dict (copy; original unchanged).
        """
        new_weights = dict(current_weights)
        changes = review_result.suggested_weight_changes

        for key, delta in changes.items():
            if key not in new_weights:
                logger.debug("Skipping unknown weight key: %s", key)
                continue
            clamped = float(np.clip(delta, -self.max_weight_change, self.max_weight_change))
            new_weights[key] = max(_MIN_WEIGHT, new_weights[key] + clamped)

        # Re-normalise
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}

        return new_weights

    def log_to_mlflow(
        self,
        review_result: ReviewResult,
        mlflow_manager,
        artifact_name: str = "llm_review.json",
    ) -> None:
        """Upload review result as a JSON artifact to MLflow."""
        try:
            review_dict = {
                "strategy": review_result.strategy,
                "risk": review_result.risk,
                "execution": review_result.execution,
                "data": review_result.data,
                "improvement": review_result.improvement,
                "suggested_weight_changes": review_result.suggested_weight_changes,
                "confidence_score": review_result.confidence_score,
                "reviewed_at": review_result.reviewed_at,
            }
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(review_dict, f, indent=2)
                tmp_path = f.name
            mlflow_manager.log_artifact(tmp_path, artifact_name)
            mlflow_manager.log_metric(
                "review_confidence", review_result.confidence_score
            )
            logger.info("LLM review logged to MLflow as %s", artifact_name)
        except Exception as e:
            logger.warning("Failed to log review to MLflow: %s", e)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, summary: AgentBehaviorSummary) -> str:
        weight_str = json.dumps(summary.current_reward_weights, indent=2)
        fi_str = (
            json.dumps(summary.feature_importances, indent=2)
            if summary.feature_importances
            else "Not available"
        )
        trade_sample = summary.trades[:10]  # first 10 for brevity
        trades_str = json.dumps(trade_sample, indent=2)

        prompt = f"""You are a quantitative trading analyst reviewing the behavior of a reinforcement learning trading agent.

## Agent Performance Summary
- Symbol: {summary.symbol}
- Period: {summary.period_label}
- Total Return: {summary.total_return:.2%}
- Sharpe Ratio: {summary.sharpe_ratio:.3f}
- Max Drawdown: {summary.max_drawdown:.2%}
- Win Rate: {summary.win_rate:.2%}
- Avg Holding Period: {summary.avg_holding_steps:.1f} steps
- Num Trades: {len(summary.trades)}

## Action Distribution
- Buy actions: {summary.action_distribution['buy_pct']:.1%}
- Sell actions: {summary.action_distribution['sell_pct']:.1%}
- Hold actions: {summary.action_distribution['hold_pct']:.1%}
- Mean action magnitude: {summary.action_distribution['mean_action']:.4f}

## Current Reward Weights
{weight_str}

## Feature Importances
{fi_str}

## Recent Trades (sample)
{trades_str}

---

Please provide a structured review across EXACTLY these 5 perspectives. Your response MUST be valid JSON with this exact schema:

{{
  "strategy": "<Is the agent's behavior consistent with a coherent trading strategy? What strategy seems to be emerging?>",
  "risk": "<What tail risks, correlation dangers, or drawdown patterns exist?>",
  "execution": "<Is turnover excessive? Are entries/exits well-timed?>",
  "data": "<Which features seem most/least important based on observed behavior?>",
  "improvement": "<What specific parameter changes would improve performance?>",
  "suggested_weight_changes": {{
    "<weight_key>": <float delta, e.g. 0.03 to increase or -0.02 to decrease>
  }},
  "confidence_score": <float 0.0-1.0, your confidence in the assessment>
}}

Only include weight keys that exist in the current reward weights. Each suggested change must be between -0.05 and 0.05.
Respond with ONLY the JSON object, no additional text."""
        return prompt

    # ------------------------------------------------------------------
    # Claude API call
    # ------------------------------------------------------------------

    def _call_claude(self, prompt: str) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for LLMReviewPanel. "
                "Install with: pip install anthropic"
            ) from exc

        client = anthropic.Anthropic(api_key=self._api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> ReviewResult:
        """Parse Claude's JSON response into a ReviewResult."""
        try:
            # Strip markdown code fences if present
            text = raw.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                text = "\n".join(
                    l for l in lines if not l.startswith("```")
                ).strip()
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON; using raw text")
            return ReviewResult(
                strategy=raw,
                raw_response=raw,
                confidence_score=0.0,
            )

        # Clamp suggested weight changes
        raw_changes = data.get("suggested_weight_changes", {})
        clamped_changes = {
            k: float(np.clip(v, -self.max_weight_change, self.max_weight_change))
            for k, v in raw_changes.items()
        }

        return ReviewResult(
            strategy=str(data.get("strategy", "")),
            risk=str(data.get("risk", "")),
            execution=str(data.get("execution", "")),
            data=str(data.get("data", "")),
            improvement=str(data.get("improvement", "")),
            suggested_weight_changes=clamped_changes,
            confidence_score=float(
                np.clip(data.get("confidence_score", 0.0), 0.0, 1.0)
            ),
            raw_response=raw,
        )

    # ------------------------------------------------------------------
    # Dry-run helpers
    # ------------------------------------------------------------------

    def _dry_run_result(self) -> ReviewResult:
        d = _DRY_RUN_RESPONSE
        changes = {
            k: float(np.clip(v, -self.max_weight_change, self.max_weight_change))
            for k, v in d["suggested_weight_changes"].items()
        }
        return ReviewResult(
            strategy=d["strategy"],
            risk=d["risk"],
            execution=d["execution"],
            data=d["data"],
            improvement=d["improvement"],
            suggested_weight_changes=changes,
            confidence_score=float(d["confidence_score"]),
            raw_response=json.dumps(d),
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="LLM Review Panel CLI")
    parser.add_argument(
        "--summary-file",
        help="Path to JSON file with AgentBehaviorSummary fields",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Skip API call, use canned response"
    )
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    args = parser.parse_args()

    panel = LLMReviewPanel(dry_run=args.dry_run, model=args.model)

    if args.summary_file:
        with open(args.summary_file) as f:
            data = json.load(f)
        summary = AgentBehaviorSummary(**data)
    else:
        # Minimal synthetic summary for demo
        rng = np.random.default_rng(42)
        prices = 50000.0 + np.cumsum(rng.normal(0, 200, 200))
        summary = AgentBehaviorSummary(
            actions=rng.uniform(-1, 1, 200).tolist(),
            portfolio_values=prices.tolist(),
            trades=[
                {"side": "buy", "price": 50200.0, "quantity": 0.1, "pnl": 0.0},
                {"side": "sell", "price": 51000.0, "quantity": 0.1, "pnl": 80.0},
            ],
            current_reward_weights={
                "pnl": 0.35,
                "sharpe": 0.25,
                "downside_risk": 0.15,
                "drawdown": 0.15,
                "transaction_cost": 0.10,
            },
            symbol="BTC/USDT",
            period_label="demo",
        )

    result = panel.review(summary)
    print(json.dumps(
        {
            "strategy": result.strategy,
            "risk": result.risk,
            "execution": result.execution,
            "data": result.data,
            "improvement": result.improvement,
            "suggested_weight_changes": result.suggested_weight_changes,
            "confidence_score": result.confidence_score,
        },
        indent=2,
    ))


if __name__ == "__main__":
    _main()
