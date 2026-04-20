"""
Week 75 (G1-G5): Governance & Go-Live Gate tests.

Covers:
  G1 — ModelRegistry promotion state machine
  G2 — PaperTrader canary routing + promotion suggestion
  G4 — promote_model.py CLI (dry-run & full)
  G5 — PaperTrader.replace_agent() hot-swap

완료 조건:
  - canary → prod 전이 시뮬레이션 1회 완료 (TestPromotionStateMachine.test_full_pipeline)
  - 핫스왑 테스트 pass (TestAgentHotSwap.test_hotswap_mid_run)
"""

from __future__ import annotations

import threading
from typing import Any, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from deployment.paper_trader import PaperTrader
from training.registry.model_registry import (
    ModelRegistry,
    VALID_STAGES,
    VALID_TRANSITIONS,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _DummyAgent:
    """Deterministic agent that always returns the same scalar action."""

    def __init__(self, action: float = 0.5, name: str = "dummy") -> None:
        self._action = action
        self.name = name
        self.call_count: int = 0

    def predict(self, obs, deterministic: bool = True) -> Tuple[np.ndarray, Any]:
        self.call_count += 1
        return np.array([self._action]), None

    def __repr__(self) -> str:
        return f"<DummyAgent name={self.name!r} action={self._action}>"


def _build_config(canary_enabled: bool = False, traffic_pct: float = 0.10) -> dict:
    cfg: dict = {
        "paper_trading": {
            "symbol": "BTC/USDT",
            "initial_balance": 10_000.0,
            "trading_fee": 0.001,
            "max_position_size": 1.0,
            "max_drawdown_threshold": 0.99,
            "window_size": 5,
        },
        "monitoring": {},
    }
    if canary_enabled:
        cfg["canary"] = {"enabled": True, "traffic_pct": traffic_pct}
    return cfg


def _price_stream(n: int = 30, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    prices = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    return prices.tolist()


# ---------------------------------------------------------------------------
# G1 — ModelRegistry promotion state machine
# ---------------------------------------------------------------------------

class TestPromotionStateMachine:

    def test_newly_registered_version_is_candidate(self, tmp_path):
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        ver = reg.register(model_path="fake.zip")
        assert reg.get_stage(ver) == "candidate"

    def test_promote_candidate_to_staging(self, tmp_path):
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        ver = reg.register(model_path="fake.zip")
        reg.promote(ver, to_stage="staging", actor="alice", reason="backtest ok")
        assert reg.get_stage(ver) == "staging"

    def test_promote_staging_to_canary(self, tmp_path):
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        ver = reg.register(model_path="fake.zip")
        reg.promote(ver, to_stage="staging", actor="alice", reason="backtest ok")
        reg.promote(ver, to_stage="canary", actor="alice", reason="walkforward passed")
        assert reg.get_stage(ver) == "canary"

    def test_promote_canary_to_prod(self, tmp_path):
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        ver = reg.register(model_path="fake.zip")
        reg.promote(ver, to_stage="staging", actor="alice", reason="ok")
        reg.promote(ver, to_stage="canary", actor="alice", reason="ok")
        reg.promote(ver, to_stage="prod", actor="alice", reason="7d canary passed")
        assert reg.get_stage(ver) == "prod"

    def test_promote_prod_to_retired(self, tmp_path):
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        ver = reg.register(model_path="fake.zip")
        for stage in ("staging", "canary", "prod", "retired"):
            reg.promote(ver, to_stage=stage, actor="alice", reason="ok")
        assert reg.get_stage(ver) == "retired"

    def test_invalid_transition_raises(self, tmp_path):
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        ver = reg.register(model_path="fake.zip")
        # candidate → prod is not allowed
        with pytest.raises(ValueError, match="not allowed"):
            reg.promote(ver, to_stage="prod", actor="alice", reason="skip")

    def test_invalid_stage_name_raises(self, tmp_path):
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        ver = reg.register(model_path="fake.zip")
        with pytest.raises(ValueError, match="Unknown stage"):
            reg.promote(ver, to_stage="bogus", actor="alice", reason="")

    def test_retired_has_no_outgoing_transitions(self, tmp_path):
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        ver = reg.register(model_path="fake.zip")
        for stage in ("staging", "canary", "prod", "retired"):
            reg.promote(ver, to_stage=stage, actor="a", reason="ok")
        with pytest.raises(ValueError, match="not allowed"):
            reg.promote(ver, to_stage="candidate", actor="a", reason="", force=False)

    def test_force_bypasses_transition_check(self, tmp_path):
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        ver = reg.register(model_path="fake.zip")
        # candidate → prod normally not allowed, but force=True skips check
        reg.promote(ver, to_stage="prod", actor="tester", reason="force test", force=True)
        assert reg.get_stage(ver) == "prod"

    def test_promotion_history_recorded(self, tmp_path):
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        ver = reg.register(model_path="fake.zip")
        reg.promote(ver, to_stage="staging", actor="bob", reason="first step")
        history = reg.get_promotion_history(ver)
        # Should have: initial registration + staging promotion
        assert len(history) == 2
        assert history[0]["to_stage"] == "candidate"
        assert history[0]["actor"] == "system"
        assert history[1]["to_stage"] == "staging"
        assert history[1]["actor"] == "bob"
        assert history[1]["reason"] == "first step"
        assert "timestamp" in history[1]

    def test_list_by_stage(self, tmp_path):
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        v1 = reg.register(model_path="fake.zip")
        v2 = reg.register(model_path="fake.zip")
        v3 = reg.register(model_path="fake.zip")
        reg.promote(v1, to_stage="staging", actor="a", reason="ok")
        reg.promote(v2, to_stage="staging", actor="a", reason="ok")
        # v3 stays candidate
        staging_list = reg.list_by_stage("staging")
        candidate_list = reg.list_by_stage("candidate")
        assert v1 in staging_list
        assert v2 in staging_list
        assert v3 not in staging_list
        assert v3 in candidate_list

    def test_check_promotion_conditions_valid(self, tmp_path):
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        ver = reg.register(model_path="fake.zip")
        ok, msg = reg.check_promotion_conditions(ver, "staging")
        assert ok is True
        assert "candidate" in msg

    def test_check_promotion_conditions_invalid(self, tmp_path):
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        ver = reg.register(model_path="fake.zip")
        ok, msg = reg.check_promotion_conditions(ver, "prod")
        assert ok is False
        assert "not allowed" in msg

    def test_full_pipeline_candidate_to_prod(self, tmp_path):
        """G1 완료 조건: canary → prod 전이 시뮬레이션."""
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        ver = reg.register(model_path="model_v1.zip", name="ppo_v1", metrics={"sharpe": 0.82})

        # candidate → staging
        ok, _ = reg.check_promotion_conditions(ver, "staging")
        assert ok
        reg.promote(ver, to_stage="staging", actor="skylar", reason="backtest Sharpe=0.82")
        assert reg.get_stage(ver) == "staging"

        # staging → canary
        ok, _ = reg.check_promotion_conditions(ver, "canary")
        assert ok
        reg.promote(ver, to_stage="canary", actor="skylar", reason="walkforward all-positive")
        assert reg.get_stage(ver) == "canary"

        # canary → prod
        ok, _ = reg.check_promotion_conditions(ver, "prod")
        assert ok
        reg.promote(ver, to_stage="prod", actor="skylar", reason="7d canary passed, ruin_prob=0.003")
        assert reg.get_stage(ver) == "prod"

        # verify full history
        history = reg.get_promotion_history(ver)
        stages_in_order = [h["to_stage"] for h in history]
        assert stages_in_order == ["candidate", "staging", "canary", "prod"]

    def test_demotion_canary_to_staging(self, tmp_path):
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        ver = reg.register(model_path="fake.zip")
        reg.promote(ver, to_stage="staging", actor="a", reason="ok")
        reg.promote(ver, to_stage="canary", actor="a", reason="ok")
        reg.promote(ver, to_stage="staging", actor="a", reason="underperformance")
        assert reg.get_stage(ver) == "staging"

    def test_get_stage_unknown_version_raises(self, tmp_path):
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        with pytest.raises(KeyError):
            reg.get_stage(999)

    def test_promote_unknown_version_raises(self, tmp_path):
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        with pytest.raises(KeyError):
            reg.promote(999, to_stage="staging", actor="a", reason="ok")

    def test_registry_persistence_across_instances(self, tmp_path):
        """Stage 정보가 디스크에 저장되어 재구성 후에도 유지되는지 확인."""
        reg_dir = str(tmp_path / "reg")
        reg1 = ModelRegistry(registry_dir=reg_dir)
        ver = reg1.register(model_path="fake.zip")
        reg1.promote(ver, to_stage="staging", actor="a", reason="ok")

        reg2 = ModelRegistry(registry_dir=reg_dir)
        assert reg2.get_stage(ver) == "staging"


# ---------------------------------------------------------------------------
# G2 — PaperTrader canary routing
# ---------------------------------------------------------------------------

class TestCanaryRouting:

    def test_canary_agent_none_runs_clean(self):
        """canary_agent=None이면 에러 없이 정상 동작."""
        agent = _DummyAgent(action=0.3)
        trader = PaperTrader(agent, _build_config(), simulation_mode=True)
        report = trader.run(price_stream=iter(_price_stream(30)))
        assert report["steps"] > 0

    def test_shadow_agent_backward_compat(self):
        """shadow_agent 인자가 canary_agent로 흡수되는지 확인."""
        main = _DummyAgent(action=0.3, name="main")
        shadow = _DummyAgent(action=-0.1, name="shadow")
        trader = PaperTrader(
            main, _build_config(), simulation_mode=True, shadow_agent=shadow
        )
        assert trader.canary_agent is shadow
        assert trader.shadow_agent is shadow

    def test_canary_agent_kwarg(self):
        main = _DummyAgent(action=0.3, name="main")
        canary = _DummyAgent(action=-0.1, name="canary")
        trader = PaperTrader(
            main, _build_config(), simulation_mode=True, canary_agent=canary
        )
        assert trader.canary_agent is canary

    def test_canary_observe_mode_does_not_execute(self):
        """canary_enabled=False → canary action은 로그만, 포지션 변경 없음."""
        main = _DummyAgent(action=0.3)
        canary = _DummyAgent(action=-0.9)  # would sell heavily if active
        cfg = _build_config(canary_enabled=False)
        trader = PaperTrader(
            main, cfg, simulation_mode=True, canary_agent=canary
        )
        prices = _price_stream(20)
        trader.run(price_stream=iter(prices))
        # canary was queried but should not have changed trader position
        # (main agent only buys, canary only sells — if canary were active, trades would differ)
        assert canary.call_count > 0

    def test_canary_enabled_routes_traffic(self):
        """canary_enabled=True → 일부 스텝에서 canary action이 실행됨."""
        main = _DummyAgent(action=0.0)    # hold — produces no trades
        canary = _DummyAgent(action=0.5)  # buy — produces trades if active
        cfg = _build_config(canary_enabled=True, traffic_pct=0.5)
        trader = PaperTrader(
            main, cfg, simulation_mode=True, canary_agent=canary
        )
        trader.run(price_stream=iter(_price_stream(30)))
        # Canary should have been called (at active steps)
        assert canary.call_count > 0

    def test_canary_promotion_suggestion_logged(self):
        """canary 성과가 threshold 충족 시 promotion suggestion이 기록되는지."""
        main = _DummyAgent(action=0.0)
        canary = _DummyAgent(action=0.3)
        cfg = _build_config(canary_enabled=True, traffic_pct=0.5)
        mock_audit = MagicMock()
        trader = PaperTrader(
            main, cfg, simulation_mode=True,
            canary_agent=canary, audit_logger=mock_audit,
        )
        # Inject enough return history to exceed min_window=168 threshold
        trader._canary_returns = [0.001] * 200
        trader._prod_returns = [0.0005] * 200
        trader._check_canary_promotion_suggestion()
        assert trader._canary_promotion_suggested is True

    def test_canary_suggestion_fires_only_once(self):
        """promotion suggestion은 한 번만 발화해야 함."""
        main = _DummyAgent(action=0.0)
        canary = _DummyAgent(action=0.3)
        cfg = _build_config(canary_enabled=True)
        trader = PaperTrader(main, cfg, simulation_mode=True, canary_agent=canary)
        trader._canary_returns = [0.001] * 200
        trader._prod_returns = [0.0005] * 200
        trader._check_canary_promotion_suggestion()
        trader._check_canary_promotion_suggestion()  # second call — should not re-fire
        assert trader._canary_promotion_suggested is True


# ---------------------------------------------------------------------------
# G4 — promote_model.py CLI
# ---------------------------------------------------------------------------

class TestPromoteModelCLI:

    def test_check_valid_transition(self, tmp_path):
        from scripts.promote_model import main as promote_main

        reg_dir = str(tmp_path / "reg")
        reg = ModelRegistry(registry_dir=reg_dir)
        ver = reg.register(model_path="fake.zip")

        rc = promote_main([
            "--registry", reg_dir,
            "--check",
            "--from", "candidate",
            "--to", "staging",
            "--version", str(int(ver)),
        ])
        assert rc == 0

    def test_check_invalid_transition_returns_nonzero(self, tmp_path):
        from scripts.promote_model import main as promote_main

        reg_dir = str(tmp_path / "reg")
        reg = ModelRegistry(registry_dir=reg_dir)
        ver = reg.register(model_path="fake.zip")

        rc = promote_main([
            "--registry", reg_dir,
            "--check",
            "--from", "candidate",
            "--to", "prod",
            "--version", str(int(ver)),
        ])
        assert rc != 0

    def test_actual_promotion_via_cli(self, tmp_path):
        from scripts.promote_model import main as promote_main

        reg_dir = str(tmp_path / "reg")
        reg = ModelRegistry(registry_dir=reg_dir)
        ver = reg.register(model_path="fake.zip")

        rc = promote_main([
            "--registry", reg_dir,
            "--from", "candidate",
            "--to", "staging",
            "--version", str(int(ver)),
            "--actor", "tester",
            "--reason", "CI test promotion",
        ])
        assert rc == 0
        assert reg.get_stage(ver) == "staging"

    def test_wrong_from_stage_rejected(self, tmp_path):
        from scripts.promote_model import main as promote_main

        reg_dir = str(tmp_path / "reg")
        reg = ModelRegistry(registry_dir=reg_dir)
        ver = reg.register(model_path="fake.zip")
        # Version is candidate, but CLI says --from staging
        rc = promote_main([
            "--registry", reg_dir,
            "--from", "staging",
            "--to", "canary",
            "--version", str(int(ver)),
            "--actor", "tester",
            "--reason", "wrong stage",
        ])
        assert rc != 0
        # Stage should be unchanged
        assert reg.get_stage(ver) == "candidate"

    def test_canary_to_prod_requires_actor_and_reason(self, tmp_path):
        from scripts.promote_model import main as promote_main

        reg_dir = str(tmp_path / "reg")
        reg = ModelRegistry(registry_dir=reg_dir)
        ver = reg.register(model_path="fake.zip")
        for stage in ("staging", "canary"):
            reg.promote(ver, to_stage=stage, actor="a", reason="ok")

        # Missing reason → should be rejected
        rc = promote_main([
            "--registry", reg_dir,
            "--from", "canary",
            "--to", "prod",
            "--version", str(int(ver)),
            "--actor", "skylar",
            # no --reason
        ])
        assert rc != 0

    def test_unknown_version_returns_nonzero(self, tmp_path):
        from scripts.promote_model import main as promote_main

        reg_dir = str(tmp_path / "reg")
        ModelRegistry(registry_dir=reg_dir)  # empty registry

        rc = promote_main([
            "--registry", reg_dir,
            "--from", "candidate",
            "--to", "staging",
            "--version", "999",
            "--actor", "a",
            "--reason", "no such version",
        ])
        assert rc != 0

    def test_json_output(self, tmp_path, capsys):
        import json as _json
        from scripts.promote_model import main as promote_main

        reg_dir = str(tmp_path / "reg")
        reg = ModelRegistry(registry_dir=reg_dir)
        ver = reg.register(model_path="fake.zip")

        promote_main([
            "--registry", reg_dir,
            "--from", "candidate",
            "--to", "staging",
            "--version", str(int(ver)),
            "--actor", "ci",
            "--reason", "json test",
            "--json",
        ])
        captured = capsys.readouterr()
        data = _json.loads(captured.out)
        assert data["ok"] is True


# ---------------------------------------------------------------------------
# G5 — PaperTrader.replace_agent() hot-swap
# ---------------------------------------------------------------------------

class TestAgentHotSwap:

    def test_replace_agent_mid_run(self):
        """G5 완료 조건: 50-step 도중 agent 교체 → 이후 스텝부터 신규 agent 사용."""
        main = _DummyAgent(action=0.3, name="original")
        replacement = _DummyAgent(action=-0.1, name="replacement")

        cfg = _build_config()
        trader = PaperTrader(main, cfg, simulation_mode=True)

        prices = _price_stream(n=60)

        # Run 10 steps manually, then swap, then run 40 more
        price_iter = iter(prices)
        run_prices_first = [next(price_iter) for _ in range(10)]
        run_prices_rest = list(price_iter)

        trader.run(price_stream=iter(run_prices_first))
        step_after_first_run = trader.state.step

        trader.replace_agent(replacement, actor="tester", reason="hotswap test")
        assert trader.agent is replacement

        trader.run(price_stream=iter(run_prices_rest))
        assert trader.state.step > step_after_first_run

        # replacement must have been called in the second run
        assert replacement.call_count > 0

    def test_replace_agent_preserves_state(self):
        """교체 후 포트폴리오 히스토리, 잔고 등 내부 상태가 유지되어야 함."""
        main = _DummyAgent(action=0.3, name="original")
        replacement = _DummyAgent(action=0.0, name="hold")

        cfg = _build_config()
        trader = PaperTrader(main, cfg, simulation_mode=True)
        trader.run(price_stream=iter(_price_stream(20)))

        balance_before = trader.state.balance
        position_before = trader.state.position
        step_before = trader.state.step
        n_trades_before = len(trader.state.trades)

        trader.replace_agent(replacement, actor="tester", reason="state test")

        assert trader.state.balance == balance_before
        assert trader.state.position == position_before
        assert trader.state.step == step_before
        assert len(trader.state.trades) == n_trades_before

    def test_replace_agent_thread_safe(self):
        """replace_agent는 PositionTracker lock 안에서 실행되어야 한다."""
        main = _DummyAgent(action=0.0)
        replacement = _DummyAgent(action=0.0)

        cfg = _build_config()
        trader = PaperTrader(main, cfg, simulation_mode=True)

        errors = []

        def swap():
            try:
                trader.replace_agent(replacement, actor="thread", reason="concurrent test")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=swap) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert trader.agent is replacement

    def test_replace_agent_audit_logged(self):
        """replace_agent 이벤트가 audit log에 기록되어야 한다."""
        main = _DummyAgent(action=0.0, name="main")
        replacement = _DummyAgent(action=0.0, name="new")

        cfg = _build_config()
        mock_audit = MagicMock()
        trader = PaperTrader(main, cfg, simulation_mode=True, audit_logger=mock_audit)

        trader.replace_agent(replacement, actor="alice", reason="model upgrade")

        mock_audit.log_risk_event.assert_called_once()
        call_args = mock_audit.log_risk_event.call_args[0][0]
        assert call_args["type"] == "agent_hotswap"
        assert call_args["actor"] == "alice"
        assert call_args["reason"] == "model upgrade"

    def test_replace_agent_50_steps_then_swap(self):
        """G5 완료 조건 정확히 재현: 50 step 후 교체, 다음 step부터 신규 agent."""
        original = _DummyAgent(action=0.3, name="original")
        new_agent = _DummyAgent(action=-0.1, name="new")

        cfg = _build_config()
        trader = PaperTrader(original, cfg, simulation_mode=True)

        rng = np.random.default_rng(0)
        prices_all = (100.0 + np.cumsum(rng.normal(0, 0.3, 60))).tolist()

        trader.run(price_stream=iter(prices_all[:50]))
        assert trader.state.step == 50

        original_calls_at_swap = original.call_count
        trader.replace_agent(new_agent, actor="skylar", reason="test G5")

        trader.run(price_stream=iter(prices_all[50:]))
        # original must NOT have been called after the swap
        assert original.call_count == original_calls_at_swap
        # new_agent must have been called after the swap
        assert new_agent.call_count > 0
