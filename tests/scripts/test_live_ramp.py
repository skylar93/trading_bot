"""E3: run_paper_trader.py live-ramp guard wiring tests."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _make_fake_handler():
    """A logging.FileHandler mock whose .level is an int so the log machinery works."""
    h = MagicMock()
    h.level = 0
    h.filters = []
    return h


def _gate_pass():
    """Return a passing LiveSignalGate check result."""
    result = MagicMock()
    result.passed = True
    result.evidence_pack_age_days = 1.0
    result.failures = []
    return result


def _run_main(exchange_mode: str, extra_config: dict | None = None):
    """
    Invoke run_paper_trader.main() up to the PaperTrader construction and
    return the config dict that was used to create the trader.
    """
    captured = {}

    class _FakeTrader:
        def __init__(self, agent, config, simulation_mode=False):
            captured["config"] = config
            self._warmup_guard = None

        def run(self, **kwargs):
            return {}

        def _trigger_shutdown(self, *a):
            pass

    base_config = {
        "agent": {"checkpoint": "fake_checkpoint", "algo": "PPO"},
        "paper_trading": {"initial_balance": 10_000},
    }
    if extra_config:
        base_config.update(extra_config)

    fake_agent = MagicMock()
    fake_algo_cls = MagicMock()
    fake_algo_cls.load = MagicMock(return_value=fake_agent)
    fake_sb3 = types.ModuleType("stable_baselines3")
    fake_sb3.PPO = fake_algo_cls
    fake_sb3.SAC = MagicMock()
    fake_sb3.TD3 = MagicMock()

    fake_gate_cls = MagicMock()
    fake_gate_cls.return_value.check.return_value = _gate_pass()
    fake_gate_module = types.ModuleType("deployment.governance.live_signal_gate")
    fake_gate_module.LiveSignalGate = fake_gate_cls

    argv = ["run_paper_trader.py", "--config", "fake.yaml",
            "--exchange-mode", exchange_mode]

    # Remove cached module so patches applied below take effect
    import sys as _sys
    _sys.modules.pop("scripts.run_paper_trader", None)

    with (
        patch("yaml.safe_load", return_value=base_config),
        patch("builtins.open", MagicMock(return_value=MagicMock(
            __enter__=MagicMock(return_value=MagicMock()),
            __exit__=MagicMock(return_value=False),
        ))),
        patch.dict("sys.modules", {
            "stable_baselines3": fake_sb3,
            "deployment.governance.live_signal_gate": fake_gate_module,
        }),
        patch("deployment.paper_trader.PaperTrader", _FakeTrader),
        patch("logging.FileHandler", return_value=_make_fake_handler()),
        patch("pathlib.Path.mkdir"),
        patch("pathlib.Path.write_text"),
        patch("sys.argv", argv),
    ):
        from scripts import run_paper_trader
        # Patch write/remove PID after import so names resolve correctly
        with (
            patch.object(run_paper_trader, "_write_pid", lambda *a: None),
            patch.object(run_paper_trader, "_remove_pid", lambda *a: None),
        ):
            try:
                run_paper_trader.main()
            except SystemExit:
                pass

    return captured.get("config", {})


class TestLiveRampConfig:
    def test_live_mode_gets_e3_defaults(self):
        """live mode → size_fraction=0.3, progress_alerts=True."""
        config = _run_main("live")
        warmup = config.get("warmup", {})
        assert warmup.get("enabled") is True
        assert warmup.get("size_fraction") == 0.3
        assert warmup.get("progress_alerts") is True

    def test_paper_mode_gets_e2_defaults(self):
        """paper mode → size_fraction=0.5, progress_alerts=False."""
        config = _run_main("paper")
        warmup = config.get("warmup", {})
        assert warmup.get("enabled") is True
        assert warmup.get("size_fraction") == 0.5
        assert warmup.get("progress_alerts") is False

    def test_sandbox_mode_gets_e2_defaults(self):
        """sandbox also gets E2 defaults (not the tighter live cap)."""
        config = _run_main("sandbox")
        warmup = config.get("warmup", {})
        assert warmup.get("size_fraction") == 0.5
        assert warmup.get("progress_alerts") is False

    def test_warmup_minutes_default(self):
        config = _run_main("live")
        assert config["warmup"]["warmup_minutes"] == 30

    def test_max_qps_default(self):
        config = _run_main("live")
        assert config["warmup"]["max_qps"] == 1.0

    def test_config_override_respected(self):
        """If config already sets size_fraction, it should not be overwritten."""
        config = _run_main("live", extra_config={"warmup": {"size_fraction": 0.1}})
        # setdefault behaviour: pre-existing value is preserved
        assert config["warmup"]["size_fraction"] == 0.1


class TestLiveRampProgressAlerts:
    def test_progress_alerts_fired_each_minute(self):
        """WarmupGuard fires a progress alert for each completed minute."""
        import time
        from deployment.execution.warmup_guard import WarmupGuard

        alerter = MagicMock()
        guard = WarmupGuard(
            warmup_minutes=3,
            size_fraction=0.3,
            max_qps=100.0,
            progress_alerts=True,
            alerter=alerter,
        )
        # start() calls logger.warning — skip it so no log-level issue
        guard._start_time = time.monotonic() - 65
        alerter.reset_mock()

        guard.check(0.5)
        calls = [c[0][0] for c in alerter.send_alert.call_args_list]
        assert any("ramp progress" in m for m in calls)

    def test_no_progress_alerts_in_paper_mode(self):
        import time
        from deployment.execution.warmup_guard import WarmupGuard

        alerter = MagicMock()
        guard = WarmupGuard(
            warmup_minutes=3,
            size_fraction=0.5,
            max_qps=100.0,
            progress_alerts=False,
            alerter=alerter,
        )
        guard._start_time = time.monotonic() - 65
        alerter.reset_mock()

        guard.check(0.5)
        calls = [c[0][0] for c in alerter.send_alert.call_args_list]
        assert not any("ramp progress" in m for m in calls)
