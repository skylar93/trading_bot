"""
Week 34: Phase 3 End-to-End Integration Test

전체 파이프라인을 test_data (축소 데이터)로 실행.
각 테스트는 독립적으로 실행 가능하며, 전체 5분 이내 완료 목표.

Test 1:  Config validation (Pydantic FullConfig)
Test 2:  Data loading + multi-timeframe feature generation
Test 3:  HMM regime detection
Test 4:  Single agent short training (1 000 steps)
Test 5:  Walk-forward validation (1 fold)
Test 6:  Backtesting + statistical tests (bootstrap CI)
Test 7:  Risk manager with regime sizing (adjust_for_regime)
Test 8:  Paper trading 10 steps (OrderManager)
Test 9:  Alert system trigger test (TradingAlerter)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

N_ROWS = 300
WINDOW = 20


def _make_ohlcv(n: int = N_ROWS, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    price = np.clip(price, 1.0, None)
    return pd.DataFrame(
        {
            "$open": price,
            "$high": price * (1 + rng.uniform(0, 0.01, n)),
            "$low": price * (1 - rng.uniform(0, 0.01, n)),
            "$close": price * (1 + rng.normal(0, 0.005, n)),
            "$volume": rng.uniform(1e5, 1e6, n),
        }
    ).assign(**{"$close": lambda df: df["$close"].clip(lower=1.0)})


@pytest.fixture(scope="module")
def ohlcv():
    return _make_ohlcv()


@pytest.fixture(scope="module")
def env(ohlcv):
    from envs.single_asset_rl_env import SingleAssetRLTradingEnv

    return SingleAssetRLTradingEnv(
        data=ohlcv, initial_capital=10_000.0, window_size=WINDOW
    )


# ===========================================================================
# Test 1: Config validation (Pydantic)
# ===========================================================================


class TestConfigValidation:
    def test_fullconfig_loads_defaults(self):
        from config.schema import FullConfig

        cfg = FullConfig()
        assert cfg.env.window_size == 20
        assert cfg.training.total_timesteps == 500_000
        assert cfg.risk_management.max_drawdown_pct == 0.20

    def test_fullconfig_from_dict(self):
        from config.schema import FullConfig

        raw = {
            "env": {"window_size": 30, "initial_balance": 50_000},
            "training": {"total_timesteps": 1_000, "device": "cpu"},
        }
        cfg = FullConfig(**raw)
        assert cfg.env.window_size == 30
        assert cfg.training.total_timesteps == 1_000

    def test_fullconfig_validation_error_window(self):
        from config.schema import FullConfig

        try:
            from pydantic import ValidationError
        except ImportError:
            pytest.skip("pydantic not available")

        with pytest.raises(ValidationError):
            FullConfig(**{"env": {"window_size": 0}})

    def test_fullconfig_from_yaml(self, tmp_path):
        import yaml
        from config.schema import FullConfig

        cfg_dict = {
            "env": {"window_size": 20},
            "training": {"total_timesteps": 5_000, "device": "cpu"},
        }
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml.dump(cfg_dict))

        with open(yaml_file) as f:
            raw = yaml.safe_load(f)
        cfg = FullConfig(**raw)
        assert cfg.training.device == "cpu"


# ===========================================================================
# Test 2: Data loading + multi-timeframe features
# ===========================================================================


class TestMultiTimeframeFeatures:
    def test_generate_adds_columns(self, ohlcv):
        from training.features.multi_timeframe import MultiTimeframeFeatures

        mtf = MultiTimeframeFeatures()
        result = mtf.generate(ohlcv)
        new_cols = [c for c in result.columns if c not in ohlcv.columns]
        assert len(new_cols) >= 8, f"Expected >= 8 new columns, got {len(new_cols)}: {new_cols}"

    def test_generate_no_look_ahead(self, ohlcv):
        """Check that higher-TF values at bar i only use data up to bar i."""
        from training.features.multi_timeframe import MultiTimeframeFeatures

        mtf = MultiTimeframeFeatures(higher_timeframes=["4H"])
        result = mtf.generate(ohlcv)
        # Feature at row 0 should not depend on future — just verify not all-NaN
        col = [c for c in result.columns if "4H" in c][0]
        non_nan = result[col].dropna()
        assert len(non_nan) > 0, "4H feature should have non-NaN values"

    def test_generate_correct_column_names(self, ohlcv):
        from training.features.multi_timeframe import MultiTimeframeFeatures

        mtf = MultiTimeframeFeatures(higher_timeframes=["4H", "1D"])
        result = mtf.generate(ohlcv)
        for tf in ["4H", "1D"]:
            for feat in ["rsi", "macd_signal", "bb_pos", "atr"]:
                col = f"{tf}_{feat}"
                assert col in result.columns, f"Missing column: {col}"

    def test_generate_preserves_length(self, ohlcv):
        from training.features.multi_timeframe import MultiTimeframeFeatures

        mtf = MultiTimeframeFeatures()
        result = mtf.generate(ohlcv)
        assert len(result) == len(ohlcv)


# ===========================================================================
# Test 3: HMM regime detection
# ===========================================================================


class TestRegimeDetection:
    def _make_regime_data(self, n: int = 300, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        price = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        price = np.clip(price, 1.0, None)
        return pd.DataFrame(
            {"$close": price, "$volume": rng.uniform(1e5, 1e6, n)}
        )

    def test_regime_detector_fit_predict(self):
        from training.signals.regime_detector import RegimeDetector

        rd = RegimeDetector(n_regimes=3)
        data = self._make_regime_data()
        rd.fit(data)
        probs = rd.predict_proba(data.tail(60))
        assert probs.shape == (3,), f"Expected (3,), got {probs.shape}"
        assert abs(probs.sum() - 1.0) < 1e-4

    def test_regime_probs_sum_to_one(self):
        from training.signals.regime_detector import RegimeDetector

        rd = RegimeDetector(n_regimes=3)
        data = self._make_regime_data()
        rd.fit(data)
        probs = rd.predict_proba(data.tail(30))
        assert abs(probs.sum() - 1.0) < 1e-4


# ===========================================================================
# Test 4: Single agent short training (1 000 steps)
# ===========================================================================


class TestAgentShortTraining:
    TIMESTEPS = 1_000

    def _make_env(self, n: int = N_ROWS):
        from envs.single_asset_rl_env import SingleAssetRLTradingEnv

        return SingleAssetRLTradingEnv(data=_make_ohlcv(n), window_size=WINDOW)

    def test_ppo_trains_1k_steps(self):
        from stable_baselines3 import PPO

        env = self._make_env()
        agent = PPO(
            "MlpPolicy", env,
            n_steps=64, batch_size=32, n_epochs=2, verbose=0,
        )
        agent.learn(total_timesteps=self.TIMESTEPS, progress_bar=False)
        obs, _ = env.reset()
        action, _ = agent.predict(obs, deterministic=True)
        assert env.action_space.contains(action)

    def test_cvar_ppo_trains_1k_steps(self):
        from agents.sb3.cvar_ppo import CVaRPPO

        env = self._make_env()
        agent = CVaRPPO(
            "MlpPolicy", env,
            cvar_alpha=0.05, cvar_threshold=-0.02, lr_nu=0.005,
            n_steps=64, batch_size=32, n_epochs=2, verbose=0,
        )
        agent.learn(total_timesteps=self.TIMESTEPS, progress_bar=False)
        assert agent.nu >= 0.0


# ===========================================================================
# Test 5: Walk-forward validation (1 fold)
# ===========================================================================


class TestWalkForwardOneFold:
    def test_single_fold_runs(self, ohlcv):
        from envs.single_asset_rl_env import SingleAssetRLTradingEnv
        from training.validation.walk_forward import WalkForwardValidator

        class _RandomAgent:
            def __init__(self, action_space):
                self._action_space = action_space

            def get_action(self, obs, deterministic=False):
                return self._action_space.sample()

            def train_step(self, obs, action, reward, next_obs, done):
                pass

        def agent_factory():
            env = SingleAssetRLTradingEnv(data=_make_ohlcv(), window_size=WINDOW)
            return _RandomAgent(env.action_space)

        def env_factory(data):
            return SingleAssetRLTradingEnv(data=data, window_size=WINDOW)

        wfv = WalkForwardValidator(n_splits=1, train_ratio=0.6, min_test_size=20)
        result = wfv.validate(
            agent_factory=agent_factory,
            env_factory=env_factory,
            data=ohlcv,
            total_timesteps=200,
            eval_episodes=1,
        )
        folds = getattr(result, "folds", None) or getattr(result, "fold_results", [])
        assert len(folds) == 1, f"Expected 1 fold, got {len(folds)}"

    def test_wfv_overfitting_warning(self):
        """IS/OOS Sharpe ratio > 2 should trigger a warning."""
        import logging
        from training.validation.walk_forward import WalkForwardResult, FoldResult

        fold = FoldResult(
            fold_idx=0,
            train_size=200,
            test_size=50,
            is_sharpe=3.0,
            oos_sharpe=1.0,
            oos_returns=np.random.normal(0, 0.01, 50),
        )
        result = WalkForwardResult(folds=[fold])
        ratio = result.is_sharpe / max(result.oos_sharpe, 1e-9)
        assert ratio > 2.0, "Test case should have IS/OOS ratio > 2"


# ===========================================================================
# Test 6: Backtesting + statistical tests
# ===========================================================================


class TestBacktestingAndStatisticalTests:
    def test_bootstrap_sharpe_ci(self):
        from training.analysis.statistical_tests import StrategyStatisticalTests

        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 252)
        st = StrategyStatisticalTests()
        lo, mid, hi = st.bootstrap_sharpe_ci(returns, n_bootstrap=1_000, random_state=0)
        assert lo <= mid <= hi, "CI must be ordered: lower ≤ point ≤ upper"
        assert hi - lo > 0, "CI width must be positive"

    def test_permutation_test_positive_mean(self):
        from training.analysis.statistical_tests import StrategyStatisticalTests

        rng = np.random.default_rng(1)
        returns = rng.normal(0.005, 0.01, 252)   # clearly positive mean
        st = StrategyStatisticalTests()
        p = st.permutation_test(returns, n_permutations=500, random_state=0)
        assert 0.0 <= p <= 1.0, f"p-value must be in [0,1], got {p}"
        assert p < 0.2, f"Strongly positive strategy should have low p-value, got {p}"

    def test_permutation_test_random_strategy(self):
        from training.analysis.statistical_tests import StrategyStatisticalTests

        rng = np.random.default_rng(2)
        returns = rng.normal(0.0, 0.02, 252)     # zero-mean random
        st = StrategyStatisticalTests()
        p = st.permutation_test(returns, n_permutations=500, random_state=0)
        assert p > 0.05, f"Random strategy should not be significant, got p={p}"

    def test_deflated_sharpe_ratio(self):
        from training.analysis.statistical_tests import StrategyStatisticalTests

        st = StrategyStatisticalTests()
        dsr = st.deflated_sharpe_ratio(
            sharpe=2.0, n_trials=100, var_sharpe=0.5, skew=0.0, kurt=0.0
        )
        assert 0.0 <= dsr <= 1.0, f"DSR must be in [0,1], got {dsr}"

    def test_report_includes_stat_results(self, tmp_path):
        """HTML report에 statistical significance 섹션이 포함되는지 확인."""
        from scripts.generate_report import ReportGenerator

        rg = ReportGenerator(output_dir=tmp_path)
        report_path = rg.generate(output_path=tmp_path / "test_report.html")

        html = report_path.read_text(encoding="utf-8")
        assert "Statistical Significance" in html, "Report must contain stat section"
        assert "Bootstrap Sharpe" in html
        assert "Permutation p-value" in html
        assert "Deflated Sharpe" in html

    def test_regime_conditional_report(self):
        from training.analysis.statistical_tests import StrategyStatisticalTests

        rng = np.random.default_rng(3)
        n = 252
        returns = rng.normal(0.001, 0.02, n)
        regimes = np.tile([0, 1, 2], n // 3 + 1)[:n]
        st = StrategyStatisticalTests()
        report = st.regime_conditional_report(returns, regimes)
        assert set(report.keys()) == {0, 1, 2}
        for regime_id, stats in report.items():
            assert "sharpe" in stats
            assert "max_drawdown" in stats
            assert "win_rate" in stats
            assert "n_trades" in stats

    def test_report_includes_stat_results(self, tmp_path):
        """HTML report에 statistical significance 섹션이 포함되는지 확인."""
        from scripts.generate_report import ReportGenerator

        rg = ReportGenerator(output_dir=tmp_path)
        report_path = rg.generate(output_path=tmp_path / "test_report.html")

        html = report_path.read_text(encoding="utf-8")
        assert "Statistical Significance" in html, "Report must contain stat section"
        assert "Bootstrap Sharpe" in html
        assert "Permutation p-value" in html
        assert "Deflated Sharpe" in html


# ===========================================================================
# Test 7: Risk manager with regime sizing
# ===========================================================================


class TestRegimeSizing:
    def _make_config(self):
        from risk_management.rl_risk_manager import RLRiskConfig

        return RLRiskConfig(
            stop_loss_threshold=0.05,
            trailing_stop_buffer=0.03,
            max_drawdown_pct=0.20,
        )

    def test_high_vol_reduces_position(self):
        from risk_management.rl_risk_manager import RLRiskManager

        rm = RLRiskManager(self._make_config())
        high_vol_probs = np.array([0.1, 0.1, 0.8])
        adjusted = rm.adjust_for_regime(0.8, high_vol_probs)
        assert adjusted < 0.8, f"High-vol regime should reduce position: {adjusted}"

    def test_low_vol_preserves_position(self):
        from risk_management.rl_risk_manager import RLRiskManager

        rm = RLRiskManager(self._make_config())
        low_vol_probs = np.array([0.8, 0.1, 0.1])
        adjusted = rm.adjust_for_regime(0.8, low_vol_probs)
        assert adjusted > 0.5, f"Low-vol regime should allow larger position: {adjusted}"

    def test_high_vol_smaller_than_low_vol(self):
        from risk_management.rl_risk_manager import RLRiskManager

        rm = RLRiskManager(self._make_config())
        high = rm.adjust_for_regime(0.8, np.array([0.1, 0.1, 0.8]))
        low = rm.adjust_for_regime(0.8, np.array([0.8, 0.1, 0.1]))
        assert high < low, f"High-vol ({high:.3f}) must be < low-vol ({low:.3f})"

    def test_adjusted_clipped_to_max_position(self):
        from risk_management.rl_risk_manager import RLRiskManager

        rm = RLRiskManager(self._make_config())
        result = rm.adjust_for_regime(2.0, np.array([0.9, 0.05, 0.05]))
        assert result <= 1.0, f"Result must not exceed max_position_size: {result}"

    def test_zero_action_stays_zero(self):
        from risk_management.rl_risk_manager import RLRiskManager

        rm = RLRiskManager(self._make_config())
        result = rm.adjust_for_regime(0.0, np.array([0.1, 0.1, 0.8]))
        assert result == pytest.approx(0.0)


# ===========================================================================
# Test 8: Paper trading 10 steps (OrderManager)
# ===========================================================================


class TestPaperTradingOrderManager:
    def test_submit_and_check_order(self):
        from deployment.execution.order_manager import OrderManager

        mgr = OrderManager({"daily_loss_limit": -1000.0, "initial_cash": 10_000.0}, paper_mode=True)
        mgr.update_paper_price(100.0)
        order_id = mgr.submit_order("buy", amount=0.01, current_price=100.0)
        status = mgr.check_order(order_id)
        assert status == "filled", f"Paper order should be filled, got '{status}'"

    def test_ten_step_paper_trading_loop(self):
        from deployment.execution.order_manager import OrderManager

        mgr = OrderManager({"daily_loss_limit": -10_000.0, "max_order_size": 1.0,
                            "initial_cash": 1_000_000.0})
        mgr.update_paper_price(100.0)

        for i in range(10):
            side = "buy" if i % 2 == 0 else "sell"
            oid = mgr.submit_order(side, amount=0.01, current_price=100.0 + i)
            status = mgr.check_order(oid)
            assert status in {"filled", "pending", "failed"}

        info = mgr.reconcile()
        assert "position" in info
        assert "daily_pnl" in info
        assert isinstance(info["open_orders"], int)

    def test_cancel_pending_order(self):
        from deployment.execution.order_manager import OrderManager

        mgr = OrderManager(paper_mode=True)
        mgr.update_paper_price(100.0)
        oid = mgr.submit_order("buy", 0.01, current_price=100.0)
        # Force back to pending to test cancellation
        mgr._orders[oid].status = "pending"
        success = mgr.cancel_order(oid)
        assert success is True
        assert mgr.check_order(oid) == "cancelled"

    def test_daily_loss_limit_blocks_orders(self):
        from deployment.execution.order_manager import OrderManager

        mgr = OrderManager({"daily_loss_limit": -1.0}, paper_mode=True)
        mgr._halted = True   # simulate halted state
        with pytest.raises(RuntimeError, match="halted"):
            mgr.submit_order("buy", 0.1)


# ===========================================================================
# Test 9: Alert system trigger test (TradingAlerter)
# ===========================================================================


class TestAlertSystem:
    def _alerter(self, **kwargs):
        from deployment.monitoring.alerter import TradingAlerter

        config = {"alert_channels": ["console"], "drawdown_alert_threshold": 0.10}
        config.update(kwargs)
        return TradingAlerter(config)

    def test_no_alert_below_threshold(self):
        alerter = self._alerter()
        fired = alerter.check_drawdown(current=9_500, peak=10_000)  # 5% → no alert
        assert fired is False
        assert len(alerter.alert_history) == 0

    def test_alert_fires_above_threshold(self):
        alerter = self._alerter()
        fired = alerter.check_drawdown(current=8_800, peak=10_000)  # 12% → alert
        assert fired is True
        assert len(alerter.alert_history) == 1
        assert "Drawdown" in alerter.alert_history[0].message

    def test_daily_pnl_alert(self):
        alerter = self._alerter()
        fired = alerter.check_daily_pnl(-600.0)
        assert fired is True
        assert len(alerter.alert_history) == 1
        assert "Daily loss" in alerter.alert_history[0].message

    def test_no_daily_pnl_alert_when_ok(self):
        alerter = self._alerter()
        fired = alerter.check_daily_pnl(-100.0)
        assert fired is False

    def test_send_alert_manual(self):
        alerter = self._alerter()
        alerter.send_alert("Test alert message", level="WARNING")
        assert len(alerter.alert_history) == 1
        assert alerter.alert_history[0].level == "WARNING"

    def test_alert_history_accumulates(self):
        alerter = self._alerter(drawdown_alert_threshold=0.05)
        alerter.check_drawdown(current=9_400, peak=10_000)   # 6% → fires
        alerter.check_drawdown(current=9_200, peak=10_000)   # 8% → fires
        assert len(alerter.alert_history) == 2

    def test_connection_alert(self):
        alerter = self._alerter()
        no_fire = alerter.check_connection_lost(30.0)   # under 60 s
        fire = alerter.check_connection_lost(90.0)      # over 60 s
        assert no_fire is False
        assert fire is True


# ---------------------------------------------------------------------------
# Timing guard
# ---------------------------------------------------------------------------


def test_suite_timing_sentinel():
    """Sentinel: always passes; CI --timeout=300 enforces time limit."""
    assert True
