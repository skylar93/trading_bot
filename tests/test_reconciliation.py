"""Test backtesting ↔ live reconciliation."""
import pytest
from training.analysis.reconciliation import ReconciliationReport, NormalisedMetrics


class TestNormalisedMetrics:
    def test_from_backtest_format(self):
        """BaseBacktester._calculate_metrics() output format."""
        report = {
            "total_return": -0.05,
            "sharpe_ratio": -0.3,
            "sortino_ratio": -0.2,
            "max_drawdown": 0.15,
            "total_trades": 42,
            "win_rate": 0.4,
            "final_portfolio_value": 9500.0,
        }
        nm = ReconciliationReport._normalise(report, source="backtest")
        assert nm.total_return == -0.05
        assert nm.num_trades == 42  # maps from total_trades
        assert nm.source == "backtest"

    def test_from_paper_trader_format(self):
        """PaperTrader.generate_report() output format."""
        report = {
            "total_return": -0.03,
            "sharpe_ratio": -0.1,
            "max_drawdown": 0.12,
            "num_trades": 38,
            "win_rate": 0.42,
            "total_fees": 15.5,
            "avg_fill_slippage": 0.001,
            "final_portfolio_value": 9700.0,
        }
        nm = ReconciliationReport._normalise(report, source="live")
        assert nm.num_trades == 38
        assert nm.avg_fill_slippage == 0.001


class TestReconciliationReport:
    def test_no_warnings_on_similar_results(self):
        bt = {"total_return": 0.10, "sharpe_ratio": 1.2, "max_drawdown": 0.08,
              "num_trades": 50, "win_rate": 0.55, "total_fees": 10.0,
              "final_portfolio_value": 11000.0}
        lv = {"total_return": 0.09, "sharpe_ratio": 1.1, "max_drawdown": 0.09,
              "num_trades": 48, "win_rate": 0.54, "total_fees": 12.0,
              "avg_fill_slippage": 0.0005, "final_portfolio_value": 10900.0}
        report = ReconciliationReport.from_reports(bt, lv)
        assert len(report.warnings) == 0

    def test_return_divergence_warning(self):
        bt = {"total_return": 0.10, "sharpe_ratio": 1.0, "max_drawdown": 0.05,
              "num_trades": 50, "win_rate": 0.5, "total_fees": 0, "final_portfolio_value": 11000}
        lv = {"total_return": -0.02, "sharpe_ratio": -0.1, "max_drawdown": 0.08,
              "num_trades": 50, "win_rate": 0.5, "total_fees": 0, "final_portfolio_value": 9800}
        report = ReconciliationReport.from_reports(bt, lv)
        assert any("Return divergence" in w for w in report.warnings)

    def test_high_slippage_warning(self):
        bt = {"total_return": 0.05, "sharpe_ratio": 0.5, "max_drawdown": 0.05,
              "num_trades": 10, "win_rate": 0.5, "total_fees": 0, "final_portfolio_value": 10500}
        lv = {"total_return": 0.04, "sharpe_ratio": 0.4, "max_drawdown": 0.06,
              "num_trades": 10, "win_rate": 0.5, "total_fees": 0,
              "avg_fill_slippage": 0.005, "final_portfolio_value": 10400}
        report = ReconciliationReport.from_reports(bt, lv)
        assert any("slippage" in w.lower() for w in report.warnings)

    def test_trade_count_divergence_warning(self):
        bt = {"total_return": 0.05, "sharpe_ratio": 0.5, "max_drawdown": 0.05,
              "num_trades": 100, "win_rate": 0.5, "total_fees": 0, "final_portfolio_value": 10500}
        lv = {"total_return": 0.04, "sharpe_ratio": 0.4, "max_drawdown": 0.06,
              "num_trades": 50, "win_rate": 0.5, "total_fees": 0, "final_portfolio_value": 10400}
        report = ReconciliationReport.from_reports(bt, lv)
        assert any("Trade count" in w for w in report.warnings)

    def test_summary_format(self):
        bt = {"total_return": 0.10, "sharpe_ratio": 1.0, "max_drawdown": 0.08,
              "num_trades": 50, "win_rate": 0.55, "total_fees": 10.0, "final_portfolio_value": 11000}
        lv = {"total_return": 0.08, "sharpe_ratio": 0.9, "max_drawdown": 0.09,
              "num_trades": 48, "win_rate": 0.53, "total_fees": 12.0, "final_portfolio_value": 10800}
        report = ReconciliationReport.from_reports(bt, lv)
        text = report.summary()
        assert "RECONCILIATION REPORT" in text
        assert "total_return" in text

    def test_to_json(self, tmp_path):
        bt = {"total_return": 0.10, "sharpe_ratio": 1.0, "max_drawdown": 0.05,
              "num_trades": 50, "win_rate": 0.5, "total_fees": 0, "final_portfolio_value": 11000}
        lv = {"total_return": 0.08, "sharpe_ratio": 0.9, "max_drawdown": 0.06,
              "num_trades": 48, "win_rate": 0.5, "total_fees": 0, "final_portfolio_value": 10800}
        report = ReconciliationReport.from_reports(bt, lv)
        out_path = str(tmp_path / "reconciliation.json")
        data = report.to_json(out_path)
        assert "backtest" in data
        assert "live" in data
        assert "deltas" in data
        import json
        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded["backtest"]["total_return"] == 0.10
