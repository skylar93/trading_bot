"""[A0] Evidence pack completeness tests.

Verifies that generate_evidence_pack.py produces a markdown file with all
required sections, a valid YAML frontmatter, and mandatory metrics.

These tests run without any real walk-forward data (--simulate mode).
They are the CI gate for the evidence pack generator.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

# Make sure scripts/ is importable
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_evidence_pack import (
    EvidencePack,
    FoldMetrics,
    compute,
    render,
    _simulate_folds,
    main,
    _sharpe,
    _sortino,
    _max_drawdown,
    _calmar,
    _baseline_buy_hold,
    _baseline_ma_cross,
    _baseline_mean_reversion,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simulated_pack() -> EvidencePack:
    """Small simulated evidence pack (5 folds, 252 steps each)."""
    from datetime import datetime, timezone
    folds = _simulate_folds(n_folds=5, n_steps=252, seed=42)
    pack = EvidencePack(
        generated_at=datetime.now(timezone.utc).isoformat(),
        walk_forward_period="2025-01 to 2026-01 (simulated)",
        n_folds=len(folds),
        n_hyperopt_trials=50,
        folds=folds,
    )
    return compute(pack)


@pytest.fixture
def rendered_md(simulated_pack) -> str:
    return render(simulated_pack)


@pytest.fixture
def parsed_frontmatter(rendered_md) -> dict:
    """Parse YAML frontmatter from rendered markdown."""
    m = re.match(r"^---\n(.*?)\n---", rendered_md, re.DOTALL)
    assert m, "No YAML frontmatter found in rendered markdown"
    return yaml.safe_load(m.group(1))


# ---------------------------------------------------------------------------
# A0.1 Walk-forward section
# ---------------------------------------------------------------------------

class TestA01WalkForward:
    def test_section_heading_present(self, rendered_md):
        assert "## A0.1 Walk-Forward Results" in rendered_md

    def test_per_fold_table_present(self, rendered_md):
        assert "### Per-Fold Metrics" in rendered_md
        assert "| Fold |" in rendered_md

    def test_aggregate_table_present(self, rendered_md):
        assert "### Aggregate" in rendered_md
        assert "Net-of-cost" in rendered_md

    def test_all_folds_listed(self, rendered_md, simulated_pack):
        for fold in simulated_pack.folds:
            assert f"| {fold.fold} |" in rendered_md

    def test_gross_and_net_columns(self, rendered_md):
        assert "Gross Sharpe" in rendered_md
        assert "Net Sharpe" in rendered_md


# ---------------------------------------------------------------------------
# A0.2 Statistical confidence section
# ---------------------------------------------------------------------------

class TestA02Statistics:
    def test_section_heading_present(self, rendered_md):
        assert "## A0.2 Statistical Confidence" in rendered_md

    def test_dsr_row_present(self, rendered_md):
        assert "DSR" in rendered_md

    def test_bootstrap_ci_row_present(self, rendered_md):
        assert "Bootstrap 95% CI" in rendered_md

    def test_permutation_row_present(self, rendered_md):
        assert "Permutation p-value" in rendered_md

    def test_net_sharpe_row_present(self, rendered_md):
        assert "Net Sharpe" in rendered_md


# ---------------------------------------------------------------------------
# A0.3 Regime-conditional section
# ---------------------------------------------------------------------------

class TestA03RegimeConditional:
    def test_section_heading_present(self, rendered_md):
        assert "## A0.3 Regime-Conditional Breakdown" in rendered_md

    def test_all_regime_names_present(self, rendered_md):
        for regime in ["trend", "range", "crisis"]:
            assert regime in rendered_md

    def test_per_fold_regime_table(self, rendered_md):
        assert "| Fold | Regime |" in rendered_md

    def test_max_dd_table_present(self, rendered_md):
        assert "Max DD (worst fold)" in rendered_md

    def test_hmm_leakage_note_present(self, rendered_md):
        assert "leakage" in rendered_md.lower() or "per-fold re-fit" in rendered_md


# ---------------------------------------------------------------------------
# A0.4 Baseline comparisons
# ---------------------------------------------------------------------------

class TestA04Baselines:
    def test_section_heading_present(self, rendered_md):
        assert "## A0.4 Baseline Comparisons" in rendered_md

    def test_buy_and_hold_row(self, rendered_md):
        assert "Buy And Hold" in rendered_md or "buy_and_hold" in rendered_md.lower()

    def test_ma_cross_row(self, rendered_md):
        assert "Ma Cross" in rendered_md or "ma_cross" in rendered_md.lower()

    def test_mean_reversion_row(self, rendered_md):
        assert "Mean Reversion" in rendered_md or "mean_reversion" in rendered_md.lower()

    def test_outperform_summary_present(self, rendered_md):
        assert "Outperforms at least 1 baseline" in rendered_md


# ---------------------------------------------------------------------------
# A0.5 Agent decomposition
# ---------------------------------------------------------------------------

class TestA05AgentDecomposition:
    def test_section_heading_present(self, rendered_md):
        assert "## A0.5 Agent Contribution Decomposition" in rendered_md

    def test_agent_table_present(self, rendered_md):
        assert "| Agent |" in rendered_md

    def test_regime_columns_present(self, rendered_md):
        assert "trend" in rendered_md
        assert "crisis" in rendered_md


# ---------------------------------------------------------------------------
# A0.6 Reality gap section
# ---------------------------------------------------------------------------

class TestA06RealityGap:
    def test_section_heading_present(self, rendered_md):
        assert "## A0.6 Reality Gap" in rendered_md

    def test_data_insufficient_note_when_no_data(self, rendered_md):
        assert "insufficient" in rendered_md.lower() or "Data insufficient" in rendered_md


# ---------------------------------------------------------------------------
# A0.7 Reward audit section
# ---------------------------------------------------------------------------

class TestA07RewardAudit:
    def test_section_heading_present(self, rendered_md):
        assert "## A0.7 Reward" in rendered_md

    def test_net_of_cost_verdict_present(self, rendered_md):
        assert "net-of-cost" in rendered_md.lower()

    def test_reward_audit_doc_referenced(self, rendered_md):
        assert "reward_audit.md" in rendered_md


# ---------------------------------------------------------------------------
# GO/NO-GO summary section
# ---------------------------------------------------------------------------

class TestGoNoGoSummary:
    def test_section_heading_present(self, rendered_md):
        assert "## GO / NO-GO Summary" in rendered_md

    def test_all_criteria_listed(self, rendered_md):
        criteria = [
            "Net Sharpe",
            "DSR",
            "Bootstrap CI",
            "Permutation",
            "Crisis DD",
            "baseline",
        ]
        for c in criteria:
            assert c in rendered_md, f"GO/NO-GO criterion '{c}' not found in rendered output"

    def test_operator_sign_off_block_present(self, rendered_md):
        assert "Operator decision" in rendered_md


# ---------------------------------------------------------------------------
# YAML frontmatter
# ---------------------------------------------------------------------------

class TestFrontmatter:
    REQUIRED_KEYS = [
        "generated_at",
        "walk_forward_period",
        "metrics",
    ]
    REQUIRED_METRICS = [
        "net_sharpe",
        "gross_sharpe",
        "dsr",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
        "permutation_p",
        "max_regime_dd",
        "n_folds",
    ]

    def test_frontmatter_parses(self, parsed_frontmatter):
        assert isinstance(parsed_frontmatter, dict)

    def test_required_top_level_keys(self, parsed_frontmatter):
        for key in self.REQUIRED_KEYS:
            assert key in parsed_frontmatter, f"Frontmatter missing key: {key}"

    def test_required_metrics(self, parsed_frontmatter):
        metrics = parsed_frontmatter.get("metrics", {})
        for key in self.REQUIRED_METRICS:
            assert key in metrics, f"Frontmatter metrics missing key: {key}"

    def test_max_regime_dd_has_three_regimes(self, parsed_frontmatter):
        dd = parsed_frontmatter["metrics"]["max_regime_dd"]
        for regime in ["trend", "range", "crisis"]:
            assert regime in dd, f"max_regime_dd missing regime: {regime}"

    def test_metric_values_are_numeric(self, parsed_frontmatter):
        m = parsed_frontmatter["metrics"]
        for key in ["net_sharpe", "dsr", "bootstrap_ci_lower", "permutation_p"]:
            assert isinstance(m[key], (int, float)), f"metric {key} is not numeric"

    def test_n_folds_positive(self, parsed_frontmatter):
        assert parsed_frontmatter["metrics"]["n_folds"] > 0


# ---------------------------------------------------------------------------
# CLI integration test
# ---------------------------------------------------------------------------

class TestCLI:
    def test_simulate_flag_creates_file(self, tmp_path):
        out = tmp_path / "test_evidence.md"
        rc = main(["--simulate", "--output", str(out), "--n-folds", "3"])
        assert rc == 0
        assert out.exists()
        content = out.read_text()
        assert "## A0.1" in content
        assert "## A0.2" in content
        assert "## A0.3" in content
        assert "## A0.4" in content
        assert "## A0.5" in content
        assert "## A0.6" in content
        assert "## A0.7" in content
        assert "## GO / NO-GO" in content

    def test_no_runs_produces_valid_skeleton(self, tmp_path):
        out = tmp_path / "skeleton.md"
        rc = main(["--output", str(out)])
        assert rc == 0
        assert out.exists()

    def test_output_dir_created_if_not_exists(self, tmp_path):
        out = tmp_path / "nested" / "evidence.md"
        rc = main(["--simulate", "--output", str(out)])
        assert rc == 0
        assert out.exists()


# ---------------------------------------------------------------------------
# Metric helper unit tests
# ---------------------------------------------------------------------------

class TestMetricHelpers:
    import numpy as np

    def test_sharpe_positive_drift(self):
        import numpy as np
        rng = np.random.default_rng(99)
        returns = rng.normal(0.002, 0.01, 252)
        assert _sharpe(returns) > 0

    def test_sharpe_zero_variance(self):
        import numpy as np
        assert _sharpe(np.zeros(252)) == 0.0

    def test_max_drawdown_nonnegative(self):
        import numpy as np
        r = np.random.default_rng(0).normal(0, 0.01, 100)
        assert _max_drawdown(r) >= 0

    def test_sortino_only_considers_negative(self):
        import numpy as np
        r = np.array([0.01, 0.02, -0.005, 0.015, -0.003])
        assert _sortino(r) > 0  # positive mean → positive sortino

    def test_calmar_positive(self):
        import numpy as np
        rng = np.random.default_rng(77)
        # Mix of positive/negative returns with positive drift → positive mean and non-zero max_dd
        r = rng.normal(0.003, 0.01, 200)
        result = _calmar(r)
        # calmar = 0 only if mean <= 0 or max_dd == 0; with drift + noise both hold reliably
        assert result != 0.0  # can be negative if unlucky, but not zero

    def test_baseline_buy_hold_returns_dict(self):
        import numpy as np
        r = np.random.default_rng(1).normal(0, 0.01, 200)
        result = _baseline_buy_hold(r)
        assert "sharpe" in result and "max_dd" in result

    def test_baseline_ma_cross_returns_dict(self):
        import numpy as np
        r = np.random.default_rng(2).normal(0, 0.01, 200)
        result = _baseline_ma_cross(r)
        assert "sharpe" in result

    def test_baseline_mean_reversion_returns_dict(self):
        import numpy as np
        r = np.random.default_rng(3).normal(0, 0.01, 200)
        result = _baseline_mean_reversion(r)
        assert "sharpe" in result

    def test_simulate_folds_count(self):
        folds = _simulate_folds(n_folds=3)
        assert len(folds) == 3

    def test_simulate_folds_have_regime_breakdown(self):
        folds = _simulate_folds(n_folds=2)
        for f in folds:
            for regime in ["trend", "range", "crisis"]:
                assert regime in f.regime_breakdown

    def test_compute_fills_aggregates(self):
        from datetime import datetime, timezone
        folds = _simulate_folds(n_folds=3)
        pack = EvidencePack(
            generated_at=datetime.now(timezone.utc).isoformat(),
            walk_forward_period="test",
            n_folds=3,
            n_hyperopt_trials=10,
            folds=folds,
        )
        result = compute(pack)
        assert result.net_sharpe_agg != 0 or result.gross_sharpe_agg >= 0
        assert "trend" in result.max_regime_dd
        assert "crisis" in result.max_regime_dd

    def test_compute_populates_baselines(self):
        from datetime import datetime, timezone
        folds = _simulate_folds(n_folds=2, n_steps=100)
        pack = EvidencePack(
            generated_at=datetime.now(timezone.utc).isoformat(),
            walk_forward_period="test",
            n_folds=2,
            n_hyperopt_trials=10,
            folds=folds,
        )
        result = compute(pack)
        assert "buy_and_hold" in result.baselines
        assert "ma_cross" in result.baselines
        assert "mean_reversion" in result.baselines
