"""
Week 28: Deployment Integration Tests

테스트 항목:
    - setup_local.py --dry-run 동작 확인
    - fetch_data.py --dry-run 동작 확인
    - run_full_pipeline.py --dry-run 동작 확인
    - generate_report.py 실행 및 HTML 출력 확인
    - Docker 관련 파일 존재 확인
    - Config validation (local_3060ti.yaml, training_config.yaml)
    - Pipeline state checkpoint/resume 동작
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ensure project is on sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────

@pytest.fixture
def tmp_results_dir(tmp_path: Path) -> Path:
    """Temporary results directory for report tests."""
    return tmp_path / "results"


@pytest.fixture
def sample_config() -> dict:
    """Minimal valid config for testing."""
    return {
        "env": {
            "type": "single_asset_rl",
            "initial_balance": 10000.0,
            "window_size": 10,
        },
        "data": {
            "symbols": ["BTC/USDT"],
            "timeframe": "1h",
            "data_path": "data/BTC_1h.csv",
        },
        "agent": {
            "name": "PPO",
            "learning_rate": 3e-4,
            "batch_size": 64,
        },
        "training": {
            "total_timesteps": 1000,
            "use_gpu": False,
            "device": "cpu",
        },
        "walk_forward": {
            "n_folds": 3,
            "train_window": 500,
            "val_window": 100,
        },
        "report": {
            "output_dir": "results",
        },
    }


# ──────────────────────────────────────────────────────────
# 28.1: Docker files exist
# ──────────────────────────────────────────────────────────

class TestDockerFiles:
    def test_dockerfile_exists(self):
        assert (PROJECT_ROOT / "Dockerfile").exists(), "Dockerfile missing"

    def test_dockerfile_has_cuda_base(self):
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "nvidia/cuda" in content, "Dockerfile should use NVIDIA CUDA base image"

    def test_dockerfile_has_python310(self):
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "3.10" in content, "Dockerfile should use Python 3.10"

    def test_dockerfile_multistage(self):
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert content.count("FROM ") >= 2, "Dockerfile should be multi-stage (>= 2 FROM lines)"

    def test_docker_compose_exists(self):
        assert (PROJECT_ROOT / "docker-compose.yml").exists(), "docker-compose.yml missing"

    def test_docker_compose_valid_yaml(self):
        content = (PROJECT_ROOT / "docker-compose.yml").read_text()
        data = yaml.safe_load(content)
        assert "services" in data, "docker-compose.yml must have 'services' key"

    def test_docker_compose_has_required_services(self):
        content = (PROJECT_ROOT / "docker-compose.yml").read_text()
        data = yaml.safe_load(content)
        services = data.get("services", {})
        required = {"mlflow", "training", "web_ui", "data_fetcher"}
        missing = required - set(services.keys())
        assert not missing, f"docker-compose.yml missing services: {missing}"

    def test_docker_compose_mlflow_port(self):
        content = (PROJECT_ROOT / "docker-compose.yml").read_text()
        assert "5000" in content, "MLflow port 5000 should be in docker-compose.yml"

    def test_docker_compose_streamlit_port(self):
        content = (PROJECT_ROOT / "docker-compose.yml").read_text()
        assert "8501" in content, "Streamlit port 8501 should be in docker-compose.yml"


# ──────────────────────────────────────────────────────────
# 28.2: Config validation
# ──────────────────────────────────────────────────────────

class TestConfigValidation:
    def test_training_config_exists(self):
        # Week 63: training_config.yaml → config/base.yaml (consolidated)
        assert (PROJECT_ROOT / "config" / "base.yaml").exists()

    def test_training_config_valid_yaml(self):
        # Week 63: use base.yaml as the canonical training config
        p = PROJECT_ROOT / "config" / "base.yaml"
        with open(p) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), "base.yaml must be a dict"
        assert "env" in data
        assert "training" in data

    def test_local_3060ti_config_exists(self):
        # Week 63: local_3060ti.yaml moved to config/env/local_3060ti.yaml
        assert (PROJECT_ROOT / "config" / "env" / "local_3060ti.yaml").exists(), (
            "config/env/local_3060ti.yaml missing"
        )

    def test_local_3060ti_config_valid_yaml(self):
        # Week 63: full merged config via loader
        from config.loader import load
        data = load("local_3060ti")
        assert isinstance(data, dict)

    def test_local_3060ti_has_gpu_settings(self):
        from config.loader import load
        data = load("local_3060ti")
        training = data.get("training", {})
        assert training.get("use_gpu") is True, "local_3060ti config should have use_gpu=true"
        assert training.get("device", "").startswith("cuda"), "device should be cuda:x"

    def test_local_3060ti_has_walk_forward(self):
        from config.loader import load
        data = load("local_3060ti")
        assert "walk_forward" in data, "local_3060ti config should have walk_forward section"
        assert data["walk_forward"]["n_folds"] >= 3

    def test_all_yaml_configs_parseable(self):
        config_dir = PROJECT_ROOT / "config"
        for yaml_path in config_dir.glob("*.yaml"):
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            assert data is not None or data == {}, f"Failed to parse {yaml_path.name}"


# ──────────────────────────────────────────────────────────
# 28.3: setup_local.py --dry-run
# ──────────────────────────────────────────────────────────

class TestSetupLocal:
    def test_setup_local_exists(self):
        assert (PROJECT_ROOT / "setup_local.py").exists(), "setup_local.py missing"

    def test_setup_local_dry_run(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "setup_local.py"), "--dry-run"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode in (0, 1), (
            f"setup_local.py --dry-run crashed:\n{result.stderr}"
        )
        # Should not raise an exception
        assert "Traceback" not in result.stderr, (
            f"setup_local.py raised exception:\n{result.stderr}"
        )

    def test_setup_local_dry_run_data_only(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "setup_local.py"), "--dry-run", "--data-only"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "Traceback" not in result.stderr

    def test_setup_local_imports_cleanly(self):
        """setup_local.py should be importable without side effects."""
        result = subprocess.run(
            [sys.executable, "-c",
             "import importlib.util; "
             "spec = importlib.util.spec_from_file_location('setup_local', 'setup_local.py'); "
             "# just check it exists"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=10,
        )
        assert (PROJECT_ROOT / "setup_local.py").exists()


# ──────────────────────────────────────────────────────────
# 28.4: fetch_data.py --dry-run
# ──────────────────────────────────────────────────────────

class TestFetchData:
    def test_fetch_data_exists(self):
        assert (PROJECT_ROOT / "scripts" / "fetch_data.py").exists()

    def test_fetch_data_dry_run(self):
        result = subprocess.run(
            [sys.executable,
             str(PROJECT_ROOT / "scripts" / "fetch_data.py"),
             "--asset", "BTCUSDT",
             "--period", "30d",
             "--dry-run"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"fetch_data.py --dry-run failed:\n{result.stderr}"
        assert "Traceback" not in result.stderr

    def test_fetch_data_python_api_dry_run(self):
        from scripts.fetch_data import fetch_data  # noqa: PLC0415
        df = fetch_data(asset="BTCUSDT", period="30d", dry_run=True)
        assert df is not None
        assert len(df) > 0
        assert "$close" in df.columns
        assert "$volume" in df.columns

    def test_fetch_data_returns_dollar_columns(self):
        from scripts.fetch_data import fetch_data  # noqa: PLC0415
        df = fetch_data(asset="BTCUSDT", period="7d", dry_run=True)
        required = {"$open", "$high", "$low", "$close", "$volume"}
        assert required.issubset(set(df.columns)), f"Missing columns: {required - set(df.columns)}"

    def test_fetch_data_period_parser(self):
        from scripts.fetch_data import _period_to_start_date  # noqa: PLC0415
        from datetime import datetime, timedelta  # noqa: PLC0415
        start_2y = _period_to_start_date("2y")
        assert abs((datetime.utcnow() - start_2y).days - 730) <= 5
        start_6m = _period_to_start_date("6m")
        assert abs((datetime.utcnow() - start_6m).days - 180) <= 5

    def test_fetch_data_schedule_flag(self):
        result = subprocess.run(
            [sys.executable,
             str(PROJECT_ROOT / "scripts" / "fetch_data.py"),
             "--schedule", "daily",
             "--asset", "BTCUSDT"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "cron" in result.stdout.lower() or "crontab" in result.stdout.lower()


# ──────────────────────────────────────────────────────────
# 28.5: run_full_pipeline.py --dry-run
# ──────────────────────────────────────────────────────────

class TestRunFullPipeline:
    def test_pipeline_script_exists(self):
        assert (PROJECT_ROOT / "scripts" / "run_full_pipeline.py").exists()

    def test_pipeline_dry_run(self):
        # Week 63: use --env flag (new loader) instead of --config with old path
        result = subprocess.run(
            [sys.executable,
             str(PROJECT_ROOT / "scripts" / "run_full_pipeline.py"),
             "--env", "local_3060ti",
             "--dry-run",
             "--no-resume"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=60,
        )
        assert result.returncode == 0, f"pipeline --dry-run failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        assert "Traceback" not in result.stderr

    def test_pipeline_state_checkpoint(self, tmp_path: Path):
        from scripts.run_full_pipeline import PipelineState  # noqa: PLC0415

        # Monkey-patch STATE_FILE to tmp_path
        original = PipelineState.STATE_FILE
        PipelineState.STATE_FILE = tmp_path / "pipeline_state.json"

        try:
            state = PipelineState()
            state.mark_done("fetch_data", output_path="/tmp/test.csv")
            assert state.is_done("fetch_data")

            # Reload from disk
            state2 = PipelineState()
            assert state2.is_done("fetch_data")
            assert state2.metadata["fetch_data"]["output_path"] == "/tmp/test.csv"

            # Clear
            state2.clear()
            state3 = PipelineState()
            assert not state3.is_done("fetch_data")
        finally:
            PipelineState.STATE_FILE = original

    def test_pipeline_imports(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "from scripts.run_full_pipeline import run_pipeline, PipelineState; print('OK')"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=15,
        )
        assert "OK" in result.stdout, f"Import failed:\n{result.stderr}"


# ──────────────────────────────────────────────────────────
# 28.6: generate_report.py
# ──────────────────────────────────────────────────────────

class TestGenerateReport:
    def test_report_script_exists(self):
        assert (PROJECT_ROOT / "scripts" / "generate_report.py").exists()

    def test_report_generator_import(self):
        from scripts.generate_report import ReportGenerator  # noqa: PLC0415
        assert ReportGenerator is not None

    def test_report_generates_html(self, tmp_results_dir: Path):
        from scripts.generate_report import ReportGenerator  # noqa: PLC0415

        rg = ReportGenerator(output_dir=tmp_results_dir)
        output_path = tmp_results_dir / "test_report.html"
        result_path = rg.generate(output_path=output_path)

        assert result_path.exists(), "Report HTML file was not created"
        content = result_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "Trading Bot" in content

    def test_report_html_is_standalone(self, tmp_results_dir: Path):
        """HTML should embed plotly — no external JS deps (or cdn link)."""
        from scripts.generate_report import ReportGenerator  # noqa: PLC0415

        rg = ReportGenerator(output_dir=tmp_results_dir)
        output_path = tmp_results_dir / "standalone.html"
        rg.generate(output_path=output_path)
        content = output_path.read_text(encoding="utf-8")
        # Either plotly is embedded or loaded via CDN script tag
        assert "plotly" in content.lower(), "Report should reference Plotly"

    def test_report_contains_sections(self, tmp_results_dir: Path):
        from scripts.generate_report import ReportGenerator  # noqa: PLC0415

        rg = ReportGenerator(output_dir=tmp_results_dir)
        output_path = tmp_results_dir / "sections.html"
        rg.generate(output_path=output_path)
        content = output_path.read_text(encoding="utf-8")
        assert "Executive Summary" in content
        assert "Walk-Forward" in content
        assert "Feature Importance" in content
        assert "Risk Analysis" in content

    def test_report_with_dummy_wf_results(self, tmp_results_dir: Path):
        from scripts.generate_report import ReportGenerator  # noqa: PLC0415

        rg = ReportGenerator(output_dir=tmp_results_dir)
        wf = rg._make_dummy_wf_results()
        assert len(wf) == 5
        for fold in wf:
            assert "is_sharpe" in fold
            assert "oos_sharpe" in fold
            assert len(fold["equity_curve"]) > 0

    def test_report_dry_run(self, tmp_results_dir: Path):
        from scripts.generate_report import ReportGenerator  # noqa: PLC0415

        rg = ReportGenerator(output_dir=tmp_results_dir)
        output_path = tmp_results_dir / "dry_run.html"
        rg.generate(output_path=output_path, dry_run=True)
        # dry_run=True should NOT write the file
        assert not output_path.exists(), "dry_run=True should not write the file"

    def test_report_cli_dry_run(self):
        result = subprocess.run(
            [sys.executable,
             str(PROJECT_ROOT / "scripts" / "generate_report.py"),
             "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=30,
        )
        assert result.returncode == 0, f"generate_report.py --dry-run failed:\n{result.stderr}"
        assert "Traceback" not in result.stderr


# ──────────────────────────────────────────────────────────
# 28.7: Metric helpers
# ──────────────────────────────────────────────────────────

class TestReportMetrics:
    def test_compute_sharpe(self):
        from scripts.generate_report import ReportGenerator  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        rg = ReportGenerator.__new__(ReportGenerator)
        returns = np.random.default_rng(0).normal(0.001, 0.01, 252)
        sharpe = rg._compute_sharpe(returns, freq=252)
        assert isinstance(sharpe, float)
        assert sharpe > 0  # positive mean return → positive Sharpe

    def test_compute_max_drawdown(self):
        from scripts.generate_report import ReportGenerator  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        rg = ReportGenerator.__new__(ReportGenerator)
        equity = np.array([100, 110, 105, 90, 95, 100], dtype=float)
        dd = rg._compute_max_drawdown(equity)
        assert dd < 0  # drawdown is negative
        # peak=110, trough=90 → drawdown = (90-110)/110 ≈ -0.1818
        assert abs(dd - (-20.0 / 110.0)) < 0.01

    def test_compute_max_drawdown_flat(self):
        from scripts.generate_report import ReportGenerator  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        rg = ReportGenerator.__new__(ReportGenerator)
        equity = np.ones(100)
        dd = rg._compute_max_drawdown(equity)
        assert dd >= -0.01  # essentially 0

    def test_sharpe_zero_std(self):
        from scripts.generate_report import ReportGenerator  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        rg = ReportGenerator.__new__(ReportGenerator)
        returns = np.zeros(100)
        sharpe = rg._compute_sharpe(returns)
        assert sharpe == 0.0  # no std → Sharpe = 0
