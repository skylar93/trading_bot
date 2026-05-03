"""
tests/config/test_config_loader.py — Week 63 (S40)

Config Consolidation 테스트:
- 모든 env config이 loader.load()로 성공적으로 로드됨
- deep_merge가 올바르게 동작함
- schema validation이 잘못된 값을 거부함
- load_raw()가 레거시 경로를 처리함
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

# Project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import _deep_merge, load, load_raw, _list_envs


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

ENV_DIR = PROJECT_ROOT / "config" / "env"
ALL_ENVS = [p.stem for p in ENV_DIR.glob("*.yaml")]


# ────────────────────────────────────────────────────────────
# S40-A: 모든 env config이 loader로 로드 & schema validate 통과
# ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("env", ALL_ENVS)
def test_load_all_envs(env: str) -> None:
    """각 env config이 오류 없이 load()되어야 한다."""
    cfg = load(env)
    assert isinstance(cfg, dict), f"load({env!r}) should return dict"
    # 최소 필수 top-level 키 존재 확인
    assert "env" in cfg, "merged config must have 'env' key"
    assert "training" in cfg, "merged config must have 'training' key"
    assert "risk_management" in cfg, "merged config must have 'risk_management' key"
    assert "monitoring" in cfg, "merged config must have 'monitoring' key"


def test_load_default_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """환경변수 없을 때 기본 env(local_3060ti)로 로드된다."""
    monkeypatch.delenv("TRADING_BOT_ENV", raising=False)
    cfg = load()
    assert cfg is not None


def test_load_env_via_envvar(monkeypatch: pytest.MonkeyPatch) -> None:
    """TRADING_BOT_ENV 환경변수가 env 선택에 사용된다."""
    monkeypatch.setenv("TRADING_BOT_ENV", "test")
    cfg = load()
    # test env sets mlflow.enabled = false
    assert cfg.get("mlflow", {}).get("enabled") is False


def test_load_unknown_env_raises() -> None:
    """존재하지 않는 env는 FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Env override not found"):
        load("nonexistent_env_xyz")


# ────────────────────────────────────────────────────────────
# S40-B: deep_merge 정확성
# ────────────────────────────────────────────────────────────

def test_deep_merge_override_scalar() -> None:
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    over = {"a": 99, "b": {"c": 200}}
    result = _deep_merge(base, over)
    assert result["a"] == 99
    assert result["b"]["c"] == 200
    assert result["b"]["d"] == 3  # preserved


def test_deep_merge_adds_new_keys() -> None:
    base = {"a": 1}
    over = {"b": {"x": 10}}
    result = _deep_merge(base, over)
    assert result["a"] == 1
    assert result["b"]["x"] == 10


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"b": 1}}
    over = {"a": {"c": 2}}
    _deep_merge(base, over)
    assert "c" not in base["a"]


def test_deep_merge_list_is_replaced() -> None:
    """List는 재귀 merge 없이 통째로 교체된다."""
    base = {"agents": [1, 2, 3]}
    over = {"agents": [4, 5]}
    result = _deep_merge(base, over)
    assert result["agents"] == [4, 5]


# ────────────────────────────────────────────────────────────
# S40-C: env override 우선순위 검증
# ────────────────────────────────────────────────────────────

def test_test_env_overrides_training_timesteps() -> None:
    """test env는 total_timesteps를 10000으로 오버라이드해야 한다."""
    cfg = load("test")
    assert cfg["training"]["total_timesteps"] == 10000


def test_test_env_disables_mlflow() -> None:
    cfg = load("test")
    assert cfg["mlflow"]["enabled"] is False


def test_test_env_disables_persistence() -> None:
    cfg = load("test")
    assert cfg["persistence"]["enabled"] is False


def test_local_3060ti_env_sets_cuda() -> None:
    cfg = load("local_3060ti")
    assert cfg["training"]["device"] == "cuda:0"


def test_local_m2_env_sets_mps() -> None:
    cfg = load("local_m2")
    assert cfg["training"]["device"] == "mps"


def test_local_m2_disables_ensemble() -> None:
    cfg = load("local_m2")
    assert cfg["ensemble"]["enabled"] is False


def test_uw_gpu_enables_parallel_training() -> None:
    cfg = load("uw_gpu")
    assert cfg["training"]["ensemble_training"] == "parallel"


# ────────────────────────────────────────────────────────────
# S40-D: Schema validation (FullConfig)
# ────────────────────────────────────────────────────────────

def test_schema_rejects_invalid_trading_fee() -> None:
    """잘못된 trading_fee는 ValidationError를 일으켜야 한다."""
    from pydantic import ValidationError
    from config.schema import FullConfig

    with pytest.raises(ValidationError):
        FullConfig(**{"env": {"trading_fee": 0.5}})  # > 0.1 → invalid


def test_schema_rejects_bad_algo_type() -> None:
    from pydantic import ValidationError
    from config.schema import FullConfig

    with pytest.raises(ValidationError):
        FullConfig(**{"agent": {"algo_type": "invalid_algo"}})


def test_schema_rejects_stop_loss_gte_drawdown() -> None:
    """stop_loss_threshold >= max_drawdown_pct → ValidationError."""
    from pydantic import ValidationError
    from config.schema import FullConfig

    with pytest.raises(ValidationError):
        FullConfig(**{
            "risk_management": {
                "stop_loss_threshold": 0.25,
                "trailing_stop_buffer": 0.03,
                "max_drawdown_pct": 0.20,
            }
        })


# ────────────────────────────────────────────────────────────
# S40-E: load_raw (레거시 호환)
# ────────────────────────────────────────────────────────────

def test_load_raw_valid_file(tmp_path: Path) -> None:
    """load_raw는 단일 YAML 파일을 로드해야 한다."""
    cfg_file = tmp_path / "custom.yaml"
    cfg_file.write_text(
        "env:\n  window_size: 20\ntraining:\n  total_timesteps: 1000\n"
        "agent:\n  algo_type: sb3_ppo\npaths:\n  checkpoint_dir: ckpt\n"
    )
    result = load_raw(str(cfg_file))
    assert result["env"]["window_size"] == 20
    assert result["training"]["total_timesteps"] == 1000


def test_load_raw_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_raw("/nonexistent/path/config.yaml")


# ────────────────────────────────────────────────────────────
# S40-F: _list_envs
# ────────────────────────────────────────────────────────────

def test_list_envs_returns_all_env_names() -> None:
    envs = _list_envs()
    assert "local_3060ti" in envs
    assert "local_m2" in envs
    assert "test" in envs
    assert "uw_gpu" in envs


# ────────────────────────────────────────────────────────────
# S40-G: base config 파일 수 검증 (50% 감소 조건)
# ────────────────────────────────────────────────────────────

def test_config_file_count_reduced() -> None:
    """config/ 전체 YAML 파일 수 ≤ 12 (Phase 8-Alpha: futures_maker.yaml 추가로 +1)."""
    config_dir = PROJECT_ROOT / "config"
    all_yamls = list(config_dir.rglob("*.yaml"))
    count = len(all_yamls)
    assert count <= 12, (
        f"Expected ≤12 config YAML files, "
        f"found {count}: {[str(p.relative_to(config_dir)) for p in all_yamls]}"
    )


def test_old_config_files_deleted() -> None:
    """삭제되어야 할 구 config 파일이 없어야 한다."""
    config_dir = PROJECT_ROOT / "config"
    deleted = [
        "default_config.yaml",
        "training_config.yaml",
        "ensemble.yaml",
        "single_agent.yaml",
        "sac.yaml",
        "td3.yaml",
        "flag_trader.yaml",
        "risk_management.yaml",
        "paper_trading.yaml",
        "test_config.yaml",
        "local_3060ti.yaml",
        "local_m2.yaml",
    ]
    for fname in deleted:
        fpath = config_dir / fname
        assert not fpath.exists(), f"Old config file should have been deleted: {fname}"
