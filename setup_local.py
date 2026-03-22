#!/usr/bin/env python3
"""
One-click local setup script for the Trading Bot.

Usage:
    python setup_local.py --gpu 3060ti
    python setup_local.py --gpu m2
    python setup_local.py --gpu cpu
    python setup_local.py --gpu 3060ti --skip-optional
    python setup_local.py --gpu 3060ti --data-only

Flags:
    --gpu [3060ti|m1|m2|cpu]    Target hardware profile
    --skip-optional             Skip transformers, hmmlearn (faster install)
    --data-only                 Only set up data directories, skip training check
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# ─────────────────────────────────────────────
# Color helpers (works on Mac/Linux; stripped on Windows)
# ─────────────────────────────────────────────
_NO_COLOR = platform.system() == "Windows"


def _c(text: str, code: str) -> str:
    if _NO_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def ok(msg: str) -> None:
    print(_c(f"  ✓ {msg}", "32"))


def warn(msg: str) -> None:
    print(_c(f"  ⚠ {msg}", "33"))


def err(msg: str) -> None:
    print(_c(f"  ✗ {msg}", "31"))


def info(msg: str) -> None:
    print(_c(f"  → {msg}", "36"))


def header(msg: str) -> None:
    print(_c(f"\n{'='*60}\n  {msg}\n{'='*60}", "1"))


# ─────────────────────────────────────────────
# Step 1: Python version check
# ─────────────────────────────────────────────

def check_python() -> bool:
    header("Step 1 / 9 — Python 버전 확인")
    major, minor = sys.version_info[:2]
    info(f"Python {major}.{minor} 감지")
    if (major, minor) < (3, 10):
        err(f"Python 3.10+ 필요. 현재: {major}.{minor}")
        err("https://www.python.org/downloads/ 에서 업그레이드하세요.")
        return False
    ok(f"Python {major}.{minor} — OK")
    return True


# ─────────────────────────────────────────────
# Step 2: CUDA / device detection
# ─────────────────────────────────────────────

def detect_device(gpu_arg: str) -> dict:
    """Returns device info dict based on --gpu flag and actual hardware."""
    header("Step 2 / 9 — GPU / 디바이스 감지")

    result = {
        "gpu_arg": gpu_arg,
        "torch_index_url": None,
        "device": "cpu",
        "config": "config/default_config.yaml",
    }

    if gpu_arg in ("3060ti",):
        # Try to detect CUDA
        cuda_available = shutil.which("nvidia-smi") is not None
        if cuda_available:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
                ok(f"NVIDIA GPU 감지: {out}")
            except Exception:
                warn("nvidia-smi 실행 실패 — CUDA 설치 확인 필요")
            result["torch_index_url"] = "https://download.pytorch.org/whl/cu121"
            result["device"] = "cuda"
            result["config"] = "config/local_3060ti.yaml"
        else:
            warn("nvidia-smi 없음 — CPU 모드로 fallback")
            result["device"] = "cpu"
            result["config"] = "config/default_config.yaml"

    elif gpu_arg in ("m1", "m2"):
        info("Apple Silicon 감지 — MPS 사용")
        result["device"] = "mps"
        result["config"] = "config/local_m2.yaml"

    else:  # cpu
        info("CPU 모드로 설정")
        result["device"] = "cpu"
        result["config"] = "config/default_config.yaml"

    info(f"디바이스: {result['device']}  |  config: {result['config']}")
    return result


# ─────────────────────────────────────────────
# Step 3: venv / conda env 생성
# ─────────────────────────────────────────────

def setup_env(project_root: Path) -> tuple[Path, bool]:
    """Returns (python_executable, is_new_env)."""
    header("Step 3 / 9 — 가상환경 설정")

    venv_path = project_root / ".venv"
    python_in_venv = venv_path / ("Scripts" if platform.system() == "Windows" else "bin") / "python"

    if venv_path.exists():
        ok(f"기존 venv 발견: {venv_path}")
        return python_in_venv, False

    # Check conda first
    conda_exe = shutil.which("conda")
    if conda_exe:
        info("Conda 감지 — conda env 생성 시도")
        ret = subprocess.call(
            [conda_exe, "create", "-n", "trading_bot", "python=3.10", "-y"],
            stdout=subprocess.DEVNULL,
        )
        if ret == 0:
            ok("Conda env 'trading_bot' 생성 완료")
            warn("conda activate trading_bot 후 다시 실행하세요.")
            sys.exit(0)
        warn("conda env 생성 실패 — venv로 fallback")

    info(f"venv 생성: {venv_path}")
    subprocess.check_call([sys.executable, "-m", "venv", str(venv_path)])
    ok(f"venv 생성 완료: {venv_path}")
    return python_in_venv, True


# ─────────────────────────────────────────────
# Step 4: PyTorch 설치
# ─────────────────────────────────────────────

def install_pytorch(python: Path, device_info: dict) -> bool:
    header("Step 4 / 9 — PyTorch 설치")

    # Check if already installed
    ret = subprocess.call(
        [str(python), "-c", "import torch; print(torch.__version__)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ret == 0:
        ok("PyTorch 이미 설치됨 — 건너뜀")
        return True

    cmd = [str(python), "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
    index_url = device_info.get("torch_index_url")
    if index_url:
        cmd += ["--index-url", index_url]

    info(f"PyTorch 설치 중 (device={device_info['device']})...")
    ret = subprocess.call(cmd)
    if ret != 0:
        err("PyTorch 설치 실패")
        return False
    ok("PyTorch 설치 완료")
    return True


# ─────────────────────────────────────────────
# Step 5: requirements.txt 설치
# ─────────────────────────────────────────────

def install_requirements(python: Path, project_root: Path, skip_optional: bool) -> bool:
    header("Step 5 / 9 — requirements.txt 설치")

    req_file = project_root / "requirements.txt"
    if not req_file.exists():
        err(f"requirements.txt 없음: {req_file}")
        return False

    info("핵심 패키지 설치 중...")
    ret = subprocess.call([str(python), "-m", "pip", "install", "-r", str(req_file)])
    if ret != 0:
        err("requirements.txt 설치 실패")
        return False
    ok("핵심 패키지 설치 완료")

    if not skip_optional:
        optional_pkgs = [
            ("hmmlearn>=0.3.0", "HMM regime detection"),
            ("river>=0.21.0",   "ADWIN drift detection"),
        ]
        for pkg, desc in optional_pkgs:
            info(f"Optional: {desc} ({pkg})")
            ret = subprocess.call(
                [str(python), "-m", "pip", "install", pkg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if ret == 0:
                ok(f"  {desc} 설치 완료")
            else:
                warn(f"  {desc} 설치 실패 — fallback 모드 사용")
    else:
        warn("--skip-optional: transformers, hmmlearn 건너뜀")

    return True


# ─────────────────────────────────────────────
# Step 6: 데이터 디렉토리 생성
# ─────────────────────────────────────────────

def create_data_dirs(project_root: Path) -> None:
    header("Step 6 / 9 — 데이터 디렉토리 생성")

    dirs = [
        project_root / "data" / "raw",
        project_root / "data" / "processed",
        project_root / "data" / "cache",
        project_root / "checkpoints",
        project_root / "logs",
        project_root / "mlruns",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        ok(f"  {d.relative_to(project_root)}")


# ─────────────────────────────────────────────
# Step 7: config 복사 (없을 때만)
# ─────────────────────────────────────────────

def copy_config(project_root: Path, device_info: dict) -> None:
    header("Step 7 / 9 — Config 확인")

    src = project_root / device_info["config"]
    dst = project_root / "config" / "active_config.yaml"

    if dst.exists():
        ok(f"active_config.yaml 이미 존재 — 유지")
    else:
        if src.exists():
            shutil.copy(src, dst)
            ok(f"복사: {src.name} → config/active_config.yaml")
        else:
            warn(f"기본 config 없음: {src} — default_config.yaml 사용")
            fallback = project_root / "config" / "default_config.yaml"
            if fallback.exists():
                shutil.copy(fallback, dst)


# ─────────────────────────────────────────────
# Step 8: sanity check
# ─────────────────────────────────────────────

def run_sanity_check(python: Path, project_root: Path, device_info: dict) -> bool:
    header("Step 8 / 9 — Sanity check")

    checks = [
        # (description, code_snippet)
        ("numpy import",    "import numpy; print(numpy.__version__)"),
        ("pandas import",   "import pandas; print(pandas.__version__)"),
        ("gymnasium import","import gymnasium; print(gymnasium.__version__)"),
        ("SB3 import",      "import stable_baselines3; print(stable_baselines3.__version__)"),
    ]

    # Add device-specific torch check
    if device_info["device"] == "cuda":
        checks.append((
            "CUDA available",
            "import torch; assert torch.cuda.is_available(), 'CUDA not available'; "
            "print(f'CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')",
        ))
    elif device_info["device"] == "mps":
        checks.append((
            "MPS available",
            "import torch; assert torch.backends.mps.is_available(), 'MPS not available'; "
            "print('MPS OK')",
        ))
    else:
        checks.append(("torch import", "import torch; print(torch.__version__)"))

    all_ok = True
    for desc, code in checks:
        ret = subprocess.call(
            [str(python), "-c", code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(project_root),
        )
        if ret == 0:
            ok(f"  {desc}")
        else:
            err(f"  {desc} — 실패")
            all_ok = False

    return all_ok


# ─────────────────────────────────────────────
# Step 9: Quick 100-step training smoke test
# ─────────────────────────────────────────────

def run_smoke_test(python: Path, project_root: Path) -> bool:
    header("Step 9 / 9 — 빠른 학습 smoke test (100 steps)")

    smoke_code = """
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent if '__file__' in dir() else '.'))
import numpy as np
import pandas as pd
from envs.single_asset_rl_env import SingleAssetRLTradingEnv
from stable_baselines3 import PPO

# 최소 데이터 생성 (200 rows)
np.random.seed(42)
n = 200
data = pd.DataFrame({
    '$open':   100 + np.cumsum(np.random.randn(n) * 0.5),
    '$high':   100 + np.cumsum(np.random.randn(n) * 0.5) + 1,
    '$low':    100 + np.cumsum(np.random.randn(n) * 0.5) - 1,
    '$close':  100 + np.cumsum(np.random.randn(n) * 0.5),
    '$volume': np.random.randint(1000, 10000, n).astype(float),
})
data['$high'] = data[['$open', '$high', '$close']].max(axis=1)
data['$low']  = data[['$open', '$low',  '$close']].min(axis=1)

env = SingleAssetRLTradingEnv(data=data, initial_capital=10000, window_size=20)
model = PPO("MlpPolicy", env, verbose=0, n_steps=64, batch_size=32)
model.learn(total_timesteps=100)
print("smoke test passed")
"""

    smoke_file = project_root / "_smoke_test_tmp.py"
    try:
        smoke_file.write_text(smoke_code)
        ret = subprocess.call(
            [str(python), str(smoke_file)],
            cwd=str(project_root),
            env={**os.environ, "PYTHONPATH": str(project_root)},
        )
        return ret == 0
    finally:
        smoke_file.unlink(missing_ok=True)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Trading Bot — one-click local setup")
    parser.add_argument(
        "--gpu",
        choices=["3060ti", "m1", "m2", "cpu"],
        default="cpu",
        help="Target hardware profile",
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="transformers, hmmlearn 등 optional 패키지 건너뜀",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="데이터 디렉토리만 생성, 학습 테스트 건너뜀",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent

    print(_c("\n╔══════════════════════════════════════════════╗", "1"))
    print(_c("║   Trading Bot — Local Setup                  ║", "1"))
    print(_c("╚══════════════════════════════════════════════╝", "1"))
    info(f"프로젝트 루트: {project_root}")
    info(f"GPU 프로파일: {args.gpu}")

    # 1. Python check
    if not check_python():
        sys.exit(1)

    # 2. Device detection
    device_info = detect_device(args.gpu)

    # 3. Env setup
    python_exe, is_new = setup_env(project_root)
    info(f"Python 실행 파일: {python_exe}")

    # 4. PyTorch
    if not args.data_only:
        if not install_pytorch(python_exe, device_info):
            sys.exit(1)

    # 5. Requirements
    if not args.data_only:
        if not install_requirements(python_exe, project_root, args.skip_optional):
            sys.exit(1)

    # 6. Data dirs
    create_data_dirs(project_root)

    # 7. Config
    copy_config(project_root, device_info)

    if args.data_only:
        header("완료 (--data-only)")
        ok("데이터 디렉토리 설정 완료")
        sys.exit(0)

    # 8. Sanity check
    check_ok = run_sanity_check(python_exe, project_root, device_info)

    # 9. Smoke test
    smoke_ok = False
    if check_ok:
        smoke_ok = run_smoke_test(python_exe, project_root)
    else:
        warn("sanity check 실패 — smoke test 건너뜀")

    # ── 최종 리포트 ──
    print(_c("\n" + "="*60, "1"))
    print(_c("  Setup 결과", "1"))
    print(_c("="*60, "1"))
    if check_ok and smoke_ok:
        ok("모든 검증 통과 — 준비 완료!")
        print()
        info("다음 단계:")
        info(f"  1. python scripts/fetch_data.py --asset BTCUSDT --period 1y")
        info(f"  2. python -m training.train_pipeline --config {device_info['config']}")
        info(f"  3. streamlit run deployment/web_interface/app.py")
    elif check_ok:
        warn("패키지 OK, smoke test 실패 — 환경은 준비됨")
        warn("docs/USER_GUIDE.md 트러블슈팅 섹션 참고")
    else:
        err("설정 실패 — 위 에러 메시지를 확인하세요")
        err("docs/USER_GUIDE.md 트러블슈팅 섹션 참고")
        sys.exit(1)


if __name__ == "__main__":
    main()
