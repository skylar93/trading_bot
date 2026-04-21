"""
DVC initialization helper — Week 79 (H8).

Run once after cloning the repo:
    python scripts/setup_dvc.py

What it does:
  1. Runs `dvc init` (idempotent)
  2. Configures local cache path
  3. Adds data/raw/ and data/processed/ to DVC tracking
  4. Creates .dvcignore with sensible defaults
  5. Prints next steps
"""

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=ROOT, check=check, capture_output=False)


def main() -> None:
    print("=== DVC Setup (Week 79 / H8) ===\n")

    # 1. init
    print("[1/5] Initialising DVC (idempotent)...")
    run(["dvc", "init", "--no-scm"], check=False)  # --no-scm if already git-managed

    # 2. local cache
    cache_dir = ROOT / ".dvc" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    run(["dvc", "config", "cache.dir", str(cache_dir)])

    # 3. add directories
    print("\n[3/5] Adding data directories to DVC tracking...")
    for d in ["data/raw", "data/processed"]:
        path = ROOT / d
        path.mkdir(parents=True, exist_ok=True)
        (path / ".gitkeep").touch()
        # Only add if there are actual files
        files = [f for f in path.iterdir() if f.name != ".gitkeep"]
        if files:
            run(["dvc", "add", str(d)])
        else:
            print(f"  Skipping {d} (empty — add data files first)")

    # 4. .dvcignore
    dvcignore = ROOT / ".dvcignore"
    if not dvcignore.exists():
        print("\n[4/5] Writing .dvcignore...")
        dvcignore.write_text(textwrap.dedent("""\
            # DVC ignore rules
            **/.DS_Store
            **/__pycache__
            **/*.pyc
            **/mlruns
            **/wandb
        """))
    else:
        print("\n[4/5] .dvcignore already exists — skipping")

    # 5. next steps
    print("\n[5/5] Done.\n")
    print(textwrap.dedent("""
    Next steps:
      - Add raw data:     cp your_data.csv data/raw/
      - Run pipeline:     dvc repro
      - Commit DVC files: git add data/raw.dvc data/processed.dvc dvc.yaml .dvcignore
                          git commit -m "track data with DVC (H8)"
      - Push cache:       dvc push   (after configuring a remote with `dvc remote add`)

    To pin a model to a dataset version:
      - After training:   git log --oneline -1   (note the commit hash)
      - In model meta:    {"data_commit": "<hash>", "dvc_lock": "dvc.lock"}
    """))


if __name__ == "__main__":
    main()
