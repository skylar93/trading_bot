"""I12-c: Tests for scripts/verify_doc_commands.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_doc_commands import verify_file, main


class TestVerifyFile:
    def test_valid_bash_block_passes(self, tmp_path: Path):
        doc = tmp_path / "test.md"
        doc.write_text("```bash\necho hello\n```\n")
        failures = verify_file(doc)
        assert failures == []

    def test_broken_bash_block_is_caught(self, tmp_path: Path):
        doc = tmp_path / "bad.md"
        doc.write_text("```bash\nif true\nthen\n# missing fi\n```\n")
        failures = verify_file(doc)
        assert len(failures) == 1
        assert "bad.md" in failures[0]

    def test_multiple_blocks_all_valid(self, tmp_path: Path):
        doc = tmp_path / "multi.md"
        doc.write_text(
            "```bash\necho a\n```\nsome text\n```bash\nls -la\n```\n"
        )
        failures = verify_file(doc)
        assert failures == []

    def test_one_bad_among_valid_blocks(self, tmp_path: Path):
        doc = tmp_path / "mixed.md"
        doc.write_text(
            "```bash\necho good\n```\n"
            "```bash\nif true\nthen\n# no fi\n```\n"
            "```bash\npwd\n```\n"
        )
        failures = verify_file(doc)
        assert len(failures) == 1
        assert "block 1" in failures[0]

    def test_missing_file_returns_failure(self, tmp_path: Path):
        missing = tmp_path / "nonexistent.md"
        failures = verify_file(missing)
        assert len(failures) == 1
        assert "not found" in failures[0]

    def test_no_bash_blocks_passes(self, tmp_path: Path):
        doc = tmp_path / "no_blocks.md"
        doc.write_text("# Just prose\n\nNo code blocks here.\n")
        failures = verify_file(doc)
        assert failures == []

    def test_non_bash_blocks_are_ignored(self, tmp_path: Path):
        doc = tmp_path / "python_block.md"
        doc.write_text("```python\nif True\n    pass\n```\n")
        failures = verify_file(doc)
        assert failures == []


class TestMain:
    def test_main_returns_0_for_valid_docs(self, tmp_path: Path):
        doc = tmp_path / "valid.md"
        doc.write_text("```bash\necho ok\n```\n")
        rc = main([doc])
        assert rc == 0

    def test_main_returns_1_for_broken_doc(self, tmp_path: Path):
        doc = tmp_path / "broken.md"
        doc.write_text("```bash\nif true\nthen\n```\n")
        rc = main([doc])
        assert rc == 1

    def test_main_returns_1_for_missing_file(self, tmp_path: Path):
        missing = tmp_path / "gone.md"
        rc = main([missing])
        assert rc == 1
