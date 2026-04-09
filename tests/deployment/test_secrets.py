"""
Tests for Week 58: Secrets Management (S12-S15).

Covers:
  - EnvSecretProvider: normal lookup, missing key
  - FileSecretProvider: normal lookup, missing key, missing file, invalid JSON
  - KeychainSecretProvider: graceful import error path
  - get_default_provider: backend selection via env var
  - config_resolver.resolve_secrets: nested dict, list, missing secret
  - Config file: paper_trading.yaml must NOT contain plaintext api_key/api_secret
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deployment.secrets.secret_provider import (
    EnvSecretProvider,
    FileSecretProvider,
    SecretProvider,
    get_default_provider,
)
from deployment.secrets.config_resolver import resolve_secrets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_secrets_file(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "secrets.json"
    p.write_text(json.dumps(data))
    return p


# ---------------------------------------------------------------------------
# S12 — SecretProvider implementations
# ---------------------------------------------------------------------------

class TestEnvSecretProvider:
    def test_get_existing_key(self, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "my_secret_value")
        provider = EnvSecretProvider()
        assert provider.get("TEST_API_KEY") == "my_secret_value"

    def test_get_missing_key_raises_keyerror(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_KEY_XYZ", raising=False)
        provider = EnvSecretProvider()
        with pytest.raises(KeyError, match="NONEXISTENT_KEY_XYZ"):
            provider.get("NONEXISTENT_KEY_XYZ")

    def test_get_optional_returns_default_on_missing(self, monkeypatch):
        monkeypatch.delenv("MISSING_KEY_ABC", raising=False)
        provider = EnvSecretProvider()
        assert provider.get_optional("MISSING_KEY_ABC", default="fallback") == "fallback"

    def test_get_optional_returns_value_when_present(self, monkeypatch):
        monkeypatch.setenv("PRESENT_KEY", "real_value")
        provider = EnvSecretProvider()
        assert provider.get_optional("PRESENT_KEY") == "real_value"

    def test_is_secret_provider_subclass(self):
        assert issubclass(EnvSecretProvider, SecretProvider)


class TestFileSecretProvider:
    def test_get_existing_key(self, tmp_path):
        p = _make_secrets_file(tmp_path, {"MY_KEY": "my_val"})
        provider = FileSecretProvider(path=p)
        assert provider.get("MY_KEY") == "my_val"

    def test_get_missing_key_raises_keyerror(self, tmp_path):
        p = _make_secrets_file(tmp_path, {"OTHER_KEY": "val"})
        provider = FileSecretProvider(path=p)
        with pytest.raises(KeyError, match="MISSING_KEY"):
            provider.get("MISSING_KEY")

    def test_missing_file_raises_file_not_found(self, tmp_path):
        provider = FileSecretProvider(path=tmp_path / "nonexistent.json")
        with pytest.raises(FileNotFoundError):
            provider.get("ANY_KEY")

    def test_invalid_json_raises_value_error(self, tmp_path):
        p = tmp_path / "secrets.json"
        p.write_text("not-valid-json")
        provider = FileSecretProvider(path=p)
        with pytest.raises(Exception):  # json.JSONDecodeError subclasses ValueError
            provider.get("ANY_KEY")

    def test_non_dict_json_raises_value_error(self, tmp_path):
        p = tmp_path / "secrets.json"
        p.write_text('["list", "not", "dict"]')
        provider = FileSecretProvider(path=p)
        with pytest.raises(ValueError, match="JSON object"):
            provider.get("ANY_KEY")

    def test_cache_is_used_on_second_call(self, tmp_path):
        p = _make_secrets_file(tmp_path, {"K": "V"})
        provider = FileSecretProvider(path=p)
        provider.get("K")  # populates cache
        # Overwrite file with different content — cache should still return old value
        p.write_text(json.dumps({"K": "NEW_V"}))
        assert provider.get("K") == "V"

    def test_invalidate_cache_forces_reload(self, tmp_path):
        p = _make_secrets_file(tmp_path, {"K": "V"})
        provider = FileSecretProvider(path=p)
        provider.get("K")
        p.write_text(json.dumps({"K": "NEW_V"}))
        provider.invalidate_cache()
        assert provider.get("K") == "NEW_V"

    def test_default_path_uses_home_dir(self):
        provider = FileSecretProvider()
        expected = Path.home() / ".trading_bot" / "secrets.json"
        assert provider._path == expected


class TestKeychainSecretProvider:
    def test_import_error_raised_without_keyring(self):
        """If keyring is not installed, KeychainSecretProvider.__init__ raises ImportError."""
        from deployment.secrets.secret_provider import KeychainSecretProvider

        with patch.dict("sys.modules", {"keyring": None}):  # type: ignore[dict-item]
            with pytest.raises(ImportError, match="keyring"):
                KeychainSecretProvider()

    def test_get_delegates_to_keyring(self):
        """With a mocked keyring, get() returns the stored password."""
        from deployment.secrets.secret_provider import KeychainSecretProvider

        mock_keyring = MagicMock()
        mock_keyring.get_password.return_value = "stored_secret"

        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            provider = KeychainSecretProvider()
            result = provider.get("MY_SERVICE_KEY")
            assert result == "stored_secret"
            mock_keyring.get_password.assert_called_once_with("trading_bot", "MY_SERVICE_KEY")

    def test_missing_keychain_entry_raises_keyerror(self):
        from deployment.secrets.secret_provider import KeychainSecretProvider

        mock_keyring = MagicMock()
        mock_keyring.get_password.return_value = None  # key not set

        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            provider = KeychainSecretProvider()
            with pytest.raises(KeyError, match="MY_MISSING_KEY"):
                provider.get("MY_MISSING_KEY")


class TestGetDefaultProvider:
    def test_default_returns_env_provider(self, monkeypatch):
        monkeypatch.delenv("TRADING_BOT_SECRET_BACKEND", raising=False)
        provider = get_default_provider()
        assert isinstance(provider, EnvSecretProvider)

    def test_env_backend_explicit(self, monkeypatch):
        monkeypatch.setenv("TRADING_BOT_SECRET_BACKEND", "env")
        provider = get_default_provider()
        assert isinstance(provider, EnvSecretProvider)

    def test_file_backend(self, monkeypatch):
        monkeypatch.setenv("TRADING_BOT_SECRET_BACKEND", "file")
        provider = get_default_provider()
        assert isinstance(provider, FileSecretProvider)

    def test_unknown_backend_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("TRADING_BOT_SECRET_BACKEND", "nonexistent_backend")
        provider = get_default_provider()
        assert isinstance(provider, EnvSecretProvider)


# ---------------------------------------------------------------------------
# S13 — Config resolver
# ---------------------------------------------------------------------------

class TestConfigResolver:
    def _make_provider(self, secrets: dict) -> SecretProvider:
        """Return a FileSecretProvider backed by a temp file."""
        import tempfile, json, pathlib
        tf = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(secrets, tf)
        tf.close()
        return FileSecretProvider(path=tf.name)

    def test_resolves_top_level_ref(self):
        provider = self._make_provider({"MY_KEY": "resolved_value"})
        config = {"api_key_ref": "MY_KEY", "other": "unchanged"}
        result = resolve_secrets(config, provider)
        assert result["api_key"] == "resolved_value"
        assert result["other"] == "unchanged"
        assert "api_key_ref" not in result

    def test_resolves_nested_ref(self):
        provider = self._make_provider({"NESTED_KEY": "nested_val"})
        config = {"section": {"api_key_ref": "NESTED_KEY"}}
        result = resolve_secrets(config, provider)
        assert result["section"]["api_key"] == "nested_val"

    def test_resolves_ref_in_list_dict(self):
        provider = self._make_provider({"LIST_KEY": "list_val"})
        config = {"items": [{"api_key_ref": "LIST_KEY"}]}
        result = resolve_secrets(config, provider)
        assert result["items"][0]["api_key"] == "list_val"

    def test_does_not_modify_original(self):
        provider = self._make_provider({"K": "V"})
        config = {"api_key_ref": "K"}
        original = config.copy()
        resolve_secrets(config, provider)
        assert config == original

    def test_non_ref_keys_pass_through(self):
        provider = self._make_provider({})
        config = {"symbol": "BTC/USDT", "fee": 0.001}
        result = resolve_secrets(config, provider)
        assert result == {"symbol": "BTC/USDT", "fee": 0.001}

    def test_empty_ref_value_is_kept_as_ref(self):
        """Empty string ref value: key is NOT resolved (treated as absent)."""
        provider = self._make_provider({})
        config = {"api_key_ref": ""}
        result = resolve_secrets(config, provider)
        # empty _ref is left as-is (no secret lookup)
        assert "api_key_ref" in result
        assert "api_key" not in result

    def test_missing_secret_raises_key_error(self):
        provider = self._make_provider({})  # empty — key not present
        config = {"api_key_ref": "MISSING_SECRET_KEY"}
        with pytest.raises(KeyError, match="MISSING_SECRET_KEY"):
            resolve_secrets(config, provider)

    def test_uses_default_provider_when_none(self, monkeypatch):
        monkeypatch.setenv("SOME_SECRET_123", "hello")
        monkeypatch.setenv("TRADING_BOT_SECRET_BACKEND", "env")
        config = {"token_ref": "SOME_SECRET_123"}
        result = resolve_secrets(config)  # no provider argument
        assert result["token"] == "hello"


# ---------------------------------------------------------------------------
# S14 — Config file audit: no plaintext credentials
# ---------------------------------------------------------------------------

class TestNoPlaintextCredentialsInConfigs:
    """Ensure no YAML config file contains plaintext api_key / api_secret values."""

    def _get_config_dir(self) -> Path:
        here = Path(__file__).parent
        root = here.parent.parent  # tests/ -> repo root
        return root / "config"

    def test_no_plaintext_api_key_in_yaml_configs(self):
        import re
        config_dir = self._get_config_dir()
        pattern = re.compile(r'^\s*api_(key|secret)\s*:\s*["\'][^"\']{1,}["\']', re.MULTILINE)
        violations = []
        for yaml_file in config_dir.glob("**/*.yaml"):
            content = yaml_file.read_text()
            if pattern.search(content):
                violations.append(str(yaml_file))
        assert violations == [], (
            f"Plaintext API credentials found in config files: {violations}\n"
            "Use 'api_key_ref: \"ENV_VAR_NAME\"' instead."
        )

    def test_paper_trading_yaml_uses_secret_ref(self):
        config_dir = self._get_config_dir()
        content = (config_dir / "paper_trading.yaml").read_text()
        assert "api_key_ref" in content, "paper_trading.yaml should use api_key_ref"
        assert "api_secret_ref" in content, "paper_trading.yaml should use api_secret_ref"
        assert "api_key:" not in content or "api_key_ref:" in content

    def test_paper_trading_yaml_no_bare_api_key(self):
        import re
        config_dir = self._get_config_dir()
        content = (config_dir / "paper_trading.yaml").read_text()
        # Lines like `  api_key: "something"` or `  api_key: ''` with non-empty value
        bad = re.findall(r'^\s*api_(key|secret)\s*:\s*["\'][^"\']+["\']', content, re.MULTILINE)
        assert bad == [], f"Plaintext credentials found: {bad}"
