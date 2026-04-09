"""
Secret providers for trading bot credential management.

Three implementations:
  - EnvSecretProvider   : reads from os.environ (recommended for production)
  - KeychainSecretProvider: reads from macOS Keychain via the `keyring` library
  - FileSecretProvider  : reads from ~/.trading_bot/secrets.json (dev only, gitignored)

Usage
-----
  provider = EnvSecretProvider()
  api_key = provider.get("EXCHANGE_BINANCE_KEY")

Config integration
------------------
Config files should use:
    secret_ref: "EXCHANGE_BINANCE_KEY"
instead of:
    api_key: "actual-key-here"

The config loader (deployment/secrets/config_resolver.py) resolves secret_refs
at load time via the active provider.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_SECRETS_FILE = Path.home() / ".trading_bot" / "secrets.json"


class SecretProvider(ABC):
    """Abstract base class for secret retrieval."""

    @abstractmethod
    def get(self, key: str) -> str:
        """Return the secret value for *key*.

        Raises
        ------
        KeyError
            If the key does not exist in this provider.
        """

    def get_optional(self, key: str, default: str = "") -> str:
        """Return the secret value for *key*, or *default* if not found."""
        try:
            return self.get(key)
        except KeyError:
            return default


class EnvSecretProvider(SecretProvider):
    """Reads secrets from environment variables.

    This is the recommended provider for production / CI.  Set e.g.::

        export EXCHANGE_BINANCE_KEY=abc123
        export EXCHANGE_BINANCE_SECRET=xyz789
    """

    def get(self, key: str) -> str:
        try:
            return os.environ[key]
        except KeyError:
            raise KeyError(
                f"Secret '{key}' not found in environment variables. "
                f"Set the env var before running the bot."
            )


class KeychainSecretProvider(SecretProvider):
    """Reads secrets from the macOS Keychain via the `keyring` library.

    Requires: ``pip install keyring``

    Secrets are stored under service name ``"trading_bot"``.
    Store a secret via::

        python -c "import keyring; keyring.set_password('trading_bot', 'EXCHANGE_BINANCE_KEY', 'abc123')"
    """

    _SERVICE = "trading_bot"

    def __init__(self) -> None:
        try:
            import keyring  # noqa: F401 — validate at init time
            self._keyring = keyring
        except ImportError as exc:
            raise ImportError(
                "KeychainSecretProvider requires the 'keyring' package: "
                "pip install keyring"
            ) from exc

    def get(self, key: str) -> str:
        value = self._keyring.get_password(self._SERVICE, key)
        if value is None:
            raise KeyError(
                f"Secret '{key}' not found in keychain service '{self._SERVICE}'. "
                f"Add it with: "
                f"python -c \"import keyring; keyring.set_password('{self._SERVICE}', '{key}', '<value>')\""
            )
        return value


class FileSecretProvider(SecretProvider):
    """Reads secrets from a JSON file on disk.

    Default location: ``~/.trading_bot/secrets.json``

    This file MUST be gitignored and kept out of version control.
    It is intended for local development only.

    File format::

        {
            "EXCHANGE_BINANCE_KEY": "abc123",
            "EXCHANGE_BINANCE_SECRET": "xyz789"
        }
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else _DEFAULT_SECRETS_FILE
        self._cache: dict[str, str] | None = None

    def _load(self) -> dict[str, str]:
        if self._cache is not None:
            return self._cache
        if not self._path.exists():
            raise FileNotFoundError(
                f"Secrets file not found: {self._path}. "
                f"Create it with the required key/value pairs."
            )
        with open(self._path) as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"Secrets file must be a JSON object: {self._path}")
        self._cache = {str(k): str(v) for k, v in data.items()}
        return self._cache

    def get(self, key: str) -> str:
        secrets = self._load()
        if key not in secrets:
            raise KeyError(
                f"Secret '{key}' not found in {self._path}. "
                f"Add it to the secrets file."
            )
        return secrets[key]

    def invalidate_cache(self) -> None:
        """Force re-read of the secrets file on next access."""
        self._cache = None


def get_default_provider() -> SecretProvider:
    """Return the most appropriate provider for the current environment.

    Priority:
    1. ``TRADING_BOT_SECRET_BACKEND`` env var: ``"env"`` | ``"keychain"`` | ``"file"``
    2. If unset, falls back to ``EnvSecretProvider``.
    """
    backend = os.environ.get("TRADING_BOT_SECRET_BACKEND", "env").lower()
    if backend == "env":
        return EnvSecretProvider()
    if backend == "keychain":
        return KeychainSecretProvider()
    if backend == "file":
        return FileSecretProvider()
    logger.warning(
        "Unknown TRADING_BOT_SECRET_BACKEND=%r, falling back to EnvSecretProvider",
        backend,
    )
    return EnvSecretProvider()
