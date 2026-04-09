"""Secrets management for trading bot deployment."""
from .secret_provider import (
    SecretProvider,
    EnvSecretProvider,
    FileSecretProvider,
    get_default_provider,
)

try:
    from .secret_provider import KeychainSecretProvider
except ImportError:
    pass  # keyring not installed

__all__ = [
    "SecretProvider",
    "EnvSecretProvider",
    "FileSecretProvider",
    "get_default_provider",
]
