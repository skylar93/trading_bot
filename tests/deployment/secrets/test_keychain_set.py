"""I5-b: KeychainSecretProvider.set() and .delete() subprocess argument tests."""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, call, patch

import pytest


@pytest.fixture()
def provider():
    with patch.dict("sys.modules", {"keyring": MagicMock()}):
        from deployment.secrets.secret_provider import KeychainSecretProvider
        return KeychainSecretProvider()


class TestKeychainSet:
    def test_set_calls_security_add_generic_password(self, provider):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            provider.set("MY_KEY", "my_secret_value")

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "security"
        assert "add-generic-password" in args
        assert "-s" in args
        assert "trading_bot" in args
        assert "-a" in args
        assert "MY_KEY" in args
        assert "-w" in args
        assert "my_secret_value" in args
        assert "-U" in args  # update flag

    def test_set_uses_check_true(self, provider):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            provider.set("K", "V")

        kwargs = mock_run.call_args[1]
        assert kwargs.get("check") is True

    def test_set_captures_output(self, provider):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            provider.set("K", "V")

        kwargs = mock_run.call_args[1]
        assert kwargs.get("capture_output") is True


class TestKeychainDelete:
    def test_delete_calls_security_delete_generic_password(self, provider):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            provider.delete("MY_KEY")

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "security"
        assert "delete-generic-password" in args
        assert "-s" in args
        assert "trading_bot" in args
        assert "-a" in args
        assert "MY_KEY" in args

    def test_delete_silently_ignores_item_not_found(self, provider):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=44)  # item not found
            # Should not raise
            provider.delete("NONEXISTENT_KEY")

    def test_delete_silently_ignores_rc_zero(self, provider):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            provider.delete("EXISTING_KEY")  # no exception
