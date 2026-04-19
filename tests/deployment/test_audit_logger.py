"""
Tests for Week 57: Immutable Audit Log (S7-S11)

Covers:
- S7: AuditLogger — basic write, chain integrity, replay
- S8: verify_audit_log script — valid chain passes, tampered chain fails
- S9: OrderManager / RLRiskManager integration
- S10: concurrency (10 threads × 100 records), 1000-record chain
- S11: Observation hash helper
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from deployment.audit.audit_logger import AuditLogger, _GENESIS_HASH, _sha256

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logger(tmp_path: Path, fsync: bool = False) -> AuditLogger:
    log_file = str(tmp_path / "audit.jsonl")
    return AuditLogger(log_path=log_file, fsync=fsync)


def _read_records(log_path: str) -> list[dict]:
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _verify_chain(records: list[dict]) -> bool:
    prev = _GENESIS_HASH
    for rec in records:
        expected = _sha256(prev, rec["payload"])
        if rec["hash"] != expected:
            return False
        prev = rec["hash"]
    return True


# ---------------------------------------------------------------------------
# S7: AuditLogger basic tests
# ---------------------------------------------------------------------------


class TestAuditLoggerBasic:
    def test_creates_file(self, tmp_path):
        al = _make_logger(tmp_path)
        al.log_risk_event({"event": "test"})
        al.close()
        assert os.path.exists(str(tmp_path / "audit.jsonl"))

    def test_record_has_required_fields(self, tmp_path):
        al = _make_logger(tmp_path)
        al.log_risk_event({"event": "drawdown_breach", "value": 0.1})
        al.close()
        records = _read_records(str(tmp_path / "audit.jsonl"))
        assert len(records) == 1
        rec = records[0]
        assert "ts" in rec
        assert "type" in rec
        assert "payload" in rec
        assert "hash" in rec
        assert rec["type"] == "risk_event"

    def test_single_record_chain(self, tmp_path):
        al = _make_logger(tmp_path)
        al.log_risk_event({"event": "test"})
        al.close()
        records = _read_records(str(tmp_path / "audit.jsonl"))
        assert _verify_chain(records)

    def test_multi_record_chain(self, tmp_path):
        al = _make_logger(tmp_path)
        for i in range(20):
            al.log_risk_event({"event": "test", "seq": i})
        al.close()
        records = _read_records(str(tmp_path / "audit.jsonl"))
        assert len(records) == 20
        assert _verify_chain(records)

    def test_log_order(self, tmp_path):
        from deployment.execution.order_manager import Order

        al = _make_logger(tmp_path)
        order = Order(
            order_id="abc123",
            side="buy",
            amount=0.01,
            order_type="market",
            limit_price=None,
            status="pending",
        )
        al.log_order(order)
        al.close()
        records = _read_records(str(tmp_path / "audit.jsonl"))
        assert records[0]["type"] == "order"
        assert records[0]["payload"]["order_id"] == "abc123"

    def test_log_fill(self, tmp_path):
        al = _make_logger(tmp_path)
        al.log_fill({"order_id": "x1", "side": "sell", "amount": 0.5, "price": 100.0})
        al.close()
        records = _read_records(str(tmp_path / "audit.jsonl"))
        assert records[0]["type"] == "fill"

    def test_log_model_decision(self, tmp_path):
        al = _make_logger(tmp_path)
        obs = np.random.rand(10).astype(np.float32)
        obs_hash = hashlib.sha256(obs.tobytes()).hexdigest()
        al.log_model_decision(action=1, obs_hash=obs_hash)
        al.close()
        records = _read_records(str(tmp_path / "audit.jsonl"))
        assert records[0]["type"] == "model_decision"
        assert records[0]["payload"]["obs_hash"] == obs_hash
        assert records[0]["payload"]["action"] == 1

    def test_replay_chain_on_reopen(self, tmp_path):
        """After reopening an existing log, new records chain from the last hash."""
        log_file = str(tmp_path / "audit.jsonl")
        al = AuditLogger(log_path=log_file)
        al.log_risk_event({"event": "first"})
        al.close()

        al2 = AuditLogger(log_path=log_file)
        al2.log_risk_event({"event": "second"})
        al2.close()

        records = _read_records(log_file)
        assert len(records) == 2
        assert _verify_chain(records)

    def test_context_manager(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        with AuditLogger(log_file) as al:
            al.log_risk_event({"event": "ctx"})
        records = _read_records(log_file)
        assert len(records) == 1


# ---------------------------------------------------------------------------
# S8: verify_audit_log script
# ---------------------------------------------------------------------------


class TestVerifyAuditLogScript:
    _script = str(
        Path(__file__).parent.parent.parent / "scripts" / "verify_audit_log.py"
    )

    def test_valid_chain_exits_zero(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        with AuditLogger(log_file) as al:
            for i in range(10):
                al.log_risk_event({"seq": i})

        result = subprocess.run(
            [sys.executable, self._script, log_file], capture_output=True
        )
        assert result.returncode == 0, result.stderr.decode()

    def test_tampered_chain_exits_one(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        with AuditLogger(log_file) as al:
            for i in range(5):
                al.log_risk_event({"seq": i})

        # Tamper: overwrite second record's payload
        records = _read_records(log_file)
        records[1]["payload"]["tampered"] = True
        with open(log_file, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        result = subprocess.run(
            [sys.executable, self._script, log_file], capture_output=True
        )
        assert result.returncode == 1

    def test_empty_file_exits_zero(self, tmp_path):
        log_file = str(tmp_path / "empty.jsonl")
        Path(log_file).touch()
        result = subprocess.run(
            [sys.executable, self._script, log_file], capture_output=True
        )
        assert result.returncode == 0

    def test_missing_file_exits_one(self, tmp_path):
        result = subprocess.run(
            [sys.executable, self._script, str(tmp_path / "nofile.jsonl")],
            capture_output=True,
        )
        assert result.returncode == 1


# ---------------------------------------------------------------------------
# S10: 1000-record chain + concurrency
# ---------------------------------------------------------------------------


class TestAuditLoggerScale:
    def test_1000_records_chain_valid(self, tmp_path):
        log_file = str(tmp_path / "big.jsonl")
        with AuditLogger(log_file) as al:
            for i in range(1000):
                al.log_risk_event({"event": "bulk", "seq": i})
        records = _read_records(log_file)
        assert len(records) == 1000
        assert _verify_chain(records)

    def test_concurrent_writes_chain_valid(self, tmp_path):
        """10 threads × 100 records = 1000 total; chain must be intact."""
        log_file = str(tmp_path / "concurrent.jsonl")
        al = AuditLogger(log_file)

        errors: list[Exception] = []

        def writer(thread_id: int) -> None:
            try:
                for i in range(100):
                    al.log_risk_event({"thread": thread_id, "seq": i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        al.close()

        assert not errors, f"Thread errors: {errors}"
        records = _read_records(log_file)
        assert len(records) == 1000
        assert _verify_chain(records), "Chain broken after concurrent writes"


# ---------------------------------------------------------------------------
# S9: OrderManager integration
# ---------------------------------------------------------------------------


class TestOrderManagerAuditIntegration:
    def test_order_and_fill_logged(self, tmp_path):
        from deployment.execution.order_manager import OrderManager

        log_file = str(tmp_path / "order_audit.jsonl")
        al = AuditLogger(log_file)
        mgr = OrderManager(paper_mode=True, audit_logger=al)
        mgr.submit_order("buy", amount=0.01, current_price=100.0)
        al.close()

        records = _read_records(log_file)
        types = [r["type"] for r in records]
        assert "order" in types
        assert "fill" in types

    def test_chain_valid_after_orders(self, tmp_path):
        from deployment.execution.order_manager import OrderManager

        log_file = str(tmp_path / "chain.jsonl")
        al = AuditLogger(log_file)
        mgr = OrderManager(paper_mode=True, audit_logger=al)
        for _ in range(10):
            mgr.submit_order("buy", amount=0.001, current_price=50.0)
        al.close()

        records = _read_records(log_file)
        assert _verify_chain(records)

    def test_no_audit_logger_no_crash(self, tmp_path):
        """OrderManager without audit_logger must work normally."""
        from deployment.execution.order_manager import OrderManager

        mgr = OrderManager(paper_mode=True)
        order_id = mgr.submit_order("buy", amount=0.01, current_price=100.0)
        assert mgr.check_order(order_id) == "filled"


# ---------------------------------------------------------------------------
# S9: RLRiskManager integration
# ---------------------------------------------------------------------------


class TestRLRiskManagerAuditIntegration:
    def _make_risk_manager(self, audit_logger=None):
        from risk_management.rl_risk_manager import RLRiskConfig, RLRiskManager

        cfg = RLRiskConfig(
            use_stop_loss=True,
            stop_loss_threshold=0.05,
            use_trailing_stop=True,
            trailing_stop_buffer=0.05,
            max_drawdown_pct=0.10,
        )
        return RLRiskManager(config=cfg, audit_logger=audit_logger)

    def test_stop_loss_logged(self, tmp_path):
        log_file = str(tmp_path / "risk_audit.jsonl")
        al = AuditLogger(log_file)
        rm = self._make_risk_manager(audit_logger=al)
        # 10% drop → stop_loss_threshold=0.05 → triggers
        rm.check_stop_loss("agent_0", position_size=1.0, entry_price=100.0, current_price=89.0)
        al.close()
        records = _read_records(log_file)
        assert any(r["payload"].get("event") == "stop_loss" for r in records)

    def test_trailing_stop_logged(self, tmp_path):
        log_file = str(tmp_path / "trail_audit.jsonl")
        al = AuditLogger(log_file)
        rm = self._make_risk_manager(audit_logger=al)
        rm.position_highest_values["agent_0_BTC"] = 100.0
        rm.check_trailing_stop("agent_0", asset="BTC", position_size=1.0, current_price=93.0)
        al.close()
        records = _read_records(log_file)
        assert any(r["payload"].get("event") == "trailing_stop" for r in records)

    def test_drawdown_breach_logged(self, tmp_path):
        log_file = str(tmp_path / "dd_audit.jsonl")
        al = AuditLogger(log_file)
        rm = self._make_risk_manager(audit_logger=al)
        # peak=100, current=85 → 15% drawdown > 10% threshold
        rm.check_drawdown(100.0, 85.0)
        al.close()
        records = _read_records(log_file)
        assert any(r["payload"].get("event") == "drawdown_breach" for r in records)

    def test_no_audit_logger_no_crash(self):
        rm = self._make_risk_manager(audit_logger=None)
        # should not raise
        triggered = rm.check_stop_loss("a", 1.0, 100.0, 85.0)
        assert triggered


# ---------------------------------------------------------------------------
# S11: Observation hash
# ---------------------------------------------------------------------------


class TestObservationHash:
    def test_obs_hash_is_sha256(self, tmp_path):
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        expected = hashlib.sha256(obs.tobytes()).hexdigest()
        assert len(expected) == 64
        # Verify it is deterministic
        assert hashlib.sha256(obs.tobytes()).hexdigest() == expected

    def test_model_decision_stores_obs_hash(self, tmp_path):
        log_file = str(tmp_path / "model.jsonl")
        obs = np.random.rand(20).astype(np.float32)
        obs_hash = hashlib.sha256(obs.tobytes()).hexdigest()

        with AuditLogger(log_file) as al:
            al.log_model_decision(action=2, obs_hash=obs_hash)

        records = _read_records(log_file)
        assert records[0]["payload"]["obs_hash"] == obs_hash
        # Full obs is NOT stored
        assert "observation" not in records[0]["payload"]

    def test_obs_hash_does_not_store_full_obs(self, tmp_path):
        """Make sure we only store the hash, not the raw observation data."""
        log_file = str(tmp_path / "model2.jsonl")
        obs = np.ones(1000, dtype=np.float64)
        obs_hash = hashlib.sha256(obs.tobytes()).hexdigest()

        with AuditLogger(log_file) as al:
            al.log_model_decision(action=0, obs_hash=obs_hash)

        size = os.path.getsize(log_file)
        # Full obs would be ~8000 bytes; log should be tiny
        assert size < 500, f"Log file unexpectedly large ({size} bytes) — obs may have been stored"


# ---------------------------------------------------------------------------
# F6: Credential redaction (Week 72)
# ---------------------------------------------------------------------------

class TestCredentialRedaction:
    def test_api_key_is_redacted(self, tmp_path):
        log_file = str(tmp_path / "redact.jsonl")
        with AuditLogger(log_file) as al:
            al.log_risk_event({"event": "test", "api_key": "super_secret_key"})

        records = _read_records(log_file)
        payload = records[0]["payload"]
        assert payload["api_key"] == "***REDACTED***"
        assert "super_secret_key" not in json.dumps(payload)

    def test_api_secret_is_redacted(self, tmp_path):
        log_file = str(tmp_path / "redact2.jsonl")
        with AuditLogger(log_file) as al:
            al.log_risk_event({"event": "test", "api_secret": "my_secret_123"})

        records = _read_records(log_file)
        assert records[0]["payload"]["api_secret"] == "***REDACTED***"

    def test_nested_credential_is_redacted(self, tmp_path):
        log_file = str(tmp_path / "redact3.jsonl")
        with AuditLogger(log_file) as al:
            al.log_risk_event({
                "event": "exchange_init",
                "config": {"api_key": "nested_key", "symbol": "BTC/USDT"},
            })

        records = _read_records(log_file)
        cfg = records[0]["payload"]["config"]
        assert cfg["api_key"] == "***REDACTED***"
        assert cfg["symbol"] == "BTC/USDT"  # non-sensitive key preserved

    def test_non_credential_fields_preserved(self, tmp_path):
        log_file = str(tmp_path / "redact4.jsonl")
        with AuditLogger(log_file) as al:
            al.log_risk_event({"event": "halt", "reason": "drawdown", "value": 0.25})

        records = _read_records(log_file)
        payload = records[0]["payload"]
        assert payload["reason"] == "drawdown"
        assert payload["value"] == 0.25

    def test_redaction_applied_before_hash(self, tmp_path):
        """Hash must be computed on the redacted payload (chain integrity)."""
        log_file = str(tmp_path / "redact5.jsonl")
        with AuditLogger(log_file) as al:
            al.log_risk_event({"api_key": "secret"})

        records = _read_records(log_file)
        payload = records[0]["payload"]
        stored_hash = records[0]["hash"]
        # Recompute expected hash over the redacted payload
        expected = _sha256(_GENESIS_HASH, payload)
        assert stored_hash == expected

    def test_chain_remains_valid_after_redaction(self, tmp_path):
        """Full chain integrity check after redaction."""
        log_file = str(tmp_path / "redact6.jsonl")
        with AuditLogger(log_file) as al:
            al.log_risk_event({"step": 1, "api_key": "secret1"})
            al.log_risk_event({"step": 2, "api_secret": "secret2"})
            al.log_risk_event({"step": 3, "event": "normal"})

        records = _read_records(log_file)
        assert _verify_chain(records) is True
