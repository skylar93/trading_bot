# Week 81: Idempotency Concurrency Analysis

**Date**: 2026-04-22  
**Test**: `tests/deployment/test_live_risk_enforcement.py::TestIdempotencyKey::test_concurrent_duplicate_keys`

## Summary

The idempotency test was historically marked as flaky (passes in isolation, occasional race in concurrent runs). After 100-run stress test with `pytest-repeat`, 0 failures were recorded.

**Conclusion**: No fix required. The existing implementation is correct.

## Implementation

`deployment/execution/order_manager.py:317-326` uses `dict.setdefault` inside `self._lock`:

```python
with self._lock:
    registered_id = self._idempotency_map.setdefault(
        idempotency_key, _pre_order_id
    )
    if registered_id != _pre_order_id:
        return registered_id  # duplicate — return original ID
```

The lock wraps the entire `setdefault + branch` block, making it fully atomic. Ten threads racing with the same key always resolve to a single order ID.

## Stress Test Results

```
Command: pytest --count=100 tests/deployment/test_live_risk_enforcement.py::TestIdempotencyKey::test_concurrent_duplicate_keys
Result: 100 passed in 0.65s
Failures: 0 / 100
```

Test scenario: 10 threads simultaneously call `submit_order` with identical `idempotency_key`. All 10 threads must return the same `order_id`.

## Decision

- No threading.Lock change needed (lock already present).
- The prior "flaky" observation was likely a test environment artifact.
- `pytest-repeat>=0.9.3` added to `requirements.txt` for future stress runs.
- No flaky marker exists in `pytest.ini` — nothing to remove.
