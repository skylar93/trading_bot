# Kill Switch Drill — 2026-04-23

**Environment**: Paper (simulation_mode=True)
**Executed by**: skylar (solo dev)
**Script**: `scripts/kill_switch.py` / `PaperTrader._trigger_shutdown()`

## Scenario

Fire the kill switch against a running PaperTrader instance and verify
that shutdown completes within 5 seconds with all state cleanly recorded.

## Steps Performed

1. Started PaperTrader in simulation mode (initial_balance=$100, window_size=5)
2. Waited 0.5s for the run loop to begin processing prices
3. Called `trader._trigger_shutdown("kill switch drill")` and measured wall-clock time
4. Verified `trader.state.shutdown_triggered == True`
5. Verified `trader.state.shutdown_reason == "kill switch drill"`
6. Verified PID file cleaned up (state/drill_kill_test2.pid)

## Results

| Metric | Expected | Actual | Pass |
|--------|----------|--------|------|
| Shutdown elapsed | < 5.0 s | **0.003 s** | ✅ |
| `shutdown_triggered` | True | True | ✅ |
| shutdown_reason | set | "kill switch drill" | ✅ |
| PID file cleaned | absent | absent | ✅ |

## Timing Breakdown

- `_trigger_shutdown()` call → event set: ~0.001 s
- Run-loop detects event + exits: ~0.002 s
- Total wall-clock: **0.003 s**

## Issues Found

None.

## Notes

Shutdown is event-driven (`threading.Event.set()`), so response time is
sub-millisecond in simulation mode.  On live exchange the bottleneck would be
the cancel-all-orders network round-trip (~80 ms per the capacity baseline).
Even in that case the 5s SLA has ample margin.

---

*Next drill recommended: after any change to `PaperTrader.run()` or `OrderManager.submit_order()`*
