# Launchd Auto-Restart for Autonomous 72h Drill

Registers `autonomous_72h_drill.py` as a launchd agent so it survives
crashes and reboots during a 72h run.

## Prerequisites

1. The drill script exists at `~/Desktop/trading_bot/scripts/autonomous_72h_drill.py`
2. Python is at `/Users/skylar/anaconda3/bin/python`
3. `logs/` directory is writable

If your paths differ, edit `com.tradingbot.drill.plist` before registering.

---

## Register (one-time)

```bash
# Copy plist to LaunchAgents
cp scripts/launchd/com.tradingbot.drill.plist ~/Library/LaunchAgents/

# Load agent (does NOT start it yet — RunAtLoad is false)
launchctl load ~/Library/LaunchAgents/com.tradingbot.drill.plist
```

## Start the drill

```bash
launchctl start com.tradingbot.drill
```

The drill will restart automatically within 5 seconds if it crashes.

## Stop the drill

```bash
launchctl stop com.tradingbot.drill
```

This sends SIGTERM; `autonomous_72h_drill.py` handles it gracefully.

## Unregister (after drill completes)

```bash
launchctl unload ~/Library/LaunchAgents/com.tradingbot.drill.plist
rm ~/Library/LaunchAgents/com.tradingbot.drill.plist
```

## Check status

```bash
launchctl list | grep tradingbot
# Output: PID  exit_code  com.tradingbot.drill
# PID > 0 means running; exit_code 0 means last run clean
```

## Logs

| File | Contents |
|------|----------|
| `logs/drill_stdout.log` | Drill INFO/WARNING messages |
| `logs/drill_stderr.log` | Python errors / tracebacks |
| `logs/drill_snapshots.jsonl` | 15-min portfolio snapshots |
| `logs/fault_injection.jsonl` | Fault injector events |
| `logs/alerts.jsonl` | All alert events |

## Restart limit

The plist sets `ThrottleInterval=5s`. launchd enforces a 10-crash/hour
limit internally. If the drill crashes more than 10 times in an hour,
launchd will throttle and wait ~1 min before retrying.
