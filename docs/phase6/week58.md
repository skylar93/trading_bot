# Week 58 Retro — Secrets Management (S12-S15)

**Date**: 2026-04-09
**Branch**: claude/quizzical-faraday
**Sections**: S12-S15

---

## What was done

### S12 — SecretProvider interface + 3 implementations
- `deployment/secrets/secret_provider.py`
  - Abstract base `SecretProvider` with `get(key) -> str` and `get_optional(key, default)`
  - `EnvSecretProvider`: reads `os.environ[key]` (production default)
  - `KeychainSecretProvider`: delegates to macOS `keyring` library (optional import)
  - `FileSecretProvider`: reads `~/.trading_bot/secrets.json`, caches on first read, `invalidate_cache()` available
  - `get_default_provider()`: picks backend based on `TRADING_BOT_SECRET_BACKEND` env var (`env` | `keychain` | `file`)

### S12b — Config resolver
- `deployment/secrets/config_resolver.py`
  - `resolve_secrets(config, provider)` deep-copies the config dict and replaces every `*_ref: "KEY_NAME"` leaf with `provider.get("KEY_NAME")`
  - Convention: `api_key_ref` → resolves to `api_key`, `api_secret_ref` → `api_secret`
  - Empty `_ref` values are left untouched (simulation / no-credential mode)

### S13 — Config migration
- `config/paper_trading.yaml`: replaced `api_key: ""` / `api_secret: ""` with `api_key_ref: "EXCHANGE_BINANCE_KEY"` / `api_secret_ref: "EXCHANGE_BINANCE_SECRET"`
- `config/local_3060ti.yaml`: added `secrets: { backend: "env" }` block

### S14 — .gitignore + pre-commit hook
- `.gitignore` additions: `secrets.json`, `state/`, `*.db`, `audit_log/`, `audit_logs/`
- `.git/hooks/pre-commit`: scans staged files for `api_(key|secret)\s*[:=]\s*['"][^'"]{1,}['"]` pattern; rejects commit if found

### S15 — Tests
- `tests/deployment/test_secrets.py`: 31 tests covering all providers, resolver, and config audit

---

## Test results

```
1448 passed, 19 skipped, 0 failed
```

Baseline was 1386 (Phase 5). No regressions introduced.

---

## Gotchas

1. **Module reload breaks isinstance**: The `KeychainSecretProvider` tests initially used `importlib.reload()` to swap in a mock `keyring`. This invalidated the original class references, causing `isinstance` checks in `TestGetDefaultProvider` to fail (object is an instance of the *reloaded* class, not the *imported* class). Fixed by using `patch.dict("sys.modules", {"keyring": mock_keyring})` directly without reload — the `__init__` re-imports `keyring` at construction time so the mock is picked up.

2. **Worktree `.git` is a file, not a dir**: `worktrees/<name>/.git` is a pointer file. The hook must go in the main repo's `.git/hooks/`, which all worktrees share.

3. **Empty `_ref` as simulation mode**: When `api_key_ref: ""` the resolver intentionally skips resolution (treats it as "no credential needed"). This allows YAML-level opt-out without removing the key.

---

## What was NOT done (out of scope)

- Wiring `resolve_secrets()` into the live config loading path (done at call sites as needed in later weeks)
- Cloud secret backends (AWS Secrets Manager, Vault) — Phase 7 candidate
- Rotating / expiring secrets — Phase 7 candidate
