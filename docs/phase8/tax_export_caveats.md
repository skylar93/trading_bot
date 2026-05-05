# Tax Export Caveats

**Script**: `scripts/export_tax_report.py`
**Format**: `audit_log/audit.jsonl` → CSV (korea | us-1099b | generic)

## Known gaps between audit log fields and tax form requirements

### Symbol not recorded per fill

The `Order` dataclass (and therefore every fill record in the audit log) does **not** store the trading symbol (e.g., `BTC/USDT`). The symbol is part of the `OrderManager` config, not the fill payload. As a result, the `symbol` column in `korea` format is always blank. Before filing, the operator must add the symbol manually or cross-reference `exchange_order_id` against the exchange's own trade history export.

### FX rate — no live source

The `korea` format requires KRW-denominated proceeds and cost basis. The script converts using rates from `config/fx_rates.yaml` (see example below). If that file is absent or a date-specific rate is missing, the script falls back to 1.0 and prints a `WARNING` to stderr. **This will produce incorrect KRW figures.** The operator must populate `fx_rates.yaml` with daily USD/KRW rates from a reliable source (e.g., Bank of Korea API, or manually from the exchange at fill time) before generating a final report.

Example `config/fx_rates.yaml` format:
```yaml
USD_KRW: 1350.0                   # fallback rate used when no dated rate exists
USD_KRW_2026-03-01: 1348.5        # date-specific rate takes priority
USD_KRW_2026-06-15: 1362.0
```

### Fee not included in FIFO-matched rows

Per-fill fee data exists in the raw audit payload but is not forwarded to individual FIFO-matched rows (the matched row represents one lot consumed from a buy, not the original fill event). The `fee_krw` column is therefore always `0.0` in `korea` format. For accurate tax reporting, sum fees separately using the `generic` format export and reconcile manually.

### FIFO assumption

This script uses FIFO (first-in, first-out) lot identification. South Korean tax guidance (소득세법 시행령) allows specific-lot identification (개별법), which can yield a different gain/loss figure. This script does not support specific-lot matching. Consult a tax accountant to confirm the acceptable method for your situation.

### Partial-year opens

Buys made before the export year but matched to sells within the year will show `cost_basis_usd=NaN` and trigger a `WARNING`, because the matching lot is outside the audit log scope for that year. Run `--year` over the full holding period or export multiple years sequentially with accumulated lot state (not currently supported).

---

**The operator must validate this output with a qualified tax accountant before filing.** This script is an aid for record-keeping, not a certified tax document generator.
