# Reward Market Scanner

Shadow/backtest-only scanner foundation for selecting Polymarket markets for later PMBT replay batches and Homerun shadow-forward logging.

Safety invariants:

- no live trading
- no signing, submitting, cancelling, or placing orders
- no wallet private keys, Polymarket L2 credentials, GitHub PATs, or secrets required
- public/read-only scan artifacts only

The scanner consumes the latest autonomous market-scan JSON under `/opt/polymarket-lab/data/market_scans/` and writes timestamped manifests under `/opt/polymarket-lab/autoresearch/reward_scanner/manifests/`.

Scored features include:

- Yes/No CLOB spreads
- two-sided top/book depth, including `depth_2c` when present
- volume, 24h volume, and liquidity
- market age and time to end
- strategy-fit volatility proxy
- explicit reward evidence (`clobRewards`, `rewardsMinSize`, `rewardsMaxSpread`, `umaReward`) vs public proxy-only labels
- accidental-fill risk flags for tail-price, wide-spread, thin-depth, and one-sided-depth markets
- fail-closed token mapping checks: exactly two unique Yes/No CLOB token IDs are required for backtest queue eligibility
- manifest provenance (`input_scan_path`, optional strategy/report path, and `source_paths`) plus summary counts and stable candidate ranks

Run locally in the PMBT worktree:

```bash
./.venv/bin/python scripts/build_reward_market_manifest.py --limit 20
./.venv/bin/python -m pytest tests/test_reward_market_scanner.py
```

A manifest is a research queue, not a go-live approval. Live promotion remains blocked until separate user approval plus geoblock, secrets, execution-adapter, risk-limit, and dry-run evidence gates.

## Reward Maker Shadow Lifecycle V2

`scripts/low_fill_reward_shadow_lifecycle.py` emits shadow-only virtual maker
quote evidence for low-fill reward research. It defines post-only virtual quotes
from manifest top-of-book snapshots, measures reward-band scheduled time,
classifies would-fill evidence as `conservative`, `optimistic`, or `unknown`,
and writes fail-closed EV readiness scenarios.

The writer produces:

- `low_fill_reward_shadow_quote_log_*.jsonl` with one virtual quote record per
  snapshot side, including book touch, reward terms, queue-ahead proxy,
  would-fill classification, immediate exit loss proxy, 1m/5m/15m markout
  placeholders, and explicit shadow safety fields.
- `low_fill_reward_would_fill_*.json/.md` summarizing conservative,
  optimistic, and unknown fill evidence.
- `low_fill_reward_exit_risk_*.json/.md` summarizing immediate exit-loss and
  missing markout evidence.
- `low_fill_reward_ev_readiness_*.json/.md` with base/worst/best cases.

EV readiness fails closed unless reward amount, reward-score denominator share,
time in band, conservative would-fill evidence, and exit-loss evidence are all
available. The reports do not make a profitability claim from shadow diagnostics
alone.
