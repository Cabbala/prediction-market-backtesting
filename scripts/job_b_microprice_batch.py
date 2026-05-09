from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
from dataclasses import dataclass, asdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    from _script_helpers import ensure_repo_root
else:
    from ._script_helpers import ensure_repo_root

ensure_repo_root(__file__)

from prediction_market_extensions.backtesting._execution_config import (  # noqa: E402
    ExecutionModelConfig,
    StaticLatencyConfig,
)
from prediction_market_extensions.backtesting._prediction_market_runner import (  # noqa: E402
    MarketDataConfig,
    run_single_market_backtest,
)
from prediction_market_extensions.backtesting.data_sources import Book, PMXT, Polymarket  # noqa: E402
from strategies import BookMicropriceImbalanceConfig, BookMicropriceImbalanceStrategy  # noqa: E402

SAFETY_MODE = "backtest_only_no_live_trading"
DEFAULT_SOURCES = (
    "local:/opt/polymarket-lab/data/pmxt/raw",
    "archive:r2v2.pmxt.dev",
    "archive:r2.pmxt.dev",
)


@dataclass(frozen=True)
class Candidate:
    slug: str
    question: str
    token_index: int
    condition_id: str | None
    scan_mid: float | None
    scan_spread: float | None
    scan_imbalance5: float | None
    liquidity: float | None
    source_strategy: str


@dataclass(frozen=True)
class BacktestAttempt:
    slug: str
    question: str
    token_index: int
    source_strategy: str
    params: dict[str, Any]
    status: str
    result: dict[str, Any] | None
    error: str | None


def _utc_now() -> datetime:
    return datetime.now(tz=UTC).replace(microsecond=0)


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _candidate_slug(raw: dict[str, Any]) -> str | None:
    value = raw.get("market_slug") or raw.get("slug")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _strategy_key(value: Any) -> str:
    if value is None:
        return ""
    key = "".join(ch for ch in str(value).lower() if ch.isalnum())
    aliases = {
        "micropriceoptimizer": "microprice",
        "micropriceorderbookimbalance": "microprice",
        "bookmicropriceimbalance": "microprice",
    }
    return aliases.get(key, key)


def _strategy_matches(requested: str | None, actual: str | None) -> bool:
    requested_key = _strategy_key(requested)
    if not requested_key:
        return True
    actual_key = _strategy_key(actual)
    return requested_key == actual_key


def _normalize_candidate(raw: dict[str, Any], *, source_strategy: str) -> Candidate | None:
    slug = _candidate_slug(raw)
    if slug is None:
        return None
    token_index = raw.get("token_index", 0)
    try:
        token_index = int(token_index)
    except (TypeError, ValueError):
        token_index = 0
    return Candidate(
        slug=slug,
        question=str(raw.get("question") or slug),
        token_index=token_index,
        condition_id=raw.get("condition_id") if isinstance(raw.get("condition_id"), str) else None,
        scan_mid=_parse_float(raw.get("scan_mid") or raw.get("yes_probability")),
        scan_spread=_parse_float(raw.get("scan_spread") or raw.get("avg_spread")),
        scan_imbalance5=_parse_float(raw.get("scan_imbalance5")),
        liquidity=_parse_float(raw.get("scan_liquidity") or raw.get("liquidity")),
        source_strategy=source_strategy,
    )


def load_candidates(manifest_path: Path, *, strategy: str, max_candidates: int) -> list[Candidate]:
    payload = json.loads(manifest_path.read_text())
    candidates: list[Candidate] = []
    seen: set[tuple[str, int]] = set()

    if isinstance(payload, dict):
        batches = payload.get("batches")
        if isinstance(batches, list):
            for batch in batches:
                if not isinstance(batch, dict):
                    continue
                source_strategy = str(batch.get("strategy") or "unknown")
                if not _strategy_matches(strategy, source_strategy):
                    continue
                markets = batch.get("markets")
                if not isinstance(markets, list):
                    continue
                for raw in markets:
                    if not isinstance(raw, dict):
                        continue
                    cand = _normalize_candidate(raw, source_strategy=source_strategy)
                    if cand is None:
                        continue
                    key = (cand.slug, cand.token_index)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(cand)
                    if len(candidates) >= max_candidates:
                        return candidates
        raw_candidates = payload.get("candidates")
        if isinstance(raw_candidates, list) and not candidates:
            for raw in raw_candidates:
                if not isinstance(raw, dict):
                    continue
                source_strategy = str(
                    raw.get("source_strategy")
                    or raw.get("strategy")
                    or payload.get("strategy")
                    or "reward_manifest"
                )
                if not _strategy_matches(strategy, source_strategy):
                    continue
                coverage = raw.get("coverage")
                if isinstance(coverage, dict):
                    if coverage.get("status") != "pass":
                        continue
                    if int(coverage.get("gap_hours_missing") or 0) != 0:
                        continue
                    book_events = coverage.get("book_events")
                    min_book_events = coverage.get("min_book_events")
                    if book_events is not None and min_book_events is not None:
                        try:
                            if int(book_events) < int(min_book_events):
                                continue
                        except (TypeError, ValueError):
                            continue
                cand = _normalize_candidate(raw, source_strategy=source_strategy)
                if cand is None:
                    continue
                key = (cand.slug, cand.token_index)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(cand)
                if len(candidates) >= max_candidates:
                    return candidates
    elif isinstance(payload, list):
        for raw in payload:
            if not isinstance(raw, dict):
                continue
            cand = _normalize_candidate(raw, source_strategy="list_manifest")
            if cand is None:
                continue
            key = (cand.slug, cand.token_index)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(cand)
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def microprice_param_grid() -> list[dict[str, Any]]:
    return [
        {"depth_levels": 1, "entry_imbalance": 0.55, "exit_imbalance": 0.50, "min_microprice_edge": 0.0005, "quote_lifetime_seconds": 10.0},
        {"depth_levels": 3, "entry_imbalance": 0.57, "exit_imbalance": 0.50, "min_microprice_edge": 0.0010, "quote_lifetime_seconds": 30.0},
        {"depth_levels": 5, "entry_imbalance": 0.60, "exit_imbalance": 0.52, "min_microprice_edge": 0.0015, "quote_lifetime_seconds": 60.0},
        {"depth_levels": 3, "entry_imbalance": 0.62, "exit_imbalance": 0.54, "min_microprice_edge": 0.0020, "quote_lifetime_seconds": 30.0},
    ]


def _safe_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    safe: dict[str, Any] = {}
    for key, value in result.items():
        if key.lower() in {"private_key", "secret", "token", "api_key"}:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, list):
            safe[key] = value[:20]
        elif isinstance(value, dict):
            safe[key] = {str(k): v for k, v in list(value.items())[:50] if isinstance(v, (str, int, float, bool)) or v is None}
        else:
            safe[key] = repr(value)
    return safe


async def run_attempt(candidate: Candidate, params: dict[str, Any], *, start_time: str, end_time: str, min_book_events: int) -> BacktestAttempt:
    def strategy_factory(instrument_id):  # type: ignore[no-untyped-def]
        return BookMicropriceImbalanceStrategy(
            BookMicropriceImbalanceConfig(
                instrument_id=instrument_id,
                trade_size=Decimal("5"),
                depth_levels=int(params["depth_levels"]),
                entry_imbalance=float(params["entry_imbalance"]),
                exit_imbalance=float(params["exit_imbalance"]),
                min_microprice_edge=float(params["min_microprice_edge"]),
                max_spread=0.05,
                max_entry_price=0.95,
                max_expected_slippage=0.02,
                min_holding_updates=0,
                reentry_cooldown_updates=0,
                min_holding_seconds=float(params["quote_lifetime_seconds"]),
                reentry_cooldown_seconds=float(params["quote_lifetime_seconds"]),
                take_profit=0.01,
                stop_loss=0.015,
            )
        )

    try:
        result = await run_single_market_backtest(
            name="job_b_microprice_batch",
            data=MarketDataConfig(platform=Polymarket, data_type=Book, vendor=PMXT, sources=DEFAULT_SOURCES),
            market_slug=candidate.slug,
            token_index=candidate.token_index,
            start_time=start_time,
            end_time=end_time,
            min_book_events=min_book_events,
            min_price_range=0.0,
            probability_window=128,
            initial_cash=100.0,
            emit_summary=False,
            nautilus_log_level="ERROR",
            strategy_factory=strategy_factory,
            execution=ExecutionModelConfig(
                queue_position=True,
                latency_model=StaticLatencyConfig(
                    base_latency_ms=75.0,
                    insert_latency_ms=10.0,
                    update_latency_ms=5.0,
                    cancel_latency_ms=5.0,
                ),
            ),
        )
        return BacktestAttempt(
            slug=candidate.slug,
            question=candidate.question,
            token_index=candidate.token_index,
            source_strategy=candidate.source_strategy,
            params=params,
            status="skipped_no_coverage" if result is None else "completed",
            result=_safe_result(result),
            error=None,
        )
    except Exception as exc:  # bounded research runner: record and continue to next candidate
        return BacktestAttempt(
            slug=candidate.slug,
            question=candidate.question,
            token_index=candidate.token_index,
            source_strategy=candidate.source_strategy,
            params=params,
            status="error",
            result=None,
            error=f"{type(exc).__name__}: {exc}",
        )


async def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    candidates = load_candidates(args.manifest, strategy=args.strategy, max_candidates=args.max_candidates)
    selected_params = microprice_param_grid()[: args.max_param_sets]
    attempts: list[BacktestAttempt] = []
    for candidate in candidates:
        for params in selected_params:
            attempts.append(
                await asyncio.wait_for(
                    run_attempt(
                        candidate,
                        params,
                        start_time=args.start_time,
                        end_time=args.end_time,
                        min_book_events=args.min_book_events,
                    ),
                    timeout=args.per_attempt_timeout_secs,
                )
            )
    completed = sum(1 for attempt in attempts if attempt.status == "completed")
    skipped = sum(1 for attempt in attempts if attempt.status == "skipped_no_coverage")
    errored = sum(1 for attempt in attempts if attempt.status == "error")
    return {
        "generated_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
        "mode": SAFETY_MODE,
        "safety": {
            "live_trading": False,
            "orders_submitted": False,
            "credentials_required": False,
            "worker_trading_started": False,
        },
        "manifest": str(args.manifest),
        "window": {"start_time": args.start_time, "end_time": args.end_time},
        "candidate_count": len(candidates),
        "parameter_set_count": len(selected_params),
        "attempt_count": len(attempts),
        "completed_count": completed,
        "skipped_no_coverage_count": skipped,
        "error_count": errored,
        "candidates": [asdict(candidate) for candidate in candidates],
        "attempts": [asdict(attempt) for attempt in attempts],
    }


def _write_outputs(summary: dict[str, Any], output_dir: Path, timestamp: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"job_B_microprice_batch_{timestamp}.json"
    csv_path = output_dir / f"job_B_microprice_batch_{timestamp}.csv"
    md_path = output_dir / f"job_B_microprice_batch_{timestamp}.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    attempts = summary.get("attempts", [])
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["slug", "token_index", "source_strategy", "status", "params", "pnl", "fills", "quotes", "error"],
        )
        writer.writeheader()
        for attempt in attempts:
            result = attempt.get("result") or {}
            writer.writerow(
                {
                    "slug": attempt.get("slug"),
                    "token_index": attempt.get("token_index"),
                    "source_strategy": attempt.get("source_strategy"),
                    "status": attempt.get("status"),
                    "params": json.dumps(attempt.get("params", {}), sort_keys=True),
                    "pnl": result.get("pnl"),
                    "fills": result.get("fills"),
                    "quotes": result.get("quotes") or result.get("book_events"),
                    "error": attempt.get("error"),
                }
            )
    lines = [
        "# Job B Microprice / Orderbook Imbalance Batch",
        "",
        f"- generated_at_utc: {summary['generated_at_utc']}",
        f"- mode: {summary['mode']}",
        f"- manifest: {summary['manifest']}",
        f"- window: {summary['window']['start_time']} -> {summary['window']['end_time']}",
        f"- candidates: {summary['candidate_count']}",
        f"- parameter_sets: {summary['parameter_set_count']}",
        f"- attempts: {summary['attempt_count']}",
        f"- completed: {summary['completed_count']}",
        f"- skipped_no_coverage: {summary['skipped_no_coverage_count']}",
        f"- errors: {summary['error_count']}",
        "",
        "Safety: backtest/shadow only; no live trading, signing, cancellation, order submission, credentials, or worker-trading.",
        "",
        "## Attempts",
        "",
        "| status | market | params | result/error |",
        "|---|---|---|---|",
    ]
    for attempt in attempts:
        result = attempt.get("result") or {}
        detail = attempt.get("error") or f"pnl={result.get('pnl')} fills={result.get('fills')} quotes={result.get('quotes') or result.get('book_events')}"
        lines.append(
            f"| {attempt.get('status')} | {attempt.get('slug')} | `{json.dumps(attempt.get('params', {}), sort_keys=True)}` | {detail} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Job B bounded PMBT microprice batch runner (backtest-only).")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strategy", default="microprice_optimizer")
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--max-param-sets", type=int, default=2)
    parser.add_argument("--start-time", default=None)
    parser.add_argument("--end-time", default=None)
    parser.add_argument("--lookback-hours", type=int, default=1)
    parser.add_argument("--min-book-events", type=int, default=100)
    parser.add_argument("--per-attempt-timeout-secs", type=int, default=120)
    args = parser.parse_args()

    if args.max_candidates < 1 or args.max_param_sets < 1:
        raise SystemExit("max-candidates and max-param-sets must be >= 1")
    if args.end_time is None:
        end = _utc_now().replace(minute=0, second=0) - timedelta(hours=3)
        args.end_time = end.isoformat().replace("+00:00", "Z")
    else:
        end = datetime.fromisoformat(args.end_time.replace("Z", "+00:00"))
    if args.start_time is None:
        start = end - timedelta(hours=args.lookback_hours)
        args.start_time = start.isoformat().replace("+00:00", "Z")

    timestamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    summary = asyncio.run(run_batch(args))
    summary["output_files"] = _write_outputs(summary, args.output_dir, timestamp)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["candidate_count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
