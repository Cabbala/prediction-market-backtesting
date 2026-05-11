from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from glob import glob
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
DEFAULT_PASS_MANIFEST_GLOB = (
    "/opt/polymarket-lab/autoresearch/backtests/job_B_pmxt_l2_coverage_pass_*.json"
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
    coverage_start_time: str | None = None
    coverage_end_time: str | None = None
    coverage_book_events: int | None = None
    coverage_min_book_events: int | None = None
    manifest_rank: int | None = None
    selection_policy: str = "non_extreme_tail_priority_then_liquidity"


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
    diagnostics: dict[str, Any] | None = None


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


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _candidate_slug(raw: dict[str, Any]) -> str | None:
    value = raw.get("market_slug") or raw.get("slug")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [value]
    return []


def _first_present(raw: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = raw.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


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


def _strategy_from_row(raw: dict[str, Any], payload: dict[str, Any], requested: str) -> str:
    explicit = raw.get("source_strategy") or raw.get("strategy") or raw.get("strategy_name")
    if explicit is not None:
        return str(explicit)

    tags = (
        _as_list(raw.get("strategy_tags"))
        + _as_list(raw.get("strategy_fits"))
        + _as_list(raw.get("source_tags"))
        + _as_list(raw.get("source_tag"))
    )
    for tag in tags:
        if _strategy_matches(requested, str(tag)):
            return str(tag)
    if tags:
        return str(tags[0])

    payload_strategy = payload.get("strategy")
    if payload_strategy is not None:
        return str(payload_strategy)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    if str(metadata.get("artifact_type") or "").startswith("job_B_microprice"):
        return "Microprice"
    return "reward_manifest"


def _side_book_mid(raw: dict[str, Any], side: str) -> float | None:
    book_value = raw.get(f"{side}_book")
    if isinstance(book_value, dict):
        mid = _parse_float(book_value.get("mid"))
        if mid is not None:
            return mid
        bid = _parse_float(_first_present(book_value, "bid", "best_bid"))
        ask = _parse_float(_first_present(book_value, "ask", "best_ask"))
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
    else:
        compact_mid = _parse_float(book_value)
        if compact_mid is not None:
            return compact_mid

    bid = _parse_float(_first_present(raw, f"{side}_bid", f"{side}_best_bid"))
    ask = _parse_float(_first_present(raw, f"{side}_ask", f"{side}_best_ask"))
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    return None


def _side_book_spread(raw: dict[str, Any], side: str) -> float | None:
    book_value = raw.get(f"{side}_book")
    if isinstance(book_value, dict):
        spread = _parse_float(book_value.get("spread"))
        if spread is not None:
            return spread
        bid = _parse_float(_first_present(book_value, "bid", "best_bid"))
        ask = _parse_float(_first_present(book_value, "ask", "best_ask"))
        if bid is not None and ask is not None:
            return max(0.0, ask - bid)

    spread = _parse_float(raw.get(f"{side}_spread"))
    if spread is not None:
        return spread
    bid = _parse_float(_first_present(raw, f"{side}_bid", f"{side}_best_bid"))
    ask = _parse_float(_first_present(raw, f"{side}_ask", f"{side}_best_ask"))
    if bid is not None and ask is not None:
        return max(0.0, ask - bid)
    return None


def _normalize_candidate(
    raw: dict[str, Any],
    *,
    source_strategy: str,
    manifest_rank: int | None = None,
    default_coverage_window: dict[str, Any] | None = None,
    default_min_book_events: int | None = None,
) -> Candidate | None:
    slug = _candidate_slug(raw)
    if slug is None:
        return None
    token_index = raw.get("token_index", 0)
    try:
        token_index = int(token_index)
    except (TypeError, ValueError):
        token_index = 0
    coverage = raw.get("coverage") if isinstance(raw.get("coverage"), dict) else {}
    coverage_window = coverage.get("window") if isinstance(coverage.get("window"), dict) else {}
    if not coverage_window and default_coverage_window:
        coverage_window = default_coverage_window
    coverage_book_events = _parse_int(coverage.get("book_events"))
    coverage_min_book_events = _parse_int(coverage.get("min_book_events"))
    if coverage_min_book_events is None:
        coverage_min_book_events = default_min_book_events
    scan_mid = _parse_float(_first_present(raw, "scan_mid", "yes_probability", "yes_mid"))
    if scan_mid is None:
        scan_mid = _side_book_mid(raw, "yes")
    scan_spread = _parse_float(_first_present(raw, "scan_spread", "avg_spread", "spread"))
    if scan_spread is None:
        scan_spread = _side_book_spread(raw, "yes")
    return Candidate(
        slug=slug,
        question=str(raw.get("question") or slug),
        token_index=token_index,
        condition_id=raw.get("condition_id") if isinstance(raw.get("condition_id"), str) else None,
        scan_mid=scan_mid,
        scan_spread=scan_spread,
        scan_imbalance5=_parse_float(raw.get("scan_imbalance5")),
        liquidity=_parse_float(
            raw.get("scan_liquidity") or raw.get("liquidity") or raw.get("liquidityNum")
        ),
        source_strategy=source_strategy,
        coverage_start_time=coverage_window.get("start_time")
        if isinstance(coverage_window.get("start_time"), str)
        else None,
        coverage_end_time=coverage_window.get("end_time")
        if isinstance(coverage_window.get("end_time"), str)
        else None,
        coverage_book_events=coverage_book_events,
        coverage_min_book_events=coverage_min_book_events,
        manifest_rank=manifest_rank,
    )


def _candidate_priority(candidate: Candidate) -> tuple[int, float, float]:
    """Prefer non-extreme-tail candidates while preserving fail-closed filtering.

    Ultra-low Yes mids around one tick can dominate liquidity/reward scans but are
    poor first probes for microprice PnL/fill discovery. Keep them eligible, but
    rank 0.005-0.25 mids first, then deeper liquidity and tighter spreads.
    """
    mid = candidate.scan_mid
    if mid is None:
        bucket = 2
        mid_distance = 1.0
    elif 0.005 <= mid <= 0.25:
        bucket = 0
        mid_distance = abs(mid - 0.03)
    elif 0.003 <= mid < 0.005:
        bucket = 1
        mid_distance = 0.005 - mid
    else:
        bucket = 2
        mid_distance = abs((mid or 0.0) - 0.03)
    liquidity_score = -(candidate.liquidity or 0.0)
    spread_score = candidate.scan_spread if candidate.scan_spread is not None else 999.0
    return (bucket, mid_distance + spread_score, liquidity_score)


def load_candidates(manifest_path: Path, *, strategy: str, max_candidates: int) -> list[Candidate]:
    payload = json.loads(manifest_path.read_text())
    candidates: list[Candidate] = []
    seen: set[tuple[str, int]] = set()

    if isinstance(payload, dict):
        manifest_window = payload.get("window") if isinstance(payload.get("window"), dict) else None
        manifest_min_book_events = _parse_int(payload.get("min_book_events"))
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
                for manifest_idx, raw in enumerate(markets):
                    if not isinstance(raw, dict):
                        continue
                    cand = _normalize_candidate(
                        raw,
                        source_strategy=source_strategy,
                        manifest_rank=manifest_idx,
                        default_coverage_window=manifest_window,
                        default_min_book_events=manifest_min_book_events,
                    )
                    if cand is None:
                        continue
                    key = (cand.slug, cand.token_index)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(cand)
        raw_candidates = payload.get("candidates")
        if isinstance(raw_candidates, list) and not candidates:
            for manifest_idx, raw in enumerate(raw_candidates):
                if not isinstance(raw, dict):
                    continue
                source_strategy = _strategy_from_row(raw, payload, strategy)
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
                cand = _normalize_candidate(
                    raw,
                    source_strategy=source_strategy,
                    manifest_rank=manifest_idx,
                    default_coverage_window=manifest_window,
                    default_min_book_events=manifest_min_book_events,
                )
                if cand is None:
                    continue
                key = (cand.slug, cand.token_index)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(cand)
    elif isinstance(payload, list):
        for manifest_idx, raw in enumerate(payload):
            if not isinstance(raw, dict):
                continue
            cand = _normalize_candidate(
                raw, source_strategy="list_manifest", manifest_rank=manifest_idx
            )
            if cand is None:
                continue
            key = (cand.slug, cand.token_index)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(cand)
    candidates.sort(key=_candidate_priority)
    return candidates[:max_candidates]


def _declared_candidate_count(manifest_path: Path) -> int:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return 0
    candidate_count = _parse_int(payload.get("candidate_count"))
    if candidate_count is not None:
        return candidate_count
    candidates = payload.get("candidates")
    return len(candidates) if isinstance(candidates, list) else 0


def select_latest_non_empty_pass_manifest(
    manifest_glob: str,
    *,
    strategy: str,
) -> tuple[Path | None, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    manifest_paths = [
        Path(path)
        for path in sorted(
            glob(manifest_glob),
            key=lambda value: Path(value).stat().st_mtime,
            reverse=True,
        )
    ]
    for path in manifest_paths:
        record: dict[str, Any] = {"path": str(path), "selected": False}
        try:
            declared_count = _declared_candidate_count(path)
            record["candidate_count"] = declared_count
            if declared_count <= 0:
                record["reason"] = "skipped_zero_candidates"
                records.append(record)
                continue
            matching_candidates = load_candidates(path, strategy=strategy, max_candidates=1)
            record["matching_candidate_count"] = len(matching_candidates)
            if not matching_candidates:
                record["reason"] = "skipped_no_matching_pass_candidates"
                records.append(record)
                continue
        except Exception as exc:  # pass manifest discovery must fail closed
            record["reason"] = "skipped_unreadable_or_invalid"
            record["error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
            continue
        record["selected"] = True
        record["reason"] = "selected_newest_non_empty_pass_manifest"
        records.append(record)
        return path, records
    return None, records


def microprice_param_grid() -> list[dict[str, Any]]:
    return [
        {
            "depth_levels": 1,
            "entry_imbalance": 0.55,
            "exit_imbalance": 0.50,
            "min_microprice_edge": 0.0005,
            "quote_lifetime_seconds": 10.0,
        },
        {
            "depth_levels": 3,
            "entry_imbalance": 0.57,
            "exit_imbalance": 0.50,
            "min_microprice_edge": 0.0010,
            "quote_lifetime_seconds": 30.0,
        },
        {
            "depth_levels": 5,
            "entry_imbalance": 0.60,
            "exit_imbalance": 0.52,
            "min_microprice_edge": 0.0015,
            "quote_lifetime_seconds": 60.0,
        },
        {
            "depth_levels": 3,
            "entry_imbalance": 0.62,
            "exit_imbalance": 0.54,
            "min_microprice_edge": 0.0020,
            "quote_lifetime_seconds": 30.0,
        },
    ]


def _safe_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        str(key): _safe_json_value(value)
        for key, value in result.items()
        if str(key).lower() not in {"private_key", "secret", "token", "api_key"}
    }


def _safe_json_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        return repr(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_safe_json_value(item, depth=depth + 1) for item in value[:50]]
    if isinstance(value, dict):
        return {
            str(k): _safe_json_value(v, depth=depth + 1)
            for k, v in list(value.items())[:100]
            if str(k).lower() not in {"private_key", "secret", "token", "api_key"}
        }
    return repr(value)


def _safe_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_count(value: Any) -> int:
    parsed = _parse_int(value)
    return parsed if parsed is not None else 0


def _safe_nested_float(mapping: dict[str, Any], key: str) -> float | None:
    return _parse_float(mapping.get(key))


def _tick_cost_diagnostic(
    *, scan_mid: float | None, scan_spread: float | None, observed_min_spread: float | None
) -> dict[str, Any]:
    effective_spread = scan_spread if scan_spread is not None else observed_min_spread
    ratio = (
        effective_spread / scan_mid
        if scan_mid is not None and scan_mid > 0 and effective_spread is not None
        else None
    )
    blocked = bool(ratio is not None and ratio >= 0.20)
    return {
        "blocked": blocked,
        "scan_mid": scan_mid,
        "scan_spread": scan_spread,
        "observed_min_spread": observed_min_spread,
        "effective_spread": effective_spread,
        "spread_to_mid_ratio": ratio,
        "threshold_ratio": 0.20,
    }


def _classify_no_order_diagnostics(
    *,
    status: str,
    fills: float | None,
    strategy_order_count: float | None,
    scan_mid: float | None,
    scan_spread: float | None,
    params: dict[str, Any],
    strategy_diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    diagnostics = strategy_diagnostics or {}
    observed = _safe_mapping(diagnostics.get("observed"))
    thresholds = _safe_mapping(diagnostics.get("thresholds"))
    entry_block_counts = _safe_mapping(diagnostics.get("entry_block_counts"))
    book_signal_count = _safe_count(diagnostics.get("book_signal_count"))
    flat_evaluation_count = _safe_count(diagnostics.get("flat_evaluation_count"))
    entry_signal_count = _safe_count(diagnostics.get("entry_signal_count"))
    no_fill_completed = status == "completed" and (fills or 0.0) == 0.0
    order_count = strategy_order_count if strategy_order_count is not None else 0.0

    min_spread = _safe_nested_float(observed, "min_spread")
    max_microprice_edge = _safe_nested_float(observed, "max_microprice_edge")
    max_spread_threshold = _parse_float(thresholds.get("max_spread"))
    min_edge_threshold = _parse_float(
        thresholds.get("min_microprice_edge") or params.get("min_microprice_edge")
    )
    tick_cost = _tick_cost_diagnostic(
        scan_mid=scan_mid, scan_spread=scan_spread, observed_min_spread=min_spread
    )
    spread_blocked = bool(
        no_fill_completed
        and entry_signal_count == 0
        and (
            _safe_count(entry_block_counts.get("spread")) > 0
            or (
                min_spread is not None
                and max_spread_threshold is not None
                and min_spread > max_spread_threshold
            )
        )
    )
    edge_blocked = bool(
        no_fill_completed
        and entry_signal_count == 0
        and (
            _safe_count(entry_block_counts.get("microprice_edge")) > 0
            or (
                max_microprice_edge is not None
                and min_edge_threshold is not None
                and max_microprice_edge < min_edge_threshold
            )
        )
    )
    queue_blocked = bool(no_fill_completed and (order_count > 0 or entry_signal_count > 0))
    fill_opportunity_blocked = bool(no_fill_completed and entry_signal_count == 0)
    causes: list[str] = []
    if no_fill_completed and tick_cost["blocked"]:
        causes.append("tick_cost_blocker")
    if spread_blocked:
        causes.append("spread_blocker")
    if edge_blocked:
        causes.append("edge_blocker")
    if queue_blocked:
        causes.append("queue_blocker")
    if fill_opportunity_blocked:
        causes.append("fill_opportunity_blocker")
    if no_fill_completed and not causes:
        causes.append("unexplained_no_order_or_fill")

    primary_cause = None
    for candidate_cause in (
        "queue_blocker",
        "edge_blocker",
        "spread_blocker",
        "fill_opportunity_blocker",
        "tick_cost_blocker",
        "unexplained_no_order_or_fill",
    ):
        if candidate_cause in causes:
            primary_cause = candidate_cause
            break

    return {
        "eligible": no_fill_completed,
        "primary_cause": primary_cause,
        "causes": causes,
        "cause_counts": {cause: 1 for cause in causes},
        "blockers": {
            "tick_cost": tick_cost,
            "spread": {
                "blocked": spread_blocked,
                "block_count": _safe_count(entry_block_counts.get("spread")),
                "observed_min_spread": min_spread,
                "threshold": max_spread_threshold,
            },
            "edge": {
                "blocked": edge_blocked,
                "block_count": _safe_count(entry_block_counts.get("microprice_edge")),
                "observed_max_microprice_edge": max_microprice_edge,
                "threshold": min_edge_threshold,
            },
            "queue": {
                "blocked": queue_blocked,
                "strategy_order_count": strategy_order_count,
                "entry_signal_count": entry_signal_count,
                "fills": fills,
            },
            "fill_opportunity": {
                "blocked": fill_opportunity_blocked,
                "book_signal_count": book_signal_count,
                "flat_evaluation_count": flat_evaluation_count,
                "entry_signal_count": entry_signal_count,
            },
        },
    }


def _diagnose_attempt(
    candidate: Candidate,
    params: dict[str, Any],
    result: dict[str, Any] | None,
    *,
    start_time: str,
    end_time: str,
    min_book_events: int,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    fills = _parse_float(result.get("fills") if result else None) if result else None
    pnl = _parse_float(result.get("pnl") if result else None) if result else None
    book_events = _parse_float(result.get("book_events") if result else None) if result else None
    portfolio_stats = result.get("portfolio_stats") if result else None
    if not isinstance(portfolio_stats, dict):
        portfolio_stats = {}
    strategy_order_count = _parse_float(portfolio_stats.get("total_orders"))
    mid = candidate.scan_mid
    spread = candidate.scan_spread
    tail_bucket = (
        "unknown"
        if mid is None
        else "ultra_low_tail"
        if mid < 0.005
        else "low_tail"
        if mid < 0.01
        else "non_extreme_tail"
        if mid <= 0.25
        else "high_probability"
    )
    suspected_causes: list[str] = []
    if status == "skipped_no_coverage" or (
        book_events is not None and book_events < min_book_events
    ):
        suspected_causes.append("coverage_insufficient")
    if status == "completed" and (fills or 0) == 0:
        suspected_causes.append("zero_fills_with_coverage")
        if tail_bucket == "ultra_low_tail":
            suspected_causes.append("one_tick_relative_cost_too_large")
        if strategy_order_count == 0:
            suspected_causes.append("strategy_generated_no_simulated_orders")
        elif strategy_order_count is not None and strategy_order_count > 0:
            suspected_causes.append("simulated_orders_not_filled_or_not_crossed")
        else:
            suspected_causes.append("order_intent_count_not_instrumented")
        suspected_causes.append("needs_multi_window_multi_candidate_replay")
    if error:
        suspected_causes.append("runner_error")
    strategy_diagnostics = (
        result.get("strategy_diagnostics")
        if result and isinstance(result.get("strategy_diagnostics"), dict)
        else None
    )
    no_order = _classify_no_order_diagnostics(
        status=status,
        fills=fills,
        strategy_order_count=strategy_order_count,
        scan_mid=mid,
        scan_spread=spread,
        params=params,
        strategy_diagnostics=strategy_diagnostics,
    )
    for cause in no_order.get("causes", []):
        if cause not in suspected_causes:
            suspected_causes.append(cause)
    return {
        "window": {"start_time": start_time, "end_time": end_time},
        "min_book_events": min_book_events,
        "scan_mid": mid,
        "scan_spread": spread,
        "liquidity": candidate.liquidity,
        "tail_bucket": tail_bucket,
        "coverage_book_events": candidate.coverage_book_events,
        "result_book_events": book_events,
        "fills": fills,
        "pnl": pnl,
        "strategy_order_count": strategy_order_count,
        "params": params,
        "strategy_diagnostics": strategy_diagnostics,
        "no_order": no_order,
        "no_order_primary_cause": no_order.get("primary_cause"),
        "no_order_causes": no_order.get("causes", []),
        "suspected_causes": suspected_causes,
    }


def _aggregate_diagnostics(attempts: list[BacktestAttempt]) -> dict[str, Any]:
    completed = [a for a in attempts if a.status == "completed"]
    zero_fill = [a for a in completed if ((a.result or {}).get("fills") in {0, 0.0, None})]
    causes: dict[str, int] = {}
    no_order_cause_counts: dict[str, int] = {}
    no_order_primary_cause_counts: dict[str, int] = {}
    for attempt in attempts:
        diagnostics = attempt.diagnostics or {}
        for cause in diagnostics.get("suspected_causes", []):
            causes[cause] = causes.get(cause, 0) + 1
        no_order = diagnostics.get("no_order")
        if isinstance(no_order, dict):
            for cause, count in (no_order.get("cause_counts") or {}).items():
                no_order_cause_counts[str(cause)] = no_order_cause_counts.get(str(cause), 0) + int(
                    count or 0
                )
            primary_cause = no_order.get("primary_cause")
            if primary_cause:
                key = str(primary_cause)
                no_order_primary_cause_counts[key] = no_order_primary_cause_counts.get(key, 0) + 1
    return {
        "completed_attempts": len(completed),
        "zero_fill_completed_attempts": len(zero_fill),
        "suspected_cause_counts": causes,
        "no_order_cause_counts": no_order_cause_counts,
        "no_order_primary_cause_counts": no_order_primary_cause_counts,
        "profit_opportunity_demonstrated": any(
            ((a.result or {}).get("pnl") or 0) > 0 and ((a.result or {}).get("fills") or 0) > 0
            for a in attempts
            if a.status == "completed"
        ),
    }


def _candidate_replay_request(candidate: Candidate, args: argparse.Namespace) -> dict[str, Any]:
    start_time = candidate.coverage_start_time or args.start_time
    end_time = candidate.coverage_end_time or args.end_time
    min_book_events = (
        candidate.coverage_min_book_events
        if candidate.coverage_min_book_events is not None
        else args.min_book_events
    )
    return {
        "slug": candidate.slug,
        "token_index": candidate.token_index,
        "source_strategy": candidate.source_strategy,
        "manifest_rank": candidate.manifest_rank,
        "window": {"start_time": start_time, "end_time": end_time},
        "min_book_events": min_book_events,
        "coverage_book_events": candidate.coverage_book_events,
    }


def _unique_replay_windows(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str | None, str | None, int | None], dict[str, Any]] = {}
    for request in requests:
        window = request["window"]
        key = (
            window.get("start_time"),
            window.get("end_time"),
            request.get("min_book_events"),
        )
        unique.setdefault(
            key,
            {
                "window": {
                    "start_time": window.get("start_time"),
                    "end_time": window.get("end_time"),
                },
                "min_book_events": request.get("min_book_events"),
                "candidate_count": 0,
            },
        )
        unique[key]["candidate_count"] += 1
    return list(unique.values())


def _attempt_replay_key(
    attempt: BacktestAttempt,
) -> tuple[str | None, str | None, int | None]:
    diagnostics = attempt.diagnostics or {}
    window = diagnostics.get("window") if isinstance(diagnostics.get("window"), dict) else {}
    return (
        window.get("start_time"),
        window.get("end_time"),
        _parse_int(diagnostics.get("min_book_events")),
    )


def _build_exact_window_metadata(
    *,
    requested_window: dict[str, str],
    requested_min_book_events: int,
    replay_requests: list[dict[str, Any]],
    attempts: list[BacktestAttempt],
) -> dict[str, Any]:
    selected_windows = _unique_replay_windows(replay_requests)
    warnings: list[str] = []
    selected_window: dict[str, str | None] | None = None
    selected_min_book_events: int | None = None
    root_window: dict[str, str | None] = dict(requested_window)
    root_min_book_events: int | None = requested_min_book_events
    status = "verified"

    if not replay_requests:
        status = "fail_closed_no_candidates"
        warnings.append("no_candidate_coverage_pass_manifest")
    elif len(selected_windows) != 1:
        status = "fail_closed"
        warnings.append("exact_window_mismatch: multiple selected windows or min_book_events")
    else:
        selected = selected_windows[0]
        selected_window = dict(selected["window"])
        selected_min_book_events = _parse_int(selected.get("min_book_events"))
        root_window = dict(selected_window)
        root_min_book_events = selected_min_book_events
        expected_key = (
            selected_window.get("start_time"),
            selected_window.get("end_time"),
            selected_min_book_events,
        )
        mismatches = [
            {
                "slug": attempt.slug,
                "token_index": attempt.token_index,
                "attempt_window": (attempt.diagnostics or {}).get("window"),
                "attempt_min_book_events": (attempt.diagnostics or {}).get("min_book_events"),
            }
            for attempt in attempts
            if _attempt_replay_key(attempt) != expected_key
        ]
        if attempts and not mismatches:
            status = "verified"
        else:
            status = "fail_closed"
            warnings.append("exact_window_mismatch")
            if not attempts:
                warnings.append("no_attempts_generated")

    return {
        "status": status,
        "requested_window": requested_window,
        "requested_min_book_events": requested_min_book_events,
        "selected_window": selected_window,
        "selected_min_book_events": selected_min_book_events,
        "selected_windows": selected_windows,
        "root_window": root_window,
        "root_min_book_events": root_min_book_events,
        "warnings": warnings,
    }


async def run_attempt(
    candidate: Candidate,
    params: dict[str, Any],
    *,
    start_time: str,
    end_time: str,
    min_book_events: int,
) -> BacktestAttempt:
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
            data=MarketDataConfig(
                platform=Polymarket, data_type=Book, vendor=PMXT, sources=DEFAULT_SOURCES
            ),
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
        safe_result = _safe_result(result)
        return BacktestAttempt(
            slug=candidate.slug,
            question=candidate.question,
            token_index=candidate.token_index,
            source_strategy=candidate.source_strategy,
            params=params,
            status="skipped_no_coverage" if result is None else "completed",
            result=safe_result,
            error=None,
            diagnostics=_diagnose_attempt(
                candidate,
                params,
                safe_result,
                start_time=start_time,
                end_time=end_time,
                min_book_events=min_book_events,
                status="skipped_no_coverage" if result is None else "completed",
            ),
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
            diagnostics=_diagnose_attempt(
                candidate,
                params,
                None,
                start_time=start_time,
                end_time=end_time,
                min_book_events=min_book_events,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            ),
        )


async def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    candidates = load_candidates(
        args.manifest, strategy=args.strategy, max_candidates=args.max_candidates
    )
    selected_params = microprice_param_grid()[: args.max_param_sets]
    attempts: list[BacktestAttempt] = []
    replay_requests = [_candidate_replay_request(candidate, args) for candidate in candidates]
    for candidate in candidates:
        replay_request = _candidate_replay_request(candidate, args)
        replay_window = replay_request["window"]
        for params in selected_params:
            attempts.append(
                await asyncio.wait_for(
                    run_attempt(
                        candidate,
                        params,
                        start_time=replay_window["start_time"],
                        end_time=replay_window["end_time"],
                        min_book_events=int(replay_request["min_book_events"]),
                    ),
                    timeout=args.per_attempt_timeout_secs,
                )
            )
    completed = sum(1 for attempt in attempts if attempt.status == "completed")
    skipped = sum(1 for attempt in attempts if attempt.status == "skipped_no_coverage")
    errored = sum(1 for attempt in attempts if attempt.status == "error")
    requested_window = {"start_time": args.start_time, "end_time": args.end_time}
    exact_window = _build_exact_window_metadata(
        requested_window=requested_window,
        requested_min_book_events=args.min_book_events,
        replay_requests=replay_requests,
        attempts=attempts,
    )
    return {
        "generated_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
        "mode": SAFETY_MODE,
        "safety": {
            "live_trading": False,
            "orders_submitted": False,
            "orders_signed": False,
            "credentials_required": False,
            "worker_trading_started": False,
            "live_trading_worker_started": False,
        },
        "manifest": str(args.manifest),
        "manifest_selection": getattr(
            args,
            "manifest_selection",
            [{"path": str(args.manifest), "selected": True, "reason": "explicit_manifest"}],
        ),
        "window": exact_window["root_window"],
        "requested_window": requested_window,
        "selected_window": exact_window["selected_window"],
        "selected_windows": exact_window["selected_windows"],
        "min_book_events": exact_window["root_min_book_events"],
        "requested_min_book_events": args.min_book_events,
        "selected_min_book_events": exact_window["selected_min_book_events"],
        "exact_window_status": exact_window["status"],
        "warnings": exact_window["warnings"],
        "exact_window": exact_window,
        "candidate_count": len(candidates),
        "parameter_set_count": len(selected_params),
        "attempt_count": len(attempts),
        "completed_count": completed,
        "skipped_no_coverage_count": skipped,
        "error_count": errored,
        "candidate_selection": {
            "policy": "non_extreme_tail_priority_then_liquidity",
            "preferred_yes_mid_range": [0.005, 0.25],
            "ultra_low_tail_retained_but_deprioritized": True,
        },
        "candidate_replay_requests": replay_requests,
        "candidates": [asdict(candidate) for candidate in candidates],
        "diagnostics": _aggregate_diagnostics(attempts),
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
            fieldnames=[
                "slug",
                "token_index",
                "source_strategy",
                "status",
                "params",
                "pnl",
                "fills",
                "book_events",
                "strategy_order_count",
                "no_order_primary_cause",
                "no_order_causes",
                "diagnostics",
                "error",
            ],
        )
        writer.writeheader()
        for attempt in attempts:
            result = attempt.get("result") or {}
            diagnostics = attempt.get("diagnostics") or {}
            no_order = diagnostics.get("no_order") or {}
            writer.writerow(
                {
                    "slug": attempt.get("slug"),
                    "token_index": attempt.get("token_index"),
                    "source_strategy": attempt.get("source_strategy"),
                    "status": attempt.get("status"),
                    "params": json.dumps(attempt.get("params", {}), sort_keys=True),
                    "pnl": result.get("pnl"),
                    "fills": result.get("fills"),
                    "book_events": result.get("book_events"),
                    "strategy_order_count": diagnostics.get("strategy_order_count"),
                    "no_order_primary_cause": no_order.get("primary_cause"),
                    "no_order_causes": json.dumps(no_order.get("causes", []), sort_keys=True),
                    "diagnostics": json.dumps(diagnostics, sort_keys=True),
                    "error": attempt.get("error"),
                }
            )
    aggregate_diagnostics = summary.get("diagnostics", {})
    lines = [
        "# Job B Microprice / Orderbook Imbalance Batch",
        "",
        f"- generated_at_utc: {summary['generated_at_utc']}",
        f"- mode: {summary['mode']}",
        f"- manifest: {summary['manifest']}",
        f"- window: {summary['window']['start_time']} -> {summary['window']['end_time']}",
        f"- min_book_events: {summary.get('min_book_events')}",
        f"- requested_window: {summary['requested_window']['start_time']} -> {summary['requested_window']['end_time']}",
        f"- requested_min_book_events: {summary.get('requested_min_book_events')}",
        f"- selected_window: {json.dumps(summary.get('selected_window'), sort_keys=True)}",
        f"- selected_min_book_events: {summary.get('selected_min_book_events')}",
        f"- exact_window_status: {summary.get('exact_window_status')}",
        f"- warnings: {json.dumps(summary.get('warnings', []), sort_keys=True)}",
        f"- candidates: {summary['candidate_count']}",
        f"- parameter_sets: {summary['parameter_set_count']}",
        f"- attempts: {summary['attempt_count']}",
        f"- completed: {summary['completed_count']}",
        f"- skipped_no_coverage: {summary['skipped_no_coverage_count']}",
        f"- errors: {summary['error_count']}",
        f"- no_order_cause_counts: {json.dumps(aggregate_diagnostics.get('no_order_cause_counts', {}), sort_keys=True)}",
        f"- no_order_primary_cause_counts: {json.dumps(aggregate_diagnostics.get('no_order_primary_cause_counts', {}), sort_keys=True)}",
        f"- diagnostics: {json.dumps(aggregate_diagnostics, sort_keys=True)}",
        "",
        "Safety: backtest/shadow only; no live trading, signing, cancellation, order submission, credentials, or worker-trading.",
        "",
        "## Attempts",
        "",
        "| status | market | params | no-order diagnostics | result/error |",
        "|---|---|---|---|---|",
    ]
    for attempt in attempts:
        result = attempt.get("result") or {}
        diagnostics = attempt.get("diagnostics") or {}
        causes = ",".join(diagnostics.get("suspected_causes", []))
        order_count = diagnostics.get("strategy_order_count")
        no_order = diagnostics.get("no_order") or {}
        blockers = no_order.get("blockers") or {}
        blocker_summary = {
            "primary": no_order.get("primary_cause"),
            "causes": no_order.get("causes", []),
            "tick_cost": (blockers.get("tick_cost") or {}).get("blocked"),
            "spread": (blockers.get("spread") or {}).get("blocked"),
            "edge": (blockers.get("edge") or {}).get("blocked"),
            "queue": (blockers.get("queue") or {}).get("blocked"),
            "fill_opportunity": (blockers.get("fill_opportunity") or {}).get("blocked"),
        }
        detail = attempt.get("error") or (
            f"pnl={result.get('pnl')} fills={result.get('fills')} "
            f"book_events={result.get('book_events')} strategy_orders={order_count} causes={causes}"
        )
        lines.append(
            f"| {attempt.get('status')} | {attempt.get('slug')} | `{json.dumps(attempt.get('params', {}), sort_keys=True)}` | {json.dumps(blocker_summary, sort_keys=True)} | {detail} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Job B bounded PMBT microprice batch runner (backtest-only)."
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--manifest-glob",
        default=DEFAULT_PASS_MANIFEST_GLOB,
        help="Pass-manifest glob used when --manifest is omitted; zero-candidate files are skipped.",
    )
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
    if args.manifest is None:
        selected_manifest, selection_records = select_latest_non_empty_pass_manifest(
            args.manifest_glob,
            strategy=args.strategy,
        )
        args.manifest_selection = selection_records
        if selected_manifest is None:
            print(
                json.dumps(
                    {
                        "error": "no_non_empty_pass_manifest",
                        "manifest_glob": args.manifest_glob,
                        "manifest_selection": selection_records,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        args.manifest = selected_manifest
    else:
        args.manifest_selection = [
            {"path": str(args.manifest), "selected": True, "reason": "explicit_manifest"}
        ]
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
