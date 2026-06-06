from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping
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
class ReplayWindow:
    start_time: str
    end_time: str
    min_book_events: int | None = None
    source: str | None = None
    candidate_count: int | None = None
    book_events: int | None = None
    provenance: str = "manifest"


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
    replay_windows: tuple[ReplayWindow, ...] = ()


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


def _replay_window_from_mapping(
    raw: dict[str, Any],
    *,
    default_min_book_events: int | None = None,
    provenance: str,
) -> ReplayWindow | None:
    start_time = raw.get("start_time")
    end_time = raw.get("end_time")
    if not isinstance(start_time, str) or not start_time.strip():
        return None
    if not isinstance(end_time, str) or not end_time.strip():
        return None
    min_book_events = _parse_int(raw.get("min_book_events"))
    if min_book_events is None:
        min_book_events = default_min_book_events
    return ReplayWindow(
        start_time=start_time.strip(),
        end_time=end_time.strip(),
        min_book_events=min_book_events,
        source=raw.get("source")
        if isinstance(raw.get("source"), str)
        else raw.get("label")
        if isinstance(raw.get("label"), str)
        else None,
        candidate_count=_parse_int(raw.get("candidate_count")),
        book_events=_parse_int(raw.get("book_events")),
        provenance=provenance,
    )


def _dedupe_replay_windows(windows: list[ReplayWindow]) -> tuple[ReplayWindow, ...]:
    deduped: list[ReplayWindow] = []
    seen: set[tuple[str, str, int | None]] = set()
    for window in windows:
        key = (window.start_time, window.end_time, window.min_book_events)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(window)
    return tuple(deduped)


def _manifest_replay_windows(payload: dict[str, Any]) -> tuple[ReplayWindow, ...]:
    default_min_book_events = _parse_int(payload.get("min_book_events"))
    windows: list[ReplayWindow] = []
    manifest_window = payload.get("window")
    if isinstance(manifest_window, dict):
        parsed = _replay_window_from_mapping(
            manifest_window,
            default_min_book_events=default_min_book_events,
            provenance="manifest_window",
        )
        if parsed is not None:
            windows.append(parsed)

    guidance = payload.get("coverage_first_guidance")
    if isinstance(guidance, dict):
        for key in ("recommended_windows", "recommended_pmxt_windows"):
            for raw_window in _as_list(guidance.get(key)):
                if not isinstance(raw_window, dict):
                    continue
                parsed = _replay_window_from_mapping(
                    raw_window,
                    default_min_book_events=default_min_book_events,
                    provenance=f"coverage_first_guidance.{key}",
                )
                if parsed is not None:
                    windows.append(parsed)
    return _dedupe_replay_windows(windows)


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
    default_replay_windows: tuple[ReplayWindow, ...] = (),
    prefer_default_replay_windows: bool = False,
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
    candidate_coverage_window = (
        coverage.get("window") if isinstance(coverage.get("window"), dict) else {}
    )
    coverage_window = (
        {}
        if prefer_default_replay_windows and default_replay_windows
        else candidate_coverage_window
    )
    coverage_book_events = _parse_int(coverage.get("book_events"))
    coverage_min_book_events = _parse_int(coverage.get("min_book_events"))
    if coverage_min_book_events is None:
        coverage_min_book_events = default_min_book_events
    if prefer_default_replay_windows and default_replay_windows:
        coverage_book_events = default_replay_windows[0].book_events
    replay_windows: list[ReplayWindow] = []
    if prefer_default_replay_windows and default_replay_windows:
        replay_windows.extend(default_replay_windows)
    else:
        coverage_windows = coverage.get("windows")
        if isinstance(coverage_windows, list):
            for raw_window in coverage_windows:
                if not isinstance(raw_window, dict):
                    continue
                parsed = _replay_window_from_mapping(
                    raw_window,
                    default_min_book_events=coverage_min_book_events,
                    provenance="candidate_coverage_windows",
                )
                if parsed is not None:
                    replay_windows.append(parsed)
        if candidate_coverage_window:
            parsed = _replay_window_from_mapping(
                candidate_coverage_window,
                default_min_book_events=coverage_min_book_events,
                provenance="candidate_coverage_window",
            )
            if parsed is not None:
                replay_windows.append(parsed)
    if not replay_windows:
        replay_windows.extend(default_replay_windows)
    replay_window_tuple = _dedupe_replay_windows(replay_windows)
    if not coverage_window and replay_window_tuple:
        primary_window = replay_window_tuple[0]
        coverage_window = {
            "start_time": primary_window.start_time,
            "end_time": primary_window.end_time,
        }
    if not coverage_window and default_coverage_window:
        coverage_window = default_coverage_window
    if coverage_min_book_events is None and replay_window_tuple:
        coverage_min_book_events = replay_window_tuple[0].min_book_events
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
        replay_windows=replay_window_tuple,
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
        manifest_replay_windows = _manifest_replay_windows(payload)
        manifest_window = (
            {
                "start_time": manifest_replay_windows[0].start_time,
                "end_time": manifest_replay_windows[0].end_time,
            }
            if manifest_replay_windows
            else payload.get("window")
            if isinstance(payload.get("window"), dict)
            else None
        )
        manifest_min_book_events = (
            manifest_replay_windows[0].min_book_events
            if manifest_replay_windows and manifest_replay_windows[0].min_book_events is not None
            else _parse_int(payload.get("min_book_events"))
        )
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
                        default_replay_windows=manifest_replay_windows,
                        prefer_default_replay_windows=bool(manifest_replay_windows),
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
                    default_replay_windows=manifest_replay_windows,
                    prefer_default_replay_windows=bool(manifest_replay_windows),
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
                record["classification"] = "no_pass"
                records.append(record)
                continue
            matching_candidates = load_candidates(path, strategy=strategy, max_candidates=1)
            record["matching_candidate_count"] = len(matching_candidates)
            if not matching_candidates:
                record["reason"] = "skipped_no_matching_pass_candidates"
                record["classification"] = "no_pass"
                records.append(record)
                continue
        except Exception as exc:  # pass manifest discovery must fail closed
            record["reason"] = "skipped_unreadable_or_invalid"
            record["error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
            continue
        record["selected"] = True
        record["reason"] = "selected_newest_non_empty_pass_manifest"
        record["classification"] = "pass"
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


def _safe_mapping_from(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _safe_fill_events(result: dict[str, Any]) -> list[dict[str, Any]]:
    fill_events = result.get("fill_events")
    if not isinstance(fill_events, list):
        return []
    return [dict(event) for event in fill_events if isinstance(event, Mapping)]


def _weighted_fill_stats(fill_events: list[dict[str, Any]], action: str) -> dict[str, Any]:
    quantity = 0.0
    notional = 0.0
    commissions = 0.0
    for event in fill_events:
        if str(event.get("action") or "").lower() != action:
            continue
        price = _parse_float(event.get("price"))
        qty = _parse_float(event.get("quantity"))
        if price is None or qty is None:
            continue
        quantity += qty
        notional += price * qty
        commissions += _parse_float(event.get("commission")) or 0.0
    return {
        "quantity": quantity,
        "notional": notional,
        "average_price": notional / quantity if quantity > 0 else None,
        "commission": commissions,
    }


def _fill_time_span_seconds(fill_events: list[dict[str, Any]]) -> float | None:
    timestamps: list[float] = []
    for event in fill_events:
        timestamp = event.get("timestamp")
        if not isinstance(timestamp, str):
            continue
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        timestamps.append(parsed.timestamp())
    if len(timestamps) < 2:
        return None
    return max(timestamps) - min(timestamps)


def _terminal_price(result: dict[str, Any]) -> float | None:
    for key in ("last", "entry_min"):
        price = _parse_float(result.get(key))
        if price is not None:
            return price
    return None


def _price_move_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    fill_events = _safe_fill_events(result)
    buy = _weighted_fill_stats(fill_events, "buy")
    sell = _weighted_fill_stats(fill_events, "sell")
    terminal_price = _terminal_price(result)
    avg_buy = _parse_float(buy.get("average_price"))
    avg_sell = _parse_float(sell.get("average_price"))
    round_trip_edge = avg_sell - avg_buy if avg_buy is not None and avg_sell is not None else None
    terminal_long_markout = (
        terminal_price - avg_buy if terminal_price is not None and avg_buy is not None else None
    )
    available = round_trip_edge is not None or terminal_long_markout is not None
    adverse = bool(
        (round_trip_edge is not None and round_trip_edge < 0)
        or (terminal_long_markout is not None and terminal_long_markout < 0)
    )
    return {
        "available": available,
        "adverse_selection_or_markout_detected": adverse,
        "fill_event_count": len(fill_events),
        "buy_quantity": buy["quantity"],
        "sell_quantity": sell["quantity"],
        "average_buy_price": avg_buy,
        "average_sell_price": avg_sell,
        "terminal_price": terminal_price,
        "round_trip_price_edge": round_trip_edge,
        "terminal_long_markout": terminal_long_markout,
        "commission_total": buy["commission"] + sell["commission"],
        "fill_time_span_seconds": _fill_time_span_seconds(fill_events),
    }


def _spread_tick_cost_attribution(
    *,
    diagnostics: dict[str, Any],
    result: dict[str, Any],
    price_move: dict[str, Any],
) -> dict[str, Any]:
    strategy_diagnostics = _safe_mapping_from(diagnostics.get("strategy_diagnostics"))
    observed = _safe_mapping_from(strategy_diagnostics.get("observed"))
    thresholds = _safe_mapping_from(strategy_diagnostics.get("thresholds"))
    no_order = _safe_mapping_from(diagnostics.get("no_order"))
    blockers = _safe_mapping_from(no_order.get("blockers"))
    tick_cost = _safe_mapping_from(blockers.get("tick_cost"))
    scan_mid = _parse_float(diagnostics.get("scan_mid"))
    scan_spread = _parse_float(diagnostics.get("scan_spread"))
    observed_min_spread = _parse_float(observed.get("min_spread"))
    observed_max_spread = _parse_float(observed.get("max_spread"))
    max_spread_threshold = _parse_float(thresholds.get("max_spread"))
    spread_to_mid_ratio = _parse_float(tick_cost.get("spread_to_mid_ratio"))
    if spread_to_mid_ratio is None and scan_mid is not None and scan_mid > 0:
        effective_spread = scan_spread if scan_spread is not None else observed_min_spread
        spread_to_mid_ratio = effective_spread / scan_mid if effective_spread is not None else None
    threshold_ratio = _parse_float(tick_cost.get("threshold_ratio")) or 0.20
    commission_total = _parse_float(price_move.get("commission_total")) or 0.0
    fill_events = _safe_fill_events(result)
    fill_notional = 0.0
    for event in fill_events:
        price = _parse_float(event.get("price"))
        quantity = _parse_float(event.get("quantity"))
        if price is not None and quantity is not None:
            fill_notional += price * quantity
    commission_to_notional_ratio = commission_total / fill_notional if fill_notional > 0 else None
    round_trip_edge = _parse_float(price_move.get("round_trip_price_edge"))
    effective_spread = scan_spread if scan_spread is not None else observed_min_spread
    spread_move_exceeded = bool(
        round_trip_edge is not None
        and effective_spread is not None
        and abs(round_trip_edge) >= effective_spread
    )
    too_large = bool(
        tick_cost.get("blocked") is True
        or (spread_to_mid_ratio is not None and spread_to_mid_ratio >= threshold_ratio)
        or (
            observed_max_spread is not None
            and max_spread_threshold is not None
            and observed_max_spread > max_spread_threshold
        )
        or (commission_to_notional_ratio is not None and commission_to_notional_ratio >= 0.005)
        or spread_move_exceeded
    )
    return {
        "spread_tick_cost_too_large": too_large,
        "scan_mid": scan_mid,
        "scan_spread": scan_spread,
        "observed_min_spread": observed_min_spread,
        "observed_max_spread": observed_max_spread,
        "max_spread_threshold": max_spread_threshold,
        "spread_to_mid_ratio": spread_to_mid_ratio,
        "threshold_ratio": threshold_ratio,
        "commission_total": commission_total,
        "fill_notional": fill_notional,
        "commission_to_notional_ratio": commission_to_notional_ratio,
        "round_trip_move_exceeded_effective_spread": spread_move_exceeded,
    }


def _queue_fill_timing_attribution(
    *,
    diagnostics: dict[str, Any],
    result: dict[str, Any],
    params: dict[str, Any],
    price_move: dict[str, Any],
) -> dict[str, Any]:
    strategy_diagnostics = _safe_mapping_from(diagnostics.get("strategy_diagnostics"))
    entry_block_counts = _safe_mapping_from(strategy_diagnostics.get("entry_block_counts"))
    fills = _parse_float(diagnostics.get("fills"))
    strategy_order_count = _parse_float(diagnostics.get("strategy_order_count"))
    entry_signal_count = _parse_int(strategy_diagnostics.get("entry_signal_count"))
    fill_span_seconds = _parse_float(price_move.get("fill_time_span_seconds"))
    quote_lifetime_seconds = _parse_float(params.get("quote_lifetime_seconds"))
    order_fill_ratio = (
        fills / strategy_order_count
        if fills is not None and strategy_order_count is not None and strategy_order_count > 0
        else None
    )
    pending_order_blocks = _safe_count(entry_block_counts.get("pending_order"))
    cooldown_blocks = _safe_count(entry_block_counts.get("reentry_cooldown_seconds")) + _safe_count(
        entry_block_counts.get("reentry_cooldown_updates")
    )
    suspected = bool(
        (fills or 0.0) > 0
        and (
            pending_order_blocks > 0
            or cooldown_blocks > 0
            or (
                fill_span_seconds is not None
                and quote_lifetime_seconds is not None
                and fill_span_seconds > quote_lifetime_seconds
            )
            or (order_fill_ratio is not None and order_fill_ratio < 1.0)
        )
    )
    return {
        "queue_or_fill_timing_suspected": suspected,
        "fills": fills,
        "strategy_order_count": strategy_order_count,
        "entry_signal_count": entry_signal_count,
        "order_fill_ratio": order_fill_ratio,
        "fill_time_span_seconds": fill_span_seconds,
        "quote_lifetime_seconds": quote_lifetime_seconds,
        "pending_order_block_count": pending_order_blocks,
        "cooldown_block_count": cooldown_blocks,
    }


def _parameter_candidate_bucket(
    *,
    slug: str | None,
    token_index: int | None,
    source_strategy: str | None,
    diagnostics: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "slug": slug,
        "token_index": token_index,
        "source_strategy": source_strategy,
        "tail_bucket": diagnostics.get("tail_bucket") or "unknown",
        "scan_mid": diagnostics.get("scan_mid"),
        "scan_spread": diagnostics.get("scan_spread"),
        "liquidity": diagnostics.get("liquidity"),
        "params": dict(params),
        "bucket_key": (
            f"{diagnostics.get('tail_bucket') or 'unknown'}|"
            f"depth={params.get('depth_levels')}|"
            f"edge={params.get('min_microprice_edge')}|"
            f"entry={params.get('entry_imbalance')}|"
            f"hold={params.get('quote_lifetime_seconds')}"
        ),
    }


def _negative_pnl_attempt_attribution(
    *,
    slug: str | None,
    token_index: int | None,
    source_strategy: str | None,
    status: str,
    params: dict[str, Any],
    result: dict[str, Any] | None,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    result = result or {}
    fills = _parse_float(diagnostics.get("fills"))
    pnl = _parse_float(diagnostics.get("pnl"))
    strategy_order_count = _parse_float(diagnostics.get("strategy_order_count"))
    eligible = bool(
        status == "completed"
        and pnl is not None
        and pnl <= 0
        and ((fills or 0.0) > 0 or (strategy_order_count or 0.0) > 0)
    )
    price_move = _price_move_diagnostics(result)
    spread_tick_cost = _spread_tick_cost_attribution(
        diagnostics=diagnostics, result=result, price_move=price_move
    )
    queue_fill_timing = _queue_fill_timing_attribution(
        diagnostics=diagnostics, result=result, params=params, price_move=price_move
    )
    parameter_bucket = _parameter_candidate_bucket(
        slug=slug,
        token_index=token_index,
        source_strategy=source_strategy,
        diagnostics=diagnostics,
        params=params,
    )
    causes: list[str] = []
    if eligible and price_move["adverse_selection_or_markout_detected"]:
        causes.append("adverse_selection_markout")
    if eligible and spread_tick_cost["spread_tick_cost_too_large"]:
        causes.append("spread_tick_cost_too_large")
    if eligible and queue_fill_timing["queue_or_fill_timing_suspected"]:
        causes.append("queue_fill_timing")
    if eligible:
        causes.append("parameter_candidate_bucket")
    missing: list[str] = []
    if eligible and not price_move["available"]:
        missing.append("markout_or_round_trip_price_move_unavailable")
    if eligible and not diagnostics.get("strategy_diagnostics"):
        missing.append("strategy_diagnostics_unavailable")
    if eligible and diagnostics.get("scan_mid") is None:
        missing.append("scan_mid_unavailable")
    unknown = bool(eligible and not any(cause != "parameter_candidate_bucket" for cause in causes))
    if unknown:
        causes.append("unknown_insufficient_evidence")
    primary_cause = None
    for candidate_cause in (
        "adverse_selection_markout",
        "spread_tick_cost_too_large",
        "queue_fill_timing",
        "unknown_insufficient_evidence",
        "parameter_candidate_bucket",
    ):
        if candidate_cause in causes:
            primary_cause = candidate_cause
            break
    return {
        "schema_version": 1,
        "eligible": eligible,
        "classification": "non_positive_pnl_after_fills" if eligible else "not_applicable",
        "fail_closed": True,
        "primary_cause": primary_cause,
        "causes": causes,
        "cause_counts": {cause: 1 for cause in causes},
        "pnl": pnl,
        "fills": fills,
        "strategy_order_count": strategy_order_count,
        "adverse_selection_markout": price_move,
        "spread_tick_cost": spread_tick_cost,
        "queue_fill_timing": queue_fill_timing,
        "parameter_candidate_bucket": parameter_bucket,
        "unknown_insufficient_evidence": {
            "blocked": unknown,
            "fail_closed": True,
            "missing_evidence": missing,
        },
    }


def _attempt_record(attempt: BacktestAttempt | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(attempt, BacktestAttempt):
        return {
            "slug": attempt.slug,
            "question": attempt.question,
            "token_index": attempt.token_index,
            "source_strategy": attempt.source_strategy,
            "params": attempt.params,
            "status": attempt.status,
            "result": attempt.result,
            "error": attempt.error,
            "diagnostics": attempt.diagnostics,
        }
    return dict(attempt)


def _attribution_from_attempt_record(
    attempt: BacktestAttempt | Mapping[str, Any],
) -> dict[str, Any]:
    record = _attempt_record(attempt)
    diagnostics = _safe_mapping_from(record.get("diagnostics"))
    result = _safe_mapping_from(record.get("result"))
    params = _safe_mapping_from(record.get("params") or diagnostics.get("params"))
    existing = diagnostics.get("negative_pnl_attribution")
    if isinstance(existing, Mapping):
        return dict(existing)
    return _negative_pnl_attempt_attribution(
        slug=record.get("slug") if isinstance(record.get("slug"), str) else None,
        token_index=_parse_int(record.get("token_index")),
        source_strategy=(
            record.get("source_strategy")
            if isinstance(record.get("source_strategy"), str)
            else None
        ),
        status=str(record.get("status") or ""),
        params=params,
        result=result,
        diagnostics=diagnostics,
    )


def _negative_pnl_attribution_summary(
    attempts: list[BacktestAttempt] | list[Mapping[str, Any]],
    fills_orders_pnl: dict[str, Any],
) -> dict[str, Any]:
    completed_pnl_sum = _parse_float(fills_orders_pnl.get("completed_pnl_sum"))
    total_fills = _parse_float(fills_orders_pnl.get("total_fills")) or 0.0
    total_strategy_orders = _parse_float(fills_orders_pnl.get("total_strategy_orders")) or 0.0
    attributions: list[dict[str, Any]] = []
    cause_counts: dict[str, int] = {}
    primary_cause_counts: dict[str, int] = {}
    for attempt in attempts:
        attribution = _attribution_from_attempt_record(attempt)
        if not attribution.get("eligible"):
            continue
        attributions.append(attribution)
        for cause, count in (attribution.get("cause_counts") or {}).items():
            cause_counts[str(cause)] = cause_counts.get(str(cause), 0) + int(count or 0)
        primary_cause = attribution.get("primary_cause")
        if primary_cause:
            key = str(primary_cause)
            primary_cause_counts[key] = primary_cause_counts.get(key, 0) + 1

    aggregate_eligible = bool(
        completed_pnl_sum is not None
        and completed_pnl_sum <= 0
        and (total_fills > 0 or total_strategy_orders > 0)
    )
    if not aggregate_eligible:
        classification = (
            "not_applicable_positive_aggregate_pnl"
            if completed_pnl_sum is not None and completed_pnl_sum > 0
            else "not_applicable_no_orders_or_fills"
        )
    elif attributions:
        classification = "non_positive_pnl_after_orders_or_fills"
    else:
        classification = "unknown_insufficient_evidence_fail_closed"
        cause_counts["unknown_insufficient_evidence"] = (
            cause_counts.get("unknown_insufficient_evidence", 0) + 1
        )

    primary_cause = None
    for candidate_cause in (
        "adverse_selection_markout",
        "spread_tick_cost_too_large",
        "queue_fill_timing",
        "unknown_insufficient_evidence",
        "parameter_candidate_bucket",
    ):
        if cause_counts.get(candidate_cause):
            primary_cause = candidate_cause
            break
    unknown_count = cause_counts.get("unknown_insufficient_evidence", 0)
    return {
        "schema_version": 1,
        "eligible": aggregate_eligible,
        "classification": classification,
        "fail_closed": True,
        "primary_cause": primary_cause,
        "cause_counts": cause_counts,
        "primary_cause_counts": primary_cause_counts,
        "eligible_attempt_count": len(attributions),
        "completed_pnl_sum": completed_pnl_sum,
        "total_fills": total_fills,
        "total_strategy_orders": total_strategy_orders,
        "adverse_selection_markout_available_attempts": sum(
            1
            for attribution in attributions
            if _safe_mapping_from(attribution.get("adverse_selection_markout")).get("available")
        ),
        "spread_tick_cost_too_large_attempts": sum(
            1
            for attribution in attributions
            if _safe_mapping_from(attribution.get("spread_tick_cost")).get(
                "spread_tick_cost_too_large"
            )
        ),
        "queue_fill_timing_suspected_attempts": sum(
            1
            for attribution in attributions
            if _safe_mapping_from(attribution.get("queue_fill_timing")).get(
                "queue_or_fill_timing_suspected"
            )
        ),
        "unknown_insufficient_evidence_attempts": unknown_count,
        "attempts": [
            {
                "slug": _safe_mapping_from(attribution.get("parameter_candidate_bucket")).get(
                    "slug"
                ),
                "token_index": _safe_mapping_from(
                    attribution.get("parameter_candidate_bucket")
                ).get("token_index"),
                "pnl": attribution.get("pnl"),
                "fills": attribution.get("fills"),
                "strategy_order_count": attribution.get("strategy_order_count"),
                "primary_cause": attribution.get("primary_cause"),
                "causes": attribution.get("causes", []),
                "parameter_candidate_bucket": attribution.get("parameter_candidate_bucket"),
            }
            for attribution in attributions
        ],
    }


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
    negative_pnl_attribution = _negative_pnl_attempt_attribution(
        slug=candidate.slug,
        token_index=candidate.token_index,
        source_strategy=candidate.source_strategy,
        status=status,
        params=params,
        result=result,
        diagnostics={
            "scan_mid": mid,
            "scan_spread": spread,
            "liquidity": candidate.liquidity,
            "tail_bucket": tail_bucket,
            "fills": fills,
            "pnl": pnl,
            "strategy_order_count": strategy_order_count,
            "params": params,
            "strategy_diagnostics": strategy_diagnostics,
            "no_order": no_order,
        },
    )
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
        "negative_pnl_attribution": negative_pnl_attribution,
        "suspected_causes": suspected_causes,
    }


def _aggregate_diagnostics(attempts: list[BacktestAttempt]) -> dict[str, Any]:
    completed = [a for a in attempts if a.status == "completed"]
    zero_fill = [a for a in completed if ((a.result or {}).get("fills") in {0, 0.0, None})]
    causes: dict[str, int] = {}
    no_order_cause_counts: dict[str, int] = {}
    no_order_primary_cause_counts: dict[str, int] = {}
    tail_bucket_counts: dict[str, int] = {}
    tick_cost_bucket_counts: dict[str, int] = {}
    total_fills = 0.0
    total_strategy_orders = 0.0
    completed_pnl = 0.0
    completed_pnl_values: list[float] = []
    for attempt in attempts:
        diagnostics = attempt.diagnostics or {}
        for cause in diagnostics.get("suspected_causes", []):
            causes[cause] = causes.get(cause, 0) + 1
        tail_bucket = diagnostics.get("tail_bucket")
        if tail_bucket:
            key = str(tail_bucket)
            tail_bucket_counts[key] = tail_bucket_counts.get(key, 0) + 1
        fills = _parse_float(diagnostics.get("fills"))
        if fills is not None:
            total_fills += fills
        strategy_order_count = _parse_float(diagnostics.get("strategy_order_count"))
        if strategy_order_count is not None:
            total_strategy_orders += strategy_order_count
        pnl = _parse_float(diagnostics.get("pnl"))
        if attempt.status == "completed" and pnl is not None:
            completed_pnl += pnl
            completed_pnl_values.append(pnl)
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
            blockers = no_order.get("blockers")
            tick_cost = blockers.get("tick_cost") if isinstance(blockers, dict) else None
            if isinstance(tick_cost, dict):
                if tick_cost.get("blocked") is True:
                    bucket = "blocked_ge_threshold"
                elif tick_cost.get("spread_to_mid_ratio") is None:
                    bucket = "unknown"
                else:
                    bucket = "not_blocked"
            else:
                bucket = "unknown"
            tick_cost_bucket_counts[bucket] = tick_cost_bucket_counts.get(bucket, 0) + 1
        else:
            tick_cost_bucket_counts["unknown"] = tick_cost_bucket_counts.get("unknown", 0) + 1
    fills_orders_pnl = {
        "total_fills": total_fills,
        "total_strategy_orders": total_strategy_orders,
        "completed_pnl_sum": completed_pnl,
        "completed_positive_pnl_attempts": sum(1 for value in completed_pnl_values if value > 0),
        "completed_negative_pnl_attempts": sum(1 for value in completed_pnl_values if value < 0),
        "completed_zero_pnl_attempts": sum(1 for value in completed_pnl_values if value == 0),
    }
    negative_pnl_summary = _negative_pnl_attribution_summary(attempts, fills_orders_pnl)
    return {
        "completed_attempts": len(completed),
        "zero_fill_completed_attempts": len(zero_fill),
        "suspected_cause_counts": causes,
        "no_order_cause_counts": no_order_cause_counts,
        "no_order_primary_cause_counts": no_order_primary_cause_counts,
        "tail_bucket_counts": tail_bucket_counts,
        "tick_cost_bucket_counts": tick_cost_bucket_counts,
        "fills_orders_pnl": fills_orders_pnl,
        "negative_pnl_attribution_summary": negative_pnl_summary,
        "profit_opportunity_demonstrated": any(
            ((a.result or {}).get("pnl") or 0) > 0 and ((a.result or {}).get("fills") or 0) > 0
            for a in attempts
            if a.status == "completed"
        ),
    }


def _fallback_replay_window(candidate: Candidate, args: argparse.Namespace) -> ReplayWindow:
    return ReplayWindow(
        start_time=candidate.coverage_start_time or args.start_time,
        end_time=candidate.coverage_end_time or args.end_time,
        min_book_events=(
            candidate.coverage_min_book_events
            if candidate.coverage_min_book_events is not None
            else args.min_book_events
        ),
        book_events=candidate.coverage_book_events,
        provenance="candidate_or_cli_fallback",
    )


def _candidate_replay_requests(
    candidate: Candidate, args: argparse.Namespace
) -> list[dict[str, Any]]:
    window_policy = getattr(args, "window_policy", "candidate")
    if window_policy == "all":
        windows = candidate.replay_windows or (_fallback_replay_window(candidate, args),)
    else:
        windows = (
            (candidate.replay_windows[0],)
            if candidate.replay_windows
            else (_fallback_replay_window(candidate, args),)
        )

    requests: list[dict[str, Any]] = []
    for window_index, window in enumerate(windows):
        min_book_events = (
            window.min_book_events
            if window.min_book_events is not None
            else candidate.coverage_min_book_events
            if candidate.coverage_min_book_events is not None
            else args.min_book_events
        )
        requests.append(
            {
                "slug": candidate.slug,
                "token_index": candidate.token_index,
                "source_strategy": candidate.source_strategy,
                "manifest_rank": candidate.manifest_rank,
                "window_index": window_index,
                "window_source": window.source,
                "window_provenance": window.provenance,
                "window": {"start_time": window.start_time, "end_time": window.end_time},
                "min_book_events": min_book_events,
                "coverage_book_events": (
                    window.book_events
                    if window.book_events is not None
                    else candidate.coverage_book_events
                ),
            }
        )
    return requests


def _candidate_replay_request(candidate: Candidate, args: argparse.Namespace) -> dict[str, Any]:
    return _candidate_replay_requests(candidate, args)[0]


def _candidate_from_mapping(raw: dict[str, Any]) -> Candidate:
    replay_windows = tuple(
        ReplayWindow(**window)
        for window in raw.get("replay_windows", ())
        if isinstance(window, dict)
    )
    return Candidate(
        slug=str(raw["slug"]),
        question=str(raw.get("question") or raw["slug"]),
        token_index=int(raw.get("token_index", 0)),
        condition_id=raw.get("condition_id") if isinstance(raw.get("condition_id"), str) else None,
        scan_mid=_parse_float(raw.get("scan_mid")),
        scan_spread=_parse_float(raw.get("scan_spread")),
        scan_imbalance5=_parse_float(raw.get("scan_imbalance5")),
        liquidity=_parse_float(raw.get("liquidity")),
        source_strategy=str(raw.get("source_strategy") or "Microprice"),
        coverage_start_time=raw.get("coverage_start_time")
        if isinstance(raw.get("coverage_start_time"), str)
        else None,
        coverage_end_time=raw.get("coverage_end_time")
        if isinstance(raw.get("coverage_end_time"), str)
        else None,
        coverage_book_events=_parse_int(raw.get("coverage_book_events")),
        coverage_min_book_events=_parse_int(raw.get("coverage_min_book_events")),
        manifest_rank=_parse_int(raw.get("manifest_rank")),
        selection_policy=str(
            raw.get("selection_policy") or "non_extreme_tail_priority_then_liquidity"
        ),
        replay_windows=replay_windows,
    )


def _attempt_from_mapping(raw: dict[str, Any]) -> BacktestAttempt:
    return BacktestAttempt(
        slug=str(raw["slug"]),
        question=str(raw.get("question") or raw["slug"]),
        token_index=int(raw.get("token_index", 0)),
        source_strategy=str(raw.get("source_strategy") or "Microprice"),
        params=dict(raw.get("params") or {}),
        status=str(raw.get("status") or "error"),
        result=raw.get("result") if isinstance(raw.get("result"), dict) else None,
        error=raw.get("error") if isinstance(raw.get("error"), str) else None,
        diagnostics=raw.get("diagnostics") if isinstance(raw.get("diagnostics"), dict) else None,
    )


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
                "request_count": 0,
            },
        )
        unique[key]["request_count"] += 1
        unique[key]["candidate_count"] = len(
            {
                (candidate.get("slug"), candidate.get("token_index"))
                for candidate in requests
                if (
                    candidate["window"].get("start_time"),
                    candidate["window"].get("end_time"),
                    candidate.get("min_book_events"),
                )
                == key
            }
        )
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
    blockers: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    if not replay_requests:
        status = "fail_closed_no_candidates"
        warnings.append("no_candidate_coverage_pass_manifest")
        blockers.append(
            {
                "type": "coverage_pass_manifest_no_candidates",
                "message": "No positive coverage-pass candidate replay requests were available.",
            }
        )
    else:
        selected = selected_windows[0]
        selected_window = dict(selected["window"])
        selected_min_book_events = _parse_int(selected.get("min_book_events"))
        root_window = dict(selected_window)
        root_min_book_events = selected_min_book_events
        expected_keys = {
            (
                window["window"].get("start_time"),
                window["window"].get("end_time"),
                _parse_int(window.get("min_book_events")),
            )
            for window in selected_windows
        }
        mismatches = [
            {
                "slug": attempt.slug,
                "token_index": attempt.token_index,
                "attempt_window": (attempt.diagnostics or {}).get("window"),
                "attempt_min_book_events": (attempt.diagnostics or {}).get("min_book_events"),
            }
            for attempt in attempts
            if _attempt_replay_key(attempt) not in expected_keys
        ]
        if attempts and not mismatches:
            status = "verified"
        else:
            status = "fail_closed"
            warnings.append("exact_window_mismatch")
            warnings.append("exact_window_metadata_blocker")
            blockers.append(
                {
                    "type": "exact_window_metadata_blocker",
                    "message": (
                        "One or more Job B attempts did not report the selected coverage-pass "
                        "manifest replay window and min_book_events."
                    ),
                    "mismatches": mismatches,
                }
            )
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
        "blockers": blockers,
        "mismatches": mismatches,
    }


def _classification_for_exact_window(
    *,
    candidate_count: int,
    exact_window_status: str,
) -> str:
    if candidate_count <= 0 or exact_window_status != "verified":
        return "blocked"
    return "diagnostic_only"


def _canonical_timestamp(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _window_key_from_record(
    record: dict[str, Any],
) -> tuple[str | None, str | None, int | None]:
    window = record.get("window") if isinstance(record.get("window"), dict) else {}
    return (
        _canonical_timestamp(window.get("start_time")),
        _canonical_timestamp(window.get("end_time")),
        _parse_int(record.get("min_book_events")),
    )


def _expected_window_records_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for window in _manifest_replay_windows(payload):
        records.append(
            {
                "window": {
                    "start_time": window.start_time,
                    "end_time": window.end_time,
                },
                "min_book_events": window.min_book_events,
                "window_source": window.source,
                "window_provenance": window.provenance,
                "candidate_count": window.candidate_count,
                "book_events": window.book_events,
            }
        )
    return records


def _add_artifact_window_record(
    records: list[dict[str, Any]],
    *,
    window: Any,
    min_book_events: Any,
    source: str,
    slug: str | None = None,
    token_index: int | None = None,
) -> None:
    if not isinstance(window, dict):
        return
    start_time = window.get("start_time")
    end_time = window.get("end_time")
    if not isinstance(start_time, str) or not isinstance(end_time, str):
        return
    record: dict[str, Any] = {
        "source": source,
        "window": {"start_time": start_time, "end_time": end_time},
        "min_book_events": _parse_int(min_book_events),
    }
    if slug is not None:
        record["slug"] = slug
    if token_index is not None:
        record["token_index"] = token_index
    records.append(record)


def _artifact_window_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    _add_artifact_window_record(
        records,
        window=payload.get("window"),
        min_book_events=payload.get("min_book_events"),
        source="artifact.window",
    )
    _add_artifact_window_record(
        records,
        window=payload.get("selected_window"),
        min_book_events=payload.get("selected_min_book_events"),
        source="artifact.selected_window",
    )
    selected_windows = payload.get("selected_windows")
    if isinstance(selected_windows, list):
        for index, selected in enumerate(selected_windows):
            if not isinstance(selected, dict):
                continue
            _add_artifact_window_record(
                records,
                window=selected.get("window"),
                min_book_events=selected.get("min_book_events"),
                source=f"artifact.selected_windows[{index}]",
            )
    attempts = payload.get("attempts")
    if isinstance(attempts, list):
        for index, attempt in enumerate(attempts):
            if not isinstance(attempt, dict):
                continue
            diagnostics = (
                attempt.get("diagnostics") if isinstance(attempt.get("diagnostics"), dict) else {}
            )
            _add_artifact_window_record(
                records,
                window=diagnostics.get("window"),
                min_book_events=diagnostics.get("min_book_events"),
                source=f"artifact.attempts[{index}].diagnostics",
                slug=attempt.get("slug") if isinstance(attempt.get("slug"), str) else None,
                token_index=_parse_int(attempt.get("token_index")),
            )
    return records


def build_exact_window_validation_report(
    *,
    manifest_path: Path,
    artifact_path: Path,
    command: list[str],
    manifest_selection: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, dict):
        raise TypeError("manifest must be a JSON object")
    if not isinstance(artifact_payload, dict):
        raise TypeError("artifact must be a JSON object")

    expected_records = _expected_window_records_from_payload(manifest_payload)
    actual_records = _artifact_window_records(artifact_payload)
    expected_keys = {_window_key_from_record(record) for record in expected_records}
    actual_mismatches = [
        record for record in actual_records if _window_key_from_record(record) not in expected_keys
    ]
    missing_expected_metadata = [
        record
        for record in expected_records
        if _window_key_from_record(record)[0] is None
        or _window_key_from_record(record)[1] is None
        or _window_key_from_record(record)[2] is None
    ]
    candidate_count = _declared_candidate_count(manifest_path)
    warnings: list[str] = []
    blockers: list[dict[str, Any]] = []

    if candidate_count <= 0:
        warnings.append("coverage_pass_manifest_no_pass")
        blockers.append(
            {
                "type": "coverage_pass_manifest_no_pass",
                "message": "Selected coverage-pass manifest declares zero candidates.",
            }
        )
    if not expected_records or missing_expected_metadata:
        warnings.append("selected_pass_window_metadata_missing")
        blockers.append(
            {
                "type": "selected_pass_window_metadata_missing",
                "message": (
                    "Selected coverage-pass manifest does not declare a complete "
                    "window/start/end/min_book_events tuple."
                ),
                "records": missing_expected_metadata,
            }
        )
    if not actual_records:
        warnings.append("job_b_artifact_window_metadata_missing")
        blockers.append(
            {
                "type": "job_b_artifact_window_metadata_missing",
                "message": "Job B artifact does not expose comparable replay window metadata.",
            }
        )
    if actual_mismatches:
        warnings.extend(["exact_window_mismatch", "exact_window_metadata_blocker"])
        blockers.append(
            {
                "type": "exact_window_metadata_blocker",
                "message": (
                    "Job B artifact replay window/min_book_events differs from the "
                    "selected coverage-pass manifest."
                ),
                "mismatches": actual_mismatches,
            }
        )

    exact_window_status = "verified" if not blockers else "fail_closed"
    classification = _classification_for_exact_window(
        candidate_count=candidate_count,
        exact_window_status=exact_window_status,
    )
    artifact_fills_orders_pnl = (
        artifact_payload.get("fills_orders_pnl")
        if isinstance(artifact_payload.get("fills_orders_pnl"), dict)
        else {}
    )
    artifact_negative_pnl_summary = artifact_payload.get("negative_pnl_attribution_summary")
    if not isinstance(artifact_negative_pnl_summary, dict):
        attempts = (
            artifact_payload.get("attempts")
            if isinstance(artifact_payload.get("attempts"), list)
            else []
        )
        artifact_negative_pnl_summary = _negative_pnl_attribution_summary(
            [attempt for attempt in attempts if isinstance(attempt, Mapping)],
            artifact_fills_orders_pnl,
        )
    return {
        "generated_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
        "mode": SAFETY_MODE,
        "classification": classification,
        "live_ready": False,
        "no_profit_claim": True,
        "exact_window_status": exact_window_status,
        "pass_manifest_status": "pass" if candidate_count > 0 else "no_pass",
        "manifest": str(manifest_path),
        "artifact": str(artifact_path),
        "negative_pnl_attribution_summary": artifact_negative_pnl_summary,
        "manifest_selection": manifest_selection
        or [{"path": str(manifest_path), "selected": True, "reason": "explicit_manifest"}],
        "expected_selected_windows": expected_records,
        "artifact_windows": actual_records,
        "warnings": warnings,
        "blockers": blockers,
        "commands": [" ".join(command)],
        "safety": {
            "live_trading": False,
            "orders_submitted": False,
            "orders_signed": False,
            "orders_cancelled": False,
            "credentials_required": False,
            "worker_trading_started": False,
            "live_trading_worker_started": False,
        },
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


async def _run_attempt_with_timeout(
    candidate: Candidate,
    params: dict[str, Any],
    *,
    replay_request: dict[str, Any],
    timeout_seconds: int,
) -> BacktestAttempt:
    replay_window = replay_request["window"]
    min_book_events = int(replay_request["min_book_events"])
    try:
        return await asyncio.wait_for(
            run_attempt(
                candidate,
                params,
                start_time=replay_window["start_time"],
                end_time=replay_window["end_time"],
                min_book_events=min_book_events,
            ),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        error = f"TimeoutError: timed out after {timeout_seconds} seconds"
        return BacktestAttempt(
            slug=candidate.slug,
            question=candidate.question,
            token_index=candidate.token_index,
            source_strategy=candidate.source_strategy,
            params=params,
            status="error",
            result=None,
            error=error,
            diagnostics=_diagnose_attempt(
                candidate,
                params,
                None,
                start_time=replay_window["start_time"],
                end_time=replay_window["end_time"],
                min_book_events=min_book_events,
                status="error",
                error=error,
            ),
        )


def _attempt_error(
    candidate: Candidate,
    params: dict[str, Any],
    *,
    replay_request: dict[str, Any],
    error: str,
) -> BacktestAttempt:
    replay_window = replay_request["window"]
    min_book_events = int(replay_request["min_book_events"])
    return BacktestAttempt(
        slug=candidate.slug,
        question=candidate.question,
        token_index=candidate.token_index,
        source_strategy=candidate.source_strategy,
        params=params,
        status="error",
        result=None,
        error=error,
        diagnostics=_diagnose_attempt(
            candidate,
            params,
            None,
            start_time=replay_window["start_time"],
            end_time=replay_window["end_time"],
            min_book_events=min_book_events,
            status="error",
            error=error,
        ),
    )


def _decode_subprocess_output(data: bytes, *, limit: int = 4000) -> str:
    text = data.decode("utf-8", errors="replace").strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


async def _run_attempt_in_subprocess(
    candidate: Candidate,
    params: dict[str, Any],
    *,
    replay_request: dict[str, Any],
    timeout_seconds: int,
) -> BacktestAttempt:
    payload = {
        "candidate": asdict(candidate),
        "params": params,
        "replay_request": replay_request,
    }
    try:
        with tempfile.TemporaryDirectory(prefix="job_b_microprice_attempt_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            request_path = tmp_path / "request.json"
            output_path = tmp_path / "attempt.json"
            request_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(Path(__file__).resolve()),
                "--single-attempt-input",
                str(request_path),
                "--single-attempt-output",
                str(output_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout_seconds
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                return _attempt_error(
                    candidate,
                    params,
                    replay_request=replay_request,
                    error=f"TimeoutError: isolated attempt timed out after {timeout_seconds} seconds",
                )
            if process.returncode != 0:
                stdout_text = _decode_subprocess_output(stdout)
                stderr_text = _decode_subprocess_output(stderr)
                return _attempt_error(
                    candidate,
                    params,
                    replay_request=replay_request,
                    error=(
                        f"SubprocessError: isolated attempt exited {process.returncode}; "
                        f"stdout={stdout_text!r}; stderr={stderr_text!r}"
                    ),
                )
            if not output_path.exists():
                return _attempt_error(
                    candidate,
                    params,
                    replay_request=replay_request,
                    error="SubprocessError: isolated attempt produced no output JSON",
                )
            raw_attempt = json.loads(output_path.read_text(encoding="utf-8"))
            if not isinstance(raw_attempt, dict):
                return _attempt_error(
                    candidate,
                    params,
                    replay_request=replay_request,
                    error="SubprocessError: isolated attempt output was not an object",
                )
            return _attempt_from_mapping(raw_attempt)
    except Exception as exc:
        return _attempt_error(
            candidate,
            params,
            replay_request=replay_request,
            error=f"{type(exc).__name__}: isolated attempt wrapper failed: {exc}",
        )


def _run_single_attempt_cli(input_path: Path, output_path: Path) -> int:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("single-attempt payload must be a JSON object")
    raw_candidate = payload.get("candidate")
    replay_request = payload.get("replay_request")
    if not isinstance(raw_candidate, dict):
        raise TypeError("single-attempt payload missing candidate object")
    if not isinstance(replay_request, dict):
        raise TypeError("single-attempt payload missing replay_request object")
    candidate = _candidate_from_mapping(raw_candidate)
    params = dict(payload.get("params") or {})
    replay_window = replay_request["window"]
    attempt = asyncio.run(
        run_attempt(
            candidate,
            params,
            start_time=replay_window["start_time"],
            end_time=replay_window["end_time"],
            min_book_events=int(replay_request["min_book_events"]),
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(attempt), indent=2, sort_keys=True), encoding="utf-8")
    return 0


async def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    candidates = load_candidates(
        args.manifest, strategy=args.strategy, max_candidates=args.max_candidates
    )
    selected_params = microprice_param_grid()[: args.max_param_sets]
    attempts: list[BacktestAttempt] = []
    candidate_requests = [
        (candidate, replay_request)
        for candidate in candidates
        for replay_request in _candidate_replay_requests(candidate, args)
    ]
    replay_requests = [replay_request for _, replay_request in candidate_requests]
    attempt_isolation = getattr(args, "attempt_isolation", "inline")
    for candidate, replay_request in candidate_requests:
        for params in selected_params:
            runner = (
                _run_attempt_in_subprocess
                if attempt_isolation == "subprocess"
                else _run_attempt_with_timeout
            )
            attempts.append(
                await runner(
                    candidate,
                    params,
                    replay_request=replay_request,
                    timeout_seconds=args.per_attempt_timeout_secs,
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
    aggregate_diagnostics = _aggregate_diagnostics(attempts)
    classification = _classification_for_exact_window(
        candidate_count=len(candidates),
        exact_window_status=str(exact_window["status"]),
    )
    return {
        "generated_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
        "mode": SAFETY_MODE,
        "classification": classification,
        "live_ready": False,
        "no_profit_claim": True,
        "pass_manifest_status": "pass" if candidates else "no_pass",
        "safety": {
            "live_trading": False,
            "orders_submitted": False,
            "orders_signed": False,
            "orders_cancelled": False,
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
        "blockers": exact_window["blockers"],
        "exact_window": exact_window,
        "attempt_isolation": attempt_isolation,
        "candidate_count": len(candidates),
        "parameter_set_count": len(selected_params),
        "attempt_count": len(attempts),
        "completed_count": completed,
        "skipped_count": skipped,
        "skipped_no_coverage_count": skipped,
        "error_count": errored,
        "completed": completed,
        "skipped": skipped,
        "skipped_no_coverage": skipped,
        "errors": errored,
        "fills_orders_pnl": aggregate_diagnostics["fills_orders_pnl"],
        "negative_pnl_attribution_summary": aggregate_diagnostics[
            "negative_pnl_attribution_summary"
        ],
        "tail_bucket_counts": aggregate_diagnostics["tail_bucket_counts"],
        "tick_cost_buckets": aggregate_diagnostics["tick_cost_bucket_counts"],
        "candidate_selection": {
            "policy": "non_extreme_tail_priority_then_liquidity",
            "preferred_yes_mid_range": [0.005, 0.25],
            "ultra_low_tail_retained_but_deprioritized": True,
            "window_policy": getattr(args, "window_policy", "candidate"),
        },
        "candidate_replay_requests": replay_requests,
        "candidates": [asdict(candidate) for candidate in candidates],
        "diagnostics": aggregate_diagnostics,
        "attempts": [asdict(attempt) for attempt in attempts],
    }


def _write_outputs(summary: dict[str, Any], output_dir: Path, timestamp: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"job_B_microprice_batch_{timestamp}.json"
    csv_path = output_dir / f"job_B_microprice_batch_{timestamp}.csv"
    md_path = output_dir / f"job_B_microprice_batch_{timestamp}.md"
    output_files = {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}
    json_summary = dict(summary)
    json_summary["output_files"] = output_files
    json_path.write_text(json.dumps(json_summary, indent=2, sort_keys=True), encoding="utf-8")
    attempts = summary.get("attempts", [])
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "slug",
                "token_index",
                "source_strategy",
                "status",
                "window",
                "min_book_events",
                "params",
                "pnl",
                "fills",
                "book_events",
                "strategy_order_count",
                "tail_bucket",
                "tick_cost_blocked",
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
                    "window": json.dumps(diagnostics.get("window"), sort_keys=True),
                    "min_book_events": diagnostics.get("min_book_events"),
                    "params": json.dumps(attempt.get("params", {}), sort_keys=True),
                    "pnl": result.get("pnl"),
                    "fills": result.get("fills"),
                    "book_events": result.get("book_events"),
                    "strategy_order_count": diagnostics.get("strategy_order_count"),
                    "tail_bucket": diagnostics.get("tail_bucket"),
                    "tick_cost_blocked": (
                        ((no_order.get("blockers") or {}).get("tick_cost") or {}).get("blocked")
                    ),
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
        f"- classification: {summary.get('classification')}",
        f"- live_ready: {str(summary.get('live_ready')).lower()}",
        f"- no_profit_claim: {str(summary.get('no_profit_claim')).lower()}",
        f"- pass_manifest_status: {summary.get('pass_manifest_status')}",
        f"- manifest: {summary['manifest']}",
        f"- window: {summary['window']['start_time']} -> {summary['window']['end_time']}",
        f"- min_book_events: {summary.get('min_book_events')}",
        f"- requested_window: {summary['requested_window']['start_time']} -> {summary['requested_window']['end_time']}",
        f"- requested_min_book_events: {summary.get('requested_min_book_events')}",
        f"- selected_window: {json.dumps(summary.get('selected_window'), sort_keys=True)}",
        f"- selected_min_book_events: {summary.get('selected_min_book_events')}",
        f"- exact_window_status: {summary.get('exact_window_status')}",
        f"- warnings: {json.dumps(summary.get('warnings', []), sort_keys=True)}",
        f"- blockers: {json.dumps(summary.get('blockers', []), sort_keys=True)}",
        f"- attempt_isolation: {summary.get('attempt_isolation')}",
        f"- safety: {json.dumps(summary.get('safety', {}), sort_keys=True)}",
        f"- orders_submitted={str(summary.get('safety', {}).get('orders_submitted')).lower()}",
        f"- orders_signed={str(summary.get('safety', {}).get('orders_signed')).lower()}",
        f"- orders_cancelled={str(summary.get('safety', {}).get('orders_cancelled')).lower()}",
        f"- credentials_required={str(summary.get('safety', {}).get('credentials_required')).lower()}",
        f"- live_trading_worker_started={str(summary.get('safety', {}).get('live_trading_worker_started')).lower()}",
        f"- worker_trading_started={str(summary.get('safety', {}).get('worker_trading_started')).lower()}",
        f"- candidates: {summary['candidate_count']}",
        f"- window_policy: {summary.get('candidate_selection', {}).get('window_policy')}",
        f"- parameter_sets: {summary['parameter_set_count']}",
        f"- attempts: {summary['attempt_count']}",
        f"- completed: {summary.get('completed', summary['completed_count'])}",
        f"- skipped: {summary.get('skipped', summary.get('skipped_count'))}",
        f"- skipped_no_coverage: {summary.get('skipped_no_coverage', summary['skipped_no_coverage_count'])}",
        f"- errors: {summary.get('errors', summary['error_count'])}",
        f"- fills_orders_pnl: {json.dumps(summary.get('fills_orders_pnl', {}), sort_keys=True)}",
        f"- negative_pnl_attribution_summary: {json.dumps(summary.get('negative_pnl_attribution_summary', {}), sort_keys=True)}",
        f"- tail_bucket_counts: {json.dumps(summary.get('tail_bucket_counts', {}), sort_keys=True)}",
        f"- tick_cost_buckets: {json.dumps(summary.get('tick_cost_buckets', {}), sort_keys=True)}",
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
    return output_files


def _write_validation_outputs(
    report: dict[str, Any], output_dir: Path, timestamp: str
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"job_B_microprice_exact_window_validation_{timestamp}.json"
    md_path = output_dir / f"job_B_microprice_exact_window_validation_{timestamp}.md"
    output_files = {"json": str(json_path), "markdown": str(md_path)}
    json_report = dict(report)
    json_report["output_files"] = output_files
    json_path.write_text(json.dumps(json_report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Job B Exact-Window Validation",
        "",
        f"- generated_at_utc: {report.get('generated_at_utc')}",
        f"- mode: {report.get('mode')}",
        f"- classification: {report.get('classification')}",
        f"- live_ready: {str(report.get('live_ready')).lower()}",
        f"- no_profit_claim: {str(report.get('no_profit_claim')).lower()}",
        f"- exact_window_status: {report.get('exact_window_status')}",
        f"- pass_manifest_status: {report.get('pass_manifest_status')}",
        f"- manifest: {report.get('manifest')}",
        f"- artifact: {report.get('artifact')}",
        f"- expected_selected_windows: {json.dumps(report.get('expected_selected_windows', []), sort_keys=True)}",
        f"- artifact_windows: {json.dumps(report.get('artifact_windows', []), sort_keys=True)}",
        f"- warnings: {json.dumps(report.get('warnings', []), sort_keys=True)}",
        f"- blockers: {json.dumps(report.get('blockers', []), sort_keys=True)}",
        f"- commands: {json.dumps(report.get('commands', []), sort_keys=True)}",
        f"- safety: {json.dumps(report.get('safety', {}), sort_keys=True)}",
        f"- orders_submitted={str(report.get('safety', {}).get('orders_submitted')).lower()}",
        f"- orders_signed={str(report.get('safety', {}).get('orders_signed')).lower()}",
        f"- orders_cancelled={str(report.get('safety', {}).get('orders_cancelled')).lower()}",
        f"- credentials_required={str(report.get('safety', {}).get('credentials_required')).lower()}",
        f"- live_trading_worker_started={str(report.get('safety', {}).get('live_trading_worker_started')).lower()}",
        f"- worker_trading_started={str(report.get('safety', {}).get('worker_trading_started')).lower()}",
        f"- negative_pnl_attribution_summary: {json.dumps(report.get('negative_pnl_attribution_summary', {}), sort_keys=True)}",
        "",
        "Safety: backtest/shadow only; no live trading, signing, cancellation, order submission, credentials, or worker-trading.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Job B bounded PMBT microprice batch runner (backtest-only)."
    )
    parser.add_argument("--single-attempt-input", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-attempt-output", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--validate-artifact",
        type=Path,
        default=None,
        help=(
            "Validate an existing Job B artifact against the selected coverage-pass "
            "manifest window without running a backtest."
        ),
    )
    parser.add_argument(
        "--manifest-glob",
        default=DEFAULT_PASS_MANIFEST_GLOB,
        help="Pass-manifest glob used when --manifest is omitted; zero-candidate files are skipped.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--strategy", default="microprice_optimizer")
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--max-param-sets", type=int, default=2)
    parser.add_argument("--start-time", default=None)
    parser.add_argument("--end-time", default=None)
    parser.add_argument("--lookback-hours", type=int, default=1)
    parser.add_argument("--min-book-events", type=int, default=100)
    parser.add_argument("--per-attempt-timeout-secs", type=int, default=120)
    parser.add_argument(
        "--window-policy",
        choices=("candidate", "all"),
        default="candidate",
        help=(
            "candidate runs each candidate's primary explicit manifest/coverage window; "
            "all runs every explicit manifest/coverage-first guidance window."
        ),
    )
    parser.add_argument(
        "--attempt-isolation",
        choices=("subprocess", "inline"),
        default="subprocess",
        help=(
            "Run each candidate/parameter attempt in a fresh Python subprocess by default "
            "so Nautilus/Rust logging is initialized once per process."
        ),
    )
    args = parser.parse_args()

    if args.single_attempt_input is not None or args.single_attempt_output is not None:
        if args.single_attempt_input is None or args.single_attempt_output is None:
            parser.error(
                "--single-attempt-input and --single-attempt-output must be provided together"
            )
        return _run_single_attempt_cli(args.single_attempt_input, args.single_attempt_output)

    if args.output_dir is None:
        parser.error("--output-dir is required")
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
                        "classification": "blocked",
                        "error": "no_non_empty_pass_manifest",
                        "manifest_glob": args.manifest_glob,
                        "manifest_selection": selection_records,
                        "pass_manifest_status": "no_pass",
                        "safety": {
                            "live_trading": False,
                            "orders_submitted": False,
                            "orders_signed": False,
                            "orders_cancelled": False,
                            "credentials_required": False,
                            "worker_trading_started": False,
                            "live_trading_worker_started": False,
                        },
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
    if args.validate_artifact is not None:
        timestamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
        report = build_exact_window_validation_report(
            manifest_path=args.manifest,
            artifact_path=args.validate_artifact,
            command=[sys.executable, *sys.argv],
            manifest_selection=args.manifest_selection,
        )
        report["output_files"] = _write_validation_outputs(report, args.output_dir, timestamp)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["classification"] != "blocked" else 2
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
    return 0 if summary["candidate_count"] > 0 and summary["classification"] != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
