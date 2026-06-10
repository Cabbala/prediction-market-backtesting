from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import multiprocessing as mp
import queue as queue_module
import traceback
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, datetime, timedelta
from glob import glob
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    from _script_helpers import ensure_repo_root
else:
    from ._script_helpers import ensure_repo_root

ensure_repo_root(__file__)

from prediction_market_extensions.adapters.prediction_market import ReplayLoadRequest  # noqa: E402
from prediction_market_extensions.backtesting.data_sources import replay_adapters as replay_module  # noqa: E402
from prediction_market_extensions.backtesting.data_sources.replay_adapters import (  # noqa: E402
    PolymarketPMXTBookReplayAdapter,
)
from scripts.job_b_microprice_batch import (  # noqa: E402
    DEFAULT_SOURCES as JOB_B_DEFAULT_SOURCES,
)
from scripts.job_b_microprice_batch import (  # noqa: E402
    SAFETY_MODE,
    Candidate,
    load_candidates,
)

DEFAULT_OUTPUT_DIR = Path("/opt/polymarket-lab/reports/backtests/pmxt-coverage")
DEFAULT_PASS_MANIFEST_DIR = Path("/opt/polymarket-lab/autoresearch/backtests")
DEFAULT_SOURCES = JOB_B_DEFAULT_SOURCES
REQUIRED_SAFETY_FIELDS = (
    "orders_submitted",
    "orders_signed",
    "orders_cancelled",
    "credentials_required",
    "live_trading_worker_started",
    "worker_trading_started",
)
DEFAULT_PROCESS_TIMEOUT_GRACE_SECONDS = 1


@dataclass(frozen=True)
class CoverageProbeResult:
    slug: str
    market_slug: str
    question: str
    token_index: int
    source_strategy: str
    status: str
    book_events: int
    min_book_events: int
    count_key: str | None = None
    market_key: str | None = None
    market_id: str | None = None
    price_min: float | None = None
    price_max: float | None = None
    price_range: float | None = None
    message: str | None = None
    gap_hours_missing: int = 0
    gap_warning: str | None = None
    candidate: dict[str, Any] | None = None
    window_start_time: str | None = None
    window_end_time: str | None = None
    diagnostic_category: str = "unknown"
    selection_phase: str = "primary_exact"
    source_manifest: str | None = None
    manifest_rank: int | None = None
    window_source: str | None = None
    window_provenance: str | None = None


@dataclass(frozen=True)
class ProbeTarget:
    candidate: Candidate
    start_time: str
    end_time: str
    min_book_events: int
    source_manifest: str
    selection_phase: str
    window_source: str | None = None
    window_provenance: str | None = None


def _utc_now() -> datetime:
    return datetime.now(tz=UTC).replace(microsecond=0)


def _normalize_iso8601(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a valid ISO timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _timestamp(value: str | None) -> str:
    return value or _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _finite_prices(prices: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(price) for price in prices if math.isfinite(float(price)))


def _safety_fields(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    safety = {
        "live_trading": False,
        "strategy_execution": False,
        **{field: False for field in REQUIRED_SAFETY_FIELDS},
    }
    if extra:
        safety.update(extra)
        for field in REQUIRED_SAFETY_FIELDS:
            safety[field] = False
    return safety


def _safety_top_level_fields(extra: dict[str, Any] | None = None) -> dict[str, bool]:
    safety = _safety_fields(extra)
    return {field: bool(safety[field]) for field in REQUIRED_SAFETY_FIELDS}


def _looks_like_pmxt_raw_download_failure(message: str | None) -> bool:
    if not message:
        return False
    text = message.lower()
    source_hint = any(
        hint in text
        for hint in (
            "pmxt",
            "r2.pmxt",
            "r2v2.pmxt",
            "archive.pmxt",
            "parquet",
            "raw",
        )
    )
    failure_hint = any(
        hint in text
        for hint in (
            "download",
            "http",
            "urlopen",
            "connection",
            "connectionreset",
            "connection refused",
            "temporary failure",
            "timed out",
            "timeout",
            "403",
            "404",
            "429",
            "500",
            "502",
            "503",
            "504",
        )
    )
    return source_hint and failure_hint


def _diagnostic_category(
    *,
    status: str,
    book_events: int,
    min_book_events: int,
    message: str | None,
    gap_hours_missing: int,
) -> str:
    if status == "pass":
        return "successful_pass_count"
    if _looks_like_pmxt_raw_download_failure(message):
        return "pmxt_raw_download_failure"
    if gap_hours_missing:
        return "pmxt_missing_hour_gap"
    if status == "no_coverage" and book_events <= 0:
        return "no_pmxt_l2_book_data"
    if status == "no_coverage" and book_events < min_book_events:
        return "min_book_events_not_met"
    if status == "error":
        return "probe_error"
    return "unknown"


def _candidate_payload(candidate: Candidate) -> dict[str, Any]:
    return {
        "slug": candidate.slug,
        "market_slug": candidate.slug,
        "question": candidate.question,
        "token_index": candidate.token_index,
        "condition_id": candidate.condition_id,
        "scan_mid": candidate.scan_mid,
        "scan_spread": candidate.scan_spread,
        "scan_imbalance5": candidate.scan_imbalance5,
        "liquidity": candidate.liquidity,
        "source_strategy": candidate.source_strategy,
        "manifest_rank": candidate.manifest_rank,
    }


@contextmanager
def _coverage_only_replay_load():  # type: ignore[no-untyped-def]
    original_load_trade_ticks = replay_module._load_trade_ticks

    async def _skip_trade_ticks(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return ()

    replay_module._load_trade_ticks = _skip_trade_ticks
    try:
        yield
    finally:
        replay_module._load_trade_ticks = original_load_trade_ticks


def _result_from_candidate(
    candidate: Candidate,
    *,
    status: str,
    book_events: int,
    min_book_events: int,
    count_key: str | None = None,
    market_key: str | None = None,
    market_id: str | None = None,
    prices: tuple[float, ...] = (),
    message: str | None = None,
    gap_hours_missing: int = 0,
    gap_warning: str | None = None,
    window_start_time: str | None = None,
    window_end_time: str | None = None,
    diagnostic_category: str | None = None,
    selection_phase: str = "primary_exact",
    source_manifest: str | None = None,
    window_source: str | None = None,
    window_provenance: str | None = None,
) -> CoverageProbeResult:
    finite_prices = _finite_prices(prices)
    price_min = min(finite_prices) if finite_prices else None
    price_max = max(finite_prices) if finite_prices else None
    category = diagnostic_category or _diagnostic_category(
        status=status,
        book_events=int(book_events),
        min_book_events=int(min_book_events),
        message=message,
        gap_hours_missing=int(gap_hours_missing),
    )
    return CoverageProbeResult(
        slug=candidate.slug,
        market_slug=candidate.slug,
        question=candidate.question,
        token_index=candidate.token_index,
        source_strategy=candidate.source_strategy,
        status=status,
        book_events=int(book_events),
        min_book_events=int(min_book_events),
        count_key=count_key,
        market_key=market_key,
        market_id=market_id,
        price_min=price_min,
        price_max=price_max,
        price_range=(price_max - price_min)
        if price_min is not None and price_max is not None
        else None,
        message=message,
        gap_hours_missing=int(gap_hours_missing),
        gap_warning=gap_warning,
        candidate=_candidate_payload(candidate),
        window_start_time=window_start_time,
        window_end_time=window_end_time,
        diagnostic_category=category,
        selection_phase=selection_phase,
        source_manifest=source_manifest,
        manifest_rank=candidate.manifest_rank,
        window_source=window_source,
        window_provenance=window_provenance,
    )


def _coverage_payload(result: CoverageProbeResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "diagnostic_category": result.diagnostic_category,
        "selection_phase": result.selection_phase,
        "source_manifest": result.source_manifest,
        "window": {
            "start_time": result.window_start_time,
            "end_time": result.window_end_time,
        },
        "window_source": result.window_source,
        "window_provenance": result.window_provenance,
        "book_events": result.book_events,
        "min_book_events": result.min_book_events,
        "count_key": result.count_key,
        "market_key": result.market_key,
        "market_id": result.market_id,
        "price_min": result.price_min,
        "price_max": result.price_max,
        "price_range": result.price_range,
        "message": result.message,
        "gap_hours_missing": result.gap_hours_missing,
        "gap_warning": result.gap_warning,
    }


def _summary_result_row(result: CoverageProbeResult) -> dict[str, Any]:
    row = asdict(result)
    row["coverage"] = _coverage_payload(result)
    row["safety"] = _safety_fields()
    row.update(_safety_top_level_fields())
    return row


def _coverage_result_from_row(row: dict[str, Any]) -> CoverageProbeResult:
    field_names = {field.name for field in fields(CoverageProbeResult)}
    return CoverageProbeResult(**{key: value for key, value in row.items() if key in field_names})


def _timeout_result_from_target(
    target: ProbeTarget,
    *,
    timeout_seconds: int,
    message: str | None = None,
) -> CoverageProbeResult:
    return _result_from_candidate(
        target.candidate,
        status="error",
        book_events=0,
        min_book_events=target.min_book_events,
        message=message or f"Timed out after {timeout_seconds} seconds.",
        diagnostic_category="probe_runtime_timeout",
        selection_phase=target.selection_phase,
        source_manifest=target.source_manifest,
        window_start_time=target.start_time,
        window_end_time=target.end_time,
        window_source=target.window_source,
        window_provenance=target.window_provenance,
    )


async def probe_candidate(
    candidate: Candidate,
    *,
    start_time: str,
    end_time: str,
    min_book_events: int,
    sources: tuple[str, ...],
) -> CoverageProbeResult:
    adapter = PolymarketPMXTBookReplayAdapter()
    replay = adapter.build_single_market_replay(
        field_values={
            "market_slug": candidate.slug,
            "token_index": candidate.token_index,
            "start_time": start_time,
            "end_time": end_time,
        }
    )
    request = ReplayLoadRequest(
        min_record_count=0,
        min_price_range=0.0,
        default_start_time=start_time,
        default_end_time=end_time,
    )

    caught_gap_warning: str | None = None
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", UserWarning)
            with adapter.configure_sources(sources=sources), _coverage_only_replay_load():
                loaded = await adapter.load_replay(replay, request=request)
        for caught in caught_warnings:
            text = str(caught.message)
            if "PMXT:" in text and "archive hour(s) missing" in text:
                caught_gap_warning = text
                break
    except Exception as exc:  # bounded coverage probe: record and keep scanning
        return _result_from_candidate(
            candidate,
            status="error",
            book_events=0,
            min_book_events=min_book_events,
            message=f"{type(exc).__name__}: {exc}",
            window_start_time=start_time,
            window_end_time=end_time,
        )

    if loaded is None:
        return _result_from_candidate(
            candidate,
            status="no_coverage",
            book_events=0,
            min_book_events=min_book_events,
            message="No PMXT L2 book replay was loaded for the requested window.",
            window_start_time=start_time,
            window_end_time=end_time,
        )

    coverage = loaded.coverage_stats
    book_events = int(coverage.count)
    gap_hours_missing = 0
    if caught_gap_warning is not None:
        try:
            gap_hours_missing = int(
                caught_gap_warning.split("PMXT:", 1)[1].split("archive", 1)[0].strip()
            )
        except (IndexError, ValueError):
            gap_hours_missing = 1
    status = "pass" if book_events >= min_book_events and gap_hours_missing == 0 else "no_coverage"
    if status == "pass":
        message = None
    elif gap_hours_missing:
        message = f"{gap_hours_missing} PMXT archive hour(s) missing in requested/load window"
    else:
        message = f"{book_events} book events < {min_book_events} required"
    return _result_from_candidate(
        candidate,
        status=status,
        book_events=book_events,
        min_book_events=min_book_events,
        count_key=coverage.count_key,
        market_key=coverage.market_key,
        market_id=coverage.market_id,
        prices=coverage.prices,
        message=message,
        gap_hours_missing=gap_hours_missing,
        gap_warning=caught_gap_warning,
        window_start_time=start_time,
        window_end_time=end_time,
    )


async def _probe_with_timeout(
    candidate: Candidate,
    *,
    start_time: str,
    end_time: str,
    min_book_events: int,
    sources: tuple[str, ...],
    timeout_seconds: int,
    selection_phase: str = "primary_exact",
    source_manifest: str | None = None,
    window_source: str | None = None,
    window_provenance: str | None = None,
) -> CoverageProbeResult:
    try:
        result = await asyncio.wait_for(
            probe_candidate(
                candidate,
                start_time=start_time,
                end_time=end_time,
                min_book_events=min_book_events,
                sources=sources,
            ),
            timeout=timeout_seconds,
        )
        return replace(
            result,
            window_start_time=result.window_start_time or start_time,
            window_end_time=result.window_end_time or end_time,
            selection_phase=selection_phase,
            source_manifest=source_manifest,
            manifest_rank=candidate.manifest_rank,
            window_source=window_source,
            window_provenance=window_provenance,
            diagnostic_category=(
                result.diagnostic_category
                if result.diagnostic_category != "unknown"
                else _diagnostic_category(
                    status=result.status,
                    book_events=result.book_events,
                    min_book_events=result.min_book_events,
                    message=result.message,
                    gap_hours_missing=result.gap_hours_missing,
                )
            ),
        )
    except TimeoutError:
        return _timeout_result_from_target(
            ProbeTarget(
                candidate=candidate,
                start_time=start_time,
                end_time=end_time,
                min_book_events=min_book_events,
                source_manifest=source_manifest or "",
                selection_phase=selection_phase,
                window_source=window_source,
                window_provenance=window_provenance,
            ),
            timeout_seconds=timeout_seconds,
        )


def _multiprocessing_context() -> mp.context.BaseContext:
    methods = mp.get_all_start_methods()
    method = "fork" if "fork" in methods else methods[0]
    return mp.get_context(method)


def _probe_target_child(
    output_queue: mp.Queue,
    target: ProbeTarget,
    sources: tuple[str, ...],
    timeout_seconds: int,
) -> None:
    try:
        result = asyncio.run(
            _probe_with_timeout(
                target.candidate,
                start_time=target.start_time,
                end_time=target.end_time,
                min_book_events=target.min_book_events,
                sources=sources,
                timeout_seconds=timeout_seconds,
                selection_phase=target.selection_phase,
                source_manifest=target.source_manifest,
                window_source=target.window_source,
                window_provenance=target.window_provenance,
            )
        )
        output_queue.put({"ok": True, "result": asdict(result)})
    except BaseException as exc:  # child isolation: parent records and keeps scanning
        output_queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }
        )


def _run_target_in_process(
    target: ProbeTarget,
    *,
    sources: tuple[str, ...],
    timeout_seconds: int,
    timeout_grace_seconds: int,
) -> CoverageProbeResult:
    context = _multiprocessing_context()
    output_queue: mp.Queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_probe_target_child,
        args=(output_queue, target, sources, timeout_seconds),
    )
    process.daemon = True
    process.start()
    process.join(timeout=max(1, int(timeout_seconds) + int(timeout_grace_seconds)))
    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=1)
        return _timeout_result_from_target(
            target,
            timeout_seconds=timeout_seconds,
            message=(
                f"Timed out after {timeout_seconds} seconds; child process was terminated "
                "before a candidate/window result was returned."
            ),
        )

    try:
        payload = output_queue.get_nowait()
    except queue_module.Empty:
        return _result_from_candidate(
            target.candidate,
            status="error",
            book_events=0,
            min_book_events=target.min_book_events,
            message=(
                "Probe child process exited without returning a candidate/window result "
                f"(exitcode={process.exitcode})."
            ),
            diagnostic_category="probe_error",
            selection_phase=target.selection_phase,
            source_manifest=target.source_manifest,
            window_start_time=target.start_time,
            window_end_time=target.end_time,
            window_source=target.window_source,
            window_provenance=target.window_provenance,
        )

    if payload.get("ok"):
        return CoverageProbeResult(**payload["result"])
    return _result_from_candidate(
        target.candidate,
        status="error",
        book_events=0,
        min_book_events=target.min_book_events,
        message=str(payload.get("error") or "Probe child process failed."),
        diagnostic_category="probe_error",
        selection_phase=target.selection_phase,
        source_manifest=target.source_manifest,
        window_start_time=target.start_time,
        window_end_time=target.end_time,
        window_source=target.window_source,
        window_provenance=target.window_provenance,
    )


def _pass_manifest_candidate(result: CoverageProbeResult) -> dict[str, Any]:
    candidate = dict(result.candidate or {})
    candidate.update(
        {
            "slug": result.slug,
            "market_slug": result.market_slug,
            "question": result.question,
            "token_index": result.token_index,
            "source_strategy": result.source_strategy,
            "coverage": _coverage_payload(result),
            "safety": _safety_fields(),
            **_safety_top_level_fields(),
        }
    )
    return candidate


def _build_pass_manifest(summary: dict[str, Any]) -> dict[str, Any]:
    pass_rows = [row for row in summary["results"] if row.get("status") == "pass"]
    deduped_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in sorted(pass_rows, key=lambda r: int(r.get("book_events") or 0), reverse=True):
        key = (str(row.get("slug")), int(row.get("token_index") or 0))
        if key in seen:
            continue
        seen.add(key)
        deduped_rows.append(row)
    pass_results = [_coverage_result_from_row(row) for row in deduped_rows]
    safety = _safety_fields(
        summary.get("safety") if isinstance(summary.get("safety"), dict) else None
    )
    return {
        "schema_version": 1,
        "mode": "shadow/backtest-only",
        "safety": safety,
        **_safety_top_level_fields(safety),
        "strategy": summary["strategy"],
        "source_manifest": summary.get("source_manifest") or summary.get("manifest"),
        "manifest_selection": summary.get("manifest_selection", []),
        "window": summary["window"],
        "windows": summary.get("windows", [summary["window"]]),
        "min_book_events": summary["min_book_events"],
        "sources": summary.get("sources", []),
        "generated_at": summary.get("generated_at") or summary.get("generated_at_utc"),
        "metadata": {
            "pass_window_count": len(pass_rows),
            "unique_pass_market_count": len(deduped_rows),
            "dedupe_policy": "one best-book-events pass window per slug/token_index",
            "diagnostic_counts": summary.get("diagnostic_counts", {}),
        },
        "candidate_count": len(deduped_rows),
        "candidates": [_pass_manifest_candidate(result) for result in pass_results],
    }


def _build_probe_windows(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Return one or more UTC ISO windows to probe.

    The default stays backward compatible: exactly the requested
    --start-time/--end-time.  When --recent-window-count > 1, scan additional
    same-sized windows stepping backwards from the requested end_time. This
    expands coverage discovery without relaxing fail-closed event thresholds.
    """
    start_time = _normalize_iso8601(args.start_time)
    end_time = _normalize_iso8601(args.end_time)
    if start_time >= end_time:
        raise ValueError(f"start_time {start_time} must be earlier than end_time {end_time}")
    start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
    width = end_dt - start_dt
    if width <= timedelta(0):
        raise ValueError("probe window width must be positive")
    count = int(getattr(args, "recent_window_count", 1) or 1)
    step_hours = int(getattr(args, "window_step_hours", 1) or 1)
    step = timedelta(hours=step_hours)
    windows: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for idx in range(count):
        window_end = end_dt - (step * idx)
        window_start = window_end - width
        pair = (
            window_start.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            window_end.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        )
        if pair not in seen:
            seen.add(pair)
            windows.append(pair)
    return windows


def _int_arg(args: argparse.Namespace, name: str, default: int) -> int:
    value = getattr(args, name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _target_key(target: ProbeTarget) -> tuple[str, str, int, str, str, str]:
    return (
        target.candidate.slug,
        str(target.candidate.token_index),
        int(target.min_book_events),
        target.start_time,
        target.end_time,
        target.source_manifest,
    )


def _append_target(
    targets: list[ProbeTarget],
    seen: set[tuple[str, str, int, str, str, str]],
    target: ProbeTarget,
) -> None:
    key = _target_key(target)
    if key in seen:
        return
    seen.add(key)
    targets.append(target)


def _candidate_alternate_windows(
    candidate: Candidate,
    *,
    cli_windows: list[tuple[str, str]],
    min_book_events: int,
    primary_window: tuple[str, str],
    max_alternate_windows: int,
) -> list[tuple[str, str, int, str | None, str]]:
    windows: list[tuple[str, str, int, str | None, str]] = []
    seen: set[tuple[str, str, int]] = set()

    def add_window(
        start_time: str,
        end_time: str,
        events: int | None,
        source: str | None,
        provenance: str,
    ) -> None:
        if (start_time, end_time) == primary_window:
            return
        threshold = int(events if events is not None else min_book_events)
        key = (start_time, end_time, threshold)
        if key in seen:
            return
        seen.add(key)
        windows.append((start_time, end_time, threshold, source, provenance))

    for index, (start_time, end_time) in enumerate(cli_windows[1:], start=1):
        add_window(start_time, end_time, min_book_events, None, f"recent_window_cli[{index}]")
    for window in candidate.replay_windows:
        add_window(
            window.start_time,
            window.end_time,
            window.min_book_events,
            window.source,
            window.provenance,
        )
    return windows[:max_alternate_windows]


def _recent_manifest_paths(
    *,
    manifest_glob: str | None,
    primary_manifest: Path,
    max_count: int,
) -> list[Path]:
    if not manifest_glob or max_count <= 0:
        return []
    primary_resolved = primary_manifest.resolve()
    paths = [
        Path(path)
        for path in sorted(
            glob(manifest_glob),
            key=lambda value: Path(value).stat().st_mtime,
            reverse=True,
        )
    ]
    selected: list[Path] = []
    for path in paths:
        try:
            if path.resolve() == primary_resolved:
                continue
        except OSError:
            continue
        selected.append(path)
        if len(selected) >= max_count:
            break
    return selected


def _load_candidates_fail_closed(
    manifest_path: Path,
    *,
    strategy: str,
    max_candidates: int,
) -> tuple[list[Candidate], dict[str, Any]]:
    record: dict[str, Any] = {"path": str(manifest_path), "selected": False}
    try:
        candidates = load_candidates(
            manifest_path,
            strategy=strategy,
            max_candidates=max_candidates,
        )
    except Exception as exc:
        record["reason"] = "skipped_unreadable_or_invalid"
        record["error"] = f"{type(exc).__name__}: {exc}"
        return [], record
    record["candidate_count"] = len(candidates)
    record["selected"] = bool(candidates)
    record["reason"] = "loaded_candidates" if candidates else "skipped_no_eligible_candidates"
    return candidates, record


def _build_probe_targets(
    args: argparse.Namespace,
) -> tuple[list[ProbeTarget], list[ProbeTarget], list[dict[str, Any]], list[tuple[str, str]]]:
    cli_windows = _build_probe_windows(args)
    primary_window = cli_windows[0]
    max_candidates = _int_arg(args, "max_candidates", 1)
    expanded_max_candidates = max(
        max_candidates,
        _int_arg(args, "expanded_max_candidates", max_candidates),
    )
    max_alternate_windows = _int_arg(args, "max_alternate_windows", 3)
    max_alternate_probes = _int_arg(args, "max_alternate_probes", 120)
    alternate_manifest_count = _int_arg(args, "alternate_manifest_count", 0)
    alternate_manifest_glob = getattr(args, "alternate_manifest_glob", None)

    primary_candidates, primary_record = _load_candidates_fail_closed(
        args.manifest,
        strategy=args.strategy,
        max_candidates=max_candidates,
    )
    primary_record["role"] = "primary"
    expanded_candidates, expanded_record = _load_candidates_fail_closed(
        args.manifest,
        strategy=args.strategy,
        max_candidates=expanded_max_candidates,
    )
    expanded_record["role"] = "primary_expanded_candidate_pool"
    manifest_records = [primary_record]
    if expanded_max_candidates > max_candidates:
        manifest_records.append(expanded_record)

    primary_targets: list[ProbeTarget] = []
    expansion_targets: list[ProbeTarget] = []
    seen_primary: set[tuple[str, str, int, str, str, str]] = set()
    seen_expansion: set[tuple[str, str, int, str, str, str]] = set()
    for candidate in primary_candidates:
        _append_target(
            primary_targets,
            seen_primary,
            ProbeTarget(
                candidate=candidate,
                start_time=primary_window[0],
                end_time=primary_window[1],
                min_book_events=int(args.min_book_events),
                source_manifest=str(args.manifest),
                selection_phase="primary_exact",
                window_provenance="cli_exact_window",
            ),
        )

    primary_keys = {(candidate.slug, candidate.token_index) for candidate in primary_candidates}
    expansion_candidates = list(expanded_candidates)
    for candidate in expansion_candidates:
        candidate_key = (candidate.slug, candidate.token_index)
        if candidate_key not in primary_keys:
            _append_target(
                expansion_targets,
                seen_expansion,
                ProbeTarget(
                    candidate=candidate,
                    start_time=primary_window[0],
                    end_time=primary_window[1],
                    min_book_events=int(args.min_book_events),
                    source_manifest=str(args.manifest),
                    selection_phase="zero_pass_expanded_candidate",
                    window_provenance="cli_exact_window",
                ),
            )
        phase = (
            "zero_pass_expanded_candidate_window"
            if candidate_key not in primary_keys
            else "zero_pass_expanded_window"
        )
        for start_time, end_time, threshold, source, provenance in _candidate_alternate_windows(
            candidate,
            cli_windows=cli_windows,
            min_book_events=int(args.min_book_events),
            primary_window=primary_window,
            max_alternate_windows=max_alternate_windows,
        ):
            _append_target(
                expansion_targets,
                seen_expansion,
                ProbeTarget(
                    candidate=candidate,
                    start_time=start_time,
                    end_time=end_time,
                    min_book_events=threshold,
                    source_manifest=str(args.manifest),
                    selection_phase=phase,
                    window_source=source,
                    window_provenance=provenance,
                ),
            )

    for path in _recent_manifest_paths(
        manifest_glob=alternate_manifest_glob,
        primary_manifest=args.manifest,
        max_count=alternate_manifest_count,
    ):
        candidates, record = _load_candidates_fail_closed(
            path,
            strategy=args.strategy,
            max_candidates=expanded_max_candidates,
        )
        record["role"] = "alternate_recent_manifest"
        manifest_records.append(record)
        for candidate in candidates:
            for start_time, end_time, threshold, source, provenance in _candidate_alternate_windows(
                candidate,
                cli_windows=cli_windows,
                min_book_events=int(args.min_book_events),
                primary_window=("", ""),
                max_alternate_windows=max(1, max_alternate_windows),
            ) or [
                (
                    primary_window[0],
                    primary_window[1],
                    int(args.min_book_events),
                    None,
                    "cli_exact_window",
                )
            ]:
                _append_target(
                    expansion_targets,
                    seen_expansion,
                    ProbeTarget(
                        candidate=candidate,
                        start_time=start_time,
                        end_time=end_time,
                        min_book_events=threshold,
                        source_manifest=str(path),
                        selection_phase="zero_pass_recent_manifest",
                        window_source=source,
                        window_provenance=provenance,
                    ),
                )

    return primary_targets, expansion_targets[:max_alternate_probes], manifest_records, cli_windows


async def _run_targets(
    targets: list[ProbeTarget],
    *,
    sources: tuple[str, ...],
    timeout_seconds: int,
    process_isolation: bool,
    timeout_grace_seconds: int,
) -> list[CoverageProbeResult]:
    results: list[CoverageProbeResult] = []
    for target in targets:
        if process_isolation:
            results.append(
                _run_target_in_process(
                    target,
                    sources=sources,
                    timeout_seconds=timeout_seconds,
                    timeout_grace_seconds=timeout_grace_seconds,
                )
            )
        else:
            results.append(
                await _probe_with_timeout(
                    target.candidate,
                    start_time=target.start_time,
                    end_time=target.end_time,
                    min_book_events=target.min_book_events,
                    sources=sources,
                    timeout_seconds=timeout_seconds,
                    selection_phase=target.selection_phase,
                    source_manifest=target.source_manifest,
                    window_source=target.window_source,
                    window_provenance=target.window_provenance,
                )
            )
    return results


def _diagnostic_counts(
    results: list[CoverageProbeResult], *, no_eligible_candidates: bool
) -> dict[str, int]:
    counts: dict[str, int] = {}
    if no_eligible_candidates:
        counts["no_eligible_input_candidates"] = 1
    for result in results:
        counts[result.diagnostic_category] = counts.get(result.diagnostic_category, 0) + 1
    for category in (
        "no_eligible_input_candidates",
        "pmxt_raw_download_failure",
        "probe_runtime_timeout",
        "no_pmxt_l2_book_data",
        "min_book_events_not_met",
        "successful_pass_count",
    ):
        counts.setdefault(category, 0)
    return counts


def _probed_windows(
    results: list[CoverageProbeResult],
    fallback_windows: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str | None, int | None]] = set()
    for result in results:
        key = (result.window_start_time, result.window_end_time, result.min_book_events)
        if key in seen:
            continue
        seen.add(key)
        windows.append(
            {
                "start_time": result.window_start_time,
                "end_time": result.window_end_time,
                "min_book_events": result.min_book_events,
                "source": result.window_source,
                "provenance": result.window_provenance,
            }
        )
    if windows:
        return windows
    return [
        {
            "start_time": start_time,
            "end_time": end_time,
            "min_book_events": None,
            "source": None,
            "provenance": "cli_window",
        }
        for start_time, end_time in fallback_windows
    ]


async def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    primary_targets, expansion_targets, manifest_records, windows = _build_probe_targets(args)
    primary_start_time, primary_end_time = windows[0]
    sources = tuple(args.sources or DEFAULT_SOURCES)
    process_isolation = bool(getattr(args, "process_isolation", True))
    timeout_grace_seconds = _int_arg(
        args,
        "process_timeout_grace_seconds",
        DEFAULT_PROCESS_TIMEOUT_GRACE_SECONDS,
    )
    primary_results = await _run_targets(
        primary_targets,
        sources=sources,
        timeout_seconds=args.timeout_seconds,
        process_isolation=process_isolation,
        timeout_grace_seconds=timeout_grace_seconds,
    )
    expansion_triggered = bool(
        primary_results
        and not any(result.status == "pass" for result in primary_results)
        and expansion_targets
    )
    expansion_results = (
        await _run_targets(
            expansion_targets,
            sources=sources,
            timeout_seconds=args.timeout_seconds,
            process_isolation=process_isolation,
            timeout_grace_seconds=timeout_grace_seconds,
        )
        if expansion_triggered
        else []
    )
    results = primary_results + expansion_results

    pass_count = sum(1 for result in results if result.status == "pass")
    no_coverage_count = sum(1 for result in results if result.status == "no_coverage")
    error_count = sum(1 for result in results if result.status == "error")
    pass_markets = {
        (result.slug, result.token_index) for result in results if result.status == "pass"
    }
    probed_candidates = {(result.slug, result.token_index) for result in results}
    no_eligible_candidates = not primary_targets and not expansion_targets
    diagnostic_counts = _diagnostic_counts(
        results,
        no_eligible_candidates=no_eligible_candidates,
    )
    probed_windows = _probed_windows(results, windows)
    safety = _safety_fields()
    return {
        "schema_version": 1,
        "generated_at": _utc_now().isoformat().replace("+00:00", "Z"),
        "mode": SAFETY_MODE,
        "safety": safety,
        **_safety_top_level_fields(safety),
        "strategy": args.strategy,
        "source_manifest": str(args.manifest),
        "manifest_selection": manifest_records,
        "window": {"start_time": primary_start_time, "end_time": primary_end_time},
        "windows": probed_windows,
        "min_book_events": args.min_book_events,
        "sources": list(sources),
        "candidate_count": len(probed_candidates),
        "primary_candidate_count": len(primary_targets),
        "expansion_candidate_count": max(0, len(probed_candidates) - len(primary_targets)),
        "probe_window_count": len(probed_windows),
        "probe_count": len(results),
        "primary_probe_count": len(primary_results),
        "expansion_probe_count": len(expansion_results),
        "expansion_triggered": expansion_triggered,
        "expansion_reason": "primary_exact_zero_pass" if expansion_triggered else None,
        "max_alternate_probes": _int_arg(args, "max_alternate_probes", 120),
        "pass_market_count": len(pass_markets),
        "pass_count": pass_count,
        "no_coverage_count": no_coverage_count,
        "error_count": error_count,
        "diagnostic_counts": diagnostic_counts,
        "diagnostics": {
            "no_eligible_input_candidates": no_eligible_candidates,
            "pmxt_raw_download_failure_count": diagnostic_counts["pmxt_raw_download_failure"],
            "probe_runtime_timeout_count": diagnostic_counts["probe_runtime_timeout"],
            "no_pmxt_l2_book_data_count": diagnostic_counts["no_pmxt_l2_book_data"],
            "min_book_events_not_met_count": diagnostic_counts["min_book_events_not_met"],
            "successful_pass_count": diagnostic_counts["successful_pass_count"],
            "process_isolation": process_isolation,
            "process_timeout_grace_seconds": timeout_grace_seconds,
        },
        "results": [_summary_result_row(result) for result in results],
    }


def _write_csv(summary: dict[str, Any], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "slug",
                "token_index",
                "source_strategy",
                "status",
                "diagnostic_category",
                "selection_phase",
                "source_manifest",
                "book_events",
                "min_book_events",
                "count_key",
                "market_id",
                "price_min",
                "price_max",
                "price_range",
                "message",
                "gap_hours_missing",
                "gap_warning",
                "window_start_time",
                "window_end_time",
                "window_provenance",
                "window_source",
                "orders_submitted",
                "orders_signed",
                "orders_cancelled",
                "credentials_required",
                "live_trading_worker_started",
                "worker_trading_started",
            ],
        )
        writer.writeheader()
        for row in summary["results"]:
            writer.writerow({field: row.get(field) for field in writer.fieldnames})


def _write_markdown(summary: dict[str, Any], md_path: Path, pass_manifest_path: Path) -> None:
    lines = [
        "# PMXT L2 Coverage Probe",
        "",
        f"- generated_at: {summary['generated_at']}",
        f"- mode: {summary['mode']}",
        f"- source_manifest: {summary['source_manifest']}",
        f"- window: {summary['window']['start_time']} -> {summary['window']['end_time']}",
        f"- probe_windows: {summary.get('probe_window_count', 1)}",
        f"- probes: {summary.get('probe_count', summary['candidate_count'])}",
        f"- pass_markets: {summary.get('pass_market_count', summary['pass_count'])}",
        f"- sources: {', '.join(summary['sources'])}",
        f"- min_book_events: {summary['min_book_events']}",
        f"- candidates: {summary['candidate_count']}",
        f"- pass: {summary['pass_count']}",
        f"- no_coverage: {summary['no_coverage_count']}",
        f"- errors: {summary['error_count']}",
        f"- diagnostic_counts: {json.dumps(summary.get('diagnostic_counts', {}), sort_keys=True)}",
        f"- expansion_triggered: {summary.get('expansion_triggered')}",
        f"- expansion_reason: {summary.get('expansion_reason')}",
        f"- pass_manifest: {pass_manifest_path}",
        "- pass_policy: book_events >= min_book_events and no PMXT missing-hour gap warnings",
        "",
        "Safety fields:",
        f"- orders_submitted={str(summary['safety']['orders_submitted']).lower()}",
        f"- orders_signed={str(summary['safety']['orders_signed']).lower()}",
        f"- orders_cancelled={str(summary['safety']['orders_cancelled']).lower()}",
        f"- credentials_required={str(summary['safety']['credentials_required']).lower()}",
        f"- live_trading_worker_started={str(summary['safety']['live_trading_worker_started']).lower()}",
        f"- worker_trading_started={str(summary['safety']['worker_trading_started']).lower()}",
        "",
        "## Candidates",
        "",
        "| status | diagnostic | phase | market | token | window | book_events | message |",
        "|---|---|---|---|---:|---|---:|---|",
    ]
    for row in summary["results"]:
        lines.append(
            f"| {row.get('status')} | {row.get('diagnostic_category')} | "
            f"{row.get('selection_phase')} | {row.get('slug')} | {row.get('token_index')} | "
            f"{row.get('window_start_time')} -> {row.get('window_end_time')} | "
            f"{row.get('book_events')} | {row.get('message') or ''} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    summary: dict[str, Any],
    *,
    output_dir: Path,
    pass_manifest_dir: Path,
    timestamp: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pass_manifest_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"pmxt_l2_coverage_{timestamp}.json"
    csv_path = output_dir / f"pmxt_l2_coverage_{timestamp}.csv"
    md_path = output_dir / f"pmxt_l2_coverage_{timestamp}.md"
    pass_manifest_path = pass_manifest_dir / f"job_B_pmxt_l2_coverage_pass_{timestamp}.json"

    pass_manifest = _build_pass_manifest(summary)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(summary, csv_path)
    pass_manifest_path.write_text(
        json.dumps(pass_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(summary, md_path, pass_manifest_path)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "pass_manifest": str(pass_manifest_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe PMXT L2 book coverage for Job B candidates."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pass-manifest-dir", type=Path, default=DEFAULT_PASS_MANIFEST_DIR)
    parser.add_argument("--start-time", required=True)
    parser.add_argument("--end-time", required=True)
    parser.add_argument("--strategy", default="microprice_optimizer")
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--min-book-events", type=int, default=50)
    parser.add_argument("--sources", action="append", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument(
        "--recent-window-count",
        type=int,
        default=1,
        help="Probe this many same-sized recent windows ending at --end-time and stepping backwards.",
    )
    parser.add_argument(
        "--window-step-hours",
        type=int,
        default=1,
        help="Hours to step backwards between recent windows.",
    )
    parser.add_argument(
        "--expanded-max-candidates",
        type=int,
        default=None,
        help=(
            "After the primary exact window has zero passes, expand up to this many "
            "ranked candidates from the primary manifest."
        ),
    )
    parser.add_argument(
        "--max-alternate-windows",
        type=int,
        default=3,
        help=(
            "After primary zero-pass, probe up to this many alternate windows per candidate "
            "from CLI recent windows and manifest coverage guidance."
        ),
    )
    parser.add_argument(
        "--max-alternate-probes",
        type=int,
        default=120,
        help="Hard cap on zero-pass expansion probes.",
    )
    parser.add_argument(
        "--alternate-manifest-glob",
        default=None,
        help="Optional recent manifest glob to consider only after primary exact zero-pass.",
    )
    parser.add_argument(
        "--alternate-manifest-count",
        type=int,
        default=0,
        help="Maximum recent manifests from --alternate-manifest-glob to consider.",
    )
    parser.add_argument("--timestamp", default=None, help="Override output timestamp for tests.")
    parser.add_argument(
        "--no-process-isolation",
        action="store_false",
        dest="process_isolation",
        help=(
            "Disable per-candidate child-process isolation. Intended only for local debugging; "
            "the default isolates stuck candidate/window probes so artifacts can still be written."
        ),
    )
    parser.set_defaults(process_isolation=True)
    parser.add_argument(
        "--process-timeout-grace-seconds",
        type=int,
        default=DEFAULT_PROCESS_TIMEOUT_GRACE_SECONDS,
        help=(
            "Additional seconds the parent waits for a child to serialize a timeout row before "
            "terminating the child process."
        ),
    )
    parser.add_argument("--fail-on-errors", action="store_true")
    parser.add_argument("--fail-on-no-pass", action="store_true")
    args = parser.parse_args(argv)
    if args.max_candidates < 1:
        raise SystemExit("max-candidates must be >= 1")
    if args.min_book_events < 0:
        raise SystemExit("min-book-events must be >= 0")
    if args.timeout_seconds < 1:
        raise SystemExit("timeout-seconds must be >= 1")
    if args.process_timeout_grace_seconds < 0:
        raise SystemExit("process-timeout-grace-seconds must be >= 0")
    if args.recent_window_count < 1:
        raise SystemExit("recent-window-count must be >= 1")
    if args.window_step_hours < 1:
        raise SystemExit("window-step-hours must be >= 1")
    if args.expanded_max_candidates is not None and args.expanded_max_candidates < 1:
        raise SystemExit("expanded-max-candidates must be >= 1")
    if args.max_alternate_windows < 0:
        raise SystemExit("max-alternate-windows must be >= 0")
    if args.max_alternate_probes < 0:
        raise SystemExit("max-alternate-probes must be >= 0")
    if args.alternate_manifest_count < 0:
        raise SystemExit("alternate-manifest-count must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = asyncio.run(run_probe(args))
    output_files = write_outputs(
        summary,
        output_dir=args.output_dir,
        pass_manifest_dir=args.pass_manifest_dir,
        timestamp=_timestamp(args.timestamp),
    )
    summary["output_files"] = output_files
    print(
        json.dumps(
            {
                "output_files": output_files,
                "candidate_count": summary["candidate_count"],
                "pass_count": summary["pass_count"],
                "pass_market_count": summary.get("pass_market_count"),
                "probe_count": summary.get("probe_count"),
                "expansion_triggered": summary.get("expansion_triggered"),
                "diagnostic_counts": summary.get("diagnostic_counts"),
            },
            sort_keys=True,
        )
    )
    if args.fail_on_errors and summary["error_count"]:
        return 1
    if args.fail_on_no_pass and summary["pass_count"] == 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
