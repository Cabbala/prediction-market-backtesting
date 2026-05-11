from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
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
) -> CoverageProbeResult:
    finite_prices = _finite_prices(prices)
    price_min = min(finite_prices) if finite_prices else None
    price_max = max(finite_prices) if finite_prices else None
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
) -> CoverageProbeResult:
    try:
        return await asyncio.wait_for(
            probe_candidate(
                candidate,
                start_time=start_time,
                end_time=end_time,
                min_book_events=min_book_events,
                sources=sources,
            ),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        return _result_from_candidate(
            candidate,
            status="error",
            book_events=0,
            min_book_events=min_book_events,
            message=f"Timed out after {timeout_seconds} seconds.",
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
            "coverage": {
                "status": result.status,
                "window": {
                    "start_time": result.window_start_time,
                    "end_time": result.window_end_time,
                },
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
            },
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
    pass_results = [CoverageProbeResult(**row) for row in deduped_rows]
    return {
        "schema_version": 1,
        "mode": "shadow/backtest-only",
        "safety": summary["safety"],
        "strategy": summary["strategy"],
        "source_manifest": summary.get("source_manifest") or summary.get("manifest"),
        "window": summary["window"],
        "windows": summary.get("windows", [summary["window"]]),
        "min_book_events": summary["min_book_events"],
        "sources": summary.get("sources", []),
        "generated_at": summary.get("generated_at") or summary.get("generated_at_utc"),
        "metadata": {
            "pass_window_count": len(pass_rows),
            "unique_pass_market_count": len(deduped_rows),
            "dedupe_policy": "one best-book-events pass window per slug/token_index",
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


async def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    windows = _build_probe_windows(args)
    primary_start_time, primary_end_time = windows[0]
    sources = tuple(args.sources or DEFAULT_SOURCES)
    candidates = load_candidates(
        args.manifest, strategy=args.strategy, max_candidates=args.max_candidates
    )
    results: list[CoverageProbeResult] = []

    for candidate in candidates:
        for start_time, end_time in windows:
            results.append(
                await _probe_with_timeout(
                    candidate,
                    start_time=start_time,
                    end_time=end_time,
                    min_book_events=args.min_book_events,
                    sources=sources,
                    timeout_seconds=args.timeout_seconds,
                )
            )

    pass_count = sum(1 for result in results if result.status == "pass")
    no_coverage_count = sum(1 for result in results if result.status == "no_coverage")
    error_count = sum(1 for result in results if result.status == "error")
    pass_markets = {
        (result.slug, result.token_index) for result in results if result.status == "pass"
    }
    return {
        "schema_version": 1,
        "generated_at": _utc_now().isoformat().replace("+00:00", "Z"),
        "mode": SAFETY_MODE,
        "safety": {
            "live_trading": False,
            "orders_submitted": False,
            "orders_signed": False,
            "credentials_required": False,
            "worker_trading_started": False,
            "live_trading_worker_started": False,
            "strategy_execution": False,
        },
        "strategy": args.strategy,
        "source_manifest": str(args.manifest),
        "window": {"start_time": primary_start_time, "end_time": primary_end_time},
        "windows": [{"start_time": start, "end_time": end} for start, end in windows],
        "min_book_events": args.min_book_events,
        "sources": list(sources),
        "candidate_count": len(candidates),
        "probe_window_count": len(windows),
        "probe_count": len(results),
        "pass_market_count": len(pass_markets),
        "pass_count": pass_count,
        "no_coverage_count": no_coverage_count,
        "error_count": error_count,
        "results": [asdict(result) for result in results],
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
        f"- pass_manifest: {pass_manifest_path}",
        "- pass_policy: book_events >= min_book_events and no PMXT missing-hour gap warnings",
        "",
        "Safety: shadow/backtest only; no live trading, signing, cancellation, order submission, "
        "credentials, worker-trading, or strategy execution.",
        "",
        "## Candidates",
        "",
        "| status | market | token | window | book_events | message |",
        "|---|---|---:|---|---:|---|",
    ]
    for row in summary["results"]:
        lines.append(
            f"| {row.get('status')} | {row.get('slug')} | {row.get('token_index')} | "
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
    parser.add_argument("--timestamp", default=None, help="Override output timestamp for tests.")
    parser.add_argument("--fail-on-errors", action="store_true")
    parser.add_argument("--fail-on-no-pass", action="store_true")
    args = parser.parse_args(argv)
    if args.max_candidates < 1:
        raise SystemExit("max-candidates must be >= 1")
    if args.min_book_events < 0:
        raise SystemExit("min-book-events must be >= 0")
    if args.timeout_seconds < 1:
        raise SystemExit("timeout-seconds must be >= 1")
    if args.recent_window_count < 1:
        raise SystemExit("recent-window-count must be >= 1")
    if args.window_step_hours < 1:
        raise SystemExit("window-step-hours must be >= 1")
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
