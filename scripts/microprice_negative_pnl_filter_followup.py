from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    from _script_helpers import ensure_repo_root
else:
    from ._script_helpers import ensure_repo_root

ensure_repo_root(__file__)

try:
    from scripts.job_b_microprice_batch import SAFETY_MODE  # noqa: E402
except ImportError:  # pragma: no cover - direct-script fallback is covered by smoke checks
    SAFETY_MODE = "shadow_backtest_only_no_live_orders"

DEFAULT_OUTPUT_DIR = Path("/opt/polymarket-lab/reports/backtests/job-B-microprice")
REQUIRED_SAFETY_FIELDS = (
    "orders_submitted",
    "orders_signed",
    "orders_cancelled",
    "credentials_required",
    "live_trading_worker_started",
    "worker_trading_started",
)
PROMOTION_REJECTION = "fail_closed_exclude_from_replay_promotion"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _parse_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sum_float(values: Iterable[Any]) -> float:
    total = 0.0
    for value in values:
        parsed = _parse_float(value)
        if parsed is not None:
            total += parsed
    return total


def _safety_fields() -> dict[str, bool]:
    return {
        "orders_submitted": False,
        "orders_signed": False,
        "orders_cancelled": False,
        "credentials_required": False,
        "live_trading_worker_started": False,
        "worker_trading_started": False,
    }


def _artifact_attempts(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(attempt)
        for attempt in _as_list(artifact.get("attempts"))
        if isinstance(attempt, Mapping)
    ]


def _negative_attribution_from_attempt(attempt: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = _as_mapping(attempt.get("diagnostics"))
    attribution = _as_mapping(diagnostics.get("negative_pnl_attribution"))
    if not attribution:
        return {}
    bucket = _as_mapping(attribution.get("parameter_candidate_bucket"))
    if not bucket:
        bucket = {
            "slug": attempt.get("slug"),
            "token_index": _parse_int(attempt.get("token_index")),
            "source_strategy": attempt.get("source_strategy"),
            "params": _as_mapping(attempt.get("params")),
            "bucket_key": None,
        }
    enriched = dict(attribution)
    enriched["parameter_candidate_bucket"] = bucket
    enriched.setdefault("slug", bucket.get("slug") or attempt.get("slug"))
    token_index = bucket.get("token_index")
    if token_index is None:
        token_index = _parse_int(attempt.get("token_index"))
    enriched.setdefault("token_index", token_index)
    enriched.setdefault("params", bucket.get("params") or _as_mapping(attempt.get("params")))
    return enriched


def _eligible_negative_attributions(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    attributions: list[dict[str, Any]] = []
    for attempt in _artifact_attempts(artifact):
        attribution = _negative_attribution_from_attempt(attempt)
        if attribution.get("eligible") is True:
            attributions.append(attribution)
    return attributions


def _artifact_metrics(artifact: Mapping[str, Any]) -> dict[str, Any]:
    fills_orders_pnl = _as_mapping(artifact.get("fills_orders_pnl"))
    attempts = _artifact_attempts(artifact)
    completed_attempts = [
        attempt for attempt in attempts if str(attempt.get("status") or "") == "completed"
    ]
    p_values = [
        _parse_float(_as_mapping(attempt.get("diagnostics")).get("pnl"))
        for attempt in completed_attempts
    ]
    p_values = [value for value in p_values if value is not None]
    completed_pnl_sum = _parse_float(fills_orders_pnl.get("completed_pnl_sum"))
    if completed_pnl_sum is None:
        completed_pnl_sum = _sum_float(p_values)
    total_fills = _parse_float(fills_orders_pnl.get("total_fills"))
    if total_fills is None:
        total_fills = _sum_float(
            _as_mapping(attempt.get("diagnostics")).get("fills") for attempt in attempts
        )
    positive_attempts = _parse_int(fills_orders_pnl.get("completed_positive_pnl_attempts"))
    if positive_attempts is None:
        positive_attempts = sum(1 for value in p_values if value > 0)
    negative_attempts = _parse_int(fills_orders_pnl.get("completed_negative_pnl_attempts"))
    if negative_attempts is None:
        negative_attempts = sum(1 for value in p_values if value < 0)
    return {
        "attempt_count": _parse_int(artifact.get("attempt_count")) or len(attempts),
        "completed_attempt_count": _parse_int(artifact.get("completed_count"))
        or _parse_int(artifact.get("completed"))
        or len(completed_attempts),
        "total_fills": total_fills,
        "completed_pnl_sum": completed_pnl_sum,
        "completed_positive_pnl_attempts": positive_attempts,
        "completed_negative_pnl_attempts": negative_attempts,
        "exact_window_status": artifact.get("exact_window_status"),
        "source_classification": artifact.get("classification"),
        "source_live_ready": artifact.get("live_ready"),
        "source_no_profit_claim": artifact.get("no_profit_claim"),
    }


def _market_key(bucket: Mapping[str, Any]) -> str:
    slug = bucket.get("slug") or "unknown"
    token_index = bucket.get("token_index")
    return f"{slug}#{token_index if token_index is not None else 'unknown'}"


def _bucket_key(bucket: Mapping[str, Any]) -> str:
    key = bucket.get("bucket_key")
    if isinstance(key, str) and key:
        return key
    slug = bucket.get("slug") or "unknown"
    token_index = bucket.get("token_index")
    params = json.dumps(_as_mapping(bucket.get("params")), sort_keys=True)
    return f"{slug}#{token_index if token_index is not None else 'unknown'}|{params}"


def _cause_counts(attributions: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for attribution in attributions:
        for cause in _as_list(attribution.get("causes")):
            counts[str(cause)] += 1
    return dict(counts)


def _grouped_filters(attributions: list[dict[str, Any]], *, group: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attribution in attributions:
        bucket = _as_mapping(attribution.get("parameter_candidate_bucket"))
        key = _market_key(bucket) if group == "market" else _bucket_key(bucket)
        grouped[key].append(attribution)

    filters: list[dict[str, Any]] = []
    for key in sorted(grouped):
        records = grouped[key]
        buckets = [_as_mapping(record.get("parameter_candidate_bucket")) for record in records]
        causes = _cause_counts(records)
        p_values = [_parse_float(record.get("pnl")) for record in records]
        fills = [_parse_float(record.get("fills")) for record in records]
        first_bucket = buckets[0] if buckets else {}
        filters.append(
            {
                "key": key,
                "filter_status": PROMOTION_REJECTION,
                "fail_closed": True,
                "reason_codes": sorted(causes),
                "eligible_attempt_count": len(records),
                "pnl_sum": _sum_float(p_values),
                "fills_sum": _sum_float(fills),
                "positive_pnl_attempt_count": sum(
                    1 for value in p_values if value is not None and value > 0
                ),
                "adverse_selection_markout_attempts": causes.get("adverse_selection_markout", 0),
                "spread_tick_cost_too_large_attempts": causes.get("spread_tick_cost_too_large", 0),
                "queue_fill_timing_attempts": causes.get("queue_fill_timing", 0),
                "parameter_candidate_bucket_attempts": causes.get("parameter_candidate_bucket", 0),
                "slug": first_bucket.get("slug"),
                "token_index": first_bucket.get("token_index"),
                "source_strategy": first_bucket.get("source_strategy"),
                "tail_bucket": first_bucket.get("tail_bucket"),
                "scan_mid": first_bucket.get("scan_mid"),
                "scan_spread": first_bucket.get("scan_spread"),
                "params": first_bucket.get("params"),
                "parameter_bucket_keys": sorted({_bucket_key(bucket) for bucket in buckets}),
            }
        )
    return filters


def _recommendations(attributions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    causes = _cause_counts(attributions)
    recommendations: list[dict[str, Any]] = []
    if causes.get("adverse_selection_markout", 0) > 0:
        recommendations.append(
            {
                "name": "block_adverse_selection_markout_bucket",
                "filter_status": PROMOTION_REJECTION,
                "fail_closed": True,
                "affected_attempt_count": causes["adverse_selection_markout"],
                "rationale": (
                    "At least one filled attempt had adverse round-trip or terminal markout."
                ),
            }
        )
    if causes.get("spread_tick_cost_too_large", 0) > 0:
        recommendations.append(
            {
                "name": "block_spread_tick_cost_bucket",
                "filter_status": PROMOTION_REJECTION,
                "fail_closed": True,
                "affected_attempt_count": causes["spread_tick_cost_too_large"],
                "rationale": (
                    "Spread, tick movement, or commission cost overwhelmed the observed fill."
                ),
            }
        )
    if attributions:
        recommendations.append(
            {
                "name": "block_negative_parameter_market_buckets",
                "filter_status": PROMOTION_REJECTION,
                "fail_closed": True,
                "affected_attempt_count": len(attributions),
                "rationale": (
                    "Parameter/market buckets with non-positive after-cost fills are not "
                    "eligible for replay promotion."
                ),
            }
        )
    else:
        recommendations.append(
            {
                "name": "negative_pnl_attribution_missing_fail_closed",
                "filter_status": PROMOTION_REJECTION,
                "fail_closed": True,
                "affected_attempt_count": 0,
                "rationale": (
                    "No eligible negative_pnl_attribution records were available to justify "
                    "promotion."
                ),
            }
        )
    recommendations.append(
        {
            "name": "require_positive_after_costs_out_of_sample",
            "filter_status": "required_before_any_future_promotion",
            "fail_closed": True,
            "affected_attempt_count": len(attributions),
            "rationale": (
                "A future artifact needs exact-window coverage, fills, positive PnL after costs, "
                "and cleared negative-PnL attribution before promotion."
            ),
        }
    )
    return recommendations


def _not_live_ready_reason_codes(
    *, metrics: Mapping[str, Any], attributions: list[dict[str, Any]]
) -> list[str]:
    reasons: list[str] = ["shadow_backtest_only"]
    aggregate_pnl = _parse_float(metrics.get("completed_pnl_sum"))
    positive_attempts = _parse_int(metrics.get("completed_positive_pnl_attempts")) or 0
    if aggregate_pnl is None or aggregate_pnl <= 0:
        reasons.append("aggregate_pnl_not_positive_after_costs")
    if positive_attempts <= 0:
        reasons.append("zero_positive_pnl_attempts")
    if attributions:
        reasons.append("eligible_negative_pnl_attribution_detected")
    else:
        reasons.append("negative_pnl_attribution_missing_or_not_eligible")
    causes = _cause_counts(attributions)
    if causes.get("adverse_selection_markout", 0) > 0:
        reasons.append("adverse_selection_markout_detected")
    if causes.get("spread_tick_cost_too_large", 0) > 0:
        reasons.append("spread_tick_cost_too_large_detected")
    return sorted(dict.fromkeys(reasons))


def build_filter_followup_report(
    *,
    artifact_path: Path,
    command: list[str] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    raw_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact = _as_mapping(raw_artifact)
    artifact_shape_status = "recognized"
    artifact_shape_warnings: list[dict[str, Any]] = []
    if not isinstance(raw_artifact, Mapping):
        artifact_shape_status = "unknown_fail_closed"
        artifact_shape_warnings.append(
            {
                "type": "unknown_artifact_json_shape",
                "message": "Job B artifact is not a JSON object; diagnostics fail closed.",
                "json_type": type(raw_artifact).__name__,
            }
        )

    attributions = _eligible_negative_attributions(artifact)
    metrics = _artifact_metrics(artifact)
    cause_counts = _cause_counts(attributions)
    reason_codes = _not_live_ready_reason_codes(metrics=metrics, attributions=attributions)
    if artifact_shape_warnings:
        reason_codes = sorted(dict.fromkeys([*reason_codes, "unknown_artifact_json_shape"]))
    report = {
        "schema_version": 1,
        "generated_at_utc": (generated_at or _utc_now()).isoformat().replace("+00:00", "Z"),
        "mode": SAFETY_MODE,
        "classification": "diagnostic_only",
        "adopted": False,
        "diagnostic_only": True,
        "rejected": False,
        "live_ready": False,
        "replay_promotion_ready": False,
        "no_profit_claim": True,
        "profit_opportunity_demonstrated": False,
        "artifact": str(artifact_path),
        "artifact_shape_status": artifact_shape_status,
        "artifact_shape_warnings": artifact_shape_warnings,
        "source_metrics": metrics,
        "not_live_ready_reason_codes": reason_codes,
        "negative_pnl_attribution_summary": _as_mapping(
            artifact.get("negative_pnl_attribution_summary")
        ),
        "filter_diagnostics": {
            "fail_closed": True,
            "filter_status": PROMOTION_REJECTION,
            "eligible_negative_pnl_attempt_count": len(attributions),
            "cause_counts": cause_counts,
            "market_filters": _grouped_filters(attributions, group="market"),
            "parameter_bucket_filters": _grouped_filters(attributions, group="parameter"),
            "recommendations": _recommendations(attributions),
        },
        "next_hook": (
            "Feed these diagnostic filters into the next Microprice replay-promotion gate; "
            "rerun only with exact-window coverage, fills, positive after-cost PnL, and "
            "cleared adverse-selection/spread-cost attribution."
        ),
        "commands": [" ".join(command)] if command else [],
        "safety": _safety_fields(),
    }
    # Scheduled-job consumers require these exact safety booleans at top level,
    # not only nested under ``safety``.
    report.update(_safety_fields())
    source_safety = _as_mapping(artifact.get("safety"))
    report["source_safety"] = {
        field: bool(source_safety.get(field)) for field in REQUIRED_SAFETY_FIELDS
    }
    return report


def _write_markdown(report: Mapping[str, Any], md_path: Path) -> None:
    safety = _as_mapping(report.get("safety"))
    metrics = _as_mapping(report.get("source_metrics"))
    diagnostics = _as_mapping(report.get("filter_diagnostics"))
    lines = [
        "# Microprice Negative-PnL Filter Follow-up",
        "",
        f"- generated_at_utc: {report.get('generated_at_utc')}",
        f"- mode: {report.get('mode')}",
        f"- classification: {report.get('classification')}",
        f"- live_ready: {str(report.get('live_ready')).lower()}",
        f"- replay_promotion_ready: {str(report.get('replay_promotion_ready')).lower()}",
        f"- no_profit_claim: {str(report.get('no_profit_claim')).lower()}",
        f"- profit_opportunity_demonstrated: {str(report.get('profit_opportunity_demonstrated')).lower()}",
        f"- artifact: {report.get('artifact')}",
        f"- completed_pnl_sum: {metrics.get('completed_pnl_sum')}",
        f"- total_fills: {metrics.get('total_fills')}",
        f"- completed_positive_pnl_attempts: {metrics.get('completed_positive_pnl_attempts')}",
        f"- exact_window_status: {metrics.get('exact_window_status')}",
        f"- not_live_ready_reason_codes: {json.dumps(report.get('not_live_ready_reason_codes', []), sort_keys=True)}",
        f"- filter_status: {diagnostics.get('filter_status')}",
        f"- fail_closed: {str(diagnostics.get('fail_closed')).lower()}",
        f"- safety: {json.dumps(safety, sort_keys=True)}",
        f"- orders_submitted={str(safety.get('orders_submitted')).lower()}",
        f"- orders_signed={str(safety.get('orders_signed')).lower()}",
        f"- orders_cancelled={str(safety.get('orders_cancelled')).lower()}",
        f"- credentials_required={str(safety.get('credentials_required')).lower()}",
        f"- live_trading_worker_started={str(safety.get('live_trading_worker_started')).lower()}",
        f"- worker_trading_started={str(safety.get('worker_trading_started')).lower()}",
        "",
        "Safety: shadow/backtest only; no live trading, signing, cancellation, order submission, credentials, or worker-trading.",
        "",
        "## Recommendations",
        "",
        "| name | status | affected_attempts | rationale |",
        "|---|---:|---:|---|",
    ]
    for recommendation in _as_list(diagnostics.get("recommendations")):
        rec = _as_mapping(recommendation)
        lines.append(
            f"| {rec.get('name')} | {rec.get('filter_status')} | "
            f"{rec.get('affected_attempt_count')} | {rec.get('rationale')} |"
        )
    lines.extend(
        [
            "",
            "## Market Filters",
            "",
            "| key | status | pnl_sum | fills_sum | reasons |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for market_filter in _as_list(diagnostics.get("market_filters")):
        row = _as_mapping(market_filter)
        lines.append(
            f"| {row.get('key')} | {row.get('filter_status')} | {row.get('pnl_sum')} | "
            f"{row.get('fills_sum')} | {json.dumps(row.get('reason_codes', []), sort_keys=True)} |"
        )
    lines.extend(
        [
            "",
            "## Parameter Bucket Filters",
            "",
            "| key | status | pnl_sum | fills_sum | reasons |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for bucket_filter in _as_list(diagnostics.get("parameter_bucket_filters")):
        row = _as_mapping(bucket_filter)
        lines.append(
            f"| `{row.get('key')}` | {row.get('filter_status')} | {row.get('pnl_sum')} | "
            f"{row.get('fills_sum')} | {json.dumps(row.get('reason_codes', []), sort_keys=True)} |"
        )
    lines.extend(["", f"Next hook: {report.get('next_hook')}", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def write_filter_followup_outputs(
    report: dict[str, Any], output_dir: Path, timestamp: str
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"microprice_negative_pnl_filter_followup_{timestamp}.json"
    md_path = output_dir / f"microprice_negative_pnl_filter_followup_{timestamp}.md"
    output_files = {"json": str(json_path), "markdown": str(md_path)}
    payload = dict(report)
    payload["output_files"] = output_files
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(payload, md_path)
    return output_files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build shadow/backtest-only Microprice negative-PnL filter diagnostics "
            "from a Job B artifact."
        )
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args(argv)

    timestamp = args.timestamp or _utc_now().strftime("%Y%m%dT%H%M%SZ")
    command = [sys.executable, *sys.argv] if argv is None else [sys.executable, __file__, *argv]
    report = build_filter_followup_report(artifact_path=args.artifact, command=command)
    report["output_files"] = write_filter_followup_outputs(report, args.output_dir, timestamp)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
