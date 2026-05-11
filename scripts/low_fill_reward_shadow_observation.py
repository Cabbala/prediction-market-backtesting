from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

SAFETY_MODE = "shadow_observation_only_no_live_orders"
AGGREGATION_MODE = "shadow_quote_multi_snapshot_aggregation_no_live_orders"
DEFAULT_OUTPUT_DIR = Path("/opt/polymarket-lab/reports/reward_scanner/shadow_observations")
DEFAULT_JSON_GLOB = (
    "/opt/polymarket-lab/autoresearch/reward_scanner/manifests/reward_scanner_manifest_*.json"
)
SAFETY_FALSE_FIELDS = {
    "orders_submitted": False,
    "orders_signed": False,
    "orders_cancelled": False,
    "credentials_required": False,
    "live_trading_worker_started": False,
}


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


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _safety_fields() -> dict[str, bool]:
    return {
        "live_trading": False,
        "orders_submitted": False,
        "orders_signed": False,
        "orders_cancelled": False,
        "credentials_required": False,
        "live_trading_worker_started": False,
        "worker_trading_started": False,
    }


def _latest_manifest(pattern: str) -> Path:
    candidates = [p for p in Path("/").glob(pattern.lstrip("/")) if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"no manifest files match {pattern}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _iter_candidates(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in (
        "low_fill_reward_maker_watchlist",
        "low_fill_reward_candidates",
        "watchlist",
        "coverage_probe_candidates",
        "backtest_ready_candidates",
        "candidates",
    ):
        value = payload.get(key)
        if isinstance(value, list) and value:
            return [row for row in value if isinstance(row, dict)]
    buckets = payload.get("buckets")
    if isinstance(buckets, dict):
        rows: list[dict[str, Any]] = []
        for key in (
            "low_fill_reward_maker_watchlist",
            "coverage_probe_candidates",
            "backtest_ready_candidates",
        ):
            value = buckets.get(key)
            if isinstance(value, list):
                rows.extend(row for row in value if isinstance(row, dict))
        return rows
    return []


def _candidate_slug(row: dict[str, Any]) -> str:
    return str(row.get("market_slug") or row.get("slug") or row.get("question") or "unknown")


def _features(row: dict[str, Any]) -> dict[str, Any]:
    features = row.get("features")
    return features if isinstance(features, dict) else {}


def _first_float(row: dict[str, Any], *keys: str) -> float | None:
    features = _features(row)
    for key in keys:
        value = _parse_float(row.get(key))
        if value is not None:
            return value
        value = _parse_float(features.get(key))
        if value is not None:
            return value
    return None


def _reward_evidence(row: dict[str, Any]) -> dict[str, Any]:
    evidence = row.get("reward_evidence") if isinstance(row.get("reward_evidence"), dict) else {}
    nested = (
        evidence.get("reward_evidence") if isinstance(evidence.get("reward_evidence"), dict) else {}
    )
    merged = dict(evidence)
    merged.update(nested)
    return merged


def _reward_max_spread(row: dict[str, Any]) -> float | None:
    evidence = _reward_evidence(row)
    features = _features(row)
    return _parse_float(
        row.get("rewardsMaxSpread")
        or features.get("rewardsMaxSpread")
        or features.get("reward_max_spread")
        or evidence.get("rewardsMaxSpread")
    )


def _has_reward_evidence(row: dict[str, Any]) -> bool:
    evidence = _reward_evidence(row)
    features = _features(row)
    fee_reward_category = features.get("fee_reward_category")
    explicit_category = fee_reward_category in {
        "explicit_clob_rewards",
        "explicit_gamma_reward_terms",
        "explicit_uma_reward_hint",
        "explicit_reward_hint",
    }
    return bool(
        evidence
        or row.get("clobRewards")
        or row.get("rewardsMinSize")
        or row.get("umaReward")
        or features.get("has_explicit_reward_evidence")
        or explicit_category
    )


def _observation(row: dict[str, Any]) -> dict[str, Any]:
    yes_mid = _first_float(row, "yes_mid", "scan_mid", "yes_probability")
    spread = _first_float(
        row,
        "spread",
        "scan_spread",
        "avg_spread",
        "max_spread",
        "yes_spread",
    )
    liquidity = _first_float(row, "liquidity", "scan_liquidity")
    reward_max_spread = _reward_max_spread(row)
    has_reward_evidence = _has_reward_evidence(row)
    if has_reward_evidence and spread is not None and reward_max_spread is not None:
        in_band: bool | None = spread <= reward_max_spread
        time_in_band_basis = "single_snapshot_proxy_not_duration"
        time_in_band_status = "measured_single_snapshot_not_duration"
    else:
        in_band = None
        time_in_band_basis = "unknown_missing_reward_spread_or_quote"
        time_in_band_status = "unknown_missing_reward_spread_or_quote"
    tail_bucket = (
        "unknown"
        if yes_mid is None
        else "ultra_low_tail"
        if yes_mid < 0.005
        else "low_tail"
        if yes_mid < 0.01
        else "non_extreme_tail"
        if yes_mid <= 0.25
        else "high_probability"
    )
    accidental_fill_risk = "unknown"
    if yes_mid is not None:
        if yes_mid < 0.005:
            accidental_fill_risk = "high_relative_tick_cost"
        elif yes_mid < 0.01:
            accidental_fill_risk = "medium_tail_tick_cost"
        else:
            accidental_fill_risk = "normal_requires_l2_fill_model"
    exit_risk = "unknown"
    if liquidity is not None and liquidity < 5000:
        exit_risk = "thin_liquidity"
    elif tail_bucket in {"ultra_low_tail", "low_tail"}:
        exit_risk = "tail_exit_slippage_risk"
    elif liquidity is not None:
        exit_risk = "liquidity_proxy_ok_needs_l2"
    return {
        "slug": _candidate_slug(row),
        "question": row.get("question"),
        "yes_mid": yes_mid,
        "spread": spread,
        "liquidity": liquidity,
        "reward_max_spread": reward_max_spread,
        "has_reward_evidence": has_reward_evidence,
        "time_in_band_observed": in_band,
        "time_in_band_basis": time_in_band_basis,
        "time_in_band": {
            "status": time_in_band_status,
            "sample_count": 1,
            "in_band": in_band,
            "in_band_snapshot_count": 1 if in_band is True else 0,
            "observed_fraction": (1.0 if in_band else 0.0) if in_band is not None else None,
            "basis": time_in_band_basis,
        },
        "would_have_filled": "unknown_requires_l2_or_shadow_quote_log",
        "accidental_fill_risk": accidental_fill_risk,
        "exit_risk": exit_risk,
        "orders_submitted": False,
        "orders_signed": False,
        "orders_cancelled": False,
        "credentials_required": False,
        "live_trading_worker_started": False,
    }


def _unknown_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized == "" or normalized == "unknown" or normalized.startswith("unknown_")
    if isinstance(value, dict):
        status = value.get("status")
        if isinstance(status, str) and status.startswith("unknown"):
            return True
        return value.get("value") is None and value.get("in_band") is None
    return False


def _safe_observation_input(path: Path, payload: dict[str, Any]) -> None:
    safety = payload.get("safety")
    if not isinstance(safety, dict):
        raise ValueError(f"{path} is missing a shadow safety block")
    unsafe_fields = {
        "live_trading": safety.get("live_trading"),
        "orders_submitted": safety.get("orders_submitted"),
        "orders_signed": safety.get("orders_signed"),
        "orders_cancelled": safety.get("orders_cancelled"),
        "credentials_required": safety.get("credentials_required"),
        "live_trading_worker_started": safety.get(
            "live_trading_worker_started", safety.get("worker_trading_started")
        ),
    }
    for key, value in unsafe_fields.items():
        if value not in (False, None):
            raise ValueError(f"{path} is not shadow-safe: {key}={value!r}")


def _load_snapshot(path: Path) -> tuple[datetime, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    _safe_observation_input(path, payload)
    observed_at = _parse_datetime(payload.get("generated_at_utc"))
    if observed_at is None:
        raise ValueError(f"{path} is missing a valid generated_at_utc timestamp")
    observations = payload.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError(f"{path} has no observation rows to aggregate")
    seen_slugs: set[str] = set()
    rows: list[dict[str, Any]] = []
    for raw in observations:
        if not isinstance(raw, dict):
            raise ValueError(f"{path} contains a non-object observation row")
        slug = _candidate_slug(raw)
        if slug == "unknown":
            raise ValueError(f"{path} contains an observation without a stable slug")
        if slug in seen_slugs:
            raise ValueError(f"{path} contains duplicate observation rows for {slug}")
        seen_slugs.add(slug)
        row = dict(raw)
        row["source_snapshot_file"] = str(path)
        row["source_observed_at_utc"] = observed_at.isoformat().replace("+00:00", "Z")
        rows.append(row)
    return observed_at, rows


def _snapshot_time_in_band(row: dict[str, Any]) -> bool | None:
    measurement = row.get("time_in_band")
    if isinstance(measurement, dict):
        value = _parse_bool(measurement.get("in_band"))
        status = measurement.get("status")
        if value is not None and isinstance(status, str) and status.startswith("measured"):
            return value
    value = _parse_bool(row.get("time_in_band_observed"))
    if value is True:
        return True
    if value is False:
        spread = _parse_float(row.get("spread"))
        reward_max_spread = _parse_float(row.get("reward_max_spread"))
        if (
            row.get("has_reward_evidence") is True
            and spread is not None
            and reward_max_spread is not None
        ):
            return False
    return None


def _snapshot_would_have_filled(row: dict[str, Any]) -> bool | None:
    value = row.get("would_have_filled")
    if isinstance(value, dict):
        return _parse_bool(value.get("value"))
    parsed = _parse_bool(value)
    if parsed is not None:
        return parsed
    shadow_quote = row.get("shadow_quote")
    if isinstance(shadow_quote, dict):
        return _parse_bool(shadow_quote.get("would_have_filled"))
    return None


def _risk_value(row: dict[str, Any], field: str) -> str | None:
    value = row.get(field)
    if isinstance(value, dict):
        value = value.get("value") or value.get("latest") or value.get("worst_observed")
    if _unknown_value(value):
        return None
    return str(value)


def _categorical_measurement(
    rows: Iterable[dict[str, Any]], field: str, severity: dict[str, int]
) -> dict[str, Any]:
    values = [_risk_value(row, field) for row in rows]
    known = [value for value in values if value is not None]
    if not known:
        return {
            "status": f"unknown_no_{field}_measurements",
            "value": None,
            "observed_values": [],
        }
    unique = sorted(set(known), key=lambda value: (-severity.get(value, 0), value))
    return {
        "status": "measured_from_shadow_snapshots",
        "value": unique[0],
        "worst_observed": unique[0],
        "latest": known[-1],
        "observed_values": unique,
    }


def _time_in_band_measurement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured: list[tuple[datetime, bool]] = []
    for row in rows:
        observed_at = _parse_datetime(row.get("source_observed_at_utc"))
        value = _snapshot_time_in_band(row)
        if observed_at is not None and value is not None:
            measured.append((observed_at, value))
    in_band_count = sum(1 for _, value in measured if value)
    if len(measured) < 2:
        status = (
            "unknown_single_measured_snapshot_not_duration"
            if measured
            else "unknown_no_measured_reward_band_snapshots"
        )
        return {
            "status": status,
            "sample_count": len(rows),
            "measured_snapshot_count": len(measured),
            "in_band_snapshot_count": in_band_count,
            "observed_fraction": None,
            "observed_window_seconds": None,
            "basis": "requires_at_least_two_reward_band_snapshots",
        }
    first = min(observed_at for observed_at, _ in measured)
    last = max(observed_at for observed_at, _ in measured)
    window_seconds = int((last - first).total_seconds())
    if window_seconds <= 0:
        return {
            "status": "unknown_non_positive_observation_window",
            "sample_count": len(rows),
            "measured_snapshot_count": len(measured),
            "in_band_snapshot_count": in_band_count,
            "observed_fraction": None,
            "observed_window_seconds": window_seconds,
            "basis": "requires_positive_time_between_snapshots",
        }
    observed_fraction = in_band_count / len(measured)
    return {
        "status": "measured_multi_snapshot_proxy_not_continuous",
        "sample_count": len(rows),
        "measured_snapshot_count": len(measured),
        "in_band_snapshot_count": in_band_count,
        "observed_fraction": round(observed_fraction, 6),
        "observed_window_seconds": window_seconds,
        "approx_in_band_seconds": round(window_seconds * observed_fraction, 6),
        "basis": "discrete_shadow_snapshots_not_continuous_l2_replay",
    }


def _would_have_filled_measurement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [_snapshot_would_have_filled(row) for row in rows]
    known = [value for value in measured if value is not None]
    if not known:
        return {
            "status": "unknown_no_l2_or_shadow_quote_fill_log",
            "value": None,
            "filled_snapshot_count": 0,
            "measured_snapshot_count": 0,
            "basis": "requires_l2_replay_or_shadow_quote_lifecycle_log",
        }
    filled = sum(1 for value in known if value)
    return {
        "status": "measured_from_shadow_quote_log",
        "value": filled > 0,
        "filled_snapshot_count": filled,
        "measured_snapshot_count": len(known),
        "basis": "explicit_shadow_quote_fill_observations",
    }


def _latest_float(rows: list[dict[str, Any]], key: str) -> float | None:
    for row in reversed(rows):
        value = _parse_float(row.get(key))
        if value is not None:
            return value
    return None


def _aggregate_candidate(slug: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    accidental_severity = {
        "high_relative_tick_cost": 50,
        "tight_spread_possible_fill": 40,
        "medium_tail_tick_cost": 30,
        "moderate": 20,
        "normal_requires_l2_fill_model": 10,
    }
    exit_severity = {
        "thin_liquidity": 40,
        "tail_exit_slippage_risk": 30,
        "liquidity_proxy_ok_needs_l2": 10,
    }
    question = next((row.get("question") for row in rows if row.get("question")), None)
    observed_times = [
        row["source_observed_at_utc"]
        for row in rows
        if isinstance(row.get("source_observed_at_utc"), str)
    ]
    return {
        "slug": slug,
        "question": question,
        "source_observation_count": len(rows),
        "first_observed_at_utc": observed_times[0] if observed_times else None,
        "last_observed_at_utc": observed_times[-1] if observed_times else None,
        "latest_yes_mid": _latest_float(rows, "yes_mid"),
        "latest_spread": _latest_float(rows, "spread"),
        "latest_liquidity": _latest_float(rows, "liquidity"),
        "latest_reward_max_spread": _latest_float(rows, "reward_max_spread"),
        "time_in_band": _time_in_band_measurement(rows),
        "would_have_filled": _would_have_filled_measurement(rows),
        "accidental_fill_risk": _categorical_measurement(
            rows, "accidental_fill_risk", accidental_severity
        ),
        "exit_risk": _categorical_measurement(rows, "exit_risk", exit_severity),
        **SAFETY_FALSE_FIELDS,
    }


def build_aggregation_report(snapshot_paths: list[Path], *, limit: int) -> dict[str, Any]:
    if not snapshot_paths:
        raise ValueError("at least one snapshot path is required")
    loaded = [_load_snapshot(path) for path in snapshot_paths]
    loaded.sort(key=lambda item: item[0])
    grouped: dict[str, list[dict[str, Any]]] = {}
    source_files: list[str] = []
    for _, rows in loaded:
        for row in rows:
            source_file = row["source_snapshot_file"]
            if source_file not in source_files:
                source_files.append(source_file)
            grouped.setdefault(_candidate_slug(row), []).append(row)
    observations = [
        _aggregate_candidate(slug, rows)
        for slug, rows in sorted(
            grouped.items(), key=lambda item: item[1][0].get("source_observed_at_utc", "")
        )
    ][:limit]
    would_have_filled_known_count = sum(
        1 for row in observations if row["would_have_filled"]["status"].startswith("measured")
    )
    classification = "diagnostic_only"
    if not observations:
        classification = "blocked"
    elif would_have_filled_known_count > 0:
        classification = "adopted"
    return {
        "schema_version": 2,
        "generated_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
        "mode": AGGREGATION_MODE,
        "classification": classification,
        "safety": _safety_fields(),
        "source_snapshot_files": source_files,
        "snapshot_file_count": len(source_files),
        "candidate_count": len(observations),
        "observations": observations,
        "summary": {
            "candidate_count": len(observations),
            "snapshot_file_count": len(source_files),
            "total_snapshot_observation_rows": sum(len(rows) for _, rows in loaded),
            "time_in_band_measured_count": sum(
                1 for row in observations if row["time_in_band"]["status"].startswith("measured")
            ),
            "time_in_band_unknown_count": sum(
                1 for row in observations if row["time_in_band"]["status"].startswith("unknown")
            ),
            "would_have_filled_known_count": would_have_filled_known_count,
            "would_have_filled_unknown_count": sum(
                1
                for row in observations
                if row["would_have_filled"]["status"].startswith("unknown")
            ),
            "accidental_fill_risk_known_count": sum(
                1
                for row in observations
                if row["accidental_fill_risk"]["status"].startswith("measured")
            ),
            "exit_risk_known_count": sum(
                1 for row in observations if row["exit_risk"]["status"].startswith("measured")
            ),
        },
    }


def build_report(manifest: Path, *, limit: int) -> dict[str, Any]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = _iter_candidates(payload)[:limit]
    observations = [_observation(row) for row in rows]
    return {
        "schema_version": 1,
        "generated_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
        "mode": SAFETY_MODE,
        "safety": _safety_fields(),
        "source_manifest": str(manifest),
        "candidate_count": len(rows),
        "observations": observations,
        "summary": {
            "time_in_band_snapshot_count": sum(
                1 for row in observations if row["time_in_band_observed"]
            ),
            "would_have_filled_known_count": 0,
            "needs_l2_or_shadow_quote_log_count": len(observations),
            "high_relative_tick_cost_count": sum(
                1
                for row in observations
                if row["accidental_fill_risk"] == "high_relative_tick_cost"
            ),
        },
    }


def write_outputs(
    report: dict[str, Any], output_dir: Path, timestamp: str, *, prefix: str | None = None
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or (
        "low_fill_reward_shadow_quote_aggregation"
        if report.get("mode") == AGGREGATION_MODE
        else "low_fill_reward_shadow_observation"
    )
    json_path = output_dir / f"{prefix}_{timestamp}.json"
    md_path = output_dir / f"{prefix}_{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    safety = report["safety"]
    if report.get("mode") == AGGREGATION_MODE:
        lines = [
            "# Low-fill Reward Maker Multi-snapshot Shadow Quote Aggregation",
            "",
            f"- mode: {report['mode']}",
            f"- classification: {report['classification']}",
            f"- snapshot_files: {report['snapshot_file_count']}",
            f"- candidates: {report['candidate_count']}",
            f"- summary: {json.dumps(report['summary'], sort_keys=True)}",
            "",
            "Safety fields:",
            f"- orders_submitted={str(safety['orders_submitted']).lower()}",
            f"- orders_signed={str(safety['orders_signed']).lower()}",
            f"- orders_cancelled={str(safety['orders_cancelled']).lower()}",
            f"- credentials_required={str(safety['credentials_required']).lower()}",
            f"- live_trading_worker_started={str(safety['live_trading_worker_started']).lower()}",
            "",
            "No reward or profit claim is made without observed fill/reward evidence.",
            "",
            "| slug | time_in_band | would_have_filled | accidental_fill_risk | exit_risk |",
            "|---|---|---|---|---|",
        ]
        for row in report["observations"]:
            time_in_band = row["time_in_band"]
            would_fill = row["would_have_filled"]
            fill_risk = row["accidental_fill_risk"]
            exit_risk = row["exit_risk"]
            time_summary = (
                f"{time_in_band['status']} "
                f"({time_in_band['in_band_snapshot_count']}/"
                f"{time_in_band['measured_snapshot_count']})"
            )
            lines.append(
                f"| {row['slug']} | {time_summary} | {would_fill['status']} | "
                f"{fill_risk['status']}: {fill_risk['value']} | "
                f"{exit_risk['status']}: {exit_risk['value']} |"
            )
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"json": str(json_path), "markdown": str(md_path)}

    lines = [
        "# Low-fill Reward Maker Shadow Observation",
        "",
        f"- mode: {report['mode']}",
        f"- source_manifest: {report['source_manifest']}",
        f"- candidates: {report['candidate_count']}",
        f"- summary: {json.dumps(report['summary'], sort_keys=True)}",
        "",
        "Safety fields:",
        f"- orders_submitted={str(safety['orders_submitted']).lower()}",
        f"- orders_signed={str(safety['orders_signed']).lower()}",
        f"- orders_cancelled={str(safety['orders_cancelled']).lower()}",
        f"- credentials_required={str(safety['credentials_required']).lower()}",
        f"- live_trading_worker_started={str(safety['live_trading_worker_started']).lower()}",
        "",
        "| slug | time_in_band_snapshot | would_have_filled | accidental_fill_risk | exit_risk |",
        "|---|---:|---|---|---|",
    ]
    for row in report["observations"]:
        lines.append(
            f"| {row['slug']} | {row['time_in_band_observed']} | {row['would_have_filled']} | {row['accidental_fill_risk']} | {row['exit_risk']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def _expand_paths(paths: list[Path] | None, globs: list[str] | None) -> list[Path]:
    expanded: list[Path] = []
    for path in paths or []:
        if path not in expanded:
            expanded.append(path)
    for pattern in globs or []:
        if pattern.startswith("/"):
            matches = sorted(Path("/").glob(pattern.lstrip("/")))
        else:
            matches = sorted(Path().glob(pattern))
        for path in matches:
            if path.is_file() and path not in expanded:
                expanded.append(path)
    return expanded


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build shadow-only low-fill reward-maker observation report."
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--manifest-glob", default=DEFAULT_JSON_GLOB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument(
        "--snapshot",
        action="append",
        type=Path,
        default=None,
        help="Prior low-fill shadow observation JSON to aggregate. Repeat for multiple snapshots.",
    )
    parser.add_argument(
        "--snapshots-glob",
        action="append",
        default=None,
        help="Glob of prior low-fill shadow observation JSON files to aggregate.",
    )
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("limit must be >= 1")
    timestamp = args.timestamp or _utc_now().strftime("%Y%m%dT%H%M%SZ")
    snapshot_paths = _expand_paths(args.snapshot, args.snapshots_glob)
    if snapshot_paths:
        report = build_aggregation_report(snapshot_paths, limit=args.limit)
        report["output_files"] = write_outputs(
            report,
            args.output_dir,
            timestamp,
            prefix="low_fill_reward_shadow_quote_aggregation",
        )
    else:
        manifest = args.manifest or _latest_manifest(args.manifest_glob)
        report = build_report(manifest, limit=args.limit)
        report["output_files"] = write_outputs(
            report,
            args.output_dir,
            timestamp,
            prefix="low_fill_reward_shadow_observation",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
