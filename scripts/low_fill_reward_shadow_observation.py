from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SAFETY_MODE = "shadow_observation_only_no_live_orders"
DEFAULT_OUTPUT_DIR = Path("/opt/polymarket-lab/reports/reward_scanner/shadow_observations")
DEFAULT_JSON_GLOB = (
    "/opt/polymarket-lab/autoresearch/reward_scanner/manifests/reward_scanner_manifest_*.json"
)


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
    return _parse_float(row.get("rewardsMaxSpread") or evidence.get("rewardsMaxSpread"))


def _observation(row: dict[str, Any]) -> dict[str, Any]:
    yes_mid = _parse_float(row.get("yes_mid") or row.get("scan_mid") or row.get("yes_probability"))
    spread = _parse_float(
        row.get("spread")
        or row.get("scan_spread")
        or row.get("avg_spread")
        or row.get("max_spread")
    )
    liquidity = _parse_float(row.get("liquidity") or row.get("scan_liquidity"))
    reward_max_spread = _reward_max_spread(row)
    evidence = _reward_evidence(row)
    has_reward_evidence = bool(
        evidence or row.get("clobRewards") or row.get("rewardsMinSize") or row.get("umaReward")
    )
    in_band = bool(
        has_reward_evidence
        and spread is not None
        and reward_max_spread is not None
        and spread <= reward_max_spread
    )
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
        "time_in_band_basis": "single_snapshot_proxy_not_duration",
        "would_have_filled": "unknown_requires_l2_or_shadow_quote_log",
        "accidental_fill_risk": accidental_fill_risk,
        "exit_risk": exit_risk,
        "orders_submitted": False,
        "orders_signed": False,
        "credentials_required": False,
    }


def build_report(manifest: Path, *, limit: int) -> dict[str, Any]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = _iter_candidates(payload)[:limit]
    observations = [_observation(row) for row in rows]
    return {
        "schema_version": 1,
        "generated_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
        "mode": SAFETY_MODE,
        "safety": {
            "live_trading": False,
            "orders_submitted": False,
            "orders_signed": False,
            "orders_cancelled": False,
            "credentials_required": False,
            "worker_trading_started": False,
        },
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


def write_outputs(report: dict[str, Any], output_dir: Path, timestamp: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"low_fill_reward_shadow_observation_{timestamp}.json"
    md_path = output_dir / f"low_fill_reward_shadow_observation_{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Low-fill Reward Maker Shadow Observation",
        "",
        f"- mode: {report['mode']}",
        f"- source_manifest: {report['source_manifest']}",
        f"- candidates: {report['candidate_count']}",
        f"- summary: {json.dumps(report['summary'], sort_keys=True)}",
        "",
        "Safety: shadow observation only; no live trading, signing, cancellation, order submission, credentials, or worker-trading.",
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build shadow-only low-fill reward-maker observation report."
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--manifest-glob", default=DEFAULT_JSON_GLOB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("limit must be >= 1")
    manifest = args.manifest or _latest_manifest(args.manifest_glob)
    timestamp = args.timestamp or _utc_now().strftime("%Y%m%dT%H%M%SZ")
    report = build_report(manifest, limit=args.limit)
    report["output_files"] = write_outputs(report, args.output_dir, timestamp)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
