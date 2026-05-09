from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SAFETY_MODE = "shadow_only_no_live_trading"
DEFAULT_OUTPUT_DIR = Path("/opt/polymarket-lab/data/reward_shadow")
DEFAULT_REPORT_DIR = Path("/opt/polymarket-lab/reports/reward_shadow")


def _utc_now() -> datetime:
    return datetime.now(tz=UTC).replace(microsecond=0)


def _parse_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _book_side(raw: dict[str, Any], side: str) -> dict[str, float | None]:
    book = raw.get(f"{side}_book")
    if not isinstance(book, dict):
        book = {}
    return {
        "bid": _parse_float(book.get("bid") or book.get("best_bid")),
        "ask": _parse_float(book.get("ask") or book.get("best_ask")),
        "bid_size": _parse_float(book.get("bid_size")),
        "ask_size": _parse_float(book.get("ask_size")),
        "depth_bid_5c": _parse_float(book.get("depth_bid_5c") or book.get("depth_bid_top10")),
        "depth_ask_5c": _parse_float(book.get("depth_ask_5c") or book.get("depth_ask_top10")),
    }


def _reward_fields(raw: dict[str, Any]) -> dict[str, Any]:
    evidence = raw.get("reward_evidence")
    if not isinstance(evidence, dict):
        evidence = {}
    allow = {"clobRewards", "rewardsMinSize", "rewardsMaxSpread", "umaReward"}
    return {k: evidence.get(k) for k in allow if k in evidence}


def build_shadow_record(raw: dict[str, Any], *, generated_at: str, quote_size: float) -> dict[str, Any] | None:
    slug = raw.get("slug") or raw.get("market_slug")
    if not isinstance(slug, str) or not slug:
        return None
    yes = _book_side(raw, "yes")
    no = _book_side(raw, "no")
    yes_bid = yes["bid"]
    yes_ask = yes["ask"]
    yes_spread = _parse_float(raw.get("yes_spread"))
    yes_mid = _parse_float(raw.get("yes_mid"))
    rewards_max_spread = _parse_float(_reward_fields(raw).get("rewardsMaxSpread"))
    quoted_price = yes_bid
    in_band = False
    if yes_spread is not None:
        if rewards_max_spread is None:
            in_band = True
        else:
            in_band = yes_spread <= rewards_max_spread
    accidental_fill_risk = "unknown"
    if yes_bid is not None and yes_ask is not None:
        if yes_bid <= 0.01:
            accidental_fill_risk = "high_relative_tick_cost"
        elif yes_spread is not None and yes_spread <= 0.001:
            accidental_fill_risk = "tight_spread_possible_fill"
        else:
            accidental_fill_risk = "moderate"
    return {
        "generated_at_utc": generated_at,
        "mode": SAFETY_MODE,
        "slug": slug,
        "question": raw.get("question") or slug,
        "condition_id": raw.get("condition_id") or raw.get("conditionId"),
        "yes_token_id": raw.get("yes_token_id"),
        "no_token_id": raw.get("no_token_id"),
        "source_market_url": raw.get("source_market_url"),
        "yes_mid": yes_mid,
        "yes_spread": yes_spread,
        "yes_book": yes,
        "no_book": no,
        "reward_evidence": _reward_fields(raw),
        "shadow_quote": {
            "side": "yes",
            "action": "post_only_bid_observation",
            "price": quoted_price,
            "size": quote_size,
            "in_reward_band_now": in_band,
            "time_in_band_seconds_observed": 0,
            "would_have_filled": "unknown_single_snapshot",
            "accidental_fill_risk": accidental_fill_risk,
            "exit_risk": "unknown_until_multisnapshot_or_replay",
        },
        "safety": {
            "live_trading": False,
            "orders_submitted": False,
            "orders_signed": False,
            "credentials_required": False,
            "worker_trading_started": False,
        },
    }


def load_scan_candidates(scan_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(scan_path.read_text())
    candidates = payload.get("candidates") or payload.get("top_candidates") or []
    return [c for c in candidates if isinstance(c, dict)]


def write_outputs(records: list[dict[str, Any]], *, output_dir: Path, report_dir: Path, timestamp: str, source_scan: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"reward_shadow_{timestamp}.jsonl"
    json_path = output_dir / f"reward_shadow_{timestamp}.json"
    md_path = report_dir / f"reward_shadow_{timestamp}.md"
    jsonl_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    summary = {
        "generated_at_utc": timestamp,
        "mode": SAFETY_MODE,
        "source_scan": str(source_scan),
        "record_count": len(records),
        "safety": {"live_trading": False, "orders_submitted": False, "orders_signed": False, "credentials_required": False},
        "records": records,
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    lines = ["# Reward Maker Shadow Observation", "", f"- source_scan: {source_scan}", f"- records: {len(records)}", "- safety: shadow only; no live trading, signing, cancellation, order submission, credentials, or worker-trading.", "", "| slug | yes_mid | spread | quote | in_band | fill_risk |", "|---|---:|---:|---:|---|---|"]
    for r in records:
        q = r["shadow_quote"]
        lines.append(f"| {r['slug']} | {r.get('yes_mid')} | {r.get('yes_spread')} | {q.get('price')} | {q.get('in_reward_band_now')} | {q.get('accidental_fill_risk')} |")
    md_path.write_text("\n".join(lines) + "\n")
    return {"jsonl": str(jsonl_path), "json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build reward-maker shadow-only observation records from a market scan.")
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--quote-size", type=float, default=5.0)
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()
    timestamp = args.timestamp or _utc_now().strftime("%Y%m%dT%H%M%SZ")
    generated_at = _utc_now().isoformat().replace("+00:00", "Z")
    records = []
    for raw in load_scan_candidates(args.scan):
        if not _reward_fields(raw):
            continue
        rec = build_shadow_record(raw, generated_at=generated_at, quote_size=args.quote_size)
        if rec is not None:
            records.append(rec)
        if len(records) >= args.limit:
            break
    outputs = write_outputs(records, output_dir=args.output_dir, report_dir=args.report_dir, timestamp=timestamp, source_scan=args.scan)
    print(json.dumps({"record_count": len(records), "output_files": outputs, "mode": SAFETY_MODE}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
