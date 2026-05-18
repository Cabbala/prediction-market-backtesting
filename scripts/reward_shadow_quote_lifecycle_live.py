#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1] if SCRIPT_PATH.parent.name == "scripts" else Path.cwd()
BASE_SCRIPT = ROOT / "scripts" / "low_fill_reward_shadow_lifecycle.py"
spec = importlib.util.spec_from_file_location("low_fill_reward_shadow_lifecycle", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
assert spec and spec.loader
sys.modules[spec.name] = base
spec.loader.exec_module(base)  # type: ignore[union-attr]

SAFETY_MODE = "shadow_quote_lifecycle_live_read_only_no_orders"
DEFAULT_OUTPUT_ROOT = Path("/opt/polymarket-lab/reports/reward_scanner/shadow_quote_lifecycle_live")
USER_AGENT = "HermesPolymarketShadowQuoteLifecycle/1.0"


def utc_now() -> datetime:
    return datetime.now(tz=UTC).replace(microsecond=0)


def iso_z(dt: datetime | None = None) -> str:
    return (dt or utc_now()).isoformat().replace("+00:00", "Z")


def stamp(dt: datetime | None = None) -> str:
    return (dt or utc_now()).strftime("%Y%m%dT%H%M%SZ")


def safety_object() -> dict[str, bool]:
    return {
        "live_trading": False,
        "orders_submitted": False,
        "orders_signed": False,
        "orders_cancelled": False,
        "credentials_required": False,
        "worker_trading_started": False,
        "live_trading_worker_started": False,
    }


def http_json(url: str, *, method: str = "GET", body: Any = None, timeout: float = 15.0) -> Any:
    data = None
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def as_float(v: Any) -> float | None:
    if v in (None, ""):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def side_levels(book: Mapping[str, Any], side: str) -> list[dict[str, float]]:
    raw = book.get(side) or book.get(side.rstrip("s")) or []
    out: list[dict[str, float]] = []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return out
    for lvl in raw:
        if isinstance(lvl, Mapping):
            p = as_float(lvl.get("price") or lvl.get("p"))
            s = as_float(lvl.get("size") or lvl.get("s"))
        elif isinstance(lvl, Sequence) and len(lvl) >= 2:
            p = as_float(lvl[0])
            s = as_float(lvl[1])
        else:
            p = s = None
        if p is not None and s is not None:
            out.append({"price": p, "size": s})
    if side.startswith("bid"):
        out.sort(key=lambda x: x["price"], reverse=True)
    else:
        out.sort(key=lambda x: x["price"])
    return out


def book_summary(raw: Mapping[str, Any], token_id: str) -> dict[str, Any]:
    bids = side_levels(raw, "bids")
    asks = side_levels(raw, "asks")
    best_bid = max((x["price"] for x in bids), default=None)
    best_ask = min((x["price"] for x in asks), default=None)
    return {
        "token_id": token_id,
        "asset_id": raw.get("asset_id") or raw.get("token_id") or token_id,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_levels": bids[:20],
        "ask_levels": asks[:20],
        "raw_level_count": {"bids": len(bids), "asks": len(asks)},
    }


def fetch_books(token_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    ids = [str(t) for t in token_ids if t]
    if not ids:
        return {}
    payload = [{"token_id": t} for t in ids]
    data = http_json("https://clob.polymarket.com/books", method="POST", body=payload)
    if isinstance(data, Mapping):
        rows = data.get("data") or data.get("books") or []
    else:
        rows = data
    out: dict[str, dict[str, Any]] = {}
    for raw in rows if isinstance(rows, Sequence) else []:
        if not isinstance(raw, Mapping):
            continue
        asset = str(raw.get("asset_id") or raw.get("token_id") or raw.get("market") or "")
        if asset:
            out[asset] = book_summary(raw, asset)
    # Some endpoints preserve request order but omit asset_id; map fallback by order.
    if len(out) < len(ids) and isinstance(rows, Sequence):
        for token_id, raw in zip(ids, rows):
            if isinstance(raw, Mapping) and token_id not in out:
                out[token_id] = book_summary(raw, token_id)
    return out


def fetch_trades(condition_id: str, *, start_ts: int | None = None, end_ts: int | None = None, limit: int = 500) -> list[dict[str, Any]]:
    q = {"market": condition_id, "limit": str(min(limit, 500)), "offset": "0"}
    url = "https://data-api.polymarket.com/trades?" + urllib.parse.urlencode(q)
    data = http_json(url)
    trades = data if isinstance(data, list) else data.get("data", []) if isinstance(data, Mapping) else []
    out: list[dict[str, Any]] = []
    for t in trades:
        if not isinstance(t, Mapping):
            continue
        ts = as_float(t.get("timestamp") or t.get("time") or t.get("createdAt"))
        if ts is not None:
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
        out.append(dict(t))
    return out


def outcome_matches_trade(trade: Mapping[str, Any], side: str, token_id: str | None) -> bool:
    asset = str(trade.get("asset") or trade.get("asset_id") or trade.get("token_id") or "")
    if token_id and asset and asset == str(token_id):
        return True
    outcome = str(trade.get("outcome") or trade.get("outcomeName") or trade.get("tokenOutcome") or "").casefold()
    return outcome == side.casefold() if outcome else True


def queue_ahead_for_bid(levels: Sequence[Mapping[str, float]], price: float, tick: float) -> float:
    # For a post-only bid, all displayed bid depth at prices at-or-better than our limit is ahead.
    eps = max(tick / 2.0, 1e-9)
    return sum(float(level.get("size") or 0.0) for level in levels if float(level.get("price") or 0.0) >= price - eps)


def trade_consumption_for_bid(trades: Sequence[Mapping[str, Any]], *, side: str, token_id: str | None, price: float, tick: float) -> tuple[float, int, list[dict[str, Any]]]:
    eps = max(tick / 2.0, 1e-9)
    matched: list[dict[str, Any]] = []
    total = 0.0
    for t in trades:
        if not outcome_matches_trade(t, side, token_id):
            continue
        p = as_float(t.get("price"))
        s = as_float(t.get("size") or t.get("amount") or t.get("shares"))
        if p is None or s is None:
            continue
        # Conservative maker-bid evidence: observed executions at or below our bid can deplete bids at our price.
        if p <= price + eps:
            total += s
            matched.append({"timestamp": t.get("timestamp"), "price": p, "size": s, "outcome": t.get("outcome"), "asset": t.get("asset") or t.get("asset_id") or t.get("token_id")})
    return total, len(matched), matched[:20]


def build_live_quote(candidate: Any, side: str, book: Mapping[str, Any], *, quote_size: float, tick_size: float) -> dict[str, Any]:
    bids = list(book.get("bid_levels") or [])
    asks = list(book.get("ask_levels") or [])
    best_bid = as_float(book.get("best_bid"))
    best_ask = as_float(book.get("best_ask"))
    mid = (best_bid + best_ask) / 2.0 if best_bid is not None and best_ask is not None else None
    spread = best_ask - best_bid if best_bid is not None and best_ask is not None else None
    min_size = candidate.reward_min_size or quote_size
    size = max(quote_size, min_size)
    price = max(tick_size, min(1.0 - tick_size, best_bid)) if best_bid is not None else None
    post_only_cross = bool(price is not None and best_ask is not None and price >= best_ask)
    q_ahead = queue_ahead_for_bid(bids, price, tick_size) if price is not None else None
    reward_band = "unknown"
    if candidate.reward_max_spread is not None and spread is not None:
        reward_band = "in_band" if spread <= candidate.reward_max_spread else "out_of_band_spread"
    return {
        "side": side,
        "token_id": candidate.yes_token_id if side == "yes" else candidate.no_token_id,
        "quote_price": price,
        "quote_size": size,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread": spread,
        "top_l2": {"bids": bids[:10], "asks": asks[:10]},
        "queue_ahead_size": q_ahead,
        "reward_band_status": reward_band,
        "required_size_met": size >= min_size,
        "post_only_would_cross": post_only_cross,
        "scoring_eligible_estimate": bool(price is not None and not post_only_cross and reward_band == "in_band" and size >= min_size),
    }


def classify_quote(quote: dict[str, Any], trades: Sequence[Mapping[str, Any]], future_book: Mapping[str, Any] | None, *, tick_size: float) -> dict[str, Any]:
    price = as_float(quote.get("quote_price"))
    size = as_float(quote.get("quote_size")) or 0.0
    q_ahead = as_float(quote.get("queue_ahead_size"))
    side = str(quote.get("side"))
    token_id = quote.get("token_id")
    if quote.get("post_only_would_cross"):
        classification = "post_only_cancel_reprice"
        known = False
        prob = None
        basis = "post_only_quote_would_cross_current_ask_not_maker_fill"
        consumed = 0.0
        matched_count = 0
        matched_sample = []
    elif price is None or q_ahead is None:
        classification = "unknown"
        known = False
        prob = None
        basis = "missing_live_book_or_queue_ahead"
        consumed = 0.0
        matched_count = 0
        matched_sample = []
    else:
        consumed, matched_count, matched_sample = trade_consumption_for_bid(trades, side=side, token_id=str(token_id) if token_id else None, price=price, tick=tick_size)
        if consumed >= q_ahead + size and matched_count > 0:
            classification = "conservative"
            known = True
            prob = 1.0
            basis = "trade_volume_at_or_through_quote_exceeded_queue_ahead_plus_own_size"
        elif consumed > 0 or matched_count > 0:
            classification = "optimistic"
            known = False
            prob = None
            basis = "trade_touched_quote_price_but_did_not_clear_queue_ahead_plus_own_size"
        else:
            classification = "unknown"
            known = False
            prob = None
            basis = "no_trade_tape_evidence_reached_quote_before_next_snapshot"
    exit_bid = as_float((future_book or {}).get("best_bid"))
    markout = None
    exit_loss = None
    if price is not None and exit_bid is not None:
        markout = exit_bid - price
        exit_loss = max(0.0, price - exit_bid)
    return {
        "classification": classification,
        "known": known,
        "estimate": True if classification == "conservative" else "unknown",
        "probability": prob,
        "basis": basis,
        "trade_window": {
            "matched_trade_count": matched_count,
            "matched_trade_size_at_or_through_quote": round(consumed, 6),
            "matched_trade_sample": matched_sample,
        },
        "queue_evidence": {
            "queue_ahead_size": q_ahead,
            "own_size": size,
            "required_consumption_for_conservative_fill": round((q_ahead or 0.0) + size, 6),
        },
        "exit_risk": {
            "future_best_bid": exit_bid,
            "markout_from_quote": round(markout, 6) if markout is not None else None,
            "immediate_exit_loss": round(exit_loss, 6) if exit_loss is not None else None,
        },
    }


def quote_sides(candidate: Any) -> list[str]:
    return base.quote_sides(candidate)


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_manifest or base.latest_source_manifest()
    payload = base.load_source(source)
    candidates = base.select_candidates(payload, source_path=str(source), max_candidates=args.max_candidates)
    ts = args.timestamp or stamp()
    output_dir = args.output_dir / ts
    output_dir.mkdir(parents=True, exist_ok=True)
    lifecycle: list[dict[str, Any]] = []
    quote_records: list[dict[str, Any]] = []
    offsets = [min(args.duration_secs, i * args.interval_secs) for i in range(int(math.ceil(args.duration_secs / args.interval_secs)) + 1)] if args.duration_secs > 0 else [0]
    if not offsets:
        offsets = [0]
    started = utc_now()
    previous_records: list[dict[str, Any]] = []
    previous_epoch_ts: int | None = None
    for idx, offset in enumerate(offsets):
        if idx > 0 and not args.no_sleep:
            prev = offsets[idx - 1]
            time.sleep(max(0.0, offset - prev))
        now = utc_now()
        epoch_ts = int(now.timestamp())
        token_ids = []
        for c in candidates:
            token_ids.extend([c.yes_token_id, c.no_token_id])
        books = fetch_books([t for t in token_ids if t])
        # finalize previous interval with trades up to this snapshot and current/future book
        if previous_records and previous_epoch_ts is not None:
            trades_by_condition: dict[str, list[dict[str, Any]]] = {}
            for c in candidates:
                if c.condition_id and c.condition_id not in trades_by_condition:
                    try:
                        trades_by_condition[c.condition_id] = fetch_trades(c.condition_id, start_ts=previous_epoch_ts, end_ts=epoch_ts, limit=args.trade_limit)
                    except Exception as e:
                        trades_by_condition[c.condition_id] = [{"error": f"trade_fetch_failed: {e}"}]
            for rec in previous_records:
                c_condition = rec.get("condition_id")
                token = rec.get("quote", {}).get("token_id")
                future_book = books.get(str(token)) if token else None
                trades = trades_by_condition.get(str(c_condition), [])
                if trades and isinstance(trades[0], Mapping) and "error" in trades[0]:
                    rec["would_fill"] = {"classification": "unknown", "known": False, "basis": trades[0]["error"]}
                else:
                    rec["would_fill"] = classify_quote(rec["quote"], trades, future_book, tick_size=args.tick_size)
                rec["finalized_at_utc"] = iso_z(now)
                quote_records.append(rec)
        previous_records = []
        for c in candidates:
            candidate_snapshot = {"slug": c.slug, "condition_id": c.condition_id, "question": c.question, "source_url": c.source_url}
            for side in quote_sides(c):
                token_id = c.yes_token_id if side == "yes" else c.no_token_id
                book = books.get(str(token_id)) if token_id else None
                if not book:
                    quote = {"side": side, "token_id": token_id, "quote_price": None, "scoring_eligible_estimate": False, "missing": "live_book_unavailable"}
                else:
                    quote = build_live_quote(c, side, book, quote_size=args.quote_size, tick_size=args.tick_size)
                previous_records.append({
                    "snapshot_index": idx,
                    "quoted_at_utc": iso_z(now),
                    "quoted_at_epoch": epoch_ts,
                    "slug": c.slug,
                    "condition_id": c.condition_id,
                    "question": c.question,
                    "source_url": c.source_url,
                    "quote": quote,
                    "safety": safety_object(),
                })
            lifecycle.append({"snapshot_index": idx, "snapshot_at_utc": iso_z(now), "candidate": candidate_snapshot, "book_fetch_token_count": len(token_ids)})
        previous_epoch_ts = epoch_ts
    # finalize last snapshot with a zero/short terminal trade window unless --duration 0; fetch current books/trades once more
    if previous_records:
        now = utc_now()
        epoch_ts = int(now.timestamp())
        token_ids = []
        for c in candidates:
            token_ids.extend([c.yes_token_id, c.no_token_id])
        books = fetch_books([t for t in token_ids if t])
        trades_by_condition = {}
        for c in candidates:
            if c.condition_id and c.condition_id not in trades_by_condition:
                try:
                    trades_by_condition[c.condition_id] = fetch_trades(c.condition_id, start_ts=previous_epoch_ts, end_ts=epoch_ts, limit=args.trade_limit)
                except Exception as e:
                    trades_by_condition[c.condition_id] = [{"error": f"trade_fetch_failed: {e}"}]
        for rec in previous_records:
            token = rec.get("quote", {}).get("token_id")
            trades = trades_by_condition.get(str(rec.get("condition_id")), [])
            if trades and isinstance(trades[0], Mapping) and "error" in trades[0]:
                rec["would_fill"] = {"classification": "unknown", "known": False, "basis": trades[0]["error"]}
            else:
                rec["would_fill"] = classify_quote(rec["quote"], trades, books.get(str(token)) if token else None, tick_size=args.tick_size)
            rec["finalized_at_utc"] = iso_z(now)
            quote_records.append(rec)
    counts: dict[str, int] = {}
    for r in quote_records:
        cls = str((r.get("would_fill") or {}).get("classification") or "unknown")
        counts[cls] = counts.get(cls, 0) + 1
    computable = [r for r in quote_records if (r.get("would_fill") or {}).get("classification") == "conservative"]
    report = {
        "classification": "shadow_quote_lifecycle_live_diagnostic_only",
        "safety_mode": SAFETY_MODE,
        "safety": safety_object(),
        "source_manifest": str(source),
        "started_at_utc": iso_z(started),
        "finished_at_utc": iso_z(),
        "duration_secs": args.duration_secs,
        "interval_secs": args.interval_secs,
        "candidate_count": len(candidates),
        "quote_record_count": len(quote_records),
        "would_fill_classification_counts": counts,
        "conservative_would_fill_count": counts.get("conservative", 0),
        "optimistic_would_fill_count": counts.get("optimistic", 0),
        "unknown_would_fill_count": counts.get("unknown", 0),
        "post_only_cancel_reprice_count": counts.get("post_only_cancel_reprice", 0),
        "reward_ev_computable_count": len(computable),
        "profitable_edge_demonstrated": False,
        "live_ready": False,
        "notes": [
            "Read-only public CLOB/Data API observation; no orders, no signatures, no credentials.",
            "Conservative would-fill requires trade size at/through quote to exceed displayed queue ahead plus own size.",
            "Reward EV remains not a profit claim until reward amount/share and conservative exit-risk evidence are sufficient.",
        ],
    }
    qpath = output_dir / f"shadow_quote_records_{ts}.jsonl"
    with qpath.open("w", encoding="utf-8") as f:
        for r in quote_records:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    jpath = output_dir / f"shadow_quote_lifecycle_live_{ts}.json"
    mpath = output_dir / f"shadow_quote_lifecycle_live_{ts}.md"
    report["output_files"] = {"json": str(jpath), "markdown": str(mpath), "quote_records_jsonl": str(qpath)}
    jpath.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mpath.write_text("\n".join([
        "# Shadow Quote Lifecycle Live Evidence",
        "",
        f"- classification: {report['classification']}",
        f"- source_manifest: {report['source_manifest']}",
        f"- candidate_count: {report['candidate_count']}",
        f"- quote_record_count: {report['quote_record_count']}",
        f"- would_fill_classification_counts: {report['would_fill_classification_counts']}",
        f"- reward_ev_computable_count: {report['reward_ev_computable_count']}",
        f"- profitable_edge_demonstrated: {report['profitable_edge_demonstrated']}",
        f"- safety: {report['safety']}",
    ]) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live read-only shadow quote lifecycle recorder with L2/queue/trade evidence.")
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--duration-secs", type=float, default=300.0)
    parser.add_argument("--interval-secs", type=float, default=60.0)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--quote-size", type=float, default=5.0)
    parser.add_argument("--tick-size", type=float, default=0.001)
    parser.add_argument("--trade-limit", type=int, default=500)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--no-sleep", action="store_true")
    args = parser.parse_args(argv)
    if args.duration_secs < 0:
        parser.error("--duration-secs must be >= 0")
    if args.interval_secs <= 0:
        parser.error("--interval-secs must be > 0")
    if args.duration_secs > 0 and args.interval_secs > args.duration_secs:
        parser.error("--interval-secs must be <= --duration-secs")
    if not 1 <= args.max_candidates <= 10:
        parser.error("--max-candidates must be between 1 and 10 for live evidence runs")
    report = run(args)
    print(json.dumps({
        "classification": report["classification"],
        "candidate_count": report["candidate_count"],
        "quote_record_count": report["quote_record_count"],
        "would_fill_classification_counts": report["would_fill_classification_counts"],
        "reward_ev_computable_count": report["reward_ev_computable_count"],
        "profitable_edge_demonstrated": report["profitable_edge_demonstrated"],
        "safety": report["safety"],
        "output_files": report["output_files"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
