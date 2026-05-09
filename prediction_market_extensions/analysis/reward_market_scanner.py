"""Read-only Polymarket reward-market scanner primitives.

This module converts autonomous market-scan artifacts into a bounded manifest for
later PMBT backtests or Homerun shadow-forward logging. It never reads secrets
and contains no order placement/signing code.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _parse_jsonish(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default if default is not None else value
    return value


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _levels(book: dict[str, Any], side: str) -> list[dict[str, float]]:
    raw = book.get(side) or book.get(f"{side}s") or book.get(f"{side}_levels") or []
    if isinstance(raw, (int, float)):
        return []
    out: list[dict[str, float]] = []
    for row in raw or []:
        if isinstance(row, dict):
            price = _float(row.get("price") or row.get("p"))
            size = _float(row.get("size") or row.get("s") or row.get("quantity"))
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            price, size = _float(row[0]), _float(row[1])
        else:
            continue
        if price > 0 and size > 0:
            out.append({"price": price, "size": size})
    return out


def _level_count(book: dict[str, Any], key: str, parsed_levels: list[dict[str, float]]) -> int:
    value = book.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return len(value)
    return len(parsed_levels)


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _book_metrics(book: dict[str, Any]) -> dict[str, Any]:
    bids = _levels(book, "bid")
    asks = _levels(book, "ask")
    # Current public scan artifacts may store top-of-book as either
    # best_bid/best_ask or compact bid/ask scalar fields. Accept both, but
    # never infer a book when one side is missing.
    best_bid = _float(_first_present(book, ("best_bid", "bid")), 0.0) or (max((x["price"] for x in bids), default=0.0))
    best_ask = _float(_first_present(book, ("best_ask", "ask")), 0.0) or (min((x["price"] for x in asks), default=0.0))
    spread = _float(book.get("spread"), 0.0) or (best_ask - best_bid if best_ask and best_bid else 0.0)
    depth_5c_bid = sum(x["size"] for x in bids if best_bid and x["price"] >= max(0.0, best_bid - 0.05))
    depth_5c_ask = sum(x["size"] for x in asks if best_ask and x["price"] <= min(1.0, best_ask + 0.05))
    # Some autonomous scans persist bounded depth aggregates rather than raw
    # price levels. Prefer exact level-derived 5c depth when present, then use
    # the conservative bounded aggregate as a depth proxy for scoring.
    depth_5c_bid = depth_5c_bid or _float(_first_present(book, ("depth_bid_5c", "depth_bid_top10", "depth_bid_2c", "bid_size")), 0.0)
    depth_5c_ask = depth_5c_ask or _float(_first_present(book, ("depth_ask_5c", "depth_ask_top10", "depth_ask_2c", "ask_size")), 0.0)
    return {
        "token_id": str(book.get("token_id") or book.get("asset_id") or ""),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": max(0.0, spread),
        "bid_levels": _level_count(book, "bid_levels", bids),
        "ask_levels": _level_count(book, "ask_levels", asks),
        "depth_5c_bid": depth_5c_bid,
        "depth_5c_ask": depth_5c_ask,
        "ok": bool(book.get("ok", True)) and best_bid > 0 and best_ask > 0 and best_ask > best_bid,
    }

REWARD_FIELD_ALLOWLIST = {
    "rewardsMinSize",
    "rewardsMaxSpread",
    "umaReward",
    "clobRewards",
    "clobRewards_count",
    "clobRewards_daily_rate_total",
    "liquidityRewards",
    "text",
}


def _reward_fields(candidate: dict[str, Any]) -> dict[str, Any]:
    evidence = candidate.get("reward_evidence") or {}
    if isinstance(evidence, str):
        evidence = {"text": evidence}
    out = {k: v for k, v in evidence.items() if k in REWARD_FIELD_ALLOWLIST} if isinstance(evidence, dict) else {}
    for key in REWARD_FIELD_ALLOWLIST - {"text"}:
        if candidate.get(key) not in (None, "", [], {}):
            out[key] = candidate[key]
    return out


def _string_list(value: Any) -> list[str]:
    parsed = _parse_jsonish(value, value)
    if parsed is None:
        return []
    if isinstance(parsed, str):
        return [parsed]
    if isinstance(parsed, list):
        return [str(x) for x in parsed if str(x)]
    return [str(parsed)]


def _books_from_flat_candidate(candidate: dict[str, Any], tokens: list[str]) -> list[dict[str, Any]]:
    """Build minimal book rows from flattened/named scan fields when full books are absent."""
    named_books = (candidate.get("yes_book"), candidate.get("no_book"))
    if all(isinstance(book, dict) for book in named_books):
        books = []
        for book, token in zip(named_books, tokens, strict=True):
            observed_ids = {str(book.get(key)) for key in ("token_id", "asset_id") if book.get(key)}
            if observed_ids and observed_ids != {token}:
                return []
            row = dict(book)
            row.setdefault("token_id", token)
            row.setdefault("asset_id", token)
            books.append(row)
        return books

    prefixes = (("yes", tokens[0]), ("no", tokens[1]))
    books: list[dict[str, Any]] = []
    for prefix, token in prefixes:
        best_bid = candidate.get(f"{prefix}_best_bid")
        best_ask = candidate.get(f"{prefix}_best_ask")
        if best_bid in (None, "") or best_ask in (None, ""):
            return []
        books.append({
            "token_id": token,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": candidate.get(f"{prefix}_spread"),
            "bid_levels": candidate.get(f"{prefix}_bid_levels", 0),
            "ask_levels": candidate.get(f"{prefix}_ask_levels", 0),
            "depth_bid_2c": candidate.get(f"{prefix}_depth_bid_2c"),
            "depth_ask_2c": candidate.get(f"{prefix}_depth_ask_2c"),
        })
    return books


def canonical_yes_no_tokens(candidate: dict[str, Any]) -> tuple[list[str] | None, list[str], str | None]:
    canonical_tokens = _parse_jsonish(candidate.get("clobTokenIds_yes_no"), None)
    side_tokens = None
    if candidate.get("yes_token_id") not in (None, "") or candidate.get("no_token_id") not in (None, ""):
        side_tokens = [candidate.get("yes_token_id"), candidate.get("no_token_id")]
    raw_tokens = _parse_jsonish(candidate.get("clobTokenIds") or candidate.get("clob_token_ids"), []) or []
    outcomes = _parse_jsonish(candidate.get("outcomes") or candidate.get("outcome_names"), []) or []
    outcomes = [str(x).strip().lower() for x in outcomes]
    tokens_source = canonical_tokens if canonical_tokens is not None else (side_tokens if side_tokens is not None else raw_tokens)
    tokens = [str(x) for x in (tokens_source or []) if str(x)]
    if len(tokens) != 2 or len(set(tokens)) != 2:
        return None, outcomes, "requires_exactly_two_unique_clob_token_ids"
    if canonical_tokens is not None or side_tokens is not None:
        # Side-scoped token fields are already explicit Yes/No evidence in the
        # current autonomous scan schema. Keep fail-closed uniqueness checks, but
        # do not require a redundant outcomes array for these canonical sources.
        return tokens, outcomes or ["yes", "no"], None
    if len(outcomes) != 2 or set(outcomes) != {"yes", "no"}:
        return None, outcomes, "requires_explicit_binary_yes_no_outcomes"
    yes_idx = outcomes.index("yes")
    no_idx = outcomes.index("no")
    return [tokens[yes_idx], tokens[no_idx]], outcomes, None


def iter_candidates(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for key in ("candidates", "top_candidates", "markets"):
        rows = payload.get(key)
        if isinstance(rows, list):
            yield from (x for x in rows if isinstance(x, dict))
            return
    events = payload.get("events")
    if isinstance(events, list):
        for event in events:
            for market in event.get("markets", []) if isinstance(event, dict) else []:
                if isinstance(market, dict):
                    merged = dict(market)
                    merged.setdefault("event_title", event.get("title") or event.get("slug"))
                    yield merged


@dataclass
class RewardMarketRow:
    market_id: str
    condition_id: str
    slug: str
    question: str
    yes_token_id: str
    no_token_id: str
    end_time_utc: str | None
    age_days_proxy: float | None
    hours_to_end: float | None
    yes_price: float
    avg_spread: float
    max_spread: float
    min_two_sided_depth_5c: float
    volume: float
    liquidity: float
    volume_24h: float
    volatility_proxy: float
    reward_evidence: dict[str, Any]
    reward_category: str
    accidental_fill_risk_flags: list[str]
    strategy_fits: list[str]
    source_tags: list[str]
    score: float


class RewardMarketScanner:
    def __init__(self, now: datetime | None = None) -> None:
        self.now = now or datetime.now(timezone.utc)

    def score_candidate(self, candidate: dict[str, Any]) -> RewardMarketRow | dict[str, Any]:
        tokens, outcomes, token_error = canonical_yes_no_tokens(candidate)
        end = _dt(candidate.get("endDate") or candidate.get("end_date") or candidate.get("end_time_utc"))
        if token_error:
            return {"skip_reason": token_error, "question": candidate.get("question"), "slug": candidate.get("slug")}
        if end and end <= self.now:
            return {"skip_reason": "expired_market", "question": candidate.get("question"), "slug": candidate.get("slug")}
        books = candidate.get("books") or []
        if not books:
            books = _books_from_flat_candidate(candidate, tokens)
        book_metrics = [_book_metrics(b) for b in books if isinstance(b, dict)]
        if len(book_metrics) != 2 or not all(b["ok"] for b in book_metrics):
            return {"skip_reason": "requires_complete_two_sided_books", "question": candidate.get("question"), "slug": candidate.get("slug")}
        books_by_token = {b["token_id"]: b for b in book_metrics if b["token_id"]}
        if set(books_by_token) != set(tokens):
            return {"skip_reason": "book_token_ids_must_match_yes_no_tokens", "question": candidate.get("question"), "slug": candidate.get("slug")}
        book_metrics = [books_by_token[tokens[0]], books_by_token[tokens[1]]]
        spreads = [b["spread"] for b in book_metrics]
        depths = [min(b["depth_5c_bid"], b["depth_5c_ask"]) for b in book_metrics]
        avg_spread = _float(candidate.get("avg_spread"), sum(spreads) / len(spreads))
        max_spread = _float(candidate.get("max_spread"), max(spreads))
        min_depth = min(depths) if depths else 0.0
        volume = _float(candidate.get("volume") or candidate.get("volumeNum"))
        liquidity = _float(candidate.get("liquidity") or candidate.get("liquidityNum"))
        volume_24h = _float(candidate.get("volume24hr") or candidate.get("volume_24h"))
        yes_price = _float(candidate.get("yes_price")) or (book_metrics[0]["best_bid"] + book_metrics[0]["best_ask"]) / 2
        hours_to_end = ((end - self.now).total_seconds() / 3600.0) if end else None
        rewards = _reward_fields(candidate)
        flags: list[str] = []
        if max_spread > 0.03:
            flags.append("wide_spread")
        if min_depth < 25:
            flags.append("thin_two_sided_depth")
        if hours_to_end is not None and hours_to_end < 48:
            flags.append("near_expiry")
        if yes_price < 0.03 or yes_price > 0.97:
            flags.append("tail_price_gap_risk")
        if not rewards:
            flags.append("no_explicit_reward_evidence")
        reward_category = "explicit_reward" if rewards else "liquidity_proxy"
        volatility_proxy = volume_24h / max(volume, 1.0)
        score = 0.0
        score += 3.0 if rewards else 0.5
        score += min(2.0, math.log10(max(liquidity, 1.0)) / 3.0)
        score += min(2.0, math.log10(max(volume_24h, 1.0)) / 3.0)
        score += min(1.5, min_depth / 1000.0)
        score -= min(3.0, max_spread * 40.0)
        score -= 0.35 * len(flags)
        return RewardMarketRow(
            market_id=str(candidate.get("id") or candidate.get("market_id") or ""),
            condition_id=str(candidate.get("conditionId") or candidate.get("condition_id") or ""),
            slug=str(candidate.get("slug") or ""),
            question=str(candidate.get("question") or candidate.get("title") or ""),
            yes_token_id=tokens[0], no_token_id=tokens[1],
            end_time_utc=end.isoformat().replace("+00:00", "Z") if end else None,
            age_days_proxy=None,
            hours_to_end=hours_to_end,
            yes_price=yes_price,
            avg_spread=avg_spread,
            max_spread=max_spread,
            min_two_sided_depth_5c=min_depth,
            volume=volume,
            liquidity=liquidity,
            volume_24h=volume_24h,
            volatility_proxy=volatility_proxy,
            reward_evidence=rewards,
            reward_category=reward_category,
            accidental_fill_risk_flags=flags,
            strategy_fits=_string_list(candidate.get("strategy_fits") or candidate.get("strategy_fit")),
            source_tags=_string_list(candidate.get("source_tags")),
            score=round(score, 6),
        )

    def build_manifest(self, scan_payload: dict[str, Any], *, source_path: str, limit: int = 25) -> dict[str, Any]:
        rows: list[RewardMarketRow] = []
        skipped: list[dict[str, Any]] = []
        for candidate in iter_candidates(scan_payload):
            result = self.score_candidate(candidate)
            if isinstance(result, RewardMarketRow):
                rows.append(result)
            else:
                skipped.append(result)
        rows.sort(key=lambda r: r.score, reverse=True)
        if limit >= 0:
            rows = rows[:limit]
        return {
            "schema_version": "reward-market-manifest/v1",
            "generated_at_utc": self.now.isoformat().replace("+00:00", "Z"),
            "source_path": source_path,
            "safety_mode": "shadow_backtest_only_no_live_orders",
            "eligibility_rules": [
                "exactly_two_unique_clob_token_ids",
                "explicit_binary_yes_no_outcomes",
                "unexpired_market",
                "complete_two_sided_books_or_flat_top_of_book_for_both_tokens",
                "explicit_reward_evidence_preferred_but_not_required_for_proxy_rows",
            ],
            "scoring_notes": [
                "reward evidence, liquidity, 24h volume, and measured two-sided 5c depth increase score",
                "wide spread, thin/unknown depth, near expiry, tail price, and missing reward evidence add risk flags and reduce score",
                "flat top-of-book rows are accepted for candidate triage but carry unknown-depth risk until full books are available",
                "manifest is for PMBT/backtest/shadow logging only",
            ],
            "candidates": [asdict(r) for r in rows],
            "skipped": skipped[:50],
            "summary": {"eligible_count": len(rows), "skipped_count": len(skipped)},
        }


def load_scan(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if p.suffix == ".jsonl":
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        return {"candidates": rows, "metadata": {"jsonl_source": str(p)}}
    return json.loads(p.read_text())
