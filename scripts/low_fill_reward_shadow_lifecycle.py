from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SAFETY_MODE = "shadow_backtest_lifecycle_only_no_live_orders"
DEFAULT_OUTPUT_DIR = Path("/opt/polymarket-lab/reports/reward_scanner/shadow_lifecycle")
DEFAULT_SOURCE_PATTERNS = (
    "/opt/polymarket-lab/autoresearch/reward_scanner/manifests/reward_scanner_manifest_*.json",
    "/opt/polymarket-lab/reports/strategy_candidates/strategy_candidate_summary_*.json",
    "/opt/polymarket-lab/reports/strategy_candidates/strategy_candidates_*.json",
    "/opt/polymarket-lab/reports/strategy_candidates/strategy_candidate_report_*.json",
    "/opt/polymarket-lab/reports/autonomous/job-A-reward-scanner/job_A_reward_scanner_*.json",
    "/opt/polymarket-lab/autoresearch/reward_scanner/manifests/reward_market_manifest_*.json",
)
DEFAULT_MAX_CANDIDATES = 20
DEFAULT_DURATION_SECS = 60.0
DEFAULT_INTERVAL_SECS = 15.0
DEFAULT_QUOTE_SIZE = 5.0
DEFAULT_TICK_SIZE = 0.001
REWARD_EVIDENCE_KEYS = (
    "clobRewards",
    "rewards",
    "rewardsMinSize",
    "rewardsMaxSpread",
    "umaReward",
)
TOKEN_BLOCKERS = {
    "duplicate_yes_no_token_ids",
    "incomplete_side_scoped_yes_no_token_ids",
    "invalid_or_duplicate_clob_token_ids",
    "missing_yes_no_outcome_mapping_for_clob_token_ids",
    "non_binary_outcome_mapping",
}


@dataclass(frozen=True)
class BookSide:
    bid: float | None
    ask: float | None
    bid_size: float | None
    ask_size: float | None
    depth_bid: float | None
    depth_ask: float | None
    token_id: str | None


@dataclass(frozen=True)
class Candidate:
    slug: str
    question: str | None
    condition_id: str | None
    source_url: str | None
    yes_token_id: str | None
    no_token_id: str | None
    yes_mid: float | None
    no_mid: float | None
    yes_spread: float | None
    no_spread: float | None
    liquidity: float | None
    volume: float | None
    reward_min_size: float | None
    reward_max_spread_raw: float | None
    reward_max_spread: float | None
    reward_value_raw: Any
    yes_book: BookSide
    no_book: BookSide
    source_rank: float | None
    source_path: str | None
    source_tags: tuple[str, ...]
    source_blockers: tuple[str, ...]


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


def _utc_now() -> datetime:
    return datetime.now(tz=UTC).replace(microsecond=0)


def _utc_stamp(dt: datetime | None = None) -> str:
    return (dt or _utc_now()).strftime("%Y%m%dT%H%M%SZ")


def _iso_z(dt: datetime | None = None) -> str:
    return (dt or _utc_now()).isoformat().replace("+00:00", "Z")


def _parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _clean_string(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _first_present(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _is_present(value: Any) -> bool:
    return value not in (None, "", [], {}, False)


def _strategy_tags(row: Mapping[str, Any]) -> tuple[str, ...]:
    value = row.get("strategy_fits") or row.get("source_tags") or row.get("source_tag")
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in _as_list(value) if str(item).strip())


def _blockers(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(item) for item in _as_list(row.get("blockers")) if str(item).strip())


def _reward_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}

    def add_allowlisted(source: Mapping[str, Any]) -> None:
        for key in REWARD_EVIDENCE_KEYS:
            if _is_present(source.get(key)):
                evidence[key] = source[key]

    raw = row.get("reward_evidence")
    if isinstance(raw, Mapping):
        add_allowlisted(raw)
        nested = raw.get("reward_evidence")
        if isinstance(nested, Mapping):
            add_allowlisted(nested)
    add_allowlisted(row)
    rewards = evidence.get("clobRewards") or evidence.get("rewards")
    if isinstance(rewards, Sequence) and not isinstance(rewards, (str, bytes, bytearray)):
        for reward in rewards:
            if not isinstance(reward, Mapping):
                continue
            if not _is_present(evidence.get("rewardsMinSize")):
                min_size = reward.get("min_size") or reward.get("minSize")
                if _is_present(min_size):
                    evidence["rewardsMinSize"] = min_size
            if not _is_present(evidence.get("rewardsMaxSpread")):
                max_spread = reward.get("max_spread") or reward.get("maxSpread")
                if _is_present(max_spread):
                    evidence["rewardsMaxSpread"] = max_spread
            if not _is_present(evidence.get("umaReward")):
                reward_value = reward.get("reward") or reward.get("amount")
                if _is_present(reward_value):
                    evidence["umaReward"] = reward_value
    return evidence


def _has_reward_candidate_signal(row: Mapping[str, Any]) -> bool:
    if _reward_evidence(row):
        return True
    watchlist = row.get("low_fill_reward_maker_watchlist")
    if isinstance(watchlist, Mapping) and any(bool(value) for value in watchlist.values()):
        return True
    tags = {tag.casefold() for tag in _strategy_tags(row)}
    return "low-fill-probability liquidity-reward maker" in tags


def _normalize_reward_spread(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    if value > 1.0:
        return value / 100.0
    return value


def _book_side(row: Mapping[str, Any], side: str) -> BookSide:
    raw = _as_mapping(row.get(f"{side}_book"))

    def first_book_value(*keys: str) -> Any:
        value = _first_present(raw, *keys)
        if value not in (None, "", [], {}):
            return value
        return _first_present(row, *(f"{side}_{key}" for key in keys))

    token_id = _clean_string(
        raw.get("token_id")
        or raw.get("asset_id")
        or row.get(f"{side}_token_id")
        or row.get(f"{side}_asset_id")
    )
    return BookSide(
        bid=_parse_float(first_book_value("bid", "best_bid")),
        ask=_parse_float(first_book_value("ask", "best_ask")),
        bid_size=_parse_float(first_book_value("bid_size", "best_bid_size", "bid_size_proxy")),
        ask_size=_parse_float(first_book_value("ask_size", "best_ask_size", "ask_size_proxy")),
        depth_bid=_parse_float(
            first_book_value("depth_bid_2c", "depth_bid_5c", "depth_bid_top10", "depth_bid")
        ),
        depth_ask=_parse_float(
            first_book_value("depth_ask_2c", "depth_ask_5c", "depth_ask_top10", "depth_ask")
        ),
        token_id=token_id,
    )


def _spread_from_book(book: BookSide) -> float | None:
    if book.bid is None or book.ask is None:
        return None
    return max(0.0, book.ask - book.bid)


def _mid_from_book(book: BookSide) -> float | None:
    if book.bid is None or book.ask is None:
        return None
    return (book.bid + book.ask) / 2.0


def _candidate_slug(row: Mapping[str, Any]) -> str:
    return str(
        _first_present(row, "slug", "market_slug", "question", "condition_id", "market_id")
        or "unknown"
    )


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return []
        value = parsed
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _outcomes(row: Mapping[str, Any]) -> list[str]:
    return [str(item).strip() for item in _json_list(row.get("outcomes")) if str(item).strip()]


def _raw_clob_token_ids(row: Mapping[str, Any]) -> list[str]:
    raw = row.get("clob_token_ids") or row.get("clobTokenIds")
    return [str(item).strip() for item in _json_list(raw) if str(item).strip()]


def _compact_book_mid(row: Mapping[str, Any], side: str) -> float | None:
    raw = row.get(f"{side}_book")
    if isinstance(raw, Mapping):
        return _parse_float(raw.get("mid"))
    return _parse_float(raw)


def _yes_no_token_ids(
    row: Mapping[str, Any], yes_book: BookSide, no_book: BookSide
) -> tuple[str | None, str | None, tuple[str, ...]]:
    side_yes = _clean_string(_first_present(row, "yes_token_id", "yes_asset_id"))
    side_no = _clean_string(_first_present(row, "no_token_id", "no_asset_id"))
    yes_token = side_yes or yes_book.token_id
    no_token = side_no or no_book.token_id
    if yes_token is not None or no_token is not None:
        if yes_token is None or no_token is None:
            return yes_token, no_token, ("incomplete_side_scoped_yes_no_token_ids",)
        if yes_token == no_token:
            return yes_token, no_token, ("duplicate_yes_no_token_ids",)
        return yes_token, no_token, ()

    raw_ids = _raw_clob_token_ids(row)
    if not raw_ids:
        return None, None, ()
    if len(raw_ids) != 2 or len(set(raw_ids)) != 2:
        return None, None, ("invalid_or_duplicate_clob_token_ids",)

    outcomes = _outcomes(row)
    normalized = [outcome.casefold() for outcome in outcomes]
    if not outcomes:
        return None, None, ("missing_yes_no_outcome_mapping_for_clob_token_ids",)
    if len(outcomes) != 2 or set(normalized) != {"yes", "no"}:
        return None, None, ("non_binary_outcome_mapping",)
    by_outcome = {outcome.casefold(): token_id for outcome, token_id in zip(outcomes, raw_ids)}
    return by_outcome["yes"], by_outcome["no"], ()


def normalize_candidate(row: Mapping[str, Any], *, source_path: str | None = None) -> Candidate:
    yes_book = _book_side(row, "yes")
    no_book = _book_side(row, "no")
    yes_token_id, no_token_id, token_blockers = _yes_no_token_ids(row, yes_book, no_book)
    evidence = _reward_evidence(row)
    reward_max_spread_raw = _parse_float(evidence.get("rewardsMaxSpread"))
    reward_min_size = _parse_float(evidence.get("rewardsMinSize"))
    yes_mid = _parse_float(_first_present(row, "yes_mid", "yes_probability", "yes_price"))
    no_mid = _parse_float(_first_present(row, "no_mid", "no_probability", "no_price"))
    if yes_mid is None:
        yes_mid = _compact_book_mid(row, "yes")
    if yes_mid is None:
        yes_mid = _mid_from_book(yes_book)
    if no_mid is None:
        no_mid = _compact_book_mid(row, "no")
    if no_mid is None:
        no_mid = _mid_from_book(no_book)
    yes_spread = _parse_float(_first_present(row, "yes_spread", "spread", "avg_spread"))
    no_spread = _parse_float(_first_present(row, "no_spread", "spread", "avg_spread"))
    if yes_spread is None:
        yes_spread = _spread_from_book(yes_book)
    if no_spread is None:
        no_spread = _spread_from_book(no_book)
    return Candidate(
        slug=_candidate_slug(row),
        question=_clean_string(row.get("question")),
        condition_id=_clean_string(_first_present(row, "condition_id", "conditionId")),
        source_url=_clean_string(
            _first_present(row, "source_market_url", "source_url", "market_url")
        ),
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        yes_mid=yes_mid,
        no_mid=no_mid,
        yes_spread=yes_spread,
        no_spread=no_spread,
        liquidity=_parse_float(_first_present(row, "liquidity", "liquidityNum")),
        volume=_parse_float(_first_present(row, "volume", "volumeNum", "volume24hr")),
        reward_min_size=reward_min_size,
        reward_max_spread_raw=reward_max_spread_raw,
        reward_max_spread=_normalize_reward_spread(reward_max_spread_raw),
        reward_value_raw=evidence.get("umaReward") or evidence.get("reward"),
        yes_book=yes_book,
        no_book=no_book,
        source_rank=_parse_float(
            _first_present(row, "rank", "selection_score", "score", "reward_proxy_score")
        ),
        source_path=_clean_string(
            row.get("source_path") or row.get("source_artifact") or source_path
        ),
        source_tags=_strategy_tags(row),
        source_blockers=tuple(dict.fromkeys((*_blockers(row), *token_blockers))),
    )


def _candidate_rows(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return [row for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        return []

    rows: list[Mapping[str, Any]] = []
    for key in (
        "low_fill_reward_maker_watchlist",
        "low_fill_reward_candidates",
        "reward_watchlist",
        "top_candidates",
        "candidates",
        "backtest_ready_candidates",
        "coverage_probe_candidates",
        "watchlist",
        "sample_watchlist",
        "sample_backtest_ready",
        "coverage_probe",
    ):
        value = payload.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            rows.extend(row for row in value if isinstance(row, Mapping))

    buckets = payload.get("buckets")
    if isinstance(buckets, Mapping):
        for value in buckets.values():
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                rows.extend(row for row in value if isinstance(row, Mapping))

    seen: set[str] = set()
    deduped: list[Mapping[str, Any]] = []
    for row in rows:
        slug = _candidate_slug(row)
        if slug in seen:
            continue
        seen.add(slug)
        deduped.append(row)
    return deduped


def _has_book_snapshot(candidate: Candidate) -> bool:
    return (
        candidate.yes_book.bid is not None
        and candidate.yes_book.ask is not None
        and candidate.no_book.bid is not None
        and candidate.no_book.ask is not None
    )


def select_candidates(
    payload: Any,
    *,
    source_path: str | None = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[Candidate]:
    selected: list[Candidate] = []
    for row in _candidate_rows(payload):
        if not _has_reward_candidate_signal(row):
            continue
        selected.append(normalize_candidate(row, source_path=source_path))
    selected.sort(
        key=lambda c: (
            0 if _has_book_snapshot(c) else 1,
            -(c.source_rank or 0.0),
            c.slug,
        )
    )
    return selected[:max_candidates]


def _json_candidates_by_mtime(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(path for path in Path("/").glob(pattern.lstrip("/")) if path.is_file())
    return sorted(set(paths), key=lambda path: path.stat().st_mtime, reverse=True)


def latest_source_manifest(patterns: Iterable[str] = DEFAULT_SOURCE_PATTERNS) -> Path:
    fallback: Path | None = None
    for path in _json_candidates_by_mtime(patterns):
        fallback = fallback or path
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        candidates = select_candidates(payload, source_path=str(path), max_candidates=10)
        if len(candidates) >= 10 and any(_has_book_snapshot(candidate) for candidate in candidates):
            return path
    if fallback is not None:
        return fallback
    raise FileNotFoundError("no source candidate artifacts found")


def load_source(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _side_values(candidate: Candidate, side: str) -> tuple[BookSide, float | None, float | None]:
    if side == "yes":
        return candidate.yes_book, candidate.yes_mid, candidate.yes_spread
    return candidate.no_book, candidate.no_mid, candidate.no_spread


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


def quote_sides(candidate: Candidate) -> list[str]:
    midpoint = candidate.yes_mid
    if midpoint is None:
        return ["yes"]
    if midpoint < 0.10 or midpoint > 0.90:
        return ["yes", "no"]
    if candidate.no_mid is not None and candidate.no_mid < midpoint:
        return ["no"]
    return ["yes"]


def _risk_label(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 0.75:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def build_quote(
    candidate: Candidate,
    side: str,
    *,
    quote_size: float = DEFAULT_QUOTE_SIZE,
    tick_size: float = DEFAULT_TICK_SIZE,
) -> dict[str, Any]:
    book, mid, spread = _side_values(candidate, side)
    min_size = candidate.reward_min_size or quote_size
    size = max(quote_size, min_size)
    price: float | None = None
    if book.bid is not None:
        price = max(tick_size, min(1.0 - tick_size, book.bid))
    elif book.ask is not None:
        price = max(tick_size, min(1.0 - tick_size, book.ask - tick_size))

    reward_band_status = "unknown_no_reward_max_spread"
    if candidate.reward_max_spread is not None and spread is not None:
        reward_band_status = (
            "in_band" if spread <= candidate.reward_max_spread else "out_of_band_spread"
        )
    elif spread is None:
        reward_band_status = "unknown_missing_spread"

    required_size_met = size >= min_size
    if price is None:
        touch_cross_status = "no_quote_missing_book"
    elif book.ask is not None and price >= book.ask:
        touch_cross_status = "would_cross_or_take_current_ask"
    elif book.bid is not None and math.isclose(price, book.bid, abs_tol=1e-12):
        touch_cross_status = "resting_at_best_bid"
    elif book.bid is not None and price < book.bid:
        touch_cross_status = "behind_best_bid"
    else:
        touch_cross_status = "inside_spread_post_only"

    known_fill = touch_cross_status == "would_cross_or_take_current_ask"
    if known_fill:
        would_have_filled = {
            "known": True,
            "estimate": True,
            "basis": "hypothetical_bid_crosses_current_ask",
        }
    elif price is None:
        would_have_filled = {
            "known": False,
            "estimate": "unknown",
            "basis": "missing_book_snapshot",
        }
    else:
        would_have_filled = {
            "known": False,
            "estimate": "unknown",
            "basis": "requires_trade_tape_or_l2_queue_position",
        }

    cancel_or_reprice_reason = "none_hold_quote"
    if price is None:
        cancel_or_reprice_reason = "no_quote_missing_book"
    elif known_fill:
        cancel_or_reprice_reason = "cancel_post_only_quote_would_cross"
    elif reward_band_status == "out_of_band_spread":
        cancel_or_reprice_reason = "reprice_spread_outside_reward_band"
    elif not required_size_met:
        cancel_or_reprice_reason = "resize_required_size_not_met"

    relative_tick_cost = _safe_div(tick_size, mid)
    relative_half_spread = _safe_div((spread / 2.0) if spread is not None else None, mid)
    same_side_size = book.bid_size if price is not None and book.bid is not None else None
    queue_ahead_proxy = _safe_div(size, same_side_size)
    depth_near_touch = None
    if book.depth_bid is not None or book.depth_ask is not None:
        depth_near_touch = (book.depth_bid or 0.0) + (book.depth_ask or 0.0)
    depth_coverage = _safe_div(size, depth_near_touch)

    risk_score_parts = [
        min(1.0, relative_tick_cost or 0.0),
        min(1.0, (relative_half_spread or 0.0) * 2.0),
        min(1.0, (queue_ahead_proxy or 0.0) * 10.0),
        min(1.0, (depth_coverage or 0.0) * 10.0),
    ]
    if mid is not None and (mid <= 0.01 or mid >= 0.99):
        risk_score_parts.append(0.75)
    adverse_selection_score = max(risk_score_parts) if risk_score_parts else None
    exit_slippage_proxy = relative_half_spread

    scoring_eligible = (
        price is not None
        and not known_fill
        and reward_band_status == "in_band"
        and required_size_met
    )
    reward_score_proxy = 0.0
    if scoring_eligible and candidate.reward_max_spread is not None and spread is not None:
        tightness = max(0.0, 1.0 - min(1.0, spread / candidate.reward_max_spread))
        reward_score_proxy = size * tightness

    return {
        "side": side,
        "token_id": candidate.yes_token_id if side == "yes" else candidate.no_token_id,
        "action": "post_only_bid_shadow_observation",
        "price": _round(price),
        "size": _round(size),
        "mid": _round(mid),
        "best_bid": _round(book.bid),
        "best_ask": _round(book.ask),
        "spread": _round(spread),
        "reward_min_size": _round(candidate.reward_min_size),
        "reward_max_spread_raw": _round(candidate.reward_max_spread_raw),
        "reward_max_spread_decimal": _round(candidate.reward_max_spread),
        "reward_band_status": reward_band_status,
        "required_size_met": required_size_met,
        "hypothetical_quote_touch_cross_status": touch_cross_status,
        "would_have_filled_estimate": would_have_filled,
        "cancel_or_reprice_reason": cancel_or_reprice_reason,
        "scoring_eligible_estimate": scoring_eligible,
        "reward_score_proxy": _round(reward_score_proxy),
        "adverse_selection_risk_proxy": {
            "score": _round(adverse_selection_score),
            "label": _risk_label(adverse_selection_score),
            "relative_tick_cost": _round(relative_tick_cost),
            "queue_ahead_size_ratio": _round(queue_ahead_proxy),
        },
        "exit_slippage_proxy": {
            "half_spread_over_mid": _round(exit_slippage_proxy),
            "label": _risk_label(exit_slippage_proxy),
        },
    }


def evaluate_snapshot(
    candidate: Candidate,
    *,
    snapshot_index: int,
    snapshot_at_utc: str,
    snapshot_weight_secs: float,
    quote_size: float = DEFAULT_QUOTE_SIZE,
    tick_size: float = DEFAULT_TICK_SIZE,
) -> dict[str, Any]:
    sides = quote_sides(candidate)
    quotes = [
        build_quote(candidate, side, quote_size=quote_size, tick_size=tick_size) for side in sides
    ]
    double_sided_required = len(sides) == 2
    eligible_quotes = [quote for quote in quotes if quote["scoring_eligible_estimate"]]
    scoring_eligible = (
        len(eligible_quotes) == len(quotes) if double_sided_required else bool(eligible_quotes)
    )
    if any(quote["cancel_or_reprice_reason"] != "none_hold_quote" for quote in quotes):
        cancel_reason = next(
            quote["cancel_or_reprice_reason"]
            for quote in quotes
            if quote["cancel_or_reprice_reason"] != "none_hold_quote"
        )
    else:
        cancel_reason = "none_hold_quote"
    fill_known = any(quote["would_have_filled_estimate"]["known"] for quote in quotes)
    fill_estimate = any(quote["would_have_filled_estimate"]["estimate"] is True for quote in quotes)
    reward_score_proxy = sum(float(quote["reward_score_proxy"] or 0.0) for quote in quotes)
    max_adverse_risk = max(
        (float(quote["adverse_selection_risk_proxy"]["score"] or 0.0) for quote in quotes),
        default=0.0,
    )
    max_exit_slippage = max(
        (float(quote["exit_slippage_proxy"]["half_spread_over_mid"] or 0.0) for quote in quotes),
        default=0.0,
    )
    return {
        "snapshot_index": snapshot_index,
        "snapshot_at_utc": snapshot_at_utc,
        "slug": candidate.slug,
        "question": candidate.question,
        "condition_id": candidate.condition_id,
        "source_url": candidate.source_url,
        "snapshot_source": "source_manifest_top_of_book_snapshot",
        "trade_snapshot_status": "not_available_in_source_artifact",
        "double_sided_required": double_sided_required,
        "quote_sides": sides,
        "quotes": quotes,
        "scoring_eligible_estimate": scoring_eligible,
        "reward_band_status": ("in_band" if scoring_eligible else "not_scoring_or_unknown"),
        "required_size_met": all(quote["required_size_met"] for quote in quotes),
        "hypothetical_quote_touch_cross_status": ";".join(
            quote["hypothetical_quote_touch_cross_status"] for quote in quotes
        ),
        "would_have_filled_estimate": {
            "known": fill_known,
            "estimate": fill_estimate if fill_known else "unknown",
            "basis": (
                "current_book_cross" if fill_known else "requires_trade_tape_or_l2_queue_position"
            ),
        },
        "cancel_or_reprice_reason": cancel_reason,
        "time_in_band_contribution_secs": _round(snapshot_weight_secs if scoring_eligible else 0.0),
        "reward_score_proxy": _round(reward_score_proxy),
        "adverse_selection_risk_proxy": {
            "score": _round(max_adverse_risk),
            "label": _risk_label(max_adverse_risk),
        },
        "exit_slippage_proxy": {
            "half_spread_over_mid": _round(max_exit_slippage),
            "label": _risk_label(max_exit_slippage),
        },
        "safety": safety_object(),
    }


def _snapshot_schedule(duration_secs: float, interval_secs: float) -> list[float]:
    if duration_secs <= 0:
        return [0.0]
    count = int(math.ceil(duration_secs / interval_secs)) + 1
    return [min(duration_secs, i * interval_secs) for i in range(count)]


def collect_snapshots(
    candidates: Sequence[Candidate],
    *,
    duration_secs: float,
    interval_secs: float,
    quote_size: float = DEFAULT_QUOTE_SIZE,
    tick_size: float = DEFAULT_TICK_SIZE,
    sleep: bool = True,
) -> list[dict[str, Any]]:
    offsets = _snapshot_schedule(duration_secs, interval_secs)
    snapshots: list[dict[str, Any]] = []
    previous_offset = 0.0
    for snapshot_index, offset in enumerate(offsets):
        if sleep and snapshot_index > 0:
            time.sleep(max(0.0, offset - previous_offset))
        previous_offset = offset
        snapshot_at_utc = _iso_z()
        next_offset = (
            offsets[snapshot_index + 1] if snapshot_index + 1 < len(offsets) else duration_secs
        )
        weight = max(0.0, next_offset - offset)
        for candidate in candidates:
            snapshots.append(
                evaluate_snapshot(
                    candidate,
                    snapshot_index=snapshot_index,
                    snapshot_at_utc=snapshot_at_utc,
                    snapshot_weight_secs=weight,
                    quote_size=quote_size,
                    tick_size=tick_size,
                )
            )
    return snapshots


def aggregate_candidates(
    candidates: Sequence[Candidate],
    snapshots: Sequence[Mapping[str, Any]],
    *,
    duration_secs: float,
) -> list[dict[str, Any]]:
    by_slug: dict[str, list[Mapping[str, Any]]] = {candidate.slug: [] for candidate in candidates}
    for snapshot in snapshots:
        slug = str(snapshot.get("slug"))
        if slug in by_slug:
            by_slug[slug].append(snapshot)

    raw: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_snapshots = by_slug.get(candidate.slug, [])
        time_in_band_secs = sum(
            float(snapshot.get("time_in_band_contribution_secs") or 0.0)
            for snapshot in candidate_snapshots
        )
        reward_score_proxy = sum(
            float(snapshot.get("reward_score_proxy") or 0.0) for snapshot in candidate_snapshots
        )
        known_fill_count = sum(
            1
            for snapshot in candidate_snapshots
            if _as_mapping(snapshot.get("would_have_filled_estimate")).get("known") is True
        )
        estimated_fill_count = sum(
            1
            for snapshot in candidate_snapshots
            if _as_mapping(snapshot.get("would_have_filled_estimate")).get("estimate") is True
        )
        eligible_count = sum(
            1
            for snapshot in candidate_snapshots
            if snapshot.get("scoring_eligible_estimate") is True
        )
        max_adverse = max(
            (
                float(_as_mapping(snapshot.get("adverse_selection_risk_proxy")).get("score") or 0.0)
                for snapshot in candidate_snapshots
            ),
            default=0.0,
        )
        max_exit = max(
            (
                float(
                    _as_mapping(snapshot.get("exit_slippage_proxy")).get("half_spread_over_mid")
                    or 0.0
                )
                for snapshot in candidate_snapshots
            ),
            default=0.0,
        )
        raw.append(
            {
                "slug": candidate.slug,
                "question": candidate.question,
                "condition_id": candidate.condition_id,
                "source_url": candidate.source_url,
                "yes_mid": _round(candidate.yes_mid),
                "no_mid": _round(candidate.no_mid),
                "reward_min_size": _round(candidate.reward_min_size),
                "reward_max_spread_raw": _round(candidate.reward_max_spread_raw),
                "reward_max_spread_decimal": _round(candidate.reward_max_spread),
                "double_sided_required": quote_sides(candidate) == ["yes", "no"],
                "snapshot_count": len(candidate_snapshots),
                "scoring_eligible_snapshot_count": eligible_count,
                "time_in_band_secs": _round(time_in_band_secs),
                "time_in_band_ratio": _round(_safe_div(time_in_band_secs, duration_secs)),
                "would_have_filled_known_count": known_fill_count,
                "would_have_filled_estimated_true_count": estimated_fill_count,
                "would_have_filled_status": (
                    "known"
                    if known_fill_count
                    else "unknown_requires_trade_tape_or_l2_queue_position"
                ),
                "adverse_selection_risk": {
                    "score": _round(max_adverse),
                    "label": _risk_label(max_adverse),
                },
                "exit_slippage_proxy": {
                    "half_spread_over_mid": _round(max_exit),
                    "label": _risk_label(max_exit),
                },
                "reward_score_proxy": _round(reward_score_proxy),
                "expected_reward_ev_minus_expected_loss_classification": (
                    "unknown_missing_reward_amount_or_fill_loss_distribution"
                ),
                "source_blockers": list(candidate.source_blockers),
                "safety": safety_object(),
            }
        )

    total_reward_score = sum(float(row["reward_score_proxy"] or 0.0) for row in raw)
    for row in raw:
        row["estimated_reward_score_share_proxy"] = _round(
            _safe_div(float(row["reward_score_proxy"] or 0.0), total_reward_score)
        )
    return raw


def build_report(
    source_manifest: Path,
    *,
    duration_secs: float,
    interval_secs: float,
    max_candidates: int,
    quote_size: float = DEFAULT_QUOTE_SIZE,
    tick_size: float = DEFAULT_TICK_SIZE,
    sleep: bool = True,
) -> dict[str, Any]:
    payload = load_source(source_manifest)
    candidates = select_candidates(
        payload, source_path=str(source_manifest), max_candidates=max_candidates
    )
    snapshots = collect_snapshots(
        candidates,
        duration_secs=duration_secs,
        interval_secs=interval_secs,
        quote_size=quote_size,
        tick_size=tick_size,
        sleep=sleep,
    )
    candidate_table = aggregate_candidates(candidates, snapshots, duration_secs=duration_secs)
    blocker_reasons: list[str] = []
    if len(candidates) < 10 and max_candidates >= 10:
        blocker_reasons.append("fewer_than_10_reward_candidates_selected")
    if not any(_has_book_snapshot(candidate) for candidate in candidates):
        blocker_reasons.append("selected_candidates_missing_top_of_book_snapshots")
    token_blockers = sorted(
        {
            blocker
            for candidate in candidates
            for blocker in candidate.source_blockers
            if blocker in TOKEN_BLOCKERS
        }
    )
    blocker_reasons.extend(token_blockers)
    known_fill_count = sum(row["would_have_filled_known_count"] for row in candidate_table)
    classification = "diagnostic_only"
    if blocker_reasons:
        classification = "blocked"
    return {
        "schema_version": 1,
        "generated_at_utc": _iso_z(),
        "mode": SAFETY_MODE,
        "classification": classification,
        "source_manifest": str(source_manifest),
        "duration_secs": duration_secs,
        "interval_secs": interval_secs,
        "candidate_count": len(candidates),
        "scheduled_snapshot_count": len(_snapshot_schedule(duration_secs, interval_secs)),
        "snapshot_count": len(snapshots),
        "quote_size_default": quote_size,
        "tick_size": tick_size,
        "safety": safety_object(),
        "no_profit_claim": True,
        "profit_verdict": "no_profit_claim_shadow_diagnostic_only",
        "blocker_reasons": blocker_reasons,
        "summary": {
            "scoring_eligible_candidate_count": sum(
                1 for row in candidate_table if row["scoring_eligible_snapshot_count"] > 0
            ),
            "would_have_filled_known_count": known_fill_count,
            "would_have_filled_unknown_count": max(0, len(candidate_table) - known_fill_count),
            "expected_reward_ev_minus_expected_loss": (
                "unknown_missing_reward_amount_or_fill_loss_distribution"
            ),
            "next_hook": "run a 2-6h shadow observation with trade tape/L2 queue evidence before any profitability claim",
        },
        "candidate_table": candidate_table,
        "snapshots": snapshots,
    }


def _csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in _as_list(report.get("candidate_table")):
        if not isinstance(candidate, Mapping):
            continue
        adverse = _as_mapping(candidate.get("adverse_selection_risk"))
        exit_slippage = _as_mapping(candidate.get("exit_slippage_proxy"))
        rows.append(
            {
                "slug": candidate.get("slug"),
                "yes_mid": candidate.get("yes_mid"),
                "no_mid": candidate.get("no_mid"),
                "double_sided_required": candidate.get("double_sided_required"),
                "snapshot_count": candidate.get("snapshot_count"),
                "scoring_eligible_snapshot_count": candidate.get("scoring_eligible_snapshot_count"),
                "time_in_band_secs": candidate.get("time_in_band_secs"),
                "time_in_band_ratio": candidate.get("time_in_band_ratio"),
                "would_have_filled_status": candidate.get("would_have_filled_status"),
                "reward_score_proxy": candidate.get("reward_score_proxy"),
                "estimated_reward_score_share_proxy": candidate.get(
                    "estimated_reward_score_share_proxy"
                ),
                "adverse_selection_risk_label": adverse.get("label"),
                "adverse_selection_risk_score": adverse.get("score"),
                "exit_slippage_proxy_label": exit_slippage.get("label"),
                "exit_slippage_proxy_half_spread_over_mid": exit_slippage.get(
                    "half_spread_over_mid"
                ),
                "expected_reward_ev_minus_expected_loss_classification": candidate.get(
                    "expected_reward_ev_minus_expected_loss_classification"
                ),
                "orders_submitted": False,
                "orders_signed": False,
                "orders_cancelled": False,
                "credentials_required": False,
                "live_trading": False,
                "worker_trading_started": False,
                "live_trading_worker_started": False,
            }
        )
    return rows


def _markdown_report(report: Mapping[str, Any]) -> str:
    safety = _as_mapping(report.get("safety"))
    summary = _as_mapping(report.get("summary"))
    lines = [
        "# Low-fill Reward Shadow Quote Lifecycle",
        "",
        f"- mode: {report.get('mode')}",
        f"- classification: {report.get('classification')}",
        f"- source_manifest: {report.get('source_manifest')}",
        f"- duration_secs: {report.get('duration_secs')}",
        f"- interval_secs: {report.get('interval_secs')}",
        f"- candidates: {report.get('candidate_count')}",
        f"- scheduled_snapshots: {report.get('scheduled_snapshot_count')}",
        f"- candidate_snapshots: {report.get('snapshot_count')}",
        f"- no_profit_claim: {report.get('no_profit_claim')}",
        ("- safety: " + ", ".join(f"{key}={value}" for key, value in sorted(safety.items()))),
        "",
        "## Summary",
        "",
        f"- scoring_eligible_candidate_count: {summary.get('scoring_eligible_candidate_count')}",
        f"- would_have_filled_known_count: {summary.get('would_have_filled_known_count')}",
        f"- would_have_filled_unknown_count: {summary.get('would_have_filled_unknown_count')}",
        (
            "- expected_reward_ev_minus_expected_loss: "
            f"{summary.get('expected_reward_ev_minus_expected_loss')}"
        ),
        f"- next_hook: {summary.get('next_hook')}",
        "",
        "No profitability claim is made. Missing trade tape/L2 queue evidence remains unknown.",
        "",
        "## Candidate Table",
        "",
        "| slug | yes_mid | double_sided | eligible_snapshots | time_in_band_secs | fill_status | reward_share_proxy | adverse_risk | exit_slippage |",
        "|---|---:|---|---:|---:|---|---:|---|---|",
    ]
    for row in _as_list(report.get("candidate_table")):
        if not isinstance(row, Mapping):
            continue
        adverse = _as_mapping(row.get("adverse_selection_risk"))
        exit_slippage = _as_mapping(row.get("exit_slippage_proxy"))
        lines.append(
            "| "
            f"{row.get('slug')} | "
            f"{row.get('yes_mid')} | "
            f"{row.get('double_sided_required')} | "
            f"{row.get('scoring_eligible_snapshot_count')} | "
            f"{row.get('time_in_band_secs')} | "
            f"{row.get('would_have_filled_status')} | "
            f"{row.get('estimated_reward_score_share_proxy')} | "
            f"{adverse.get('label')} | "
            f"{exit_slippage.get('label')} |"
        )
    blockers = _as_list(report.get("blocker_reasons"))
    if blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in blockers)
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], output_dir: Path, timestamp: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"low_fill_reward_shadow_lifecycle_{timestamp}.json"
    csv_path = output_dir / f"low_fill_reward_shadow_lifecycle_{timestamp}.csv"
    md_path = output_dir / f"low_fill_reward_shadow_lifecycle_{timestamp}.md"
    report["output_files"] = {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = _csv_rows(report)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()) if rows else ["slug"])
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    return report["output_files"]


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a shadow-only low-fill reward quote lifecycle observation."
    )
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--duration-secs", type=_non_negative_float, default=DEFAULT_DURATION_SECS)
    parser.add_argument("--interval-secs", type=_positive_float, default=DEFAULT_INTERVAL_SECS)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--quote-size", type=_positive_float, default=DEFAULT_QUOTE_SIZE)
    parser.add_argument("--tick-size", type=_positive_float, default=DEFAULT_TICK_SIZE)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument(
        "--no-sleep",
        action="store_true",
        help="Evaluate the configured lifecycle schedule without waiting between snapshots.",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.max_candidates <= 30:
        parser.error("--max-candidates must be between 1 and 30")
    if args.duration_secs > 0 and args.interval_secs > args.duration_secs:
        parser.error("--interval-secs must be <= --duration-secs when duration is positive")

    source_manifest = args.source_manifest or latest_source_manifest()
    timestamp = args.timestamp or _utc_stamp()
    report = build_report(
        source_manifest,
        duration_secs=args.duration_secs,
        interval_secs=args.interval_secs,
        max_candidates=args.max_candidates,
        quote_size=args.quote_size,
        tick_size=args.tick_size,
        sleep=not args.no_sleep,
    )
    output_files = write_outputs(report, args.output_dir, timestamp)
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "candidate_count": report["candidate_count"],
                "snapshot_count": report["snapshot_count"],
                "safety": report["safety"],
                "source_manifest": report["source_manifest"],
                "output_files": output_files,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
