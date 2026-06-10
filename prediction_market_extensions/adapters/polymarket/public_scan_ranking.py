from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

MICROPRICE = "microprice"
VOLATILITY_SPIKE_DEEP_LIMIT_MAKER = "volatility_spike_deep_limit_maker"
LOW_FILL_LIQUIDITY_REWARD_MAKER = "low_fill_liquidity_reward_maker"
STRATEGY_KEYS = (
    MICROPRICE,
    VOLATILITY_SPIKE_DEEP_LIMIT_MAKER,
    LOW_FILL_LIQUIDITY_REWARD_MAKER,
)

PUBLIC_SCAN_RANKING_SCHEMA_VERSION = "polymarket.public-scan.strategy-ranking.v2"
PUBLIC_SCAN_SAFETY_FIELDS = {
    "orders_submitted": False,
    "orders_signed": False,
    "orders_cancelled": False,
    "credentials_required": False,
    "live_trading_worker_started": False,
    "worker_trading_started": False,
}


@dataclass(frozen=True)
class PublicScanRankingRules:
    """Objective-specific public scan ranking rules.

    These rules are read-only scan heuristics for shadow/backtest handoffs.
    They do not claim edge or live trading eligibility.
    """

    tight_spread: float = 0.03
    usable_spread: float = 0.06
    microprice_min_mid: float = 0.08
    microprice_max_mid: float = 0.92
    volatility_min_mid: float = 0.05
    volatility_max_mid: float = 0.95
    extreme_tail_mid: float = 0.02
    reward_tail_mid: float = 0.08
    high_liquidity: float = 500.0
    high_volume_24h: float = 1_000.0
    high_volume_total: float = 10_000.0
    liquidity_full_score: float = 1_000_000.0
    volume_full_score: float = 1_000_000.0
    depth_full_score: float = 50_000.0


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _as_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _first_present(candidate: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = candidate.get(key)
        if value not in (None, ""):
            return value
    return None


def _book(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    book = candidate.get("yes_book")
    return book if isinstance(book, Mapping) else {}


def _has_yes_book_prices(candidate: Mapping[str, Any]) -> bool:
    book = _book(candidate)
    return _book_first(book, "best_bid", "bid", "best_bid_yes") not in (None, "") and _book_first(
        book, "best_ask", "ask", "best_ask_yes"
    ) not in (None, "")


def _book_first(book: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = book.get(key)
        if value not in (None, ""):
            return value
    return None


def _mid(candidate: Mapping[str, Any]) -> float | None:
    book = _book(candidate)
    mid = _as_optional_float(_book_first(book, "mid", "yes_mid", "mid_yes"))
    if mid is not None:
        return mid
    return _as_optional_float(
        _first_present(
            candidate,
            "yes_mid_gamma",
            "yes_mid",
            "mid_yes",
            "mid",
            "yes_mid_from_outcome_prices",
        )
    )


def _spread(candidate: Mapping[str, Any]) -> float | None:
    book = _book(candidate)
    spread = _as_optional_float(_book_first(book, "spread", "spread_yes", "yes_spread"))
    if spread is not None:
        return spread
    top_level_spread = _as_optional_float(
        _first_present(candidate, "spread", "spread_yes", "yes_spread")
    )
    if top_level_spread is not None:
        return top_level_spread
    bid = _as_optional_float(
        _book_first(book, "best_bid", "bid", "best_bid_yes") or candidate.get("best_bid_yes")
    )
    ask = _as_optional_float(
        _book_first(book, "best_ask", "ask", "best_ask_yes") or candidate.get("best_ask_yes")
    )
    if bid is not None and ask is not None:
        return max(0.0, ask - bid)
    return None


def _top_book_depth(candidate: Mapping[str, Any]) -> float:
    book = _book(candidate)
    bid_size = _as_float(
        _book_first(book, "bid_size", "best_bid_size", "top_bid_size_yes")
        or candidate.get("top_bid_size_yes")
    )
    ask_size = _as_float(
        _book_first(book, "ask_size", "best_ask_size", "top_ask_size_yes")
        or candidate.get("top_ask_size_yes")
    )
    explicit_depth = _as_float(
        _book_first(book, "depth", "depth_2c", "depth_bid_top10", "depth_ask_top10")
        or candidate.get("depth_1c_bid_yes")
    ) + _as_float(candidate.get("depth_1c_ask_yes"))
    return max(bid_size + ask_size, explicit_depth)


def _is_present(value: Any) -> bool:
    return value not in (None, "", [], {}, False, "false", "False")


def _reward_eligible(candidate: Mapping[str, Any]) -> bool:
    for key in ("reward_eligible_signal", "reward_eligible", "rewards_eligible"):
        if key in candidate:
            return bool(candidate.get(key))
    reward = candidate.get("reward")
    if isinstance(reward, Mapping) and bool(reward.get("reward_eligible_heuristic")):
        return True
    reward_fields = candidate.get("reward_fields")
    if isinstance(reward_fields, Mapping) and any(
        _is_present(value) for value in reward_fields.values()
    ):
        return True
    return any(
        _is_present(candidate.get(key))
        for key in (
            "rewards",
            "rewardsMinSize",
            "rewardsMaxSpread",
            "rewardMinSize",
            "rewardMaxSpread",
            "liquidityRewards",
        )
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _log_component(value: float, full_score_value: float) -> float:
    if value <= 0:
        return 0.0
    return _clamp01(math.log10(value + 1.0) / math.log10(full_score_value + 1.0))


def _tight_component(spread: float | None, rules: PublicScanRankingRules) -> float:
    if spread is None:
        return 0.0
    return _clamp01(1.0 - min(spread, rules.usable_spread) / rules.usable_spread)


def _center_component(mid: float | None) -> float:
    if mid is None:
        return 0.0
    return _clamp01(1.0 - abs(mid - 0.5) / 0.5)


def _tail_component(mid: float | None, threshold: float) -> float:
    if mid is None:
        return 0.0
    tail_distance = min(mid, 1.0 - mid)
    return _clamp01(1.0 - tail_distance / threshold)


def _low_fill_proxy(liquidity: float, volume_24h: float) -> float:
    if liquidity <= 0:
        return 0.0 if volume_24h > 0 else 0.5
    return _clamp01(1.0 - volume_24h / liquidity)


def public_scan_features(
    candidate: Mapping[str, Any], rules: PublicScanRankingRules | None = None
) -> dict[str, Any]:
    rules = rules or PublicScanRankingRules()
    mid = _mid(candidate)
    spread = _spread(candidate)
    liquidity = _as_float(_first_present(candidate, "liquidity", "liquidityNum"))
    volume_24h = _as_float(
        _first_present(candidate, "volume24hr", "volume_24h", "volume24h", "volume24Hr")
    )
    volume = _as_float(_first_present(candidate, "volume", "volume_total", "volumeNum"))
    depth = _top_book_depth(candidate)
    tail_distance = None if mid is None else min(mid, 1.0 - mid)
    return {
        "mid": mid,
        "spread": spread,
        "liquidity": liquidity,
        "volume_24h": volume_24h,
        "volume": volume,
        "top_book_depth": depth,
        "has_yes_book_prices": _has_yes_book_prices(candidate),
        "reward_eligible": _reward_eligible(candidate),
        "tight_spread": spread is not None and spread <= rules.tight_spread,
        "usable_spread": spread is not None and spread <= rules.usable_spread,
        "balanced_microprice": (
            mid is not None and rules.microprice_min_mid <= mid <= rules.microprice_max_mid
        ),
        "balanced_volatility": (
            mid is not None and rules.volatility_min_mid <= mid <= rules.volatility_max_mid
        ),
        "extreme_tail": tail_distance is not None and tail_distance <= rules.extreme_tail_mid,
        "reward_tail": tail_distance is not None and tail_distance <= rules.reward_tail_mid,
        "tail_distance": tail_distance,
        "high_liquidity_or_volume": (
            liquidity >= rules.high_liquidity
            or volume_24h >= rules.high_volume_24h
            or volume >= rules.high_volume_total
        ),
        "components": {
            "tight_spread": _tight_component(spread, rules),
            "centered_price": _center_component(mid),
            "reward_tail": _tail_component(mid, rules.reward_tail_mid),
            "liquidity": _log_component(liquidity, rules.liquidity_full_score),
            "volume_24h": _log_component(volume_24h, rules.volume_full_score),
            "volume": _log_component(volume, rules.volume_full_score),
            "top_book_depth": _log_component(depth, rules.depth_full_score),
            "low_fill_proxy": _low_fill_proxy(liquidity, volume_24h),
        },
    }


def _ranking_record(
    *,
    score: float,
    bucket: str,
    bucket_priority: int,
    eligible_for_handoff: bool,
    filters_passed: Mapping[str, bool],
    downrank_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "score": round(score, 6),
        "bucket": bucket,
        "bucket_priority": bucket_priority,
        "eligible_for_handoff": eligible_for_handoff,
        "filters_passed": dict(filters_passed),
        "downrank_reasons": list(downrank_reasons),
    }


def _microprice_ranking(features: Mapping[str, Any]) -> dict[str, Any]:
    components = features["components"]
    score = 100.0 * (
        0.34 * components["centered_price"]
        + 0.26 * components["tight_spread"]
        + 0.15 * components["liquidity"]
        + 0.15 * components["volume_24h"]
        + 0.10 * components["top_book_depth"]
    )
    downrank_reasons: list[str] = []
    filters = {
        "has_yes_book_prices": bool(features["has_yes_book_prices"]),
        "balanced_price": bool(features["balanced_microprice"]),
        "tight_spread": bool(features["tight_spread"]),
        "high_liquidity_or_volume": bool(features["high_liquidity_or_volume"]),
        "not_extreme_tail": not bool(features["extreme_tail"]),
    }
    if not features["has_yes_book_prices"]:
        downrank_reasons.append("missing_yes_book_prices")
        return _ranking_record(
            score=score * 0.20,
            bucket="missing_book_fail_closed",
            bucket_priority=-1,
            eligible_for_handoff=False,
            filters_passed=filters,
            downrank_reasons=downrank_reasons,
        )
    if features["extreme_tail"]:
        downrank_reasons.append("extreme_tail_price")
        score *= 0.70
    elif not features["balanced_microprice"]:
        downrank_reasons.append("outside_balanced_price_band")
        score *= 0.85
    if not features["tight_spread"]:
        downrank_reasons.append("spread_not_tight")
    if not features["high_liquidity_or_volume"]:
        downrank_reasons.append("low_activity_or_liquidity")

    if (
        filters["balanced_price"]
        and filters["tight_spread"]
        and filters["high_liquidity_or_volume"]
    ):
        return _ranking_record(
            score=score,
            bucket="balanced_tight_order_book",
            bucket_priority=3,
            eligible_for_handoff=True,
            filters_passed=filters,
            downrank_reasons=downrank_reasons,
        )
    if features["extreme_tail"]:
        return _ranking_record(
            score=score,
            bucket="separated_tail_watch",
            bucket_priority=0,
            eligible_for_handoff=False,
            filters_passed=filters,
            downrank_reasons=downrank_reasons,
        )
    return _ranking_record(
        score=score,
        bucket="secondary_microprice_watch",
        bucket_priority=1,
        eligible_for_handoff=False,
        filters_passed=filters,
        downrank_reasons=downrank_reasons,
    )


def _volatility_ranking(features: Mapping[str, Any]) -> dict[str, Any]:
    components = features["components"]
    activity = max(components["volume_24h"], components["volume"])
    score = 100.0 * (
        0.30 * activity
        + 0.22 * components["liquidity"]
        + 0.20 * components["tight_spread"]
        + 0.18 * components["centered_price"]
        + 0.10 * components["top_book_depth"]
    )
    downrank_reasons: list[str] = []
    filters = {
        "has_yes_book_prices": bool(features["has_yes_book_prices"]),
        "usable_spread": bool(features["usable_spread"]),
        "active_liquidity_or_volume": bool(features["high_liquidity_or_volume"]),
        "non_extreme_price": not bool(features["extreme_tail"]),
        "volatility_price_band": bool(features["balanced_volatility"]),
    }
    if not features["has_yes_book_prices"]:
        downrank_reasons.append("missing_yes_book_prices")
        return _ranking_record(
            score=score * 0.20,
            bucket="missing_book_fail_closed",
            bucket_priority=-1,
            eligible_for_handoff=False,
            filters_passed=filters,
            downrank_reasons=downrank_reasons,
        )
    if features["extreme_tail"]:
        downrank_reasons.append("extreme_tail_price")
        score *= 0.75
    elif not features["balanced_volatility"]:
        downrank_reasons.append("outside_volatility_price_band")
        score *= 0.9
    if not features["usable_spread"]:
        downrank_reasons.append("spread_not_usable")
    if not features["high_liquidity_or_volume"]:
        downrank_reasons.append("low_activity_or_liquidity")

    if all(filters.values()):
        return _ranking_record(
            score=score,
            bucket="active_non_tail_deep_limit_watch",
            bucket_priority=3,
            eligible_for_handoff=True,
            filters_passed=filters,
            downrank_reasons=downrank_reasons,
        )
    if features["extreme_tail"]:
        return _ranking_record(
            score=score,
            bucket="separated_tail_watch",
            bucket_priority=0,
            eligible_for_handoff=False,
            filters_passed=filters,
            downrank_reasons=downrank_reasons,
        )
    return _ranking_record(
        score=score,
        bucket="secondary_volatility_watch",
        bucket_priority=1,
        eligible_for_handoff=False,
        filters_passed=filters,
        downrank_reasons=downrank_reasons,
    )


def _reward_ranking(features: Mapping[str, Any]) -> dict[str, Any]:
    components = features["components"]
    reward_component = 1.0 if features["reward_eligible"] else 0.0
    score = 100.0 * (
        0.34 * reward_component
        + 0.26 * components["reward_tail"]
        + 0.14 * components["tight_spread"]
        + 0.12 * components["liquidity"]
        + 0.08 * components["low_fill_proxy"]
        + 0.06 * components["top_book_depth"]
    )
    downrank_reasons: list[str] = []
    filters = {
        "has_yes_book_prices": bool(features["has_yes_book_prices"]),
        "reward_signal_present": bool(features["reward_eligible"]),
        "usable_spread": bool(features["usable_spread"]),
        "extreme_tail_reward_watchlist": bool(features["extreme_tail"]),
        "tail_reward_observation": bool(features["reward_tail"]),
        "high_liquidity_or_volume": bool(features["high_liquidity_or_volume"]),
    }
    if not features["has_yes_book_prices"]:
        downrank_reasons.append("missing_yes_book_prices")
        return _ranking_record(
            score=score * 0.20,
            bucket="missing_book_fail_closed",
            bucket_priority=-1,
            eligible_for_handoff=False,
            filters_passed=filters,
            downrank_reasons=downrank_reasons,
        )
    if not features["reward_eligible"]:
        downrank_reasons.append("missing_public_reward_signal")
        score *= 0.65
    if not features["usable_spread"]:
        downrank_reasons.append("spread_not_usable_for_reward_observation")
    if features["reward_eligible"] and features["extreme_tail"]:
        return _ranking_record(
            score=score,
            bucket="extreme_tail_reward_watchlist",
            bucket_priority=4,
            eligible_for_handoff=True,
            filters_passed=filters,
            downrank_reasons=downrank_reasons,
        )
    if features["reward_eligible"] and features["reward_tail"]:
        return _ranking_record(
            score=score,
            bucket="tail_reward_observation",
            bucket_priority=3,
            eligible_for_handoff=True,
            filters_passed=filters,
            downrank_reasons=downrank_reasons,
        )
    if features["reward_eligible"]:
        return _ranking_record(
            score=score,
            bucket="reward_observation",
            bucket_priority=2,
            eligible_for_handoff=True,
            filters_passed=filters,
            downrank_reasons=downrank_reasons,
        )
    if features["reward_tail"]:
        return _ranking_record(
            score=score,
            bucket="public_tail_proxy_no_reward_terms",
            bucket_priority=1,
            eligible_for_handoff=False,
            filters_passed=filters,
            downrank_reasons=downrank_reasons,
        )
    return _ranking_record(
        score=score,
        bucket="diagnostic_only_no_reward_tail",
        bucket_priority=0,
        eligible_for_handoff=False,
        filters_passed=filters,
        downrank_reasons=downrank_reasons,
    )


def apply_strategy_rankings(
    candidate: Mapping[str, Any], rules: PublicScanRankingRules | None = None
) -> dict[str, Any]:
    rules = rules or PublicScanRankingRules()
    row = dict(candidate)
    features = public_scan_features(candidate, rules)
    rankings = {
        MICROPRICE: _microprice_ranking(features),
        VOLATILITY_SPIKE_DEEP_LIMIT_MAKER: _volatility_ranking(features),
        LOW_FILL_LIQUIDITY_REWARD_MAKER: _reward_ranking(features),
    }
    row["objective_features"] = features
    row["strategy_rankings"] = rankings
    row["objective_buckets"] = {
        strategy: ranking["bucket"] for strategy, ranking in rankings.items()
    }
    row["scores"] = {strategy: rankings[strategy]["score"] for strategy in STRATEGY_KEYS}
    return row


def _strategy_sort_key(candidate: Mapping[str, Any], strategy: str) -> tuple[Any, ...]:
    ranking = candidate["strategy_rankings"][strategy]
    features = candidate["objective_features"]
    return (
        ranking["bucket_priority"],
        ranking["score"],
        features["volume_24h"],
        features["liquidity"],
        str(candidate.get("slug") or candidate.get("market_id") or candidate.get("id") or ""),
    )


def _candidate_key(candidate: Mapping[str, Any]) -> str:
    for key in ("slug", "market_id", "id", "condition_id", "conditionId"):
        value = candidate.get(key)
        if value not in (None, ""):
            return str(value)
    return str(candidate.get("question") or "")


def _strategy_bucket_counts(
    candidates: Sequence[Mapping[str, Any]], strategy: str
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        bucket = str(candidate["strategy_rankings"][strategy]["bucket"])
        counts[bucket] = counts.get(bucket, 0) + 1
    return dict(sorted(counts.items()))


def _top_strategy_overlap_diagnostics(
    top_candidates: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    top_keys = {
        strategy: {_candidate_key(candidate) for candidate in top_candidates.get(strategy, [])}
        for strategy in STRATEGY_KEYS
    }
    top_slugs = {
        strategy: {
            _candidate_key(candidate): str(candidate.get("slug") or _candidate_key(candidate))
            for candidate in top_candidates.get(strategy, [])
        }
        for strategy in STRATEGY_KEYS
    }
    low_fill = top_keys[LOW_FILL_LIQUIDITY_REWARD_MAKER]
    microprice_overlap = sorted(low_fill & top_keys[MICROPRICE])
    volatility_overlap = sorted(low_fill & top_keys[VOLATILITY_SPIKE_DEEP_LIMIT_MAKER])
    return {
        "low_fill_microprice_overlap_count": len(microprice_overlap),
        "low_fill_microprice_overlap_slugs": [
            top_slugs[LOW_FILL_LIQUIDITY_REWARD_MAKER].get(key, key) for key in microprice_overlap
        ],
        "low_fill_volatility_overlap_count": len(volatility_overlap),
        "low_fill_volatility_overlap_slugs": [
            top_slugs[LOW_FILL_LIQUIDITY_REWARD_MAKER].get(key, key) for key in volatility_overlap
        ],
    }


def _strategy_ranking_diagnostics(
    scored: Sequence[Mapping[str, Any]],
    top_candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    rules: PublicScanRankingRules,
) -> dict[str, Any]:
    low_fill_top = top_candidates.get(LOW_FILL_LIQUIDITY_REWARD_MAKER, [])
    low_fill_bucket_counts = _strategy_bucket_counts(low_fill_top, LOW_FILL_LIQUIDITY_REWARD_MAKER)
    return {
        "schema_version": PUBLIC_SCAN_RANKING_SCHEMA_VERSION,
        "shadow_backtest_only": True,
        "safety_fields": dict(PUBLIC_SCAN_SAFETY_FIELDS),
        "candidate_count": len(scored),
        "top_candidate_counts": {
            strategy: len(top_candidates.get(strategy, [])) for strategy in STRATEGY_KEYS
        },
        "all_candidate_bucket_counts": {
            strategy: _strategy_bucket_counts(scored, strategy) for strategy in STRATEGY_KEYS
        },
        "top_candidate_bucket_counts": {
            strategy: _strategy_bucket_counts(top_candidates.get(strategy, []), strategy)
            for strategy in STRATEGY_KEYS
        },
        "low_fill_reward_tail": {
            "extreme_tail_threshold": rules.extreme_tail_mid,
            "reward_tail_threshold": rules.reward_tail_mid,
            "extreme_tail_watchlist_count": low_fill_bucket_counts.get(
                "extreme_tail_reward_watchlist", 0
            ),
            "tail_observation_count": low_fill_bucket_counts.get("tail_reward_observation", 0),
            "missing_book_fail_closed_count": low_fill_bucket_counts.get(
                "missing_book_fail_closed", 0
            ),
        },
        "overlap_diagnostics": _top_strategy_overlap_diagnostics(top_candidates),
    }


def build_strategy_ranked_scan(
    candidates: Sequence[Mapping[str, Any]],
    *,
    limit: int = 15,
    rules: PublicScanRankingRules | None = None,
) -> dict[str, Any]:
    rules = rules or PublicScanRankingRules()
    scored = [apply_strategy_rankings(candidate, rules) for candidate in candidates]
    top_candidates: dict[str, list[dict[str, Any]]] = {}
    for strategy in STRATEGY_KEYS:
        ordered = sorted(scored, key=lambda row: _strategy_sort_key(row, strategy), reverse=True)
        for rank, row in enumerate(ordered, start=1):
            row["strategy_rankings"][strategy]["rank"] = rank
        top_candidates[strategy] = ordered[: max(0, limit)]

    return {
        "objective_specific_ranking": strategy_ranking_metadata(rules),
        "ranking_diagnostics": _strategy_ranking_diagnostics(scored, top_candidates, rules),
        "top_candidates": top_candidates,
        "all_scored_candidates": scored,
    }


def strategy_ranking_metadata(
    rules: PublicScanRankingRules | None = None,
) -> dict[str, Any]:
    rules = rules or PublicScanRankingRules()
    return {
        "schema_version": PUBLIC_SCAN_RANKING_SCHEMA_VERSION,
        "shadow_backtest_only": True,
        "safety_fields": dict(PUBLIC_SCAN_SAFETY_FIELDS),
        "rules": asdict(rules),
        "sort_order": "bucket_priority desc, objective_score desc, volume_24h desc, liquidity desc",
        "strategies": {
            MICROPRICE: {
                "objective": "Balanced tight-spread L2 markets for Microprice/order-book imbalance backtests.",
                "primary_bucket": "balanced_tight_order_book",
                "filters": {
                    "balanced_price": f"{rules.microprice_min_mid:.2f} <= yes_mid <= {rules.microprice_max_mid:.2f}",
                    "tight_spread": f"yes spread <= {rules.tight_spread:.2f}",
                    "high_liquidity_or_volume": (
                        f"liquidity >= {rules.high_liquidity:.0f} OR "
                        f"volume_24h >= {rules.high_volume_24h:.0f} OR "
                        f"volume >= {rules.high_volume_total:.0f}"
                    ),
                    "tail_policy": "extreme tails are retained in separated_tail_watch, not primary handoff",
                },
            },
            VOLATILITY_SPIKE_DEEP_LIMIT_MAKER: {
                "objective": "Active non-tail markets with usable spread/depth for volatility-spike deep limit-maker backtests.",
                "primary_bucket": "active_non_tail_deep_limit_watch",
                "filters": {
                    "volatility_price_band": f"{rules.volatility_min_mid:.2f} <= yes_mid <= {rules.volatility_max_mid:.2f}",
                    "usable_spread": f"yes spread <= {rules.usable_spread:.2f}",
                    "activity": "volume/liquidity components dominate after price-band eligibility",
                    "tail_policy": "extreme tails are separated from the primary volatility handoff",
                },
            },
            LOW_FILL_LIQUIDITY_REWARD_MAKER: {
                "objective": "Reward-signal markets, especially extreme tails, for low-fill liquidity reward observation only.",
                "primary_bucket": "extreme_tail_reward_watchlist",
                "filters": {
                    "reward_signal_present": "Gamma/public reward fields present or scan reward flag true",
                    "extreme_tail_reward_watchlist": f"min(yes_mid, 1 - yes_mid) <= {rules.extreme_tail_mid:.2f}",
                    "tail_reward_observation": f"min(yes_mid, 1 - yes_mid) <= {rules.reward_tail_mid:.2f}",
                    "usable_spread": f"yes spread <= {rules.usable_spread:.2f}",
                    "tail_policy": "extreme tails are reported as a separate reward watchlist, not as balanced Microprice/volatility candidates",
                    "missing_book_policy": "missing yes best bid/ask rows fail closed for all strategy handoffs",
                },
            },
        },
    }
