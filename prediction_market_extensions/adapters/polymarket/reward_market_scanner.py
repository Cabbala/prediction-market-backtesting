from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from prediction_market_extensions.adapters.polymarket.public_scan_ranking import (
    LOW_FILL_LIQUIDITY_REWARD_MAKER,
    MICROPRICE,
    VOLATILITY_SPIKE_DEEP_LIMIT_MAKER,
)

SHADOW_MODE = "SHADOW_BACKTEST_ONLY_NO_LIVE_TRADING"
MANIFEST_SCHEMA_VERSION = "polymarket.reward-market-manifest.v1"
CANONICAL_MANIFEST_PREFIX = "reward_scanner_manifest"
LEGACY_MANIFEST_PREFIX = "reward_market_manifest"
LOW_FILL_EXTREME_TAIL_THRESHOLD = 0.02
LOW_FILL_REWARD_TAIL_THRESHOLD = 0.08
REWARD_EVIDENCE_KEYS = (
    "clobRewards",
    "rewards",
    "rewardsMinSize",
    "rewardsMaxSpread",
    "umaReward",
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _as_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)


def _first_present(candidate: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in candidate and candidate[key] not in (None, ""):
            return candidate[key]
    return None


def _first_book_value(book: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = book.get(key)
        if value not in (None, ""):
            return value
    return None


def _is_present(value: Any) -> bool:
    return value not in (None, "", [], {}, False)


def _book_from_scan_books(candidate: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    books = candidate.get("books")
    if not isinstance(books, Sequence) or isinstance(books, (str, bytes)):
        return {}
    idx = 0 if side == "yes" else 1
    if len(books) <= idx or not isinstance(books[idx], Mapping):
        return {}
    raw = books[idx]
    return {
        "ok": True,
        "best_bid": _first_book_value(raw, "best_bid", "bid"),
        "best_ask": _first_book_value(raw, "best_ask", "ask"),
        "best_bid_size": _first_book_value(raw, "best_bid_size", "bid_size"),
        "best_ask_size": _first_book_value(raw, "best_ask_size", "ask_size"),
        "depth_2c": _first_book_value(raw, "depth_2c", "depth"),
        "depth_bid": _first_book_value(
            raw, "depth_bid", "depth_bid_2c", "depth_bid_5c", "depth_bid_top10"
        ),
        "depth_ask": _first_book_value(
            raw, "depth_ask", "depth_ask_2c", "depth_ask_5c", "depth_ask_top10"
        ),
        "depth_bid_top10": raw.get("depth_bid_top10"),
        "depth_ask_top10": raw.get("depth_ask_top10"),
        "mid": raw.get("mid"),
        "spread": raw.get("spread"),
        "token_id": raw.get("token_id"),
    }


def _book(candidate: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    value = candidate.get(f"{side}_book")
    if isinstance(value, Mapping):
        return {
            **value,
            "best_bid": _first_book_value(value, "best_bid", "bid"),
            "best_ask": _first_book_value(value, "best_ask", "ask"),
            "best_bid_size": _first_book_value(value, "best_bid_size", "bid_size"),
            "best_ask_size": _first_book_value(value, "best_ask_size", "ask_size"),
            "depth_bid": _first_book_value(
                value, "depth_bid", "depth_bid_2c", "depth_bid_5c", "depth_bid_top10"
            ),
            "depth_ask": _first_book_value(
                value, "depth_ask", "depth_ask_2c", "depth_ask_5c", "depth_ask_top10"
            ),
        }
    from_books = _book_from_scan_books(candidate, side)
    if from_books:
        return from_books

    # Some autonomous scan artifacts persist only flat top-of-book fields.
    # Keep this read-only/fail-closed: use fields only when they are explicit
    # and side-scoped; never infer missing prices from the opposite token.
    prefix = f"{side}_"
    flat_keys = (
        "token_id",
        "best_bid",
        "best_ask",
        "bid",
        "ask",
        "best_bid_size",
        "best_ask_size",
        "bid_size",
        "ask_size",
        "depth_2c",
        "depth_bid",
        "depth_ask",
        "depth_bid_2c",
        "depth_ask_2c",
        "depth_bid_5c",
        "depth_ask_5c",
        "depth_bid_top10",
        "depth_ask_top10",
        "mid",
        "spread",
    )
    flat = {
        key: candidate.get(prefix + key)
        for key in flat_keys
        if candidate.get(prefix + key) not in (None, "")
    }
    if flat:
        if "mid" not in flat and not isinstance(value, bool):
            compact_mid = _as_float(value, default=-1.0)
            if compact_mid >= 0:
                flat["mid"] = compact_mid
        return flat
    if not isinstance(value, bool):
        compact_mid = _as_float(value, default=-1.0)
        if compact_mid >= 0:
            return {"mid": compact_mid}
    return flat


def _reward_evidence(candidate: Mapping[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}

    def add_allowlisted(source: Mapping[str, Any]) -> None:
        for key in REWARD_EVIDENCE_KEYS:
            value = source.get(key)
            if _is_present(value):
                evidence[key] = value

    raw = candidate.get("reward_evidence")
    if isinstance(raw, Mapping):
        add_allowlisted(raw)
        nested = raw.get("reward_evidence")
        if isinstance(nested, Mapping):
            add_allowlisted(nested)
    add_allowlisted(candidate)

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


def _has_book_prices(candidate: Mapping[str, Any], side: str) -> bool:
    book = _book(candidate, side)
    return book.get("best_bid") not in (None, "") and book.get("best_ask") not in (None, "")


def _spread(candidate: Mapping[str, Any], side: str) -> float:
    book = _book(candidate, side)
    spread = _as_float(book.get("spread"), default=-1.0)
    if spread >= 0:
        return spread
    bid = _as_float(_first_book_value(book, "best_bid", "bid"), default=0.0)
    ask = _as_float(_first_book_value(book, "best_ask", "ask"), default=0.0)
    return max(0.0, ask - bid) if ask and bid else 0.0


def _depth(candidate: Mapping[str, Any], side: str) -> float:
    book = _book(candidate, side)
    top_depth = _as_float(_first_book_value(book, "best_bid_size", "bid_size")) + _as_float(
        _first_book_value(book, "best_ask_size", "ask_size")
    )
    bid_depth = _as_float(
        _first_book_value(book, "depth_bid", "depth_bid_2c", "depth_bid_5c", "depth_bid_top10")
    )
    ask_depth = _as_float(
        _first_book_value(book, "depth_ask", "depth_ask_2c", "depth_ask_5c", "depth_ask_top10")
    )
    paired_depth = bid_depth + ask_depth
    return max(top_depth, paired_depth, _as_float(book.get("depth_2c")))


def _mid(candidate: Mapping[str, Any], side: str) -> float | None:
    book = _book(candidate, side)
    mid = _as_float(book.get("mid"), default=-1.0)
    if mid >= 0:
        return mid
    prices = candidate.get("outcomePrices") or candidate.get("outcome_prices")
    idx = 0 if side == "yes" else 1
    if isinstance(prices, Sequence) and not isinstance(prices, (str, bytes)) and len(prices) > idx:
        parsed = _as_float(prices[idx], default=-1.0)
        return parsed if parsed >= 0 else None
    return None


def _has_explicit_reward_evidence(candidate: Mapping[str, Any]) -> bool:
    if _reward_evidence(candidate):
        return True
    if candidate.get("reward_hint"):
        return True
    fits = candidate.get("strategy_fits") or []
    return (
        isinstance(fits, Sequence)
        and not isinstance(fits, (str, bytes))
        and "reward_eligible_candidate" in fits
    )


def _reward_category(candidate: Mapping[str, Any]) -> str:
    evidence = _reward_evidence(candidate)
    if _is_present(evidence.get("clobRewards")) or _is_present(evidence.get("rewards")):
        return "explicit_clob_rewards"
    if _is_present(evidence.get("rewardsMinSize")) or _is_present(evidence.get("rewardsMaxSpread")):
        return "explicit_gamma_reward_terms"
    if _is_present(evidence.get("umaReward")):
        return "explicit_uma_reward_hint"
    if candidate.get("reward_hint"):
        return "explicit_reward_hint"
    return "public_proxy_only_reward_unverified"


def _side_scoped_token_ids(candidate: Mapping[str, Any]) -> list[str]:
    yes_book = candidate.get("yes_book")
    no_book = candidate.get("no_book")
    yes_token = candidate.get("yes_token_id") or (
        yes_book.get("token_id") if isinstance(yes_book, Mapping) else None
    )
    no_token = candidate.get("no_token_id") or (
        no_book.get("token_id") if isinstance(no_book, Mapping) else None
    )
    if (
        yes_token not in (None, "")
        and no_token not in (None, "")
        and str(yes_token) != str(no_token)
    ):
        return [str(yes_token), str(no_token)]
    return []


def _outcomes(candidate: Mapping[str, Any]) -> list[str]:
    raw = candidate.get("outcomes")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return []
        raw = parsed
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(outcome).strip() for outcome in raw if str(outcome).strip()]

    # Current autonomous scan artifacts may omit the raw Gamma ``outcomes``
    # field while persisting explicit side-scoped token ids. Treat the field
    # names themselves as the Yes/No mapping, but only when both unique token
    # ids are present. This preserves fail-closed behavior for ambiguous
    # clobTokenIds-only artifacts and does not infer prices from the opposite
    # side.
    if _side_scoped_token_ids(candidate):
        return ["Yes", "No"]
    return []


def _has_yes_no_outcomes(candidate: Mapping[str, Any]) -> bool:
    outcomes = _outcomes(candidate)
    normalized = [outcome.casefold() for outcome in outcomes]
    return len(outcomes) == 2 and set(normalized) == {"yes", "no"}


def _candidate_token_ids(candidate: Mapping[str, Any]) -> list[str]:
    raw = candidate.get("clob_token_ids") or candidate.get("clobTokenIds")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return []
        raw = parsed
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(token_id) for token_id in raw if token_id not in (None, "")]
    books = candidate.get("books")
    if isinstance(books, Sequence) and not isinstance(books, (str, bytes)):
        book_ids = [
            str(book.get("token_id"))
            for book in books
            if isinstance(book, Mapping) and book.get("token_id") not in (None, "")
        ]
        if book_ids:
            return book_ids
        if len(books) > 0:
            return []
    return _side_scoped_token_ids(candidate)


def _clob_token_ids(candidate: Mapping[str, Any]) -> list[str]:
    ids = _candidate_token_ids(candidate)
    outcomes = _outcomes(candidate)
    if len(ids) != 2 or len(set(ids)) != 2 or not _has_yes_no_outcomes(candidate):
        return []
    raw = candidate.get("clob_token_ids") or candidate.get("clobTokenIds")
    side_scoped_ids = _side_scoped_token_ids(candidate)
    if raw in (None, "", []) and ids == side_scoped_ids:
        return side_scoped_ids
    by_outcome = {
        outcome.casefold(): token_id for outcome, token_id in zip(outcomes, ids, strict=True)
    }
    return [by_outcome["yes"], by_outcome["no"]]


def _strategy_fits(candidate: Mapping[str, Any]) -> set[str]:
    value = candidate.get("strategy_fits")
    if isinstance(value, str):
        return {value}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {str(v) for v in value}
    fit_map = candidate.get("strategy_fit")
    if isinstance(fit_map, str):
        return {fit_map}
    if isinstance(fit_map, Mapping):
        return {str(k) for k, enabled in fit_map.items() if enabled}
    source_tags = candidate.get("source_tags")
    if isinstance(source_tags, str):
        return {source_tags}
    if isinstance(source_tags, Sequence) and not isinstance(source_tags, (str, bytes)):
        return {str(v) for v in source_tags}
    return set()


def _source_row_key(row: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("market_id", "id", "condition_id", "conditionId", "slug"):
        value = row.get(key)
        if value not in (None, ""):
            parts.append(f"{key}:{value}")
    if parts:
        return "|".join(parts)
    question = row.get("question")
    if question not in (None, ""):
        return f"question:{question}"
    return json.dumps(row, sort_keys=True, default=str)


def _type_name(value: Any) -> str | None:
    return type(value).__name__ if value is not None else None


def _normalized_strategy_label(value: Any) -> str:
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())


def _canonical_strategy_bucket(value: Any) -> str:
    aliases = {
        "microprice": MICROPRICE,
        "volatilityspikedeeplimitmaker": VOLATILITY_SPIKE_DEEP_LIMIT_MAKER,
        "volatilityspike": VOLATILITY_SPIKE_DEEP_LIMIT_MAKER,
        "deeplimitmaker": VOLATILITY_SPIKE_DEEP_LIMIT_MAKER,
        "lowfillrewardmaker": LOW_FILL_LIQUIDITY_REWARD_MAKER,
        "lowfillliquidityrewardmaker": LOW_FILL_LIQUIDITY_REWARD_MAKER,
        "lowfillprobabilityliquidityrewardmaker": LOW_FILL_LIQUIDITY_REWARD_MAKER,
        "lowfillliquidityreward": LOW_FILL_LIQUIDITY_REWARD_MAKER,
    }
    return aliases.get(_normalized_strategy_label(value), str(value))


def _canonical_strategy_buckets(values: Sequence[Any]) -> list[str]:
    buckets: list[str] = []
    for value in values:
        canonical = _canonical_strategy_bucket(value)
        if canonical and canonical not in buckets:
            buckets.append(canonical)
    return buckets


def _tail_distance(candidate: Mapping[str, Any]) -> float | None:
    mid = _mid(candidate, "yes")
    if mid is None:
        return None
    return min(mid, 1.0 - mid)


def _low_fill_reward_tail_bucket(
    candidate: Mapping[str, Any],
    canonical_strategy_buckets: Sequence[str],
    *,
    extreme_tail_threshold: float = LOW_FILL_EXTREME_TAIL_THRESHOLD,
    reward_tail_threshold: float = LOW_FILL_REWARD_TAIL_THRESHOLD,
) -> str:
    if LOW_FILL_LIQUIDITY_REWARD_MAKER not in canonical_strategy_buckets:
        return "not_low_fill_reward_source"
    tail_distance = _tail_distance(candidate)
    if tail_distance is None:
        return "low_fill_missing_yes_mid_fail_closed"
    if tail_distance <= extreme_tail_threshold:
        return "low_fill_extreme_tail_watchlist"
    if tail_distance <= reward_tail_threshold:
        return "low_fill_tail_observation"
    return "low_fill_non_tail_reward_observation"


def _source_strategy_overlap(
    canonical_strategy_buckets: Sequence[str],
) -> dict[str, bool]:
    bucket_set = set(canonical_strategy_buckets)
    low_fill = LOW_FILL_LIQUIDITY_REWARD_MAKER in bucket_set
    return {
        "overlaps_low_fill_and_microprice": low_fill and MICROPRICE in bucket_set,
        "overlaps_low_fill_and_volatility": low_fill
        and VOLATILITY_SPIKE_DEEP_LIMIT_MAKER in bucket_set,
    }


def _strategy_overlap_diagnostics(
    rows: Sequence[tuple[Mapping[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    by_strategy: dict[str, dict[str, str]] = {
        MICROPRICE: {},
        VOLATILITY_SPIKE_DEEP_LIMIT_MAKER: {},
        LOW_FILL_LIQUIDITY_REWARD_MAKER: {},
    }
    for row, provenance in rows:
        key = str(provenance["deduplication_key"])
        slug = str(row.get("slug") or row.get("market_id") or row.get("id") or key)
        for bucket in provenance.get("canonical_strategy_buckets", []):
            if bucket in by_strategy:
                by_strategy[bucket][key] = slug

    low_fill_keys = set(by_strategy[LOW_FILL_LIQUIDITY_REWARD_MAKER])
    microprice_overlap = sorted(low_fill_keys & set(by_strategy[MICROPRICE]))
    volatility_overlap = sorted(low_fill_keys & set(by_strategy[VOLATILITY_SPIKE_DEEP_LIMIT_MAKER]))
    return {
        "low_fill_microprice_overlap_count": len(microprice_overlap),
        "low_fill_microprice_overlap_slugs": [
            by_strategy[LOW_FILL_LIQUIDITY_REWARD_MAKER][key] for key in microprice_overlap
        ],
        "low_fill_volatility_overlap_count": len(volatility_overlap),
        "low_fill_volatility_overlap_slugs": [
            by_strategy[LOW_FILL_LIQUIDITY_REWARD_MAKER][key] for key in volatility_overlap
        ],
    }


def _count_canonical_strategy_buckets(
    rows: Sequence[tuple[Mapping[str, Any], dict[str, Any]]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, provenance in rows:
        for bucket in provenance.get("canonical_strategy_buckets", []):
            counts[bucket] = counts.get(bucket, 0) + 1
    return dict(sorted(counts.items()))


def _source_low_fill_tail_diagnostics(
    rows: Sequence[tuple[Mapping[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    bucket_counts: dict[str, int] = {}
    missing_book_count = 0
    for row, provenance in rows:
        canonical_buckets = provenance.get("canonical_strategy_buckets", [])
        if LOW_FILL_LIQUIDITY_REWARD_MAKER not in canonical_buckets:
            continue
        bucket = _low_fill_reward_tail_bucket(row, canonical_buckets)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if not (_has_book_prices(row, "yes") and _has_book_prices(row, "no")):
            missing_book_count += 1
    return {
        "thresholds": {
            "extreme_tail": LOW_FILL_EXTREME_TAIL_THRESHOLD,
            "reward_tail": LOW_FILL_REWARD_TAIL_THRESHOLD,
        },
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "low_fill_extreme_tail_watchlist_count": bucket_counts.get(
            "low_fill_extreme_tail_watchlist", 0
        ),
        "low_fill_missing_book_source_count": missing_book_count,
    }


def _source_exclusion(
    *,
    source_key: str,
    reason: str,
    source_shape: str,
    strategy_bucket: str | None = None,
    value: Any = None,
) -> dict[str, Any]:
    exclusion: dict[str, Any] = {
        "source_key": source_key,
        "reason": reason,
        "source_shape": source_shape,
    }
    if strategy_bucket is not None:
        exclusion["strategy_bucket"] = strategy_bucket
    if value is not None:
        exclusion["value_type"] = type(value).__name__
    return exclusion


def _scan_rows_from_sequence(
    value: Any,
    *,
    source_key: str,
    source_shape: str,
    strategy_bucket: str | None = None,
) -> tuple[list[tuple[Mapping[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return [], [
            _source_exclusion(
                source_key=source_key,
                reason="candidate_container_not_list",
                source_shape=source_shape,
                strategy_bucket=strategy_bucket,
                value=value,
            )
        ]

    rows: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    exclusions: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        row_source_key = f"{source_key}[{index}]"
        if not isinstance(row, Mapping):
            exclusions.append(
                _source_exclusion(
                    source_key=row_source_key,
                    reason="non_object_candidate_row",
                    source_shape=source_shape,
                    strategy_bucket=strategy_bucket,
                    value=row,
                )
            )
            continue
        row_key = _source_row_key(row)
        rows.append(
            (
                row,
                {
                    "source_key": row_source_key,
                    "source_shape": source_shape,
                    "strategy_buckets": [strategy_bucket] if strategy_bucket else [],
                    "canonical_strategy_buckets": (
                        _canonical_strategy_buckets([strategy_bucket]) if strategy_bucket else []
                    ),
                    "deduplication_key": row_key,
                    "duplicate_source_keys": [],
                },
            )
        )
    return rows, exclusions


def _deduplicate_source_rows(
    rows: Sequence[tuple[Mapping[str, Any], dict[str, Any]]],
) -> tuple[list[tuple[Mapping[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    by_key: dict[str, tuple[Mapping[str, Any], dict[str, Any]]] = {}
    duplicate_rows: list[dict[str, Any]] = []
    for row, provenance in rows:
        row_key = str(provenance["deduplication_key"])
        existing = by_key.get(row_key)
        if existing is None:
            stored_provenance = dict(provenance)
            stored_provenance["strategy_buckets"] = list(provenance.get("strategy_buckets", []))
            stored_provenance["canonical_strategy_buckets"] = list(
                provenance.get("canonical_strategy_buckets", [])
            )
            stored_provenance["duplicate_source_keys"] = list(
                provenance.get("duplicate_source_keys", [])
            )
            by_key[row_key] = (row, stored_provenance)
            continue
        _, existing_provenance = existing
        existing_buckets = existing_provenance.setdefault("strategy_buckets", [])
        for bucket in provenance.get("strategy_buckets", []):
            if bucket not in existing_buckets:
                existing_buckets.append(bucket)
        existing_canonical_buckets = existing_provenance.setdefault(
            "canonical_strategy_buckets", []
        )
        for bucket in provenance.get("canonical_strategy_buckets", []):
            if bucket not in existing_canonical_buckets:
                existing_canonical_buckets.append(bucket)
        duplicate_source_keys = existing_provenance.setdefault("duplicate_source_keys", [])
        duplicate_source_keys.append(provenance["source_key"])
        duplicate_rows.append(
            {
                "source_key": provenance["source_key"],
                "deduplicated_into": existing_provenance["source_key"],
                "deduplication_key": row_key,
                "strategy_buckets": provenance.get("strategy_buckets", []),
                "canonical_strategy_buckets": provenance.get("canonical_strategy_buckets", []),
                "reason": "duplicate_candidate_row",
            }
        )
    return list(by_key.values()), duplicate_rows


def _count_reasons(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _source_state_classification(diagnostics: Mapping[str, Any]) -> str:
    if diagnostics.get("normalized_candidate_count"):
        selected_source = diagnostics.get("selected_source")
        if selected_source == "top_candidates_strategy_buckets":
            return "strategy_keyed_top_candidates_normalized"
        return "list_shaped_candidates_normalized"
    if diagnostics.get("source_candidate_container_count") or diagnostics.get(
        "strategy_bucket_candidate_count"
    ):
        return "source_rows_failed_closed_before_scoring"
    if diagnostics.get("excluded_source_row_count"):
        return "source_schema_or_provenance_failed_closed"
    return "source_empty_zero_candidates"


def _scan_candidates_with_diagnostics(
    scan: Mapping[str, Any],
) -> tuple[list[tuple[Mapping[str, Any], dict[str, Any]]], dict[str, Any]]:
    top_candidates = scan.get("top_candidates")
    candidates = scan.get("candidates")
    diagnostics: dict[str, Any] = {
        "top_candidates_type": _type_name(top_candidates),
        "candidates_type": _type_name(candidates),
        "selected_source": None,
        "strategy_bucket_counts": {},
        "canonical_strategy_bucket_counts": {},
        "canonical_strategy_bucket_unique_counts": {},
        "strategy_bucket_aliases": {},
        "strategy_overlap_diagnostics": {},
        "low_fill_tail_diagnostics": {},
        "strategy_bucket_candidate_count": 0,
        "strategy_bucket_unique_candidate_count": 0,
        "source_candidate_container_count": 0,
        "normalized_candidate_count": 0,
        "duplicate_source_row_count": 0,
        "excluded_source_row_count": 0,
        "exclusion_reason_counts": {},
        "duplicate_source_rows": [],
        "source_candidate_exclusions": [],
    }

    raw_rows: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    exclusions: list[dict[str, Any]] = []
    selected_source: str | None = None
    if isinstance(top_candidates, Sequence) and not isinstance(
        top_candidates, (str, bytes, bytearray)
    ):
        selected_source = "top_candidates_list"
        raw_rows, exclusions = _scan_rows_from_sequence(
            top_candidates,
            source_key="top_candidates",
            source_shape="list_top_candidates",
        )
        diagnostics["source_candidate_container_count"] = len(top_candidates)
    elif isinstance(top_candidates, Mapping):
        selected_source = "top_candidates_strategy_buckets"
        for bucket, bucket_rows in top_candidates.items():
            bucket_name = str(bucket)
            if isinstance(bucket_rows, Sequence) and not isinstance(
                bucket_rows, (str, bytes, bytearray)
            ):
                diagnostics["strategy_bucket_counts"][bucket_name] = sum(
                    1 for row in bucket_rows if isinstance(row, Mapping)
                )
                diagnostics["strategy_bucket_candidate_count"] += len(bucket_rows)
            bucket_candidates, bucket_exclusions = _scan_rows_from_sequence(
                bucket_rows,
                source_key=f"top_candidates.{bucket_name}",
                source_shape="strategy_keyed_top_candidates",
                strategy_bucket=bucket_name,
            )
            raw_rows.extend(bucket_candidates)
            exclusions.extend(bucket_exclusions)
    elif isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes, bytearray)):
        selected_source = "candidates_list"
        raw_rows, exclusions = _scan_rows_from_sequence(
            candidates,
            source_key="candidates",
            source_shape="list_candidates",
        )
        diagnostics["source_candidate_container_count"] = len(candidates)
    elif top_candidates is not None:
        exclusions.append(
            _source_exclusion(
                source_key="top_candidates",
                reason="candidate_container_not_list_or_strategy_mapping",
                source_shape="unsupported_top_candidates",
                value=top_candidates,
            )
        )
    elif candidates is not None:
        exclusions.append(
            _source_exclusion(
                source_key="candidates",
                reason="candidate_container_not_list",
                source_shape="unsupported_candidates",
                value=candidates,
            )
        )

    deduplicated_rows, duplicate_rows = _deduplicate_source_rows(raw_rows)
    diagnostics["selected_source"] = selected_source
    diagnostics["strategy_bucket_counts"] = dict(
        sorted(diagnostics["strategy_bucket_counts"].items())
    )
    diagnostics["strategy_bucket_unique_candidate_count"] = len(
        {str(provenance["deduplication_key"]) for _, provenance in raw_rows}
    )
    diagnostics["canonical_strategy_bucket_counts"] = _count_canonical_strategy_buckets(raw_rows)
    diagnostics["canonical_strategy_bucket_unique_counts"] = _count_canonical_strategy_buckets(
        deduplicated_rows
    )
    diagnostics["strategy_bucket_aliases"] = dict(
        sorted(
            {
                raw_bucket: _canonical_strategy_bucket(raw_bucket)
                for _, provenance in raw_rows
                for raw_bucket in provenance.get("strategy_buckets", [])
            }.items()
        )
    )
    diagnostics["strategy_overlap_diagnostics"] = _strategy_overlap_diagnostics(deduplicated_rows)
    diagnostics["low_fill_tail_diagnostics"] = _source_low_fill_tail_diagnostics(deduplicated_rows)
    diagnostics["normalized_candidate_count"] = len(deduplicated_rows)
    diagnostics["duplicate_source_row_count"] = len(duplicate_rows)
    diagnostics["excluded_source_row_count"] = len(exclusions)
    diagnostics["duplicate_source_rows"] = duplicate_rows
    diagnostics["source_candidate_exclusions"] = exclusions
    diagnostics["exclusion_reason_counts"] = _count_reasons(exclusions)
    diagnostics["source_state_classification"] = _source_state_classification(diagnostics)
    return deduplicated_rows, diagnostics


def _book_token_ids_match(candidate: Mapping[str, Any], clob_token_ids: Sequence[str]) -> bool:
    if len(clob_token_ids) != 2:
        return False
    yes_token = _book(candidate, "yes").get("token_id") or candidate.get("yes_token_id")
    no_token = _book(candidate, "no").get("token_id") or candidate.get("no_token_id")
    if yes_token in (None, "") and no_token in (None, ""):
        return True
    return [str(yes_token), str(no_token)] == [str(clob_token_ids[0]), str(clob_token_ids[1])]


def _canonical_book_side(
    candidate: Mapping[str, Any], side: str, token_id: str | None
) -> dict[str, Any]:
    book = _book(candidate, side)
    best_bid = _as_optional_float(_first_book_value(book, "best_bid", "bid"))
    best_ask = _as_optional_float(_first_book_value(book, "best_ask", "ask"))
    best_bid_size = _as_optional_float(_first_book_value(book, "best_bid_size", "bid_size"))
    best_ask_size = _as_optional_float(_first_book_value(book, "best_ask_size", "ask_size"))
    depth_bid = _as_optional_float(
        _first_book_value(book, "depth_bid", "depth_bid_2c", "depth_bid_5c", "depth_bid_top10")
    )
    depth_ask = _as_optional_float(
        _first_book_value(book, "depth_ask", "depth_ask_2c", "depth_ask_5c", "depth_ask_top10")
    )
    depth_2c = _as_optional_float(book.get("depth_2c"))
    top_depth = (best_bid_size or 0.0) + (best_ask_size or 0.0)
    paired_depth = (depth_bid or 0.0) + (depth_ask or 0.0)
    depth_proxy = max(top_depth, paired_depth, depth_2c or 0.0)
    mid = _as_optional_float(book.get("mid"))
    if mid is None and best_bid is not None and best_ask is not None:
        mid = (best_bid + best_ask) / 2.0
    spread = _as_optional_float(book.get("spread"))
    if spread is None and best_bid is not None and best_ask is not None:
        spread = max(0.0, best_ask - best_bid)
    resolved_token = (
        str(token_id)
        if token_id not in (None, "")
        else str(book.get("token_id") or candidate.get(f"{side}_token_id") or "") or None
    )
    return {
        "side": side,
        "token_id": resolved_token,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "best_bid_size": best_bid_size,
        "best_ask_size": best_ask_size,
        "depth_bid": depth_bid,
        "depth_ask": depth_ask,
        "depth_2c": depth_2c,
        "depth_proxy": depth_proxy if depth_proxy > 0 else None,
        "mid": mid,
        "spread": spread,
        "present": bool(book),
    }


def _book_provenance(
    candidate: Mapping[str, Any],
    clob_token_ids: Sequence[str],
    *,
    source_artifact_path: str | None,
    source_timestamp_utc: Any,
) -> dict[str, Any]:
    yes_token = str(clob_token_ids[0]) if len(clob_token_ids) == 2 else None
    no_token = str(clob_token_ids[1]) if len(clob_token_ids) == 2 else None
    sides = {
        "yes": _canonical_book_side(candidate, "yes", yes_token),
        "no": _canonical_book_side(candidate, "no", no_token),
    }
    fail_closed_reasons: list[str] = []
    for side, side_book in sides.items():
        for field in ("token_id", "best_bid", "best_ask"):
            if side_book.get(field) in (None, ""):
                fail_closed_reasons.append(f"missing_{side}_{field}")
    if (
        candidate.get("complete_books") is False
        or candidate.get("complete_yes_no_clob_books") is False
    ):
        fail_closed_reasons.append("source_marked_incomplete_yes_no_clob_books")
    if len(clob_token_ids) != 2:
        fail_closed_reasons.append("missing_canonical_yes_no_clob_token_ids")
    elif not _book_token_ids_match(candidate, clob_token_ids):
        fail_closed_reasons.append("book_token_ids_do_not_match_yes_no_mapping")
    fail_closed_reasons = sorted(set(fail_closed_reasons))
    complete = not fail_closed_reasons
    return {
        "status": "complete" if complete else "incomplete_fail_closed",
        "complete": complete,
        "source_artifact_path": source_artifact_path,
        "source_timestamp_utc": source_timestamp_utc,
        "fail_closed_reasons": fail_closed_reasons,
        "sides": sides,
    }


def _token_provenance(
    candidate: Mapping[str, Any],
    clob_token_ids: Sequence[str],
    book_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    sides = book_provenance.get("sides")
    side_books = sides if isinstance(sides, Mapping) else {}
    yes_book = side_books.get("yes") if isinstance(side_books.get("yes"), Mapping) else {}
    no_book = side_books.get("no") if isinstance(side_books.get("no"), Mapping) else {}
    canonical_complete = len(clob_token_ids) == 2
    fail_closed_reasons = list(book_provenance.get("fail_closed_reasons") or [])
    if (
        not canonical_complete
        and "missing_canonical_yes_no_clob_token_ids" not in fail_closed_reasons
    ):
        fail_closed_reasons.append("missing_canonical_yes_no_clob_token_ids")
    return {
        "status": "complete" if canonical_complete else "incomplete_fail_closed",
        "canonical_complete": canonical_complete,
        "canonical_yes_token_id": clob_token_ids[0] if canonical_complete else None,
        "canonical_no_token_id": clob_token_ids[1] if canonical_complete else None,
        "canonical_clob_token_ids": list(clob_token_ids),
        "source_clob_token_ids": _candidate_token_ids(candidate),
        "side_book_token_ids": {
            "yes": yes_book.get("token_id"),
            "no": no_book.get("token_id"),
        },
        "fail_closed_reasons": sorted(set(str(reason) for reason in fail_closed_reasons)),
    }


@dataclass(frozen=True)
class RewardScoreRules:
    """Public-data proxy scoring rules for reward/backtest market selection.

    These rules never imply live trading eligibility. They only rank markets for later
    PMBT replay batches and Homerun shadow-forward logging.
    """

    min_liquidity: float = 25_000.0
    min_volume: float = 25_000.0
    max_side_spread: float = 0.02
    ideal_spread: float = 0.001
    min_depth: float = 5_000.0
    min_days_to_end: float = 3.0
    max_days_to_end: float = 365.0
    min_market_age_hours: float = 1.0
    tail_price_threshold: float = 0.02
    reward_tail_threshold: float = 0.08


def accidental_fill_risk_flags(
    candidate: Mapping[str, Any], rules: RewardScoreRules | None = None
) -> list[str]:
    rules = rules or RewardScoreRules()
    flags: list[str] = []
    yes_mid = _mid(candidate, "yes")
    no_mid = _mid(candidate, "no")
    yes_spread = _spread(candidate, "yes")
    no_spread = _spread(candidate, "no")
    yes_depth = _depth(candidate, "yes")
    no_depth = _depth(candidate, "no")

    for label, mid in (("yes", yes_mid), ("no", no_mid)):
        if mid is not None and (
            mid <= rules.tail_price_threshold or mid >= 1 - rules.tail_price_threshold
        ):
            flags.append(f"{label}_tail_price_accidental_fill_risk")
    if yes_spread > rules.max_side_spread or no_spread > rules.max_side_spread:
        flags.append("wide_spread_adverse_selection_risk")
    if min(yes_depth, no_depth) < rules.min_depth:
        flags.append("thin_top_of_book_fill_risk")
    if yes_depth and no_depth:
        ratio = max(yes_depth, no_depth) / max(1.0, min(yes_depth, no_depth))
        if ratio >= 5:
            flags.append("one_sided_depth_queue_risk")

    return sorted(set(flags))


def score_candidate(
    candidate: Mapping[str, Any],
    *,
    generated_at: datetime | None = None,
    rules: RewardScoreRules | None = None,
) -> dict[str, Any]:
    rules = rules or RewardScoreRules()
    generated_at = generated_at or datetime.now(UTC)
    yes_spread = _spread(candidate, "yes")
    no_spread = _spread(candidate, "no")
    yes_depth = _depth(candidate, "yes")
    no_depth = _depth(candidate, "no")
    liquidity = _as_float(_first_present(candidate, "liquidity", "liquidityNum"))
    volume = _as_float(_first_present(candidate, "volume", "volumeNum"))
    volume_24h = _as_float(_first_present(candidate, "volume24hr", "volume24h", "volume24hrClob"))
    end_dt = _parse_dt(_first_present(candidate, "end_date", "endDate"))
    created_dt = _parse_dt(
        _first_present(candidate, "createdAt", "created_at", "startDate", "start_date")
    )
    days_to_end = ((end_dt - generated_at).total_seconds() / 86400.0) if end_dt else None
    market_age_days = (
        ((generated_at - created_dt).total_seconds() / 86400.0) if created_dt else None
    )
    avg_spread = (yes_spread + no_spread) / 2.0

    spread_component = max(0.0, 1.0 - avg_spread / rules.max_side_spread) * 30.0
    tight_reward_component = (
        max(0.0, 1.0 - abs(avg_spread - rules.ideal_spread) / rules.max_side_spread) * 10.0
    )
    depth_component = min(20.0, math.log10(max(1.0, min(yes_depth, no_depth))) * 4.0)
    liquidity_component = min(15.0, math.log10(max(1.0, liquidity)) * 2.5)
    volume_component = min(10.0, math.log10(max(1.0, max(volume, volume_24h))) * 1.5)
    horizon_component = 0.0
    if days_to_end is not None:
        if rules.min_days_to_end <= days_to_end <= rules.max_days_to_end:
            horizon_component = 10.0
        elif days_to_end > 0:
            horizon_component = 4.0
    age_component = 0.0
    if market_age_days is None or market_age_days * 24.0 >= rules.min_market_age_hours:
        age_component = 3.0
    fits = _strategy_fits(candidate)
    volatility_proxy = (
        5.0 if {"volatility_spike_deep_limit_maker", "deep_limit_maker"} & fits else 0.0
    )
    reward_hint_component = 8.0 if _has_explicit_reward_evidence(candidate) else 0.0

    risk_flags = accidental_fill_risk_flags(candidate, rules)
    risk_penalty = min(18.0, 3.0 * len(risk_flags))
    score = (
        spread_component
        + tight_reward_component
        + depth_component
        + liquidity_component
        + volume_component
        + horizon_component
        + age_component
        + volatility_proxy
        + reward_hint_component
        - risk_penalty
    )

    complete_books = (
        bool(candidate.get("complete_books", candidate.get("complete_yes_no_clob_books", True)))
        and _has_book_prices(candidate, "yes")
        and _has_book_prices(candidate, "no")
    )
    clob_token_ids = _clob_token_ids(candidate)
    has_yes_no_outcomes = _has_yes_no_outcomes(candidate)
    book_token_ids_match = _book_token_ids_match(candidate, clob_token_ids)
    blockers: list[str] = []
    if not complete_books:
        blockers.append("missing_complete_yes_no_clob_books")
    if len(clob_token_ids) != 2:
        blockers.append("invalid_or_missing_yes_no_clob_token_ids")
    if len(clob_token_ids) == 2 and not book_token_ids_match:
        blockers.append("book_token_ids_do_not_match_yes_no_mapping")
    if not has_yes_no_outcomes:
        blockers.append("invalid_yes_no_outcome_mapping")
    if liquidity < rules.min_liquidity:
        blockers.append("liquidity_below_proxy_threshold")
    if volume < rules.min_volume and volume_24h < rules.min_volume:
        blockers.append("volume_below_proxy_threshold")
    if yes_spread > rules.max_side_spread or no_spread > rules.max_side_spread:
        blockers.append("spread_too_wide_for_reward_proxy")
    if days_to_end is not None and days_to_end <= 0:
        blockers.append("expired_or_resolved")
    if market_age_days is not None and market_age_days * 24.0 < rules.min_market_age_hours:
        blockers.append("market_too_new_for_stable_backtest_queue")

    return {
        "reward_proxy_score": round(score, 6),
        "eligible_for_backtest_queue": not blockers,
        "blockers": blockers,
        "features": {
            "yes_spread": yes_spread,
            "no_spread": no_spread,
            "avg_spread": avg_spread,
            "min_top_book_depth": min(yes_depth, no_depth),
            "yes_top_book_depth": yes_depth,
            "no_top_book_depth": no_depth,
            "volume": volume,
            "volume_24h": volume_24h,
            "liquidity": liquidity,
            "days_to_end": None if days_to_end is None else round(days_to_end, 3),
            "market_age_days": None if market_age_days is None else round(market_age_days, 3),
            "volatility_proxy": volatility_proxy,
            "fee_reward_category": _reward_category(candidate),
            "has_explicit_reward_evidence": _has_explicit_reward_evidence(candidate),
            "has_complete_clob_token_ids": len(clob_token_ids) == 2,
            "has_yes_no_outcomes": has_yes_no_outcomes,
            "book_token_ids_match_yes_no_mapping": book_token_ids_match,
        },
        "accidental_fill_risk_flags": risk_flags,
    }


def _scan_candidates(scan: Mapping[str, Any]) -> Sequence[Any]:
    return [row for row, _ in _scan_candidates_with_diagnostics(scan)[0]]


def _scan_timestamp(metadata: Mapping[str, Any]) -> Any:
    return (
        metadata.get("timestamp_utc")
        or metadata.get("utc_timestamp")
        or metadata.get("generated_at_utc")
    )


def _source_provenance_limitations(metadata: Mapping[str, Any]) -> list[str]:
    limitations: list[str] = []
    geoblock = metadata.get("geoblock")
    if isinstance(geoblock, Mapping) and geoblock.get("ok") is not True:
        limitations.append("geoblock_status_provenance_limited")
    return limitations


def _candidate_fail_closed_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = [str(reason) for reason in row.get("blockers", [])]
    book_provenance = row.get("book_provenance")
    if isinstance(book_provenance, Mapping):
        reasons.extend(str(reason) for reason in book_provenance.get("fail_closed_reasons", []))
    token_provenance = row.get("token_provenance")
    if isinstance(token_provenance, Mapping):
        reasons.extend(str(reason) for reason in token_provenance.get("fail_closed_reasons", []))
    return sorted(set(reasons))


def _manifest_strategy_bucket_diagnostics(
    scored: Sequence[Mapping[str, Any]],
    rules: RewardScoreRules,
) -> dict[str, Any]:
    canonical_counts: dict[str, int] = {}
    low_fill_tail_bucket_counts: dict[str, int] = {}
    low_fill_missing_book_fail_closed_count = 0
    low_fill_microprice_overlap_slugs: list[str] = []
    low_fill_volatility_overlap_slugs: list[str] = []
    for row in scored:
        canonical_buckets = row.get("canonical_source_strategy_buckets", [])
        if not isinstance(canonical_buckets, Sequence) or isinstance(
            canonical_buckets, (str, bytes, bytearray)
        ):
            canonical_buckets = []
        for bucket in canonical_buckets:
            bucket_name = str(bucket)
            canonical_counts[bucket_name] = canonical_counts.get(bucket_name, 0) + 1

        reward_tail_bucket = str(row.get("low_fill_reward_tail_bucket"))
        if LOW_FILL_LIQUIDITY_REWARD_MAKER in canonical_buckets:
            low_fill_tail_bucket_counts[reward_tail_bucket] = (
                low_fill_tail_bucket_counts.get(reward_tail_bucket, 0) + 1
            )
            if "missing_complete_yes_no_clob_books" in row.get("blockers", []):
                low_fill_missing_book_fail_closed_count += 1
            if MICROPRICE in canonical_buckets:
                low_fill_microprice_overlap_slugs.append(str(row.get("slug") or row["market_id"]))
            if VOLATILITY_SPIKE_DEEP_LIMIT_MAKER in canonical_buckets:
                low_fill_volatility_overlap_slugs.append(str(row.get("slug") or row["market_id"]))

    return {
        "thresholds": {
            "extreme_tail": rules.tail_price_threshold,
            "reward_tail": rules.reward_tail_threshold,
        },
        "canonical_strategy_bucket_counts": dict(sorted(canonical_counts.items())),
        "low_fill_reward_tail_bucket_counts": dict(sorted(low_fill_tail_bucket_counts.items())),
        "low_fill_extreme_tail_watchlist_count": low_fill_tail_bucket_counts.get(
            "low_fill_extreme_tail_watchlist", 0
        ),
        "low_fill_missing_book_fail_closed_count": low_fill_missing_book_fail_closed_count,
        "low_fill_microprice_overlap_count": len(low_fill_microprice_overlap_slugs),
        "low_fill_microprice_overlap_slugs": sorted(low_fill_microprice_overlap_slugs),
        "low_fill_volatility_overlap_count": len(low_fill_volatility_overlap_slugs),
        "low_fill_volatility_overlap_slugs": sorted(low_fill_volatility_overlap_slugs),
    }


def build_reward_manifest(
    scan: Mapping[str, Any],
    *,
    limit: int = 25,
    rules: RewardScoreRules | None = None,
    source_artifact_path: str | None = None,
) -> dict[str, Any]:
    rules = rules or RewardScoreRules()
    metadata = scan.get("metadata") if isinstance(scan.get("metadata"), Mapping) else {}
    metadata_safety = metadata.get("safety") if isinstance(metadata.get("safety"), Mapping) else {}
    source_scan_timestamp_utc = _scan_timestamp(metadata)
    source_scan_mode = metadata.get("mode") or metadata_safety.get("mode")
    source_artifacts = metadata.get("sources", metadata.get("data_sources", {}))
    generated_at = _parse_dt(source_scan_timestamp_utc) or datetime.now(UTC)
    scored: list[dict[str, Any]] = []
    source_rows, source_candidate_diagnostics = _scan_candidates_with_diagnostics(scan)
    source_provenance_limitations = _source_provenance_limitations(metadata)
    for candidate, source_candidate_provenance in source_rows:
        score = score_candidate(candidate, generated_at=generated_at, rules=rules)
        clob_token_ids = _clob_token_ids(candidate)
        provenance = _book_provenance(
            candidate,
            clob_token_ids,
            source_artifact_path=(
                source_artifact_path
                or _first_present(candidate, "source_path", "source_artifact", "source_scan")
            ),
            source_timestamp_utc=_scan_timestamp(metadata),
        )
        canonical_strategy_buckets = list(
            source_candidate_provenance.get("canonical_strategy_buckets", [])
        )
        low_fill_reward_tail_bucket = _low_fill_reward_tail_bucket(
            candidate,
            canonical_strategy_buckets,
            extreme_tail_threshold=rules.tail_price_threshold,
            reward_tail_threshold=rules.reward_tail_threshold,
        )
        scored.append(
            {
                "market_id": str(_first_present(candidate, "market_id", "id") or ""),
                "condition_id": _first_present(candidate, "condition_id", "conditionId"),
                "slug": candidate.get("slug"),
                "question": candidate.get("question"),
                "source_candidate_provenance": source_candidate_provenance,
                "source_strategy_buckets": source_candidate_provenance.get("strategy_buckets", []),
                "canonical_source_strategy_buckets": canonical_strategy_buckets,
                "source_strategy_overlap": _source_strategy_overlap(canonical_strategy_buckets),
                "low_fill_reward_tail_bucket": low_fill_reward_tail_bucket,
                "low_fill_reward_tail_diagnostic": {
                    "bucket": low_fill_reward_tail_bucket,
                    "tail_distance": _tail_distance(candidate),
                    "extreme_tail_threshold": rules.tail_price_threshold,
                    "reward_tail_threshold": rules.reward_tail_threshold,
                    "canonical_source_strategy_buckets": canonical_strategy_buckets,
                },
                "clob_token_ids": clob_token_ids,
                "source_clob_token_ids": _candidate_token_ids(candidate),
                "yes_token_id": clob_token_ids[0] if len(clob_token_ids) == 2 else None,
                "no_token_id": clob_token_ids[1] if len(clob_token_ids) == 2 else None,
                "outcomes": _outcomes(candidate),
                "reward_evidence": _reward_evidence(candidate),
                "book_provenance": provenance,
                "token_provenance": _token_provenance(candidate, clob_token_ids, provenance),
                "yes_book": provenance["sides"]["yes"],
                "no_book": provenance["sides"]["no"],
                "source_candidate_score": _first_present(candidate, "candidate_score", "score"),
                "source_url": _first_present(candidate, "url", "source_market_url"),
                **score,
            }
        )
        fail_closed_reasons = _candidate_fail_closed_reasons(scored[-1])
        scored[-1]["candidate_diagnostic"] = {
            "included_in_manifest_diagnostics": True,
            "excluded_from_backtest_queue": not scored[-1]["eligible_for_backtest_queue"],
            "exclusion_reasons": list(scored[-1]["blockers"]),
            "fail_closed_reasons": fail_closed_reasons,
            "provenance_limitations": list(source_provenance_limitations),
            "source_strategy_buckets": list(scored[-1]["source_strategy_buckets"]),
            "canonical_source_strategy_buckets": canonical_strategy_buckets,
            "low_fill_reward_tail_bucket": low_fill_reward_tail_bucket,
            "source_strategy_overlap": scored[-1]["source_strategy_overlap"],
            "source_key": source_candidate_provenance.get("source_key"),
            "source_shape": source_candidate_provenance.get("source_shape"),
        }
    scored.sort(
        key=lambda row: (row["eligible_for_backtest_queue"], row["reward_proxy_score"]),
        reverse=True,
    )
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank
    eligible_count = sum(1 for row in scored if row["eligible_for_backtest_queue"])
    explicit_reward_count = sum(
        1 for row in scored if row.get("features", {}).get("has_explicit_reward_evidence")
    )
    blocked_count = len(scored) - eligible_count
    strategy_bucket_diagnostics = _manifest_strategy_bucket_diagnostics(scored, rules)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "mode": SHADOW_MODE,
        "manifest_kind": "reward_scanner_manifest",
        "canonical_manifest_prefix": CANONICAL_MANIFEST_PREFIX,
        "legacy_manifest_prefixes": [LEGACY_MANIFEST_PREFIX],
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_scan_timestamp_utc": source_scan_timestamp_utc,
        "source_scan_mode": source_scan_mode,
        "source_artifacts": source_artifacts,
        "source_provenance": {
            "source_artifact_path": source_artifact_path,
            "source_scan_timestamp_utc": source_scan_timestamp_utc,
            "source_scan_mode": source_scan_mode,
            "source_artifacts": source_artifacts,
            "provenance_limitations": source_provenance_limitations,
        },
        "summary": {
            "candidate_count": len(scored),
            "manifest_candidate_count": len(scored[: max(0, limit)]),
            "eligible_for_backtest_queue_count": eligible_count,
            "blocked_count": blocked_count,
            "explicit_reward_evidence_count": explicit_reward_count,
            "source_normalized_candidate_count": source_candidate_diagnostics[
                "normalized_candidate_count"
            ],
            "source_candidate_exclusion_count": source_candidate_diagnostics[
                "excluded_source_row_count"
            ],
            "source_duplicate_candidate_count": source_candidate_diagnostics[
                "duplicate_source_row_count"
            ],
            "source_strategy_bucket_candidate_count": source_candidate_diagnostics[
                "strategy_bucket_candidate_count"
            ],
            "source_strategy_bucket_unique_candidate_count": source_candidate_diagnostics[
                "strategy_bucket_unique_candidate_count"
            ],
            "low_fill_extreme_tail_watchlist_count": strategy_bucket_diagnostics[
                "low_fill_extreme_tail_watchlist_count"
            ],
            "low_fill_microprice_overlap_count": strategy_bucket_diagnostics[
                "low_fill_microprice_overlap_count"
            ],
            "low_fill_volatility_overlap_count": strategy_bucket_diagnostics[
                "low_fill_volatility_overlap_count"
            ],
            "low_fill_missing_book_fail_closed_count": strategy_bucket_diagnostics[
                "low_fill_missing_book_fail_closed_count"
            ],
        },
        "source_candidate_diagnostics": source_candidate_diagnostics,
        "strategy_bucket_diagnostics": strategy_bucket_diagnostics,
        "source_provenance_limitations": source_provenance_limitations,
        "safety": {
            "live_trading": False,
            "submit_orders": False,
            "sign_orders": False,
            "requires_secrets": False,
            "orders_submitted": False,
            "orders_signed": False,
            "orders_cancelled": False,
            "credentials_required": False,
            "live_trading_worker_started": False,
            "worker_trading_started": False,
            "intended_uses": ["PMBT_BACKTEST_QUEUE", "HOMERUN_SHADOW_FORWARD_LOGGING"],
        },
        "live_trading": False,
        "orders_submitted": False,
        "orders_signed": False,
        "orders_cancelled": False,
        "credentials_required": False,
        "live_trading_worker_started": False,
        "worker_trading_started": False,
        "scoring_rules": asdict(rules),
        "candidates": scored[: max(0, limit)],
    }


def load_scan(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError("scan artifact must be a JSON object")
    return value


def _manifest_stamp(manifest: Mapping[str, Any]) -> str:
    generated_at = manifest.get("generated_at_utc")
    parsed = _parse_dt(generated_at) if isinstance(generated_at, str) else None
    return (parsed or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")


def write_manifest(
    manifest: Mapping[str, Any],
    output_dir: str | Path,
    *,
    prefix: str = CANONICAL_MANIFEST_PREFIX,
    legacy_prefix: str | None = LEGACY_MANIFEST_PREFIX,
) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = _manifest_stamp(manifest)
    manifest_path = out / f"{prefix}_{stamp}.json"
    rules_path = out / f"{prefix}_rules_{stamp}.md"
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    rules = manifest.get("scoring_rules", {})
    safety = manifest.get("safety", {})
    safety_fields = (
        "orders_submitted",
        "orders_signed",
        "orders_cancelled",
        "credentials_required",
        "live_trading_worker_started",
        "worker_trading_started",
    )
    safety_text = "\n".join(
        f"- {field}={str(safety.get(field)).lower()}" for field in safety_fields
    )
    rules_text = (
        "# Reward Market Scanner Scoring Rules\n\n"
        "Mode: SHADOW/BACKTEST ONLY. No live trading, signing, order placement, or secret access.\n\n"
        "This scanner ranks public/read-only Polymarket candidates for later PMBT backtests and Homerun shadow-forward logging. "
        "Reward eligibility is proxy-only unless Polymarket reward metadata is explicitly available.\n\n"
        "## Rule parameters\n"
        + "\n".join(f"- `{key}`: `{value}`" for key, value in sorted(rules.items()))
        + "\n\n## Risk flags\n"
        "- `*_tail_price_accidental_fill_risk`: very low/high probability outcomes can have asymmetric loss if filled.\n"
        "- `wide_spread_adverse_selection_risk`: spread is too wide for reward proxy assumptions.\n"
        "- `thin_top_of_book_fill_risk`: visible top-of-book depth is below proxy threshold.\n"
        "- `one_sided_depth_queue_risk`: queue/depth imbalance may create unfavorable fills.\n"
        "\n## Safety Booleans\n"
        f"{safety_text}\n"
    )
    rules_path.write_text(rules_text, encoding="utf-8")
    if (
        prefix == CANONICAL_MANIFEST_PREFIX
        and legacy_prefix
        and legacy_prefix != CANONICAL_MANIFEST_PREFIX
    ):
        (out / f"{legacy_prefix}_{stamp}.json").write_text(manifest_text, encoding="utf-8")
        (out / f"{legacy_prefix}_rules_{stamp}.md").write_text(rules_text, encoding="utf-8")
    return manifest_path, rules_path
