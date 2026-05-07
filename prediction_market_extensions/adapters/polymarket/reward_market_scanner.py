from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

SHADOW_MODE = "SHADOW_BACKTEST_ONLY_NO_LIVE_TRADING"
MANIFEST_SCHEMA_VERSION = "polymarket.reward-market-manifest.v1"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


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
        "best_bid": raw.get("best_bid"),
        "best_ask": raw.get("best_ask"),
        "best_bid_size": raw.get("best_bid_size", raw.get("bid_size")),
        "best_ask_size": raw.get("best_ask_size", raw.get("ask_size")),
        "depth_2c": raw.get("depth_2c"),
        "mid": raw.get("mid"),
        "spread": raw.get("spread"),
        "token_id": raw.get("token_id"),
    }


def _book(candidate: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    value = candidate.get(f"{side}_book")
    if isinstance(value, Mapping):
        return value
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
        "best_bid_size",
        "best_ask_size",
        "bid_size",
        "ask_size",
        "depth_2c",
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
    return flat


def _spread(candidate: Mapping[str, Any], side: str) -> float:
    book = _book(candidate, side)
    spread = _as_float(book.get("spread"), default=-1.0)
    if spread >= 0:
        return spread
    bid = _as_float(book.get("best_bid"), default=0.0)
    ask = _as_float(book.get("best_ask"), default=0.0)
    return max(0.0, ask - bid) if ask and bid else 0.0


def _depth(candidate: Mapping[str, Any], side: str) -> float:
    book = _book(candidate, side)
    top_depth = _as_float(book.get("best_bid_size", book.get("bid_size"))) + _as_float(
        book.get("best_ask_size", book.get("ask_size"))
    )
    top10_depth = _as_float(book.get("depth_bid_top10")) + _as_float(book.get("depth_ask_top10"))
    return max(top_depth, top10_depth, _as_float(book.get("depth_2c")))


def _mid(candidate: Mapping[str, Any], side: str) -> float | None:
    book = _book(candidate, side)
    mid = _as_float(book.get("mid"), default=-1.0)
    if mid >= 0:
        return mid
    prices = candidate.get("outcomePrices")
    idx = 0 if side == "yes" else 1
    if isinstance(prices, Sequence) and not isinstance(prices, (str, bytes)) and len(prices) > idx:
        parsed = _as_float(prices[idx], default=-1.0)
        return parsed if parsed >= 0 else None
    return None


def _has_explicit_reward_evidence(candidate: Mapping[str, Any]) -> bool:
    reward_evidence = candidate.get("reward_evidence")
    if isinstance(reward_evidence, Mapping) and any(
        value not in (None, "", [], {}, False) for value in reward_evidence.values()
    ):
        return True
    if candidate.get("reward_hint"):
        return True
    for key in ("clobRewards", "rewards", "rewardsMinSize", "rewardsMaxSpread", "umaReward"):
        value = candidate.get(key)
        if value not in (None, "", [], {}, False):
            return True
    fits = candidate.get("strategy_fits") or []
    return isinstance(fits, Sequence) and not isinstance(fits, (str, bytes)) and "reward_eligible_candidate" in fits


def _reward_category(candidate: Mapping[str, Any]) -> str:
    reward_evidence = candidate.get("reward_evidence")
    if isinstance(reward_evidence, Mapping):
        if reward_evidence.get("clobRewards") not in (None, "", [], {}, False):
            return "explicit_clob_rewards"
        if reward_evidence.get("rewardsMinSize") not in (None, "", [], {}, False) or reward_evidence.get(
            "rewardsMaxSpread"
        ) not in (None, "", [], {}, False):
            return "explicit_gamma_reward_terms"
        if reward_evidence.get("umaReward") not in (None, "", [], {}, False):
            return "explicit_uma_reward_hint"
    if candidate.get("clobRewards") not in (None, "", [], {}, False):
        return "explicit_clob_rewards"
    if candidate.get("rewardsMinSize") not in (None, "", [], {}, False) or candidate.get(
        "rewardsMaxSpread"
    ) not in (None, "", [], {}, False):
        return "explicit_gamma_reward_terms"
    if candidate.get("umaReward") not in (None, "", [], {}, False):
        return "explicit_uma_reward_hint"
    if candidate.get("reward_hint"):
        return "explicit_reward_hint"
    return "public_proxy_only_reward_unverified"


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
        return [
            str(book.get("token_id"))
            for book in books
            if isinstance(book, Mapping) and book.get("token_id") not in (None, "")
        ]
    yes_token = _book(candidate, "yes").get("token_id")
    no_token = _book(candidate, "no").get("token_id")
    return [str(token_id) for token_id in (yes_token, no_token) if token_id not in (None, "")]


def _clob_token_ids(candidate: Mapping[str, Any]) -> list[str]:
    ids = _candidate_token_ids(candidate)
    outcomes = _outcomes(candidate)
    if len(ids) != 2 or len(set(ids)) != 2 or not _has_yes_no_outcomes(candidate):
        return []
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


def _book_token_ids_match(candidate: Mapping[str, Any], clob_token_ids: Sequence[str]) -> bool:
    if len(clob_token_ids) != 2:
        return False
    yes_token = _book(candidate, "yes").get("token_id") or candidate.get("yes_token_id")
    no_token = _book(candidate, "no").get("token_id") or candidate.get("no_token_id")
    if yes_token in (None, "") and no_token in (None, ""):
        return True
    return [str(yes_token), str(no_token)] == [str(clob_token_ids[0]), str(clob_token_ids[1])]


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
        bool(candidate.get("complete_books", True))
        and bool(_book(candidate, "yes"))
        and bool(_book(candidate, "no"))
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
    for key in ("top_candidates", "candidates"):
        value = scan.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return value
    return []


def _scan_timestamp(metadata: Mapping[str, Any]) -> Any:
    return (
        metadata.get("timestamp_utc")
        or metadata.get("utc_timestamp")
        or metadata.get("generated_at_utc")
    )


def build_reward_manifest(
    scan: Mapping[str, Any], *, limit: int = 25, rules: RewardScoreRules | None = None
) -> dict[str, Any]:
    rules = rules or RewardScoreRules()
    metadata = scan.get("metadata") if isinstance(scan.get("metadata"), Mapping) else {}
    generated_at = _parse_dt(_scan_timestamp(metadata)) or datetime.now(UTC)
    scored: list[dict[str, Any]] = []
    for candidate in _scan_candidates(scan):
        if not isinstance(candidate, Mapping):
            continue
        score = score_candidate(candidate, generated_at=generated_at, rules=rules)
        scored.append(
            {
                "market_id": str(_first_present(candidate, "market_id", "id") or ""),
                "condition_id": _first_present(candidate, "condition_id", "conditionId"),
                "slug": candidate.get("slug"),
                "question": candidate.get("question"),
                "clob_token_ids": _clob_token_ids(candidate),
                "outcomes": _outcomes(candidate),
                "source_candidate_score": _first_present(candidate, "candidate_score", "score"),
                "source_url": _first_present(candidate, "url", "source_market_url"),
                **score,
            }
        )
    scored.sort(key=lambda row: row["reward_proxy_score"], reverse=True)
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank
    eligible_count = sum(1 for row in scored if row["eligible_for_backtest_queue"])
    explicit_reward_count = sum(
        1 for row in scored if row.get("features", {}).get("has_explicit_reward_evidence")
    )
    blocked_count = len(scored) - eligible_count
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "mode": SHADOW_MODE,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_scan_timestamp_utc": _scan_timestamp(metadata),
        "source_scan_mode": metadata.get("mode") or metadata.get("safety", {}).get("mode"),
        "source_artifacts": metadata.get("sources", metadata.get("data_sources", {})),
        "summary": {
            "candidate_count": len(scored),
            "manifest_candidate_count": len(scored[: max(0, limit)]),
            "eligible_for_backtest_queue_count": eligible_count,
            "blocked_count": blocked_count,
            "explicit_reward_evidence_count": explicit_reward_count,
        },
        "safety": {
            "live_trading": False,
            "submit_orders": False,
            "sign_orders": False,
            "requires_secrets": False,
            "intended_uses": ["PMBT_BACKTEST_QUEUE", "HOMERUN_SHADOW_FORWARD_LOGGING"],
        },
        "scoring_rules": asdict(rules),
        "candidates": scored[: max(0, limit)],
    }


def load_scan(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError("scan artifact must be a JSON object")
    return value


def write_manifest(
    manifest: Mapping[str, Any], output_dir: str | Path, *, prefix: str = "reward_market_manifest"
) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = out / f"{prefix}_{stamp}.json"
    rules_path = out / f"{prefix}_rules_{stamp}.md"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rules = manifest.get("scoring_rules", {})
    rules_path.write_text(
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
        "- `one_sided_depth_queue_risk`: queue/depth imbalance may create unfavorable fills.\n",
        encoding="utf-8",
    )
    return manifest_path, rules_path
