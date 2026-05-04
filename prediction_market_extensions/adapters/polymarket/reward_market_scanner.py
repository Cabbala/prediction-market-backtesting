from __future__ import annotations

import json
import math
from dataclasses import dataclass
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


def _book(candidate: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    value = candidate.get(f"{side}_book")
    return value if isinstance(value, Mapping) else {}


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
    return _as_float(book.get("best_bid_size")) + _as_float(book.get("best_ask_size"))


def _mid(candidate: Mapping[str, Any], side: str) -> float | None:
    book = _book(candidate, side)
    mid = _as_float(book.get("mid"), default=-1.0)
    return mid if mid >= 0 else None


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
    tail_price_threshold: float = 0.02


def accidental_fill_risk_flags(candidate: Mapping[str, Any], rules: RewardScoreRules | None = None) -> list[str]:
    rules = rules or RewardScoreRules()
    flags: list[str] = []
    yes_mid = _mid(candidate, "yes")
    no_mid = _mid(candidate, "no")
    yes_spread = _spread(candidate, "yes")
    no_spread = _spread(candidate, "no")
    yes_depth = _depth(candidate, "yes")
    no_depth = _depth(candidate, "no")

    for label, mid in (("yes", yes_mid), ("no", no_mid)):
        if mid is not None and (mid <= rules.tail_price_threshold or mid >= 1 - rules.tail_price_threshold):
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
    liquidity = _as_float(candidate.get("liquidity"))
    volume = _as_float(candidate.get("volume"))
    end_dt = _parse_dt(candidate.get("end_date") or candidate.get("endDate"))
    days_to_end = ((end_dt - generated_at).total_seconds() / 86400.0) if end_dt else None

    spread_component = max(0.0, 1.0 - ((yes_spread + no_spread) / 2.0) / rules.max_side_spread) * 30.0
    tight_reward_component = max(0.0, 1.0 - abs(((yes_spread + no_spread) / 2.0) - rules.ideal_spread) / rules.max_side_spread) * 10.0
    depth_component = min(20.0, math.log10(max(1.0, min(yes_depth, no_depth))) * 4.0)
    liquidity_component = min(15.0, math.log10(max(1.0, liquidity)) * 2.5)
    volume_component = min(10.0, math.log10(max(1.0, volume)) * 1.5)
    horizon_component = 0.0
    if days_to_end is not None:
        if rules.min_days_to_end <= days_to_end <= rules.max_days_to_end:
            horizon_component = 10.0
        elif days_to_end > 0:
            horizon_component = 4.0
    volatility_proxy = 5.0 if (candidate.get("strategy_fit") or {}).get("volatility_spike_deep_limit_maker") else 0.0
    reward_hint_component = 5.0 if bool(candidate.get("reward_hint")) else 0.0

    risk_flags = accidental_fill_risk_flags(candidate, rules)
    risk_penalty = min(15.0, 3.0 * len(risk_flags))
    score = (
        spread_component
        + tight_reward_component
        + depth_component
        + liquidity_component
        + volume_component
        + horizon_component
        + volatility_proxy
        + reward_hint_component
        - risk_penalty
    )

    complete_books = bool(candidate.get("complete_books")) and _book(candidate, "yes").get("ok", True) and _book(candidate, "no").get("ok", True)
    blockers: list[str] = []
    if not complete_books:
        blockers.append("missing_complete_yes_no_clob_books")
    if liquidity < rules.min_liquidity:
        blockers.append("liquidity_below_proxy_threshold")
    if volume < rules.min_volume:
        blockers.append("volume_below_proxy_threshold")
    if yes_spread > rules.max_side_spread or no_spread > rules.max_side_spread:
        blockers.append("spread_too_wide_for_reward_proxy")
    if days_to_end is not None and days_to_end <= 0:
        blockers.append("expired_or_resolved")

    return {
        "reward_proxy_score": round(score, 6),
        "eligible_for_backtest_queue": not blockers,
        "blockers": blockers,
        "features": {
            "yes_spread": yes_spread,
            "no_spread": no_spread,
            "min_top_book_depth": min(yes_depth, no_depth),
            "yes_top_book_depth": yes_depth,
            "no_top_book_depth": no_depth,
            "volume": volume,
            "liquidity": liquidity,
            "days_to_end": None if days_to_end is None else round(days_to_end, 3),
            "market_age_days": None,
            "volatility_proxy": volatility_proxy,
            "fee_reward_category": "gamma_reward_hint" if bool(candidate.get("reward_hint")) else "public_proxy_only_reward_unverified",
        },
        "accidental_fill_risk_flags": risk_flags,
    }


def build_reward_manifest(scan: Mapping[str, Any], *, limit: int = 25, rules: RewardScoreRules | None = None) -> dict[str, Any]:
    rules = rules or RewardScoreRules()
    metadata = scan.get("metadata") if isinstance(scan.get("metadata"), Mapping) else {}
    generated_at = _parse_dt(metadata.get("timestamp_utc")) or datetime.now(UTC)
    candidates = scan.get("top_candidates") if isinstance(scan.get("top_candidates"), Sequence) else []
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        score = score_candidate(candidate, generated_at=generated_at, rules=rules)
        scored.append(
            {
                "market_id": str(candidate.get("market_id", "")),
                "condition_id": candidate.get("condition_id"),
                "slug": candidate.get("slug"),
                "question": candidate.get("question"),
                "clob_token_ids": candidate.get("clob_token_ids") or [],
                "outcomes": candidate.get("outcomes") or ["Yes", "No"],
                "source_candidate_score": candidate.get("candidate_score"),
                **score,
            }
        )
    scored.sort(key=lambda row: row["reward_proxy_score"], reverse=True)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "mode": SHADOW_MODE,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_scan_timestamp_utc": metadata.get("timestamp_utc"),
        "source_scan_mode": metadata.get("mode"),
        "source_artifacts": metadata.get("sources", {}),
        "safety": {
            "live_trading": False,
            "submit_orders": False,
            "sign_orders": False,
            "requires_secrets": False,
            "intended_uses": ["PMBT_BACKTEST_QUEUE", "HOMERUN_SHADOW_FORWARD_LOGGING"],
        },
        "scoring_rules": rules.__dict__,
        "candidates": scored[:limit],
    }


def load_scan(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError("scan artifact must be a JSON object")
    return value


def write_manifest(manifest: Mapping[str, Any], output_dir: str | Path, *, prefix: str = "reward_market_manifest") -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = out / f"{prefix}_{stamp}.json"
    rules_path = out / f"{prefix}_rules_{stamp}.md"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
