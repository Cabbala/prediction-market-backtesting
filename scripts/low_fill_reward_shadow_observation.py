from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

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
    "worker_trading_started": False,
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


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


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


def _book_provenance(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("book_provenance")
    return value if isinstance(value, Mapping) else {}


def _provenance_side(row: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    provenance = _book_provenance(row)
    sides = provenance.get("sides")
    if isinstance(sides, Mapping):
        value = sides.get(side)
        if isinstance(value, Mapping):
            return value
    value = row.get(f"{side}_book")
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return [value] if value else []
        value = parsed
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray, Mapping)):
        return [str(item) for item in value if item not in (None, "")]
    return []


def _top_book(row: Mapping[str, Any], side: str) -> dict[str, Any]:
    side_book = _provenance_side(row, side)
    return {
        "side": side,
        "token_id": side_book.get("token_id") or row.get(f"{side}_token_id"),
        "best_bid": _parse_float(side_book.get("best_bid") or side_book.get("bid")),
        "best_ask": _parse_float(side_book.get("best_ask") or side_book.get("ask")),
        "best_bid_size": _parse_float(side_book.get("best_bid_size") or side_book.get("bid_size")),
        "best_ask_size": _parse_float(side_book.get("best_ask_size") or side_book.get("ask_size")),
        "depth_bid": _parse_float(side_book.get("depth_bid")),
        "depth_ask": _parse_float(side_book.get("depth_ask")),
        "depth_proxy": _parse_float(side_book.get("depth_proxy")),
        "mid": _parse_float(side_book.get("mid")),
        "spread": _parse_float(side_book.get("spread")),
        "present": bool(side_book.get("present", bool(side_book))),
    }


def _token_provenance(row: Mapping[str, Any]) -> dict[str, Any]:
    explicit = row.get("token_provenance")
    explicit_map = explicit if isinstance(explicit, Mapping) else {}
    canonical_ids = _string_list(
        row.get("clob_token_ids") or explicit_map.get("canonical_clob_token_ids")
    )
    canonical_yes = (
        row.get("yes_token_id")
        or explicit_map.get("canonical_yes_token_id")
        or (canonical_ids[0] if len(canonical_ids) == 2 else None)
    )
    canonical_no = (
        row.get("no_token_id")
        or explicit_map.get("canonical_no_token_id")
        or (canonical_ids[1] if len(canonical_ids) == 2 else None)
    )
    yes_book = _top_book(row, "yes")
    no_book = _top_book(row, "no")
    source_ids = _string_list(
        row.get("source_clob_token_ids")
        or explicit_map.get("source_clob_token_ids")
        or row.get("clobTokenIds")
    )
    if not source_ids and len(canonical_ids) == 2:
        source_ids = list(canonical_ids)
    fail_closed_reasons = set(
        str(reason) for reason in explicit_map.get("fail_closed_reasons") or []
    )
    provenance = _book_provenance(row)
    fail_closed_reasons.update(
        str(reason) for reason in provenance.get("fail_closed_reasons") or []
    )
    if (
        (canonical_yes in (None, "") or canonical_no in (None, ""))
        and provenance.get("complete") is True
        and yes_book.get("token_id") not in (None, "")
        and no_book.get("token_id") not in (None, "")
        and str(yes_book["token_id"]) != str(no_book["token_id"])
    ):
        canonical_yes = yes_book["token_id"]
        canonical_no = no_book["token_id"]
    canonical_complete = (
        canonical_yes not in (None, "")
        and canonical_no not in (None, "")
        and str(canonical_yes) != str(canonical_no)
    )
    if not canonical_complete:
        fail_closed_reasons.add("missing_canonical_yes_no_clob_token_ids")
    book_complete = bool(provenance.get("complete")) if provenance else False
    status = "complete" if canonical_complete and book_complete else "incomplete_fail_closed"
    return {
        "status": status,
        "canonical_complete": canonical_complete,
        "canonical_yes_token_id": str(canonical_yes) if canonical_yes not in (None, "") else None,
        "canonical_no_token_id": str(canonical_no) if canonical_no not in (None, "") else None,
        "canonical_clob_token_ids": (
            [str(canonical_yes), str(canonical_no)] if canonical_complete else []
        ),
        "source_clob_token_ids": source_ids,
        "side_book_token_ids": {
            "yes": yes_book.get("token_id"),
            "no": no_book.get("token_id"),
        },
        "fail_closed_reasons": sorted(fail_closed_reasons),
    }


def _observation_key(row: Mapping[str, Any]) -> str:
    slug = _candidate_slug(dict(row))
    token_provenance = _token_provenance(row)
    token_ids = (
        token_provenance["canonical_clob_token_ids"] or token_provenance["source_clob_token_ids"]
    )
    if len(token_ids) == 2:
        return f"{slug}|yes={token_ids[0]}|no={token_ids[1]}"
    return slug


def _first_side_float(row: dict[str, Any], side: str, *keys: str) -> float | None:
    side_book = _provenance_side(row, side)
    for key in keys:
        value = _parse_float(side_book.get(key))
        if value is not None:
            return value
    return None


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


def _evidence_sources(
    row: Mapping[str, Any], evidence_rows: Iterable[Mapping[str, Any]] = ()
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    source_manifest = (
        row.get("source_manifest") or row.get("source_path") or row.get("source_artifact")
    )
    if source_manifest not in (None, ""):
        sources.append({"type": "source_manifest", "path": str(source_manifest)})
    provenance = _book_provenance(row)
    if provenance:
        sources.append(
            {
                "type": "book_provenance",
                "path": provenance.get("source_artifact_path"),
                "timestamp_utc": provenance.get("source_timestamp_utc"),
                "status": provenance.get("status"),
                "complete": provenance.get("complete"),
                "fail_closed_reasons": provenance.get("fail_closed_reasons", []),
            }
        )
    if _reward_evidence(dict(row)):
        sources.append({"type": "reward_terms", "path": source_manifest})
    for evidence in evidence_rows:
        source_file = evidence.get("source_evidence_file") or evidence.get("source_snapshot_file")
        sources.append(
            {
                "type": str(evidence.get("evidence_type") or "shadow_evidence_log"),
                "path": str(source_file) if source_file not in (None, "") else None,
                "timestamp_utc": evidence.get("source_observed_at_utc")
                or evidence.get("generated_at_utc"),
            }
        )
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for source in sources:
        key = (source.get("type"), source.get("path"), source.get("timestamp_utc"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def _reward_value(row: Mapping[str, Any]) -> float | None:
    evidence = _reward_evidence(dict(row))
    return _parse_float(
        evidence.get("umaReward")
        or evidence.get("reward")
        or row.get("reward_value")
        or row.get("reward_amount")
    )


def _reward_score_share(row: Mapping[str, Any]) -> float | None:
    return _parse_float(
        row.get("reward_score_share_proxy")
        or row.get("estimated_reward_score_share_proxy")
        or row.get("estimated_reward_score_share")
    )


def _time_in_band_fraction(row: Mapping[str, Any], measurement: Mapping[str, Any]) -> float | None:
    direct = _parse_float(row.get("time_in_band_ratio"))
    if direct is not None:
        return max(0.0, min(1.0, direct))
    observed_fraction = _parse_float(measurement.get("observed_fraction"))
    if observed_fraction is not None and str(measurement.get("status", "")).startswith(
        "measured_multi_snapshot"
    ):
        return max(0.0, min(1.0, observed_fraction))
    return None


def _exit_loss_proxy(
    row: Mapping[str, Any], *, yes_mid: float | None, spread: float | None
) -> dict[str, Any]:
    half_spread_over_mid = None
    if yes_mid is not None and yes_mid > 0 and spread is not None:
        half_spread_over_mid = (spread / 2.0) / yes_mid
    if half_spread_over_mid is None:
        return {
            "status": "unknown_missing_mid_or_spread",
            "expected_exit_loss_proxy": None,
            "half_spread_over_mid": None,
            "basis": "requires_top_of_book_mid_and_spread_or_l2_exit_path",
        }
    return {
        "status": "proxy_from_top_of_book_half_spread",
        "expected_exit_loss_proxy": _round(half_spread_over_mid),
        "half_spread_over_mid": _round(half_spread_over_mid),
        "basis": "top_of_book_half_spread_over_mid_proxy_not_realized_exit",
    }


def _reward_ev_fields(
    row: Mapping[str, Any],
    *,
    would_have_filled_probability: float | None,
    exit_loss_proxy: Mapping[str, Any],
    time_in_band: Mapping[str, Any],
    fill_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    reward_value = _reward_value(row)
    reward_score_share = _reward_score_share(row)
    time_fraction = _time_in_band_fraction(row, time_in_band)
    expected_exit_loss = _parse_float(exit_loss_proxy.get("expected_exit_loss_proxy"))
    conservative_fill_count = int(fill_evidence.get("conservative_would_fill_snapshot_count") or 0)
    missing: list[str] = []
    if would_have_filled_probability is None:
        missing.append("would_have_filled_probability")
    if conservative_fill_count <= 0:
        missing.append("conservative_would_fill_evidence")
    if reward_value is None:
        missing.append("reward_amount")
    if reward_score_share is None:
        missing.append("reward_denominator_or_score_share")
    if time_fraction is None:
        missing.append("time_in_band")
    if expected_exit_loss is None:
        missing.append("exit_loss_proxy")
    if missing:
        return {
            "reward_ev_status": "reward_ev_not_yet_computable",
            "expected_reward_ev_minus_loss": None,
            "missing_reward_ev_inputs": missing,
            "reward_ev_inputs": {
                "reward_amount": _round(reward_value),
                "reward_score_share_proxy": _round(reward_score_share),
                "time_in_band_fraction": _round(time_fraction),
                "would_have_filled_probability": _round(would_have_filled_probability),
                "conservative_would_fill_snapshot_count": conservative_fill_count,
                "expected_exit_loss_proxy": _round(expected_exit_loss),
            },
        }
    assert reward_value is not None
    assert reward_score_share is not None
    assert time_fraction is not None
    assert expected_exit_loss is not None
    assert would_have_filled_probability is not None
    return {
        "reward_ev_status": "computable_shadow_proxy_not_profit_claim",
        "expected_reward_ev_minus_loss": _round(
            reward_value * reward_score_share * time_fraction
            - would_have_filled_probability * expected_exit_loss
        ),
        "missing_reward_ev_inputs": [],
        "reward_ev_inputs": {
            "reward_amount": _round(reward_value),
            "reward_score_share_proxy": _round(reward_score_share),
            "time_in_band_fraction": _round(time_fraction),
            "would_have_filled_probability": _round(would_have_filled_probability),
            "conservative_would_fill_snapshot_count": conservative_fill_count,
            "expected_exit_loss_proxy": _round(expected_exit_loss),
        },
    }


def _observation(
    row: dict[str, Any], evidence_rows: Iterable[Mapping[str, Any]] = ()
) -> dict[str, Any]:
    evidence_row_list = [dict(evidence) for evidence in evidence_rows]
    yes_mid = _first_float(row, "yes_mid", "scan_mid", "yes_probability")
    if yes_mid is None:
        yes_mid = _first_side_float(row, "yes", "mid")
    spread = _first_float(
        row,
        "spread",
        "scan_spread",
        "avg_spread",
        "max_spread",
        "yes_spread",
    )
    if spread is None:
        spread = _first_side_float(row, "yes", "spread")
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
    exit_loss_proxy = (
        _exit_loss_proxy_measurement(evidence_row_list)
        if evidence_row_list
        else _exit_loss_proxy(row, yes_mid=yes_mid, spread=spread)
    )
    would_have_filled = (
        _would_have_filled_measurement(evidence_row_list)
        if evidence_row_list
        else {
            "status": "unknown_requires_l2_or_shadow_quote_log",
            "value": None,
            "probability": None,
            "basis": "requires_l2_replay_or_shadow_quote_lifecycle_log",
        }
    )
    would_have_filled_probability = would_have_filled.get("probability")
    would_have_filled_status = str(would_have_filled["status"])
    token_provenance = _token_provenance(row)
    yes_book = _top_book(row, "yes")
    no_book = _top_book(row, "no")
    time_in_band = {
        "status": time_in_band_status,
        "sample_count": 1,
        "in_band": in_band,
        "in_band_snapshot_count": 1 if in_band is True else 0,
        "observed_fraction": (1.0 if in_band else 0.0) if in_band is not None else None,
        "basis": time_in_band_basis,
    }
    fill_evidence = _fill_evidence_summary(evidence_row_list or [row])
    ev_fields = _reward_ev_fields(
        row,
        would_have_filled_probability=would_have_filled_probability,
        exit_loss_proxy=exit_loss_proxy,
        time_in_band=time_in_band,
        fill_evidence=fill_evidence,
    )
    return {
        "slug": _candidate_slug(row),
        "market_slug": _candidate_slug(row),
        "observation_key": _observation_key(row),
        "question": row.get("question"),
        "yes_token_id": token_provenance["canonical_yes_token_id"],
        "no_token_id": token_provenance["canonical_no_token_id"],
        "canonical_yes_token_id": token_provenance["canonical_yes_token_id"],
        "canonical_no_token_id": token_provenance["canonical_no_token_id"],
        "canonical_clob_token_ids": token_provenance["canonical_clob_token_ids"],
        "source_clob_token_ids": token_provenance["source_clob_token_ids"],
        "token_provenance": token_provenance,
        "book_provenance": _book_provenance(row),
        "book_provenance_status": _book_provenance(row).get("status"),
        "book_provenance_complete": bool(_book_provenance(row).get("complete")),
        "yes_book": yes_book,
        "no_book": no_book,
        "evidence_sources": _evidence_sources(row, evidence_row_list),
        "yes_mid": yes_mid,
        "spread": spread,
        "liquidity": liquidity,
        "reward_max_spread": reward_max_spread,
        "has_reward_evidence": has_reward_evidence,
        "time_in_band_observed": in_band,
        "time_in_band_basis": time_in_band_basis,
        "time_in_band": time_in_band,
        "would_have_filled": "unknown_requires_l2_or_shadow_quote_log",
        "would_have_filled_status": would_have_filled_status,
        "would_have_filled_probability": would_have_filled_probability,
        "fill_evidence": fill_evidence,
        "accidental_fill_risk": accidental_fill_risk,
        "exit_risk": exit_risk,
        "exit_loss_proxy": exit_loss_proxy,
        **ev_fields,
        "orders_submitted": False,
        "orders_signed": False,
        "orders_cancelled": False,
        "credentials_required": False,
        "live_trading_worker_started": False,
        "worker_trading_started": False,
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
        "worker_trading_started": safety.get("worker_trading_started"),
    }
    for key, value in unsafe_fields.items():
        if value not in (False, None):
            raise ValueError(f"{path} is not shadow-safe: {key}={value!r}")


def _safe_evidence_row(path: Path, row: Mapping[str, Any]) -> None:
    safety = row.get("safety")
    if isinstance(safety, Mapping):
        source = safety
    else:
        source = row
    unsafe_fields = {
        "live_trading": source.get("live_trading"),
        "orders_submitted": source.get("orders_submitted"),
        "orders_signed": source.get("orders_signed"),
        "orders_cancelled": source.get("orders_cancelled"),
        "credentials_required": source.get("credentials_required"),
        "worker_trading_started": source.get("worker_trading_started"),
        "live_trading_worker_started": source.get("live_trading_worker_started"),
    }
    for key, value in unsafe_fields.items():
        if value not in (False, None):
            raise ValueError(f"{path} evidence is not shadow-safe: {key}={value!r}")


def _evidence_rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for key, evidence_type in (
        ("records", "reward_shadow_log"),
        ("observations", "low_fill_shadow_observation"),
        ("snapshots", "low_fill_lifecycle_snapshot"),
        ("candidate_table", "low_fill_lifecycle_candidate"),
    ):
        value = payload.get(key)
        if isinstance(value, list):
            for row in value:
                if isinstance(row, Mapping):
                    copied = dict(row)
                    copied.setdefault("evidence_type", evidence_type)
                    rows.append(copied)
    if not rows and payload.get("slug"):
        rows.append(dict(payload))
    return rows


def _load_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _load_evidence_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = _load_json_or_jsonl(path)
        if isinstance(payload, Mapping):
            _safe_evidence_row(path, payload)
        for row in _evidence_rows_from_payload(payload):
            _safe_evidence_row(path, row)
            row["source_evidence_file"] = str(path)
            if "source_observed_at_utc" not in row:
                row["source_observed_at_utc"] = row.get("generated_at_utc")
            rows.append(row)
    return rows


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
    seen_keys: set[str] = set()
    rows: list[dict[str, Any]] = []
    for raw in observations:
        if not isinstance(raw, dict):
            raise ValueError(f"{path} contains a non-object observation row")
        slug = _candidate_slug(raw)
        if slug == "unknown":
            raise ValueError(f"{path} contains an observation without a stable slug")
        row = dict(raw)
        observation_key = _observation_key(row)
        if observation_key in seen_keys:
            raise ValueError(f"{path} contains duplicate observation rows for {observation_key}")
        seen_keys.add(observation_key)
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
    would_fill = row.get("would_fill")
    if isinstance(would_fill, dict) and would_fill.get("known") is True:
        return _parse_bool(would_fill.get("estimate"))
    estimate = row.get("would_have_filled_estimate")
    if isinstance(estimate, dict) and estimate.get("known") is True:
        return _parse_bool(estimate.get("estimate"))
    return None


def _would_fill_classification(row: Mapping[str, Any]) -> str:
    direct = row.get("would_fill_classification")
    if direct not in (None, ""):
        return str(direct)
    would_fill = row.get("would_fill")
    if isinstance(would_fill, Mapping) and would_fill.get("classification") not in (None, ""):
        return str(would_fill["classification"])
    shadow_quote = row.get("shadow_quote")
    if isinstance(shadow_quote, Mapping) and shadow_quote.get("would_fill_classification") not in (
        None,
        "",
    ):
        return str(shadow_quote["would_fill_classification"])
    return "unknown"


def _touch_cross_status(row: Mapping[str, Any]) -> str:
    for value in (
        row.get("hypothetical_quote_touch_cross_status"),
        row.get("touch_cross_status"),
    ):
        if value not in (None, ""):
            return str(value)
    return "unknown"


def _cancel_or_reprice_reason(row: Mapping[str, Any]) -> str:
    value = row.get("cancel_or_reprice_reason")
    return str(value) if value not in (None, "") else "unknown"


def _has_markout_placeholder(row: Mapping[str, Any]) -> bool:
    markout = row.get("markout_proxy")
    if isinstance(markout, Mapping):
        return True
    exit_risk = row.get("exit_risk")
    if isinstance(exit_risk, Mapping) and isinstance(exit_risk.get("markout_proxy"), Mapping):
        return True
    return False


def _fill_evidence_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    classifications = [_would_fill_classification(row) for row in rows]
    optimistic_count = sum(1 for value in classifications if value == "optimistic")
    unknown_count = sum(
        1 for value in classifications if value not in {"conservative", "optimistic"}
    )
    touch_statuses = [_touch_cross_status(row) for row in rows]
    cancel_reprice_reasons = [_cancel_or_reprice_reason(row) for row in rows]
    known_fill_values = [_snapshot_would_have_filled(row) for row in rows]
    known_fill_count = sum(1 for value in known_fill_values if value is not None)
    conservative_fill_count = sum(
        1
        for row, known in zip(rows, known_fill_values, strict=True)
        if known is not None and _would_fill_classification(row) == "conservative"
    )
    post_only_cross_count = sum(
        1 for status in touch_statuses if "would_cross_or_take_current_ask" in status
    )
    cancel_or_reprice_count = sum(
        1 for reason in cancel_reprice_reasons if reason not in {"none_hold_quote", "unknown", ""}
    )
    status = "unknown_no_would_fill_evidence"
    if conservative_fill_count:
        status = "conservative_would_fill_evidence_present"
    elif known_fill_count:
        status = "diagnostic_proxy_would_fill_only"
    return {
        "status": status,
        "sample_count": len(rows),
        "known_would_fill_snapshot_count": known_fill_count,
        "conservative_would_fill_snapshot_count": conservative_fill_count,
        "optimistic_would_fill_snapshot_count": optimistic_count,
        "unknown_would_fill_snapshot_count": unknown_count,
        "post_only_cross_snapshot_count": post_only_cross_count,
        "cancel_or_reprice_snapshot_count": cancel_or_reprice_count,
        "touch_cross_statuses": sorted(set(touch_statuses)),
        "cancel_or_reprice_reasons": sorted(set(cancel_reprice_reasons)),
        "markout_placeholder_count": sum(1 for row in rows if _has_markout_placeholder(row)),
        "basis": (
            "conservative maker fill requires explicit L2/trade evidence; "
            "optimistic post-only shadow quotes remain diagnostic"
        ),
    }


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
            "probability": None,
            "filled_snapshot_count": 0,
            "measured_snapshot_count": 0,
            "basis": "requires_l2_replay_or_shadow_quote_lifecycle_log",
        }
    filled = sum(1 for value in known if value)
    probability = filled / len(known)
    return {
        "status": "measured_from_shadow_quote_log",
        "value": filled > 0,
        "probability": round(probability, 6),
        "filled_snapshot_count": filled,
        "measured_snapshot_count": len(known),
        "basis": "explicit_shadow_quote_fill_observations",
    }


def _exit_loss_proxy_measurement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    proxies: list[float] = []
    for row in rows:
        exit_loss = row.get("exit_loss_proxy")
        if isinstance(exit_loss, Mapping):
            value = _parse_float(
                exit_loss.get("expected_exit_loss_proxy") or exit_loss.get("half_spread_over_mid")
            )
            if value is not None:
                proxies.append(value)
        exit_slippage = row.get("exit_slippage_proxy")
        if isinstance(exit_slippage, Mapping):
            value = _parse_float(exit_slippage.get("half_spread_over_mid"))
            if value is not None:
                proxies.append(value)
    if not proxies:
        latest_mid = _latest_float(rows, "yes_mid")
        latest_spread = _latest_float(rows, "spread")
        return _exit_loss_proxy({}, yes_mid=latest_mid, spread=latest_spread)
    return {
        "status": "measured_or_proxy_from_shadow_evidence",
        "expected_exit_loss_proxy": round(max(proxies), 6),
        "half_spread_over_mid": round(max(proxies), 6),
        "basis": "worst_observed_shadow_exit_loss_proxy",
    }


def _latest_float(rows: list[dict[str, Any]], key: str) -> float | None:
    for row in reversed(rows):
        value = _parse_float(row.get(key))
        if value is not None:
            return value
    return None


def _aggregate_candidate(observation_key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
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
    latest = rows[-1] if rows else {}
    slug = _candidate_slug(latest) if latest else observation_key
    question = next((row.get("question") for row in rows if row.get("question")), None)
    observed_times = [
        row["source_observed_at_utc"]
        for row in rows
        if isinstance(row.get("source_observed_at_utc"), str)
    ]
    would_have_filled = _would_have_filled_measurement(rows)
    exit_loss_proxy = _exit_loss_proxy_measurement(rows)
    time_in_band = _time_in_band_measurement(rows)
    fill_evidence = _fill_evidence_summary(rows)
    token_provenance = _token_provenance(latest)
    yes_book = _top_book(latest, "yes")
    no_book = _top_book(latest, "no")
    ev_fields = _reward_ev_fields(
        latest,
        would_have_filled_probability=would_have_filled.get("probability"),
        exit_loss_proxy=exit_loss_proxy,
        time_in_band=time_in_band,
        fill_evidence=fill_evidence,
    )
    return {
        "slug": slug,
        "market_slug": slug,
        "observation_key": observation_key,
        "question": question,
        "yes_token_id": token_provenance["canonical_yes_token_id"],
        "no_token_id": token_provenance["canonical_no_token_id"],
        "canonical_yes_token_id": token_provenance["canonical_yes_token_id"],
        "canonical_no_token_id": token_provenance["canonical_no_token_id"],
        "canonical_clob_token_ids": token_provenance["canonical_clob_token_ids"],
        "source_clob_token_ids": token_provenance["source_clob_token_ids"],
        "token_provenance": token_provenance,
        "book_provenance": _book_provenance(latest),
        "book_provenance_status": _book_provenance(latest).get("status"),
        "book_provenance_complete": bool(_book_provenance(latest).get("complete")),
        "yes_book": yes_book,
        "no_book": no_book,
        "evidence_sources": _evidence_sources(latest, rows),
        "source_observation_count": len(rows),
        "first_observed_at_utc": observed_times[0] if observed_times else None,
        "last_observed_at_utc": observed_times[-1] if observed_times else None,
        "latest_yes_mid": _latest_float(rows, "yes_mid"),
        "latest_spread": _latest_float(rows, "spread"),
        "latest_liquidity": _latest_float(rows, "liquidity"),
        "latest_reward_max_spread": _latest_float(rows, "reward_max_spread"),
        "time_in_band": time_in_band,
        "would_have_filled": would_have_filled,
        "would_have_filled_status": would_have_filled["status"],
        "would_have_filled_probability": would_have_filled.get("probability"),
        "fill_evidence": fill_evidence,
        "accidental_fill_risk": _categorical_measurement(
            rows, "accidental_fill_risk", accidental_severity
        ),
        "exit_risk": _categorical_measurement(rows, "exit_risk", exit_severity),
        "exit_loss_proxy": exit_loss_proxy,
        **ev_fields,
        **SAFETY_FALSE_FIELDS,
    }


def build_aggregation_report(
    snapshot_paths: list[Path], *, limit: int, evidence_paths: list[Path] | None = None
) -> dict[str, Any]:
    evidence_paths = evidence_paths or []
    if not snapshot_paths and not evidence_paths:
        raise ValueError("at least one snapshot or evidence path is required")
    loaded = [_load_snapshot(path) for path in snapshot_paths]
    loaded.sort(key=lambda item: item[0])
    evidence_rows = _load_evidence_rows(evidence_paths)
    grouped: dict[str, list[dict[str, Any]]] = {}
    grouped_by_slug: dict[str, list[str]] = {}
    source_files: list[str] = []
    for _, rows in loaded:
        for row in rows:
            source_file = row["source_snapshot_file"]
            if source_file not in source_files:
                source_files.append(source_file)
            observation_key = _observation_key(row)
            grouped.setdefault(observation_key, []).append(row)
            slug_keys = grouped_by_slug.setdefault(_candidate_slug(row), [])
            if observation_key not in slug_keys:
                slug_keys.append(observation_key)
    source_evidence_files: list[str] = []
    for row in evidence_rows:
        source_file = row["source_evidence_file"]
        if source_file not in source_evidence_files:
            source_evidence_files.append(source_file)
        slug = _candidate_slug(row)
        observation_key = _observation_key(row)
        if snapshot_paths and observation_key not in grouped:
            slug_keys = grouped_by_slug.get(slug, [])
            if len(slug_keys) != 1:
                continue
            observation_key = slug_keys[0]
        grouped.setdefault(observation_key, []).append(row)
        slug_keys = grouped_by_slug.setdefault(slug, [])
        if observation_key not in slug_keys:
            slug_keys.append(observation_key)
    observations = [
        _aggregate_candidate(observation_key, rows)
        for observation_key, rows in sorted(
            grouped.items(), key=lambda item: item[1][0].get("source_observed_at_utc", "")
        )
    ][:limit]
    would_have_filled_known_count = sum(
        1 for row in observations if row["would_have_filled"]["status"].startswith("measured")
    )
    reward_ev_computable_count = sum(
        1
        for row in observations
        if row["reward_ev_status"] == "computable_shadow_proxy_not_profit_claim"
    )
    classification = "diagnostic_only"
    if not observations:
        classification = "blocked"
    elif reward_ev_computable_count > 0:
        classification = "adopted"
    return {
        "schema_version": 2,
        "generated_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
        "mode": AGGREGATION_MODE,
        "classification": classification,
        "safety": _safety_fields(),
        **_safety_fields(),
        "source_snapshot_files": source_files,
        "source_evidence_files": source_evidence_files,
        "snapshot_file_count": len(source_files),
        "evidence_file_count": len(source_evidence_files),
        "candidate_count": len(observations),
        "observations": observations,
        "summary": {
            "candidate_count": len(observations),
            "snapshot_file_count": len(source_files),
            "evidence_file_count": len(source_evidence_files),
            "total_snapshot_observation_rows": sum(len(rows) for _, rows in loaded),
            "total_evidence_rows": len(evidence_rows),
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
            "conservative_would_fill_candidate_count": sum(
                1
                for row in observations
                if row["fill_evidence"]["conservative_would_fill_snapshot_count"] > 0
            ),
            "post_only_cross_or_cancel_reprice_candidate_count": sum(
                1
                for row in observations
                if row["fill_evidence"]["post_only_cross_snapshot_count"] > 0
                or row["fill_evidence"]["cancel_or_reprice_snapshot_count"] > 0
            ),
            "accidental_fill_risk_known_count": sum(
                1
                for row in observations
                if row["accidental_fill_risk"]["status"].startswith("measured")
            ),
            "exit_risk_known_count": sum(
                1 for row in observations if row["exit_risk"]["status"].startswith("measured")
            ),
            "reward_ev_computable_count": reward_ev_computable_count,
        },
    }


def build_report(
    manifest: Path, *, limit: int, evidence_paths: list[Path] | None = None
) -> dict[str, Any]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = _iter_candidates(payload)[:limit]
    evidence_by_slug: dict[str, list[dict[str, Any]]] = {}
    source_evidence_files: list[str] = []
    for evidence in _load_evidence_rows(evidence_paths or []):
        source_file = evidence["source_evidence_file"]
        if source_file not in source_evidence_files:
            source_evidence_files.append(source_file)
        evidence_by_slug.setdefault(_candidate_slug(evidence), []).append(evidence)
    observations = [
        _observation(row, evidence_by_slug.get(_candidate_slug(row), ())) for row in rows
    ]
    would_have_filled_known_count = sum(
        1 for row in observations if row["would_have_filled_status"].startswith("measured")
    )
    classification = "diagnostic_only" if observations else "blocked"
    return {
        "schema_version": 1,
        "generated_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
        "mode": SAFETY_MODE,
        "classification": classification,
        "safety": _safety_fields(),
        **_safety_fields(),
        "source_manifest": str(manifest),
        "source_evidence_files": source_evidence_files,
        "candidate_count": len(rows),
        "observations": observations,
        "summary": {
            "time_in_band_snapshot_count": sum(
                1 for row in observations if row["time_in_band_observed"]
            ),
            "would_have_filled_known_count": would_have_filled_known_count,
            "needs_l2_or_shadow_quote_log_count": sum(
                1 for row in observations if row["would_have_filled_status"].startswith("unknown")
            ),
            "reward_ev_computable_count": sum(
                1
                for row in observations
                if row["reward_ev_status"] == "computable_shadow_proxy_not_profit_claim"
            ),
            "high_relative_tick_cost_count": sum(
                1
                for row in observations
                if row["accidental_fill_risk"] == "high_relative_tick_cost"
            ),
            "classification": classification,
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
            f"- worker_trading_started={str(safety['worker_trading_started']).lower()}",
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
        f"- classification: {report['classification']}",
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
        f"- worker_trading_started={str(safety['worker_trading_started']).lower()}",
        "",
        "No reward, profit, or live-ready claim is made without multi-snapshot L2/shadow-quote fill and exit-risk evidence.",
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
    parser.add_argument(
        "--evidence-log",
        action="append",
        type=Path,
        default=None,
        help="Shadow-only JSON/JSONL evidence log to aggregate by slug.",
    )
    parser.add_argument(
        "--evidence-glob",
        action="append",
        default=None,
        help="Glob of shadow-only JSON/JSONL evidence logs to aggregate by slug.",
    )
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("limit must be >= 1")
    timestamp = args.timestamp or _utc_now().strftime("%Y%m%dT%H%M%SZ")
    snapshot_paths = _expand_paths(args.snapshot, args.snapshots_glob)
    evidence_paths = _expand_paths(args.evidence_log, args.evidence_glob)
    if snapshot_paths:
        report = build_aggregation_report(
            snapshot_paths, limit=args.limit, evidence_paths=evidence_paths
        )
        report["output_files"] = write_outputs(
            report,
            args.output_dir,
            timestamp,
            prefix="low_fill_reward_shadow_quote_aggregation",
        )
    else:
        manifest = args.manifest or _latest_manifest(args.manifest_glob)
        report = build_report(manifest, limit=args.limit, evidence_paths=evidence_paths)
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
