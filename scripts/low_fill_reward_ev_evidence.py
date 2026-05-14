from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

SAFETY_MODE = "reward_ev_evidence_shadow_backtest_only_no_live_orders"
DEFAULT_OBSERVATION_GLOB = (
    "/opt/polymarket-lab/reports/reward_scanner/shadow_observations/"
    "low_fill_reward_shadow_observation_*.json"
)
DEFAULT_OUTPUT_DIR = Path("/opt/polymarket-lab/reports/reward_scanner/ev_evidence")
DEFAULT_TICK_SIZE = 0.001
HIGH_RELATIVE_TICK_COST = 0.20
MEDIUM_RELATIVE_TICK_COST = 0.10
SAFETY_FALSE_FIELDS = (
    "orders_submitted",
    "orders_signed",
    "orders_cancelled",
    "credentials_required",
    "live_trading_worker_started",
    "worker_trading_started",
)
REAL_FILL_EVIDENCE_MARKERS = (
    "l2_queue",
    "l2 queue",
    "trade_tape",
    "trade tape",
    "l2_replay",
    "l2 replay",
    "orderbookdeltas",
    "order_book_deltas",
)


def _utc_now() -> datetime:
    return datetime.now(tz=UTC).replace(microsecond=0)


def _utc_stamp(dt: datetime | None = None) -> str:
    return (dt or _utc_now()).strftime("%Y%m%dT%H%M%SZ")


def safety_object() -> dict[str, bool]:
    return {
        "live_trading": False,
        "orders_submitted": False,
        "orders_signed": False,
        "orders_cancelled": False,
        "credentials_required": False,
        "live_trading_worker_started": False,
        "worker_trading_started": False,
    }


def _parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _candidate_slug(row: Mapping[str, Any]) -> str:
    return str(
        row.get("slug")
        or row.get("market_slug")
        or row.get("question")
        or row.get("condition_id")
        or "unknown"
    )


def _latest_path(pattern: str) -> Path:
    matches = sorted(Path("/").glob(pattern.lstrip("/")), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"no files match {pattern}")
    return matches[-1]


def _safe_payload(path: Path, payload: Mapping[str, Any]) -> None:
    safety = _as_mapping(payload.get("safety"))
    if not safety:
        raise ValueError(f"{path} is missing a shadow safety block")
    unsafe = {
        "live_trading": safety.get("live_trading"),
        "orders_submitted": safety.get("orders_submitted"),
        "orders_signed": safety.get("orders_signed"),
        "orders_cancelled": safety.get("orders_cancelled"),
        "credentials_required": safety.get("credentials_required"),
        "live_trading_worker_started": safety.get("live_trading_worker_started"),
        "worker_trading_started": safety.get("worker_trading_started"),
    }
    for key, value in unsafe.items():
        if value not in (False, None):
            raise ValueError(f"{path} is not shadow-safe: {key}={value!r}")


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _candidate_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("observations", "candidate_table", "candidates", "top_candidates"):
        value = payload.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            rows = [dict(row) for row in value if isinstance(row, Mapping)]
            if rows:
                return rows
    return []


def _manifest_candidate_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in (
        "candidates",
        "top_candidates",
        "low_fill_reward_maker_watchlist",
        "low_fill_reward_candidates",
        "watchlist",
    ):
        value = payload.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            rows.extend(dict(row) for row in value if isinstance(row, Mapping))
    buckets = payload.get("buckets")
    if isinstance(buckets, Mapping):
        for value in buckets.values():
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                rows.extend(dict(row) for row in value if isinstance(row, Mapping))
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        deduped.setdefault(_candidate_slug(row), row)
    return list(deduped.values())


def _reward_terms(row: Mapping[str, Any]) -> dict[str, Any]:
    terms: dict[str, Any] = {}

    def add(source: Mapping[str, Any]) -> None:
        for key in (
            "reward",
            "reward_amount",
            "umaReward",
            "rewardsMinSize",
            "rewardsMaxSpread",
            "clobRewards",
            "rewards",
        ):
            value = source.get(key)
            if value not in (None, "", [], {}):
                terms[key] = value

    raw = row.get("reward_evidence")
    if isinstance(raw, Mapping):
        add(raw)
        nested = raw.get("reward_evidence")
        if isinstance(nested, Mapping):
            add(nested)
    add(row)
    rewards = terms.get("clobRewards") or terms.get("rewards")
    for reward in _as_sequence(rewards):
        if not isinstance(reward, Mapping):
            continue
        if "rewardsMinSize" not in terms:
            min_size = reward.get("min_size") or reward.get("minSize")
            if min_size not in (None, ""):
                terms["rewardsMinSize"] = min_size
        if "rewardsMaxSpread" not in terms:
            max_spread = reward.get("max_spread") or reward.get("maxSpread")
            if max_spread not in (None, ""):
                terms["rewardsMaxSpread"] = max_spread
        if "umaReward" not in terms:
            amount = reward.get("reward") or reward.get("amount")
            if amount not in (None, ""):
                terms["umaReward"] = amount
    return terms


def _reward_amount(terms: Mapping[str, Any]) -> float | None:
    return _parse_float(terms.get("umaReward") or terms.get("reward") or terms.get("reward_amount"))


def _normalize_reward_spread(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    return value / 100.0 if value > 1.0 else value


def _time_in_band_ratio(row: Mapping[str, Any]) -> tuple[float | None, str]:
    direct = _parse_float(row.get("time_in_band_ratio"))
    if direct is not None:
        return max(0.0, min(1.0, direct)), "reported_time_in_band_ratio"
    measurement = row.get("time_in_band")
    if isinstance(measurement, Mapping):
        observed = _parse_float(measurement.get("observed_fraction"))
        if observed is not None:
            return max(0.0, min(1.0, observed)), str(
                measurement.get("status") or "reported_time_in_band_proxy"
            )
        in_band = measurement.get("in_band")
        if isinstance(in_band, bool):
            return (1.0 if in_band else 0.0), str(
                measurement.get("status") or "single_snapshot_time_in_band_proxy"
            )
    observed_bool = row.get("time_in_band_observed")
    if isinstance(observed_bool, bool):
        return (1.0 if observed_bool else 0.0), "single_snapshot_time_in_band_proxy"
    return None, "unknown_missing_time_in_band_evidence"


def _expected_exit_loss_proxy(row: Mapping[str, Any]) -> tuple[float | None, str, str]:
    loss = row.get("exit_loss_proxy")
    if isinstance(loss, Mapping):
        value = _parse_float(
            loss.get("expected_exit_loss_proxy") or loss.get("half_spread_over_mid")
        )
        status = str(loss.get("status") or "reported_exit_loss_proxy")
        basis = str(loss.get("basis") or "reported_exit_loss_proxy")
        return value, status, basis
    slippage = row.get("exit_slippage_proxy")
    if isinstance(slippage, Mapping):
        value = _parse_float(slippage.get("half_spread_over_mid"))
        return value, "reported_exit_slippage_proxy", "half_spread_over_mid"
    return None, "unknown_missing_expected_exit_loss_proxy", "requires_exit_loss_proxy"


def _relative_tick_cost(yes_mid: float | None, tick_size: float) -> tuple[float | None, str]:
    if yes_mid is None or yes_mid <= 0:
        return None, "unknown"
    relative = tick_size / yes_mid
    if relative >= HIGH_RELATIVE_TICK_COST:
        return relative, "high"
    if relative >= MEDIUM_RELATIVE_TICK_COST:
        return relative, "medium"
    return relative, "normal"


def _would_fill_probability(row: Mapping[str, Any]) -> float | None:
    direct = _parse_float(row.get("would_have_filled_probability"))
    if direct is not None:
        return max(0.0, min(1.0, direct))
    would_fill = row.get("would_have_filled")
    if isinstance(would_fill, Mapping):
        nested = _parse_float(would_fill.get("probability"))
        if nested is not None:
            return max(0.0, min(1.0, nested))
    return None


def _would_fill_status(row: Mapping[str, Any]) -> str:
    status = row.get("would_have_filled_status")
    if isinstance(status, str) and status:
        return status
    would_fill = row.get("would_have_filled")
    if isinstance(would_fill, Mapping) and would_fill.get("status"):
        return str(would_fill["status"])
    if isinstance(would_fill, str) and would_fill:
        return would_fill
    return "unknown_missing_would_fill_evidence"


def _has_real_would_fill_evidence(row: Mapping[str, Any]) -> bool:
    status = _would_fill_status(row)
    if status.startswith("unknown"):
        return False
    would_fill = row.get("would_have_filled")
    basis = would_fill.get("basis") if isinstance(would_fill, Mapping) else None
    source_text = " ".join(
        str(part).lower()
        for part in (
            status,
            basis,
            row.get("snapshot_source"),
            row.get("trade_snapshot_status"),
            row.get("source_fill_evidence_type"),
        )
        if part not in (None, "")
    )
    return any(marker in source_text for marker in REAL_FILL_EVIDENCE_MARKERS)


def _would_fill_evidence_status(row: Mapping[str, Any]) -> str:
    status = _would_fill_status(row)
    if _has_real_would_fill_evidence(row):
        return "real_l2_or_trade_evidence"
    if status.startswith("unknown"):
        return "missing_l2_or_trade_evidence"
    return "proxy_not_real_l2_or_trade_evidence"


def _raw_score_proxy(
    row: Mapping[str, Any],
    terms: Mapping[str, Any],
    *,
    time_in_band_ratio: float | None,
) -> tuple[float | None, str]:
    reported = _parse_float(
        row.get("reward_score_proxy") or row.get("estimated_reward_score_proxy")
    )
    if reported is not None:
        return reported, "reported_reward_score_proxy"
    if time_in_band_ratio is None:
        return None, "missing_time_in_band_ratio_proxy"
    spread = _parse_float(row.get("spread") or row.get("latest_spread"))
    max_spread = _normalize_reward_spread(
        _parse_float(
            row.get("reward_max_spread")
            or row.get("reward_max_spread_decimal")
            or terms.get("rewardsMaxSpread")
        )
    )
    if spread is None or max_spread is None or max_spread <= 0:
        return None, "missing_reward_spread_terms_for_score_proxy"
    min_size = _parse_float(row.get("reward_min_size") or terms.get("rewardsMinSize")) or 1.0
    tightness = max(0.0, 1.0 - min(1.0, spread / max_spread))
    return time_in_band_ratio * tightness * min_size, "top_of_book_time_in_band_score_proxy"


def _source_manifest_path(observation_path: Path, payload: Mapping[str, Any]) -> Path | None:
    value = payload.get("source_manifest")
    if isinstance(value, str) and value:
        path = Path(value)
        return path if path.exists() else None
    sibling = observation_path.with_name(observation_path.name.replace("observation", "manifest"))
    return sibling if sibling.exists() else None


def _manifest_terms_by_slug(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = _load_json_object(path)
    terms: dict[str, dict[str, Any]] = {}
    for row in _manifest_candidate_rows(payload):
        terms[_candidate_slug(row)] = _reward_terms(row)
    return terms


def _merge_terms(
    row: Mapping[str, Any], manifest_terms: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    merged = dict(manifest_terms.get(_candidate_slug(row), {}))
    merged.update(_reward_terms(row))
    return merged


def _base_candidate(
    row: Mapping[str, Any],
    terms: Mapping[str, Any],
    *,
    tick_size: float,
    exclude_high_tick_cost: bool,
) -> dict[str, Any]:
    yes_mid = _parse_float(row.get("yes_mid") or row.get("latest_yes_mid"))
    time_ratio, time_status = _time_in_band_ratio(row)
    raw_score, score_basis = _raw_score_proxy(row, terms, time_in_band_ratio=time_ratio)
    fill_probability = _would_fill_probability(row)
    fill_evidence_status = _would_fill_evidence_status(row)
    real_fill_evidence = fill_evidence_status == "real_l2_or_trade_evidence"
    exit_loss, exit_status, exit_basis = _expected_exit_loss_proxy(row)
    relative_tick, tick_bucket = _relative_tick_cost(yes_mid, tick_size)
    excluded = exclude_high_tick_cost and tick_bucket == "high"
    reward_amount = _reward_amount(terms)
    missing: list[str] = []
    if excluded:
        missing.append("excluded_high_relative_tick_cost")
    if not real_fill_evidence:
        missing.append("missing_l2_or_trade_fill_evidence")
    if fill_probability is None:
        missing.append("missing_would_fill_probability_proxy")
    if raw_score is None:
        missing.append(score_basis)
    if exit_loss is None:
        missing.append("missing_expected_exit_loss_proxy")
    if reward_amount is None:
        missing.append("missing_reward_amount")
    return {
        "slug": _candidate_slug(row),
        "question": row.get("question"),
        "yes_mid": _round(yes_mid),
        "spread": _round(_parse_float(row.get("spread") or row.get("latest_spread"))),
        "reward_amount": _round(reward_amount),
        "reward_min_size": _round(
            _parse_float(row.get("reward_min_size") or terms.get("rewardsMinSize"))
        ),
        "reward_max_spread_decimal": _round(
            _normalize_reward_spread(
                _parse_float(
                    row.get("reward_max_spread")
                    or row.get("reward_max_spread_decimal")
                    or terms.get("rewardsMaxSpread")
                )
            )
        ),
        "would_fill_evidence_status": fill_evidence_status,
        "would_have_filled_status": _would_fill_status(row),
        "would_fill_probability_proxy": _round(fill_probability),
        "time_in_band_ratio_proxy": _round(time_ratio),
        "time_in_band_status": time_status,
        "reward_score_proxy": _round(raw_score),
        "reward_score_proxy_basis": score_basis,
        "reward_score_share_proxy": None,
        "reward_score_share_proxy_basis": None,
        "expected_exit_loss_proxy": _round(exit_loss),
        "expected_exit_loss_proxy_status": exit_status,
        "expected_exit_loss_proxy_basis": exit_basis,
        "relative_tick_cost": _round(relative_tick),
        "relative_tick_cost_bucket": tick_bucket,
        "excluded_from_ev_rank": excluded,
        "excluded_from_ev_rank_reason": "high_relative_tick_cost" if excluded else None,
        "expected_reward_ev_minus_loss": None,
        "expected_reward_ev_minus_loss_status": "pending_score_share_proxy",
        "not_computable_reason": None,
        "not_computable_reasons": missing,
        "candidate_ev_rank": None,
        "evidence_sources": row.get("evidence_sources")
        if isinstance(row.get("evidence_sources"), list)
        else [],
        **{field: False for field in SAFETY_FALSE_FIELDS},
    }


def build_report(
    observation_path: Path,
    *,
    manifest_path: Path | None = None,
    tick_size: float = DEFAULT_TICK_SIZE,
    exclude_high_tick_cost: bool = True,
) -> dict[str, Any]:
    payload = _load_json_object(observation_path)
    _safe_payload(observation_path, payload)
    rows = _candidate_rows(payload)
    manifest = manifest_path or _source_manifest_path(observation_path, payload)
    manifest_terms = _manifest_terms_by_slug(manifest)
    candidates = [
        _base_candidate(
            row,
            _merge_terms(row, manifest_terms),
            tick_size=tick_size,
            exclude_high_tick_cost=exclude_high_tick_cost,
        )
        for row in rows
    ]
    denominator = sum(
        float(row["reward_score_proxy"] or 0.0)
        for row in candidates
        if not row["excluded_from_ev_rank"]
    )
    for row in candidates:
        if row["excluded_from_ev_rank"]:
            row["reward_score_share_proxy_basis"] = "excluded_high_relative_tick_cost"
        elif denominator > 0 and row["reward_score_proxy"] is not None:
            row["reward_score_share_proxy"] = _round(float(row["reward_score_proxy"]) / denominator)
            row["reward_score_share_proxy_basis"] = "share_of_non_excluded_score_proxy"
        else:
            row["not_computable_reasons"].append("missing_reward_score_share_proxy")
            row["reward_score_share_proxy_basis"] = "missing_reward_score_denominator_proxy"

        if (
            row["reward_score_share_proxy"] is None
            and "missing_reward_score_share_proxy" not in row["not_computable_reasons"]
        ):
            row["not_computable_reasons"].append("missing_reward_score_share_proxy")
        if row["not_computable_reasons"]:
            primary = row["not_computable_reasons"][0]
            row["not_computable_reason"] = primary
            row["expected_reward_ev_minus_loss_status"] = f"not_computable_{primary}"
            continue

        reward_amount = float(row["reward_amount"])
        fill_probability = float(row["would_fill_probability_proxy"])
        time_ratio = float(row["time_in_band_ratio_proxy"])
        reward_share = float(row["reward_score_share_proxy"])
        exit_loss = float(row["expected_exit_loss_proxy"])
        row["expected_reward_ev_minus_loss"] = _round(
            reward_amount * reward_share * time_ratio - fill_probability * exit_loss
        )
        row["expected_reward_ev_minus_loss_status"] = (
            "computable_shadow_ev_evidence_not_profit_claim"
        )

    computable = [
        row
        for row in candidates
        if row["expected_reward_ev_minus_loss_status"]
        == "computable_shadow_ev_evidence_not_profit_claim"
    ]
    computable.sort(
        key=lambda row: float(row["expected_reward_ev_minus_loss"] or 0.0), reverse=True
    )
    for rank, row in enumerate(computable, start=1):
        row["candidate_ev_rank"] = rank

    summary = {
        "candidate_count": len(candidates),
        "reward_ev_computable_count": len(computable),
        "would_have_filled_known_count": sum(
            1
            for row in candidates
            if row["would_fill_evidence_status"] == "real_l2_or_trade_evidence"
        ),
        "high_relative_tick_cost_count": sum(
            1 for row in candidates if row["relative_tick_cost_bucket"] == "high"
        ),
        "excluded_high_tick_cost_count": sum(
            1 for row in candidates if row["excluded_from_ev_rank"]
        ),
        "missing_l2_or_trade_evidence_count": sum(
            1
            for row in candidates
            if row["would_fill_evidence_status"] != "real_l2_or_trade_evidence"
        ),
        "reward_score_share_proxy_count": sum(
            1 for row in candidates if row["reward_score_share_proxy"] is not None
        ),
        "expected_exit_loss_proxy_count": sum(
            1 for row in candidates if row["expected_exit_loss_proxy"] is not None
        ),
    }
    classification = "blocked" if not candidates else "diagnostic_only"
    if summary["reward_ev_computable_count"] > 0:
        classification = "adopted"
    return {
        "schema_version": 1,
        "generated_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
        "mode": SAFETY_MODE,
        "classification": classification,
        "source_observation": str(observation_path),
        "source_manifest": str(manifest) if manifest else None,
        "tick_size": tick_size,
        "exclude_high_tick_cost": exclude_high_tick_cost,
        "safety": safety_object(),
        "no_profit_claim": True,
        "profit_verdict": "no_profit_claim_shadow_diagnostic_only",
        "summary": summary,
        "candidates": candidates,
    }


def _markdown_report(report: Mapping[str, Any]) -> str:
    safety = _as_mapping(report.get("safety"))
    summary = _as_mapping(report.get("summary"))
    lines = [
        "# Low-fill Reward EV Evidence",
        "",
        f"- mode: {report.get('mode')}",
        f"- classification: {report.get('classification')}",
        f"- source_observation: {report.get('source_observation')}",
        f"- source_manifest: {report.get('source_manifest')}",
        f"- no_profit_claim: {report.get('no_profit_claim')}",
        "",
        "Safety fields:",
        f"- orders_submitted={str(safety.get('orders_submitted')).lower()}",
        f"- orders_signed={str(safety.get('orders_signed')).lower()}",
        f"- orders_cancelled={str(safety.get('orders_cancelled')).lower()}",
        f"- credentials_required={str(safety.get('credentials_required')).lower()}",
        f"- live_trading_worker_started={str(safety.get('live_trading_worker_started')).lower()}",
        f"- worker_trading_started={str(safety.get('worker_trading_started')).lower()}",
        "",
        "## Summary",
        "",
        f"- reward_ev_computable_count: {summary.get('reward_ev_computable_count')}",
        f"- would_have_filled_known_count: {summary.get('would_have_filled_known_count')}",
        f"- high_relative_tick_cost_count: {summary.get('high_relative_tick_cost_count')}",
        f"- excluded_high_tick_cost_count: {summary.get('excluded_high_tick_cost_count')}",
        f"- missing_l2_or_trade_evidence_count: {summary.get('missing_l2_or_trade_evidence_count')}",
        "",
        "No profitability claim is made unless real L2/trade fill evidence is present.",
        "",
        "## Candidates",
        "",
        "| slug | tick_bucket | fill_evidence | time_ratio | reward_share | exit_loss | ev_status | reason |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in _as_sequence(report.get("candidates")):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| "
            f"{row.get('slug')} | "
            f"{row.get('relative_tick_cost_bucket')} | "
            f"{row.get('would_fill_evidence_status')} | "
            f"{row.get('time_in_band_ratio_proxy')} | "
            f"{row.get('reward_score_share_proxy')} | "
            f"{row.get('expected_exit_loss_proxy')} | "
            f"{row.get('expected_reward_ev_minus_loss_status')} | "
            f"{row.get('not_computable_reason')} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], output_dir: Path, timestamp: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"low_fill_reward_ev_evidence_{timestamp}.json"
    md_path = output_dir / f"low_fill_reward_ev_evidence_{timestamp}.md"
    report["output_files"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    return report["output_files"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build shadow/backtest-only low-fill reward EV evidence report."
    )
    parser.add_argument("--observation", type=Path, default=None)
    parser.add_argument("--observation-glob", default=DEFAULT_OBSERVATION_GLOB)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--tick-size", type=float, default=DEFAULT_TICK_SIZE)
    parser.add_argument(
        "--include-high-tick-cost",
        action="store_true",
        help="Do not exclude high relative tick-cost candidates from EV ranking.",
    )
    args = parser.parse_args(argv)
    if args.tick_size <= 0:
        parser.error("--tick-size must be positive")
    observation = args.observation or _latest_path(args.observation_glob)
    timestamp = args.timestamp or _utc_stamp()
    report = build_report(
        observation,
        manifest_path=args.manifest,
        tick_size=args.tick_size,
        exclude_high_tick_cost=not args.include_high_tick_cost,
    )
    output_files = write_outputs(report, args.output_dir, timestamp)
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "source_observation": report["source_observation"],
                "source_manifest": report["source_manifest"],
                "summary": report["summary"],
                "safety": report["safety"],
                "output_files": output_files,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
