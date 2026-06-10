from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.low_fill_reward_shadow_observation import (
    AGGREGATION_MODE,
    _observation,
    build_aggregation_report,
    build_report,
    write_outputs,
)


def test_nested_reward_evidence_sets_time_in_band_and_safety_flags() -> None:
    row = {
        "slug": "reward-market",
        "yes_mid": 0.02,
        "spread": 0.001,
        "liquidity": 10000,
        "reward_evidence": {"reward_evidence": {"rewardsMaxSpread": 2.5}},
    }

    obs = _observation(row)

    assert obs["reward_max_spread"] == 2.5
    assert obs["time_in_band_observed"] is True
    assert obs["would_have_filled"] == "unknown_requires_l2_or_shadow_quote_log"
    assert obs["would_have_filled_status"] == "unknown_requires_l2_or_shadow_quote_log"
    assert obs["would_have_filled_probability"] is None
    assert obs["reward_ev_status"] == "reward_ev_not_yet_computable"
    assert obs["expected_reward_ev_minus_loss"] is None
    assert "would_have_filled_probability" in obs["missing_reward_ev_inputs"]
    assert "conservative_would_fill_evidence" in obs["missing_reward_ev_inputs"]
    assert "reward_denominator_or_score_share" in obs["missing_reward_ev_inputs"]
    assert "time_in_band" in obs["missing_reward_ev_inputs"]
    assert obs["accidental_fill_risk"] == "normal_requires_l2_fill_model"
    assert obs["exit_risk"] == "liquidity_proxy_ok_needs_l2"
    assert obs["orders_submitted"] is False
    assert obs["orders_signed"] is False
    assert obs["orders_cancelled"] is False
    assert obs["credentials_required"] is False
    assert obs["live_trading_worker_started"] is False
    assert obs["worker_trading_started"] is False


def test_ultra_low_tail_observation_flags_relative_tick_and_exit_risk() -> None:
    row = {
        "slug": "tail-market",
        "yes_mid": 0.0035,
        "spread": 0.001,
        "liquidity": 1_000_000,
        "reward_evidence": {"reward_evidence": {"rewardsMaxSpread": 2.5}},
    }

    obs = _observation(row)

    assert obs["time_in_band_observed"] is True
    assert obs["accidental_fill_risk"] == "high_relative_tick_cost"
    assert obs["exit_risk"] == "tail_exit_slippage_risk"


def test_missing_reward_band_terms_are_unknown_not_false() -> None:
    row = {
        "slug": "reward-market",
        "features": {
            "avg_spread": 0.001,
            "fee_reward_category": "explicit_gamma_reward_terms",
            "has_explicit_reward_evidence": True,
            "liquidity": 1_000_000,
        },
    }

    obs = _observation(row)

    assert obs["has_reward_evidence"] is True
    assert obs["spread"] == 0.001
    assert obs["time_in_band_observed"] is None
    assert obs["time_in_band"]["status"] == "unknown_missing_reward_spread_or_quote"


def _write_snapshot(path: Path, *, generated_at: str, observations: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at_utc": generated_at,
                "mode": "shadow_observation_only_no_live_orders",
                "safety": {
                    "live_trading": False,
                    "orders_submitted": False,
                    "orders_signed": False,
                    "orders_cancelled": False,
                    "credentials_required": False,
                    "worker_trading_started": False,
                },
                "source_manifest": "fixture.json",
                "candidate_count": len(observations),
                "observations": observations,
            }
        ),
        encoding="utf-8",
    )


def test_multi_snapshot_aggregation_accepts_current_observation_artifacts(tmp_path: Path) -> None:
    first = tmp_path / "low_fill_reward_shadow_observation_1.json"
    second = tmp_path / "low_fill_reward_shadow_observation_2.json"
    base_observation = {
        "slug": "reward-market",
        "question": "Will reward happen?",
        "yes_mid": 0.02,
        "spread": 0.001,
        "reward_max_spread": 0.002,
        "has_reward_evidence": True,
        "would_have_filled": "unknown_requires_l2_or_shadow_quote_log",
        "accidental_fill_risk": "normal_requires_l2_fill_model",
        "exit_risk": "liquidity_proxy_ok_needs_l2",
    }
    _write_snapshot(
        first,
        generated_at="2026-05-11T00:00:00Z",
        observations=[base_observation | {"time_in_band_observed": True}],
    )
    _write_snapshot(
        second,
        generated_at="2026-05-11T00:05:00Z",
        observations=[base_observation | {"time_in_band_observed": False}],
    )

    report = build_aggregation_report([first, second], limit=10)
    row = report["observations"][0]

    assert report["mode"] == AGGREGATION_MODE
    assert report["classification"] == "diagnostic_only"
    assert report["safety"]["orders_submitted"] is False
    assert report["safety"]["orders_signed"] is False
    assert report["safety"]["orders_cancelled"] is False
    assert report["safety"]["credentials_required"] is False
    assert report["safety"]["live_trading_worker_started"] is False
    assert report["orders_submitted"] is False
    assert report["orders_signed"] is False
    assert report["orders_cancelled"] is False
    assert report["credentials_required"] is False
    assert report["live_trading_worker_started"] is False
    assert report["worker_trading_started"] is False
    assert row["time_in_band"] == {
        "status": "measured_multi_snapshot_proxy_not_continuous",
        "sample_count": 2,
        "measured_snapshot_count": 2,
        "in_band_snapshot_count": 1,
        "observed_fraction": 0.5,
        "observed_window_seconds": 300,
        "approx_in_band_seconds": 150.0,
        "basis": "discrete_shadow_snapshots_not_continuous_l2_replay",
    }
    assert row["would_have_filled"]["status"] == "unknown_no_l2_or_shadow_quote_fill_log"
    assert row["would_have_filled_status"] == "unknown_no_l2_or_shadow_quote_fill_log"
    assert row["would_have_filled_probability"] is None
    assert row["reward_ev_status"] == "reward_ev_not_yet_computable"
    assert row["accidental_fill_risk"]["status"] == "measured_from_shadow_snapshots"
    assert row["exit_risk"]["status"] == "measured_from_shadow_snapshots"
    assert row["fill_evidence"]["status"] == "unknown_no_would_fill_evidence"
    assert row["fill_evidence"]["unknown_would_fill_snapshot_count"] == 2
    assert row["orders_submitted"] is False
    assert row["orders_signed"] is False
    assert row["orders_cancelled"] is False
    assert row["credentials_required"] is False
    assert row["live_trading_worker_started"] is False
    assert row["worker_trading_started"] is False


def test_single_snapshot_aggregation_keeps_time_and_fill_unknown(tmp_path: Path) -> None:
    snapshot = tmp_path / "low_fill_reward_shadow_observation_1.json"
    _write_snapshot(
        snapshot,
        generated_at="2026-05-11T00:00:00Z",
        observations=[
            {
                "slug": "reward-market",
                "time_in_band_observed": True,
                "spread": 0.001,
                "reward_max_spread": 0.002,
                "has_reward_evidence": True,
                "would_have_filled": "unknown_requires_l2_or_shadow_quote_log",
                "accidental_fill_risk": "unknown",
                "exit_risk": "unknown",
            }
        ],
    )

    report = build_aggregation_report([snapshot], limit=10)
    row = report["observations"][0]

    assert row["time_in_band"]["status"] == "unknown_single_measured_snapshot_not_duration"
    assert row["would_have_filled"]["status"] == "unknown_no_l2_or_shadow_quote_fill_log"
    assert row["accidental_fill_risk"]["status"] == "unknown_no_accidental_fill_risk_measurements"
    assert row["exit_risk"]["status"] == "unknown_no_exit_risk_measurements"


def test_multi_snapshot_aggregation_fails_closed_on_ambiguous_input(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe.json"
    unsafe.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-05-11T00:00:00Z",
                "safety": {
                    "live_trading": False,
                    "orders_submitted": True,
                    "orders_signed": False,
                    "orders_cancelled": False,
                    "credentials_required": False,
                },
                "observations": [{"slug": "reward-market"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="orders_submitted=True"):
        build_aggregation_report([unsafe], limit=10)


def test_aggregation_markdown_writes_explicit_safety_fields(tmp_path: Path) -> None:
    snapshot = tmp_path / "low_fill_reward_shadow_observation_1.json"
    _write_snapshot(
        snapshot,
        generated_at="2026-05-11T00:00:00Z",
        observations=[
            {
                "slug": "reward-market",
                "time_in_band_observed": None,
                "would_have_filled": "unknown_requires_l2_or_shadow_quote_log",
                "accidental_fill_risk": "unknown",
                "exit_risk": "unknown",
            }
        ],
    )
    report = build_aggregation_report([snapshot], limit=10)

    output_files = write_outputs(report, tmp_path / "out", "20260511T000000Z")
    md = Path(output_files["markdown"]).read_text(encoding="utf-8")

    assert "- orders_submitted=false" in md
    assert "- orders_signed=false" in md
    assert "- orders_cancelled=false" in md
    assert "- credentials_required=false" in md
    assert "- live_trading_worker_started=false" in md
    assert "No reward or profit claim" in md


def test_aggregation_keeps_proxy_fill_evidence_diagnostic_without_strict_ev_inputs(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "low_fill_reward_shadow_observation_1.json"
    _write_snapshot(
        snapshot,
        generated_at="2026-05-11T00:00:00Z",
        observations=[
            {
                "slug": "reward-market",
                "time_in_band_observed": True,
                "spread": 0.001,
                "yes_mid": 0.02,
                "reward_max_spread": 0.002,
                "reward_evidence": {"umaReward": "5"},
                "has_reward_evidence": True,
                "would_have_filled": "unknown_requires_l2_or_shadow_quote_log",
                "accidental_fill_risk": "normal_requires_l2_fill_model",
                "exit_risk": "liquidity_proxy_ok_needs_l2",
            }
        ],
    )
    evidence = tmp_path / "shadow_evidence.jsonl"
    evidence.write_text(
        json.dumps(
            {
                "slug": "reward-market",
                "generated_at_utc": "2026-05-11T00:00:30Z",
                "reward_evidence": {"umaReward": "5"},
                "shadow_quote": {"would_have_filled": True},
                "exit_loss_proxy": {"expected_exit_loss_proxy": 0.25},
                "safety": {
                    "live_trading": False,
                    "orders_submitted": False,
                    "orders_signed": False,
                    "orders_cancelled": False,
                    "credentials_required": False,
                    "worker_trading_started": False,
                    "live_trading_worker_started": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_aggregation_report([snapshot], limit=10, evidence_paths=[evidence])
    row = report["observations"][0]

    assert report["summary"]["evidence_file_count"] == 1
    assert row["would_have_filled_status"] == "measured_from_shadow_quote_log"
    assert row["would_have_filled_probability"] == 1.0
    assert row["fill_evidence"]["status"] == "diagnostic_proxy_would_fill_only"
    assert row["reward_ev_status"] == "reward_ev_not_yet_computable"
    assert row["expected_reward_ev_minus_loss"] is None
    assert "conservative_would_fill_evidence" in row["missing_reward_ev_inputs"]
    assert "reward_denominator_or_score_share" in row["missing_reward_ev_inputs"]
    assert "time_in_band" in row["missing_reward_ev_inputs"]
    assert any(source["path"] == str(evidence) for source in row["evidence_sources"])


def test_aggregation_computes_ev_only_with_conservative_fill_time_share_and_exit(
    tmp_path: Path,
) -> None:
    first = tmp_path / "low_fill_reward_shadow_observation_1.json"
    second = tmp_path / "low_fill_reward_shadow_observation_2.json"
    base_observation = {
        "slug": "reward-market",
        "time_in_band_observed": True,
        "spread": 0.001,
        "yes_mid": 0.02,
        "reward_max_spread": 0.002,
        "has_reward_evidence": True,
        "would_have_filled": "unknown_requires_l2_or_shadow_quote_log",
        "accidental_fill_risk": "normal_requires_l2_fill_model",
        "exit_risk": "liquidity_proxy_ok_needs_l2",
    }
    _write_snapshot(first, generated_at="2026-05-11T00:00:00Z", observations=[base_observation])
    _write_snapshot(second, generated_at="2026-05-11T00:05:00Z", observations=[base_observation])
    evidence = tmp_path / "shadow_evidence.jsonl"
    evidence.write_text(
        json.dumps(
            {
                "slug": "reward-market",
                "generated_at_utc": "2026-05-11T00:05:30Z",
                "reward_evidence": {"umaReward": "5"},
                "reward_score_share_proxy": 0.5,
                "would_fill": {
                    "classification": "conservative",
                    "known": True,
                    "estimate": True,
                    "probability": 1.0,
                    "basis": "l2_queue_plus_trade_tape_replay",
                },
                "exit_loss_proxy": {"expected_exit_loss_proxy": 0.25},
                "safety": {
                    "live_trading": False,
                    "orders_submitted": False,
                    "orders_signed": False,
                    "orders_cancelled": False,
                    "credentials_required": False,
                    "worker_trading_started": False,
                    "live_trading_worker_started": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_aggregation_report([first, second], limit=10, evidence_paths=[evidence])
    row = report["observations"][0]

    assert report["classification"] == "adopted"
    assert row["fill_evidence"]["status"] == "conservative_would_fill_evidence_present"
    assert row["reward_ev_status"] == "computable_shadow_proxy_not_profit_claim"
    assert row["expected_reward_ev_minus_loss"] == 2.25
    assert row["missing_reward_ev_inputs"] == []


def test_manifest_observation_reads_book_provenance_and_keeps_ev_unknown(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "safety": {
                    "live_trading": False,
                    "orders_submitted": False,
                    "orders_signed": False,
                    "orders_cancelled": False,
                    "credentials_required": False,
                    "live_trading_worker_started": False,
                },
                "candidates": [
                    {
                        "slug": "reward-market",
                        "question": "Reward?",
                        "liquidity": 10_000,
                        "reward_evidence": {"rewardsMaxSpread": 0.002, "umaReward": "5"},
                        "book_provenance": {
                            "complete": True,
                            "status": "complete",
                            "source_artifact_path": "scan.json",
                            "source_timestamp_utc": "2026-05-12T00:00:00Z",
                            "fail_closed_reasons": [],
                            "sides": {
                                "yes": {
                                    "side": "yes",
                                    "token_id": "yes",
                                    "mid": 0.02,
                                    "spread": 0.001,
                                    "best_bid": 0.0195,
                                    "best_ask": 0.0205,
                                },
                                "no": {
                                    "side": "no",
                                    "token_id": "no",
                                    "mid": 0.98,
                                    "spread": 0.001,
                                    "best_bid": 0.9795,
                                    "best_ask": 0.9805,
                                },
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_report(manifest, limit=1)
    row = report["observations"][0]

    assert report["classification"] == "diagnostic_only"
    assert report["summary"]["classification"] == "diagnostic_only"
    assert row["yes_mid"] == 0.02
    assert row["spread"] == 0.001
    assert row["evidence_sources"][0]["type"] == "book_provenance"
    assert row["reward_ev_status"] == "reward_ev_not_yet_computable"
    assert "would_have_filled_probability" in row["missing_reward_ev_inputs"]
    assert row["yes_token_id"] == "yes"
    assert row["no_token_id"] == "no"
    assert row["book_provenance_complete"] is True
    assert row["yes_book"]["best_bid"] == 0.0195
    assert row["no_book"]["best_ask"] == 0.9805

    output_files = write_outputs(report, tmp_path / "out", "20260512T000000Z")
    md = Path(output_files["markdown"]).read_text(encoding="utf-8")
    assert "- classification: diagnostic_only" in md
    assert "No reward, profit, or live-ready claim" in md


def test_current_fresh_manifest_shape_preserves_source_tokens_but_fails_closed(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    yes_token = "20257190540739490630509657713144742134547949967093643458458133445357169845406"
    no_token = "1770840559776249239623005379825945674336282130390798724203946923853499387834"
    manifest.write_text(
        json.dumps(
            {
                "safety": {
                    "live_trading": False,
                    "orders_submitted": False,
                    "orders_signed": False,
                    "orders_cancelled": False,
                    "credentials_required": False,
                    "live_trading_worker_started": False,
                    "worker_trading_started": False,
                },
                "candidates": [
                    {
                        "slug": "will-the-new-york-knicks-win-the-2026-nba-finals",
                        "question": "Will the New York Knicks win the 2026 NBA Finals?",
                        "source_clob_token_ids": [yes_token, no_token],
                        "clob_token_ids": [],
                        "book_provenance": {
                            "complete": False,
                            "status": "incomplete_fail_closed",
                            "source_artifact_path": "public_market_scan.json",
                            "fail_closed_reasons": [
                                "missing_canonical_yes_no_clob_token_ids",
                                "missing_no_best_ask",
                                "missing_no_best_bid",
                                "missing_no_token_id",
                            ],
                            "sides": {
                                "yes": {
                                    "side": "yes",
                                    "token_id": yes_token,
                                    "best_bid": 0.79,
                                    "best_ask": 0.791,
                                    "mid": 0.7905,
                                    "spread": 0.001,
                                    "present": True,
                                },
                                "no": {
                                    "side": "no",
                                    "token_id": None,
                                    "best_bid": None,
                                    "best_ask": None,
                                    "present": False,
                                },
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_report(manifest, limit=1)
    row = report["observations"][0]

    assert row["source_clob_token_ids"] == [yes_token, no_token]
    assert row["canonical_clob_token_ids"] == []
    assert row["yes_token_id"] is None
    assert row["no_token_id"] is None
    assert row["yes_book"]["token_id"] == yes_token
    assert row["book_provenance_status"] == "incomplete_fail_closed"
    assert row["token_provenance"]["status"] == "incomplete_fail_closed"
    assert "missing_no_best_ask" in row["token_provenance"]["fail_closed_reasons"]
    assert row["reward_ev_status"] == "reward_ev_not_yet_computable"


def test_multi_snapshot_aggregation_groups_by_slug_and_token_pair(tmp_path: Path) -> None:
    first = tmp_path / "low_fill_reward_shadow_observation_1.json"
    second = tmp_path / "low_fill_reward_shadow_observation_2.json"
    base = {
        "slug": "same-slug",
        "time_in_band_observed": True,
        "spread": 0.001,
        "reward_max_spread": 0.002,
        "has_reward_evidence": True,
        "would_have_filled": "unknown_requires_l2_or_shadow_quote_log",
    }
    _write_snapshot(
        first,
        generated_at="2026-05-11T00:00:00Z",
        observations=[
            base | {"source_clob_token_ids": ["yes-a", "no-a"]},
            base | {"source_clob_token_ids": ["yes-b", "no-b"]},
        ],
    )
    _write_snapshot(
        second,
        generated_at="2026-05-11T00:05:00Z",
        observations=[
            base | {"source_clob_token_ids": ["yes-a", "no-a"]},
            base | {"source_clob_token_ids": ["yes-b", "no-b"]},
        ],
    )

    report = build_aggregation_report([first, second], limit=10)
    keys = {row["observation_key"] for row in report["observations"]}

    assert keys == {
        "same-slug|yes=yes-a|no=no-a",
        "same-slug|yes=yes-b|no=no-b",
    }
    assert all(row["source_observation_count"] == 2 for row in report["observations"])
