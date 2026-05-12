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
    assert obs["reward_ev_status"] == "not_computable_missing_inputs"
    assert obs["expected_reward_ev_minus_loss"] is None
    assert "would_have_filled_probability" in obs["missing_reward_ev_inputs"]
    assert obs["accidental_fill_risk"] == "normal_requires_l2_fill_model"
    assert obs["exit_risk"] == "liquidity_proxy_ok_needs_l2"
    assert obs["orders_submitted"] is False
    assert obs["orders_signed"] is False
    assert obs["orders_cancelled"] is False
    assert obs["credentials_required"] is False
    assert obs["live_trading_worker_started"] is False


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
    assert row["reward_ev_status"] == "not_computable_missing_inputs"
    assert row["accidental_fill_risk"]["status"] == "measured_from_shadow_snapshots"
    assert row["exit_risk"]["status"] == "measured_from_shadow_snapshots"
    assert row["orders_submitted"] is False
    assert row["orders_signed"] is False
    assert row["orders_cancelled"] is False
    assert row["credentials_required"] is False
    assert row["live_trading_worker_started"] is False


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


def test_aggregation_uses_shadow_evidence_for_fill_probability_and_reward_ev(
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
    assert row["reward_ev_status"] == "computable_shadow_proxy_not_profit_claim"
    assert row["expected_reward_ev_minus_loss"] == 4.75
    assert row["missing_reward_ev_inputs"] == []
    assert any(source["path"] == str(evidence) for source in row["evidence_sources"])


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

    assert row["yes_mid"] == 0.02
    assert row["spread"] == 0.001
    assert row["evidence_sources"][0]["type"] == "book_provenance"
    assert row["reward_ev_status"] == "not_computable_missing_inputs"
    assert "would_have_filled_probability" in row["missing_reward_ev_inputs"]
