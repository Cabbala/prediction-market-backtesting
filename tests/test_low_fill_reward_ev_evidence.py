from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.low_fill_reward_ev_evidence import build_report, write_outputs


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _safe_observation(rows: list[dict], *, manifest: Path | None = None) -> dict:
    return {
        "schema_version": 1,
        "generated_at_utc": "2026-05-14T00:00:00Z",
        "mode": "shadow_observation_only_no_live_orders",
        "safety": {
            "live_trading": False,
            "orders_submitted": False,
            "orders_signed": False,
            "orders_cancelled": False,
            "credentials_required": False,
            "live_trading_worker_started": False,
            "worker_trading_started": False,
        },
        "source_manifest": str(manifest) if manifest else None,
        "candidate_count": len(rows),
        "observations": rows,
    }


def _manifest(rows: list[dict]) -> dict:
    return {
        "safety": {
            "live_trading": False,
            "orders_submitted": False,
            "orders_signed": False,
            "orders_cancelled": False,
            "credentials_required": False,
            "live_trading_worker_started": False,
            "worker_trading_started": False,
        },
        "candidates": rows,
    }


def test_ev_evidence_fails_closed_without_l2_and_counts_high_tick_cost(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    observation = tmp_path / "observation.json"
    _write_json(
        manifest,
        _manifest(
            [
                {
                    "slug": "high-tail",
                    "reward_evidence": {
                        "rewardsMaxSpread": 2.5,
                        "rewardsMinSize": 100,
                        "umaReward": "5",
                    },
                },
                {
                    "slug": "medium-tail",
                    "reward_evidence": {
                        "rewardsMaxSpread": 2.5,
                        "rewardsMinSize": 100,
                        "umaReward": "5",
                    },
                },
            ]
        ),
    )
    _write_json(
        observation,
        _safe_observation(
            [
                {
                    "slug": "high-tail",
                    "yes_mid": 0.0035,
                    "spread": 0.001,
                    "time_in_band_observed": True,
                    "would_have_filled_status": "unknown_requires_l2_or_shadow_quote_log",
                    "would_have_filled_probability": None,
                    "exit_loss_proxy": {"expected_exit_loss_proxy": 0.14},
                },
                {
                    "slug": "medium-tail",
                    "yes_mid": 0.0055,
                    "spread": 0.001,
                    "time_in_band_observed": True,
                    "would_have_filled_status": "unknown_requires_l2_or_shadow_quote_log",
                    "would_have_filled_probability": None,
                    "exit_loss_proxy": {"expected_exit_loss_proxy": 0.09},
                },
            ],
            manifest=manifest,
        ),
    )

    report = build_report(observation)
    by_slug = {row["slug"]: row for row in report["candidates"]}

    assert report["classification"] == "diagnostic_only"
    assert report["summary"]["candidate_count"] == 2
    assert report["summary"]["reward_ev_computable_count"] == 0
    assert report["summary"]["would_have_filled_known_count"] == 0
    assert report["summary"]["high_relative_tick_cost_count"] == 1
    assert report["summary"]["excluded_high_tick_cost_count"] == 1
    assert report["summary"]["missing_l2_or_trade_evidence_count"] == 2
    assert by_slug["high-tail"]["relative_tick_cost_bucket"] == "high"
    assert by_slug["high-tail"]["excluded_from_ev_rank"] is True
    assert by_slug["medium-tail"]["relative_tick_cost_bucket"] == "medium"
    assert by_slug["medium-tail"]["reward_score_share_proxy"] == 1.0
    assert by_slug["medium-tail"]["would_fill_evidence_status"] == ("missing_l2_or_trade_evidence")
    assert by_slug["medium-tail"]["expected_reward_ev_minus_loss_status"] == (
        "not_computable_missing_l2_or_trade_fill_evidence"
    )
    assert "missing_l2_or_trade_fill_evidence" in by_slug["medium-tail"]["not_computable_reasons"]
    assert by_slug["medium-tail"]["ev_readiness"]["status"] == "fail_closed_not_computable"
    assert (
        "real_l2_or_trade_would_fill_evidence"
        in by_slug["medium-tail"]["ev_readiness"]["missing_inputs"]
    )
    assert report["profit_verdict"] == "profitable_edge_not_established_fail_closed"
    assert report["safety"]["orders_submitted"] is False


def test_ev_evidence_computes_only_with_real_fill_evidence(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    observation = tmp_path / "observation.json"
    _write_json(
        manifest,
        _manifest(
            [
                {
                    "slug": "normal-market",
                    "reward_evidence": {
                        "rewardsMaxSpread": 2.5,
                        "rewardsMinSize": 10,
                        "umaReward": "5",
                    },
                }
            ]
        ),
    )
    _write_json(
        observation,
        _safe_observation(
            [
                {
                    "slug": "normal-market",
                    "yes_mid": 0.02,
                    "spread": 0.001,
                    "time_in_band": {
                        "status": "measured_multi_snapshot_proxy_not_continuous",
                        "observed_fraction": 0.5,
                    },
                    "would_have_filled_status": "measured_from_l2_trade_tape",
                    "would_have_filled": {
                        "status": "measured_from_l2_trade_tape",
                        "probability": 0.2,
                        "basis": "l2_queue_plus_trade_tape_replay",
                    },
                    "exit_loss_proxy": {"expected_exit_loss_proxy": 0.25},
                }
            ],
            manifest=manifest,
        ),
    )

    report = build_report(observation)
    row = report["candidates"][0]

    assert report["classification"] == "adopted"
    assert report["summary"]["reward_ev_computable_count"] == 1
    assert report["summary"]["would_have_filled_known_count"] == 1
    assert row["candidate_ev_rank"] == 1
    assert row["would_fill_evidence_status"] == "real_l2_or_trade_evidence"
    assert row["time_in_band_ratio_proxy"] == 0.5
    assert row["reward_score_share_proxy"] == 1.0
    assert row["expected_reward_ev_minus_loss_status"] == (
        "computable_shadow_ev_evidence_not_profit_claim"
    )
    assert row["expected_reward_ev_minus_loss"] == 2.45
    assert row["ev_readiness"]["base_case_ev"] == 2.45
    assert row["ev_readiness"]["worst_case_ev"] == 2.4
    assert row["ev_readiness"]["best_case_ev"] == 2.5
    assert row["ev_readiness"]["profitable_edge"] is True
    assert report["summary"]["strict_profitable_edge_count"] == 1
    assert row["not_computable_reasons"] == []


def test_ev_evidence_rejects_unsafe_observation(tmp_path: Path) -> None:
    observation = tmp_path / "observation.json"
    _write_json(
        observation,
        {
            "safety": {
                "live_trading": False,
                "orders_submitted": True,
                "orders_signed": False,
                "orders_cancelled": False,
                "credentials_required": False,
                "live_trading_worker_started": False,
                "worker_trading_started": False,
            },
            "observations": [{"slug": "unsafe"}],
        },
    )

    with pytest.raises(ValueError, match="orders_submitted=True"):
        build_report(observation)


def test_ev_evidence_markdown_writes_required_safety_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    observation = tmp_path / "observation.json"
    _write_json(
        manifest,
        _manifest(
            [
                {
                    "slug": "medium-tail",
                    "reward_evidence": {
                        "rewardsMaxSpread": 2.5,
                        "rewardsMinSize": 100,
                        "umaReward": "5",
                    },
                }
            ]
        ),
    )
    _write_json(
        observation,
        _safe_observation(
            [
                {
                    "slug": "medium-tail",
                    "yes_mid": 0.0055,
                    "spread": 0.001,
                    "time_in_band_observed": True,
                    "would_have_filled_status": "unknown_requires_l2_or_shadow_quote_log",
                    "exit_loss_proxy": {"expected_exit_loss_proxy": 0.09},
                }
            ],
            manifest=manifest,
        ),
    )
    report = build_report(observation)

    outputs = write_outputs(report, tmp_path / "reports", "20260514T000000Z")
    md = Path(outputs["markdown"]).read_text(encoding="utf-8")

    assert "- orders_submitted=false" in md
    assert "- orders_signed=false" in md
    assert "- orders_cancelled=false" in md
    assert "- credentials_required=false" in md
    assert "- live_trading_worker_started=false" in md
    assert "- worker_trading_started=false" in md
    assert "No profitability claim" in md
