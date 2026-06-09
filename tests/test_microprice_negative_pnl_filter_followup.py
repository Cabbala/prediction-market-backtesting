from __future__ import annotations

import json
from datetime import UTC, datetime

from scripts import microprice_negative_pnl_filter_followup as followup


def _write_artifact(tmp_path, payload):
    artifact = tmp_path / "job_B_microprice_batch.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    return artifact


def _negative_attempt(params):
    return {
        "slug": "filled-loss",
        "token_index": 0,
        "source_strategy": "Microprice",
        "status": "completed",
        "params": params,
        "diagnostics": {
            "pnl": -0.28315,
            "fills": 2,
            "negative_pnl_attribution": {
                "schema_version": 1,
                "eligible": True,
                "classification": "non_positive_pnl_after_fills",
                "fail_closed": True,
                "primary_cause": "adverse_selection_markout",
                "causes": [
                    "adverse_selection_markout",
                    "spread_tick_cost_too_large",
                    "queue_fill_timing",
                    "parameter_candidate_bucket",
                ],
                "cause_counts": {
                    "adverse_selection_markout": 1,
                    "spread_tick_cost_too_large": 1,
                    "queue_fill_timing": 1,
                    "parameter_candidate_bucket": 1,
                },
                "pnl": -0.28315,
                "fills": 2,
                "strategy_order_count": 2,
                "adverse_selection_markout": {
                    "available": True,
                    "adverse_selection_or_markout_detected": True,
                    "round_trip_price_edge": -0.05,
                    "terminal_long_markout": -0.0495,
                },
                "spread_tick_cost": {
                    "spread_tick_cost_too_large": True,
                    "commission_to_notional_ratio": 0.0947,
                    "round_trip_move_exceeded_effective_spread": True,
                },
                "parameter_candidate_bucket": {
                    "slug": "filled-loss",
                    "token_index": 0,
                    "source_strategy": "Microprice",
                    "tail_bucket": "non_extreme_tail",
                    "scan_mid": 0.0125,
                    "scan_spread": 0.001,
                    "params": params,
                    "bucket_key": ("non_extreme_tail|depth=1|edge=0.0005|entry=0.55|hold=10.0"),
                },
            },
        },
    }


def test_build_filter_followup_report_blocks_negative_markout_and_spread_buckets(tmp_path):
    params = {
        "depth_levels": 1,
        "entry_imbalance": 0.55,
        "exit_imbalance": 0.5,
        "min_microprice_edge": 0.0005,
        "quote_lifetime_seconds": 10.0,
    }
    artifact = _write_artifact(
        tmp_path,
        {
            "classification": "diagnostic_only",
            "live_ready": False,
            "no_profit_claim": True,
            "attempt_count": 1,
            "completed_count": 1,
            "exact_window_status": "verified",
            "fills_orders_pnl": {
                "total_fills": 2,
                "completed_pnl_sum": -0.28315,
                "completed_positive_pnl_attempts": 0,
                "completed_negative_pnl_attempts": 1,
            },
            "attempts": [_negative_attempt(params)],
            "negative_pnl_attribution_summary": {
                "eligible": True,
                "classification": "non_positive_pnl_after_orders_or_fills",
                "cause_counts": {
                    "adverse_selection_markout": 1,
                    "spread_tick_cost_too_large": 1,
                },
            },
            "safety": {
                "orders_submitted": False,
                "orders_signed": False,
                "orders_cancelled": False,
                "credentials_required": False,
                "live_trading_worker_started": False,
                "worker_trading_started": False,
            },
        },
    )

    report = followup.build_filter_followup_report(
        artifact_path=artifact,
        command=["python", "scripts/microprice_negative_pnl_filter_followup.py"],
        generated_at=datetime(2026, 6, 6, tzinfo=UTC),
    )

    assert report["classification"] == "diagnostic_only"
    assert report["live_ready"] is False
    assert report["replay_promotion_ready"] is False
    assert report["no_profit_claim"] is True
    assert report["profit_opportunity_demonstrated"] is False
    assert report["source_metrics"]["completed_pnl_sum"] == -0.28315
    assert "zero_positive_pnl_attempts" in report["not_live_ready_reason_codes"]
    assert "adverse_selection_markout_detected" in report["not_live_ready_reason_codes"]
    assert "spread_tick_cost_too_large_detected" in report["not_live_ready_reason_codes"]
    assert all(report["safety"][field] is False for field in followup.REQUIRED_SAFETY_FIELDS)

    diagnostics = report["filter_diagnostics"]
    assert diagnostics["fail_closed"] is True
    assert diagnostics["filter_status"] == followup.PROMOTION_REJECTION
    assert diagnostics["cause_counts"]["adverse_selection_markout"] == 1
    assert diagnostics["cause_counts"]["spread_tick_cost_too_large"] == 1
    assert diagnostics["market_filters"][0]["filter_status"] == followup.PROMOTION_REJECTION
    assert diagnostics["market_filters"][0]["key"] == "filled-loss#0"
    assert (
        diagnostics["parameter_bucket_filters"][0]["filter_status"] == followup.PROMOTION_REJECTION
    )
    recommendation_names = {item["name"] for item in diagnostics["recommendations"]}
    assert "block_adverse_selection_markout_bucket" in recommendation_names
    assert "block_spread_tick_cost_bucket" in recommendation_names
    assert "require_positive_after_costs_out_of_sample" in recommendation_names


def test_build_filter_followup_report_fail_closes_when_attribution_missing(tmp_path):
    artifact = _write_artifact(
        tmp_path,
        {
            "classification": "diagnostic_only",
            "live_ready": False,
            "attempt_count": 1,
            "completed_count": 1,
            "fills_orders_pnl": {
                "total_fills": 1,
                "completed_pnl_sum": -0.01,
                "completed_positive_pnl_attempts": 0,
            },
            "attempts": [{"status": "completed", "diagnostics": {"pnl": -0.01, "fills": 1}}],
        },
    )

    report = followup.build_filter_followup_report(artifact_path=artifact)

    assert report["classification"] == "diagnostic_only"
    assert report["live_ready"] is False
    assert report["filter_diagnostics"]["fail_closed"] is True
    assert report["filter_diagnostics"]["market_filters"] == []
    assert report["filter_diagnostics"]["parameter_bucket_filters"] == []
    assert report["filter_diagnostics"]["recommendations"][0]["name"] == (
        "negative_pnl_attribution_missing_fail_closed"
    )
    assert (
        "negative_pnl_attribution_missing_or_not_eligible" in report["not_live_ready_reason_codes"]
    )


def test_build_filter_followup_report_unknown_shape_fail_closes(tmp_path):
    artifact = _write_artifact(tmp_path, [{"unexpected": "list-artifact"}])

    report = followup.build_filter_followup_report(artifact_path=artifact)

    assert report["classification"] == "diagnostic_only"
    assert report["live_ready"] is False
    assert report["artifact_shape_status"] == "unknown_fail_closed"
    assert "unknown_artifact_json_shape" in report["not_live_ready_reason_codes"]
    assert report["filter_diagnostics"]["fail_closed"] is True
    assert report["filter_diagnostics"]["eligible_negative_pnl_attempt_count"] == 0
    assert report["filter_diagnostics"]["recommendations"][0]["name"] == (
        "negative_pnl_attribution_missing_fail_closed"
    )
    assert all(report["safety"][field] is False for field in followup.REQUIRED_SAFETY_FIELDS)
    assert all(report[field] is False for field in followup.REQUIRED_SAFETY_FIELDS)


def test_cli_writes_json_and_markdown_reports(tmp_path, capsys):
    params = {
        "depth_levels": 1,
        "entry_imbalance": 0.55,
        "exit_imbalance": 0.5,
        "min_microprice_edge": 0.0005,
        "quote_lifetime_seconds": 10.0,
    }
    artifact = _write_artifact(
        tmp_path,
        {
            "classification": "diagnostic_only",
            "live_ready": False,
            "fills_orders_pnl": {
                "total_fills": 2,
                "completed_pnl_sum": -0.28315,
                "completed_positive_pnl_attempts": 0,
            },
            "attempts": [_negative_attempt(params)],
        },
    )
    output_dir = tmp_path / "reports"

    exit_code = followup.main(
        [
            "--artifact",
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--timestamp",
            "20260606T000000Z",
        ]
    )

    assert exit_code == 0
    emitted = json.loads(capsys.readouterr().out)
    json_path = output_dir / "microprice_negative_pnl_filter_followup_20260606T000000Z.json"
    md_path = output_dir / "microprice_negative_pnl_filter_followup_20260606T000000Z.md"
    assert emitted["output_files"]["json"] == str(json_path)
    assert emitted["output_files"]["markdown"] == str(md_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["live_ready"] is False
    markdown = md_path.read_text(encoding="utf-8")
    assert "orders_submitted=false" in markdown
    assert "credentials_required=false" in markdown
