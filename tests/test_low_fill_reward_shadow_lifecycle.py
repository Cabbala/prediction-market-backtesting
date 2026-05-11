from __future__ import annotations

import csv
import json

from scripts.low_fill_reward_shadow_lifecycle import (
    build_report,
    select_candidates,
    write_outputs,
)


def _candidate(slug: str = "tail-reward") -> dict[str, object]:
    return {
        "slug": slug,
        "question": "Will the tail outcome happen?",
        "condition_id": "0xabc",
        "yes_token_id": "yes-token",
        "no_token_id": "no-token",
        "yes_mid": 0.0035,
        "no_mid": 0.9965,
        "yes_book": {
            "bid": 0.003,
            "ask": 0.004,
            "bid_size": 10_000,
            "ask_size": 5_000,
            "depth_bid_2c": 20_000,
            "depth_ask_2c": 10_000,
        },
        "no_book": {
            "bid": 0.996,
            "ask": 0.997,
            "bid_size": 5_000,
            "ask_size": 10_000,
            "depth_bid_2c": 10_000,
            "depth_ask_2c": 20_000,
        },
        "spread": 0.001,
        "liquidity": 1_000_000,
        "volume": 2_000_000,
        "reward_evidence": {
            "reward_evidence": {
                "rewardsMaxSpread": 2.5,
                "rewardsMinSize": 100,
                "umaReward": "5",
            }
        },
        "strategy_fits": ["low-fill-probability liquidity-reward maker"],
    }


def test_select_candidates_normalizes_nested_reward_and_requires_double_sided_quote() -> None:
    candidates = select_candidates({"candidates": [_candidate()]}, max_candidates=1)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.reward_min_size == 100
    assert candidate.reward_max_spread_raw == 2.5
    assert candidate.reward_max_spread == 0.025
    assert candidate.yes_book.bid == 0.003
    assert candidate.no_book.ask == 0.997
    assert candidate.yes_token_id == "yes-token"
    assert candidate.no_token_id == "no-token"


def test_select_candidates_accepts_generic_tokens_flat_books_and_scalar_mids() -> None:
    row = _candidate()
    row.pop("yes_token_id")
    row.pop("no_token_id")
    row.pop("yes_mid")
    row.pop("no_mid")
    row["clobTokenIds"] = '["yes-generic", "no-generic"]'
    row["outcomes"] = ["Yes", "No"]
    row["yes_book"] = "0.0035"
    row["no_book"] = 0.9965
    row["yes_best_bid"] = 0.003
    row["yes_best_ask"] = 0.004
    row["no_best_bid"] = 0.996
    row["no_best_ask"] = 0.997
    row["reward_evidence"] = {
        "rewardsMaxSpread": 2.5,
        "rewardsMinSize": 100,
        "unexpected": "drop",
    }

    candidates = select_candidates({"candidates": [row]}, max_candidates=1)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.yes_token_id == "yes-generic"
    assert candidate.no_token_id == "no-generic"
    assert candidate.yes_mid == 0.0035
    assert candidate.no_mid == 0.9965
    assert candidate.yes_book.bid == 0.003
    assert candidate.no_book.ask == 0.997
    assert candidate.reward_min_size == 100
    assert candidate.reward_max_spread_raw == 2.5
    assert candidate.source_blockers == ()


def test_duplicate_side_tokens_block_lifecycle_report_without_dropping_candidate(tmp_path) -> None:
    source = tmp_path / "source.json"
    row = _candidate()
    row["no_token_id"] = row["yes_token_id"]
    source.write_text(json.dumps({"candidates": [row]}), encoding="utf-8")

    report = build_report(
        source,
        duration_secs=0,
        interval_secs=1,
        max_candidates=1,
        sleep=False,
    )

    assert report["candidate_count"] == 1
    assert report["classification"] == "blocked"
    assert "duplicate_yes_no_token_ids" in report["blocker_reasons"]
    assert report["candidate_table"][0]["source_blockers"] == ["duplicate_yes_no_token_ids"]


def test_build_report_scores_lifecycle_without_live_actions(tmp_path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"candidates": [_candidate()]}), encoding="utf-8")

    report = build_report(
        source,
        duration_secs=2,
        interval_secs=1,
        max_candidates=1,
        sleep=False,
    )

    assert report["classification"] == "diagnostic_only"
    assert report["safety"] == {
        "live_trading": False,
        "orders_submitted": False,
        "orders_signed": False,
        "orders_cancelled": False,
        "credentials_required": False,
        "worker_trading_started": False,
        "live_trading_worker_started": False,
    }
    assert report["scheduled_snapshot_count"] == 3
    assert report["snapshot_count"] == 3
    row = report["candidate_table"][0]
    assert row["double_sided_required"] is True
    assert row["time_in_band_secs"] == 2
    assert row["would_have_filled_status"] == "unknown_requires_trade_tape_or_l2_queue_position"
    assert row["expected_reward_ev_minus_expected_loss_classification"] == (
        "unknown_missing_reward_amount_or_fill_loss_distribution"
    )
    first_snapshot = report["snapshots"][0]
    assert first_snapshot["scoring_eligible_estimate"] is True
    assert first_snapshot["would_have_filled_estimate"]["known"] is False
    assert first_snapshot["safety"]["orders_submitted"] is False


def test_missing_book_fails_closed_as_blocked_unknown(tmp_path) -> None:
    source = tmp_path / "source.json"
    row = _candidate()
    row.pop("yes_book")
    row.pop("no_book")
    source.write_text(json.dumps({"candidates": [row]}), encoding="utf-8")

    report = build_report(
        source,
        duration_secs=0,
        interval_secs=1,
        max_candidates=1,
        sleep=False,
    )

    assert report["classification"] == "blocked"
    assert "selected_candidates_missing_top_of_book_snapshots" in report["blocker_reasons"]
    snapshot = report["snapshots"][0]
    assert snapshot["scoring_eligible_estimate"] is False
    assert snapshot["cancel_or_reprice_reason"] == "no_quote_missing_book"
    assert snapshot["would_have_filled_estimate"]["estimate"] == "unknown"


def test_write_outputs_includes_csv_and_safety_columns(tmp_path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"candidates": [_candidate()]}), encoding="utf-8")
    report = build_report(
        source,
        duration_secs=0,
        interval_secs=1,
        max_candidates=1,
        sleep=False,
    )

    outputs = write_outputs(report, tmp_path / "reports", "20260511T000000Z")

    assert set(outputs) == {"json", "csv", "markdown"}
    with open(outputs["csv"], newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert rows[0]["orders_submitted"] == "False"
    assert rows[0]["orders_signed"] == "False"
    assert rows[0]["orders_cancelled"] == "False"
    assert rows[0]["credentials_required"] == "False"
    assert rows[0]["live_trading"] == "False"
    assert rows[0]["worker_trading_started"] == "False"
    assert rows[0]["live_trading_worker_started"] == "False"
