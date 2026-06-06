from __future__ import annotations

import json
from pathlib import Path

import pytest

from prediction_market_extensions.adapters.polymarket.reward_market_scanner import (
    CANONICAL_MANIFEST_PREFIX,
    LEGACY_MANIFEST_PREFIX,
    SHADOW_MODE,
    accidental_fill_risk_flags,
    build_reward_manifest,
    load_scan,
    score_candidate,
)
from scripts.build_reward_market_manifest import main as build_manifest_main


def _candidate(**overrides):
    base = {
        "market_id": "1",
        "condition_id": "0xabc",
        "slug": "demo-market",
        "question": "Will demo happen?",
        "clob_token_ids": ["yes", "no"],
        "outcomes": ["Yes", "No"],
        "complete_books": True,
        "end_date": "2026-07-20T00:00:00Z",
        "createdAt": "2026-05-01T00:00:00Z",
        "liquidity": 100000.0,
        "volume": 200000.0,
        "reward_hint": False,
        "strategy_fit": {"volatility_spike_deep_limit_maker": True},
        "yes_book": {
            "ok": True,
            "best_bid": 0.49,
            "best_ask": 0.491,
            "best_bid_size": 10000,
            "best_ask_size": 12000,
            "mid": 0.4905,
            "spread": 0.001,
        },
        "no_book": {
            "ok": True,
            "best_bid": 0.509,
            "best_ask": 0.510,
            "best_bid_size": 9000,
            "best_ask_size": 11000,
            "mid": 0.5095,
            "spread": 0.001,
        },
    }
    base.update(overrides)
    return base


def test_score_candidate_keeps_reward_scanner_shadow_safe() -> None:
    scored = score_candidate(_candidate())

    assert scored["eligible_for_backtest_queue"] is True
    assert scored["features"]["fee_reward_category"] == "public_proxy_only_reward_unverified"
    assert scored["features"]["min_top_book_depth"] == 20000
    assert scored["features"]["market_age_days"] is not None
    assert scored["accidental_fill_risk_flags"] == []


def test_score_candidate_reads_latest_scan_books_and_explicit_reward_terms() -> None:
    candidate = {
        "id": "558960",
        "conditionId": "0xabc",
        "slug": "world-cup-tail",
        "question": "Will team win?",
        "outcomes": ["Yes", "No"],
        "endDate": "2026-07-20T00:00:00Z",
        "liquidity": 5_000_000,
        "volume24hr": 50_000,
        "rewardsMinSize": 100,
        "rewardsMaxSpread": 2.5,
        "strategy_fits": ["microprice", "reward_eligible_candidate"],
        "books": [
            {
                "token_id": "yes",
                "best_bid": 0.002,
                "best_ask": 0.003,
                "bid_size": 1000,
                "ask_size": 2000,
                "depth_2c": 8000,
                "mid": 0.0025,
                "spread": 0.001,
            },
            {
                "token_id": "no",
                "best_bid": 0.997,
                "best_ask": 0.998,
                "bid_size": 2000,
                "ask_size": 1000,
                "depth_2c": 8000,
                "mid": 0.9975,
                "spread": 0.001,
            },
        ],
    }

    scored = score_candidate(candidate)

    assert scored["features"]["fee_reward_category"] == "explicit_gamma_reward_terms"
    assert scored["features"]["has_explicit_reward_evidence"] is True
    assert scored["features"]["has_complete_clob_token_ids"] is True
    assert scored["features"]["volume_24h"] == 50000
    assert scored["features"]["min_top_book_depth"] == 8000
    assert scored["accidental_fill_risk_flags"] == [
        "no_tail_price_accidental_fill_risk",
        "yes_tail_price_accidental_fill_risk",
    ]


def test_accidental_fill_risk_flags_tail_and_thin_books() -> None:
    candidate = _candidate(
        yes_book={
            "ok": True,
            "best_bid": 0.002,
            "best_ask": 0.003,
            "best_bid_size": 50,
            "best_ask_size": 60,
            "mid": 0.0025,
            "spread": 0.001,
        },
        no_book={
            "ok": True,
            "best_bid": 0.997,
            "best_ask": 0.998,
            "best_bid_size": 50,
            "best_ask_size": 60,
            "mid": 0.9975,
            "spread": 0.001,
        },
    )

    assert accidental_fill_risk_flags(candidate) == [
        "no_tail_price_accidental_fill_risk",
        "thin_top_of_book_fill_risk",
        "yes_tail_price_accidental_fill_risk",
    ]


def test_build_reward_manifest_accepts_candidates_key_and_declares_no_live_trading() -> None:
    scan = {
        "metadata": {"utc_timestamp": "2026-05-04T06:01:34+00:00", "mode": SHADOW_MODE},
        "candidates": [
            _candidate(
                market_id="wide",
                yes_book={
                    "ok": True,
                    "best_bid": 0.45,
                    "best_ask": 0.50,
                    "best_bid_size": 100,
                    "best_ask_size": 100,
                    "mid": 0.475,
                    "spread": 0.05,
                },
            ),
            _candidate(market_id="tight", rewardsMinSize=100),
        ],
    }

    manifest = build_reward_manifest(scan, limit=2)

    assert manifest["mode"] == SHADOW_MODE
    assert manifest["manifest_kind"] == "reward_scanner_manifest"
    assert manifest["canonical_manifest_prefix"] == CANONICAL_MANIFEST_PREFIX
    assert manifest["legacy_manifest_prefixes"] == [LEGACY_MANIFEST_PREFIX]
    assert manifest["source_provenance"] == {
        "source_artifact_path": None,
        "source_scan_timestamp_utc": "2026-05-04T06:01:34+00:00",
        "source_scan_mode": SHADOW_MODE,
        "source_artifacts": {},
    }
    assert manifest["safety"] == {
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
    }
    assert manifest["orders_submitted"] is False
    assert manifest["orders_signed"] is False
    assert manifest["orders_cancelled"] is False
    assert manifest["credentials_required"] is False
    assert manifest["live_trading_worker_started"] is False
    assert manifest["worker_trading_started"] is False
    assert [candidate["market_id"] for candidate in manifest["candidates"]] == ["tight", "wide"]
    assert "spread_too_wide_for_reward_proxy" in manifest["candidates"][1]["blockers"]


def test_invalid_clob_token_ids_block_backtest_queue() -> None:
    for token_ids in ([], ["yes"], ["same", "same"], ["yes", "no", "maybe"]):
        scored = score_candidate(_candidate(clob_token_ids=token_ids))

        assert scored["eligible_for_backtest_queue"] is False
        assert "invalid_or_missing_yes_no_clob_token_ids" in scored["blockers"]
        assert scored["features"]["has_complete_clob_token_ids"] is False


def test_reversed_outcome_order_canonicalizes_yes_no_token_ids() -> None:
    manifest = build_reward_manifest(
        {
            "metadata": {"utc_timestamp": "2026-05-04T06:01:34+00:00", "mode": SHADOW_MODE},
            "candidates": [
                _candidate(clob_token_ids=["no-token", "yes-token"], outcomes=["No", "Yes"])
            ],
        },
        limit=1,
    )

    candidate = manifest["candidates"][0]
    assert candidate["clob_token_ids"] == ["yes-token", "no-token"]
    assert candidate["eligible_for_backtest_queue"] is True


def test_json_string_clob_token_ids_and_binary_outcomes_are_supported() -> None:
    manifest = build_reward_manifest(
        {
            "metadata": {"utc_timestamp": "2026-05-04T06:01:34+00:00", "mode": SHADOW_MODE},
            "candidates": [
                _candidate(
                    clob_token_ids=None,
                    clobTokenIds='["yes-token", "no-token"]',
                    outcomes='["Yes", "No"]',
                )
            ],
        },
        limit=1,
    )

    candidate = manifest["candidates"][0]
    assert candidate["clob_token_ids"] == ["yes-token", "no-token"]
    assert candidate["eligible_for_backtest_queue"] is True
    assert candidate["features"]["has_yes_no_outcomes"] is True


def test_missing_malformed_or_non_binary_outcomes_block_backtest_queue() -> None:
    for outcomes in (None, "not-json", ["Red", "Blue"]):
        scored = score_candidate(_candidate(outcomes=outcomes))

        assert scored["eligible_for_backtest_queue"] is False
        assert "invalid_yes_no_outcome_mapping" in scored["blockers"]
        assert "invalid_or_missing_yes_no_clob_token_ids" in scored["blockers"]


def test_manifest_summary_and_rank_are_emitted() -> None:
    scan = {
        "metadata": {"utc_timestamp": "2026-05-05T00:00:00Z"},
        "candidates": [
            {
                "id": "m1",
                "conditionId": "c1",
                "question": "Will test happen?",
                "outcomes": ["Yes", "No"],
                "clobTokenIds": ["1", "2"],
                "volume": 100000,
                "liquidity": 100000,
                "endDate": "2026-06-05T00:00:00Z",
                "yes_book": {
                    "best_bid": 0.49,
                    "best_ask": 0.50,
                    "best_bid_size": 6000,
                    "best_ask_size": 6000,
                },
                "no_book": {
                    "best_bid": 0.50,
                    "best_ask": 0.51,
                    "best_bid_size": 6000,
                    "best_ask_size": 6000,
                },
                "rewardsMinSize": 100,
            }
        ],
    }
    manifest = build_reward_manifest(scan)
    assert manifest["summary"]["candidate_count"] == 1
    assert manifest["summary"]["manifest_candidate_count"] == 1
    assert manifest["summary"]["eligible_for_backtest_queue_count"] == 1
    assert manifest["summary"]["explicit_reward_evidence_count"] == 1
    assert manifest["candidates"][0]["rank"] == 1


def test_negative_limit_returns_no_candidates_and_nonnegative_summary() -> None:
    scan = {
        "metadata": {"utc_timestamp": "2026-05-05T00:00:00Z"},
        "candidates": [_candidate(market_id="1"), _candidate(market_id="2")],
    }
    manifest = build_reward_manifest(scan, limit=-1)
    assert manifest["summary"]["candidate_count"] == 2
    assert manifest["summary"]["manifest_candidate_count"] == 0
    assert manifest["candidates"] == []


def test_build_reward_manifest_main_writes_source_paths(tmp_path: Path) -> None:
    scan_path = tmp_path / "scan.json"
    report_path = tmp_path / "strategy.md"
    output_dir = tmp_path / "out"
    scan_path.write_text(
        json.dumps(
            {"metadata": {"utc_timestamp": "2026-05-05T00:00:00Z"}, "candidates": [_candidate()]}
        ),
        encoding="utf-8",
    )
    report_path.write_text("# strategy\n", encoding="utf-8")

    assert (
        build_manifest_main(
            [
                "--input",
                str(scan_path),
                "--strategy-report",
                str(report_path),
                "--output-dir",
                str(output_dir),
                "--limit",
                "1",
            ]
        )
        == 0
    )
    manifest_paths = sorted(output_dir.glob(f"{CANONICAL_MANIFEST_PREFIX}_*.json"))
    legacy_manifest_paths = sorted(output_dir.glob(f"{LEGACY_MANIFEST_PREFIX}_*.json"))
    rules_paths = sorted(output_dir.glob(f"{CANONICAL_MANIFEST_PREFIX}_rules_*.md"))
    legacy_rules_paths = sorted(output_dir.glob(f"{LEGACY_MANIFEST_PREFIX}_rules_*.md"))
    assert len(manifest_paths) == 1
    assert len(legacy_manifest_paths) == 1
    assert len(rules_paths) == 1
    assert len(legacy_rules_paths) == 1
    assert manifest_paths[0].name.replace(CANONICAL_MANIFEST_PREFIX, LEGACY_MANIFEST_PREFIX) == (
        legacy_manifest_paths[0].name
    )
    assert rules_paths[0].name.replace(CANONICAL_MANIFEST_PREFIX, LEGACY_MANIFEST_PREFIX) == (
        legacy_rules_paths[0].name
    )
    manifest = load_scan(manifest_paths[0])
    legacy_manifest = load_scan(legacy_manifest_paths[0])
    assert legacy_manifest == manifest
    assert manifest["input_scan_path"] == str(scan_path)
    assert manifest["input_strategy_report_path"] == str(report_path)
    assert manifest["source_paths"] == {
        "market_scan_json": str(scan_path),
        "strategy_or_scan_report_md": str(report_path),
    }
    assert manifest["source_provenance"]["source_artifact_path"] == str(scan_path)
    assert manifest["canonical_manifest_prefix"] == CANONICAL_MANIFEST_PREFIX
    assert manifest["legacy_manifest_prefixes"] == [LEGACY_MANIFEST_PREFIX]
    assert manifest["safety"]["orders_submitted"] is False
    assert manifest["safety"]["orders_signed"] is False
    assert manifest["safety"]["orders_cancelled"] is False
    assert manifest["safety"]["credentials_required"] is False
    assert manifest["safety"]["live_trading_worker_started"] is False


def test_build_reward_manifest_main_rejects_negative_limit(tmp_path: Path) -> None:
    scan_path = tmp_path / "scan.json"
    scan_path.write_text(json.dumps({"metadata": {}, "candidates": []}), encoding="utf-8")
    with pytest.raises(SystemExit):
        build_manifest_main(
            ["--input", str(scan_path), "--output-dir", str(tmp_path / "out"), "--limit", "-1"]
        )


def test_latest_scan_shape_uses_liquidity_num_top10_depth_and_reward_evidence() -> None:
    candidate = _candidate(
        liquidity=None,
        liquidityNum=250_000,
        volume=None,
        volumeNum=125_000,
        reward_evidence={"rewardsMinSize": "100", "rewardsMaxSpread": "3.0"},
        source_market_url="https://polymarket.com/event/demo/demo-market",
        source_tags="reward_scan",
        yes_book={
            "token_id": "yes",
            "best_bid": 0.003,
            "best_ask": 0.004,
            "spread": 0.001,
            "mid": 0.0035,
            "bid_levels": 3,
            "ask_levels": 77,
            "depth_bid_top10": 3_000_000,
            "depth_ask_top10": 2_000_000,
        },
        no_book={
            "token_id": "no",
            "best_bid": 0.996,
            "best_ask": 0.997,
            "spread": 0.001,
            "mid": 0.9965,
            "bid_levels": 77,
            "ask_levels": 3,
            "depth_bid_top10": 2_000_000,
            "depth_ask_top10": 3_000_000,
        },
    )

    manifest = build_reward_manifest({"metadata": {}, "candidates": [candidate]}, limit=1)
    row = manifest["candidates"][0]

    assert row["source_url"] == "https://polymarket.com/event/demo/demo-market"
    assert row["features"]["liquidity"] == 250_000
    assert row["features"]["volume"] == 125_000
    assert row["features"]["min_top_book_depth"] == 5_000_000
    assert row["features"]["fee_reward_category"] == "explicit_gamma_reward_terms"
    assert row["features"]["has_explicit_reward_evidence"] is True
    assert row["features"]["book_token_ids_match_yes_no_mapping"] is True
    assert row["reward_evidence"] == {"rewardsMinSize": "100", "rewardsMaxSpread": "3.0"}


def test_latest_scan_shape_preserves_raw_tokens_without_canonical_mapping() -> None:
    candidate = {
        "id": "553858",
        "condition_id": "0xabc",
        "slug": "will-the-new-york-knicks-win-the-2026-nba-finals",
        "question": "Will the New York Knicks win the 2026 NBA Finals?",
        "liquidity_num": 453532.94,
        "volume_24h": 1400427.53,
        "yes_mid": 0.7905,
        "clob_token_ids": ["yes-source-token", "no-source-token"],
        "yes_book": {
            "token_id": "yes-source-token",
            "best_bid": 0.79,
            "best_ask": 0.791,
            "spread": 0.001,
            "mid": 0.7905,
        },
    }

    manifest = build_reward_manifest({"metadata": {}, "top_candidates": [candidate]}, limit=1)
    row = manifest["candidates"][0]

    assert row["source_clob_token_ids"] == ["yes-source-token", "no-source-token"]
    assert row["clob_token_ids"] == []
    assert row["yes_token_id"] is None
    assert row["no_token_id"] is None
    assert row["yes_book"]["token_id"] == "yes-source-token"
    assert row["token_provenance"]["source_clob_token_ids"] == [
        "yes-source-token",
        "no-source-token",
    ]
    assert row["token_provenance"]["side_book_token_ids"] == {
        "yes": "yes-source-token",
        "no": None,
    }
    assert row["token_provenance"]["status"] == "incomplete_fail_closed"
    assert (
        "missing_canonical_yes_no_clob_token_ids" in row["token_provenance"]["fail_closed_reasons"]
    )
    assert "missing_no_best_bid" in row["book_provenance"]["fail_closed_reasons"]
    assert row["eligible_for_backtest_queue"] is False


def test_current_scan_bid_ask_books_emit_complete_book_provenance() -> None:
    candidate = _candidate(
        liquidity=None,
        liquidityNum=250_000,
        volume=None,
        volumeNum=125_000,
        complete_yes_no_clob_books=True,
        reward_evidence={"rewardsMinSize": "100", "rewardsMaxSpread": "3.0"},
        yes_book={
            "token_id": "yes",
            "bid": 0.003,
            "ask": 0.004,
            "bid_size": 3_000_000,
            "ask_size": 2_000_000,
            "mid": 0.0035,
            "spread": 0.001,
            "present": True,
        },
        no_book={
            "token_id": "no",
            "bid": 0.996,
            "ask": 0.997,
            "bid_size": 2_000_000,
            "ask_size": 3_000_000,
            "mid": 0.9965,
            "spread": 0.001,
            "present": True,
        },
    )

    manifest = build_reward_manifest(
        {"metadata": {"utc_timestamp": "2026-05-12T06:02:46Z"}, "candidates": [candidate]},
        limit=1,
        source_artifact_path="/tmp/current-scan.json",
    )
    row = manifest["candidates"][0]

    assert row["eligible_for_backtest_queue"] is True
    assert row["blockers"] == []
    assert row["book_provenance"]["complete"] is True
    assert row["book_provenance"]["source_artifact_path"] == "/tmp/current-scan.json"
    assert row["book_provenance"]["source_timestamp_utc"] == "2026-05-12T06:02:46Z"
    assert row["book_provenance"]["sides"]["yes"]["side"] == "yes"
    assert row["book_provenance"]["sides"]["yes"]["best_bid"] == 0.003
    assert row["book_provenance"]["sides"]["yes"]["best_ask"] == 0.004
    assert row["book_provenance"]["sides"]["yes"]["depth_proxy"] == 5_000_000


def test_flat_top_of_book_fields_are_supported_without_fabricating_books() -> None:
    candidate = {
        "id": "flat-1",
        "conditionId": "0xflat",
        "question": "Will flat happen?",
        "outcomes": ["Yes", "No"],
        "clobTokenIds": ["yes-flat", "no-flat"],
        "yes_token_id": "yes-flat",
        "no_token_id": "no-flat",
        "yes_best_bid": 0.49,
        "yes_best_ask": 0.491,
        "yes_best_bid_size": 6000,
        "yes_best_ask_size": 7000,
        "no_best_bid": 0.509,
        "no_best_ask": 0.510,
        "no_best_bid_size": 6000,
        "no_best_ask_size": 7000,
        "liquidityNum": 50_000,
        "volumeNum": 60_000,
        "endDate": "2026-07-20T00:00:00Z",
    }

    scored = score_candidate(candidate)

    assert scored["eligible_for_backtest_queue"] is True
    assert scored["features"]["yes_spread"] == pytest.approx(0.001)
    assert scored["features"]["min_top_book_depth"] == 13_000


def test_compact_scalar_books_keep_side_token_candidate_but_fail_closed_on_missing_book() -> None:
    manifest = build_reward_manifest(
        {
            "metadata": {"utc_timestamp": "2026-05-04T06:01:34+00:00", "mode": SHADOW_MODE},
            "candidates": [
                {
                    "id": "compact-book",
                    "conditionId": "0xcompact",
                    "slug": "compact-book-market",
                    "question": "Will compact scalar books normalize?",
                    "yes_token_id": "yes-compact-token",
                    "no_token_id": "no-compact-token",
                    "yes_book": "0.49",
                    "no_book": 0.51,
                    "liquidityNum": 50_000,
                    "volumeNum": 60_000,
                    "endDate": "2026-07-20T00:00:00Z",
                    "reward_evidence": {
                        "rewardsMinSize": "100",
                        "rewardsMaxSpread": "2.5",
                        "unexpected": "drop",
                    },
                }
            ],
        },
        limit=1,
    )

    row = manifest["candidates"][0]
    assert manifest["summary"]["candidate_count"] == 1
    assert row["clob_token_ids"] == ["yes-compact-token", "no-compact-token"]
    assert row["yes_token_id"] == "yes-compact-token"
    assert row["no_token_id"] == "no-compact-token"
    assert row["features"]["has_complete_clob_token_ids"] is True
    assert row["features"]["has_yes_no_outcomes"] is True
    assert row["features"]["yes_spread"] == 0.0
    assert row["features"]["has_explicit_reward_evidence"] is True
    assert row["reward_evidence"] == {"rewardsMinSize": "100", "rewardsMaxSpread": "2.5"}
    assert row["eligible_for_backtest_queue"] is False
    assert "missing_complete_yes_no_clob_books" in row["blockers"]
    assert row["book_provenance"]["complete"] is False
    assert "missing_yes_best_bid" in row["book_provenance"]["fail_closed_reasons"]
    assert "missing_no_best_ask" in row["book_provenance"]["fail_closed_reasons"]


def test_flat_side_token_fields_provide_explicit_yes_no_mapping() -> None:
    candidate = {
        "id": "flat-side-token-1",
        "conditionId": "0xflat-side-token",
        "question": "Will side-scoped tokens be accepted?",
        "yes_token_id": "yes-side-token",
        "no_token_id": "no-side-token",
        "yes_best_bid": 0.49,
        "yes_best_ask": 0.491,
        "yes_best_bid_size": 6000,
        "yes_best_ask_size": 7000,
        "no_best_bid": 0.509,
        "no_best_ask": 0.510,
        "no_best_bid_size": 6000,
        "no_best_ask_size": 7000,
        "liquidityNum": 50_000,
        "volumeNum": 60_000,
        "endDate": "2026-07-20T00:00:00Z",
    }

    scored = score_candidate(candidate)

    assert scored["eligible_for_backtest_queue"] is True
    assert scored["features"]["has_complete_clob_token_ids"] is True
    assert scored["features"]["has_yes_no_outcomes"] is True
    assert scored["features"]["book_token_ids_match_yes_no_mapping"] is True


def test_flat_side_token_fields_with_explicit_outcomes_but_no_clob_token_ids() -> None:
    candidate = {
        "id": "flat-side-token-outcomes",
        "conditionId": "0xflat-side-token-outcomes",
        "question": "Will side-scoped tokens plus outcomes be accepted?",
        "outcomes": ["Yes", "No"],
        "yes_token_id": "yes-side-token",
        "no_token_id": "no-side-token",
        "yes_best_bid": 0.49,
        "yes_best_ask": 0.491,
        "yes_best_bid_size": 6000,
        "yes_best_ask_size": 7000,
        "no_best_bid": 0.509,
        "no_best_ask": 0.510,
        "no_best_bid_size": 6000,
        "no_best_ask_size": 7000,
        "liquidityNum": 50_000,
        "volumeNum": 60_000,
        "endDate": "2026-07-20T00:00:00Z",
    }

    scored = score_candidate(candidate)

    assert scored["eligible_for_backtest_queue"] is True
    assert scored["features"]["has_complete_clob_token_ids"] is True
    assert scored["features"]["has_yes_no_outcomes"] is True
    assert scored["features"]["book_token_ids_match_yes_no_mapping"] is True


def test_side_scoped_tokens_ignore_reversed_outcomes_without_raw_clob_ids() -> None:
    candidate = {
        "id": "side-token-reversed-outcomes",
        "conditionId": "0xside-token-reversed-outcomes",
        "question": "Will side-scoped tokens stay canonical?",
        "outcomes": ["No", "Yes"],
        "yes_token_id": "yes-side-token",
        "no_token_id": "no-side-token",
        "yes_best_bid": 0.49,
        "yes_best_ask": 0.491,
        "yes_best_bid_size": 6000,
        "yes_best_ask_size": 7000,
        "no_best_bid": 0.509,
        "no_best_ask": 0.510,
        "no_best_bid_size": 6000,
        "no_best_ask_size": 7000,
        "liquidityNum": 50_000,
        "volumeNum": 60_000,
        "endDate": "2026-07-20T00:00:00Z",
    }

    manifest = build_reward_manifest({"metadata": {}, "candidates": [candidate]}, limit=1)
    row = manifest["candidates"][0]

    assert row["eligible_for_backtest_queue"] is True
    assert row["clob_token_ids"] == ["yes-side-token", "no-side-token"]


def test_positional_books_without_token_ids_do_not_use_side_token_fallback() -> None:
    candidate = {
        "id": "positional-books-with-side-tokens",
        "conditionId": "0xpositional-books-with-side-tokens",
        "question": "Are positional books without token IDs accepted?",
        "outcomes": ["Yes", "No"],
        "yes_token_id": "yes-side-token",
        "no_token_id": "no-side-token",
        "books": [
            {
                "best_bid": 0.49,
                "best_ask": 0.491,
                "best_bid_size": 6000,
                "best_ask_size": 7000,
            },
            {
                "best_bid": 0.509,
                "best_ask": 0.510,
                "best_bid_size": 6000,
                "best_ask_size": 7000,
            },
        ],
        "liquidityNum": 50_000,
        "volumeNum": 60_000,
        "endDate": "2026-07-20T00:00:00Z",
    }

    scored = score_candidate(candidate)

    assert scored["eligible_for_backtest_queue"] is False
    assert "invalid_or_missing_yes_no_clob_token_ids" in scored["blockers"]
    assert scored["features"]["has_complete_clob_token_ids"] is False


def test_token_only_side_fields_do_not_make_complete_books() -> None:
    candidate = {
        "id": "token-only",
        "conditionId": "0xtokenonly",
        "question": "Do token-only side fields pass?",
        "yes_token_id": "yes-token",
        "no_token_id": "no-token",
        "liquidityNum": 50_000,
        "volumeNum": 60_000,
        "endDate": "2026-07-20T00:00:00Z",
    }

    scored = score_candidate(candidate)

    assert scored["eligible_for_backtest_queue"] is False
    assert "missing_complete_yes_no_clob_books" in scored["blockers"]
    assert scored["features"]["has_complete_clob_token_ids"] is True
    assert scored["features"]["has_yes_no_outcomes"] is True


def test_positional_books_without_outcomes_remain_fail_closed() -> None:
    candidate = {
        "id": "books-no-outcomes",
        "conditionId": "0xbooks",
        "question": "Are positional books ambiguous?",
        "books": [
            {
                "token_id": "token-a",
                "best_bid": 0.49,
                "best_ask": 0.491,
                "best_bid_size": 6000,
                "best_ask_size": 7000,
            },
            {
                "token_id": "token-b",
                "best_bid": 0.509,
                "best_ask": 0.510,
                "best_bid_size": 6000,
                "best_ask_size": 7000,
            },
        ],
        "liquidityNum": 50_000,
        "volumeNum": 60_000,
        "endDate": "2026-07-20T00:00:00Z",
    }

    scored = score_candidate(candidate)

    assert scored["eligible_for_backtest_queue"] is False
    assert "invalid_or_missing_yes_no_clob_token_ids" in scored["blockers"]
    assert "invalid_yes_no_outcome_mapping" in scored["blockers"]


def test_invalid_duplicate_and_non_binary_tokens_are_reported_not_truncated() -> None:
    manifest = build_reward_manifest(
        {
            "metadata": {"utc_timestamp": "2026-05-04T06:01:34+00:00", "mode": SHADOW_MODE},
            "candidates": [
                _candidate(market_id="duplicate", clob_token_ids=["same", "same"]),
                _candidate(market_id="non-binary", outcomes=["Red", "Blue"]),
            ],
        },
        limit=2,
    )

    assert manifest["summary"]["candidate_count"] == 2
    for row in manifest["candidates"]:
        assert row["clob_token_ids"] == []
        assert row["eligible_for_backtest_queue"] is False
        assert "invalid_or_missing_yes_no_clob_token_ids" in row["blockers"]
    assert "invalid_yes_no_outcome_mapping" in manifest["candidates"][1]["blockers"]


def test_book_token_mismatch_blocks_backtest_queue() -> None:
    scored = score_candidate(
        _candidate(
            clob_token_ids=["yes-token", "no-token"],
            yes_book={
                "token_id": "no-token",
                "best_bid": 0.49,
                "best_ask": 0.491,
                "best_bid_size": 6000,
                "best_ask_size": 6000,
            },
            no_book={
                "token_id": "yes-token",
                "best_bid": 0.509,
                "best_ask": 0.510,
                "best_bid_size": 6000,
                "best_ask_size": 6000,
            },
        )
    )

    assert scored["eligible_for_backtest_queue"] is False
    assert "book_token_ids_do_not_match_yes_no_mapping" in scored["blockers"]
