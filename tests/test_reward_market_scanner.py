from datetime import datetime, timezone

from prediction_market_extensions.analysis.reward_market_scanner import RewardMarketScanner, canonical_yes_no_tokens


def _candidate(**overrides):
    base = {
        "id": "m1",
        "conditionId": "0xabc",
        "slug": "example",
        "question": "Will example happen?",
        "clobTokenIds": ["yes-token", "no-token"],
        "outcomes": ["Yes", "No"],
        "endDate": "2026-12-31T00:00:00Z",
        "volume": 100000,
        "volume24hr": 12000,
        "liquidity": 5000,
        "reward_evidence": {"umaReward": "5"},
        "books": [
            {"token_id": "yes-token", "best_bid": 0.49, "best_ask": 0.50, "bid_levels": 4, "ask_levels": 4},
            {"token_id": "no-token", "best_bid": 0.50, "best_ask": 0.51, "bid_levels": 4, "ask_levels": 4},
        ],
    }
    base.update(overrides)
    return base


def test_canonical_yes_no_tokens_accepts_reversed_outcomes():
    tokens, outcomes, error = canonical_yes_no_tokens(_candidate(clobTokenIds=["no", "yes"], outcomes=["No", "Yes"]))
    assert error is None
    assert tokens == ["yes", "no"]


def test_canonical_yes_no_tokens_rejects_duplicate_or_missing_outcomes():
    assert canonical_yes_no_tokens(_candidate(clobTokenIds=["a", "a"]))[2] == "requires_exactly_two_unique_clob_token_ids"
    assert canonical_yes_no_tokens(_candidate(outcomes=["Up", "Down"]))[2] == "requires_explicit_binary_yes_no_outcomes"


def test_build_manifest_scores_and_flags_shadow_only():
    scanner = RewardMarketScanner(now=datetime(2026, 5, 6, tzinfo=timezone.utc))
    manifest = scanner.build_manifest({"candidates": [_candidate()]}, source_path="scan.json")
    assert manifest["safety_mode"] == "shadow_backtest_only_no_live_orders"
    assert manifest["summary"]["eligible_count"] == 1
    row = manifest["candidates"][0]
    assert row["yes_token_id"] == "yes-token"
    assert row["no_token_id"] == "no-token"
    assert row["reward_category"] == "explicit_reward"
    assert row["score"] > 0


def test_build_manifest_fails_closed_without_books():
    scanner = RewardMarketScanner(now=datetime(2026, 5, 6, tzinfo=timezone.utc))
    manifest = scanner.build_manifest({"candidates": [_candidate(books=[])]}, source_path="scan.json")
    assert manifest["summary"]["eligible_count"] == 0
    assert manifest["skipped"][0]["skip_reason"] == "requires_complete_two_sided_books"



def test_book_token_ids_must_match_yes_no_tokens():
    scanner = RewardMarketScanner(now=datetime(2026, 5, 6, tzinfo=timezone.utc))
    bad = _candidate(books=[
        {"token_id": "other-yes", "best_bid": 0.49, "best_ask": 0.50},
        {"token_id": "other-no", "best_bid": 0.50, "best_ask": 0.51},
    ])
    manifest = scanner.build_manifest({"candidates": [bad]}, source_path="scan.json")
    assert manifest["summary"]["eligible_count"] == 0
    assert manifest["skipped"][0]["skip_reason"] == "book_token_ids_must_match_yes_no_tokens"


def test_flat_scan_best_bid_ask_fields_are_accepted_when_books_absent():
    scanner = RewardMarketScanner(now=datetime(2026, 5, 6, tzinfo=timezone.utc))
    candidate = _candidate(
        books=[],
        yes_best_bid=0.49,
        yes_best_ask=0.50,
        yes_bid_levels=3,
        yes_ask_levels=4,
        no_best_bid=0.50,
        no_best_ask=0.51,
        no_bid_levels=2,
        no_ask_levels=5,
    )
    manifest = scanner.build_manifest({"candidates": [candidate]}, source_path="scan.json")
    assert manifest["summary"]["eligible_count"] == 1
    row = manifest["candidates"][0]
    assert round(row["max_spread"], 6) == 0.01
    assert "thin_two_sided_depth" in row["accidental_fill_risk_flags"]


def test_level_arrays_do_not_crash_and_string_fields_are_lists():
    scanner = RewardMarketScanner(now=datetime(2026, 5, 6, tzinfo=timezone.utc))
    candidate = _candidate(
        strategy_fit="maker",
        source_tags="scanner",
        reward_evidence={"umaReward": "5", "clobRewards_count": 1, "api_secret": "must-not-persist"},
        books=[
            {"token_id": "yes-token", "bid_levels": [[0.49, 100]], "ask_levels": [[0.50, 100]]},
            {"token_id": "no-token", "bid_levels": [[0.50, 100]], "ask_levels": [[0.51, 100]]},
        ],
    )
    manifest = scanner.build_manifest({"candidates": [candidate]}, source_path="scan.json")
    row = manifest["candidates"][0]
    assert row["strategy_fits"] == ["maker"]
    assert row["source_tags"] == ["scanner"]
    assert row["reward_evidence"] == {"umaReward": "5", "clobRewards_count": 1}
    assert row["min_two_sided_depth_5c"] == 100
