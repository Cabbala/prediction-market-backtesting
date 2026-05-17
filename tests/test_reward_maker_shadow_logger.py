from __future__ import annotations

from scripts.reward_maker_shadow_logger import build_shadow_record


def test_build_shadow_record_is_shadow_only_and_uses_compact_books():
    record = build_shadow_record(
        {
            "slug": "example-market",
            "question": "Example?",
            "conditionId": "0xabc",
            "yes_token_id": "yes-token",
            "no_token_id": "no-token",
            "yes_mid": 0.08,
            "yes_spread": 0.001,
            "yes_book": {"bid": 0.08, "ask": 0.081, "bid_size": 100, "ask_size": 50},
            "no_book": {"bid": 0.919, "ask": 0.92},
            "reward_evidence": {
                "rewardsMaxSpread": 0.002,
                "rewardsMinSize": 10,
                "unexpected": "drop",
            },
        },
        generated_at="2026-05-09T00:00:00Z",
        quote_size=5.0,
    )

    assert record is not None
    assert record["mode"] == "shadow_only_no_live_trading"
    assert record["shadow_quote"]["in_reward_band_now"] is True
    assert record["shadow_quote"]["would_have_filled"] == "unknown_single_snapshot"
    assert record["shadow_quote"]["would_fill_classification"] == "optimistic"
    assert record["safety"]["orders_submitted"] is False
    assert record["safety"]["orders_cancelled"] is False
    assert record["safety"]["live_trading_worker_started"] is False
    assert record["reward_evidence"] == {"rewardsMaxSpread": 0.002, "rewardsMinSize": 10}
