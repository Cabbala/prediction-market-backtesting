from __future__ import annotations

from prediction_market_extensions.adapters.polymarket.public_scan_ranking import (
    LOW_FILL_LIQUIDITY_REWARD_MAKER,
    MICROPRICE,
    PUBLIC_SCAN_RANKING_SCHEMA_VERSION,
    PUBLIC_SCAN_SAFETY_FIELDS,
    VOLATILITY_SPIKE_DEEP_LIMIT_MAKER,
    build_strategy_ranked_scan,
)


def _candidate(
    slug: str,
    *,
    mid: float,
    spread: float = 0.001,
    liquidity: float = 1_000_000.0,
    volume24hr: float = 100_000.0,
    reward: bool = True,
) -> dict[str, object]:
    bid = max(0.0, mid - spread / 2.0)
    ask = min(1.0, mid + spread / 2.0)
    return {
        "market_id": slug,
        "slug": slug,
        "question": f"Will {slug} happen?",
        "yes_book": {
            "best_bid": bid,
            "best_ask": ask,
            "bid_size": 25_000.0,
            "ask_size": 25_000.0,
            "spread": spread,
            "mid": mid,
        },
        "yes_mid_gamma": mid,
        "liquidity": liquidity,
        "volume24hr": volume24hr,
        "volume": volume24hr * 2,
        "reward_eligible_signal": reward,
        "reward_fields": {"rewardsMinSize": 100, "rewardsMaxSpread": 2.5} if reward else {},
    }


def test_strategy_rankings_separate_extreme_tail_from_microprice_but_keep_reward() -> None:
    balanced = _candidate("balanced-tight-book", mid=0.50, liquidity=250_000)
    extreme_tail = _candidate(
        "extreme-tail-reward-book",
        mid=0.004,
        liquidity=5_000_000,
        volume24hr=750_000,
    )

    scan = build_strategy_ranked_scan([extreme_tail, balanced], limit=2)

    microprice = scan["top_candidates"][MICROPRICE]
    assert [row["slug"] for row in microprice] == [
        "balanced-tight-book",
        "extreme-tail-reward-book",
    ]
    assert microprice[0]["strategy_rankings"][MICROPRICE]["bucket"] == ("balanced_tight_order_book")
    assert microprice[1]["strategy_rankings"][MICROPRICE]["bucket"] == ("separated_tail_watch")
    assert (
        "extreme_tail_price" in microprice[1]["strategy_rankings"][MICROPRICE]["downrank_reasons"]
    )
    assert microprice[1]["strategy_rankings"][MICROPRICE]["eligible_for_handoff"] is False

    reward = scan["top_candidates"][LOW_FILL_LIQUIDITY_REWARD_MAKER]
    assert reward[0]["slug"] == "extreme-tail-reward-book"
    assert reward[0]["strategy_rankings"][LOW_FILL_LIQUIDITY_REWARD_MAKER]["bucket"] == (
        "extreme_tail_reward_watchlist"
    )
    assert (
        reward[0]["strategy_rankings"][LOW_FILL_LIQUIDITY_REWARD_MAKER]["eligible_for_handoff"]
        is True
    )

    volatility = scan["top_candidates"][VOLATILITY_SPIKE_DEEP_LIMIT_MAKER]
    assert volatility[0]["slug"] == "balanced-tight-book"
    assert volatility[1]["strategy_rankings"][VOLATILITY_SPIKE_DEEP_LIMIT_MAKER]["bucket"] == (
        "separated_tail_watch"
    )
    diagnostics = scan["ranking_diagnostics"]
    assert diagnostics["low_fill_reward_tail"]["extreme_tail_watchlist_count"] == 1
    assert diagnostics["overlap_diagnostics"]["low_fill_microprice_overlap_count"] == 2
    assert diagnostics["top_candidate_bucket_counts"][LOW_FILL_LIQUIDITY_REWARD_MAKER] == {
        "extreme_tail_reward_watchlist": 1,
        "reward_observation": 1,
    }


def test_strategy_ranking_metadata_records_filters_and_shadow_safety_fields() -> None:
    scan = build_strategy_ranked_scan([_candidate("balanced-tight-book", mid=0.50)], limit=1)
    metadata = scan["objective_specific_ranking"]

    assert metadata["schema_version"] == PUBLIC_SCAN_RANKING_SCHEMA_VERSION
    assert metadata["safety_fields"] == PUBLIC_SCAN_SAFETY_FIELDS
    assert set(metadata["strategies"]) == {
        MICROPRICE,
        VOLATILITY_SPIKE_DEEP_LIMIT_MAKER,
        LOW_FILL_LIQUIDITY_REWARD_MAKER,
    }
    assert metadata["strategies"][MICROPRICE]["primary_bucket"] == "balanced_tight_order_book"
    assert "tail_policy" in metadata["strategies"][MICROPRICE]["filters"]
    assert metadata["strategies"][LOW_FILL_LIQUIDITY_REWARD_MAKER]["primary_bucket"] == (
        "extreme_tail_reward_watchlist"
    )
    assert all(value is False for value in metadata["safety_fields"].values())


def test_strategy_rankings_fail_closed_for_missing_book_prices() -> None:
    missing_book = {
        "market_id": "missing-book",
        "slug": "missing-book",
        "question": "Will missing book happen?",
        "yes_mid_gamma": 0.50,
        "liquidity": 2_000_000.0,
        "volume24hr": 500_000.0,
        "reward_eligible_signal": True,
    }

    scan = build_strategy_ranked_scan([missing_book], limit=1)
    row = scan["top_candidates"][LOW_FILL_LIQUIDITY_REWARD_MAKER][0]

    for strategy in (
        MICROPRICE,
        VOLATILITY_SPIKE_DEEP_LIMIT_MAKER,
        LOW_FILL_LIQUIDITY_REWARD_MAKER,
    ):
        ranking = row["strategy_rankings"][strategy]
        assert ranking["bucket"] == "missing_book_fail_closed"
        assert ranking["eligible_for_handoff"] is False
        assert "missing_yes_book_prices" in ranking["downrank_reasons"]
    assert (
        scan["ranking_diagnostics"]["low_fill_reward_tail"]["missing_book_fail_closed_count"] == 1
    )
