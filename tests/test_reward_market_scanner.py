from __future__ import annotations

from prediction_market_extensions.adapters.polymarket.reward_market_scanner import (
    SHADOW_MODE,
    accidental_fill_risk_flags,
    build_reward_manifest,
    score_candidate,
)


def _candidate(**overrides):
    base = {
        "market_id": "1",
        "condition_id": "0xabc",
        "slug": "demo-market",
        "question": "Will demo happen?",
        "clob_token_ids": ["yes", "no"],
        "complete_books": True,
        "end_date": "2026-07-20T00:00:00Z",
        "createdAt": "2026-05-01T00:00:00Z",
        "liquidity": 100000.0,
        "volume": 200000.0,
        "reward_hint": False,
        "strategy_fit": {"volatility_spike_deep_limit_maker": True},
        "yes_book": {"ok": True, "best_bid": 0.49, "best_ask": 0.491, "best_bid_size": 10000, "best_ask_size": 12000, "mid": 0.4905, "spread": 0.001},
        "no_book": {"ok": True, "best_bid": 0.509, "best_ask": 0.510, "best_bid_size": 9000, "best_ask_size": 11000, "mid": 0.5095, "spread": 0.001},
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
        "endDate": "2026-07-20T00:00:00Z",
        "liquidity": 5_000_000,
        "volume24hr": 50_000,
        "rewardsMinSize": 100,
        "rewardsMaxSpread": 2.5,
        "strategy_fits": ["microprice", "reward_eligible_candidate"],
        "books": [
            {"token_id": "yes", "best_bid": 0.002, "best_ask": 0.003, "bid_size": 1000, "ask_size": 2000, "depth_2c": 8000, "mid": 0.0025, "spread": 0.001},
            {"token_id": "no", "best_bid": 0.997, "best_ask": 0.998, "bid_size": 2000, "ask_size": 1000, "depth_2c": 8000, "mid": 0.9975, "spread": 0.001},
        ],
    }

    scored = score_candidate(candidate)

    assert scored["features"]["fee_reward_category"] == "explicit_gamma_reward_terms"
    assert scored["features"]["has_explicit_reward_evidence"] is True
    assert scored["features"]["volume_24h"] == 50000
    assert scored["features"]["min_top_book_depth"] == 8000
    assert scored["accidental_fill_risk_flags"] == [
        "no_tail_price_accidental_fill_risk",
        "yes_tail_price_accidental_fill_risk",
    ]


def test_accidental_fill_risk_flags_tail_and_thin_books() -> None:
    candidate = _candidate(
        yes_book={"ok": True, "best_bid": 0.002, "best_ask": 0.003, "best_bid_size": 50, "best_ask_size": 60, "mid": 0.0025, "spread": 0.001},
        no_book={"ok": True, "best_bid": 0.997, "best_ask": 0.998, "best_bid_size": 50, "best_ask_size": 60, "mid": 0.9975, "spread": 0.001},
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
            _candidate(market_id="wide", yes_book={"ok": True, "best_bid": 0.45, "best_ask": 0.50, "best_bid_size": 100, "best_ask_size": 100, "mid": 0.475, "spread": 0.05}),
            _candidate(market_id="tight", rewardsMinSize=100),
        ],
    }

    manifest = build_reward_manifest(scan, limit=2)

    assert manifest["mode"] == SHADOW_MODE
    assert manifest["safety"] == {
        "live_trading": False,
        "submit_orders": False,
        "sign_orders": False,
        "requires_secrets": False,
        "intended_uses": ["PMBT_BACKTEST_QUEUE", "HOMERUN_SHADOW_FORWARD_LOGGING"],
    }
    assert [candidate["market_id"] for candidate in manifest["candidates"]] == ["tight", "wide"]
    assert "spread_too_wide_for_reward_proxy" in manifest["candidates"][1]["blockers"]
