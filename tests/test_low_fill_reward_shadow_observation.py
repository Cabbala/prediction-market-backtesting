from __future__ import annotations

from scripts.low_fill_reward_shadow_observation import _observation


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
    assert obs["accidental_fill_risk"] == "normal_requires_l2_fill_model"
    assert obs["exit_risk"] == "liquidity_proxy_ok_needs_l2"
    assert obs["orders_submitted"] is False
    assert obs["orders_signed"] is False
    assert obs["credentials_required"] is False


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
