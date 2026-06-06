from __future__ import annotations

import asyncio
import json
import os
from argparse import Namespace

from scripts import job_b_microprice_batch
from scripts.job_b_microprice_batch import load_candidates


def test_load_candidates_accepts_coverage_pass_manifest_strategy_alias(tmp_path):
    manifest = tmp_path / "coverage_pass.json"
    manifest.write_text(
        json.dumps(
            {
                "strategy": "Microprice",
                "candidates": [
                    {
                        "slug": "will-wti-crude-oil-wti-hit-high-200-in-may",
                        "market_slug": "will-wti-crude-oil-wti-hit-high-200-in-may",
                        "question": "Will WTI Crude Oil (WTI) hit (HIGH) $200 in May?",
                        "token_index": 0,
                        "source_strategy": "Microprice",
                        "coverage": {
                            "status": "pass",
                            "book_events": 577,
                            "min_book_events": 50,
                            "gap_hours_missing": 0,
                        },
                    }
                ],
            }
        )
    )

    candidates = load_candidates(manifest, strategy="microprice_optimizer", max_candidates=5)

    assert len(candidates) == 1
    assert candidates[0].slug == "will-wti-crude-oil-wti-hit-high-200-in-may"
    assert candidates[0].source_strategy == "Microprice"


def test_load_candidates_rejects_failed_coverage_even_with_alias(tmp_path):
    manifest = tmp_path / "coverage_fail.json"
    manifest.write_text(
        json.dumps(
            {
                "strategy": "Microprice",
                "candidates": [
                    {
                        "slug": "hantavirus-pandemic-in-2026",
                        "source_strategy": "Microprice",
                        "coverage": {
                            "status": "no_coverage",
                            "book_events": 0,
                            "min_book_events": 50,
                            "gap_hours_missing": 0,
                        },
                    }
                ],
            }
        )
    )

    assert load_candidates(manifest, strategy="microprice_optimizer", max_candidates=5) == []


def test_load_candidates_prioritizes_non_extreme_tail_after_coverage_pass(tmp_path):
    manifest = tmp_path / "coverage_pass_priority.json"
    manifest.write_text(
        json.dumps(
            {
                "strategy": "Microprice",
                "candidates": [
                    {
                        "slug": "ultra-tail",
                        "source_strategy": "Microprice",
                        "scan_mid": 0.0035,
                        "liquidity": 1000000,
                        "coverage": {"status": "pass", "book_events": 100, "min_book_events": 50},
                    },
                    {
                        "slug": "non-extreme",
                        "source_strategy": "Microprice",
                        "scan_mid": 0.02,
                        "liquidity": 100,
                        "coverage": {"status": "pass", "book_events": 100, "min_book_events": 50},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    candidates = load_candidates(manifest, strategy="microprice_optimizer", max_candidates=2)

    assert [candidate.slug for candidate in candidates] == ["non-extreme", "ultra-tail"]


def test_load_candidates_preserves_candidate_coverage_window(tmp_path):
    manifest = tmp_path / "coverage_window.json"
    manifest.write_text(
        json.dumps(
            {
                "strategy": "Microprice",
                "candidates": [
                    {
                        "slug": "covered-window",
                        "source_strategy": "Microprice",
                        "coverage": {
                            "status": "pass",
                            "book_events": 80,
                            "min_book_events": 50,
                            "window": {
                                "start_time": "2026-03-22T08:00:00Z",
                                "end_time": "2026-03-22T09:00:00Z",
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    candidates = load_candidates(manifest, strategy="microprice_optimizer", max_candidates=1)

    assert candidates[0].coverage_start_time == "2026-03-22T08:00:00Z"
    assert candidates[0].coverage_end_time == "2026-03-22T09:00:00Z"
    assert candidates[0].coverage_book_events == 80
    assert candidates[0].coverage_min_book_events == 50


def test_load_candidates_preserves_manifest_recommended_windows(tmp_path):
    manifest = tmp_path / "microprice_handoff.json"
    manifest.write_text(
        json.dumps(
            {
                "strategy": "Microprice",
                "window": {
                    "start_time": "2026-05-10T08:00:00Z",
                    "end_time": "2026-05-10T09:00:00Z",
                    "min_book_events": 50,
                    "source": "known-pass-window",
                    "book_events": 108,
                },
                "min_book_events": 500,
                "coverage_first_guidance": {
                    "recommended_windows": [
                        {
                            "start_time": "2026-05-10T08:00:00Z",
                            "end_time": "2026-05-10T09:00:00Z",
                            "min_book_events": 50,
                        },
                        {
                            "start_time": "2026-05-11T14:00:00Z",
                            "end_time": "2026-05-11T15:00:00Z",
                            "min_book_events": 500,
                        },
                    ]
                },
                "candidates": [
                    {
                        "slug": "candidate-a",
                        "source_strategy": "Microprice",
                        "yes_mid": 0.02,
                        "yes_spread": 0.001,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    candidates = load_candidates(manifest, strategy="microprice_optimizer", max_candidates=1)

    assert candidates[0].coverage_start_time == "2026-05-10T08:00:00Z"
    assert candidates[0].coverage_end_time == "2026-05-10T09:00:00Z"
    assert candidates[0].coverage_min_book_events == 50
    assert [
        (window.start_time, window.end_time, window.min_book_events)
        for window in candidates[0].replay_windows
    ] == [
        ("2026-05-10T08:00:00Z", "2026-05-10T09:00:00Z", 50),
        ("2026-05-11T14:00:00Z", "2026-05-11T15:00:00Z", 500),
    ]


def test_load_candidates_preserves_manifest_recommended_pmxt_windows(tmp_path):
    manifest = tmp_path / "microprice_pmxt_handoff.json"
    manifest.write_text(
        json.dumps(
            {
                "strategy": "Microprice",
                "window": {
                    "start_time": "2026-05-11T23:04:20Z",
                    "end_time": "2026-05-12T00:04:20Z",
                    "label": "last_1h_minus_30m",
                },
                "min_book_events": 200,
                "coverage_first_guidance": {
                    "recommended_pmxt_windows": [
                        {
                            "start_time": "2026-05-11T23:04:20Z",
                            "end_time": "2026-05-12T00:04:20Z",
                            "label": "last_1h_minus_30m",
                        },
                        {
                            "start_time": "2026-05-11T21:04:20Z",
                            "end_time": "2026-05-12T00:04:20Z",
                            "label": "last_3h_minus_30m",
                        },
                    ]
                },
                "candidates": [
                    {
                        "slug": "candidate-a",
                        "source_strategy": "Microprice",
                        "yes_mid": 0.02,
                        "yes_spread": 0.001,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    candidates = load_candidates(manifest, strategy="microprice_optimizer", max_candidates=1)

    assert [
        (window.start_time, window.end_time, window.min_book_events, window.source)
        for window in candidates[0].replay_windows
    ] == [
        (
            "2026-05-11T23:04:20Z",
            "2026-05-12T00:04:20Z",
            200,
            "last_1h_minus_30m",
        ),
        (
            "2026-05-11T21:04:20Z",
            "2026-05-12T00:04:20Z",
            200,
            "last_3h_minus_30m",
        ),
    ]


def test_load_candidates_accepts_market_scan_strategy_tags_and_compact_scalar_books(tmp_path):
    manifest = tmp_path / "market_scan.json"
    manifest.write_text(
        json.dumps(
            {
                "metadata": {"utc_timestamp": "2026-05-11T14:03:46Z"},
                "candidates": [
                    {
                        "slug": "microprice-market",
                        "question": "Microprice?",
                        "strategy_fits": [
                            "Microprice",
                            "low-fill-probability liquidity-reward maker",
                        ],
                        "yes_token_id": "yes-token",
                        "no_token_id": "no-token",
                        "yes_book": "0.011",
                        "no_book": 0.989,
                        "liquidityNum": 1_000_000,
                    },
                    {
                        "slug": "reward-only-market",
                        "question": "Reward only?",
                        "strategy_fits": ["low-fill-probability liquidity-reward maker"],
                        "yes_token_id": "yes-token-2",
                        "no_token_id": "no-token-2",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    candidates = load_candidates(manifest, strategy="microprice_optimizer", max_candidates=5)

    assert len(candidates) == 1
    assert candidates[0].slug == "microprice-market"
    assert candidates[0].source_strategy == "Microprice"
    assert candidates[0].scan_mid == 0.011
    assert candidates[0].liquidity == 1_000_000


def test_select_latest_non_empty_pass_manifest_skips_zero_candidate_files(tmp_path):
    newer_zero = tmp_path / "job_B_pmxt_l2_coverage_pass_20260511T083729Z.json"
    older_non_empty = tmp_path / "job_B_pmxt_l2_coverage_pass_20260511T004317Z.json"
    newer_zero.write_text(
        json.dumps(
            {
                "strategy": "Microprice",
                "candidate_count": 0,
                "candidates": [],
            }
        ),
        encoding="utf-8",
    )
    older_non_empty.write_text(
        json.dumps(
            {
                "strategy": "Microprice",
                "candidate_count": 1,
                "window": {
                    "start_time": "2026-05-10T08:00:00Z",
                    "end_time": "2026-05-10T09:00:00Z",
                },
                "min_book_events": 50,
                "candidates": [
                    {
                        "slug": "covered-window",
                        "source_strategy": "Microprice",
                        "coverage": {
                            "status": "pass",
                            "book_events": 108,
                            "min_book_events": 50,
                            "window": {
                                "start_time": "2026-05-10T08:00:00Z",
                                "end_time": "2026-05-10T09:00:00Z",
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    os.utime(older_non_empty, (100.0, 100.0))
    os.utime(newer_zero, (200.0, 200.0))

    selected, records = job_b_microprice_batch.select_latest_non_empty_pass_manifest(
        str(tmp_path / "job_B_pmxt_l2_coverage_pass_*.json"),
        strategy="microprice_optimizer",
    )

    assert selected == older_non_empty
    assert records[0]["path"] == str(newer_zero)
    assert records[0]["reason"] == "skipped_zero_candidates"
    assert records[0]["classification"] == "no_pass"
    assert records[1]["path"] == str(older_non_empty)
    assert records[1]["reason"] == "selected_newest_non_empty_pass_manifest"
    assert records[1]["classification"] == "pass"


def test_run_batch_uses_pass_manifest_exact_window_and_min_book_events(monkeypatch, tmp_path):
    manifest = tmp_path / "coverage_window.json"
    manifest.write_text(
        json.dumps(
            {
                "strategy": "Microprice",
                "candidate_count": 1,
                "window": {
                    "start_time": "2026-05-10T08:00:00Z",
                    "end_time": "2026-05-10T09:00:00Z",
                },
                "min_book_events": 50,
                "candidates": [
                    {
                        "slug": "covered-window",
                        "source_strategy": "Microprice",
                        "coverage": {
                            "status": "pass",
                            "book_events": 108,
                            "min_book_events": 50,
                            "window": {
                                "start_time": "2026-05-10T07:00:00Z",
                                "end_time": "2026-05-10T08:00:00Z",
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calls = []

    async def _fake_run_attempt(candidate, params, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return job_b_microprice_batch.BacktestAttempt(
            slug=candidate.slug,
            question=candidate.question,
            token_index=candidate.token_index,
            source_strategy=candidate.source_strategy,
            params=params,
            status="completed",
            result={"fills": 0, "pnl": 0.0, "book_events": 108},
            error=None,
            diagnostics={
                "window": {
                    "start_time": kwargs["start_time"],
                    "end_time": kwargs["end_time"],
                },
                "min_book_events": kwargs["min_book_events"],
                "suspected_causes": [],
            },
        )

    monkeypatch.setattr(job_b_microprice_batch, "run_attempt", _fake_run_attempt)
    args = Namespace(
        manifest=manifest,
        strategy="microprice_optimizer",
        max_candidates=1,
        max_param_sets=1,
        start_time="2026-05-10T20:00:00Z",
        end_time="2026-05-10T21:00:00Z",
        min_book_events=500,
        per_attempt_timeout_secs=5,
        manifest_selection=[{"path": str(manifest), "selected": True}],
    )

    summary = asyncio.run(job_b_microprice_batch.run_batch(args))

    assert calls == [
        {
            "start_time": "2026-05-10T08:00:00Z",
            "end_time": "2026-05-10T09:00:00Z",
            "min_book_events": 50,
        }
    ]
    assert summary["attempt_count"] == 1
    assert summary["window"] == {
        "start_time": "2026-05-10T08:00:00Z",
        "end_time": "2026-05-10T09:00:00Z",
    }
    assert summary["requested_window"] == {
        "start_time": "2026-05-10T20:00:00Z",
        "end_time": "2026-05-10T21:00:00Z",
    }
    assert summary["min_book_events"] == 50
    assert summary["requested_min_book_events"] == 500
    assert summary["selected_min_book_events"] == 50
    assert summary["exact_window_status"] == "verified"
    assert summary["classification"] == "diagnostic_only"
    assert summary["pass_manifest_status"] == "pass"
    assert summary["warnings"] == []
    assert summary["candidate_replay_requests"][0]["window_provenance"] == "manifest_window"
    assert summary["safety"]["orders_submitted"] is False
    assert summary["safety"]["orders_signed"] is False
    assert summary["safety"]["credentials_required"] is False
    assert summary["safety"]["worker_trading_started"] is False
    assert summary["safety"]["live_trading_worker_started"] is False


def test_run_batch_attempts_multiple_candidates_and_manifest_windows(monkeypatch, tmp_path):
    manifest = tmp_path / "microprice_multi_window.json"
    manifest.write_text(
        json.dumps(
            {
                "strategy": "Microprice",
                "window": {
                    "start_time": "2026-05-10T08:00:00Z",
                    "end_time": "2026-05-10T09:00:00Z",
                    "min_book_events": 50,
                },
                "coverage_first_guidance": {
                    "recommended_windows": [
                        {
                            "start_time": "2026-05-10T08:00:00Z",
                            "end_time": "2026-05-10T09:00:00Z",
                            "min_book_events": 50,
                        },
                        {
                            "start_time": "2026-05-11T14:00:00Z",
                            "end_time": "2026-05-11T15:00:00Z",
                            "min_book_events": 500,
                        },
                    ]
                },
                "candidates": [
                    {
                        "slug": "candidate-a",
                        "source_strategy": "Microprice",
                        "yes_mid": 0.02,
                        "yes_spread": 0.001,
                    },
                    {
                        "slug": "candidate-b",
                        "source_strategy": "Microprice",
                        "yes_mid": 0.03,
                        "yes_spread": 0.001,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    calls = []

    async def _fake_run_attempt(candidate, params, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((candidate.slug, kwargs))
        return job_b_microprice_batch.BacktestAttempt(
            slug=candidate.slug,
            question=candidate.question,
            token_index=candidate.token_index,
            source_strategy=candidate.source_strategy,
            params=params,
            status="completed",
            result={"fills": 0, "pnl": 0.0, "book_events": kwargs["min_book_events"]},
            error=None,
            diagnostics={
                "window": {
                    "start_time": kwargs["start_time"],
                    "end_time": kwargs["end_time"],
                },
                "min_book_events": kwargs["min_book_events"],
                "fills": 0,
                "pnl": 0.0,
                "strategy_order_count": 0,
                "tail_bucket": "non_extreme_tail",
                "no_order": {
                    "primary_cause": "fill_opportunity_blocker",
                    "causes": ["fill_opportunity_blocker"],
                    "cause_counts": {"fill_opportunity_blocker": 1},
                    "blockers": {
                        "tick_cost": {
                            "blocked": False,
                            "spread_to_mid_ratio": 0.05,
                        }
                    },
                },
                "suspected_causes": ["zero_fills_with_coverage"],
            },
        )

    monkeypatch.setattr(job_b_microprice_batch, "run_attempt", _fake_run_attempt)
    args = Namespace(
        manifest=manifest,
        strategy="microprice_optimizer",
        max_candidates=2,
        max_param_sets=1,
        start_time="2026-05-10T20:00:00Z",
        end_time="2026-05-10T21:00:00Z",
        min_book_events=999,
        per_attempt_timeout_secs=5,
        window_policy="all",
    )

    summary = asyncio.run(job_b_microprice_batch.run_batch(args))

    assert len(calls) == 4
    assert summary["candidate_count"] == 2
    assert summary["attempt_count"] == 4
    assert summary["completed"] == 4
    assert summary["skipped_no_coverage"] == 0
    assert summary["errors"] == 0
    assert summary["exact_window_status"] == "verified"
    assert [
        (window["window"]["start_time"], window["window"]["end_time"], window["min_book_events"])
        for window in summary["selected_windows"]
    ] == [
        ("2026-05-10T08:00:00Z", "2026-05-10T09:00:00Z", 50),
        ("2026-05-11T14:00:00Z", "2026-05-11T15:00:00Z", 500),
    ]
    assert summary["tail_bucket_counts"] == {"non_extreme_tail": 4}
    assert summary["tick_cost_buckets"] == {"not_blocked": 4}
    assert summary["fills_orders_pnl"]["total_fills"] == 0
    assert summary["fills_orders_pnl"]["total_strategy_orders"] == 0
    assert summary["safety"]["orders_cancelled"] is False


def test_run_batch_can_use_subprocess_attempt_isolation(monkeypatch, tmp_path):
    manifest = tmp_path / "microprice_isolated.json"
    manifest.write_text(
        json.dumps(
            {
                "strategy": "Microprice",
                "candidate_count": 1,
                "candidates": [
                    {
                        "slug": "isolated-candidate",
                        "source_strategy": "Microprice",
                        "coverage": {
                            "status": "pass",
                            "book_events": 108,
                            "min_book_events": 50,
                            "window": {
                                "start_time": "2026-05-10T08:00:00Z",
                                "end_time": "2026-05-10T09:00:00Z",
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calls = []

    class _FakeProcess:
        def __init__(self, cmd):  # type: ignore[no-untyped-def]
            self.cmd = list(cmd)
            self.returncode = 0

        async def communicate(self):  # type: ignore[no-untyped-def]
            input_path = self.cmd[self.cmd.index("--single-attempt-input") + 1]
            output_path = self.cmd[self.cmd.index("--single-attempt-output") + 1]
            with open(input_path, encoding="utf-8") as fh:
                request = json.load(fh)
            calls.append(request)
            replay_window = request["replay_request"]["window"]
            attempt = job_b_microprice_batch.BacktestAttempt(
                slug=request["candidate"]["slug"],
                question=request["candidate"]["question"],
                token_index=request["candidate"]["token_index"],
                source_strategy=request["candidate"]["source_strategy"],
                params=request["params"],
                status="completed",
                result={"fills": 1, "pnl": -0.1, "book_events": 108},
                error=None,
                diagnostics={
                    "window": replay_window,
                    "min_book_events": request["replay_request"]["min_book_events"],
                    "fills": 1,
                    "pnl": -0.1,
                    "strategy_order_count": 1,
                    "tail_bucket": "unknown",
                    "no_order": {"causes": [], "blockers": {}},
                    "suspected_causes": [],
                },
            )
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(job_b_microprice_batch.asdict(attempt), fh)
            return b"isolated child complete", b""

        def kill(self):  # type: ignore[no-untyped-def]
            self.returncode = -9

        async def wait(self):  # type: ignore[no-untyped-def]
            return self.returncode

    async def _fake_create_subprocess_exec(*cmd, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["stdout"] == asyncio.subprocess.PIPE
        assert kwargs["stderr"] == asyncio.subprocess.PIPE
        assert "env" in kwargs
        return _FakeProcess(cmd)

    monkeypatch.setattr(
        job_b_microprice_batch.asyncio,
        "create_subprocess_exec",
        _fake_create_subprocess_exec,
    )
    args = Namespace(
        manifest=manifest,
        strategy="microprice_optimizer",
        max_candidates=1,
        max_param_sets=1,
        start_time="2026-05-10T20:00:00Z",
        end_time="2026-05-10T21:00:00Z",
        min_book_events=500,
        per_attempt_timeout_secs=5,
        window_policy="candidate",
        attempt_isolation="subprocess",
    )

    summary = asyncio.run(job_b_microprice_batch.run_batch(args))

    assert len(calls) == 1
    assert calls[0]["candidate"]["slug"] == "isolated-candidate"
    assert calls[0]["replay_request"]["window"] == {
        "start_time": "2026-05-10T08:00:00Z",
        "end_time": "2026-05-10T09:00:00Z",
    }
    assert summary["attempt_isolation"] == "subprocess"
    assert summary["attempt_count"] == 1
    assert summary["completed"] == 1
    assert summary["skipped"] == 0
    assert summary["errors"] == 0
    assert summary["exact_window_status"] == "verified"
    assert summary["safety"]["orders_submitted"] is False
    assert summary["safety"]["credentials_required"] is False


def test_run_batch_fail_closes_missing_pmxt_coverage(monkeypatch, tmp_path):
    manifest = tmp_path / "microprice_no_coverage.json"
    manifest.write_text(
        json.dumps(
            {
                "strategy": "Microprice",
                "window": {
                    "start_time": "2026-05-10T08:00:00Z",
                    "end_time": "2026-05-10T09:00:00Z",
                    "min_book_events": 500,
                },
                "candidates": [
                    {
                        "slug": "candidate-no-coverage",
                        "source_strategy": "Microprice",
                        "yes_mid": 0.02,
                        "yes_spread": 0.001,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    async def _fake_run_attempt(candidate, params, **kwargs):  # type: ignore[no-untyped-def]
        diagnostics = job_b_microprice_batch._diagnose_attempt(
            candidate,
            params,
            None,
            start_time=kwargs["start_time"],
            end_time=kwargs["end_time"],
            min_book_events=kwargs["min_book_events"],
            status="skipped_no_coverage",
        )
        return job_b_microprice_batch.BacktestAttempt(
            slug=candidate.slug,
            question=candidate.question,
            token_index=candidate.token_index,
            source_strategy=candidate.source_strategy,
            params=params,
            status="skipped_no_coverage",
            result=None,
            error=None,
            diagnostics=diagnostics,
        )

    monkeypatch.setattr(job_b_microprice_batch, "run_attempt", _fake_run_attempt)
    args = Namespace(
        manifest=manifest,
        strategy="microprice_optimizer",
        max_candidates=1,
        max_param_sets=1,
        start_time="2026-05-10T20:00:00Z",
        end_time="2026-05-10T21:00:00Z",
        min_book_events=50,
        per_attempt_timeout_secs=5,
        window_policy="candidate",
    )

    summary = asyncio.run(job_b_microprice_batch.run_batch(args))

    assert summary["completed"] == 0
    assert summary["skipped_no_coverage"] == 1
    assert summary["errors"] == 0
    assert summary["attempts"][0]["diagnostics"]["suspected_causes"] == ["coverage_insufficient"]
    assert summary["diagnostics"]["profit_opportunity_demonstrated"] is False


def test_diagnose_attempt_distinguishes_no_order_blocker_buckets():
    candidate = job_b_microprice_batch.Candidate(
        slug="covered-window",
        question="Covered market",
        token_index=0,
        condition_id=None,
        scan_mid=0.0035,
        scan_spread=0.001,
        scan_imbalance5=None,
        liquidity=1_000_000.0,
        source_strategy="Microprice",
        coverage_book_events=108,
        coverage_min_book_events=50,
    )
    params = {
        "depth_levels": 1,
        "entry_imbalance": 0.55,
        "exit_imbalance": 0.50,
        "min_microprice_edge": 0.0005,
        "quote_lifetime_seconds": 10.0,
    }
    result = {
        "fills": 0,
        "pnl": 0.0,
        "book_events": 108,
        "portfolio_stats": {"total_orders": 0},
        "strategy_diagnostics": {
            "book_signal_count": 108,
            "flat_evaluation_count": 108,
            "entry_signal_count": 0,
            "entry_block_counts": {"microprice_edge": 108, "spread": 0},
            "observed": {"min_spread": 0.001, "max_microprice_edge": 0.0},
            "thresholds": {"max_spread": 0.05, "min_microprice_edge": 0.0005},
        },
    }

    diagnostics = job_b_microprice_batch._diagnose_attempt(
        candidate,
        params,
        result,
        start_time="2026-05-10T08:00:00Z",
        end_time="2026-05-10T09:00:00Z",
        min_book_events=50,
        status="completed",
    )

    blockers = diagnostics["no_order"]["blockers"]
    assert blockers["tick_cost"]["blocked"] is True
    assert blockers["edge"]["blocked"] is True
    assert blockers["spread"]["blocked"] is False
    assert blockers["queue"]["blocked"] is False
    assert blockers["fill_opportunity"]["blocked"] is True
    assert diagnostics["no_order_primary_cause"] == "edge_blocker"
    assert "tick_cost_blocker" in diagnostics["no_order_causes"]


def test_diagnose_attempt_distinguishes_spread_and_queue_blockers():
    candidate = job_b_microprice_batch.Candidate(
        slug="covered-window",
        question="Covered market",
        token_index=0,
        condition_id=None,
        scan_mid=0.5,
        scan_spread=0.01,
        scan_imbalance5=None,
        liquidity=1_000_000.0,
        source_strategy="Microprice",
    )
    params = {
        "depth_levels": 1,
        "entry_imbalance": 0.55,
        "exit_imbalance": 0.50,
        "min_microprice_edge": 0.0005,
        "quote_lifetime_seconds": 10.0,
    }
    spread_result = {
        "fills": 0,
        "pnl": 0.0,
        "book_events": 108,
        "portfolio_stats": {"total_orders": 0},
        "strategy_diagnostics": {
            "book_signal_count": 108,
            "flat_evaluation_count": 108,
            "entry_signal_count": 0,
            "entry_block_counts": {"spread": 108, "microprice_edge": 0},
            "observed": {"min_spread": 0.06, "max_microprice_edge": 0.003},
            "thresholds": {"max_spread": 0.05, "min_microprice_edge": 0.0005},
        },
    }
    queue_result = {
        "fills": 0,
        "pnl": 0.0,
        "book_events": 108,
        "portfolio_stats": {"total_orders": 2},
        "strategy_diagnostics": {
            "book_signal_count": 108,
            "flat_evaluation_count": 108,
            "entry_signal_count": 2,
            "entry_block_counts": {"spread": 0, "microprice_edge": 0},
            "observed": {"min_spread": 0.01, "max_microprice_edge": 0.003},
            "thresholds": {"max_spread": 0.05, "min_microprice_edge": 0.0005},
        },
    }

    spread_diagnostics = job_b_microprice_batch._diagnose_attempt(
        candidate,
        params,
        spread_result,
        start_time="2026-05-10T08:00:00Z",
        end_time="2026-05-10T09:00:00Z",
        min_book_events=50,
        status="completed",
    )
    queue_diagnostics = job_b_microprice_batch._diagnose_attempt(
        candidate,
        params,
        queue_result,
        start_time="2026-05-10T08:00:00Z",
        end_time="2026-05-10T09:00:00Z",
        min_book_events=50,
        status="completed",
    )
    aggregate = job_b_microprice_batch._aggregate_diagnostics(
        [
            job_b_microprice_batch.BacktestAttempt(
                slug="spread",
                question="spread",
                token_index=0,
                source_strategy="Microprice",
                params=params,
                status="completed",
                result=spread_result,
                error=None,
                diagnostics=spread_diagnostics,
            ),
            job_b_microprice_batch.BacktestAttempt(
                slug="queue",
                question="queue",
                token_index=0,
                source_strategy="Microprice",
                params=params,
                status="completed",
                result=queue_result,
                error=None,
                diagnostics=queue_diagnostics,
            ),
        ]
    )

    assert spread_diagnostics["no_order"]["blockers"]["spread"]["blocked"] is True
    assert spread_diagnostics["no_order"]["blockers"]["queue"]["blocked"] is False
    assert queue_diagnostics["no_order"]["blockers"]["queue"]["blocked"] is True
    assert queue_diagnostics["no_order"]["blockers"]["fill_opportunity"]["blocked"] is False
    assert aggregate["no_order_cause_counts"]["spread_blocker"] == 1
    assert aggregate["no_order_cause_counts"]["queue_blocker"] == 1


def test_diagnose_attempt_attributes_filled_negative_pnl_fail_closed():
    candidate = job_b_microprice_batch.Candidate(
        slug="filled-loss",
        question="Filled loss market",
        token_index=0,
        condition_id=None,
        scan_mid=0.0125,
        scan_spread=0.001,
        scan_imbalance5=None,
        liquidity=1_000_000.0,
        source_strategy="Microprice",
    )
    params = {
        "depth_levels": 1,
        "entry_imbalance": 0.55,
        "exit_imbalance": 0.50,
        "min_microprice_edge": 0.0005,
        "quote_lifetime_seconds": 10.0,
    }
    result = {
        "fills": 2,
        "pnl": -0.28315,
        "book_events": 179,
        "last": 0.0095,
        "fill_events": [
            {
                "action": "buy",
                "price": 0.06,
                "quantity": 5.0,
                "commission": 0.0282,
                "timestamp": "2026-05-25T08:01:32.339000+00:00",
            },
            {
                "action": "sell",
                "price": 0.01,
                "quantity": 5.0,
                "commission": 0.00495,
                "timestamp": "2026-05-25T08:02:17.569000+00:00",
            },
        ],
        "portfolio_stats": {"total_orders": 2},
        "strategy_diagnostics": {
            "book_signal_count": 177,
            "flat_evaluation_count": 174,
            "entry_signal_count": 1,
            "entry_block_counts": {
                "microprice_edge": 169,
                "spread": 3,
                "pending_order": 2,
                "reentry_cooldown_seconds": 1,
            },
            "observed": {
                "min_spread": 0.001,
                "max_spread": 0.59,
                "max_microprice_edge": 0.2886326247,
            },
            "thresholds": {"max_spread": 0.05, "min_microprice_edge": 0.0005},
        },
    }

    diagnostics = job_b_microprice_batch._diagnose_attempt(
        candidate,
        params,
        result,
        start_time="2026-05-25T08:00:00Z",
        end_time="2026-05-25T09:00:00Z",
        min_book_events=50,
        status="completed",
    )
    attribution = diagnostics["negative_pnl_attribution"]
    aggregate = job_b_microprice_batch._aggregate_diagnostics(
        [
            job_b_microprice_batch.BacktestAttempt(
                slug="filled-loss",
                question="filled",
                token_index=0,
                source_strategy="Microprice",
                params=params,
                status="completed",
                result=result,
                error=None,
                diagnostics=diagnostics,
            )
        ]
    )

    assert attribution["eligible"] is True
    assert attribution["fail_closed"] is True
    assert attribution["primary_cause"] == "adverse_selection_markout"
    assert "adverse_selection_markout" in attribution["causes"]
    assert "spread_tick_cost_too_large" in attribution["causes"]
    assert "queue_fill_timing" in attribution["causes"]
    assert "parameter_candidate_bucket" in attribution["causes"]
    assert round(attribution["adverse_selection_markout"]["round_trip_price_edge"], 6) == -0.05
    assert attribution["spread_tick_cost"]["spread_tick_cost_too_large"] is True
    assert attribution["queue_fill_timing"]["queue_or_fill_timing_suspected"] is True
    summary = aggregate["negative_pnl_attribution_summary"]
    assert summary["eligible"] is True
    assert summary["classification"] == "non_positive_pnl_after_orders_or_fills"
    assert summary["cause_counts"]["adverse_selection_markout"] == 1
    assert summary["cause_counts"]["spread_tick_cost_too_large"] == 1
    assert summary["cause_counts"]["queue_fill_timing"] == 1


def test_run_batch_fail_closes_on_exact_window_mismatch(monkeypatch, tmp_path):
    manifest = tmp_path / "coverage_window.json"
    manifest.write_text(
        json.dumps(
            {
                "strategy": "Microprice",
                "candidate_count": 1,
                "candidates": [
                    {
                        "slug": "covered-window",
                        "source_strategy": "Microprice",
                        "coverage": {
                            "status": "pass",
                            "book_events": 108,
                            "min_book_events": 50,
                            "window": {
                                "start_time": "2026-05-10T08:00:00Z",
                                "end_time": "2026-05-10T09:00:00Z",
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    async def _fake_run_attempt(candidate, params, **kwargs):  # type: ignore[no-untyped-def]
        return job_b_microprice_batch.BacktestAttempt(
            slug=candidate.slug,
            question=candidate.question,
            token_index=candidate.token_index,
            source_strategy=candidate.source_strategy,
            params=params,
            status="completed",
            result={"fills": 0, "pnl": 0.0, "book_events": 108},
            error=None,
            diagnostics={
                "window": {
                    "start_time": "2026-05-10T20:00:00Z",
                    "end_time": "2026-05-10T21:00:00Z",
                },
                "min_book_events": 50,
                "suspected_causes": [],
            },
        )

    monkeypatch.setattr(job_b_microprice_batch, "run_attempt", _fake_run_attempt)
    args = Namespace(
        manifest=manifest,
        strategy="microprice_optimizer",
        max_candidates=1,
        max_param_sets=1,
        start_time="2026-05-10T20:00:00Z",
        end_time="2026-05-10T21:00:00Z",
        min_book_events=500,
        per_attempt_timeout_secs=5,
    )

    summary = asyncio.run(job_b_microprice_batch.run_batch(args))

    assert summary["exact_window_status"] == "fail_closed"
    assert summary["classification"] == "blocked"
    assert "exact_window_mismatch" in summary["warnings"]
    assert "exact_window_metadata_blocker" in summary["warnings"]
    assert summary["blockers"][0]["type"] == "exact_window_metadata_blocker"


def test_validate_artifact_fail_closes_against_selected_manifest_window(tmp_path):
    manifest = tmp_path / "coverage_window.json"
    artifact = tmp_path / "job_b_artifact.json"
    manifest.write_text(
        json.dumps(
            {
                "strategy": "Microprice",
                "candidate_count": 1,
                "window": {
                    "start_time": "2026-05-10T08:00:00Z",
                    "end_time": "2026-05-10T09:00:00Z",
                },
                "min_book_events": 50,
                "candidates": [
                    {
                        "slug": "covered-window",
                        "source_strategy": "Microprice",
                        "coverage": {
                            "status": "pass",
                            "book_events": 108,
                            "min_book_events": 50,
                            "window": {
                                "start_time": "2026-05-10T07:00:00Z",
                                "end_time": "2026-05-10T08:00:00Z",
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    artifact.write_text(
        json.dumps(
            {
                "window": {
                    "start_time": "2026-05-10T07:00:00Z",
                    "end_time": "2026-05-10T08:00:00Z",
                },
                "min_book_events": 50,
                "attempts": [
                    {
                        "slug": "covered-window",
                        "token_index": 0,
                        "diagnostics": {
                            "window": {
                                "start_time": "2026-05-10T07:00:00Z",
                                "end_time": "2026-05-10T08:00:00Z",
                            },
                            "min_book_events": 50,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = job_b_microprice_batch.build_exact_window_validation_report(
        manifest_path=manifest,
        artifact_path=artifact,
        command=["python", "scripts/job_b_microprice_batch.py", "--validate-artifact"],
    )

    assert report["classification"] == "blocked"
    assert report["exact_window_status"] == "fail_closed"
    assert report["pass_manifest_status"] == "pass"
    assert report["expected_selected_windows"][0]["window"] == {
        "start_time": "2026-05-10T08:00:00Z",
        "end_time": "2026-05-10T09:00:00Z",
    }
    assert "exact_window_metadata_blocker" in report["warnings"]
    assert report["blockers"][0]["type"] == "exact_window_metadata_blocker"


def test_validate_artifact_derives_negative_pnl_attribution_from_legacy_artifact(tmp_path):
    manifest = tmp_path / "coverage_window.json"
    artifact = tmp_path / "job_b_artifact.json"
    manifest.write_text(
        json.dumps(
            {
                "strategy": "Microprice",
                "candidate_count": 1,
                "window": {
                    "start_time": "2026-05-25T08:00:00Z",
                    "end_time": "2026-05-25T09:00:00Z",
                },
                "min_book_events": 50,
                "candidates": [
                    {
                        "slug": "filled-loss",
                        "source_strategy": "Microprice",
                        "coverage": {
                            "status": "pass",
                            "book_events": 179,
                            "min_book_events": 50,
                            "window": {
                                "start_time": "2026-05-25T08:00:00Z",
                                "end_time": "2026-05-25T09:00:00Z",
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    artifact.write_text(
        json.dumps(
            {
                "classification": "diagnostic_only",
                "window": {
                    "start_time": "2026-05-25T08:00:00Z",
                    "end_time": "2026-05-25T09:00:00Z",
                },
                "min_book_events": 50,
                "fills_orders_pnl": {
                    "total_fills": 2,
                    "total_strategy_orders": 2,
                    "completed_pnl_sum": -0.28315,
                    "completed_positive_pnl_attempts": 0,
                    "completed_negative_pnl_attempts": 1,
                    "completed_zero_pnl_attempts": 0,
                },
                "attempts": [
                    {
                        "slug": "filled-loss",
                        "token_index": 0,
                        "source_strategy": "Microprice",
                        "status": "completed",
                        "params": {
                            "depth_levels": 1,
                            "entry_imbalance": 0.55,
                            "exit_imbalance": 0.50,
                            "min_microprice_edge": 0.0005,
                            "quote_lifetime_seconds": 10.0,
                        },
                        "result": {
                            "fills": 2,
                            "pnl": -0.28315,
                            "last": 0.0095,
                            "fill_events": [
                                {
                                    "action": "buy",
                                    "price": 0.06,
                                    "quantity": 5.0,
                                    "commission": 0.0282,
                                    "timestamp": "2026-05-25T08:01:32.339000+00:00",
                                },
                                {
                                    "action": "sell",
                                    "price": 0.01,
                                    "quantity": 5.0,
                                    "commission": 0.00495,
                                    "timestamp": "2026-05-25T08:02:17.569000+00:00",
                                },
                            ],
                        },
                        "diagnostics": {
                            "window": {
                                "start_time": "2026-05-25T08:00:00Z",
                                "end_time": "2026-05-25T09:00:00Z",
                            },
                            "min_book_events": 50,
                            "fills": 2,
                            "pnl": -0.28315,
                            "strategy_order_count": 2,
                            "scan_mid": 0.0125,
                            "scan_spread": 0.001,
                            "tail_bucket": "non_extreme_tail",
                            "strategy_diagnostics": {
                                "entry_signal_count": 1,
                                "entry_block_counts": {
                                    "spread": 3,
                                    "pending_order": 2,
                                    "reentry_cooldown_seconds": 1,
                                },
                                "observed": {
                                    "min_spread": 0.001,
                                    "max_spread": 0.59,
                                },
                                "thresholds": {"max_spread": 0.05},
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = job_b_microprice_batch.build_exact_window_validation_report(
        manifest_path=manifest,
        artifact_path=artifact,
        command=["python", "scripts/job_b_microprice_batch.py", "--validate-artifact"],
    )

    summary = report["negative_pnl_attribution_summary"]
    assert report["classification"] == "diagnostic_only"
    assert report["exact_window_status"] == "verified"
    assert report["live_ready"] is False
    assert summary["eligible"] is True
    assert summary["classification"] == "non_positive_pnl_after_orders_or_fills"
    assert summary["cause_counts"]["adverse_selection_markout"] == 1
    assert summary["cause_counts"]["spread_tick_cost_too_large"] == 1
    assert summary["cause_counts"]["queue_fill_timing"] == 1
