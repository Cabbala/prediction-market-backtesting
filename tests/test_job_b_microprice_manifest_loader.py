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
    assert records[1]["path"] == str(older_non_empty)
    assert records[1]["reason"] == "selected_newest_non_empty_pass_manifest"


def test_run_batch_uses_pass_manifest_exact_window_and_min_book_events(
    monkeypatch, tmp_path
):
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
    assert summary["warnings"] == []
    assert summary["safety"]["orders_submitted"] is False
    assert summary["safety"]["orders_signed"] is False
    assert summary["safety"]["credentials_required"] is False
    assert summary["safety"]["worker_trading_started"] is False
    assert summary["safety"]["live_trading_worker_started"] is False


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
    assert "exact_window_mismatch" in summary["warnings"]
