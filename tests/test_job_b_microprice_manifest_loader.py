from __future__ import annotations

import json

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
