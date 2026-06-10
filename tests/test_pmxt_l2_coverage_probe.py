from __future__ import annotations

import asyncio
import csv
import json
import time
from argparse import Namespace
from pathlib import Path

from scripts import job_b_microprice_batch
from scripts import probe_pmxt_l2_coverage


def _write_source_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "batches": [
                    {
                        "strategy": "microprice_optimizer",
                        "markets": [
                            {
                                "market_slug": "covered-market",
                                "question": "Covered?",
                                "token_index": 0,
                                "scan_mid": 0.51,
                            },
                            {
                                "market_slug": "thin-market",
                                "question": "Thin?",
                                "token_index": 1,
                            },
                            {
                                "market_slug": "broken-market",
                                "question": "Broken?",
                                "token_index": 0,
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_probe_writes_outputs_and_pass_manifest_with_only_covered_candidates(
    monkeypatch,
    tmp_path,
) -> None:
    source_manifest = tmp_path / "source_manifest.json"
    _write_source_manifest(source_manifest)

    async def _fake_probe_candidate(candidate, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["sources"] == probe_pmxt_l2_coverage.DEFAULT_SOURCES
        if candidate.slug == "covered-market":
            return probe_pmxt_l2_coverage._result_from_candidate(
                candidate,
                status="pass",
                book_events=75,
                min_book_events=50,
                count_key="book_events",
                market_key="slug",
                market_id=candidate.slug,
                prices=(0.40, 0.55),
            )
        if candidate.slug == "thin-market":
            return probe_pmxt_l2_coverage._result_from_candidate(
                candidate,
                status="no_coverage",
                book_events=12,
                min_book_events=50,
                message="12 book events < 50 required",
            )
        return probe_pmxt_l2_coverage._result_from_candidate(
            candidate,
            status="error",
            book_events=0,
            min_book_events=50,
            message="HTTPError downloading PMXT raw parquet from r2.pmxt.dev: 503",
        )

    monkeypatch.setattr(probe_pmxt_l2_coverage, "probe_candidate", _fake_probe_candidate)

    args = Namespace(
        manifest=source_manifest,
        output_dir=tmp_path / "reports",
        pass_manifest_dir=tmp_path / "pass_manifests",
        start_time="2026-03-22T09:00:00Z",
        end_time="2026-03-22T10:00:00Z",
        strategy="microprice_optimizer",
        max_candidates=6,
        min_book_events=50,
        sources=None,
        timeout_seconds=5,
        recent_window_count=1,
        window_step_hours=1,
    )

    summary = asyncio.run(probe_pmxt_l2_coverage.run_probe(args))
    output_files = probe_pmxt_l2_coverage.write_outputs(
        summary,
        output_dir=args.output_dir,
        pass_manifest_dir=args.pass_manifest_dir,
        timestamp="20260504T000000Z",
    )

    assert summary["candidate_count"] == 3
    assert summary["pass_count"] == 1
    assert summary["no_coverage_count"] == 1
    assert summary["error_count"] == 1
    assert summary["diagnostic_counts"]["successful_pass_count"] == 1
    assert summary["diagnostic_counts"]["min_book_events_not_met"] == 1
    assert summary["diagnostic_counts"]["pmxt_raw_download_failure"] == 1
    assert Path(output_files["json"]).exists()
    assert Path(output_files["csv"]).exists()
    assert Path(output_files["markdown"]).exists()
    assert Path(output_files["pass_manifest"]).exists()

    csv_rows = list(csv.DictReader(Path(output_files["csv"]).open(encoding="utf-8")))
    assert [row["status"] for row in csv_rows] == ["pass", "no_coverage", "error"]
    assert csv_rows[0]["book_events"] == "75"

    pass_manifest = json.loads(Path(output_files["pass_manifest"]).read_text(encoding="utf-8"))
    assert pass_manifest["mode"] == "shadow/backtest-only"
    assert pass_manifest["orders_submitted"] is False
    assert pass_manifest["orders_signed"] is False
    assert pass_manifest["orders_cancelled"] is False
    assert pass_manifest["credentials_required"] is False
    assert pass_manifest["worker_trading_started"] is False
    assert pass_manifest["live_trading_worker_started"] is False
    assert pass_manifest["safety"]["orders_submitted"] is False
    assert pass_manifest["safety"]["orders_signed"] is False
    assert pass_manifest["safety"]["orders_cancelled"] is False
    assert pass_manifest["safety"]["credentials_required"] is False
    assert pass_manifest["safety"]["worker_trading_started"] is False
    assert pass_manifest["safety"]["live_trading_worker_started"] is False
    assert pass_manifest["source_manifest"] == str(source_manifest)
    assert pass_manifest["min_book_events"] == 50
    assert [candidate["market_slug"] for candidate in pass_manifest["candidates"]] == [
        "covered-market"
    ]
    assert pass_manifest["candidates"][0]["source_strategy"] == "microprice_optimizer"
    assert pass_manifest["candidates"][0]["coverage"]["book_events"] == 75
    assert pass_manifest["candidates"][0]["coverage"]["window"] == {
        "start_time": "2026-03-22T09:00:00Z",
        "end_time": "2026-03-22T10:00:00Z",
    }
    assert pass_manifest["candidates"][0]["coverage"]["diagnostic_category"] == (
        "successful_pass_count"
    )
    assert pass_manifest["candidates"][0]["safety"]["orders_submitted"] is False
    assert pass_manifest["candidates"][0]["orders_submitted"] is False
    md = Path(output_files["markdown"]).read_text(encoding="utf-8")
    assert "- orders_submitted=false" in md
    assert "- orders_signed=false" in md
    assert "- orders_cancelled=false" in md
    assert "- credentials_required=false" in md
    assert "- live_trading_worker_started=false" in md
    assert "- worker_trading_started=false" in md

    loaded = job_b_microprice_batch.load_candidates(
        Path(output_files["pass_manifest"]),
        strategy="microprice_optimizer",
        max_candidates=6,
    )
    assert [(candidate.slug, candidate.token_index) for candidate in loaded] == [
        ("covered-market", 0)
    ]
    assert loaded[0].source_strategy == "microprice_optimizer"


def test_process_timeout_preserves_partial_candidate_window_outputs(monkeypatch, tmp_path) -> None:
    source_manifest = tmp_path / "source_manifest.json"
    _write_source_manifest(source_manifest)

    async def _fake_probe_candidate(candidate, **kwargs):  # type: ignore[no-untyped-def]
        if candidate.slug == "covered-market":
            return probe_pmxt_l2_coverage._result_from_candidate(
                candidate,
                status="pass",
                book_events=75,
                min_book_events=50,
                window_start_time=kwargs["start_time"],
                window_end_time=kwargs["end_time"],
            )
        if candidate.slug == "broken-market":
            time.sleep(5)
        return probe_pmxt_l2_coverage._result_from_candidate(
            candidate,
            status="no_coverage",
            book_events=0,
            min_book_events=50,
            message="No PMXT L2 book replay was loaded for the requested window.",
            window_start_time=kwargs["start_time"],
            window_end_time=kwargs["end_time"],
        )

    monkeypatch.setattr(probe_pmxt_l2_coverage, "probe_candidate", _fake_probe_candidate)
    args = Namespace(
        manifest=source_manifest,
        output_dir=tmp_path / "reports",
        pass_manifest_dir=tmp_path / "pass_manifests",
        start_time="2026-03-22T09:00:00Z",
        end_time="2026-03-22T10:00:00Z",
        strategy="microprice_optimizer",
        max_candidates=3,
        min_book_events=50,
        sources=None,
        timeout_seconds=1,
        recent_window_count=1,
        window_step_hours=1,
        process_isolation=True,
        process_timeout_grace_seconds=0,
    )

    started_at = time.monotonic()
    summary = asyncio.run(probe_pmxt_l2_coverage.run_probe(args))
    elapsed = time.monotonic() - started_at
    output_files = probe_pmxt_l2_coverage.write_outputs(
        summary,
        output_dir=args.output_dir,
        pass_manifest_dir=args.pass_manifest_dir,
        timestamp="20260504T000006Z",
    )

    assert elapsed < 4
    assert summary["orders_submitted"] is False
    assert summary["orders_signed"] is False
    assert summary["orders_cancelled"] is False
    assert summary["credentials_required"] is False
    assert summary["worker_trading_started"] is False
    assert summary["live_trading_worker_started"] is False
    assert summary["pass_count"] == 1
    assert summary["no_coverage_count"] == 1
    assert summary["error_count"] == 1
    assert summary["diagnostic_counts"]["probe_runtime_timeout"] == 1
    assert [row["slug"] for row in summary["results"]] == [
        "covered-market",
        "thin-market",
        "broken-market",
    ]
    timeout_row = summary["results"][2]
    assert timeout_row["status"] == "error"
    assert timeout_row["diagnostic_category"] == "probe_runtime_timeout"
    assert timeout_row["coverage"]["window"] == {
        "start_time": "2026-03-22T09:00:00Z",
        "end_time": "2026-03-22T10:00:00Z",
    }
    assert timeout_row["safety"]["orders_submitted"] is False
    assert timeout_row["orders_submitted"] is False

    for artifact in output_files.values():
        assert Path(artifact).exists()
    json_report = json.loads(Path(output_files["json"]).read_text(encoding="utf-8"))
    assert json_report["diagnostic_counts"]["probe_runtime_timeout"] == 1
    assert json_report["results"][2]["coverage"]["window"]["start_time"] == ("2026-03-22T09:00:00Z")
    csv_rows = list(csv.DictReader(Path(output_files["csv"]).open(encoding="utf-8")))
    assert csv_rows[2]["diagnostic_category"] == "probe_runtime_timeout"
    assert csv_rows[2]["window_start_time"] == "2026-03-22T09:00:00Z"
    assert csv_rows[2]["orders_submitted"] == "False"
    pass_manifest = json.loads(Path(output_files["pass_manifest"]).read_text(encoding="utf-8"))
    assert pass_manifest["candidate_count"] == 1
    assert pass_manifest["orders_submitted"] is False
    assert pass_manifest["candidates"][0]["coverage"]["window"] == {
        "start_time": "2026-03-22T09:00:00Z",
        "end_time": "2026-03-22T10:00:00Z",
    }


def test_job_b_load_candidates_preserves_top_level_source_strategy(tmp_path) -> None:
    pass_manifest = tmp_path / "coverage_pass_manifest.json"
    pass_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "shadow/backtest-only",
                "strategy": "microprice_optimizer",
                "candidates": [
                    {
                        "market_slug": "pass-market",
                        "question": "Pass?",
                        "token_index": 1,
                        "source_strategy": "microprice_optimizer",
                        "coverage": {"status": "pass", "book_events": 51},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    candidates = job_b_microprice_batch.load_candidates(
        pass_manifest,
        strategy="microprice_optimizer",
        max_candidates=6,
    )

    assert len(candidates) == 1
    assert candidates[0].slug == "pass-market"
    assert candidates[0].question == "Pass?"
    assert candidates[0].token_index == 1
    assert candidates[0].source_strategy == "microprice_optimizer"


def test_gap_warning_prevents_pass_manifest_even_with_enough_events(monkeypatch, tmp_path) -> None:
    source_manifest = tmp_path / "source_manifest.json"
    _write_source_manifest(source_manifest)

    async def _fake_probe_candidate(candidate, **kwargs):  # type: ignore[no-untyped-def]
        return probe_pmxt_l2_coverage._result_from_candidate(
            candidate,
            status="no_coverage",
            book_events=100,
            min_book_events=50,
            message="1 PMXT archive hour(s) missing in requested/load window",
            gap_hours_missing=1,
            gap_warning="PMXT: 1 archive hour(s) missing for market x",
        )

    monkeypatch.setattr(probe_pmxt_l2_coverage, "probe_candidate", _fake_probe_candidate)
    args = Namespace(
        manifest=source_manifest,
        output_dir=tmp_path / "reports",
        pass_manifest_dir=tmp_path / "pass_manifests",
        start_time="2026-03-22T09:00:00Z",
        end_time="2026-03-22T10:00:00Z",
        strategy="microprice_optimizer",
        max_candidates=1,
        min_book_events=50,
        sources=None,
        timeout_seconds=5,
        recent_window_count=1,
        window_step_hours=1,
    )

    summary = asyncio.run(probe_pmxt_l2_coverage.run_probe(args))
    output_files = probe_pmxt_l2_coverage.write_outputs(
        summary,
        output_dir=args.output_dir,
        pass_manifest_dir=args.pass_manifest_dir,
        timestamp="20260504T000001Z",
    )

    assert summary["pass_count"] == 0
    assert summary["results"][0]["gap_hours_missing"] == 1
    pass_manifest = json.loads(Path(output_files["pass_manifest"]).read_text(encoding="utf-8"))
    assert pass_manifest["candidates"] == []


def test_job_b_load_candidates_filters_top_level_strategy_mismatch(tmp_path) -> None:
    pass_manifest = tmp_path / "coverage_pass_manifest_mismatch.json"
    pass_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "shadow/backtest-only",
                "strategy": "volatility_spike",
                "candidates": [
                    {
                        "market_slug": "wrong-strategy-market",
                        "question": "Wrong?",
                        "token_index": 0,
                        "source_strategy": "volatility_spike",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    candidates = job_b_microprice_batch.load_candidates(
        pass_manifest,
        strategy="microprice_optimizer",
        max_candidates=6,
    )
    assert candidates == []


def test_fail_on_no_pass_fails_even_with_zero_candidates(tmp_path) -> None:
    empty_manifest = tmp_path / "empty.json"
    empty_manifest.write_text(
        json.dumps({"strategy": "microprice_optimizer", "candidates": []}), encoding="utf-8"
    )
    args = probe_pmxt_l2_coverage.parse_args(
        [
            "--manifest",
            str(empty_manifest),
            "--start-time",
            "2026-03-22T09:00:00Z",
            "--end-time",
            "2026-03-22T10:00:00Z",
            "--output-dir",
            str(tmp_path / "reports"),
            "--pass-manifest-dir",
            str(tmp_path / "pass"),
            "--timestamp",
            "20260504T000002Z",
            "--fail-on-no-pass",
        ]
    )
    summary = asyncio.run(probe_pmxt_l2_coverage.run_probe(args))
    assert summary["candidate_count"] == 0
    assert summary["pass_count"] == 0
    assert summary["diagnostics"]["no_eligible_input_candidates"] is True
    assert summary["diagnostic_counts"]["no_eligible_input_candidates"] == 1
    # Exercise the final exit policy through main, not only run_probe.
    assert (
        probe_pmxt_l2_coverage.main(
            [
                "--manifest",
                str(empty_manifest),
                "--start-time",
                "2026-03-22T09:00:00Z",
                "--end-time",
                "2026-03-22T10:00:00Z",
                "--output-dir",
                str(tmp_path / "reports2"),
                "--pass-manifest-dir",
                str(tmp_path / "pass2"),
                "--timestamp",
                "20260504T000003Z",
                "--fail-on-no-pass",
            ]
        )
        == 3
    )


def test_job_b_load_candidates_rejects_non_pass_coverage(tmp_path) -> None:
    manifest = tmp_path / "coverage_not_pass.json"
    manifest.write_text(
        json.dumps(
            {
                "strategy": "microprice_optimizer",
                "candidates": [
                    {
                        "market_slug": "bad-coverage",
                        "source_strategy": "microprice_optimizer",
                        "coverage": {
                            "status": "no_coverage",
                            "book_events": 100,
                            "min_book_events": 50,
                        },
                    },
                    {
                        "market_slug": "gap-coverage",
                        "source_strategy": "microprice_optimizer",
                        "coverage": {
                            "status": "pass",
                            "book_events": 100,
                            "min_book_events": 50,
                            "gap_hours_missing": 1,
                        },
                    },
                    {
                        "market_slug": "thin-coverage",
                        "source_strategy": "microprice_optimizer",
                        "coverage": {"status": "pass", "book_events": 49, "min_book_events": 50},
                    },
                    {
                        "market_slug": "good-coverage",
                        "source_strategy": "microprice_optimizer",
                        "coverage": {
                            "status": "pass",
                            "book_events": 50,
                            "min_book_events": 50,
                            "gap_hours_missing": 0,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    candidates = job_b_microprice_batch.load_candidates(
        manifest,
        strategy="microprice_optimizer",
        max_candidates=10,
    )
    assert [candidate.slug for candidate in candidates] == ["good-coverage"]


def test_recent_window_count_probes_multiple_windows_and_keeps_candidate_window(
    monkeypatch, tmp_path
) -> None:
    source_manifest = tmp_path / "source_manifest.json"
    _write_source_manifest(source_manifest)

    async def _fake_probe_candidate(candidate, **kwargs):  # type: ignore[no-untyped-def]
        if candidate.slug == "covered-market" and kwargs["start_time"] == "2026-03-22T08:00:00Z":
            return probe_pmxt_l2_coverage._result_from_candidate(
                candidate,
                status="pass",
                book_events=80,
                min_book_events=50,
                window_start_time=kwargs["start_time"],
                window_end_time=kwargs["end_time"],
            )
        return probe_pmxt_l2_coverage._result_from_candidate(
            candidate,
            status="no_coverage",
            book_events=0,
            min_book_events=50,
            window_start_time=kwargs["start_time"],
            window_end_time=kwargs["end_time"],
        )

    monkeypatch.setattr(probe_pmxt_l2_coverage, "probe_candidate", _fake_probe_candidate)
    args = Namespace(
        manifest=source_manifest,
        output_dir=tmp_path / "reports",
        pass_manifest_dir=tmp_path / "pass_manifests",
        start_time="2026-03-22T09:00:00Z",
        end_time="2026-03-22T10:00:00Z",
        strategy="microprice_optimizer",
        max_candidates=1,
        min_book_events=50,
        sources=None,
        timeout_seconds=5,
        recent_window_count=2,
        window_step_hours=1,
    )

    summary = asyncio.run(probe_pmxt_l2_coverage.run_probe(args))
    output_files = probe_pmxt_l2_coverage.write_outputs(
        summary,
        output_dir=args.output_dir,
        pass_manifest_dir=args.pass_manifest_dir,
        timestamp="20260504T000004Z",
    )

    assert summary["candidate_count"] == 1
    assert summary["probe_window_count"] == 2
    assert summary["probe_count"] == 2
    assert summary["expansion_triggered"] is True
    assert summary["expansion_reason"] == "primary_exact_zero_pass"
    assert summary["pass_count"] == 1
    assert summary["diagnostic_counts"]["no_pmxt_l2_book_data"] == 1
    assert summary["diagnostic_counts"]["successful_pass_count"] == 1
    pass_manifest = json.loads(Path(output_files["pass_manifest"]).read_text(encoding="utf-8"))
    assert pass_manifest["candidates"][0]["coverage"]["window"] == {
        "start_time": "2026-03-22T08:00:00Z",
        "end_time": "2026-03-22T09:00:00Z",
    }
    assert pass_manifest["candidates"][0]["coverage"]["min_book_events"] == 50
    assert pass_manifest["candidates"][0]["coverage"]["selection_phase"] == (
        "zero_pass_expanded_window"
    )


def test_zero_pass_expands_to_additional_ranked_candidates(monkeypatch, tmp_path) -> None:
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "strategy": "microprice_optimizer",
                "candidates": [
                    {"market_slug": "first-thin", "question": "Thin?", "token_index": 0},
                    {"market_slug": "second-covered", "question": "Covered?", "token_index": 0},
                ],
            }
        ),
        encoding="utf-8",
    )

    async def _fake_probe_candidate(candidate, **kwargs):  # type: ignore[no-untyped-def]
        if candidate.slug == "second-covered":
            return probe_pmxt_l2_coverage._result_from_candidate(
                candidate,
                status="pass",
                book_events=90,
                min_book_events=50,
                window_start_time=kwargs["start_time"],
                window_end_time=kwargs["end_time"],
            )
        return probe_pmxt_l2_coverage._result_from_candidate(
            candidate,
            status="no_coverage",
            book_events=0,
            min_book_events=50,
            message="No PMXT L2 book replay was loaded for the requested window.",
            window_start_time=kwargs["start_time"],
            window_end_time=kwargs["end_time"],
        )

    monkeypatch.setattr(probe_pmxt_l2_coverage, "probe_candidate", _fake_probe_candidate)
    args = Namespace(
        manifest=source_manifest,
        output_dir=tmp_path / "reports",
        pass_manifest_dir=tmp_path / "pass_manifests",
        start_time="2026-03-22T09:00:00Z",
        end_time="2026-03-22T10:00:00Z",
        strategy="microprice_optimizer",
        max_candidates=1,
        expanded_max_candidates=2,
        min_book_events=50,
        sources=None,
        timeout_seconds=5,
        recent_window_count=1,
        window_step_hours=1,
        max_alternate_windows=0,
        max_alternate_probes=10,
        alternate_manifest_glob=None,
        alternate_manifest_count=0,
    )

    summary = asyncio.run(probe_pmxt_l2_coverage.run_probe(args))
    output_files = probe_pmxt_l2_coverage.write_outputs(
        summary,
        output_dir=args.output_dir,
        pass_manifest_dir=args.pass_manifest_dir,
        timestamp="20260504T000005Z",
    )

    assert summary["primary_probe_count"] == 1
    assert summary["expansion_probe_count"] == 1
    assert summary["expansion_triggered"] is True
    assert summary["pass_count"] == 1
    assert [row["selection_phase"] for row in summary["results"]] == [
        "primary_exact",
        "zero_pass_expanded_candidate",
    ]
    pass_manifest = json.loads(Path(output_files["pass_manifest"]).read_text(encoding="utf-8"))
    assert [candidate["market_slug"] for candidate in pass_manifest["candidates"]] == [
        "second-covered"
    ]
    assert pass_manifest["candidates"][0]["coverage"]["window"] == {
        "start_time": "2026-03-22T09:00:00Z",
        "end_time": "2026-03-22T10:00:00Z",
    }


def test_pass_manifest_dedupes_multiple_pass_windows_for_same_market(tmp_path) -> None:
    summary = {
        "generated_at_utc": "2026-03-22T00:00:00Z",
        "manifest": "source.json",
        "strategy": "Microprice",
        "window": {"start_time": "2026-03-22T09:00:00Z", "end_time": "2026-03-22T10:00:00Z"},
        "windows": [
            {"start_time": "2026-03-22T09:00:00Z", "end_time": "2026-03-22T10:00:00Z"},
            {"start_time": "2026-03-22T08:00:00Z", "end_time": "2026-03-22T09:00:00Z"},
        ],
        "min_book_events": 50,
        "safety": {"live_trading": False},
        "results": [
            {
                "status": "pass",
                "slug": "same",
                "market_slug": "same",
                "question": "same?",
                "source_strategy": "Microprice",
                "token_index": 0,
                "book_events": 60,
                "min_book_events": 50,
            },
            {
                "status": "pass",
                "slug": "same",
                "market_slug": "same",
                "question": "same?",
                "source_strategy": "Microprice",
                "token_index": 0,
                "book_events": 90,
                "min_book_events": 50,
            },
        ],
    }

    manifest = probe_pmxt_l2_coverage._build_pass_manifest(summary)

    assert manifest["candidate_count"] == 1
    assert manifest["metadata"]["pass_window_count"] == 2
    assert manifest["metadata"]["unique_pass_market_count"] == 1
    assert manifest["candidates"][0]["coverage"]["book_events"] == 90
