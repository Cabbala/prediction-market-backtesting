from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.reward_scanner_zero_diff_diagnostic import (
    DiagnosticError,
    build_diagnostic,
    write_outputs,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _manifest(path: Path, *, scan_path: Path, generated_at: str, candidates: list[dict]) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "polymarket.reward-market-manifest.v1",
            "manifest_kind": "reward_scanner_manifest",
            "generated_at_utc": generated_at,
            "input_scan_path": str(scan_path),
            "summary": {
                "candidate_count": len(candidates),
                "manifest_candidate_count": len(candidates),
                "eligible_for_backtest_queue_count": 0,
                "blocked_count": len(candidates),
                "explicit_reward_evidence_count": 0,
            },
            "safety": {
                "orders_submitted": False,
                "orders_signed": False,
                "orders_cancelled": False,
                "credentials_required": False,
                "live_trading_worker_started": False,
                "worker_trading_started": False,
            },
            "orders_submitted": False,
            "orders_signed": False,
            "orders_cancelled": False,
            "credentials_required": False,
            "live_trading_worker_started": False,
            "worker_trading_started": False,
            "candidates": candidates,
        },
    )


def _candidate(slug: str) -> dict:
    return {
        "id": slug,
        "condition_id": f"0x{slug}",
        "slug": slug,
        "question": f"Will {slug} happen?",
        "outcomes": ["Yes", "No"],
        "clob_token_ids": [f"{slug}-yes", f"{slug}-no"],
    }


def test_zero_vs_prior_classifies_strategy_keyed_source_shape(tmp_path: Path) -> None:
    latest_scan = _write_json(
        tmp_path / "public_market_scan_20260607T030338Z.json",
        {
            "metadata": {
                "artifact_timestamp": "20260607T030338Z",
                "candidate_count": 65,
                "source_counts": {"gamma": 90},
            },
            "top_candidates": {
                "microprice": [_candidate("shared"), _candidate("micro")],
                "low_fill_liquidity_reward_maker": [_candidate("shared"), _candidate("reward")],
            },
        },
    )
    latest_scan.with_name("public_market_scan_candidates_20260607T030338Z.jsonl").write_text(
        "{}\n{}\n{}\n",
        encoding="utf-8",
    )
    prior_scan = _write_json(
        tmp_path / "public_market_scan_20260606T154656Z.json",
        {"candidate_count": 25, "top_candidates": [_candidate("prior")]},
    )
    latest_manifest = _manifest(
        tmp_path / "reward_scanner_manifest_20260607T034639Z.json",
        scan_path=latest_scan,
        generated_at="2026-06-07T03:46:39Z",
        candidates=[],
    )
    prior_manifest = _manifest(
        tmp_path / "reward_scanner_manifest_20260606T154656Z.json",
        scan_path=prior_scan,
        generated_at="2026-06-06T15:46:56Z",
        candidates=[_candidate("prior")],
    )
    observation = _write_json(
        tmp_path / "low_fill_reward_shadow_observation_20260607T034639Z.json",
        {
            "candidate_count": 0,
            "observations": [],
            "summary": {"reward_ev_computable_count": 0},
            "safety": {
                "orders_submitted": False,
                "orders_signed": False,
                "orders_cancelled": False,
                "credentials_required": False,
                "live_trading_worker_started": False,
                "worker_trading_started": False,
            },
        },
    )

    report = build_diagnostic(
        latest_manifest_path=latest_manifest,
        prior_manifest_path=prior_manifest,
        observation_path=observation,
    )

    assert report["classification"] == "diagnostic_only_adoptable_source_change"
    assert report["cause_code"] == "source_schema_mismatch_strategy_keyed_top_candidates"
    assert report["counts"]["latest_manifest_candidate_count"] == 0
    assert report["counts"]["prior_manifest_candidate_count"] == 1
    assert report["counts"]["latest_source_reported_candidate_count"] == 65
    assert report["counts"]["latest_source_strategy_bucket_candidate_count"] == 4
    assert report["counts"]["latest_source_strategy_bucket_unique_candidate_count"] == 3
    assert report["counts"]["latest_source_sidecar_jsonl_line_count"] == 3
    assert report["counts"]["latest_observation_reward_ev_computable_count"] == 0
    for field in (
        "orders_submitted",
        "orders_signed",
        "orders_cancelled",
        "credentials_required",
        "live_trading_worker_started",
        "worker_trading_started",
    ):
        assert report[field] is False
        assert report["safety"][field] is False


def test_write_outputs_writes_json_and_markdown(tmp_path: Path) -> None:
    latest_scan = _write_json(
        tmp_path / "public_market_scan_20260607T030338Z.json",
        {
            "metadata": {"candidate_count": 1},
            "top_candidates": {"low_fill_liquidity_reward_maker": [_candidate("reward")]},
        },
    )
    prior_scan = _write_json(
        tmp_path / "public_market_scan_20260606T154656Z.json",
        {"top_candidates": [_candidate("prior")]},
    )
    report = build_diagnostic(
        latest_manifest_path=_manifest(
            tmp_path / "reward_scanner_manifest_20260607T034639Z.json",
            scan_path=latest_scan,
            generated_at="2026-06-07T03:46:39Z",
            candidates=[],
        ),
        prior_manifest_path=_manifest(
            tmp_path / "reward_scanner_manifest_20260606T154656Z.json",
            scan_path=prior_scan,
            generated_at="2026-06-06T15:46:56Z",
            candidates=[_candidate("prior")],
        ),
    )

    outputs = write_outputs(report, tmp_path / "out", "20260607T040329Z")

    json_report = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
    markdown_report = Path(outputs["markdown"]).read_text(encoding="utf-8")
    assert json_report["cause_code"] == "source_schema_mismatch_strategy_keyed_top_candidates"
    assert json_report["orders_submitted"] is False
    assert "## Likely Cause" in markdown_report
    assert "orders_submitted=false" in markdown_report


def test_missing_artifact_fails_closed(tmp_path: Path) -> None:
    prior_scan = _write_json(
        tmp_path / "public_market_scan_20260606T154656Z.json",
        {"top_candidates": [_candidate("prior")]},
    )
    prior_manifest = _manifest(
        tmp_path / "reward_scanner_manifest_20260606T154656Z.json",
        scan_path=prior_scan,
        generated_at="2026-06-06T15:46:56Z",
        candidates=[_candidate("prior")],
    )

    with pytest.raises(DiagnosticError, match="missing artifact"):
        build_diagnostic(
            latest_manifest_path=tmp_path / "missing.json",
            prior_manifest_path=prior_manifest,
        )


def test_builder_visible_latest_source_fails_closed_as_ambiguous(tmp_path: Path) -> None:
    latest_scan = _write_json(
        tmp_path / "public_market_scan_20260607T030338Z.json",
        {"top_candidates": [_candidate("visible")]},
    )
    prior_scan = _write_json(
        tmp_path / "public_market_scan_20260606T154656Z.json",
        {"top_candidates": [_candidate("prior")]},
    )

    with pytest.raises(DiagnosticError, match="builder-visible candidates"):
        build_diagnostic(
            latest_manifest_path=_manifest(
                tmp_path / "reward_scanner_manifest_20260607T034639Z.json",
                scan_path=latest_scan,
                generated_at="2026-06-07T03:46:39Z",
                candidates=[],
            ),
            prior_manifest_path=_manifest(
                tmp_path / "reward_scanner_manifest_20260606T154656Z.json",
                scan_path=prior_scan,
                generated_at="2026-06-06T15:46:56Z",
                candidates=[_candidate("prior")],
            ),
        )
