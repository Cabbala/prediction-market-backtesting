from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_MANIFEST_GLOB = (
    "/opt/polymarket-lab/autoresearch/reward_scanner/manifests/reward_scanner_manifest_*.json"
)
DEFAULT_OBSERVATION_DIR = Path("/opt/polymarket-lab/reports/reward_scanner/shadow_observations")
DEFAULT_OUTPUT_DIR = Path("/opt/polymarket-lab/reports/reward_scanner/diagnostics")
MODE = "reward_scanner_zero_vs_prior_nonempty_diagnostic_shadow_backtest_only"
SCHEMA_VERSION = "polymarket.reward-scanner-zero-diff-diagnostic.v1"
SAFETY_FALSE_FIELDS = (
    "orders_submitted",
    "orders_signed",
    "orders_cancelled",
    "credentials_required",
    "live_trading_worker_started",
    "worker_trading_started",
)
STAMP_RE = re.compile(r"(\d{8}T\d{6}Z)")


class DiagnosticError(RuntimeError):
    """Raised when the differential cannot be explained without guessing."""


def _utc_now() -> datetime:
    return datetime.now(tz=UTC).replace(microsecond=0)


def _utc_stamp(dt: datetime | None = None) -> str:
    return (dt or _utc_now()).strftime("%Y%m%dT%H%M%SZ")


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _stamp_from_path(path: Path) -> str | None:
    match = STAMP_RE.search(path.name)
    return match.group(1) if match else None


def _time_from_path(path: Path) -> datetime | None:
    stamp = _stamp_from_path(path)
    if not stamp:
        return None
    return datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)


def _artifact_time(path: Path, payload: Mapping[str, Any]) -> datetime | None:
    return (
        _parse_dt(payload.get("generated_at_utc"))
        or _parse_dt(payload.get("source_scan_timestamp_utc"))
        or _time_from_path(path)
    )


def safety_object() -> dict[str, bool]:
    return {
        "live_trading": False,
        "orders_submitted": False,
        "orders_signed": False,
        "orders_cancelled": False,
        "credentials_required": False,
        "live_trading_worker_started": False,
        "worker_trading_started": False,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DiagnosticError(f"missing artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DiagnosticError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DiagnosticError(f"artifact must be a JSON object: {path}")
    return payload


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _rows_from_sequence(value: Any) -> list[Mapping[str, Any]]:
    if not _is_sequence(value):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _row_key(row: Mapping[str, Any]) -> str:
    for key in ("condition_id", "conditionId", "market_id", "id", "slug", "question"):
        value = row.get(key)
        if value not in (None, ""):
            return f"{key}:{value}"
    return json.dumps(row, sort_keys=True, default=str)


def _manifest_candidate_count(payload: Mapping[str, Any]) -> int:
    candidates = payload.get("candidates")
    if _is_sequence(candidates):
        return len(candidates)
    summary = _as_mapping(payload.get("summary"))
    for key in ("manifest_candidate_count", "candidate_count"):
        value = summary.get(key)
        if isinstance(value, int):
            return value
    return 0


def _summary_counts(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_mapping(payload.get("summary"))
    return {
        "candidate_count": summary.get("candidate_count"),
        "manifest_candidate_count": summary.get("manifest_candidate_count"),
        "eligible_for_backtest_queue_count": summary.get("eligible_for_backtest_queue_count"),
        "blocked_count": summary.get("blocked_count"),
        "explicit_reward_evidence_count": summary.get("explicit_reward_evidence_count"),
        "candidate_rows_len": len(payload.get("candidates"))
        if _is_sequence(payload.get("candidates"))
        else 0,
    }


def _source_scan_path(manifest: Mapping[str, Any]) -> Path:
    direct = manifest.get("input_scan_path")
    source_paths = _as_mapping(manifest.get("source_paths"))
    provenance = _as_mapping(manifest.get("source_provenance"))
    value = direct or source_paths.get("market_scan_json") or provenance.get("source_artifact_path")
    if not isinstance(value, str) or not value:
        raise DiagnosticError("manifest is missing input_scan_path/source_paths.market_scan_json")
    return Path(value)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _jsonl_sidecar_path(scan_path: Path) -> Path | None:
    name = scan_path.name
    if not name.startswith("public_market_scan_") or scan_path.suffix != ".json":
        return None
    return scan_path.with_name(
        name.replace("public_market_scan_", "public_market_scan_candidates_", 1)
    ).with_suffix(".jsonl")


def _jsonl_line_count(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def scan_profile(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path)
    metadata = _as_mapping(payload.get("metadata"))
    top_candidates = payload.get("top_candidates")
    candidates = payload.get("candidates")

    builder_key = None
    builder_visible_count = 0
    for key, value in (("top_candidates", top_candidates), ("candidates", candidates)):
        if _is_sequence(value):
            builder_key = key
            builder_visible_count = len(_rows_from_sequence(value))
            break

    bucket_counts: dict[str, int] = {}
    bucket_unique: dict[str, Mapping[str, Any]] = {}
    if isinstance(top_candidates, Mapping):
        for key, value in sorted(top_candidates.items()):
            rows = _rows_from_sequence(value)
            bucket_counts[str(key)] = len(rows)
            for row in rows:
                bucket_unique.setdefault(_row_key(row), row)

    sidecar = _jsonl_sidecar_path(path)
    sidecar_lines = _jsonl_line_count(sidecar)
    source_reported_count = _first_present(
        payload.get("candidate_count"),
        metadata.get("candidate_count"),
        payload.get("market_count"),
        metadata.get("market_count"),
    )
    bucket_total = sum(bucket_counts.values())
    return {
        "path": str(path),
        "exists": True,
        "top_candidates_type": type(top_candidates).__name__
        if top_candidates is not None
        else None,
        "candidates_type": type(candidates).__name__ if candidates is not None else None,
        "manifest_builder_visible_key": builder_key,
        "manifest_builder_visible_candidate_count": builder_visible_count,
        "strategy_bucket_counts": bucket_counts,
        "strategy_bucket_candidate_count": bucket_total,
        "strategy_bucket_unique_candidate_count": len(bucket_unique),
        "source_reported_candidate_count": source_reported_count,
        "source_timestamp_utc": _first_present(
            metadata.get("timestamp_utc"),
            metadata.get("utc_timestamp"),
            metadata.get("generated_at_utc"),
            metadata.get("artifact_timestamp"),
        ),
        "source_counts": metadata.get("source_counts") or payload.get("source_counts"),
        "sidecar_jsonl_path": str(sidecar) if sidecar else None,
        "sidecar_jsonl_line_count": sidecar_lines,
    }


def _observation_path_for_manifest(manifest_path: Path, observation_dir: Path) -> Path | None:
    stamp = _stamp_from_path(manifest_path)
    if not stamp:
        return None
    candidate = observation_dir / f"low_fill_reward_shadow_observation_{stamp}.json"
    return candidate if candidate.exists() else None


def _observation_profile(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = _load_json_object(path)
    summary = _as_mapping(payload.get("summary"))
    return {
        "path": str(path),
        "candidate_count": payload.get("candidate_count"),
        "observation_rows_len": len(payload.get("observations"))
        if _is_sequence(payload.get("observations"))
        else 0,
        "reward_ev_computable_count": summary.get("reward_ev_computable_count"),
        "summary": dict(summary),
        "safety": dict(_as_mapping(payload.get("safety"))),
    }


def _validate_false_safety(report: Mapping[str, Any]) -> None:
    for field in SAFETY_FALSE_FIELDS:
        if report.get(field) is not False:
            raise DiagnosticError(f"top-level safety field is not false: {field}")


def _classify(
    *,
    latest_scan: Mapping[str, Any],
    prior_scan: Mapping[str, Any],
) -> tuple[str, str, str]:
    latest_builder_visible = latest_scan["manifest_builder_visible_candidate_count"]
    latest_bucket_count = latest_scan["strategy_bucket_candidate_count"]
    latest_source_signal = _first_present(
        latest_scan.get("source_reported_candidate_count"),
        latest_scan.get("sidecar_jsonl_line_count"),
        latest_bucket_count,
    )
    if latest_builder_visible > 0:
        raise DiagnosticError(
            "ambiguous zero manifest: latest source has builder-visible candidates but "
            "latest manifest has zero candidates"
        )
    if latest_scan["top_candidates_type"] == "dict" and latest_bucket_count > 0:
        return (
            "diagnostic_only_adoptable_source_change",
            "source_schema_mismatch_strategy_keyed_top_candidates",
            "Latest scan top_candidates is a strategy-keyed object, while the reward "
            "manifest builder consumes only list-shaped top_candidates/candidates. "
            "The prior source scan used list-shaped top_candidates and produced a "
            "non-empty manifest.",
        )
    if not latest_source_signal:
        return (
            "diagnostic_only_upstream_source_empty",
            "latest_source_scan_has_no_candidate_signal",
            "Latest source artifacts expose no candidate rows or candidate-count signal; "
            "the zero manifest is consistent with an empty upstream scan.",
        )
    if prior_scan["manifest_builder_visible_candidate_count"] <= 0:
        raise DiagnosticError("ambiguous prior manifest: prior source has no builder-visible rows")
    raise DiagnosticError("ambiguous zero manifest: no supported source-shape explanation found")


def _manifest_record(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "generated_at_utc": payload.get("generated_at_utc"),
        "source_scan_timestamp_utc": payload.get("source_scan_timestamp_utc"),
        "candidate_count": _manifest_candidate_count(payload),
        "summary": _summary_counts(payload),
        "safety": dict(_as_mapping(payload.get("safety"))),
    }


def build_diagnostic(
    *,
    latest_manifest_path: Path,
    prior_manifest_path: Path,
    observation_path: Path | None = None,
    observation_dir: Path = DEFAULT_OBSERVATION_DIR,
) -> dict[str, Any]:
    latest_manifest = _load_json_object(latest_manifest_path)
    prior_manifest = _load_json_object(prior_manifest_path)
    if latest_manifest_path.resolve() == prior_manifest_path.resolve():
        raise DiagnosticError("latest and prior manifests must be different artifacts")

    latest_count = _manifest_candidate_count(latest_manifest)
    prior_count = _manifest_candidate_count(prior_manifest)
    if latest_count != 0:
        raise DiagnosticError(f"latest manifest is not zero-candidate: count={latest_count}")
    if prior_count <= 0:
        raise DiagnosticError(f"prior manifest is not non-empty: count={prior_count}")

    latest_time = _artifact_time(latest_manifest_path, latest_manifest)
    prior_time = _artifact_time(prior_manifest_path, prior_manifest)
    if latest_time and prior_time and latest_time <= prior_time:
        raise DiagnosticError(
            "latest manifest must be newer than prior manifest for this differential"
        )

    latest_scan = scan_profile(_source_scan_path(latest_manifest))
    prior_scan = scan_profile(_source_scan_path(prior_manifest))
    classification, cause_code, likely_cause = _classify(
        latest_scan=latest_scan,
        prior_scan=prior_scan,
    )
    obs_path = observation_path or _observation_path_for_manifest(
        latest_manifest_path, observation_dir
    )
    observation = _observation_profile(obs_path)
    latest_scan_candidates = _first_present(
        latest_scan.get("source_reported_candidate_count"),
        latest_scan.get("sidecar_jsonl_line_count"),
        latest_scan.get("strategy_bucket_candidate_count"),
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
        "mode": MODE,
        "classification": classification,
        "cause_code": cause_code,
        "likely_cause": likely_cause,
        "diagnostic_only": True,
        "adoptable_source_change": cause_code
        == "source_schema_mismatch_strategy_keyed_top_candidates",
        "no_profit_claim": True,
        "no_live_readiness_claim": True,
        "profit_or_live_readiness_verdict": "no_profit_or_live_readiness_claim_made",
        "artifacts_compared": {
            "latest_manifest": str(latest_manifest_path),
            "prior_manifest": str(prior_manifest_path),
            "latest_source_scan": latest_scan["path"],
            "prior_source_scan": prior_scan["path"],
            "latest_observation": observation["path"] if observation else None,
        },
        "counts": {
            "latest_manifest_candidate_count": latest_count,
            "prior_manifest_candidate_count": prior_count,
            "latest_source_reported_candidate_count": latest_scan.get(
                "source_reported_candidate_count"
            ),
            "latest_source_strategy_bucket_candidate_count": latest_scan.get(
                "strategy_bucket_candidate_count"
            ),
            "latest_source_strategy_bucket_unique_candidate_count": latest_scan.get(
                "strategy_bucket_unique_candidate_count"
            ),
            "latest_source_sidecar_jsonl_line_count": latest_scan.get("sidecar_jsonl_line_count"),
            "latest_source_candidate_signal_count": latest_scan_candidates,
            "latest_manifest_builder_visible_source_count": latest_scan.get(
                "manifest_builder_visible_candidate_count"
            ),
            "prior_manifest_builder_visible_source_count": prior_scan.get(
                "manifest_builder_visible_candidate_count"
            ),
            "latest_observation_candidate_count": observation.get("candidate_count")
            if observation
            else None,
            "latest_observation_reward_ev_computable_count": observation.get(
                "reward_ev_computable_count"
            )
            if observation
            else None,
        },
        "latest_manifest": _manifest_record(latest_manifest_path, latest_manifest),
        "prior_manifest": _manifest_record(prior_manifest_path, prior_manifest),
        "latest_source_scan_profile": latest_scan,
        "prior_source_scan_profile": prior_scan,
        "latest_observation_profile": observation,
        "next_hook": (
            "Teach reward manifest extraction to flatten the strategy-keyed "
            "top_candidates object, de-duplicate rows, and keep fail-closed safety "
            "tests before rerunning Job A/supervisor diagnostics."
        ),
        "safety": safety_object(),
        "live_trading": False,
        **{field: False for field in SAFETY_FALSE_FIELDS},
    }
    _validate_false_safety(report)
    return report


def _latest_zero_and_prior_nonempty(manifest_glob: str) -> tuple[Path, Path]:
    paths = [Path(path) for path in glob.glob(manifest_glob)]
    if not paths:
        raise DiagnosticError(f"no manifests match {manifest_glob}")
    payloads: list[tuple[datetime, Path, dict[str, Any]]] = []
    for path in paths:
        payload = _load_json_object(path)
        artifact_time = _artifact_time(path, payload)
        if artifact_time is None:
            raise DiagnosticError(f"manifest has no parseable timestamp: {path}")
        payloads.append((artifact_time, path, payload))
    payloads.sort(key=lambda item: (item[0], str(item[1])))
    latest_time, latest_path, latest_payload = payloads[-1]
    if _manifest_candidate_count(latest_payload) != 0:
        raise DiagnosticError(
            f"latest manifest is not zero-candidate: {latest_path} "
            f"count={_manifest_candidate_count(latest_payload)}"
        )
    for prior_time, prior_path, prior_payload in reversed(payloads[:-1]):
        if prior_time >= latest_time:
            continue
        if _manifest_candidate_count(prior_payload) > 0:
            return latest_path, prior_path
    raise DiagnosticError(f"no prior non-empty manifest found before {latest_path}")


def _markdown_report(report: Mapping[str, Any]) -> str:
    counts = _as_mapping(report.get("counts"))
    artifacts = _as_mapping(report.get("artifacts_compared"))
    latest_scan = _as_mapping(report.get("latest_source_scan_profile"))
    safety = _as_mapping(report.get("safety"))
    lines = [
        "# Reward Scanner Zero vs Prior Non-empty Diagnostic",
        "",
        f"- mode: {report.get('mode')}",
        f"- classification: {report.get('classification')}",
        f"- cause_code: {report.get('cause_code')}",
        f"- no_profit_claim: {report.get('no_profit_claim')}",
        f"- no_live_readiness_claim: {report.get('no_live_readiness_claim')}",
        "",
        "## Artifacts",
        "",
        f"- latest_manifest: `{artifacts.get('latest_manifest')}`",
        f"- prior_manifest: `{artifacts.get('prior_manifest')}`",
        f"- latest_source_scan: `{artifacts.get('latest_source_scan')}`",
        f"- prior_source_scan: `{artifacts.get('prior_source_scan')}`",
        f"- latest_observation: `{artifacts.get('latest_observation')}`",
        "",
        "## Counts",
        "",
        f"- latest_manifest_candidate_count: {counts.get('latest_manifest_candidate_count')}",
        f"- prior_manifest_candidate_count: {counts.get('prior_manifest_candidate_count')}",
        f"- latest_source_reported_candidate_count: {counts.get('latest_source_reported_candidate_count')}",
        f"- latest_source_strategy_bucket_candidate_count: {counts.get('latest_source_strategy_bucket_candidate_count')}",
        f"- latest_source_strategy_bucket_unique_candidate_count: {counts.get('latest_source_strategy_bucket_unique_candidate_count')}",
        f"- latest_source_sidecar_jsonl_line_count: {counts.get('latest_source_sidecar_jsonl_line_count')}",
        f"- latest_manifest_builder_visible_source_count: {counts.get('latest_manifest_builder_visible_source_count')}",
        f"- latest_observation_candidate_count: {counts.get('latest_observation_candidate_count')}",
        f"- latest_observation_reward_ev_computable_count: {counts.get('latest_observation_reward_ev_computable_count')}",
        "",
        "## Likely Cause",
        "",
        str(report.get("likely_cause")),
        "",
        "Latest source scan shape:",
        f"- top_candidates_type: {latest_scan.get('top_candidates_type')}",
        f"- strategy_bucket_counts: {json.dumps(latest_scan.get('strategy_bucket_counts'), sort_keys=True)}",
        f"- manifest_builder_visible_key: {latest_scan.get('manifest_builder_visible_key')}",
        "",
        "## Safety",
        "",
        f"- orders_submitted={str(safety.get('orders_submitted')).lower()}",
        f"- orders_signed={str(safety.get('orders_signed')).lower()}",
        f"- orders_cancelled={str(safety.get('orders_cancelled')).lower()}",
        f"- credentials_required={str(safety.get('credentials_required')).lower()}",
        f"- live_trading_worker_started={str(safety.get('live_trading_worker_started')).lower()}",
        f"- worker_trading_started={str(safety.get('worker_trading_started')).lower()}",
        "",
        "## Next Hook",
        "",
        str(report.get("next_hook")),
        "",
        "This is a shadow/backtest diagnostic only. It does not claim profit, live readiness, or order safety beyond the explicit false safety flags above.",
    ]
    return "\n".join(lines) + "\n"


def _blocker_report(error: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
        "mode": MODE,
        "classification": "blocked_fail_closed",
        "cause_code": "diagnostic_artifacts_missing_or_ambiguous",
        "blocker": error,
        "diagnostic_only": True,
        "adoptable_source_change": False,
        "no_profit_claim": True,
        "no_live_readiness_claim": True,
        "profit_or_live_readiness_verdict": "no_profit_or_live_readiness_claim_made",
        "safety": safety_object(),
        "live_trading": False,
        **{field: False for field in SAFETY_FALSE_FIELDS},
    }


def write_outputs(report: dict[str, Any], output_dir: Path, timestamp: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"reward_scanner_zero_diff_diagnostic_{timestamp}.json"
    md_path = output_dir / f"reward_scanner_zero_diff_diagnostic_{timestamp}.md"
    report["output_files"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    return report["output_files"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose a latest-zero vs prior-nonempty reward scanner manifest "
            "differential using shadow/backtest-only artifacts."
        )
    )
    parser.add_argument("--latest-manifest", type=Path, default=None)
    parser.add_argument("--prior-manifest", type=Path, default=None)
    parser.add_argument("--manifest-glob", default=DEFAULT_MANIFEST_GLOB)
    parser.add_argument("--observation", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args(argv)

    timestamp = args.timestamp or _utc_stamp()
    try:
        latest_manifest = args.latest_manifest
        prior_manifest = args.prior_manifest
        if latest_manifest is None or prior_manifest is None:
            latest_manifest, prior_manifest = _latest_zero_and_prior_nonempty(args.manifest_glob)
        report = build_diagnostic(
            latest_manifest_path=latest_manifest,
            prior_manifest_path=prior_manifest,
            observation_path=args.observation,
        )
        output_files = write_outputs(report, args.output_dir, timestamp)
        print(
            json.dumps(
                {
                    "classification": report["classification"],
                    "cause_code": report["cause_code"],
                    "counts": report["counts"],
                    "safety": report["safety"],
                    "output_files": output_files,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except DiagnosticError as exc:
        report = _blocker_report(str(exc))
        output_files = write_outputs(report, args.output_dir, timestamp)
        print(
            json.dumps(
                {
                    "classification": report["classification"],
                    "cause_code": report["cause_code"],
                    "blocker": report["blocker"],
                    "safety": report["safety"],
                    "output_files": output_files,
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
