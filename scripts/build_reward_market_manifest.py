from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prediction_market_extensions.adapters.polymarket.reward_market_scanner import (  # noqa: E402
    build_reward_manifest,
    load_scan,
    write_manifest,
)

DEFAULT_SCAN_DIR = Path("/opt/polymarket-lab/data/market_scans")
DEFAULT_REPORT_DIR = Path("/opt/polymarket-lab/reports")
DEFAULT_OUTPUT_DIR = Path("/opt/polymarket-lab/autoresearch/reward_scanner/manifests")


def latest_report(report_dir: Path = DEFAULT_REPORT_DIR) -> Path | None:
    if not report_dir.exists():
        return None
    candidates = [
        p
        for p in report_dir.rglob("*.md")
        if p.is_file()
        and (
            "strategy" in str(p).lower()
            or "candidate" in str(p).lower()
            or "market_scan" in str(p).lower()
        )
    ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def latest_scan(scan_dir: Path = DEFAULT_SCAN_DIR) -> Path:
    candidates = [p for p in scan_dir.glob("*.json") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"no JSON scan artifacts found under {scan_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a shadow/backtest-only Polymarket reward-market candidate manifest."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Market scan JSON artifact. Defaults to latest /opt/polymarket-lab/data/market_scans/*.json",
    )
    parser.add_argument(
        "--strategy-report",
        type=Path,
        help="Optional strategy candidate/market report path. Defaults to latest markdown report under /opt/polymarket-lab/reports.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be non-negative")

    scan_path = args.input or latest_scan()
    report_path = args.strategy_report or latest_report()
    manifest = build_reward_manifest(
        load_scan(scan_path), limit=args.limit, source_artifact_path=str(scan_path)
    )
    manifest["input_scan_path"] = str(scan_path)
    manifest["input_strategy_report_path"] = str(report_path) if report_path else None
    manifest["source_paths"] = {
        "market_scan_json": str(scan_path),
        "strategy_or_scan_report_md": str(report_path) if report_path else None,
    }
    manifest_path, rules_path = write_manifest(manifest, args.output_dir)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "rules": str(rules_path),
                "candidates": len(manifest["candidates"]),
                "mode": manifest["mode"],
                "strategy_report": str(report_path) if report_path else None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
