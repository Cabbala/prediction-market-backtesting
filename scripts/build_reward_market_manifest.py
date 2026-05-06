#!/usr/bin/env python3
"""Build a reward-market candidate manifest from read-only scan artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prediction_market_extensions.analysis.reward_market_scanner import RewardMarketScanner, load_scan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", required=True, help="Path to market scan JSON/JSONL artifact")
    parser.add_argument("--output-dir", default="/opt/polymarket-lab/autoresearch/reward_scanner/manifests")
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()
    payload = load_scan(args.scan)
    manifest = RewardMarketScanner().build_manifest(payload, source_path=args.scan, limit=args.limit)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = manifest["generated_at_utc"].replace("-", "").replace(":", "").replace("Z", "Z")
    json_path = out_dir / f"reward_market_manifest_{ts}.json"
    rules_path = out_dir / f"reward_market_manifest_rules_{ts}.md"
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    rules_path.write_text("\n".join([
        "# Reward Market Manifest Rules",
        "",
        f"Generated: {manifest['generated_at_utc']}",
        f"Source: {args.scan}",
        "Mode: SHADOW/BACKTEST ONLY; no live orders/signing/cancel paths.",
        "",
        "## Eligibility",
        *[f"- {rule}" for rule in manifest["eligibility_rules"]],
        "",
        "## Scoring",
        *[f"- {note}" for note in manifest["scoring_notes"]],
        "",
        f"Eligible candidates: {manifest['summary']['eligible_count']}",
        f"Skipped candidates: {manifest['summary']['skipped_count']}",
        "",
    ]) + "\n")
    print(json.dumps({"manifest": str(json_path), "rules": str(rules_path), "summary": manifest["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
