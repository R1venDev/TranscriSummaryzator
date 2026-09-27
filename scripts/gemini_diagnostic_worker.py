#!/usr/bin/env python3
"""Register exactly two private Gemini Batch probes; scheduler collects them."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from summary.gemini_v1.diagnostic import inspect_run, submit_pair


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("--manifest", type=Path, required=True)
    submit.add_argument("--private-root", type=Path, required=True)
    submit.add_argument("--output-dir", type=Path, required=True)
    status = commands.add_parser("status")
    status.add_argument("--run-id", required=True)
    status.add_argument("--private-root", type=Path, required=True)
    status.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outcome = (submit_pair(plan_path=args.manifest,
                           private_root=args.private_root,
                           output_dir=args.output_dir)
               if args.command == "submit" else
               inspect_run(private_root=args.private_root,
                           output_dir=args.output_dir, run_id=args.run_id))
    print(json.dumps(outcome, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if args.command == "status" or outcome["status"] == "registered" else 2


if __name__ == "__main__":
    raise SystemExit(main())
