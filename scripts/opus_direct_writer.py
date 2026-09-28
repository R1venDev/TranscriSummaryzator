#!/usr/bin/env python3
"""Admin-only isolated Opus writer comparison; never publishes production."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from summary.opus_v1.direct_writer import authorize, submit


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("authorize", "submit"):
        part = sub.add_parser(command)
        part.add_argument("--transcript", type=Path, required=True)
        part.add_argument("--output", type=Path, required=True,
                          help="Isolated experiment output identity, not a production pointer")
        part.add_argument("--private-root", type=Path, required=True)
        part.add_argument("--run-id", required=True)
        if command == "authorize":
            part.add_argument("--authorization-ref", required=True)
    args = parser.parse_args()
    common = {"transcript_path": args.transcript, "output_dir": args.output,
              "private_root": args.private_root, "run_id": args.run_id}
    result = (authorize(**common, authorization_ref=args.authorization_ref)
              if args.command == "authorize" else submit(**common))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if args.command == "authorize" or result["status"] in {
        "submitted", "direct_complete", "reserved", "polling", "submitted"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
