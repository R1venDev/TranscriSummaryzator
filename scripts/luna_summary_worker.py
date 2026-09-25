#!/usr/bin/env python3
"""Isolated summary-only entry point; never imports speech model runtimes."""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from summary.luna_v1.engine import poll_once, submit

def _write_status(path: Path, value: dict) -> None:
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("submit")
    create.add_argument("--transcript", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--private-root", type=Path, required=True)
    create.add_argument("--force-nonce")
    poll = sub.add_parser("poll")
    poll.add_argument("--private-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "submit":
        outcome = submit(transcript_path=args.transcript, output_dir=args.output,
                         private_root=args.private_root, force_nonce=args.force_nonce)
        _write_status(args.output / "summary_luna_attempt.json", outcome)
        print("SUMMARY_PROGRESS " + json.dumps({
            "stage": "summary_" + outcome["status"],
            "progress": 100 if outcome["status"] == "accepted_cache_hit" else 5,
            "detail": outcome["status"],
        }, ensure_ascii=False), flush=True)
        return 0 if outcome["status"] in {"submitted", "pending", "accepted_cache_hit", "accepted"} else 2
    outcomes = poll_once(private_root=args.private_root)
    print(json.dumps(outcomes, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
