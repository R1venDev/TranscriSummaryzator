#!/usr/bin/env python3
"""Replay deterministic publication gates without repeating model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from summary.verifier import runtime_quality_gates, verify_generated_items


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir
    items = load(run / "public_items.json").get("items", [])
    plan = load(run / "summary_plan.json")
    state_path = run / "semantics" / "meeting_state.v2.json"
    if not state_path.is_file():
        state_path = run / "meeting_state.v2.json"
    state = load(state_path)
    report = verify_generated_items(items, plan.get("public_sentence_plans", []), state.get("claims", []))
    markdown_path = run / "summary.md"
    if markdown_path.is_file():
        markdown = markdown_path.read_text(encoding="utf-8")
        report["quality_gates"] = runtime_quality_gates(report, markdown, hashlib.sha256(markdown.encode()).hexdigest(), items, plan)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report.get("passed") and report.get("quality_gates", {"passed": True}).get("passed") else 1)


if __name__ == "__main__":
    main()
