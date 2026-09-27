#!/usr/bin/env python3
"""Freeze two comparable, diagnostic-only Gemini Batch request bodies.

The private fixture rubric is deliberately not read by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from summary.gemini_v1.reconcile_contract import draft_units


PROMPTS = ROOT / "summary" / "gemini_v1"
CAP = 5_000
# Pre-output, source-only probe added to both arms after the original seven
# inventory items were frozen. The original fixture files remain unchanged.
SUPPLEMENTAL_PROBE = {
    "item_id": "D01-I001", "segment_id": "S01", "kind": "action",
    "claim": "Yachoy упоминает попытки в обсуждаемой работе.",
    "source_ids": ["U00126"], "speaker": "@Yachoy", "actor": None,
    "recipient": None, "action": None,
    "modality": None, "condition": None, "alternatives": [],
    "correction_of": [],
    "uncertainty": "Предмет прежних попыток в этой реплике назван неполно.",
    "source_quote": "Это были попытки, они получились неуспешными.",
}


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_verified(path: Path, expected_sha: str) -> object:
    data = path.read_bytes()
    if sha(data) != expected_sha or path.is_symlink():
        raise ValueError(f"fixture identity changed: {path.name}")
    return json.loads(data)


def write_private_new(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def pointer_unit_ids(pointer: str) -> list[str]:
    parts = pointer.strip("/").split("/")
    if len(parts) == 2 and parts[0] == "tasks":
        return [f"tasks:{parts[1]}:action", f"tasks:{parts[1]}:relations"]
    if len(parts) == 2 and parts[0] in {"main", "technical", "questions",
                                         "ideas", "timecodes", "verification"}:
        return [f"{parts[0]}:{parts[1]}"]
    if len(parts) == 3 and parts[0] == "chapters" and parts[2] == "summary":
        return [f"chapters:{parts[1]}:summary"]
    if len(parts) == 4 and parts[0] == "chapters" and parts[2] == "details":
        return [f"chapters:{parts[1]}:detail:{parts[3]}"]
    raise ValueError(f"unsupported frozen draft pointer: {pointer}")


def build(fixture_dir: Path, request_dir: Path, run_id: str) -> dict:
    manifest_path = fixture_dir / "manifest.json"
    fixture_manifest_bytes = manifest_path.read_bytes()
    fixture_manifest = json.loads(fixture_manifest_bytes)
    if fixture_manifest.get("fixture_id") != run_id:
        raise ValueError("fixture run identity mismatch")
    files = fixture_manifest["files"]
    # Never open rubric.hidden.json here.
    windows = read_verified(fixture_dir / "source_windows.json",
                            files["source_windows.json"]["sha256"])
    selected = read_verified(fixture_dir / "selected_inventory.json",
                             files["selected_inventory.json"]["sha256"])
    draft = read_verified(fixture_dir / "draft_document.json",
                          files["draft_document.json"]["sha256"])
    selected_units = read_verified(fixture_dir / "draft_units.json",
                                   files["draft_units.json"]["sha256"])
    source_sha = fixture_manifest["source"]["sha256"]
    draft_sha = fixture_manifest["draft"]["sha256"]
    if windows["source_sha256"] != source_sha or selected_units["draft_sha256"] != draft_sha:
        raise ValueError("source or draft fixture identity mismatch")
    if sha((fixture_dir / "draft_document.json").read_bytes()) != draft_sha:
        raise ValueError("draft revision changed")

    window_rows = sorted(windows["windows"], key=lambda row: row["start_id"])
    # Speaker labels and U IDs preserve the semantic input. Word-level identity
    # and timestamps stay in the frozen host fixture, not the model request.
    source_rows = [{"id": utterance["id"], "speaker": utterance["speaker"],
                    "text": utterance["text"]}
                   for window in window_rows for utterance in window["utterances"]]
    source_ids = [row["id"] for row in source_rows]
    if (len(source_rows) != 98 or len(source_ids) != len(set(source_ids))
            or source_ids != sorted(source_ids)):
        raise ValueError("diagnostic source order or count changed")
    original_items = selected["items"]
    if ([item["item_id"] for item in original_items]
            != fixture_manifest["selected_inventory_item_ids"]):
        raise ValueError("original selected inventory changed")
    source_by_id = {row["id"]: row for row in source_rows}
    if (SUPPLEMENTAL_PROBE["source_ids"] != ["U00126"]
            or SUPPLEMENTAL_PROBE["source_quote"] not in source_by_id["U00126"]["text"]
            or source_by_id["U00126"]["speaker"] != SUPPLEMENTAL_PROBE["speaker"]):
        raise ValueError("supplemental source-only probe changed")
    items = [*original_items, SUPPLEMENTAL_PROBE]
    item_ids = [item["item_id"] for item in items]
    if (len(items) != 8 or len(item_ids) != len(set(item_ids))):
        raise ValueError("selected inventory changed")
    if any(not set(item["source_ids"]).issubset(source_ids) for item in items):
        raise ValueError("inventory citation absent from frozen excerpts")

    all_units = draft_units(draft)
    all_unit_ids = [unit["unit_id"] for unit in all_units]
    selected_unit_ids = set()
    for row in selected_units["units"]:
        selected_unit_ids.update(pointer_unit_ids(row["json_pointer"]))
    if not selected_unit_ids.issubset(all_unit_ids):
        raise ValueError("frozen draft unit unavailable")
    visible_units = [unit for unit in all_units
                     if unit["unit_id"] in selected_unit_ids]
    coverage = []
    for window in window_rows:
        ids = {utterance["id"] for utterance in window["utterances"]}
        coverage.append({
            "window_id": window["window_id"],
            "start_id": window["start_id"],
            "end_id": window["end_id"],
            "item_ids": [
                item["item_id"] for item in items
                if ids.intersection(item["source_ids"])
            ],
        })
    payload = {
        "MODE": "reconcile",
        "SOURCE_EXCERPTS": source_rows,
        "EXCERPT_SCOPE": {
            "selection": "six_frozen_disjoint_real_windows",
            "included_utterance_count": len(source_rows),
            "total_utterance_count": fixture_manifest["source"]["utterance_count"],
            "source_sha256": source_sha,
        },
        "SOURCE_WINDOWS_TO_RECHECK": coverage,
        "INDEPENDENT_SOURCE_INVENTORY": {
            "schema_version": "gemini_source_inventory_merged_v2",
            "source_sha256": source_sha,
            "coverage": coverage,
            "items": items,
            "risk_warnings": [],
        },
        "SOURCE_RISK_WARNINGS": {"uncited_by_source": {}, "other": []},
        "DRAFT_DOCUMENT": draft,
        "DRAFT_UNITS": visible_units,
        "DRAFT_REFERENCE_UNIT_IDS": all_unit_ids,
        "PRIOR_FINDINGS": [],
        "RECONCILE_SCOPE": {
            "mode": "frozen_diagnostic",
            "primary_window_ids": [row["window_id"] for row in coverage],
            "primary_source_ids": source_ids,
        },
        "TASK": (
            "Проверь каждый пункт перечня по показанным первичным репликам "
            "и сравни с тем же полным черновиком. Проверь также все окна "
            "на новые активные задачи и неверные утверждения. "
            "Это выборочная диагностика без публикации; не делай выводов "
            "о непоказанных репликах. Верни формат, заданный системной "
            "инструкцией и схемой ответа."
        ),
    }
    user_content = canonical(payload).decode("utf-8")
    requests = {}
    for name, prompt_file, schema_file, schema_name in (
        ("baseline", "prompt_reconcile_v2.md",
         "output_schema_reconcile_v2.json", "gemini_inventory_reconcile_v2"),
        ("compact", "prompt_diagnostic_compact_v1.md",
         "output_schema_diagnostic_compact_v1.json",
         "gemini_diagnostic_compact_v1"),
    ):
        prompt = (PROMPTS / prompt_file).read_text(encoding="utf-8")
        schema = json.loads((PROMPTS / schema_file).read_text(encoding="utf-8"))
        request = {
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True,
                                "schema": schema},
            },
            "max_completion_tokens": CAP,
            "reasoning": {"effort": "medium"},
            "plugins": [],
        }
        path = request_dir / f"{name}.json"
        data = canonical(request) + b"\n"
        write_private_new(path, data)
        requests[name] = {
            "path": str(path), "sha256": sha(data),
            "prompt_sha256": sha(prompt.encode("utf-8")),
            "schema_sha256": sha(canonical(schema)),
        }
    request_manifest = {
        "run_id": run_id,
        "source_sha256": source_sha,
        "draft_sha256": draft_sha,
        "cases": item_ids,
        "requests": {name: f"{name}.json" for name in requests},
    }
    receipt = {
        "run_id": run_id,
        "fixture_manifest_sha256": sha(fixture_manifest_bytes),
        "fixture_dir": str(fixture_dir),
        "same_user_content_sha256": sha(user_content.encode("utf-8")),
        "source_utterances": len(source_rows),
        "output_cap_tokens_per_call": CAP,
        "request_receipts": requests,
        "supplemental_source_only_probe": {
            "item_id": SUPPLEMENTAL_PROBE["item_id"],
            "sha256": sha(canonical(SUPPLEMENTAL_PROBE)),
            "provenance": "fixture_authored_before_paid_output",
        },
        "rubric_excluded": True,
        "no_publication": True,
    }
    manifest_out = request_dir / "manifest.json"
    write_private_new(manifest_out, canonical(request_manifest) + b"\n")
    write_private_new(request_dir / "request_receipt.json", canonical(receipt) + b"\n")
    return {"manifest": str(manifest_out), "manifest_sha256": sha(manifest_out.read_bytes()),
            "user_content_sha256": receipt["same_user_content_sha256"],
            "request_sha256": {key: value["sha256"] for key, value in requests.items()},
            "items": item_ids, "utterances": len(source_rows),
            "draft_units": len(visible_units)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    args.requests.mkdir(mode=0o700, parents=True, exist_ok=False)
    print(json.dumps(build(args.fixtures, args.requests, args.run_id),
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
