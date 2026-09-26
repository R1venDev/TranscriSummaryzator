"""Source-only Gemini inventory with exact, non-overlapping primary coverage.

This module deliberately never sees a Luna draft. It verifies source coordinates,
report shape, and accounting only. It cannot establish entailment or guarantee
that the model noticed every material fact in the transcript.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


INVENTORY_SCHEMA_ID = "gemini_source_inventory_v1"
MERGED_INVENTORY_SCHEMA_ID = "gemini_source_inventory_merged_v1"
INVENTORY_PROMPT_PATH = Path(__file__).with_name("inventory_prompt_v1.md")
INVENTORY_SCHEMA = json.loads(
    Path(__file__).with_name("inventory_schema_v1.json").read_text(encoding="utf-8")
)

_ITEM_KEYS = {
    "kind", "claim", "source_ids", "speaker", "actor", "recipient", "action",
    "modality", "condition", "alternatives", "correction_of", "uncertainty",
}
_ITEM_KINDS = {
    "action", "decision", "constraint", "question", "answer", "correction",
    "technical", "context",
}
_MODALITIES = {
    "proposed", "committed", "conditional", "completed", "cancelled",
    "rejected", "open", "answered", "observation", "explanation",
    "hypothesis", "unknown",
}
_ASSESSMENTS = {"material_items", "no_material_items", "uncertain"}
_COVERAGE_KEYS = {"window_id", "start_id", "end_id", "assessment", "item_indices"}
_REPORT_KEYS = {"schema_version", "segment_id", "coverage", "items"}
_SHORT_REPLY_PREFIXES = (
    "да", "нет", "но", "точнее", "подожди", "погоди", "поправка",
    "а вот", "то есть", "вернее", "however", "actually", "yes", "no",
)


def _source_utterances(source_text: str) -> tuple[dict, list[dict]]:
    try:
        source = json.loads(source_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("inventory source is not canonical JSON") from exc
    if not isinstance(source, dict) or not isinstance(source.get("utterances"), list):
        raise ValueError("inventory source has no utterance list")
    utterances = source["utterances"]
    if not utterances:
        raise ValueError("inventory source has no utterances")
    ids = []
    previous_start = -1
    for number, row in enumerate(utterances):
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                or not isinstance(row.get("text"), str)
                or type(row.get("start_ms")) is not int
                or type(row.get("end_ms")) is not int
                or row["start_ms"] < previous_start
                or row["end_ms"] < row["start_ms"]):
            raise ValueError(f"inventory source utterance {number} is malformed")
        previous_start = row["start_ms"]
        ids.append(row["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("inventory source has duplicate IDs")
    return source, utterances


def _cut_score(utterances: list[dict], boundary: int, target: int) -> float:
    """Prefer a nearby pause; avoid separating short replies/corrections.

    This is boundary hygiene, not a semantic classifier. Explicit overlap is
    required because an apparently safe pause may still split a dependency.
    """
    before, after = utterances[boundary - 1], utterances[boundary]
    gap_ms = max(0, after["start_ms"] - before["end_ms"])
    next_text = after["text"].strip().casefold()
    followup = len(next_text) < 90 and any(
        next_text == cue or next_text.startswith(cue + " ") or next_text.startswith(cue + ",")
        for cue in _SHORT_REPLY_PREFIXES
    )
    punctuation = before["text"].rstrip().endswith((".", "!", "?", "…"))
    return abs(boundary - target) - min(gap_ms, 30_000) / 2_500 + (8 if followup else 0) - (1 if punctuation else 0)


def _boundaries(utterances: list[dict], count: int) -> list[int]:
    length = len(utterances)
    boundaries = [0]
    radius = max(4, min(20, length // (count * 4)))
    for part in range(1, count):
        target = round(length * part / count)
        lower = max(boundaries[-1] + 1, target - radius)
        upper = min(length - (count - part), target + radius)
        candidates = range(lower, upper + 1)
        boundary = min(candidates, key=lambda candidate: (
            _cut_score(utterances, candidate, target), abs(candidate - target), candidate
        ))
        boundaries.append(boundary)
    boundaries.append(length)
    return boundaries


def _windows(primary: list[dict], segment_id: str, count: int) -> list[dict]:
    count = min(count, len(primary))
    width, remainder = divmod(len(primary), count)
    result = []
    position = 0
    for number in range(count):
        size = width + (number < remainder)
        rows = primary[position:position + size]
        result.append({
            "window_id": f"{segment_id}-W{number + 1:02d}",
            "start_id": rows[0]["id"], "end_id": rows[-1]["id"],
            "utterance_count": len(rows),
        })
        position += size
    return result


def plan_inventory_segments(
    source_text: str, *, count: int = 3, overlap: int = 4,
    windows_per_segment: int = 4,
) -> list[dict]:
    """Plan 2–3 contiguous primary segments with marked contextual overlap.

    Use the exact canonical text returned by ``load_source``. Every utterance
    is primary in exactly one segment. No entire utterance is truncated.
    Pauses help choose boundaries; overlap protects short replies and
    corrections when a boundary still crosses a semantic dependency.
    """
    if type(count) is not int or count not in (2, 3):
        raise ValueError("inventory count must be 2 or 3")
    if type(overlap) is not int or not 0 <= overlap <= 12:
        raise ValueError("inventory overlap must be 0..12 utterances")
    if type(windows_per_segment) is not int or not 1 <= windows_per_segment <= 12:
        raise ValueError("inventory windows_per_segment must be 1..12")
    source, utterances = _source_utterances(source_text)
    count = min(count, len(utterances))
    boundaries = _boundaries(utterances, count) if count > 1 else [0, len(utterances)]
    fingerprint = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    segments = []
    for number, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
        segment_id = f"S{number:02d}"
        primary = utterances[start:end]
        before = utterances[max(0, start - overlap):start]
        after = utterances[end:min(len(utterances), end + overlap)]
        segments.append({
            "segment_id": segment_id,
            "source_fingerprint": fingerprint,
            "source_name": source.get("source_name"),
            "meeting_date": source.get("meeting_date"),
            "participants_by_transcript": source.get("participants_by_transcript", []),
            "context_before": before,
            "primary_utterances": primary,
            "context_after": after,
            "coverage_windows": _windows(primary, segment_id, windows_per_segment),
        })
    return segments


def build_inventory_input(segment: dict) -> str:
    """Canonical source-only message; trusted task follows all untrusted text."""
    if not isinstance(segment, dict) or not segment.get("primary_utterances"):
        raise ValueError("inventory segment is empty")
    fields = (
        "segment_id", "source_fingerprint", "source_name", "meeting_date",
        "participants_by_transcript", "context_before", "primary_utterances",
        "context_after", "coverage_windows",
    )
    if set(segment) != set(fields):
        raise ValueError("inventory segment fields changed")
    payload = {
        "SOURCE_SEGMENT": {key: segment[key] for key in fields},
        "TASK": (
            "По PRIMARY_UTTERANCES составь независимый от черновика перечень "
            "существенных смыслов и заполни все COVERAGE_WINDOWS. "
            "CONTEXT_BEFORE/AFTER нужны для связей и поздних поправок; "
            "они не являются дополнительными primary-репликами."
        ),
    }
    # Preserve before → primary → after reading order while keeping deterministic
    # insertion order for semantic cache identity.
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _string_or_null(value: object, where: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{where}: expected nonempty string or null")


def _ids(value: object, visible: set[str], where: str, *, allow_empty: bool) -> list[str]:
    if (not isinstance(value, list) or (not allow_empty and not value)
            or any(not isinstance(item, str) or item not in visible for item in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{where}: duplicate, unknown, or missing source ID")
    return value


def validate_inventory_report(report: dict, segment: dict) -> dict:
    """Reject impossible coordinates and malformed bookkeeping, not meaning."""
    if not isinstance(report, dict) or set(report) != _REPORT_KEYS:
        raise ValueError("inventory report fields are invalid")
    if report["schema_version"] != INVENTORY_SCHEMA_ID:
        raise ValueError("inventory report schema version mismatch")
    if report["segment_id"] != segment["segment_id"]:
        raise ValueError("inventory report segment mismatch")
    primary = segment["primary_utterances"]
    primary_ids = [row["id"] for row in primary]
    primary_set = set(primary_ids)
    visible = primary_set | {row["id"] for row in segment["context_before"]} | {
        row["id"] for row in segment["context_after"]
    }
    items, coverage = report["items"], report["coverage"]
    if not isinstance(items, list) or len(items) > 256:
        raise ValueError("inventory items must be a bounded array")
    if not isinstance(coverage, list) or len(coverage) != len(segment["coverage_windows"]):
        raise ValueError("inventory coverage must include every window")
    for number, item in enumerate(items):
        where = f"inventory items[{number}]"
        if not isinstance(item, dict) or set(item) != _ITEM_KEYS:
            raise ValueError(f"{where}: fields are invalid")
        if (not isinstance(item["kind"], str) or item["kind"] not in _ITEM_KINDS
                or not isinstance(item["modality"], str)
                or item["modality"] not in _MODALITIES):
            raise ValueError(f"{where}: kind or modality is invalid")
        if not isinstance(item["claim"], str) or not item["claim"].strip():
            raise ValueError(f"{where}: empty claim")
        cited = _ids(item["source_ids"], visible, where, allow_empty=False)
        if not primary_set.intersection(cited):
            raise ValueError(f"{where}: context-only item has no primary support")
        _ids(item["correction_of"], visible, f"{where}.correction_of", allow_empty=True)
        for field in ("speaker", "actor", "recipient", "action", "condition", "uncertainty"):
            _string_or_null(item[field], f"{where}.{field}")
        if item["speaker"] is not None:
            source_speakers = {
                row["speaker"] for row in (
                    segment["context_before"] + primary + segment["context_after"]
                ) if row["id"] in cited
            }
            if item["speaker"] not in source_speakers:
                raise ValueError(f"{where}: cited speaker mismatch")
        alternatives = item["alternatives"]
        if (not isinstance(alternatives, list)
                or any(not isinstance(option, str) or not option.strip() for option in alternatives)
                or len(set(alternatives)) != len(alternatives)):
            raise ValueError(f"{where}: alternatives are malformed")
        if item["kind"] == "action" and item["action"] is None:
            raise ValueError(f"{where}: an action needs its object and operation")

    linked: set[int] = set()
    positions = {source_id: position for position, source_id in enumerate(primary_ids)}
    for number, (row, expected) in enumerate(zip(coverage, segment["coverage_windows"])):
        where = f"inventory coverage[{number}]"
        if not isinstance(row, dict) or set(row) != _COVERAGE_KEYS:
            raise ValueError(f"{where}: fields are invalid")
        if any(row[key] != expected[key] for key in ("window_id", "start_id", "end_id")):
            raise ValueError(f"{where}: window identity mismatch")
        if not isinstance(row["assessment"], str) or row["assessment"] not in _ASSESSMENTS:
            raise ValueError(f"{where}: assessment is invalid")
        indices = row["item_indices"]
        if (not isinstance(indices, list)
                or any(type(pointer) is not int or pointer < 0 or pointer >= len(items)
                       for pointer in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError(f"{where}: invalid item indices")
        if row["assessment"] == "no_material_items" and indices:
            raise ValueError(f"{where}: no-material label conflicts with items")
        if row["assessment"] == "material_items" and not indices:
            raise ValueError(f"{where}: material label needs items")
        members = set(primary_ids[positions[expected["start_id"]]:
                                  positions[expected["end_id"]] + 1])
        for pointer in indices:
            if not members.intersection(items[pointer]["source_ids"]):
                raise ValueError(f"{where}: item has no source in this window")
        linked.update(indices)
    if linked != set(range(len(items))):
        raise ValueError("inventory has unlinked items")
    return report


def merge_inventory_reports(reports: list[dict], segments: list[dict],
                            source_index: dict) -> dict:
    """Merge all source-only results with host IDs and exact primary coverage."""
    if not isinstance(reports, list) or len(reports) != len(segments):
        raise ValueError("inventory needs one report per segment")
    validate_inventory_plan(segments, source_index)
    all_items = []
    all_coverage = []
    for report, segment in zip(reports, segments):
        validate_inventory_report(report, segment)
        item_ids = []
        for number, item in enumerate(report["items"], 1):
            item_id = f"{segment['segment_id']}-I{number:03d}"
            item_ids.append(item_id)
            all_items.append({"item_id": item_id, "segment_id": segment["segment_id"], **item})
            if len(all_items) > 512:
                raise ValueError("merged inventory exceeds reconciliation limit")
        for row in report["coverage"]:
            all_coverage.append({
                "segment_id": segment["segment_id"],
                "window_id": row["window_id"], "start_id": row["start_id"],
                "end_id": row["end_id"], "assessment": row["assessment"],
                "item_ids": [item_ids[pointer] for pointer in row["item_indices"]],
            })
    return {
        "schema_version": MERGED_INVENTORY_SCHEMA_ID,
        "source_sha256": source_index["source_sha256"],
        "source_fingerprint": segments[0]["source_fingerprint"],
        "primary_utterance_count": len(source_index["by_id"]),
        "segment_count": len(segments),
        "coverage": all_coverage,
        "items": all_items,
    }


def validate_inventory_plan(segments: list[dict], source_index: dict) -> list[dict]:
    """Preflight exact segment bytes/coordinates against a loaded source.

    This can run before any API dispatch. It excludes a stale or manipulated
    plan even when its source IDs happen to match the current transcript.
    """
    if not isinstance(segments, list) or not segments:
        raise ValueError("inventory has no segments")
    source_ids = list(source_index["by_id"])
    primary_ids = [row["id"] for segment in segments for row in segment["primary_utterances"]]
    if primary_ids != source_ids or len(set(primary_ids)) != len(primary_ids):
        raise ValueError("inventory segments do not cover the complete source exactly once")
    expected_source = {
        "source_kind": "TRANSCRIPT_SOURCE",
        "source_name": source_index["source_name"],
        "meeting_date": source_index["meeting_date"],
        "duration_ms": source_index["duration_ms"],
        "participants_by_transcript": source_index["participants"],
        "unattributed_speech": source_index["unattributed_speech"],
        "utterances": list(source_index["by_id"].values()),
    }
    canonical = json.dumps(expected_source, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    cursor = 0
    for number, segment in enumerate(segments, 1):
        if segment["segment_id"] != f"S{number:02d}":
            raise ValueError("inventory segment identity or order changed")
        if segment["source_fingerprint"] != fingerprint:
            raise ValueError("inventory source fingerprint mismatch")
        if (segment["source_name"] != source_index["source_name"]
                or segment["meeting_date"] != source_index["meeting_date"]
                or segment["participants_by_transcript"] != source_index["participants"]):
            raise ValueError("inventory segment metadata differs from canonical source")
        for row in (segment["context_before"] + segment["primary_utterances"]
                    + segment["context_after"]):
            if source_index["by_id"].get(row["id"]) != row:
                raise ValueError("inventory segment differs from canonical source")
        before_ids = [row["id"] for row in segment["context_before"]]
        after_ids = [row["id"] for row in segment["context_after"]]
        end = cursor + len(segment["primary_utterances"])
        if (len(before_ids) > 12 or len(after_ids) > 12
                or before_ids != source_ids[cursor - len(before_ids):cursor]
                or after_ids != source_ids[end:end + len(after_ids)]):
            raise ValueError("inventory context is not adjacent marked overlap")
        expected_windows = _windows(segment["primary_utterances"],
                                    segment["segment_id"],
                                    len(segment["coverage_windows"]))
        if segment["coverage_windows"] != expected_windows:
            raise ValueError("inventory coverage windows differ from primary source")
        cursor = end
    return segments
