"""Opus evidence checks over the existing Luna targeted-patch contract.

Quote presence is a mechanical check, not proof that a finding is entailed by
the cited speech. Semantic review remains the judge model's responsibility.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

from ..gemini_v1.inventory_contract import _windows, validate_inventory_plan
from ..luna_v1.audit import (
    AUDIT_SCHEMA_ID, _working_draft, apply_audit, coverage_warnings, source_windows,
    validate_audit,
)

OPUS_AUDIT_SCHEMA_ID = "opus_summary_audit_v1"
OPUS_AUDIT_PROMPT_PATH = Path(__file__).with_name("prompt_audit_v1.md")
OPUS_AUDIT_SCHEMA = json.loads(
    Path(__file__).with_name("output_schema_audit_v1.json").read_text(encoding="utf-8")
)
OPUS_SEGMENT_SCHEMA_ID = "opus_segment_audit_v2"
OPUS_SEGMENT_PROMPT_PATH = Path(__file__).with_name("prompt_audit_v2.md")
OPUS_SEGMENT_SCHEMA = json.loads(
    Path(__file__).with_name("output_schema_audit_v2.json").read_text(encoding="utf-8")
)
OPUS_SEGMENT_SCHEMA_ID_V3 = "opus_segment_audit_v3"
OPUS_SEGMENT_PROMPT_PATH_V3 = Path(__file__).with_name("prompt_audit_v3.md")
OPUS_SEGMENT_SCHEMA_V3 = json.loads(
    Path(__file__).with_name("output_schema_audit_v3.json").read_text(encoding="utf-8")
)
_COVERAGE_KEYS = {"window_id", "start_id", "end_id", "salient",
                  "source_quote", "draft_coverage", "finding_indices"}
_FINDING_KEYS = {"severity", "kind", "description", "evidence_quote",
                 "source_ids", "affected", "status", "patch_indices"}
_MAX_QUOTE_LENGTH = 240
_SOURCE_ID = re.compile(r"U[0-9]{5}\Z")


def _item_source_refs(value: object) -> set[str]:
    """Collect source coordinates inside an item, including nested task fields."""
    if isinstance(value, list):
        refs: set[str] = set()
        for item in value:
            refs.update(_item_source_refs(item))
        return refs
    if not isinstance(value, dict):
        return set()
    refs = set()
    for key, item in value.items():
        if key in {"start_id", "end_id"} and isinstance(item, str) and _SOURCE_ID.fullmatch(item):
            refs.add(item)
        elif key == "source_ids" and isinstance(item, list):
            refs.update(source_id for source_id in item
                        if isinstance(source_id, str) and _SOURCE_ID.fullmatch(source_id))
        elif isinstance(item, (dict, list)):
            refs.update(_item_source_refs(item))
    return refs


def _quote(value: object, where: str) -> str:
    if (not isinstance(value, str) or not value.strip()
            or len(value) > _MAX_QUOTE_LENGTH):
        raise ValueError(f"{where}: invalid source quote")
    return value.strip()


def legacy_report_v1(report: dict) -> dict:
    """Project Opus metadata away for the tested Luna patch validator."""
    if (not isinstance(report, dict)
            or set(report) != {"schema_version", "coverage", "findings", "patches"}
            or report.get("schema_version") != OPUS_AUDIT_SCHEMA_ID
            or not isinstance(report.get("coverage"), list)
            or not isinstance(report.get("findings"), list)):
        raise ValueError("Opus audit: malformed report")
    projected = copy.deepcopy(report)
    projected["schema_version"] = AUDIT_SCHEMA_ID
    for number, row in enumerate(projected["coverage"]):
        if not isinstance(row, dict) or set(row) != _COVERAGE_KEYS:
            raise ValueError(f"Opus coverage[{number}]: fields")
        del row["source_quote"]
    for number, finding in enumerate(projected["findings"]):
        if not isinstance(finding, dict) or set(finding) != _FINDING_KEYS:
            raise ValueError(f"Opus findings[{number}]: fields")
        del finding["evidence_quote"]
    return projected


def _mechanical_report(report: dict, source_index: dict, mode: str) -> dict:
    projected = legacy_report_v1(report)
    if mode == "verify":
        # Luna's patch validator also requires full-source coverage bookkeeping.
        # The focused Opus verify report intentionally covers only supplied
        # evidence windows. Temporary rows satisfy the mechanical validator;
        # they are never returned, stored, or represented as model review.
        projected["coverage"] = [{
            "window_id": window["window_id"],
            "start_id": window["start_id"],
            "end_id": window["end_id"],
            "salient": "Адресная проверка отдельных прежних находок",
            "draft_coverage": "covered", "finding_indices": [],
        } for window in source_windows(source_index)]
    return projected


def validate_opus_audit(report: dict, draft: dict, source_index: dict,
                        *, mode: str = "audit",
                        expected_windows: list[dict] | None = None,
                        verify_source_ids: set[str] | None = None) -> dict:
    """Require exact evidence for every finding and safe original-draft patches."""
    projected = _mechanical_report(report, source_index, mode)
    validate_audit(projected, draft, source_index, mode=mode)
    by_id = source_index["by_id"]
    if verify_source_ids is not None:
        if (mode != "verify" or not isinstance(verify_source_ids, set)
                or not verify_source_ids or not verify_source_ids.issubset(by_id)):
            raise ValueError("verify: invalid_source_scope")
    for number, finding in enumerate(report["findings"]):
        if (verify_source_ids is not None
                and not set(finding["source_ids"]).issubset(verify_source_ids)):
            raise ValueError(f"findings[{number}]: source_outside_verify_scope")
        quote = _quote(finding["evidence_quote"], f"findings[{number}].evidence_quote")
        if not any(quote in by_id[source_id]["text"] for source_id in finding["source_ids"]):
            raise ValueError(f"findings[{number}]: evidence_quote_not_in_source")
    by_id_order = list(by_id)
    previous_end = -1
    for number, row in enumerate(report["coverage"]):
        _quote(row["source_quote"], f"coverage[{number}].source_quote")
        if mode == "verify":
            if (row["window_id"] != f"V{number + 1:02d}"
                    or row["start_id"] not in by_id or row["end_id"] not in by_id):
                raise ValueError(f"coverage[{number}]: verify_window_identity")
            start = by_id_order.index(row["start_id"])
            end = by_id_order.index(row["end_id"])
            if start > end or start <= previous_end:
                raise ValueError(f"coverage[{number}]: verify_window_order")
            previous_end = end
    if mode == "verify" and not report["coverage"]:
        raise ValueError("verify: coverage_missing")
    if expected_windows is not None:
        if (mode != "verify" or not isinstance(expected_windows, list)
                or [(row["window_id"], row["start_id"], row["end_id"])
                    for row in report["coverage"]]
                != [(row.get("window_id"), row.get("start_id"), row.get("end_id"))
                    for row in expected_windows if isinstance(row, dict)]
                or len(report["coverage"]) != len(expected_windows)):
            raise ValueError("verify: coverage_scope_mismatch")
    if verify_source_ids is not None:
        for number, patch in enumerate(report["patches"]):
            if patch["operation"] == "remove":
                continue
            section, index = patch["section"], patch["index"]
            updated = json.loads(patch["item_json"])
            original = (draft["meeting"] if section == "meeting" else
                        draft[section][index] if patch["operation"] == "replace" else {})
            new_refs = _item_source_refs(updated) - _item_source_refs(original)
            if not new_refs.issubset(verify_source_ids):
                raise ValueError(f"patches[{number}]: new_source_outside_verify_scope")
    return report


def coverage_warnings_opus(report: dict, source_index: dict) -> list[dict]:
    """Keep coverage quote errors diagnostic so valid patches can survive."""
    projected = legacy_report_v1(report)
    full_windows = source_windows(source_index)
    is_full_audit = (len(report["coverage"]) == len(full_windows)
                     and all(row["window_id"] == expected["window_id"]
                             and row["start_id"] == expected["start_id"]
                             and row["end_id"] == expected["end_id"]
                             for row, expected in zip(report["coverage"], full_windows)))
    warnings = coverage_warnings(projected, source_index) if is_full_audit else []
    all_ids = list(source_index["by_id"])
    for number, row in enumerate(report["coverage"]):
        start, end = all_ids.index(row["start_id"]), all_ids.index(row["end_id"])
        quote = row["source_quote"].strip()
        if not any(quote in source_index["by_id"][source_id]["text"]
                   for source_id in all_ids[start:end + 1]):
            warnings.append({"code": "coverage_quote_not_in_window", "coverage_index": number})
        if not is_full_audit:
            members = set(all_ids[start:end + 1])
            for pointer in row["finding_indices"]:
                if type(pointer) is not int or pointer < 0 or pointer >= len(report["findings"]):
                    warnings.append({"code": "unknown_finding_index", "coverage_index": number})
                elif not members.intersection(report["findings"][pointer]["source_ids"]):
                    warnings.append({"code": "finding_outside_window", "coverage_index": number})
            if row["draft_coverage"] in {"partial", "missing"} and not any(
                type(pointer) is int and 0 <= pointer < len(report["findings"])
                and report["findings"][pointer]["kind"] == "omission"
                and members.intersection(report["findings"][pointer]["source_ids"])
                for pointer in row["finding_indices"]
            ):
                warnings.append({"code": "coverage_without_omission", "coverage_index": number})
    return warnings


def apply_opus_audit(draft: dict, report: dict, source_index: dict,
                     *, mode: str = "audit",
                     expected_windows: list[dict] | None = None,
                     verify_source_ids: set[str] | None = None) -> tuple[dict, list[dict]]:
    """Apply only addressed patches, preserving the original draft."""
    validate_opus_audit(report, draft, source_index, mode=mode,
                        expected_windows=expected_windows,
                        verify_source_ids=verify_source_ids)
    revised, _ = apply_audit(draft, _mechanical_report(report, source_index, mode),
                             source_index, mode=mode)
    unresolved = copy.deepcopy([finding for finding in report["findings"]
                                if finding["status"] == "unresolved"])
    return revised, unresolved


_SEGMENT_REPORT_KEYS = {"schema_version", "segment_id", "coverage", "findings", "patches"}
_SEGMENT_COVERAGE_KEYS = _COVERAGE_KEYS | {"draft_evidence", "reviewed_anchor_ids"}
_DRAFT_EVIDENCE_KEYS = {"section", "index", "quote"}
_DRAFT_SECTIONS = {"meeting", "main", "timecodes", "tasks", "questions",
                   "technical", "ideas", "verification", "chapters"}
_DRAFT_COVERAGE = {"covered", "partial", "missing", "no_material_item"}
_RISK_PATTERNS = {
    "number": re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?:\s*(?:%|тыс\.?|млн|мс|сек|мин|час|дн(?:я|ей)?))?", re.I),
    "condition": re.compile(r"\b(?:если|когда|при|пока|в случае|только если|if|when|unless)\b", re.I),
    "alternative": re.compile(r"\b(?:или|либо|or|either)\b", re.I),
    "negation": re.compile(r"\b(?:не|нет|без|not|no)\b", re.I),
    "role_or_status": re.compile(
        r"\b(?:предлага\w*|обеща\w*|сделаю|сделает|нужно|надо|должен|решили|"
        r"отмен\w*|готов\w*|поруч\w*|propos\w*|commit\w*|cancel\w*)\b", re.I),
    "correction": re.compile(r"\b(?:поправк\w*|точнее|вернее|вместо|исправ\w*|actually|correction)\b", re.I),
}


def risk_anchors_for_primary(primary: list[dict]) -> list[dict]:
    """Deterministic lexical cues, not model answers or evidence of review."""
    anchors = []
    for row in primary:
        kinds = [kind for kind, pattern in _RISK_PATTERNS.items()
                 if pattern.search(row["text"])]
        if kinds:
            anchors.append({"anchor_id": f"R{len(anchors) + 1:03d}",
                            "source_id": row["id"], "kinds": kinds})
    return anchors


def _segment_source_scope(segment: dict, source_index: dict) -> tuple[list[str], set[str]]:
    """Check one exact inventory segment without pretending it covers the rest."""
    by_id = source_index.get("by_id") if isinstance(source_index, dict) else None
    if not isinstance(by_id, dict) or not by_id:
        raise ValueError("Opus segment source index is invalid")
    source = {
        "source_kind": "TRANSCRIPT_SOURCE",
        "source_name": source_index["source_name"],
        "meeting_date": source_index["meeting_date"],
        "duration_ms": source_index["duration_ms"],
        "participants_by_transcript": source_index["participants"],
        "unattributed_speech": source_index["unattributed_speech"],
        "utterances": list(by_id.values()),
    }
    canonical = json.dumps(source, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    if (not isinstance(segment, dict)
            or segment.get("source_fingerprint") != hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            or segment.get("source_name") != source_index["source_name"]
            or segment.get("meeting_date") != source_index["meeting_date"]
            or segment.get("participants_by_transcript") != source_index["participants"]):
        raise ValueError("Opus segment differs from canonical source")
    primary = segment.get("primary_utterances")
    before = segment.get("context_before")
    after = segment.get("context_after")
    windows = segment.get("coverage_windows")
    if (not isinstance(primary, list) or not primary
            or not isinstance(before, list) or not isinstance(after, list)
            or not isinstance(windows, list) or not windows
            or len(before) > 12 or len(after) > 12):
        raise ValueError("Opus segment primary/context is invalid")
    source_ids = list(by_id)
    start_id = primary[0].get("id") if isinstance(primary[0], dict) else None
    if start_id not in by_id:
        raise ValueError("Opus segment primary start is unknown")
    start = source_ids.index(start_id)
    end = start + len(primary)
    rows = list(by_id.values())
    if (start < len(before)
            or primary != rows[start:end]
            or before != rows[start - len(before):start]
            or after != rows[end:end + len(after)]
            or windows != _windows(primary, segment["segment_id"], len(windows))):
        raise ValueError("Opus segment differs from canonical primary/context")
    primary_ids = [row["id"] for row in primary]
    visible = set(primary_ids) | {row["id"] for row in before + after}
    return primary_ids, visible


def _draft_string_fields(value: object):
    """Quotes must occur in actual text fields, never just in source IDs."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for child in value:
            yield from _draft_string_fields(child)
    elif isinstance(value, dict):
        for key, child in value.items():
            if key not in {"source_ids", "start_id", "end_id", "id", "action_id"}:
                yield from _draft_string_fields(child)


def _segment_mechanical_report(report: dict, source_index: dict) -> dict:
    """Use the existing patch validator with clearly synthetic bookkeeping."""
    return {
        "schema_version": AUDIT_SCHEMA_ID,
        "coverage": [{
            "window_id": window["window_id"],
            "start_id": window["start_id"],
            "end_id": window["end_id"],
            "salient": "Служебная проверка формы сегментных патчей",
            "draft_coverage": "partial",
            "finding_indices": [],
        } for window in source_windows(source_index)],
        "findings": [{key: copy.deepcopy(value) for key, value in finding.items()
                      if key != "evidence_quote"}
                     for finding in report["findings"]],
        "patches": copy.deepcopy(report["patches"]),
    }


def validate_opus_segment_report(report: dict, segment: dict, draft: dict,
                                 source_index: dict, *,
                                 expected_schema_id: str = OPUS_SEGMENT_SCHEMA_ID) -> dict:
    """Check exact primary windows, source quotes and current-draft quotes.

    These mechanical checks cannot prove that a quote entails a model claim.
    """
    primary_ids, visible_ids = _segment_source_scope(segment, source_index)
    primary_set = set(primary_ids)
    if expected_schema_id not in {OPUS_SEGMENT_SCHEMA_ID, OPUS_SEGMENT_SCHEMA_ID_V3}:
        raise ValueError("unknown Opus segment schema")
    if (not isinstance(report, dict) or set(report) != _SEGMENT_REPORT_KEYS
            or report.get("schema_version") != expected_schema_id
            or report.get("segment_id") != segment["segment_id"]
            or not isinstance(report.get("coverage"), list)
            or not isinstance(report.get("findings"), list)
            or not isinstance(report.get("patches"), list)):
        raise ValueError("Opus segment report shape or identity is invalid")
    windows = segment["coverage_windows"]
    if len(report["coverage"]) != len(windows):
        raise ValueError("Opus segment coverage must include every primary window")
    if any(not isinstance(finding, dict) or set(finding) != _FINDING_KEYS
           for finding in report["findings"]):
        raise ValueError("Opus segment finding fields are invalid")
    prepared = _working_draft(draft)
    by_id = source_index["by_id"]
    positions = {source_id: number for number, source_id in enumerate(primary_ids)}
    anchors = risk_anchors_for_primary(segment["primary_utterances"])
    for number, (row, window) in enumerate(zip(report["coverage"], windows)):
        where = f"coverage[{number}]"
        if (not isinstance(row, dict) or set(row) != _SEGMENT_COVERAGE_KEYS
                or any(row[key] != window[key] for key in ("window_id", "start_id", "end_id"))
                or not isinstance(row["salient"], str) or not row["salient"].strip()
                or not isinstance(row["draft_coverage"], str)
                or row["draft_coverage"] not in _DRAFT_COVERAGE):
            raise ValueError(f"{where}: primary window or fields mismatch")
        start, end = positions[window["start_id"]], positions[window["end_id"]]
        members = set(primary_ids[start:end + 1])
        source_quote = _quote(row["source_quote"], f"{where}.source_quote")
        if not any(source_quote in by_id[source_id]["text"] for source_id in members):
            raise ValueError(f"{where}: source_quote_not_in_primary_window")
        expected_anchors = {item["anchor_id"] for item in anchors
                            if item["source_id"] in members}
        reviewed_anchors = row["reviewed_anchor_ids"]
        if (not isinstance(reviewed_anchors, list)
                or any(not isinstance(item, str) for item in reviewed_anchors)
                or len(reviewed_anchors) != len(set(reviewed_anchors))
                or set(reviewed_anchors) != expected_anchors):
            raise ValueError(f"{where}: reviewed_anchor_ids do not match this window")
        evidence = row["draft_evidence"]
        if (not isinstance(evidence, list) or len(evidence) > 32
                or (row["draft_coverage"] == "covered" and not evidence)
                or (row["draft_coverage"] in {"missing", "no_material_item"} and evidence)):
            raise ValueError(f"{where}: draft_evidence and coverage disagree")
        for item_number, item in enumerate(evidence):
            item_where = f"{where}.draft_evidence[{item_number}]"
            if (not isinstance(item, dict) or set(item) != _DRAFT_EVIDENCE_KEYS
                    or not isinstance(item["section"], str)
                    or item["section"] not in _DRAFT_SECTIONS
                    or type(item["index"]) is not int or item["index"] < 0):
                raise ValueError(f"{item_where}: invalid draft coordinate")
            section, index = item["section"], item["index"]
            if section == "meeting":
                if index != 0:
                    raise ValueError(f"{item_where}: meeting index must be zero")
                target = prepared["meeting"]
            else:
                if index >= len(prepared[section]):
                    raise ValueError(f"{item_where}: index outside draft")
                target = prepared[section][index]
            quote = _quote(item["quote"], f"{item_where}.quote")
            if _SOURCE_ID.fullmatch(quote) or not any(
                    quote in value for value in _draft_string_fields(target)):
                raise ValueError(f"{item_where}: draft_quote_not_in_target")
        links = row["finding_indices"]
        if (not isinstance(links, list)
                or any(type(pointer) is not int or pointer < 0
                       or pointer >= len(report["findings"]) for pointer in links)
                or len(links) != len(set(links))):
            raise ValueError(f"{where}: invalid finding indices")
        for pointer in links:
            if not members.intersection(report["findings"][pointer].get("source_ids", [])):
                raise ValueError(f"{where}: finding outside primary window")
        if row["draft_coverage"] in {"partial", "missing"} and not links:
            raise ValueError(f"{where}: partial/missing needs a finding")
    linked = {pointer for row in report["coverage"] for pointer in row["finding_indices"]}
    if linked != set(range(len(report["findings"]))):
        raise ValueError("Opus segment has an unlinked finding")
    validate_audit(_segment_mechanical_report(report, source_index), draft,
                   source_index, mode="audit")
    for number, finding in enumerate(report["findings"]):
        where = f"findings[{number}]"
        if not isinstance(finding, dict) or set(finding) != _FINDING_KEYS:
            raise ValueError(f"{where}: fields")
        ids = finding["source_ids"]
        if (not isinstance(ids, list) or not ids
                or len(ids) != len(set(ids))
                or not set(ids).issubset(visible_ids)
                or not set(ids).intersection(primary_set)):
            raise ValueError(f"{where}: source outside primary/context scope")
        quote = _quote(finding["evidence_quote"], f"{where}.evidence_quote")
        if not any(quote in by_id[source_id]["text"] for source_id in ids):
            raise ValueError(f"{where}: evidence_quote_not_in_source")
    for number, patch in enumerate(report["patches"]):
        if patch["operation"] == "remove":
            continue
        section, index = patch["section"], patch["index"]
        updated = json.loads(patch["item_json"])
        original = (prepared["meeting"] if section == "meeting" else
                    prepared[section][index] if patch["operation"] == "replace" else {})
        new_refs = _item_source_refs(updated) - _item_source_refs(original)
        if not new_refs.issubset(visible_ids):
            raise ValueError(f"patches[{number}]: new_source_outside_segment_scope")
    return report


def merge_opus_segment_reports(reports: list[dict], segments: list[dict],
                               draft: dict, source_index: dict, *,
                               expected_schema_id: str = OPUS_SEGMENT_SCHEMA_ID,
                               ) -> tuple[dict, list[dict]]:
    """Project all segment findings to a legacy patch report without lost conflicts.

    The original segment reports remain the authoritative coverage evidence. Full
    windows here are derived bookkeeping for the existing patch applier.
    """
    if (not isinstance(reports, list) or not isinstance(segments, list)
            or len(reports) != len(segments) or not reports):
        raise ValueError("Opus merge requires every segment report")
    validate_inventory_plan(segments, source_index)
    for report, segment in zip(reports, segments):
        validate_opus_segment_report(report, segment, draft, source_index,
                                     expected_schema_id=expected_schema_id)
    patches_with_origin = [(segment["segment_id"], pointer, copy.deepcopy(patch))
                           for report, segment in zip(reports, segments)
                           for pointer, patch in enumerate(report["patches"])]
    targets: dict[tuple[str, int], list[int]] = {}
    for number, (_segment_id, _pointer, patch) in enumerate(patches_with_origin):
        targets.setdefault((patch["section"], patch["index"]), []).append(number)
    conflict_targets = {target: positions for target, positions in targets.items()
                        if len(positions) > 1}
    conflict_blocked = {position for positions in conflict_targets.values()
                        for position in positions}
    prepared = _working_draft(draft)
    visible_by_segment = {
        segment["segment_id"]: _segment_source_scope(segment, source_index)[1]
        for segment in segments
    }
    cross_scope: dict[int, set[str]] = {}
    for number, (segment_id, _pointer, patch) in enumerate(patches_with_origin):
        if patch["operation"] not in {"replace", "remove"}:
            continue
        section, index = patch["section"], patch["index"]
        original = prepared["meeting"] if section == "meeting" else prepared[section][index]
        unseen_refs = _item_source_refs(original) - visible_by_segment[segment_id]
        if unseen_refs:
            cross_scope[number] = unseen_refs
    blocked = conflict_blocked | set(cross_scope)
    offsets = []
    cursor = 0
    for report in reports:
        offsets.append(cursor)
        cursor += len(report["patches"])
    # A finding is indivisible: if one of its corrections conflicts, retain
    # its evidence as unresolved rather than applying a partial correction.
    changed = True
    while changed:
        changed = False
        for offset, report in zip(offsets, reports):
            for finding in report["findings"]:
                indices = {offset + pointer for pointer in finding["patch_indices"]}
                if indices & blocked and not indices.issubset(blocked):
                    blocked.update(indices)
                    changed = True
    kept = [number for number in range(len(patches_with_origin)) if number not in blocked]
    remap = {old: new for new, old in enumerate(kept)}
    findings = []
    for offset, report in zip(offsets, reports):
        for finding in report["findings"]:
            projected = {key: copy.deepcopy(value) for key, value in finding.items()
                         if key != "evidence_quote"}
            old_indices = [offset + pointer for pointer in finding["patch_indices"]]
            if any(pointer in blocked for pointer in old_indices):
                projected["status"] = "unresolved"
                projected["patch_indices"] = []
                if any(pointer in cross_scope for pointer in old_indices):
                    projected["description"] += (
                        " Автоматическая правка не применена: элемент также "
                        "ссылается на реплики вне проверенного участка."
                    )
                elif any(pointer in conflict_blocked for pointer in old_indices):
                    projected["description"] += (
                        " Автоматическая правка не применена: несколько проверок "
                        "затронули один элемент черновика."
                    )
                else:
                    projected["description"] += (
                        " Автоматическая правка не применена вместе со связанной "
                        "конфликтующей правкой."
                    )
            else:
                projected["patch_indices"] = [remap[pointer] for pointer in old_indices]
            findings.append(projected)
    all_ids = list(source_index["by_id"])
    window_status_by_id = {}
    for report, segment in zip(reports, segments):
        primary_ids = [row["id"] for row in segment["primary_utterances"]]
        positions = {source_id: number for number, source_id in enumerate(primary_ids)}
        for row in report["coverage"]:
            start, end = positions[row["start_id"]], positions[row["end_id"]]
            for source_id in primary_ids[start:end + 1]:
                window_status_by_id[source_id] = row["draft_coverage"]
    coverage = []
    for window in source_windows(source_index):
        start, end = all_ids.index(window["start_id"]), all_ids.index(window["end_id"])
        members = set(all_ids[start:end + 1])
        statuses = {window_status_by_id[source_id] for source_id in members}
        status = (next(iter(statuses)) if len(statuses) == 1 else "partial")
        linked = [number for number, finding in enumerate(findings)
                  if members.intersection(finding["source_ids"])]
        coverage.append({
            "window_id": window["window_id"],
            "start_id": window["start_id"],
            "end_id": window["end_id"],
            "salient": "Служебная агрегация проверенных сегментных окон",
            "draft_coverage": status,
            "finding_indices": linked,
        })
    merged = {
        "schema_version": AUDIT_SCHEMA_ID,
        "coverage": coverage,
        "findings": findings,
        "patches": [patches_with_origin[number][2] for number in kept],
    }
    validate_audit(merged, draft, source_index, mode="audit")
    warnings = [{
        "code": "cross_segment_patch_conflict",
        "section": target[0], "index": target[1],
        "segment_ids": sorted({patches_with_origin[position][0]
                               for position in positions}),
    } for target, positions in sorted(conflict_targets.items())]
    warnings.extend({
        "code": "cross_scope_patch_requires_review",
        "section": patches_with_origin[number][2]["section"],
        "index": patches_with_origin[number][2]["index"],
        "segment_id": patches_with_origin[number][0],
        "outside_source_ids": sorted(unseen_refs),
    } for number, unseen_refs in sorted(cross_scope.items()))
    return merged, warnings
