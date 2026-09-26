"""One source segment + the full draft: source inventory and local reconciliation.

The model performs the semantic review. These checks establish exact source
coordinates, complete mechanical bookkeeping, and safe whole-item patch shape;
they do not claim that a cited utterance entails a generated claim.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .inventory_contract import (
    INVENTORY_SCHEMA_ID,
    _windows,
    validate_inventory_plan,
    validate_inventory_report,
)
from .reconcile_contract import _citations_in, draft_units
from ..luna_v1.audit import (
    AUDIT_SCHEMA_ID,
    _working_draft,
    apply_audit,
    source_windows,
    validate_audit,
)


SEGMENT_REVIEW_SCHEMA_ID = "gemini_segment_review_v1"
SEGMENT_REVIEW_PROMPT_PATH = Path(__file__).with_name("segment_review_prompt_v1.md")
SEGMENT_REVIEW_SCHEMA = json.loads(
    Path(__file__).with_name("segment_review_schema_v1.json").read_text(encoding="utf-8")
)

_REPORT_KEYS = {
    "schema_version", "segment_id", "coverage", "items", "item_assessments",
    "draft_assessments", "findings", "patches",
}
_ASSESSMENT_KEYS = {"item_index", "status", "draft_targets", "finding_indices"}
_DRAFT_KEYS = {"unit_id", "status", "source_ids", "item_indices", "finding_indices"}
_TARGET_KEYS = {"section", "index"}
_ITEM_STATES = {"represented", "partial", "missing", "contradicted", "uncertain"}
_DRAFT_STATES = {"supported", "partial", "unsupported", "uncertain"}
_SECTIONS = {
    "meeting", "main", "timecodes", "tasks", "questions", "technical",
    "ideas", "verification", "chapters",
}
_ITEM_KEYS = {
    "kind", "claim", "source_ids", "speaker", "actor", "recipient",
    "action", "modality", "condition", "alternatives", "correction_of",
    "uncertainty",
}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _source_segment(source_text: str, segment: dict, source_index: dict) -> tuple[list[dict], set[str], set[str]]:
    """Check that the entire submitted primary span and overlap are exact bytes.

    Callers still validate the whole plan with ``validate_inventory_plan``;
    this local check prevents a stale or altered individual segment dispatch.
    """
    try:
        source = json.loads(source_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("segment review source is not JSON") from exc
    if (not isinstance(source, dict) or not isinstance(source.get("utterances"), list)
            or _canonical(source) != source_text):
        raise ValueError("segment review source is not canonical")
    if (not isinstance(segment, dict) or not isinstance(segment.get("segment_id"), str)
            or segment.get("source_fingerprint") != hashlib.sha256(source_text.encode("utf-8")).hexdigest()):
        raise ValueError("segment review source fingerprint differs")
    rows = source["utterances"]
    if (not rows or not isinstance(source_index, dict)
            or list(source_index.get("by_id", {}).values()) != rows):
        raise ValueError("segment review source index differs")
    if (segment.get("source_name") != source.get("source_name")
            or segment.get("meeting_date") != source.get("meeting_date")
            or segment.get("participants_by_transcript") != source.get("participants_by_transcript")):
        raise ValueError("segment review source metadata differs")
    positions = {row["id"]: number for number, row in enumerate(rows)}
    if len(positions) != len(rows):
        raise ValueError("segment review source has duplicate IDs")
    primary = segment.get("primary_utterances")
    before = segment.get("context_before")
    after = segment.get("context_after")
    if (not isinstance(primary, list) or not primary
            or not isinstance(before, list) or not isinstance(after, list)
            or len(before) > 12 or len(after) > 12):
        raise ValueError("segment review primary or context is invalid")
    first_id = primary[0].get("id") if isinstance(primary[0], dict) else None
    if first_id not in positions:
        raise ValueError("segment review primary ID is unknown")
    start = positions[first_id]
    end = start + len(primary)
    if (primary != rows[start:end]
            or before != rows[start - len(before):start]
            or after != rows[end:end + len(after)]
            or (before and start < len(before))):
        raise ValueError("segment review source span or adjacent overlap differs")
    windows = segment.get("coverage_windows")
    if (not isinstance(windows, list) or not windows
            or windows != _windows(primary, segment["segment_id"], len(windows))):
        raise ValueError("segment review coverage windows differ")
    primary_ids = {row["id"] for row in primary}
    visible_ids = primary_ids | {row["id"] for row in before + after}
    return rows, primary_ids, visible_ids


def draft_units_for_segment(draft: dict, segment: dict, source_index: dict) -> list[dict]:
    """Checklist for claims tied to this primary span, plus uncited claims.

    The complete draft is still sent. Uncited units are checked in every
    segment because no coordinate can safely assign them to only one part.
    """
    prepared = _working_draft(draft)
    primary_ids = {row["id"] for row in segment["primary_utterances"]}
    positions = {source_id: number for number, source_id in enumerate(source_index["by_id"])}
    selected = []
    for unit in draft_units(prepared):
        section, index, unit_id = unit["section"], unit["index"], unit["unit_id"]
        item = prepared["meeting"] if section == "meeting" else prepared[section][index]
        if section == "tasks" and unit_id.endswith(":action") and isinstance(item, dict):
            item = {key: item.get(key) for key in ("title", "description", "source_ids", "field_sources")}
        elif section == "tasks" and unit_id.endswith(":relations") and isinstance(item, dict):
            item = {key: item.get(key) for key in (
                "discussion_status", "assignee", "due", "recipient", "priority",
                "source_ids", "field_sources",
            )}
        elif section == "chapters" and isinstance(item, dict):
            if ":detail:" in unit_id:
                item = item["details"][int(unit_id.rsplit(":", 1)[1])]
            elif unit_id.endswith(":summary"):
                item = {key: item.get(key) for key in (
                    "topic", "summary", "source_ids", "start_id", "end_id",
                )}
        citations = _citations_in(item, positions, f"DRAFT_UNITS.{unit_id}")
        if not citations or citations.intersection(primary_ids):
            selected.append(unit)
    return selected


def build_segment_review_input(source_text: str, segment: dict, draft: dict,
                               source_index: dict) -> str:
    """Canonical private payload: exact segment + full draft, no prior verdicts.

    The source comes before the draft so the model inventories source meanings
    before comparing them. The task instruction follows both untrusted data
    fields and the pinned system prompt is supplied separately by the caller.
    """
    _source_segment(source_text, segment, source_index)
    prepared = _working_draft(draft)
    units = draft_units_for_segment(prepared, segment, source_index)
    payload = {
        "MODE": "segment_review",
        "SOURCE_SEGMENT": segment,
        "DRAFT_DOCUMENT": prepared,
        "DRAFT_UNITS_TO_CHECK": units,
        "TASK": (
            "Сначала независимо перечисли существенные смыслы PRIMARY_UTTERANCES "
            "и оцени каждое окно. Затем для каждого смысла проверь его отражение "
            "в полном черновике и проверь все DRAFT_UNITS_TO_CHECK по исходнику. "
            "Верни адресные findings и только безопасные whole-item patches."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _inventory_view(report: dict) -> dict:
    return {"schema_version": INVENTORY_SCHEMA_ID,
            "segment_id": report["segment_id"],
            "coverage": report["coverage"], "items": report["items"]}


def segment_inventory_view(report: dict) -> dict:
    """Deterministic adapter for ``merge_inventory_reports`` and focused verify."""
    if not isinstance(report, dict) or not all(
        key in report for key in ("segment_id", "coverage", "items")
    ):
        raise ValueError("segment review has no inventory view")
    return copy.deepcopy(_inventory_view(report))


def _legacy_report(report: dict, source_index: dict) -> dict:
    """Reuse v1 patch safety without presenting artificial coverage as evidence."""
    return {
        "schema_version": AUDIT_SCHEMA_ID,
        "coverage": [
            {"window_id": row["window_id"], "start_id": row["start_id"],
             "end_id": row["end_id"], "salient": "Механическая проверка патчей",
             "draft_coverage": "covered", "finding_indices": []}
            for row in source_windows(source_index)
        ],
        "findings": copy.deepcopy(report["findings"]),
        "patches": copy.deepcopy(report["patches"]),
    }


def validate_segment_review_report(report: dict, segment: dict, draft: dict,
                                   source_index: dict) -> dict:
    """Validate coordinates and exact assessment coverage, not semantic truth."""
    if (not isinstance(report, dict) or set(report) != _REPORT_KEYS
            or report.get("schema_version") != SEGMENT_REVIEW_SCHEMA_ID
            or report.get("segment_id") != segment.get("segment_id")):
        raise ValueError("segment review report shape, version, or segment differs")
    validate_inventory_report(_inventory_view(report), segment)
    prepared = _working_draft(draft)
    units = draft_units_for_segment(prepared, segment, source_index)
    item_rows = report["item_assessments"]
    draft_rows = report["draft_assessments"]
    if (not isinstance(item_rows, list) or len(item_rows) != len(report["items"])
            or not isinstance(draft_rows, list) or len(draft_rows) != len(units)):
        raise ValueError("segment review omitted an item or draft assessment")
    known_findings = len(report["findings"]) if isinstance(report["findings"], list) else 0
    def finding_links(value: object, where: str) -> None:
        if (not isinstance(value, list)
                or any(type(index) is not int or index < 0 or index >= known_findings for index in value)
                or len(set(value)) != len(value)):
            raise ValueError(f"{where}: invalid finding links")

    for number, row in enumerate(item_rows):
        where = f"item_assessments[{number}]"
        if (not isinstance(row, dict) or set(row) != _ASSESSMENT_KEYS
                or row["item_index"] != number or type(row["item_index"]) is not int
                or row["status"] not in _ITEM_STATES
                or not isinstance(row["draft_targets"], list)):
            raise ValueError(f"{where}: missing, duplicate, or invalid item assessment")
        finding_links(row["finding_indices"], where)
        for target in row["draft_targets"]:
            if not isinstance(target, dict) or set(target) != _TARGET_KEYS:
                raise ValueError(f"{where}: invalid draft target")
            section, index = target["section"], target["index"]
            if (section not in _SECTIONS or type(index) is not int or index < 0
                    or (section == "meeting" and index != 0)
                    or (section != "meeting" and index >= len(prepared[section]))):
                raise ValueError(f"{where}: unknown draft target")
    expected = [unit["unit_id"] for unit in units]
    for number, (row, unit_id) in enumerate(zip(draft_rows, expected)):
        where = f"draft_assessments[{number}]"
        if (not isinstance(row, dict) or set(row) != _DRAFT_KEYS
                or row["unit_id"] != unit_id or row["status"] not in _DRAFT_STATES
                or not isinstance(row["source_ids"], list)
                or not isinstance(row["item_indices"], list)):
            raise ValueError(f"{where}: missing, duplicate, or invalid draft assessment")
        finding_links(row["finding_indices"], where)
        if (any(not isinstance(source_id, str) or source_id not in source_index["by_id"]
                for source_id in row["source_ids"])
                or len(set(row["source_ids"])) != len(row["source_ids"])):
            raise ValueError(f"{where}: unknown or duplicate source ID")
        if (any(type(index) is not int or index < 0 or index >= len(report["items"])
                for index in row["item_indices"])
                or len(set(row["item_indices"])) != len(row["item_indices"])):
            raise ValueError(f"{where}: invalid item index")

    primary_ids = {row["id"] for row in segment["primary_utterances"]}
    visible_ids = primary_ids | {row["id"] for row in segment["context_before"] + segment["context_after"]}
    for number, finding in enumerate(report["findings"]):
        citations = finding.get("source_ids", []) if isinstance(finding, dict) else []
        if (not isinstance(citations, list) or not set(citations).issubset(visible_ids)
                or not set(citations).intersection(primary_ids)):
            raise ValueError(f"findings[{number}]: source outside this primary segment")
        if finding.get("status") == "unresolved" and finding.get("patch_indices"):
            raise ValueError(f"findings[{number}]: unresolved finding cannot apply patches")
    for number, row in enumerate(draft_rows):
        if not set(row["source_ids"]).issubset(visible_ids):
            raise ValueError(f"draft_assessments[{number}]: source outside segment")
    validate_audit(_legacy_report(report, source_index), prepared, source_index)
    return report


def segment_review_warnings(report: dict, segment: dict, draft: dict,
                            source_index: dict) -> list[dict]:
    """Visible incompleteness and questionable bookkeeping; never a fact judge."""
    validate_segment_review_report(report, segment, draft, source_index)
    warnings = []
    for number, row in enumerate(report["coverage"]):
        if row["assessment"] == "uncertain":
            warnings.append({"code": "source_window_uncertain", "window_id": row["window_id"]})
    for number, row in enumerate(report["item_assessments"]):
        if row["status"] in {"partial", "missing", "contradicted", "uncertain"} and not row["finding_indices"]:
            warnings.append({"code": "unexplained_item_defect", "item_index": number})
        if row["status"] == "represented" and not row["draft_targets"]:
            warnings.append({"code": "represented_without_target", "item_index": number})
        if row["status"] == "missing" and row["draft_targets"]:
            warnings.append({"code": "missing_with_draft_target", "item_index": number})
        item = report["items"][number]
        if (item["kind"] == "action" and item["modality"] in {"proposed", "committed", "conditional", "open"}
                and row["status"] == "represented"
                and not any(target["section"] == "tasks" for target in row["draft_targets"])):
            warnings.append({"code": "action_without_task_target", "item_index": number})
    for number, row in enumerate(report["draft_assessments"]):
        if row["status"] in {"partial", "unsupported", "uncertain"} and not row["finding_indices"]:
            warnings.append({"code": "unexplained_draft_defect", "unit_id": row["unit_id"]})
        if row["status"] == "supported" and not row["source_ids"] and row["unit_id"] != "meeting:0":
            warnings.append({"code": "supported_without_source", "unit_id": row["unit_id"]})
    linked = {index for row in report["item_assessments"] + report["draft_assessments"]
              for index in row["finding_indices"]}
    for number in set(range(len(report["findings"]))) - linked:
        warnings.append({"code": "finding_not_linked_to_assessment", "finding_index": number})
    if not report["items"] and len(segment["primary_utterances"]) > 1:
        warnings.append({"code": "all_segment_inventory_empty", "segment_id": segment["segment_id"]})
    return warnings


def apply_segment_review_report(draft: dict, report: dict, segment: dict,
                                source_index: dict) -> tuple[dict, list[dict]]:
    """Apply a single segment's whole-item patches to its pinned original draft.

    Multiple segment reports must be reconciled for duplicate patch targets
    before application; never apply them sequentially to shifted indices.
    """
    validate_segment_review_report(report, segment, draft, source_index)
    return apply_audit(draft, _legacy_report(report, source_index), source_index)


def merge_segment_review_reports(reports: list[dict], segments: list[dict],
                                 draft: dict, source_index: dict, *,
                                 allow_partial: bool = False) -> tuple[dict, list[dict]]:
    """Combine safe nonconflicting patches against the same pinned Luna draft.

    Patches on the same original target with different operations/content are
    withheld and their findings become unresolved. An already unresolved
    affected target also blocks automatic replacement by another segment.
    Returned report can be passed to ``apply_audit`` once; callers must never
    apply segment patches sequentially because inserts/removals shift indices.
    """
    if (not isinstance(reports, list) or not isinstance(segments, list)
            or not reports or (len(reports) > len(segments) if allow_partial
                              else len(reports) != len(segments))):
        raise ValueError("segment reviews need one report per segment")
    # The entire frozen source plan is required even when only its completed
    # prefix can be used after a later segment becomes unavailable.
    validate_inventory_plan(segments, source_index)
    for report, segment in zip(reports, segments):
        validate_segment_review_report(report, segment, draft, source_index)
    prepared = _working_draft(draft)
    patch_records: list[tuple[int, int, dict]] = []
    for segment_number, report in enumerate(reports):
        for local_number, patch in enumerate(report["patches"]):
            patch_records.append((segment_number, local_number, patch))
    by_target: dict[tuple[str, int], list[tuple[int, int, dict]]] = {}
    for record in patch_records:
        patch = record[2]
        by_target.setdefault((patch["section"], patch["index"]), []).append(record)

    blocked_targets = {
        target for target, records in by_target.items()
        if len({_canonical(record[2]) for record in records}) > 1
    }
    # An explicit unresolved finding against a current element is evidence
    # that another segment's replacement is not yet safe to apply blindly.
    for report in reports:
        for finding in report["findings"]:
            if finding["status"] == "unresolved" and not finding["patch_indices"]:
                blocked_targets.update(
                    (target["section"], target["index"])
                    for target in finding["affected"]
                )
    # A duplicate insert aimed at a different index can create the same card
    # twice. Keep both findings visible and withhold these uncertain inserts.
    insert_values: dict[tuple[str, str], set[tuple[str, int]]] = {}
    for target, records in by_target.items():
        patch = records[0][2]
        if patch["operation"] == "insert":
            insert_values.setdefault((patch["section"], patch["item_json"]), set()).add(target)
    for targets in insert_values.values():
        if len(targets) > 1:
            blocked_targets.update(targets)

    # If one patch of a multi-patch finding conflicts, withhold all of its
    # patches and every other segment's patch on those same original targets.
    # Repeat until shared patch links stop propagating the hold.
    blocked_refs: set[tuple[int, int]] = {
        (segment_number, local_number)
        for segment_number, local_number, patch in patch_records
        if (patch["section"], patch["index"]) in blocked_targets
    }
    changed = True
    while changed:
        changed = False
        blocked_target_now = {
            (patch["section"], patch["index"])
            for segment_number, local_number, patch in patch_records
            if (segment_number, local_number) in blocked_refs
        }
        for segment_number, local_number, patch in patch_records:
            ref = (segment_number, local_number)
            if (patch["section"], patch["index"]) in blocked_target_now and ref not in blocked_refs:
                blocked_refs.add(ref)
                changed = True
        for segment_number, report in enumerate(reports):
            for finding in report["findings"]:
                refs = {(segment_number, pointer) for pointer in finding["patch_indices"]}
                if refs.intersection(blocked_refs) and not refs.issubset(blocked_refs):
                    blocked_refs.update(refs)
                    changed = True
    blocked_targets = {
        (patch["section"], patch["index"])
        for segment_number, local_number, patch in patch_records
        if (segment_number, local_number) in blocked_refs
    }
    warnings = [
        {"code": "cross_segment_patch_conflict", "section": section, "index": index}
        for section, index in sorted(blocked_targets)
    ]

    merged_patches: list[dict] = []
    index_for_target: dict[tuple[str, int], int] = {}
    for segment_number, local_number, patch in patch_records:
        if (segment_number, local_number) in blocked_refs:
            continue
        target = (patch["section"], patch["index"])
        if target not in index_for_target:
            index_for_target[target] = len(merged_patches)
            merged_patches.append(copy.deepcopy(patch))

    merged_findings = []
    for segment_number, report in enumerate(reports):
        for finding in report["findings"]:
            merged = copy.deepcopy(finding)
            original_refs = {(segment_number, pointer) for pointer in finding["patch_indices"]}
            if original_refs.intersection(blocked_refs):
                merged["status"] = "unresolved"
                merged["patch_indices"] = []
            else:
                merged["patch_indices"] = list(dict.fromkeys(
                    index_for_target[
                        (report["patches"][pointer]["section"],
                         report["patches"][pointer]["index"])
                    ] for pointer in finding["patch_indices"]
                ))
            merged_findings.append(merged)
    combined = {
        "schema_version": AUDIT_SCHEMA_ID,
        "coverage": [
            {"window_id": row["window_id"], "start_id": row["start_id"],
             "end_id": row["end_id"], "salient": "Механическая проверка патчей",
             "draft_coverage": "covered", "finding_indices": []}
            for row in source_windows(source_index)
        ],
        "findings": merged_findings,
        "patches": merged_patches,
    }
    validate_audit(combined, prepared, source_index)
    return combined, warnings
