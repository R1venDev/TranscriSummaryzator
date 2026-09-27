"""Gemini v2 source inventory layered on the immutable v1 patch contract.

The inventory makes an all-covered report mechanically inspectable. These
checks establish coordinate and report consistency, never semantic entailment.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from ..luna_v1.audit import (
    AUDIT_SCHEMA_ID,
    apply_audit,
    coverage_warnings,
    source_windows,
    validate_audit,
)


GEMINI_AUDIT_SCHEMA_ID_V2 = "gemini_summary_audit_v2"
GEMINI_AUDIT_PROMPT_PATH_V2 = Path(__file__).with_name("prompt_audit_v2.md")
GEMINI_AUDIT_SCHEMA_V2 = json.loads(
    Path(__file__).with_name("output_schema_audit_v2.json").read_text(encoding="utf-8")
)

_MATERIAL_KINDS = {"action", "decision", "constraint", "question", "correction", "technical"}
_COVERAGE_KEYS = {
    "window_id", "start_id", "end_id", "salient", "draft_coverage",
    "finding_indices", "material_items",
}
_V1_COVERAGE_KEYS = _COVERAGE_KEYS - {"material_items"}
_ITEM_KEYS = {"kind", "claim", "source_ids", "draft_targets", "finding_indices"}
_TARGET_KEYS = {"section", "index"}
_SECTIONS = {"meeting", "main", "timecodes", "tasks", "questions", "technical", "ideas", "verification", "chapters"}


def legacy_report_v1(report: dict) -> dict:
    """Project a v2 report to v1 for the unchanged patch validator/applicator."""
    if not isinstance(report, dict) or report.get("schema_version") != GEMINI_AUDIT_SCHEMA_ID_V2:
        raise ValueError("Gemini audit v2: wrong schema version")
    projected = copy.deepcopy(report)
    projected["schema_version"] = AUDIT_SCHEMA_ID
    if not isinstance(projected.get("coverage"), list):
        raise ValueError("Gemini audit v2: coverage must be an array")
    for number, row in enumerate(projected["coverage"]):
        if not isinstance(row, dict):
            raise ValueError("Gemini audit v2: coverage row must be an object")
        # Inventory metadata is diagnostic. Preserve the exact v1 coverage
        # fields for its patch-safety validator, even if an inventory field is
        # missing, extra, or malformed.
        projected["coverage"][number] = {
            key: value for key, value in row.items() if key in _V1_COVERAGE_KEYS
        }
    return projected


def _is_index(value: object) -> bool:
    return type(value) is int and value >= 0


def validate_gemini_audit_v2(report: dict, draft: dict, source_index: dict,
                             mode: str = "audit") -> dict:
    """Keep v1 finding/patch safety mandatory; inventory is diagnostic.

    Malformed inventory must not discard valid source-referenced corrections.
    Call coverage_warnings_gemini_v2 and retain its warnings beside the raw
    report; warnings downgrade the quality status. The report is not mutated.
    """
    validate_audit(legacy_report_v1(report), draft, source_index, mode=mode)
    return report


def apply_gemini_audit_v2(draft: dict, report: dict, source_index: dict,
                          mode: str = "audit") -> tuple[dict, list[dict]]:
    """Apply unchanged v1 patch mechanics after validating the v2 inventory."""
    validate_gemini_audit_v2(report, draft, source_index, mode=mode)
    return apply_audit(draft, legacy_report_v1(report), source_index, mode=mode)


def coverage_warnings_gemini_v2(report: dict, source_index: dict,
                                draft: dict | None = None) -> list[dict]:
    """Report inventory defects without discarding already validated patches.

    Call after validate_gemini_audit_v2. A citation/target link is only a
    mechanical consistency check and does not establish source entailment.
    """
    warnings = coverage_warnings(legacy_report_v1(report), source_index)
    all_ids = list(source_index["by_id"])
    positions = {source_id: number for number, source_id in enumerate(all_ids)}
    findings = report["findings"]
    linked_findings: set[int] = set()
    if len(all_ids) > 1 and all(
        row["draft_coverage"] == "no_material_item" for row in report["coverage"]
    ):
        warnings.append({"code": "all_windows_no_material"})

    for number, (row, expected) in enumerate(zip(report["coverage"], source_windows(source_index))):
        if set(row) != _COVERAGE_KEYS:
            warnings.append({"code": "material_inventory_fields", "coverage_index": number})
        if "material_items" not in row:
            warnings.append({"code": "material_inventory_missing", "coverage_index": number})
            continue
        items = row["material_items"]
        if not isinstance(items, list):
            warnings.append({"code": "material_inventory_not_array", "coverage_index": number})
            continue
        if not items and row["draft_coverage"] != "no_material_item":
            warnings.append({"code": "material_inventory_empty", "coverage_index": number})
        if row["draft_coverage"] == "no_material_item" and (items or row["finding_indices"]):
            warnings.append({"code": "no_material_item_conflict", "coverage_index": number})

        start = positions[expected["start_id"]]
        end = positions[expected["end_id"]]
        window_ids = set(all_ids[start:end + 1])
        row_item_findings: set[int] = set()
        for item_number, item in enumerate(items):
            location = {"coverage_index": number, "material_index": item_number}
            if not isinstance(item, dict) or set(item) != _ITEM_KEYS:
                warnings.append({"code": "material_item_fields", **location})
                if not isinstance(item, dict):
                    continue
            kind = item.get("kind")
            if not isinstance(kind, str) or kind not in _MATERIAL_KINDS:
                warnings.append({"code": "material_kind_invalid", **location})
            claim = item.get("claim")
            if not isinstance(claim, str) or not claim.strip():
                warnings.append({"code": "material_claim_empty", **location})

            raw_ids = item.get("source_ids")
            ids = set(raw_ids) if (isinstance(raw_ids, list) and raw_ids
                                    and all(isinstance(value, str) for value in raw_ids)) else set()
            if (not ids or len(ids) != len(raw_ids)
                    or any(value not in positions for value in ids)):
                warnings.append({"code": "material_source_invalid", **location})
            if not window_ids.intersection(ids):
                warnings.append({"code": "material_source_outside_window", **location})

            seen_targets: set[tuple[str, int]] = set()
            targets = item.get("draft_targets")
            if not isinstance(targets, list):
                warnings.append({"code": "material_targets_not_array", **location})
                targets = []
            for target_number, target in enumerate(targets):
                target_location = {**location, "target_index": target_number}
                if not isinstance(target, dict) or set(target) != _TARGET_KEYS:
                    warnings.append({"code": "material_target_fields", **target_location})
                    continue
                section, position = target.get("section"), target.get("index")
                if (not isinstance(section, str) or section not in _SECTIONS
                        or not _is_index(position)
                        or (section == "meeting" and position != 0)
                        or (draft is not None and section != "meeting"
                            and position >= len(draft.get(section, [])))):
                    warnings.append({"code": "material_target_invalid", **target_location})
                    continue
                key = (section, position)
                if key in seen_targets:
                    warnings.append({"code": "material_target_duplicate", **target_location})
                seen_targets.add(key)

            seen_links: set[int] = set()
            links = item.get("finding_indices")
            if not isinstance(links, list):
                warnings.append({"code": "material_findings_not_array", **location})
                links = []
            for link_number, pointer in enumerate(links):
                link_location = {**location, "finding_index": link_number}
                if not _is_index(pointer) or pointer >= len(findings):
                    warnings.append({"code": "material_finding_invalid", **link_location})
                    continue
                if pointer in seen_links:
                    warnings.append({"code": "material_finding_duplicate", **link_location})
                seen_links.add(pointer)
                if not ids.intersection(findings[pointer]["source_ids"]):
                    warnings.append({"code": "material_finding_source_mismatch", **link_location})
                if not window_ids.intersection(findings[pointer]["source_ids"]):
                    warnings.append({"code": "material_finding_outside_window", **link_location})
            if not seen_targets and not seen_links:
                warnings.append({"code": "material_unmapped", **location})
            if kind == "action" and not any(section == "tasks" for section, _ in seen_targets) and not seen_links:
                warnings.append({"code": "action_missing_task_or_finding", **location})
            row_item_findings.update(seen_links)

        if row_item_findings != set(row["finding_indices"]):
            warnings.append({"code": "material_finding_links_mismatch", "coverage_index": number})
        linked_findings.update(row_item_findings)
    if linked_findings != set(range(len(findings))):
        warnings.append({"code": "material_findings_unlinked"})
    return warnings
