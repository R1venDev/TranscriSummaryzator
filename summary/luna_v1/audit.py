"""Pure, versioned source-audit contract and deterministic section patching.

The model identifies semantic problems; these local checks only verify report
shape, references and mechanical application. They cannot prove entailment or
detect an omission that the model did not report.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from .contract import SCHEMA_ID, validate_document


AUDIT_SCHEMA_ID = "luna_summary_audit_v1"
AUDIT_PROMPT_PATH = Path(__file__).with_name("prompt_audit_v3.md")
AUDIT_SCHEMA = json.loads(Path(__file__).with_name("output_schema_audit_v1.json").read_text(encoding="utf-8"))
_ARRAY_SECTIONS = ("main", "timecodes", "tasks", "questions", "technical", "ideas", "verification", "chapters")
_SECTIONS = ("meeting",) + _ARRAY_SECTIONS
_KINDS = {"unsupported_claim", "omission", "role", "modality", "alternative", "condition", "number", "time", "late_correction", "duplication", "other"}
_SEVERITIES = {"critical", "major", "minor"}
_OPERATIONS = {"insert", "replace", "remove"}
_COVERAGE = {"covered", "partial", "missing", "no_material_item"}
_MAX_WINDOWS = 12
_MAX_FINDINGS = 128
_MAX_PATCHES = 128
_MAX_VERIFY_PATCHES = 64
_UNRESOLVED_REASON = "Автоматическая сверка не смогла надёжно исправить или подтвердить этот смысл; проверьте указанные реплики."


def _working_draft(draft: dict) -> dict:
    """Permit a parseable, partly malformed v1 draft to be fixed by patches.

    Missing top-level section arrays become empty. Invalid nested items stay
    intact for the auditor to replace. Final validation remains mandatory.
    """
    if not isinstance(draft, dict):
        raise ValueError("audit draft must be an object")
    prepared = copy.deepcopy(draft)
    if prepared.get("schema_version") not in (None, SCHEMA_ID):
        raise ValueError("audit draft has an unsupported schema version")
    prepared.setdefault("schema_version", SCHEMA_ID)
    prepared.setdefault("meeting", {"topic": "", "project": None})
    if not isinstance(prepared["meeting"], dict):
        raise ValueError("audit meeting must be an object")
    for section in _ARRAY_SECTIONS:
        prepared.setdefault(section, [])
        if not isinstance(prepared[section], list):
            raise ValueError(f"audit {section} must be an array")
    return prepared


def _windows_from_ids(source_ids: list[str]) -> list[dict]:
    if not source_ids or any(not isinstance(item, str) or not item for item in source_ids):
        raise ValueError("audit source has no valid utterance IDs")
    count = min(_MAX_WINDOWS, len(source_ids))
    width, remainder = divmod(len(source_ids), count)
    windows = []
    cursor = 0
    for number in range(count):
        size = width + (1 if number < remainder else 0)
        ids = source_ids[cursor:cursor + size]
        windows.append({"window_id": f"W{number + 1:02d}",
                        "start_id": ids[0], "end_id": ids[-1], "utterance_count": size})
        cursor += size
    return windows


def source_windows(source_index: dict) -> list[dict]:
    """Expose the exact balanced source coverage required from every audit."""
    return _windows_from_ids(list(source_index["by_id"]))


def build_audit_input(source_text: str, draft: dict, *, mode: str = "audit",
                      prior_findings: list[dict] | tuple[dict, ...] = ()) -> str:
    """Build one stable data message; the system prompt is sent separately."""
    if mode not in {"audit", "verify"}:
        raise ValueError("unknown audit mode")
    source = json.loads(source_text)
    if not isinstance(source, dict) or not isinstance(source.get("utterances"), list):
        raise ValueError("audit source must have utterances")
    utterances = source["utterances"]
    if any(not isinstance(item, dict) or "id" not in item for item in utterances):
        raise ValueError("audit source has malformed utterances")
    if not isinstance(prior_findings, (list, tuple)) or any(not isinstance(item, dict) for item in prior_findings):
        raise ValueError("prior findings must be an array of objects")
    payload = {"MODE": mode, "TRANSCRIPT_SOURCE": source,
               "SOURCE_WINDOWS": _windows_from_ids([item["id"] for item in utterances]),
               "DRAFT_DOCUMENT": _working_draft(draft), "PRIOR_FINDINGS": list(prior_findings)}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _exact_keys(value: object, expected: set[str], location: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{location}: unexpected or missing fields")
    return value


def _index(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{location}: index must be a non-negative integer")
    return value


def _source_ids(ids: object, index: dict, location: str) -> None:
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"{location}: source_ids must be nonempty")
    known = index.get("by_id", {})
    if any(not isinstance(source_id, str) or source_id not in known for source_id in ids):
        raise ValueError(f"{location}: unknown source ID")


def validate_audit(report: dict, draft: dict, source_index: dict, *, mode: str = "audit") -> dict:
    """Validate a model report against the original draft and known source IDs."""
    if mode not in {"audit", "verify"}:
        raise ValueError("unknown audit mode")
    prepared = _working_draft(draft)
    _exact_keys(report, {"schema_version", "coverage", "findings", "patches"}, "audit")
    if report["schema_version"] != AUDIT_SCHEMA_ID:
        raise ValueError("audit: wrong schema version")
    findings, patches = report["findings"], report["patches"]
    if not isinstance(findings, list) or len(findings) > _MAX_FINDINGS:
        raise ValueError("audit: invalid findings array")
    if not isinstance(patches, list) or len(patches) > _MAX_PATCHES:
        raise ValueError("audit: invalid patches array")
    if mode == "verify" and len(patches) > _MAX_VERIFY_PATCHES:
        raise ValueError("verify mode has too many corrective patches")

    targets: set[tuple[str, int]] = set()
    for number, patch in enumerate(patches):
        where = f"patches[{number}]"
        _exact_keys(patch, {"section", "operation", "index", "item_json"}, where)
        section, operation = patch["section"], patch["operation"]
        if not isinstance(section, str) or not isinstance(operation, str) or section not in _SECTIONS or operation not in _OPERATIONS:
            raise ValueError(f"{where}: invalid section or operation")
        position = _index(patch["index"], f"{where}.index")
        if (section, position) in targets:
            raise ValueError(f"{where}: duplicate target")
        targets.add((section, position))
        if section == "meeting":
            if operation != "replace" or position != 0:
                raise ValueError(f"{where}: meeting only supports replace at index 0")
        else:
            size = len(prepared[section])
            if position > size or (operation != "insert" and position == size):
                raise ValueError(f"{where}: index exceeds original section")
        if operation == "remove":
            if patch["item_json"] is not None:
                raise ValueError(f"{where}: remove must use null item_json")
        else:
            if not isinstance(patch["item_json"], str):
                raise ValueError(f"{where}: item_json must be a string")
            try:
                item = json.loads(patch["item_json"])
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError(f"{where}: malformed item_json") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{where}: item_json must encode an object")

    referenced: set[int] = set()
    for number, finding in enumerate(findings):
        where = f"findings[{number}]"
        _exact_keys(finding, {"severity", "kind", "description", "source_ids", "affected", "status", "patch_indices"}, where)
        if (not isinstance(finding["severity"], str) or not isinstance(finding["kind"], str)
                or finding["severity"] not in _SEVERITIES or finding["kind"] not in _KINDS):
            raise ValueError(f"{where}: invalid severity or kind")
        if not isinstance(finding["description"], str) or not finding["description"].strip():
            raise ValueError(f"{where}: empty description")
        _source_ids(finding["source_ids"], source_index, where)
        affected = finding["affected"]
        if not isinstance(affected, list):
            raise ValueError(f"{where}: affected must be an array")
        affected_targets: set[tuple[str, int]] = set()
        for target_number, target in enumerate(affected):
            target_where = f"{where}.affected[{target_number}]"
            _exact_keys(target, {"section", "index"}, target_where)
            section = target["section"]
            if not isinstance(section, str) or section not in _SECTIONS:
                raise ValueError(f"{target_where}: invalid section")
            position = _index(target["index"], f"{target_where}.index")
            if section == "meeting":
                if position != 0:
                    raise ValueError(f"{target_where}: meeting index must be 0")
            elif position >= len(prepared[section]):
                raise ValueError(f"{target_where}: affected index exceeds original section")
            if (section, position) in affected_targets:
                raise ValueError(f"{target_where}: duplicate affected target")
            affected_targets.add((section, position))
        indices = finding["patch_indices"]
        if not isinstance(indices, list):
            raise ValueError(f"{where}: patch_indices must be an array")
        for pointer in indices:
            if _index(pointer, f"{where}.patch_indices") >= len(patches):
                raise ValueError(f"{where}: unknown patch index")
        if len(set(indices)) != len(indices):
            raise ValueError(f"{where}: duplicate patch index")
        if finding["status"] not in {"repaired", "unresolved"}:
            raise ValueError(f"{where}: invalid status")
        if finding["status"] == "repaired" and not indices:
            raise ValueError(f"{where}: repaired finding needs a patch")
        # The kind names the source problem, not whether the draft asserts a
        # conflicting claim. For example an unknown executor can be a role
        # finding even when the draft prudently leaves assignee null.
        if affected_targets:
            linked_patches = {
                (patches[pointer]["section"], patches[pointer]["index"]): patches[pointer]
                for pointer in indices
                if patches[pointer]["operation"] in {"replace", "remove"}
            }
            linked_targets = set(linked_patches)
            if not affected_targets.issubset(linked_targets):
                raise ValueError(f"{where}: every confident affected claim needs a corrective patch")
            for section, position in affected_targets:
                patch = linked_patches[(section, position)]
                if patch["operation"] == "replace":
                    original = prepared["meeting"] if section == "meeting" else prepared[section][position]
                    if json.loads(patch["item_json"]) == original:
                        raise ValueError(f"{where}: corrective patch cannot be a no-op")
        referenced.update(indices)
    if referenced != set(range(len(patches))):
        raise ValueError("audit: every patch must explain a repaired finding")

    coverage = report["coverage"]
    windows = source_windows(source_index)
    if not isinstance(coverage, list) or len(coverage) != len(windows):
        raise ValueError("audit: one coverage row is required for each source window")
    for number, (row, expected) in enumerate(zip(coverage, windows)):
        where = f"coverage[{number}]"
        _exact_keys(row, {"window_id", "start_id", "end_id", "salient", "draft_coverage", "finding_indices"}, where)
        if any(row[key] != expected[key] for key in ("window_id", "start_id", "end_id")):
            raise ValueError(f"{where}: source window identity mismatch")
        if not isinstance(row["salient"], str) or not row["salient"].strip():
            raise ValueError(f"{where}: empty salient content")
        if not isinstance(row["draft_coverage"], str) or row["draft_coverage"] not in _COVERAGE:
            raise ValueError(f"{where}: invalid draft coverage")
        links = row["finding_indices"]
        if not isinstance(links, list) or len(set(_index(item, f"{where}.finding_indices") for item in links)) != len(links):
            raise ValueError(f"{where}: invalid finding indices")
        # Window links and coverage labels are model bookkeeping. They do not
        # drive patches or the published document, so a mistaken association
        # (including a reference past the findings array) must not discard
        # otherwise valid source-referenced corrections. Call coverage_warnings
        # after validation to retain these diagnostics.
    return report


def coverage_warnings(report: dict, source_index: dict) -> list[dict]:
    """Describe nonfatal coverage bookkeeping errors in a validated report.

    Call after validate_audit; finding source IDs and patch targets are already
    checked there. This function never alters the model's report.
    """
    windows = source_windows(source_index)
    all_ids = list(source_index["by_id"])
    warnings = []
    for number, (row, expected) in enumerate(zip(report["coverage"], windows)):
        members = set(all_ids[all_ids.index(expected["start_id"]):all_ids.index(expected["end_id"]) + 1])
        valid_links = []
        for linked in row["finding_indices"]:
            if linked >= len(report["findings"]):
                warnings.append({"code": "unknown_finding_index", "coverage_index": number,
                                 "finding_index": linked})
                continue
            valid_links.append(linked)
            if not (set(report["findings"][linked]["source_ids"]) & members):
                warnings.append({"code": "finding_outside_window", "coverage_index": number,
                                 "finding_index": linked})
        if row["draft_coverage"] in {"partial", "missing"} and not any(
                report["findings"][linked]["kind"] == "omission"
                and set(report["findings"][linked]["source_ids"]) & members
                for linked in valid_links):
            warnings.append({"code": "coverage_without_omission", "coverage_index": number})
    return warnings


def _annotate_unresolved(document: dict, findings: list[dict]) -> None:
    """Keep unresolved source issues visible in the existing v1 document."""
    existing = {(item["text"], tuple(item["source_ids"])) for item in document["verification"]}
    for finding in findings:
        if finding["status"] != "unresolved":
            continue
        text = "Автоматическая проверка: " + finding["description"].strip()
        key = (text, tuple(finding["source_ids"]))
        if key not in existing:
            document["verification"].append({"text": text, "why_unresolved": _UNRESOLVED_REASON,
                                             "source_ids": list(finding["source_ids"])})
            existing.add(key)


def apply_audit(draft: dict, report: dict, source_index: dict, *, mode: str = "audit") -> tuple[dict, list[dict]]:
    """Apply a source-audit report without mutating the sealed draft.

    All indices refer to the original draft. Unresolved findings are added to
    the public verification section and also returned for private diagnostics.
    """
    validate_audit(report, draft, source_index, mode=mode)
    revised = _working_draft(draft)
    grouped: dict[str, list[dict]] = {}
    for patch in report["patches"]:
        grouped.setdefault(patch["section"], []).append(patch)
    for section, patches in grouped.items():
        for patch in sorted(patches, key=lambda item: item["index"], reverse=True):
            operation, position = patch["operation"], patch["index"]
            item = json.loads(patch["item_json"]) if operation != "remove" else None
            if section == "meeting":
                revised["meeting"] = item
            elif operation == "insert":
                revised[section].insert(position, item)
            elif operation == "replace":
                revised[section][position] = item
            else:
                del revised[section][position]
    unresolved = copy.deepcopy([finding for finding in report["findings"] if finding["status"] == "unresolved"])
    _annotate_unresolved(revised, unresolved)
    validate_document(revised, source_index)
    return revised, unresolved
