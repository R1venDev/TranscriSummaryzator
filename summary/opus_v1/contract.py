"""Opus evidence checks over the existing Luna targeted-patch contract.

Quote presence is a mechanical check, not proof that a finding is entailed by
the cited speech. Semantic review remains the judge model's responsibility.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from ..luna_v1.audit import (
    AUDIT_SCHEMA_ID, apply_audit, coverage_warnings, source_windows,
    validate_audit,
)

OPUS_AUDIT_SCHEMA_ID = "opus_summary_audit_v1"
OPUS_AUDIT_PROMPT_PATH = Path(__file__).with_name("prompt_audit_v1.md")
OPUS_AUDIT_SCHEMA = json.loads(
    Path(__file__).with_name("output_schema_audit_v1.json").read_text(encoding="utf-8")
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
