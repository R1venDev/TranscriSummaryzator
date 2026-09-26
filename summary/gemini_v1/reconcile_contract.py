"""Source-grounded, bidirectional Gemini reconciliation of an independent inventory.

This module deliberately keeps semantic judgment in the model. Local checks
establish source identity, citation coordinates, report accounting, and safe
application of whole-item patches. A cited utterance is not proof of a claim.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from ..luna_v1.audit import (
    AUDIT_SCHEMA_ID,
    _working_draft,
    apply_audit,
    source_windows,
    validate_audit,
)


RECONCILE_SCHEMA_ID = "gemini_inventory_reconcile_v1"
RECONCILE_PROMPT_PATH = Path(__file__).with_name("prompt_reconcile_v1.md")
RECONCILE_SCHEMA = json.loads(
    Path(__file__).with_name("output_schema_reconcile_v1.json").read_text(encoding="utf-8")
)

_ARRAY_SECTIONS = (
    "main", "timecodes", "tasks", "questions", "technical", "ideas",
    "verification", "chapters",
)
_SECTIONS = {"meeting", *_ARRAY_SECTIONS}
_INVENTORY_STATES = {"represented", "partial", "missing", "contradicted", "uncertain"}
_DRAFT_STATES = {"supported", "supported_uninventoried", "partial", "unsupported", "uncertain"}
_TARGET_KEYS = {"section", "index"}
_INVENTORY_ROW_KEYS = {"item_id", "status", "draft_targets", "finding_indices"}
_DRAFT_ROW_KEYS = {"unit_id", "status", "source_ids", "inventory_ids", "finding_indices"}
_WINDOW_ROW_KEYS = {"window_id", "status", "source_ids", "finding_indices"}
_WINDOW_STATES = {"confirmed_no_material", "material_in_inventory", "new_material", "uncertain"}
_MAX_ROWS = 512
_CONTEXT_CUES = (
    "да", "нет", "но", "точнее", "вернее", "поправка", "отмена", "подожди",
    "погоди", "то есть", "это", "этот", "эта", "они", "он", "она", "там",
    "тогда", "если", "поэтому", "в таком случае", "yes", "no", "but",
    "however", "actually", "if", "so",
)
_CONTEXTUAL_INVENTORY_KINDS = {"answer", "correction", "question"}
_CONTEXTUAL_MODALITIES = {"conditional", "cancelled", "rejected", "answered"}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _source(source_text: str) -> tuple[dict, list[dict], dict[str, int]]:
    try:
        source = json.loads(source_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("reconcile source is not JSON") from exc
    if not isinstance(source, dict) or not isinstance(source.get("utterances"), list):
        raise ValueError("reconcile source has no utterance list")
    utterances = source["utterances"]
    if not utterances or any(not isinstance(row, dict) or not isinstance(row.get("id"), str)
                             or not row["id"] for row in utterances):
        raise ValueError("reconcile source has invalid utterances")
    positions = {row["id"]: number for number, row in enumerate(utterances)}
    if len(positions) != len(utterances):
        raise ValueError("reconcile source contains duplicate utterance IDs")
    # The production entry supplies load_source's canonical serialization. A
    # transformed excerpt must not masquerade as the complete source here.
    if _canonical(source) != source_text:
        raise ValueError("reconcile source is not canonical")
    return source, utterances, positions


def _inventory_items(inventory: dict, positions: dict[str, int],
                     *, source_fingerprint: str | None = None,
                     source_sha256: str | None = None) -> dict[str, dict]:
    if not isinstance(inventory, dict) or inventory.get("schema_version") != "gemini_source_inventory_merged_v1":
        raise ValueError("reconcile requires merged source inventory v1")
    if source_fingerprint is not None and inventory.get("source_fingerprint") != source_fingerprint:
        raise ValueError("inventory source fingerprint differs from transcript")
    if source_sha256 is not None and inventory.get("source_sha256") != source_sha256:
        raise ValueError("inventory raw source SHA differs from transcript")
    items = inventory.get("items")
    if not isinstance(items, list) or len(items) > _MAX_ROWS:
        raise ValueError("inventory items are invalid")
    if inventory.get("primary_utterance_count") != len(positions):
        raise ValueError("inventory primary coverage differs from transcript")
    if (type(inventory.get("segment_count")) is not int
            or inventory["segment_count"] < 1 or inventory["segment_count"] > 3):
        raise ValueError("inventory segment count is invalid")
    by_id = {}
    for number, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("item_id"), str) or not item["item_id"]:
            raise ValueError(f"inventory item {number} has no ID")
        item_id = item["item_id"]
        if item_id in by_id:
            raise ValueError("inventory contains duplicate item IDs")
        citations = item.get("source_ids")
        if (not isinstance(citations, list) or not citations
                or any(not isinstance(source_id, str) or source_id not in positions
                       for source_id in citations)
                or len(set(citations)) != len(citations)):
            raise ValueError(f"inventory item {item_id} has invalid source IDs")
        by_id[item_id] = item
    return by_id


def _flagged_windows(inventory: dict, positions: dict[str, int]) -> list[tuple[dict, range]]:
    coverage = inventory.get("coverage")
    if not isinstance(coverage, list):
        raise ValueError("inventory coverage is invalid")
    flagged = []
    seen = set()
    for number, row in enumerate(coverage):
        if not isinstance(row, dict) or not isinstance(row.get("window_id"), str):
            raise ValueError(f"inventory coverage window {number} is invalid")
        window_id = row["window_id"]
        if window_id in seen:
            raise ValueError("inventory coverage has duplicate window IDs")
        seen.add(window_id)
        if (not isinstance(row.get("assessment"), str)
                or row["assessment"] not in {"material_items", "no_material_items", "uncertain"}):
            raise ValueError(f"inventory coverage window {number} has invalid assessment")
        start_id, end_id = row.get("start_id"), row.get("end_id")
        if (not isinstance(start_id, str) or not isinstance(end_id, str)
                or start_id not in positions or end_id not in positions
                or positions[start_id] > positions[end_id]):
            raise ValueError(f"inventory coverage window {number} has invalid bounds")
        if row["assessment"] in {"uncertain", "no_material_items"}:
            flagged.append((row, range(positions[start_id], positions[end_id] + 1)))
    return flagged


def _citations_in(value: object, positions: dict[str, int], path: str) -> set[str]:
    citations: set[str] = set()

    def walk(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                where = f"{path}.{key}"
                if key in {"start_id", "end_id"}:
                    if nested is not None:
                        if not isinstance(nested, str) or nested not in positions:
                            raise ValueError(f"{where}: unknown draft source ID")
                        citations.add(nested)
                elif key == "source_ids":
                    if not isinstance(nested, list) or any(
                        not isinstance(source_id, str) or source_id not in positions
                        for source_id in nested
                    ):
                        raise ValueError(f"{where}: invalid draft source IDs")
                    citations.update(nested)
                elif key == "field_sources":
                    if not isinstance(nested, dict):
                        raise ValueError(f"{where}: invalid field sources")
                    for field, field_ids in nested.items():
                        if not isinstance(field_ids, list) or any(
                            not isinstance(source_id, str) or source_id not in positions
                            for source_id in field_ids
                        ):
                            raise ValueError(f"{where}.{field}: invalid draft source IDs")
                        citations.update(field_ids)
                else:
                    walk(nested, where)
        elif isinstance(value, list):
            for number, nested in enumerate(value):
                walk(nested, f"{path}[{number}]")

    walk(value, path)
    return citations


def _draft_citations(draft: dict, positions: dict[str, int]) -> set[str]:
    return _citations_in(draft, positions, "DRAFT_DOCUMENT")


def draft_units(draft: dict) -> list[dict]:
    """Enumerate a stable checklist of content units without paraphrasing it."""
    prepared = _working_draft(draft)
    units = [{"unit_id": "meeting:0", "section": "meeting", "index": 0,
              "focus": "topic_project_date"}]
    for section in _ARRAY_SECTIONS:
        for number, item in enumerate(prepared[section]):
            if section == "tasks":
                units.extend([
                    {"unit_id": f"tasks:{number}:action", "section": section,
                     "index": number, "focus": "action_object_scope_alternatives"},
                    {"unit_id": f"tasks:{number}:relations", "section": section,
                     "index": number, "focus": "actor_recipient_status_condition_due"},
                ])
            elif section == "chapters" and isinstance(item, dict):
                units.append({"unit_id": f"chapters:{number}:summary", "section": section,
                              "index": number, "focus": "visible_summary"})
                details = item.get("details")
                if isinstance(details, list):
                    for detail_number, _ in enumerate(details):
                        units.append({"unit_id": f"chapters:{number}:detail:{detail_number}",
                                      "section": section, "index": number,
                                      "focus": "disclosed_detail"})
                else:
                    units.append({"unit_id": f"chapters:{number}:details_malformed",
                                  "section": section, "index": number,
                                  "focus": "repair_disclosed_details"})
            else:
                units.append({"unit_id": f"{section}:{number}", "section": section,
                              "index": number, "focus": "content"})
    return units


def _unit_value(draft: dict, unit: dict) -> object:
    if unit["section"] == "meeting":
        return draft["meeting"]
    item = draft[unit["section"]][unit["index"]]
    unit_id = unit["unit_id"]
    if not isinstance(item, dict):
        return item
    if unit_id.endswith(":action") and unit["section"] == "tasks":
        return {key: item.get(key) for key in ("title", "description", "source_ids", "field_sources")}
    if unit_id.endswith(":relations") and unit["section"] == "tasks":
        return {key: item.get(key) for key in (
            "discussion_status", "assignee", "due", "recipient", "priority",
            "source_ids", "field_sources",
        )}
    if unit["section"] == "chapters" and isinstance(item, dict):
        if unit_id.endswith(":details_malformed"):
            return item.get("details")
        if ":detail:" in unit_id:
            detail_number = int(unit_id.rsplit(":", 1)[1])
            return item["details"][detail_number]
        return {key: item.get(key) for key in ("topic", "summary", "source_ids", "start_id", "end_id")}
    return item


def _focused_verify_scope(prepared: dict, inventory: dict, prior_findings: list[dict],
                          positions: dict[str, int]) -> tuple[list[dict], list[dict], set[str]]:
    """Select source-linked inventory and draft units for a short verification."""
    focus_ids = set()
    affected = set()
    for number, finding in enumerate(prior_findings):
        source_ids = finding.get("source_ids")
        if not isinstance(source_ids, list) or any(
            not isinstance(source_id, str) or source_id not in positions for source_id in source_ids
        ):
            raise ValueError(f"prior finding {number} has invalid source IDs")
        focus_ids.update(source_ids)
        for target in finding.get("affected", []):
            if isinstance(target, dict):
                affected.add((target.get("section"), target.get("index")))
    if not focus_ids:
        raise ValueError("verify needs prior source-referenced findings")
    seed_ids = frozenset(focus_ids)
    focused_items = [item for item in inventory["items"]
                     if set(item["source_ids"]).intersection(seed_ids)]
    focused_units = []
    for unit in draft_units(prepared):
        unit_citations = _citations_in(_unit_value(prepared, unit), positions,
                                      f"DRAFT_UNITS.{unit['unit_id']}")
        if (unit_citations.intersection(seed_ids)
                or (unit["section"], unit["index"]) in affected):
            focused_units.append(unit)
            focus_ids.update(unit_citations)
    for item in focused_items:
        focus_ids.update(item["source_ids"])
    return focused_items, focused_units, focus_ids


def _context_sensitive(utterance: dict) -> bool:
    """Cheap generic cue for adjacent context; never used to omit a citation."""
    text = " ".join(str(utterance.get("text", "")).split()).casefold()
    if len(text) <= 90:
        return True
    return any(text == cue or text.startswith(cue + " ") or text.startswith(cue + ",")
               for cue in _CONTEXT_CUES)


def _selected_excerpt_positions(citations: set[str], items: list[dict],
                                utterances: list[dict], positions: dict[str, int]) -> set[int]:
    selected = {positions[citation] for citation in citations}
    contextual_ids = {
        source_id for item in items
        if (item.get("kind") in _CONTEXTUAL_INVENTORY_KINDS
            or item.get("modality") in _CONTEXTUAL_MODALITIES)
        for source_id in item["source_ids"]
    }
    for citation in citations:
        center = positions[citation]
        if citation in contextual_ids or _context_sensitive(utterances[center]):
            selected.update(range(max(0, center - 1), min(len(utterances), center + 2)))
    return selected


def build_reconcile_input(source_text: str, draft: dict, inventory: dict, *,
                          mode: str = "reconcile",
                          prior_findings: list[dict] | tuple[dict, ...] = ()) -> str:
    """Return a deterministic, bounded-evidence reconciliation message.

    Every inventory citation and draft citation contributes its exact source
    utterance plus one immediate neighbor on each side. There is no text
    truncation, silent omission of a cited utterance, or model-made source ID.
    If the union spans the whole source, the actual full size is exposed and
    remains subject to the caller's pre-dispatch budget guard.
    """
    if mode not in {"reconcile", "verify"}:
        raise ValueError("unknown reconciliation mode")
    if not isinstance(prior_findings, (list, tuple)) or any(
        not isinstance(item, dict) for item in prior_findings
    ):
        raise ValueError("prior findings must be an array of objects")
    source, utterances, positions = _source(source_text)
    fingerprint = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    inventory_by_id = _inventory_items(inventory, positions, source_fingerprint=fingerprint)
    prepared = _working_draft(draft)
    # Validate every draft citation even when verify sends only a focused
    # subset. Otherwise a malformed unseen citation could hide behind focus.
    all_draft_citations = _draft_citations(prepared, positions)
    if mode == "verify":
        focused_items, units, citations = _focused_verify_scope(
            prepared, inventory, list(prior_findings), positions
        )
        focused_inventory = {
            "schema_version": inventory["schema_version"],
            "source_sha256": inventory["source_sha256"],
            "source_fingerprint": inventory["source_fingerprint"],
            "scope": "focused_verify",
            "items": focused_items,
        }
        flagged = []
    else:
        units = draft_units(prepared)
        citations = set(all_draft_citations)
        for item in inventory_by_id.values():
            citations.update(item["source_ids"])
        focused_inventory = inventory
        flagged = _flagged_windows(inventory, positions)
    selected_positions = _selected_excerpt_positions(
        citations, focused_inventory["items"], utterances, positions
    )
    contextual_neighbor_count = len(selected_positions) - len(citations)
    before_recheck = len(selected_positions)
    for _, window_positions in flagged:
        selected_positions.update(window_positions)
    recheck_window_additional_count = len(selected_positions) - before_recheck
    excerpts = [utterances[number] for number in sorted(selected_positions)]
    metadata = {key: value for key, value in source.items() if key != "utterances"}
    task = (
        "Сопоставь независимый перечень фактов с черновиком в обоих направлениях. "
        "Верни все существенные расхождения и минимальные безопасные patches."
        if mode == "reconcile" else
        "Повторно проверь адресно исправленные и спорные элементы по исходным "
        "репликам; сохрани оставшиеся расхождения и безопасные patches."
    )
    payload = {
        "SOURCE_METADATA": metadata,
        "SOURCE_EXCERPTS": excerpts,
        "EXCERPT_SCOPE": {
            "selection": "all_citations_plus_one_neighbor_for_short_discourse_or_dependent_inventory_items",
            "cited_utterance_count": len(citations),
            "contextual_neighbor_count": contextual_neighbor_count,
            "recheck_window_additional_count": recheck_window_additional_count,
            "included_utterance_count": len(excerpts),
            "total_utterance_count": len(utterances),
        },
        "INDEPENDENT_SOURCE_INVENTORY": focused_inventory,
        "SOURCE_WINDOWS_TO_RECHECK": [
            {"window_id": row["window_id"], "start_id": row["start_id"],
             "end_id": row["end_id"], "previous_assessment": row["assessment"]}
            for row, _ in flagged
        ],
        "DRAFT_DOCUMENT": prepared,
        "DRAFT_UNITS": units,
        "PRIOR_FINDINGS": list(prior_findings),
        "MODE": mode,
        "TASK": task,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _legacy_report(report: dict, source_index: dict) -> dict:
    """Reuse v1 patch safety with synthetic *mechanical* window placeholders.

    These rows are never presented as model coverage or published evidence;
    the independent source-inventory and reconciliation rows carry coverage.
    """
    windows = source_windows(source_index)
    return {
        "schema_version": AUDIT_SCHEMA_ID,
        "coverage": [
            {"window_id": row["window_id"], "start_id": row["start_id"],
             "end_id": row["end_id"], "salient": "Механическая проверка патчей",
             "draft_coverage": "covered", "finding_indices": []}
            for row in windows
        ],
        "findings": copy.deepcopy(report["findings"]),
        "patches": copy.deepcopy(report["patches"]),
    }


def _strict_report_shape(report: dict) -> None:
    if not isinstance(report, dict) or set(report) != {
        "schema_version", "source_window_assessments", "inventory_assessments",
        "draft_assessments", "findings", "patches"
    } or report.get("schema_version") != RECONCILE_SCHEMA_ID:
        raise ValueError("reconciliation report has wrong shape or version")
    for field in ("source_window_assessments", "inventory_assessments", "draft_assessments",
                  "findings", "patches"):
        if not isinstance(report[field], list):
            raise ValueError(f"reconciliation {field} must be an array")
    if (len(report["source_window_assessments"]) > 36
            or len(report["inventory_assessments"]) > _MAX_ROWS
            or len(report["draft_assessments"]) > _MAX_ROWS):
        raise ValueError("reconciliation has too many assessment rows")
    for number, row in enumerate(report["source_window_assessments"]):
        if not isinstance(row, dict) or set(row) != _WINDOW_ROW_KEYS:
            raise ValueError(f"source_window_assessments[{number}] has wrong fields")
        if (not isinstance(row["window_id"], str)
                or not isinstance(row["status"], str)
                or row["status"] not in _WINDOW_STATES
                or not isinstance(row["source_ids"], list)
                or not isinstance(row["finding_indices"], list)):
            raise ValueError(f"source_window_assessments[{number}] has invalid fields")
    for number, row in enumerate(report["inventory_assessments"]):
        if not isinstance(row, dict) or set(row) != _INVENTORY_ROW_KEYS:
            raise ValueError(f"inventory_assessments[{number}] has wrong fields")
        if (not isinstance(row["item_id"], str)
                or not isinstance(row["status"], str)
                or row["status"] not in _INVENTORY_STATES):
            raise ValueError(f"inventory_assessments[{number}] has invalid identity or status")
        if not isinstance(row["draft_targets"], list) or not isinstance(row["finding_indices"], list):
            raise ValueError(f"inventory_assessments[{number}] has invalid links")
    for number, row in enumerate(report["draft_assessments"]):
        if not isinstance(row, dict) or set(row) != _DRAFT_ROW_KEYS:
            raise ValueError(f"draft_assessments[{number}] has wrong fields")
        if (not isinstance(row["unit_id"], str)
                or not isinstance(row["status"], str)
                or row["status"] not in _DRAFT_STATES):
            raise ValueError(f"draft_assessments[{number}] has invalid identity or status")
        if any(not isinstance(row[field], list) for field in ("source_ids", "inventory_ids", "finding_indices")):
            raise ValueError(f"draft_assessments[{number}] has invalid links")


def validate_reconciliation_report(report: dict, draft: dict, inventory: dict,
                                   source_index: dict, *, mode: str = "reconcile") -> dict:
    """Reject unsafe patches and wrong-source inventory; retain raw assessments."""
    if mode not in {"reconcile", "verify"}:
        raise ValueError("unknown reconciliation mode")
    if not isinstance(source_index, dict) or not isinstance(source_index.get("by_id"), dict):
        raise ValueError("reconciliation has no source index")
    _inventory_items(inventory, {source_id: number for number, source_id in enumerate(source_index["by_id"])},
                     source_sha256=source_index.get("source_sha256"))
    _strict_report_shape(report)
    validate_audit(_legacy_report(report, source_index), draft, source_index,
                   mode="verify" if mode == "verify" else "audit")
    return report


def reconciliation_warnings(report: dict, draft: dict, inventory: dict,
                            source_index: dict, *, mode: str = "reconcile",
                            prior_findings: list[dict] | tuple[dict, ...] = ()) -> list[dict]:
    """Nonfatal bookkeeping diagnostics; a warning cannot prove a fact false."""
    validate_reconciliation_report(report, draft, inventory, source_index, mode=mode)
    prepared = _working_draft(draft)
    expected_inventory = {item["item_id"]: item for item in inventory["items"]}
    expected_units = {unit["unit_id"]: unit for unit in draft_units(prepared)}
    known_source = set(source_index["by_id"])
    positions = {source_id: number for number, source_id in enumerate(source_index["by_id"])}
    source_rows = list(source_index["by_id"].values())
    if mode == "verify" and prior_findings:
        focused_items, _, cited = _focused_verify_scope(
            prepared, inventory, list(prior_findings), positions
        )
        visible = {source_rows[number]["id"] for number in _selected_excerpt_positions(
            cited, focused_items, source_rows, positions
        )}
    elif mode == "reconcile":
        cited = _draft_citations(prepared, positions)
        for item in inventory["items"]:
            cited.update(item["source_ids"])
        selected = _selected_excerpt_positions(
            cited, inventory["items"], source_rows, positions
        )
        flagged = _flagged_windows(inventory, positions)
        for _, window_positions in flagged:
            selected.update(window_positions)
        visible = {source_rows[number]["id"] for number in selected}
    else:
        visible = known_source  # Verify without prior input cannot reconstruct its focus.
    findings_count = len(report["findings"])
    warnings: list[dict] = []

    def warn(code: str, **location: object) -> None:
        warnings.append({"code": code, **location})

    seen_inventory: set[str] = set()
    linked_findings: set[int] = set()
    inventory_rows: dict[str, dict] = {}
    if mode == "reconcile":
        flagged = _flagged_windows(inventory, positions)
        flagged_by_id = {row["window_id"]: (row, set(source_rows[number]["id"]
                                                      for number in window_positions))
                         for row, window_positions in flagged}
        seen_windows = set()
        for number, row in enumerate(report["source_window_assessments"]):
            window_id = row["window_id"]
            if window_id not in flagged_by_id:
                warn("unknown_source_window_assessment", row=number, window_id=window_id)
            if window_id in seen_windows:
                warn("duplicate_source_window_assessment", row=number, window_id=window_id)
            seen_windows.add(window_id)
            actual_ids = row["source_ids"]
            if (not actual_ids or any(not isinstance(source_id, str) or source_id not in known_source
                                      for source_id in actual_ids)):
                warn("invalid_source_window_assessment_sources", row=number)
            elif (window_id in flagged_by_id
                  and not set(actual_ids).intersection(flagged_by_id[window_id][1])):
                warn("source_window_assessment_outside_window", row=number)
            links = row["finding_indices"]
            if any(type(pointer) is not int or pointer < 0 or pointer >= findings_count
                   for pointer in links):
                warn("invalid_source_window_finding_link", row=number)
            linked_findings.update(pointer for pointer in links
                                   if type(pointer) is int and 0 <= pointer < findings_count)
            if row["status"] in {"new_material", "uncertain"} and not links:
                warn("unexplained_source_window_defect", row=number, window_id=window_id)
            if window_id in flagged_by_id:
                original_assessment = flagged_by_id[window_id][0]["assessment"]
                if original_assessment == "no_material_items" and row["status"] != "confirmed_no_material":
                    warn("no_material_window_unverified", window_id=window_id)
        for window_id, (window, _) in flagged_by_id.items():
            if window["assessment"] == "uncertain":
                warn("source_inventory_uncertain_window", window_id=window_id)
            if window_id not in seen_windows:
                warn("source_window_unassessed", window_id=window_id)
                if window["assessment"] == "no_material_items":
                    warn("no_material_window_unverified", window_id=window_id)
        if (len(source_index["by_id"]) > 1 and inventory["coverage"]
                and all(row["assessment"] == "no_material_items"
                        for row in inventory["coverage"])):
            warn("all_inventory_empty")
    for number, row in enumerate(report["inventory_assessments"]):
        item_id = row["item_id"]
        if item_id not in expected_inventory:
            warn("unknown_inventory_item", row=number, item_id=item_id)
        if item_id in seen_inventory:
            warn("duplicate_inventory_assessment", row=number, item_id=item_id)
        seen_inventory.add(item_id)
        inventory_rows[item_id] = row
        links = row["finding_indices"]
        linked_findings.update(pointer for pointer in links
                               if type(pointer) is int and 0 <= pointer < findings_count)
        if any(type(pointer) is not int or pointer < 0 or pointer >= findings_count for pointer in links):
            warn("invalid_inventory_finding_link", row=number)
        if row["status"] in {"partial", "missing", "contradicted", "uncertain"} and not links:
            warn("unexplained_inventory_defect", row=number, item_id=item_id)
        targets = row["draft_targets"]
        if row["status"] == "represented" and not targets:
            warn("represented_without_draft_target", row=number, item_id=item_id)
        if row["status"] == "missing" and targets:
            warn("missing_with_draft_target", row=number, item_id=item_id)
        for target_number, target in enumerate(targets):
            section = target.get("section") if isinstance(target, dict) else None
            index = target.get("index") if isinstance(target, dict) else None
            if (not isinstance(target, dict) or set(target) != _TARGET_KEYS
                    or not isinstance(section, str) or section not in _SECTIONS
                    or type(index) is not int or index < 0
                    or (section == "meeting" and index != 0)
                    or (section != "meeting" and index >= len(prepared[section]))):
                warn("invalid_inventory_draft_target", row=number, target=target_number)
        item = expected_inventory.get(item_id)
        if (item and item.get("kind") == "action"
                and item.get("modality") in {"proposed", "committed", "conditional", "open"}
                and row["status"] == "represented"
                and not any(isinstance(target, dict) and target.get("section") == "tasks"
                            for target in targets)):
            warn("active_action_without_task_target", row=number, item_id=item_id)
    if mode == "reconcile":
        for item_id in expected_inventory.keys() - seen_inventory:
            warn("inventory_item_unassessed", item_id=item_id)

    seen_units: set[str] = set()
    draft_rows: dict[str, dict] = {}
    for number, row in enumerate(report["draft_assessments"]):
        unit_id = row["unit_id"]
        if unit_id not in expected_units:
            warn("unknown_draft_unit", row=number, unit_id=unit_id)
        if unit_id in seen_units:
            warn("duplicate_draft_assessment", row=number, unit_id=unit_id)
        seen_units.add(unit_id)
        draft_rows[unit_id] = row
        if any(not isinstance(source_id, str) or source_id not in known_source
               for source_id in row["source_ids"]):
            warn("invalid_draft_assessment_source", row=number)
        elif any(source_id not in visible for source_id in row["source_ids"]):
            warn("draft_assessment_source_not_supplied", row=number)
        if any(not isinstance(item_id, str) or item_id not in expected_inventory
               for item_id in row["inventory_ids"]):
            warn("invalid_draft_assessment_inventory_link", row=number)
        links = row["finding_indices"]
        linked_findings.update(pointer for pointer in links
                               if type(pointer) is int and 0 <= pointer < findings_count)
        if any(type(pointer) is not int or pointer < 0 or pointer >= findings_count for pointer in links):
            warn("invalid_draft_finding_link", row=number)
        if row["status"] in {"partial", "unsupported", "uncertain"} and not links:
            warn("unexplained_draft_defect", row=number, unit_id=unit_id)
        if row["status"] in {"supported", "supported_uninventoried"} and not row["source_ids"] and unit_id != "meeting:0":
            warn("supported_without_source", row=number, unit_id=unit_id)
        if row["status"] == "supported_uninventoried" and row["inventory_ids"]:
            warn("uninventoried_with_inventory_link", row=number, unit_id=unit_id)
    if mode == "reconcile":
        for unit_id in expected_units.keys() - seen_units:
            warn("draft_unit_unassessed", unit_id=unit_id)
        if not report["inventory_assessments"] and expected_inventory:
            warn("all_inventory_unassessed")
        if not report["draft_assessments"] and expected_units:
            warn("all_draft_unassessed")
    for item_id, row in inventory_rows.items():
        if row["status"] != "represented":
            continue
        for target in row["draft_targets"]:
            if not isinstance(target, dict):
                continue
            for unit in expected_units.values():
                if (unit["section"], unit["index"]) != (target.get("section"), target.get("index")):
                    continue
                assessed = draft_rows.get(unit["unit_id"])
                if assessed and assessed["status"] in {"partial", "unsupported", "uncertain"}:
                    warn("represented_target_not_supported", item_id=item_id,
                         unit_id=unit["unit_id"])
    for number in set(range(findings_count)) - linked_findings:
        warn("finding_not_linked_to_assessment", finding_index=number)
    for number, finding in enumerate(report["findings"]):
        if any(source_id not in visible for source_id in finding["source_ids"]):
            warn("finding_source_not_supplied", finding_index=number)
    return warnings


def apply_reconciliation_report(draft: dict, report: dict, inventory: dict,
                                source_index: dict, *, mode: str = "reconcile") -> tuple[dict, list[dict]]:
    """Apply only mechanically valid source-referenced whole-item patches."""
    validate_reconciliation_report(report, draft, inventory, source_index, mode=mode)
    return apply_audit(draft, _legacy_report(report, source_index), source_index,
                       mode="verify" if mode == "verify" else "audit")
