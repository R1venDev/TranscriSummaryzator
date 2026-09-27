"""Versioned evidence-bearing source-inventory reconciliation.

The v1 prompt/schema remain available for immutable historical jobs. New
jobs use these explicit entry points so a replay cannot silently switch
contracts when the default quality policy changes.
"""

from __future__ import annotations

import copy
import hashlib
import json

from ..luna_v1.audit import _working_draft
from .reconcile_contract import (
    RECONCILE_PROMPT_PATH_V2,
    RECONCILE_SCHEMA_ID_V2,
    RECONCILE_SCHEMA_V2,
    _citations_in,
    _inventory_items,
    _source,
    _unit_value,
    apply_reconciliation_report,
    build_reconcile_input,
    draft_units,
    reconciliation_warnings,
    validate_reconciliation_report,
)


def _require_v2_inventory(inventory: dict) -> None:
    if (not isinstance(inventory, dict)
            or inventory.get("schema_version") != "gemini_source_inventory_merged_v2"):
        raise ValueError("reconcile v2 requires merged source inventory v2")


def _require_v2_report(report: dict) -> None:
    if (not isinstance(report, dict)
            or report.get("schema_version") != RECONCILE_SCHEMA_ID_V2):
        raise ValueError("reconcile v2 requires evidence-bearing report v2")


def build_reconcile_input_v2(source_text: str, draft: dict, inventory: dict, *,
                             mode: str = "reconcile", prior_findings=()) -> str:
    _require_v2_inventory(inventory)
    return build_reconcile_input(source_text, draft, inventory, mode=mode,
                                 prior_findings=prior_findings)


def validate_reconciliation_report_v2(report: dict, draft: dict, inventory: dict,
                                      source_index: dict, *, mode: str = "reconcile") -> dict:
    _require_v2_inventory(inventory)
    _require_v2_report(report)
    return validate_reconciliation_report(report, draft, inventory, source_index,
                                          mode=mode)


def reconciliation_warnings_v2(report: dict, draft: dict, inventory: dict,
                               source_index: dict, *, mode: str = "reconcile",
                               prior_findings=()) -> list[dict]:
    validate_reconciliation_report_v2(report, draft, inventory, source_index,
                                      mode=mode)
    return reconciliation_warnings(report, draft, inventory, source_index,
                                   mode=mode, prior_findings=prior_findings)


def apply_reconciliation_report_v2(draft: dict, report: dict, inventory: dict,
                                   source_index: dict, *, mode: str = "reconcile") -> tuple[dict, list[dict]]:
    validate_reconciliation_report_v2(report, draft, inventory, source_index,
                                      mode=mode)
    return apply_reconciliation_report(draft, report, inventory, source_index,
                                       mode=mode)


def _unit_citations(prepared: dict, unit: dict, positions: dict[str, int]) -> set[str]:
    return _citations_in(_unit_value(prepared, unit), positions,
                         f"DRAFT_UNITS.{unit['unit_id']}")


def partition_reconcile_targets(source_text: str, source_index: dict, draft: dict,
                                inventory: dict, *,
                                cut_after_window_id: str | None = None) -> list[dict]:
    """Split a v2 inventory by primary windows without changing source identity.

    Windows partition the complete transcript exactly once. An item or risk
    warning with citations on both sides appears in both targets. Draft units
    follow their citations, while citation-free units appear in both targets.
    Every target carries the full draft so patch indices remain original.
    """
    _require_v2_inventory(inventory)
    _, utterances, positions = _source(source_text)
    source_ids = [row["id"] for row in utterances]
    if (not isinstance(source_index, dict)
            or list(source_index.get("by_id", {})) != source_ids
            or source_index.get("source_sha256") != inventory.get("source_sha256")):
        raise ValueError("reconcile partition source index differs from inventory")
    fingerprint = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    _inventory_items(inventory, positions, source_fingerprint=fingerprint,
                     source_sha256=source_index["source_sha256"])
    prepared = _working_draft(draft)
    units = draft_units(prepared)
    unit_sources = {unit["unit_id"]: _unit_citations(prepared, unit, positions)
                    for unit in units}
    coverage = inventory.get("coverage")
    if not isinstance(coverage, list) or len(coverage) < 2:
        raise ValueError("reconcile partition needs at least two source windows")
    next_position = 0
    seen_windows = set()
    for row in coverage:
        if (not isinstance(row, dict) or not isinstance(row.get("window_id"), str)
                or row["window_id"] in seen_windows
                or row.get("start_id") not in positions
                or row.get("end_id") not in positions):
            raise ValueError("reconcile partition has invalid source window")
        seen_windows.add(row["window_id"])
        start, end = positions[row["start_id"]], positions[row["end_id"]]
        if start != next_position or end < start:
            raise ValueError("reconcile partition windows do not cover source in order")
        next_position = end + 1
    if next_position != len(utterances):
        raise ValueError("reconcile partition windows omit primary source")

    def targets_at(cut: int) -> list[dict]:
        boundaries = (0, cut, len(coverage))
        targets = []
        item_counts = {item["item_id"]: 0 for item in inventory["items"]}
        warning_counts = [0] * len(inventory["risk_warnings"])
        unit_counts = {unit["unit_id"]: 0 for unit in units}
        for part in (1, 2):
            windows = coverage[boundaries[part - 1]:boundaries[part]]
            first = positions[windows[0]["start_id"]]
            last = positions[windows[-1]["end_id"]]
            primary_ids = source_ids[first:last + 1]
            primary_set = set(primary_ids)
            items = []
            for item in inventory["items"]:
                references = set(item["source_ids"]) | set(item.get("correction_of", []))
                if references.intersection(primary_set):
                    items.append(copy.deepcopy(item))
                    item_counts[item["item_id"]] += 1
            warnings = []
            for number, warning in enumerate(inventory["risk_warnings"]):
                if set(warning["source_ids"]).intersection(primary_set):
                    warnings.append(copy.deepcopy(warning))
                    warning_counts[number] += 1
            selected_units = []
            for unit in units:
                citations = unit_sources[unit["unit_id"]]
                if not citations or citations.intersection(primary_set):
                    selected_units.append(unit)
                    unit_counts[unit["unit_id"]] += 1

            cited = set()
            for item in items:
                cited.update(item["source_ids"])
                cited.update(item.get("correction_of", []))
            for warning in warnings:
                cited.update(warning["source_ids"])
            for unit in selected_units:
                cited.update(unit_sources[unit["unit_id"]])
            excerpt_positions = set(range(first, last + 1))
            # Four turns around the cut preserve inventory's context overlap.
            excerpt_positions.update(range(max(0, first - 4), min(len(source_ids), last + 5)))
            for source_id in cited:
                center = positions[source_id]
                excerpt_positions.update(range(max(0, center - 1),
                                               min(len(source_ids), center + 2)))
            excerpt_ids = [source_ids[number] for number in sorted(excerpt_positions)]
            scoped_inventory = {key: copy.deepcopy(value) for key, value in inventory.items()
                                if key not in {"coverage", "items", "risk_warnings"}}
            scoped_inventory["coverage"] = copy.deepcopy(windows)
            scoped_inventory["items"] = items
            scoped_inventory["risk_warnings"] = warnings
            targets.append({
                "draft": copy.deepcopy(prepared),
                "inventory": scoped_inventory,
                "scope": {
                    "part": part,
                    "total_parts": 2,
                    "cut_after_window_id": coverage[cut - 1]["window_id"],
                    "primary_window_ids": [row["window_id"] for row in windows],
                    "primary_source_ids": primary_ids,
                    "excerpt_source_ids": excerpt_ids,
                    "draft_unit_ids": [unit["unit_id"] for unit in selected_units],
                    "all_draft_unit_ids": [unit["unit_id"] for unit in units],
                },
            })
        if (any(count == 0 for count in item_counts.values())
                or any(count == 0 for count in warning_counts)
                or any(count == 0 for count in unit_counts.values())):
            raise ValueError("reconcile partition silently omitted inventory or draft scope")
        if (targets[0]["scope"]["primary_source_ids"]
                + targets[1]["scope"]["primary_source_ids"] != source_ids):
            raise ValueError("reconcile partition primary source differs from transcript")
        return targets

    # For the 12-window real transcript, inspect cuts after windows 4..8.
    # Tiny fixtures keep a stable midpoint split. This uses only serialized
    # request size, never model judgment or paid routing, and pins the chosen
    # cut in both target scopes.
    allowed_cuts = (range(len(coverage) // 3, 2 * len(coverage) // 3 + 1)
                    if len(coverage) >= 8 else [len(coverage) // 2])
    if cut_after_window_id is None:
        cuts = allowed_cuts
    else:
        matches = [number for number, row in enumerate(coverage, 1)
                   if row["window_id"] == cut_after_window_id]
        if len(matches) != 1 or matches[0] not in allowed_cuts:
            raise ValueError("reconcile partition pinned cut is invalid")
        cuts = matches
    prompt = RECONCILE_PROMPT_PATH_V2.read_text(encoding="utf-8")

    def route_input_score(content: str) -> int:
        body = {
            "messages": [{"role": "system", "content": prompt},
                         {"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": RECONCILE_SCHEMA_ID_V2, "strict": True,
                "schema": RECONCILE_SCHEMA_V2}},
            "max_completion_tokens": 14_000,
            "reasoning": {"effort": "medium"}, "plugins": [],
        }
        serialized = json.dumps(body, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"))
        byte_count = len(serialized.encode("utf-8"))
        return max(len(serialized), (byte_count * 3 + 3) // 4)

    choices = []
    for cut in cuts:
        targets = targets_at(cut)
        scores = [route_input_score(build_reconcile_target_input_v2(source_text, target))
                  for target in targets]
        choices.append(((max(scores), abs(scores[0] - scores[1]),
                         abs(2 * cut - len(coverage)), cut), targets))
    return min(choices, key=lambda choice: choice[0])[1]


def _target_scope(target: dict, source_ids: list[str]) -> tuple[dict, dict, dict]:
    if not isinstance(target, dict) or not isinstance(target.get("scope"), dict):
        raise ValueError("reconcile target has no scope")
    scope = target["scope"]
    inventory, draft = target.get("inventory"), target.get("draft")
    _require_v2_inventory(inventory)
    prepared = _working_draft(draft)
    if (scope.get("part") not in (1, 2) or scope.get("total_parts") != 2
            or not isinstance(scope.get("primary_source_ids"), list)
            or not isinstance(scope.get("excerpt_source_ids"), list)
            or not isinstance(scope.get("draft_unit_ids"), list)
            or not isinstance(scope.get("all_draft_unit_ids"), list)):
        raise ValueError("reconcile target has invalid scope")
    primary = scope["primary_source_ids"]
    excerpts = scope["excerpt_source_ids"]
    if (not primary or len(primary) != len(set(primary))
            or len(excerpts) != len(set(excerpts))
            or any(source_id not in source_ids for source_id in excerpts)
            or not set(primary).issubset(excerpts)
            or excerpts != [source_id for source_id in source_ids if source_id in set(excerpts)]):
        raise ValueError("reconcile target excerpt scope is invalid")
    windows = inventory.get("coverage")
    if (not isinstance(windows, list) or not windows
            or any(not isinstance(row, dict) for row in windows)
            or scope.get("primary_window_ids") != [row.get("window_id") for row in windows]
            or primary[0] != windows[0].get("start_id")
            or primary[-1] != windows[-1].get("end_id")):
        raise ValueError("reconcile target window scope is invalid")
    positions = {source_id: number for number, source_id in enumerate(source_ids)}
    first, last = positions[primary[0]], positions[primary[-1]]
    if (primary != source_ids[first:last + 1]
            or not set(source_ids[max(0, first - 4):min(len(source_ids), last + 5)])
            .issubset(excerpts)):
        raise ValueError("reconcile target primary or context scope is invalid")
    next_position = first
    for row in windows:
        start_id, end_id = row.get("start_id"), row.get("end_id")
        if (start_id not in positions or end_id not in positions
                or positions[start_id] != next_position
                or positions[end_id] < next_position):
            raise ValueError("reconcile target windows do not cover primary scope")
        next_position = positions[end_id] + 1
    if next_position != last + 1:
        raise ValueError("reconcile target windows do not cover primary scope")
    units = draft_units(prepared)
    all_ids = [unit["unit_id"] for unit in units]
    if (scope["all_draft_unit_ids"] != all_ids
            or len(scope["draft_unit_ids"]) != len(set(scope["draft_unit_ids"]))
            or any(unit_id not in all_ids for unit_id in scope["draft_unit_ids"])):
        raise ValueError("reconcile target draft scope is invalid")
    return scope, inventory, prepared


def _project_risk_warnings(warnings: list[dict]) -> dict:
    """Group deterministic lexical anchors by U; keep exceptional rows whole.

    An uncited anchor's code/status and segment/window are reconstructable from
    this field and the primary coverage rows. Quote mismatches retain their
    original fields because their item index and cited set matter separately.
    """
    uncited_by_source: dict[str, list[list]] = {}
    other = []
    for warning in warnings:
        if (warning.get("code") == "source_risk_anchor_uncited"
                and warning.get("status") == "uncertain"
                and len(warning.get("source_ids", [])) == 1):
            source_id = warning["source_ids"][0]
            uncited_by_source.setdefault(source_id, []).append(
                [warning["risk_kind"], warning["anchors"]]
            )
        else:
            other.append(copy.deepcopy(warning))
    return {"uncited_by_source": uncited_by_source, "other": other}


def _project_inventory(inventory: dict) -> dict:
    """Omit only null/empty item fields and the separately supplied warnings."""
    nullable = {"speaker", "actor", "recipient", "action", "condition", "uncertainty"}
    arrays = {"alternatives", "correction_of"}
    projected = {key: copy.deepcopy(value) for key, value in inventory.items()
                 if key not in {"items", "coverage", "risk_warnings"}}
    projected["coverage"] = [
        {key: copy.deepcopy(value) for key, value in row.items()
         if not (key == "item_ids" and value == [])}
        for row in inventory["coverage"]
    ]
    projected["items"] = [
        {key: copy.deepcopy(value) for key, value in item.items()
         if not ((key in nullable and value is None)
                 or (key in arrays and value == []))}
        for item in inventory["items"]
    ]
    return projected


def build_reconcile_target_input_v2(source_text: str, target: dict) -> str:
    """Build one bounded half request while retaining exact source excerpts."""
    source, utterances, positions = _source(source_text)
    scope, inventory, prepared = _target_scope(target, [row["id"] for row in utterances])
    # Reuse all canonical source, draft, and inventory identity checks before
    # replacing the all-draft-citation excerpt selection with this target's
    # complete primary windows and explicit cross-boundary context.
    payload = json.loads(build_reconcile_input_v2(source_text, prepared, inventory))
    visible = set(scope["excerpt_source_ids"])
    unit_by_id = {unit["unit_id"]: unit for unit in draft_units(prepared)}
    for item in inventory["items"]:
        if not (set(item["source_ids"]) | set(item.get("correction_of", []))).issubset(visible):
            raise ValueError("reconcile target omits inventory citation")
    for warning in inventory["risk_warnings"]:
        if not set(warning["source_ids"]).issubset(visible):
            raise ValueError("reconcile target omits risk-warning citation")
    for unit_id in scope["draft_unit_ids"]:
        if not _unit_citations(prepared, unit_by_id[unit_id], positions).issubset(visible):
            raise ValueError("reconcile target omits draft-unit citation")
    # Exact text, speaker, source ID and order are retained. Millisecond
    # coordinates remain in the local source index for navigation and patches.
    payload["SOURCE_EXCERPTS"] = [
        {key: copy.deepcopy(value) for key, value in row.items()
         if key not in {"start_ms", "end_ms"}}
        for row in utterances if row["id"] in visible
    ]
    payload["EXCERPT_SCOPE"] = {
        "selection": "partition_primary_windows_plus_context_and_scoped_citations",
        "primary_utterance_count": len(scope["primary_source_ids"]),
        "context_utterance_count": len(visible) - len(scope["primary_source_ids"]),
        "included_utterance_count": len(visible),
        "total_utterance_count": len(utterances),
    }
    payload["INDEPENDENT_SOURCE_INVENTORY"] = _project_inventory(inventory)
    payload["SOURCE_RISK_WARNINGS"] = _project_risk_warnings(inventory["risk_warnings"])
    payload["DRAFT_UNITS"] = [unit_by_id[unit_id] for unit_id in scope["draft_unit_ids"]]
    payload["DRAFT_REFERENCE_UNIT_IDS"] = scope["all_draft_unit_ids"]
    task = payload.pop("TASK")
    payload["RECONCILE_SCOPE"] = {
        "part": scope["part"], "total_parts": 2,
        "cut_after_window_id": scope["cut_after_window_id"],
        "primary_start_id": scope["primary_source_ids"][0],
        "primary_end_id": scope["primary_source_ids"][-1],
        "primary_window_ids": scope["primary_window_ids"],
    }
    payload["TASK"] = task + (
        " Оцени все переданные SOURCE_WINDOWS_TO_RECHECK, пункты перечня и "
        "DRAFT_UNITS только для этой части; соседние реплики служат контекстом. "
        "Для отражения пункта перечня можно цитировать любой видимый текстовый "
        "unit полного DRAFT_DOCUMENT, используя его ID из DRAFT_REFERENCE_UNIT_IDS."
    )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def validate_reconciliation_target_report_v2(report: dict, target: dict,
                                             source_index: dict) -> dict:
    """Require every scoped assessment and reject citations absent from input."""
    source_ids = list(source_index.get("by_id", {})) if isinstance(source_index, dict) else []
    scope, inventory, prepared = _target_scope(target, source_ids)
    validate_reconciliation_report_v2(report, prepared, inventory, source_index)
    expected = (
        [row["window_id"] for row in inventory["coverage"]],
        [item["item_id"] for item in inventory["items"]],
        scope["draft_unit_ids"],
    )
    actual = (
        [row["window_id"] for row in report["source_window_assessments"]],
        [row["item_id"] for row in report["inventory_assessments"]],
        [row["unit_id"] for row in report["draft_assessments"]],
    )
    for label, wanted, received in zip(("source windows", "inventory items", "draft units"),
                                       expected, actual):
        if len(wanted) != len(received) or set(wanted) != set(received):
            raise ValueError(f"reconcile target {label} were not assessed exactly once")
    visible = set(scope["excerpt_source_ids"])
    primary = set(scope["primary_source_ids"])
    positions = {source_id: number for number, source_id in enumerate(source_ids)}
    scoped_units = set(scope["draft_unit_ids"])
    units_by_element: dict[tuple[str, int], set[str]] = {}
    for unit in draft_units(prepared):
        units_by_element.setdefault((unit["section"], unit["index"]), set()).add(
            unit["unit_id"])
    for row in report["source_window_assessments"]:
        if not row["source_ids"] or not set(row["source_ids"]).issubset(visible):
            raise ValueError("reconcile target window cites source outside excerpts")
    for row in report["draft_assessments"]:
        if not set(row["source_ids"]).issubset(visible):
            raise ValueError("reconcile target draft cites source outside excerpts")
    for finding in report["findings"]:
        if not set(finding["source_ids"]).issubset(visible):
            raise ValueError("reconcile target finding cites source outside excerpts")
        if not set(finding["source_ids"]).intersection(primary):
            raise ValueError("reconcile target finding has no primary source")
    for patch in report["patches"]:
        if patch["operation"] in {"replace", "remove"}:
            element = (patch["section"], patch["index"])
            target_units = units_by_element.get(element, set())
            if not target_units or not target_units.issubset(scoped_units):
                raise ValueError("reconcile patch target is outside scoped draft units")
            original = (prepared["meeting"] if patch["section"] == "meeting"
                        else prepared[patch["section"]][patch["index"]])
            original_citations = _citations_in(original, positions,
                                               "reconcile target original")
            if not original_citations.issubset(visible):
                raise ValueError("reconcile patch original cites source outside excerpts")
        if patch["item_json"] is None:
            continue
        citations = _citations_in(json.loads(patch["item_json"]), positions,
                                  "reconcile target patch")
        if not citations.issubset(visible):
            raise ValueError("reconcile target patch cites source outside excerpts")
        if not citations.intersection(primary):
            raise ValueError("reconcile target patch has no primary source")
    return report


def reconciliation_target_warnings_v2(report: dict, target: dict,
                                      source_index: dict) -> list[dict]:
    """Return ordinary diagnostics without requiring the other half's units."""
    validate_reconciliation_target_report_v2(report, target, source_index)
    scope = target["scope"]
    expected = set(scope["draft_unit_ids"])
    return [warning for warning in reconciliation_warnings_v2(
        report, target["draft"], target["inventory"], source_index)
        if warning.get("code") != "draft_unit_unassessed"
        or warning.get("unit_id") in expected]
