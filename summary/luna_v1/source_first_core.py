"""Deterministic source-first planning and report validation for Luna Batch.

The model's reports are proposals.  This module validates coordinates and
coverage, but never claims that a cited phrase entails a conclusion.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from .batch_stage_contracts_v1 import parse_stage_report
from .contract import validate_document
from .source import load_source


POLICY_VERSION = "luna_batch_source_first_v1"
# Size cores by source text plus framing; actual K/M are computed per input.
# Inputs exceeding the item policy block rather than losing source content.
CORE_TARGET_CHARS = 36_000
CONTEXT_RECORDS = 3
MAX_PLANNED_ITEMS = 12
MAX_RECOVERY_ITEMS = 2


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class SourceSnapshot:
    transcript_path: Path
    source_sha256: str
    source_text: str
    index: dict
    records: tuple[dict, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(record["id"] for record in self.records)


def load_snapshot(transcript_path: Path) -> SourceSnapshot:
    """Keep exact source text for quote checks alongside the writer projection."""
    path = Path(transcript_path)
    projected, index, sha = load_source(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    utterances = raw["utterances"]
    projected_rows = json.loads(projected)["utterances"]
    if len(utterances) != len(projected_rows):
        raise ValueError("source_snapshot_projection_mismatch")
    records = []
    for item, row in zip(utterances, projected_rows, strict=True):
        record = {
            "id": row["id"], "speaker_id": item.get("speaker"),
            "speaker_label": row["speaker"], "start_ms": row["start_ms"],
            "end_ms": row["end_ms"], "text": item["text"],
        }
        if row.get("needs_review") is True:
            record["needs_review"] = True
        if not isinstance(record["text"], str):
            raise ValueError("source_record_invalid")
        records.append(record)
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)) or tuple(ids) != tuple(index["by_id"]):
        raise ValueError("source_snapshot_ids_invalid")
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise ValueError("source_changed_during_snapshot")
    return SourceSnapshot(path, sha, projected, index, tuple(records))


def plan_packets(snapshot: SourceSnapshot, *, target_chars: int = CORE_TARGET_CHARS,
                 overlap: int = CONTEXT_RECORDS) -> tuple[dict, ...]:
    """Consecutive, exhaustive core packets with limited contextual overlap.

    An unusually long utterance remains intact and triggers route capacity
    checking against the actual request.  It is never truncated.
    """
    if target_chars < 1 or overlap < 0:
        raise ValueError("invalid_partition_profile")
    rows = snapshot.records
    boundaries: list[tuple[int, int]] = []
    start = 0
    size = 0
    for idx, row in enumerate(rows):
        cost = len(row["text"]) + 160
        if idx > start and size + cost > target_chars:
            boundaries.append((start, idx))
            start, size = idx, 0
        size += cost
    boundaries.append((start, len(rows)))
    packets = []
    for number, (lo, hi) in enumerate(boundaries, 1):
        left = max(0, lo - overlap)
        right = min(len(rows), hi + overlap)
        packets.append({
            "packet_id": f"P{number:03d}",
            "core_ids": [row["id"] for row in rows[lo:hi]],
            "context_ids": [row["id"] for row in (*rows[left:lo], *rows[hi:right])],
            "records": [deepcopy(row) for row in rows[left:right]],
            "core_start": lo, "core_end": hi,
        })
    flattened = [u for packet in packets for u in packet["core_ids"]]
    if flattened != list(snapshot.ids) or len(flattened) != len(set(flattened)):
        raise ValueError("source_core_coverage_invalid")
    return tuple(packets)


def plan_dimensions(packets: tuple[dict, ...]) -> dict:
    k = len(packets)
    m = k
    planned = 1 + k + m + 1 + 2
    if planned > MAX_PLANNED_ITEMS:
        return {"status": "dimension_budget_blocked", "K": k, "M": m,
                "planned_items": planned,
                "reason": "source_requires_more_than_current_12_item_policy"}
    return {"status": "ready", "K": k, "M": m,
            "planned_items": planned,
            "maximum_items": min(MAX_PLANNED_ITEMS, planned + MAX_RECOVERY_ITEMS),
            "batch_posts": 6, "maximum_batch_posts": 6}


def _one_of_each(items: list[dict], key: str, expected: set[str], label: str) -> dict[str, dict]:
    seen: dict[str, dict] = {}
    for item in items:
        identity = item[key]
        if identity not in expected or identity in seen:
            raise ValueError(f"{label}_identity_invalid")
        seen[identity] = item
    if set(seen) != expected:
        raise ValueError(f"{label}_coverage_incomplete")
    return seen


def validate_evidence(evidence: list[dict], snapshot: SourceSnapshot,
                      *, permitted_ids: set[str] | None = None) -> dict[str, dict]:
    exact = {row["id"]: row["text"] for row in snapshot.records}
    known = set(exact) if permitted_ids is None else permitted_ids
    checked = {}
    for item in evidence:
        evidence_id, u_id, quote = item["evidence_id"], item["u_id"], item["quote"]
        if (not evidence_id or evidence_id in checked or u_id not in known
                or not isinstance(quote, str) or not quote.strip()
                or exact[u_id].count(quote) != 1):
            raise ValueError("evidence_quote_or_identity_invalid")
        checked[evidence_id] = {**item, "source_offset": exact[u_id].index(quote),
                                "source_sha256": snapshot.source_sha256}
    return checked


def _validate_units(units: list[dict], evidence: dict[str, dict],
                    *, namespace: str) -> tuple[dict[str, dict], dict[str, dict]]:
    normalized = {}
    facets = {}
    for unit in units:
        local_id = unit["unit_id"]
        if not local_id or f"{namespace}:{local_id}" in normalized:
            raise ValueError("source_unit_identity_invalid")
        if not unit["text"].strip() or not unit["evidence_ids"]:
            raise ValueError("source_unit_content_invalid")
        if any(ref not in evidence for ref in unit["evidence_ids"]):
            raise ValueError("source_unit_evidence_missing")
        permanent = f"{namespace}:{local_id}"
        normalized[permanent] = {**unit, "unit_id": permanent}
        for facet in unit["facets"]:
            facet_id = f"{permanent}:{facet['facet_id']}"
            if (not facet["facet_id"] or facet_id in facets or not facet["value"].strip()
                    or not facet["evidence_ids"]
                    or any(ref not in evidence for ref in facet["evidence_ids"])):
                raise ValueError("source_facet_invalid")
            facets[facet_id] = {**facet, "facet_id": facet_id, "unit_id": permanent}
    return normalized, facets


def validate_inventory(raw: dict, packet: dict, snapshot: SourceSnapshot) -> dict:
    report = parse_stage_report("extract", raw).model_dump(mode="json")
    source_ids = {row["id"] for row in packet["records"]}
    evidence = validate_evidence(report["evidence"], snapshot, permitted_ids=source_ids)
    namespace = packet["packet_id"]
    units, facets = _validate_units(report["units"], evidence, namespace=namespace)
    core = set(packet["core_ids"])
    for unit in report["units"]:
        if not any(evidence[ref]["u_id"] in core for ref in unit["evidence_ids"]):
            raise ValueError("source_unit_without_core_evidence")
    local_units = {unit["unit_id"] for unit in report["units"]}
    accounts = _one_of_each(report["source_accounting"], "u_id",
                            set(packet["core_ids"]), "source_accounting")
    for row in accounts.values():
        if any(unit_id not in local_units for unit_id in row["unit_ids"]):
            raise ValueError("accounting_unknown_unit")
        if row["disposition"] == "content" and not row["unit_ids"]:
            raise ValueError("content_without_unit")
        if row["disposition"] == "content" and not any(
                any(evidence[ref]["u_id"] == row["u_id"]
                    for ref in next(unit for unit in report["units"]
                                    if unit["unit_id"] == local_id)["evidence_ids"])
                for local_id in row["unit_ids"]):
            raise ValueError("content_unit_lacks_core_evidence")
    if any(u not in packet["core_ids"] for u in report["unprocessed_ids"]):
        raise ValueError("unprocessed_source_unknown")
    links = []
    for link in report["open_links"]:
        if (link["unit_id"] not in local_units
                or any(ref not in evidence for ref in link["evidence_ids"])):
            raise ValueError("open_link_invalid")
        links.append({**link, "unit_id": f"{namespace}:{link['unit_id']}"})
    # An explicitly uncertain disposition is still unreviewed source, even
    # when the report claims to have processed every ID.
    complete = (report["complete"] and not report["unprocessed_ids"]
                and not any(row["disposition"] == "uncertain"
                            for row in accounts.values()))
    return {"packet_id": namespace, "complete": complete,
            "evidence": evidence, "units": units, "facets": facets,
            "source_accounting": accounts, "open_links": links,
            "unprocessed_ids": report["unprocessed_ids"], "native_report": raw}


def normalize_inventories(inventories: list[dict], packets: tuple[dict, ...],
                          snapshot: SourceSnapshot) -> dict:
    by_packet = _one_of_each(inventories, "packet_id",
                             {packet["packet_id"] for packet in packets}, "inventory_packet")
    units, facets, evidence, accounting, links = {}, {}, {}, {}, []
    complete = True
    for packet in packets:
        item = by_packet[packet["packet_id"]]
        complete &= bool(item["complete"])
        prefix = packet["packet_id"] + ":"
        for identity, value in item["units"].items():
            if identity in units:
                raise ValueError("inventory_identity_collision")
            copy = deepcopy(value)
            copy["evidence_ids"] = [prefix + ref for ref in copy["evidence_ids"]]
            for facet in copy["facets"]:
                facet["evidence_ids"] = [prefix + ref for ref in facet["evidence_ids"]]
                facet["facet_id"] = identity + ":" + facet["facet_id"]
            units[identity] = copy
        for identity, value in item["facets"].items():
            if identity in facets:
                raise ValueError("inventory_identity_collision")
            copy = deepcopy(value)
            copy["evidence_ids"] = [prefix + ref for ref in copy["evidence_ids"]]
            facets[identity] = copy
        for ref, value in item["evidence"].items():
            key = prefix + ref
            evidence[key] = {**value, "evidence_id": key}
        for u_id, account in item["source_accounting"].items():
            if u_id in accounting:
                raise ValueError("source_core_duplicate")
            copy = deepcopy(account)
            copy["unit_ids"] = [prefix + ref for ref in copy["unit_ids"]]
            accounting[u_id] = copy
        for link in item["open_links"]:
            copy = deepcopy(link)
            copy["evidence_ids"] = [prefix + ref for ref in copy["evidence_ids"]]
            links.append(copy)
    if set(accounting) != set(snapshot.ids):
        raise ValueError("source_core_not_exhaustive")
    return {"complete": complete, "units": units, "facets": facets,
            "evidence": evidence, "source_accounting": accounting,
            "open_links": links, "packet_ids": list(by_packet)}


def build_surfaces(document: dict, source_index: dict) -> dict[str, dict]:
    """Enumerate every model-authored assertion, including task fields."""
    validate_document(document, source_index)
    registry = {}

    def add(path: str, value: Any, sources: list[str], context: str) -> None:
        identity = "s-" + hashlib.sha256(path.encode()).hexdigest()[:16]
        registry[identity] = {"surface_id": identity, "entity_id": path.rsplit(".", 1)[0],
                              "field_key": path, "text_or_scalar": value,
                              "source_ids": sources, "parent_context": context}

    add("meeting.topic", document["meeting"]["topic"], [], "meeting")
    add("meeting.project", document["meeting"]["project"], [], "meeting")
    for section in ("main", "questions", "technical", "ideas", "verification"):
        for number, item in enumerate(document[section]):
            add(f"{section}.{number}.text", item["text"], item["source_ids"], section)
            if section == "verification":
                add(f"{section}.{number}.why_unresolved", item["why_unresolved"],
                    item["source_ids"], section)
    for section in ("timecodes", "chapters"):
        for number, item in enumerate(document[section]):
            refs = [item["start_id"], *([item["end_id"]] if item["end_id"] else [])]
            add(f"{section}.{number}.topic", item["topic"], refs, section)
            if section == "chapters":
                add(f"chapters.{number}.summary", item["summary"], item["source_ids"],
                    item["topic"])
                for detail_no, detail in enumerate(item["details"]):
                    add(f"chapters.{number}.details.{detail_no}.text", detail["text"],
                        detail["source_ids"], item["topic"])
    for number, task in enumerate(document["tasks"]):
        action_sources = task["field_sources"]["action"]
        for field in ("title", "description", "discussion_status", "assignee",
                      "due", "priority", "recipient"):
            sources = task["field_sources"].get(field, action_sources) or action_sources
            add(f"tasks.{number}.{field}", task[field], sources, task["title"])
    return registry


def partition_surfaces(registry: dict[str, dict], packets: tuple[dict, ...]) -> tuple[list[dict], ...]:
    if not packets:
        raise ValueError("invalid_audit_count")
    bins: list[list[dict]] = [[] for _ in packets]
    for number, surface in enumerate(registry.values()):
        refs = set(surface["source_ids"])
        scores = [len(refs & set(packet["core_ids"])) for packet in packets]
        best = max(scores)
        choice = scores.index(best) if best else number % len(packets)
        bins[choice].append(surface)
    if {item["surface_id"] for group in bins for item in group} != set(registry):
        raise ValueError("surface_partition_incomplete")
    return tuple(bins)


def validate_audit(raw: dict, *, packet: dict, packet_inventory: dict,
                   surfaces: list[dict], full_registry: dict[str, dict],
                   snapshot: SourceSnapshot) -> dict:
    report = parse_stage_report("audit", raw).model_dump(mode="json")
    evidence = validate_evidence(report["evidence"], snapshot,
                                 permitted_ids={row["id"] for row in packet["records"]})
    units = packet_inventory["units"]
    facets = packet_inventory["facets"]
    source_checks = _one_of_each(report["source_checks"], "unit_id",
                                 set(units), "source_check")
    expected_surfaces = {row["surface_id"] for row in surfaces}
    document_checks = _one_of_each(report["document_checks"], "surface_id",
                                   expected_surfaces, "document_check")
    additional, _ = _validate_units(report["additional_units"], evidence,
                                    namespace=packet["packet_id"] + ":extra")
    additional_prefix = packet["packet_id"] + ":extra:"
    def known_unit(ref: str) -> str:
        if ref in units or ref in additional:
            return ref
        if additional_prefix + ref in additional:
            return additional_prefix + ref
        raise ValueError("audit_finding_unknown_unit")
    finding_ids = set()
    for finding in report["findings"]:
        if (not finding["finding_id"] or finding["finding_id"] in finding_ids
                or not finding["evidence_ids"]
                or not (finding["affected_surface_ids"] or finding["affected_unit_ids"])
                or any(s not in full_registry for s in finding["affected_surface_ids"])
                or any(e not in evidence for e in finding["evidence_ids"])):
            raise ValueError("audit_finding_invalid")
        for ref in finding["affected_unit_ids"]:
            known_unit(ref)
        finding_ids.add(finding["finding_id"])
    context_ids = set()
    for request in report["context_requests"]:
        if (not request["request_id"] or request["request_id"] in context_ids
                or any(ref not in snapshot.ids for ref in request["known_source_ids"])
                or any(ref not in set(units) | set(facets) | set(full_registry)
                       for ref in request["affected_ids"])):
            raise ValueError("context_request_invalid")
        context_ids.add(request["request_id"])
    findings_by_id = {row["finding_id"]: row for row in report["findings"]}

    def negative_explained(identity: str, ids: list[str], *, unit: bool) -> bool:
        field = "affected_unit_ids" if unit else "affected_surface_ids"
        related = ({identity} | {facet_id for facet_id, facet in facets.items()
                                 if facet["unit_id"] == identity}) if unit else {identity}
        return (any(identity in findings_by_id[ref][field] for ref in ids)
                or any(related.intersection(row["affected_ids"])
                       for row in report["context_requests"]))

    for unit_id, check in source_checks.items():
        facet_checks = _one_of_each(check["facet_checks"], "facet_id",
                                    {facet_id for facet_id in facets if facets[facet_id]["unit_id"] == unit_id},
                                    "facet_check")
        if any(ref not in finding_ids for ref in check["finding_ids"]):
            raise ValueError("source_check_finding_invalid")
        negative = (check["inventory_verdict"] != "source_supported"
                    or check["coverage"] in {"partial", "absent", "uncertain"}
                    or any(facet_check["verdict"] in {"missing", "distorted", "uncertain"}
                           for facet_check in facet_checks.values()))
        if negative and not negative_explained(unit_id, check["finding_ids"], unit=True):
            raise ValueError("negative_source_verdict_unaccounted")
        if check["coverage"] == "full":
            if (check["inventory_verdict"] != "source_supported"
                    or not facet_checks
                    or not any(facet_check["verdict"] == "preserved"
                               for facet_check in facet_checks.values())
                    or any(facet_check["verdict"] not in {"preserved", "not_applicable"}
                           for facet_check in facet_checks.values())
                    or any(facet_check["verdict"] == "preserved"
                           and not facet_check["document_evidence"]
                           for facet_check in facet_checks.values())):
                raise ValueError("unsupported_full_coverage")
        for facet_check in facet_checks.values():
            for citation in facet_check["document_evidence"]:
                surface = full_registry.get(citation["surface_id"])
                if (surface is None or not isinstance(surface["text_or_scalar"], str)
                        or not citation["quote"]
                        or citation["quote"] not in surface["text_or_scalar"]):
                    raise ValueError("document_quote_invalid")
    for check in document_checks.values():
        if any(ref not in finding_ids for ref in check["finding_ids"]):
            raise ValueError("document_check_finding_invalid")
        negative = any(claim["verdict"] != "supported" for claim in check["claims"])
        if negative and not negative_explained(check["surface_id"],
                                               check["finding_ids"], unit=False):
            raise ValueError("negative_document_verdict_unaccounted")
        surface_value = full_registry[check["surface_id"]]["text_or_scalar"]
        if surface_value not in (None, "") and not check["claims"]:
            raise ValueError("document_surface_without_claim_check")
        for claim in check["claims"]:
            if any(ref not in evidence for ref in claim["evidence_ids"]):
                raise ValueError("document_claim_evidence_invalid")
            if claim["verdict"] == "supported" and not claim["evidence_ids"]:
                raise ValueError("document_claim_unsupported")
    if any(identity not in set(units) | expected_surfaces
           for identity in report["unprocessed_ids"]):
        raise ValueError("audit_unprocessed_identity_invalid")
    complete = report["complete"] and not report["unprocessed_ids"]
    prefix = packet["packet_id"] + ":audit:"
    normalized_evidence = {prefix + key: {**value, "evidence_id": prefix + key}
                           for key, value in evidence.items()}
    for check in source_checks.values():
        check["finding_ids"] = [prefix + ref for ref in check["finding_ids"]]
    for check in document_checks.values():
        check["finding_ids"] = [prefix + ref for ref in check["finding_ids"]]
        for claim in check["claims"]:
            claim["evidence_ids"] = [prefix + ref for ref in claim["evidence_ids"]]
    findings = []
    for finding in report["findings"]:
        findings.append({**finding, "finding_id": prefix + finding["finding_id"],
                         "affected_unit_ids": [known_unit(ref) for ref in finding["affected_unit_ids"]],
                         "evidence_ids": [prefix + ref for ref in finding["evidence_ids"]]})
    contexts = []
    for request in report["context_requests"]:
        contexts.append({**request, "request_id": prefix + request["request_id"]})
    for unit in additional.values():
        unit["evidence_ids"] = [prefix + ref for ref in unit["evidence_ids"]]
        for facet in unit["facets"]:
            facet["evidence_ids"] = [prefix + ref for ref in facet["evidence_ids"]]
    return {"packet_id": packet["packet_id"], "complete": complete,
            "evidence": normalized_evidence, "source_checks": source_checks,
            "document_checks": document_checks,
            "findings": findings, "context_requests": contexts,
            "additional_units": additional, "native_report": raw}


def validate_global(raw: dict, *, snapshot: SourceSnapshot,
                    expected_context_ids: set[str], full_registry: dict[str, dict],
                    expected_link_ids: set[str] | None = None,
                    expected_unit_ids: set[str] | None = None) -> dict:
    report = parse_stage_report("global", raw).model_dump(mode="json")
    evidence = validate_evidence(report["evidence"], snapshot)
    resolutions = _one_of_each(report["resolutions"], "request_id",
                               expected_context_ids, "global_resolution")
    for row in resolutions.values():
        if (any(ref not in evidence for ref in row["evidence_ids"])
                or any(surface_id not in full_registry for surface_id in row["affected_surface_ids"])):
            raise ValueError("global_resolution_invalid")
    additional, _ = _validate_units(report["additional_units"], evidence,
                                    namespace="global:extra")
    def known_unit(ref: str) -> str:
        if expected_unit_ids is None or ref in expected_unit_ids or ref in additional:
            return ref
        if "global:extra:" + ref in additional:
            return "global:extra:" + ref
        raise ValueError("global_unknown_unit")
    finding_ids = set()
    for finding in report["findings"]:
        if (finding["finding_id"] in finding_ids
                or any(ref not in evidence for ref in finding["evidence_ids"])
                or any(surface_id not in full_registry for surface_id in finding["affected_surface_ids"])):
            raise ValueError("global_finding_invalid")
        for ref in finding["affected_unit_ids"]:
            known_unit(ref)
        finding_ids.add(finding["finding_id"])
    if expected_link_ids is not None:
        _one_of_each(report["link_checks"], "link_id", expected_link_ids, "global_link")
    for link in report["link_checks"]:
        if any(ref not in evidence for ref in link["evidence_ids"]):
            raise ValueError("global_link_evidence_invalid")
        for ref in link["unit_ids"]:
            known_unit(ref)
    if any(surface_id not in full_registry for surface_id in report["affected_surfaces"]):
        raise ValueError("global_affected_surface_unknown")
    if any(identity not in expected_context_ids for identity in report["unprocessed_ids"]):
        raise ValueError("global_unprocessed_unknown")
    complete = report["complete"] and not report["unprocessed_ids"]
    prefix = "global:"
    normalized_evidence = {prefix + key: {**value, "evidence_id": prefix + key}
                           for key, value in evidence.items()}
    for resolution in resolutions.values():
        resolution["evidence_ids"] = [prefix + ref for ref in resolution["evidence_ids"]]
    findings = [{**finding, "finding_id": prefix + finding["finding_id"],
                 "affected_unit_ids": [known_unit(ref) for ref in finding["affected_unit_ids"]],
                 "evidence_ids": [prefix + ref for ref in finding["evidence_ids"]]}
                for finding in report["findings"]]
    links = [{**link, "unit_ids": [known_unit(ref) for ref in link["unit_ids"]],
              "evidence_ids": [prefix + ref for ref in link["evidence_ids"]]}
             for link in report["link_checks"]]
    for unit in additional.values():
        unit["evidence_ids"] = [prefix + ref for ref in unit["evidence_ids"]]
        for facet in unit["facets"]:
            facet["evidence_ids"] = [prefix + ref for ref in facet["evidence_ids"]]
    return {"complete": complete, "evidence": normalized_evidence, "resolutions": resolutions,
            "findings": findings, "additional_units": additional,
            "link_checks": links, "native_report": raw}


_ADDABLE_SECTIONS = frozenset({"main", "timecodes", "tasks", "questions", "technical",
                               "ideas", "verification", "chapters"})
_REPLACEABLE_TASK_FIELDS = frozenset({"title", "description", "discussion_status",
                                      "assignee", "due", "priority", "recipient"})


def _field_target(document: dict, path: str) -> tuple[Any, str | int]:
    parts = path.split(".")
    node: Any = document
    for part in parts[:-1]:
        if isinstance(node, list):
            if not part.isdigit() or int(part) >= len(node):
                raise ValueError("patch_target_missing")
            node = node[int(part)]
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise ValueError("patch_target_missing")
    last: str | int = parts[-1]
    if isinstance(node, list):
        if not last.isdigit() or int(last) >= len(node):
            raise ValueError("patch_target_missing")
        last = int(last)
    elif not isinstance(node, dict) or last not in node:
        raise ValueError("patch_target_missing")
    return node, last


def validate_patch_plan(raw: dict, *, findings: dict[str, dict],
                        registry: dict[str, dict], known_evidence_ids: set[str]) -> dict:
    """Authorize only app-issued targets and a bounded subset of operations."""
    report = parse_stage_report("repair", raw).model_dump(mode="json")
    bundles = _one_of_each(report["bundles"], "bundle_id",
                           {bundle["bundle_id"] for bundle in report["bundles"]},
                           "patch_bundle")
    if len(bundles) != len(report["bundles"]):
        raise ValueError("patch_bundle_duplicate")
    if any(ref not in findings for ref in report["unresolved"] + report["unprocessed_finding_ids"]):
        raise ValueError("patch_finding_unknown")
    for bundle_id, bundle in bundles.items():
        if (not bundle_id or not bundle["finding_ids"] or not bundle["operations"]
                or any(ref not in findings for ref in bundle["finding_ids"])
                or any(ref not in known_evidence_ids for ref in bundle["evidence_ids"])
                or any(ref not in registry for ref in bundle["affected_surface_ids"])):
            raise ValueError("patch_bundle_not_grounded")
        if any(dep == bundle_id or dep not in bundles for dep in bundle["dependency_bundle_ids"]):
            raise ValueError("patch_dependency_invalid")
        for operation in bundle["operations"]:
            kind, target, field, value_json = (operation["kind"], operation["target_id"],
                                                operation["field_key"], operation["value_json"])
            if kind in {"replace_field", "update_task"}:
                if (target not in registry or field != registry[target]["field_key"]
                        or target not in bundle["affected_surface_ids"]
                        or value_json is None or operation["position_after_id"] is not None
                        or operation["temp_id"] is not None or operation["lineage_ids"]):
                    raise ValueError("patch_target_unauthorized")
                if kind == "update_task" and not field.startswith("tasks."):
                    raise ValueError("patch_task_target_invalid")
            elif kind in {"add_section_item", "create_task"}:
                if (target is not None or field not in _ADDABLE_SECTIONS
                        or (kind == "create_task") != (field == "tasks")
                        or value_json is None or operation["lineage_ids"]):
                    raise ValueError("patch_add_unauthorized")
                if operation["position_after_id"] is not None:
                    anchor = registry.get(operation["position_after_id"])
                    if anchor is None or not anchor["field_key"].startswith(field + "."):
                        raise ValueError("patch_position_unauthorized")
            else:
                # Relation/split/merge need a separately reviewed app-owned
                # action lineage contract; unresolved is safer than guessing.
                raise ValueError("patch_relation_operation_unsupported")
            try:
                json.loads(value_json)
            except (TypeError, ValueError):
                raise ValueError("patch_value_json_invalid") from None
    visited, active = set(), set()
    def visit(node: str) -> None:
        if node in active:
            raise ValueError("patch_dependency_cycle")
        if node in visited:
            return
        active.add(node)
        for dependency in bundles[node]["dependency_bundle_ids"]:
            visit(dependency)
        active.remove(node)
        visited.add(node)
    for root in bundles:
        visit(root)
    accounted = [ref for bundle in bundles.values() for ref in bundle["finding_ids"]]
    accounted.extend(report["unresolved"])
    accounted.extend(report["unprocessed_finding_ids"])
    if Counter(accounted) != Counter({key: 1 for key in findings}):
        raise ValueError("patch_findings_not_accounted")
    return {"complete": report["complete"] and not report["unprocessed_finding_ids"],
            "bundles": bundles, "unresolved": report["unresolved"],
            "unprocessed_finding_ids": report["unprocessed_finding_ids"],
            "native_report": raw}


def stage_patch_candidate(document: dict, plan: dict, registry: dict[str, dict],
                          source_index: dict,
                          evidence_lookup: dict[str, dict] | None = None) -> tuple[dict, dict]:
    """Apply candidate operations to a copy and retain expected-before hashes."""
    candidate = deepcopy(document)
    before_hashes: dict[str, list[dict]] = {}
    touched = {}
    insertion_anchors = set()
    prior_insertions: dict[str, list[int]] = {}
    for bundle_id in plan["bundles"]:
        before_hashes[bundle_id] = []
    # Replacements address D0 paths. Apply them before any insertion can shift
    # a later index in the same section.
    operations = [(bundle_id, op)
                  for bundle_id, bundle in plan["bundles"].items()
                  for op in bundle["operations"]]
    operations.sort(key=lambda pair: pair[1]["kind"] not in {"replace_field", "update_task"})
    for bundle_id, op in operations:
            bundle = plan["bundles"][bundle_id]
            value = json.loads(op["value_json"])
            if op["kind"] in {"replace_field", "update_task"}:
                field = registry[op["target_id"]]["field_key"]
                if field in touched and touched[field] != bundle_id:
                    raise ValueError("overlapping_patch_bundles")
                touched[field] = bundle_id
                old_container, old_key = _field_target(document, field)
                container, key = _field_target(candidate, field)
                before_hashes[bundle_id].append({
                    "target_id": op["target_id"], "expected_before_hash": digest(old_container[old_key]),
                    "value_hash": digest(value),
                })
                container[key] = value
                if (field.startswith("tasks.") and field.rsplit(".", 1)[-1]
                        in {"assignee", "due", "priority", "recipient"}
                        and value is not None):
                    if evidence_lookup is None or not bundle["evidence_ids"]:
                        raise ValueError("task_metadata_repair_without_evidence")
                    cited = []
                    for evidence_id in bundle["evidence_ids"]:
                        evidence = evidence_lookup.get(evidence_id)
                        if not isinstance(evidence, dict) or not evidence.get("u_id"):
                            raise ValueError("task_metadata_repair_evidence_unknown")
                        cited.append(evidence["u_id"])
                    task = candidate["tasks"][int(field.split(".")[1])]
                    field_name = field.rsplit(".", 1)[-1]
                    task["field_sources"][field_name] = list(dict.fromkeys(
                        [*task["field_sources"][field_name], *cited]))
                    task["source_ids"] = list(dict.fromkeys([*task["source_ids"], *cited]))
            else:
                section = op["field_key"]
                if not isinstance(value, dict):
                    raise ValueError("new_section_item_invalid")
                before_hashes[bundle_id].append({
                    "target_id": section, "expected_before_hash": digest(document[section]),
                    "value_hash": digest(value),
                })
                after = op["position_after_id"]
                if after is None:
                    candidate[section].append(value)
                else:
                    if (section, after) in insertion_anchors:
                        raise ValueError("reused_patch_insertion_anchor")
                    insertion_anchors.add((section, after))
                    anchor_path = registry[after]["field_key"].split(".")
                    if len(anchor_path) < 3 or not anchor_path[1].isdigit():
                        raise ValueError("patch_position_unauthorized")
                    original_index = int(anchor_path[1])
                    shift = sum(index <= original_index
                                for index in prior_insertions.get(section, []))
                    candidate[section].insert(original_index + 1 + shift, value)
                    prior_insertions.setdefault(section, []).append(original_index)
    validate_document(candidate, source_index)
    return candidate, before_hashes


def remap_mark_targets(findings: list[dict], *, plan: dict | None,
                       accepted: set[str], old_registry: dict[str, dict],
                       new_registry: dict[str, dict]) -> list[dict]:
    """Translate D0 surface IDs after accepted insertions, or refuse to guess."""
    insertion_after: dict[str, list[int]] = {}
    if plan:
        for bundle_id, bundle in plan["bundles"].items():
            if bundle_id not in accepted:
                continue
            for op in bundle["operations"]:
                if op["kind"] not in {"add_section_item", "create_task"}:
                    continue
                anchor = op["position_after_id"]
                if anchor is None:
                    continue  # append cannot move an original item
                path = old_registry[anchor]["field_key"].split(".")
                insertion_after.setdefault(op["field_key"], []).append(int(path[1]))
    by_path = {row["field_key"]: identity for identity, row in new_registry.items()}
    mapped = []
    for finding in findings:
        copy = deepcopy(finding)
        copy["affected_surface_ids"] = []
        for identity in finding.get("affected_surface_ids", []):
            old = old_registry.get(identity)
            if old is None:
                raise ValueError("mark_surface_unknown")
            parts = old["field_key"].split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                original_index = int(parts[1])
                parts[1] = str(original_index + sum(
                    after < original_index for after in insertion_after.get(parts[0], [])))
            new_identity = by_path.get(".".join(parts))
            if new_identity is None:
                raise ValueError("mark_surface_shift_unresolved")
            copy["affected_surface_ids"].append(new_identity)
        mapped.append(copy)
    return mapped


def validate_verification(raw: dict, *, plan: dict, snapshot: SourceSnapshot,
                          registry: dict[str, dict]) -> dict:
    report = parse_stage_report("verify", raw).model_dump(mode="json")
    checks = _one_of_each(report["bundle_checks"], "bundle_id",
                          set(plan["bundles"]), "verification_bundle")
    evidence = {}
    for check in checks.values():
        for evidence_id, value in validate_evidence(check["evidence"], snapshot).items():
            if evidence_id in evidence:
                raise ValueError("verification_evidence_id_repeated")
            evidence[evidence_id] = value
        if any(ref not in registry for ref in check["preserved_surface_ids"]):
            raise ValueError("verification_surface_invalid")
        if check["verdict"] == "accept" and (not check["evidence"] or check["regressions"]):
            raise ValueError("verification_accept_unsupported")
    new_ids = set()
    for finding in report["new_findings"]:
        if (not finding["finding_id"] or finding["finding_id"] in new_ids
                or any(ref not in evidence for ref in finding["evidence_ids"])
                or any(ref not in registry for ref in finding["affected_surface_ids"])):
            raise ValueError("verification_new_finding_invalid")
        new_ids.add(finding["finding_id"])
    if any(ref not in plan["bundles"] for ref in report["unprocessed_bundle_ids"]):
        raise ValueError("verification_unprocessed_unknown")
    accepted = {bundle_id for bundle_id, check in checks.items()
                if check["verdict"] == "accept" and bundle_id not in report["unprocessed_bundle_ids"]}
    # Treat every connected dependency component as one atomic group.
    neighbors = {bundle_id: set(bundle["dependency_bundle_ids"])
                 for bundle_id, bundle in plan["bundles"].items()}
    for bundle_id, refs in list(neighbors.items()):
        for ref in refs:
            neighbors[ref].add(bundle_id)
    visited = set()
    grouped_acceptance = set()
    for root in neighbors:
        if root in visited:
            continue
        group, stack = set(), [root]
        while stack:
            node = stack.pop()
            if node in group:
                continue
            group.add(node)
            stack.extend(neighbors[node])
        visited.update(group)
        if group <= accepted:
            grouped_acceptance.update(group)
    accepted = grouped_acceptance
    if not report["complete"] or report["new_findings"]:
        accepted.clear()
    prefix = "verify:"
    normalized_evidence = {prefix + key: {**value, "evidence_id": prefix + key}
                           for key, value in evidence.items()}
    new_findings = [{**finding, "finding_id": prefix + finding["finding_id"],
                     "evidence_ids": [prefix + ref for ref in finding["evidence_ids"]]}
                    for finding in report["new_findings"]]
    return {"complete": report["complete"] and not report["unprocessed_bundle_ids"],
            "accepted": accepted, "rejected": set(plan["bundles"]) - accepted,
            "new_findings": new_findings, "evidence": normalized_evidence,
            "native_report": raw}


def accepted_patch_document(document: dict, plan: dict, verification: dict,
                            registry: dict[str, dict], source_index: dict,
                            evidence_lookup: dict[str, dict] | None = None) -> dict:
    selected = {key: value for key, value in plan["bundles"].items()
                if key in verification["accepted"]}
    if not selected:
        return deepcopy(document)
    candidate, _ = stage_patch_candidate(document, {"bundles": selected}, registry,
                                          source_index, evidence_lookup)
    return candidate


def safe_mark_document(document: dict, findings: list[dict], registry: dict[str, dict],
                       snapshot: SourceSnapshot, evidence_lookup: dict[str, dict]) -> tuple[dict, list[dict]]:
    """Put explicit caution on contested claims in a copy of a usable draft.

    A review failure never silently removes a task or promotes a model finding
    to source truth.  This projection is a local safety annotation, with the
    native D0 retained separately.  Optional actor/metadata values called
    into question are cleared, while the question is shown to the reader.
    """
    candidate = deepcopy(document)
    marks = []
    seen_questions = set()
    for finding in findings:
        cited = []
        for evidence_id in finding.get("evidence_ids", []):
            row = evidence_lookup.get(evidence_id)
            if isinstance(row, dict) and row.get("u_id") in snapshot.index["by_id"]:
                cited.append(row["u_id"])
        affected = []
        for surface_id in finding.get("affected_surface_ids", []):
            surface = registry.get(surface_id)
            if surface is None:
                continue
            path = surface["field_key"]
            affected.append(path)
            container, key = _field_target(candidate, path)
            value = container[key]
            if path.startswith("tasks.") and path.rsplit(".", 1)[-1] in {
                    "assignee", "due", "priority", "recipient"}:
                container[key] = None
            elif path.startswith("tasks.") and path.endswith(".discussion_status"):
                container[key] = "unknown"
            elif isinstance(value, str) and value and not value.startswith("Требует проверки по источнику: "):
                container[key] = "Требует проверки по источнику: " + value
            cited.extend(surface["source_ids"])
        cited = list(dict.fromkeys(ref for ref in cited if ref in snapshot.index["by_id"]))
        problem = finding.get("problem", "Смысловое утверждение требует проверки")
        if not isinstance(problem, str) or not problem.strip():
            problem = "Смысловое утверждение требует проверки"
        mark = {"finding_id": finding.get("finding_id"), "affected_fields": affected,
                "source_ids": cited, "problem": problem}
        marks.append(mark)
        identity = (problem, tuple(cited))
        if cited and identity not in seen_questions:
            seen_questions.add(identity)
            candidate["verification"].append({
                "text": "Проверить утверждение: " + problem,
                "why_unresolved": "Автоматическая сверка не подтвердила точный смысл; исходная формулировка сохранена с оговоркой.",
                "source_ids": cited,
            })
    validate_document(candidate, snapshot.index)
    return candidate, marks
