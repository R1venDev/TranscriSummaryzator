"""Independent, grounded findings survive an incomplete typed review.

Never claim positive coverage from a salvaged audit. Unknown/duplicate IDs and
bad quotes are discarded with provenance, not guessed or silently repaired.
Truncated JSON is not accepted here. Raw reports remain immutable.
"""
from __future__ import annotations

from collections import Counter

from .batch_stage_contracts_v1 import parse_stage_report
from .source_first_core import SourceSnapshot, _validate_units, validate_evidence, digest


def salvage_review(raw: dict, *, stage: str, snapshot: SourceSnapshot,
                   registry: dict, known_evidence: dict, known_units: set[str],
                   packet: dict | None = None, known_contexts: set[str] | None = None,
                   known_links: set[str] | None = None, known_context_targets: set[str] | None = None) -> dict:
    report = parse_stage_report(stage, raw).model_dump(mode="json")
    discarded = []
    permitted = {row["id"] for row in packet["records"]} if packet else None
    prefix = packet["packet_id"] + ":audit:" if packet else "global:"
    namespace = packet["packet_id"] + ":extra" if packet else "global:extra"

    def discard(kind, identity, reason):
        discarded.append({"kind": kind, "id": identity, "reason": reason})

    evidence = {}
    for key, value in known_evidence.items():
        try:
            evidence.update(validate_evidence([{field: value[field] for field in
                ("evidence_id", "u_id", "quote")}], snapshot, permitted_ids=permitted))
        except (ValueError, KeyError, TypeError):
            discard("input_evidence", key, "source_quote_invalid")
    counts = Counter(row["evidence_id"] for row in report["evidence"])
    for row in report["evidence"]:
        key = row["evidence_id"]
        try:
            if counts[key] != 1:
                raise ValueError("duplicate_identity")
            emitted = validate_evidence([row], snapshot, permitted_ids=permitted)[key]
            if key in evidence:
                if emitted["u_id"] != evidence[key]["u_id"]:
                    raise ValueError("reference_collision")
                if emitted["quote"] != evidence[key]["quote"]:
                    emitted["input_quote_sha256"] = digest(evidence[key]["quote"])
                    emitted["reference_resolution"] = "report_local_span_same_u_v1"
            evidence[key] = emitted
        except (ValueError, KeyError, TypeError) as exc:
            evidence.pop(key, None)  # An ambiguous ID cannot inherit a guess.
            discard("evidence", key, str(exc))
    additional = {}
    unit_counts = Counter(row["unit_id"] for row in report["additional_units"])
    for row in report["additional_units"]:
        try:
            if unit_counts[row["unit_id"]] != 1:
                raise ValueError("duplicate_identity")
            units, _ = _validate_units([row], evidence, namespace=namespace)
            additional.update(units)
        except (ValueError, KeyError, TypeError) as exc:
            discard("additional_unit", row["unit_id"], str(exc))

    def unit_ref(ref):
        if ref in known_units or ref in additional:
            return ref
        if namespace + ":" + ref in additional:
            return namespace + ":" + ref
        raise ValueError("unknown_unit")

    def refs_valid(row):
        if any(ref not in evidence for ref in row["evidence_ids"]):
            raise ValueError("evidence_missing")
        if any(ref not in registry for ref in row.get("affected_surface_ids", [])):
            raise ValueError("unknown_surface")

    findings = []
    counts = Counter(row["finding_id"] for row in report["findings"])
    for row in report["findings"]:
        try:
            identity = row["finding_id"]
            if not identity or counts[identity] != 1 or not row["evidence_ids"]:
                raise ValueError("finding_identity_or_evidence_invalid")
            if not (row["affected_surface_ids"] or row["affected_unit_ids"]):
                raise ValueError("finding_without_target")
            refs_valid(row)
            findings.append({**row, "finding_id": prefix + identity,
                "affected_unit_ids": [unit_ref(ref) for ref in row["affected_unit_ids"]],
                "evidence_ids": [prefix + ref for ref in row["evidence_ids"]]})
        except (ValueError, KeyError, TypeError) as exc:
            discard("finding", row["finding_id"], str(exc))
    result = {"complete": False, "salvaged": True, "discarded": discarded,
        "findings": findings, "native_report": raw,
        "evidence": {prefix + key: {**row, "evidence_id": prefix + key}
                     for key, row in evidence.items()}, "additional_units": additional}
    for unit in additional.values():
        unit["evidence_ids"] = [prefix + ref for ref in unit["evidence_ids"]]
        for facet in unit["facets"]:
            facet["evidence_ids"] = [prefix + ref for ref in facet["evidence_ids"]]
    if stage == "audit":
        contexts = []
        counts = Counter(row["request_id"] for row in report["context_requests"])
        for row in report["context_requests"]:
            if (not row["request_id"] or counts[row["request_id"]] != 1
                    or any(ref not in snapshot.ids for ref in row["known_source_ids"])
                    or any(ref not in (known_context_targets or set()) for ref in row["affected_ids"])):
                discard("context_request", row["request_id"], "identity_invalid")
                continue
            contexts.append({**row, "request_id": prefix + row["request_id"]})
        result.update(packet_id=packet["packet_id"], source_checks={},
                      document_checks={}, context_requests=contexts)
        discard("coverage", packet["packet_id"], "positive_coverage_not_salvaged")
    else:
        resolutions, links = {}, []
        for field, id_field, expected in (("resolutions", "request_id", known_contexts or set()),
                                          ("link_checks", "link_id", known_links or set())):
            counts = Counter(row[id_field] for row in report[field])
            for row in report[field]:
                identity = row[id_field]
                try:
                    if identity not in expected or counts[identity] != 1:
                        raise ValueError("unknown_or_duplicate_identity")
                    refs_valid(row)
                    copy = {**row, "evidence_ids": [prefix + ref for ref in row["evidence_ids"]]}
                    if field == "link_checks":
                        copy["unit_ids"] = [unit_ref(ref) for ref in row["unit_ids"]]
                        links.append(copy)
                    else:
                        resolutions[identity] = copy
                except (ValueError, KeyError, TypeError) as exc:
                    discard(field, identity, str(exc))
            retained = set(resolutions) if field == "resolutions" else {row["link_id"] for row in links}
            for identity in sorted(expected - retained):
                discard(field, identity, "missing_verdict")
        result.update(resolutions=resolutions, link_checks=links)
    return result
