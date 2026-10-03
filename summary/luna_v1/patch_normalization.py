"""Lossless repair-envelope normalization before independent patch verification.

No meaning is inferred here. Coordinates come from the application registry;
text merges accept only identical or disjoint edits against the same D0.
The exact native report stays immutable and every transformation is recorded.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from difflib import SequenceMatcher
import json
import re

from .batch_stage_contracts_v1 import parse_stage_report
from .contract import validate_document
from .source_first_core import digest, stage_patch_candidate, validate_patch_plan

NORMALIZATION_VERSION = "repair_envelope_v1"
TASK_FIELDS = ("title", "description", "discussion_status", "assignee", "due",
               "priority", "recipient")


def _merge_text(base: str, variants: list[str]) -> str:
    """Three-way merge; never choose a winner for differing overlapping edits."""
    tokens = re.findall(r"\s+|\S+", base)
    edits = []
    for variant in dict.fromkeys(variants):
        changed = re.findall(r"\s+|\S+", variant)
        for tag, lo, hi, a, b in SequenceMatcher(None, tokens, changed,
                                                autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            edit = (lo, hi, tuple(changed[a:b]))
            if edit in edits:
                continue
            for x, y, replacement in edits:
                # Insertions at the same point need identical bytes. An
                # insertion at a replacement boundary is otherwise disjoint.
                overlap = (max(lo, x) < min(hi, y) or
                    lo == hi == x == y or lo == hi and x < lo < y or
                    x == y and lo < x < hi)
                if overlap:
                    raise ValueError("patch_text_merge_conflict")
            edits.append(edit)
    for lo, hi, replacement in sorted(edits, key=lambda row: (row[0], row[1]),
                                       reverse=True):
        tokens[lo:hi] = replacement
    return "".join(tokens)


def normalize_patch_report(raw: dict, *, document: dict, registry: dict,
                           findings: dict, evidence: dict, source_index: dict
                           ) -> tuple[dict, dict]:
    """Return a validated plan plus provenance; isolate invalid dependency groups.

    Whole task objects may only address an existing app-issued task root. Their
    changed business fields become registry-addressed scalar operations. The
    app retains action identity and derives provenance from bundle evidence.
    Unknown IDs, unauthorized paths, source IDs, UUIDs and conflicting edits
    are never guessed or silently accepted.
    """
    report = parse_stage_report("repair", raw).model_dump(mode="json")
    provenance = {"version": NORMALIZATION_VERSION, "native_sha256": digest(raw),
                  "transforms": [], "discarded": [], "bundle_lineage": {}}
    for field in ("unresolved", "unprocessed_finding_ids"):
        resolved = []
        for value in report[field]:
            identity = value
            if value not in findings:
                prefix, separator, explanation = value.partition(" — ")
                if separator and prefix in findings:
                    identity = prefix
                    provenance["transforms"].append({"kind": "id_explanation",
                        "field": field, "id": prefix, "explanation": explanation})
                else:
                    raise ValueError("patch_finding_unknown")
            resolved.append(identity)
        report[field] = resolved
    bundles = report["bundles"]
    if len({b["bundle_id"] for b in bundles}) != len(bundles):
        raise ValueError("patch_bundle_duplicate")
    accounted = [f for b in bundles for f in b["finding_ids"]]
    accounted += report["unresolved"] + report["unprocessed_finding_ids"]
    if Counter(accounted) != Counter({key: 1 for key in findings}):
        raise ValueError("patch_findings_not_accounted")
    by_path = {s["field_key"]: identity for identity, s in registry.items()}
    errors = {}
    for bundle in bundles:
        original_ops = bundle["operations"]
        converted = []
        try:
            for op in original_ops:
                if (op["kind"] in {"replace_field", "update_task"}
                        and op["target_id"] not in registry
                        and op["field_key"] in by_path):
                    resolved = by_path[op["field_key"]]
                    if op["target_id"] in {registry[resolved]["entity_id"], op["field_key"]}:
                        provenance["transforms"].append({"kind": "entity_field_to_surface",
                            "native_target_id": op["target_id"], "field_key": op["field_key"],
                            "surface_id": resolved})
                        op = {**op, "target_id": resolved}
                if op["kind"] != "update_task" or op["target_id"] in registry:
                    converted.append(op)
                    continue
                roots = {s["entity_id"] for s in registry.values()
                         if s["field_key"].startswith("tasks.")}
                root = op["target_id"]
                if (root not in roots or op["field_key"] is not None or
                        op["position_after_id"] is not None or op["temp_id"] is not None
                        or op["lineage_ids"]):
                    raise ValueError("patch_task_root_unauthorized")
                index = int(root.split(".")[1])
                task = json.loads(op["value_json"])
                check = deepcopy(document)
                check["tasks"][index] = task
                validate_document(check, source_index)
                for name in TASK_FIELDS:
                    if task[name] == document["tasks"][index][name]:
                        continue
                    target = by_path[root + "." + name]
                    converted.append({**op, "target_id": target,
                        "field_key": root + "." + name,
                        "value_json": json.dumps(task[name], ensure_ascii=False)})
                    if target not in bundle["affected_surface_ids"]:
                        bundle["affected_surface_ids"].append(target)
                provenance["transforms"].append({"kind": "task_to_fields",
                    "bundle_id": bundle["bundle_id"], "task_root": root,
                    "native_operation_sha256": digest(op),
                    "provenance_policy": "retain_action_anchors_derive_changed_fields_from_evidence"})
            bundle["operations"] = converted
            # Validate each bundle without pretending its dependencies vanished.
            probe = {**report, "bundles": [{**bundle, "dependency_bundle_ids": []}],
                "unresolved": [f for f in findings if f not in bundle["finding_ids"]],
                "unprocessed_finding_ids": []}
            validate_patch_plan(probe, findings=findings, registry=registry,
                                known_evidence_ids=set(evidence))
        except (ValueError, KeyError, TypeError) as exc:
            errors[bundle["bundle_id"]] = str(exc)

    # Shared targets and explicit dependencies form atomic components. This
    # also prevents accepting one side of a conflicting paragraph replacement.
    by_id = {b["bundle_id"]: b for b in bundles}
    active, done = set(), set()
    def visit(node):
        if node in active:
            errors[node] = "patch_dependency_cycle"
            return
        if node in done:
            return
        active.add(node)
        for dep in by_id[node]["dependency_bundle_ids"]:
            if dep in by_id:
                visit(dep)
        active.remove(node)
        done.add(node)
    for node in by_id:
        visit(node)
    neighbors = {identity: set() for identity in by_id}
    target_owner = {}
    for identity, bundle in by_id.items():
        for dep in bundle["dependency_bundle_ids"]:
            if dep not in by_id or dep == identity:
                errors[identity] = "patch_dependency_invalid"
            else:
                neighbors[identity].add(dep)
                neighbors[dep].add(identity)
        for op in bundle["operations"]:
            target = ("field", op["target_id"]) if op["kind"] in {"update_task", "replace_field"} else None
            if (op["kind"] in {"add_section_item", "create_task"}
                    and op["position_after_id"] is not None):
                target = ("insertion", op["field_key"], op["position_after_id"])
            if target is None:
                continue
            if target in target_owner:
                other = target_owner[target]
                neighbors[identity].add(other)
                neighbors[other].add(identity)
            target_owner[target] = identity
    visited, normalized = set(), []
    for identity in by_id:
        if identity in visited:
            continue
        stack, group = [identity], set()
        while stack:
            node = stack.pop()
            if node in group:
                continue
            group.add(node)
            stack.extend(neighbors[node])
        visited.update(group)
        members = [by_id[key] for key in by_id if key in group]
        try:
            if group & errors.keys():
                raise ValueError(next(errors[k] for k in by_id if k in group and k in errors))
            merged = deepcopy(members[0])
            for field in ("finding_ids", "evidence_ids", "affected_surface_ids"):
                merged[field] = list(dict.fromkeys(v for b in members for v in b[field]))
            merged["dependency_bundle_ids"] = []  # Entire component is now one bundle.
            if len(members) > 1:
                merged["bundle_id"] = "joined-" + digest([b["bundle_id"] for b in members])[:16]
                merged["preservation_notes"] = "\n".join(b["preservation_notes"] for b in members)
            ops, positions = [], {}
            for member in members:
                for op in member["operations"]:
                    target = op["target_id"] if op["kind"] in {"update_task", "replace_field"} else None
                    if target is None or target not in positions:
                        if target is not None:
                            positions[target] = len(ops)
                        ops.append(deepcopy(op))
                        continue
                    previous = ops[positions[target]]
                    left, right = json.loads(previous["value_json"]), json.loads(op["value_json"])
                    if left != right:
                        original = registry[target]["text_or_scalar"]
                        if not all(isinstance(v, str) for v in (original, left, right)):
                            raise ValueError("patch_scalar_merge_conflict")
                        previous["value_json"] = json.dumps(_merge_text(original, [left, right]),
                                                             ensure_ascii=False)
                    provenance["transforms"].append({"kind": "shared_target_merge",
                        "target_id": target, "native_bundle_ids": [b["bundle_id"] for b in members],
                        "merged_value_sha256": digest(json.loads(previous["value_json"]))})
            merged["operations"] = ops
            probe = {**report, "bundles": [merged],
                "unresolved": [f for f in findings if f not in merged["finding_ids"]],
                "unprocessed_finding_ids": []}
            plan = validate_patch_plan(probe, findings=findings, registry=registry,
                                       known_evidence_ids=set(evidence))
            stage_patch_candidate(document, plan, registry, source_index, evidence)
            normalized.append(merged)
            provenance["bundle_lineage"][merged["bundle_id"]] = [b["bundle_id"] for b in members]
        except (ValueError, KeyError, TypeError) as exc:
            provenance["discarded"].append({"bundle_ids": [b["bundle_id"] for b in members],
                "finding_ids": [f for b in members for f in b["finding_ids"]], "reason": str(exc)})
            report["unprocessed_finding_ids"].extend(f for b in members for f in b["finding_ids"])
    report["bundles"] = normalized
    report["complete"] = report["complete"] and not provenance["discarded"]
    plan = validate_patch_plan(report, findings=findings, registry=registry,
                               known_evidence_ids=set(evidence))
    plan["normalization"] = provenance
    return plan, provenance
