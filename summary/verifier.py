"""Deterministic plan and atomic/relation surface guards."""
from __future__ import annotations
import re
from semantics.graph import cross_episode_allowed

CAUSAL_RE = re.compile(r"(?iu)\b(?:из-за|поэтому|привел[оа]? к|в результате|для этого)\b")
NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,:]\d+)*(?:\s*%)?")


def relation_markers(text):
    """Return relation wording that was already present in a source claim."""
    return {match.casefold() for match in CAUSAL_RE.findall(text or "")}


def verify_sentence_plan(plan, claims, relations):
    by_id = {x.get("claim_id"): x for x in claims}
    errors = []
    if any(x not in by_id for x in plan.get("claim_ids", [])):
        errors.append("unknown_claim")
    if not cross_episode_allowed(plan.get("claim_ids", []), plan.get("relation_ids", []), claims, relations):
        errors.append("cross_episode_without_relation")
    relation_ids = {x.get("relation_id") for x in relations}
    if any(x not in relation_ids for x in plan.get("relation_ids", [])):
        errors.append("unknown_relation")
    return {"passed": not errors, "errors": errors}


def audit_realization(text, plan):
    allowed_numbers = {x.replace(" ", "") for x in plan.get("allowed_numbers", [])}
    found_numbers = {x.replace(" ", "") for x in NUMBER_RE.findall(text or "")}
    errors = []
    if not found_numbers.issubset(allowed_numbers):
        errors.append("unplanned_number")
    found_relations = relation_markers(text)
    allowed_relations = {str(x).casefold() for x in plan.get("allowed_relation_markers", [])}
    if found_relations and not plan.get("relation_ids") and not found_relations.issubset(allowed_relations):
        errors.append("unsupported_relation_language")
    return {"passed": not errors, "errors": errors, "atomic_claims": list(plan.get("claim_ids", [])), "relations": list(plan.get("relation_ids", []))}
