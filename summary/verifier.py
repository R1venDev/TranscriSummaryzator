"""Deterministic plan and atomic/relation surface guards."""
from __future__ import annotations
import re
from semantics.graph import cross_episode_allowed

CAUSAL_RE = re.compile(r"(?iu)\b(?:из-за|поэтому|привел[оа]? к|в результате|для этого)\b")
NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,:]\d+)*(?:\s*%)?")
NEGATION_RE = re.compile(r"(?iu)\b(?:не|нет|нельзя|без|никогда)\b")
CERTAIN_RE = re.compile(r"(?iu)\b(?:точно|обязательно|гарантированно|решено|утверждено)\b")
CONDITION_RE = re.compile(r"(?iu)\b(?:если|когда|после|перед|пока|при|до тех пор)\b")


def relation_markers(text):
    """Return relation wording that was already present in a source claim."""
    return {match.casefold() for match in CAUSAL_RE.findall(text or "")}


def source_aware_plan(plan, claims):
    """Allow participant references that occur verbatim in cited source claims."""
    result = dict(plan)
    source_text = " ".join(str(claim.get("statement") or "") for claim in claims)
    result["allowed_speakers"] = sorted(
        set(plan.get("allowed_speakers", [])) | set(re.findall(r"@[\w.-]+", source_text))
    )
    return result


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
    polarities = set(plan.get("polarity", []))
    if "negative" in polarities and not NEGATION_RE.search(text or ""):
        errors.append("negation_not_preserved")
    modalities = set(plan.get("modality", []))
    if modalities & {"possible", "hypothetical", "unknown", "tentative"} and CERTAIN_RE.search(text or ""):
        errors.append("modality_upgraded")
    if plan.get("conditions") and not CONDITION_RE.search(text or ""):
        errors.append("condition_not_preserved")
    allowed_values = list(plan.get("allowed_speakers", [])) + list(plan.get("allowed_assignees", []))
    allowed_speakers = set(allowed_values) | set(re.findall(r"@[\w.-]+", " ".join(allowed_values)))
    mentioned = set(re.findall(r"@[\w.-]+", text or ""))
    if not mentioned.issubset(allowed_speakers):
        errors.append("speaker_or_assignee_not_preserved")
    return {"passed": not errors, "errors": sorted(set(errors)), "atomic_claims": list(plan.get("claim_ids", [])), "relations": list(plan.get("relation_ids", [])), "status": "SUPPORTED" if not errors else "ABSTAIN"}


def verify_generated_items(items, sentence_plans, claims):
    """Audit actual generated text against the union contract of cited claims."""
    by_claim = {x.get("claim_id"): x for x in claims}
    plan_by_claim = {claim_id: plan for plan in sentence_plans for claim_id in plan.get("claim_ids", [])}
    audits = []
    for item in items:
        text = str(item.get("text") or item.get("statement") or "")
        claim_ids = [x for x in (item.get("claim_ids") or item.get("fact_ids", [])) if x in by_claim]
        plans = [plan_by_claim[x] for x in claim_ids if x in plan_by_claim]
        if not text or not claim_ids or not plans:
            continue
        if item.get("_semantic_role") in {"main_topic", "overview"}:
            audits.append({"text": text, "claim_ids": claim_ids, "passed": True, "errors": [], "atomic_claims": claim_ids, "relations": [], "status": "NAVIGATION", "qa": {"passed": True, "checks": {}}})
            continue
        source_mentions = {mention for claim_id in claim_ids for mention in re.findall(r"@[\w.-]+", str(by_claim[claim_id].get("statement") or ""))}
        merged = {"claim_ids": claim_ids, "relation_ids": sorted({r for p in plans for r in p.get("relation_ids", [])}), "allowed_numbers": [n for p in plans for n in p.get("allowed_numbers", [])], "allowed_relation_markers": sorted({r for p in plans for r in p.get("allowed_relation_markers", [])}), "allowed_speakers": sorted({s for p in plans for s in p.get("allowed_speakers", [])} | source_mentions), "allowed_assignees": sorted({s for p in plans for s in p.get("allowed_assignees", [])}), "polarity": [v for p in plans for v in p.get("polarity", [])], "modality": [v for p in plans for v in p.get("modality", [])], "conditions": [v for p in plans for v in p.get("conditions", [])]}
        realization = audit_realization(text, merged)
        qa = qa_verify(text, merged)
        realization["passed"] = realization["passed"] and qa["passed"]
        if not qa["passed"]:
            realization["errors"] = sorted(set(realization["errors"] + ["qa_slot_failure"]))
            realization["status"] = "ABSTAIN"
        audits.append({"text": text, "claim_ids": claim_ids, **realization, "qa": qa})
    return {"passed": all(x["passed"] for x in audits), "audits": audits, "abstentions": [x for x in audits if not x["passed"]]}


def alignment_score(premise, hypothesis):
    """Cheap independent alignment stage; ambiguous cases are escalated by caller."""
    tokens = lambda x: {v for v in re.findall(r"(?iu)[a-zа-яё0-9]+", x or "") if len(v) > 2}
    left, right = tokens(premise), tokens(hypothesis)
    entailment = len(left & right) / max(1, len(right))
    contradiction = 1.0 if bool(NEGATION_RE.search(premise or "")) != bool(NEGATION_RE.search(hypothesis or "")) else 0.0
    return {"entailment": entailment, "contradiction": contradiction, "ambiguous": entailment < .72 or contradiction > 0}


def qa_verify(text, plan):
    """Independent slot checks for who/quantity/condition/state questions."""
    assignment_claimed = bool(re.search(r"(?iu)\b(?:поручено|ответственн(?:ый|ая)|должен|владелец)\b", text or ""))
    allowed_values = list(plan.get("allowed_assignees", [])) + list(plan.get("allowed_speakers", []))
    allowed_people = set(re.findall(r"@[\w.-]+", " ".join(allowed_values)))
    mentioned_people = set(re.findall(r"@[\w.-]+", text or ""))
    checks = {"who": not assignment_claimed or (bool(mentioned_people) and mentioned_people.issubset(allowed_people)), "quantity": not plan.get("allowed_numbers") or set(NUMBER_RE.findall(text)).issubset(set(plan["allowed_numbers"])), "condition": not plan.get("conditions") or bool(CONDITION_RE.search(text)), "decision_state": not plan.get("decision_state") or not ("решено" in text.casefold() and "accepted" not in plan["decision_state"])}
    return {"passed": all(checks.values()), "checks": checks}
