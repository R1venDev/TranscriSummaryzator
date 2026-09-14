"""Deterministic plan and atomic/relation surface guards."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import hashlib
import re
from semantics.graph import cross_episode_allowed

CAUSAL_RE = re.compile(r"(?iu)\b(?:из-за|поэтому|привел[оа]? к|в результате|для этого)\b")
NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,:]\d+)*(?:\s*%)?")
NEGATION_RE = re.compile(r"(?iu)\b(?:не|нет|нельзя|без|никогда)\b")
CERTAIN_RE = re.compile(r"(?iu)\b(?:точно|обязательно|гарантированно|решено|утверждено)\b")
CONDITION_RE = re.compile(r"(?iu)\b(?:если|когда|после|перед|пока|при|до тех пор)\b")


@dataclass(frozen=True)
class PublicItem:
    public_id: str
    section: str
    text: str
    claim_ids: list[str]
    evidence_ids: list[str]
    content_kind: str
    social_state: str
    lifecycle: str = "active"
    relation_ids: list[str] = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


def can_publish_as_decision(item):
    return bool(item.get("lifecycle", "active") == "active"
                and item.get("decision_status") == "accepted"
                and item.get("decision_evidence_ids", item.get("evidence_ids"))
                and item.get("acceptance_check", "entailed") == "entailed")


def build_public_items(meeting_graph, summary_plan):
    """Create the complete public contract before formatting Markdown."""
    claims = {x["claim_id"]: x for x in meeting_graph.get("claims", [])}
    view_plans = summary_plan.get("view_plans", {})
    task_states = {x["proposition_id"]: x for x in meeting_graph.get("task_states", [])}
    question_states = {x["proposition_id"]: x for x in meeting_graph.get("question_states", [])}
    sections = []
    def add(section, claim, social_state=None, text=None):
        if claim.get("lifecycle", "active") != "active" or not claim.get("evidence_ids"):
            return
        sections.append(PublicItem(
            public_id=f"PI{len(sections)+1:05d}", section=section,
            text=str(text or claim.get("statement") or "").strip(), claim_ids=[claim["claim_id"]],
            evidence_ids=list(claim.get("evidence_ids", [])), content_kind=claim.get("content_kind") or claim.get("kind"),
            social_state=social_state or claim.get("social_state", "candidate"), lifecycle=claim.get("lifecycle", "active"),
        ).as_dict() | {"start": claim.get("start", 0), "task_state": task_states.get(claim.get("proposition_id"), {}), "question_state": question_states.get(claim.get("proposition_id"), {})})
    def selected(view):
        return [claims[x] for x in view_plans.get(view, {}).get("selected_claim_ids", []) if x in claims]

    def attributed_text(claim):
        text = str(claim.get("statement") or "").strip()
        if re.search(r"(?iu)(?:\b100\s*%|\b100\s+процент)", text) and len(claim.get("speaker_refs", [])) == 1:
            text = f"По словам {claim['speaker_refs'][0]}, {text[:1].lower() + text[1:]}"
        if claim.get("risk", {}).get("recognition", 0) >= .65:
            text += " ⚠ Формулировка или термин требуют проверки."
        return text
    for claim in selected("executive")[:7]:
        add("overview", claim, text=attributed_text(claim))
    for claim in selected("executive"):
        if can_publish_as_decision(claim):
            add("decisions", claim, "accepted", attributed_text(claim))
    for claim in selected("technical"):
        if claim.get("content_kind") in {"trading_rule", "system_rule"}:
            add("rules", claim, "accepted" if can_publish_as_decision(claim) else "described", attributed_text(claim))
    proposed = 0
    confirmed = {"self_committed", "explicit_self_commitment", "assigned", "accepted", "in_progress", "blocked", "completed"}
    for claim in selected("tasks"):
        status = claim.get("task_status", "idea")
        state = task_states.get(claim.get("proposition_id"), {})
        details = [attributed_text(claim)]
        if state.get("assignee"):
            details.append(f"исполнитель: {state['assignee']}")
        if state.get("current_scope"):
            details.append(f"текущий scope: {state['current_scope']}")
        conditions = [x.get("antecedent") or x.get("text") for x in state.get("conditions", []) if isinstance(x, dict) and (x.get("antecedent") or x.get("text"))]
        if conditions:
            details.append("условие: если " + "; ".join(conditions))
        details.append(f"статус: {status}")
        task_text = " — ".join(details)
        if status in confirmed:
            add("tasks", claim, status, task_text)
        elif status in {"proposed", "idea"} and proposed < 3:
            add("tasks", claim, status, task_text); proposed += 1
    for claim in selected("questions")[:10]:
        if claim.get("content_kind") in {"question", "schedule"} and claim.get("question_status") not in {"answered", "rhetorical", "superseded"}:
            state = question_states.get(claim.get("proposition_id"), {})
            missing = state.get("missing_slots", [])
            suffix = f" — остаются открыты поля: {', '.join(missing)}" if missing else ""
            add("questions", claim, claim.get("question_status", "unanswered"), attributed_text(claim) + suffix)
    for claim in selected("experiments"):
        add("experiments", claim, text=attributed_text(claim))
    seen_minutes = set()
    for claim in selected("minutes"):
        key = (claim.get("proposition_id"), claim.get("social_state"), tuple(claim.get("evidence_ids", [])))
        if key not in seen_minutes and len(seen_minutes) < 16:
            add("minutes", claim, text=attributed_text(claim)); seen_minutes.add(key)
    return sections


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
    allowed_scopes = [x.casefold() for x in plan.get("time_scope", []) if isinstance(x, str) and x.strip()]
    if allowed_scopes and not any(scope in (text or "").casefold() for scope in allowed_scopes):
        errors.append("time_scope_not_preserved")
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
            errors = (["empty_public_text"] if not text else []) + (["orphan_public_item"] if not claim_ids else []) + (["claim_outside_plan"] if claim_ids and not plans else [])
            audits.append({"text": text, "claim_ids": claim_ids, "passed": False, "errors": errors, "atomic_claims": claim_ids, "relations": [], "status": "ABSTAIN", "qa": {"passed": False, "checks": {}}})
            continue
        source_mentions = {mention for claim_id in claim_ids for mention in re.findall(r"@[\w.-]+", str(by_claim[claim_id].get("statement") or ""))}
        merged = {"claim_ids": claim_ids, "relation_ids": sorted({r for p in plans for r in p.get("relation_ids", [])}), "allowed_numbers": [n for p in plans for n in p.get("allowed_numbers", [])], "allowed_relation_markers": sorted({r for p in plans for r in p.get("allowed_relation_markers", [])}), "allowed_speakers": sorted({s for p in plans for s in p.get("allowed_speakers", [])} | source_mentions), "allowed_assignees": sorted({s for p in plans for s in p.get("allowed_assignees", [])}), "polarity": [v for p in plans for v in p.get("polarity", [])], "modality": [v for p in plans for v in p.get("modality", [])], "conditions": [v for p in plans for v in p.get("conditions", [])], "time_scope": [v for p in plans for v in p.get("time_scope", [])]}
        realization = audit_realization(text, merged)
        cited = [by_claim[x] for x in claim_ids]
        if item.get("section") and not all(str(x.get("statement") or "").casefold().strip(". ") in text.casefold() for x in cited):
            realization["errors"].append("cleaned_text_semantic_drift")
        if any(x.get("lifecycle", "active") != "active" for x in cited):
            realization["errors"].append("inactive_claim_published")
        if item.get("section") == "decisions" and not all(can_publish_as_decision(x) for x in cited):
            realization["errors"].append("status_upgrade")
        if item.get("section") == "tasks" and any(x.get("task_status") in {"idea", "proposed", "superseded"} for x in cited) and item.get("social_state") in {"accepted", "self_committed", "assigned"}:
            realization["errors"].append("task_status_upgrade")
        qa = qa_verify(text, merged)
        realization["errors"] = sorted(set(realization["errors"]))
        realization["passed"] = not realization["errors"] and qa["passed"]
        if not qa["passed"]:
            realization["errors"] = sorted(set(realization["errors"] + ["qa_slot_failure"]))
            realization["status"] = "ABSTAIN"
        audits.append({"text": text, "claim_ids": claim_ids, **realization, "qa": qa})
    return {"passed": all(x["passed"] for x in audits), "audits": audits, "abstentions": [x for x in audits if not x["passed"]]}


def runtime_quality_gates(report, artifact_text, verified_hash=None):
    audits = report.get("audits", [])
    counters = {
        "unsupported_public_items": sum(not x.get("passed") for x in audits),
        "orphan_public_items": sum("orphan_public_item" in x.get("errors", []) for x in audits),
        "status_upgrades": sum(bool({"status_upgrade", "task_status_upgrade"} & set(x.get("errors", []))) for x in audits),
        "superseded_items_published": sum("inactive_claim_published" in x.get("errors", []) for x in audits),
        "number_or_negation_mismatches": sum(bool({"unplanned_number", "negation_not_preserved"} & set(x.get("errors", []))) for x in audits),
        "cross_episode_merges_without_relation": sum("cross_episode_without_relation" in x.get("errors", []) for x in audits),
        "unknown_assignee_publications": sum("speaker_or_assignee_not_preserved" in x.get("errors", []) for x in audits),
    }
    artifact_hash = hashlib.sha256(artifact_text.encode("utf-8")).hexdigest()
    counters["verified_artifact_hash"] = verified_hash or artifact_hash
    counters["final_artifact_hash"] = artifact_hash
    return {"passed": all(value == 0 for key, value in counters.items() if key.endswith("items") or key.endswith("upgrades") or key.endswith("published") or key.endswith("mismatches") or key.endswith("relation") or key.endswith("publications")) and counters["verified_artifact_hash"] == artifact_hash, **counters}


def diff_public_items(previous, current):
    """Non-blocking shadow diff for release review and regression triage."""
    key = lambda x: (x.get("section"), tuple(x.get("claim_ids", [])))
    before, after = {key(x): x for x in previous or []}, {key(x): x for x in current or []}
    return {
        "added": [after[x] for x in sorted(after.keys() - before.keys())],
        "removed": [before[x] for x in sorted(before.keys() - after.keys())],
        "changed": [{"before": before[x], "after": after[x]} for x in sorted(before.keys() & after.keys()) if before[x].get("text") != after[x].get("text") or before[x].get("social_state") != after[x].get("social_state")],
        "section_counts_before": {section: sum(x.get("section") == section for x in previous or []) for section in sorted({x.get("section") for x in previous or []})},
        "section_counts_after": {section: sum(x.get("section") == section for x in current or []) for section in sorted({x.get("section") for x in current or []})},
    }


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
