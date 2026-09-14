"""Deterministic plan and atomic/relation surface guards."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import hashlib
import re
from semantics.graph import cross_episode_allowed
from summary.policy import RULE_KINDS, TECHNICAL_KINDS

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
    source_word_ids: list[str]
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
    task_states = {x["task_id"]: x for x in meeting_graph.get("task_states", [])}
    question_states = {x["proposition_id"]: x for x in meeting_graph.get("question_states", [])}
    planned_relations = {}
    for sentence in summary_plan.get("public_sentence_plans", []):
        for claim_id in sentence.get("claim_ids", []):
            planned_relations.setdefault(claim_id, []).extend(sentence.get("relation_ids", []))
    sections = []
    materialized = {}
    def add(section, claim, social_state=None, text=None, claim_ids=None, relation_ids=None, extra_evidence=None):
        if claim.get("lifecycle", "active") != "active" or not claim.get("evidence_ids"):
            return
        task_state = task_states.get(claim.get("canonical_task_state_id"), {})
        evidence_ids = task_state.get("evidence_ids", claim.get("evidence_ids", [])) if section == "tasks" else claim.get("evidence_ids", [])
        source_word_ids = task_state.get("source_word_ids", claim.get("source_word_ids", [])) if section == "tasks" else claim.get("source_word_ids", [])
        dialogue = claim.get("dialogue_evidence", []) if section in {"tasks", "questions"} else []
        evidence_ids = list(dict.fromkeys(evidence_ids + [x.get("id") for x in dialogue if x.get("id")]))
        source_word_ids = list(dict.fromkeys(source_word_ids + [w for x in dialogue for w in x.get("source_word_ids", [])]))
        cited_ids = list(dict.fromkeys(claim_ids or [claim["claim_id"]]))
        retained_relations = list(relation_ids or [])
        retained_relations.extend(value for claim_id in cited_ids for value in planned_relations.get(claim_id, []))
        if section == "tasks":
            retained_relations.extend(task_state.get("scope_relation_ids", []))
            retained_relations.extend(task_state.get("acceptance_relation_ids", []))
        public = PublicItem(
            public_id=f"PI{len(sections)+1:05d}", section=section,
            text=str(text or claim.get("statement") or "").strip(), claim_ids=cited_ids,
            evidence_ids=list(dict.fromkeys(evidence_ids + list(extra_evidence or []))), source_word_ids=list(source_word_ids), content_kind=claim.get("content_kind") or claim.get("kind"),
            social_state=social_state or claim.get("social_state", "candidate"), lifecycle=claim.get("lifecycle", "active"), relation_ids=list(dict.fromkeys(retained_relations)),
        ).as_dict() | {"start": claim.get("primary_evidence_start", claim.get("start", 0)), "task_state_id": claim.get("canonical_task_state_id"), "task_state": task_state, "question_state": question_states.get(claim.get("proposition_id"), {}), "topic_entities": [x.get("canonical_name") or x.get("name") for x in claim.get("entities", []) if isinstance(x, dict) and (x.get("canonical_name") or x.get("name"))]}
        sections.append(public)
        for claim_id in cited_ids:
            materialized.setdefault(section, {})[claim_id] = {"status": "published", "public_id": public["public_id"]}
    def selected(view):
        return [claims[x] for x in view_plans.get(view, {}).get("selected_claim_ids", []) if x in claims]

    def attributed_text(claim):
        text = str(claim.get("statement") or "").strip()
        if re.search(r"(?iu)(?:\b100\s*%|\b100\s+процент|\b(?:всегда|никогда|невозможно|нельзя|гарантированно)\b)", text) and len(claim.get("speaker_refs", [])) == 1:
            text = f"По словам {claim['speaker_refs'][0]}, {text[:1].lower() + text[1:]}"
        if claim.get("risk", {}).get("recognition", 0) >= .65:
            text += " ⚠ Формулировка или термин требуют проверки."
        return text
    # Overview is a compact mix of state, obstacle and next step. Repeating a
    # short task outcome across sections is useful; verbatim duplicates are not.
    overview = []
    executive_claims = selected("executive")
    category_order = (
        {"current_state", "observation", "experimental_result"},
        {"problem", "blocker", "constraint"},
        {"action", "follow_up", "decision"},
        set(TECHNICAL_KINDS),
    )
    overview_order, overview_seen = [], set()
    for kinds in category_order:
        candidate = next((x for x in executive_claims if x.get("content_kind") in kinds and x["claim_id"] not in overview_seen), None)
        if candidate:
            overview_order.append(candidate); overview_seen.add(candidate["claim_id"])
    overview_order.extend(x for x in executive_claims if x["claim_id"] not in overview_seen)
    for claim in overview_order:
        if claim.get("content_kind") in {"question", "schedule"} or claim.get("risk", {}).get("recognition", 0) >= .65:
            continue
        if re.search(r"(?iu)^\s*(?:это|так|вот\s+эт\w+|они|он|она)\b", claim.get("statement", "")) and not claim.get("entities"):
            continue
        tokens = set(re.findall(r"(?iu)[a-zа-яё0-9]+", claim.get("statement", "").casefold()))
        if any(len(tokens & old) / max(1, min(len(tokens), len(old))) >= .55 for old in (x[1] for x in overview)):
            continue
        overview.append((claim, tokens))
        if len(overview) == 5: break
    for claim, _ in overview:
        add("overview", claim, text=attributed_text(claim))
    for claim in selected("executive"):
        if can_publish_as_decision(claim):
            add("decisions", claim, "accepted", attributed_text(claim))
    for claim in selected("technical"):
        if claim.get("content_kind") in RULE_KINDS:
            add("rules", claim, "accepted" if can_publish_as_decision(claim) else "described", attributed_text(claim))
        elif claim.get("content_kind") in TECHNICAL_KINDS:
            add("technical", claim, "observation", attributed_text(claim))
    proposed, emitted_tasks = 0, set()
    confirmed = {"self_committed", "explicit_self_commitment", "assigned", "accepted", "in_progress", "blocked", "completed"}
    for claim in selected("tasks"):
        state = task_states.get(claim.get("canonical_task_state_id"), {})
        if not state or state.get("task_id") in emitted_tasks: continue
        emitted_tasks.add(state["task_id"])
        status = state.get("status", "idea")
        description = str(state.get("deliverable") or state.get("description") or claim.get("statement") or "")
        details = [description]
        if state.get("assignee"):
            details.append(f"исполнитель: {state['assignee']}")
        if state.get("current_scope"):
            details.append(f"объём: {state['current_scope']}")
        if state.get("data_origin"):
            details.append(f"период данных: {state['data_origin']}")
        if state.get("deadline"):
            due = state["deadline"].get("text") if isinstance(state["deadline"], dict) else state["deadline"]
            if due: details.append(f"срок: {due}")
        conditions = [x.get("antecedent") or x.get("text") for x in state.get("conditions", []) if isinstance(x, dict) and (x.get("antecedent") or x.get("text"))]
        if conditions:
            details.append("условие: если " + "; ".join(conditions))
        labels = {"self_committed": "участник взял на себя", "proposed": "предложено, не подтверждено", "idea": "идея, не подтверждена", "assigned_pending": "назначение ожидает подтверждения", "assigned": "назначено", "accepted": "согласовано", "completed": "выполнено", "blocked": "заблокировано"}
        if status in labels:
            details.append(f"статус: {labels[status]}")
        task_text = " — ".join(details)
        if status in confirmed:
            add("tasks", claim, status, task_text)
        elif status in {"proposed", "idea", "assigned_pending"} and proposed < 3:
            add("tasks", claim, status, task_text); proposed += 1
    for claim in selected("questions"):
        if claim.get("content_kind") in {"question", "schedule"} and claim.get("question_status") not in {"answered", "rhetorical", "superseded"}:
            state = question_states.get(claim.get("proposition_id"), {})
            missing = [x for x in state.get("missing_slot_labels", []) if x]
            suffix = f" — не уточнено: {', '.join(missing)}" if missing else ""
            add("questions", claim, claim.get("question_status", "unanswered"), attributed_text(claim) + suffix, relation_ids=state.get("answer_relation_ids", []), extra_evidence=state.get("answer_evidence_ids", []))
    for claim in selected("experiments"):
        if claim.get("content_kind") == "hypothesis" and re.search(r"(?iu)\b(?:нельзя|невозможно|ограничен)\b", claim.get("statement", "")):
            add("technical", claim, "constraint", attributed_text(claim))
        else:
            add("experiments", claim, text=attributed_text(claim))
    seen_minutes = set()
    for claim in sorted(selected("minutes"), key=lambda x: (float(x.get("start", 0)), x.get("claim_id", ""))):
        key = (claim.get("proposition_id"), claim.get("social_state"), tuple(claim.get("evidence_ids", [])))
        if key not in seen_minutes:
            add("minutes", claim, text=attributed_text(claim)); seen_minutes.add(key)
    for view, view_plan in view_plans.items():
        for claim_id in view_plan.get("selected_claim_ids", []):
            published = next((values[claim_id] for values in materialized.values() if claim_id in values), None)
            view_plan.setdefault("dispositions", {})[claim_id] = published or {"status": "excluded", "reason": "editorial_route_or_dedup"}
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
        source_tokens = {v for x in cited for v in re.findall(r"(?iu)[a-zа-яё0-9]+", str(x.get("statement") or "").casefold()) if len(v) > 2}
        text_tokens = {v for v in re.findall(r"(?iu)[a-zа-яё0-9]+", text.casefold()) if len(v) > 2}
        if item.get("section") and len(source_tokens & text_tokens) / max(1, min(len(source_tokens), len(text_tokens))) < .45:
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


def publication_audit(report, artifact_text, items=None, summary_plan=None, verified_hash=None):
    items = items or []
    summary_plan = summary_plan or {}
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
    def tokens(value): return {x for x in re.findall(r"(?iu)[a-zа-яё0-9]+", str(value or "").casefold()) if len(x) > 3}
    duplicates = 0
    for index, left in enumerate(items):
        for right in items[index + 1:]:
            if left.get("section") != right.get("section"): continue
            a, b = tokens(left.get("text")), tokens(right.get("text"))
            duplicates += bool(a and b and len(a & b) / max(1, min(len(a), len(b))) >= .8)
    minute_starts = [float(x.get("start", 0)) for x in items if x.get("section") == "minutes"]
    task_ids = [x.get("task_state_id") for x in items if x.get("section") == "tasks"]
    internal = re.compile(r"(?iu)\b(?:self_committed|assigned_pending|additional_tools|rhythmic_entry_implementation|high_tf_result|stop_loss_options)\b")
    overview_tokens = set().union(*(tokens(x.get("text")) for x in items if x.get("section") == "overview"))
    task_tokens = set().union(*(tokens(x.get("text")) for x in items if x.get("section") == "tasks"))
    rendered_counts, current = {}, None
    heading_sections = {"Краткое описание — что изменилось после встречи": "overview", "Принятые решения": "decisions", "Упомянутые действующие правила": "rules", "Задачи и следующие шаги": "tasks", "Открытые вопросы": "questions", "Технические выводы и ограничения": "technical", "Гипотезы и эксперименты": "experiments", "Хронология встречи": "minutes"}
    for line in artifact_text.splitlines():
        if line.startswith("## "): current = heading_sections.get(line[3:].strip())
        elif line.startswith("- ") and current: rendered_counts[current] = rendered_counts.get(current, 0) + 1
    item_counts = {section: sum(x.get("section") == section for x in items) for section in sorted({x.get("section") for x in items})}
    state_conflicts = sum(bool(
        x.get("section") == "tasks" and (not x.get("task_state_id") or x.get("social_state") != x.get("task_state", {}).get("status") or (x.get("task_state", {}).get("current_scope") and str(x["task_state"]["current_scope"]).casefold() not in str(x.get("text") or "").casefold()))
    )
        for x in items
    )
    counters.update({
        "duplicate_items": duplicates,
        "answered_questions_published_as_open": sum(x.get("section") == "questions" and x.get("question_state", {}).get("status") in {"answered", "rhetorical", "superseded"} for x in items),
        "unconfirmed_tasks_published_as_committed": sum(x.get("section") == "tasks" and x.get("social_state") == "self_committed" and x.get("task_state", {}).get("commitment_strength") != "explicit" for x in items),
        "duplicate_task_states": len([x for x in task_ids if x]) - len({x for x in task_ids if x}),
        "chronology_inversions": sum(a > b for a, b in zip(minute_starts, minute_starts[1:])),
        "internal_labels_exposed": sum(bool(internal.search(str(x.get("text") or ""))) for x in items),
        "missing_public_provenance": sum(not x.get("evidence_ids") or not x.get("source_word_ids") for x in items),
        "planner_budget_violations": sum(int(v.get("selected_count", len(v.get("selected_claim_ids", [])))) > int(v.get("budget", 0)) for v in summary_plan.get("view_plans", {}).values()),
        "state_conflicts": state_conflicts,
        "task_without_deliverable": sum(x.get("section") == "tasks" and not str(x.get("task_state", {}).get("deliverable") or "").strip() for x in items),
        "overview_task_overlap": (len(overview_tokens & task_tokens) / max(1, len(overview_tokens))) if overview_tokens else 0,
        "section_count_mismatches": sum(item_counts.get(k, 0) != rendered_counts.get(k, 0) for k in set(item_counts) | set(rendered_counts)),
    })
    counters["section_counts"] = item_counts
    counters["rendered_section_counts"] = rendered_counts
    counters["planner_overflow"] = {name: value.get("overflow_count", 0) for name, value in summary_plan.get("view_plans", {}).items()}
    artifact_hash = hashlib.sha256(artifact_text.encode("utf-8")).hexdigest()
    counters["verified_artifact_hash"] = verified_hash or artifact_hash
    counters["final_artifact_hash"] = artifact_hash
    unexplained = sum(1 for view in summary_plan.get("view_plans", {}).values() for claim_id in view.get("selected_claim_ids", []) if claim_id not in view.get("dispositions", {}))
    public_claims = {claim_id for item in items for claim_id in item.get("claim_ids", [])}
    candidate_ids = set(summary_plan.get("commitment_candidate_ids", []))
    counters.update({"unexplained_selected_claims": unexplained, "unexplained_commitment_candidates": len(candidate_ids - public_claims), "published_unique_claims": len(public_claims)})
    integrity_keys = {"unsupported_public_items", "orphan_public_items", "status_upgrades", "superseded_items_published", "number_or_negation_mismatches", "cross_episode_merges_without_relation", "unknown_assignee_publications", "duplicate_items", "answered_questions_published_as_open", "unconfirmed_tasks_published_as_committed", "duplicate_task_states", "chronology_inversions", "internal_labels_exposed", "missing_public_provenance", "planner_budget_violations", "state_conflicts", "task_without_deliverable", "section_count_mismatches"}
    integrity = all(counters.get(key, 0) == 0 for key in integrity_keys) and counters["verified_artifact_hash"] == artifact_hash
    grounding = counters["unsupported_public_items"] == counters["number_or_negation_mismatches"] == counters["missing_public_provenance"] == 0
    coverage = bool(items) and unexplained == 0 and counters["unexplained_commitment_candidates"] == 0
    readability = counters["duplicate_items"] == counters["internal_labels_exposed"] == 0
    return {"schema": "PublicationAudit", "schema_version": 2, "passed": integrity and grounding and coverage and readability, "dimensions": {"integrity": integrity, "grounding": grounding, "coverage": coverage, "readability": readability}, **counters}


def runtime_quality_gates(report, artifact_text, verified_hash=None, items=None, summary_plan=None):
    return publication_audit(report, artifact_text, items, summary_plan, verified_hash)


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
